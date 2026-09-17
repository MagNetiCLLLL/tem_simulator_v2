"""FIFO resource admission around the application's existing Qt workers.

Reservations are conservative ownership estimates, not a hard process RSS cap.
No new physical State, solver, process pool or live-state writer is introduced.
"""
from collections import deque
from dataclasses import dataclass, field
from threading import Event
from time import monotonic
from uuid import uuid4
import weakref

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot, QCoreApplication, QThread, QEventLoop, QEvent, Qt

from temsim.cache_memory import retained_memory_inventory
from temsim.job_events import JobEvents, traced_job, job_event, job_stage

GIB = 1024**3
DEFAULT_WORKING_BYTES = 24 * GIB
_DISPATCH_EVENT = QEvent.Type(QEvent.registerEventType())


@dataclass(frozen=True)
class ResourceClaim:
    working_bytes: int = DEFAULT_WORKING_BYTES
    vram_bytes: int = 0

    def __post_init__(self):
        if any(type(n) is not int or n < 0 for n in (self.working_bytes, self.vram_bytes)):
            raise ValueError("Job reservations must be nonnegative byte counts")


@dataclass
class Job:
    owner: object
    worker: object
    claim: ResourceClaim
    revision: int
    input_identity: str
    priority: str
    cancellation: Event
    identifier: str = field(default_factory=lambda: uuid4().hex)
    status: str = "queued"
    input_inventory: dict = field(default_factory=dict)
    adapter: object = None
    terminal: bool = False
    submitted_s: float = field(default_factory=monotonic)

    def metadata(self):
        return dict(job_id=self.identifier, request_id=self.adapter.request_id or self.input_identity,
                    owner=self.adapter.owner_name, worker=type(self.worker).__name__,
                    generation=self.revision, input_identity=self.input_identity,
                    backend=self.adapter.backend, working_bytes=self.claim.working_bytes,
                    vram_bytes=self.claim.vram_bytes)


@dataclass
class WorkerAdapter:
    """Explicit incremental contract; callbacks keep existing Qt signal APIs.

    A job terminates when run() returns, not when finished/prepared is emitted.
    Preparation success deliberately transfers the controller request to its
    solver without a controller-finished notification.
    """
    worker: object
    cancellation: Event
    claim: ResourceClaim
    revision: int
    input_identity: str
    quality: str
    roots: tuple
    fail: object
    finish: object
    release: object
    numerical_context: bool = True
    request_id: str = ""
    backend: str = "not_selected"
    owner_name: str = ""
    failure: str | None = None
    finished_notified: bool = False
    released: bool = False
    observer: object = None

    def release_once(self):
        if not self.released:
            self.released = True
            self.release()

    def finish_once(self):
        if not self.finished_notified:
            self.finished_notified = True
            self.finish()

    def fail_once(self, error):
        if self.failure is None:
            self.failure = str(error)
            self.fail(str(error))
        self.finish_once()


class _OutcomeObserver(QObject):
    """Explicit Qt receiver; no temporary callable proxies on worker signals."""
    def __init__(self, adapter, parent):
        super().__init__(parent)
        self.adapter = adapter

    @Slot(str)
    @Slot(int, str)
    @Slot(int, str, str)
    def failed(self, *args):
        self.adapter.failure = str(args[-1]) if args else "Worker reported failure"

    @Slot()
    def finished(self):
        self.adapter.finished_notified = True


def legacy_adapter(worker):
    """Confine unconverted worker conventions here, not in queue policy."""
    prefix = _signal_prefix(worker)
    signals = worker.signals
    failure = getattr(signals, "error", None)
    if failure is None:
        failure = signals.failed
    def release():
        assets = getattr(getattr(worker, "request", None), "_input_assets", None)
        if assets is not None:
            assets.close()
    return WorkerAdapter(worker, _event(worker), getattr(worker, "resource_claim", ResourceClaim()),
                         int(getattr(worker, "generation", 0)), str(getattr(worker, "job_input_identity", "")),
                         str(getattr(worker, "quality", "")), _input_roots(worker),
                         lambda message: failure.emit(*prefix, message), lambda: signals.finished.emit(*prefix), release)


def _event(worker):
    for name in ("cancel_event", "cancellation", "cancelled", "event", "_cancel", "cancel"):
        value = getattr(worker, name, None)
        if isinstance(value, Event):
            return value
    value = Event()
    worker.cancel_event = value
    return value


def _input_roots(worker):
    return tuple(getattr(worker, name) for name in
                 ("request", "snapshot", "state", "_state", "payload", "recipe", "existing_result", "record")
                 if hasattr(worker, name))


def _signal_prefix(worker):
    prefix = ()
    if hasattr(worker, "generation"):
        prefix = (worker.generation,)
        if hasattr(worker, "quality"):
            prefix += (worker.quality,)
        elif type(worker).__name__ == "DirectAlignmentWorker":
            prefix += (worker.key,)
    return prefix


class _RunnerSignals(QObject):
    done = Signal(str, object)


class _Runner(QRunnable):
    def __init__(self, job, numerical_threads, vram_budget_bytes, events):
        super().__init__()
        self.setAutoDelete(False)
        self.job = job
        self.numerical_threads = numerical_threads
        self.vram_budget_bytes = vram_budget_bytes
        self.signals = _RunnerSignals()
        self.events = events

    def run(self):
        failure = None
        try:
            with traced_job(self.events, self.job.metadata()):
                job_event("worker_entry")
                if self.job.cancellation.is_set():
                    return
                if self.job.adapter.numerical_context:
                    self._run_numerical()
                else:
                    with job_stage("worker_body"):
                        self.job.worker.run()
        except Exception as exc:
            failure = exc
        finally:
            self.signals.done.emit(self.job.identifier, failure)

    def _run_numerical(self):
        # Setup has its own timing: admission is not worker entry. Preparation
        # still uses these limits because it can build active mapped fields.
        with job_stage("library_setup", backend="CPU setup"):
            from threadpoolctl import threadpool_limits
            from temsim.physics.ray_device_cache import device_budget
            try:
                import numba
            except ImportError:
                numba = None
            limits = threadpool_limits(limits=self.numerical_threads)
            previous = numba.get_num_threads() if numba is not None else None
        try:
            with limits, device_budget(self.vram_budget_bytes):
                if previous is not None:
                    numba.set_num_threads(min(previous, self.numerical_threads))
                try:
                    with job_stage("worker_body"):
                        self.job.worker.run()
                finally:
                    if previous is not None:
                        numba.set_num_threads(previous)
        finally:
            job_event("library_cleanup")


class JobCoordinator(QObject):
    changed = Signal()

    def __init__(self, parent=None, *, ram_budget_bytes=40*GIB, vram_budget_bytes=8*GIB, numerical_threads=4):
        application = QCoreApplication.instance()
        # Queued Qt delivery and pool threads outlive Python local references.
        # Give the dispatch owner a Qt lifetime, including embedded/test users
        # that do not go through application_coordinator(). Explicit owners
        # can still dispose it with deleteLater after draining their jobs.
        super().__init__(parent if parent is not None else application)
        if application is not None:
            from temsim.gui.garbage_collection import install_gui_gc
            install_gui_gc(application)
        self.ram_budget_bytes, self.vram_budget_bytes = ram_budget_bytes, vram_budget_bytes
        self.numerical_threads = numerical_threads
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.queue = deque()
        self.active = {}
        self._completions = {}
        self.history = deque(maxlen=128)
        self.events = JobEvents()
        self._providers = []
        self._dispatch_pending = False
        self._completion_timer = QTimer(self)
        self._completion_timer.setSingleShot(True)
        self._completion_timer.setInterval(1)
        self._completion_timer.timeout.connect(self._drain_completions)

    def _schedule_dispatch(self):
        # Zero timers run after pending posted events. Bulk plot destruction
        # was observed delaying admission by >5 s before any worker entered.
        # Coalesce one high-priority admission event, not a solver/UI callback;
        # FIFO and cancellation-before-admission still apply. This cannot
        # preempt a callback already running or promise a hard GUI deadline.
        if not self._dispatch_pending:
            self._dispatch_pending = True
            QCoreApplication.postEvent(self, QEvent(_DISPATCH_EVENT), Qt.EventPriority.HighEventPriority.value)

    def event(self, event):
        if event.type() == _DISPATCH_EVENT:
            self._dispatch_pending = False
            self._dispatch()
            return True
        return super().event(event)

    def register_retained(self, owner, method):
        """Weak owners; returned roots are counted together, including aliases."""
        self._providers.append((weakref.ref(owner), method))

    def _retained_bytes(self):
        from temsim.input_assets import INPUT_ASSETS
        from temsim.physics.prepared_specimen_cache import retained_prepared_specimen_roots
        from temsim.specimen.display_cache import retained_sample_display_roots
        roots = [INPUT_ASSETS.retained_buffers(), retained_prepared_specimen_roots(), retained_sample_display_roots()]
        live = []
        for ref, method in self._providers:
            owner = ref()
            if owner is not None:
                roots.append(getattr(owner, method)())
                live.append((ref, method))
        self._providers = live
        inventory = retained_memory_inventory(*roots)
        for job in (*self.queue, *(entry[0] for entry in self.active.values())):
            for key, size in job.input_inventory.items():
                inventory[key] = max(size, inventory.get(key, 0))
        return sum(inventory.values())

    def configure(self, *, ram_budget_bytes, vram_budget_bytes=None):
        if type(ram_budget_bytes) is not int or ram_budget_bytes <= 0:
            raise ValueError("Shared worker memory budget must be positive")
        if vram_budget_bytes is not None and (type(vram_budget_bytes) is not int or vram_budget_bytes < 0):
            raise ValueError("Shared device reservation must be nonnegative")
        self.ram_budget_bytes = ram_budget_bytes
        if vram_budget_bytes is not None:
            self.vram_budget_bytes = vram_budget_bytes
        # Running jobs keep their reservations. New requests use the new cap.
        self._schedule_dispatch()

    def submit(self, owner, worker, *, claim=None):
        factory = getattr(worker, "job_adapter", None)
        adapter = factory() if factory is not None else legacy_adapter(worker)
        if claim is not None:
            adapter.claim = claim
        parent = owner.parent() if isinstance(owner, QObject) else None
        adapter.owner_name = type(parent if parent is not None else owner).__name__
        job = Job(owner, worker, adapter.claim, adapter.revision, adapter.input_identity,
                  "live" if adapter.quality in ("Preview", "Medium") else "explicit", adapter.cancellation,
                  adapter=adapter)
        worker.job_identifier = job.identifier
        adapter.observer = _OutcomeObserver(adapter, self)
        worker.signals.finished.connect(adapter.observer.finished, Qt.ConnectionType.DirectConnection)
        for name in ("failed", "error"):
            signal = getattr(worker.signals, name, None)
            if signal is not None:
                # Only scalar outcome bookkeeping runs here; UI receivers keep
                # their queued connections and their own error presentation.
                signal.connect(adapter.observer.failed, Qt.ConnectionType.DirectConnection)
        job.input_inventory = retained_memory_inventory(*adapter.roots)
        worker.setAutoDelete(False)
        self.queue.append(job)
        self.events.record("enqueue", metadata=job.metadata())
        self.changed.emit()
        self._schedule_dispatch()
        return job.identifier

    @Slot()
    def _dispatch(self):
        if self.active:
            return
        while self.queue:
            job = self.queue[0]
            self.events.record("dispatch_entry", metadata=job.metadata())
            if job.cancellation.is_set():
                self.queue.popleft()
                self.events.record("cancellation_observed", metadata=job.metadata())
                job.adapter.finish_once()
                self._record(job, "cancelled")
                continue
            with traced_job(self.events, job.metadata()), job_stage("retained_inventory", backend="CPU admission"):
                retained = self._retained_bytes()
            if (retained + job.claim.working_bytes > self.ram_budget_bytes
                    or job.claim.vram_bytes > self.vram_budget_bytes):
                self.queue.popleft()
                job.adapter.fail_once("Shared job memory reservation exceeds the configured budget; lower numerical costs or adjust Performance and cache explicitly")
                self._record(job, "rejected")
                continue
            self.queue.popleft()
            job.status = "admitted"
            self.events.record("admission", metadata=job.metadata(), retained_bytes=retained)
            runner = _Runner(job, self.numerical_threads, self.vram_budget_bytes, self.events)
            # The sender outlives delivery of its queued done event. Releasing
            # the last Python runnable reference inside that delivery must not
            # synchronously destroy its unparented QObject signal sender.
            runner.signals.setParent(self)
            runner.signals.done.connect(self._done)
            self.active[job.identifier] = (job, runner)
            self.pool.start(runner)
            self.changed.emit()
            return

    def _record(self, job, status):
        if job.terminal:
            return
        job.terminal = True
        job.status = status
        try:
            job.adapter.release_once()
        except Exception as exc:
            job.adapter.fail_once(exc)
            status = job.status = "failed"
        self.events.record("cleanup", metadata=job.metadata())
        self.history.append(dict(identifier=job.identifier, revision=job.revision,
                                 input_identity=job.input_identity, priority=job.priority, status=status,
                                 elapsed_s=monotonic()-job.submitted_s, error=job.adapter.failure))
        job.input_inventory.clear()
        self.events.record("terminal", metadata=job.metadata(), outcome=status)
        job.adapter.observer.deleteLater()
        job.adapter.observer = None
        self.changed.emit()

    @Slot(str, object)
    def _done(self, identifier, failure):
        if identifier not in self.active or identifier in self._completions:
            return
        self._completions[identifier] = failure
        self.events.record("worker_return_signal", metadata=self.active[identifier][0].metadata())
        self._drain_completions()

    @Slot()
    def _drain_completions(self):
        # A queued done signal can arrive before QRunnable.run has unwound in
        # C++. Keep the wrapper, QObject signals and pool owner alive until Qt
        # confirms return, including when a test/window closes immediately.
        if self.pool.activeThreadCount():
            self._completion_timer.start()
            return
        for identifier, failure in tuple(self._completions.items()):
            self._completions.pop(identifier)
            self._complete(identifier, failure)

    def _complete(self, identifier, failure):
        entry = self.active.pop(identifier, None)
        if entry is None:
            return
        job, runner = entry
        runner.signals.deleteLater()
        if failure is not None:
            job.adapter.fail_once(failure)
        elif job.cancellation.is_set():
            job.adapter.finish_once()
        failed = failure is not None or job.adapter.failure is not None
        self._record(job, "failed" if failed else "cancelled" if job.cancellation.is_set() else "finished")
        self._schedule_dispatch()

    def clear(self, owner, predicate=None):
        # Active ownership stays reserved until the worker returns. Cancellation
        # only sets the existing safe-boundary token; it never kills a kernel.
        for job, _ in self.active.values():
            if job.owner is owner and (predicate is None or predicate(job.worker)):
                self.events.record("cancellation_request", metadata=job.metadata())
                job.cancellation.set()
        retained = deque()
        for job in self.queue:
            if job.owner is owner and (predicate is None or predicate(job.worker)):
                self.events.record("cancellation_request", metadata=job.metadata())
                job.cancellation.set()
                job.adapter.finish_once()
                self._record(job, "cancelled")
            else:
                retained.append(job)
        self.queue = retained

    def diagnostic_snapshot(self, *, include_stacks=False):
        """Scalar ownership plus thread stacks for an observed timeout."""
        result = dict(events=self.events.snapshot(), history=list(self.history),
                      outstanding=[dict(job.metadata(), status=job.status)
                                   for job in (*self.queue, *(pair[0] for pair in self.active.values()))])
        collector = getattr(QCoreApplication.instance(), "_temsim_gui_gc", None)
        if collector is not None:
            result["gui_collections"] = list(collector.history)
        if include_stacks:
            import sys
            import traceback
            result["thread_stacks"] = {str(key): "".join(traceback.format_stack(frame))
                                      for key, frame in sys._current_frames().items()}
        return result

    def has_owner(self, owner):
        return any(job.owner is owner for job in self.queue) or any(job.owner is owner for job, _ in self.active.values())

    def statistics(self):
        from temsim.physics.ray_device_cache import DEVICE_CACHE
        return dict(queued=len(self.queue), running=len(self.active),
                    retained_ray_vram_bytes=DEVICE_CACHE.retained_bytes,
                    reserved_working_bytes=sum(job.claim.working_bytes for job, _ in self.active.values()),
                    reserved_vram_bytes=sum(job.claim.vram_bytes for job, _ in self.active.values()),
                    retained_bytes=self._retained_bytes(), ram_budget_bytes=self.ram_budget_bytes,
                    vram_budget_bytes=self.vram_budget_bytes, numerical_threads=self.numerical_threads)


_HEADLESS = None


def application_coordinator():
    global _HEADLESS
    application = QCoreApplication.instance()
    if application is None:
        # Synchronous callers/tests may construct a controller and run its
        # workers directly. Actual queued dispatch still needs an event loop.
        if _HEADLESS is None:
            _HEADLESS = JobCoordinator()
        return _HEADLESS
    result = getattr(application, "_temsim_job_coordinator", None)
    if result is None:
        from temsim.gui.garbage_collection import install_gui_gc
        install_gui_gc(application)
        result = JobCoordinator(application)
        application._temsim_job_coordinator = result
    return result


class CoordinatedPool(QObject):
    """Small QThreadPool adapter retaining the existing controller API."""
    def __init__(self, parent=None, *, coordinator=None):
        super().__init__(parent)
        self.coordinator = coordinator or application_coordinator()

    def setMaxThreadCount(self, count):
        if count != 1:
            raise ValueError("Shared admission currently permits one numerical worker")

    def maxThreadCount(self):
        return 1

    def start(self, worker, priority=0):
        return self.coordinator.submit(self, worker)

    def clear(self, predicate=None):
        self.coordinator.clear(self, predicate)

    def activeThreadCount(self):
        return sum(job.owner is self for job, _ in self.coordinator.active.values())

    def waitForDone(self, msecs=-1):
        # Bounded event processing delivers final queued signals; ownership is
        # released only after the worker returns. Never kill a solver/kernel.
        if QThread.currentThread() != self.thread():
            raise RuntimeError("Wait for worker disposal from its owning thread")
        deadline = monotonic() + (3. if msecs < 0 else msecs / 1000.)
        while self.coordinator.has_owner(self) and monotonic() < deadline:
            QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents, 10)
            self.coordinator.pool.waitForDone(10)
        return not self.coordinator.has_owner(self)
