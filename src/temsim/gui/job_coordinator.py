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

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot, QCoreApplication, QThread, QEventLoop, Qt

from temsim.cache_memory import retained_memory_inventory

GIB = 1024**3
DEFAULT_WORKING_BYTES = 24 * GIB


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


def _failure(worker, message):
    signals = worker.signals
    prefix = _signal_prefix(worker)
    target = getattr(signals, "error", None)
    if target is None:
        target = signals.failed
    target.emit(*prefix, str(message))
    signals.finished.emit(*prefix)


def _cancel_queued(job):
    job.cancellation.set()
    assets = getattr(getattr(job.worker, "request", None), "_input_assets", None)
    if assets is not None:
        assets.close()
    job.worker.signals.finished.emit(*_signal_prefix(job.worker))


class _RunnerSignals(QObject):
    done = Signal(str, object)


class _Runner(QRunnable):
    def __init__(self, job, numerical_threads, vram_budget_bytes):
        super().__init__()
        self.job = job
        self.numerical_threads = numerical_threads
        self.vram_budget_bytes = vram_budget_bytes
        self.signals = _RunnerSignals()

    def run(self):
        failure = None
        try:
            # Only one numerical worker is admitted at once, so this scoped
            # library setting cannot race another app worker's BLAS context.
            from threadpoolctl import threadpool_limits
            from temsim.physics.ray_device_cache import device_budget
            with threadpool_limits(limits=self.numerical_threads), device_budget(self.vram_budget_bytes):
                try:
                    import numba
                except ImportError:
                    numba = None
                previous = numba.get_num_threads() if numba is not None else None
                try:
                    if previous is not None:
                        numba.set_num_threads(min(previous, self.numerical_threads))
                    self.job.worker.run()
                finally:
                    if previous is not None:
                        numba.set_num_threads(previous)
        except Exception as exc:
            failure = exc
        finally:
            self.signals.done.emit(self.job.identifier, failure)


class JobCoordinator(QObject):
    changed = Signal()

    def __init__(self, parent=None, *, ram_budget_bytes=40*GIB, vram_budget_bytes=8*GIB, numerical_threads=4):
        super().__init__(parent)
        self.ram_budget_bytes, self.vram_budget_bytes = ram_budget_bytes, vram_budget_bytes
        self.numerical_threads = numerical_threads
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.queue = deque()
        self.active = {}
        self.history = deque(maxlen=128)
        self._providers = []

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
        QTimer.singleShot(0, self._dispatch)

    def submit(self, owner, worker, *, claim=None):
        identity = str(getattr(worker, "job_input_identity", ""))
        job = Job(owner, worker, claim or getattr(worker, "resource_claim", ResourceClaim()),
                  int(getattr(worker, "generation", 0)), identity,
                  "live" if getattr(worker, "quality", "") in ("Preview", "Medium") else "explicit", _event(worker))
        worker.job_identifier = job.identifier
        worker._coordinator_failure = None
        worker_ref = weakref.ref(worker)
        def remember_failure(*args):
            target = worker_ref()
            if target is not None:
                target._coordinator_failure = str(args[-1]) if args else "Worker reported failure"
        for name in ("failed", "error"):
            signal = getattr(worker.signals, name, None)
            if signal is not None:
                # Only scalar outcome bookkeeping runs here; UI receivers keep
                # their queued connections and their own error presentation.
                signal.connect(remember_failure, Qt.ConnectionType.DirectConnection)
        job.input_inventory = retained_memory_inventory(*_input_roots(worker))
        worker.setAutoDelete(False)
        self.queue.append(job)
        self.changed.emit()
        QTimer.singleShot(0, self._dispatch)
        return job.identifier

    @Slot()
    def _dispatch(self):
        if self.active:
            return
        while self.queue:
            job = self.queue[0]
            if job.cancellation.is_set():
                self.queue.popleft()
                _cancel_queued(job)
                self._record(job, "cancelled")
                continue
            retained = self._retained_bytes()
            if (retained + job.claim.working_bytes > self.ram_budget_bytes
                    or job.claim.vram_bytes > self.vram_budget_bytes):
                self.queue.popleft()
                _failure(job.worker, "Shared job memory reservation exceeds the configured budget; lower numerical costs or adjust Performance and cache explicitly")
                self._record(job, "rejected")
                continue
            self.queue.popleft()
            job.status = "running"
            runner = _Runner(job, self.numerical_threads, self.vram_budget_bytes)
            runner.signals.done.connect(self._done)
            self.active[job.identifier] = (job, runner)
            self.pool.start(runner)
            self.changed.emit()
            return

    def _record(self, job, status):
        job.status = status
        self.history.append(dict(identifier=job.identifier, revision=job.revision,
                                 input_identity=job.input_identity, priority=job.priority, status=status))
        job.input_inventory.clear()
        self.changed.emit()

    @Slot(str, object)
    def _done(self, identifier, failure):
        job, runner = self.active.pop(identifier)
        if failure is not None:
            _failure(job.worker, failure)
        failed = failure is not None or getattr(job.worker, "_coordinator_failure", None) is not None
        self._record(job, "failed" if failed else "cancelled" if job.cancellation.is_set() else "finished")
        QTimer.singleShot(0, self._dispatch)

    def clear(self, owner, predicate=None):
        retained = deque()
        for job in self.queue:
            if job.owner is owner and (predicate is None or predicate(job.worker)):
                _cancel_queued(job)
                self._record(job, "cancelled")
            else:
                retained.append(job)
        self.queue = retained

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
