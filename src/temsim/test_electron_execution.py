"""Persistent isolated execution of unchanged diagnostic electron physics.

The GUI's coordinator retains numerical CPU admission while a hidden child
process executes one job with one numerical thread. Thus the Python integrator
cannot hold the GUI interpreter's GIL, and other simulator numerical workers
cannot multiply the process-wide CPU budget. Prepared fields stay in the child;
adjustable electrons send only settings and can receive executed trajectory
prefixes before their complete result. Prefixes are observations, never caches
or independently configurable starting states.

The binary protocol is private to a directly spawned child, never a file import
or a network endpoint. Captured input graphs retain mapping-proxy immutability.
"""
from __future__ import annotations

import atexit
import copyreg
from dataclasses import dataclass, fields, is_dataclass
import io
import os
import pickle
from queue import Empty, Queue
import struct
import subprocess
import sys
from threading import Event, Lock, RLock, Thread
from types import MappingProxyType
from uuid import uuid4
import weakref


class ElectronExecutionError(RuntimeError):
    """The isolated worker failed; no trajectory was accepted."""


class ElectronExecutionCancelled(InterruptedError):
    """Cancellation includes CPU admission, preparation and transport."""


@dataclass(frozen=True)
class RemoteElectronScene:
    """Display metadata and identity of an executed field preparation.

    This is neither a field provider nor an electron source. Only the owning
    backend/process can use its opaque token for further trajectories.
    """

    token: str
    process_identity: str
    initial_position_m: tuple[float, float, float]
    initial_energy_ev: float
    default_path_length_m: float
    bounds_m: tuple[tuple[float, float, float], tuple[float, float, float]]
    diagnostic_bounds_m: tuple[tuple[float, float, float], tuple[float, float, float]]
    notes: tuple[str, ...]


def _mapping_proxy(values):
    return MappingProxyType(values)


def _reduce_mapping_proxy(value):
    return _mapping_proxy, (dict(value),)


def _dumps(value):
    stream = io.BytesIO()
    pickler = pickle.Pickler(stream, protocol=5)
    pickler.dispatch_table = {**copyreg.dispatch_table, MappingProxyType: _reduce_mapping_proxy}
    pickler.dump(value)
    return stream.getvalue()


def _send(stream, value):
    data = _dumps(value)
    for block in (struct.pack("!Q", len(data)), data):
        remaining = memoryview(block)
        while remaining:
            written = stream.write(remaining)
            if not written:
                raise BrokenPipeError("Diagnostic electron command pipe closed")
            remaining = remaining[written:]
    stream.flush()


def _read_exact(stream, size):
    parts = []
    while size:
        data = stream.read(size)
        if not data:
            raise EOFError("Diagnostic electron worker closed its output")
        parts.append(data)
        size -= len(data)
    return b"".join(parts)


def _receive(stream):
    size, = struct.unpack("!Q", _read_exact(stream, 8))
    # Internal framing guard; ordinary field meshes can legitimately be large.
    if size > 4*1024**3:
        raise ElectronExecutionError("Diagnostic worker response exceeds the 4 GiB message limit")
    return pickle.loads(_read_exact(stream, size))


def _read_responses(stream, output):
    try:
        while True:
            output.put(_receive(stream))
    except Exception as exc:  # protocol/import failures must wake the owner
        output.put(("disconnected", str(exc)))


def _freeze_result_arrays(result):
    """Restore immutable solver arrays after crossing the private pickle pipe."""
    import numpy as np
    if is_dataclass(result):
        for item in fields(result):
            array = getattr(result, item.name)
            if isinstance(array, np.ndarray):
                array.setflags(write=False)
    return result


_BACKENDS = weakref.WeakSet()


def _close_backends():
    for backend in tuple(_BACKENDS):
        backend.close()


atexit.register(_close_backends)


class ElectronExecutionBackend:
    """One lazily started process and one prepared scene per GUI controller.

    Call prepare/trace in a Qt worker thread; both methods block while keeping
    the GUI thread free. Cancellation is a callable, normally Event.is_set.
    close() may be called from the GUI thread and also interrupts admission.
    """

    def __init__(self):
        self._process = None
        self._identity = None
        self._scene_token = None
        self._responses = None
        self._reader = None
        self._lock = Lock()
        self._process_lock = RLock()
        self._closed = Event()
        self._sequence = 0
        _BACKENDS.add(self)

    @property
    def process_id(self):
        process = self._process
        return process.pid if process is not None and process.poll() is None else None

    def owns_scene(self, scene):
        return (isinstance(scene, RemoteElectronScene) and not self._closed.is_set()
                and scene.process_identity == self._identity and scene.token == self._scene_token
                and self.process_id is not None)

    def _start(self):
        with self._process_lock:
            self._start_locked()

    def _start_locked(self):
        if self._closed.is_set():
            raise ElectronExecutionCancelled("Diagnostic electron execution was closed")
        if self._process is not None and self._process.poll() is None:
            return
        self._stop_process()
        self._identity = uuid4().hex
        environment = os.environ.copy()
        # The child is always a single numerical worker, including before NumPy
        # and Numba import. The parent holds the shared admission lock meanwhile.
        for name in ("TEMSIM_CPU_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OMP_THREAD_LIMIT",
                     "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[name] = "1"
        environment["OMP_MAX_ACTIVE_LEVELS"] = "1"
        environment["OMP_NESTED"] = "FALSE"
        environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(
            os.path.abspath(path or os.getcwd()) for path in sys.path if isinstance(path, str)))
        executable = sys.executable
        if os.name == "nt" and sys.prefix != sys.base_prefix:
            # CPython's Windows venv python.exe is a redirector. Spawning it
            # would make Popen own the launcher while the numerical interpreter
            # survives terminate(). Reproduce the redirector's environment and
            # directly own its base interpreter, preserving this exact venv.
            executable = sys._base_executable
            environment["__PYVENV_LAUNCHER__"] = sys.executable
        self._responses = Queue()
        self._process = subprocess.Popen(
            [executable, "-m", "temsim.test_electron_execution", "--child", self._identity],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=environment, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            bufsize=0,
        )
        self._reader = Thread(target=_read_responses, args=(self._process.stdout, self._responses),
                              name="electron-result-reader", daemon=True)
        self._reader.start()

    def _stop_process(self):
        with self._process_lock:
            process, self._process = self._process, None
            self._identity = None
            self._scene_token = None
        if process is not None:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=.5)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()

    def close(self):
        """Permanently close this owner; no orphan worker survives app shutdown."""
        self._closed.set()
        self._stop_process()

    def _request(self, command, *, cancelled=None, progress=None):
        from temsim.cpu_resources import NumericalJobCancelled, numerical_job

        callback_error = None

        def is_cancelled():
            return (callback_error is not None or self._closed.is_set()
                    or (cancelled is not None and cancelled()))

        try:
            with numerical_job(1, cancelled=is_cancelled):
                # Existing GUI coordinator is serial; this lock also protects
                # direct callers from crossing messages between separate jobs.
                with self._lock:
                    if is_cancelled():
                        raise ElectronExecutionCancelled("Diagnostic electron request cancelled")
                    self._start()
                    process = self._process
                    if process is None:
                        raise ElectronExecutionCancelled("Diagnostic electron execution was closed")
                    self._sequence += 1
                    sequence = self._sequence
                    try:
                        _send(process.stdin, (sequence, command))
                    except (OSError, ValueError) as exc:
                        self._stop_process()
                        raise ElectronExecutionError("Could not send work to the diagnostic electron process") from exc
                    sent_cancel = False
                    while True:
                        if is_cancelled() and not sent_cancel:
                            # Field preparation includes non-interruptible native
                            # solvers. Stop that process rather than waiting with
                            # an obsolete scene; accepted scenes are never lost
                            # when cancelling an ordinary electron trajectory.
                            if command[0] in ("prepare", "install") or self._closed.is_set():
                                self._stop_process()
                                raise ElectronExecutionCancelled("Diagnostic electron request cancelled")
                            try:
                                _send(process.stdin, (sequence, ("cancel",)))
                            except (OSError, ValueError):
                                pass
                            sent_cancel = True
                        try:
                            response = self._responses.get(timeout=.025)
                        except Empty:
                            continue
                        if response[0] == "disconnected":
                            self._stop_process()
                            if is_cancelled():
                                raise ElectronExecutionCancelled("Diagnostic electron request cancelled")
                            raise ElectronExecutionError(response[1])
                        response_sequence, kind, value = response
                        if response_sequence != sequence:
                            self._stop_process()
                            raise ElectronExecutionError("Diagnostic electron response identity does not match its request")
                        if kind == "progress":
                            # A cancelled request must consume its terminal
                            # response before releasing this serial owner. Its
                            # already queued prefixes never reach a later job.
                            if not sent_cancel and not is_cancelled():
                                if progress is None:
                                    self._stop_process()
                                    raise ElectronExecutionError("Unrequested diagnostic electron progress response")
                                try:
                                    progress(value)
                                except Exception as exc:
                                    # Drain the cancelled trace even if a client
                                    # rejects a prefix; prepared fields survive.
                                    callback_error = exc
                            continue
                        if kind not in ("ok", "cancelled", "error"):
                            self._stop_process()
                            raise ElectronExecutionError("Unknown diagnostic electron response kind")
                        if callback_error is not None:
                            raise ElectronExecutionError("Diagnostic electron progress consumer failed") from callback_error
                        if sent_cancel or is_cancelled() or kind == "cancelled":
                            raise ElectronExecutionCancelled("Diagnostic electron request cancelled")
                        if kind == "error":
                            raise ElectronExecutionError(value)
                        return value
        except NumericalJobCancelled as exc:
            raise ElectronExecutionCancelled(str(exc)) from exc

    def prepare(self, state, magnetic_scene, *, z_limits_mm=None, cancelled=None):
        """Prepare the existing full electric provider once in the child."""
        scene = self._request(("prepare", state, magnetic_scene, z_limits_mm), cancelled=cancelled)
        self._scene_token = scene.token
        return scene

    def start(self, *, cancelled=None):
        """Start and admit the child without preparing any physical fields."""
        return self._request(("ready",), cancelled=cancelled)

    def install_prepared(self, scene, *, cancelled=None):
        """Transfer an already prepared scene once, without rebuilding fields."""
        handle = self._request(("install", scene), cancelled=cancelled)
        self._scene_token = handle.token
        return handle

    def trace(self, scene, settings, *, cancelled=None, progress=None):
        """Execute full-precision transport, optionally reporting real prefixes.

        ``progress`` runs on this calling worker thread, with immutable
        TestElectronTrajectory arrays, ``reason='in_progress'`` and
        ``completed=False``. Only this method's returned terminal result may be
        cached as an accepted trajectory. Cancellation drains this request's
        terminal response before allowing another request to use the scene.
        """
        if not isinstance(scene, RemoteElectronScene):
            raise TypeError("Isolated electron execution requires its prepared scene handle")
        if scene.process_identity == self._identity and scene.token != self._scene_token:
            raise ElectronExecutionError("The requested prepared fields were replaced; capture the fields again")
        if not self.owns_scene(scene):
            raise ElectronExecutionError("The prepared diagnostic fields are unavailable; capture the fields again")
        def report_prefix(result):
            if getattr(result, "reason", None) != "in_progress" or getattr(result, "completed", True):
                raise ElectronExecutionError("Diagnostic progress must contain an unfinished executed prefix")
            progress(_freeze_result_arrays(result))

        result = self._request(("trace", scene.token, settings, progress is not None),
                               cancelled=cancelled, progress=report_prefix if progress is not None else None)
        if getattr(result, "reason", None) == "in_progress":
            raise ElectronExecutionError("Diagnostic execution returned an unfinished prefix as its final result")
        return _freeze_result_arrays(result)


def _scene_metadata(scene, token, identity):
    bounds = getattr(scene, "diagnostic_bounds_m", scene.bounds_m)
    return RemoteElectronScene(token, identity, tuple(scene.initial_position_m), float(scene.initial_energy_ev),
        float(scene.default_path_length_m), tuple(tuple(float(v) for v in row) for row in scene.bounds_m),
        tuple(tuple(float(v) for v in row) for row in bounds), tuple(scene.notes))


def _child_main(identity):
    # Resolve reducers/classes by the importable module name, never __main__.
    from temsim.test_electron_execution import _child_loop
    return _child_loop(identity)


def _child_loop(identity):
    output, input_stream = sys.stdout.buffer, sys.stdin.buffer
    sys.stdout = sys.stderr  # Incidental provider prints cannot corrupt framing.
    from temsim.cpu_resources import initialize_cpu_resources, numerical_job
    initialize_cpu_resources()
    # Initialize native numerical extensions before the IPC reader blocks on a
    # Windows pipe. This also separates one-time runtime startup from field work.
    with numerical_job(1):
        pass
    jobs = Queue()
    cancellations = {}
    cancellation_lock = Lock()

    def receive_jobs():
        try:
            while True:
                sequence, command = _receive(input_stream)
                with cancellation_lock:
                    if command[0] == "cancel":
                        event = cancellations.get(sequence)
                        if event is not None:
                            event.set()
                        continue
                    event = Event()
                    cancellations[sequence] = event
                jobs.put((sequence, command, event))
        except Exception:  # malformed/import-failed requests cannot strand jobs.get()
            with cancellation_lock:
                for event in cancellations.values():
                    event.set()
            jobs.put(None)

    Thread(target=receive_jobs, name="electron-command-reader", daemon=True).start()
    scene, token = None, None
    while (job := jobs.get()) is not None:
        sequence, command, event = job
        kind, value = "ok", None
        try:
            with numerical_job(1, cancelled=event.is_set) as receipt:
                if command[0] == "ready":
                    value = {"process_id": os.getpid(), "cpu": receipt.to_dict(),
                             "python_executable": sys.executable, "python_prefix": sys.prefix}
                elif command[0] in ("prepare", "install"):
                    if command[0] == "prepare":
                        from temsim.test_electron_scene import prepare_test_electron_scene
                        magnetic = command[2]
                        if magnetic is None:
                            from temsim.magnetic_field_scene import prepare_magnetic_scene
                            magnetic = prepare_magnetic_scene(command[1], z_limits_mm=command[3])
                        scene = prepare_test_electron_scene(command[1], magnetic, z_limits_mm=command[3])
                    else:
                        scene = command[1]
                    token = uuid4().hex
                    value = _scene_metadata(scene, token, identity)
                elif command[0] == "trace":
                    if scene is None or command[1] != token:
                        raise ElectronExecutionError("The requested prepared fields were replaced; capture the fields again")
                    from temsim.magnetic_test_particle import trace_test_electron

                    def report_prefix(result):
                        if not event.is_set():
                            _send(output, (sequence, "progress", result))

                    kwargs = {"cancelled": event.is_set}
                    if command[3]:
                        kwargs["progress"] = report_prefix
                    value = trace_test_electron(scene, command[2], **kwargs)
                else:
                    raise ElectronExecutionError("Unknown diagnostic electron worker command")
        except Exception as exc:
            kind, value = "error", f"{type(exc).__name__}: {exc}"
        if event.is_set():
            kind, value = "cancelled", None
        with cancellation_lock:
            cancellations.pop(sequence, None)
        try:
            _send(output, (sequence, kind, value))
        except (BrokenPipeError, OSError):
            return 0
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--child":
        raise SystemExit("This module is an internal diagnostic-electron worker")
    raise SystemExit(_child_main(sys.argv[2]))
