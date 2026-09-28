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
from dataclasses import asdict, dataclass, fields, is_dataclass
from contextlib import nullcontext
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
import math
from time import monotonic, perf_counter
import traceback


class ElectronExecutionError(RuntimeError):
    """The isolated worker failed; no trajectory was accepted."""

    def __init__(self, message, *, diagnostic=None, fields_lost=False):
        super().__init__(message)
        self.diagnostic = diagnostic
        self.fields_lost = bool(fields_lost)


class ElectronExecutionCancelled(InterruptedError):
    """Cancellation includes CPU admission, preparation and transport."""

    def __init__(self, message, *, diagnostic=None, fields_lost=False):
        super().__init__(message)
        self.diagnostic = diagnostic
        self.fields_lost = bool(fields_lost)


@dataclass(frozen=True)
class ElectronExecutionPolicy:
    """Wall-clock budgets are independent of trajectory publication cadence."""

    startup_timeout_s: float = 60.
    prepare_timeout_s: float = 1800.
    trace_timeout_s: float = 600.
    send_timeout_s: float = 60.
    cancel_grace_s: float = 2.
    terminate_grace_s: float = .5
    poll_interval_s: float = .025

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0.:
                raise ValueError(f"{field.name} must be finite positive seconds")

    def timeout_for(self, stage):
        if stage in ("prepare", "install"):
            return self.prepare_timeout_s
        return self.trace_timeout_s if stage == "trace" else self.startup_timeout_s


class _PendingCommandWrite:
    """Keep a cancelled serializer inside numerical admission until it exits."""

    def __init__(self, thread):
        self.thread = thread
        self.pid = os.getpid()
        self.label = f"command writer in PID {self.pid}"

    def poll(self):
        return None if self.thread.is_alive() else 0


@dataclass(frozen=True)
class _ProfiledCommand:
    """Private opt-in envelope; the outer (sequence, command) stays unchanged."""

    command: tuple


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
    physical_identity: str | None = None
    numerical_identity: str | None = None
    transport_identity: str | None = None
    support_metadata: tuple = ()


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


def _write_payload(stream, data):
    for block in (struct.pack("!Q", len(data)), data):
        remaining = memoryview(block)
        while remaining:
            written = stream.write(remaining)
            if not written:
                raise BrokenPipeError("Diagnostic electron command pipe closed")
            remaining = remaining[written:]
    stream.flush()


def _send(stream, value, *, performance=None):
    with performance.stage("command_or_progress_serialization") if performance is not None else nullcontext():
        data = _dumps(value)
    if performance is not None:
        performance.increment("serialized_bytes", len(data))
    with performance.stage("command_or_progress_write") if performance is not None else nullcontext():
        _write_payload(stream, data)


def _read_exact(stream, size):
    parts = []
    while size:
        data = stream.read(size)
        if not data:
            raise EOFError("Diagnostic electron worker closed its output")
        parts.append(data)
        size -= len(data)
    return b"".join(parts)


def _receive(stream, *, performance_getter=None):
    size, = struct.unpack("!Q", _read_exact(stream, 8))
    # Internal framing guard; ordinary field meshes can legitimately be large.
    if size > 4*1024**3:
        raise ElectronExecutionError("Diagnostic worker response exceeds the 4 GiB message limit")
    performance = performance_getter() if performance_getter is not None else None
    # Excludes waiting for the next frame header. The payload interval measures
    # the actual receiving pipe, and may overlap the producer's write interval.
    with performance.stage("response_payload_read") if performance is not None else nullcontext():
        data = _read_exact(stream, size)
    if performance is not None:
        performance.increment("received_bytes", size)
    with performance.stage("response_deserialization") if performance is not None else nullcontext():
        return pickle.loads(data)


def _read_responses(stream, output, performance_getter=None):
    try:
        while True:
            output.put(_receive(stream, performance_getter=performance_getter))
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

    def __init__(self, *, policy=None, diagnostic_root=None, measure_performance=False):
        self.policy = policy or ElectronExecutionPolicy()
        self._diagnostic_root = diagnostic_root
        self._diagnostics = None
        self.last_diagnostic = None
        self.last_diagnostic_path = None
        self.measure_performance = bool(measure_performance)
        self.last_performance = None
        self._performance = None
        self._child_performance = None
        self._profile_lock = Lock()
        self._cleanup_error = None
        self._active_request = False
        self._request_context = {}
        self._remote_scene = None
        self._process = None
        self._identity = None
        self._scene_token = None
        self._responses = None
        self._reader = None
        self._command_write = None
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
        if self._cleanup_error and self._process is not None and self._process.poll() is None:
            raise ElectronExecutionError(self._cleanup_error, fields_lost=True)
        if self._process is not None and self._process.poll() is None:
            return
        self._stop_process()
        self._identity = uuid4().hex
        from temsim.electron_execution_diagnostics import ElectronRunDiagnostics
        self._diagnostics = ElectronRunDiagnostics(root=self._diagnostic_root, run_id=self._identity)
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
            self._child_command(executable, self._identity),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=environment, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            bufsize=0,
        )
        self._diagnostics.start_stderr_drain(self._process.stderr)
        self._reader = Thread(target=_read_responses, args=(self._process.stdout, self._responses,
                              lambda: self._performance),
                              name="electron-result-reader", daemon=True)
        self._reader.start()

    def _child_command(self, executable, identity):
        return [executable, "-m", "temsim.test_electron_execution", "--child", identity]

    def _stop_process(self):
        with self._process_lock:
            process = self._process
            self._identity = None
            self._scene_token = None
            self._remote_scene = None
            self._cleanup_error = None
            code = None
            if process is not None:
                for action_name in ("terminate", "kill"):
                    if process.poll() is not None:
                        break
                    try:
                        getattr(process, action_name)()
                        process.wait(timeout=self.policy.terminate_grace_s)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
                code = process.poll()
                if code is None:
                    # Keep ownership of an unreaped child; never create another
                    # numerical worker on top of it or claim successful cleanup.
                    self._cleanup_error = "Diagnostic process could not be stopped; restart is blocked while it remains alive"
                    from temsim.cpu_resources import retain_unreaped_numerical_process
                    retain_unreaped_numerical_process(process)
                    self._finish_command_write()
                    return None
                self._process = None
                for name in ("stdin", "stdout"):
                    stream = getattr(process, name, None)
                    if stream is not None:
                        try:
                            stream.close()
                        except OSError:
                            pass
            self._finish_command_write()
            if self._diagnostics is not None:
                self._diagnostics.close(wait_s=self.policy.terminate_grace_s)
            # The drain thread owns stderr until EOF, even when its bounded
            # join expires. Closing it here would discard the crash tail.
            if self._diagnostics is None and process is not None and getattr(process, "stderr", None) is not None:
                process.stderr.close()
            return code

    def _finish_command_write(self):
        pending = self._command_write
        if pending is None:
            return
        if pending.thread.ident is not None:
            pending.thread.join(timeout=self.policy.terminate_grace_s)
        if pending.poll() is None:
            # Stopping the child releases a pipe write, but cannot interrupt a
            # Python pickle operation. Retain that thread and block new work.
            from temsim.cpu_resources import retain_unreaped_numerical_process
            retain_unreaped_numerical_process(pending)
        else:
            self._command_write = None

    def close(self):
        """Permanently close this owner; no orphan worker survives app shutdown."""
        self._closed.set()
        self._stop_process()

    def restart(self):
        """Explicit retry discards the old handle; preparation starts a new child."""
        if self._closed.is_set():
            raise ElectronExecutionCancelled("Diagnostic electron execution was closed")
        if self._active_request:
            raise ElectronExecutionError("Wait for the current diagnostic request to finish before retrying")
        self._stop_process()
        if self._cleanup_error:
            raise ElectronExecutionError(self._cleanup_error, fields_lost=True)

    def _failure(self, message, *, reason, cancelled=False, fields_lost=False,
                 exception_type=None, traceback_text="", exit_code=None):
        if fields_lost:
            stopped = self._stop_process()
            exit_code = stopped if stopped is not None else exit_code
            if self._cleanup_error:
                message += "; " + self._cleanup_error
            if self._command_write is not None and self._command_write.poll() is None:
                message += "; command serialization is still finishing, so further numerical work is blocked"
            message += ". Prepared fields are unavailable; capture the fields again."
        values = {**self._request_context, "exception_type": exception_type or (
            "ElectronExecutionCancelled" if cancelled else "ElectronExecutionError"),
            "message": message, "traceback_text": traceback_text, "exit_code": exit_code,
            "reason": reason, "fields_lost": fields_lost}
        if self._diagnostics is not None:
            try:
                diagnostic = self._diagnostics.record_failure(**values)
            except (OSError, ValueError) as error:
                trace = values.pop("traceback_text", "")
                diagnostic = {**values, "traceback": trace, "log_error": str(error)}
        else:
            trace = values.pop("traceback_text", "")
            diagnostic = {**values, "traceback": trace}
        self.last_diagnostic = diagnostic
        self.last_diagnostic_path = diagnostic.get("report_path")
        error_type = ElectronExecutionCancelled if cancelled else ElectronExecutionError
        return error_type(message, diagnostic=diagnostic, fields_lost=fields_lost)

    def _send_bounded(self, process, payload, *, is_cancelled, cancellation_message=False, consumer_error=None):
        """Observe cancellation while a large command is serialized or written.

        The child owns unbuffered pipes; terminating it releases a blocked write.
        No second request may use this pipe before the writer has finished.
        """
        delivered = Queue(maxsize=1)
        def write():
            try:
                _send(process.stdin, payload, performance=self._performance)
                delivered.put(None)
            except Exception as error:
                delivered.put(error)
        writer = Thread(target=write, name="electron-command-writer", daemon=True)
        self._command_write = _PendingCommandWrite(writer)
        writer.start()
        limit = self.policy.cancel_grace_s if cancellation_message else self.policy.send_timeout_s
        deadline = monotonic() + limit
        trace_cancel_deadline = None
        while True:
            if not cancellation_message and is_cancelled():
                if self._request_context.get("stage") == "trace" and not self._closed.is_set():
                    # Rapid edits may cancel while this small command is being
                    # written. Finish its frame before sending the ordinary
                    # cancellation frame; never interleave two pipe writers.
                    # A stuck writer still loses the scene within cancel grace.
                    if trace_cancel_deadline is None:
                        trace_cancel_deadline = monotonic() + self.policy.cancel_grace_s
                        deadline = min(deadline, trace_cancel_deadline)
                else:
                    raise self._failure("Diagnostic request cancelled during transmission", reason="cancelled_during_send",
                                        cancelled=True, fields_lost=True)
            if monotonic() >= deadline:
                raise self._failure("Diagnostic process did not receive its command within the configured wait",
                                    reason="send_timeout", cancelled=(trace_cancel_deadline is not None or
                                        cancellation_message and consumer_error is None), fields_lost=True)
            try:
                error = delivered.get(timeout=self.policy.poll_interval_s)
            except Empty:
                if process.poll() is not None:
                    raise self._failure("Diagnostic process closed during transmission", reason="process_exit",
                                        fields_lost=True, exit_code=process.poll())
                continue
            if error is not None:
                raise self._failure("Could not send work to the diagnostic electron process", reason="send_error",
                                    fields_lost=True, exception_type=type(error).__name__,
                                    traceback_text="".join(traceback.format_exception(error))) from error
            self._finish_command_write()
            return

    def _request(self, command, *, cancelled=None, progress=None, request_metadata=None, scene_handle=None):
        if not self.measure_performance:
            self.last_performance = None
            return self._request_unmeasured(command, cancelled=cancelled, progress=progress,
                request_metadata=request_metadata, scene_handle=scene_handle)
        while not self._profile_lock.acquire(timeout=self.policy.poll_interval_s):
            if self._closed.is_set() or cancelled is not None and cancelled():
                raise ElectronExecutionCancelled("Diagnostic performance request cancelled before admission")
        try:
            return self._request_profiled(command, cancelled=cancelled, progress=progress,
                request_metadata=request_metadata, scene_handle=scene_handle)
        finally:
            self._profile_lock.release()

    def _request_profiled(self, command, *, cancelled=None, progress=None, request_metadata=None, scene_handle=None):
        from temsim.electron_execution_performance import ExecutionPerformance, ACCOUNTING
        # The public backend already serializes numerical requests. This
        # recorder observes that admission wait too, without admitting new work.
        recorder = ExecutionPerformance(self._sequence + 1, command[0], scope="parent")
        self._performance, self._child_performance = recorder, None
        failure = None
        try:
            return self._request_unmeasured(command, cancelled=cancelled, progress=progress,
                request_metadata=request_metadata, scene_handle=scene_handle)
        except BaseException as error:
            failure = {"exception_type": type(error).__name__, "message": str(error)}
            raise
        finally:
            parent = recorder.snapshot()
            child = self._child_performance or {"events": [], "stages": [], "counts": {},
                                               "actual_backend": "not_executed"}
            counts = dict(parent["counts"])
            for key, value in child["counts"].items():
                counts[key] = counts.get(key, 0) + value
            self.last_performance = {"request_kind": command[0], "sequence": recorder.sequence,
                "request_wall_s": max(0., perf_counter()-recorder.started),
                "parent_events": parent["events"], "child_events": child["events"],
                "counts": counts, "stages": parent["stages"] + child["stages"],
                "actual_backend": child["actual_backend"], "diagnostic": failure,
                "child_measurement_received": self._child_performance is not None,
                "field_solver_coverage": child.get("field_solver_coverage"), "accounting": ACCOUNTING}
            self._performance, self._child_performance = None, None

    def _request_unmeasured(self, command, *, cancelled=None, progress=None, request_metadata=None, scene_handle=None):
        from temsim.cpu_resources import NumericalJobCancelled, numerical_job
        from temsim.electron_execution_protocol import (
            ElectronProtocolValueError, validate_scene_metadata, validate_trajectory_payload,
        )

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
                    self._sequence += 1
                    sequence = self._sequence
                    stage = command[0]
                    # A failed preparation has not produced a new field. Do
                    # not attribute that request to the previously cached scene.
                    scene = ((scene_handle or self._remote_scene) if stage == "trace" else
                             command[1] if stage == "install" else None)
                    settings = command[2] if stage == "trace" and len(command) > 2 else None
                    from temsim.diagnostic_execution_identity import trajectory_execution_identity
                    self._request_context = dict(sequence=sequence, stage=stage, backend="isolated_cpu",
                        physical_identity=getattr(scene, "physical_identity", None),
                        numerical_identity=getattr(scene, "numerical_identity", None),
                        execution_identity=(trajectory_execution_identity(scene, settings,
                                            use_compiled=command[4] if len(command) > 4 else True)
                                            if settings is not None and scene is not None else None),
                        parameter_revision=(request_metadata or {}).get("parameter_revision"),
                        settings=asdict(settings) if is_dataclass(settings) else None,
                        cache_directory=os.environ.get("NUMBA_CACHE_DIR"), request_metadata=request_metadata)
                    self._active_request = True
                    try:
                        if scene_handle is not None:
                            if (scene_handle.process_identity == self._identity
                                    and scene_handle.token != self._scene_token):
                                raise self._failure("The requested prepared fields were replaced; capture the fields again",
                                                    reason="scene_mismatch")
                            if not self.owns_scene(scene_handle):
                                raise self._failure("The prepared diagnostic fields are unavailable; capture the fields again",
                                    reason="scene_unavailable", fields_lost=self.process_id is None)
                        process_boundary = "process_start" if self.process_id is None else "process_reuse"
                        with self._performance.stage(process_boundary) if self._performance is not None else nullcontext():
                            self._start()
                        process, responses = self._process, self._responses
                        if process is None:
                            raise ElectronExecutionCancelled("Diagnostic electron execution was closed", fields_lost=True)
                        if stage == "trace" and len(command) > 1 and command[1] != self._scene_token:
                            raise self._failure("The requested prepared fields were replaced; capture the fields again",
                                                reason="scene_mismatch")
                        if stage in ("prepare", "install"):
                            self._scene_token = None
                            self._remote_scene = None
                        outgoing = _ProfiledCommand(command) if self._performance is not None else command
                        self._send_bounded(process, (sequence, outgoing), is_cancelled=is_cancelled)
                        deadline = monotonic() + self.policy.timeout_for(stage)
                        cancel_deadline = None
                        while True:
                            if is_cancelled() and cancel_deadline is None:
                                if stage != "trace" or self._closed.is_set():
                                    raise self._failure("Diagnostic electron request cancelled", reason="cancelled_preparation",
                                                        cancelled=True, fields_lost=True)
                                self._send_bounded(process, (sequence, ("cancel",)), is_cancelled=is_cancelled,
                                                   cancellation_message=True, consumer_error=callback_error)
                                cancel_deadline = monotonic() + self.policy.cancel_grace_s
                            if cancel_deadline is not None and monotonic() >= cancel_deadline:
                                raise self._failure("Diagnostic process did not respond to cancellation", reason="cancel_timeout",
                                                    cancelled=callback_error is None, fields_lost=True)
                            if cancel_deadline is None and monotonic() >= deadline:
                                raise self._failure(f"Diagnostic {stage} exceeded its configured time budget",
                                                    reason="stage_timeout", fields_lost=True)
                            try:
                                response = responses.get(timeout=self.policy.poll_interval_s)
                            except Empty:
                                code = process.poll()
                                if code is not None:
                                    raise self._failure("Diagnostic electron worker closed its output",
                                                        reason="process_exit", fields_lost=True,
                                                        cancelled=is_cancelled(), exit_code=code)
                                continue
                            if (isinstance(response, tuple) and len(response) == 2
                                    and response[0] == "disconnected"):
                                if process.poll() is None:
                                    try:
                                        process.wait(timeout=min(.1, self.policy.terminate_grace_s))
                                    except subprocess.TimeoutExpired:
                                        pass
                                raise self._failure(str(response[1]), reason="disconnected", fields_lost=True,
                                                    cancelled=is_cancelled(), exit_code=process.poll())
                            if (not isinstance(response, tuple) or len(response) != 3
                                    or type(response[0]) is not int or not isinstance(response[1], str)):
                                raise self._failure("Malformed diagnostic electron response", reason="protocol_error", fields_lost=True)
                            response_sequence, kind, value = response
                            if response_sequence != sequence:
                                raise self._failure("Diagnostic electron response identity does not match its request",
                                                    reason="protocol_error", fields_lost=True)
                            if kind == "performance":
                                if self._performance is None or self._child_performance is not None:
                                    raise self._failure("Unrequested or duplicate diagnostic performance response",
                                                        reason="protocol_error", fields_lost=True)
                                from temsim.electron_execution_performance import validate_performance_payload
                                try:
                                    self._child_performance = validate_performance_payload(value)
                                except ValueError as exc:
                                    raise self._failure(str(exc), reason="protocol_error", fields_lost=True) from exc
                                continue
                            if kind == "progress":
                                if progress is None:
                                    raise self._failure("Unrequested diagnostic electron progress response",
                                                        reason="protocol_error", fields_lost=True)
                                if cancel_deadline is None and not is_cancelled():
                                    try:
                                        if stage == "trace" and settings is not None:
                                            validate_trajectory_payload(value, settings=settings, scene=scene,
                                                execution_identity=self._request_context["execution_identity"], progress=True)
                                        progress(value)
                                    except ElectronProtocolValueError as exc:
                                        raise self._failure(str(exc), reason="protocol_error", fields_lost=True,
                                                            exception_type=type(exc).__name__) from exc
                                    except Exception as exc:
                                        callback_error = exc
                                continue
                            if kind not in ("ok", "cancelled", "error"):
                                raise self._failure("Unknown diagnostic electron response kind", reason="protocol_error", fields_lost=True)
                            if callback_error is not None:
                                raise self._failure("Diagnostic electron progress consumer failed", reason="progress_consumer",
                                    exception_type=type(callback_error).__name__,
                                    traceback_text="".join(traceback.format_exception(callback_error))) from callback_error
                            if cancel_deadline is not None or is_cancelled() or kind == "cancelled":
                                raise ElectronExecutionCancelled("Diagnostic electron request cancelled")
                            if kind == "error":
                                details = value if isinstance(value, dict) else {"message": str(value)}
                                raise self._failure(str(details.get("message", "Diagnostic worker error")),
                                    reason="child_exception", exception_type=details.get("exception_type"),
                                    traceback_text=str(details.get("traceback", "")))
                            if self._performance is not None and self._child_performance is None:
                                raise self._failure("Diagnostic worker omitted requested performance observations",
                                                    reason="protocol_error", fields_lost=True)
                            with self._performance.stage("result_acceptance") if self._performance is not None else nullcontext():
                                if stage in ("prepare", "install"):
                                    validate_scene_metadata(value, process_identity=self._identity)
                                    self._scene_token = value.token
                                    self._remote_scene = value
                                elif stage == "trace" and settings is not None:
                                    validate_trajectory_payload(value, settings=settings, scene=scene,
                                        execution_identity=self._request_context["execution_identity"])
                            return value
                    except ElectronProtocolValueError as exc:
                        raise self._failure(str(exc), reason="protocol_error", fields_lost=True,
                                            exception_type=type(exc).__name__) from exc
                    except (ElectronExecutionError, ElectronExecutionCancelled):
                        raise
                    except Exception as exc:
                        raise self._failure(f"Diagnostic request failed: {exc}", reason="request_error", fields_lost=True,
                            exception_type=type(exc).__name__, traceback_text=traceback.format_exc()) from exc
                    finally:
                        self._active_request = False
        except NumericalJobCancelled as exc:
            raise ElectronExecutionCancelled(str(exc)) from exc

    def prepare(self, state, magnetic_scene, *, z_limits_mm=None, cancelled=None, request_metadata=None):
        """Prepare the existing full electric provider once in the child."""
        return self._request(("prepare", state, magnetic_scene, z_limits_mm), cancelled=cancelled,
                             request_metadata=request_metadata)

    def start(self, *, cancelled=None):
        """Start and admit the child without preparing any physical fields."""
        return self._request(("ready",), cancelled=cancelled)

    def install_prepared(self, scene, *, cancelled=None, request_metadata=None):
        """Transfer an already prepared scene once, without rebuilding fields."""
        return self._request(("install", scene), cancelled=cancelled, request_metadata=request_metadata)

    def trace(self, scene, settings, *, cancelled=None, progress=None, request_metadata=None, use_compiled=True):
        """Execute full-precision transport, optionally reporting real prefixes.

        ``progress`` runs on this calling worker thread, with immutable
        TestElectronTrajectory arrays, ``reason='in_progress'`` and
        ``completed=False``. Only this method's returned terminal result may be
        cached as an accepted trajectory. Cancellation drains this request's
        terminal response before allowing another request to use the scene.
        """
        if not isinstance(scene, RemoteElectronScene):
            raise TypeError("Isolated electron execution requires its prepared scene handle")
        def report_prefix(result):
            from temsim.magnetic_test_particle import TestElectronTrajectory
            if not isinstance(result, TestElectronTrajectory):
                raise ElectronExecutionError("Diagnostic progress has an invalid trajectory type")
            if getattr(result, "reason", None) != "in_progress" or getattr(result, "completed", True):
                raise ElectronExecutionError("Diagnostic progress must contain an unfinished executed prefix")
            progress(_freeze_result_arrays(result))

        command = ("trace", scene.token, settings, progress is not None)
        if not use_compiled:
            command += (False,)
        result = self._request(command,
                               cancelled=cancelled, progress=report_prefix if progress is not None else None,
                               request_metadata=request_metadata, scene_handle=scene)
        if getattr(result, "reason", None) == "in_progress":
            raise self._failure("Diagnostic execution returned an unfinished prefix as its final result",
                                reason="protocol_error", fields_lost=True)
        from temsim.magnetic_test_particle import TestElectronTrajectory
        if not isinstance(result, TestElectronTrajectory):
            raise self._failure("Diagnostic execution returned an invalid trajectory", reason="protocol_error", fields_lost=True)
        return _freeze_result_arrays(result)


def _scene_metadata(scene, token, identity):
    bounds = getattr(scene, "diagnostic_bounds_m", scene.bounds_m)
    return RemoteElectronScene(token, identity, tuple(scene.initial_position_m), float(scene.initial_energy_ev),
        float(scene.default_path_length_m), tuple(tuple(float(v) for v in row) for row in scene.bounds_m),
        tuple(tuple(float(v) for v in row) for row in bounds), tuple(scene.notes),
        physical_identity=getattr(scene, "physical_identity", None),
        numerical_identity=getattr(scene, "numerical_identity", None),
        transport_identity=getattr(scene, "transport_identity", None),
        support_metadata=getattr(scene, "support_metadata", ()))


def _child_main(identity):
    # Resolve reducers/classes by the importable module name, never __main__.
    from temsim.test_electron_execution import _child_loop
    return _child_loop(identity)


def _child_loop(identity):
    output, input_stream = sys.stdout.buffer, sys.stdin.buffer
    sys.stdout = sys.stderr  # Incidental provider prints cannot corrupt framing.
    import faulthandler
    faulthandler.enable(file=sys.stderr, all_threads=True)
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
                measured = isinstance(command, _ProfiledCommand)
                if measured:
                    command = command.command
                with cancellation_lock:
                    if command[0] == "cancel":
                        event = cancellations.get(sequence)
                        if event is not None:
                            event.set()
                        continue
                    event = Event()
                    cancellations[sequence] = event
                jobs.put((sequence, command, event, measured))
        except Exception:  # malformed/import-failed requests cannot strand jobs.get()
            traceback.print_exc(file=sys.stderr)
            with cancellation_lock:
                for event in cancellations.values():
                    event.set()
            jobs.put(None)

    Thread(target=receive_jobs, name="electron-command-reader", daemon=True).start()
    scene, token = None, None
    while (job := jobs.get()) is not None:
        sequence, command, event, measured = job
        kind, value = "ok", None
        performance = None
        if measured:
            from temsim.electron_execution_performance import ExecutionPerformance
            performance = ExecutionPerformance(sequence, command[0], scope="child")
        boundary = {"ready": "runtime_ready", "prepare": "field_prepare_or_load",
                    "install": "field_install", "trace": "particle_transport"}.get(command[0], "request")
        try:
            with (performance.observe() if performance is not None else nullcontext()), \
                    numerical_job(1, cancelled=event.is_set) as receipt, \
                    (performance.stage(boundary) if performance is not None else nullcontext()):
                if command[0] == "ready":
                    value = {"process_id": os.getpid(), "cpu": receipt.to_dict(),
                             "python_executable": sys.executable, "python_prefix": sys.prefix}
                elif command[0] in ("prepare", "install"):
                    scene, token = None, None
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
                            if performance is not None:
                                performance.increment("progress_messages")
                            _send(output, (sequence, "progress", result), performance=performance)

                    kwargs = {"cancelled": event.is_set}
                    if command[3]:
                        kwargs["progress"] = report_prefix
                    if len(command) > 4:
                        kwargs["use_compiled"] = command[4]
                    if performance is not None:
                        performance.increment("trajectory_executions")
                    value = trace_test_electron(scene, command[2], **kwargs)
                else:
                    raise ElectronExecutionError("Unknown diagnostic electron worker command")
        except Exception as exc:
            detail = traceback.format_exc()
            print(detail, file=sys.stderr, flush=True)
            kind, value = "error", {"exception_type": type(exc).__name__,
                                    "message": f"{type(exc).__name__}: {exc}", "traceback": detail}
        if event.is_set():
            kind, value = "cancelled", None
        with cancellation_lock:
            cancellations.pop(sequence, None)
        try:
            if performance is None:
                _send(output, (sequence, kind, value))
            else:
                with performance.stage("result_serialization"):
                    data = _dumps((sequence, kind, value))
                performance.increment("serialized_bytes", len(data))
                # These scalar observations precede the unchanged terminal
                # result. Its pipe read is measured by the parent; do not
                # invent a producer write duration before writing it.
                _send(output, (sequence, "performance", performance.snapshot()))
                _write_payload(output, data)
        except (BrokenPipeError, OSError):
            return 0
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--child":
        raise SystemExit("This module is an internal diagnostic-electron worker")
    raise SystemExit(_child_main(sys.argv[2]))
