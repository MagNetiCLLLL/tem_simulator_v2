"""Bounded evidence for diagnostic electron workers; no physics or recovery.

The stderr reader always drains the pipe, including after a disk error. Only
this module's closed/dead-owner diagnostic runs are eligible for retention;
compiler caches, numerical results and arbitrary directories are untouched.
Exports contain a small allowlisted report and redact local absolute paths.
"""
from __future__ import annotations

import codecs
from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from threading import Event, RLock, Thread
from uuid import uuid4

from temsim.job_events import job_event


SCHEMA = "electron-worker-diagnostics-v1"
_RETENTION_LOCK = RLock()
_SETTINGS = ("kinetic_energy_ev", "position_m", "polar_angle_deg", "azimuth_angle_deg",
             "max_path_length_m", "step_m", "max_steps", "relative_tolerance", "position_tolerance_m")
_REQUEST_KEYS = {"operation", "request_id", "record_id", "generation", "field_revision", "parameter_revision",
    "settings_revision", "source_revision", "input_digest", "worker_pid", "compiled_requested", "execution_mode",
    "cancellation_grace_s", "stage_timeout_s", "cache_mode", "process_identity", "scene_token",
    "execution_generation", "field_generation"}
_REPORT_KEYS = {"schema", "run_id", "utc", "sequence", "stage", "backend", "physical_identity", "numerical_identity",
    "execution_identity", "parameter_revision", "settings", "request_metadata", "exception_type", "message",
    "traceback", "exit_code", "reason", "stderr_tail", "log_path", "report_path", "log_error", "truncation",
    "fields_lost", "cache_directory", "stderr_bytes_read", "stderr_drain_complete"}
_CREDENTIAL = re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|passwd|secret|authorization)\b\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)")
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*")
_API_TOKEN = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")
_URL_AUTH = re.compile(r"(\b[a-zA-Z][a-zA-Z0-9+.-]*://)[^\s/@:]+:[^\s/@]+@")
_QUOTED_PATH = re.compile(r"([\"'])(?:[A-Za-z]:[\\/]|\\\\|/)[^\r\n\"']*\1")
_WINDOWS_PATH = re.compile(r"(?<![\w])[A-Za-z]:[\\/][^\r\n\"'<>|]*|\\\\[^\r\n\"'<>|]+")
_POSIX_PATH = re.compile(r"(?<![\w:/>])/(?:[^\s/\"'<>]+/)*[^\s\"'<>]*")


def default_diagnostics_root():
    local = str(os.environ.get("LOCALAPPDATA", "")).strip()
    return (Path(local) if local else Path.home()/".cache")/"TEM Simulator v2"/"electron_diagnostics"


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _credentials(text):
    text = _BEARER.sub("Bearer [REDACTED]", str(text))
    text = _CREDENTIAL.sub("[REDACTED]", text)
    text = _API_TOKEN.sub("<credential omitted>", text)
    return _URL_AUTH.sub(r"\1<credential omitted>@", text)


def _limit(text, size):
    data = str(text).encode("utf-8", errors="replace")
    if len(data) <= size:
        return data.decode("utf-8"), False
    # Keep both the exception heading and its terminal frames/message.
    marker = b"\n... [diagnostic text truncated by byte budget] ...\n"
    room = max(0, size-len(marker))
    return (data[:room//2].decode("utf-8", errors="ignore")+marker.decode()
            +data[-(room-room//2):].decode("utf-8", errors="ignore") if room else marker.decode()[:size]), True


def _scalar(value, maximum=512):
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else "<nonfinite>"
    if isinstance(value, str):
        return _limit(_credentials(value), maximum)[0]
    return None  # Never stringify an array, environment, model or arbitrary object.


def _settings(value):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        from temsim.magnetic_test_particle import TestElectronSettings
        if not isinstance(value, TestElectronSettings):
            return {}
        value = {name: getattr(value, name) for name in _SETTINGS}
    output = {}
    for name in _SETTINGS:
        if name not in value:
            continue
        item = value[name]
        if name == "position_m":
            if isinstance(item, (tuple, list)) and len(item) == 3 and all(type(v) in (int, float) for v in item):
                output[name] = [_scalar(v) for v in item]
        elif item is None or type(item) in (str, bool, int, float):
            output[name] = _scalar(item)
    return output


def _metadata(value):
    if not isinstance(value, Mapping):
        return {}
    return {name: _scalar(value[name]) for name in sorted(_REQUEST_KEYS & value.keys())
            if value[name] is None or type(value[name]) in (str, bool, int, float)}


def _path_label(value):
    name = value.strip("\"' ").replace("\\", "/").rsplit("/", 1)[-1]
    # Only diagnostic/code file names are retained; directory/user names are not.
    if re.fullmatch(r"[A-Za-z0-9_.-]+\.(?:py|pyc|pyd|dll|so|log|json|toml|npz|nbc|nbi)", name):
        return "<LOCAL_PATH>/"+name
    return "<LOCAL_PATH>"


def redact_local_paths(text):
    text = _credentials(text)
    text = _QUOTED_PATH.sub(lambda match: match.group(1)+_path_label(match.group())+match.group(1), text)
    text = _WINDOWS_PATH.sub(lambda match: _path_label(match.group()), text)
    return _POSIX_PATH.sub(lambda match: _path_label(match.group()), text)


def _report_copy(diagnostic, *, redact=False):
    if not isinstance(diagnostic, Mapping):
        raise TypeError("A captured diagnostic report dictionary is required")
    output = {}
    for key in _REPORT_KEYS & diagnostic.keys():
        value = diagnostic[key]
        if key == "settings":
            output[key] = _settings(value)
        elif key == "request_metadata":
            output[key] = _metadata(value)
        elif key == "truncation":
            output[key] = {str(name): bool(flag) for name, flag in value.items()
                           if name in ("message", "traceback", "stderr_tail", "request_metadata", "settings")} if isinstance(value, Mapping) else {}
        elif value is None or type(value) in (str, bool, int, float):
            output[key] = _scalar(value, 256*1024)
    def clean(value):
        if isinstance(value, str):
            return redact_local_paths(value) if redact else _credentials(value)
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value
    return clean(output)


def format_failure_report(diagnostic, *, redact=False):
    return json.dumps(_report_copy(diagnostic, redact=redact), indent=2, ensure_ascii=False, allow_nan=False)


def export_failure_report(destination, diagnostic):
    """Export the captured error, even if its worker has since been replaced."""
    path = Path(destination)
    text = format_failure_report(diagnostic, redact=True)
    temporary = path.with_name(path.name+"."+uuid4().hex+".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _owner_alive(pid):
    if type(pid) is not int or pid <= 0:
        return True
    if pid == os.getpid():
        return True
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() != 87  # Access denied is not proof of death.
        try:
            code = wintypes.DWORD()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


class ElectronRunDiagnostics:
    def __init__(self, root=None, run_id=None, *, max_file_bytes=256*1024, backups=2,
                 max_runs=12, total_bytes=16*1024**2, max_reports=8, max_report_bytes=256*1024):
        for name, value, minimum in (("max_file_bytes", max_file_bytes, 128), ("backups", backups, 0),
                ("max_runs", max_runs, 1), ("total_bytes", total_bytes, 1024),
                ("max_reports", max_reports, 1), ("max_report_bytes", max_report_bytes, 8192)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.root = Path(root) if root is not None else default_diagnostics_root()
        self.run_id = uuid4().hex if run_id is None else str(run_id)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", self.run_id):
            raise ValueError("Diagnostic run ID must be a safe local identifier")
        self.directory = self.root/("run-"+self.run_id)
        self.log_path = self.directory/"stderr.log"
        self.max_file_bytes, self.backups = max_file_bytes, backups
        self.max_runs, self.total_bytes = max_runs, total_bytes
        self.max_reports, self.max_report_bytes = max_reports, max_report_bytes
        self.last_report = self.last_report_path = None
        self.log_error = None
        self.stderr_bytes_read = 0
        self._tail = b""
        self._stream = self._reader = None
        self._lock = RLock()
        self._done = Event()
        self._closing = False
        self._eof = False
        self._reports = []
        self._counter = 0
        self._manifest = {"schema": SCHEMA, "run_id": self.run_id, "owner_pid": os.getpid(),
                          "created_utc": _utc(), "closed": False}
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            # Refuse collisions instead of appending one worker to another.
            self.directory.mkdir(exist_ok=False)
            self._write_manifest()
        except FileExistsError:
            raise
        except OSError as exc:
            self._disk_error(exc)
        self._prune()

    def _disk_error(self, error):
        self.log_error = _limit(_credentials(f"{type(error).__name__}: {error}"), 2048)[0]

    def _write_manifest(self):
        try:
            temporary = self.directory/"run.json.tmp"
            temporary.write_text(json.dumps(self._manifest, allow_nan=False), encoding="utf-8")
            temporary.replace(self.directory/"run.json")
        except OSError as exc:
            self._disk_error(exc)

    def _prune(self):
        with _RETENTION_LOCK:
            try:
                rows = []
                root = self.root.resolve()
                for folder in self.root.glob("run-*"):
                    if folder.is_symlink() or not folder.is_dir() or folder.resolve().parent != root:
                        continue
                    try:
                        marker = json.loads((folder/"run.json").read_text(encoding="utf-8"))
                        if marker.get("schema") != SCHEMA or folder.name != "run-"+str(marker.get("run_id")):
                            continue
                        files = list(folder.iterdir())
                        if any(not p.is_file() or p.is_symlink() or not re.fullmatch(r"(?:run\.json|stderr\.log(?:\.\d+)?|failure-\d+\.json)", p.name) for p in files):
                            continue
                        eligible = folder != self.directory and (bool(marker.get("closed")) or not _owner_alive(marker.get("owner_pid")))
                        rows.append((str(marker.get("created_utc", "")), folder, files, sum(p.stat().st_size for p in files), eligible))
                    except (OSError, ValueError, TypeError):
                        continue
                rows.sort(key=lambda item: item[0])
                count, size = len(rows), sum(item[3] for item in rows)
                for _, folder, files, byte_count, eligible in rows:
                    if count <= self.max_runs and size <= self.total_bytes:
                        break
                    if not eligible:
                        continue
                    # Only known files in an inspected direct child, never a
                    # recursive removal of user-supplied/cache directories.
                    for item in files:
                        item.unlink(missing_ok=True)
                    folder.rmdir()
                    count -= 1
                    size -= byte_count
            except OSError as exc:
                self._disk_error(exc)

    def _rotate(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        if self.backups:
            oldest = self.directory/f"stderr.log.{self.backups}"
            oldest.unlink(missing_ok=True)
            for index in range(self.backups-1, 0, -1):
                source = self.directory/f"stderr.log.{index}"
                if source.exists():
                    source.replace(self.directory/f"stderr.log.{index+1}")
            if self.log_path.exists():
                self.log_path.replace(self.directory/"stderr.log.1")
        else:
            self.log_path.unlink(missing_ok=True)

    def _write_stderr(self, text):
        data = _credentials(text).encode("utf-8", errors="replace")
        with self._lock:
            self._tail = (self._tail+data)[-32*1024:]
            try:
                while data:
                    if self._stream is None:
                        self._stream = self.log_path.open("ab")
                    room = self.max_file_bytes-self._stream.tell()
                    if room <= 0:
                        self._rotate()
                        continue
                    chunk, data = data[:room], data[room:]
                    self._stream.write(chunk)
                    self._stream.flush()
            except OSError as exc:
                self._disk_error(exc)
                if self._stream is not None:
                    try:
                        self._stream.close()
                    except OSError:
                        pass
                    self._stream = None

    def start_stderr_drain(self, binary_stream):
        with self._lock:
            if self._reader is not None or self._closing:
                raise RuntimeError("Each worker run owns exactly one stderr drain")
            self._reader = Thread(target=self._drain, args=(binary_stream,), name="electron-stderr-reader", daemon=True)
            self._reader.start()
            return self._reader

    def _drain(self, source):
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        pending, omitted, sensitive = "", 0, False
        def emit(line):
            if omitted:
                line = ("[oversized stderr line with credential text omitted]\n" if sensitive else
                        f"[stderr line: {omitted} leading characters omitted]\n"+line)
            self._write_stderr(line)
        try:
            while True:
                data = source.read(64*1024)
                if not data:
                    self._eof = True
                    pending += decoder.decode(b"", final=True)
                    if pending:
                        emit(pending)
                    break
                self.stderr_bytes_read += len(data)
                pending += decoder.decode(data)
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    if len(line) > 8192:
                        omitted += len(line)-8192
                        sensitive = sensitive or bool(_CREDENTIAL.search(line) or _BEARER.search(line) or _API_TOKEN.search(line) or _URL_AUTH.search(line))
                        line = line[-8192:]
                    emit(line+"\n")
                    omitted, sensitive = 0, False
                if len(pending) > 8192:
                    omitted += len(pending)-8192
                    sensitive = sensitive or bool(_CREDENTIAL.search(pending) or _BEARER.search(pending) or _API_TOKEN.search(pending) or _URL_AUTH.search(pending))
                    pending = pending[-8192:]
        except Exception as exc:
            self._disk_error(exc)
        finally:
            try:
                source.close()
            except OSError:
                pass
            with self._lock:
                if self._stream is not None:
                    try:
                        self._stream.close()
                    except OSError as exc:
                        self._disk_error(exc)
                    self._stream = None
                self._done.set()
                if self._closing:
                    self._finish()

    def tail_text(self):
        with self._lock:
            return self._tail.decode("utf-8", errors="replace")

    def record_failure(self, *, sequence=None, stage="unknown", backend="diagnostic electron worker",
            physical_identity=None, numerical_identity=None, execution_identity=None, parameter_revision=None,
            settings=None, cache_directory=None, request_metadata=None, exception_type="UnknownError", message="",
            traceback_text="", exit_code=None, reason="", fields_lost=False):
        with self._lock:
            self._counter += 1
            path = self.directory/f"failure-{self._counter:06d}.json"
            report = {"schema": SCHEMA, "run_id": self.run_id, "utc": _utc(),
                "sequence": _scalar(sequence), "stage": _scalar(stage), "backend": _scalar(backend),
                "physical_identity": _scalar(physical_identity), "numerical_identity": _scalar(numerical_identity),
                "execution_identity": _scalar(execution_identity), "parameter_revision": _scalar(parameter_revision),
                "settings": _settings(settings), "request_metadata": _metadata(request_metadata),
                "exception_type": _scalar(exception_type), "exit_code": _scalar(exit_code), "reason": _scalar(reason),
                "fields_lost": bool(fields_lost), "cache_directory": _scalar(str(cache_directory)) if cache_directory is not None else None,
                "log_path": str(self.log_path), "report_path": str(path), "log_error": self.log_error,
                "stderr_bytes_read": self.stderr_bytes_read, "stderr_drain_complete": self._eof, "truncation": {}}
            for key, value, budget in (("message", message, 32*1024), ("traceback", traceback_text, 192*1024),
                                       ("stderr_tail", self.tail_text(), 32*1024)):
                report[key], report["truncation"][key] = _limit(_credentials(value), budget)
            def encoded():
                return json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
            while len(encoded()) > self.max_report_bytes:
                key = max(("message", "traceback", "stderr_tail"), key=lambda name: len(report[name]))
                if len(report[key].encode("utf-8")) > 256:
                    report[key], _ = _limit(report[key], len(report[key].encode("utf-8"))//2)
                    report["truncation"][key] = True
                elif report["request_metadata"]:
                    report["request_metadata"] = {}
                    report["truncation"]["request_metadata"] = True
                elif report["settings"]:
                    report["settings"] = {}
                    report["truncation"]["settings"] = True
                else:
                    raise ValueError("Diagnostic report budget cannot fit its bounded metadata")
            try:
                path.write_bytes(encoded())
                self.last_report_path = path
                self._reports.append(path)
                while len(self._reports) > self.max_reports:
                    self._reports.pop(0).unlink(missing_ok=True)
            except OSError as exc:
                self._disk_error(exc)
                self.last_report_path = None
                report["report_path"] = None
                report["log_error"] = self.log_error
            self.last_report = json.loads(json.dumps(report, allow_nan=False))
            job_event("electron_worker_failure", run_id=self.run_id, stage=report["stage"],
                      sequence=report["sequence"], exception_type=report["exception_type"], fields_lost=bool(fields_lost))
            return json.loads(json.dumps(report, allow_nan=False))

    def _finish(self):
        self._manifest.update(closed=True, closed_utc=_utc(), stderr_eof=self._eof)
        self._write_manifest()
        self._prune()

    def close(self, wait_s=.5):
        if not math.isfinite(wait_s) or wait_s < 0:
            raise ValueError("Diagnostic drain wait must be finite and nonnegative")
        self._closing = True
        reader = self._reader
        if reader is not None:
            reader.join(wait_s)
        if reader is None or self._done.is_set():
            with self._lock:
                self._finish()
        # A live reader remains draining until EOF, never abandoned/disabled
        # just because the bounded close wait has expired.

    def export_report(self, destination, failure=None):
        report = self.last_report if failure is None else failure
        if report is None:
            raise ValueError("No captured worker failure is available")
        return export_failure_report(destination, report)
