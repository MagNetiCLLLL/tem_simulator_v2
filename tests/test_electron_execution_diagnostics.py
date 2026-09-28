"""Filesystem/pipe evidence fixtures; no numerical solver or child worker."""
from collections import deque
import io
import json
from pathlib import Path
from queue import Queue

import pytest

from temsim.electron_execution_diagnostics import (
    ElectronRunDiagnostics, export_failure_report, format_failure_report, redact_local_paths,
)


def drain(log, data):
    reader = log.start_stderr_drain(io.BytesIO(data))
    reader.join(3.)
    assert not reader.is_alive()
    log.close()


def test_stderr_rotation_keeps_tail_and_has_per_file_byte_limits(tmp_path):
    log = ElectronRunDiagnostics(tmp_path, max_file_bytes=128, backups=2)
    assert log.last_report is None and log.last_report_path is None
    payload = b"field diagnostic line\n"*4096+b"FINAL_STDERR_MARKER\n"
    drain(log, payload)
    files = list(log.directory.glob("stderr.log*"))
    assert len(files) == 3
    assert all(path.stat().st_size <= 128 for path in files)
    assert "FINAL_STDERR_MARKER" in log.tail_text()
    assert b"FINAL_STDERR_MARKER" in log.log_path.read_bytes()
    assert log.stderr_bytes_read == len(payload)
    assert json.loads((log.directory/"run.json").read_text())["closed"]


def test_split_utf8_and_invalid_bytes_do_not_break_drain(tmp_path):
    encoded = "电场 \N{GREEK SMALL LETTER PHI}\n".encode()
    pieces = deque(bytes([item]) for item in encoded)
    pieces.extend((b"\xff\n", b"EOF marker", b""))
    class Chunks:
        def read(self, count):
            return pieces.popleft()
        def close(self):
            pass
    log = ElectronRunDiagnostics(tmp_path)
    reader = log.start_stderr_drain(Chunks())
    reader.join(3.)
    log.close()
    assert "电场 φ" in log.tail_text()
    assert "�" in log.tail_text()
    assert "EOF marker" in log.tail_text()


def test_oversized_line_keeps_bounded_end_and_explicit_omission(tmp_path):
    log = ElectronRunDiagnostics(tmp_path, max_file_bytes=1024)
    payload = b"x"*(512*1024)+b"TERMINAL_MARKER\n"
    drain(log, payload)
    assert "TERMINAL_MARKER" in log.tail_text()
    assert "omitted" in log.tail_text()
    assert len(log.tail_text().encode()) <= 32*1024


def test_disk_failure_does_not_stop_pipe_drain_or_lose_memory_tail(tmp_path, monkeypatch):
    log = ElectronRunDiagnostics(tmp_path)
    original = Path.open
    def failed_open(path, *args, **kwargs):
        if path.name == "stderr.log":
            raise OSError("injected log disk failure")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", failed_open)
    payload = b"line\n"*50000+b"STILL_DRAINED\n"
    drain(log, payload)
    assert log.stderr_bytes_read == len(payload)
    assert "STILL_DRAINED" in log.tail_text()
    assert "injected log disk failure" in log.log_error
    report = log.record_failure(message="child died", exit_code=29)
    assert report["exit_code"] == 29
    assert report["stderr_drain_complete"]
    assert "injected log disk failure" in report["log_error"]


def test_close_never_disables_live_drain_before_eof(tmp_path):
    queue = Queue()
    class Pipe:
        def read(self, count):
            return queue.get(timeout=3.)
        def close(self):
            pass
    log = ElectronRunDiagnostics(tmp_path)
    reader = log.start_stderr_drain(Pipe())
    log.close(wait_s=0.)
    assert reader.is_alive()
    assert not json.loads((log.directory/"run.json").read_text())["closed"]
    queue.put(b"AFTER_CLOSE_REQUEST\n")
    queue.put(b"")
    reader.join(3.)
    assert not reader.is_alive()
    assert "AFTER_CLOSE_REQUEST" in log.tail_text()
    assert json.loads((log.directory/"run.json").read_text())["closed"]


def test_report_budget_and_failure_count_are_bounded(tmp_path):
    log = ElectronRunDiagnostics(tmp_path, max_reports=2, max_report_bytes=8192)
    for index in range(6):
        report = log.record_failure(sequence=index, exception_type="InjectedError", message="error"*5000,
                                    traceback_text="trace frame\n"*20000)
    assert len(list(log.directory.glob("failure-*.json"))) == 2
    assert all(path.stat().st_size <= 8192 for path in log.directory.glob("failure-*.json"))
    assert report["truncation"]["traceback"]
    assert "truncated" in report["traceback"]
    assert log.last_report_path == Path(report["report_path"])
    assert json.loads(log.last_report_path.read_text())["sequence"] == 5
    log.close()


def test_original_traceback_paths_remain_local_but_exports_are_sanitized(tmp_path):
    log = ElectronRunDiagnostics(tmp_path)
    full_trace = 'Traceback (most recent call last):\n  File "C:\\Users\\Alice\\work\\solver.py", line 12\nValueError: invalid field\n'
    report = log.record_failure(sequence=9, stage="trace", exception_type="ValueError", message="invalid field",
        traceback_text=full_trace, exit_code=29, fields_lost=True, physical_identity="physical123",
        numerical_identity="numerical456", execution_identity="execution789", parameter_revision=7,
        cache_directory="/home/alice/.cache/private-cache",
        settings={"position_m": [0., 0., 0.], "kinetic_energy_ev": .3, "environ": {"PASSWORD": "forbidden"}},
        request_metadata={"generation": 2, "worker_pid": 789, "env": {"TOKEN": "forbidden"}})
    assert report["traceback"] == full_trace
    assert json.loads(format_failure_report(report))["traceback"] == full_trace
    local_bytes = log.last_report_path.read_bytes()
    destination = export_failure_report(tmp_path/"export.json", report)
    text = destination.read_text()
    assert "Alice" not in text and "alice" not in text and "forbidden" not in text
    exported = json.loads(text)
    assert exported["physical_identity"] == "physical123"
    assert exported["exit_code"] == 29 and exported["fields_lost"]
    assert "solver.py" in exported["traceback"]
    assert "<LOCAL_PATH>" in exported["traceback"]
    assert log.last_report_path.read_bytes() == local_bytes
    assert exported["settings"] == {"position_m": [0., 0., 0.], "kinetic_energy_ev": .3}
    assert exported["request_metadata"] == {"generation": 2, "worker_pid": 789}
    log.close()


@pytest.mark.parametrize("value", ['File "C:\\Users\\Alice\\project\\solver.py", line 3',
    "cache=C:\\Users\\Alice\\OneDrive - University\\cache", "/home/alice/work/solver.py",
    'File "\\\\server\\private\\Alice\\solver.py"', "'/Users/alice/My Project/solver.py'"])
def test_export_path_redaction_does_not_retain_personal_directories(value):
    text = redact_local_paths(value)
    assert "Alice" not in text and "alice" not in text
    assert "server" not in text and "University" not in text
    assert "<LOCAL_PATH>" in text


def test_known_credentials_and_unnecessary_arrays_are_not_logged(tmp_path):
    class NeverStringify:
        def __str__(self):
            pytest.fail("Raw payload/model must not be stringified")
    log = ElectronRunDiagnostics(tmp_path)
    drain(log, b'api_key="TOP_SECRET_VALUE"\nAuthorization: Bearer ABCDEFGHIJKLMN\n')
    report = log.record_failure(message="password='SECRET WITH SPACE'",
        settings={"position_m": NeverStringify(), "kinetic_energy_ev": .3, "raw_array": NeverStringify()},
        request_metadata={"generation": 1, "payload": NeverStringify(), "env": {"SECRET": "nope"}})
    raw = log.log_path.read_bytes()+log.last_report_path.read_bytes()
    for secret in (b"TOP_SECRET_VALUE", b"ABCDEFGHIJKLMN", b"SECRET WITH SPACE", b"nope"):
        assert secret not in raw
    assert report["settings"] == {"kinetic_energy_ev": .3}
    assert report["request_metadata"] == {"generation": 1}


def test_retention_preserves_active_and_foreign_directories(tmp_path):
    first = ElectronRunDiagnostics(tmp_path, run_id="first", max_runs=1)
    first.record_failure(message="first")
    first.close()
    foreign = tmp_path/"run-user-not-a-log"
    foreign.mkdir()
    (foreign/"data.txt").write_text("user data")
    active = ElectronRunDiagnostics(tmp_path, run_id="active", max_runs=1)
    newest = ElectronRunDiagnostics(tmp_path, run_id="newest", max_runs=1)
    assert not first.directory.exists()
    assert active.directory.exists() and newest.directory.exists()
    assert (foreign/"data.txt").read_text() == "user data"
    active.close()
    newest.close()
    assert not active.directory.exists()
    assert newest.directory.exists() and foreign.exists()


def test_retired_worker_error_can_export_after_local_retention(tmp_path):
    first = ElectronRunDiagnostics(tmp_path, run_id="old", max_runs=1)
    captured = first.record_failure(message="old run failed", fields_lost=True)
    first.close()
    new = ElectronRunDiagnostics(tmp_path, run_id="new", max_runs=1)
    assert not first.directory.exists()
    exported = export_failure_report(tmp_path/"old-error.json", captured)
    assert json.loads(exported.read_text())["run_id"] == "old"
    new.close()


@pytest.mark.parametrize("identifier", ["../escape", "absolute/path", "A"*81, "", "C:\\somewhere"])
def test_run_identifier_cannot_escape_diagnostics_root(tmp_path, identifier):
    with pytest.raises(ValueError, match="identifier"):
        ElectronRunDiagnostics(tmp_path, run_id=identifier)
