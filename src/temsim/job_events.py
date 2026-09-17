"""Bounded scalar lifecycle evidence, independent of widgets and physical state."""
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock, get_ident
from time import monotonic

_CURRENT = ContextVar("temsim_job_trace", default=None)


class JobEvents:
    def __init__(self, maximum=4096):
        self._rows = deque(maxlen=maximum)
        self._lock = RLock()

    def record(self, event, *, metadata=None, timestamp=None, **details):
        row = dict(metadata or {})
        row.update(details)
        if any(value is not None and not isinstance(value, (str, int, float, bool)) for value in row.values()):
            raise TypeError("Job evidence accepts scalar metadata, never arrays or model objects")
        row.update(event=str(event), monotonic_s=monotonic() if timestamp is None else timestamp,
                   thread_id=get_ident())
        with self._lock:
            self._rows.append(row)

    def snapshot(self):
        with self._lock:
            return [dict(row) for row in self._rows]


@contextmanager
def traced_job(recorder, metadata):
    token = _CURRENT.set((recorder, metadata))
    try:
        yield
    finally:
        _CURRENT.reset(token)


def job_event(event, **details):
    current = _CURRENT.get()
    if current is not None:
        current[0].record(event, metadata=current[1], **details)


@contextmanager
def job_stage(name, *, backend="not_selected"):
    job_event("stage_entry", stage=name, backend=backend)
    try:
        yield
    except BaseException:
        job_event("stage_exit", stage=name, backend=backend, outcome="raised")
        raise
    else:
        job_event("stage_exit", stage=name, backend=backend, outcome="returned")
