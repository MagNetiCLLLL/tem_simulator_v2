"""Scoped optical status messages; never part of physical state or cache keys."""
from contextlib import contextmanager
from contextvars import ContextVar

_CALLBACK = ContextVar("optical_transport_progress", default=None)


@contextmanager
def transport_progress(callback):
    token = _CALLBACK.set(callback)
    try:
        yield
    finally:
        _CALLBACK.reset(token)


def transport_progress_active():
    return _CALLBACK.get() is not None


def report_transport_progress(label):
    callback = _CALLBACK.get()
    if callback is not None:
        callback(label)
