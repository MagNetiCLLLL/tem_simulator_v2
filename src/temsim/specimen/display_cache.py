"""Bounded, exact CIF display-data cache; independent of physics results.

Entries contain immutable atom positions, element numbers and bonds in nm.
No propagation result, detector signal or OpenGL object is stored here.
"""

from collections import OrderedDict
import hashlib
from pathlib import Path
from numbers import Integral
from threading import RLock
import sys

import numpy as np


_lock = RLock()
_entries: OrderedDict[tuple, tuple[tuple, int]] = OrderedDict()
_budget_bytes = 128 * 1024**2
_used_bytes = 0
_hits = 0
_misses = 0
_MAX_ENTRIES = 128


def configure_sample_display_cache(*, budget_bytes: int) -> None:
    """Set the RAM limit; zero clears and disables this display-only cache."""
    global _budget_bytes, _used_bytes
    if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, Integral) or budget_bytes < 0:
        raise ValueError("Sample display cache budget must be a non-negative integer.")
    budget = int(budget_bytes)
    with _lock:
        _budget_bytes = budget
        while _entries and (_used_bytes > budget or budget == 0):
            _key, (_value, size) = _entries.popitem(last=False)
            _used_bytes -= size


def sample_display_cache_info() -> dict[str, int]:
    """Return lightweight process-local counters and the retained-data budget."""
    with _lock:
        return dict(hits=_hits, misses=_misses, entries=len(_entries),
                    used_bytes=_used_bytes, budget_bytes=_budget_bytes)


def cif_display_fingerprint(path: Path) -> str:
    """Read the actual bytes, detecting even same-size/same-mtime replacement."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _entry_size(value, seen=None) -> int:
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        # Include backing bytes and view headers, with shared bases counted once.
        return sys.getsizeof(value) + (_entry_size(value.base, seen) if value.base is not None else 0)
    if isinstance(value, (tuple, list)):
        return sys.getsizeof(value) + sum(_entry_size(item, seen) for item in value)
    return sys.getsizeof(value)


def cached_cif_display(key: tuple, build) -> tuple:
    """Reuse exact immutable data; failed or oversize builds are never retained."""
    global _hits, _misses, _used_bytes
    with _lock:
        entry = _entries.get(key) if _budget_bytes else None
        if entry is not None:
            _entries.move_to_end(key)
            _hits += 1
            return entry[0]
        _misses += 1
    # Building outside the lock avoids blocking preference changes / readers.
    value = build()
    for item in value:
        if isinstance(item, np.ndarray):
            item.setflags(write=False)
    size = _entry_size(key) + _entry_size(value)
    with _lock:
        if _budget_bytes and size <= _budget_bytes:
            # Bytes-backed views cannot be made writeable again by a caller.
            # Only admitted entries pay this copy, not disabled/oversize ones.
            try:
                cached_value = tuple(
                    np.frombuffer(item.tobytes(order="C"), dtype=item.dtype).reshape(item.shape)
                    if isinstance(item, np.ndarray) else item
                    for item in value
                )
            except MemoryError:
                # Admission is optional: keep the successful read-only display
                # build rather than turning cache pressure into a CIF failure.
                # Builder errors above intentionally still propagate.
                return value
            value = cached_value
            size = _entry_size(key) + _entry_size(value)
            if size > _budget_bytes:
                return value
            old = _entries.pop(key, None)
            if old is not None:
                _used_bytes -= old[1]
            _entries[key] = (value, size)
            _used_bytes += size
            while _used_bytes > _budget_bytes or len(_entries) > _MAX_ENTRIES:
                _old_key, (_old_value, old_size) = _entries.popitem(last=False)
                _used_bytes -= old_size
    return value
