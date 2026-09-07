"""Bounded process-local retention of immutable specimen-potential products.

This cache contains potentials, not incident/exit waves or detector signals.
Keys must describe every potential input. Concurrent requests for the same key
share one build; clear/reconfiguration cannot let an old build repopulate it.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from threading import RLock
from time import perf_counter

import numpy as np

from temsim.cache_memory import RetainedMemoryLedger, retained_memory_inventory


DEFAULT_PREPARED_SPECIMEN_CACHE_BUDGET_BYTES = 256 * 1024**2
DEFAULT_PREPARED_SPECIMEN_CACHE_ENTRIES = 8
PREPARED_SPECIMEN_CACHE_SCHEMA = "finite-lab-roi-potential-v1"


def content_identity(raw_path: str) -> tuple[str, str] | None:
    """Hash current bytes, never relying on timestamp/size for correctness."""
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    try:
        digest = sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return str(path.resolve()), digest.hexdigest()
    except OSError:
        # The original builder owns the actionable input error. This identity
        # never matches a previously readable input at the same path.
        return str(path.absolute()), "unavailable"


def exact_identity(payload: dict) -> str:
    def encode(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Unsupported prepared-specimen identity: {type(value).__name__}")

    encoded = json.dumps(
        {"schema": PREPARED_SPECIMEN_CACHE_SCHEMA, "inputs": payload},
        sort_keys=True, separators=(",", ":"), allow_nan=False, default=encode,
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def _array_fields(value):
    return (
        value.x_angstrom, value.y_angstrom,
        *value.potential_configurations_v_angstrom,
        value.mean_projected_potential_v_angstrom,
        value.slice_thicknesses_angstrom,
    )


def _snapshot(value, *, immutable_buffers: bool):
    """Detach mutable metadata and share repeated array fields only once."""
    arrays = {}
    for array in _array_fields(value):
        if array is None or id(array) in arrays:
            continue
        source = np.asarray(array)
        if source.dtype.hasobject:
            raise TypeError("Prepared specimen arrays must have numeric storage")
        if immutable_buffers:
            # bytes-backed storage cannot be made writeable via setflags or
            # through the returned array's base chain.
            stored = np.frombuffer(source.tobytes(order="C"), dtype=source.dtype)
            stored = stored.reshape(source.shape)
        else:
            # Oversized/disabled results are not retained. Avoid a second huge
            # potential allocation solely for a cache that cannot admit it.
            source.setflags(write=False)
            stored = source.view()
            stored.setflags(write=False)
        arrays[id(array)] = stored
    return replace(
        value,
        x_angstrom=arrays[id(value.x_angstrom)],
        y_angstrom=arrays[id(value.y_angstrom)],
        potential_configurations_v_angstrom=tuple(
            arrays[id(array)] for array in value.potential_configurations_v_angstrom
        ),
        mean_projected_potential_v_angstrom=arrays[id(value.mean_projected_potential_v_angstrom)],
        slice_thicknesses_angstrom=(
            arrays[id(value.slice_thicknesses_angstrom)]
            if value.slice_thicknesses_angstrom is not None else None
        ),
        metrics=deepcopy(value.metrics),
    )


class PreparedSpecimenCache:
    """Thread-safe exact LRU cache; retained bytes exclude active callers."""

    def __init__(self, *, budget_bytes=DEFAULT_PREPARED_SPECIMEN_CACHE_BUDGET_BYTES,
                 max_entries=DEFAULT_PREPARED_SPECIMEN_CACHE_ENTRIES):
        self._lock = RLock()
        self._entries = OrderedDict()
        self._inflight = {}
        self._memory = RetainedMemoryLedger()
        self._generation = 0
        self._hits = self._misses = self._builds = self._waits = 0
        self._failures = self._evictions = self._skipped = 0
        self.configure(budget_bytes=budget_bytes, max_entries=max_entries)

    def configure(self, *, budget_bytes=None, max_entries=None):
        for name, value, minimum in (
            ("budget_bytes", budget_bytes, 0), ("max_entries", max_entries, 1),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        with self._lock:
            if budget_bytes is not None:
                self._budget = int(budget_bytes)
            if max_entries is not None:
                self._limit = int(max_entries)
            self._generation += 1
            self._trim()

    def _trim(self):
        while self._entries and (
            len(self._entries) > self._limit or self._memory.total_bytes > self._budget
        ):
            key, _value = self._entries.popitem(last=False)
            self._memory.remove(key)
            self._evictions += 1

    def clear(self):
        with self._lock:
            self._generation += 1
            self._entries.clear()
            self._memory.clear()
            # Running callers still receive their result, but new requests do
            # not join a build begun before clear().
            self._inflight.clear()
            self._hits = self._misses = self._builds = self._waits = 0
            self._failures = self._evictions = self._skipped = 0

    def info(self):
        with self._lock:
            return {
                "budget_bytes": self._budget, "max_entries": self._limit,
                "used_bytes": self._memory.total_bytes, "entries": len(self._entries),
                "hits": self._hits, "misses": self._misses, "builds": self._builds,
                "waits": self._waits, "failures": self._failures,
                "evictions": self._evictions, "skipped": self._skipped,
            }

    def get_or_build(self, key, builder, *, still_valid=lambda: True):
        """Return (detached product, hit, build seconds, retained bytes)."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                self._hits += 1
                return replace(entry, metrics=deepcopy(entry.metrics)), True, 0.0, self._memory.total_bytes
            future = self._inflight.get(key) if self._budget and key is not None else None
            if future is not None:
                self._hits += 1
                self._waits += 1
                owner = False
            else:
                self._misses += 1
                self._builds += 1
                future = Future()
                if self._budget and key is not None:
                    self._inflight[key] = future
                owner = True
            generation = self._generation
        if not owner:
            value = future.result()
            return replace(value, metrics=deepcopy(value.metrics)), True, 0.0, self.info()["used_bytes"]

        started = perf_counter()
        try:
            value = builder()
            build_seconds = perf_counter() - started
            try:
                valid = key is not None and bool(still_valid())
            except (TypeError, ValueError, OverflowError, OSError):
                # A previously valid external input may have changed or been
                # removed while building. Keep this call's result unretained.
                valid = False
            # A temporarily unavailable atomistic builder can return a named
            # qualitative fallback. It is usable for this call, not a success
            # that should suppress retrying the requested backend later.
            if (value.metrics.get("atomistic_requested")
                    and value.metrics.get("atomistic_fallback_reason")
                    and not value.metrics.get("atomistic_applied")):
                valid = False
            estimate = sum(retained_memory_inventory(key, value).values())
            with self._lock:
                can_admit = valid and generation == self._generation and 0 < estimate <= self._budget
            try:
                value = _snapshot(value, immutable_buffers=can_admit)
            except MemoryError:
                if not can_admit:
                    raise
                # Cache ownership is optional. Do not fail a completed build
                # solely because its immutable retention copy cannot fit.
                can_admit = False
                value = _snapshot(value, immutable_buffers=False)
            if can_admit:
                inventory = retained_memory_inventory(key, value)
                with self._lock:
                    if generation == self._generation and sum(inventory.values()) <= self._budget:
                        self._entries[key] = value
                        self._memory.replace_inventory(key, inventory)
                        self._trim()
                    else:
                        self._skipped += 1
            else:
                with self._lock:
                    self._skipped += 1
            future.set_result(value)
        except BaseException as exc:
            with self._lock:
                self._failures += 1
            future.set_exception(exc)
            raise
        finally:
            with self._lock:
                if self._inflight.get(key) is future:
                    self._inflight.pop(key, None)
        return replace(value, metrics=deepcopy(value.metrics)), False, build_seconds, self.info()["used_bytes"]


_CACHE = PreparedSpecimenCache()


def configure_prepared_specimen_cache(*, budget_bytes=None, max_entries=None):
    _CACHE.configure(budget_bytes=budget_bytes, max_entries=max_entries)


def prepared_specimen_cache_info():
    return _CACHE.info()


def clear_prepared_specimen_cache():
    _CACHE.clear()


def cached_prepared_specimen(key, builder, *, still_valid=lambda: True):
    return _CACHE.get_or_build(key, builder, still_valid=still_valid)
