"""Bounded cache preferences, independent of physical state and workspace layouts.

Budgets are retention limits, not preallocations or whole-process memory limits.
Only the managed RAM caches are counted against the hardware safety allowance.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import os
import sys
from typing import Protocol


MIB = 1024**2
GIB = 1024**3
SETTINGS_ROOT = "performance_cache/v1"
_DETECT = object()


class Settings(Protocol):
    def value(self, key: str, defaultValue=None): ...
    def setValue(self, key: str, value): ...
    def sync(self): ...


def detect_total_memory_bytes() -> int | None:
    """Read installed physical RAM without allocations or optional packages."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", wintypes.DWORD), ("load", wintypes.DWORD),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            total = int(status.total_physical)
        else:
            total = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
        return total if total >= 128 * MIB else None
    except (AttributeError, OSError, ValueError, OverflowError):
        return None


@dataclass(frozen=True)
class CachePreferences:
    high_cache_budget_bytes: int = 8 * GIB
    tuning_cache_budget_bytes: int = GIB
    ray_display_cache_budget_bytes: int = 512 * MIB
    disk_cache_budget_bytes: int = 16 * GIB
    high_cache_limit: int = 32
    tuning_cache_limit: int = 128

    @property
    def managed_ram_budget_bytes(self) -> int:
        return (self.high_cache_budget_bytes + self.tuning_cache_budget_bytes
                + self.ray_display_cache_budget_bytes)

    def controller_kwargs(self) -> dict[str, int]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "ray_display_cache_budget_bytes"
        }


def default_cache_preferences(total_memory_bytes: int | None = None) -> CachePreferences:
    """Hardware-conscious defaults; unknown hardware keeps bounded fixed defaults."""
    if total_memory_bytes is None:
        return CachePreferences()
    total = max(6 * MIB, int(total_memory_bytes))
    return CachePreferences(
        high_cache_budget_bytes=max(MIB, min(8 * GIB, total // 4)),
        tuning_cache_budget_bytes=max(MIB, min(GIB, total // 16)),
        ray_display_cache_budget_bytes=max(MIB, min(512 * MIB, total // 32)),
    )


def validate_cache_preferences(prefs: CachePreferences,
                               total_memory_bytes: int | None = None) -> None:
    """Reject invalid limits and RAM budgets above half the detected physical RAM."""
    if not isinstance(prefs, CachePreferences):
        raise ValueError("Expected cache preferences")
    for name in ("high_cache_budget_bytes", "tuning_cache_budget_bytes",
                 "ray_display_cache_budget_bytes", "disk_cache_budget_bytes"):
        value = getattr(prefs, name)
        maximum = (1024 if name == "disk_cache_budget_bytes" else 256) * GIB
        if type(value) is not int or not MIB <= value <= maximum:
            raise ValueError(f"{name}: cache limits must be positive, bounded byte counts")
    for name in ("high_cache_limit", "tuning_cache_limit"):
        value = getattr(prefs, name)
        if type(value) is not int or not 1 <= value <= 4096:
            raise ValueError(f"{name}: entry limit must be between 1 and 4096")
    if total_memory_bytes is not None and prefs.managed_ram_budget_bytes > total_memory_bytes // 2:
        raise ValueError("Managed RAM caches must total no more than half the detected physical RAM")


def _saved_integer(value) -> int:
    if type(value) is int:
        return value
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return int(value)
    raise ValueError("Invalid saved cache limit")


def load_cache_preferences(settings: Settings, total_memory_bytes=_DETECT) -> CachePreferences:
    """Corrupt or oversized saved preferences fall back to safe defaults as a group."""
    total = detect_total_memory_bytes() if total_memory_bytes is _DETECT else total_memory_bytes
    defaults = default_cache_preferences(total)
    try:
        preferences = CachePreferences(**{
            item.name: _saved_integer(settings.value(f"{SETTINGS_ROOT}/{item.name}", getattr(defaults, item.name)))
            for item in fields(defaults)
        })
        validate_cache_preferences(preferences, total)
        return preferences
    except (TypeError, ValueError, OverflowError):
        return defaults


def save_cache_preferences(settings: Settings, prefs: CachePreferences, total_memory_bytes=_DETECT) -> None:
    total = detect_total_memory_bytes() if total_memory_bytes is _DETECT else total_memory_bytes
    validate_cache_preferences(prefs, total)
    for item in fields(prefs):
        settings.setValue(f"{SETTINGS_ROOT}/{item.name}", getattr(prefs, item.name))
    settings.sync()
    status_method = getattr(settings, "status", None)
    if callable(status_method):
        status = status_method()
        if getattr(status, "value", status) != 0:
            raise OSError("Cache preferences could not be saved")
