"""Numerical execution budgets; no electron or downstream optical inputs."""
from dataclasses import dataclass
import os

GiB = 1024**3


def available_physical_memory():
    """Return (available, total) bytes, or None if the OS cannot report them."""
    if os.name == "nt":
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)]+[
                (name, ctypes.c_ulonglong) for name in
                ("total", "available", "total_page", "available_page", "total_virtual", "available_virtual", "extended")]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.available), int(status.total)
    else:
        try:
            from pathlib import Path
            fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            return int(fields["MemAvailable"].split()[0])*1024, int(fields["MemTotal"].split()[0])*1024
        except (OSError, KeyError, ValueError):
            pass
    return None


def check_available_memory(additional_bytes):
    """Conservative allocation preflight, not a reservation or total-RSS cap."""
    reported = available_physical_memory()
    if reported is None:
        return
    available, total = reported
    reserve = min(4*GiB, max(512*1024**2, int(total*.05)))
    if int(additional_bytes)+reserve > available:
        raise MemoryError(f"Wave allocation needs approximately {int(additional_bytes)} additional bytes; "
            f"available physical memory={available}, system reserve={reserve}. "
            "Committed segments remain reusable; free memory before resuming the same calculation.")


@dataclass(frozen=True)
class WaveExecutionOptions:
    segmented: bool = True
    segment_steps: int = 128
    cache_directory: str = ".temsim-wave-cache"
    maximum_ram_cache_bytes: int = 8*GiB
    maximum_disk_cache_bytes: int = 192*GiB

    def validate(self):
        if type(self.segmented) is not bool:
            raise ValueError("Segmented execution must be boolean")
        for name in ("segment_steps", "maximum_ram_cache_bytes", "maximum_disk_cache_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.cache_directory, str) or not self.cache_directory.strip():
            raise ValueError("Execution needs a local cache directory")
        return self


@dataclass(frozen=True)
class InelasticWaveNumerics:
    method: str = "trajectories"
    trajectories_per_mode: int = 32
    seed: int = 0

    def validate(self):
        if self.method not in ("zero_loss", "trajectories"):
            raise ValueError("Inelastic wave method must be zero_loss or trajectories")
        for name, limit in (("trajectories_per_mode", 1000000), ("seed", 2**63-1)):
            value = getattr(self, name)
            minimum = 1 if name == "trajectories_per_mode" else 0
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= limit:
                raise ValueError(f"Invalid inelastic {name}")
        return self


@dataclass(frozen=True)
class ScanWaveNumerics:
    stride: int = 1
    dwell_samples: int = 1
    maximum_positions: int = 4096
    frame_index: int = 0

    def validate(self):
        for name in ("stride", "dwell_samples", "maximum_positions", "frame_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "frame_index" else 1):
                raise ValueError(f"Invalid scan {name}")
        return self
