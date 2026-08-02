"""Public electron-gun interfaces shared by every gun family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class EmissionBundle:
    """Electron phase space emitted at a gun source surface."""

    x_m: np.ndarray
    y_m: np.ndarray
    tx_rad: np.ndarray
    ty_rad: np.ndarray
    energy_offset_ev: np.ndarray
    weight: np.ndarray
    ray_id: np.ndarray


@dataclass(frozen=True)
class GunExitBundle:
    """Electron phase space resolved at the gun exit plane."""

    x_m: np.ndarray
    y_m: np.ndarray
    tx_rad: np.ndarray
    ty_rad: np.ndarray
    energy_offset_ev: np.ndarray
    weight: np.ndarray
    ray_id: np.ndarray
    alive: np.ndarray


@dataclass(frozen=True)
class GunEqualTimeFront:
    """One equal-time electron front suitable for a future arc overlay."""

    time_s: float
    z_mm: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    tx_rad: np.ndarray
    ty_rad: np.ndarray
    alive: np.ndarray
    completed: np.ndarray


@dataclass(frozen=True)
class GunEqualTimeHistory:
    """Simultaneously emitted rays sampled at common laboratory times.

    ``z_mm``/``x_m``/``y_m`` are time-by-ray arrays. A completed ray is held at
    its boundary crossing and marked in ``completed``; a future wavefront view
    can omit completed rays or use the snapshots before that crossing.
    """

    time_s: np.ndarray
    z_mm: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    tx_rad: np.ndarray
    ty_rad: np.ndarray
    alive: np.ndarray
    completed: np.ndarray

    def sample(self, time_s: float) -> GunEqualTimeFront:
        """Linearly interpolate the simultaneous beam state at one time."""

        requested = float(time_s)
        if not np.isfinite(requested):
            raise ValueError("Equal-time front time must be finite")
        upper = int(np.searchsorted(self.time_s, requested, side="left"))
        if upper <= 0:
            lower = upper = 0
            fraction = 0.0
        elif upper >= self.time_s.size:
            lower = upper = self.time_s.size - 1
            fraction = 0.0
        else:
            lower = upper - 1
            duration = float(self.time_s[upper] - self.time_s[lower])
            fraction = 0.0 if duration <= 0.0 else (
                requested - float(self.time_s[lower])
            ) / duration

        def interpolate(values):
            values = np.asarray(values)
            if lower == upper:
                return values[lower].copy()
            return (
                values[lower]
                + fraction * (values[upper] - values[lower])
            )

        state_index = lower if fraction < 1.0 else upper
        return GunEqualTimeFront(
            time_s=float(np.clip(requested, self.time_s[0], self.time_s[-1])),
            z_mm=interpolate(self.z_mm),
            x_m=interpolate(self.x_m),
            y_m=interpolate(self.y_m),
            tx_rad=interpolate(self.tx_rad),
            ty_rad=interpolate(self.ty_rad),
            alive=np.asarray(self.alive[state_index], dtype=bool).copy(),
            completed=np.asarray(
                self.completed[state_index], dtype=bool
            ).copy(),
        )


@dataclass(frozen=True)
class GunPlaneArrival:
    """Per-ray crossing coordinates and arrival times at one important plane."""

    key: str
    name: str
    z_mm: float
    time_s: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    reached: np.ndarray
    transmitted: np.ndarray


@dataclass(frozen=True)
class GunTraceResult:
    """Complete gun history plus the beam handed to the column."""

    z_mm: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    tx_rad: np.ndarray
    ty_rad: np.ndarray
    exit_bundle: GunExitBundle
    blocked_z_mm: np.ndarray
    blocked_key: tuple[str, ...]
    emitted_current_a: float
    dpa_transmitted_current_a: float
    c1_transmitted_current_a: float
    monochromator_transmitted_current_a: float | None = None
    output_energy_fwhm_ev: float | None = None
    slit_dispersion_um_per_ev: float | None = None
    slit_x_m: np.ndarray | None = None
    slit_reached: np.ndarray | None = None
    equal_time_history: GunEqualTimeHistory | None = None
    plane_arrivals: tuple[GunPlaneArrival, ...] = ()

    def equal_time_front_at_plane(self, key: str) -> GunEqualTimeFront:
        """Sample the beam when the median ray reaches an important plane."""

        plane = next(
            (item for item in self.plane_arrivals if item.key == str(key)),
            None,
        )
        if plane is None:
            raise KeyError(f"Unknown gun timing plane: {key}")
        if self.equal_time_history is None:
            raise ValueError("Electron-gun equal-time history is unavailable")
        valid = plane.reached & np.isfinite(plane.time_s)
        if not np.any(valid):
            raise ValueError(f"No rays reached gun timing plane: {key}")
        reference_time = float(np.median(plane.time_s[valid]))
        return self.equal_time_history.sample(reference_time)


@runtime_checkable
class ElectronGunAssembly(Protocol):
    """Stable mounting interface used by the rest of the microscope."""

    type_key: str
    display_name: str

    @property
    def components(self):
        ...

    @property
    def exit_plane_z_mm(self) -> float:
        ...

    @property
    def nominal_exit_energy_ev(self) -> float:
        ...

    @property
    def emitted_current_a(self) -> float:
        ...

    @property
    def ray_count(self) -> int:
        ...

    @property
    def diagnostic_waist_region_mm(self) -> tuple[float, float]:
        ...

    def validate(self):
        ...

    def emit(self, count: int | None = None) -> EmissionBundle:
        ...

    def trace_to_exit(self, count: int | None = None) -> GunTraceResult:
        ...

    def draw_layout(self):
        ...

    def to_dict(self) -> dict:
        ...


_GUN_FACTORIES: dict[str, object] = {}


def register_electron_gun(type_key: str, factory):
    key = str(type_key).strip()
    if not key:
        raise ValueError("Electron-gun type key must not be empty.")
    _GUN_FACTORIES[key] = factory


def registered_electron_gun_types() -> tuple[str, ...]:
    return tuple(sorted(_GUN_FACTORIES))


def create_electron_gun(type_key: str = "cold_feg", data: dict | None = None):
    key = str(type_key)
    try:
        factory = _GUN_FACTORIES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown electron-gun type: {key}") from exc
    return factory(data)
