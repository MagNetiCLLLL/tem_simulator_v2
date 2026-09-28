"""Instrument magnetic inputs shared by particle transport and diagnostics.

An electron's initial conditions never alter this captured field.  Finite
deflector coils retain the configured signed integral and effective length;
their uniform interior is a model assumption, not a measured fringe profile.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import numpy as np

from temsim import input_io


@lru_cache(maxsize=1)
def _implementation_identity():
    return sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class ColumnDipoleField:
    key: str
    lower_m: float
    upper_m: float
    bx_t: float
    by_t: float
    event_z_mm: float
    event_dx_rad: float
    event_dy_rad: float
    reference_momentum: float
    captured_time_s: float | None = None

    def field_at_global_positions_t(self, points):
        points = np.asarray(points, dtype=float)
        result = np.zeros_like(points)
        active = (points[..., 2] >= self.lower_m) & (points[..., 2] <= self.upper_m)
        result[..., 0] = np.where(active, self.bx_t, 0.)
        result[..., 1] = np.where(active, self.by_t, 0.)
        return result


@input_io.using_state_inputs
def column_dipole_fields(state):
    """Freeze the actual scan/deflection commands once at instrument energy.

    The same records supply both the main beam's interval forces and the
    diagnostic Lorentz field. No query depends on a test electron's energy.
    """
    from temsim.physics.core import electron
    components = (*getattr(state, "stigmators", ()),
                  *getattr(state, "corrector_elements", ()),
                  *getattr(state, "deflectors", ()))
    charge, momentum, _ = electron(state) if components else (-1., 0., 0.)
    result, seen = [], set()
    for component in components:
        key = str(component.key)
        if key in seen or not bool(getattr(component, "enabled", False)):
            continue
        seen.add(key)
        captured_time = None
        if hasattr(component, "kick_events"):
            try:
                captured_time = float(getattr(state, "simulation_time_s", 0.))
                events = component.kick_events(time_s=captured_time)
            except TypeError:
                captured_time = None
                events = component.kick_events()
        elif all(hasattr(component, name) for name in
                 ("upper_z_mm", "lower_z_mm", "upper_x_mrad", "upper_y_mrad", "lower_x_mrad", "lower_y_mrad")):
            events = ((component.upper_z_mm, component.upper_x_mrad*1e-3, component.upper_y_mrad*1e-3),
                      (component.lower_z_mm, component.lower_x_mrad*1e-3, component.lower_y_mrad*1e-3))
        else:
            continue
        events = tuple(events)
        if not events:
            continue
        thickness = float(getattr(component, "effective_thickness_mm", getattr(component, "thickness_mm", 0.)))
        if not np.isfinite(thickness) or thickness <= 0.:
            raise ValueError(f"{key}: magnetic deflection requires a positive effective coil thickness")
        length = thickness*1e-3
        for index, (z, dx, dy) in enumerate(events):
            if not np.isfinite((z, dx, dy)).all():
                raise ValueError(f"{key}: deflection commands must be finite")
            result.append(ColumnDipoleField(
                f"{key}:{index}", (z-.5*thickness)*1e-3, (z+.5*thickness)*1e-3,
                momentum/charge*dy/length, -momentum/charge*dx/length,
                float(z), float(dx), float(dy), float(momentum), captured_time))
    return tuple(result)


@dataclass(frozen=True)
class InstrumentMagneticField:
    """Frozen full component sum, independent of any plotting/cutoff window.

    Analytic fields retain their declared finite axial support and their
    existing near-axis law. Diagnostics enforce its validity radius before
    querying. Production callers retain their own transport validity checks.
    Registered maps keep the provider's existing zero-outside-volume rule.
    """
    _sources: tuple
    physical_identity: str | None
    numerical_identity: str | None

    def identity_for_axial_range(self, lower_m, upper_m):
        """Bind exactly the possible contributors, including currently zero ones."""
        from temsim.diagnostic_field_identity import identity_digest
        if not np.isfinite((lower_m, upper_m)).all() or upper_m <= lower_m:
            raise ValueError("Magnetic dependency range must be finite and increasing")
        sources = tuple(source for source in self._sources
                        if source.bounds_m[0, 2] <= upper_m and source.bounds_m[1, 2] >= lower_m)
        identities = tuple(getattr(source.identity, "numerical_identity", None) for source in sources)
        if any(value is None for value in identities):
            return None
        return identity_digest("instrument-magnetic-axial-dependency-v1", {
            "range_m": (float(lower_m), float(upper_m)),
            "sources_in_sum_order": identities,
            "implementation": _implementation_identity(),
        })

    def field_at_global_positions_t(self, positions):
        points = np.asarray(positions, dtype=float)
        if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
            raise ValueError("Instrument magnetic positions must be finite XYZ metres")
        flat = points.reshape(-1, 3)
        total = np.zeros_like(flat)
        for source in self._sources:
            if source.known_zero:
                continue
            active = ((flat[:, 2] >= source.bounds_m[0, 2])
                      & (flat[:, 2] <= source.bounds_m[1, 2]))
            if not active.any():
                continue
            values = np.asarray(source.provider.field_at_global_positions_t(flat[active]), dtype=float)
            if values.shape != (int(active.sum()), 3) or not np.isfinite(values).all():
                raise ValueError(f"{source.key}: invalid captured instrument magnetic field")
            total[active] += values
        return total.reshape(points.shape)


@input_io.using_state_inputs
def capture_instrument_magnetic_field(state):
    """Use exactly the provider graph displayed and queried by diagnostics."""
    from temsim.magnetic_field_scene import prepare_magnetic_scene
    if hasattr(state, "sync_objective"):
        state.sync_objective()
    # Production is allowed to resolve configured FEM fields. The display
    # entry point remains read-only and only accepts an already solved cache.
    if getattr(state, "lens_field_map_descriptors", {}):
        from temsim.component_keys import CONDENSER_LENS_KEYS
        from temsim.physics.lens_field_provider import resolve_runtime_lens_field_provider
        for lens in state.lenses:
            if bool(getattr(lens, "enabled", True)):
                native = state.condenser_system[lens.key] if lens.key in CONDENSER_LENS_KEYS else lens
                resolve_runtime_lens_field_provider(state, lens.key, native)
    scene = prepare_magnetic_scene(state)
    from temsim.diagnostic_field_identity import identity_digest
    numerical = (identity_digest("instrument-magnetic-numerical-v1", {
        "source_graph": scene.numerical_identity, "implementation": _implementation_identity(),
    }) if scene.numerical_identity is not None else None)
    return InstrumentMagneticField(scene._sources, scene.physical_identity, numerical)


def events_overlapping_interval(state, events, lower_mm, upper_mm):
    """Include a finite coil even when its centre is outside this segment."""
    coils = column_dipole_fields(state)
    ranges = {}
    for coil in coils:
        event = (coil.event_z_mm, coil.event_dx_rad, coil.event_dy_rad)
        ranges.setdefault(event, []).append((coil.lower_m*1e3, coil.upper_m*1e3))
    result = []
    for event in events:
        event = tuple(map(float, event))
        supports = ranges.get(event)
        if supports:
            overlap = any(low < upper_mm and high > lower_mm for low, high in supports)
        else:
            overlap = lower_mm <= event[0] <= upper_mm
        if overlap:
            result.append(event)
    return tuple(result)


def active_column_events(state):
    """Commands for every captured finite column coil, retaining identity rows.

    Keep coincident coils separate: summing their centre-plane commands would
    lose their individual physical lengths and prevent finite-field matching.
    Gun fields are already continuous providers and have no column kick rows.
    """
    return tuple((coil.event_z_mm, coil.event_dx_rad, coil.event_dy_rad)
                 for coil in column_dipole_fields(state))


def _captured_gun_sources(state):
    """Same finite source records as the full graph, without resolving lenses."""
    from types import SimpleNamespace
    from temsim.magnetic_field_scene import _extra_sources
    gun = getattr(state, "electron_gun", None)
    if gun is None:
        return ()
    return tuple(_extra_sources(SimpleNamespace(electron_gun=gun), (-np.inf, np.inf), []))


def gun_magnetic_support_edges_mm(state):
    """Keep the gun fields when their physical support crosses the gun exit."""
    sources = _captured_gun_sources(state)
    edges = {float(v)*1e3 for source in sources for v in source.bounds_m[:, 2]}
    # Resolve both edges of each smooth coil, including the plateau boundaries.
    gun = getattr(state, "electron_gun", None)
    for name in ("deflector", "stigmator"):
        component = getattr(gun, name, None)
        if component is None:
            continue
        soft = float(component.soft_edge_mm)
        if name == "deflector":
            centers = (component.upper_center_from_tip_mm+component.field_center_offset_mm,
                       component.lower_center_from_tip_mm+component.field_center_offset_mm)
            half = .5*component.coil_length_mm
        else:
            centers = (component.optical_reference_from_tip_mm,)
            half = .5*component.effective_length_mm
        for center in centers:
            edges.update(center+offset for offset in (-half-soft, -half, half, half+soft))
    return tuple(sorted(edges))


def gun_paraxial_fields(state, z_mm):
    """Axial dipole and linear quadrupole coefficients of the same gun B.

    Gun coils and the crossed-field selector have uniform transverse B;
    the gun stigmator has an exactly linear transverse law. The conversion
    below evaluates those original providers, without fitting a second field.
    """
    z = np.asarray(z_mm, dtype=float)
    bx, by, kx, ky, kxy = (np.zeros_like(z) for _ in range(5))
    sources = _captured_gun_sources(state)
    if not sources or not z.size:
        return bx, by, kx, ky, kxy
    from temsim.physics.core import electron
    charge, momentum, _ = electron(state)
    scale = charge/momentum
    flat_z = z.reshape(-1)*1e-3
    outputs = [array.reshape(-1) for array in (bx, by, kx, ky, kxy)]
    radius = 1e-6
    for source in sources:
        if source.known_zero:
            continue
        active = (flat_z >= source.bounds_m[0, 2]) & (flat_z <= source.bounds_m[1, 2])
        if not active.any():
            continue
        points = np.zeros((int(active.sum()), 3))
        points[:, 2] = flat_z[active]
        axis = source.provider.field_at_global_positions_t(points)
        points[:, 0] = radius
        along_x = source.provider.field_at_global_positions_t(points)
        points[:, 0] = 0.
        points[:, 1] = radius
        along_y = source.provider.field_at_global_positions_t(points)
        if np.any(axis[:, 2] != 0.) or not np.isfinite((axis, along_x, along_y)).all():
            raise ValueError("Gun magnetic provider is outside the transverse dipole/quadrupole contract")
        outputs[0][active] += axis[:, 0]
        outputs[1][active] += axis[:, 1]
        outputs[2][active] += scale*(along_x[:, 1]-axis[:, 1])/radius
        outputs[3][active] -= scale*(along_y[:, 0]-axis[:, 0])/radius
        outputs[4][active] += scale*(along_y[:, 1]-axis[:, 1])/radius
    return bx, by, kx, ky, kxy
