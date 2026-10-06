"""One captured instrument electrostatic solution, independent of observation.

The grounded column liner and its assembled axial downstream end define the
cold-FEG solve domain. An installed curved filter owns the path beyond its
declared entrance handoff, not the recording module's layout envelope. A user
cutoff never changes the grid or downstream Dirichlet boundary.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import numpy as np
from temsim import input_io
from temsim.physics.grounded_tip_field import GroundedTipField


_GROUNDED_CONSTANT_METHODS = tuple((name, getattr(GroundedTipField, name)) for name in (
    "_interpolate", "potential_rise_v_at_global_positions", "field_at_global_positions_v_per_m"))


def instrument_electric_end_mm(state):
    """Return the fixed mechanical end inside the actual connected liner."""
    gun = state.electron_gun
    from temsim.physics.gun_field_environment import ensure_gun_field_environment
    ensure_gun_field_environment(gun)
    assembly = getattr(state, "_resolved_assembly", None)
    rows = getattr(gun, "_grounded_outlet_liner_segments", ())
    ends = [float(row["end_z_mm"] if isinstance(row, dict) else row.end_z_mm) for row in rows]
    if assembly is not None and hasattr(assembly, "exit_z_mm"):
        end = float(assembly.exit_z_mm)
        from temsim.component_keys import ENERGY_FILTER_ENTRANCE_APERTURE
        parts = {part.key: part for part in getattr(assembly, "parts", ())}
        interface = parts.get("energy_filter")
        entrance = parts.get(ENERGY_FILTER_ENTRANCE_APERTURE)
        if interface is not None or entrance is not None:
            if interface is None or entrance is None:
                raise ValueError("Instrument electric domain requires the complete Energy Filter entrance interface")
            handoff = float(interface.center_z_mm)
            if (not np.isfinite(handoff) or not np.isclose(
                    handoff, float(entrance.center_z_mm), rtol=0., atol=1e-9)
                    or handoff > end):
                raise ValueError("Energy Filter axial handoff must coincide with its resolved entrance aperture inside the assembly")
            # This installed mechanical coordinate is independent of runtime
            # recording choices and observation cutoffs. Never silently shrink
            # it to a truncated liner: missing coverage remains a hard error.
            end = handoff
    elif ends:
        # Standalone physical-input fixtures can carry the same resolved liner
        # without owning a State. This is still mechanical, never a view limit.
        end = max(ends)
    else:
        raise ValueError("Instrument electric field needs a resolved mechanical domain")
    if not np.isfinite(end) or end <= float(gun.exit_plane_z_mm):
        raise ValueError("Instrument electric domain must extend beyond the physical gun exit")
    if getattr(gun, "type_key", "") == "cold_feg" and (
            not ends or not np.isfinite(ends).all() or max(ends) < end-1e-10):
        raise ValueError("Instrument electric domain is not covered by the connected grounded liner")
    return end


def configure_instrument_electric_domain(gun, end_mm):
    """Configure a captured/scoped gun; do not solve or modify its electrodes."""
    if getattr(gun, "type_key", "") == "cold_feg" and getattr(gun.emitter, "surface_model", None) is None:
        from temsim.physics.gun_field_environment import ensure_gun_field_environment
        ensure_gun_field_environment(gun)
        gun._instrument_electric_end_mm = float(end_mm)


@dataclass(frozen=True)
class InstrumentElectricField:
    provider: object = field(repr=False)
    base_field: object = field(repr=False)
    gun_snapshot: object = field(repr=False)
    bounds_m: np.ndarray
    physical_identity: str | None
    numerical_identity: str | None
    request_identity: str | None
    notes: tuple[str, ...]

    def interpolate(self, positions):
        """Return potential rise in V and its electric field in V/m together."""
        points = np.asarray(positions, dtype=float)
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError("Instrument electric positions must be finite XYZ metres")
        if (np.any(points < self.bounds_m[0]) or np.any(points > self.bounds_m[1])
                or np.any(np.hypot(points[..., 0], points[..., 1]) > self.bounds_m[1, 0])):
            raise ValueError("Position is outside the fixed instrument electric domain")
        base = self.base_field
        interpolate = getattr(base, "interpolate", None)
        if callable(interpolate):
            potential, electric = interpolate(points)
        else:
            potential_query = getattr(base, "potential_rise_v_at_global_positions", None)
            if potential_query is None:
                potential_query = base.potential_v_at_global_positions
            potential = potential_query(points)
            electric = base.field_at_global_positions_v_per_m(points)
        wien = getattr(self.provider, "wien_field", None)
        if wien is not None:
            electric = electric+wien.field_at_global_positions_v_per_m(points)
            potential = potential+wien.potential_v_at_global_positions(points)
        return potential, electric

    def potential_rise_v_at_global_positions(self, positions):
        return self.interpolate(positions)[0]

    def field_at_global_positions_v_per_m(self, positions):
        return self.interpolate(positions)[1]

    def is_constant_on_interval(self, lower_mm, upper_mm):
        """Prove exact historical grounded continuation, never weak tail E."""
        base = self.base_field
        if (not np.isfinite((lower_mm, upper_mm)).all() or upper_mm < lower_mm
                or upper_mm*1e-3 > self.bounds_m[1, 2]):
            return False
        return (self.provider is base and type(base) is GroundedTipField
                and lower_mm*1e-3 >= float(base.z[-1])
                and all(getattr(getattr(base, name), "__func__", None) is original
                        for name, original in _GROUNDED_CONSTANT_METHODS))


@input_io.using_state_inputs
def capture_instrument_electric_field(state):
    """Capture physical inputs and reuse the canonical immutable field cache.

    Called by numerical workers. Reading a saved/partial result or changing a
    plotting window must not call this method from the UI thread.
    """
    original = getattr(state, "electron_gun", None)
    if original is None or not hasattr(type(original), "electric_field"):
        raise ValueError("This electron gun has no declared full electric-field provider")
    end_mm = instrument_electric_end_mm(state)
    # Reuse immutable scalar-field arrays; deepcopy would make them writable.
    memo = {id(state): state}
    for name in ("_closed_gun_field", "_continuous_gun_field", "_trace_cache"):
        value = getattr(original, name, None)
        if value is not None:
            memo[id(value)] = value
    gun = deepcopy(original, memo)
    # Field capture needs inputs, never the potentially multi-GB executed gun
    # population. Excluding it during copy avoids both time and memory spikes.
    if hasattr(gun, "_trace_cache"):
        gun._trace_cache = None
        gun._trace_cache_key = None
    configure_instrument_electric_domain(gun, end_mm)
    provider = gun.electric_field
    base = getattr(provider, "base_field", provider)
    r, z = getattr(base, "r", None), getattr(base, "z", None)
    from temsim.physics.grounded_tip_field import GroundedTipField
    from temsim.physics.closed_gun_field import ClosedGunField
    historical_grounded_exit = isinstance(base, GroundedTipField)
    if r is not None and z is not None:
        radius, lower, upper = float(r[-1]), float(z[0]), float(z[-1])
        if historical_grounded_exit:
            upper = end_mm*1e-3  # Existing provider explicitly defines its grounded continuation.
        elif upper < end_mm*1e-3-1e-13:
            raise ValueError("Electric solution does not cover the fixed instrument domain")
    else:
        radius = .5e-3*min(float(part.mechanical_clear_bore_diameter_mm) for part in gun.bore_components)
        lower, upper = 0., end_mm*1e-3
    if isinstance(base, ClosedGunField):
        lower = min(lower, float(gun.emitter.mechanical_center_from_tip_mm)*1e-3)
    bounds = np.array(((-radius, -radius, lower), (radius, radius, upper)))
    bounds.setflags(write=False)
    from temsim.diagnostic_field_identity import electric_field_identity
    identity = electric_field_identity(provider)
    notes = [f"Shared instrument electric domain ends at the fixed mechanical boundary {end_mm:.6g} mm; observation cutoffs do not alter its solved field."]
    if r is not None and z is not None:
        notes.append(f"Immutable scalar solution: {z[0]*1e3:.6g}–{z[-1]*1e3:.6g} mm; {len(r)*len(z):,} mesh nodes.")
    if historical_grounded_exit:
        notes.append("Historical surface-model field retains its declared grounded gun-exit boundary and zero-field continuation; its physical boundary model is unchanged.")
    return InstrumentElectricField(provider, base, gun, bounds, identity.physical_id,
                                   identity.numerical_id, identity.request_id, tuple(notes))
