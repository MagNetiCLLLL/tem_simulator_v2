"""Frozen full-field scene for one independent diagnostic electron.

The existing gun electric provider owns extraction, gun focusing and every
accelerator electrode. This adapter does not define an accelerated source and
does not transport the microscope's particle population. This deterministic
field trajectory excludes specimen and detector interactions. Physical stops
remain active.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import math

import numpy as np

from temsim import input_io
from temsim.magnetic_field_scene import MagneticSceneField, MagneticSourceRegion


def _point(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("Diagnostic electron position must be finite XYZ metres")
    return result


def _frozen(value):
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class _Bore:
    key: str
    lower_m: float
    upper_m: float
    inner_m: float
    outer_m: float = math.inf


def _quadratic_roots(a, b, c):
    if a == 0.:
        return () if b == 0. else (-c/b,)
    discriminant = b*b-4.*a*c
    if discriminant < 0.:
        return ()
    root = math.sqrt(discriminant)
    q = -.5*(b+math.copysign(root, b))
    return ((-b/(2.*a),) if q == 0. else (q/a, c/q))


def _bore_intercept(start, end, bore):
    """First entry into the same finite circular bore/shell used by hardware.

    Partition the chronological chord at axial faces and cylinder crossings.
    This handles backward, transverse and thin-shell crossings without sorting
    the electron history by Z or relying only on its final position.
    """
    if max(start[2], end[2]) < bore.lower_m or min(start[2], end[2]) > bore.upper_m:
        return None
    delta = end-start
    if bore.lower_m == bore.upper_m and delta[2] != 0.:
        fraction = (bore.lower_m-start[2])/delta[2]
        point = start+fraction*delta
        radius = math.hypot(point[0], point[1])
        return float(fraction) if 0. <= fraction <= 1. and bore.inner_m <= radius <= bore.outer_m else None
    cuts = [0., 1.]
    if delta[2] != 0.:
        cuts.extend((face-start[2])/delta[2] for face in (bore.lower_m, bore.upper_m))
    a = float(delta[:2]@delta[:2])
    b = 2.*float(start[:2]@delta[:2])
    radius_squared = float(start[:2]@start[:2])
    for radius in (bore.inner_m, bore.outer_m):
        if math.isfinite(radius):
            cuts.extend(_quadratic_roots(a, b, radius_squared-radius*radius))
    cuts = sorted(set(float(v) for v in cuts if 0. <= v <= 1.))
    for first, last in zip(cuts[:-1], cuts[1:]):
        point = start+.5*(first+last)*delta
        radial = math.hypot(point[0], point[1])
        if bore.lower_m <= point[2] <= bore.upper_m and bore.inner_m <= radial <= bore.outer_m:
            return first
    # A stop reached precisely at the end is a physical contact too.
    radial = math.hypot(end[0], end[1])
    if bore.lower_m <= end[2] <= bore.upper_m and bore.inner_m <= radial <= bore.outer_m:
        return 1.
    return None


def _active_apertures(state, gun):
    candidates = [getattr(gun, name, None) for name in ("dpa_aperture", "c1_aperture")]
    candidates.extend(getattr(state, "apertures", ()))
    pulser = getattr(state, "nanopulser", None)
    if pulser is not None and bool(getattr(pulser, "installed", False)):
        candidates.append(pulser.aperture)
    seen, result = set(), []
    for item in candidates:
        if item is None or item.key in seen:
            continue
        seen.add(item.key)
        if bool(getattr(item, "enabled", True)) and bool(getattr(item, "installed", True)):
            result.append(deepcopy(item, {id(state): state}))
    return tuple(result)


def _physical_bores(state, gun, electric_base):
    rows = []
    assembly = getattr(state, "_resolved_assembly", None)
    for segment in getattr(assembly, "vacuum_bore_segments", ()):
        rows.append(_Bore("column_wall", float(segment.start_z_mm)*1e-3,
                          float(segment.end_z_mm)*1e-3, float(segment.inner_diameter_mm)*.5e-3))
    for part in getattr(gun, "bore_components", ()):
        center = float(part.mechanical_center_from_tip_mm)*1e-3
        half = float(part.mechanical_length_mm)*.5e-3
        outer = float(getattr(part, "mechanical_outer_diameter_mm", math.inf))*.5e-3
        rows.append(_Bore(str(part.key), center-half, center+half,
                          float(part.mechanical_clear_bore_diameter_mm)*.5e-3, outer))
    pulser = getattr(state, "nanopulser", None)
    if pulser is not None and bool(getattr(pulser, "installed", False)):
        center = float(pulser.mechanical_center_from_tip_mm)*1e-3
        half = float(pulser.mechanical_length_mm)*.5e-3
        rows.append(_Bore(str(pulser.key), center-half, center+half,
                          float(pulser.mechanical_clear_bore_diameter_mm)*.5e-3,
                          float(pulser.mechanical_outer_diameter_mm)*.5e-3))
    # Actual annular electrodes can be thinner than the enclosing gun bodies.
    for row in getattr(electric_base, "request", {}).get("rings", ()):
        if isinstance(row, dict):
            values = (row["key"], row["start_m"], row["stop_m"], row["inner_m"], row["outer_m"])
        else:
            values = tuple(row[:5])
        rows.append(_Bore(str(values[0]), *(float(value) for value in values[1:])))
    for row in getattr(electric_base, "request", {}).get("grounded_liner", ()):
        rows.append(_Bore(str(row["key"]), float(row["start_m"]), float(row["stop_m"]), float(row["inner_m"])))
    for row in rows:
        if (not all(math.isfinite(v) for v in (row.lower_m, row.upper_m, row.inner_m))
                or row.upper_m < row.lower_m or row.inner_m <= 0. or row.outer_m <= row.inner_m):
            raise ValueError(f"{row.key}: invalid captured aperture or wall geometry")
    return tuple(rows)


def _prepare_electric_provider(state, stop_m):
    original = getattr(state, "electron_gun", None)
    if original is None or not hasattr(type(original), "electric_field"):
        raise ValueError("This electron gun has no declared full electric-field provider")
    # Field products are immutable shared caches. Clone physical inputs without
    # duplicating their potentially large numerical arrays or mutating state.
    memo = {id(state): state}
    for name in ("_closed_gun_field", "_continuous_gun_field"):
        cached = getattr(original, name, None)
        if cached is not None:
            memo[id(cached)] = cached
    gun = deepcopy(original, memo)
    preparation = []
    if getattr(gun, "type_key", "") == "cold_feg" and getattr(gun.emitter, "surface_model", None) is None:
        from temsim.physics.gun_field_environment import ensure_gun_field_environment
        ensure_gun_field_environment(gun)
        old = float(gun._gun_field_exit_extension_mm)
        extension = max(old, stop_m*1e3-float(gun.exit_plane_z_mm))
        gun._gun_field_exit_extension_mm = extension
        if float(gun.emitter.curvature_nm_inv) == 0.:
            from temsim.physics.closed_gun_field import closed_field_request, mesh_axes
            request = closed_field_request(gun)
        else:
            from temsim.physics.continuous_gun_field import continuous_field_request, mesh_axes
            request = continuous_field_request(gun)
        r, z = mesh_axes(request)  # Admission before the existing cached solve.
        preparation.append(f"Electric field domain {z[0]*1e3:.6g}–{z[-1]*1e3:.6g} mm; {len(r)*len(z):,} mesh nodes. Existing dependency-bound field cache reused or prepared once.")
    electric = gun.electric_field
    return gun, electric, tuple(preparation)


@dataclass(frozen=True)
class TestElectronScene:
    magnetic_scene: MagneticSceneField
    electric_provider: object
    electric_base: object
    diagnostic_bounds_m: np.ndarray
    electric_bounds_m: np.ndarray
    initial_position_m: tuple[float, float, float]
    initial_energy_ev: float
    default_path_length_m: float
    notes: tuple[str, ...]
    _apertures: tuple
    _bores: tuple[_Bore, ...]
    _electric_regions: tuple[MagneticSourceRegion, ...]
    _flat_cathode: bool = False
    _post_exit_ground: bool = False
    has_electric_field: bool = True
    _unsupported_stops: tuple[tuple[float, str], ...] = ()
    _bore_axial_bounds_m: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        # Only reject disjoint axial intervals here. Every remaining shell,
        # thin electrode and bore still uses the original exact interception.
        # Inclusive comparisons preserve contacts exactly on either face.
        bounds = _frozen([(bore.lower_m, bore.upper_m) for bore in self._bores]).reshape(-1, 2)
        object.__setattr__(self, "_bore_axial_bounds_m", bounds)

    @property
    def bounds_m(self):
        return self.diagnostic_bounds_m

    @property
    def source_regions(self):
        return (*self.magnetic_scene.source_regions, *self._electric_regions)

    @property
    def diagnostic_sampling_regions(self):
        return self.magnetic_scene.diagnostic_sampling_regions

    def diagnostic_position_is_valid(self, position):
        point = _point(position)
        if np.any(point < self.diagnostic_bounds_m[0]) or np.any(point > self.diagnostic_bounds_m[1]):
            return False
        radius = math.hypot(point[0], point[1])
        if radius > self.electric_bounds_m[1, 0]:
            return False
        if point[2] < self.electric_bounds_m[0, 2] and not self._flat_cathode:
            return False
        if point[2] > self.electric_bounds_m[1, 2] and not self._post_exit_ground:
            return False
        return self.magnetic_scene.diagnostic_position_is_valid(point)

    def diagnostic_fields_at_global_position(self, position):
        point = _point(position)
        if not self.diagnostic_position_is_valid(point):
            return None
        points = point[None, :]
        base = self.electric_base
        interpolate = getattr(base, "interpolate", None)
        if callable(interpolate):
            potential, electric = interpolate(points)
        else:
            potential_function = getattr(base, "potential_rise_v_at_global_positions", None)
            if potential_function is None:
                potential_function = base.potential_v_at_global_positions
            potential = potential_function(points)
            electric = base.field_at_global_positions_v_per_m(points)
        electric = np.asarray(electric, dtype=float)
        potential = np.asarray(potential, dtype=float)
        wien = getattr(self.electric_provider, "wien_field", None)
        if wien is not None:
            electric = electric + wien.field_at_global_positions_v_per_m(points)
            potential = potential + wien.potential_v_at_global_positions(points)
        magnetic = self.magnetic_scene.diagnostic_field_at_global_position_t(point)
        if magnetic is None:
            return None
        if (electric.shape != (1, 3) or potential.shape != (1,)
                or not np.isfinite(electric).all() or not np.isfinite(potential).all()):
            raise ValueError("Captured electric provider returned invalid field or potential")
        return magnetic, electric[0], float(potential[0])

    def diagnostic_field_at_global_position_t(self, position):
        return self.magnetic_scene.diagnostic_field_at_global_position_t(position)

    def diagnostic_spatial_step_m(self, position, direction, requested):
        """Local E-grid resolution; nanometre tip cells do not cap the column."""
        point, direction = _point(position), _point(direction)
        step = float(requested)
        base = self.electric_base
        r, z = getattr(base, "r", None), getattr(base, "z", None)
        if r is not None and z is not None and point[2] <= z[-1]:
            radius = math.hypot(point[0], point[1])
            ir = int(np.clip(np.searchsorted(r, radius, side="right")-1, 0, len(r)-2))
            iz = int(np.clip(np.searchsorted(z, point[2], side="right")-1, 0, len(z)-2))
            transverse = math.hypot(direction[0], direction[1])
            if transverse > 1e-14:
                step = min(step, .5*float(r[ir+1]-r[ir])/transverse)
                for face, neighbour in ((float(r[ir]), ir-1), (float(r[ir+1]), ir+1)):
                    if 0 <= neighbour < len(r)-1:
                        into_next = .5*float(r[neighbour+1]-r[neighbour])
                        step = min(step, (abs(radius-face)+into_next)/transverse)
            if abs(direction[2]) > 1e-14:
                following = min(len(z)-2, iz+1) if direction[2] > 0. else max(0, iz-1)
                step = min(step, .5*float(z[iz+1]-z[iz])/abs(direction[2]))
                face = float(z[iz+1] if direction[2] > 0. else z[iz])
                into_next = .5*float(z[following+1]-z[following])
                step = min(step, (abs(face-point[2])+into_next)/abs(direction[2]))
            # Do not pin an accelerating particle to the exact next cell face:
            # a shorter accepted time step would approach it asymptotically.
            # A lookahead reaches at most halfway into a finer adjacent cell;
            # its tiny spacing must not constrain the entire preceding cell.
            # Step-doubling then resolves field changes across the face.
        return step

    def _tip_intercept(self, start, end):
        if min(start[2], end[2]) >= 0.:
            return None
        delta = end-start
        if self._flat_cathode:
            if start[2] < 0.:
                return 0.
            return float(-start[2]/delta[2]) if delta[2] < 0. else None
        mask = getattr(self.electric_base, "tip_material_mask", None)
        if mask is None:
            return None
        surface = getattr(self.electric_base, "_metal_surface_z", None)
        geometry = getattr(self.electric_base, "geometry", None)
        if surface is None and geometry is not None:
            surface = getattr(geometry, "surface_z_m", None)
        # The same represented surface is used for a curved source launch and
        # return. The minimum catches enter-and-leave chords with vacuum ends.
        if surface is not None:
            from scipy.optimize import minimize_scalar
            def signed(t):
                p = start+t*delta
                value = np.asarray(surface(np.array([math.hypot(p[0], p[1])]))).reshape(-1)[0]
                return float(p[2]-value)
            result = minimize_scalar(signed, bounds=(0., 1.), method="bounded", options={"xatol": 1e-14})
            candidates = sorted((0., float(result.x), 1.))
        else:
            candidates = (0., .5, 1.)
        hit = next((t for t in candidates if bool(mask((start+t*delta)[None, :])[0])), None)
        if hit is None:
            return None
        low, high = 0., hit
        for _ in range(48):
            mid = .5*(low+high)
            if bool(mask((start+mid*delta)[None, :])[0]):
                high = mid
            else:
                low = mid
        return high

    def diagnostic_segment_stop(self, start_m, end_m):
        start, end = _point(start_m), _point(end_m)
        delta = end-start
        hits = []
        for plane, reason in self._unsupported_stops:
            if delta[2] > 0. and start[2] <= plane <= end[2]:
                hits.append((float((plane-start[2])/delta[2]), reason))
        tip = self._tip_intercept(start, end)
        if tip is not None:
            hits.append((tip, "tip_return"))
        lower_z, upper_z = min(start[2], end[2]), max(start[2], end[2])
        bounds = self._bore_axial_bounds_m
        overlapping = np.flatnonzero((bounds[:, 0] <= upper_z) & (bounds[:, 1] >= lower_z))
        for index in overlapping:
            bore = self._bores[index]
            fraction = _bore_intercept(start, end, bore)
            if fraction is not None:
                hits.append((fraction, f"hardware:{bore.key}"))
        for aperture in self._apertures:
            plane = float(aperture.z_mm)*1e-3
            if delta[2] == 0.:
                if start[2] != plane:
                    continue
                if not self._aperture_passes(aperture, start[:2]):
                    fraction = 0.
                elif self._aperture_passes(aperture, end[:2]):
                    continue
                else:
                    low, high = 0., 1.
                    for _ in range(48):
                        mid = .5*(low+high)
                        if self._aperture_passes(aperture, (start+mid*delta)[:2]):
                            low = mid
                        else:
                            high = mid
                    hits.append((high, f"aperture:{aperture.key}"))
                    continue
            else:
                fraction = (plane-start[2])/delta[2]
                if not 0. <= fraction <= 1.:
                    continue
            passes = self._aperture_passes(aperture, (start+fraction*delta)[:2])
            if not passes:
                hits.append((float(fraction), f"aperture:{aperture.key}"))
        return min(hits, key=lambda pair: pair[0]) if hits else None

    @staticmethod
    def _aperture_passes(aperture, xy_m):
        xy = np.asarray(xy_m)*1e3
        if callable(getattr(aperture, "transmission_mask", None)):
            return bool(np.asarray(aperture.transmission_mask(xy[:1], xy[1:])).reshape(-1)[0])
        if float(getattr(aperture, "radius_mm", 1.)) <= 0.:
            return False
        return math.hypot(xy[0]-float(aperture.offset_x_mm), xy[1]-float(aperture.offset_y_mm)) <= aperture.radius_mm


TestElectronScene.__test__ = False


@input_io.using_state_inputs
def prepare_test_electron_scene(state, magnetic_scene, *, z_limits_mm=None):
    """Capture complete electric optics once, reusing their existing field cache.

    A longer numerical electric domain uses the same physical grounded liner,
    electrode geometry and voltages. It is never a zero-field substitution at
    the earlier gun exit. Called in the numerical worker, not the GUI thread.
    """
    if not isinstance(magnetic_scene, MagneticSceneField):
        raise TypeError("An already captured MagneticSceneField is required")
    limits = (magnetic_scene.diagnostic_bounds_m if z_limits_mm is None
              else np.asarray(z_limits_mm, dtype=float)*1e-3)
    if z_limits_mm is None:
        limits = np.asarray(limits if limits is not None else magnetic_scene.bounds_m)[:, 2]
    if np.shape(limits) != (2,) or not np.isfinite(limits).all() or limits[1] <= limits[0]:
        raise ValueError("Electric diagnostic needs two increasing finite axial limits")
    requested_stop = float(limits[1])
    notes = list(magnetic_scene.notes)
    unsupported_stops = []
    if bool(getattr(state, "energy_filter_installed", False)):
        entrance = next((float(a.z_mm)*1e-3 for a in getattr(state, "apertures", ())
                         if a.key == "energy_filter_entrance_aperture" and bool(getattr(a, "installed", False))), None)
        if entrance is None:
            raise ValueError("Installed energy filter has no declared entrance plane")
        requested_stop = min(requested_stop, entrance)
        unsupported_stops.append((entrance, "unsupported_field:energy_filter"))
        notes.append("The installed energy filter bends the column coordinate system; this diagnostic stops at its entrance.")
    pulser = getattr(state, "nanopulser", None)
    if pulser is not None and bool(getattr(pulser, "installed", False)) and bool(getattr(pulser, "blanked", False)):
        pulser.validate()
        entrance = (float(pulser.z_mm)-.5*float(pulser.plate_length_mm))*1e-3
        if entrance <= 0.:
            raise ValueError("The active electrostatic beam blanker has no continuous electric-field provider at the physical tip; its existing thin-kick model is unchanged.")
        requested_stop = min(requested_stop, entrance)
        unsupported_stops.append((entrance, "unsupported_field:electrostatic_blanker"))
        notes.append(f"Active electrostatic beam blanker: continuous electric field is unavailable; this diagnostic stops before the plate entrance at {entrance*1e3:.6g} mm. The main thin-kick model is unchanged.")
    gun, electric, preparation = _prepare_electric_provider(state, requested_stop)
    base = getattr(electric, "base_field", electric)
    from temsim.physics.closed_gun_field import ClosedGunField
    from temsim.physics.grounded_tip_field import GroundedTipField
    from temsim.optics.electron_gun.electrostatic import FegElectrostaticField
    flat = isinstance(base, ClosedGunField)
    post_exit_ground = isinstance(base, GroundedTipField)
    r, z = getattr(base, "r", None), getattr(base, "z", None)
    if r is not None and z is not None:
        electric_bounds = np.array(((-r[-1], -r[-1], z[0]), (r[-1], r[-1], z[-1])))
    elif isinstance(base, FegElectrostaticField):
        radius = .5e-3*min(float(c.mechanical_clear_bore_diameter_mm) for c in gun.bore_components)
        electric_bounds = np.array(((-radius, -radius, 0.), (radius, radius, requested_stop)))
        notes.append("Analytic gun electric potential uses the existing near-axis radial expansion; the hardware bore bounds this diagnostic.")
    else:
        raise ValueError("Electric provider has no supported explicit diagnostic domain")
    initial = np.zeros((1, 3))
    if callable(getattr(base, "surface_mesh_positions", None)):
        initial, error = base.surface_mesh_positions(initial)
        notes.append(f"Tip launch uses the existing represented surface; displacement from the analytic apex {error:.6g} m.")
    model = getattr(gun.emitter, "surface_model", None)
    energy = float(model.emission.mean_energy_ev if model is not None else gun.emitter.emission_energy_ev)
    if not math.isfinite(energy) or energy <= 0.:
        raise ValueError("The physical emitter must define a positive local emission energy")
    bounds = electric_bounds.copy()
    bounds[1, 2] = requested_stop if post_exit_ground else min(requested_stop, electric_bounds[1, 2])
    if flat:
        bounds[0, 2] = min(bounds[0, 2], float(gun.emitter.mechanical_center_from_tip_mm)*1e-3)
    if bounds[1, 2] <= float(initial[0, 2]):
        raise ValueError("The calculation cutoff must lie after the physical tip")
    expanded_magnetic = replace(magnetic_scene, diagnostic_bounds_m=_frozen(bounds))
    electric_region = MagneticSourceRegion("gun_electric_field", "Gun electrostatic field", "electric", _frozen(electric_bounds))
    notes.extend(preparation)
    notes.extend(("Virtual electron starts at the physical tip with its local emission energy; edited diagnostic settings never replace the microscope source.",
                  "Captured supported electric and magnetic optics and hardware stops are included; specimen and detector interactions are excluded. Unsupported downstream fields stop this diagnostic explicitly.",
                  "Electric and magnetic inputs remain frozen when only the diagnostic electron is adjusted; unknown field domains stop propagation."))
    return TestElectronScene(expanded_magnetic, electric, base, _frozen(bounds), _frozen(electric_bounds),
                             tuple(float(v) for v in initial[0]), energy,
                             float(bounds[1, 2]-initial[0, 2]), tuple(notes),
                             _active_apertures(state, gun), _physical_bores(state, gun, base),
                             (electric_region,), flat, post_exit_ground,
                             _unsupported_stops=tuple(unsupported_stops))


__all__ = ["TestElectronScene", "prepare_test_electron_scene"]
