"""Finite, snapshot-owned vector fields for a display-only magnetic scene.

Coordinates are global right-handed XYZ in metres; +Z is downstream and B is
in tesla. This combines the existing lens, stigmator, corrector and gun fields.
Column deflectors, transported as angular kicks, have explicitly labelled
finite-coil equivalents with the same signed field integral; they are display
models, not a reconstruction of unmodelled fields inside magnetic material. The
analytic provider's first-order off-axis expansion is restricted to a small
near-axis cylinder.  Imported/generated maps retain their registered volume.
No particle propagation or magnetostatic solve is started by this module.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from types import SimpleNamespace

import numpy as np

from temsim import input_io
from temsim.component_keys import CONDENSER_LENS_KEYS
from temsim.physics.lens_field_provider import (
    FrozenMappedField,
    MappedLensFieldProvider,
    _provider_geometry_token,
    lens_geometry_binding,
    resolve_runtime_lens_field_provider,
)


def _points(values):
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Magnetic scene positions must be finite N by 3 XYZ metres")
    return points


def _frozen_array(values):
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MagneticSourceRegion:
    key: str
    label: str
    category: str
    bounds_m: np.ndarray


@dataclass(frozen=True)
class MagneticDiagnosticSamplingRegion:
    """Spatial resolution limits for a test electron in an active field."""

    bounds_m: np.ndarray
    transverse_step_m: float
    step_m: float


@dataclass(frozen=True)
class MagneticDiagnosticProfile:
    """Total field along the axis and on a declared small transverse ring.

    ``transverse_rms_t`` is sqrt(mean(Bx**2 + By**2)) on that ring. The
    gradient is the Frobenius norm / sqrt(2) of the on-axis transverse 2x2
    field Jacobian, using centred differences over the same radius. It equals
    abs(G) for Bx=G*y, By=G*x. It is not an on-axis field amplitude.
    """

    z_mm: np.ndarray
    on_axis_t: np.ndarray
    transverse_rms_t: np.ndarray
    transverse_gradient_rms_t_per_m: np.ndarray
    probe_radius_m: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class _Source:
    key: str
    provider: object
    bounds_m: np.ndarray
    radius_m: float
    field_map: object | None = None
    category: str = "lens"
    label: str = "Round lens"
    known_zero: bool = False

    def contains(self, points):
        valid = np.all((points >= self.bounds_m[0]) & (points <= self.bounds_m[1]), axis=1)
        if self.field_map is None:
            return valid & (np.hypot(points[:, 0], points[:, 1]) <= self.radius_m)
        local = self.field_map.registration.positions_global_to_local_m(points)
        if self.field_map.map_type == "axisymmetric_rz":
            coordinates = np.column_stack((np.hypot(local[:, 0], local[:, 1]), local[:, 2]))
        else:
            coordinates = local
        for index, axis in enumerate(self.field_map.axes_m):
            valid &= (coordinates[:, index] >= axis[0]) & (coordinates[:, index] <= axis[-1])
        return valid


@dataclass(frozen=True)
class MagneticSceneField:
    """Bounded vector query for a captured calculation, independent of the GUI.

    ``contains`` admits the provider-domain union only where every axially
    overlapping analytic contribution remains inside its near-axis radius;
    it does not claim that the field is nonzero. Missing map support is never
    extrapolated. The
    scene sums active contributions before tracing lines. A transverse
    quadrupole can have zero field on axis and still alter the total off-axis
    field. The post-column filter's bent coordinate system is not included.
    """

    bounds_m: np.ndarray
    transverse_radius_m: float
    source_keys: tuple[str, ...]
    notes: tuple[str, ...]
    seed_regions_m: tuple[np.ndarray, ...]
    _sources: tuple[_Source, ...]
    diagnostic_bounds_m: np.ndarray | None = None

    def _diagnostic_sources_at_position(self, position):
        point = np.asarray(position, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Test electron position must be finite XYZ metres")
        bounds = self.diagnostic_bounds_m if self.diagnostic_bounds_m is not None else self.bounds_m
        if any(point[i] < bounds[0, i] or point[i] > bounds[1, i] for i in range(3)):
            return None
        active = []
        radius = float(np.hypot(point[0], point[1]))
        for source in self._sources:
            if source.known_zero or not source.bounds_m[0, 2] <= point[2] <= source.bounds_m[1, 2]:
                continue
            if source.field_map is None:
                if (radius > source.radius_m or
                        any(point[i] < source.bounds_m[0, i] or point[i] > source.bounds_m[1, i] for i in (0, 1))):
                    return None
            elif not source.contains(point[None, :])[0]:
                return None
            active.append(source)
        return tuple(active)

    def diagnostic_position_is_valid(self, position):
        """Check test-electron support without evaluating the field again."""
        return self._diagnostic_sources_at_position(position) is not None

    @property
    def diagnostic_sampling_regions(self):
        """Resolve native map cells and the analytic near-axis scale.

        Map registrations rotate coordinates without changing lengths, so
        half the smallest native cell spacing is conservative for every
        propagation direction. This also resolves nonuniform native grids.
        Analytic fields use a transverse displacement bound instead, avoiding
        needless tiny steps for nearly axial electrons.
        """
        regions = []
        for source in self._sources:
            if source.known_zero:
                continue
            if source.field_map is not None:
                spacing = min(float(np.min(np.diff(axis))) for axis in source.field_map.axes_m)
                regions.append(MagneticDiagnosticSamplingRegion(source.bounds_m, np.inf, .5*spacing))
            else:
                regions.append(MagneticDiagnosticSamplingRegion(source.bounds_m, source.radius_m/8., np.inf))
        return tuple(regions)

    def diagnostic_field_at_global_position_t(self, position):
        """Single-point field for a magnetic-only virtual test electron.

        Return ``None`` at an unsupported position. Axial gaps inside the
        explicit display box have no active source and are zero in this finite
        field model. An active source's missing radial/map support is unknown,
        never zero. This stricter diagnostic domain does not change field-line
        sampling or the main particle transport. Fields remain frozen at the
        captured settings even when the test electron's energy is changed.
        """
        active = self._diagnostic_sources_at_position(position)
        if active is None:
            return None
        point = np.asarray(position, dtype=float)
        values = np.zeros(3)
        for source in active:
            field = np.asarray(source.provider.field_at_global_positions_t(point[None, :]), dtype=float)
            if field.shape != (1, 3) or not np.isfinite(field).all():
                raise ValueError(f"{source.key}: magnetic provider returned invalid XYZ field values")
            values += field[0]
        return values

    @property
    def source_categories(self):
        return tuple(source.category for source in self._sources)

    @property
    def source_regions(self):
        return tuple(MagneticSourceRegion(source.key, source.label, source.category, source.bounds_m)
                     for source in self._sources)

    def contains(self, points):
        points = _points(points)
        valid = np.zeros(len(points), dtype=bool)
        for source in self._sources:
            valid |= source.contains(points)
        # A near-axis display boundary is not a physical disappearance of a
        # lens field. At overlapping supports the complete sum is admissible
        # only inside every analytic contributor's radial validity region.
        # Otherwise terminate the line instead of bending it with a partial
        # sum. Finite imported maps retain their native zero-outside contract.
        radius = np.hypot(points[:, 0], points[:, 1])
        for source in self._sources:
            if source.field_map is None and not source.known_zero:
                active_z = ((points[:, 2] >= source.bounds_m[0, 2])
                            & (points[:, 2] <= source.bounds_m[1, 2]))
                valid &= ~(active_z & (radius > source.radius_m))
        return valid

    def field_at_global_positions_t(self, points):
        points = _points(points)
        field = np.zeros_like(points)
        admitted = self.contains(points)
        for source in self._sources:
            if source.known_zero:
                continue
            valid = admitted & source.contains(points)
            if np.any(valid):
                values = np.asarray(source.provider.field_at_global_positions_t(points[valid]), dtype=float)
                if values.shape != (int(np.count_nonzero(valid)), 3) or not np.isfinite(values).all():
                    raise ValueError(f"{source.key}: magnetic provider returned invalid XYZ field values")
                field[valid] += values
        return field


def _resolved_provider(state, lens):
    """Generated fields must already exist in this calculation snapshot."""
    key = str(lens.key)
    native = state.condenser_system[key] if key in CONDENSER_LENS_KEYS else lens
    from temsim.simulation_modes import uses_field_maps
    recipe = (getattr(state, "lens_field_map_descriptors", {}) or {}).get(key, {})
    solver = recipe.get("solver") if uses_field_maps(state) else None
    if solver not in {"axisymmetric_linear_fem", "axisymmetric_nonlinear_fem"}:
        return resolve_runtime_lens_field_provider(state, key, native)
    cache_name = ("_runtime_nonlinear_provider_cache" if solver == "axisymmetric_nonlinear_fem"
                  else "_runtime_lens_field_provider_cache")
    cached = (getattr(state, cache_name, {}) or {}).get(key)
    if not (isinstance(cached, tuple) and len(cached) == 2 and isinstance(cached[1], MappedLensFieldProvider)):
        raise ValueError(f"{key}: captured magnetic field cache is missing; recalculate before viewing fields")
    provider = cached[1]
    if solver == "axisymmetric_linear_fem":
        current = _provider_geometry_token(state, key, native)
        matched = current == cached[0]
    else:
        from temsim.physics.nonlinear_circuits import nonlinear_state_fingerprint
        diagnostic = (getattr(state, "_field_provider_diagnostics", {}) or {}).get(key, {})
        matched = diagnostic.get("joint_state_fingerprint") == nonlinear_state_fingerprint(state)
    if not matched or provider.binding.geometry_fingerprint != lens_geometry_binding(state, key, native).geometry_fingerprint:
        raise ValueError(f"{key}: captured field inputs changed; recalculate before viewing fields")
    provider.excitation_scale()  # Also enforces the frozen nonlinear operating point.
    return provider


def _near_axis_radius_m(provider):
    native = getattr(provider, "native_provider", provider)
    source = getattr(native, "lens", native)
    widths = []
    for scale_name, terms_name in (("a_mm", "gaussian"), ("upper_a_mm", "upper_gaussian"),
                                   ("lower_a_mm", "lower_gaussian")):
        scale = float(getattr(source, scale_name, 0.0))
        for term in getattr(source, terms_name, ()):
            value = abs(float(getattr(term, "sigma", 0.0)) * scale)
            if np.isfinite(value) and value > 0:
                widths.append(value)
    if not widths:
        for name in ("effective_length_mm", "length_mm", "pole_gap_mm", "inner_face_gap_mm"):
            value = float(getattr(native, name, getattr(source, name, 0.0)))
            if np.isfinite(value) and value > 0:
                widths.append(value / 2.355)
    if not widths:
        raise ValueError(f"{getattr(provider, 'lens_key', 'Lens')}: no finite axial scale for a near-axis magnetic scene")
    # A display-domain restriction, not an error estimate or a new field law.
    radius_mm = 0.1 * min(widths)
    for name in ("bore_diameter_mm", "pole_piece_bore_diameter_mm"):
        value = float(getattr(native, name, getattr(source, name, 0.0)))
        if np.isfinite(value) and value > 0:
            radius_mm = min(radius_mm, value * 0.5)
    return radius_mm * 1e-3


def _mapped_bounds(field_map):
    if field_map.map_type == "axisymmetric_rz":
        radius = float(field_map.axes_m[0][-1])
        ranges = ((-radius, radius), (-radius, radius), field_map.axes_m[1][[0, -1]])
    else:
        ranges = tuple(axis[[0, -1]] for axis in field_map.axes_m)
        radius = .5 * min(float(axis[-1] - axis[0]) for axis in field_map.axes_m[:2])
    corners = np.asarray(tuple(product(*ranges)), dtype=float)
    global_corners = (corners @ field_map.registration.rotation_array.T
                      + field_map.registration.origin_array_m)
    return np.vstack((global_corners.min(axis=0), global_corners.max(axis=0))), radius


@dataclass(frozen=True)
class _MultipoleField:
    """The same effective multipole-to-B conversion as specimen Lorentz transport."""

    state: object
    momentum_over_charge: float

    def field_at_global_positions_t(self, points):
        from temsim.physics.core import (
            multipole_focusing_fields, skew_quadrupole_field,
            hexapole_field_components,
        )
        points = np.asarray(points, dtype=float)
        z = points[..., 2] * 1e3
        kx, ky = multipole_focusing_fields(z, self.state)
        kxy = skew_quadrupole_field(z, self.state)
        hn, hs = hexapole_field_components(z, self.state)
        x, y = points[..., 0], points[..., 1]
        u, v = x*x-y*y, 2*x*y
        result = np.zeros_like(points)
        result[..., 0] = self.momentum_over_charge * (-ky*y-kxy*x+hn*v-hs*u)
        result[..., 1] = self.momentum_over_charge * (kx*x+kxy*y+hn*u+hs*v)
        return result


@dataclass(frozen=True)
class _EquivalentDeflectorField:
    """Display-only uniform field over the configured effective coil length.

    At the captured reference momentum, d(theta_x)/dz=-q*By/p and
    d(theta_y)/dz=q*Bx/p. No fringe shape or magnetic circuit is inferred
    from an integrated angular command. Runtime still owns the original kick.
    """

    low_m: float
    high_m: float
    bx_t: float
    by_t: float

    def field_at_global_positions_t(self, points):
        points = np.asarray(points, dtype=float)
        result = np.zeros_like(points)
        active = (points[..., 2] >= self.low_m) & (points[..., 2] <= self.high_m)
        result[..., 0] = np.where(active, self.bx_t, 0.)
        result[..., 1] = np.where(active, self.by_t, 0.)
        return result


def _component_support_mm(component):
    support = getattr(component, "field_support_mm", None)
    if callable(support):
        support = support()
    if support is None:
        length = float(getattr(component, "effective_length_mm", getattr(component, "length_mm", 0.)))
        center = float(getattr(component, "z_mm", np.nan))
        half = 7. * abs(length) / 2.355
        support = (center-half, center+half)
    values = np.asarray(support, dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all() or values[1] <= values[0]:
        raise ValueError(f"{component.key}: magnetic component needs finite axial support")
    return values


def _component_radius_m(component, *, axial_scale_mm):
    # An explicit display validity boundary for ideal near-axis polynomials.
    # No field is extended into an unknown bore or magnetic material.
    radius_mm = .1 * float(axial_scale_mm)
    bore_mm = float(getattr(component, "mechanical_clear_bore_diameter_mm", 0.))
    if bore_mm > 0.:
        radius_mm = min(radius_mm, .5*bore_mm)
    if not np.isfinite(radius_mm) or radius_mm <= 0.:
        raise ValueError(f"{component.key}: magnetic component needs a finite transverse domain")
    return radius_mm * 1e-3


def _extra_sources(state, z_clip, notes):
    """Freeze effective column operators and directly modelled gun magnets."""
    from temsim.physics.core import electron
    components = (*getattr(state, "stigmators", ()), *getattr(state, "corrector_elements", ()),
                  *getattr(state, "deflectors", ()))
    if components:
        charge, momentum, _ = electron(state)
        momentum_over_charge = momentum / charge
    else:
        momentum_over_charge = 0.
    sources = []

    def append(key, provider, support_mm, radius, category, label, *, known_zero=False):
        low, high = np.asarray(support_mm, dtype=float) * 1e-3
        low, high = max(low, z_clip[0]), min(high, z_clip[1])
        if high <= low:
            return
        bounds = _frozen_array(((-radius, -radius, low), (radius, radius, high)))
        sources.append(_Source(str(key), provider, bounds, radius, None, category, str(label), bool(known_zero)))

    seen = set()
    for original in components:
        key = str(original.key)
        if key in seen or not bool(getattr(original, "enabled", False)):
            continue
        seen.add(key)
        component = deepcopy(original, {id(state): state})
        label = str(getattr(component, "name", getattr(component, "label", key)))
        is_stigmator = hasattr(component, "quadrupole_tensor_m2")
        is_corrector = any(hasattr(component, name) for name in
                           ("quadrupole_strength_m2", "hexapole_strength_m3", "hexapole_strength_components_m3"))
        if is_stigmator or is_corrector:
            support = _component_support_mm(component)
            radius = _component_radius_m(component, axial_scale_mm=(support[1]-support[0])/14.)
            context = SimpleNamespace(stigmators=(component,) if is_stigmator else (),
                                      corrector_elements=() if is_stigmator else (component,),
                                      simulation_mode=getattr(state, "simulation_mode", "custom"))
            from temsim.optics.quadrupole import QuadrupoleComponent
            from temsim.optics.hexapole import HexapoleComponent
            from temsim.simulation_modes import is_ideal
            # Exact zeros of the configured Gaussian operators do not have a
            # radial approximation error and must not truncate another field.
            known_zero = (
                is_stigmator and getattr(component, "field_model", None) == "normal_skew"
                and component.strength_x_percent == 0. and component.strength_y_percent == 0.
            ) or (
                isinstance(component, QuadrupoleComponent)
                and type(component).quadrupole_strength_m2 is QuadrupoleComponent.quadrupole_strength_m2
                and component.strength_m2 == 0.
            ) or (
                isinstance(component, HexapoleComponent)
                and type(component).hexapole_strength_components_m3 is HexapoleComponent.hexapole_strength_components_m3
                and (component.strength_m3 == 0. or is_ideal(context))
            )
            append(key, _MultipoleField(context, momentum_over_charge), support, radius,
                   "stigmator" if is_stigmator else "corrector", label, known_zero=known_zero)
            notes.append(f"{label}: existing effective multipole field at the captured reference energy; near-axis radius <= {radius*1e3:.6g} mm.")
        if hasattr(component, "kick_events"):
            try:
                events = component.kick_events(time_s=float(getattr(state, "simulation_time_s", 0.)))
            except TypeError:
                events = component.kick_events()
        elif all(hasattr(component, name) for name in
                 ("upper_z_mm", "lower_z_mm", "upper_x_mrad", "upper_y_mrad", "lower_x_mrad", "lower_y_mrad")):
            events = ((component.upper_z_mm, component.upper_x_mrad*1e-3, component.upper_y_mrad*1e-3),
                      (component.lower_z_mm, component.lower_x_mrad*1e-3, component.lower_y_mrad*1e-3))
        else:
            continue
        events = tuple(events)
        if not events:
            # Layout-only virtual controls do not own a magnetic field.
            continue
        thickness_mm = float(getattr(component, "effective_thickness_mm", getattr(component, "thickness_mm", 0.)))
        if not np.isfinite(thickness_mm) or thickness_mm <= 0.:
            raise ValueError(f"{label}: deflector display needs its configured effective coil thickness")
        radius = _component_radius_m(component, axial_scale_mm=thickness_mm)
        length_m = thickness_mm*1e-3
        for index, (z_mm, dx_rad, dy_rad) in enumerate(events):
            if not np.isfinite((z_mm, dx_rad, dy_rad)).all():
                raise ValueError(f"{label}: deflector kick must be finite")
            support = (z_mm-.5*thickness_mm, z_mm+.5*thickness_mm)
            provider = _EquivalentDeflectorField(support[0]*1e-3, support[1]*1e-3,
                                                 momentum_over_charge*dy_rad/length_m,
                                                 -momentum_over_charge*dx_rad/length_m)
            append(f"{key}:{index}", provider, support, radius, "deflector", label,
                   known_zero=(dx_rad == 0. and dy_rad == 0.))
        notes.append(f"{label}: display-only equivalent finite-coil field; its signed integral reproduces the existing kick at {getattr(state, 'beam_voltage_kv', 0.):.6g} kV and time {getattr(state, 'simulation_time_s', 0.):.6g} s. The effective coil thickness is {thickness_mm:.6g} mm; no fringe shape is inferred.")

    gun = getattr(state, "electron_gun", None)
    if gun is not None:
        for name, category in (("deflector", "deflector"), ("stigmator", "stigmator")):
            original = getattr(gun, name, None)
            if original is None or not hasattr(original, "field_at_global_positions_t"):
                continue
            if not bool(getattr(original, "enabled", False)) and not bool(getattr(original, "beam_blanked", False)):
                continue
            component = deepcopy(original)
            edge = float(component.soft_edge_mm)
            if name == "deflector":
                half = .5*float(component.coil_length_mm)+edge
                centers = np.array((component.upper_center_from_tip_mm, component.lower_center_from_tip_mm)) + component.field_center_offset_mm
                support = (centers.min()-half, centers.max()+half)
                scale_mm = component.coil_length_mm
            else:
                half = .5*float(component.effective_length_mm)+edge
                support = (component.optical_reference_from_tip_mm-half, component.optical_reference_from_tip_mm+half)
                scale_mm = component.effective_length_mm
            radius = _component_radius_m(component, axial_scale_mm=scale_mm)
            known_zero = (component.gradient_t_per_m == 0. if name == "stigmator" else
                          not component.beam_blanked and all(getattr(component, item) == 0.
                          for item in ("upper_field_x_mt", "upper_field_y_mt", "lower_field_x_mt", "lower_field_y_mt")))
            append(component.key, component, support, radius, category, component.name, known_zero=known_zero)
            notes.append(f"{component.name}: existing finite gun magnetic provider, including the captured blanking/alignment setting.")
        if bool(getattr(gun, "monochromator_installed", False)):
            mono = gun.monochromator
            component = mono.wien
            if bool(component.enabled):
                radius = _component_radius_m(component, axial_scale_mm=component.active_length_mm)
                append(component.key, deepcopy(mono.field_provider), component.field_support_mm,
                       radius, "crossed_field", component.name)
                notes.append("Crossed-field velocity selector: the existing magnetic provider; its electric contribution is not a magnetic field line.")
    return sources


@input_io.using_state_inputs
def prepare_magnetic_scene(state, *, z_limits_mm=None):
    """Capture the total enabled magnetic graph for bounded field-line sampling.

    Analytic sampling is cheap. Recipe-backed FEM fields are admitted only from
    their dependency-checked captured cache; opening this view cannot launch a
    new solve. Map values and excitation scales are frozen without copying their
    numerical arrays. Ordinary analytic objects are copied while preserving the
    snapshot's shared state owner (their field laws use the copied lens inputs).
    """
    if z_limits_mm is None:
        z_clip = (-np.inf, np.inf)
    else:
        limits = np.asarray(z_limits_mm, dtype=float)
        if limits.shape != (2,) or not np.isfinite(limits).all() or limits[1] <= limits[0]:
            raise ValueError("Magnetic scene axial limits must be two increasing finite millimetre positions")
        z_clip = tuple(limits * 1e-3)
    lenses = {str(lens.key): lens for lens in getattr(state, "lenses", ())}
    sources, notes, included = [], ["Total captured magnetic field; +Z downstream, right-handed XYZ metres, B in tesla."], set()
    for key in lenses:
        lens = lenses[key]
        if not bool(getattr(lens, "enabled", True)):
            continue
        provider = _resolved_provider(state, lens)
        if getattr(provider, "model_status", "") == "included_in_joint_nonlinear_field":
            diagnostic = (getattr(state, "_field_provider_diagnostics", {}) or {}).get(key, {})
            owner = str(diagnostic.get("joint_field_owner", ""))
            if owner not in lenses:
                raise ValueError(f"{key}: the captured joint magnetic-field owner is unavailable")
            notes.append(f"{key} belongs to the joint field owned by {owner}; this contribution is counted once.")
            continue
        if key in included:
            continue
        included.add(key)
        if isinstance(provider, MappedLensFieldProvider):
            field_map = provider.field_map
            bounds, radius = _mapped_bounds(field_map)
            frozen_provider = FrozenMappedField.from_provider(provider)
            notes.append(f"{key}: {provider.model_status}; finite registered {field_map.map_type} map; no extrapolation.")
        else:
            field_map = None
            radius = _near_axis_radius_m(provider)
            support = np.asarray(provider.field_support_mm(), dtype=float) * 1e-3
            if support.shape != (2,) or not np.isfinite(support).all() or support[1] <= support[0]:
                raise ValueError(f"{key}: magnetic provider has no finite axial support")
            bounds = np.array(((-radius, -radius, support[0]), (radius, radius, support[1])))
            frozen_provider = deepcopy(provider, {id(state): state})
            notes.append(f"{key}: near-axis first-order field, radius <= {radius * 1e3:.6g} mm and 0.1 of the narrowest axial width; higher radial orders are not modelled.")
        bounds[0, 2] = max(bounds[0, 2], z_clip[0])
        bounds[1, 2] = min(bounds[1, 2], z_clip[1])
        if np.any(bounds[1] <= bounds[0]):
            continue
        label = str(getattr(lens, "name", getattr(lens, "label", key)))
        sources.append(_Source(key, frozen_provider, _frozen_array(bounds), radius, field_map, "lens", label))
    sources.extend(_extra_sources(state, z_clip, notes))
    notes.append("The post-column energy filter uses a separate bent coordinate system and is outside this straight-column scene.")
    if sources:
        bounds = np.vstack((np.min([source.bounds_m[0] for source in sources], axis=0),
                            np.max([source.bounds_m[1] for source in sources], axis=0)))
        radius = max(source.radius_m for source in sources)
    else:
        radius = 1e-6
        finite_clip = z_clip if np.isfinite(z_clip).all() else (0.0, 1e-3)
        bounds = np.array(((-radius, -radius, finite_clip[0]), (radius, radius, finite_clip[1])))
        notes.append("No enabled magnetic component overlaps this display region.")
    diagnostic_bounds = bounds.copy()
    if np.isfinite(z_clip).all():
        diagnostic_bounds[:, 2] = z_clip
    return MagneticSceneField(_frozen_array(bounds), float(radius), tuple(source.key for source in sources),
                              tuple(notes), tuple(source.bounds_m for source in sources), tuple(sources),
                              _frozen_array(diagnostic_bounds))


def sample_magnetic_diagnostic(scene, z_mm, *, probe_radius_m=1e-5):
    """Query the frozen total field without creating fields or transporting rays.

    The eight-point ring resolves dipole/quadrupole/hexapole transverse RMS.
    Its nominal radius is reduced locally until every sample is in the same
    valid total-field domain; the actual radius is returned for honest labels.
    Missing ring support is NaN, not an invented zero field. On-axis samples
    outside the scene's bounded provider union are zero with ``valid=False``;
    that mask must be used to distinguish display support from measured zero.
    """
    z = np.asarray(z_mm, dtype=float)
    nominal = float(probe_radius_m)
    if z.ndim != 1 or not np.isfinite(z).all() or not np.isfinite(nominal) or nominal <= 0.:
        raise ValueError("Magnetic diagnostic needs finite axial positions and a positive probe radius")
    center = np.column_stack((np.zeros((len(z), 2)), z*1e-3))
    valid = scene.contains(center)
    on_axis = scene.field_at_global_positions_t(center)
    radius = np.full(len(z), nominal)
    angles = np.arange(8)*np.pi/4.
    offsets = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(8)))
    ring_valid = np.zeros(len(z), dtype=bool)
    for _ in range(24):
        ring = center[:, None, :] + radius[:, None, None]*offsets[None, :, :]
        ring_valid = scene.contains(ring.reshape(-1, 3)).reshape(-1, 8).all(axis=1)
        unresolved = valid & ~ring_valid
        if not unresolved.any():
            break
        radius[unresolved] *= .5
    sampled = valid & ring_valid
    values = scene.field_at_global_positions_t(ring.reshape(-1, 3)).reshape(-1, 8, 3)
    rms = np.sqrt(np.mean(np.sum(values[..., :2]**2, axis=2), axis=1))
    dx = (values[:, 0, :2]-values[:, 4, :2])/(2.*radius[:, None])
    dy = (values[:, 2, :2]-values[:, 6, :2])/(2.*radius[:, None])
    gradient = np.sqrt((np.sum(dx*dx, axis=1)+np.sum(dy*dy, axis=1))/2.)
    rms[~sampled] = np.nan
    gradient[~sampled] = np.nan
    radius[~sampled] = np.nan
    sampled.setflags(write=False)
    return MagneticDiagnosticProfile(_frozen_array(z), _frozen_array(on_axis), _frozen_array(rms),
                                     _frozen_array(gradient), _frozen_array(radius), sampled)
