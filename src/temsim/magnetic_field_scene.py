"""Finite, snapshot-owned vector fields for a display-only magnetic scene.

Coordinates are global right-handed XYZ in metres; +Z is downstream and B is
in tesla. This combines the existing lens, stigmator, corrector and gun fields.
Column deflectors use the same finite-coil field as particle transport, with
the configured signed integral and effective length. This is not a
reconstruction of unmodelled fields inside magnetic material. The
analytic provider's first-order off-axis expansion is restricted to a small
near-axis cylinder.  Imported/generated maps retain their registered volume.
No particle propagation or magnetostatic solve is started by this module.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from hashlib import sha256
import inspect
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy._lib._array_api import array_namespace

from temsim import input_io
from temsim.component_keys import CONDENSER_LENS_KEYS
from temsim.physics.instrument_magnetic import ColumnDipoleField, column_dipole_fields
from temsim.physics.lens_field_provider import (
    CoordinateRegistration,
    FrozenMappedField,
    MagneticFieldMap,
    MappedLensFieldProvider,
    _provider_geometry_token,
    lens_geometry_binding,
    resolve_runtime_lens_field_provider,
)


# Capture the admitted implementations before a caller can replace methods.
# Exact dataclass types alone do not establish which interpolation law executes.
_MAPPED_METHOD_CONTRACTS = {
    cls: tuple((name, inspect.getattr_static(cls, name)) for name in names)
    for cls, names in (
        (MappedLensFieldProvider, ("excitation_scale", "field_support_mm", "field_at_global_positions_t")),
        (FrozenMappedField, ("from_provider", "field_at_global_positions_t")),
        (MagneticFieldMap, ("field_support_mm", "field_at_global_positions_t")),
        (CoordinateRegistration, ("origin_array_m", "rotation_array", "positions_global_to_local_m", "vectors_local_to_global")),
        (RegularGridInterpolator, ("__call__", "_prepare_xi", "_find_indices", "_evaluate_linear", "_find_out_of_bounds")),
    )
}
_RGI_ARRAY_CONVERTER = array_namespace(np.empty(0)).asarray


def _mapped_methods_match(instance):
    contract = _MAPPED_METHOD_CONTRACTS.get(type(instance))
    return contract is not None and all(
        inspect.getattr_static(instance, name, None) is original
        for name, original in contract
    )


def _readonly_same_array(actual, declared):
    return (type(actual) is np.ndarray and type(declared) is np.ndarray
            and not actual.flags.writeable and not declared.flags.writeable
            and actual.dtype == declared.dtype and actual.shape == declared.shape
            and (actual is declared or np.array_equal(actual, declared)))


def _mapped_inputs_are_supported(source, original_provider):
    """Verify the consumed arrays and law, without evaluating or rebuilding B."""
    provider, field_map = source.provider, source.field_map
    if (type(original_provider) is not MappedLensFieldProvider or type(provider) is not FrozenMappedField
            or type(field_map) is not MagneticFieldMap or type(field_map.registration) is not CoordinateRegistration):
        return False
    if not all(_mapped_methods_match(item) for item in (original_provider, provider, field_map)):
        return False
    if field_map is not provider.field_map or field_map is not original_provider.field_map:
        return False
    if not _mapped_methods_match(field_map.registration):
        return False
    dimension = {"axisymmetric_rz": 2, "cartesian_xyz": 3}.get(field_map.map_type)
    if dimension is None or len(field_map.axes_m) != dimension or len(field_map.components_t) != dimension:
        return False
    if len(field_map._interpolators) != dimension:
        return False
    for interpolator, component in zip(field_map._interpolators, field_map.components_t):
        if not _mapped_methods_match(interpolator):
            return False
        if interpolator.method != "linear" or interpolator.bounds_error is not False or interpolator.fill_value != 0.:
            return False
        # SciPy stores the arrays privately in current releases; older releases
        # use direct public attributes. Read storage, never an overridable property.
        storage = vars(interpolator)
        values = storage.get("_values", storage.get("values"))
        grid = storage.get("_grid", storage.get("grid"))
        if "_asarray" in storage and storage["_asarray"] is not _RGI_ARRAY_CONVERTER:
            return False
        if not _readonly_same_array(values, component) or not isinstance(grid, tuple) or len(grid) != dimension:
            return False
        if not all(_readonly_same_array(actual, axis) for actual, axis in zip(grid, field_map.axes_m)):
            return False
    return True


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
class MagneticFieldSupport:
    """Captured model identity and support, never a physical accuracy claim."""

    key: str
    model: str
    bounds_m: tuple[tuple[float, ...], ...]
    radial_limit_m: float
    physical_identity: str | None = None
    numerical_identity: str | None = None
    reference_momentum_kg_m_s: float | None = None
    reference_charge_c: float | None = None
    captured_time_s: float | None = None
    limitation: str = "Provider identity unavailable; no reproducible field claim."


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
    identity: MagneticFieldSupport | None = None

    def contains(self, points):
        valid = np.all((points >= self.bounds_m[0]) & (points <= self.bounds_m[1]), axis=1)
        if self.field_map is None:
            local = self.analytic_local_points(points)
            if hasattr(self.provider, "native_field_support_mm"):
                lo, hi = np.asarray(self.provider.native_field_support_mm())*1e-3
                valid &= (local[:, 2] >= lo) & (local[:, 2] <= hi)
            return valid & (np.hypot(local[:, 0], local[:, 1]) <= self.radius_m)
        local = self.field_map.registration.positions_global_to_local_m(points)
        if self.field_map.map_type == "axisymmetric_rz":
            coordinates = np.column_stack((np.hypot(local[:, 0], local[:, 1]), local[:, 2]))
        else:
            coordinates = local
        for index, axis in enumerate(self.field_map.axes_m):
            valid &= (coordinates[:, index] >= axis[0]) & (coordinates[:, index] <= axis[-1])
        return valid

    def analytic_local_points(self, points):
        registration = getattr(self.provider, "registration", None)
        return (registration.positions_global_to_local_m(points)
                if registration is not None else points)

    def analytic_axial_mask(self, points):
        if hasattr(self.provider, "native_field_support_mm"):
            local = self.analytic_local_points(points)
            lo, hi = np.asarray(self.provider.native_field_support_mm())*1e-3
            return (local[:, 2] >= lo) & (local[:, 2] <= hi)
        return (points[:, 2] >= self.bounds_m[0, 2]) & (points[:, 2] <= self.bounds_m[1, 2])


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
    physical_identity: str | None = None
    numerical_identity: str | None = None
    support_metadata: tuple[MagneticFieldSupport, ...] = ()

    def with_diagnostic_bounds(self, bounds_m):
        """Change numerical support without relabelling the captured physics."""
        bounds = np.asarray(bounds_m, dtype=float)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
            raise ValueError("Magnetic diagnostic bounds must be finite increasing XYZ metres")
        result = replace(self, diagnostic_bounds_m=_frozen_array(bounds))
        return replace(result, numerical_identity=_scene_numerical_identity(result))

    def _diagnostic_sources_at_position(self, position):
        point = np.asarray(position, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Test electron position must be finite XYZ metres")
        bounds = self.diagnostic_bounds_m if self.diagnostic_bounds_m is not None else self.bounds_m
        if any(point[i] < bounds[0, i] or point[i] > bounds[1, i] for i in range(3)):
            return None
        active = []
        for source in self._sources:
            if source.known_zero or not source.bounds_m[0, 2] <= point[2] <= source.bounds_m[1, 2]:
                continue
            if source.field_map is None:
                if not source.analytic_axial_mask(point[None, :])[0]:
                    continue
                if not source.contains(point[None, :])[0]:
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
        for source in self._sources:
            if source.field_map is None and not source.known_zero:
                local = source.analytic_local_points(points)
                radius = np.hypot(local[:, 0], local[:, 1])
                active_z = source.analytic_axial_mask(points)
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
        from temsim.lens_pose import lens_pose_registration
        from temsim.physics.lens_field_provider import _identity_registration
        registration = lens_pose_registration(state, key)
        if not _identity_registration(registration):
            posed = getattr(state, "_runtime_posed_nonlinear_provider_cache", {}).get(key)
            if posed is None or posed[0] is not provider or posed[1] != registration:
                matched = False
            else:
                provider = posed[2]
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


def _profile_inputs(component, names):
    """Select consumed scalar controls; presentation and mutable owners stay out."""
    return {name: getattr(component, name) for name in names}


def _gaussian_inputs(terms):
    return tuple((float(term.amplitude), float(term.offset), float(term.sigma)) for term in terms)


@lru_cache(maxsize=64)
def _implementation_inputs(provider_type, native_type=None, component_type=None):
    """Numerical identity follows implementation files used by known laws."""
    paths = {Path(__file__), Path(inspect.getfile(provider_type))}
    for cls in (native_type, component_type):
        if cls is not None:
            # Include inherited field methods, not just the concrete class file.
            paths.update(Path(inspect.getfile(base)) for base in cls.__mro__ if base.__module__.startswith("temsim."))
    if native_type is not None:
        from temsim.optics import lens_focal_length
        paths.add(Path(lens_focal_length.__file__))
    if provider_type is _MultipoleField:
        from temsim.physics import core
        from temsim.optics import stigmator_field
        paths.update((Path(core.__file__), Path(stigmator_field.__file__)))
    from temsim.optics.electron_gun.alignment import GunDeflector, GunStigmator
    if provider_type in (GunDeflector, GunStigmator):
        from temsim.optics.electron_gun import electrostatic
        paths.add(Path(electrostatic.__file__))
    return {str(path.relative_to(Path(__file__).parent)).replace("\\", "/"): sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def _source_field_inputs(source, original_provider, reference):
    """Admit only explicitly understood field laws, not arbitrary callbacks.

    The existing compiled-law registry supplies the same concrete-type and
    method guards; using it here does not request compilation or transport.
    Map arrays are hashed from the captured numerical data, not a path or a
    caller-provided cache label. Unknown laws remain usable but unidentified.
    """
    from temsim.diagnostic_field_identity import field_array_digest
    from temsim.test_electron_compiled_laws import (
        lens_family, magnetic_provider_is_supported, multipole_is_supported,
    )
    from temsim.physics.lens_field_provider import GeometryAwareAnalyticFieldProvider
    from temsim.optics.electron_gun.alignment import GunDeflector, GunStigmator

    provider = source.provider
    from temsim.physics.posed_column_fields import FrozenPosedMultipole
    if type(provider) is FrozenPosedMultipole:
        return ("posed_native_multipole", asdict(provider), {},
                "Rigid native transverse multipole; original axial envelope and finite support; local paraxial model.")
    if type(original_provider) is MappedLensFieldProvider and type(provider) is FrozenMappedField:
        field_map = source.field_map
        if not _mapped_inputs_are_supported(source, original_provider):
            return None
        binding = original_provider.binding
        if not binding.geometry_fingerprint or not field_map.geometry_fingerprint:
            return None
        physical = {
            "geometry_binding": binding.geometry_fingerprint,
            "map_geometry_binding": field_map.geometry_fingerprint,
            "excitation_scaling": original_provider.excitation_scaling,
            "scale": provider.scale,
            "reference_excitation_percent": field_map.reference_excitation_percent,
            "reference_polarity": field_map.reference_polarity,
            "circuit_operating_point": original_provider.circuit_operating_point,
            "origin_global_m": field_map.registration.origin_global_m,
            "rotation_local_to_global": field_map.registration.rotation_local_to_global,
        }
        numerical = {
            "map_type": field_map.map_type,
            "axes_m": tuple(field_array_digest(axis) for axis in field_map.axes_m),
            "components_t": tuple(field_array_digest(values) for values in field_map.components_t),
            "interpolation": "linear; no extrapolation; native map zero-outside",
        }
        return "registered_mapped_field", physical, numerical, "Finite registered map; missing active map support is unknown, not zero."
    if not magnetic_provider_is_supported(provider, allow_rigid_pose=True):
        return None
    if type(provider) is GeometryAwareAnalyticFieldProvider:
        native = provider.native_provider
        family = lens_family(native)
        if family is None or not provider.binding.geometry_fingerprint:
            return None
        if family == "objective":
            physical = _profile_inputs(native, ("enabled", "percent", "polarity"))
            for prefix in ("upper", "lower"):
                physical.update(_profile_inputs(native, (
                    prefix + "_b0_t", prefix + "_a_mm", prefix + "_field_center_z_mm")))
                physical[prefix + "_gaussian"] = _gaussian_inputs(getattr(native, prefix + "_gaussian"))
        else:
            lens = getattr(native, "lens", native)
            physical = _profile_inputs(lens, ("enabled", "z_mm", "b0_t", "a_mm", "percent", "polarity"))
            physical["normalise_profile_peak"] = bool(getattr(lens, "normalise_profile_peak", False))
            physical["gaussian"] = _gaussian_inputs(lens.gaussian)
        physical["geometry_binding"] = provider.binding.geometry_fingerprint
        physical["family"] = family
        physical["origin_global_m"] = provider.registration.origin_global_m
        physical["rotation_local_to_global"] = provider.registration.rotation_local_to_global
        support = provider.native_field_support_mm()
        numerical = {"axial_support_mm": tuple(support),
                     "axial_derivative_step_mm": max(abs(support[1] - support[0]), 1.) * 1e-6}
        return "near_axis_first_order", physical, numerical, "First-order off-axis expansion; higher radial orders unmodelled; radius is a support restriction, not an error bound."
    if type(provider) is ColumnDipoleField:
        if reference is None:
            return None  # A B value alone cannot reconstruct its reference momentum/time.
        physical = {**reference, "coil_range_m": (provider.lower_m, provider.upper_m),
                    "bx_t": provider.bx_t, "by_t": provider.by_t}
        return "finite_coil_dipole", physical, {}, "Shared uniform finite-coil field at captured momentum/time; fringe shape is unmodelled."
    if type(provider) is _MultipoleField:
        components = (*provider.state.stigmators, *provider.state.corrector_elements)
        if len(components) != 1 or not multipole_is_supported(components[0]):
            return None
        component = components[0]
        from temsim.optics.quadrupole import QuadrupoleComponent
        from temsim.optics.hexapole import HexapoleComponent
        from temsim.simulation_modes import is_ideal
        if isinstance(component, QuadrupoleComponent):
            names = ("enabled", "z_mm", "effective_length_mm", "strength_m2")
            family = "quadrupole"
        elif isinstance(component, HexapoleComponent):
            names = ("enabled", "z_mm", "effective_length_mm", "strength_m3", "orientation_rad")
            family = "hexapole"
        else:
            names = ("enabled", "z_mm", "length_mm", "field_model", "max_strength_m2",
                     "strength_x_percent", "strength_y_percent", "channel_x_angle_deg", "channel_y_angle_deg")
            family = "stigmator"
        physical = {**_profile_inputs(component, names), "family": family,
                    "momentum_over_charge": provider.momentum_over_charge}
        if family == "hexapole":
            physical["ideal_mode_disables_hexapole"] = is_ideal(provider.state)
        if reference is not None:
            physical.update(reference)
        return "effective_multipole", physical, {}, "Existing near-axis effective multipole at captured reference momentum; no magnetic-material field reconstruction."
    if type(provider) is GunDeflector:
        names = ("enabled", "beam_blanked", "upper_center_from_tip_mm", "lower_center_from_tip_mm",
                 "coil_length_mm", "field_center_offset_mm", "soft_edge_mm")
        physical = _profile_inputs(provider, names)
        physical.update(_profile_inputs(provider, ("blanking_field_y_mt",) if provider.beam_blanked else
            ("upper_field_x_mt", "upper_field_y_mt", "lower_field_x_mt", "lower_field_y_mt")))
        return "finite_gun_deflector", physical, {}, "Existing smooth finite gun field, including captured blanking setting."
    if type(provider) is GunStigmator:
        physical = _profile_inputs(provider, ("enabled", "optical_reference_from_tip_mm", "effective_length_mm",
                                               "soft_edge_mm", "gradient_t_per_m", "rotation_deg"))
        return "finite_gun_stigmator", physical, {}, "Existing finite quadrupole gun field inside the declared radial support."
    return None


def _bind_source_identity(source, *, original_provider=None, reference=None):
    from temsim.diagnostic_field_identity import identity_digest

    bounds = tuple(tuple(float(value) for value in row) for row in source.bounds_m)
    inputs = _source_field_inputs(source, original_provider, reference)
    if inputs is None:
        support = MagneticFieldSupport(source.key, "unknown_provider", bounds, source.radius_m)
    else:
        model, physical, numerical, limitation = inputs
        provider = source.provider
        native = getattr(provider, "native_provider", None)
        components = ((*provider.state.stigmators, *provider.state.corrector_elements)
                      if type(provider) is _MultipoleField else ())
        implementation = _implementation_inputs(type(provider), type(native) if native is not None else None,
                                                  type(components[0]) if components else None)
        libraries = {"numpy": np.__version__}
        if source.field_map is not None:
            import scipy
            libraries["scipy"] = scipy.__version__
        physical_id = identity_digest("magnetic-source-physical-v1", {"key": source.key, "model": model, "inputs": physical})
        numerical_id = identity_digest("magnetic-source-numerical-v1", {
            "physical_identity": physical_id, "bounds_m": bounds, "radius_m": source.radius_m,
            "known_zero": source.known_zero, "inputs": numerical, "arithmetic": "float64",
            "implementation_sha256": implementation, "libraries": libraries,
        })
        reference = reference or {}
        support = MagneticFieldSupport(source.key, model, bounds, source.radius_m,
            physical_id, numerical_id, reference.get("reference_momentum_kg_m_s"),
            reference.get("reference_charge_c"), reference.get("captured_time_s"), limitation)
    return replace(source, identity=support)


def _scene_numerical_identity(scene):
    from temsim.diagnostic_field_identity import identity_digest

    if scene.physical_identity is None or any(item.numerical_identity is None for item in scene.support_metadata):
        return None
    return identity_digest("magnetic-scene-numerical-v1", {
        "physical_identity": scene.physical_identity,
        "sources_in_sum_order": tuple(item.numerical_identity for item in scene.support_metadata),
        "bounds_m": scene.bounds_m.tolist(),
        "diagnostic_bounds_m": (scene.diagnostic_bounds_m if scene.diagnostic_bounds_m is not None else scene.bounds_m).tolist(),
        "support_rule": "all active axial contributions required; gaps zero; filter bent coordinates unsupported",
    })


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
    from temsim.physics.posed_column_fields import capture_posed_column_fields
    posed = capture_posed_column_fields(state)
    posed_components = {field.component_key for field in posed}
    posed_coils = {field.lens_key for field in posed if field.event_z_mm is not None}

    def append(key, provider, support_mm, radius, category, label, *, known_zero=False, reference=None, global_bounds=None):
        low, high = np.asarray(support_mm, dtype=float) * 1e-3
        low, high = max(low, z_clip[0]), min(high, z_clip[1])
        if high <= low:
            return
        if global_bounds is None:
            bounds = _frozen_array(((-radius, -radius, low), (radius, radius, high)))
        else:
            bounds = np.array(global_bounds, copy=True)
            bounds[:, 2] = (low, high)
            bounds = _frozen_array(bounds)
        source = _Source(str(key), provider, bounds, radius, None, category, str(label), bool(known_zero))
        sources.append(_bind_source_identity(source, reference=reference))

    seen = set()
    for original in components:
        key = str(original.key)
        if key in seen or not bool(getattr(original, "enabled", False)):
            continue
        seen.add(key)
        if key in posed_components:
            continue
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
            from temsim.test_electron_compiled_laws import multipole_is_supported
            known_zero = known_zero and multipole_is_supported(component)
            append(key, _MultipoleField(context, momentum_over_charge), support, radius,
                   "stigmator" if is_stigmator else "corrector", label, known_zero=known_zero,
                   reference={"reference_momentum_kg_m_s": momentum, "reference_charge_c": charge})
            notes.append(f"{label}: existing effective multipole field at the captured reference energy; near-axis radius <= {radius*1e3:.6g} mm.")
    # One definition supplies production interval forces and diagnostic B.
    by_key = {str(component.key): component for component in components}
    for provider in column_dipole_fields(state):
        if provider.key in posed_coils:
            continue
        component = by_key[provider.key.rsplit(":", 1)[0]]
        label = str(getattr(component, "name", getattr(component, "label", component.key)))
        thickness_mm = (provider.upper_m-provider.lower_m)*1e3
        radius = _component_radius_m(component, axial_scale_mm=thickness_mm)
        append(provider.key, provider, (provider.lower_m*1e3, provider.upper_m*1e3),
               radius, "deflector", label,
               known_zero=(provider.bx_t == 0. and provider.by_t == 0.), reference={
                   "reference_momentum_kg_m_s": provider.reference_momentum,
                   "reference_charge_c": charge,
                   "captured_time_s": provider.captured_time_s,
                   "kick_xy_rad": (provider.event_dx_rad, provider.event_dy_rad),
               })
        notes.append(f"{label}: shared uniform finite-coil magnetic field; effective length {thickness_mm:.6g} mm. No fringe shape is inferred.")

    for provider in posed:
        component = by_key[provider.component_key]
        label = str(getattr(component, "name", getattr(component, "label", provider.component_key)))
        append(provider.lens_key, provider, provider.field_support_mm, provider.radial_support_m,
               "deflector" if provider.event_z_mm is not None else "stigmator" if hasattr(component, "quadrupole_tensor_m2") else "corrector",
               label, known_zero=not provider.scale, global_bounds=provider.bounds_m,
               reference={"reference_momentum_kg_m_s": provider.reference_momentum,
                          "reference_charge_c": charge, "captured_time_s": provider.captured_time_s})
        notes.append(f"{label}: rigidly placed native transverse magnetic field; original envelope and support retained.")

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
            from temsim.test_electron_compiled_laws import magnetic_provider_is_supported
            known_zero = known_zero and magnetic_provider_is_supported(component)
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
            native_support = getattr(provider, "native_field_support_mm", provider.field_support_mm)
            support = np.asarray(native_support(), dtype=float) * 1e-3
            if support.shape != (2,) or not np.isfinite(support).all() or support[1] <= support[0]:
                raise ValueError(f"{key}: magnetic provider has no finite axial support")
            corners = np.asarray(tuple(product((-radius, radius), (-radius, radius), support)))
            registration = getattr(provider, "registration", CoordinateRegistration())
            corners = (corners @ registration.rotation_array.T + registration.origin_array_m)
            bounds = np.vstack((corners.min(axis=0), corners.max(axis=0)))
            frozen_provider = deepcopy(provider, {id(state): state})
            notes.append(f"{key}: near-axis first-order field, radius <= {radius * 1e3:.6g} mm and 0.1 of the narrowest axial width; higher radial orders are not modelled.")
        bounds[0, 2] = max(bounds[0, 2], z_clip[0])
        bounds[1, 2] = min(bounds[1, 2], z_clip[1])
        if np.any(bounds[1] <= bounds[0]):
            continue
        label = str(getattr(lens, "name", getattr(lens, "label", key)))
        source = _Source(key, frozen_provider, _frozen_array(bounds), radius, field_map, "lens", label)
        sources.append(_bind_source_identity(source, original_provider=provider))
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
    support_metadata = tuple(source.identity for source in sources)
    if any(item.physical_identity is None for item in support_metadata):
        physical_identity = None
        notes.append("Some captured magnetic providers have unknown reproducible identities; total field identity remains unknown.")
    else:
        from temsim.diagnostic_field_identity import identity_digest
        physical_identity = identity_digest("magnetic-scene-physical-v1", {
            "sources": tuple(sorted((item.key, item.physical_identity) for item in support_metadata)),
            "coordinates": "right-handed global XYZ m; downstream +Z; B tesla",
        })
    scene = MagneticSceneField(_frozen_array(bounds), float(radius), tuple(source.key for source in sources),
                               tuple(notes), tuple(source.bounds_m for source in sources), tuple(sources),
                               _frozen_array(diagnostic_bounds), physical_identity, None, support_metadata)
    return replace(scene, numerical_identity=_scene_numerical_identity(scene))


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
