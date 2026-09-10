"""Geometry-bound magnetic-field maps and runtime lens-field selection.

The existing Gaussian lens profiles remain a useful engineering fallback, but
they are not magnetostatic solutions of an edited pole-piece geometry.  This
module therefore separates three facts that must never be conflated:

* the current assembled lens geometry;
* a measured or FEM field map bound to that exact geometry; and
* a clearly labelled provisional analytic fallback.

Global coordinates are right handed and use metres here.  The microscope beam
axis is global +Z.  Axisymmetric maps store ``(Br, Bz)`` on ``(r, z)`` grids;
Cartesian maps store ``(Bx, By, Bz)`` on ``(x, y, z)`` grids.  File loaders
accept SI-labelled columns/arrays only and never infer units from magnitudes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize_scalar


class FieldMapError(ValueError):
    """Base class for invalid or incompatible magnetic-field maps."""


class FieldMapGeometryMismatch(FieldMapError):
    """Raised when a field map is used with a different physical assembly."""


def _immutable_array(values, *, ndim: int, name: str) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=np.float64)
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise FieldMapError(f"{name} must be a finite {ndim}-D array")
    result.setflags(write=False)
    return result


def _canonical(value):
    if is_dataclass(value):
        return {
            item.name: _canonical(getattr(value, item.name))
            for item in dataclass_fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _fingerprint(value) -> str:
    encoded = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CoordinateRegistration:
    """Rigid registration from map-local coordinates to global SI coordinates."""

    origin_global_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_local_to_global: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin_global_m, dtype=float)
        rotation = np.asarray(self.rotation_local_to_global, dtype=float)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise FieldMapError("Field-map registration origin is invalid")
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise FieldMapError("Field-map registration rotation is invalid")
        if not np.allclose(
            rotation.T @ rotation, np.eye(3), rtol=0.0, atol=2.0e-10
        ) or not math.isclose(
            float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=2.0e-10
        ):
            raise FieldMapError(
                "Field-map registration must be a proper orthonormal rotation"
            )
        object.__setattr__(self, "origin_global_m", tuple(float(v) for v in origin))
        object.__setattr__(self, "rotation_local_to_global",
                           tuple(tuple(float(v) for v in row) for row in rotation))

    @property
    def origin_array_m(self) -> np.ndarray:
        return np.asarray(self.origin_global_m, dtype=float)

    @property
    def rotation_array(self) -> np.ndarray:
        return np.asarray(self.rotation_local_to_global, dtype=float)

    def positions_global_to_local_m(self, positions_m) -> np.ndarray:
        positions = np.asarray(positions_m, dtype=float)
        if positions.shape[-1:] != (3,) or not np.all(np.isfinite(positions)):
            raise FieldMapError("Field query positions must end in a finite XYZ axis")
        return (positions - self.origin_array_m) @ self.rotation_array

    def vectors_local_to_global(self, vectors) -> np.ndarray:
        local = np.asarray(vectors, dtype=float)
        return local @ self.rotation_array.T


@dataclass(frozen=True, slots=True)
class LensGeometryBinding:
    """Stable identity of one physical lens and its local assembly context."""

    lens_key: str
    geometry_fingerprint: str
    assembly_fingerprint: str
    canonical_geometry_json: str


_GEOMETRY_ATTRIBUTES = (
    "z_mm",
    "mechanical_center_from_tip_mm",
    "mechanical_length_mm",
    "assembly_length_mm",
    "mechanical_outer_diameter_mm",
    "assembly_outer_diameter_mm",
    "bore_diameter_mm",
    "pole_gap_mm",
    "inner_face_gap_mm",
    "pole_piece_bore_diameter_mm",
    "pole_piece_tip_diameter_mm",
    "upper_pole_piece_tip_diameter_mm",
    "pole_piece_outer_diameter_mm",
    "upper_pole_piece_outer_diameter_mm",
    "pole_piece_axial_length_mm",
    "upper_pole_piece_axial_length_mm",
    "pole_piece_center_separation_mm",
    "upper_pole_piece_center_z_mm",
    "lower_pole_piece_center_z_mm",
    "optical_reference_from_tip_mm",
)

_PART_GEOMETRY_FIELDS = frozenset({
    "local_start_z_mm",
    "local_center_z_mm",
    "local_end_z_mm",
    "length_mm",
    "mechanical_profile",
    "pole_piece_geometry_style",
    "pole_piece_topology",
    "mechanical_inner_diameter_mm",
    "vacuum_inner_diameter_mm",
    "mechanical_bore_diameter_mm",
    "mechanical_clear_bore_diameter_mm",
    "mechanical_tip_diameter_mm",
    "mechanical_outer_diameter_mm",
    "pole_gap_mm",
    "pole_nose_axial_length_mm",
    "pole_cone_angle_to_axis_deg",
    "pole_stem_outer_diameter_mm",
    "material_class",
    "material_regions",
    "magnetic_circuit_id",
    "magnetic_lens_keys",
    "field_source_key",
    "coil_ampere_turn_fraction",
    "magnetic_part_role",
    "upper_yoke_start_local_z_mm", "upper_yoke_end_local_z_mm",
    "lower_yoke_start_local_z_mm", "lower_yoke_end_local_z_mm",
    "mechanical_coil_axial_inset_mm",
})


_FIELD_STRUCTURE_PROFILES = frozenset({
    "magnetic_lens_assembly",
    "magnetic_lens_housing",
    "magnetic_lens_yoke",
    "magnetic_excitation_coil",
    "magnetic_pole_piece",
    "c1_c2_pole_piece_cartridge",
})

_PART_GEOMETRY_TOKENS = (
    "profile",
    "style",
    "shape",
    "topology",
    "bore",
    "gap",
    "diameter",
    "radius",
    "angle",
    "taper",
    "tip",
    "nose",
    "stem",
    "shank",
    "land",
    "fillet",
    "material",
    "permeability",
    "vacuum",
    "thickness",
)

_PART_GEOMETRY_METADATA_SUFFIXES = (
    "_source",
    "_source_urls",
    "_status",
    "_reason",
    "_evidence",
)


def _part_geometry_data(part) -> dict[str, object]:
    """Return only physical fields that can change a mapped lens body.

    Provenance prose and analytic excitation calibration are deliberately not
    geometry.  Future pole-profile/material fields are retained by name so a
    newly editable mechanical property cannot silently reuse an old map.
    """

    result: dict[str, object] = {}
    for raw_name, value in dict(getattr(part, "data", {})).items():
        name = str(raw_name)
        lowered = name.lower()
        if lowered.endswith(_PART_GEOMETRY_METADATA_SUFFIXES):
            continue
        if (lowered.startswith("field_") and lowered != "field_source_key") or lowered.endswith(
            "_excitation_percent"
        ):
            continue
        if (
            name in _PART_GEOMETRY_FIELDS
            or any(token in lowered for token in _PART_GEOMETRY_TOKENS)
        ):
            result[name] = value
    return result


def _part_is_descendant_of(
    part,
    lens_key: str,
    parts_by_key: Mapping[str, object],
) -> bool:
    """Return whether ``part`` belongs to this lens, excluding nested lenses."""

    current = part
    visited: set[str] = set()
    while getattr(current, "parent_key", None):
        parent_key = str(current.parent_key)
        if parent_key == lens_key:
            return True
        if parent_key in visited:
            return False
        visited.add(parent_key)
        parent = parts_by_key.get(parent_key)
        if parent is None:
            return False
        if str(dict(parent.data).get("mechanical_profile", "")) == (
            "magnetic_lens_assembly"
        ):
            # A separately excited lens nested in the same package owns its
            # own map and must not contaminate its parent's binding.
            return False
        current = parent
    return False


def _part_is_shared_with_lens(part, lens_key: str) -> bool:
    """Return whether manifest metadata assigns one body to this lens too.

    The integrated C1/C2 pole-piece cartridge is physically shared, while the
    TOML tree can express only one structural parent.  Its
    ``mechanical_overlap_group`` explicitly names the other magnetic assembly;
    honouring that relation prevents a C2 field map from surviving a cartridge
    geometry edit merely because C1 is the structural parent.
    """

    data = dict(getattr(part, "data", {}))
    overlap_group = str(data.get("mechanical_overlap_group", "")).strip()
    return (str(lens_key) in data.get("magnetic_lens_keys", ())
            or overlap_group == f"{str(lens_key)}_assembly")


def _overlaps_support(item, support: tuple[float, float]) -> bool:
    if len(support) != 2:
        return False
    return (
        float(getattr(item, "end_z_mm")) >= support[0]
        and float(getattr(item, "start_z_mm")) <= support[1]
    )


def _is_field_structure_part(part, lens_key: str) -> bool:
    from temsim.magnetic_circuits import is_custom_mechanical_part
    if is_custom_mechanical_part(part):
        return False
    key = str(getattr(part, "key", "")).lower()
    data = dict(getattr(part, "data", {}))
    profile = str(data.get("mechanical_profile", ""))
    if profile == "magnetic_lens_assembly":
        # A child lens inside the Objective package (for example the Mini
        # Condenser) has an independent excitation and independent map.
        return False
    if profile in _FIELD_STRUCTURE_PROFILES:
        return True
    if str(data.get("field_source_key", "")) == str(lens_key):
        return True
    return any(
        token in key
        for token in ("pole", "housing", "yoke", "excitation_coil", "cartridge")
    )


def _lens_assembly_payload(
    assembly,
    lens_key: str,
    support: tuple[float, ...],
) -> dict[str, object]:
    """Describe only the selected hardware that can affect one lens map."""

    if assembly is None:
        return {"status": "assembly_unavailable"}
    all_parts = tuple(getattr(assembly, "parts", ()))
    parts_by_key = {
        str(getattr(part, "key", "")): part for part in all_parts
    }
    lens_part = parts_by_key.get(str(lens_key))
    from temsim.magnetic_circuits import belongs_to_circuit, circuit_channels, is_custom_mechanical_part, optical_owner
    channels = set(circuit_channels(parts_by_key, str(lens_key)))
    module_key = str(getattr(lens_part, "module_key", ""))
    module = next(
        (
            item
            for item in getattr(assembly, "modules", ())
            if str(getattr(item, "key", "")) == module_key
        ),
        None,
    )
    related_parts = []
    for part in all_parts:
        if is_custom_mechanical_part(part):
            continue
        key = str(getattr(part, "key", ""))
        belongs = (
            key == lens_key
            or _part_is_descendant_of(part, str(lens_key), parts_by_key)
            or _part_is_shared_with_lens(part, str(lens_key))
            or belongs_to_circuit(part, str(lens_key), parts_by_key, channels)
        )
        if not belongs:
            continue
        if key != lens_key and not _is_field_structure_part(part, lens_key):
            continue
        physical_data = _part_geometry_data(part)
        parent = parts_by_key.get(getattr(part, "parent_key", None))
        if parent is not None and physical_data.get("mechanical_profile") in {"magnetic_lens_yoke", "magnetic_excitation_coil"}:
            from temsim.magnetic_geometry import objective_layer_intervals_mm
            intervals = objective_layer_intervals_mm(parent.data, parent.start_z_mm, physical_data["mechanical_profile"])
            if intervals:
                physical_data["material_intervals_mm"] = intervals
        if physical_data.get("mechanical_profile") == "magnetic_excitation_coil":
            physical_data.setdefault("field_source_key", optical_owner(part, parts_by_key))
        related_parts.append({
            "definition_id": getattr(part, "definition_id", ""),
            "module_key": str(getattr(part, "module_key", "")),
            "key": key,
            "parent_key": getattr(part, "parent_key", None),
            "start_z_mm": float(getattr(part, "start_z_mm")),
            "center_z_mm": float(getattr(part, "center_z_mm")),
            "end_z_mm": float(getattr(part, "end_z_mm")),
            "length_mm": float(getattr(part, "length_mm")),
            "data": physical_data,
        })
    support_pair = tuple(float(value) for value in support)
    bores = tuple(
        _canonical(segment)
        for segment in getattr(assembly, "vacuum_bore_segments", ())
        if _overlaps_support(segment, support_pair)
    )
    liners = tuple(
        _canonical(segment)
        for segment in getattr(assembly, "vacuum_liner_segments", ())
        if _overlaps_support(segment, support_pair)
    )
    return {
        "magnetic_circuit_topology": dict(getattr(lens_part, "data", {})).get("magnetic_circuit_topology", ""),
        "magnetic_circuit_channels": tuple(sorted(channels)),
        "module": (
            {
                "type": str(getattr(module, "type", "")),
                "key": str(getattr(module, "key", "")),
                "source_file": str(getattr(module, "source_file", "")),
            }
            if module is not None
            else {"key": module_key}
        ),
        "parts": tuple(related_parts),
        "overlapping_vacuum_bore_segments": bores,
        "overlapping_vacuum_liner_segments": liners,
    }


def lens_geometry_binding(
    state,
    lens_key: str,
    native_provider=None,
) -> LensGeometryBinding:
    """Fingerprint all geometry that makes a field map physically applicable.

    Excitation percentage is intentionally excluded: imported maps are scaled
    from their declared reference excitation at runtime.  Pole geometry, lens
    Z, field support, the owning module, its magnetic structure and only the
    vacuum segments overlapping that support are included.  Unrelated lenses
    elsewhere in the assembly cannot stale this map.
    """

    key = str(lens_key)
    provider = native_provider
    if provider is None:
        provider = next(
            (item for item in getattr(state, "lenses", ()) if item.key == key),
            None,
        )
    if provider is None:
        raise FieldMapError(f"Unknown lens field provider: {key}")
    geometry_source = getattr(provider, "lens", provider)
    attributes = {
        name: float(getattr(geometry_source, name))
        for name in _GEOMETRY_ATTRIBUTES
        if hasattr(geometry_source, name)
        and isinstance(getattr(geometry_source, name), (int, float))
    }
    for name in _GEOMETRY_ATTRIBUTES:
        if name not in attributes and hasattr(provider, name):
            value = getattr(provider, name)
            if isinstance(value, (int, float)):
                attributes[name] = float(value)
    try:
        support = tuple(float(value) for value in provider.field_support_mm())
    except (AttributeError, TypeError, ValueError):
        support = ()
    assembly = getattr(state, "_resolved_assembly", None)
    lens_assembly = _lens_assembly_payload(assembly, key, support)
    descriptor = getattr(state, "lens_field_map_descriptors", {}).get(key, {})
    if descriptor.get("solver") in {"axisymmetric_linear_fem", "axisymmetric_nonlinear_fem"} and assembly is not None:
        own_parts = lens_assembly["parts"]
        from temsim.part_materials import is_magnetostatic_body
        field_parts = [part for part in own_parts if is_magnetostatic_body(part["data"])
                       or part["data"].get("mechanical_profile") == "magnetic_excitation_coil"]
        neighbours = []
        if field_parts:
            lower = min(part["start_z_mm"] for part in field_parts)
            upper = max(part["end_z_mm"] for part in field_parts)
            centre, half = .5*(lower+upper), .5*(upper-lower)*float(descriptor.get("padding_factor", 2))
            own_keys = {part["key"] for part in own_parts}
            for part in assembly.parts:
                if (part.key not in own_keys and part.start_z_mm < centre+half and part.end_z_mm > centre-half
                        and is_magnetostatic_body(part.data)):
                    data = _part_geometry_data(part)
                    parent = next((p for p in assembly.parts if p.key == part.parent_key), None)
                    if parent is not None and data.get("mechanical_profile") == "magnetic_lens_yoke":
                        from temsim.magnetic_geometry import objective_layer_intervals_mm
                        intervals = objective_layer_intervals_mm(parent.data, parent.start_z_mm, data["mechanical_profile"])
                        if intervals:
                            data["material_intervals_mm"] = intervals
                    neighbours.append({"key": part.key, "start_z_mm": part.start_z_mm, "end_z_mm": part.end_z_mm,
                                       "data": data})
        # Nearby passive magnetic material affects the selected coil response.
        # Include its geometry in the identity, not only the owning lens shell.
        lens_assembly["magnetostatic_neighbours"] = tuple(neighbours)
    assembly_fingerprint = _fingerprint(lens_assembly)
    geometry = {
        "lens_key": key,
        "lens_geometry": attributes,
        "field_support_mm": support,
        "lens_assembly": lens_assembly,
        "assembly_fingerprint": assembly_fingerprint,
    }
    canonical_json = json.dumps(
        _canonical(geometry),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return LensGeometryBinding(
        lens_key=key,
        geometry_fingerprint=sha256(canonical_json.encode("utf-8")).hexdigest(),
        assembly_fingerprint=assembly_fingerprint,
        canonical_geometry_json=canonical_json,
    )


@dataclass(frozen=True, slots=True)
class FieldMapValidation:
    grid_shape: tuple[int, ...]
    minimum_spacing_m: float
    maximum_spacing_ratio: float
    rms_field_t: float
    rms_divergence_t_per_m: float
    maximum_divergence_t_per_m: float
    relative_divergence: float
    divergence_tolerance: float

    @property
    def divergence_within_tolerance(self) -> bool:
        return self.relative_divergence <= self.divergence_tolerance


@dataclass(frozen=True, slots=True)
class FieldMapProvenance:
    kind: str
    source_path: str
    source_sha256: str
    source_note: str
    coordinate_units: str = "m"
    field_units: str = "T"

    def __post_init__(self) -> None:
        if self.kind not in {"measured", "fem"}:
            raise FieldMapError("Field-map provenance must be measured or fem")
        if self.coordinate_units != "m" or self.field_units != "T":
            raise FieldMapError("Runtime field maps must use explicit SI m/T units")
        if not self.source_sha256 or len(self.source_sha256) != 64:
            raise FieldMapError("Field-map source SHA-256 is required")


@dataclass(frozen=True, slots=True)
class MagneticFieldMap:
    """One validated axisymmetric or Cartesian magnetic-field map."""

    map_type: str
    axes_m: tuple[np.ndarray, ...]
    components_t: tuple[np.ndarray, ...]
    registration: CoordinateRegistration
    geometry_fingerprint: str
    reference_excitation_percent: float
    reference_polarity: int
    provenance: FieldMapProvenance
    divergence_tolerance: float = 0.05
    validation: FieldMapValidation = field(init=False)
    content_fingerprint: str = field(init=False)
    _interpolators: tuple[RegularGridInterpolator, ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        kind = str(self.map_type).lower()
        if kind not in {"axisymmetric_rz", "cartesian_xyz"}:
            raise FieldMapError(
                "Field-map type must be axisymmetric_rz or cartesian_xyz"
            )
        expected_axes = 2 if kind == "axisymmetric_rz" else 3
        expected_components = 2 if kind == "axisymmetric_rz" else 3
        if len(self.axes_m) != expected_axes or len(
            self.components_t
        ) != expected_components:
            raise FieldMapError("Field-map axes/components do not match its type")
        axes = tuple(
            _immutable_array(axis, ndim=1, name=f"field axis {index}")
            for index, axis in enumerate(self.axes_m)
        )
        if any(axis.size < 2 or np.any(np.diff(axis) <= 0.0) for axis in axes):
            raise FieldMapError("Field-map axes must be strictly increasing")
        if kind == "axisymmetric_rz" and float(axes[0][0]) < 0.0:
            raise FieldMapError("Axisymmetric field radius cannot be negative")
        shape = tuple(axis.size for axis in axes)
        components = tuple(
            _immutable_array(component, ndim=expected_axes, name="field component")
            for component in self.components_t
        )
        if any(component.shape != shape for component in components):
            raise FieldMapError("Field-map component shapes must match the grid")
        reference = float(self.reference_excitation_percent)
        if not math.isfinite(reference) or reference <= 0.0:
            raise FieldMapError("Reference excitation percent must be positive")
        if int(self.reference_polarity) not in {-1, 1}:
            raise FieldMapError("Reference field-map polarity must be +1 or -1")
        tolerance = float(self.divergence_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise FieldMapError("Field-map divergence tolerance is invalid")
        object.__setattr__(self, "map_type", kind)
        object.__setattr__(self, "axes_m", axes)
        object.__setattr__(self, "components_t", components)
        object.__setattr__(self, "reference_excitation_percent", reference)
        object.__setattr__(self, "reference_polarity", int(self.reference_polarity))
        validation = _field_map_validation(
            kind, axes, components, divergence_tolerance=tolerance
        )
        object.__setattr__(self, "validation", validation)
        digest = sha256()
        digest.update(repr((kind, self.geometry_fingerprint,
                            reference, int(self.reference_polarity))).encode())
        for array in (self.registration.origin_array_m,
                      self.registration.rotation_array, *axes, *components):
            digest.update(repr(array.shape).encode())
            digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
        object.__setattr__(self, "content_fingerprint", digest.hexdigest())
        object.__setattr__(
            self,
            "_interpolators",
            tuple(
                RegularGridInterpolator(
                    axes,
                    component,
                    method="linear",
                    bounds_error=False,
                    fill_value=0.0,
                )
                for component in components
            ),
        )

    @property
    def field_support_mm(self) -> tuple[float, float]:
        # Project the entire map volume, not just its centre line.  A tilted
        # cylinder/box has a transverse contribution to its global Z extent.
        row = self.registration.rotation_array[2]
        origin_z = float(self.registration.origin_array_m[2])
        if self.map_type == "axisymmetric_rz":
            radial_extent = float(self.axes_m[0][-1]) * math.hypot(row[0], row[1])
            ends = row[2] * self.axes_m[1][[0, -1]]
            lo, hi = float(np.min(ends)) - radial_extent, float(np.max(ends)) + radial_extent
        else:
            projections = [coefficient * axis[[0, -1]]
                           for coefficient, axis in zip(row, self.axes_m)]
            lo = sum(float(np.min(value)) for value in projections)
            hi = sum(float(np.max(value)) for value in projections)
        return (origin_z + lo) * 1.0e3, (origin_z + hi) * 1.0e3

    def field_at_global_positions_t(self, positions_m) -> np.ndarray:
        global_positions = np.asarray(positions_m, dtype=float)
        original_shape = global_positions.shape
        if original_shape[-1:] != (3,):
            raise FieldMapError("Field-map query positions must end in XYZ")
        local = self.registration.positions_global_to_local_m(
            global_positions.reshape(-1, 3)
        )
        # SI <-> display-unit conversions and rigid transforms can round an
        # exact boundary point a few ULP outside the grid. Snap only that
        # roundoff neighbourhood; genuinely outside points still return zero.
        scale = np.maximum(np.max(np.abs(global_positions.reshape(-1, 3)), axis=1),
                           np.max(np.abs(self.registration.origin_array_m)))
        def interpolation_points(points):
            points = points.copy()
            for dimension, axis in enumerate(self.axes_m):
                tolerance = 8*np.finfo(float).eps*np.maximum(scale, np.max(np.abs(axis)))
                near = ((points[:, dimension] >= axis[0]-tolerance)
                        & (points[:, dimension] <= axis[-1]+tolerance))
                points[near, dimension] = np.clip(points[near, dimension], axis[0], axis[-1])
            return points
        if self.map_type == "cartesian_xyz":
            points = interpolation_points(local)
            local_field = np.column_stack(
                tuple(interpolator(points) for interpolator in self._interpolators)
            )
        else:
            radius = np.hypot(local[:, 0], local[:, 1])
            points = interpolation_points(np.column_stack((radius, local[:, 2])))
            br = self._interpolators[0](points)
            bz = self._interpolators[1](points)
            inverse_radius = np.divide(
                1.0,
                radius,
                out=np.zeros_like(radius),
                where=radius > 0.0,
            )
            local_field = np.column_stack(
                (br * local[:, 0] * inverse_radius,
                 br * local[:, 1] * inverse_radius,
                 bz)
            )
        global_field = self.registration.vectors_local_to_global(local_field)
        return global_field.reshape(original_shape)


@dataclass(frozen=True, slots=True)
class AxialFieldCalibrationResult:
    """Read-only scale/registration fit to sourced axial Bz measurements."""

    axial_shift_m: float
    field_scale: float
    measured_z_global_m: np.ndarray
    measured_bz_t: np.ndarray
    fitted_bz_t: np.ndarray
    residual_bz_t: np.ndarray
    rms_residual_t: float
    maximum_absolute_residual_t: float
    relative_rms_residual: float
    measurement_reference: str
    measurement_fingerprint: str
    fitted_reference_excitation_percent: float
    fitted_reference_polarity: int

    def __post_init__(self) -> None:
        arrays = tuple(
            _immutable_array(values, ndim=1, name="axial calibration array")
            for values in (
                self.measured_z_global_m,
                self.measured_bz_t,
                self.fitted_bz_t,
                self.residual_bz_t,
            )
        )
        if len({value.shape for value in arrays}) != 1 or arrays[0].size < 3:
            raise FieldMapError(
                "Axial field calibration needs at least three matching points"
            )
        if not self.measurement_reference.strip():
            raise FieldMapError("Axial measurements require a source reference")
        for name, value in (
            ("axial shift", self.axial_shift_m),
            ("field scale", self.field_scale),
            ("RMS residual", self.rms_residual_t),
            ("maximum residual", self.maximum_absolute_residual_t),
            ("relative RMS residual", self.relative_rms_residual),
            (
                "fitted reference excitation",
                self.fitted_reference_excitation_percent,
            ),
        ):
            if not math.isfinite(float(value)):
                raise FieldMapError(f"Axial calibration {name} must be finite")
        if abs(float(self.field_scale)) <= np.finfo(float).tiny:
            raise FieldMapError("Axial calibration field scale cannot be zero")
        if self.fitted_reference_excitation_percent <= 0.0:
            raise FieldMapError(
                "Fitted reference excitation percent must be positive"
            )
        if int(self.fitted_reference_polarity) not in {-1, 1}:
            raise FieldMapError("Fitted reference polarity must be +1 or -1")
        object.__setattr__(self, "measured_z_global_m", arrays[0])
        object.__setattr__(self, "measured_bz_t", arrays[1])
        object.__setattr__(self, "fitted_bz_t", arrays[2])
        object.__setattr__(self, "residual_bz_t", arrays[3])


def fit_axial_field_map_calibration(
    field_map: MagneticFieldMap,
    measured_z_global_m,
    measured_bz_t,
    *,
    maximum_axial_shift_m: float,
    measurement_reference: str,
    weights=None,
) -> AxialFieldCalibrationResult:
    """Fit map scale and axial origin to measured Bz points.

    Measurements must describe the same excitation as the imported map's
    declared reference state.  The fit changes neither the source file nor
    any material/coil parameter; it returns only a scale, registration shift
    and residual report.
    """

    z = np.asarray(measured_z_global_m, dtype=float)
    measured = np.asarray(measured_bz_t, dtype=float)
    if (
        z.ndim != 1
        or measured.shape != z.shape
        or z.size < 3
        or not np.all(np.isfinite(z))
        or not np.all(np.isfinite(measured))
        or np.any(np.diff(z) <= 0.0)
    ):
        raise FieldMapError(
            "Axial Bz calibration points must be finite, matching and ordered"
        )
    limit = float(maximum_axial_shift_m)
    if not math.isfinite(limit) or limit < 0.0:
        raise FieldMapError("Maximum axial calibration shift is invalid")
    if not str(measurement_reference).strip():
        raise FieldMapError("Axial measurements require a source reference")
    if weights is None:
        weight = np.ones_like(z)
    else:
        weight = np.asarray(weights, dtype=float)
        if (
            weight.shape != z.shape
            or not np.all(np.isfinite(weight))
            or np.any(weight <= 0.0)
        ):
            raise FieldMapError("Axial calibration weights must be positive")
    weight = weight / float(np.sum(weight))

    def model_at_shift(shift_m: float) -> np.ndarray:
        positions = np.zeros((z.size, 3), dtype=float)
        positions[:, 2] = z - float(shift_m)
        return field_map.field_at_global_positions_t(positions)[:, 2]

    def scale_and_loss(shift_m: float) -> tuple[float, float]:
        model = model_at_shift(shift_m)
        denominator = float(np.sum(weight * model * model))
        if denominator <= np.finfo(float).tiny:
            return 0.0, math.inf
        scale = float(np.sum(weight * model * measured) / denominator)
        residual = scale * model - measured
        return scale, float(np.sum(weight * residual * residual))

    if limit == 0.0:
        shift = 0.0
    else:
        optimisation = minimize_scalar(
            lambda candidate: scale_and_loss(float(candidate))[1],
            bounds=(-limit, limit),
            method="bounded",
            options={"xatol": max(limit * 1.0e-10, 1.0e-15)},
        )
        if not optimisation.success or not math.isfinite(float(optimisation.fun)):
            raise FieldMapError("Axial field calibration did not converge")
        shift = float(optimisation.x)
    scale, _loss = scale_and_loss(shift)
    if abs(scale) <= np.finfo(float).tiny:
        raise FieldMapError("Measured Bz points do not overlap a non-zero map")
    model = model_at_shift(shift)
    fitted = scale * model
    residual = fitted - measured
    rms = float(np.sqrt(np.sum(weight * residual * residual)))
    measured_rms = float(np.sqrt(np.sum(weight * measured * measured)))
    polarity_sign = 1 if scale > 0.0 else -1
    measurement_fingerprint = _fingerprint({
        "z_global_m": z,
        "bz_t": measured,
        "weights": weight,
        "reference": str(measurement_reference),
    })
    return AxialFieldCalibrationResult(
        axial_shift_m=shift,
        field_scale=scale,
        measured_z_global_m=z,
        measured_bz_t=measured,
        fitted_bz_t=fitted,
        residual_bz_t=residual,
        rms_residual_t=rms,
        maximum_absolute_residual_t=float(np.max(np.abs(residual))),
        relative_rms_residual=rms / max(measured_rms, np.finfo(float).tiny),
        measurement_reference=str(measurement_reference),
        measurement_fingerprint=measurement_fingerprint,
        fitted_reference_excitation_percent=(
            field_map.reference_excitation_percent / abs(scale)
        ),
        fitted_reference_polarity=(
            field_map.reference_polarity * polarity_sign
        ),
    )


def calibrated_field_map(
    field_map: MagneticFieldMap,
    calibration: AxialFieldCalibrationResult,
) -> MagneticFieldMap:
    """Return an in-memory calibrated map; never modify the source file."""

    origin = list(field_map.registration.origin_global_m)
    origin[2] += float(calibration.axial_shift_m)
    note = field_map.provenance.source_note.strip()
    calibration_note = (
        "axial Bz calibration "
        f"{calibration.measurement_fingerprint}; "
        f"RMS residual {calibration.rms_residual_t:.9g} T; "
        f"source {calibration.measurement_reference}"
    )
    return MagneticFieldMap(
        map_type=field_map.map_type,
        axes_m=field_map.axes_m,
        components_t=field_map.components_t,
        registration=CoordinateRegistration(
            origin_global_m=tuple(origin),
            rotation_local_to_global=(
                field_map.registration.rotation_local_to_global
            ),
        ),
        geometry_fingerprint=field_map.geometry_fingerprint,
        reference_excitation_percent=(
            calibration.fitted_reference_excitation_percent
        ),
        reference_polarity=calibration.fitted_reference_polarity,
        provenance=FieldMapProvenance(
            kind=field_map.provenance.kind,
            source_path=field_map.provenance.source_path,
            source_sha256=field_map.provenance.source_sha256,
            source_note=f"{note}; {calibration_note}" if note else calibration_note,
        ),
        divergence_tolerance=field_map.divergence_tolerance,
    )


def _spacing_summary(axes: tuple[np.ndarray, ...]) -> tuple[float, float]:
    spacings = np.concatenate(tuple(np.diff(axis) for axis in axes))
    minimum = float(np.min(spacings))
    ratio = float(np.max(spacings) / minimum)
    return minimum, ratio


def _field_map_validation(
    map_type: str,
    axes: tuple[np.ndarray, ...],
    components: tuple[np.ndarray, ...],
    *,
    divergence_tolerance: float,
) -> FieldMapValidation:
    if map_type == "cartesian_xyz":
        bx, by, bz = components
        divergence = (
            np.gradient(bx, axes[0], axis=0, edge_order=1)
            + np.gradient(by, axes[1], axis=1, edge_order=1)
            + np.gradient(bz, axes[2], axis=2, edge_order=1)
        )
        field_squared = bx * bx + by * by + bz * bz
    else:
        radius, z_axis = axes
        br, bz = components
        radial_flux = radius[:, None] * br
        radial_term = np.empty_like(br)
        radial_term[1:] = np.gradient(
            radial_flux, radius, axis=0, edge_order=1
        )[1:] / radius[1:, None]
        if radius[0] == 0.0:
            radial_term[0] = 2.0 * np.gradient(
                br, radius, axis=0, edge_order=1
            )[0]
        else:
            radial_term[0] = np.gradient(
                radial_flux, radius, axis=0, edge_order=1
            )[0] / radius[0]
        divergence = radial_term + np.gradient(
            bz, z_axis, axis=1, edge_order=1
        )
        field_squared = br * br + bz * bz
    minimum_spacing, spacing_ratio = _spacing_summary(axes)
    rms_field = float(np.sqrt(np.mean(field_squared)))
    rms_divergence = float(np.sqrt(np.mean(divergence * divergence)))
    maximum_divergence = float(np.max(np.abs(divergence)))
    scale = max(rms_field / minimum_spacing, np.finfo(float).tiny)
    return FieldMapValidation(
        grid_shape=tuple(component for component in components[0].shape),
        minimum_spacing_m=minimum_spacing,
        maximum_spacing_ratio=spacing_ratio,
        rms_field_t=rms_field,
        rms_divergence_t_per_m=rms_divergence,
        maximum_divergence_t_per_m=maximum_divergence,
        relative_divergence=rms_divergence / scale,
        divergence_tolerance=float(divergence_tolerance),
    )


def _metadata_from_npz(document) -> dict[str, object]:
    if "metadata_json" not in document:
        return {}
    raw = np.asarray(document["metadata_json"])
    if raw.size != 1:
        raise FieldMapError("metadata_json must contain one JSON document")
    value = raw.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise FieldMapError("Field-map metadata_json must be an object")
    return parsed


def _grid_from_tidy_csv(path: Path):
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    names = tuple(table.dtype.names or ())
    axisymmetric = ("r_m", "z_m", "br_t", "bz_t")
    cartesian = ("x_m", "y_m", "z_m", "bx_t", "by_t", "bz_t")
    required = axisymmetric if set(axisymmetric) <= set(names) else cartesian
    if not set(required) <= set(names):
        raise FieldMapError(
            "CSV needs r_m,z_m,br_t,bz_t or x_m,y_m,z_m,bx_t,by_t,bz_t"
        )
    axis_names = required[:2] if required is axisymmetric else required[:3]
    component_names = required[len(axis_names):]
    axes = tuple(np.unique(np.asarray(table[name], dtype=float)) for name in axis_names)
    expected_rows = int(np.prod([axis.size for axis in axes]))
    if table.size != expected_rows:
        raise FieldMapError("CSV field map must be a complete rectangular grid")
    lookup = {
        tuple(float(table[name][index]) for name in axis_names): index
        for index in range(table.size)
    }
    mesh = np.meshgrid(*axes, indexing="ij")
    keys = zip(*(values.reshape(-1) for values in mesh), strict=True)
    order = np.asarray([lookup[tuple(key)] for key in keys], dtype=int)
    components = tuple(
        np.asarray(table[name], dtype=float)[order].reshape(
            tuple(axis.size for axis in axes)
        )
        for name in component_names
    )
    return (
        "axisymmetric_rz" if required is axisymmetric else "cartesian_xyz",
        axes,
        components,
        {},
    )


def load_magnetic_field_map(
    path: str | Path,
    *,
    geometry_binding: LensGeometryBinding,
    provenance_kind: str | None = None,
    registration: CoordinateRegistration | None = None,
    reference_excitation_percent: float | None = None,
    reference_polarity: int | None = None,
    source_note: str = "",
    divergence_tolerance: float = 0.05,
    require_divergence: bool = False,
) -> MagneticFieldMap:
    """Load an SI-labelled NPZ/CSV map and bind it to the current geometry."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FieldMapError(f"Magnetic field map does not exist: {source}")
    digest = sha256(source.read_bytes()).hexdigest()
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as document:
            metadata = _metadata_from_npz(document)
            map_type = str(metadata.get("map_type", "")).lower()
            if not map_type:
                if {"r_m", "z_m", "br_t", "bz_t"} <= set(document.files):
                    map_type = "axisymmetric_rz"
                elif {
                    "x_m", "y_m", "z_m", "bx_t", "by_t", "bz_t"
                } <= set(document.files):
                    map_type = "cartesian_xyz"
            if map_type == "axisymmetric_rz":
                axes = (document["r_m"], document["z_m"])
                components = (document["br_t"], document["bz_t"])
            elif map_type == "cartesian_xyz":
                axes = (document["x_m"], document["y_m"], document["z_m"])
                components = (document["bx_t"], document["by_t"], document["bz_t"])
            else:
                raise FieldMapError("NPZ field-map arrays do not identify a supported type")
            axes = tuple(np.array(value, copy=True) for value in axes)
            components = tuple(np.array(value, copy=True) for value in components)
    elif source.suffix.lower() == ".csv":
        map_type, axes, components, metadata = _grid_from_tidy_csv(source)
    else:
        raise FieldMapError("Magnetic field maps must be .npz or tidy .csv")

    embedded_fingerprint = str(metadata.get("geometry_fingerprint", "")).strip()
    if embedded_fingerprint and embedded_fingerprint != (
        geometry_binding.geometry_fingerprint
    ):
        raise FieldMapGeometryMismatch(
            "Imported field map metadata targets a different lens geometry"
        )
    kind = str(provenance_kind or metadata.get("provenance_kind", "")).lower()
    reference = (
        reference_excitation_percent
        if reference_excitation_percent is not None
        else metadata.get("reference_excitation_percent")
    )
    if reference is None:
        raise FieldMapError(
            "Field-map reference_excitation_percent must be supplied explicitly"
        )
    polarity = (
        reference_polarity
        if reference_polarity is not None
        else metadata.get("reference_polarity", 1)
    )
    if registration is None:
        origin = metadata.get("origin_global_m", (0.0, 0.0, 0.0))
        rotation = metadata.get(
            "rotation_local_to_global",
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
        registration = CoordinateRegistration(
            origin_global_m=tuple(float(value) for value in origin),
            rotation_local_to_global=tuple(
                tuple(float(value) for value in row) for row in rotation
            ),
        )
    result = MagneticFieldMap(
        map_type=map_type,
        axes_m=axes,
        components_t=components,
        registration=registration,
        geometry_fingerprint=geometry_binding.geometry_fingerprint,
        reference_excitation_percent=float(reference),
        reference_polarity=int(polarity),
        provenance=FieldMapProvenance(
            kind=kind,
            source_path=str(source),
            source_sha256=digest,
            source_note=str(source_note or metadata.get("source_note", "")),
        ),
        divergence_tolerance=float(divergence_tolerance),
    )
    if require_divergence and not result.validation.divergence_within_tolerance:
        raise FieldMapError(
            "Field-map divergence residual exceeds its configured tolerance: "
            f"{result.validation.relative_divergence:.6g} > "
            f"{result.validation.divergence_tolerance:.6g}"
        )
    return result


def _runtime_lens_state(provider):
    return getattr(provider, "lens", provider)


def _runtime_excitation(provider) -> tuple[bool, float, int]:
    source = _runtime_lens_state(provider)
    return (
        bool(getattr(source, "enabled", True)),
        float(getattr(source, "percent", 100.0)),
        int(getattr(source, "polarity", 1)),
    )


def field_map_descriptor(
    field_map: MagneticFieldMap,
    binding: LensGeometryBinding,
) -> dict[str, object]:
    """Return the JSON-safe loader contract copied into worker snapshots."""

    return {
        "source_path": field_map.provenance.source_path,
        "source_sha256": field_map.provenance.source_sha256,
        "content_fingerprint": field_map.content_fingerprint,
        "source_note": field_map.provenance.source_note,
        "provenance_kind": field_map.provenance.kind,
        "map_type": field_map.map_type,
        "geometry_fingerprint": binding.geometry_fingerprint,
        "assembly_fingerprint": binding.assembly_fingerprint,
        "reference_excitation_percent": (
            field_map.reference_excitation_percent
        ),
        "reference_polarity": field_map.reference_polarity,
        "origin_global_m": list(field_map.registration.origin_global_m),
        "rotation_local_to_global": [
            list(row)
            for row in field_map.registration.rotation_local_to_global
        ],
        "divergence_tolerance": field_map.divergence_tolerance,
        "require_divergence": False,
    }


def _load_descriptor_map(
    descriptor: Mapping,
    binding: LensGeometryBinding,
) -> MagneticFieldMap:
    if str(descriptor.get("geometry_fingerprint", "")) != (
        binding.geometry_fingerprint
    ) or str(descriptor.get("assembly_fingerprint", "")) != (
        binding.assembly_fingerprint
    ):
        raise FieldMapGeometryMismatch(
            "Saved field-map descriptor targets an earlier lens assembly"
        )
    path = Path(str(descriptor.get("source_path", ""))).expanduser().resolve()
    if not path.is_file():
        raise FieldMapError("Saved magnetic field-map source is unavailable")
    actual_sha = sha256(path.read_bytes()).hexdigest()
    if actual_sha != str(descriptor.get("source_sha256", "")):
        raise FieldMapError(
            "Saved magnetic field-map source changed after it was bound"
        )
    result = load_magnetic_field_map(
        path,
        geometry_binding=binding,
        provenance_kind=str(descriptor.get("provenance_kind", "")),
        registration=CoordinateRegistration(
            origin_global_m=tuple(
                float(value)
                for value in descriptor.get(
                    "origin_global_m", (0.0, 0.0, 0.0)
                )
            ),
            rotation_local_to_global=tuple(
                tuple(float(value) for value in row)
                for row in descriptor.get(
                    "rotation_local_to_global",
                    ((1.0, 0.0, 0.0),
                     (0.0, 1.0, 0.0),
                     (0.0, 0.0, 1.0)),
                )
            ),
        ),
        reference_excitation_percent=float(
            descriptor["reference_excitation_percent"]
        ),
        reference_polarity=int(descriptor.get("reference_polarity", 1)),
        source_note=str(descriptor.get("source_note", "")),
        divergence_tolerance=float(
            descriptor.get("divergence_tolerance", 0.05)
        ),
        require_divergence=bool(
            descriptor.get("require_divergence", False)
        ),
    )
    expected = descriptor.get("content_fingerprint")
    if expected is not None and result.content_fingerprint != expected:
        raise FieldMapError("Saved field-map numeric content or registration changed")
    return result


def _provider_geometry_token(
    state,
    lens_key: str,
    native_provider,
    *,
    binding: LensGeometryBinding | None = None,
) -> tuple:
    source = _runtime_lens_state(native_provider)
    values = tuple(
        (name, float(getattr(source, name)))
        for name in _GEOMETRY_ATTRIBUTES
        if hasattr(source, name)
        and isinstance(getattr(source, name), (int, float))
    )
    provider_values = tuple(
        (name, float(getattr(native_provider, name)))
        for name in _GEOMETRY_ATTRIBUTES
        if not hasattr(source, name)
        and hasattr(native_provider, name)
        and isinstance(getattr(native_provider, name), (int, float))
    )
    try:
        support = tuple(
            float(value) for value in native_provider.field_support_mm()
        )
    except (AttributeError, TypeError, ValueError):
        support = ()
    descriptor = getattr(state, "lens_field_map_descriptors", {}).get(
        str(lens_key), None
    )
    registry = getattr(state, "_lens_field_map_bindings", None)
    imported = registry.get(str(lens_key)) if isinstance(registry, dict) else None
    current_binding = binding or lens_geometry_binding(
        state, lens_key, native_provider
    )
    return (
        id(source),
        str(getattr(state, "simulation_mode", "custom")),
        values,
        provider_values,
        support,
        current_binding.geometry_fingerprint,
        _fingerprint(descriptor) if isinstance(descriptor, Mapping) else "",
        id(imported),
    )


@dataclass(frozen=True, slots=True)
class MappedLensFieldProvider:
    lens_key: str
    field_map: MagneticFieldMap
    native_provider: object
    binding: LensGeometryBinding
    model_status: str = "measured_or_fem_geometry_bound"
    excitation_scaling: str = "linear_assumption"
    circuit_sources: tuple = ()
    circuit_operating_point: tuple = ()

    def field_support_mm(self, *_args) -> tuple[float, float]:
        return self.field_map.field_support_mm

    def excitation_scale(self) -> float:
        if self.excitation_scaling == "joint_nonlinear_operating_point":
            point = tuple(_runtime_excitation(source) for source in self.circuit_sources)
            if point != self.circuit_operating_point:
                raise FieldMapError("Nonlinear circuit operating point changed; resolve a new joint field")
            return 1.0
        enabled, percent, polarity = _runtime_excitation(self.native_provider)
        if not enabled:
            return 0.0
        if self.excitation_scaling == "fixed_operating_point" and (
            not math.isclose(percent, self.field_map.reference_excitation_percent, rel_tol=0, abs_tol=1e-10)
            or polarity != self.field_map.reference_polarity
        ):
            raise FieldMapError("A saturation-dependent field map is valid only at its reference excitation and polarity; provide a new operating-point map")
        return float(
            percent / self.field_map.reference_excitation_percent
            * polarity / self.field_map.reference_polarity
        )

    def field_at_global_positions_t(self, positions_m) -> np.ndarray:
        return self.excitation_scale() * self.field_map.field_at_global_positions_t(positions_m)

    def magnetic_field_t(self, z_mm) -> np.ndarray:
        z = np.asarray(z_mm, dtype=float)
        positions = np.zeros(z.shape + (3,), dtype=float)
        positions[..., 2] = z * 1.0e-3
        return self.field_at_global_positions_t(positions)[..., 2]


@dataclass(frozen=True, slots=True)
class GeometryAwareAnalyticFieldProvider:
    """Current geometry-bound wrapper around the existing analytic profile.

    It deliberately does not pretend to solve the edited ferromagnetic body.
    Pole shape/gap/bore/tip and support are captured in ``binding`` so every
    geometry edit constructs a new provider identity.  The native Gaussian is
    used only as an explicitly provisional engineering fallback.
    """

    lens_key: str
    native_provider: object
    binding: LensGeometryBinding
    fallback_reason: str
    model_status: str = "provisional_geometry_bound_analytic_not_fem"

    def field_support_mm(self, *args) -> tuple[float, float]:
        """Return native support, with a legacy analytic-profile fallback.

        Older round-lens providers expose only ``magnetic_field_t`` plus their
        Gaussian scale.  Wrapping one of those providers must not make the
        propagation contract stricter than it was before field maps were
        introduced.  Reconstruct a finite support from the same profile data
        when possible; otherwise retain the historical all-z support.
        """

        sigma_cutoff = float(args[0]) if args else 7.0
        support_method = getattr(self.native_provider, "field_support_mm", None)
        if callable(support_method):
            try:
                support = support_method(*args)
            except TypeError:
                support = support_method()
            return tuple(float(value) for value in support)

        source = _runtime_lens_state(self.native_provider)
        centre_mm = float(getattr(source, "z_mm", 0.0))
        scale_mm = abs(float(getattr(source, "a_mm", 0.0)))
        gaussian = tuple(getattr(source, "gaussian", ()) or ())
        if gaussian and scale_mm > 0.0:
            lower = min(
                centre_mm
                + float(getattr(term, "offset", 0.0)) * scale_mm
                - sigma_cutoff
                * abs(float(getattr(term, "sigma", 0.0)))
                * scale_mm
                for term in gaussian
            )
            upper = max(
                centre_mm
                + float(getattr(term, "offset", 0.0)) * scale_mm
                + sigma_cutoff
                * abs(float(getattr(term, "sigma", 0.0)))
                * scale_mm
                for term in gaussian
            )
            return float(lower), float(upper)

        length_mm = abs(float(getattr(
            source,
            "effective_length_mm",
            getattr(source, "length_mm", 0.0),
        )))
        if length_mm > 0.0:
            sigma_mm = length_mm / 2.355
            return (
                centre_mm - sigma_cutoff * sigma_mm,
                centre_mm + sigma_cutoff * sigma_mm,
            )
        if scale_mm > 0.0:
            return (
                centre_mm - sigma_cutoff * scale_mm,
                centre_mm + sigma_cutoff * scale_mm,
            )
        return float("-inf"), float("inf")

    def magnetic_field_t(self, z_mm) -> np.ndarray:
        return np.asarray(self.native_provider.magnetic_field_t(z_mm), dtype=float)

    def field_at_global_positions_t(self, positions_m) -> np.ndarray:
        """Use the first-order divergence-free off-axis expansion.

        ``Br = -r/2 dBz/dz`` follows from ``div(B)=0`` near an axisymmetric
        axis.  Higher radial orders require a matching measured/FEM map.
        """

        positions = np.asarray(positions_m, dtype=float)
        if positions.shape[-1:] != (3,):
            raise FieldMapError("Analytic field query positions must end in XYZ")
        flat = positions.reshape(-1, 3)
        z_mm = flat[:, 2] * 1.0e3
        support = self.field_support_mm()
        support_span_mm = abs(support[1] - support[0])
        if not np.isfinite(support_span_mm):
            source = _runtime_lens_state(self.native_provider)
            support_span_mm = max(
                abs(float(getattr(source, "a_mm", 0.0))),
                abs(float(getattr(source, "effective_length_mm", 0.0))),
                abs(float(getattr(source, "length_mm", 0.0))),
                1.0,
            )
        scale_mm = max(support_span_mm, 1.0) * 1.0e-6
        plus = np.asarray(self.magnetic_field_t(z_mm + scale_mm), dtype=float)
        minus = np.asarray(self.magnetic_field_t(z_mm - scale_mm), dtype=float)
        derivative_t_per_m = (plus - minus) / (2.0 * scale_mm * 1.0e-3)
        bx = -0.5 * flat[:, 0] * derivative_t_per_m
        by = -0.5 * flat[:, 1] * derivative_t_per_m
        bz = self.magnetic_field_t(z_mm)
        return np.column_stack((bx, by, bz)).reshape(positions.shape)


def bind_imported_lens_field_map(
    state,
    lens_key: str,
    field_map: MagneticFieldMap,
    *,
    native_provider=None,
) -> LensGeometryBinding:
    """Install one imported map only after exact geometry validation."""

    binding = lens_geometry_binding(state, lens_key, native_provider)
    if field_map.geometry_fingerprint != binding.geometry_fingerprint:
        raise FieldMapGeometryMismatch(
            "Field map is not bound to the current pole-piece/lens assembly"
        )
    registry = getattr(state, "_lens_field_map_bindings", None)
    if not isinstance(registry, dict):
        registry = {}
        setattr(state, "_lens_field_map_bindings", registry)
    registry[str(lens_key)] = field_map
    descriptors = getattr(state, "lens_field_map_descriptors", None)
    if not isinstance(descriptors, dict):
        descriptors = {}
        setattr(state, "lens_field_map_descriptors", descriptors)
    descriptors[str(lens_key)] = field_map_descriptor(field_map, binding)
    cache = getattr(state, "_runtime_lens_field_provider_cache", None)
    if isinstance(cache, dict):
        cache.pop(str(lens_key), None)
    return binding


def clear_imported_lens_field_map(state, lens_key: str) -> None:
    registry = getattr(state, "_lens_field_map_bindings", None)
    if isinstance(registry, dict):
        registry.pop(str(lens_key), None)
    descriptors = getattr(state, "lens_field_map_descriptors", None)
    if isinstance(descriptors, dict):
        descriptors.pop(str(lens_key), None)
    cache = getattr(state, "_runtime_lens_field_provider_cache", None)
    if isinstance(cache, dict):
        cache.pop(str(lens_key), None)


def _validate_shared_linear_recipes(state, lens_key, binding):
    """Shared linear responses must use the same material/domain operator."""
    descriptors = getattr(state, "lens_field_map_descriptors", {})
    selected = descriptors.get(lens_key, {})
    if selected.get("solver") != "axisymmetric_linear_fem":
        return
    channels = json.loads(binding.canonical_geometry_json)["lens_assembly"].get("magnetic_circuit_channels", ())
    if len(channels) < 2:
        return

    def operator_settings(recipe):
        return (
            float(recipe["relative_permeability"]),
            tuple(sorted((key, float(value)) for key, value in recipe.get("material_permeabilities", {}).items())),
            int(recipe.get("radial_nodes", 40)), int(recipe.get("axial_nodes", 80)),
            float(recipe.get("padding_factor", 2)),
        )

    expected = operator_settings(selected)
    for channel in channels:
        other = descriptors.get(channel, {})
        if other.get("solver") == "axisymmetric_linear_fem" and operator_settings(other) != expected:
            raise FieldMapError(
                f"Shared circuit channels {lens_key} and {channel} require identical material, mesh and boundary settings; "
                "only their excitation may differ"
            )


def resolve_runtime_lens_field_provider(state, lens_key: str, native_provider):
    """Return a matching imported map or an explicit provisional fallback."""

    key = str(lens_key)
    from temsim.simulation_modes import mode_key, uses_field_maps
    selected_mode = mode_key(state)
    current = lens_geometry_binding(state, key, native_provider)
    descriptor = getattr(state, "lens_field_map_descriptors", {}).get(key, {})
    from temsim.excitation_calibration import validate_excitation_recipe
    validate_excitation_recipe(descriptor)
    if uses_field_maps(state) and descriptor.get("solver") in {"axisymmetric_linear_fem", "axisymmetric_nonlinear_fem"}:
        from temsim.geometry_effects import field_geometry_admission
        field_geometry_admission(state, current, descriptor)
    if selected_mode == "nonlinear_material" and descriptor.get("solver") != "axisymmetric_nonlinear_fem":
        raise FieldMapError(f"{key}: Nonlinear Material Field requires an explicit B-H recipe")
    if uses_field_maps(state) and descriptor.get("solver") == "axisymmetric_nonlinear_fem":
        from temsim.physics.nonlinear_circuits import resolve_nonlinear_provider
        return resolve_nonlinear_provider(state, key, native_provider, current)
    # Check before returning a cached basis too: another channel may have edited
    # the shared material/domain settings since this channel's last field solve.
    if uses_field_maps(state):
        _validate_shared_linear_recipes(state, key, current)
    token = _provider_geometry_token(
        state, key, native_provider, binding=current
    )
    provider_cache = getattr(
        state, "_runtime_lens_field_provider_cache", None
    )
    if not isinstance(provider_cache, dict):
        provider_cache = {}
        setattr(state, "_runtime_lens_field_provider_cache", provider_cache)
    cached = provider_cache.get(key)
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == token:
        if isinstance(cached[1], MappedLensFieldProvider):
            cached[1].excitation_scale()
        return cached[1]
    registry = getattr(state, "_lens_field_map_bindings", None)
    imported = registry.get(key) if isinstance(registry, dict) else None
    diagnostics = getattr(state, "_field_provider_diagnostics", None)
    if not isinstance(diagnostics, dict):
        diagnostics = {}
        setattr(state, "_field_provider_diagnostics", diagnostics)
    descriptor = getattr(state, "lens_field_map_descriptors", {}).get(
        key, None
    )
    if not uses_field_maps(state):
        provider = GeometryAwareAnalyticFieldProvider(
            key, native_provider, current, f"selected_{selected_mode}_model",
            model_status="ideal_paraxial" if selected_mode == "ideal" else "parameterized_analytic",
        )
        diagnostics[key] = {"mode": selected_mode, "model_status": provider.model_status,
                            "geometry_fingerprint": current.geometry_fingerprint}
        provider_cache[key] = (token, provider)
        return provider
    if selected_mode == "linear_geometry" and (
        not isinstance(descriptor, Mapping) or descriptor.get("solver") != "axisymmetric_linear_fem"
    ):
        raise FieldMapError(f"{key}: Linear Geometry Field requires an explicit linear FEM recipe; analytic fallback is disabled")
    descriptor_error = None
    if isinstance(descriptor, Mapping) and descriptor.get("solver") == "axisymmetric_linear_fem":
        from temsim.physics.axisymmetric_magnetostatics import solve_geometry_field_map
        # Explicit generated models never silently fall back after an invalid
        # geometry or a failed numerical solve.
        imported = solve_geometry_field_map(current, descriptor)
    if imported is None and isinstance(descriptor, Mapping):
        try:
            imported = _load_descriptor_map(descriptor, current)
            if not isinstance(registry, dict):
                registry = {}
                setattr(state, "_lens_field_map_bindings", registry)
            registry[key] = imported
        except (FieldMapError, OSError, KeyError, TypeError, ValueError) as exc:
            descriptor_error = str(exc)
    if imported is not None and (
        imported.geometry_fingerprint == current.geometry_fingerprint
    ):
        topology = json.loads(current.canonical_geometry_json)["lens_assembly"].get("magnetic_circuit_topology")
        scaling = "fixed_operating_point" if topology == "monolithic_saturated_insert" else "linear_assumption"
        provider = MappedLensFieldProvider(key, imported, native_provider, current, excitation_scaling=scaling)
        provider.excitation_scale()
        diagnostics[key] = {
            "mode": "imported_field_map",
            "model_status": provider.model_status,
            "geometry_fingerprint": current.geometry_fingerprint,
            "assembly_fingerprint": current.assembly_fingerprint,
            "source_sha256": imported.provenance.source_sha256,
            "descriptor_fingerprint": _fingerprint(descriptor),
            "excitation_scaling": scaling,
            "divergence_within_tolerance": (
                imported.validation.divergence_within_tolerance
            ),
        }
        provider_cache[key] = (
            _provider_geometry_token(
                state, key, native_provider, binding=current
            ),
            provider,
        )
        return provider
    reason = (
        "imported_map_geometry_mismatch"
        if imported is not None
        else "saved_map_unavailable_or_stale"
        if descriptor_error is not None
        else "no_matching_measured_or_fem_map"
    )
    provider = GeometryAwareAnalyticFieldProvider(
        key, native_provider, current, reason
    )
    diagnostics[key] = {
        "mode": "provisional_analytic_fallback",
        "model_status": provider.model_status,
        "reason": reason,
        "geometry_fingerprint": current.geometry_fingerprint,
        "assembly_fingerprint": current.assembly_fingerprint,
        "stale_source_sha256": (
            imported.provenance.source_sha256 if imported is not None else None
        ),
        "descriptor_error": descriptor_error,
    }
    provider_cache[key] = (
        _provider_geometry_token(
            state, key, native_provider, binding=current
        ),
        provider,
    )
    return provider


def runtime_axial_magnetic_field_t(
    state,
    lens_key: str,
    native_provider,
    z_mm,
) -> tuple[np.ndarray, object]:
    """Unified hook used by the axial solver and specimen-local field query."""

    provider = resolve_runtime_lens_field_provider(
        state, lens_key, native_provider
    )
    return np.asarray(provider.magnetic_field_t(z_mm), dtype=float), provider


def active_mapped_providers(state) -> tuple[MappedLensFieldProvider, ...]:
    """Resolve only configured maps; ordinary Gaussian runs keep their fast path."""
    from temsim.component_keys import CONDENSER_LENS_KEYS
    from temsim.simulation_modes import mode_key, uses_field_maps, linear_mode_issues, nonlinear_mode_issues
    if not uses_field_maps(state):
        return ()
    if mode_key(state) == "linear_geometry":
        issues = linear_mode_issues(state)
        if issues:
            raise FieldMapError("Linear Geometry Field is not ready: " + "; ".join(issues[:4]))
    if mode_key(state) == "nonlinear_material":
        issues = nonlinear_mode_issues(state)
        if issues:
            raise FieldMapError("Nonlinear Material Field is not ready: " + "; ".join(issues[:4]))

    keys = set(getattr(state, "lens_field_map_descriptors", {}) or {})
    keys.update(getattr(state, "_lens_field_map_bindings", {}) or {})
    result = []
    for lens in getattr(state, "lenses", ()):
        if lens.key not in keys or not bool(getattr(lens, "enabled", True)):
            continue
        native = state.condenser_system[lens.key] if lens.key in CONDENSER_LENS_KEYS else lens
        provider = resolve_runtime_lens_field_provider(state, lens.key, native)
        if isinstance(provider, MappedLensFieldProvider):
            result.append(provider)
    return tuple(result)


def supports_axisymmetric_reduction(field_map: MagneticFieldMap) -> bool:
    """Only centred, axially aligned RZ maps admit the scalar thin-lens model."""
    return (
        field_map.map_type == "axisymmetric_rz"
        and float(field_map.axes_m[0][0]) == 0.0
        and np.allclose(field_map.registration.origin_array_m[:2], 0.0, rtol=0.0, atol=1e-15)
        and np.allclose(field_map.registration.rotation_array[:2, 2], 0.0, rtol=0.0, atol=1e-12)
    )


@dataclass(frozen=True, slots=True)
class FrozenMappedField:
    """Map and excitation frozen into a restartable propagation plan."""

    lens_key: str
    field_map: MagneticFieldMap
    scale: float

    def __post_init__(self):
        if not math.isfinite(float(self.scale)):
            raise FieldMapError("Mapped excitation scale must be finite")
        object.__setattr__(self, "scale", float(self.scale))

    def maximum_step_mm(self, requested_step_mm: float, momentum_kg_m_s: float) -> float:
        """Resolve interpolation cells and bound the nominal angular advance."""
        if not (math.isfinite(requested_step_mm) and requested_step_mm > 0
                and math.isfinite(momentum_kg_m_s) and momentum_kg_m_s > 0):
            raise FieldMapError("Mapped transport step and momentum must be finite and positive")
        spacing_mm = min(float(np.min(np.diff(axis)))
                         for axis in self.field_map.axes_m) * 1e3
        peak_t = abs(self.scale)*sum(float(np.max(np.abs(component)))
                                    for component in self.field_map.components_t)
        bend_step_mm = .02*momentum_kg_m_s/(1.602176634e-19*max(peak_t,1e-30))*1e3
        return min(float(requested_step_mm), spacing_mm*.5, bend_step_mm)

    @classmethod
    def from_provider(cls, provider: MappedLensFieldProvider):
        return cls(provider.lens_key, provider.field_map, provider.excitation_scale())

    @property
    def fingerprint(self) -> str:
        return _fingerprint((self.lens_key, self.field_map.content_fingerprint, self.scale))

    def field_at_global_positions_t(self, positions_m):
        return self.scale * self.field_map.field_at_global_positions_t(positions_m)
