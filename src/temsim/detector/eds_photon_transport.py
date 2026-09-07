"""Geometry-aware EDS photon transport with explicit evidence boundaries.

The installed generic EDS definition contains measured angular acceptance but
does not contain proprietary crystal size, sensor distance or holder CAD.  The
default path therefore reports aggregate angular collection without inventing
a sensor face.  Exact per-segment hit points become available only when the
caller supplies sourced detector surfaces.  Pole pieces, support grids and
user-supplied holder solids can still shadow or attenuate those paths.

Geometry uses millimetres, photon energy uses electronvolts and directions are
right-handed unit vectors in the global microscope frame (+Z downstream).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import math
from typing import Iterable, Protocol

import numpy as np

from temsim.detector.eds_geometry import EDSDetectorArrayGeometry
from temsim.specimen.scene import SpecimenScene


# One acquisition only; different requests never share geometry or materials.
_PHOTON_GEOMETRY_CACHE_ENTRIES = 1024


class PhotonAttenuationMaterial(Protocol):
    key: str
    density_g_cm3: float

    def mass_attenuation_cm2_g(self, photon_energy_ev: float) -> float: ...


def _unit_vector(values, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite XYZ vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name} cannot be zero")
    return vector / norm


@dataclass(frozen=True, slots=True)
class EDSPhotonRay:
    photon_id: str
    origin_mm: tuple[float, float, float]
    direction: tuple[float, float, float]
    energy_ev: float
    statistical_weight: float = 1.0
    source_key: str = "sample"
    transition: str = ""
    emission_key: str = ""

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin_mm, dtype=float)
        direction = _unit_vector(self.direction, "EDS photon direction")
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("EDS photon origin must be a finite XYZ vector")
        if not str(self.photon_id).strip():
            raise ValueError("EDS photon ID cannot be empty")
        if not math.isfinite(float(self.energy_ev)) or self.energy_ev <= 0.0:
            raise ValueError("EDS photon energy must be finite and positive")
        if (
            not math.isfinite(float(self.statistical_weight))
            or self.statistical_weight < 0.0
        ):
            raise ValueError("EDS photon weight must be finite and non-negative")
        object.__setattr__(
            self, "origin_mm", tuple(float(value) for value in origin)
        )
        object.__setattr__(
            self, "direction", tuple(float(value) for value in direction)
        )


@dataclass(frozen=True, slots=True)
class PhotonMaterialInterval:
    component_key: str
    entry_distance_mm: float
    exit_distance_mm: float
    material: PhotonAttenuationMaterial | None
    hard_shadow: bool
    geometry_status: str
    provenance: str

    @property
    def path_length_mm(self) -> float:
        return max(self.exit_distance_mm - self.entry_distance_mm, 0.0)


class PhotonOccluder(Protocol):
    key: str

    def ray_intervals(
        self,
        origin_mm: np.ndarray,
        direction: np.ndarray,
        maximum_distance_mm: float,
    ) -> tuple[PhotonMaterialInterval, ...]: ...


def _quadratic_roots(a: float, b: float, c: float) -> tuple[float, ...]:
    tolerance = 64.0 * np.finfo(float).eps * max(
        abs(a), abs(b), abs(c), 1.0
    )
    if abs(a) <= tolerance:
        if abs(b) <= tolerance:
            return ()
        return (-c / b,)
    discriminant = b * b - 4.0 * a * c
    if discriminant < -tolerance:
        return ()
    root = math.sqrt(max(discriminant, 0.0))
    q = -0.5 * (b + math.copysign(root, b))
    if abs(q) <= tolerance:
        return (-b / (2.0 * a),)
    first = q / a
    second = c / q
    return (first,) if math.isclose(first, second) else (first, second)


def _radial_surface_roots(
    origin: np.ndarray,
    direction: np.ndarray,
    radius_at_z_zero_mm: float,
    radius_slope: float,
) -> tuple[float, ...]:
    """Roots of r(t) = radius_at_z_zero + radius_slope*z(t)."""

    ox, oy, oz = (float(value) for value in origin)
    dx, dy, dz = (float(value) for value in direction)
    radius_constant = radius_at_z_zero_mm + radius_slope * oz
    radius_rate = radius_slope * dz
    return _quadratic_roots(
        dx * dx + dy * dy - radius_rate * radius_rate,
        2.0 * (ox * dx + oy * dy - radius_constant * radius_rate),
        ox * ox + oy * oy - radius_constant * radius_constant,
    )


@dataclass(frozen=True, slots=True)
class AxisymmetricPolePieceOccluder:
    """Piecewise conical pole solid outside its axial electron bore."""

    key: str
    z_min_mm: float
    z_max_mm: float
    face_z_mm: float
    shoulder_z_mm: float
    bore_radius_mm: float
    tip_radius_mm: float
    body_radius_mm: float
    material: PhotonAttenuationMaterial | None = None
    hard_shadow: bool = True
    geometry_status: str = "assembly_geometry"
    provenance: str = "selected instrument TOML"

    def __post_init__(self) -> None:
        values = (
            self.z_min_mm,
            self.z_max_mm,
            self.face_z_mm,
            self.shoulder_z_mm,
            self.bore_radius_mm,
            self.tip_radius_mm,
            self.body_radius_mm,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Pole-piece photon geometry must be finite")
        if not self.z_min_mm < self.z_max_mm:
            raise ValueError("Pole-piece axial interval is reversed")
        if not self.z_min_mm <= self.face_z_mm <= self.z_max_mm:
            raise ValueError("Pole face must lie inside its axial interval")
        if not self.z_min_mm <= self.shoulder_z_mm <= self.z_max_mm:
            raise ValueError("Pole shoulder must lie inside its axial interval")
        if not 0.0 < self.bore_radius_mm < self.tip_radius_mm < self.body_radius_mm:
            raise ValueError("Pole bore, tip and body radii are invalid")
        if math.isclose(self.face_z_mm, self.shoulder_z_mm):
            raise ValueError("Pole nose must have non-zero axial length")

    def _outer_radius(self, z_mm: float) -> float:
        face = float(self.face_z_mm)
        shoulder = float(self.shoulder_z_mm)
        z = float(z_mm)
        nose_min, nose_max = sorted((face, shoulder))
        if nose_min <= z <= nose_max:
            fraction = (z - face) / (shoulder - face)
            return self.tip_radius_mm + fraction * (
                self.body_radius_mm - self.tip_radius_mm
            )
        return self.body_radius_mm

    def _distance_interval_for_z(
        self, origin: np.ndarray, direction: np.ndarray, left_z: float, right_z: float
    ) -> tuple[float, float] | None:
        if abs(float(direction[2])) <= 1.0e-15:
            if left_z <= float(origin[2]) <= right_z:
                return -math.inf, math.inf
            return None
        values = (
            (left_z - float(origin[2])) / float(direction[2]),
            (right_z - float(origin[2])) / float(direction[2]),
        )
        return min(values), max(values)

    def ray_intervals(
        self,
        origin_mm: np.ndarray,
        direction: np.ndarray,
        maximum_distance_mm: float,
    ) -> tuple[PhotonMaterialInterval, ...]:
        origin = np.asarray(origin_mm, dtype=float)
        vector = _unit_vector(direction, "photon direction")
        maximum = float(maximum_distance_mm)
        if not math.isfinite(maximum) or maximum <= 0.0:
            return ()
        z_breaks = sorted({
            float(self.z_min_mm),
            float(self.z_max_mm),
            float(self.face_z_mm),
            float(self.shoulder_z_mm),
        })
        intervals: list[tuple[float, float]] = []
        for z_left, z_right in zip(z_breaks[:-1], z_breaks[1:], strict=True):
            distance_bounds = self._distance_interval_for_z(
                origin, vector, z_left, z_right
            )
            if distance_bounds is None:
                continue
            lower = max(float(distance_bounds[0]), 0.0)
            upper = min(float(distance_bounds[1]), maximum)
            if upper <= lower:
                continue
            outer_left = self._outer_radius(z_left)
            outer_right = self._outer_radius(z_right)
            slope = (outer_right - outer_left) / (z_right - z_left)
            constant = outer_left - slope * z_left
            candidates = [lower, upper]
            candidates.extend(
                root
                for root in _radial_surface_roots(
                    origin, vector, self.bore_radius_mm, 0.0
                )
                if lower < root < upper
            )
            candidates.extend(
                root
                for root in _radial_surface_roots(origin, vector, constant, slope)
                if lower < root < upper
            )
            points = sorted(set(float(value) for value in candidates))
            for start, stop in zip(points[:-1], points[1:], strict=True):
                middle = 0.5 * (start + stop)
                point = origin + middle * vector
                radius = math.hypot(float(point[0]), float(point[1]))
                if (
                    radius >= self.bore_radius_mm - 1.0e-12
                    and radius <= self._outer_radius(float(point[2])) + 1.0e-12
                ):
                    intervals.append((start, stop))
        if not intervals:
            return ()
        intervals.sort()
        merged = [list(intervals[0])]
        for start, stop in intervals[1:]:
            if start <= merged[-1][1] + 1.0e-10:
                merged[-1][1] = max(merged[-1][1], stop)
            else:
                merged.append([start, stop])
        return tuple(
            PhotonMaterialInterval(
                component_key=self.key,
                entry_distance_mm=float(start),
                exit_distance_mm=float(stop),
                material=self.material,
                hard_shadow=bool(self.hard_shadow),
                geometry_status=self.geometry_status,
                provenance=self.provenance,
            )
            for start, stop in merged
        )


@dataclass(frozen=True, slots=True)
class AnnularPlanarLayerOccluder:
    """Finite planar support/holder layer with optional central opening."""

    key: str
    z_start_mm: float
    z_end_mm: float
    outer_radius_mm: float
    inner_radius_mm: float = 0.0
    material: PhotonAttenuationMaterial | None = None
    hard_shadow: bool = False
    geometry_status: str = "user_supplied"
    provenance: str = "user-supplied holder geometry"

    def ray_intervals(self, origin_mm, direction, maximum_distance_mm):
        origin = np.asarray(origin_mm, dtype=float)
        vector = _unit_vector(direction, "photon direction")
        if abs(float(vector[2])) <= 1.0e-15:
            return ()
        distances = sorted((
            (self.z_start_mm - float(origin[2])) / float(vector[2]),
            (self.z_end_mm - float(origin[2])) / float(vector[2]),
        ))
        start = max(distances[0], 0.0)
        stop = min(distances[1], float(maximum_distance_mm))
        if stop <= start:
            return ()
        middle = origin + 0.5 * (start + stop) * vector
        radius = math.hypot(float(middle[0]), float(middle[1]))
        if not self.inner_radius_mm <= radius <= self.outer_radius_mm:
            return ()
        return (
            PhotonMaterialInterval(
                self.key,
                start,
                stop,
                self.material,
                self.hard_shadow,
                self.geometry_status,
                self.provenance,
            ),
        )


def _slab_interval(
    origin: float,
    direction: float,
    lower: float,
    upper: float,
) -> tuple[float, float] | None:
    if abs(float(direction)) <= 1.0e-15:
        return (-math.inf, math.inf) if lower <= origin <= upper else None
    values = ((lower - origin) / direction, (upper - origin) / direction)
    return min(values), max(values)


@dataclass(frozen=True, slots=True)
class FiniteSpecimenOccluder:
    """Oriented finite specimen used for photon self-absorption paths."""

    key: str
    centre_global_mm: tuple[float, float, float]
    size_local_mm: tuple[float, float, float]
    rotation_local_to_global: tuple[tuple[float, float, float], ...]
    envelope_shape: str
    material: PhotonAttenuationMaterial
    geometry_status: str = "active_specimen_scene"
    provenance: str = "user-selected specimen source and orientation"

    def __post_init__(self) -> None:
        centre = np.asarray(self.centre_global_mm, dtype=float)
        size = np.asarray(self.size_local_mm, dtype=float)
        rotation = np.asarray(self.rotation_local_to_global, dtype=float)
        if centre.shape != (3,) or not np.all(np.isfinite(centre)):
            raise ValueError("Specimen photon centre must be finite XYZ")
        if size.shape != (3,) or not np.all(np.isfinite(size)) or np.any(size <= 0.0):
            raise ValueError("Specimen photon dimensions must be positive XYZ")
        if rotation.shape != (3, 3) or not np.allclose(
            rotation.T @ rotation, np.eye(3), rtol=0.0, atol=2.0e-10
        ):
            raise ValueError("Specimen photon registration must be orthonormal")
        if self.envelope_shape not in {"disk", "rectangle"}:
            raise ValueError("Specimen photon envelope must be disk or rectangle")

    def ray_intervals(self, origin_mm, direction, maximum_distance_mm):
        centre = np.asarray(self.centre_global_mm, dtype=float)
        rotation = np.asarray(self.rotation_local_to_global, dtype=float)
        origin = (np.asarray(origin_mm, dtype=float) - centre) @ rotation
        vector = _unit_vector(direction, "photon direction") @ rotation
        half = 0.5 * np.asarray(self.size_local_mm, dtype=float)
        z_interval = _slab_interval(
            float(origin[2]), float(vector[2]), -half[2], half[2]
        )
        if z_interval is None:
            return ()
        if self.envelope_shape == "rectangle":
            lateral = []
            for axis in (0, 1):
                interval = _slab_interval(
                    float(origin[axis]),
                    float(vector[axis]),
                    -half[axis],
                    half[axis],
                )
                if interval is None:
                    return ()
                lateral.append(interval)
            lower = max(z_interval[0], lateral[0][0], lateral[1][0], 0.0)
            upper = min(
                z_interval[1], lateral[0][1], lateral[1][1],
                float(maximum_distance_mm),
            )
        else:
            scaled_origin = origin[:2] / half[:2]
            scaled_vector = vector[:2] / half[:2]
            a = float(np.dot(scaled_vector, scaled_vector))
            b = 2.0 * float(np.dot(scaled_origin, scaled_vector))
            c = float(np.dot(scaled_origin, scaled_origin)) - 1.0
            if a <= 1.0e-30:
                if c > 0.0:
                    return ()
                radial = (-math.inf, math.inf)
            else:
                roots = _quadratic_roots(a, b, c)
                if len(roots) != 2:
                    return ()
                radial = (min(roots), max(roots))
            lower = max(z_interval[0], radial[0], 0.0)
            upper = min(
                z_interval[1], radial[1], float(maximum_distance_mm)
            )
        if upper <= lower + 1.0e-15:
            return ()
        return (
            PhotonMaterialInterval(
                self.key,
                float(lower),
                float(upper),
                self.material,
                False,
                self.geometry_status,
                self.provenance,
            ),
        )


@dataclass(frozen=True, slots=True)
class PlanarEDSDetectorSegment:
    """Sourced physical active face for exact per-segment intersection.

    ``efficiency`` is a dimensionless response relative to the array-wide
    absolute detector quantum efficiency supplied by the caller.  It can
    represent a measured segment-to-segment sensitivity or an energy-local
    surface response.  The two factors are intentionally multiplicative; this
    field must not repeat the global quantum efficiency.
    """

    segment_index: int
    center_mm: tuple[float, float, float]
    normal: tuple[float, float, float]
    u_axis: tuple[float, float, float]
    active_shape: str
    active_size_mm: tuple[float, float]
    efficiency: float = 1.0
    provenance: str = ""
    geometry_status: str = "user_supplied_or_measured"

    def __post_init__(self) -> None:
        normal = _unit_vector(self.normal, "detector normal")
        u_axis = _unit_vector(self.u_axis, "detector U axis")
        if abs(float(np.dot(normal, u_axis))) > 2.0e-10:
            raise ValueError("Detector U axis must be perpendicular to its normal")
        if self.active_shape not in {"circle", "rectangle"}:
            raise ValueError("EDS detector active shape must be circle or rectangle")
        sizes = tuple(float(value) for value in self.active_size_mm)
        if len(sizes) != 2 or min(sizes) <= 0.0 or not all(
            math.isfinite(value) for value in sizes
        ):
            raise ValueError("EDS detector active dimensions must be positive")
        if not 0.0 <= float(self.efficiency) <= 1.0:
            raise ValueError("EDS detector efficiency must be in [0, 1]")
        if not self.provenance.strip():
            raise ValueError("Physical EDS detector surfaces require provenance")

    def intersect(self, ray: EDSPhotonRay) -> tuple[float, np.ndarray] | None:
        origin = np.asarray(ray.origin_mm, dtype=float)
        direction = np.asarray(ray.direction, dtype=float)
        centre = np.asarray(self.center_mm, dtype=float)
        normal = _unit_vector(self.normal, "detector normal")
        denominator = float(np.dot(direction, normal))
        if abs(denominator) <= 1.0e-15:
            return None
        distance = float(np.dot(centre - origin, normal) / denominator)
        if distance <= 0.0:
            return None
        point = origin + distance * direction
        delta = point - centre
        u_axis = _unit_vector(self.u_axis, "detector U axis")
        v_axis = np.cross(normal, u_axis)
        u = float(np.dot(delta, u_axis))
        v = float(np.dot(delta, v_axis))
        if self.active_shape == "circle":
            inside = math.hypot(u, v) <= 0.5 * self.active_size_mm[0]
        else:
            inside = (
                abs(u) <= 0.5 * self.active_size_mm[0]
                and abs(v) <= 0.5 * self.active_size_mm[1]
            )
        return (distance, point) if inside else None


@dataclass(frozen=True, slots=True)
class EDSPhotonPathResult:
    photon: EDSPhotonRay
    detector_segment: int | None
    detector_hit_mm: tuple[float, float, float] | None
    material_intervals: tuple[PhotonMaterialInterval, ...]
    transmission: float
    detected_weight: float
    terminal_status: str
    transport_mode: str


@dataclass(frozen=True, slots=True)
class EDSPhotonTransportResult:
    paths: tuple[EDSPhotonPathResult, ...]
    expected_detected_weight_per_segment: tuple[float, ...]
    aggregate_collection_probability_per_segment: tuple[float, ...]
    geometry_complete: bool
    metrics: dict[str, object]
    expected_detected_weight_per_emission: dict[
        str, tuple[float, ...]
    ] = field(default_factory=dict)
    quadrature_weight_per_emission: dict[str, float] = field(
        default_factory=dict
    )


def _angular_segment(
    direction: np.ndarray,
    geometry: EDSDetectorArrayGeometry,
    solid_angle_sr: float,
) -> int | None:
    takeoff = math.radians(float(geometry.takeoff_angle_deg))
    argument = float(solid_angle_sr) / (4.0 * math.pi * math.cos(takeoff))
    if not 0.0 < argument <= 1.0:
        return None
    half_width = math.asin(argument)
    radial = math.hypot(float(direction[0]), float(direction[1]))
    elevation = math.atan2(-float(direction[2]), radial)
    if abs(elevation - takeoff) > half_width:
        return None
    azimuth = math.degrees(math.atan2(direction[1], direction[0])) % 360.0
    separations = tuple(
        abs((azimuth - centre + 180.0) % 360.0 - 180.0)
        for centre in geometry.azimuth_centers_deg
    )
    return int(np.argmin(separations))


def _interval_transmission(
    interval: PhotonMaterialInterval, energy_ev: float
) -> float:
    if interval.path_length_mm <= 0.0:
        return 1.0
    if interval.hard_shadow:
        return 0.0
    if interval.material is None:
        # Unknown material is never assigned an invented attenuation curve.
        return 1.0
    coefficient = float(interval.material.mass_attenuation_cm2_g(energy_ev))
    optical_depth = (
        float(interval.material.density_g_cm3)
        * coefficient
        * interval.path_length_mm
        * 0.1
    )
    return 0.0 if math.isinf(optical_depth) else math.exp(-max(optical_depth, 0.0))


@dataclass(frozen=True, slots=True)
class _EDSPhotonGeometry:
    detector_segment: int | None
    detector_hit_mm: tuple[float, float, float] | None
    material_intervals: tuple[PhotonMaterialInterval, ...]
    detector_efficiency: float
    transport_mode: str


def _trace_eds_photon_geometry(
    photon: EDSPhotonRay,
    geometry: EDSDetectorArrayGeometry,
    *,
    detector_surfaces: Iterable[PlanarEDSDetectorSegment] = (),
    occluders: Iterable[PhotonOccluder] = (),
    use_analytical_holder_solid_angle: bool = True,
    aggregate_detector_efficiency: float = 1.0,
    _geometry_validated: bool = False,
) -> _EDSPhotonGeometry:
    """Resolve energy-independent geometry without caching photon outcomes."""

    if not _geometry_validated:
        geometry.validate()
    aggregate_efficiency = float(aggregate_detector_efficiency)
    if (
        not math.isfinite(aggregate_efficiency)
        or not 0.0 <= aggregate_efficiency <= 1.0
    ):
        raise ValueError("Aggregate EDS detector efficiency must be in [0, 1]")
    surfaces = tuple(detector_surfaces)
    direction = np.asarray(photon.direction, dtype=float)
    hit_rows = tuple(
        (surface, hit)
        for surface in surfaces
        if (hit := surface.intersect(photon)) is not None
    )
    if hit_rows:
        surface, (maximum_distance, hit_point) = min(
            hit_rows, key=lambda row: row[1][0]
        )
        segment = int(surface.segment_index)
        # Photon statistical weights may already contain the array-wide
        # quantum efficiency.  A sourced face contributes only its relative
        # segment response here.
        efficiency = float(surface.efficiency)
        mode = "sourced_planar_segment_intersection"
        hit = tuple(float(value) for value in hit_point)
    elif surfaces:
        maximum_distance = math.inf
        segment = None
        efficiency = 0.0
        mode = "sourced_planar_segment_intersection"
        hit = None
    else:
        solid_angle = (
            geometry.analytical_holder_solid_angle_sr
            if use_analytical_holder_solid_angle
            else geometry.minimum_unshadowed_solid_angle_sr
        )
        segment = _angular_segment(direction, geometry, solid_angle)
        maximum_distance = 1.0e9
        efficiency = aggregate_efficiency if segment is not None else 0.0
        mode = "aggregate_angular_acceptance_no_sensor_face"
        hit = None
    intervals = tuple(
        interval
        for occluder in occluders
        for interval in occluder.ray_intervals(
            np.asarray(photon.origin_mm, dtype=float),
            direction,
            maximum_distance,
        )
    )
    intervals = tuple(
        sorted(intervals, key=lambda item: item.entry_distance_mm)
    )
    return _EDSPhotonGeometry(segment, hit, intervals, efficiency, mode)


def _apply_photon_geometry(
    photon: EDSPhotonRay, geometry: _EDSPhotonGeometry,
) -> EDSPhotonPathResult:
    """Evaluate attenuation and statistical weight anew for this photon."""

    segment = geometry.detector_segment
    hit = geometry.detector_hit_mm
    intervals = geometry.material_intervals
    efficiency = geometry.detector_efficiency
    mode = geometry.transport_mode
    transmission = 1.0
    blocked_by = None
    for interval in intervals:
        transmission *= _interval_transmission(interval, photon.energy_ev)
        if transmission <= 0.0:
            transmission = 0.0
            blocked_by = interval.component_key
            break
    detected = (
        float(photon.statistical_weight) * transmission * efficiency
        if segment is not None else 0.0
    )
    status = (
        f"blocked_by:{blocked_by}"
        if blocked_by is not None
        else "detected"
        if segment is not None
        else "outside_detector_acceptance"
    )
    return EDSPhotonPathResult(
        photon=photon,
        detector_segment=segment,
        detector_hit_mm=hit,
        material_intervals=intervals,
        transmission=transmission,
        detected_weight=detected,
        terminal_status=status,
        transport_mode=mode,
    )


def trace_eds_photon(
    photon: EDSPhotonRay,
    geometry: EDSDetectorArrayGeometry,
    *,
    detector_surfaces: Iterable[PlanarEDSDetectorSegment] = (),
    occluders: Iterable[PhotonOccluder] = (),
    use_analytical_holder_solid_angle: bool = True,
    aggregate_detector_efficiency: float = 1.0,
    _geometry_validated: bool = False,
) -> EDSPhotonPathResult:
    """Trace one photon without caches or an invented detector face."""

    resolved = _trace_eds_photon_geometry(
        photon, geometry, detector_surfaces=detector_surfaces,
        occluders=occluders,
        use_analytical_holder_solid_angle=use_analytical_holder_solid_angle,
        aggregate_detector_efficiency=aggregate_detector_efficiency,
        _geometry_validated=_geometry_validated,
    )
    return _apply_photon_geometry(photon, resolved)


def detector_quadrature_photons(
    *,
    emission_key: str,
    origin_mm: tuple[float, float, float],
    energy_ev: float,
    expected_emitted_photons: float,
    geometry: EDSDetectorArrayGeometry,
    detector_surfaces: Iterable[PlanarEDSDetectorSegment] = (),
    use_analytical_holder_solid_angle: bool = True,
    detector_efficiency: float = 1.0,
    quadrature_order: int = 1,
    source_key: str = "sample",
    transition: str = "",
) -> tuple[EDSPhotonRay, ...]:
    """Represent one isotropic line emission with deterministic ray weights.

    With sourced detector faces, midpoint area quadrature evaluates
    ``cos(theta) dA / r**2`` directly.  Without faces, the only defensible
    geometry is the measured aggregate solid angle, so equal-solid-angle
    representatives are placed inside its documented angular-band surrogate.
    In either branch the ray weights already contain the isotropic ``1/4pi``
    factor and the caller-supplied array-wide absolute detector quantum
    efficiency.  A sourced surface's ``efficiency`` is applied later, exactly
    once, as an additional relative segment response.
    """

    geometry.validate()
    key = str(emission_key).strip()
    if not key:
        raise ValueError("EDS photon emission key cannot be empty")
    origin = np.asarray(origin_mm, dtype=float)
    emitted = float(expected_emitted_photons)
    efficiency = float(detector_efficiency)
    order = int(quadrature_order)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("EDS photon emission origin must be finite XYZ")
    if not math.isfinite(emitted) or emitted < 0.0:
        raise ValueError("Expected emitted photon weight must be non-negative")
    if not math.isfinite(efficiency) or not 0.0 <= efficiency <= 1.0:
        raise ValueError("EDS detector efficiency must be in [0, 1]")
    if order <= 0:
        raise ValueError("EDS photon quadrature order must be positive")
    if emitted == 0.0 or efficiency == 0.0:
        return ()

    surfaces = tuple(detector_surfaces)
    rays: list[EDSPhotonRay] = []
    if surfaces:
        for surface_index, surface in enumerate(surfaces):
            centre = np.asarray(surface.center_mm, dtype=float)
            normal = _unit_vector(surface.normal, "detector normal")
            u_axis = _unit_vector(surface.u_axis, "detector U axis")
            v_axis = np.cross(normal, u_axis)
            if surface.active_shape == "rectangle":
                sample_rows = []
                cell_area = (
                    float(surface.active_size_mm[0])
                    * float(surface.active_size_mm[1])
                    / (order * order)
                )
                for v_index in range(order):
                    v = (
                        (v_index + 0.5) / order - 0.5
                    ) * float(surface.active_size_mm[1])
                    for u_index in range(order):
                        u = (
                            (u_index + 0.5) / order - 0.5
                        ) * float(surface.active_size_mm[0])
                        sample_rows.append((u, v, cell_area))
            else:
                count = order * order
                radius = 0.5 * float(surface.active_size_mm[0])
                cell_area = math.pi * radius * radius / count
                golden_angle = math.pi * (3.0 - math.sqrt(5.0))
                sample_rows = [
                    (
                        radius * math.sqrt((index + 0.5) / count)
                        * math.cos(index * golden_angle),
                        radius * math.sqrt((index + 0.5) / count)
                        * math.sin(index * golden_angle),
                        cell_area,
                    )
                    for index in range(count)
                ]
            for sample_index, (u, v, area_weight) in enumerate(sample_rows):
                point = centre + u * u_axis + v * v_axis
                displacement = point - origin
                distance = float(np.linalg.norm(displacement))
                if distance <= 0.0:
                    continue
                direction = displacement / distance
                incidence_cosine = max(
                    -float(np.dot(direction, normal)), 0.0
                )
                if incidence_cosine <= 0.0:
                    continue
                angular_weight = (
                    incidence_cosine * float(area_weight) / (distance * distance)
                )
                rays.append(
                    EDSPhotonRay(
                        photon_id=(
                            f"{key}:face{surface_index}:q{sample_index}"
                        ),
                        origin_mm=tuple(float(value) for value in origin),
                        direction=tuple(float(value) for value in direction),
                        energy_ev=float(energy_ev),
                        statistical_weight=(
                            emitted
                            * efficiency
                            * angular_weight
                            / (4.0 * math.pi)
                        ),
                        source_key=str(source_key),
                        transition=str(transition),
                        emission_key=key,
                    )
                )
        return tuple(rays)

    solid_angle = (
        geometry.analytical_holder_solid_angle_sr
        if use_analytical_holder_solid_angle
        else geometry.minimum_unshadowed_solid_angle_sr
    )
    takeoff = math.radians(float(geometry.takeoff_angle_deg))
    argument = solid_angle / (4.0 * math.pi * math.cos(takeoff))
    if not 0.0 < argument <= 1.0:
        raise ValueError("Configured EDS solid angle cannot form an angular band")
    half_width = math.asin(argument)
    sine_lower = math.sin(takeoff - half_width)
    sine_upper = math.sin(takeoff + half_width)
    per_segment_count = order * order
    ray_weight = (
        emitted
        * efficiency
        * solid_angle
        / (4.0 * math.pi)
        / geometry.segment_count
        / per_segment_count
    )
    nominal_azimuth_width = 2.0 * math.pi / geometry.segment_count
    for segment, centre_deg in enumerate(geometry.azimuth_centers_deg):
        centre_azimuth = math.radians(float(centre_deg))
        for elevation_index in range(order):
            sine_elevation = sine_lower + (
                (elevation_index + 0.5) / order
            ) * (sine_upper - sine_lower)
            elevation = math.asin(min(max(sine_elevation, -1.0), 1.0))
            radial = math.cos(elevation)
            for azimuth_index in range(order):
                azimuth = centre_azimuth + (
                    (azimuth_index + 0.5) / order - 0.5
                ) * nominal_azimuth_width * (1.0 - 1.0e-10)
                direction = np.asarray(
                    (
                        radial * math.cos(azimuth),
                        radial * math.sin(azimuth),
                        -math.sin(elevation),
                    ),
                    dtype=float,
                )
                # Installed segments are equally spaced.  For a future
                # irregular array, retain the sourced segment centre instead
                # of silently assigning the quadrature point to its neighbour.
                if _angular_segment(direction, geometry, solid_angle) != segment:
                    direction = np.asarray(
                        (
                            radial * math.cos(centre_azimuth),
                            radial * math.sin(centre_azimuth),
                            -math.sin(elevation),
                        ),
                        dtype=float,
                    )
                rays.append(
                    EDSPhotonRay(
                        photon_id=(
                            f"{key}:segment{segment}:"
                            f"q{elevation_index}_{azimuth_index}"
                        ),
                        origin_mm=tuple(float(value) for value in origin),
                        direction=tuple(float(value) for value in direction),
                        energy_ev=float(energy_ev),
                        statistical_weight=ray_weight,
                        source_key=str(source_key),
                        transition=str(transition),
                        emission_key=key,
                    )
                )
    return tuple(rays)


def objective_pole_occluders_from_state(
    state,
) -> tuple[AxisymmetricPolePieceOccluder, ...]:
    """Build opaque pole geometry from the active assembly, without material curves."""

    assembly = getattr(state, "_resolved_assembly", None)
    if assembly is None:
        return ()
    rows = []
    for key, upper in (
        ("objective_upper_pole", True),
        ("objective_lower_pole", False),
    ):
        try:
            part = assembly.part(key)
        except KeyError:
            continue
        data = dict(part.data)
        required = (
            "mechanical_bore_diameter_mm",
            "mechanical_tip_diameter_mm",
            "mechanical_outer_diameter_mm",
            "pole_nose_axial_length_mm",
        )
        if not all(name in data for name in required):
            continue
        face = float(part.end_z_mm if upper else part.start_z_mm)
        nose = float(data["pole_nose_axial_length_mm"])
        shoulder = face - nose if upper else face + nose
        rows.append(
            AxisymmetricPolePieceOccluder(
                key=key,
                z_min_mm=float(part.start_z_mm),
                z_max_mm=float(part.end_z_mm),
                face_z_mm=face,
                shoulder_z_mm=shoulder,
                bore_radius_mm=0.5 * float(data["mechanical_bore_diameter_mm"]),
                tip_radius_mm=0.5 * float(data["mechanical_tip_diameter_mm"]),
                body_radius_mm=0.5 * float(data["mechanical_outer_diameter_mm"]),
                hard_shadow=True,
                geometry_status=str(
                    data.get(
                        "mechanical_pole_mounting_geometry_status",
                        "selected_assembly_geometry",
                    )
                ),
                provenance=(
                    f"{part.definition_id}; opaque mechanical interception only; "
                    "no unsourced pole-alloy attenuation curve"
                ),
            )
        )
    return tuple(rows)


def support_occluder_for_ray(
    state,
    ray: EDSPhotonRay,
    *,
    scene: SpecimenScene | None = None,
):
    """Resolve the actual mesh cell crossed by a photon, when applicable."""

    scene = scene or SpecimenScene.from_state(
        state, include_eds_materials=True
    )
    if scene.support_material is None:
        return None
    z_start = scene.reference_z_mm + scene.support_top_nm * 1.0e-6
    z_end = scene.reference_z_mm + scene.support_bottom_nm * 1.0e-6
    direction = np.asarray(ray.direction, dtype=float)
    if abs(float(direction[2])) <= 1.0e-15:
        return None
    middle_z = 0.5 * (z_start + z_end)
    distance = (middle_z - float(ray.origin_mm[2])) / float(direction[2])
    if distance <= 0.0:
        return None
    point = np.asarray(ray.origin_mm, dtype=float) + distance * direction
    region = scene.support_region_at_xy(point[0] * 1.0e6, point[1] * 1.0e6)
    if region not in {"bar", "rim"}:
        return None
    return AnnularPlanarLayerOccluder(
        key=f"support:{region}",
        z_start_mm=z_start,
        z_end_mm=z_end,
        outer_radius_mm=0.5 * scene.support_grid.outer_diameter_mm,
        material=scene.support_material,
        hard_shadow=False,
        geometry_status=scene.support_grid.geometry_status,
        provenance=scene.support_grid.geometry_source_url,
    )


def specimen_occluder_from_state(
    state,
    *,
    scene: SpecimenScene | None = None,
) -> FiniteSpecimenOccluder | None:
    """Return the selected finite specimen, without inventing material data."""

    from temsim.specimen.geometry import quaternion_to_matrix

    scene = scene or SpecimenScene.from_state(
        state, include_eds_materials=True
    )
    if scene.is_vacuum or scene.sample_material is None:
        return None
    return FiniteSpecimenOccluder(
        key="sample",
        centre_global_mm=(
            scene.centre_xy_nm[0] * 1.0e-6,
            scene.centre_xy_nm[1] * 1.0e-6,
            scene.reference_z_mm,
        ),
        size_local_mm=(
            scene.size_xy_nm[0] * 1.0e-6,
            scene.size_xy_nm[1] * 1.0e-6,
            scene.thickness_nm * 1.0e-6,
        ),
        rotation_local_to_global=tuple(
            tuple(float(value) for value in row)
            for row in quaternion_to_matrix(
                scene.orientation_quaternion_wxyz
            )
        ),
        envelope_shape=scene.envelope_shape,
        material=scene.sample_material,
        provenance=(
            f"{scene.source_key}; finite specimen envelope and user orientation"
        ),
    )


def transport_eds_photons(
    state,
    photons: Iterable[EDSPhotonRay],
    geometry: EDSDetectorArrayGeometry,
    *,
    detector_surfaces: Iterable[PlanarEDSDetectorSegment] = (),
    holder_occluders: Iterable[PhotonOccluder] = (),
    include_specimen: bool = True,
    include_support: bool = True,
    use_analytical_holder_solid_angle: bool = True,
    aggregate_detector_efficiency: float = 1.0,
    maximum_stored_paths: int | None = None,
) -> EDSPhotonTransportResult:
    """Trace photons through fixed acquisition geometry with a bounded LRU.

    Only exact repeated origin/direction intersections are cached.  Every
    photon retains its own energy, weight, identity and attenuation evaluation.
    The cache is local to this call, so changed sample/material/detector settings
    cannot reuse another acquisition's geometry.
    """

    geometry.validate()
    surfaces = tuple(detector_surfaces)
    if any(
        not 0 <= int(surface.segment_index) < geometry.segment_count
        for surface in surfaces
    ):
        raise ValueError("EDS detector surface segment index is out of range")
    poles = objective_pole_occluders_from_state(state)
    holder = tuple(holder_occluders)
    stored_limit = (
        None if maximum_stored_paths is None else int(maximum_stored_paths)
    )
    if stored_limit is not None and stored_limit < 0:
        raise ValueError("Maximum stored EDS photon paths cannot be negative")
    scene = (
        SpecimenScene.from_state(state, include_eds_materials=True)
        if include_specimen or include_support
        else None
    )
    specimen = (
        specimen_occluder_from_state(state, scene=scene)
        if include_specimen
        else None
    )
    paths: list[EDSPhotonPathResult] = []
    segment_weights = np.zeros(geometry.segment_count, dtype=float)
    emission_segment_weights: dict[str, np.ndarray] = {}
    emission_quadrature_weights: dict[str, float] = {}
    photon_count = 0
    detector_hit_count = 0
    blocked_count = 0
    blocked_by_counts: dict[str, int] = {}
    blocked_input_weight_by_component: dict[str, float] = {}
    outside_count = 0
    specimen_intersection_count = 0
    specimen_path_length_mm = 0.0
    specimen_transmission_sum = 0.0
    specimen_transmission_count = 0
    total_quadrature_weight = 0.0
    total_detected_weight = 0.0
    geometry_cache: OrderedDict[
        tuple[tuple[float, float, float], tuple[float, float, float]],
        _EDSPhotonGeometry,
    ] = OrderedDict()
    for ray in photons:
        photon_count += 1
        total_quadrature_weight += float(ray.statistical_weight)
        emission_key = str(ray.emission_key or ray.photon_id)
        emission_quadrature_weights[emission_key] = (
            emission_quadrature_weights.get(emission_key, 0.0)
            + float(ray.statistical_weight)
        )
        if emission_key not in emission_segment_weights:
            emission_segment_weights[emission_key] = np.zeros(
                geometry.segment_count, dtype=float
            )
        geometry_key = (ray.origin_mm, ray.direction)
        resolved = geometry_cache.get(geometry_key)
        if resolved is None:
            support = (
                support_occluder_for_ray(state, ray, scene=scene)
                if include_support
                else None
            )
            occluders = (
                poles
                + holder
                + ((specimen,) if specimen is not None else ())
                + ((support,) if support is not None else ())
            )
            resolved = _trace_eds_photon_geometry(
                ray,
                geometry,
                detector_surfaces=surfaces,
                occluders=occluders,
                use_analytical_holder_solid_angle=use_analytical_holder_solid_angle,
                aggregate_detector_efficiency=aggregate_detector_efficiency,
                _geometry_validated=True,
            )
            if _PHOTON_GEOMETRY_CACHE_ENTRIES > 0:
                geometry_cache[geometry_key] = resolved
                if len(geometry_cache) > _PHOTON_GEOMETRY_CACHE_ENTRIES:
                    geometry_cache.popitem(last=False)
        else:
            geometry_cache.move_to_end(geometry_key)
        path = _apply_photon_geometry(ray, resolved)
        if stored_limit is None or len(paths) < stored_limit:
            paths.append(path)
        if path.detector_segment is not None:
            segment = int(path.detector_segment)
            segment_weights[segment] += path.detected_weight
            emission_segment_weights[emission_key][segment] += (
                path.detected_weight
            )
        total_detected_weight += float(path.detected_weight)
        detector_hit_count += path.terminal_status == "detected"
        is_blocked = path.terminal_status.startswith("blocked_by:")
        blocked_count += is_blocked
        if is_blocked:
            component = path.terminal_status.split(":", 1)[1]
            blocked_by_counts[component] = (
                blocked_by_counts.get(component, 0) + 1
            )
            blocked_input_weight_by_component[component] = (
                blocked_input_weight_by_component.get(component, 0.0)
                + float(ray.statistical_weight)
            )
        outside_count += (
            path.terminal_status == "outside_detector_acceptance"
        )
        for interval in path.material_intervals:
            if interval.component_key != "sample":
                continue
            specimen_intersection_count += 1
            specimen_path_length_mm += interval.path_length_mm
            specimen_transmission_sum += _interval_transmission(
                interval, path.photon.energy_ev
            )
            specimen_transmission_count += 1
    solid_angle = (
        geometry.analytical_holder_solid_angle_sr
        if use_analytical_holder_solid_angle
        else geometry.minimum_unshadowed_solid_angle_sr
    )
    aggregate = tuple(
        solid_angle / geometry.segment_count / (4.0 * math.pi)
        for _ in range(geometry.segment_count)
    )
    complete = bool(surfaces) and {
        int(surface.segment_index) for surface in surfaces
    } == set(range(geometry.segment_count))
    return EDSPhotonTransportResult(
        paths=tuple(paths),
        expected_detected_weight_per_segment=tuple(
            float(value) for value in segment_weights
        ),
        aggregate_collection_probability_per_segment=aggregate,
        geometry_complete=complete,
        metrics={
            "model": "EDS photon transport v2",
            "coordinate_system": "global right-handed XYZ; +Z downstream",
            "photon_count": int(photon_count),
            "stored_path_count": len(paths),
            "all_paths_stored": stored_limit is None or photon_count <= stored_limit,
            "total_quadrature_weight": float(total_quadrature_weight),
            "quadrature_weight_semantics": (
                "input photon weight after isotropic collection and any "
                "caller-applied global detector quantum efficiency; before "
                "attenuation and sourced-face relative response"
            ),
            "total_detected_weight": float(total_detected_weight),
            "pole_occluder_count": len(poles),
            "holder_occluder_count": len(holder),
            "finite_specimen_attenuation": specimen is not None,
            "detector_surface_count": len(surfaces),
            "surface_relative_response_applied": bool(surfaces),
            "surface_relative_response_values": tuple(
                float(surface.efficiency) for surface in surfaces
            ),
            "detector_efficiency_semantics": (
                "global absolute QE in input/quadrature weight multiplied by "
                "PlanarEDSDetectorSegment.efficiency as relative response"
                if surfaces
                else "aggregate detector efficiency applied once at acceptance"
            ),
            "geometry_complete": complete,
            "detector_hit_count": int(detector_hit_count),
            "blocked_photon_count": int(blocked_count),
            "blocked_by_component_counts": dict(
                sorted(blocked_by_counts.items())
            ),
            "blocked_input_weight_by_component": dict(
                sorted(blocked_input_weight_by_component.items())
            ),
            "outside_acceptance_count": int(outside_count),
            "specimen_intersection_count": int(specimen_intersection_count),
            "specimen_path_length_mm_sum": float(specimen_path_length_mm),
            "mean_specimen_transmission": (
                float(specimen_transmission_sum / specimen_transmission_count)
                if specimen_transmission_count else 1.0
            ),
            "sensor_geometry_status": (
                "sourced_physical_surfaces"
                if complete
                else "unavailable_no_dimensions_in_installed_definition"
            ),
            "fallback": (
                None
                if complete
                else "measured aggregate solid angle; no fabricated sensor hit point"
            ),
            "pole_material_attenuation": (
                "not evaluated; physical pole intersections are opaque shadows"
            ),
            "support_material_attenuation": bool(include_support),
        },
        expected_detected_weight_per_emission={
            key: tuple(float(value) for value in values)
            for key, values in emission_segment_weights.items()
        },
        quadrature_weight_per_emission={
            key: float(value)
            for key, value in emission_quadrature_weights.items()
        },
    )


def photons_from_sample_region(sample_region) -> tuple[EDSPhotonRay, ...]:
    """Integration hook for the existing high-accuracy 3-D sample cache."""

    return photons_from_sample_region_paths(sample_region.photon_paths)


def photons_from_sample_region_paths(paths) -> tuple[EDSPhotonRay, ...]:
    """Convert cached sample-view photon representatives without resampling."""

    return tuple(
        EDSPhotonRay(
            photon_id=f"sample-region:{index}",
            origin_mm=tuple(float(value) for value in path.positions_mm[0]),
            direction=tuple(float(value) for value in path.direction),
            energy_ev=float(path.energy_ev),
            statistical_weight=float(path.emitted_weight),
            source_key=str(path.source_key),
            transition=str(path.transition),
        )
        for index, path in enumerate(paths)
    )
