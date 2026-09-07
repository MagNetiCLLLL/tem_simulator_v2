"""TOML-backed geometry for one sample-adjacent TEM EDS detector array.

The installed detector's solid angle is a physical acceptance quantity. The
mechanical crystal envelope is deliberately kept separate because the public
reference sources do not publish its segment area, sensor distance or package
dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib
from typing import Mapping

from temsim.paths import EDS_DETECTOR_CONFIG_ROOT


EDS_DETECTOR_DEFINITION_FIELD = "eds_detector_definition_file"
EDS_DETECTOR_DEFINITION_FIELDS = (
    "eds_system_key",
    "detector_technology",
    "windowless",
    "segment_count",
    "segment_count_status",
    "azimuth_centers_deg",
    "azimuth_status",
    "takeoff_angle_deg",
    "takeoff_angle_status",
    "takeoff_angle_source",
    "minimum_unshadowed_solid_angle_sr",
    "analytical_double_tilt_holder_solid_angle_sr",
    "solid_angle_status",
    "active_area_status",
    "sample_to_sensor_distance_status",
    "mechanical_envelope_status",
    "mounting_topology",
    "mounting_topology_status",
    "eds_geometry_source",
    "eds_geometry_source_urls",
)


def load_eds_detector_definition(
    file_name: str,
    *,
    root: Path | None = None,
) -> dict[str, object]:
    """Load one non-axial detector definition from the project TOML folder."""

    definition_root = Path(
        EDS_DETECTOR_CONFIG_ROOT if root is None else root
    ).resolve()
    path = (definition_root / str(file_name)).resolve()
    if not path.is_relative_to(definition_root):
        raise ValueError(
            f"EDS detector definition escapes its config root: {file_name}"
        )
    if path.suffix.lower() != ".toml" or not path.is_file():
        raise ValueError(
            f"EDS detector definition does not exist: {file_name}"
        )
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    if int(document.get("format_version", 0)) != 1:
        raise ValueError("Unsupported EDS detector definition format")
    if document.get("definition_type") != "tem_eds_detector":
        raise ValueError("Invalid EDS detector definition type")
    definition = document.get("detector")
    if not isinstance(definition, dict):
        raise ValueError("EDS detector definition must contain [detector]")
    expected = set(EDS_DETECTOR_DEFINITION_FIELDS)
    present = set(definition)
    if missing := sorted(expected - present):
        raise ValueError(
            "Missing EDS detector definition fields: " + ", ".join(missing)
        )
    if unsupported := sorted(present - expected):
        raise ValueError(
            "Unsupported EDS detector definition fields: "
            + ", ".join(unsupported)
        )
    return dict(definition)


def resolve_eds_detector_part_data(
    data: Mapping,
    *,
    root: Path | None = None,
) -> dict[str, object]:
    """Merge one column placement row with its single detector definition."""

    result = dict(data)
    reference = str(result.get(EDS_DETECTOR_DEFINITION_FIELD, "")).strip()
    if not reference:
        return result
    definition = load_eds_detector_definition(reference, root=root)
    conflicts = sorted(set(definition) & set(result))
    if conflicts:
        raise ValueError(
            "EDS product geometry must come only from its definition TOML: "
            + ", ".join(conflicts)
        )
    result.update(definition)
    return result


@dataclass(frozen=True, slots=True)
class EDSDetectorArrayGeometry:
    """Evidence-bearing angular geometry for one installed EDS array."""

    system_key: str
    segment_count: int
    azimuth_centers_deg: tuple[float, ...]
    takeoff_angle_deg: float
    minimum_unshadowed_solid_angle_sr: float
    analytical_holder_solid_angle_sr: float
    windowless: bool

    @classmethod
    def from_part_data(cls, data: Mapping) -> "EDSDetectorArrayGeometry":
        geometry = cls(
            system_key=str(data["eds_system_key"]),
            segment_count=int(data["segment_count"]),
            azimuth_centers_deg=tuple(
                float(value) for value in data["azimuth_centers_deg"]
            ),
            takeoff_angle_deg=float(data["takeoff_angle_deg"]),
            minimum_unshadowed_solid_angle_sr=float(
                data["minimum_unshadowed_solid_angle_sr"]
            ),
            analytical_holder_solid_angle_sr=float(
                data["analytical_double_tilt_holder_solid_angle_sr"]
            ),
            windowless=bool(data["windowless"]),
        )
        geometry.validate()
        return geometry

    def validate(self) -> None:
        if self.segment_count <= 0:
            raise ValueError("EDS segment_count must be positive")
        if len(self.azimuth_centers_deg) != self.segment_count:
            raise ValueError(
                "EDS azimuth_centers_deg must contain one value per segment"
            )
        if not all(
            math.isfinite(value) and 0.0 <= value < 360.0
            for value in self.azimuth_centers_deg
        ):
            raise ValueError("EDS azimuth centers must be finite in [0, 360)")
        rounded = {round(value, 9) for value in self.azimuth_centers_deg}
        if len(rounded) != self.segment_count:
            raise ValueError("EDS azimuth centers must be unique")
        if not (
            math.isfinite(self.takeoff_angle_deg)
            and 0.0 < self.takeoff_angle_deg < 90.0
        ):
            raise ValueError("EDS take-off angle must be in (0, 90) degrees")
        unshadowed = self.minimum_unshadowed_solid_angle_sr
        holder = self.analytical_holder_solid_angle_sr
        if not (
            math.isfinite(unshadowed)
            and math.isfinite(holder)
            and 0.0 < holder <= unshadowed <= 4.0 * math.pi
        ):
            raise ValueError(
                "EDS solid angles must satisfy 0 < holder <= unshadowed <= 4pi"
            )

    @property
    def minimum_unshadowed_solid_angle_per_segment_sr(self) -> float:
        return self.minimum_unshadowed_solid_angle_sr / self.segment_count

    @property
    def analytical_holder_solid_angle_per_segment_sr(self) -> float:
        return self.analytical_holder_solid_angle_sr / self.segment_count

    @property
    def minimum_unshadowed_fraction_of_4pi(self) -> float:
        return self.minimum_unshadowed_solid_angle_sr / (4.0 * math.pi)

    def equivalent_circular_cone_half_angle_deg(
        self, *, analytical_holder: bool = False
    ) -> float:
        """Return a per-segment cone with the same solid angle.

        This is an angular acceptance summary, not an inference of detector
        area, shape or sample-to-sensor distance.
        """

        solid_angle = (
            self.analytical_holder_solid_angle_per_segment_sr
            if analytical_holder
            else self.minimum_unshadowed_solid_angle_per_segment_sr
        )
        cosine = 1.0 - solid_angle / (2.0 * math.pi)
        return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))

    def equivalent_circular_face_distance_mm(
        self,
        active_area_per_segment_mm2: float,
        *,
        analytical_holder: bool = False,
    ) -> float:
        """Return the distance for an equivalent circular, sample-facing disk.

        This inverse is exact only for a circular disk whose normal points at
        the sample.  It is deliberately not evaluated by the installed
        installed definition because the production segment area and shape are
        not public.
        """

        area = float(active_area_per_segment_mm2)
        if not math.isfinite(area) or area <= 0.0:
            raise ValueError("EDS active area must be finite and positive")
        radius = math.sqrt(area / math.pi)
        half_angle = math.radians(
            self.equivalent_circular_cone_half_angle_deg(
                analytical_holder=analytical_holder
            )
        )
        return radius / math.tan(half_angle)

    @property
    def minimum_axis_separation_deg(self) -> float:
        """Return the smallest separation of the configured segment axes."""

        elevation = math.radians(self.takeoff_angle_deg)
        cosine_elevation = math.cos(elevation)
        sine_elevation = math.sin(elevation)
        directions = tuple(
            (
                cosine_elevation * math.cos(math.radians(azimuth)),
                cosine_elevation * math.sin(math.radians(azimuth)),
                sine_elevation,
            )
            for azimuth in self.azimuth_centers_deg
        )
        separations = []
        for index, first in enumerate(directions):
            for second in directions[index + 1:]:
                dot = sum(a * b for a, b in zip(first, second, strict=True))
                separations.append(
                    math.degrees(math.acos(min(1.0, max(-1.0, dot))))
                )
        return min(separations) if separations else 180.0

    @property
    def equivalent_circular_cones_overlap(self) -> bool:
        """Whether equal circular summaries overlap on the unit sphere.

        An overlap demonstrates that the summaries cannot be literal,
        disjoint physical segment apertures; it does not invalidate the
        aggregate solid-angle measurement.
        """

        diameter = 2.0 * self.equivalent_circular_cone_half_angle_deg()
        return self.minimum_axis_separation_deg < diameter


@dataclass(frozen=True, slots=True)
class AxisymmetricPoleCenterlineAssessment:
    """Meridional clearance of one EDS centre ray past one pole surface."""

    ray_radius_at_face_mm: float
    ray_radius_at_shoulder_mm: float
    face_radial_clearance_mm: float
    shoulder_radial_clearance_mm: float
    minimum_radial_clearance_mm: float
    maximum_tip_diameter_mm_for_margin: float
    minimum_nose_axial_length_mm_for_margin: float

    @property
    def clears_requested_margin(self) -> bool:
        return self.minimum_radial_clearance_mm >= -1.0e-12


def assess_axisymmetric_pole_centerline(
    *,
    takeoff_angle_deg: float,
    pole_gap_mm: float,
    pole_bore_diameter_mm: float,
    pole_tip_diameter_mm: float,
    pole_outer_diameter_mm: float,
    pole_nose_axial_length_mm: float,
    requested_radial_margin_mm: float = 0.0,
) -> AxisymmetricPoleCenterlineAssessment:
    """Assess one straight EDS centre ray against a truncated-cone pole.

    The sample is assumed to lie at the centre of a symmetric pole gap.  The
    ray elevation is measured from the sample plane.  Because translating a
    detector along this ray leaves the ray itself unchanged, this assessment
    is independent of detector distance.  It checks only the centre ray in an
    axisymmetric meridional section, not the unpublished EDS active-area
    shape, collimator or complete 4.45 sr acceptance.
    """

    angle = float(takeoff_angle_deg)
    gap = float(pole_gap_mm)
    bore = float(pole_bore_diameter_mm)
    tip = float(pole_tip_diameter_mm)
    outer = float(pole_outer_diameter_mm)
    nose = float(pole_nose_axial_length_mm)
    margin = float(requested_radial_margin_mm)
    values = (angle, gap, bore, tip, outer, nose, margin)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("EDS/pole clearance inputs must be finite")
    if not 0.0 < angle < 90.0:
        raise ValueError("EDS take-off angle must be in (0, 90) degrees")
    if not 0.0 < bore < tip < outer:
        raise ValueError("Pole diameters must satisfy 0 < bore < tip < outer")
    if gap <= 0.0 or nose <= 0.0 or margin < 0.0:
        raise ValueError("Pole gap/nose must be positive and margin nonnegative")

    tangent = math.tan(math.radians(angle))
    half_gap = 0.5 * gap
    ray_at_face = half_gap / tangent
    ray_at_shoulder = (half_gap + nose) / tangent
    face_clearance = ray_at_face - 0.5 * tip - margin
    shoulder_clearance = ray_at_shoulder - 0.5 * outer - margin
    return AxisymmetricPoleCenterlineAssessment(
        ray_radius_at_face_mm=ray_at_face,
        ray_radius_at_shoulder_mm=ray_at_shoulder,
        face_radial_clearance_mm=face_clearance,
        shoulder_radial_clearance_mm=shoulder_clearance,
        minimum_radial_clearance_mm=min(
            face_clearance, shoulder_clearance
        ),
        maximum_tip_diameter_mm_for_margin=2.0 * (
            ray_at_face - margin
        ),
        minimum_nose_axial_length_mm_for_margin=(
            (0.5 * outer + margin) * tangent - half_gap
        ),
    )
