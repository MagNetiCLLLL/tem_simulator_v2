"""Signed first-order transverse transfer and orientation calibration.

The paraxial state ordering is ``(x, y, theta_x, theta_y)`` in one
right-handed laboratory frame.  Electrons travel from source to detector
along +Z.  The position block at a downstream plane is

``r_plane = J_img @ r_sample + J_diff @ theta_sample``.

``J_img`` is dimensionless and ``J_diff`` is expressed in m/rad.  Fixed
deflector offsets are affine terms, so a traced reference ray is subtracted
from the four basis rays before the Jacobian is assembled.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from temsim.physics.core import propagate


CALIBRATED_DETECTOR_ORIENTATION_STATUSES = frozenset({
    "column_coordinates",
    "measured_calibration",
    "service_calibration",
})


def _readonly_matrix(values, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be one finite 2x2 matrix")
    matrix = matrix.copy()
    matrix.setflags(write=False)
    return matrix


@dataclass(frozen=True, slots=True)
class LinearMapProperties:
    """Scale and orientation diagnostics for one real 2x2 map."""

    determinant: float
    isotropic_scale: float
    singular_values: tuple[float, float]
    anisotropy_ratio: float
    condition_number: float
    rank: int
    orientation_deg: float | None
    mirrored: bool


@dataclass(frozen=True, slots=True)
class DetectorFrameCalibration:
    """Map column X-Y components into one detector/display coordinate frame.

    ``axis_rotation_deg`` is the counter-clockwise angle from column +X to
    detector +U as viewed looking downstream.  Flips are then applied to the
    displayed U/V components.  An uncalibrated identity is only a placeholder;
    it must not be presented as a measured absolute detector orientation.
    """

    key: str
    axis_rotation_deg: float = 0.0
    flip_x: bool = False
    flip_y: bool = False
    uncertainty_deg: float = 180.0
    status: str = "uncalibrated_identity"
    source: str = "No absolute detector-axis calibration supplied."

    def __post_init__(self) -> None:
        angle = float(self.axis_rotation_deg)
        uncertainty = float(self.uncertainty_deg)
        if not math.isfinite(angle):
            raise ValueError("Detector-axis rotation must be finite")
        if not math.isfinite(uncertainty) or not 0.0 <= uncertainty <= 180.0:
            raise ValueError(
                "Detector-orientation uncertainty must be between 0 and 180 deg"
            )
        if not str(self.status).strip():
            raise ValueError("Detector-orientation status must not be empty")
        if not str(self.source).strip():
            raise ValueError("Detector-orientation source must not be empty")

    @property
    def is_calibrated(self) -> bool:
        return self.status in CALIBRATED_DETECTOR_ORIENTATION_STATUSES

    @property
    def column_to_detector(self) -> np.ndarray:
        angle = math.radians(float(self.axis_rotation_deg))
        # Detector components are dot products with detector basis vectors,
        # hence R(-angle), followed by any readout/display flips.
        rotation = np.array(
            [
                [math.cos(angle), math.sin(angle)],
                [-math.sin(angle), math.cos(angle)],
            ],
            dtype=float,
        )
        flips = np.diag((
            -1.0 if self.flip_x else 1.0,
            -1.0 if self.flip_y else 1.0,
        ))
        return flips @ rotation


@dataclass(frozen=True, slots=True)
class TransverseTransfer:
    """Full signed first-order 4x4 transfer in 2x2 block form."""

    source_z_mm: float
    target_z_mm: float
    j_img: np.ndarray
    j_diff_m_per_rad: np.ndarray
    k_img_rad_per_m: np.ndarray
    k_diff: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "j_img", _readonly_matrix(self.j_img, "J_img"))
        object.__setattr__(
            self,
            "j_diff_m_per_rad",
            _readonly_matrix(self.j_diff_m_per_rad, "J_diff"),
        )
        object.__setattr__(
            self,
            "k_img_rad_per_m",
            _readonly_matrix(self.k_img_rad_per_m, "K_img"),
        )
        object.__setattr__(
            self,
            "k_diff",
            _readonly_matrix(self.k_diff, "K_diff"),
        )

    @property
    def matrix(self) -> np.ndarray:
        """Return the 4x4 map in (x, y, theta_x, theta_y) ordering."""

        return np.block([
            [self.j_img, self.j_diff_m_per_rad],
            [self.k_img_rad_per_m, self.k_diff],
        ])


@dataclass(frozen=True, slots=True)
class OrientationRelation:
    """Image/diffraction direction relation for one captured mode pair."""

    image_from_diffraction: np.ndarray
    normalized_direction_map: np.ndarray
    properties: LinearMapProperties
    detector_uncertainty_deg: float | None
    calibration_status: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "image_from_diffraction",
            _readonly_matrix(
                self.image_from_diffraction,
                "Image-from-diffraction map",
            ),
        )
        object.__setattr__(
            self,
            "normalized_direction_map",
            _readonly_matrix(
                self.normalized_direction_map,
                "Normalised image-from-diffraction map",
            ),
        )


def linear_map_properties(matrix) -> LinearMapProperties:
    """Return signed orientation properties without discarding reflections.

    For a mirrored map, ``orientation_deg`` is the proper rotation left after
    factoring an input-X reflection from the nearest orthogonal factor.  The
    complete 2x2 matrix remains authoritative.
    """

    values = _readonly_matrix(matrix, "Linear map")
    left, singular_values, right_t = np.linalg.svd(values)
    maximum = float(singular_values[0])
    minimum = float(singular_values[1])
    threshold = max(maximum * 1.0e-12, 1.0e-15)
    rank = int(np.count_nonzero(singular_values > threshold))
    condition = maximum / minimum if minimum > threshold else math.inf
    determinant = float(np.linalg.det(values))
    orthogonal = left @ right_t
    mirrored = float(np.linalg.det(orthogonal)) < 0.0
    if mirrored:
        orthogonal = orthogonal @ np.diag((-1.0, 1.0))
    orientation = (
        math.degrees(math.atan2(orthogonal[1, 0], orthogonal[0, 0]))
        if rank == 2
        else None
    )
    return LinearMapProperties(
        determinant=determinant,
        isotropic_scale=math.sqrt(abs(determinant)),
        singular_values=(maximum, minimum),
        anisotropy_ratio=condition,
        condition_number=condition,
        rank=rank,
        orientation_deg=orientation,
        mirrored=mirrored,
    )


def trace_transverse_transfers(
    state,
    source_z_mm: float,
    target_z_values_mm: Iterable[float],
) -> dict[float, TransverseTransfer]:
    """Trace a reference plus four bases once and sample requested planes."""

    source = float(source_z_mm)
    if not math.isfinite(source):
        raise ValueError("Transfer source Z must be finite")
    targets = sorted({float(value) for value in target_z_values_mm})
    if any(not math.isfinite(value) for value in targets):
        raise ValueError("Transfer target Z values must be finite")
    if any(value < source for value in targets):
        raise ValueError("First-order transfer targets must not precede the source")
    if not targets:
        return {}

    result: dict[float, TransverseTransfer] = {}
    if source in targets:
        result[source] = TransverseTransfer(
            source,
            source,
            np.eye(2),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.eye(2),
        )
    downstream = [value for value in targets if value > source]
    if not downstream:
        return result

    z_mm, x, tx, y, ty = propagate(
        state,
        source,
        downstream[-1],
        np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
        include_spherical_aberration=False,
        include_hexapole=False,
        save_z_mm=downstream,
    )
    for target in downstream:
        index = int(np.argmin(np.abs(z_mm - target)))
        position = np.vstack((
            np.asarray(x[index, 1:5], dtype=float) - float(x[index, 0]),
            np.asarray(y[index, 1:5], dtype=float) - float(y[index, 0]),
        ))
        angle = np.vstack((
            np.asarray(tx[index, 1:5], dtype=float) - float(tx[index, 0]),
            np.asarray(ty[index, 1:5], dtype=float) - float(ty[index, 0]),
        ))
        result[target] = TransverseTransfer(
            source_z_mm=source,
            target_z_mm=target,
            j_img=position[:, :2],
            j_diff_m_per_rad=position[:, 2:],
            k_img_rad_per_m=angle[:, :2],
            k_diff=angle[:, 2:],
        )
    return result


def trace_transverse_transfer(
    state, source_z_mm: float, target_z_mm: float
) -> TransverseTransfer:
    """Return one signed first-order transverse transfer."""

    target = float(target_z_mm)
    return trace_transverse_transfers(state, source_z_mm, (target,))[target]


def detector_frame_from_component(component) -> DetectorFrameCalibration:
    """Read detector-axis calibration metadata from a runtime component."""

    if component is None:
        return DetectorFrameCalibration(
            key="column_xy",
            uncertainty_deg=0.0,
            status="column_coordinates",
            source="Simulator laboratory X-Y coordinates.",
        )
    return DetectorFrameCalibration(
        key=str(getattr(component, "key", "detector")),
        axis_rotation_deg=float(
            getattr(component, "detector_axis_rotation_deg", 0.0)
        ),
        flip_x=bool(getattr(component, "detector_flip_x", False)),
        flip_y=bool(getattr(component, "detector_flip_y", False)),
        uncertainty_deg=float(
            getattr(component, "detector_orientation_uncertainty_deg", 180.0)
        ),
        status=str(
            getattr(
                component,
                "detector_orientation_status",
                "uncalibrated_identity",
            )
        ),
        source=str(
            getattr(
                component,
                "detector_orientation_source",
                "No absolute detector-axis calibration supplied.",
            )
        ),
    )


def relative_image_diffraction_orientation(
    image_transfer: TransverseTransfer,
    diffraction_transfer: TransverseTransfer,
    wavelength_m: float,
    *,
    image_detector: DetectorFrameCalibration | None = None,
    diffraction_detector: DetectorFrameCalibration | None = None,
) -> OrientationRelation:
    """Map a diffraction-vector direction into the corresponding image frame.

    The small-angle relation is ``theta_sample = wavelength * g_sample``.
    The returned normalised matrix maps a measured diffraction-spot vector to
    the image direction of the same reciprocal vector.  It is a direction map,
    not a real-space length calibration.
    """

    wavelength = float(wavelength_m)
    if not math.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError("Electron wavelength must be finite and positive")
    image_frame = image_detector or detector_frame_from_component(None)
    diffraction_frame = (
        diffraction_detector or detector_frame_from_component(None)
    )
    image_map = image_frame.column_to_detector @ image_transfer.j_img
    diffraction_map = (
        diffraction_frame.column_to_detector
        @ diffraction_transfer.j_diff_m_per_rad
        * wavelength
    )
    image_properties = linear_map_properties(image_map)
    diffraction_properties = linear_map_properties(diffraction_map)
    if image_properties.rank < 2:
        raise ValueError("Captured image J_img is singular or rank deficient")
    if diffraction_properties.rank < 2:
        raise ValueError("Captured diffraction J_diff is singular or rank deficient")
    relation = np.linalg.solve(diffraction_map.T, image_map.T).T
    properties = linear_map_properties(relation)
    scale = properties.isotropic_scale
    if not math.isfinite(scale) or scale <= 1.0e-30:
        raise ValueError("Image/diffraction orientation relation is singular")
    normalised = relation / scale
    calibrated = image_frame.is_calibrated and diffraction_frame.is_calibrated
    uncertainty = (
        math.hypot(
            float(image_frame.uncertainty_deg),
            float(diffraction_frame.uncertainty_deg),
        )
        if calibrated
        else None
    )
    return OrientationRelation(
        image_from_diffraction=relation,
        normalized_direction_map=normalised,
        properties=linear_map_properties(normalised),
        detector_uncertainty_deg=uncertainty,
        calibration_status=(
            "calibrated_detector_axes"
            if calibrated
            else "uncalibrated_detector_axes"
        ),
    )
