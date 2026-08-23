"""Precalculated Direct Alignment branches and local response ratios.

The stored vectors are engineering calibration points, not OEM current tables.
Interpolation is performed in log(target) and is used only as a solver seed;
the production Direct Alignment validator remains authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from temsim.operating_modes import DirectAlignmentDefinition


@dataclass(frozen=True, slots=True)
class PrecalculatedAlignmentPoint:
    """One TOML-backed coupled-lens working point."""

    target: float
    unit: str
    branch: str
    strengths: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PrecalculatedAlignmentRatio:
    """Local response over one same-branch target interval.

    ``strength_delta_per_decade`` is preferable to a raw ratio near zero lens
    excitation. ``strength_ratios`` therefore uses ``None`` when the lower
    excitation is numerically zero.
    """

    branch: str
    lower_target: float
    upper_target: float
    target_ratio: float
    log10_span: float
    strength_ratios: tuple[float | None, ...]
    strength_delta_per_decade: tuple[float, ...]


def _target_branch(
    definition: DirectAlignmentDefinition, target: float
) -> str:
    if definition.key == "image_magnification":
        lm_maximum = float(
            definition.targets.get("lm_maximum_magnification", 1000.0)
        )
        return "lm" if float(target) <= lm_maximum else "normal"
    return "camera_length"


def _as_points(
    definition: DirectAlignmentDefinition,
    raw_targets: Iterable[object],
    raw_vectors: Iterable[object],
) -> tuple[PrecalculatedAlignmentPoint, ...]:
    targets = np.asarray(tuple(raw_targets), dtype=float)
    vectors = np.asarray(tuple(raw_vectors), dtype=float)
    raw_branches = tuple(
        str(value)
        for value in definition.targets.get("preset_branch_ids", ())
    )
    if targets.size == 0 and vectors.size == 0:
        return ()
    if (
        targets.ndim != 1
        or targets.size == 0
        or vectors.ndim != 2
        or vectors.shape[0] != targets.size
        or np.any(~np.isfinite(targets))
        or np.any(targets <= 0.0)
        or np.any(~np.isfinite(vectors))
        or np.any(vectors < 0.0)
        or np.any(np.diff(targets) <= 0.0)
    ):
        raise ValueError(
            f"{definition.key} precalculated seed table is invalid"
        )
    if raw_branches and len(raw_branches) != targets.size:
        raise ValueError(
            f"{definition.key} preset_branch_ids length is invalid"
        )
    branches = raw_branches or tuple(
        _target_branch(definition, value) for value in targets
    )
    return tuple(
        PrecalculatedAlignmentPoint(
            target=float(target),
            unit=definition.unit,
            branch=str(branch),
            strengths=tuple(float(value) for value in vector),
        )
        for target, vector, branch in zip(targets, vectors, branches)
    )


def precalculated_alignment_points(
    definition: DirectAlignmentDefinition,
) -> tuple[PrecalculatedAlignmentPoint, ...]:
    """Return the positive, ordered projector seed table for a definition."""

    if definition.key == "image_magnification":
        targets = definition.targets.get("preset_magnifications", ())
    elif definition.key == "diffraction_camera_length":
        targets = definition.targets.get("preset_camera_lengths", ())
    else:
        return ()
    return _as_points(
        definition,
        targets,
        definition.targets.get("preset_vectors", ()),
    )


def interpolated_precalculated_seed(
    definition: DirectAlignmentDefinition,
    target: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray | None:
    """Interpolate one same-branch seed without extrapolation.

    Linear interpolation in log10(target) respects the multiplicative scale of
    magnification and camera length. The returned vector is only a warm start;
    it is never a validated or committable result by itself.
    """

    requested = float(target)
    if not math.isfinite(requested) or requested <= 0.0:
        return None
    points = tuple(
        point
        for point in precalculated_alignment_points(definition)
        if point.branch == _target_branch(definition, requested)
    )
    if not points:
        return None
    vector_size = int(np.asarray(lower).size)
    if any(len(point.strengths) != vector_size for point in points):
        raise ValueError(
            f"{definition.key} precalculated vector width is invalid"
        )
    for point in points:
        if math.isclose(requested, point.target, rel_tol=1.0e-12):
            return np.clip(
                np.asarray(point.strengths, dtype=float), lower, upper
            )
    for left, right in zip(points[:-1], points[1:]):
        if left.target < requested < right.target:
            log_left = math.log10(left.target)
            log_right = math.log10(right.target)
            fraction = (
                (math.log10(requested) - log_left)
                / (log_right - log_left)
            )
            left_vector = np.asarray(left.strengths, dtype=float)
            right_vector = np.asarray(right.strengths, dtype=float)
            seed = left_vector + fraction * (right_vector - left_vector)
            return np.clip(seed, lower, upper)
    return None


def interpolated_nanoprobe_seed(
    definition: DirectAlignmentDefinition,
    target_mrad: float,
    aperture_diameter_um: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray | None:
    """Interpolate a focused C2/C3 seed on the calibrated aperture path.

    The table is one-dimensional in the physically paired path
    ``target ~= reference * aperture diameter``. It must not seed a target for
    a materially different current aperture because that would mix two
    independent controls and can select the wrong focus basin.
    """

    try:
        apertures = np.asarray(
            definition.targets.get(
                "precalculation_aperture_diameters_um", ()
            ),
            dtype=float,
        )
        targets = np.asarray(
            definition.targets.get(
                "precalculation_convergence_mrad", ()
            ),
            dtype=float,
        )
        vectors = np.asarray(
            definition.targets.get("precalculation_vectors", ()),
            dtype=float,
        )
    except (TypeError, ValueError):
        return None
    if apertures.size == 0 and targets.size == 0 and vectors.size == 0:
        return None
    if (
        apertures.ndim != 1
        or targets.shape != apertures.shape
        or vectors.shape != (targets.size, np.asarray(lower).size)
        or targets.size < 2
        or np.any(~np.isfinite(apertures))
        or np.any(~np.isfinite(targets))
        or np.any(~np.isfinite(vectors))
        or np.any(apertures <= 0.0)
        or np.any(targets <= 0.0)
        or np.any(vectors < 0.0)
        or np.any(np.diff(apertures) <= 0.0)
        or np.any(np.diff(targets) <= 0.0)
    ):
        raise ValueError("Nanoprobe precalculated table is invalid")
    requested = float(target_mrad)
    current_aperture = float(aperture_diameter_um)
    if (
        not math.isfinite(requested)
        or not math.isfinite(current_aperture)
        or requested < targets[0]
        or requested > targets[-1]
    ):
        return None
    expected_aperture = float(np.interp(requested, targets, apertures))
    match_error = abs(current_aperture - expected_aperture) / max(
        expected_aperture, 1.0e-15
    )
    tolerance = float(
        definition.targets.get(
            "precalculation_aperture_match_relative_tolerance", 0.02
        )
    )
    if match_error > tolerance:
        return None
    seed = np.asarray([
        np.interp(requested, targets, vectors[:, column])
        for column in range(vectors.shape[1])
    ])
    return np.clip(seed, lower, upper)


def precalculated_alignment_ratios(
    definition: DirectAlignmentDefinition,
) -> tuple[PrecalculatedAlignmentRatio, ...]:
    """Calculate inspectable same-branch ratios and local log slopes."""

    points = precalculated_alignment_points(definition)
    rows: list[PrecalculatedAlignmentRatio] = []
    for left, right in zip(points[:-1], points[1:]):
        if left.branch != right.branch:
            continue
        lower_strengths = np.asarray(left.strengths, dtype=float)
        upper_strengths = np.asarray(right.strengths, dtype=float)
        log10_span = math.log10(right.target / left.target)
        ratios: list[float | None] = []
        for lower_value, upper_value in zip(
            lower_strengths, upper_strengths
        ):
            ratios.append(
                None
                if abs(float(lower_value)) <= 1.0e-12
                else float(upper_value / lower_value)
            )
        delta_per_decade = (
            (upper_strengths - lower_strengths) / log10_span
        )
        rows.append(PrecalculatedAlignmentRatio(
            branch=left.branch,
            lower_target=left.target,
            upper_target=right.target,
            target_ratio=right.target / left.target,
            log10_span=log10_span,
            strength_ratios=tuple(ratios),
            strength_delta_per_decade=tuple(
                float(value) for value in delta_per_decade
            ),
        ))
    return tuple(rows)
