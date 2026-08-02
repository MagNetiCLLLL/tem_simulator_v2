"""Sequential BF/DF/HAADF signal integration and raster preview."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from temsim.component_keys import STEM_DETECTOR_KEYS
from temsim.physics.beam_observation import transverse_kick_response
from temsim.physics.stem_wave_imaging import (
    AngularDetector,
    simulate_angle_resolved_stem,
)

ELEMENTARY_CHARGE_C = 1.602176634e-19


@dataclass(frozen=True)
class CollectionAngle:
    inner_mrad: float
    outer_mrad: float
    inner_range_mrad: tuple[float, float]
    outer_range_mrad: tuple[float, float]
    anisotropic: bool


@dataclass(frozen=True)
class BeamCurrentSignal:
    key: str
    name: str
    fraction: float
    simulated_electrons: float
    current_pa: float
    electrons_per_second: float


@dataclass(frozen=True)
class DetectorSignal(BeamCurrentSignal):
    collection_angle: CollectionAngle | None


@dataclass(frozen=True)
class StemScanResult:
    scan_x_um: np.ndarray
    scan_y_um: np.ndarray
    fractions: dict[str, np.ndarray]
    detector_signals: dict[str, DetectorSignal]
    metrics: dict | None = None


def square_scan_limits(scan_x_um, scan_y_um):
    """Return equal-span X/Y limits enclosing one raster coordinate grid."""
    scan_x_um = np.asarray(scan_x_um, dtype=float)
    scan_y_um = np.asarray(scan_y_um, dtype=float)
    if scan_x_um.size == 0 or scan_y_um.size == 0:
        raise ValueError("Raster coordinates cannot be empty.")
    if not np.all(np.isfinite(scan_x_um)) or not np.all(
        np.isfinite(scan_y_um)
    ):
        raise ValueError("Raster coordinates must be finite.")
    x_min = float(scan_x_um.min())
    x_max = float(scan_x_um.max())
    y_min = float(scan_y_um.min())
    y_max = float(scan_y_um.max())
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    half_span = 0.5 * max(
        x_max - x_min,
        y_max - y_min,
        1.0e-12,
    )
    return (
        (x_center - half_span, x_center + half_span),
        (y_center - half_span, y_center + half_span),
    )


def _interpolate(values, z, requested):
    z = np.asarray(z, dtype=float)
    requested = float(requested)
    upper = int(np.searchsorted(z, requested, side="left"))
    if upper <= 0:
        return np.asarray(values[0], dtype=float)
    if upper >= len(z):
        return np.asarray(values[-1], dtype=float)
    if abs(z[upper] - requested) <= 1.0e-9:
        return np.asarray(values[upper], dtype=float)
    lower = upper - 1
    fraction = (
        (requested - z[lower])
        / max(float(z[upper] - z[lower]), 1.0e-12)
    )
    return (
        (1.0 - fraction) * np.asarray(values[lower], dtype=float)
        + fraction * np.asarray(values[upper], dtype=float)
    )


def _branch_probabilities(simulation):
    branches = tuple(simulation.branches.values())
    raw = np.asarray(
        [max(float(getattr(branch, "weight", 1.0)), 0.0)
         for branch in branches],
        dtype=float,
    )
    total = float(raw.sum())
    if total <= 0.0:
        raw[:] = 1.0
        total = float(len(raw))
    return branches, raw / total


def _normalised_ray_weights(branch):
    ray_weight = getattr(branch, "ray_weight", None)
    if ray_weight is None:
        ray_count = branch.x.shape[1]
        return np.full(
            ray_count, 1.0 / max(ray_count, 1), dtype=float
        )
    return np.asarray(ray_weight, dtype=float)


def source_current_pa(state):
    """Return the absolute emitted current represented by the ray bundle."""
    return max(
        float(state.electron_gun.emitted_current_a) * 1.0e9,
        0.0,
    ) * 1.0e3


def _current_values(state, fraction):
    fraction = max(float(fraction), 0.0)
    current_pa = source_current_pa(state) * fraction
    return (
        fraction * float(state.electron_gun.ray_count),
        current_pa,
        current_pa * 1.0e-12 / ELEMENTARY_CHARGE_C,
    )


def measure_sample_current(simulation, state):
    """Measure source current surviving to the specimen plane."""
    branch = simulation.incident
    weights = _normalised_ray_weights(branch)
    alive = np.asarray(branch.alive, dtype=bool)
    fraction = float(weights[alive].sum())
    simulated, current_pa, electrons_per_second = _current_values(
        state, fraction
    )
    return BeamCurrentSignal(
        key="sample",
        name="Sample",
        fraction=fraction,
        simulated_electrons=simulated,
        current_pa=current_pa,
        electrons_per_second=electrons_per_second,
    )


def _recorded_fraction(simulation, plane_key):
    branches, probabilities = _branch_probabilities(simulation)
    fraction = 0.0
    for branch, probability in zip(branches, probabilities):
        weights = _normalised_ray_weights(branch)
        recorded = (
            np.asarray(branch.blocked_key, dtype=object)
            == str(plane_key)
        )
        fraction += float(probability) * float(weights[recorded].sum())
    return fraction


def _detector_signal(simulation, state, plane):
    fraction = _recorded_fraction(simulation, plane.key)
    simulated, current_pa, electrons_per_second = _current_values(
        state, fraction
    )
    angle = (
        collection_angle(state, plane)
        if plane.key in STEM_DETECTOR_KEYS
        else None
    )
    return DetectorSignal(
        key=plane.key,
        name=plane.name,
        fraction=fraction,
        simulated_electrons=simulated,
        current_pa=current_pa,
        electrons_per_second=electrons_per_second,
        collection_angle=angle,
    )


def measure_recording_plane_currents(
    simulation, state, *, include_stem=True
):
    """Return actual intercepted current for every inserted recording plane."""
    return {
        plane.key: _detector_signal(simulation, state, plane)
        for plane in state.recording_planes
        if bool(getattr(plane, "inserted", False))
        and (include_stem or plane.key not in STEM_DETECTOR_KEYS)
    }


def measure_aperture_transmitted_current(
    simulation,
    state,
    aperture,
):
    """Measure current that survives through one aperture plane.

    Unlike a recording-plane signal, this counts rays continuing downstream
    of the plane and excludes electrons intercepted by the aperture itself.
    """

    branches, probabilities = _branch_probabilities(simulation)
    fraction = 0.0
    for branch, probability in zip(branches, probabilities):
        weights = _normalised_ray_weights(branch)
        blocked_z = np.asarray(branch.blocked_z, dtype=float)
        transmitted = (
            np.isnan(blocked_z)
            | (blocked_z > float(aperture.z_mm) + 1.0e-9)
        )
        fraction += (
            float(probability)
            * float(weights[transmitted].sum())
        )
    simulated, current_pa, electrons_per_second = _current_values(
        state, fraction
    )
    return BeamCurrentSignal(
        key=aperture.key,
        name=aperture.name,
        fraction=fraction,
        simulated_electrons=simulated,
        current_pa=current_pa,
        electrons_per_second=electrons_per_second,
    )


def collection_angle(state, detector):
    response = transverse_kick_response(
        state, state.sample.z_mm, detector.z_mm
    )
    singular = np.linalg.svd(response, compute_uv=False)
    singular = np.sort(np.abs(singular))
    if singular[0] <= 1.0e-15:
        nan_range = (float("nan"), float("nan"))
        return CollectionAngle(
            float("nan"), float("nan"), nan_range, nan_range, True
        )

    def angle_range(diameter_mm):
        radius_m = max(float(diameter_mm), 0.0) * 0.5e-3
        values = 1.0e3 * radius_m / singular
        return float(values[0]), float(values[1])

    inner_range = angle_range(detector.inner_diameter_mm)
    outer_range = angle_range(detector.outer_width_mm)
    anisotropic = singular[1] / singular[0] > 1.02
    return CollectionAngle(
        inner_mrad=float(np.sqrt(inner_range[0] * inner_range[1])),
        outer_mrad=float(np.sqrt(outer_range[0] * outer_range[1])),
        inner_range_mrad=inner_range,
        outer_range_mrad=outer_range,
        anisotropic=anisotropic,
    )


def measure_stem_detectors(simulation, state, detector_keys=None):
    selected = (
        set(STEM_DETECTOR_KEYS)
        if detector_keys is None
        else set(detector_keys)
    )
    result = {}
    for detector in state.stem_detectors:
        if detector.key not in selected:
            continue
        result[detector.key] = _detector_signal(
            simulation, state, detector
        )
    return result


def acquire_stem_scan(
    simulation,
    state,
    detector_keys=None,
    pixels_x=32,
    pixels_y=32,
):
    """Integrate selected detector signals over the current AC raster."""
    pixels_x = max(2, int(pixels_x))
    pixels_y = max(2, int(pixels_y))
    component = state.ac_deflector
    if not bool(component.scan_enabled):
        raise ValueError(
            "AC raster Scan must be enabled before STEM signal acquisition."
        )
    selected = [
        detector
        for detector in state.stem_detectors
        if (
            detector_keys is None
            and detector.readout_enabled
        )
        or (
            detector_keys is not None
            and detector.key in set(detector_keys)
        )
    ]
    x_factors = np.linspace(-1.0, 1.0, pixels_x)
    y_factors = np.linspace(-1.0, 1.0, pixels_y)
    kick_grid_mrad = np.empty((pixels_y, pixels_x, 2), dtype=float)
    kick_grid_mrad[:, :, 0] = (
        x_factors[None, :] * component.scan_amplitude_x_mrad
    )
    kick_grid_mrad[:, :, 1] = (
        y_factors[:, None] * component.scan_amplitude_y_mrad
    )
    baseline_scan_mrad = np.asarray(
        component.scan_kick_mrad(
            float(getattr(state, "simulation_time_s", 0.0))
        ),
        dtype=float,
    )
    descan = state.descan_deflector
    baseline_descan_scan_mrad = np.asarray(
        (
            descan.scan_kick_mrad(
                float(getattr(state, "simulation_time_s", 0.0))
            )
            if descan.enabled
            else (0.0, 0.0)
        ),
        dtype=float,
    )

    sample_response = 1.0e3 * transverse_kick_response(
        state, component.z_mm, state.sample.z_mm
    )
    sample_offsets_mm = (
        (kick_grid_mrad * 1.0e-3) @ sample_response.T
    )
    scan_x_um = sample_offsets_mm[:, :, 0] * 1.0e3
    scan_y_um = sample_offsets_mm[:, :, 1] * 1.0e3

    if bool(getattr(state.sample, "stem_wave_enabled", False)):
        ordered = sorted(selected, key=lambda detector: float(detector.z_mm))
        angular_detectors = []
        detector_angles = {}
        for detector in ordered:
            angle = collection_angle(state, detector)
            if not (
                np.isfinite(angle.inner_mrad)
                and np.isfinite(angle.outer_mrad)
            ):
                raise ValueError(
                    f"{detector.name}: collection angle is unavailable."
                )
            detector_angles[detector.key] = angle
            angular_detectors.append(
                AngularDetector(
                    detector.key,
                    angle.inner_mrad,
                    angle.outer_mrad,
                )
            )
        baseline_sample_offset_um = (
            (baseline_scan_mrad * 1.0e-3) @ sample_response.T
        ) * 1.0e3
        wave = simulate_angle_resolved_stem(
            state,
            simulation,
            angular_detectors,
            scan_x_um,
            scan_y_um,
            baseline_scan_offset_um=baseline_sample_offset_um,
        )
        incident_fraction = measure_sample_current(
            simulation, state
        ).fraction
        images = {
            key: values * incident_fraction
            for key, values in wave.fractions.items()
        }
        signals = {}
        for detector in ordered:
            fraction = float(np.mean(images[detector.key]))
            simulated, current_pa, electrons_per_second = _current_values(
                state, fraction
            )
            signals[detector.key] = DetectorSignal(
                key=detector.key,
                name=detector.name,
                fraction=fraction,
                simulated_electrons=simulated,
                current_pa=current_pa,
                electrons_per_second=electrons_per_second,
                collection_angle=detector_angles[detector.key],
            )
        metrics = dict(wave.metrics)
        metrics["incident_sample_fraction"] = incident_fraction
        metrics["mean_uncollected_fraction"] = float(
            np.mean(wave.uncollected_fraction) * incident_fraction
        )
        return StemScanResult(
            wave.scan_x_um,
            wave.scan_y_um,
            images,
            signals,
            metrics,
        )

    branches, probabilities = _branch_probabilities(simulation)
    recording_keys = {
        plane.key for plane in state.recording_planes
    }
    inserted_planes = sorted(
        (
            plane for plane in state.recording_planes
            if bool(getattr(plane, "inserted", False))
        ),
        key=lambda plane: float(plane.z_mm),
    )
    weights = []
    for branch, probability in zip(branches, probabilities):
        branch_weight = _normalised_ray_weights(branch)
        weights.append(branch_weight * float(probability))
    weights = np.concatenate(weights)

    plane_data = {}
    for plane in inserted_planes:
        positions = []
        physically_reaches = []
        for branch in branches:
            x_mm = 1.0e3 * _interpolate(
                branch.x, branch.z, plane.z_mm
            )
            y_mm = 1.0e3 * _interpolate(
                branch.y, branch.z, plane.z_mm
            )
            blocked_z = np.asarray(branch.blocked_z, dtype=float)
            blocked_key = np.asarray(branch.blocked_key, dtype=object)
            reaches = (
                np.isnan(blocked_z)
                | (blocked_z > float(plane.z_mm) + 1.0e-9)
                | np.isin(blocked_key, tuple(recording_keys))
            )
            positions.append(np.column_stack((x_mm, y_mm)))
            physically_reaches.append(reaches)
        plane_data[plane.key] = (
            np.vstack(positions),
            np.concatenate(physically_reaches),
            1.0e3 * transverse_kick_response(
                state, component.z_mm, plane.z_mm
            ),
            1.0e3 * transverse_kick_response(
                state, descan.z_mm, plane.z_mm
            ),
        )

    images = {
        detector.key: np.zeros((pixels_y, pixels_x), dtype=float)
        for detector in selected
    }
    selected_keys = set(images)
    for row in range(pixels_y):
        for column in range(pixels_x):
            delta_rad = (
                kick_grid_mrad[row, column] - baseline_scan_mrad
            ) * 1.0e-3
            scan_time_s = (
                (
                    row
                    + column / max(float(pixels_x), 1.0)
                )
                / max(float(pixels_y), 1.0)
                * float(component.scan_frame_period_s)
            )
            descan_delta_rad = (
                np.asarray(
                    (
                        descan.scan_kick_mrad(scan_time_s)
                        if descan.enabled
                        else (0.0, 0.0)
                    ),
                    dtype=float,
                )
                - baseline_descan_scan_mrad
            ) * 1.0e-3
            available = np.ones(weights.size, dtype=bool)
            for plane in inserted_planes:
                (
                    positions,
                    reaches,
                    response_mm_per_rad,
                    descan_response_mm_per_rad,
                ) = (
                    plane_data[plane.key]
                )
                shifted = (
                    positions
                    + response_mm_per_rad @ delta_rad
                    + descan_response_mm_per_rad @ descan_delta_rad
                )
                if hasattr(plane, "hit_mask"):
                    shape_hit = plane.hit_mask(
                        shifted[:, 0], shifted[:, 1]
                    )
                else:
                    radius = np.hypot(
                        shifted[:, 0], shifted[:, 1]
                    )
                    outer = float(plane.outer_width_mm) / 2.0
                    inner = float(plane.inner_diameter_mm) / 2.0
                    if str(plane.geometry).lower() == "annulus":
                        shape_hit = (
                            (radius >= inner) & (radius <= outer)
                        )
                    elif str(plane.geometry).lower() in {
                        "square", "rectangle", "camera"
                    }:
                        shape_hit = (
                            (np.abs(shifted[:, 0]) <= outer)
                            & (np.abs(shifted[:, 1]) <= outer)
                        )
                    else:
                        shape_hit = radius <= outer
                hit = available & reaches & shape_hit
                if plane.key in selected_keys:
                    images[plane.key][row, column] = float(
                        weights[hit].sum()
                    )
                available[hit] = False

    signals = measure_stem_detectors(
        simulation, state, [detector.key for detector in selected]
    )
    return StemScanResult(
        scan_x_um,
        scan_y_um,
        images,
        signals,
        {"model": "geometric_ray_preview"},
    )
