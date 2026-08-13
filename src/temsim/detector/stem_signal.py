"""Sequential BF/DF/HAADF signal integration and raster preview."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from temsim.component_keys import STEM_DETECTOR_KEYS
from temsim.physics.beam_observation import transverse_kick_response
from temsim.physics.scan_geometry import (
    calibrate_scan_system,
    paired_kick_response,
    raster_sample_grid,
)
from temsim.physics.stem_wave_imaging import (
    AngularDetector,
    PhysicalAngularDetector,
    simulate_angle_resolved_stem,
)
from temsim.physics.probe_state import ProbeState, probe_state_from_simulation
from temsim.specimen.virtual import (
    build_virtual_angular_distribution,
    virtual_density_at_scan,
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
    current_pa: dict[str, np.ndarray] | None = None
    expected_electrons: dict[str, np.ndarray] | None = None
    poisson_counts: dict[str, np.ndarray] | None = None
    dwell_time_s: float | None = None
    uncollected_fraction: np.ndarray | None = None
    absorbed_fraction: np.ndarray | None = None
    truncated_fraction: np.ndarray | None = None
    high_angle_tail_fraction: dict[str, np.ndarray] | None = None
    probe_state: ProbeState | None = None
    axis_units: tuple[str, str] = ("um", "um")
    orientation: str = "array[y, x]; laboratory +X right, +Y up"


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
    if bool(getattr(simulation, "metrics", {}).get("branch_weights_are_absolute", False)):
        if float(raw.sum()) > 1.0 + 1.0e-10:
            raise ValueError("Absolute simulation branch probabilities exceed one.")
        return branches, raw
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


def _stem_result(
    state,
    simulation,
    scan_x_um,
    scan_y_um,
    fractions,
    detector_signals,
    metrics,
    *,
    uncollected_fraction=None,
    absorbed_fraction=None,
    truncated_fraction=None,
    high_angle_tail_fraction=None,
):
    """Attach deterministic current/dose observables and optional shot noise."""

    scan_x_um = np.asarray(scan_x_um, dtype=float)
    scan_y_um = np.asarray(scan_y_um, dtype=float)
    if scan_x_um.shape != scan_y_um.shape or scan_x_um.ndim != 2:
        raise ValueError("STEM result coordinates must be matching 2-D arrays.")
    frame_period_s = float(state.ac_deflector.scan_frame_period_s)
    dwell_time_s = frame_period_s / max(scan_x_um.size, 1)
    source_pa = source_current_pa(state)
    current = {
        key: np.asarray(values, dtype=float) * source_pa
        for key, values in fractions.items()
    }
    expected = {
        key: values * 1.0e-12 * dwell_time_s / ELEMENTARY_CHARGE_C
        for key, values in current.items()
    }
    poisson = None
    if bool(getattr(state.sample, "stem_poisson_enabled", False)):
        seed = int(getattr(state.sample, "stem_poisson_seed", 0))
        if seed < 0:
            raise ValueError("STEM Poisson seed cannot be negative.")
        rng = np.random.default_rng(seed)
        poisson = {
            key: rng.poisson(np.maximum(expected[key], 0.0))
            for key in fractions
        }
    probe_state = probe_state_from_simulation(state, simulation)
    metadata = dict(metrics or {})
    metadata.update(
        {
            "scan_frame_period_s": frame_period_s,
            "dwell_time_s": dwell_time_s,
            "signal_fraction_reference": "emitted source current",
            "current_unit": "pA",
            "expected_electron_model": "current*dwell/e",
            "poisson_noise_enabled": poisson is not None,
            "poisson_seed": (
                int(getattr(state.sample, "stem_poisson_seed", 0))
                if poisson is not None
                else None
            ),
            "array_axis_order": "y,x",
            "laboratory_orientation": "+X right; +Y up; electron beam +Z",
        }
    )
    return StemScanResult(
        scan_x_um=scan_x_um,
        scan_y_um=scan_y_um,
        fractions=fractions,
        detector_signals=detector_signals,
        metrics=metadata,
        current_pa=current,
        expected_electrons=expected,
        poisson_counts=poisson,
        dwell_time_s=dwell_time_s,
        uncollected_fraction=(
            None
            if uncollected_fraction is None
            else np.asarray(uncollected_fraction, dtype=float)
        ),
        absorbed_fraction=(
            None
            if absorbed_fraction is None
            else np.asarray(absorbed_fraction, dtype=float)
        ),
        truncated_fraction=(
            None
            if truncated_fraction is None
            else np.asarray(truncated_fraction, dtype=float)
        ),
        high_angle_tail_fraction=high_angle_tail_fraction,
        probe_state=probe_state,
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
        values = np.sort(1.0e3 * radius_m / singular)
        return float(values[0]), float(values[1])

    inner_range = angle_range(detector.inner_diameter_mm)
    outer_range = angle_range(detector.outer_width_mm)
    anisotropic = bool(singular[1] / singular[0] > 1.02)
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


def physical_angular_detectors(state, detectors):
    """Resolve TOML detector shapes through the full signed 2-D transfer."""

    resolved = []
    angles = {}
    for detector in sorted(detectors, key=lambda item: float(item.z_mm)):
        angle = collection_angle(state, detector)
        if not (
            np.isfinite(angle.inner_mrad)
            and np.isfinite(angle.outer_mrad)
        ):
            raise ValueError(f"{detector.name}: collection angle is unavailable.")
        transfer = transverse_kick_response(
            state,
            state.sample.z_mm,
            detector.z_mm,
        )
        resolved.append(
            PhysicalAngularDetector(
                key=detector.key,
                detector=detector,
                sample_to_detector_m_per_rad=np.asarray(transfer, dtype=float),
                inner_mrad=angle.inner_mrad,
                outer_mrad=angle.outer_mrad,
            ).validate()
        )
        angles[detector.key] = angle
    return tuple(resolved), angles


def _detector_center_shifts_mrad(
    simulation,
    state,
    detectors,
    kick_grid_mrad,
    baseline_scan_mrad,
    scan_times_s,
    baseline_descan_scan_mrad,
):
    """Return equivalent specimen-angle shifts from the actual direct ray."""

    direct = simulation.branches.get("000")
    if direct is None and simulation.branches:
        direct = next(iter(simulation.branches.values()))
    if direct is None:
        return None
    component = state.ac_deflector
    descan = state.descan_deflector
    scan_delta_rad = (
        np.asarray(kick_grid_mrad, dtype=float)
        - np.asarray(baseline_scan_mrad, dtype=float)
    ) * 1.0e-3
    if descan.enabled:
        descan_commands_mrad = np.asarray(
            [
                descan.instantaneous_kick_mrad(float(time_s))
                for time_s in np.asarray(scan_times_s).ravel()
            ],
            dtype=float,
        ).reshape(*np.asarray(scan_times_s).shape, 2)
        descan_delta_rad = (
            descan_commands_mrad
            - np.asarray(baseline_descan_scan_mrad, dtype=float)
        ) * 1.0e-3
    else:
        descan_delta_rad = np.zeros_like(scan_delta_rad)
    result = {}
    for detector in detectors:
        base_x_values = np.asarray(
            _interpolate(direct.x, direct.z, detector.z_mm),
            dtype=float,
        )
        base_y_values = np.asarray(
            _interpolate(direct.y, direct.z, detector.z_mm),
            dtype=float,
        )
        ray_weights = _normalised_ray_weights(direct)
        blocked_z = np.asarray(direct.blocked_z, dtype=float)
        reaches = (
            np.isnan(blocked_z)
            | (blocked_z >= float(detector.z_mm) - 1.0e-9)
        )
        reaches &= np.isfinite(base_x_values) & np.isfinite(base_y_values)
        represented = float(ray_weights[reaches].sum())
        if represented <= 0.0:
            base_x_m = 0.0
            base_y_m = 0.0
        else:
            base_x_m = float(
                np.sum(ray_weights[reaches] * base_x_values[reaches])
                / represented
            )
            base_y_m = float(
                np.sum(ray_weights[reaches] * base_y_values[reaches])
                / represented
            )
        direct_displacement_m = np.empty((*scan_delta_rad.shape[:-1], 2), dtype=float)
        direct_displacement_m[..., 0] = base_x_m
        direct_displacement_m[..., 1] = base_y_m
        direct_displacement_m += np.einsum(
            "ij,...j->...i",
            paired_kick_response(state, component, detector.z_mm),
            scan_delta_rad,
        )
        if descan.enabled:
            direct_displacement_m += np.einsum(
                "ij,...j->...i",
                paired_kick_response(state, descan, detector.z_mm),
                descan_delta_rad,
            )
        sample_angle_response = transverse_kick_response(
            state,
            state.sample.z_mm,
            detector.z_mm,
        )
        equivalent_angle_rad = np.einsum(
            "ij,...j->...i",
            np.linalg.pinv(sample_angle_response),
            direct_displacement_m,
        )
        # acceptance(theta) with theta-centre is equivalent to adding the
        # measured direct-beam displacement at the physical detector plane.
        result[detector.key] = (
            -equivalent_angle_rad[..., 0] * 1.0e3,
            -equivalent_angle_rad[..., 1] * 1.0e3,
        )
    if not any(
        np.any(np.abs(values[0]) > 1.0e-12)
        or np.any(np.abs(values[1]) > 1.0e-12)
        for values in result.values()
    ):
        return None
    return result


def _virtual_stem_scan(
    simulation,
    state,
    physical_detectors,
    detector_angles,
    scan_x_um,
    scan_y_um,
    detector_center_shifts_mrad,
):
    probe = probe_state_from_simulation(state, simulation)
    interaction_enabled = bool(
        getattr(state.sample, "inserted", True)
        and getattr(state.sample, "diffraction_enabled", True)
    )
    if interaction_enabled:
        distribution = build_virtual_angular_distribution(
            state.sample,
            beam_energy_kv=state.beam_voltage_kv,
        )
        density = virtual_density_at_scan(
            state.sample,
            scan_x_um,
            scan_y_um,
            probe_sigma_nm=probe.probe_sigma_nm,
        )
        angle_x = np.r_[0.0, distribution.angle_x_mrad]
        angle_y = np.r_[0.0, distribution.angle_y_mrad]
        scatter_probabilities = distribution.probabilities
        interacting_probability = (
            distribution.scattered_probability
            + distribution.absorbed_probability
        )
        direct = 1.0 - density * interacting_probability
        absorbed = density * distribution.absorbed_probability
        approximations = tuple(
            component.approximation
            for component in distribution.components
            if component.approximation
        )
    else:
        distribution = None
        density = np.zeros_like(scan_x_um, dtype=float)
        angle_x = np.asarray((0.0,))
        angle_y = np.asarray((0.0,))
        scatter_probabilities = np.empty(0)
        direct = np.ones_like(scan_x_um, dtype=float)
        absorbed = np.zeros_like(scan_x_um, dtype=float)
        approximations = ()

    incident_fraction = measure_sample_current(simulation, state).fraction
    images = {
        detector.key: np.zeros(scan_x_um.shape, dtype=float)
        for detector in physical_detectors
    }
    uncollected = np.zeros(scan_x_um.shape, dtype=float)
    flat_density = density.ravel()
    flat_direct = direct.ravel()
    for flat_index in range(scan_x_um.size):
        local = np.r_[
            flat_direct[flat_index],
            flat_density[flat_index] * scatter_probabilities,
        ]
        available = np.ones(angle_x.size, dtype=bool)
        for detector in physical_detectors:
            if detector_center_shifts_mrad is None:
                shifted_x = angle_x
                shifted_y = angle_y
            else:
                centre_x, centre_y = detector_center_shifts_mrad[detector.key]
                shifted_x = angle_x - centre_x.ravel()[flat_index]
                shifted_y = angle_y - centre_y.ravel()[flat_index]
            hit = available & detector.acceptance_mask(shifted_x, shifted_y)
            images[detector.key].ravel()[flat_index] = (
                float(np.sum(local[hit])) * incident_fraction
            )
            available[hit] = False
        uncollected.ravel()[flat_index] = (
            float(np.sum(local[available])) * incident_fraction
        )
    absorbed_source = absorbed * incident_fraction
    pre_sample_lost_fraction = max(1.0 - incident_fraction, 0.0)
    uncollected += pre_sample_lost_fraction
    signals = {}
    for detector in physical_detectors:
        fraction = float(np.mean(images[detector.key]))
        simulated, current_pa, electrons_per_second = _current_values(
            state,
            fraction,
        )
        signals[detector.key] = DetectorSignal(
            key=detector.key,
            name=detector.detector.name,
            fraction=fraction,
            simulated_electrons=simulated,
            current_pa=current_pa,
            electrons_per_second=electrons_per_second,
            collection_angle=detector_angles[detector.key],
        )
    total = absorbed_source + uncollected
    for image in images.values():
        total += image
    conservation_error = float(np.max(np.abs(total - 1.0)))
    metrics = {
        "model": (
            "finite_virtual_absolute_probability"
            if interaction_enabled
            else "vacuum_direct_beam"
        ),
        "interaction_probability_normalised": False,
        "incident_sample_fraction": incident_fraction,
        "pre_sample_lost_fraction": pre_sample_lost_fraction,
        "mean_virtual_density": float(np.mean(density)),
        "maximum_probability_conservation_error": conservation_error,
        "virtual_probability_conserved": conservation_error <= 5.0e-10,
        "virtual_interaction_approximations": approximations,
        "rutherford_model": (
            "screened relativistic Rutherford approximation; not Mott"
            if approximations
            else None
        ),
        "scan_pixels_x": int(scan_x_um.shape[1]),
        "scan_pixels_y": int(scan_x_um.shape[0]),
        "scan_pixel_size_nm": float(state.ac_deflector.scan_pixel_size_nm),
        "scan_field_of_view_x_nm": float(
            state.ac_deflector.scan_field_of_view_x_nm
        ),
        "scan_field_of_view_y_nm": float(
            state.ac_deflector.scan_field_of_view_y_nm
        ),
        "physical_detector_masks": True,
        "sequential_detector_interception": True,
        "descan_detector_shift_applied": bool(
            detector_center_shifts_mrad is not None
        ),
    }
    return _stem_result(
        state,
        simulation,
        scan_x_um,
        scan_y_um,
        images,
        signals,
        metrics,
        uncollected_fraction=uncollected,
        absorbed_fraction=absorbed_source,
        truncated_fraction=np.zeros_like(scan_x_um, dtype=float),
        high_angle_tail_fraction={
            detector.key: np.zeros_like(scan_x_um, dtype=float)
            for detector in physical_detectors
        },
    )


def _real_high_angle_tail(
    simulation,
    state,
    physical_detectors,
    scan_x_um,
    scan_y_um,
    detector_center_shifts_mrad,
    minimum_angle_mrad,
):
    zeros = {
        detector.key: np.zeros_like(scan_x_um, dtype=float)
        for detector in physical_detectors
    }
    if not bool(getattr(state.sample, "real_high_angle_tail_enabled", False)):
        return zeros, np.zeros_like(scan_x_um), np.zeros_like(scan_x_um), None
    maximum = float(getattr(state.sample, "real_tail_max_angle_mrad", 250.0))
    if maximum <= float(minimum_angle_mrad):
        raise ValueError(
            "High-angle tail maximum must exceed strict multislice angular support."
        )
    density_atoms = float(
        getattr(state.sample, "real_tail_areal_density_atoms_nm2", 0.0)
    )
    if density_atoms <= 0.0:
        raise ValueError(
            "High-angle tail needs a positive user-supplied areal density."
        )
    row = {
        "name": "Real-sample high-angle tail",
        "kind": "physical_rutherford",
        "enabled": True,
        "atomic_number": int(getattr(state.sample, "real_tail_atomic_number", 14)),
        "areal_density_atoms_nm2": density_atoms,
        "screening_angle_mrad": float(
            getattr(state.sample, "real_tail_screening_angle_mrad", 5.0)
        ),
        "minimum_angle_mrad": float(minimum_angle_mrad) * (1.0 + 1.0e-9),
        "maximum_angle_mrad": maximum,
        "radial_samples": 128,
        "azimuth_samples": 64,
    }
    virtual_sample = SimpleNamespace(
        virtual_interactions=[row],
        virtual_diffraction_angle_mrad=5.0,
        virtual_diffraction_azimuth_deg=0.0,
        virtual_diffraction_relative_weight=1.0,
        virtual_scattering_angle_mrad=20.0,
        virtual_scattering_relative_weight=0.2,
        virtual_scattering_azimuth_samples=16,
    )
    distribution = build_virtual_angular_distribution(
        virtual_sample,
        beam_energy_kv=state.beam_voltage_kv,
    )
    probe = probe_state_from_simulation(state, simulation)
    finite_sample = SimpleNamespace(
        size_x_nm=float(state.sample.size_x_nm),
        size_y_nm=float(state.sample.size_y_nm),
        centre_x_nm=float(state.sample.centre_x_nm),
        centre_y_nm=float(state.sample.centre_y_nm),
        virtual_regions=[],
        virtual_probe_convolution_enabled=True,
    )
    density = virtual_density_at_scan(
        finite_sample,
        scan_x_um,
        scan_y_um,
        probe_sigma_nm=probe.probe_sigma_nm,
    )
    incident_fraction = measure_sample_current(simulation, state).fraction
    images = {
        detector.key: np.zeros_like(scan_x_um, dtype=float)
        for detector in physical_detectors
    }
    tail_uncollected = np.zeros_like(scan_x_um, dtype=float)
    angle_x = distribution.angle_x_mrad
    angle_y = distribution.angle_y_mrad
    for flat_index, local_density in enumerate(density.ravel()):
        local = distribution.probabilities * float(local_density)
        available = np.ones(angle_x.size, dtype=bool)
        for detector in physical_detectors:
            if detector_center_shifts_mrad is None:
                shifted_x, shifted_y = angle_x, angle_y
            else:
                centre_x, centre_y = detector_center_shifts_mrad[detector.key]
                shifted_x = angle_x - centre_x.ravel()[flat_index]
                shifted_y = angle_y - centre_y.ravel()[flat_index]
            hit = available & detector.acceptance_mask(shifted_x, shifted_y)
            images[detector.key].ravel()[flat_index] = (
                float(np.sum(local[hit])) * incident_fraction
            )
            available[hit] = False
        tail_uncollected.ravel()[flat_index] = (
            float(np.sum(local[available])) * incident_fraction
        )
    tail_probability_source = (
        density * distribution.scattered_probability * incident_fraction
    )
    metrics = {
        "model": "screened_relativistic_rutherford_approximation_not_mott",
        "minimum_angle_mrad": row["minimum_angle_mrad"],
        "maximum_angle_mrad": maximum,
        "atomic_number": row["atomic_number"],
        "areal_density_atoms_nm2": density_atoms,
        "screening_angle_mrad": row["screening_angle_mrad"],
        "integrated_cross_section_m2": distribution.components[0].parameters[
            "integrated_cross_section_m2"
        ],
    }
    return images, tail_uncollected, tail_probability_source, metrics


def acquire_stem_scan(
    simulation,
    state,
    detector_keys=None,
    pixels_x=None,
    pixels_y=None,
):
    """Integrate selected detector signals over the current AC raster."""
    component = state.ac_deflector
    if not bool(component.enabled and component.scan_enabled):
        raise ValueError(
            "AC Scan Coil and its raster drive must both be enabled before "
            "STEM signal acquisition."
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
    pixels_x = int(
        component.scan_pixels_x if pixels_x is None else pixels_x
    )
    pixels_y = int(component.scan_lines if pixels_y is None else pixels_y)
    calibrate_scan_system(state)
    x_factors, y_factors, scan_times_s = raster_sample_grid(
        component,
        pixels_x=pixels_x,
        pixels_y=pixels_y,
        maximum_count=None,
    )
    raster_factors = np.stack((x_factors, y_factors), axis=-1)
    kick_grid_mrad = np.einsum(
        "ij,...j->...i",
        np.asarray(component.scan_command_matrix_mrad, dtype=float),
        raster_factors,
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

    sample_response = 1.0e3 * paired_kick_response(
        state,
        component,
        state.sample.z_mm,
    )
    sample_offsets_mm = (
        (kick_grid_mrad * 1.0e-3) @ sample_response.T
    )
    scan_x_um = (
        sample_offsets_mm[:, :, 0] * 1.0e3
        + float(getattr(state.sample, "scan_origin_x_nm", 0.0)) * 1.0e-3
    )
    scan_y_um = (
        sample_offsets_mm[:, :, 1] * 1.0e3
        + float(getattr(state.sample, "scan_origin_y_nm", 0.0)) * 1.0e-3
    )

    physical_detectors, detector_angles = physical_angular_detectors(
        state,
        selected,
    )
    detector_center_shifts_mrad = _detector_center_shifts_mrad(
        simulation,
        state,
        [detector.detector for detector in physical_detectors],
        kick_grid_mrad,
        baseline_scan_mrad,
        scan_times_s,
        baseline_descan_scan_mrad,
    ) if physical_detectors else None

    if (
        physical_detectors
        and str(getattr(state.sample, "specimen_mode", "atomic")).lower()
        == "virtual"
    ):
        return _virtual_stem_scan(
            simulation,
            state,
            physical_detectors,
            detector_angles,
            scan_x_um,
            scan_y_um,
            detector_center_shifts_mrad,
        )

    if (
        physical_detectors
        and bool(getattr(state.sample, "stem_wave_enabled", False))
        and str(getattr(state.sample, "specimen_mode", "atomic")).lower()
        == "atomic"
    ):
        baseline_sample_offset_um = (
            (baseline_scan_mrad * 1.0e-3) @ sample_response.T
        ) * 1.0e3
        wave = simulate_angle_resolved_stem(
            state,
            simulation,
            physical_detectors,
            scan_x_um,
            scan_y_um,
            baseline_scan_offset_um=baseline_sample_offset_um,
            detector_center_shifts_mrad=detector_center_shifts_mrad,
        )
        incident_fraction = measure_sample_current(
            simulation, state
        ).fraction
        real_interactions = getattr(simulation, "real_interactions", None)
        tracked_sample_probability = (
            float(real_interactions.tracked_probability)
            if real_interactions is not None else 1.0
        )
        available_fraction = (
            incident_fraction * tracked_sample_probability
        )
        absorbed_source = np.full_like(
            scan_x_um,
            incident_fraction * (1.0 - tracked_sample_probability),
            dtype=float,
        )
        (
            tail_images,
            tail_uncollected,
            tail_probability_source,
            tail_metrics,
        ) = _real_high_angle_tail(
            simulation,
            state,
            physical_detectors,
            scan_x_um,
            scan_y_um,
            detector_center_shifts_mrad,
            wave.maximum_isotropic_angle_mrad,
        )
        tail_images = {
            key: values * tracked_sample_probability
            for key, values in tail_images.items()
        }
        tail_uncollected = (
            tail_uncollected * tracked_sample_probability
        )
        tail_probability_source = (
            tail_probability_source * tracked_sample_probability
        )
        wave_scale = np.maximum(
            1.0
            - tail_probability_source
            / max(available_fraction, 1.0e-30),
            0.0,
        )
        images = {
            key: values * available_fraction * wave_scale + tail_images[key]
            for key, values in wave.fractions.items()
        }
        signals = {}
        for detector in physical_detectors:
            fraction = float(np.mean(images[detector.key]))
            simulated, current_pa, electrons_per_second = _current_values(
                state, fraction
            )
            signals[detector.key] = DetectorSignal(
                key=detector.key,
                name=detector.detector.name,
                fraction=fraction,
                simulated_electrons=simulated,
                current_pa=current_pa,
                electrons_per_second=electrons_per_second,
                collection_angle=detector_angles[detector.key],
            )
        metrics = dict(wave.metrics)
        metrics["incident_sample_fraction"] = incident_fraction
        metrics["tracked_probability_after_inelastic_absorption"] = (
            tracked_sample_probability
        )
        metrics["inelastic_angular_transport_in_wave_scan"] = (
            "energy-loss probabilities included; coherent elastic angular "
            "distribution reused for tracked populations; compact inelastic "
            "ray angles are reported in Ray Diagram/Energy Filter"
        )
        metrics["scan_frame_period_s"] = float(
            component.scan_frame_period_s
        )
        metrics["scan_pixels_x"] = pixels_x
        metrics["scan_pixels_y"] = pixels_y
        metrics["scan_pixel_size_nm"] = float(
            component.scan_pixel_size_nm
        )
        metrics["scan_field_of_view_x_nm"] = float(
            component.scan_field_of_view_x_nm
        )
        metrics["scan_field_of_view_y_nm"] = float(
            component.scan_field_of_view_y_nm
        )
        metrics["physical_detector_masks"] = True
        metrics["sequential_detector_interception"] = True
        metrics["rutherford_tail_enabled"] = tail_metrics is not None
        metrics["rutherford_tail"] = tail_metrics
        metrics["hybrid_tail_nonoverlap_minimum_mrad"] = (
            float(wave.maximum_isotropic_angle_mrad)
            if tail_metrics is not None
            else None
        )
        uncollected = (
            wave.uncollected_fraction * available_fraction * wave_scale
            + tail_uncollected
            + max(1.0 - incident_fraction, 0.0)
        )
        metrics["pre_sample_lost_fraction"] = max(
            1.0 - incident_fraction,
            0.0,
        )
        metrics["mean_uncollected_fraction"] = float(
            np.mean(uncollected)
        )
        real_total = uncollected + absorbed_source
        for values in images.values():
            real_total = real_total + values
        real_conservation_error = float(
            np.max(np.abs(real_total - 1.0))
        )
        metrics["maximum_probability_conservation_error"] = (
            real_conservation_error
        )
        metrics["real_probability_conserved"] = (
            real_conservation_error <= 5.0e-10
        )
        return _stem_result(
            state,
            simulation,
            wave.scan_x_um,
            wave.scan_y_um,
            images,
            signals,
            metrics,
            uncollected_fraction=uncollected,
            absorbed_fraction=absorbed_source,
            truncated_fraction=(
                None
                if wave.truncated_fraction is None
                else wave.truncated_fraction * available_fraction * wave_scale
            ),
            high_angle_tail_fraction=tail_images,
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
            1.0e3 * paired_kick_response(
                state,
                component,
                plane.z_mm,
            ),
            1.0e3 * paired_kick_response(
                state,
                descan,
                plane.z_mm,
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
            scan_time_s = float(scan_times_s[row, column])
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

    signals = {}
    for detector in selected:
        fraction = float(np.mean(images[detector.key]))
        simulated, current_pa, electrons_per_second = _current_values(
            state,
            fraction,
        )
        signals[detector.key] = DetectorSignal(
            key=detector.key,
            name=detector.name,
            fraction=fraction,
            simulated_electrons=simulated,
            current_pa=current_pa,
            electrons_per_second=electrons_per_second,
            collection_angle=collection_angle(state, detector),
        )
    collected = np.zeros_like(scan_x_um, dtype=float)
    for values in images.values():
        collected += values
    uncollected = np.maximum(1.0 - collected, 0.0)
    real_interactions = getattr(simulation, "real_interactions", None)
    real_absorbed_source = (
        measure_sample_current(simulation, state).fraction
        * float(real_interactions.absorbed_probability)
        if real_interactions is not None else 0.0
    )
    absorbed = np.full_like(
        uncollected, real_absorbed_source, dtype=float
    )
    uncollected = np.maximum(uncollected - absorbed, 0.0)
    return _stem_result(
        state,
        simulation,
        scan_x_um,
        scan_y_um,
        images,
        signals,
        {
            "model": "geometric_detector_interception",
            "scan_frame_period_s": float(component.scan_frame_period_s),
            "scan_pixels_x": pixels_x,
            "scan_pixels_y": pixels_y,
            "scan_pixel_size_nm": float(component.scan_pixel_size_nm),
            "scan_field_of_view_x_nm": float(
                component.scan_field_of_view_x_nm
            ),
            "scan_field_of_view_y_nm": float(
                component.scan_field_of_view_y_nm
            ),
            "descan_applied": bool(descan.enabled and descan.scan_enabled),
            "physical_detector_masks": True,
            "sequential_detector_interception": True,
            "quantitative_model": False,
            "model_limitation": (
                "Ray preview transports material-derived inelastic event "
                "populations but does not calculate quantitative coherent "
                "elastic specimen contrast; enable wave/multislice for that."
            ),
            "real_inelastic_absorbed_source_fraction": real_absorbed_source,
        },
        uncollected_fraction=uncollected,
        absorbed_fraction=absorbed,
        truncated_fraction=np.zeros_like(uncollected),
        high_angle_tail_fraction={
            detector.key: np.zeros_like(uncollected)
            for detector in selected
        },
    )
