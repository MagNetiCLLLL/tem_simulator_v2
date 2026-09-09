"""Angle-resolved STEM wave imaging.

This is the wave-optical bridge between the specimen exit wave and the
existing BF/DF/HAADF detector geometry.  It supports symmetric multislice,
atomistic finite-projection potential slices, frozen-phonon intensity
ensembles, and a fast projected phase object.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
import math

import numpy as np

from temsim.optics.aberrations import (
    aberration_phase_rad,
    active_effective_aberrations,
    configured_probe_defocus_mm,
)

from temsim.physics.compute_backend import (
    WAVE_BACKEND_CUPY,
    WAVE_BACKEND_NUMPY,
    choose_wave_backend,
)
from temsim.physics.core import electron
from temsim.physics.multislice import propagate_multislice
from temsim.physics.stem_batching import resident_stem_batch_size
from temsim.physics.stem_cuda_pipeline import (
    release_cupy_memory_pools,
    run_resident_stem_cuda,
)
from temsim.physics.wave_fft import stem_diffraction_intensity
from temsim.physics.wave_imaging import (
    _weighted_ray_statistics,
    effective_sample_thickness_nm,
    interaction_constant_rad_per_v_angstrom,
    prepare_specimen_potentials,
)
from temsim.specimen.presets import (
    load_specimen_preset,
)
from temsim.specimen.source import (
    specimen_is_vacuum,
    specimen_structure_available,
    wave_template_preset_key,
)
from temsim.specimen.geometry import build_sample_geometry_snapshot
from temsim.specimen.envelope import sample_envelope_contains_xy
from temsim.physics.stem_sampling import (
    detector_angular_bounds,
    detector_sampling_report,
)


ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class AngularDetector:
    key: str
    inner_mrad: float
    outer_mrad: float

    def validate(self):
        if self.inner_mrad < 0.0:
            raise ValueError(f"{self.key}: inner collection angle cannot be negative.")
        if self.outer_mrad <= self.inner_mrad:
            raise ValueError(
                f"{self.key}: outer collection angle must exceed the inner angle."
            )
        return self


@dataclass(frozen=True)
class PhysicalAngularDetector:
    """A real detector plane viewed through the signed 2-D angular transfer."""

    key: str
    detector: object
    sample_to_detector_m_per_rad: np.ndarray
    inner_mrad: float
    outer_mrad: float

    def validate(self):
        transfer = np.asarray(self.sample_to_detector_m_per_rad, dtype=float)
        if transfer.shape != (2, 2) or not np.all(np.isfinite(transfer)):
            raise ValueError(
                f"{self.key}: sample-to-detector transfer must be finite and 2 by 2."
            )
        if abs(float(np.linalg.det(transfer))) <= 1.0e-18:
            raise ValueError(f"{self.key}: sample-to-detector transfer is singular.")
        if not hasattr(self.detector, "hit_mask"):
            raise ValueError(f"{self.key}: physical detector has no hit-mask geometry.")
        if self.inner_mrad < 0.0 or self.outer_mrad <= self.inner_mrad:
            raise ValueError(f"{self.key}: invalid diagnostic collection-angle range.")
        return self

    def acceptance_mask(self, angle_x_mrad, angle_y_mrad):
        angle_x = np.asarray(angle_x_mrad, dtype=float) * 1.0e-3
        angle_y = np.asarray(angle_y_mrad, dtype=float) * 1.0e-3
        if angle_x.shape != angle_y.shape:
            raise ValueError(f"{self.key}: angular coordinate grids must match.")
        transfer = np.asarray(self.sample_to_detector_m_per_rad, dtype=float)
        x_mm = 1.0e3 * (
            transfer[0, 0] * angle_x + transfer[0, 1] * angle_y
        )
        y_mm = 1.0e3 * (
            transfer[1, 0] * angle_x + transfer[1, 1] * angle_y
        )
        return np.asarray(self.detector.hit_mask(x_mm, y_mm), dtype=bool)


@dataclass(frozen=True)
class AngleResolvedStemResult:
    scan_x_um: np.ndarray
    scan_y_um: np.ndarray
    fractions: dict[str, np.ndarray]
    detector_ranges_mrad: dict[str, tuple[float, float]]
    maximum_isotropic_angle_mrad: float
    uncollected_fraction: np.ndarray
    truncated_fraction: np.ndarray | None
    metrics: dict
    fourdstem_artifact: object | None = None
    sample_overlap_fraction: np.ndarray | None = None


@dataclass(frozen=True)
class ProbeFocusState:
    """Signed sample-plane focus terms used by the coherent STEM probe."""

    ray_waist_offset_m: float
    ray_defocus_mm: float
    configured_defocus_mm: float
    effective_defocus_mm: float
    source: str


def _coherent_probe_semiangle_rad(ray_stats) -> float:
    """Use the same current-robust aperture observable as Nanoprobe alignment."""

    return max(float(ray_stats["convergence_95_rad"]), 0.0)


def _detector_mask(detector, angles, angle_x_mrad=None, angle_y_mrad=None):
    if isinstance(detector, PhysicalAngularDetector):
        if angle_x_mrad is None or angle_y_mrad is None:
            raise ValueError(
                f"{detector.key}: physical detector integration needs signed angle grids."
            )
        return detector.acceptance_mask(angle_x_mrad, angle_y_mrad)
    return (
        (angles >= float(detector.inner_mrad))
        & (angles <= float(detector.outer_mrad))
    )


def integrate_angular_intensity(
    diffraction_intensity,
    scattering_angle_mrad,
    detectors,
    *,
    valid_mask=None,
    angle_x_mrad=None,
    angle_y_mrad=None,
):
    """Integrate sequential detector bands without double counting overlap."""
    intensity = np.asarray(diffraction_intensity, dtype=float)
    angles = np.asarray(scattering_angle_mrad, dtype=float)
    if intensity.shape != angles.shape:
        raise ValueError("Diffraction intensity and angle grids must match.")
    if valid_mask is None:
        valid = np.ones_like(intensity, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != intensity.shape:
            raise ValueError("Reciprocal-space validity mask has the wrong shape.")
    intensity = np.maximum(intensity, 0.0)
    total = float(intensity.sum())
    if total <= 0.0:
        raise ValueError("Diffraction intensity must contain positive weight.")
    intensity = intensity / total
    available = np.ones_like(intensity, dtype=bool)
    fractions = {}
    masks = {}
    for detector in detectors:
        detector.validate()
        angular_band = _detector_mask(
            detector,
            angles,
            angle_x_mrad=angle_x_mrad,
            angle_y_mrad=angle_y_mrad,
        )
        mask = available & valid & angular_band
        masks[detector.key] = mask
        fractions[detector.key] = float(intensity[mask].sum())
        available[mask] = False
    return fractions, masks, float(intensity[available].sum())


def _wave_grid(state, simulation, scan_x_um, scan_y_um):
    sample_inserted = bool(getattr(state.sample, "inserted", True))
    preset_key = wave_template_preset_key(
        state.sample,
        inserted=sample_inserted,
    )
    preset = load_specimen_preset(preset_key)
    ray_stats = _weighted_ray_statistics(simulation.incident)
    # The coherent STEM probe is formed from the angular aperture below.  The
    # finite-source geometric ray radius is a source-size diagnostic, not the
    # support radius of that coherent wave; using it here can inflate the FOV
    # by orders of magnitude and remove the HAADF band from the FFT grid.
    # Use a diffraction/defocus support estimate in the same sample plane.
    _, _, wavelength_nm = electron(state)
    convergence_rad = max(_coherent_probe_semiangle_rad(ray_stats), 1.0e-12)
    configured_defocus_nm = configured_probe_defocus_mm(state) * 1.0e6
    effective_defocus_nm = (
        configured_defocus_nm
        - float(ray_stats["waist_offset_m"]) * 1.0e9
    )
    diffraction_support_nm = 1.22 * wavelength_nm / convergence_rad
    defocus_support_nm = abs(effective_defocus_nm) * math.tan(convergence_rad)
    probe_radius_99_nm = max(
        diffraction_support_nm,
        defocus_support_nm,
        1.0e-6,
    )
    padding_factor = float(
        getattr(state.sample, "wave_probe_padding_factor", 3.0)
    )
    if not math.isfinite(padding_factor) or padding_factor < 0.0:
        raise ValueError("Wave probe-padding factor must be finite and non-negative.")
    padding_nm = padding_factor * probe_radius_99_nm
    geometry = build_sample_geometry_snapshot(
        state.sample,
        scan_x_um=scan_x_um,
        scan_y_um=scan_y_um,
        probe_padding_nm=padding_nm,
        load_atoms=False,
    )
    if geometry.calculation_roi_bounds_nm is None:
        raise RuntimeError("STEM calculation ROI was not constructed.")
    x0_nm, x1_nm, y0_nm, y1_nm = geometry.calculation_roi_bounds_nm
    span_x_nm = x1_nm - x0_nm
    span_y_nm = y1_nm - y0_nm
    configured_fov = float(
        getattr(state.sample, "wave_field_of_view_angstrom", 0.0)
    )
    derived_fov_angstrom = max(span_x_nm, span_y_nm, 1.0e-6) * 10.0
    requested_fov_angstrom = max(
        derived_fov_angstrom,
        configured_fov if configured_fov > 0.0 else 0.0,
        float(preset.field_of_view_angstrom),
    )
    roi_centre_nm = (
        0.5 * (x0_nm + x1_nm),
        0.5 * (y0_nm + y1_nm),
    )
    prepared = prepare_specimen_potentials(
        state,
        preset,
        field_of_view_angstrom_override=requested_fov_angstrom,
        calculation_roi_centre_nm=roi_centre_nm,
        calculation_roi_bounds_nm=geometry.calculation_roi_bounds_nm,
    )
    prepared.metrics["sample_geometry_snapshot"] = {
        "mode": geometry.mode,
        "inserted": geometry.inserted,
        "centre_nm": geometry.centre_nm,
        "size_nm": geometry.size_nm,
        "orientation_quaternion_wxyz": (
            geometry.orientation_quaternion_wxyz
        ),
        "scan_fov_bounds_nm": geometry.scan_fov_bounds_nm,
        "calculation_roi_bounds_nm": (
            geometry.calculation_roi_bounds_nm
        ),
    }
    prepared.metrics["wave_probe_padding_model"] = (
        "max(1.22 lambda/alpha, abs(effective C1) tan(alpha)); "
        "finite-source ray radius excluded from coherent-wave support"
    )
    prepared.metrics["wave_probe_padding_radius_nm"] = probe_radius_99_nm
    prepared.metrics["wave_probe_geometric_ray_radius_99_nm"] = (
        float(ray_stats["radius_99_m"]) * 1.0e9
    )
    return preset, prepared


def probe_focus_aberrations(state, ray_stats):
    """Combine traced first-order focus with additional coherent aberrations.

    ``waist_offset_m`` is positive when the ray waist lies downstream of the
    sample.  With the Fresnel sign convention used by multislice, the wave C1
    at the sample is the negative of that offset.  Lens excitation is not
    added separately, because it is already encoded by the traced ray bundle.
    """

    waist_offset_m = float(ray_stats.get("waist_offset_m", 0.0))
    if not math.isfinite(waist_offset_m):
        raise ValueError("Probe waist offset must be finite.")
    ray_defocus_mm = -waist_offset_m * 1.0e3
    configured_defocus_mm = configured_probe_defocus_mm(state)
    effective_defocus_mm = ray_defocus_mm + configured_defocus_mm
    coefficients = replace(
        active_effective_aberrations(state, "probe"),
        c1_mm=effective_defocus_mm,
    ).validate()
    focus = ProbeFocusState(
        ray_waist_offset_m=waist_offset_m,
        ray_defocus_mm=ray_defocus_mm,
        configured_defocus_mm=configured_defocus_mm,
        effective_defocus_mm=effective_defocus_mm,
        source=(
            "sample-plane ray covariance plus additional probe C1; "
            "lens excitation is included once through the traced ray bundle"
        ),
    )
    return coefficients, focus


def _probe_spectrum(
    state,
    ray_stats,
    frequencies_x,
    frequencies_y,
    wavelength_angstrom,
):
    fx, fy = np.meshgrid(frequencies_x, frequencies_y, indexing="xy")
    frequency_squared = fx * fx + fy * fy
    frequency_step = max(
        abs(float(frequencies_x[1] - frequencies_x[0])),
        abs(float(frequencies_y[1] - frequencies_y[0])),
    )
    convergence_rad = max(
        _coherent_probe_semiangle_rad(ray_stats),
        frequency_step * wavelength_angstrom,
    )
    centre_fx = math.sin(float(ray_stats["mean_tx_rad"])) / wavelength_angstrom
    centre_fy = math.sin(float(ray_stats["mean_ty_rad"])) / wavelength_angstrom
    aperture_radius = math.sin(convergence_rad) / wavelength_angstrom
    aperture = (
        (fx - centre_fx) ** 2 + (fy - centre_fy) ** 2
        <= aperture_radius**2
    )
    if not np.any(aperture):
        nearest = np.unravel_index(
            np.argmin((fx - centre_fx) ** 2 + (fy - centre_fy) ** 2),
            fx.shape,
        )
        aperture[nearest] = True

    coefficients, _focus = probe_focus_aberrations(state, ray_stats)
    chi = aberration_phase_rad(
        fx,
        fy,
        wavelength_angstrom,
        coefficients,
    )
    return aperture.astype(complex) * np.exp(-1j * chi)


def _normalised_shifted_probe(base_spectrum, fx, fy, x_angstrom, y_angstrom):
    """Use the same translated, centred incident wave for transport and overlap."""
    x0 = np.asarray(x_angstrom, dtype=float)[:, None, None]
    y0 = np.asarray(y_angstrom, dtype=float)[:, None, None]
    shifted_spectrum = base_spectrum[None, :, :] * np.exp(
        -2j * math.pi * (fx[None, :, :] * x0 + fy[None, :, :] * y0)
    )
    probe = np.fft.ifft2(
        np.fft.ifftshift(shifted_spectrum, axes=(-2, -1)), axes=(-2, -1),
    )
    # Prepared potentials use (arange(N) - N//2) * spacing axes.
    probe = np.fft.fftshift(probe, axes=(-2, -1))
    norm = np.sqrt(np.maximum(np.sum(np.abs(probe)**2, axis=(-2, -1)), 1e-30))
    return probe / norm[:, None, None]


def simulate_angle_resolved_stem(
    state,
    simulation,
    detectors,
    scan_x_um,
    scan_y_um,
    *,
    baseline_scan_offset_um=(0.0, 0.0),
    detector_center_shifts_mrad=None,
    progress_callback: ProgressCallback | None = None,
    diffraction_sink=None,
    record_plane_plan=None,
    scan_times_s=None,
    compute_sample_overlap=False,
):
    """Form STEM images and optionally stream the complete diffraction cube.

    ``diffraction_sink`` follows the narrow ``FourDSTEMCaptureSink`` protocol:
    ``begin(calibration, valid_mask, maximum_isotropic_angle_mrad=...)``, then
    ``write_frame(y, x, probability)``, and finally ``finish()``.  Requesting
    the cube selects the complete NumPy reference route because the resident
    GPU reduction intentionally does not transfer per-probe diffraction data.

    A recording plan owns both specimen-position projection and downstream
    kicks, including its per-scan corrections. Legacy detector-centre shifts
    are used only without a plan: those absolute shifts already contain the
    specimen-position displacement and would otherwise count it twice.
    """
    last_progress_fraction = 0.0

    def report_progress(completed: int, total: int, label: str) -> None:
        nonlocal last_progress_fraction
        if progress_callback is None or int(total) <= 0:
            return
        bounded = min(max(int(completed), 0), int(total))
        fraction = bounded / int(total)
        # A failed CUDA attempt can increase the amount of work by requiring
        # a complete CPU retry. Keep the visible bar monotonic while the label
        # reports the fallback explicitly.
        fraction = max(fraction, last_progress_fraction)
        last_progress_fraction = fraction
        progress_callback(
            min(round(fraction * int(total)), int(total)),
            int(total),
            label,
        )

    report_progress(0, 1, "Preparing STEM specimen potential")
    detectors = tuple(detector.validate() for detector in detectors)
    if not detectors:
        raise ValueError("At least one angular STEM detector is required.")
    scan_x_um = np.asarray(scan_x_um, dtype=float)
    scan_y_um = np.asarray(scan_y_um, dtype=float)
    if scan_x_um.shape != scan_y_um.shape or scan_x_um.ndim != 2:
        raise ValueError("STEM scan coordinates must be matching 2-D arrays.")
    if scan_x_um.size == 0:
        raise ValueError("STEM scan coordinate arrays cannot be empty.")
    if record_plane_plan is not None and record_plane_plan.scan_times_s is not None:
        if scan_times_s is not None and not np.array_equal(scan_times_s, record_plane_plan.scan_times_s):
            raise ValueError("STEM scan times differ from the recording plan")
        scan_times_s = record_plane_plan.scan_times_s
    if scan_times_s is not None:
        scan_times_s = np.asarray(scan_times_s, dtype=float)
        if scan_times_s.shape != scan_x_um.shape or not np.all(np.isfinite(scan_times_s)):
            raise ValueError("STEM scan times must be finite and match the raster")
    if (record_plane_plan is not None and record_plane_plan.time_dependent_deflection
            and record_plane_plan.scan_times_s is None):
        raise ValueError("Dynamic STEM recording requires a plan built with scan times")

    ray_stats = _weighted_ray_statistics(simulation.incident)
    origin_x_um = float(ray_stats["mean_x_m"]) * 1.0e6 - float(
        baseline_scan_offset_um[0]
    )
    origin_y_um = float(ray_stats["mean_y_m"]) * 1.0e6 - float(
        baseline_scan_offset_um[1]
    )
    # Material ROI and probe translations share laboratory positions. A
    # steered/shifted beam must not wrap onto a centred CIF.
    preset, prepared = _wave_grid(
        state,
        simulation,
        origin_x_um + scan_x_um,
        origin_y_um + scan_y_um,
    )
    x_axis = prepared.x_angstrom
    y_axis = prepared.y_angstrom
    potential = prepared.mean_projected_potential_v_angstrom
    nx = x_axis.size
    ny = y_axis.size
    spacing_x = float(x_axis[1] - x_axis[0])
    spacing_y = float(y_axis[1] - y_axis[0])
    frequencies_x = np.fft.fftshift(np.fft.fftfreq(nx, d=spacing_x))
    frequencies_y = np.fft.fftshift(np.fft.fftfreq(ny, d=spacing_y))
    fx, fy = np.meshgrid(frequencies_x, frequencies_y, indexing="xy")
    radial_frequency = np.hypot(fx, fy)
    _, _, wavelength_nm = electron(state)
    wavelength_angstrom = wavelength_nm * 10.0
    scattering_angle_mrad = np.arcsin(
        np.clip(wavelength_angstrom * radial_frequency, 0.0, 1.0)
    ) * 1.0e3
    maximum_isotropic_frequency = min(
        abs(float(frequencies_x[0])),
        abs(float(frequencies_x[-1])),
        abs(float(frequencies_y[0])),
        abs(float(frequencies_y[-1])),
    )
    maximum_isotropic_angle_mrad = math.asin(
        min(wavelength_angstrom * maximum_isotropic_frequency, 1.0)
    ) * 1.0e3
    multislice_enabled = bool(
        getattr(state.sample, "wave_multislice_enabled", True)
    )
    total_thickness_nm = effective_sample_thickness_nm(state)
    total_thickness_angstrom = total_thickness_nm * 10.0
    target_slice_angstrom = float(
        getattr(state.sample, "wave_slice_thickness_angstrom", 2.0)
    )
    if prepared.slice_thicknesses_angstrom is not None:
        estimated_slices = int(prepared.slice_thicknesses_angstrom.size)
    else:
        estimated_slices = (
            max(
                1,
                int(
                    math.ceil(
                        total_thickness_angstrom / target_slice_angstrom
                    )
                ),
            )
            if (
                multislice_enabled
                and total_thickness_angstrom > 0.0
                and math.isfinite(target_slice_angstrom)
                and target_slice_angstrom > 0.0
            )
            else 1
        )
    configuration_count = len(
        prepared.potential_configurations_v_angstrom
    )
    wave_backend, wave_fallback_reason = choose_wave_backend(
        getattr(state, "acceleration_backend", "Auto"),
        acceleration_enabled=bool(
            getattr(state, "acceleration_enabled", True)
        ),
        work_items=(
            nx
            * ny
            * estimated_slices
            * scan_x_um.size
            * configuration_count
        ),
    )
    if multislice_enabled:
        multislice_frequency = (
            float(getattr(state.sample, "wave_bandwidth_fraction", 2.0 / 3.0))
            * min(0.5 / spacing_x, 0.5 / spacing_y)
        )
        maximum_isotropic_angle_mrad = min(
            maximum_isotropic_angle_mrad,
            math.asin(
                min(wavelength_angstrom * multislice_frequency, 1.0)
            ) * 1.0e3,
        )
    valid_reciprocal = (
        scattering_angle_mrad <= maximum_isotropic_angle_mrad
    )
    angle_x_mrad = np.arcsin(
        np.clip(wavelength_angstrom * fx, -1.0, 1.0)
    ) * 1.0e3
    angle_y_mrad = np.arcsin(
        np.clip(wavelength_angstrom * fy, -1.0, 1.0)
    ) * 1.0e3

    probe_aberrations, probe_focus = probe_focus_aberrations(state, ray_stats)
    base_spectrum = _probe_spectrum(
        state,
        ray_stats,
        frequencies_x,
        frequencies_y,
        wavelength_angstrom,
    )
    sigma = interaction_constant_rad_per_v_angstrom(state.beam_voltage_kv)
    phase_grating = (
        None if multislice_enabled else np.exp(1j * sigma * potential)
    )
    diagnostic_records = []
    fft_records = []
    fft_backend = wave_backend
    fft_fallback_reason = wave_fallback_reason
    fractions = {
        detector.key: np.zeros(scan_x_um.shape, dtype=float)
        for detector in detectors
    }
    uncollected = np.zeros(scan_x_um.shape, dtype=float)
    truncated_fraction = np.zeros(scan_x_um.shape, dtype=float)
    probe_center_mrad = np.array((ray_stats["mean_tx_rad"], ray_stats["mean_ty_rad"])) * 1e3
    sampling_positions_m = np.stack((origin_x_um + scan_x_um.ravel(),
                                     origin_y_um + scan_y_um.ravel()), axis=-1) * 1e-6
    sampling = detector_sampling_report(
        detector_angular_bounds(
            detectors,
            positions_m=sampling_positions_m,
            record_plane_plan=record_plane_plan,
            detector_center_shifts_mrad=detector_center_shifts_mrad,
        ),
        illumination_bounds_mrad=detector_angular_bounds(
            detectors,
            positions_m=sampling_positions_m,
            record_plane_plan=record_plane_plan,
            detector_center_shifts_mrad=detector_center_shifts_mrad,
            angular_origin_mrad=probe_center_mrad,
        ),
        probe_center_mrad=probe_center_mrad,
        maximum_angle_mrad=maximum_isotropic_angle_mrad,
        wavelength_angstrom=wavelength_angstrom,
        requested_fov_angstrom=float(prepared.metrics.get(
            "requested_field_of_view_angstrom", max(spacing_x * nx, spacing_y * ny))),
        requested_grid_pixels=int(prepared.metrics.get("wave_sampling_plan", {}).get(
            "pixels", getattr(state.sample, "wave_grid_pixels", 0) or preset.pixels)),
        bandwidth_fraction=(float(getattr(state.sample, "wave_bandwidth_fraction", 2 / 3))
                            if multislice_enabled else 1.0),
        probe_semiangle_mrad=_coherent_probe_semiangle_rad(ray_stats) * 1e3,
        potential_storage_bytes=int(prepared.metrics.get("potential_storage_bytes", 0)),
    )
    wave_plan = prepared.metrics.get("wave_sampling_plan")
    if wave_plan is not None:
        # Match detector sampling edits the reference Grid in Sample. Convert
        # the padded-grid proposal back to it, avoiding double FOV expansion.
        reference_fov = float(getattr(state.sample, "wave_field_of_view_angstrom", 0.0))
        if reference_fov <= 0.0:
            reference_fov = float(preset.field_of_view_angstrom)
        reference_grid = int(getattr(state.sample, "wave_grid_pixels", 0) or preset.pixels)
        proposed = sampling.get("recommended_grid_pixels")
        sampling["execution_grid_pixels"] = wave_plan["pixels"]
        sampling["requested_grid_pixels"] = reference_grid
        if proposed is not None:
            proposed_reference = max(reference_grid, 32 * math.ceil(
                proposed * reference_fov / wave_plan["field_of_view_angstrom"] / 32
            ))
            sampling["recommended_grid_pixels"] = proposed_reference
            factor = (proposed_reference / reference_grid) ** 2
            sampling["grid_area_factor"] = factor
            sampling["estimated_potential_bytes"] = math.ceil(
                int(prepared.metrics.get("potential_storage_bytes", 0)) * factor
            )
    if record_plane_plan is not None:
        physical_detector_keys = {
            plane.key
            for plane in record_plane_plan.planes
            if plane.kind == "detector"
        }
        missing_physical = {
            detector.key for detector in detectors
        } - physical_detector_keys
        if missing_physical:
            raise ValueError(
                "Runtime record-plane plan is missing STEM detector(s): "
                + ", ".join(sorted(missing_physical))
            )
    if diffraction_sink is not None:
        from temsim.physics.fourdstem import FourDSTEMCalibration

        # Mixed-plane propagation needs the actual specimen-plane position,
        # including the incident-bundle centroid.  ``scan_x/y_um`` alone are
        # raster coordinates and can omit a static alignment displacement.
        diffraction_sink.begin(
            FourDSTEMCalibration(
                origin_x_um + scan_x_um,
                origin_y_um + scan_y_um,
                angle_x_mrad,
                angle_y_mrad,
                scan_times_s=scan_times_s,
            ),
            valid_reciprocal,
            maximum_isotropic_angle_mrad=maximum_isotropic_angle_mrad,
        )
    _, detector_masks, _ = integrate_angular_intensity(
        np.ones_like(scattering_angle_mrad),
        scattering_angle_mrad,
        detectors,
        valid_mask=valid_reciprocal,
        angle_x_mrad=angle_x_mrad,
        angle_y_mrad=angle_y_mrad,
    )
    flat_detector_centers = None
    if detector_center_shifts_mrad:
        flat_detector_centers = {}
        for detector in detectors:
            values = detector_center_shifts_mrad.get(detector.key)
            if values is None:
                center_x = np.zeros(scan_x_um.shape, dtype=float)
                center_y = np.zeros(scan_y_um.shape, dtype=float)
            else:
                center_x = np.asarray(values[0], dtype=float)
                center_y = np.asarray(values[1], dtype=float)
                if (
                    center_x.shape != scan_x_um.shape
                    or center_y.shape != scan_y_um.shape
                ):
                    raise ValueError(
                        f"{detector.key}: detector-centre shift must match "
                        "the STEM raster shape."
                    )
                if not (
                    np.all(np.isfinite(center_x))
                    and np.all(np.isfinite(center_y))
                ):
                    raise ValueError(
                        f"{detector.key}: detector-centre shift must be finite."
                    )
            flat_detector_centers[detector.key] = (
                center_x.ravel(),
                center_y.ravel(),
            )

    prepared_detector_masks = None
    if record_plane_plan is not None:
        from temsim.physics.record_plane import prepare_record_plane_detector_masks

        prepared_detector_masks = prepare_record_plane_detector_masks(
            record_plane_plan,
            np.stack((angle_x_mrad, angle_y_mrad), axis=-1)[None, ...] * 1.0e-3,
        )

    def batch_detector_masks_for(start, stop):
        """One exact physical acceptance calculation, shared by CPU and GPU."""
        if prepared_detector_masks is not None:
            position_m = np.stack((
                origin_x_um + scan_x_um.ravel()[start:stop],
                origin_y_um + scan_y_um.ravel()[start:stop],
            ), axis=-1)[:, None, None, :] * 1.0e-6
            by_key = prepared_detector_masks(position_m, scan_slice=slice(start, stop))
            return {
                detector.key: by_key[detector.key] & valid_reciprocal[None, :, :]
                for detector in detectors
            }
        if flat_detector_centers is None:
            return detector_masks
        available = np.ones((stop - start, *valid_reciprocal.shape), dtype=bool)
        masks = {}
        for detector in detectors:
            center_x, center_y = flat_detector_centers[detector.key]
            shifted_x = angle_x_mrad[None, :, :] - center_x[start:stop, None, None]
            shifted_y = angle_y_mrad[None, :, :] - center_y[start:stop, None, None]
            angular_band = _detector_mask(
                detector, np.hypot(shifted_x, shifted_y),
                angle_x_mrad=shifted_x, angle_y_mrad=shifted_y,
            )
            mask = available & valid_reciprocal[None, :, :] & angular_band
            masks[detector.key] = mask
            available &= ~mask
        return masks

    roi_centre_nm = prepared.metrics.get(
        "calculation_roi_centre_nm",
        (0.0, 0.0),
    )
    flat_x_angstrom = (
        origin_x_um
        + scan_x_um.ravel()
        - float(roi_centre_nm[0]) * 1.0e-3
    ) * 1.0e4
    flat_y_angstrom = (
        origin_y_um
        + scan_y_um.ravel()
        - float(roi_centre_nm[1]) * 1.0e-3
    ) * 1.0e4
    # This is the incident coherent intensity integrated over the physical
    # projected envelope, including the configured probe defocus and shifts.
    # It supports a uniform-slab *count-budget* absorption approximation; it
    # is not an imaginary potential or slice-resolved absorptive wave solve.
    sample_overlap = None
    overlap_mask = None
    if compute_sample_overlap:
        sample_overlap = np.zeros(scan_x_um.shape, dtype=float)
        if total_thickness_nm > 0.0 and not specimen_is_vacuum(state.sample):
            overlap_mask = np.asarray(sample_envelope_contains_xy(
                state.sample,
                x_axis[None, :] * 0.1 + float(roi_centre_nm[0]),
                y_axis[:, None] * 0.1 + float(roi_centre_nm[1]),
            ), dtype=bool)
            if np.all(overlap_mask):
                sample_overlap.fill(1.0)
                overlap_mask = None
            elif not np.any(overlap_mask):
                overlap_mask = None
    flat_fractions = {
        key: values.ravel() for key, values in fractions.items()
    }
    flat_uncollected = uncollected.ravel()
    flat_truncated = truncated_fraction.ravel()
    detector_sem_numerator = {detector.key: 0.0 for detector in detectors}
    detector_sem_denominator = {detector.key: 0.0 for detector in detectors}
    batch_size = min(8, flat_x_angstrom.size)
    batch_count = int(math.ceil(flat_x_angstrom.size / batch_size))
    cpu_work_units = batch_count * configuration_count
    cpu_total_work = cpu_work_units + 2
    report_progress(1, cpu_total_work, "STEM specimen potential ready")
    resident_cuda_result = None
    resident_pipeline_metrics = {
        "cuda_resident_pipeline": False,
        "cuda_pipeline_fallback_reason": None,
    }
    if wave_backend == WAVE_BACKEND_CUPY and diffraction_sink is not None:
        capture_reason = (
            "4D-STEM capture uses the complete NumPy diffraction frames; "
            "the resident GPU detector reduction does not return a 4-D cube."
        )
        wave_backend = WAVE_BACKEND_NUMPY
        fft_backend = WAVE_BACKEND_NUMPY
        wave_fallback_reason = "; ".join(
            reason
            for reason in (wave_fallback_reason, capture_reason)
            if reason
        )
        fft_fallback_reason = wave_fallback_reason
        resident_pipeline_metrics["cuda_pipeline_fallback_reason"] = (
            capture_reason
        )
    if wave_backend == WAVE_BACKEND_CUPY:
        try:
            dynamic_masks = record_plane_plan is not None or flat_detector_centers is not None
            cuda_batch_size = resident_stem_batch_size(
                flat_x_angstrom.size, (ny, nx),
                potential_bytes=sum(
                    int(configuration.size) * np.dtype(np.float32).itemsize
                    for configuration in prepared.potential_configurations_v_angstrom
                ),
                detector_count=len(detectors),
                recording_plane_count=(0 if record_plane_plan is None else len(record_plane_plan.planes)),
                dynamic_masks=dynamic_masks,
            )
            cuda_total_work = math.ceil(flat_x_angstrom.size / cuda_batch_size) + 2
            resident_cuda_result = run_resident_stem_cuda(
                base_spectrum=base_spectrum,
                frequencies_x=frequencies_x,
                frequencies_y=frequencies_y,
                scan_x_angstrom=flat_x_angstrom,
                scan_y_angstrom=flat_y_angstrom,
                potential_configurations_v_angstrom=(
                    prepared.potential_configurations_v_angstrom
                ),
                detector_masks=detector_masks,
                detector_mask_provider=batch_detector_masks_for if dynamic_masks else None,
                valid_reciprocal_mask=valid_reciprocal,
                multislice_enabled=multislice_enabled,
                pixel_size_angstrom=(spacing_y, spacing_x),
                wavelength_angstrom=wavelength_angstrom,
                interaction_constant_rad_per_v_angstrom=sigma,
                total_thickness_angstrom=total_thickness_angstrom,
                target_slice_thickness_angstrom=target_slice_angstrom,
                slice_thicknesses_angstrom=(
                    prepared.slice_thicknesses_angstrom
                ),
                bandwidth_fraction=float(
                    getattr(
                        state.sample,
                        "wave_bandwidth_fraction",
                        2.0 / 3.0,
                    )
                ),
                batch_size=cuda_batch_size,
                fallback_reason=wave_fallback_reason,
                progress_callback=(
                    None
                    if progress_callback is None
                    else lambda completed, total, label: report_progress(
                        1 + completed,
                        total + 2,
                        label,
                    )
                ),
            )
        except Exception as exc:
            cuda_failure = (
                "Resident CuPy STEM pipeline failed: "
                f"{type(exc).__name__}: {exc}"
            )
            wave_fallback_reason = "; ".join(
                dict.fromkeys(
                    reason
                    for reason in (wave_fallback_reason, cuda_failure)
                    if reason
                )
            )
            fft_fallback_reason = wave_fallback_reason
            wave_backend = WAVE_BACKEND_NUMPY
            fft_backend = WAVE_BACKEND_NUMPY
            resident_pipeline_metrics[
                "cuda_pipeline_fallback_reason"
            ] = cuda_failure
            report_progress(
                round(last_progress_fraction * cpu_total_work),
                cpu_total_work,
                "STEM GPU failed; restarting the detector frame on CPU",
            )
            release_cupy_memory_pools()
        else:
            for key, values in resident_cuda_result.fractions_flat.items():
                flat_fractions[key][:] = values
            flat_uncollected[:] = resident_cuda_result.uncollected_flat
            if resident_cuda_result.truncated_flat is not None:
                flat_truncated[:] = resident_cuda_result.truncated_flat
            if resident_cuda_result.multislice_diagnostics is not None:
                diagnostic_records.append(
                    asdict(
                        resident_cuda_result.multislice_diagnostics
                    )
                )
            fft_records.append(resident_cuda_result.fft_diagnostics)
            resident_pipeline_metrics.update(resident_cuda_result.metrics)

    batch_starts = (
        ()
        if resident_cuda_result is not None
        else range(0, flat_x_angstrom.size, batch_size)
    )
    if resident_cuda_result is not None and overlap_mask is not None:
        # The resident CUDA reduction does not retain the incident probes.
        # Only an enabled absorption channel needs this small bounded CPU
        # pass; no potential generation or multislice is repeated.
        for start in range(0, flat_x_angstrom.size, 2):
            stop = min(start + 2, flat_x_angstrom.size)
            probe = _normalised_shifted_probe(
                base_spectrum, fx, fy,
                flat_x_angstrom[start:stop], flat_y_angstrom[start:stop],
            )
            sample_overlap.ravel()[start:stop] = np.clip(
                np.sum(np.abs(probe)**2 * overlap_mask, axis=(-2, -1)), 0.0, 1.0,
            )
    for batch_index, start in enumerate(batch_starts):
        stop = min(start + batch_size, flat_x_angstrom.size)
        normalised_probe = _normalised_shifted_probe(
            base_spectrum, fx, fy,
            flat_x_angstrom[start:stop], flat_y_angstrom[start:stop],
        )
        if overlap_mask is not None:
            sample_overlap.ravel()[start:stop] = np.clip(
                np.sum(np.abs(normalised_probe)**2 * overlap_mask, axis=(-2, -1)), 0.0, 1.0,
            )
        configuration_values = {
            detector.key: [] for detector in detectors
        }
        configuration_truncated = []
        configuration_averaged_diffraction = (
            np.zeros(
                (stop - start, *valid_reciprocal.shape),
                dtype=np.float64,
            )
            if diffraction_sink is not None
            else None
        )
        batch_detector_masks = batch_detector_masks_for(start, stop)
        for configuration_index, configuration in enumerate(
            prepared.potential_configurations_v_angstrom
        ):
            if multislice_enabled:
                explicit_slices = configuration.ndim == 3
                exit_wave, diagnostics = propagate_multislice(
                    normalised_probe,
                    configuration,
                    pixel_size_angstrom=(spacing_y, spacing_x),
                    wavelength_angstrom=wavelength_angstrom,
                    interaction_constant_rad_per_v_angstrom=sigma,
                    total_thickness_angstrom=(
                        None if explicit_slices else total_thickness_angstrom
                    ),
                    target_slice_thickness_angstrom=target_slice_angstrom,
                    slice_thicknesses_angstrom=(
                        prepared.slice_thicknesses_angstrom
                        if explicit_slices else None
                    ),
                    bandwidth_fraction=float(
                        getattr(
                            state.sample,
                            "wave_bandwidth_fraction",
                            2.0 / 3.0,
                        )
                    ),
                    compute_backend=wave_backend,
                    fallback_reason=wave_fallback_reason,
                )
                diagnostic_records.append(asdict(diagnostics))
                if diagnostics.compute_backend != wave_backend:
                    # The failed CUDA job was already recomputed on the CPU.
                    # Keep later configurations and batches on that fallback.
                    wave_backend = diagnostics.compute_backend
                    wave_fallback_reason = diagnostics.fallback_reason
                    fft_backend = diagnostics.compute_backend
                    fft_fallback_reason = diagnostics.fallback_reason
            else:
                exit_wave = normalised_probe * phase_grating

            diffraction, fft_diagnostics = stem_diffraction_intensity(
                exit_wave,
                compute_backend=fft_backend,
                fallback_reason=fft_fallback_reason,
            )
            fft_records.append(fft_diagnostics)
            if fft_diagnostics.compute_backend != fft_backend:
                fft_backend = fft_diagnostics.compute_backend
                fft_fallback_reason = fft_diagnostics.fallback_reason
            if configuration_averaged_diffraction is not None:
                configuration_averaged_diffraction += diffraction
            configuration_truncated.append(
                np.sum(
                    diffraction[:, ~valid_reciprocal],
                    axis=1,
                )
            )
            for detector in detectors:
                # Both a shared 2-D mask and a per-probe 3-D mask broadcast
                # onto the diffraction batch without indexing an extra axis.
                values = np.sum(
                    diffraction * batch_detector_masks[detector.key],
                    axis=(-2, -1),
                )
                configuration_values[detector.key].append(
                    np.clip(values, 0.0, 1.0)
                )

            completed_units = (
                batch_index * configuration_count + configuration_index + 1
            )
            phonon_detail = (
                f" · phonon {configuration_index + 1}/{configuration_count}"
                if configuration_count > 1
                else ""
            )
            report_progress(
                1 + completed_units,
                cpu_total_work,
                f"STEM probes {stop}/{flat_x_angstrom.size}{phonon_detail}",
            )

        collected = np.zeros(stop - start, dtype=float)
        for detector in detectors:
            samples = np.stack(
                configuration_values[detector.key], axis=0
            )
            values = np.mean(samples, axis=0)
            flat_fractions[detector.key][start:stop] = values
            collected += values
            if configuration_count > 1:
                standard_error = (
                    np.std(samples, axis=0, ddof=1)
                    / math.sqrt(configuration_count)
                )
                detector_sem_numerator[detector.key] += float(
                    np.sum(standard_error**2)
                )
                detector_sem_denominator[detector.key] += float(
                    np.sum(values**2)
                )
        flat_uncollected[start:stop] = np.maximum(1.0 - collected, 0.0)
        flat_truncated[start:stop] = np.mean(
            np.stack(configuration_truncated, axis=0),
            axis=0,
        )
        if configuration_averaged_diffraction is not None:
            configuration_averaged_diffraction /= configuration_count
            scan_width = int(scan_x_um.shape[1])
            for local_index, diffraction_frame in enumerate(
                configuration_averaged_diffraction
            ):
                scan_y_index, scan_x_index = divmod(
                    start + local_index,
                    scan_width,
                )
                diffraction_sink.write_frame(
                    scan_y_index,
                    scan_x_index,
                    diffraction_frame,
                )

    final_progress_total = (
        cuda_total_work
        if resident_cuda_result is not None
        else cpu_total_work
    )
    report_progress(
        final_progress_total - 1,
        final_progress_total,
        "Finalising STEM detector signals",
    )

    detector_ranges = {
        detector.key: (detector.inner_mrad, detector.outer_mrad)
        for detector in detectors
    }
    truncated = tuple(
        detector.key
        for detector in detectors
        if sampling["detectors"][detector.key]["status"] != "full"
    )
    scan_span_x_angstrom = float(np.ptp(scan_x_um)) * 1.0e4
    scan_span_y_angstrom = float(np.ptp(scan_y_um)) * 1.0e4
    field_of_view_x_angstrom = spacing_x * nx
    field_of_view_y_angstrom = spacing_y * ny
    if diagnostic_records:
        specimen_metrics = dict(diagnostic_records[0])
        for key in (
            "maximum_phase_per_slice_rad",
            "maximum_relative_intensity_change",
        ):
            specimen_metrics[key] = max(
                float(record[key]) for record in diagnostic_records
            )
        backends = {
            str(record["compute_backend"]) for record in diagnostic_records
        }
        precisions = {
            str(record["numeric_precision"]) for record in diagnostic_records
        }
        fallback_reasons = tuple(
            dict.fromkeys(
                str(record["fallback_reason"])
                for record in diagnostic_records
                if record.get("fallback_reason")
            )
        )
        specimen_metrics["compute_backend"] = (
            backends.pop()
            if len(backends) == 1
            else "Mixed (NumPy CPU + CuPy CUDA)"
        )
        specimen_metrics["numeric_precision"] = (
            precisions.pop() if len(precisions) == 1 else "mixed"
        )
        specimen_metrics["fallback_reason"] = (
            "; ".join(fallback_reasons) or None
        )
        if prepared.metrics["atomistic_applied"]:
            specimen_metrics["model"] = (
                "atomistic_frozen_phonon_multislice"
                if prepared.metrics["frozen_phonon_applied"]
                else "atomistic_static_multislice"
            )
    else:
        specimen_metrics = {
            "model": "projected_phase_object",
            "slice_count": 1 if total_thickness_nm > 0.0 else 0,
            "total_thickness_angstrom": total_thickness_angstrom,
            "slice_thickness_angstrom": total_thickness_angstrom,
            "bandwidth_fraction": 1.0,
            "maximum_isotropic_angle_mrad": maximum_isotropic_angle_mrad,
            "maximum_phase_per_slice_rad": float(
                np.max(np.abs(sigma * potential))
            ),
            "maximum_relative_intensity_change": 0.0,
            "compute_backend": (
                resident_cuda_result.fft_diagnostics.compute_backend
                if resident_cuda_result is not None
                else WAVE_BACKEND_NUMPY
            ),
            "numeric_precision": (
                resident_cuda_result.fft_diagnostics.numeric_precision
                if resident_cuda_result is not None
                else "complex128 / float64"
            ),
            "fallback_reason": (
                resident_cuda_result.fft_diagnostics.fallback_reason
                if resident_cuda_result is not None
                else wave_fallback_reason
            ),
            "pixel_size_y_angstrom": spacing_y,
            "pixel_size_x_angstrom": spacing_x,
        }
    specimen_metrics.update(prepared.metrics)
    specimen_metrics["sample_inserted"] = bool(
        getattr(state.sample, "inserted", True)
    )
    specimen_metrics["sample_interaction_applied"] = bool(
        not specimen_is_vacuum(state.sample)
        and specimen_structure_available(state.sample)
    )

    fft_backends = {record.compute_backend for record in fft_records}
    fft_precisions = {record.numeric_precision for record in fft_records}
    fft_reasons = tuple(
        dict.fromkeys(
            str(record.fallback_reason)
            for record in fft_records
            if record.fallback_reason
        )
    )
    fft_compute_backend = (
        fft_backends.pop()
        if len(fft_backends) == 1
        else "Mixed (NumPy CPU + CuPy CUDA)"
    )
    fft_numeric_precision = (
        fft_precisions.pop() if len(fft_precisions) == 1 else "mixed"
    )
    fft_fallback_reason = "; ".join(fft_reasons) or None
    actual_backends = {
        str(specimen_metrics["compute_backend"]),
        fft_compute_backend,
    }
    wave_compute_backend = (
        actual_backends.pop()
        if len(actual_backends) == 1
        else "Mixed (NumPy CPU + CuPy CUDA)"
    )
    if resident_cuda_result is not None:
        detector_relative_standard_error = (
            resident_cuda_result.detector_relative_standard_error
        )
    else:
        detector_relative_standard_error = {
            detector.key: (
                math.sqrt(
                    detector_sem_numerator[detector.key]
                    / max(
                        detector_sem_denominator[detector.key],
                        1.0e-30,
                    )
                )
                if configuration_count > 1
                else 0.0
            )
            for detector in detectors
        }
    fourdstem_artifact = (
        None if diffraction_sink is None else diffraction_sink.finish()
    )
    truncation_available = (
        resident_cuda_result is None or resident_cuda_result.truncated_flat is not None
    )
    result = AngleResolvedStemResult(
        scan_x_um=scan_x_um,
        scan_y_um=scan_y_um,
        fractions=fractions,
        detector_ranges_mrad=detector_ranges,
        maximum_isotropic_angle_mrad=maximum_isotropic_angle_mrad,
        uncollected_fraction=uncollected,
        truncated_fraction=truncated_fraction if truncation_available else None,
        metrics={
            "model": (
                "multislice_angle_resolved"
                if multislice_enabled else "thin_phase_angle_resolved"
            ),
            **{
                f"specimen_{key}": value
                for key, value in specimen_metrics.items()
            },
            "preset_key": preset.key,
            "wavelength_angstrom": wavelength_angstrom,
            "probe_ray_waist_offset_nm": (
                probe_focus.ray_waist_offset_m * 1.0e9
            ),
            "physical_detector_routing": (
                "full_signed_j_img_r_plus_j_diff_theta_sequential_stops"
                if record_plane_plan is not None
                else "angular_detector_masks"
            ),
            "record_plane_plan_fingerprint": (
                None
                if record_plane_plan is None
                else record_plane_plan.fingerprint
            ),
            "probe_ray_defocus_nm": probe_focus.ray_defocus_mm * 1.0e6,
            "probe_configured_defocus_nm": (
                probe_focus.configured_defocus_mm * 1.0e6
            ),
            "probe_effective_defocus_nm": (
                probe_focus.effective_defocus_mm * 1.0e6
            ),
            "probe_radial_wavefront_curvature_per_m": (
                ray_stats["radial_wavefront_curvature_per_m"]
            ),
            "probe_focus_source": probe_focus.source,
            "probe_effective_c1_mm": probe_aberrations.c1_mm,
            "probe_aperture_semiangle_mrad": (
                _coherent_probe_semiangle_rad(ray_stats) * 1.0e3
            ),
            "probe_aperture_observable": (
                "sample current-weighted 95 percent semi-angle"
            ),
            "grid_pixels": max(nx, ny),
            "grid_pixels_x": nx,
            "grid_pixels_y": ny,
            "pixel_size_angstrom": max(spacing_x, spacing_y),
            "pixel_size_x_angstrom": spacing_x,
            "pixel_size_y_angstrom": spacing_y,
            "field_of_view_angstrom": max(
                field_of_view_x_angstrom,
                field_of_view_y_angstrom,
            ),
            "field_of_view_x_angstrom": field_of_view_x_angstrom,
            "field_of_view_y_angstrom": field_of_view_y_angstrom,
            "scan_span_x_angstrom": scan_span_x_angstrom,
            "scan_span_y_angstrom": scan_span_y_angstrom,
            "scan_exceeds_periodic_field_of_view": bool(
                scan_span_x_angstrom > field_of_view_x_angstrom
                or scan_span_y_angstrom > field_of_view_y_angstrom
            ),
            "maximum_isotropic_angle_mrad": maximum_isotropic_angle_mrad,
            "truncated_detector_keys": truncated,
            "angular_coverage_complete": sampling["coverage_complete"],
            "detector_sampling": sampling,
            "truncated_fraction_available": truncation_available,
            "mean_truncated_fraction": (
                float(np.mean(truncated_fraction)) if truncation_available else None
            ),
            "wave_sampling_truncates_illumination": bool(
                _coherent_probe_semiangle_rad(ray_stats) * 1.0e3
                > maximum_isotropic_angle_mrad
            ),
            "wave_intensity_conservation_within_0_1_percent": bool(
                float(
                    specimen_metrics["maximum_relative_intensity_change"]
                ) <= 1.0e-3
            ),
            "multislice_enabled": multislice_enabled,
            "rutherford_tail_enabled": False,
            "descan_detector_shift_applied": bool(
                flat_detector_centers is not None
                or (record_plane_plan is not None and record_plane_plan.scan_position_offsets_m)
            ),
            "record_plane_scan_deflection_applied": bool(
                record_plane_plan is not None and record_plane_plan.scan_position_offsets_m
            ),
            "displayed_intensity_average": (
                "incoherent frozen-phonon intensity mean"
                if configuration_count > 1
                else "single configuration"
            ),
            "detector_configuration_relative_standard_error": (
                detector_relative_standard_error
            ),
            "fft_compute_backend": fft_compute_backend,
            "fft_numeric_precision": fft_numeric_precision,
            "fft_fallback_reason": fft_fallback_reason,
            "wave_compute_backend": wave_compute_backend,
            **resident_pipeline_metrics,
        },
        fourdstem_artifact=fourdstem_artifact,
        sample_overlap_fraction=sample_overlap,
    )
    report_progress(
        final_progress_total,
        final_progress_total,
        "STEM detector frame complete",
    )
    return result
