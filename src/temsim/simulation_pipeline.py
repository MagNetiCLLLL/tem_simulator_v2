"""Application-facing simulation pipeline, independent of the Tk GUI."""
from collections.abc import Callable
from dataclasses import dataclass

from temsim.detector.recording_system import ensure_recording_system
from temsim.detector.stem_signal import StemScanResult, acquire_stem_scan
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.component_names import normalise_component_names
from temsim.optics.corrector_structure import ensure_corrector_structure
from temsim.optics.energy_filter import ensure_energy_filter
from temsim.optics.energy_filter_raytrace import simulate_energy_filter
from temsim.physics.all_lens_crossovers import detect_all_lens_crossovers
from temsim.physics.simulation import Simulation, run
from temsim.physics.scan_geometry import (
    ScanGeometryResult,
    calculate_scan_geometry,
    calculate_scan_ray_paths,
)
from temsim.physics.wave_imaging import (
    WaveImagingResult,
    tem_wave_imaging_enabled,
)
from temsim.specimen.interaction_engine import run_specimen_interactions
from temsim.specimen.interaction_types import (
    SpecimenInteractionRequest,
    SpecimenInteractionResult,
)
from temsim.specimen.source import specimen_structure_available


@dataclass
class CalculationResult:
    simulation: Simulation
    energy_filter: object
    state_snapshot: object = None
    layout: object = None
    assembly: object = None
    specimen_interactions: SpecimenInteractionResult | None = None
    wave_imaging: WaveImagingResult | None = None
    scan_geometry: ScanGeometryResult | None = None
    scan_ray_paths: object | None = None
    stem_scan: StemScanResult | None = None
    lens_crossovers: tuple[dict[str, object], ...] = ()
    aperture_stops: tuple[dict[str, object], ...] = ()


def aperture_stop_records(state) -> tuple[dict[str, object], ...]:
    """Capture the exact runtime hard-edge controls used by the solver."""
    records = []
    gun = state.electron_gun
    apertures = [
        getattr(gun, name)
        for name in ("dpa_aperture", "c1_aperture")
        if getattr(gun, name, None) is not None
    ]
    apertures.extend(state.apertures)
    seen = set()
    for aperture in apertures:
        if aperture.key in seen:
            continue
        seen.add(aperture.key)
        diameter = getattr(aperture, "diameter_mm", None)
        if diameter is None:
            radius = getattr(
                aperture,
                "radius_mm",
                getattr(aperture, "effective_aperture_radius_mm", 0.0),
            )
            diameter = 2.0 * float(radius)
        else:
            radius = 0.5 * float(diameter)
        records.append(
            {
                "key": str(aperture.key),
                "name": str(aperture.name),
                "z_mm": float(aperture.z_mm),
                "diameter_mm": max(0.0, float(diameter)),
                # Kept for legacy ray-diagram readers.  Diameter is the public
                # aperture-size convention.
                "radius_mm": max(0.0, float(radius)),
                "offset_x_mm": float(getattr(aperture, "offset_x_mm", 0.0)),
                "offset_y_mm": float(getattr(aperture, "offset_y_mm", 0.0)),
                "enabled": bool(getattr(aperture, "enabled", True)),
                "installed": bool(getattr(aperture, "installed", True)),
            }
        )
    return tuple(records)


ProgressCallback = Callable[[int, int, str], None]


def _stem_frame_work_weight(state) -> int:
    """Return the number of independently completed STEM work units."""

    component = state.ac_deflector
    pixels_x = max(int(component.scan_pixels_x), 1)
    pixels_y = max(int(component.scan_lines), 1)
    sample = state.sample
    wave_requested = bool(getattr(sample, "stem_wave_enabled", False))
    if not wave_requested:
        # The geometric detector path reports one completed raster row.
        return pixels_y
    configuration_count = (
        max(int(sample.wave_frozen_phonon_configurations), 1)
        if bool(getattr(sample, "wave_atomistic_enabled", False))
        and bool(getattr(sample, "wave_frozen_phonon_enabled", False))
        else 1
    )
    # The wave path completes at most eight probe positions per batch and
    # repeats the propagation for every frozen-phonon configuration.
    batch_count = (pixels_x * pixels_y + 7) // 8
    return max(batch_count * configuration_count, 1)


def _geometric_specimen_transport_requested(state) -> bool:
    """Return whether geometric STEM needs finite-specimen exit particles."""

    sample = state.sample
    return bool(
        state.ac_deflector.enabled
        and state.ac_deflector.scan_enabled
        and getattr(sample, "inserted", False)
        and str(getattr(sample, "specimen_mode", "atomic")).strip().lower()
        == "atomic"
        and not bool(getattr(sample, "stem_wave_enabled", False))
        and specimen_structure_available(sample)
        and any(
            bool(getattr(detector, "inserted", False))
            for detector in state.stem_detectors
        )
    )


def calculate_stem_scan_frame(
    state,
    simulation,
    *,
    specimen_interactions: SpecimenInteractionResult | None = None,
    progress_callback: ProgressCallback | None = None,
):
    """Calculate exactly one detector-signal frame when AC scan is active."""

    component = state.ac_deflector
    if not bool(component.enabled and component.scan_enabled):
        return None
    return acquire_stem_scan(
        simulation,
        state,
        specimen_interactions=specimen_interactions,
        progress_callback=progress_callback,
    )


def calculate(
    state,
    *,
    progress_callback: ProgressCallback | None = None,
):
    """Normalise editable state and calculate all non-visual simulation results."""

    tem_wave_requested = tem_wave_imaging_enabled(state)
    stem_frame_requested = bool(
        state.ac_deflector.enabled and state.ac_deflector.scan_enabled
    )
    geometric_specimen_transport_requested = (
        _geometric_specimen_transport_requested(state)
    )
    stages = [
        ("Preparing state and physical layout", 1),
        ("Tracing the electron column", 1),
    ]
    if tem_wave_requested:
        stages.append(("Calculating the TEM wave image", 1))
    if geometric_specimen_transport_requested:
        stages.append(
            (
                "Transporting electrons through the specimen",
                max(int(state.electron_gun.ray_count), 1),
            )
        )
    stages.extend(
        (
            ("Tracing the energy filter", 1),
            ("Solving scan geometry", 1),
            ("Building scan-ray playback", 1),
        )
    )
    if stem_frame_requested:
        stages.append(
            (
                "Calculating the STEM detector frame",
                _stem_frame_work_weight(state),
            )
        )
    stages.append(("Finalising optical diagnostics", 1))
    total_work = sum(weight for _label, weight in stages)
    next_stage = 0
    completed_work = 0

    def report_stage() -> None:
        if progress_callback is None:
            return
        label = "Complete" if next_stage >= len(stages) else stages[next_stage][0]
        progress_callback(completed_work, total_work, label)

    def advance_stage() -> None:
        nonlocal completed_work, next_stage
        completed_work += stages[next_stage][1]
        next_stage += 1
        report_stage()

    stage_subdivisions = 10_000

    def report_current_stage_progress(
        completed: int,
        total: int,
        label: str,
    ) -> None:
        if progress_callback is None or int(total) <= 0:
            return
        bounded = min(max(int(completed), 0), int(total))
        stage_weight = stages[next_stage][1]
        progress_callback(
            completed_work * stage_subdivisions
            + round(
                stage_weight
                * stage_subdivisions
                * bounded
                / int(total)
            ),
            total_work * stage_subdivisions,
            label,
        )

    report_stage()
    ensure_recording_system(state)
    ensure_energy_filter(state)
    ensure_corrector_structure(state)
    normalise_component_names(state)
    layout = apply_physical_layout_to_state(state)
    ensure_recording_system(state)
    ensure_corrector_structure(state)
    advance_stage()
    simulation = run(state, resolved_layout=layout)
    advance_stage()
    specimen_interactions = None
    if tem_wave_requested:
        specimen_interactions = run_specimen_interactions(
            state,
            simulation,
            SpecimenInteractionRequest.tem_wave(),
        )
        wave_imaging = specimen_interactions.wave_imaging
        advance_stage()
    else:
        wave_imaging = None
    if geometric_specimen_transport_requested:
        specimen_interactions = run_specimen_interactions(
            state,
            simulation,
            SpecimenInteractionRequest.elastic_point(),
            existing_result=specimen_interactions,
            progress_callback=report_current_stage_progress,
        )
        advance_stage()
    if specimen_interactions is None:
        specimen_interactions = run_specimen_interactions(
            state,
            simulation,
            SpecimenInteractionRequest(),
        )
    energy_filter = simulate_energy_filter(state, simulation)
    advance_stage()
    scan_geometry = calculate_scan_geometry(state)
    advance_stage()
    scan_ray_paths = calculate_scan_ray_paths(state, simulation)
    advance_stage()
    stem_scan = None
    if stem_frame_requested:
        stem_scan = calculate_stem_scan_frame(
            state,
            simulation,
            specimen_interactions=specimen_interactions,
            progress_callback=report_current_stage_progress,
        )
        advance_stage()
    state.energy_filter_result = energy_filter
    lens_crossovers = detect_all_lens_crossovers(
        [simulation.incident, *simulation.branches.values()], state.lenses)
    state.all_lens_crossovers = lens_crossovers
    result = CalculationResult(
        simulation=simulation,
        energy_filter=energy_filter,
        state_snapshot=state,
        layout=layout,
        assembly=state._resolved_assembly,
        specimen_interactions=specimen_interactions,
        wave_imaging=wave_imaging,
        scan_geometry=scan_geometry,
        scan_ray_paths=scan_ray_paths,
        stem_scan=stem_scan,
        lens_crossovers=tuple(lens_crossovers),
        aperture_stops=aperture_stop_records(state),
    )
    advance_stage()
    return result
