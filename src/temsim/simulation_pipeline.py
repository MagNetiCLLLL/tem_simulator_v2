"""Application-facing simulation pipeline, independent of the Tk GUI."""
from collections.abc import Callable
from dataclasses import dataclass, replace

from temsim.calculation_cache import calculation_signatures, matching_products
from temsim.detector.recording_system import ensure_recording_system
from temsim.detector.stem_signal import StemScanResult, acquire_stem_scan
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.component_names import normalise_component_names
from temsim.optics.corrector_structure import ensure_corrector_structure
from temsim.optics.energy_filter import ensure_energy_filter
from temsim.optics.energy_filter_raytrace import simulate_energy_filter
from temsim.physics.all_lens_crossovers import detect_all_lens_crossovers
from temsim.physics.simulation import Simulation, run
from temsim.physics.beam_current import (
    column_current_limit_percent,
    effective_source_current_pa,
)
from temsim.physics.scan_geometry import (
    ScanGeometryResult,
    calculate_scan_geometry,
    calculate_scan_ray_paths,
)
from temsim.physics.wave_imaging import (
    WaveImagingResult,
    reproject_wave_image,
    tem_wave_imaging_enabled,
)
from temsim.specimen.interaction_engine import run_specimen_interactions
from temsim.specimen.interaction_types import (
    SpecimenObservable,
    SpecimenInteractionRequest,
    SpecimenInteractionResult,
    retain_specimen_observables,
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
    sample_region: object | None = None
    lens_crossovers: tuple[dict[str, object], ...] = ()
    aperture_stops: tuple[dict[str, object], ...] = ()
    model_signature: str = ""
    signatures: dict[str, str] | None = None
    calculated_products: frozenset[str] = frozenset()
    reused_products: frozenset[str] = frozenset()
    cache_hit: bool = False


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


def _eds_point_requested(state) -> bool:
    sample = state.sample
    return bool(
        getattr(sample, "inserted", False)
        and getattr(sample, "eds_enabled", False)
        and specimen_structure_available(sample)
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
    existing_result: CalculationResult | None = None,
):
    """Calculate a complete result while reusing compatible cached products.

    A previous complete result is a seed, never an output target.  Every
    product is reused only when its dependency-scoped signature matches the
    current immutable calculation snapshot.  Failed or cancelled replacement
    work therefore cannot partially mutate the previous complete result.
    """

    ensure_recording_system(state)
    ensure_energy_filter(state)
    ensure_corrector_structure(state)
    normalise_component_names(state)
    signatures = calculation_signatures(state)
    reusable = matching_products(
        getattr(existing_result, "signatures", None), signatures
    )
    tem_wave_requested = tem_wave_imaging_enabled(state)
    stem_frame_requested = bool(
        state.ac_deflector.enabled and state.ac_deflector.scan_enabled
    )
    geometric_specimen_transport_requested = (
        _geometric_specimen_transport_requested(state)
    )
    eds_point_requested = _eds_point_requested(state)
    column_reused = bool(
        existing_result is not None
        and existing_result.simulation is not None
        and "column" in reusable
    )
    incident_reused = bool(
        existing_result is not None
        and existing_result.simulation is not None
        and "incident" in reusable
    )
    wave_reused = bool(
        tem_wave_requested
        and existing_result is not None
        and existing_result.wave_imaging is not None
        and "wave" in reusable
    )
    wave_source_reused = bool(
        tem_wave_requested
        and not wave_reused
        and existing_result is not None
        and existing_result.wave_imaging is not None
        and existing_result.wave_imaging.projector_checkpoint is not None
        and "wave_source" in reusable
    )
    elastic_reused = bool(
        (geometric_specimen_transport_requested or eds_point_requested)
        and existing_result is not None
        and existing_result.specimen_interactions is not None
        and existing_result.specimen_interactions.elastic_transport is not None
        and "elastic" in reusable
    )
    eds_reused = bool(
        eds_point_requested
        and existing_result is not None
        and existing_result.specimen_interactions is not None
        and existing_result.specimen_interactions.eds_spectrum is not None
        and "eds" in reusable
    )
    energy_filter_reused = bool(
        existing_result is not None
        and "energy_filter" in reusable
    )
    scan_geometry_reused = bool(
        existing_result is not None
        and "scan" in reusable
    )
    scan_paths_reused = bool(
        existing_result is not None
        and "scan" in reusable
    )
    stem_reused = bool(
        stem_frame_requested
        and existing_result is not None
        and existing_result.stem_scan is not None
        and "stem" in reusable
    )
    sample_region_reused = bool(
        existing_result is not None
        and existing_result.sample_region is not None
        and "sample_region" in reusable
    )

    stages = [("Preparing state and physical layout", 1)]
    if not column_reused:
        stages.append(("Tracing the electron column", 1))
    if tem_wave_requested and not wave_reused:
        stages.append((
            (
                "Projecting the cached Objective wave to the recording plane"
                if wave_source_reused
                else "Calculating the TEM wave image"
            ),
            1,
        ))
    if geometric_specimen_transport_requested and not elastic_reused:
        stages.append(
            (
                "Transporting electrons through the specimen",
                max(int(state.electron_gun.ray_count), 1),
            )
        )
    if eds_point_requested and not eds_reused:
        stages.append(
            (
                "Calculating the EDS point spectrum",
                (
                    1
                    if elastic_reused
                    or geometric_specimen_transport_requested
                    else max(int(state.electron_gun.ray_count), 1)
                ),
            )
        )
    if not energy_filter_reused:
        stages.append(("Tracing the energy filter", 1))
    if not scan_geometry_reused:
        stages.append(("Solving scan geometry", 1))
    if not scan_paths_reused:
        stages.append(("Building scan-ray playback", 1))
    if stem_frame_requested and not stem_reused:
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

    calculated_products: set[str] = set()
    reused_products: set[str] = set()
    report_stage()
    layout = apply_physical_layout_to_state(state)
    ensure_recording_system(state)
    ensure_corrector_structure(state)
    advance_stage()

    if column_reused:
        previous_simulation = existing_result.simulation
        metrics = dict(previous_simulation.metrics)
        surviving_fraction = float(
            metrics.get("sample_beam_surviving_fraction", 0.0)
        )
        source_current_pa = effective_source_current_pa(state)
        metrics.update({
            "column_current_limit_percent": (
                column_current_limit_percent(state)
            ),
            "effective_source_current_pa": source_current_pa,
            "sample_surviving_current_pa": (
                source_current_pa * surviving_fraction
            ),
        })
        simulation = replace(previous_simulation, metrics=metrics)
        reused_products.add("column")
    else:
        run_kwargs = {"resolved_layout": layout}
        if existing_result is not None:
            run_kwargs["existing_simulation"] = existing_result.simulation
        simulation = run(state, **run_kwargs)
        calculated_products.add("column")
        if incident_reused:
            reused_products.add("incident")
        advance_stage()

    keep_observables: set[SpecimenObservable] = set()
    previous_interactions = (
        existing_result.specimen_interactions
        if existing_result is not None
        else None
    )
    if previous_interactions is not None and incident_reused:
        if wave_reused or wave_source_reused:
            keep_observables.add(SpecimenObservable.COHERENT_ELASTIC_WAVE)
        if "elastic" in reusable:
            keep_observables.add(SpecimenObservable.ELASTIC_TRANSPORT)
        if "eds" in reusable:
            keep_observables.add(SpecimenObservable.CHARACTERISTIC_X_RAY)
        if previous_interactions.inelastic_distribution is not None:
            keep_observables.add(SpecimenObservable.STOCHASTIC_INELASTIC)
    specimen_interactions = retain_specimen_observables(
        previous_interactions,
        frozenset(keep_observables),
    )

    if tem_wave_requested:
        if wave_reused:
            wave_imaging = existing_result.wave_imaging
            reused_products.add("wave")
        elif wave_source_reused:
            wave_imaging = reproject_wave_image(
                state, existing_result.wave_imaging
            )
            reused_products.add("wave_source")
            calculated_products.add("wave_projection")
            if specimen_interactions is not None:
                interaction_metrics = dict(specimen_interactions.metrics)
                interaction_metrics["dependency_signatures"] = signatures
                specimen_interactions = replace(
                    specimen_interactions,
                    wave_imaging=wave_imaging,
                    metrics=interaction_metrics,
                )
            advance_stage()
        else:
            specimen_interactions = run_specimen_interactions(
                state,
                simulation,
                SpecimenInteractionRequest.tem_wave(),
                existing_result=specimen_interactions,
            )
            wave_imaging = specimen_interactions.wave_imaging
            calculated_products.add("wave")
            advance_stage()
    else:
        wave_imaging = None
        if specimen_interactions is not None:
            specimen_interactions = retain_specimen_observables(
                specimen_interactions,
                frozenset(
                    specimen_interactions.completed_observables
                    - {SpecimenObservable.COHERENT_ELASTIC_WAVE}
                ),
            )

    if geometric_specimen_transport_requested:
        if elastic_reused:
            reused_products.add("elastic")
        else:
            specimen_interactions = run_specimen_interactions(
                state,
                simulation,
                SpecimenInteractionRequest.elastic_point(),
                existing_result=specimen_interactions,
                progress_callback=report_current_stage_progress,
            )
            calculated_products.add("elastic")
            advance_stage()
    if eds_point_requested:
        if eds_reused:
            reused_products.add("eds")
        else:
            from temsim.component_keys import EDS_DETECTOR_SYSTEM
            from temsim.detector.eds_geometry import EDSDetectorArrayGeometry

            detector_geometry = EDSDetectorArrayGeometry.from_part_data(
                state._resolved_assembly.part(EDS_DETECTOR_SYSTEM).data
            )
            specimen_interactions = run_specimen_interactions(
                state,
                simulation,
                SpecimenInteractionRequest.eds_point(),
                detector_geometry=detector_geometry,
                existing_result=specimen_interactions,
                progress_callback=report_current_stage_progress,
            )
            calculated_products.add("eds")
            advance_stage()
        if (
            specimen_interactions is not None
            and specimen_interactions.elastic_transport is not None
        ):
            if elastic_reused:
                reused_products.add("elastic")
            elif not geometric_specimen_transport_requested:
                calculated_products.add("elastic")
    specimen_interactions = run_specimen_interactions(
        state,
        simulation,
        SpecimenInteractionRequest(),
        existing_result=specimen_interactions,
    )

    if energy_filter_reused:
        energy_filter = existing_result.energy_filter
        reused_products.add("energy_filter")
    else:
        energy_filter = simulate_energy_filter(state, simulation)
        calculated_products.add("energy_filter")
        advance_stage()

    if scan_geometry_reused:
        scan_geometry = existing_result.scan_geometry
        reused_products.add("scan_geometry")
    else:
        scan_geometry = calculate_scan_geometry(state)
        calculated_products.add("scan_geometry")
        advance_stage()

    if scan_paths_reused:
        scan_ray_paths = existing_result.scan_ray_paths
        reused_products.add("scan_ray_paths")
    else:
        scan_ray_paths = calculate_scan_ray_paths(state, simulation)
        calculated_products.add("scan_ray_paths")
        advance_stage()

    if stem_frame_requested:
        if stem_reused:
            stem_scan = existing_result.stem_scan
            reused_products.add("stem")
        else:
            stem_scan = calculate_stem_scan_frame(
                state,
                simulation,
                specimen_interactions=specimen_interactions,
                progress_callback=report_current_stage_progress,
            )
            calculated_products.add("stem")
            advance_stage()
    else:
        stem_scan = None
    sample_region = (
        existing_result.sample_region if sample_region_reused else None
    )
    if sample_region_reused:
        reused_products.add("sample_region")

    state.energy_filter_result = energy_filter
    if column_reused and existing_result.lens_crossovers:
        lens_crossovers = existing_result.lens_crossovers
        aperture_stops = existing_result.aperture_stops
        reused_products.add("diagnostics")
    else:
        lens_crossovers = detect_all_lens_crossovers(
            [simulation.incident, *simulation.branches.values()], state.lenses
        )
        aperture_stops = aperture_stop_records(state)
        calculated_products.add("diagnostics")
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
        sample_region=sample_region,
        lens_crossovers=tuple(lens_crossovers),
        aperture_stops=tuple(aperture_stops),
        signatures=signatures,
        calculated_products=frozenset(calculated_products),
        reused_products=frozenset(reused_products),
    )
    advance_stage()
    return result
