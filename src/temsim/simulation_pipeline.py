"""Application-facing simulation pipeline, independent of the Tk GUI."""
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from time import perf_counter

from temsim.calculation_cache import calculation_signatures, matching_products
from temsim.calculation_manifest import (
    ExternalInputIdentity,
    assert_external_input_inventory_unchanged,
    capture_external_input_identities,
)
from temsim.detector.recording_system import ensure_recording_system
from temsim.detector.stem_signal import (
    StemScanResult,
    acquire_stem_scan,
    reweight_stem_scan,
)
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
    sample_illumination_absent,
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
from temsim.specimen.source import (
    specimen_interactions_active,
)
from temsim.specimen.downstream_transport import (
    GeometricSpecimenExit,
    build_geometric_specimen_exit,
    validated_geometric_specimen_exit,
)
from temsim.specimen.sample_region import (
    bind_sample_region_downstream,
    validated_sample_region_exit,
)


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
    specimen_exit: GeometricSpecimenExit | None = None
    sample_region: object | None = None
    lens_crossovers: tuple[dict[str, object], ...] = ()
    aperture_stops: tuple[dict[str, object], ...] = ()
    model_signature: str = ""
    signatures: dict[str, str] | None = None
    calculated_products: frozenset[str] = frozenset()
    reused_products: frozenset[str] = frozenset()
    cache_hit: bool = False
    performance: dict[str, object] = field(default_factory=dict)
    external_inputs: tuple[ExternalInputIdentity, ...] | None = None


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
    nanopulser = getattr(state, "nanopulser", None)
    if nanopulser is not None and bool(nanopulser.installed):
        apertures.append(nanopulser.aperture)
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
_QT_PROGRESS_SAFE_MAX = 2_000_000_000


def _progress_stage_subdivisions(total_work: int) -> int:
    """Return nested-progress resolution bounded to a Qt signed integer."""

    work = max(int(total_work), 1)
    return max(1, min(10_000, _QT_PROGRESS_SAFE_MAX // work))


def _bounded_progress_position(
    completed_work: int,
    total_work: int,
) -> tuple[int, int]:
    """Map arbitrary work counts onto Qt's signed progress range."""

    total = max(int(total_work), 1)
    completed = min(max(int(completed_work), 0), total)
    if total <= _QT_PROGRESS_SAFE_MAX:
        return completed, total
    return (
        round(_QT_PROGRESS_SAFE_MAX * completed / total),
        _QT_PROGRESS_SAFE_MAX,
    )


class _StageProgress:
    """Equal stage slots plus honest local counts, not a runtime estimate.

    Electron histories, photon events and probe batches are different units.
    They only measure their own step, never the weight of a pipeline stage.
    The outer position is monotonic even when a stage starts a new local step.
    """

    def __init__(self, stages: list[str], callback: ProgressCallback | None,
                 *, started_at: float | None = None):
        self.stages = tuple(stages)
        self.callback = callback
        self.index = 0
        self.subdivisions = _progress_stage_subdivisions(len(self.stages))
        self.total = max(len(self.stages), 1) * self.subdivisions
        self.position = 0
        self.started_at = perf_counter() if started_at is None else started_at
        self.stage_started_at = self.started_at
        self.timings: list[dict[str, object]] = []

    def _emit(self, position: int, label: str) -> None:
        if self.callback is None:
            return
        self.position = max(self.position, min(max(position, 0), self.total))
        completed, total = _bounded_progress_position(self.position, self.total)
        self.callback(completed, total, label)

    def report(self) -> None:
        label = (
            "Complete" if self.index >= len(self.stages)
            else f"Stage {self.index + 1}/{len(self.stages)} | {self.stages[self.index]}"
        )
        self._emit(self.index * self.subdivisions, label)

    def advance(self) -> None:
        now = perf_counter()
        if self.index < len(self.stages):
            self.timings.append({
                "stage": self.stages[self.index],
                "seconds": max(0.0, now - self.stage_started_at),
            })
        self.stage_started_at = now
        self.index += 1
        self.report()

    def update(self, completed: int, total: int, label: str) -> None:
        if self.callback is None or int(total) <= 0 or self.index >= len(self.stages):
            return
        total = int(total)
        completed = min(max(int(completed), 0), total)
        fraction = completed / total
        # A finished substep is not a finished stage. Only advance() may
        # complete its slot, including when photons follow electron histories.
        position = (self.index * self.subdivisions
                    + min(round(self.subdivisions * fraction), self.subdivisions - 1))
        self._emit(position, (
            f"Stage {self.index + 1}/{len(self.stages)} | {label}"
            f" | stage progress {100.0 * fraction:.1f}%"
        ))


def _geometric_specimen_transport_requested(state) -> bool:
    """Return whether geometric STEM needs finite-specimen exit particles."""

    sample = state.sample
    return bool(
        state.ac_deflector.enabled
        and state.ac_deflector.scan_enabled
        and specimen_interactions_active(sample)
        and str(getattr(sample, "specimen_mode", "atomic")).strip().lower()
        in {"atomic", "reference"}
        and not bool(getattr(sample, "stem_wave_enabled", False))
        and any(
            bool(getattr(detector, "inserted", False))
            for detector in state.stem_detectors
        )
    )


def _eds_point_requested(state) -> bool:
    sample = state.sample
    return bool(
        getattr(sample, "eds_enabled", False)
        and specimen_interactions_active(sample)
    )


def calculate_stem_scan_frame(
    state,
    simulation,
    *,
    specimen_interactions: SpecimenInteractionResult | None = None,
    geometric_specimen_exit=None,
    geometric_specimen_exit_signature: str = "",
    progress_callback: ProgressCallback | None = None,
    diffraction_sink=None,
):
    """Calculate exactly one detector-signal frame when AC scan is active."""

    component = state.ac_deflector
    if not bool(component.enabled and component.scan_enabled):
        return None
    extra = {"diffraction_sink": diffraction_sink} if diffraction_sink is not None else {}
    return acquire_stem_scan(
        simulation,
        state,
        specimen_interactions=specimen_interactions,
        geometric_specimen_exit=geometric_specimen_exit,
        geometric_specimen_exit_signature=geometric_specimen_exit_signature,
        progress_callback=progress_callback,
        **extra,
    )


def _rebind_reused_sample_region(
    sample_region,
    specimen_interactions: SpecimenInteractionResult | None,
    wave_imaging: WaveImagingResult | None,
    dependency_signatures: dict[str, str] | None = None,
):
    """Attach current shared observables to reusable particle geometry."""

    spectrum = getattr(specimen_interactions, "eds_spectrum", None)
    if (
        sample_region is None
        or specimen_interactions is None
        or spectrum is None
    ):
        return None
    metrics = dict(getattr(sample_region, "metrics", {}) or {})
    metrics["channeling_model"] = (
        "coherent wave/multislice result available; not reclassified as particles"
        if wave_imaging is not None
        else "unavailable without a wave/multislice specimen result"
    )
    if dependency_signatures is not None:
        metrics["sample_region_signature"] = str(
            dependency_signatures.get("sample_region", "")
        )
    return replace(
        sample_region,
        interactions=specimen_interactions,
        spectrum=spectrum,
        metrics=metrics,
    )


def calculate(
    state,
    *,
    progress_callback: ProgressCallback | None = None,
    existing_result: CalculationResult | None = None,
    diffraction_sink=None,
):
    """Calculate a complete result while reusing compatible cached products.

    A previous complete result is a seed, never an output target.  Every
    product is reused only when its dependency-scoped signature matches the
    current immutable calculation snapshot.  Failed or cancelled replacement
    work therefore cannot partially mutate the previous complete result.
    """

    calculation_started = perf_counter()
    ensure_recording_system(state)
    ensure_energy_filter(state)
    ensure_corrector_structure(state)
    normalise_component_names(state)
    external_inputs = capture_external_input_identities(state)
    signatures = calculation_signatures(state)
    assert_external_input_inventory_unchanged(state, external_inputs)
    reusable = matching_products(
        getattr(existing_result, "signatures", None), signatures
    )
    tem_wave_requested = tem_wave_imaging_enabled(state)
    stem_frame_requested = bool(
        state.ac_deflector.enabled and state.ac_deflector.scan_enabled
    )
    scan_geometry_requested = bool(
        (state.ac_deflector.enabled and state.ac_deflector.scan_enabled)
        or (
            state.descan_deflector.enabled
            and state.descan_deflector.scan_enabled
        )
    )
    scan_paths_requested = bool(
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
        scan_geometry_requested
        and existing_result is not None
        and existing_result.scan_geometry is not None
        and "scan_geometry" in reusable
    )
    scan_paths_reused = bool(
        scan_paths_requested
        and existing_result is not None
        and existing_result.scan_ray_paths is not None
        and "scan_ray_paths" in reusable
    )
    fourdstem_requested = bool(
        stem_frame_requested
        and (getattr(state.sample, "stem_fourdstem_enabled", False) or diffraction_sink is not None)
        and getattr(state.sample, "stem_wave_enabled", False)
        and str(getattr(state, "illumination_mode", "")).upper() == "STEM"
    )
    existing_stem_scan = (
        existing_result.stem_scan if existing_result is not None else None
    )
    existing_fourdstem = getattr(
        existing_stem_scan, "fourdstem_artifact", None
    )
    artifact_metadata = getattr(existing_fourdstem, "metadata", {}) or {}
    artifact_provenance = (
        artifact_metadata.get("provenance", {})
        if hasattr(artifact_metadata, "get")
        else {}
    )
    fourdstem_cube_reused = bool(
        fourdstem_requested
        and existing_fourdstem is not None
        and "fourdstem_cube" in reusable
        and hasattr(artifact_provenance, "get")
        and artifact_provenance.get("fourdstem_cube_state_signature")
        == signatures["fourdstem_cube"]
    )
    stem_reused = bool(
        stem_frame_requested
        and existing_stem_scan is not None
        and "stem" in reusable
        and (not fourdstem_requested or fourdstem_cube_reused)
    )
    stem_transport_reused = bool(
        stem_frame_requested
        and not stem_reused
        and existing_stem_scan is not None
        and "stem_transport" in reusable
        and (not fourdstem_requested or fourdstem_cube_reused)
    )
    from temsim.physics.diffraction_memory import can_recollect_stem
    stem_cube_recollection = bool(diffraction_sink is not None and fourdstem_cube_reused
                                  and can_recollect_stem(existing_stem_scan)
                                  and not stem_reused and not stem_transport_reused)
    existing_sample_region = (
        existing_result.sample_region
        if existing_result is not None
        else None
    )
    existing_region_metrics = dict(
        getattr(existing_sample_region, "metrics", {}) or {}
    )
    sample_region_reused = bool(
        existing_sample_region is not None
        and "sample_region" in reusable
        and str(existing_region_metrics.get("sample_region_signature", ""))
        == signatures["sample_region"]
    )
    cached_specimen_exit = validated_geometric_specimen_exit(
        (
            getattr(existing_result, "specimen_exit", None)
            if existing_result is not None
            else None
        ),
        signatures["sample_downstream"],
    )
    if cached_specimen_exit is None and sample_region_reused:
        cached_specimen_exit = validated_sample_region_exit(
            existing_sample_region,
            signatures["sample_downstream"],
        )
    sample_downstream_requested = bool(
        geometric_specimen_transport_requested or sample_region_reused
    )
    sample_downstream_reused = bool(
        sample_downstream_requested
        and cached_specimen_exit is not None
        and "sample_downstream" in reusable
    )

    stages = ["Preparing state and physical layout"]
    if not column_reused:
        stages.append("Tracing the electron column")
    if tem_wave_requested and not wave_reused:
        stages.append(
            "Projecting the cached Objective wave to the recording plane"
            if wave_source_reused
            else "Calculating the TEM wave image"
        )
    if geometric_specimen_transport_requested and not elastic_reused:
        stages.append("Transporting electrons through the specimen")
    if eds_point_requested and not eds_reused:
        stages.append("Calculating the EDS point spectrum")
    if not energy_filter_reused:
        stages.append("Tracing the energy filter")
    if scan_geometry_requested and not scan_geometry_reused:
        stages.append("Solving scan geometry")
    if scan_paths_requested and not scan_paths_reused:
        stages.append("Building scan-ray playback")
    if sample_downstream_requested and not sample_downstream_reused:
        stages.append("Propagating specimen-exit electrons downstream")
    if stem_transport_reused:
        stages.append("Updating STEM dose readout")
    elif stem_cube_recollection:
        stages.append("Recollecting cached STEM diffraction")
    elif stem_frame_requested and not stem_reused:
        stages.append("Calculating the STEM detector frame")
    stages.append("Finalising optical diagnostics")
    progress = _StageProgress(stages, progress_callback, started_at=calculation_started)
    report_stage = progress.report
    advance_stage = progress.advance
    report_current_stage_progress = progress.update

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

    no_illumination = sample_illumination_absent(simulation, state)
    if no_illumination:
        simulation = replace(simulation, metrics={
            **simulation.metrics,
            "sample_illumination_status": "no incident current",
            "sample_surviving_current_pa": 0.0,
            "specimen_products_status": "No electrons illuminate the specimen",
        })
    keep_observables: set[SpecimenObservable] = set()
    previous_interactions = (
        existing_result.specimen_interactions
        if existing_result is not None
        else None
    )
    if previous_interactions is not None and incident_reused and not no_illumination:
        if wave_reused or wave_source_reused:
            keep_observables.add(SpecimenObservable.COHERENT_ELASTIC_WAVE)
        if "elastic" in reusable:
            keep_observables.add(SpecimenObservable.ELASTIC_TRANSPORT)
        if "eds" in reusable:
            keep_observables.add(SpecimenObservable.CHARACTERISTIC_X_RAY)
        if previous_interactions.inelastic_distribution is not None:
            keep_observables.add(SpecimenObservable.STOCHASTIC_INELASTIC)
    specimen_interactions = retain_specimen_observables(
        previous_interactions if not no_illumination else None,
        frozenset(keep_observables),
    )

    if tem_wave_requested and not no_illumination:
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
        if tem_wave_requested and not wave_reused:
            advance_stage()
        if specimen_interactions is not None:
            specimen_interactions = retain_specimen_observables(
                specimen_interactions,
                frozenset(
                    specimen_interactions.completed_observables
                    - {SpecimenObservable.COHERENT_ELASTIC_WAVE}
                ),
            )

    if geometric_specimen_transport_requested and not no_illumination:
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
    elif geometric_specimen_transport_requested and not elastic_reused:
        advance_stage()
    if eds_point_requested and not no_illumination:
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
    elif eds_point_requested and not eds_reused:
        advance_stage()
    if (
        bool(getattr(state.energy_filter, "enabled", False))
        and not no_illumination
        and specimen_interactions_active(state.sample)
        and (
            specimen_interactions is None
            or specimen_interactions.inelastic_distribution is None
        )
    ):
        # The EELS/EFTEM chain must consume the same immutable specimen loss
        # distribution as every other observable.  Request it here only when
        # the shared envelope does not already carry it; the filter must never
        # independently resample or reinterpret the material.
        specimen_interactions = run_specimen_interactions(
            state,
            simulation,
            SpecimenInteractionRequest(
                observables=frozenset(
                    (SpecimenObservable.STOCHASTIC_INELASTIC,)
                )
            ),
            existing_result=specimen_interactions,
        )
    if not no_illumination:
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
        filter_args = []
        filter_kwargs = {}
        if (
            specimen_interactions is not None
            and specimen_interactions.inelastic_distribution is not None
        ):
            filter_args.append(
                specimen_interactions.inelastic_distribution
            )
        if (
            str(
                getattr(state.energy_filter, "operating_mode", "eels")
            ).lower()
            == "eftem"
            and wave_imaging is not None
        ):
            filter_kwargs["eftem_source_image"] = (
                wave_imaging.camera_electron_optical_intensity
            )
        energy_filter = simulate_energy_filter(
            state,
            simulation,
            *filter_args,
            **filter_kwargs,
        )
        calculated_products.add("energy_filter")
        advance_stage()

    if scan_geometry_reused:
        scan_geometry = existing_result.scan_geometry
        reused_products.add("scan_geometry")
    elif scan_geometry_requested:
        scan_geometry = calculate_scan_geometry(state)
        if scan_geometry is not None:
            calculated_products.add("scan_geometry")
        advance_stage()
    else:
        scan_geometry = None

    if scan_paths_reused:
        scan_ray_paths = existing_result.scan_ray_paths
        reused_products.add("scan_ray_paths")
    elif scan_paths_requested:
        scan_ray_paths = calculate_scan_ray_paths(state, simulation)
        if scan_ray_paths is not None:
            calculated_products.add("scan_ray_paths")
        advance_stage()
    else:
        scan_ray_paths = None

    # Retain a compatible first-class checkpoint even when this particular
    # request does not consume it.  It is immutable and may save a complete
    # post-specimen rebuild when the user next enables a dependent view.
    specimen_exit = (
        cached_specimen_exit
        if cached_specimen_exit is not None
        and "sample_downstream" in reusable
        else None
    )
    if sample_downstream_requested and not no_illumination:
        if sample_downstream_reused:
            specimen_exit = cached_specimen_exit
            reused_products.add("sample_downstream")
        else:
            elastic = getattr(specimen_interactions, "elastic_transport", None)
            if elastic is None:
                raise RuntimeError(
                    "Specimen-exit transport requires a current elastic result"
                )
            spectrum_elastic = getattr(
                getattr(specimen_interactions, "eds_spectrum", None),
                "elastic_transport",
                None,
            )
            if spectrum_elastic is not None and spectrum_elastic is not elastic:
                raise ValueError(
                    "EDS and downstream transport must share one elastic result"
                )
            downstream_distance_um = float(
                getattr(
                    state.sample,
                    "sample_region_downstream_distance_um",
                    0.0,
                )
            )
            save_z_mm = (
                float(existing_sample_region.exit_z_mm),
            ) if sample_region_reused else (
                (
                    float(state.sample.z_mm)
                    + downstream_distance_um * 1.0e-3
                ),
            ) if downstream_distance_um > 0.0 else ()
            specimen_exit = build_geometric_specimen_exit(
                state,
                simulation,
                elastic,
                getattr(
                    specimen_interactions,
                    "inelastic_distribution",
                    None,
                ),
                save_z_mm=save_z_mm,
                dependency_signature=signatures["sample_downstream"],
                progress_callback=report_current_stage_progress,
            )
            calculated_products.add("sample_downstream")
            advance_stage()
    elif sample_downstream_requested and not sample_downstream_reused:
        # Preserve monotonic progress when no electrons reach the specimen.
        advance_stage()

    sample_region = None
    if sample_region_reused and not no_illumination:
        sample_region = _rebind_reused_sample_region(
            existing_sample_region,
            specimen_interactions,
            wave_imaging,
            signatures,
        )
        if sample_region is not None and specimen_exit is not None:
            sample_region = bind_sample_region_downstream(
                sample_region,
                specimen_exit,
                specimen_interactions,
                expected_signature=signatures["sample_downstream"],
                wave_imaging=wave_imaging,
            )
    if sample_region is not None:
        reused_products.add("sample_region")

    if stem_frame_requested:
        if stem_reused:
            stem_scan = existing_result.stem_scan
            reused_products.add("stem")
            if fourdstem_cube_reused:
                reused_products.add("fourdstem_cube")
        elif stem_transport_reused:
            stem_scan = reweight_stem_scan(state, existing_stem_scan)
            reused_products.add("stem_transport")
            calculated_products.add("stem")
            if fourdstem_cube_reused:
                reused_products.add("fourdstem_cube")
            advance_stage()
        elif stem_cube_recollection:
            from temsim.physics.diffraction_memory import recollect_stem
            stem_scan = recollect_stem(state, existing_stem_scan)
            reused_products.add("fourdstem_cube")
            calculated_products.add("stem")
            advance_stage()
        else:
            stem_kwargs = {
                "specimen_interactions": specimen_interactions,
                "progress_callback": report_current_stage_progress,
            }
            if diffraction_sink is not None:
                stem_kwargs["diffraction_sink"] = diffraction_sink
            if specimen_exit is not None:
                stem_kwargs["geometric_specimen_exit"] = (
                    specimen_exit
                )
                stem_kwargs["geometric_specimen_exit_signature"] = (
                    signatures["sample_downstream"]
                )
            stem_scan = calculate_stem_scan_frame(
                state,
                simulation,
                **stem_kwargs,
            )
            calculated_products.add("stem")
            if getattr(stem_scan, "fourdstem_artifact", None) is not None:
                calculated_products.add("fourdstem_cube")
            advance_stage()
    else:
        stem_scan = None

    state.energy_filter_result = energy_filter
    if column_reused:
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
    from temsim.optics.aberrations import prepare_field_aberration_diagnostics
    if any(getattr(state, f"{system}_aberrations", {}).get("mode") == "field_derived"
           for system in ("probe", "image")):
        report_current_stage_progress(0, 1, "Preparing field-derived aberration diagnostics")
        prepare_field_aberration_diagnostics(
            state, getattr(existing_result, "state_snapshot", None)
        )
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
        specimen_exit=specimen_exit,
        sample_region=sample_region,
        lens_crossovers=tuple(lens_crossovers),
        aperture_stops=tuple(aperture_stops),
        signatures=signatures,
        external_inputs=external_inputs,
        calculated_products=frozenset(calculated_products),
        reused_products=frozenset(reused_products),
    )
    advance_stage()
    result.performance = {
        "pipeline_seconds": max(0.0, progress.stage_started_at - calculation_started),
        "stages": tuple(progress.timings),
        "timing_scope": "Current pipeline call; excludes GUI drawing and worker setup",
    }
    assert_external_input_inventory_unchanged(state, external_inputs)
    return result
