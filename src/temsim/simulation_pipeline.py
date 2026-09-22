"""Application-facing simulation pipeline, independent of the GUI."""
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from time import perf_counter
from types import SimpleNamespace
from temsim import input_io

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
from temsim.physics.optical_tuning import check_tuning_cancelled
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
    signatures: dict[str, str] = field(default_factory=dict)
    calculated_products: frozenset[str] = frozenset()
    reused_products: frozenset[str] = frozenset()
    cache_hit: bool = False
    performance: dict[str, object] = field(default_factory=dict)
    external_inputs: tuple[ExternalInputIdentity, ...] | None = None
    calculation_manifest: object | None = None
    working_point_parent_id: str | None = None
    particle_signals: object | None = None

    def __post_init__(self):
        if not isinstance(self.signatures, dict):
            raise TypeError("CalculationResult.signatures must be a dictionary")
        if any(not isinstance(key, str) or not key.strip()
               or not isinstance(value, str) or not value.strip()
               for key, value in self.signatures.items()):
            raise ValueError("CalculationResult signatures require nonempty string keys and values")


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
        record = {
            "key": str(aperture.key),
            "name": str(aperture.name),
            "z_mm": float(aperture.z_mm),
            "enabled": bool(getattr(aperture, "enabled", True)),
            "installed": bool(getattr(aperture, "installed", True)),
        }
        if bool(getattr(aperture, "_slit_mode", False)):
            slit = aperture._slit_profile
            record.update({
                "name": str(aperture.label),
                "shape": "two_blade_slit",
                # The solver retains the mechanical bore even with blades
                # retracted, independently of the circular aperture setting.
                "enabled": True,
                "bore_diameter_mm": float(aperture.mechanical_bore_diameter_mm),
                "slit_inserted": bool(slit.inserted),
                "slit_gap_mm": float(slit.gap_um) * 1.0e-3,
                "slit_centre_x_mm": float(slit.centre_offset_um) * 1.0e-3,
            })
        else:
            record.update({
                "shape": "circular",
                "diameter_mm": float(aperture.diameter_mm),
                "offset_x_mm": float(getattr(aperture, "offset_x_mm", 0.0)),
                "offset_y_mm": float(getattr(aperture, "offset_y_mm", 0.0)),
            })
        records.append(record)
    return tuple(records)


ProgressCallback = Callable[[int, int, str], None]
_QT_PROGRESS_SAFE_MAX = 2_000_000_000


def _cancellable_progress(state, callback: ProgressCallback | None) -> ProgressCallback:
    """Keep cooperative cancellation active without a progress display.

    Material histories, EDS tracks/photon batches and STEM rows already report
    safe boundaries. Check both sides of a display callback because it may
    synchronously request cancellation itself.
    """
    check_tuning_cancelled(state)

    def report(completed: int, total: int, label: str) -> None:
        check_tuning_cancelled(state)
        if callback is not None:
            callback(completed, total, label)
        check_tuning_cancelled(state)

    return report


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
    """Inserted material scatters classical particles even without a raster."""

    sample = state.sample
    return bool(
        specimen_interactions_active(sample)
        and str(getattr(sample, "specimen_mode", "atomic")).strip().lower()
        in {"atomic", "reference"}
        and not bool(getattr(sample, "stem_wave_enabled", False))
        and (not bool(getattr(sample, "wave_enabled", False))
             or (state.ac_deflector.enabled and state.ac_deflector.scan_enabled
                 and any(bool(getattr(detector, "inserted", False))
                         for detector in state.stem_detectors)))
    )


def _eds_point_requested(state) -> bool:
    sample = state.sample
    return bool(
        getattr(sample, "eds_enabled", False)
        and specimen_interactions_active(sample)
    )


@input_io.using_state_inputs
def calculate_stem_scan_frame(
    state,
    simulation,
    *,
    specimen_interactions: SpecimenInteractionResult | None = None,
    geometric_specimen_exit=None,
    geometric_specimen_exit_signature: str = "",
    observation_stop_z_mm: float | None = None,
    scan_calibrated: bool = False,
    progress_callback: ProgressCallback | None = None,
    diffraction_sink=None,
):
    """Calculate exactly one detector-signal frame when AC scan is active."""

    progress_callback = _cancellable_progress(state, progress_callback)
    component = state.ac_deflector
    if not bool(component.enabled and component.scan_enabled and getattr(state.sample, "stem_image_enabled", True)):
        return None
    extra = {"diffraction_sink": diffraction_sink} if diffraction_sink is not None else {}
    if observation_stop_z_mm is not None:
        extra["observation_stop_z_mm"] = observation_stop_z_mm
    if scan_calibrated:
        extra["scan_calibrated"] = True
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


def _energy_filter_source_view(simulation, specimen_exit, signatures, *, no_illumination=False):
    """Select executed downstream branches without changing the column result."""
    values = dict(vars(simulation))
    metrics = dict(getattr(simulation, "metrics", {}) or {})
    if no_illumination:
        values["branches"] = {}
        provenance = "no_incident_illumination"
        dependency = signatures["column"]
    elif specimen_exit is not None:
        checked = validated_geometric_specimen_exit(specimen_exit, signatures["sample_downstream"])
        if checked is None:
            raise ValueError("Energy-filter entrance requires a validated specimen-exit checkpoint")
        values["branches"] = {
            f"specimen_exit:{index}:{getattr(branch, 'name', '')}": branch
            for index, branch in enumerate(checked.branches)
        }
        metrics.update(checked.metrics)
        metrics["branch_weights_are_absolute"] = True
        provenance = "validated_specimen_exit"
        dependency = signatures["sample_downstream"]
    else:
        provenance = "optical_column_reference"
        dependency = signatures["column"]
    metrics.update(energy_filter_entrance_provenance=provenance,
                   energy_filter_entrance_dependency_signature=dependency)
    values["metrics"] = metrics
    return SimpleNamespace(**values)


@input_io.using_state_inputs
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
    progress_callback = _cancellable_progress(state, progress_callback)
    from temsim.physics.illumination import illumination_config
    illumination_config(state)
    from temsim.physics.source_admission import admit_requested_wave_products
    admit_requested_wave_products(state)
    ensure_recording_system(state)
    ensure_energy_filter(state)
    ensure_corrector_structure(state)
    normalise_component_names(state)
    from temsim.geometry_effects import admit_state_geometry
    admit_state_geometry(state)
    external_inputs = capture_external_input_identities(state)
    signatures = calculation_signatures(state)
    assert_external_input_inventory_unchanged(state, external_inputs)
    reusable = matching_products(
        getattr(existing_result, "signatures", None), signatures
    )
    if getattr(existing_result, "loaded_section_only", False):
        # A section archive retains transport state, not every optional output
        # or derived diagnostic of the original complete result.
        reusable = frozenset()
    tem_wave_requested = tem_wave_imaging_enabled(state)
    stem_frame_requested = bool(
        state.ac_deflector.enabled and state.ac_deflector.scan_enabled
        and getattr(state.sample, "stem_image_enabled", True)
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
    expected_filter_provenance = (
        "validated_specimen_exit" if sample_downstream_requested or (
            cached_specimen_exit is not None and "sample_downstream" in reusable
        ) else "optical_column_reference"
    )
    if column_reused and sample_illumination_absent(existing_result.simulation, state):
        expected_filter_provenance = "no_incident_illumination"
    expected_filter_dependency = signatures[
        "sample_downstream" if expected_filter_provenance == "validated_specimen_exit" else "column"
    ]
    if energy_filter_reused and bool(getattr(state.energy_filter, "enabled", False)):
        # A matching parameter signature does not turn an older optical
        # reference into a completed finite-specimen downstream calculation.
        old_filter = existing_result.energy_filter
        energy_filter_reused = bool(
            getattr(old_filter, "entrance_provenance", "") == expected_filter_provenance
            and getattr(old_filter, "entrance_dependency_signature", "") == expected_filter_dependency
        )

    eds_stage_pending = eds_point_requested and not eds_reused
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
    if eds_stage_pending:
        stages.append("Resolving the EDS point spectrum")
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
    if bool(state.energy_filter.enabled) and not energy_filter_reused:
        stages.append("Tracing the energy filter")
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
    point_request_fields = {}
    if (not no_illumination and specimen_interactions_active(state.sample)
            and not bool(state.ac_deflector.enabled and state.ac_deflector.scan_enabled)):
        from temsim.specimen.elastic_transport import incident_rays_from_simulation
        executed_point = incident_rays_from_simulation(state, simulation).original_centroid_nm
        point_request_fields = {"point_x_nm": float(executed_point[0]),
                                "point_y_nm": float(executed_point[1])}
    keep_observables: set[SpecimenObservable] = set()
    previous_interactions = (
        existing_result.specimen_interactions
        if existing_result is not None
        else None
    )
    restored_material, material_resume_cache = (None, None)
    eds_response_replayed = False
    if not no_illumination and geometric_specimen_transport_requested:
        from temsim.physics.completed_particle_section import restore_material_interactions
        restored_material, material_resume_cache = restore_material_interactions(
            state, simulation, existing_result,
            progress_callback=report_current_stage_progress, replay_response=False)
        if restored_material is not None and (not elastic_reused or (
                eds_point_requested and not eds_reused and restored_material.eds_spectrum is not None)):
            if previous_interactions is not None and elastic_reused:
                previous_interactions = replace(previous_interactions,
                    eds_spectrum=restored_material.eds_spectrum,
                    completed_observables=(previous_interactions.completed_observables
                        | {SpecimenObservable.CHARACTERISTIC_X_RAY}),
                    metrics={**previous_interactions.metrics, "dependency_signatures": signatures})
            else:
                previous_interactions = restored_material
            keep_observables.update((SpecimenObservable.ELASTIC_TRANSPORT,
                                     SpecimenObservable.STOCHASTIC_INELASTIC))
            if eds_point_requested and restored_material.eds_spectrum is not None:
                keep_observables.add(SpecimenObservable.CHARACTERISTIC_X_RAY)
                eds_reused = True
                eds_response_replayed = bool(restored_material.metrics.get("eds_response_replayed", False))
                progress.stages = [
                    ("Updating EDS dose and spectral readout" if eds_response_replayed
                     else "Restoring the cached EDS point spectrum")
                    if label == "Resolving the EDS point spectrum" else label
                    for label in progress.stages]
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
                progress_callback=report_current_stage_progress,
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
                SpecimenInteractionRequest(observables=frozenset((
                    SpecimenObservable.ELASTIC_TRANSPORT,
                    SpecimenObservable.STOCHASTIC_INELASTIC)), **point_request_fields),
                existing_result=specimen_interactions,
                progress_callback=report_current_stage_progress,
            )
            (reused_products if restored_material is not None else calculated_products).add("elastic")
            advance_stage()
    elif geometric_specimen_transport_requested and not elastic_reused:
        advance_stage()
    if eds_point_requested and not no_illumination:
        if not eds_reused and material_resume_cache is not None and specimen_interactions is not None:
            from temsim.physics.completed_particle_section import replay_material_eds
            response_candidate = getattr(material_resume_cache, "eds_spectrum", None)
            if (material_resume_cache.signatures.get("eds_response") == signatures.get("eds_response")
                    and signatures.get("eds_response")
                    and getattr(response_candidate, "response_rates", None) is not None):
                progress.stages = tuple(
                    "Updating EDS dose and spectral readout" if label == "Resolving the EDS point spectrum"
                    else label for label in progress.stages)
                report_stage()
            spectrum = replay_material_eds(material_resume_cache, signatures, state,
                specimen_interactions.incident_bundle, progress_callback=report_current_stage_progress)
            if spectrum is not None:
                observables = specimen_interactions.completed_observables | {SpecimenObservable.CHARACTERISTIC_X_RAY}
                specimen_interactions = replace(specimen_interactions, eds_spectrum=spectrum,
                    request=replace(specimen_interactions.request, observables=frozenset(observables)),
                    completed_observables=frozenset(observables),
                    metrics={**specimen_interactions.metrics, "dependency_signatures": signatures})
                eds_reused = eds_response_replayed = True
            elif "Updating EDS dose and spectral readout" in progress.stages:
                # A matching signature alone does not supply an executed
                # response. An unavailable/ineligible product runs real EDS.
                progress.stages = tuple(
                    "Resolving the EDS point spectrum" if label == "Updating EDS dose and spectral readout"
                    else label for label in progress.stages)
                report_stage()
        if eds_reused:
            if eds_response_replayed or (specimen_interactions is not None
                                         and not specimen_interactions.conservation):
                # Selective retention clears derived ledgers even for an
                # exact in-memory EDS hit. Rebuild only a missing ledger (or
                # a changed dose); all physical products remain reusable.
                specimen_interactions = run_specimen_interactions(
                    state, simulation, SpecimenInteractionRequest(observables=frozenset((
                        SpecimenObservable.CHARACTERISTIC_X_RAY,)), **point_request_fields),
                    existing_result=specimen_interactions,
                    progress_callback=report_current_stage_progress)
            if eds_response_replayed:
                reused_products.add("eds_response")
                calculated_products.add("eds")
            else:
                reused_products.add("eds")
            if eds_stage_pending:
                advance_stage()
        else:
            from temsim.component_keys import EDS_DETECTOR_SYSTEM
            from temsim.detector.eds_geometry import EDSDetectorArrayGeometry

            detector_geometry = EDSDetectorArrayGeometry.from_part_data(
                state._resolved_assembly.part(EDS_DETECTOR_SYSTEM).data
            )
            specimen_interactions = run_specimen_interactions(
                state,
                simulation,
                SpecimenInteractionRequest(observables=frozenset((
                    SpecimenObservable.CHARACTERISTIC_X_RAY,)), **point_request_fields),
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
    elif eds_stage_pending:
        advance_stage()
    if (
        (bool(getattr(state.energy_filter, "enabled", False)) or geometric_specimen_transport_requested)
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
                ), **point_request_fields,
            ),
            existing_result=specimen_interactions,
            progress_callback=report_current_stage_progress,
        )
    if not no_illumination and (specimen_interactions is None
                                or not specimen_interactions.conservation):
        # Every physical augmentation already creates the current shared
        # ledger. Only a selectively retained/empty envelope needs this last
        # pass; rebuilding a large EDS ledger again repeats no useful work.
        specimen_interactions = run_specimen_interactions(
            state,
            simulation,
            SpecimenInteractionRequest(**point_request_fields),
            existing_result=specimen_interactions,
            progress_callback=report_current_stage_progress,
        )

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
            from temsim.physics.particle_sections import section_limits
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
                existing_exit=(material_resume_cache.specimen_exit if material_resume_cache is not None else None),
                stop_z_mm=section_limits(state)[1],
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

    if energy_filter_reused:
        energy_filter = existing_result.energy_filter
        reused_products.add("energy_filter")
    elif not bool(state.energy_filter.enabled):
        # No entrance state or filter transport is consumed by an inactive
        # component. Retain the explicit empty product for cache accounting.
        energy_filter = None
        calculated_products.add("energy_filter")
    else:
        filter_args = []
        filter_kwargs = {}
        if (
            specimen_interactions is not None
            and specimen_interactions.inelastic_distribution is not None
        ):
            filter_args.append(specimen_interactions.inelastic_distribution)
        if (
            str(getattr(state.energy_filter, "operating_mode", "eels")).lower() == "eftem"
            and wave_imaging is not None
        ):
            filter_kwargs["eftem_source_image"] = wave_imaging.camera_electron_optical_intensity
        filter_simulation = _energy_filter_source_view(
            simulation, specimen_exit, signatures, no_illumination=no_illumination,
        )
        energy_filter = simulate_energy_filter(
            state, filter_simulation, *filter_args, **filter_kwargs,
        )
        calculated_products.add("energy_filter")
        advance_stage()

    # Outputs belong to CalculationResult; keeping them on State makes the
    # next parameter snapshot depend on (or try to serialize) executed results.
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
    from temsim.physics.completed_particle_section import capture_completed_particle_section
    capture_completed_particle_section(result)
    assert_external_input_inventory_unchanged(state, external_inputs)
    advance_stage()
    result.performance = {
        "pipeline_seconds": max(0.0, progress.stage_started_at - calculation_started),
        "stages": tuple(progress.timings),
        "timing_scope": "Current pipeline call; excludes GUI drawing and worker setup",
        "last_cuda_column_execution": getattr(state, "_last_ray_device_receipt", None),
    }
    return result


@input_io.using_state_inputs
def calculate_particle_section(state, target_z_mm=None, component_keys=(), *,
                               existing_result=None, progress_callback=None):
    """Execute a tip-origin classical particle section, including its specimen.

    A missing target requests the complete physical path, including an assembled
    energy filter. An explicit straight-column target stops at that exact plane;
    no detector exposure downstream of that plane is implied.
    """
    import hashlib
    import math
    import numpy as np
    from temsim.physics.particle_sections import (
        run_particle_section, section_limits, _digest_arrays,
        MaterialSectionCache, MATERIAL_SECTION_SCHEMA,
        particle_section_downstream_signature,
    )
    from temsim.physics.optical_tuning import check_tuning_cancelled
    from temsim.specimen.source import specimen_is_vacuum
    from temsim.specimen.elastic_transport import incident_rays_from_simulation
    from temsim.calculation_manifest import solver_source_identity

    started = perf_counter()
    from temsim.physics.source_admission import admit_requested_wave_products
    admit_requested_wave_products(state)
    ensure_recording_system(state)
    ensure_energy_filter(state)
    ensure_corrector_structure(state)
    normalise_component_names(state)
    from temsim.geometry_effects import admit_state_geometry
    admit_state_geometry(state)
    layout = apply_physical_layout_to_state(state)
    lower, upper = section_limits(state)
    full_path = target_z_mm is None
    target = upper if full_path else float(target_z_mm)
    if not math.isfinite(target) or not lower <= target <= upper:
        raise ValueError(f"Particle section must lie within the straight-column interval {lower:g} to {upper:g} mm")
    # Raster commands are derived from the current lens state. Resolve them
    # before the reference particles consume any AC or descan kicks.
    scan_calibration = None
    if any(bool(component.enabled and component.scan_enabled)
           for component in (state.ac_deflector, state.descan_deflector)):
        from temsim.physics.scan_geometry import calibrate_scan_system
        scan_calibration = calibrate_scan_system(state, observation_stop_z_mm=target)
    external_inputs = capture_external_input_identities(state)
    base_signatures = calculation_signatures(state)
    signatures = dict(base_signatures)
    sample_z = float(state.sample.z_mm)
    reaches_material = target > sample_z and not specimen_is_vacuum(state.sample)
    if reaches_material and not specimen_interactions_active(state.sample):
        raise ValueError("Inserted specimen requires a configured physical structure before downstream particle transport")
    keys = tuple(str(key) for key in component_keys)
    scoped = lambda label, *values: hashlib.sha256(repr((MATERIAL_SECTION_SCHEMA, label, *values)).encode()).hexdigest()
    # A truncated section must never be mistaken for a complete column product
    # by the ordinary high-accuracy pipeline or a disk-cache consumer.
    signatures["column"] = scoped("column", base_signatures["column"], target)
    signatures["sample_downstream"] = particle_section_downstream_signature(state, target)
    signatures["energy_filter"] = scoped("filter", base_signatures["energy_filter"], target, full_path)
    signatures["request"] = scoped("request", base_signatures.get("request", ""), target, keys, full_path)
    if target < sample_z:
        signatures["incident"] = scoped("incident", base_signatures["incident"], target)
    material_signature = scoped("specimen", base_signatures["elastic"], solver_source_identity())
    signatures["section_material"] = material_signature
    calculated, reused = set(), set()
    stage_times = []
    timing_stage, timing_started = None, perf_counter()

    def progress(stage, message):
        nonlocal timing_stage, timing_started
        check_tuning_cancelled(state)
        if timing_stage is None or stage != timing_stage[0]:
            now = perf_counter()
            if timing_stage is not None:
                stage_times.append({"stage": timing_stage[1], "seconds": now-timing_started})
            timing_stage, timing_started = (stage, message), now
        if progress_callback is not None:
            progress_callback(stage, 4, message)

    progress(0, "Transporting emitted electrons to the selected section")
    previous_simulation = getattr(existing_result, "simulation", None)
    simulation = run_particle_section(state,
        observation_stop_z_mm=min(target, sample_z) if reaches_material else target,
        tuning_component_keys=keys, existing_simulation=previous_simulation,
        resolved_layout=layout)
    calculated.add("column")
    if simulation.metrics.get("section_reused_prefix"):
        reused.add("incident_prefix")
    no_illumination = sample_illumination_absent(simulation, state)
    interactions = None
    specimen_exit = None
    progress(1, "Resolving specimen interactions")
    if reaches_material and not no_illumination:
        incident = simulation.incident
        incident_digest = _digest_arrays(*(getattr(incident, name)[-1]
            for name in ("x", "tx", "y", "ty", "flight_time_s")),
            incident.alive, incident.blocked_z, incident.energy_offset_ev,
            incident.ray_weight, incident.source_ray_id)
        # Preserve the actual deflected beam position. The point-interaction
        # engine defaults to a translated scan-origin beam when coordinates
        # are omitted, which is not the current non-raster physical pixel.
        bundle = incident_rays_from_simulation(state, simulation)
        point = tuple(float(value) for value in bundle.original_centroid_nm)
        request = SpecimenInteractionRequest(
            observables=frozenset((SpecimenObservable.ELASTIC_TRANSPORT,
                                   SpecimenObservable.STOCHASTIC_INELASTIC)),
            point_x_nm=point[0], point_y_nm=point[1])
        old_cache = getattr(previous_simulation, "material_section_cache", None)
        cache_compatible = (
            isinstance(old_cache, MaterialSectionCache)
            and old_cache.schema == MATERIAL_SECTION_SCHEMA
            and old_cache.signatures.get("section_material") == material_signature
            and old_cache.incident_digest == incident_digest
            and old_cache.point_xy_nm == point
        )
        previous_interactions = None
        cached_eds = None
        eds_response_replayed = False
        if cache_compatible:
            from temsim.specimen.scene import SpecimenScene
            from temsim.physics.completed_particle_section import compatible_material_eds, replay_material_eds
            if (_eds_point_requested(state)
                    and not bool(state.ac_deflector.enabled and state.ac_deflector.scan_enabled)):
                cached_eds = compatible_material_eds(old_cache, base_signatures, state)
                if cached_eds is None:
                    progress(1, "Updating EDS dose and spectral readout")
                    cached_eds = replay_material_eds(old_cache, base_signatures, state, bundle,
                        progress_callback=(lambda done, total, label: progress(1, label)))
                    eds_response_replayed = cached_eds is not None
            retained = request.observables | (
                {SpecimenObservable.CHARACTERISTIC_X_RAY} if cached_eds is not None else set())
            previous_interactions = SpecimenInteractionResult(
                request=replace(request, observables=frozenset(retained)),
                completed_observables=frozenset(retained),
                scene=SpecimenScene.from_state(state), incident_bundle=bundle,
                elastic_transport=old_cache.elastic_transport,
                inelastic_distribution=old_cache.inelastic_distribution,
                eds_spectrum=cached_eds,
                metrics={"dependency_signatures": base_signatures})
        interactions = run_specimen_interactions(state, simulation, request,
            existing_result=previous_interactions,
            progress_callback=(lambda done, total, label: progress(1, label)))
        (reused if cache_compatible else calculated).update(("elastic", "inelastic"))
        if cached_eds is not None:
            reused.add("eds_response" if eds_response_replayed else "eds")
            if eds_response_replayed:
                calculated.add("eds")
        elif _eds_point_requested(state):
            from temsim.component_keys import EDS_DETECTOR_SYSTEM
            from temsim.detector.eds_geometry import EDSDetectorArrayGeometry
            geometry = EDSDetectorArrayGeometry.from_part_data(
                state._resolved_assembly.part(EDS_DETECTOR_SYSTEM).data)
            interactions = run_specimen_interactions(state, simulation,
                SpecimenInteractionRequest.eds_point(x_nm=point[0], y_nm=point[1]),
                existing_result=interactions, detector_geometry=geometry,
                progress_callback=(lambda done, total, label: progress(1, label)))
            calculated.add("eds")
        progress(2, "Transporting scattered specimen electrons")
        if cache_compatible and old_cache.target_z_mm == target:
            specimen_exit = validated_geometric_specimen_exit(
                old_cache.specimen_exit, signatures["sample_downstream"])
        if specimen_exit is None:
            # Keep the same intermediate observation planes as a subsequent
            # complete request. Adding a display plane later must not alter
            # the already executed material-prefix integration grid.
            downstream_distance_um = float(getattr(state.sample, "sample_region_downstream_distance_um", 0.))
            material_observation = sample_z + downstream_distance_um * 1e-3
            save_planes = (target, material_observation) if (downstream_distance_um > 0.
                and sample_z <= material_observation <= target) else (target,)
            specimen_exit = build_geometric_specimen_exit(state, simulation,
                interactions.elastic_transport, interactions.inelastic_distribution,
                stop_z_mm=target, save_z_mm=save_planes,
                existing_exit=old_cache.specimen_exit if cache_compatible else None,
                tuning_component_keys=keys,
                dependency_signature=signatures["sample_downstream"],
                progress_callback=(lambda done, total, label: progress(2, label)))
            calculated.add("sample_downstream")
        else:
            reused.add("sample_downstream")
        simulation.material_section_cache = MaterialSectionCache(
            dict(signatures), incident_digest, point, interactions.elastic_transport,
            interactions.inelastic_distribution, specimen_exit, target,
            eds_spectrum=interactions.eds_spectrum)
        simulation.real_interactions = interactions.inelastic_distribution
    elif reaches_material:
        # Empty physical output is distinct from a material-free reference.
        specimen_exit = build_geometric_specimen_exit(state, simulation, None,
            stop_z_mm=target, dependency_signature=signatures["sample_downstream"])
        calculated.add("sample_downstream")
    if specimen_exit is not None:
        simulation.branches = {f"specimen_exit:{index}:{branch.name}": branch
                               for index, branch in enumerate(specimen_exit.branches)}
        simulation.metrics.update(specimen_exit.metrics)
        simulation.metrics["branch_weights_are_absolute"] = True
        resume_rows = tuple(specimen_exit.metrics.get("material_section_resume", ()))
        if resume_rows:
            starts = tuple(float(row["resume_z_mm"]) for row in resume_rows)
            simulation.metrics["section_incident_resume_z_mm"] = simulation.metrics["section_resume_z_mm"]
            # Display a common continuation plane only when every outgoing
            # branch reused its executed state; retain the full range as well.
            simulation.metrics["section_material_resume_range_mm"] = (min(starts), max(starts))
            if all(bool(row["hit"]) for row in resume_rows):
                simulation.metrics["section_resume_z_mm"] = min(starts)
                simulation.metrics["section_reused_prefix"] = True
                simulation.metrics["section_reuse_reason"] = "compatible_executed_prefix"
    # Never retain the upstream helper's assertion that scattering was omitted
    # once the current request has actually transported the finite specimen.
    simulation.metrics.update(
        particle_section=True, particle_tuning=True, optical_tuning=False,
        section_physics_scope="classical_particles_with_specimen_interactions",
        section_target_z_mm=target, section_component_keys=keys,
        section_resumable_through_z_mm=target,
        section_full_path=full_path,
        section_scan_calibration_scope=("not_consumed" if scan_calibration is None
            else "auxiliary_reference_matrices_may_extend_beyond_particle_section"),
        sample_scattering_applied=bool(reaches_material and not no_illumination),
        sample_scattering_model=("finite_geometry_elastic_x_inelastic_tensor_product"
            if reaches_material else "not_reached" if target <= sample_z else "vacuum"),
        section_downstream_provenance=("validated_specimen_exit" if specimen_exit is not None
            else "incident_only" if target <= sample_z else "vacuum_transport"))
    progress(3, "Resolving assembled downstream instruments")
    stem_scan = None
    scan_geometry = scan_ray_paths = None
    scan_requested = bool(state.ac_deflector.enabled and state.ac_deflector.scan_enabled
                          and getattr(state.sample, "stem_image_enabled", True))
    inserted_detectors = tuple(detector for detector in state.stem_detectors if detector.inserted)
    scan_planes_reached = target > sample_z and all(float(detector.z_mm) <= target for detector in inserted_detectors)
    simulation.metrics["section_scan_status"] = (
        "disabled" if not scan_requested else "detectors_not_reached" if not scan_planes_reached
        else "no_inserted_detectors" if not inserted_detectors else "no_illumination" if no_illumination else "calculated")
    if scan_requested and scan_planes_reached and inserted_detectors and not no_illumination:
        scan_geometry = calculate_scan_geometry(state, observation_stop_z_mm=target,
            calibration=scan_calibration)
        scan_ray_paths = calculate_scan_ray_paths(state, simulation, calibrated=True)
        stem_scan = calculate_stem_scan_frame(state, simulation,
            specimen_interactions=interactions, geometric_specimen_exit=specimen_exit,
            geometric_specimen_exit_signature=signatures["sample_downstream"],
            observation_stop_z_mm=target, scan_calibrated=True,
            progress_callback=(lambda done, total, label: progress(3, label)))
        calculated.add("stem_scan")
    energy_filter = None
    if full_path and bool(getattr(state.energy_filter, "enabled", False)):
        source_view = _energy_filter_source_view(simulation, specimen_exit,
            signatures, no_illumination=no_illumination)
        energy_filter = simulate_energy_filter(state, source_view,
            inelastic_distribution=getattr(interactions, "inelastic_distribution", None))
        calculated.add("energy_filter")
    diagnostic_started = perf_counter()
    lens_crossovers = detect_all_lens_crossovers(
        [simulation.incident, *simulation.branches.values()], state.lenses)
    aperture_stops = aperture_stop_records(state)
    diagnostic_seconds = perf_counter() - diagnostic_started
    result = CalculationResult(simulation=simulation, energy_filter=energy_filter,
        state_snapshot=state, layout=layout, assembly=getattr(state, "_resolved_assembly", None),
        specimen_interactions=interactions, specimen_exit=specimen_exit,
        stem_scan=stem_scan, lens_crossovers=tuple(lens_crossovers), aperture_stops=tuple(aperture_stops),
        scan_geometry=scan_geometry, scan_ray_paths=scan_ray_paths,
        signatures=signatures, external_inputs=external_inputs,
        calculated_products=frozenset(calculated), reused_products=frozenset(reused),
        performance={"pipeline_seconds": perf_counter()-started,
                     "ray_diagnostics_seconds": diagnostic_seconds,
                     "timing_scope": "Executed classical particle section"})
    assert_external_input_inventory_unchanged(state, external_inputs)
    progress(4, "Complete")
    result.performance["stages"] = tuple(stage_times)
    result.performance["gun_seconds"] = simulation.metrics.get("section_gun_seconds")
    result.performance["pipeline_seconds"] = max(0.0, perf_counter() - started)
    return result
