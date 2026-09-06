from dataclasses import replace
from pathlib import Path
import shutil
from types import SimpleNamespace
import tomllib

import pytest
import tomli_w

import temsim.gui.calculation_controller as calculation_controller_module
import temsim.simulation_pipeline as simulation_pipeline
from temsim.assembly_catalog import AssemblyCatalog
from temsim.calculation_manifest import (
    resolved_assembly_geometry_fingerprint,
)
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.gui.calculation_controller import (
    CalculationController,
    HIGH_ACCURACY_MEMORY_BUDGET_BYTES,
    estimate_calculation_memory_bytes,
)
from temsim.calculation_cache import calculation_signatures
from temsim.simulation_pipeline import CalculationResult
from temsim.optics.column import default_state
from temsim.specimen.sample_region import SampleRegionResult


def _extend_test_module(root: Path, relative: str, extension_mm: float) -> None:
    path = root / relative
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    document["ports"]["exit"]["local_z_mm"] += float(extension_mm)
    document["geometry"]["length_mm"] += float(extension_mm)
    path.write_text(tomli_w.dumps(document), encoding="utf-8")


@pytest.mark.parametrize(
    ("selection_updates", "module_path", "extension_mm"),
    (
        ({"gun": "FEG + Mono"}, "gun/FEG_Mono.toml", 37.0),
        ({"beam_blanker": "NanoPulser"}, "beam_blanker/NanoPulser.toml", 23.0),
    ),
)
def test_worker_snapshot_keeps_exact_selected_assembly_and_layout(
    tmp_path,
    monkeypatch,
    selection_updates,
    module_path,
    extension_mm,
):
    reference_catalog = AssemblyCatalog()
    root = tmp_path / "instruments"
    shutil.copytree(reference_catalog.root, root)
    _extend_test_module(root, module_path, extension_mm)
    catalog = AssemblyCatalog(root)
    selection = replace(catalog.default_selection(), **selection_updates)
    state = default_state()
    catalog.apply(state, selection)
    by_lens_key = {lens.key: lens for lens in state.lenses}
    by_lens_key["condenser_lens_1"].percent = 17.125
    by_lens_key["condenser_lens_2"].percent = 34.25
    installed_assembly = state._resolved_assembly
    live_layout = apply_physical_layout_to_state(state)
    live_layout_positions = tuple(
        (
            component.key,
            component.local_s_range_mm,
            component.optical_reference_plane_z_mm,
        )
        for component in live_layout
    )
    live_positions = {
        component.key: float(component.z_mm)
        for component in state.lenses
        if hasattr(component, "z_mm")
    }
    live_positions["sample"] = float(state.sample.z_mm)
    live_fingerprint = resolved_assembly_geometry_fingerprint(state)
    captured = {}

    def fake_calculate(worker_state, *, progress_callback):
        del progress_callback
        worker_layout = apply_physical_layout_to_state(worker_state)
        captured["assembly"] = worker_state._resolved_assembly
        captured["layout"] = tuple(
            (
                component.key,
                component.local_s_range_mm,
                component.optical_reference_plane_z_mm,
            )
            for component in worker_layout
        )
        captured["positions"] = {
            component.key: float(component.z_mm)
            for component in worker_state.lenses
            if hasattr(component, "z_mm")
        }
        captured["positions"]["sample"] = float(worker_state.sample.z_mm)
        captured["fingerprint"] = resolved_assembly_geometry_fingerprint(
            worker_state
        )
        worker_lenses = {lens.key: lens for lens in worker_state.lenses}
        captured["strengths"] = (
            float(worker_lenses["condenser_lens_1"].percent),
            float(worker_lenses["condenser_lens_2"].percent),
        )
        return object()

    monkeypatch.setattr(
        calculation_controller_module,
        "calculate",
        fake_calculate,
    )
    controller = CalculationController(persistent_cache_enabled=False)
    workers = []
    failures = []
    controller.pool.start = workers.append
    controller.failed.connect(lambda *values: failures.append(values))
    controller.submit(state, "High accuracy", 25, 5.0)
    workers[0].run()

    assert not failures, failures
    assert captured["assembly"] is installed_assembly
    assert captured["assembly"].root == root.resolve()
    assert captured["assembly"].selected_module_paths == (
        installed_assembly.selected_module_paths
    )
    assert captured["layout"] == live_layout_positions
    assert captured["positions"] == live_positions
    assert captured["fingerprint"] == live_fingerprint
    assert captured["strengths"] == pytest.approx((17.125, 34.25))


def test_high_accuracy_pipeline_reports_completed_real_stages(monkeypatch):
    state = default_state()
    state.sample.wave_enabled = False
    state.sample.eds_enabled = False
    state.ac_deflector.scan_enabled = False
    simulation = SimpleNamespace(
        incident=SimpleNamespace(),
        branches={},
    )

    monkeypatch.setattr(
        simulation_pipeline,
        "ensure_recording_system",
        lambda _state: None,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "ensure_energy_filter",
        lambda _state: None,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "ensure_corrector_structure",
        lambda _state: None,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "normalise_component_names",
        lambda _state: None,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "apply_physical_layout_to_state",
        lambda _state: "layout",
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "run",
        lambda _state, resolved_layout: simulation,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "simulate_energy_filter",
        lambda _state, _simulation: "energy-filter",
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "calculate_scan_geometry",
        lambda _state: "scan-geometry",
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "calculate_scan_ray_paths",
        lambda _state, _simulation: "scan-rays",
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "detect_all_lens_crossovers",
        lambda _branches, _lenses: (),
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "aperture_stop_records",
        lambda _state: (),
    )

    progress = []
    result = simulation_pipeline.calculate(
        state,
        progress_callback=lambda completed, total, stage: progress.append(
            (completed, total, stage)
        ),
    )

    assert result.simulation is simulation
    assert result.specimen_interactions is not None
    assert not result.specimen_interactions.completed_observables
    assert result.scan_geometry is None
    assert result.scan_ray_paths is None
    assert progress == [
        (0, 4, "Preparing state and physical layout"),
        (1, 4, "Tracing the electron column"),
        (2, 4, "Tracing the energy filter"),
        (3, 4, "Finalising optical diagnostics"),
        (4, 4, "Complete"),
    ]


def test_scan_geometry_and_playback_are_reused_independently(monkeypatch):
    state = default_state()
    state.sample.wave_enabled = False
    state.sample.eds_enabled = False
    state.ac_deflector.enabled = True
    state.ac_deflector.scan_enabled = True
    signatures = simulation_pipeline.calculation_signatures(state)
    cached_geometry = object()
    cached_paths = object()
    existing = simulation_pipeline.CalculationResult(
        simulation=object(),
        energy_filter=None,
        scan_geometry=cached_geometry,
        scan_ray_paths=cached_paths,
        signatures={
            "request": "older-request",
            "scan_geometry": signatures["scan_geometry"],
            "scan_ray_paths": "stale-playback",
        },
    )
    simulation = SimpleNamespace(
        incident=SimpleNamespace(),
        branches={},
    )
    for name in (
        "ensure_recording_system",
        "ensure_energy_filter",
        "ensure_corrector_structure",
        "normalise_component_names",
    ):
        monkeypatch.setattr(simulation_pipeline, name, lambda _state: None)
    monkeypatch.setattr(
        simulation_pipeline,
        "apply_physical_layout_to_state",
        lambda _state: "layout",
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "run",
        lambda _state, **_kwargs: simulation,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "sample_illumination_absent",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "run_specimen_interactions",
        lambda *_args, **kwargs: kwargs.get("existing_result"),
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "simulate_energy_filter",
        lambda *_args: None,
    )

    def geometry_must_not_run(_state):
        raise AssertionError("matching scan geometry was recalculated")

    rebuilt_paths = object()
    path_calls = []
    monkeypatch.setattr(
        simulation_pipeline,
        "calculate_scan_geometry",
        geometry_must_not_run,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "calculate_scan_ray_paths",
        lambda passed_state, passed_simulation: (
            path_calls.append((passed_state, passed_simulation))
            or rebuilt_paths
        ),
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "calculate_stem_scan_frame",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "detect_all_lens_crossovers",
        lambda *_args: (),
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "aperture_stop_records",
        lambda _state: (),
    )

    result = simulation_pipeline.calculate(state, existing_result=existing)

    assert result.scan_geometry is cached_geometry
    assert result.scan_ray_paths is rebuilt_paths
    assert path_calls == [(state, simulation)]
    assert "scan_geometry" in result.reused_products
    assert "scan_ray_paths" in result.calculated_products


def test_matching_scan_signature_without_artifact_recalculates(monkeypatch):
    state = default_state()
    state.sample.wave_enabled = False
    state.sample.eds_enabled = False
    state.ac_deflector.enabled = True
    state.ac_deflector.scan_enabled = True
    signatures = simulation_pipeline.calculation_signatures(state)
    existing = simulation_pipeline.CalculationResult(
        simulation=object(),
        energy_filter=None,
        scan_geometry=None,
        signatures={
            "request": "older-request",
            "scan_geometry": signatures["scan_geometry"],
        },
    )
    simulation = SimpleNamespace(incident=SimpleNamespace(), branches={})
    for name in (
        "ensure_recording_system",
        "ensure_energy_filter",
        "ensure_corrector_structure",
        "normalise_component_names",
    ):
        monkeypatch.setattr(simulation_pipeline, name, lambda _state: None)
    monkeypatch.setattr(
        simulation_pipeline,
        "apply_physical_layout_to_state",
        lambda _state: None,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "run",
        lambda _state, **_kwargs: simulation,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "sample_illumination_absent",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "run_specimen_interactions",
        lambda *_args, **kwargs: kwargs.get("existing_result"),
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "simulate_energy_filter",
        lambda *_args: None,
    )
    rebuilt_geometry = object()
    monkeypatch.setattr(
        simulation_pipeline,
        "calculate_scan_geometry",
        lambda _state: rebuilt_geometry,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "calculate_scan_ray_paths",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "calculate_stem_scan_frame",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "detect_all_lens_crossovers",
        lambda *_args: (),
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "aperture_stop_records",
        lambda _state: (),
    )

    result = simulation_pipeline.calculate(state, existing_result=existing)

    assert result.scan_geometry is rebuilt_geometry
    assert "scan_geometry" in result.calculated_products
    assert "scan_geometry" not in result.reused_products


def test_geometric_real_sample_requests_finite_specimen_transport():
    state = default_state()
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = "configured.cif"
    state.sample.inserted = True
    state.sample.stem_wave_enabled = False
    state.ac_deflector.enabled = True
    state.ac_deflector.scan_enabled = True

    assert simulation_pipeline._geometric_specimen_transport_requested(state)

    state.sample.stem_wave_enabled = True
    assert not simulation_pipeline._geometric_specimen_transport_requested(state)

    state.sample.stem_wave_enabled = False
    state.sample.specimen_mode = "virtual"
    assert not simulation_pipeline._geometric_specimen_transport_requested(state)


def test_vacuum_and_zero_thickness_skip_particle_specimen_transport():
    state = default_state()
    state.sample.specimen_mode = "virtual"
    state.sample.specimen_preset_key = "vacuum"
    state.sample.inserted = True
    state.sample.eds_enabled = True

    assert not simulation_pipeline._eds_point_requested(state)

    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = "configured.cif"
    state.sample.thickness_nm = 0.0
    state.sample.stem_wave_enabled = False
    state.ac_deflector.enabled = True
    state.ac_deflector.scan_enabled = True

    assert not simulation_pipeline._eds_point_requested(state)
    assert not simulation_pipeline._geometric_specimen_transport_requested(state)


def test_reused_sample_region_rebinds_current_shared_observables():
    old_interactions = SimpleNamespace(eds_spectrum="old-spectrum")
    old_region = SampleRegionResult(
        entry_z_mm=1.0,
        exit_z_mm=2.0,
        electron_paths=("cached-electron-path",),
        photon_paths=("cached-photon-path",),
        downstream_branches=("cached-downstream-branch",),
        spectrum="old-spectrum",
        interactions=old_interactions,
        metrics={"channeling_model": "old", "kept": 1},
    )
    current_interactions = SimpleNamespace(eds_spectrum="current-spectrum")

    rebound = simulation_pipeline._rebind_reused_sample_region(
        old_region,
        current_interactions,
        wave_imaging=object(),
    )

    assert rebound is not old_region
    assert rebound.electron_paths is old_region.electron_paths
    assert rebound.photon_paths is old_region.photon_paths
    assert rebound.downstream_branches is old_region.downstream_branches
    assert rebound.interactions is current_interactions
    assert rebound.spectrum == "current-spectrum"
    assert rebound.metrics["kept"] == 1
    assert rebound.metrics["channeling_model"].startswith("coherent wave")
    assert old_region.metrics["channeling_model"] == "old"


def test_pipeline_reprojects_only_stale_sample_downstream(monkeypatch):
    state = default_state()
    state.sample.wave_enabled = False
    state.ac_deflector.scan_enabled = False
    state.descan_deflector.scan_enabled = False
    signatures = simulation_pipeline.calculation_signatures(state)
    old_electron_paths = ("cached-electron",)
    old_photon_paths = ("cached-photon",)
    old_region = SampleRegionResult(
        entry_z_mm=1.0,
        exit_z_mm=2.0,
        electron_paths=old_electron_paths,
        photon_paths=old_photon_paths,
        downstream_branches=("old-downstream",),
        spectrum="old-spectrum",
        interactions="old-interactions",
        metrics={
            "kept": 1,
            "sample_region_signature": signatures["sample_region"],
            "sample_downstream_signature": "stale-downstream",
        },
    )
    elastic = object()
    spectrum = SimpleNamespace(elastic_transport=elastic)
    interactions = SimpleNamespace(
        elastic_transport=elastic,
        eds_spectrum=spectrum,
        inelastic_distribution=object(),
        metrics={"dependency_signatures": signatures},
        completed_observables=frozenset(),
    )
    existing = simulation_pipeline.CalculationResult(
        simulation=object(),
        energy_filter=None,
        specimen_interactions=interactions,
        sample_region=old_region,
        signatures={
            "request": "older-request",
            "elastic": signatures["elastic"],
            "eds": signatures["eds"],
            "sample_region": signatures["sample_region"],
            "sample_downstream": "stale-downstream",
        },
    )
    simulation = SimpleNamespace(incident=SimpleNamespace(), branches={})
    for name in (
        "ensure_recording_system",
        "ensure_energy_filter",
        "ensure_corrector_structure",
        "normalise_component_names",
    ):
        monkeypatch.setattr(simulation_pipeline, name, lambda _state: None)
    monkeypatch.setattr(
        simulation_pipeline,
        "apply_physical_layout_to_state",
        lambda _state: None,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "run",
        lambda _state, **_kwargs: simulation,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "sample_illumination_absent",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "retain_specimen_observables",
        lambda current, _keep: current,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "run_specimen_interactions",
        lambda *_args, **_kwargs: interactions,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "simulate_energy_filter",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "detect_all_lens_crossovers",
        lambda *_args: (),
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "aperture_stop_records",
        lambda _state: (),
    )
    calls = []
    new_downstream_branch = SimpleNamespace(weight=0.5)

    def fake_build(
        passed_state,
        passed_simulation,
        passed_elastic,
        passed_inelastic,
        **kwargs,
    ):
        calls.append((
            passed_state,
            passed_simulation,
            passed_elastic,
            passed_inelastic,
            kwargs,
        ))
        return simulation_pipeline.GeometricSpecimenExit(
            (new_downstream_branch,),
            {
                "tracked_downstream_source_probability": 0.5,
                "inelastic_absorbed_source_probability": 0.1,
            },
            dependency_signature=kwargs["dependency_signature"],
        )

    monkeypatch.setattr(
        simulation_pipeline,
        "build_geometric_specimen_exit",
        fake_build,
    )
    progress = []

    result = simulation_pipeline.calculate(
        state,
        existing_result=existing,
        progress_callback=lambda completed, total, label: progress.append(
            (completed, total, label)
        ),
    )

    assert len(calls) == 1
    assert result.sample_region.electron_paths is old_electron_paths
    assert result.sample_region.photon_paths is old_photon_paths
    assert result.sample_region.downstream_branches == (new_downstream_branch,)
    assert result.sample_region.specimen_exit is result.specimen_exit
    assert "sample_region" in result.reused_products
    assert "sample_downstream" in result.calculated_products
    assert "Propagating specimen-exit electrons downstream" in {
        label for _completed, _total, label in progress
    }


def test_pipeline_shares_cached_specimen_exit_with_geometric_stem(monkeypatch):
    state = default_state()
    state.sample.wave_enabled = False
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = "configured.cif"
    state.sample.stem_wave_enabled = False
    state.ac_deflector.enabled = True
    state.ac_deflector.scan_enabled = True
    state.descan_deflector.scan_enabled = False
    assert simulation_pipeline._geometric_specimen_transport_requested(state)

    signatures = simulation_pipeline.calculation_signatures(state)
    downstream_branches = (SimpleNamespace(weight=0.75),)
    elastic = object()
    interactions = SimpleNamespace(
        elastic_transport=elastic,
        eds_spectrum=SimpleNamespace(elastic_transport=elastic),
        inelastic_distribution=object(),
        metrics={"dependency_signatures": signatures},
        completed_observables=frozenset(),
    )
    checkpoint = simulation_pipeline.GeometricSpecimenExit(
        downstream_branches,
        {
            "tracked_downstream_source_probability": 0.75,
            "inelastic_absorbed_source_probability": 0.05,
        },
        dependency_signature=signatures["sample_downstream"],
    )
    sample_region = SampleRegionResult(
        entry_z_mm=1.0,
        exit_z_mm=2.0,
        electron_paths=("cached-electron",),
        photon_paths=("cached-photon",),
        downstream_branches=downstream_branches,
        spectrum=interactions.eds_spectrum,
        interactions=interactions,
        metrics={
            "tracked_downstream_source_probability": 0.75,
            "sample_region_signature": signatures["sample_region"],
            "sample_downstream_signature": signatures["sample_downstream"],
        },
        specimen_exit=checkpoint,
    )
    existing = simulation_pipeline.CalculationResult(
        simulation=object(),
        energy_filter=None,
        specimen_interactions=interactions,
        specimen_exit=checkpoint,
        sample_region=sample_region,
        signatures={
            "request": "older-request",
            "incident": signatures["incident"],
            "elastic": signatures["elastic"],
            "eds": signatures["eds"],
            "sample_region": signatures["sample_region"],
            "sample_downstream": signatures["sample_downstream"],
        },
    )
    simulation = SimpleNamespace(incident=SimpleNamespace(), branches={})
    for name in (
        "ensure_recording_system",
        "ensure_energy_filter",
        "ensure_corrector_structure",
        "normalise_component_names",
    ):
        monkeypatch.setattr(simulation_pipeline, name, lambda _state: None)
    monkeypatch.setattr(
        simulation_pipeline,
        "apply_physical_layout_to_state",
        lambda _state: None,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "run",
        lambda _state, **_kwargs: simulation,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "sample_illumination_absent",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "retain_specimen_observables",
        lambda current, _keep: current,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "run_specimen_interactions",
        lambda *_args, **_kwargs: interactions,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "simulate_energy_filter",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "calculate_scan_geometry",
        lambda _state: object(),
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "calculate_scan_ray_paths",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "detect_all_lens_crossovers",
        lambda *_args: (),
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "aperture_stop_records",
        lambda _state: (),
    )
    received_exit = []

    def fake_stem(
        _state,
        _simulation,
        *,
        specimen_interactions,
        geometric_specimen_exit,
        geometric_specimen_exit_signature,
        progress_callback,
    ):
        del progress_callback
        assert specimen_interactions is interactions
        assert geometric_specimen_exit_signature == (
            signatures["sample_downstream"]
        )
        received_exit.append(geometric_specimen_exit)
        return "shared-stem-frame"

    monkeypatch.setattr(
        simulation_pipeline,
        "calculate_stem_scan_frame",
        fake_stem,
    )

    result = simulation_pipeline.calculate(state, existing_result=existing)

    assert result.stem_scan == "shared-stem-frame"
    assert len(received_exit) == 1
    assert received_exit[0].branches is downstream_branches
    assert received_exit[0].metrics[
        "tracked_downstream_source_probability"
    ] == pytest.approx(0.75)
    assert "sample_region" in result.reused_products
    assert "sample_downstream" in result.reused_products


def test_pipeline_builds_one_first_class_specimen_exit_for_geometric_stem(
    monkeypatch,
):
    state = default_state()
    state.sample.wave_enabled = False
    state.sample.eds_enabled = False
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = "configured.cif"
    state.sample.stem_wave_enabled = False
    state.ac_deflector.enabled = True
    state.ac_deflector.scan_enabled = True
    state.ac_deflector.scan_pixels_x = 1
    state.ac_deflector.scan_lines = 1
    signatures = simulation_pipeline.calculation_signatures(state)
    elastic = object()
    interactions = SimpleNamespace(
        elastic_transport=elastic,
        eds_spectrum=None,
        inelastic_distribution=object(),
        metrics={},
        completed_observables=frozenset(),
    )
    simulation = SimpleNamespace(incident=SimpleNamespace(), branches={})
    for name in (
        "ensure_recording_system",
        "ensure_energy_filter",
        "ensure_corrector_structure",
        "normalise_component_names",
    ):
        monkeypatch.setattr(simulation_pipeline, name, lambda _state: None)
    monkeypatch.setattr(
        simulation_pipeline,
        "apply_physical_layout_to_state",
        lambda _state: None,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "run",
        lambda _state, **_kwargs: simulation,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "sample_illumination_absent",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "retain_specimen_observables",
        lambda current, _keep: current,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "run_specimen_interactions",
        lambda *_args, **_kwargs: interactions,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "simulate_energy_filter",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "calculate_scan_geometry",
        lambda _state: object(),
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "calculate_scan_ray_paths",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "detect_all_lens_crossovers",
        lambda *_args: (),
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "aperture_stop_records",
        lambda _state: (),
    )
    build_calls = []

    def build_exit(
        passed_state,
        passed_simulation,
        passed_elastic,
        passed_inelastic,
        **kwargs,
    ):
        build_calls.append((
            passed_state,
            passed_simulation,
            passed_elastic,
            passed_inelastic,
            kwargs,
        ))
        return simulation_pipeline.GeometricSpecimenExit(
            (),
            {
                "tracked_downstream_source_probability": 0.0,
                "inelastic_absorbed_source_probability": 0.0,
            },
            dependency_signature=kwargs["dependency_signature"],
        )

    monkeypatch.setattr(
        simulation_pipeline,
        "build_geometric_specimen_exit",
        build_exit,
    )
    stem_checkpoints = []

    def fake_stem(_state, _simulation, **kwargs):
        stem_checkpoints.append(kwargs["geometric_specimen_exit"])
        return "stem-frame"

    monkeypatch.setattr(
        simulation_pipeline,
        "calculate_stem_scan_frame",
        fake_stem,
    )

    result = simulation_pipeline.calculate(state)

    assert len(build_calls) == 1
    assert build_calls[0][4]["dependency_signature"] == (
        signatures["sample_downstream"]
    )
    assert result.specimen_exit is stem_checkpoints[0]
    assert result.specimen_exit.dependency_signature == (
        signatures["sample_downstream"]
    )
    assert "sample_downstream" in result.calculated_products


def test_high_accuracy_pipeline_maps_stem_batches_inside_stage(monkeypatch):
    state = default_state()
    state.sample.wave_enabled = False
    state.sample.eds_enabled = False
    state.ac_deflector.enabled = True
    state.ac_deflector.scan_enabled = True
    state.ac_deflector.scan_pixels_x = 32
    state.ac_deflector.scan_lines = 32
    state.sample.stem_wave_enabled = True
    state.sample.wave_atomistic_enabled = False
    simulation = SimpleNamespace(incident=SimpleNamespace(), branches={})

    monkeypatch.setattr(
        simulation_pipeline, "ensure_recording_system", lambda _state: None
    )
    monkeypatch.setattr(
        simulation_pipeline, "ensure_energy_filter", lambda _state: None
    )
    monkeypatch.setattr(
        simulation_pipeline, "ensure_corrector_structure", lambda _state: None
    )
    monkeypatch.setattr(
        simulation_pipeline, "normalise_component_names", lambda _state: None
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "apply_physical_layout_to_state",
        lambda _state: "layout",
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "run",
        lambda _state, resolved_layout: simulation,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "simulate_energy_filter",
        lambda _state, _simulation: "energy-filter",
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "calculate_scan_geometry",
        lambda _state: "scan-geometry",
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "calculate_scan_ray_paths",
        lambda _state, _simulation: "scan-rays",
    )

    received_specimen_interactions = []

    def fake_stem(
        _state,
        _simulation,
        *,
        specimen_interactions,
        progress_callback,
    ):
        received_specimen_interactions.append(specimen_interactions)
        progress_callback(0, 4, "Preparing STEM")
        progress_callback(2, 4, "STEM probes 16/32")
        progress_callback(4, 4, "STEM detector frame complete")
        return "stem-frame"

    monkeypatch.setattr(
        simulation_pipeline, "calculate_stem_scan_frame", fake_stem
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "detect_all_lens_crossovers",
        lambda _branches, _lenses: (),
    )
    monkeypatch.setattr(
        simulation_pipeline, "aperture_stop_records", lambda _state: ()
    )

    progress = []
    result = simulation_pipeline.calculate(
        state,
        progress_callback=lambda completed, total, stage: progress.append(
            (completed, total, stage)
        ),
    )

    assert result.stem_scan == "stem-frame"
    assert received_specimen_interactions == [result.specimen_interactions]
    nested = [item for item in progress if item[1] == 1_340_000]
    assert nested == [
        (50_000, 1_340_000, "Preparing STEM"),
        (690_000, 1_340_000, "STEM probes 16/32"),
        (1_330_000, 1_340_000, "STEM detector frame complete"),
    ]
    assert nested[0][0] / nested[0][1] < 0.04
    percentages = [completed / total for completed, total, _stage in progress]
    assert percentages == sorted(percentages)


def test_nested_progress_stays_inside_qt_integer_range_for_million_rays():
    total_work = 1_000_017
    subdivisions = simulation_pipeline._progress_stage_subdivisions(
        total_work
    )

    assert subdivisions > 1
    assert total_work * subdivisions <= (
        simulation_pipeline._QT_PROGRESS_SAFE_MAX
    )
    completed, total = simulation_pipeline._bounded_progress_position(
        3_000_000_000,
        4_000_000_000,
    )
    assert total == simulation_pipeline._QT_PROGRESS_SAFE_MAX
    assert completed == 1_500_000_000


def test_controller_forwards_only_current_high_accuracy_progress(monkeypatch):
    controller = CalculationController()
    workers = []
    controller.pool.start = workers.append
    updates = []
    controller.progress_changed.connect(
        lambda *values: updates.append(values)
    )

    def fake_calculate(_state, *, progress_callback):
        progress_callback(0, 2, "Preparing")
        progress_callback(1, 2, "Finalising")
        progress_callback(2, 2, "Complete")
        return object()

    monkeypatch.setattr(
        calculation_controller_module,
        "calculate",
        fake_calculate,
    )
    controller.submit(default_state(), "High accuracy", 25, 5.0)
    workers[0].run()

    assert updates == [
        ("High accuracy", 0, 2, "Preparing"),
        ("High accuracy", 1, 2, "Finalising"),
        ("High accuracy", 2, 2, "Complete"),
    ]


def test_high_accuracy_worker_automatically_loads_and_saves_incident_seed(
    monkeypatch,
):
    loaded_seed = SimpleNamespace(name="persistent-incident-seed")
    completed_simulation = SimpleNamespace(name="completed-simulation")
    calls = {"load": [], "save": [], "existing": []}

    class FakeStore:
        def get_incident_simulation_seed(self, manifest):
            calls["load"].append(manifest)
            return loaded_seed

        def put_incident_simulation_seed(self, manifest, simulation):
            calls["save"].append((manifest, simulation))
            return "stored"

    def fake_calculate(state, *, progress_callback, existing_result=None):
        calls["existing"].append(existing_result)
        return CalculationResult(
            simulation=completed_simulation,
            energy_filter=None,
            signatures=dict(calls["load"][0].calculation_signatures),
        )

    monkeypatch.setattr(
        calculation_controller_module,
        "calculate",
        fake_calculate,
    )
    controller = CalculationController(artifact_store=FakeStore())
    workers = []
    controller.pool.start = workers.append

    controller.submit(default_state(), "High accuracy", 25, 5.0)
    workers[0].run()

    assert calls["load"]
    assert calls["existing"][0].simulation is loaded_seed
    assert calls["save"]
    assert calls["save"][0][1] is completed_simulation


def test_persistent_incident_cache_failure_never_fails_calculation(monkeypatch):
    completed_simulation = SimpleNamespace(name="completed-simulation")
    calls = {"manifest": None, "calculate": 0, "save": 0}

    class FailingStore:
        def get_incident_simulation_seed(self, manifest):
            calls["manifest"] = manifest
            raise OSError("unreadable cache")

        def put_incident_simulation_seed(self, manifest, simulation):
            calls["save"] += 1
            raise OSError("unwritable cache")

    def fake_calculate(state, *, progress_callback, existing_result=None):
        calls["calculate"] += 1
        return CalculationResult(
            simulation=completed_simulation,
            energy_filter=None,
            signatures=dict(calls["manifest"].calculation_signatures),
        )

    monkeypatch.setattr(
        calculation_controller_module,
        "calculate",
        fake_calculate,
    )
    controller = CalculationController(artifact_store=FailingStore())
    workers = []
    failures = []
    results = []
    controller.pool.start = workers.append
    controller.failed.connect(lambda *values: failures.append(values))
    controller.result_ready.connect(lambda *values: results.append(values))

    controller.submit(default_state(), "High accuracy", 25, 5.0)
    workers[0].run()

    assert calls["calculate"] == 1
    assert calls["save"] == 1
    assert not failures
    assert results and results[0][1].simulation is completed_simulation


def test_preview_runs_off_the_gui_thread(qtbot):
    controller = CalculationController()
    state = default_state()
    state.objective_lens.cs_mm = 0.85
    state.objective_lens.polarity = -1

    with qtbot.waitSignal(controller.result_ready, timeout=30_000) as blocker:
        controller.submit(state, "Preview", 25, 3.0)

    quality, result, duration = blocker.args
    assert quality == "Preview"
    assert result.simulation.incident.x.shape[1] == 25
    assert "000" in result.simulation.branches
    assert {
        branch.interaction_kind
        for branch in result.simulation.branches.values()
    } == {"optical_reference"}
    assert result.simulation.metrics["branch_weights_are_absolute"] is True
    assert result.simulation.metrics["optical_tuning"]
    assert result.stem_scan is None
    assert result.lens_crossovers
    assert all(item["verified"] for item in result.lens_crossovers)
    assert result.aperture_stops
    assert result.state_snapshot.objective_lens.cs_mm == 0.85
    assert result.state_snapshot.objective_lens.polarity == -1
    assert all("diameter_mm" in item for item in result.aperture_stops)
    assert duration > 0.0


def test_high_accuracy_defaults_fit_32_gib_budget_and_extreme_request_is_rejected():
    state = default_state()
    default_estimate = estimate_calculation_memory_bytes(
        state, "High accuracy", 15_000, 0.1
    )
    assert default_estimate < HIGH_ACCURACY_MEMORY_BUDGET_BYTES

    controller = CalculationController()
    with pytest.raises(ValueError, match="32 GiB workstation"):
        controller.submit(state, "High accuracy", 1_000_000, 0.01)


def test_step_refinement_memory_tracks_axial_fields_not_z_by_ray_matrices():
    state = default_state()
    state.sample.inserted = False
    state.sample.wave_enabled = False
    coarse = estimate_calculation_memory_bytes(state, "High accuracy", 15_000, 0.1)
    fine = estimate_calculation_memory_bytes(state, "High accuracy", 15_000, 0.01)
    assert coarse < fine
    # Tenfold refinement adds only axial fields; ray histories keep their
    # independent 0.5 mm storage cadence. The former dense coefficient estimate
    # falsely needed tens of GiB and prevented a convergence calculation.
    assert fine - coarse < 512 * 1024**2
    assert fine < HIGH_ACCURACY_MEMORY_BUDGET_BYTES


def test_high_accuracy_memory_guard_includes_tem_wave_grid():
    state = default_state()
    for detector in state.stem_detectors:
        detector.inserted = False
        detector.readout_enabled = False
    state.illumination_mode = "TEM"
    state.sample.wave_enabled = True
    state.sample.wave_multislice_enabled = False
    # Wave storage itself must exceed the budget now that ray integration no
    # longer allocates nine dense axial-by-ray coefficient matrices.
    state.sample.wave_grid_pixels = 16384

    estimate = estimate_calculation_memory_bytes(
        state, "High accuracy", 15_000, 0.1
    )

    assert estimate > HIGH_ACCURACY_MEMORY_BUDGET_BYTES
    controller = CalculationController()
    with pytest.raises(ValueError, match="TEM wave grid"):
        controller.submit(state, "High accuracy", 15_000, 0.1)


def test_wave_imaging_is_disabled_only_for_preview():
    controller = CalculationController()
    captured = []
    controller.pool.start = captured.append
    state = default_state()
    state.illumination_mode = "TEM"
    state.sample.wave_enabled = True

    controller.submit(state, "High accuracy", 25, 5.0)
    controller.submit(state, "Preview", 25, 5.0)

    assert captured[0].state.sample.wave_enabled is True
    assert captured[1].state.sample.wave_enabled is False


def test_real_sample_preview_disables_synthetic_ray_scattering():
    controller = CalculationController()
    captured = []
    controller.pool.start = captured.append
    state = default_state()
    state.ac_deflector.wobble_enabled = False
    state.ac_deflector.scan_enabled = True
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = "real-sample.cif"
    state.sample.diffraction_enabled = True
    state.sample.stem_wave_enabled = True

    controller.submit(state, "Preview", 25, 5.0)

    snapshot = captured[0].state
    assert snapshot.sample.diffraction_enabled is False
    assert snapshot.sample.stem_wave_enabled is False


def test_virtual_sample_preview_defers_interaction_channels_until_high_accuracy():
    controller = CalculationController()
    captured = []
    controller.pool.start = captured.append
    state = default_state()
    state.sample.specimen_mode = "virtual"
    state.sample.diffraction_enabled = True

    controller.submit(state, "Preview", 25, 5.0)

    assert captured[0].state.sample.diffraction_enabled is False
    assert state.sample.diffraction_enabled is True


def test_high_accuracy_preserves_selected_compute_backend():
    controller = CalculationController()
    captured = []
    controller.pool.start = captured.append
    state = default_state()
    state.acceleration_enabled = False
    state.acceleration_backend = "CPU"

    controller.submit(state, "High accuracy", 25, 5.0)

    assert captured[0].state.acceleration_enabled is False
    assert captured[0].state.acceleration_backend == "CPU"
