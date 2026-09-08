from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import temsim.calculation_cache as calculation_cache
from temsim.calculation_cache import (
    calculation_signatures,
    matching_products,
    state_model_signature,
)
from temsim.gui.calculation_controller import (
    CalculationController,
    estimate_result_cache_bytes,
)
from temsim.gui.visualization import VisualizationWorkspace
from temsim.optics.column import default_state
from temsim.physics.scan_geometry import calibrate_scan_system
from temsim.simulation_pipeline import CalculationResult
from temsim.simulation_pipeline import calculate


HIGH_QUALITY = "High accuracy"
RAY_COUNT = 25
STEP_MM = 5.0


def _completed_result_for(worker) -> CalculationResult:
    return CalculationResult(
        simulation=SimpleNamespace(),
        energy_filter=SimpleNamespace(),
        state_snapshot=worker.state,
        model_signature=worker.model_signature,
        signatures=dict(worker.request_signatures),
        calculated_products=frozenset({"column", "elastic", "eds"}),
    )


def _cache_result(key: str, payload, **signatures) -> CalculationResult:
    return CalculationResult(
        simulation=SimpleNamespace(payload=payload),
        energy_filter=None,
        signatures={"request": key, **signatures},
    )


def test_high_accuracy_cache_keeps_multiple_results_and_uses_true_lru():
    controller = CalculationController(
        high_cache_limit=2,
        high_cache_budget_bytes=1024**2,
    )
    first = _cache_result("first", np.zeros(64), column="column-a")
    second = _cache_result("second", np.ones(64), column="column-b")
    third = _cache_result("third", np.full(64, 2.0), column="column-c")

    controller._cache_result(first)
    controller._cache_result(second)
    assert controller._best_seed({"column": "column-a"}) is first
    controller._cache_result(third)

    assert tuple(controller._high_cache) == ("first", "third")


def test_result_cache_budget_counts_shared_numpy_storage_once():
    owner = np.zeros(100_000, dtype=np.float64)
    first = _cache_result("first", owner)
    second = _cache_result("second", owner[1:])

    separate_total = (
        estimate_result_cache_bytes(first)
        + estimate_result_cache_bytes(second)
    )
    shared_total = estimate_result_cache_bytes(first, second)

    assert separate_total - shared_total >= owner.nbytes


def test_checkpoint_candidate_survives_when_no_complete_product_matches():
    controller = CalculationController(persistent_cache_enabled=False)
    checkpoints = SimpleNamespace(z_mm=np.asarray((500.0, 505.0)))
    candidate = _cache_result("previous", np.zeros(1), column="old-column")
    candidate.simulation.incident_checkpoints = checkpoints
    candidate.simulation.incident_plan = object()
    candidate.simulation.gun_trace = object()
    controller._cache_result(candidate)

    assert controller._best_seed({"column": "new-column"}) is candidate
    assert not matching_products(candidate.signatures, {"column": "new-column"})
    candidate.simulation.incident_checkpoints = None
    assert controller._best_seed({"column": "new-column"}) is None


def test_exact_product_seed_outranks_newer_checkpoint_only_candidate():
    controller = CalculationController(persistent_cache_enabled=False)
    exact = _cache_result("exact", np.zeros(1), column="matching-column")
    candidate = _cache_result("previous", np.zeros(1), column="old-column")
    candidate.simulation.incident_checkpoints = SimpleNamespace(z_mm=np.asarray((500.0,)))
    candidate.simulation.incident_plan = object()
    candidate.simulation.gun_trace = object()
    controller._cache_result(exact)
    controller._cache_result(candidate)
    assert controller._best_seed({"column": "matching-column"}) is exact


def test_result_cache_byte_budget_evicts_oldest_until_it_fits():
    first = _cache_result("first", np.zeros(4096, dtype=np.float64))
    second = _cache_result("second", np.ones(4096, dtype=np.float64))
    third = _cache_result("third", np.full(4096, 2.0, dtype=np.float64))
    budget = estimate_result_cache_bytes(second, third)
    controller = CalculationController(
        high_cache_limit=4,
        high_cache_budget_bytes=budget,
    )

    controller._cache_result(first)
    controller._cache_result(second)
    controller._cache_result(third)

    assert tuple(controller._high_cache) == ("second", "third")
    assert controller._high_cache_bytes() <= budget


def test_oversized_result_does_not_erase_smaller_cached_results():
    small = _cache_result("small", np.zeros(64, dtype=np.float64))
    budget = estimate_result_cache_bytes(small) + 256
    oversized = _cache_result(
        "oversized", np.zeros(budget, dtype=np.uint8)
    )
    controller = CalculationController(
        high_cache_limit=4,
        high_cache_budget_bytes=budget,
    )

    controller._cache_result(small)
    controller._cache_result(oversized)

    assert tuple(controller._high_cache) == ("small",)


def test_identical_completed_high_request_hits_cache_without_second_worker(qtbot):
    controller = CalculationController()
    workers = []
    controller.pool.start = workers.append
    state = default_state()

    controller.submit(state, HIGH_QUALITY, RAY_COUNT, STEP_MM)
    assert len(workers) == 1
    first_worker = workers[0]
    completed = _completed_result_for(first_worker)
    controller._accept_result(
        first_worker.generation,
        HIGH_QUALITY,
        completed,
        1.0,
    )
    controller._accept_finished(first_worker.generation, HIGH_QUALITY)

    with qtbot.waitSignal(controller.result_ready, timeout=1_000) as delivered:
        controller.submit(state, HIGH_QUALITY, RAY_COUNT, STEP_MM)

    assert len(workers) == 1
    quality, result, duration = delivered.args
    assert quality == HIGH_QUALITY
    assert result.cache_hit is True
    assert result.simulation is completed.simulation
    assert result.calculated_products == frozenset()
    assert {"column", "elastic", "eds"} <= result.reused_products
    assert duration == 0.0


def test_identical_inflight_high_requests_share_one_worker():
    controller = CalculationController()
    workers = []
    controller.pool.start = workers.append
    state = default_state()

    controller.submit(state, HIGH_QUALITY, RAY_COUNT, STEP_MM)
    generation = controller._generation
    running_key = controller._running_high_key
    controller.submit(state, HIGH_QUALITY, RAY_COUNT, STEP_MM)

    assert len(workers) == 1
    assert controller._generation == generation
    assert controller._running_high_key == running_key
    assert running_key == workers[0].request_signatures["request"]


def test_preview_supersedes_high_without_leaving_a_stale_running_token():
    controller = CalculationController()
    workers = []
    controller.pool.start = workers.append
    state = default_state()

    controller.submit(state, HIGH_QUALITY, RAY_COUNT, STEP_MM)
    old_high = workers[-1]
    controller.submit(state, "Preview", RAY_COUNT, STEP_MM)

    assert controller._running_high_key is None
    controller._accept_finished(old_high.generation, HIGH_QUALITY)
    controller.submit(state, HIGH_QUALITY, RAY_COUNT, STEP_MM)

    assert len(workers) == 3
    assert controller._running_high_key == (
        workers[-1].request_signatures["request"]
    )
    assert controller._running_high_generation == workers[-1].generation


@pytest.mark.parametrize(
    ("field", "replacement", "downstream_changes"),
    (
        ("sample_region_upstream_distance_um", 75.0, False),
        ("sample_region_downstream_distance_um", 80.0, True),
        ("sample_region_photon_path_count", 257, False),
        ("sample_region_secondary_path_count", 97, False),
        ("sample_region_seed", 13, False),
    ),
)
def test_sample_region_controls_invalidate_only_the_region_artifact(
    field,
    replacement,
    downstream_changes,
):
    state = default_state()
    before = calculation_signatures(state)

    setattr(state.sample, field, replacement)
    after = calculation_signatures(state)

    changed = {
        product
        for product in before
        if before[product] != after[product]
    }
    expected = {"request", "sample_region"}
    if downstream_changes:
        expected.add("sample_downstream")
    assert changed == expected
    assert "sample_region" not in matching_products(before, after)


def test_eds_only_parameter_changes_only_request_and_eds_signatures():
    state = default_state()
    before = calculation_signatures(state)

    state.sample.eds_detector_efficiency = 0.625
    after = calculation_signatures(state)

    changed = {
        product
        for product in before
        if before[product] != after[product]
    }
    assert changed == {"request", "eds", "sample_region"}
    for reusable_product in ("column", "elastic", "wave", "stem"):
        assert after[reusable_product] == before[reusable_product]


def test_energy_filter_setting_changes_only_filter_product():
    state = default_state()
    before = calculation_signatures(state)

    state.energy_filter.selected_loss_ev += 12.5
    after = calculation_signatures(state)

    changed = {
        product
        for product in before
        if before[product] != after[product]
    }
    assert changed == {"request", "energy_filter"}


def test_tem_objective_pupil_and_image_aberration_invalidate_wave_source():
    for mutate in (
        lambda state: setattr(
            state.objective_aperture,
            "radius_mm",
            state.objective_aperture.radius_mm * 1.1,
        ),
        lambda state: setattr(
            state,
            "image_aberrations",
            {"a2_mm": 0.002, "a2_azimuth_deg": 17.0},
        ),
        lambda state: setattr(
            state.objective_lens,
            "percent",
            state.objective_lens.percent + 0.25,
        ),
    ):
        state = default_state()
        before = calculation_signatures(state)

        mutate(state)
        after = calculation_signatures(state)

        assert after["wave_source"] != before["wave_source"]
        assert after["wave"] != before["wave"]


def test_elastic_transport_controls_also_invalidate_geometric_stem():
    for name, value in (
        ("eds_elastic_seed", 123456),
        ("eds_elastic_max_events", 17),
        ("eds_support_rotation_deg", 23.0),
    ):
        state = default_state()
        before = calculation_signatures(state)

        setattr(state.sample, name, value)
        after = calculation_signatures(state)

        for product in (
            "elastic",
            "eds",
            "stem",
            "sample_region",
            "sample_downstream",
        ):
            assert after[product] != before[product]
        for product in ("incident", "wave", "wave_source"):
            assert after[product] == before[product]


def test_dynamic_simulation_time_is_part_of_calculation_identity():
    state = default_state()
    before = calculation_signatures(state)

    state.simulation_time_s = 0.125
    after = calculation_signatures(state)

    assert after["request"] != before["request"]
    assert after["column"] != before["column"]
    assert after["incident"] != before["incident"]
    assert after["scan_geometry"] == before["scan_geometry"]
    assert after["scan_ray_paths"] != before["scan_ray_paths"]


def test_scan_response_products_ignore_emitter_sampling_density():
    state = default_state()
    before = calculation_signatures(state)

    state.electron_gun.emitter.ray_count += 17
    after = calculation_signatures(state)

    assert after["request"] != before["request"]
    assert after["column"] != before["column"]
    assert after["scan_geometry"] == before["scan_geometry"]
    assert after["scan_ray_paths"] == before["scan_ray_paths"]


def test_retired_virtual_density_map_does_not_change_physical_identity(tmp_path):
    density_path = tmp_path / "density.npy"
    np.save(density_path, np.zeros((4, 4), dtype=np.float32))
    state = default_state()
    state.sample.specimen_mode = "reference"
    state.sample.virtual_regions = [{
        "kind": "map",
        "map_path": str(density_path),
        "centre_x_nm": 0.0,
        "centre_y_nm": 0.0,
        "size_x_nm": 100.0,
        "size_y_nm": 100.0,
        "density": 1.0,
    }]
    before = calculation_signatures(state)

    np.save(density_path, np.ones((4, 4), dtype=np.float32))
    after = calculation_signatures(state)

    assert after == before


def test_spot_current_limit_reuses_geometry_but_invalidates_counts():
    state = default_state()
    before = calculation_signatures(state)

    state.column_current_limit_percent = 40.0
    after = calculation_signatures(state)

    changed = {
        product
        for product in before
        if before[product] != after[product]
    }
    assert changed == {
        "request",
        "eds",
        "energy_filter",
        "fourdstem_virtual_detectors",
        "sample_region",
        "stem",
    }
    for reusable_product in (
        "column",
        "elastic",
        "wave",
        "scan_geometry",
        "scan_ray_paths",
        "sample_downstream",
    ):
        assert after[reusable_product] == before[reusable_product]


def test_post_sample_lens_change_keeps_specimen_checkpoint_products():
    for lens_attribute in (
        "diffraction_lens",
        "intermediate_lens",
        "projector_lens_p1",
        "projector_lens_p2",
    ):
        state = default_state()
        before = calculation_signatures(state)

        getattr(state, lens_attribute).percent += 1.0
        after = calculation_signatures(state)

        for reusable_product in (
            "incident",
            "wave_source",
            "elastic",
            "eds",
            "sample_region",
        ):
            assert after[reusable_product] == before[reusable_product]
        for invalidated_product in (
            "column",
            "wave",
            "energy_filter",
            "sample_downstream",
            "scan_geometry",
            "scan_ray_paths",
            "stem",
        ):
            assert after[invalidated_product] != before[invalidated_product]


def _replace_assembly_part_geometry(state, key, field, value):
    parts = tuple(
        replace(
            part,
            data={**dict(part.data), field: value},
        )
        if part.key == key
        else part
        for part in state._resolved_assembly.parts
    )
    state._resolved_assembly = replace(
        state._resolved_assembly,
        parts=parts,
    )


def test_post_sample_projector_geometry_keeps_local_stage_products():
    state = default_state()
    before = calculation_signatures(state)
    pole = state._resolved_assembly.part("projector_lens_1_upper_pole")
    previous = float(pole.data.get("pole_piece_bore_diameter_mm", 20.0))

    state.projector_lens_p1.z_mm += 0.75
    _replace_assembly_part_geometry(
        state,
        "projector_lens_1_upper_pole",
        "pole_piece_bore_diameter_mm",
        previous + 0.5,
    )
    after = calculation_signatures(state)

    for reusable_product in (
        "incident",
        "wave_source",
        "elastic",
        "eds",
        "sample_region",
    ):
        assert after[reusable_product] == before[reusable_product]
    assert after["request"] != before["request"]
    assert after["column"] != before["column"]
    assert after["sample_downstream"] != before["sample_downstream"]


def test_projector_field_map_descriptor_does_not_invalidate_upstream_products(
    tmp_path,
):
    first_map = tmp_path / "projector-first.npz"
    second_map = tmp_path / "projector-second.npz"
    first_map.write_bytes(b"first projector map")
    second_map.write_bytes(b"second projector map")
    state = default_state()
    state.lens_field_map_descriptors = {
        "projector_lens_1": {
            "source_path": str(first_map),
            "source_sha256": "first-descriptor",
            "geometry_fingerprint": "projector-geometry-a",
            "reference_excitation_percent": 100.0,
        }
    }
    before = calculation_signatures(state)

    state.lens_field_map_descriptors["projector_lens_1"] = {
        "source_path": str(second_map),
        "source_sha256": "second-descriptor",
        "geometry_fingerprint": "projector-geometry-b",
        "reference_excitation_percent": 80.0,
    }
    after = calculation_signatures(state)

    for product in (
        "incident",
        "wave_source",
        "elastic",
        "eds",
        "sample_region",
        "fourdstem_cube",
        "fourdstem_virtual_detectors",
    ):
        assert after[product] == before[product]
    assert after["request"] != before["request"]
    assert after["column"] != before["column"]


def test_upstream_pole_geometry_invalidates_local_stage_products():
    state = default_state()
    before = calculation_signatures(state)
    pole = state._resolved_assembly.part("condenser_lens_1_lower_pole")
    previous = float(pole.data.get("pole_piece_bore_diameter_mm", 5.76))

    _replace_assembly_part_geometry(
        state,
        "condenser_lens_1_lower_pole",
        "pole_piece_bore_diameter_mm",
        previous + 0.25,
    )
    after = calculation_signatures(state)

    for invalidated_product in (
        "incident",
        "wave_source",
        "elastic",
        "eds",
        "sample_region",
    ):
        assert after[invalidated_product] != before[invalidated_product]


def test_projector_mode_change_keeps_specimen_checkpoint_products():
    state = default_state()
    before = calculation_signatures(state)

    state.projector_mode = "image"
    after = calculation_signatures(state)

    for reusable_product in (
        "incident",
        "wave_source",
        "elastic",
        "eds",
        "sample_region",
    ):
        assert after[reusable_product] == before[reusable_product]
    for invalidated_product in (
        "column",
        "wave",
        "energy_filter",
        "sample_downstream",
        "scan_geometry",
        "scan_ray_paths",
        "stem",
    ):
        assert after[invalidated_product] != before[invalidated_product]


def test_tem_recording_plane_switch_keeps_specimen_checkpoint_products():
    state = default_state()
    before = calculation_signatures(state)

    state.fluorescent_screen.inserted = False
    after = calculation_signatures(state)

    for reusable_product in (
        "incident",
        "wave_source",
        "elastic",
        "eds",
        "sample_region",
    ):
        assert after[reusable_product] == before[reusable_product]
    for invalidated_product in (
        "column",
        "wave",
        "energy_filter",
        "sample_downstream",
        "scan_geometry",
        "scan_ray_paths",
        "stem",
    ):
        assert after[invalidated_product] != before[invalidated_product]


def test_projector_change_reprojects_saved_objective_wave_without_specimen_recalc():
    state = default_state()
    for detector in state.stem_detectors:
        detector.inserted = False
        detector.readout_enabled = False
    state.step_mm = 5.0
    state.history_step_mm = 5.0
    state.acceleration_enabled = False
    state.illumination_mode = "TEM"
    state.projector_mode = "image"
    state.sample.eds_enabled = False
    state.sample.wave_enabled = True
    state.sample.specimen_mode = "reference"
    state.sample.specimen_preset_key = "si_110"
    state.sample.thickness_nm = 2.0
    state.sample.wave_grid_pixels = 32
    state.sample.wave_field_of_view_angstrom = 16.0
    state.sample.wave_multislice_enabled = True
    state.sample.wave_atomistic_enabled = True
    state.fluorescent_screen.inserted = False
    state.camera.inserted = True
    emitter = getattr(state.electron_gun, "emitter", None)
    if emitter is None:
        state.electron_gun.ray_count = 9
    else:
        emitter.ray_count = 9

    previous = calculate(state)
    state.intermediate_lens.percent += 2.0
    updated = calculate(state, existing_result=previous)

    assert updated.wave_imaging is not previous.wave_imaging
    assert updated.wave_imaging.exit_wave is previous.wave_imaging.exit_wave
    assert updated.specimen_interactions.wave_imaging is updated.wave_imaging
    assert "wave_projection" in updated.calculated_products
    assert "wave" not in updated.calculated_products
    assert {"incident", "wave_source"} <= updated.reused_products
    assert updated.wave_imaging.metrics["projector_checkpoint_reused"] is True


def test_seed_selection_prefers_reusable_incident_and_wave_source():
    controller = CalculationController()
    projection_seed = _cache_result(
        "projection-seed",
        np.zeros(8),
        incident="incident-a",
        wave_source="wave-source-a",
        elastic="elastic-a",
    )
    downstream_seed = _cache_result(
        "downstream-seed",
        np.ones(8),
        column="column-a",
        wave="wave-a",
        energy_filter="filter-a",
    )
    projection_seed.simulation.incident = object()
    projection_seed.specimen_interactions = SimpleNamespace(
        elastic_transport=object(),
    )
    projection_seed.wave_imaging = SimpleNamespace(
        projector_checkpoint=object(),
    )
    downstream_seed.wave_imaging = object()
    downstream_seed.energy_filter = object()
    controller._cache_result(projection_seed)
    controller._cache_result(downstream_seed)

    selected = controller._best_seed({
        "incident": "incident-a",
        "wave_source": "wave-source-a",
        "elastic": "elastic-a",
        "column": "column-a",
        "wave": "wave-a",
        "energy_filter": "filter-a",
    })

    assert selected is projection_seed


@pytest.mark.parametrize("descan_enabled", (False, True))
def test_scan_calibration_does_not_change_cache_identity(descan_enabled):
    state = default_state()
    state.ac_deflector.enabled = True
    state.ac_deflector.scan_enabled = True
    state.ac_deflector.wobble_enabled = False
    state.descan_deflector.enabled = descan_enabled
    state.descan_deflector.scan_enabled = descan_enabled
    if descan_enabled:
        state.descan_deflector.scan_pixels_x = 7
        state.descan_deflector.scan_lines = 9
        state.descan_deflector.scan_pixel_size_nm = 2.5
        state.descan_deflector.scan_frame_period_s = 0.75
    before = calculation_signatures(state)
    lower_before = (
        state.ac_deflector.lower_coil_gain,
        state.descan_deflector.lower_coil_gain,
    )

    calibrate_scan_system(state)

    lower_after = (
        state.ac_deflector.lower_coil_gain,
        state.descan_deflector.lower_coil_gain,
    )
    assert lower_after != lower_before
    assert calculation_signatures(state) == before
    if descan_enabled:
        assert state.descan_deflector.scan_pixels_x == (
            state.ac_deflector.scan_pixels_x
        )
        assert state.descan_deflector.scan_lines == state.ac_deflector.scan_lines


def test_legacy_ac_scan_amplitudes_do_not_invalidate_calibrated_scan():
    baseline = default_state()
    baseline.ac_deflector.enabled = True
    baseline.ac_deflector.scan_enabled = True
    baseline.ac_deflector.wobble_enabled = False
    changed = default_state()
    changed.ac_deflector.enabled = True
    changed.ac_deflector.scan_enabled = True
    changed.ac_deflector.wobble_enabled = False
    changed.ac_deflector.scan_amplitude_x_mrad = 7.5
    changed.ac_deflector.scan_amplitude_y_mrad = -3.25

    assert calculation_signatures(changed) == calculation_signatures(baseline)

    baseline_command, _, _ = calibrate_scan_system(baseline)
    changed_command, _, _ = calibrate_scan_system(changed)
    assert changed_command == pytest.approx(baseline_command)


def test_cif_content_change_at_same_path_changes_calculation_identity(tmp_path):
    cif_path = tmp_path / "specimen.cif"
    cif_path.write_text("data_first\n_cell_length_a 5.0\n", encoding="utf-8")
    state = default_state()
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = str(cif_path)

    before = calculation_signatures(state)
    before_model = state_model_signature(state)
    cif_path.write_text("data_second\n_cell_length_a 6.0\n", encoding="utf-8")
    after = calculation_signatures(state)
    after_model = state_model_signature(state)

    assert after["request"] != before["request"]
    assert after_model != before_model
    for product in ("incident", "column"):
        assert after[product] == before[product]
    for product in (
        "elastic",
        "eds",
        "wave",
        "wave_source",
        "stem",
        "stem_transport",
        "sample_region",
    ):
        assert after[product] != before[product]


def test_loaded_specimen_data_only_invalidates_post_column_products(
    monkeypatch,
):
    state = default_state()
    monkeypatch.setattr(
        calculation_cache,
        "_loaded_solver_input_identities",
        lambda _state: {
            "specimen_preset": "preset-a",
            "support_catalog": "support-a",
            "bote_salvat": "bote-a",
        },
    )
    before = calculation_signatures(state)
    monkeypatch.setattr(
        calculation_cache,
        "_loaded_solver_input_identities",
        lambda _state: {
            "specimen_preset": "preset-b",
            "support_catalog": "support-b",
            "bote_salvat": "bote-b",
        },
    )
    after = calculation_signatures(state)

    for product in ("incident", "column"):
        assert after[product] == before[product]
    for product in (
        "elastic",
        "eds",
        "wave",
        "wave_source",
        "stem",
        "stem_transport",
        "sample_region",
    ):
        assert after[product] != before[product]


def test_same_state_preview_does_not_replace_high_accuracy_tab_products(
    qtbot, monkeypatch
):
    workspace = VisualizationWorkspace()
    qtbot.addWidget(workspace)
    calls = {
        "energy_filter": 0,
        "scan": 0,
        "eds": 0,
        "sample_3d": 0,
        "wave": 0,
    }

    monkeypatch.setattr(workspace, "_prepare_scan_ray_playback", lambda _r: None)
    monkeypatch.setattr(
        workspace,
        "_draw_ray_diagram",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        workspace, "_update_sample_region_control_availability", lambda: None
    )
    for view in (
        workspace.physical_layout,
        workspace.magnetic_field,
        workspace.probe_aberrations,
        workspace.image_aberrations,
        workspace.optical_transfer,
        workspace.transverse_beam,
        workspace.sample_page,
    ):
        monkeypatch.setattr(view, "display_result", lambda *_args: None)

    def record(name):
        return lambda *_args, **_kwargs: calls.__setitem__(
            name, calls[name] + 1
        )

    monkeypatch.setattr(
        workspace.energy_filter, "display_result", record("energy_filter")
    )
    monkeypatch.setattr(workspace.scan_control, "display_result", record("scan"))
    monkeypatch.setattr(workspace.eds_page, "display_result", record("eds"))
    monkeypatch.setattr(
        workspace.sample_interactions_3d,
        "display_result",
        record("sample_3d"),
    )
    monkeypatch.setattr(workspace.wave_imaging, "display_result", record("wave"))

    high = SimpleNamespace(
        model_signature="same-state",
        simulation=object(),
        scan_geometry=object(),
        stem_scan=object(),
        wave_imaging=object(),
        state_snapshot=object(),
        sample_region=None,
    )
    preview = SimpleNamespace(
        model_signature="same-state",
        simulation=object(),
        scan_geometry=object(),
        stem_scan=object(),
        wave_imaging=None,
        state_snapshot=object(),
    )

    workspace.display_result(high, HIGH_QUALITY)
    workspace.display_result(preview, "Preview")

    assert calls == {
        "energy_filter": 1,
        "scan": 1,
        "eds": 1,
        "sample_3d": 1,
        "wave": 1,
    }
    assert workspace._last_result is preview
    assert workspace._high_accuracy_result is high
    assert workspace._high_accuracy_current is True
