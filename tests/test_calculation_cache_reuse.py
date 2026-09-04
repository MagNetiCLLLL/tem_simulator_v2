from types import SimpleNamespace

import numpy as np

from temsim.calculation_cache import (
    calculation_signatures,
    state_model_signature,
)
from temsim.gui.calculation_controller import (
    CalculationController,
    estimate_result_cache_bytes,
)
from temsim.gui.visualization import VisualizationWorkspace
from temsim.optics.column import default_state
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
    assert changed == {"request", "eds", "stem"}
    for reusable_product in (
        "column", "elastic", "wave", "energy_filter", "scan",
        "sample_region",
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
        for invalidated_product in ("column", "wave", "energy_filter"):
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
    for invalidated_product in ("column", "wave", "energy_filter"):
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
    for invalidated_product in ("column", "wave"):
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
    state.sample.specimen_mode = "virtual"
    state.sample.specimen_preset_key = "si_110"
    state.sample.thickness_nm = 2.0
    state.sample.wave_grid_pixels = 32
    state.sample.wave_field_of_view_angstrom = 16.0
    state.sample.wave_multislice_enabled = False
    state.sample.wave_atomistic_enabled = False
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
        workspace, "_ray_geometry_signature", lambda _result: "geometry"
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
