"""External-file edits must not publish a mixed-input complete result."""
from threading import Event

import pytest

import temsim.calculation_cache as cache
import temsim.gui.calculation_controller as controllers
from temsim.calculation_manifest import (
    assert_external_input_inventory_unchanged,
    capture_external_input_identities,
)
from temsim.optics.column import default_state
from temsim.simulation_modes import switch_mode
from temsim.simulation_pipeline import CalculationResult, calculate
from temsim.specimen import reference_catalog


@pytest.fixture
def reference_case(tmp_path, monkeypatch):
    state = default_state()
    original = reference_catalog.get_reference_sample("si_110").cif_path.read_text()
    cif = tmp_path / "si_110.cif"
    cif.write_text(original)
    monkeypatch.setattr(reference_catalog, "REFERENCE_DIRECTORY", tmp_path)
    state.sample.specimen_mode = "reference"
    state.sample.reference_sample_key = "si_110"

    def prepare(kind):
        metadata = cif.with_suffix(".toml")
        if kind == "existing_sidecar":
            metadata.write_text('thermal_sigma_angstrom = 0.085\nthermal_source = "test reference"\n')

        def mutate():
            if kind == "cif":
                cif.write_text(original.replace("5.44370237", "6.44370237"))
            elif kind == "sidecar_type_error":
                metadata.write_text('thermal_sigma_angstrom = []\n')
            elif kind == "sidecar_overflow":
                metadata.write_text('thermal_sigma_angstrom = ' + '9' * 400 + '\n')
            else:
                metadata.write_text('thermal_sigma_angstrom = 0.095\nthermal_source = "revised reference"\n')
        return state, mutate

    return prepare


@pytest.mark.parametrize("kind", ["cif", "existing_sidecar", "new_sidecar"])
def test_input_guard_checks_changes_and_new_reference_metadata(reference_case, kind):
    state, mutate = reference_case(kind)
    inputs = capture_external_input_identities(state)
    assert_external_input_inventory_unchanged(state, inputs)
    mutate()
    with pytest.raises(RuntimeError, match="Captured external inputs changed: specimen:"):
        assert_external_input_inventory_unchanged(state, inputs)


@pytest.mark.parametrize("kind,error_type", [
    ("sidecar_type_error", TypeError), ("sidecar_overflow", OverflowError),
])
def test_new_sidecar_conversion_errors_are_input_change_failures(reference_case, kind, error_type):
    state, mutate = reference_case(kind)
    inputs = capture_external_input_identities(state)
    mutate()
    with pytest.raises(RuntimeError, match="Captured external inputs changed") as failure:
        assert_external_input_inventory_unchanged(state, inputs)
    assert isinstance(failure.value.__cause__, error_type)


def test_real_pipeline_rejects_cif_change_after_signatures_are_captured(reference_case):
    state, mutate = reference_case("cif")
    switch_mode(state, "ideal")
    state.sample.wave_enabled = state.sample.stem_wave_enabled = False
    state.sample.eds_enabled = state.energy_filter.enabled = False
    state.ac_deflector.enabled = state.ac_deflector.scan_enabled = False
    state.descan_deflector.enabled = state.descan_deflector.scan_enabled = False
    state.electron_gun.emitter.ray_count = 9
    state.step_mm = state.history_step_mm = 5.0
    changed = []

    def progress(completed, total, stage):
        if not changed:
            mutate()
            changed.append(stage)

    # Real nine-ray column and specimen orchestration, with no wave/EDS scan.
    with pytest.raises(RuntimeError, match="Captured external inputs changed: specimen:cif"):
        calculate(state, progress_callback=progress)
    assert changed


@pytest.mark.parametrize("kind", ["cif", "existing_sidecar", "new_sidecar"])
@pytest.mark.parametrize("when", ["before_run", "during_run"])
def test_worker_does_not_publish_or_persist_changed_inputs(qapp, monkeypatch, reference_case, kind, when):
    state, mutate = reference_case(kind)
    signatures = cache.calculation_signatures(state)
    worker = controllers.CalculationWorker(1, "High accuracy", state,
                                           request_signatures=signatures)
    events, persisted, calls = [], [], []
    worker.signals.result.connect(lambda *_: events.append("result"))
    worker.signals.error.connect(lambda _g, _q, message: events.append(message))
    worker.signals.finished.connect(lambda *_: events.append("finished"))
    monkeypatch.setattr(worker, "_persist_incident_seed", lambda *_: persisted.append(True))

    def finish(*_args, **_kwargs):
        calls.append(True)
        mutate()
        return CalculationResult(None, None, state_snapshot=state, signatures=signatures)

    monkeypatch.setattr(controllers, "calculate", finish)
    if when == "before_run":
        mutate()
    worker.run()
    assert len(events) == 2
    assert "Captured external inputs changed" in events[0]
    assert events[1] == "finished"
    assert persisted == []
    assert calls == ([] if when == "before_run" else [True])


@pytest.mark.parametrize("when", ["before_run", "during_run"])
def test_cancelled_worker_stays_silent_even_if_inputs_changed(qapp, monkeypatch, reference_case, when):
    state, mutate = reference_case("cif")
    worker = controllers.CalculationWorker(1, "High accuracy", state)
    worker.cancel_event = Event()
    events, calls = [], []
    worker.signals.result.connect(lambda *_: events.append("result"))
    worker.signals.error.connect(lambda *_: events.append("error"))
    worker.signals.finished.connect(lambda *_: events.append("finished"))

    def finish(*_args, **_kwargs):
        calls.append(True)
        mutate()
        worker.cancel_event.set()
        return CalculationResult(None, None, state_snapshot=state)

    monkeypatch.setattr(controllers, "calculate", finish)
    if when == "before_run":
        mutate()
        worker.cancel_event.set()
    worker.run()
    assert calls == ([] if when == "before_run" else [True])
    assert events == ["finished"]


def test_result_changed_while_queued_is_not_cached_or_published(qapp, reference_case):
    state, mutate = reference_case("new_sidecar")
    controller = controllers.CalculationController(persistent_cache_enabled=False)
    previous = CalculationResult(None, None, signatures={"request": "previous"})
    controller._cache_result(previous)
    result = CalculationResult(None, None, state_snapshot=state,
        signatures=cache.calculation_signatures(state),
        external_inputs=capture_external_input_identities(state))
    delivered, errors = [], []
    controller.result_ready.connect(lambda *_: delivered.append(True))
    controller.failed.connect(lambda _q, error: errors.append(error))
    mutate()
    controller._accept_result(controller.generation, "High accuracy", result, 0.)
    assert delivered == []
    assert len(errors) == 1 and "Captured external inputs changed" in errors[0]
    assert controller.completed_high_accuracy_results() == (previous,)


@pytest.mark.parametrize("quality", ["High accuracy", "Preview"])
@pytest.mark.parametrize("kind", ["cif", "new_sidecar", "sidecar_type_error", "sidecar_overflow"])
def test_complete_cache_hit_is_checked_at_queued_delivery(qtbot, monkeypatch, reference_case, quality, kind):
    state, mutate = reference_case(kind)
    controller = controllers.CalculationController(persistent_cache_enabled=False)
    workers, delivered, errors, finished = [], [], [], []
    monkeypatch.setattr(controller.pool, "start", workers.append)
    controller.result_ready.connect(lambda _q, result, _s: delivered.append(result))
    controller.failed.connect(lambda _q, error: errors.append(error))
    controller.finished.connect(lambda *_: finished.append(True))
    controller.submit(state, quality, 9, 5.)
    worker = workers.pop()
    result = CalculationResult(None, None, state_snapshot=worker.state,
        model_signature=worker.model_signature, signatures=worker.request_signatures,
        external_inputs=worker.external_inputs)
    controller._accept_result(worker.generation, quality, result, 0.)
    controller._accept_finished(worker.generation, quality)
    delivered.clear()
    finished.clear()
    # The cache lookup is valid; the file changes before its queued callback.
    controller.started.connect(lambda *_: mutate())
    controller.submit(state, quality, 9, 5.)
    qtbot.waitUntil(lambda: bool(finished))
    assert workers == []
    assert delivered == []
    assert finished == [True]
    assert len(errors) == 1 and "Captured external inputs changed" in errors[0]
    assert (controller._high_cache[result.signatures["request"]] if quality == "High accuracy"
            else controller._tuning_cache[(quality, result.signatures["request"])]) is result
    assert controller._request_input_guard is None


def test_legacy_payload_migration_precedes_controller_input_capture(qapp, monkeypatch, reference_case):
    state, _mutate = reference_case("cif")
    payload = state.to_dict()
    payload["schema_version"] = 76
    payload["sample"]["specimen_mode"] = "virtual"
    payload["sample"]["specimen_preset_key"] = "si_110"
    restored = type(state).from_dict(payload)
    assert restored.sample.specimen_mode == "reference"
    controller = controllers.CalculationController(persistent_cache_enabled=False)
    workers = []
    monkeypatch.setattr(controller.pool, "start", workers.append)
    controller.submit(restored, "Preview", 9, 5.)
    assert workers[0].state.sample.specimen_mode == "reference"
    assert workers[0].external_inputs


def test_new_wave_and_stem_versions_keep_particle_and_eds_checkpoints(monkeypatch):
    state = default_state()
    current = cache.calculation_signatures(state)
    monkeypatch.setattr(cache, "_WAVE_SPECIMEN_SCHEMA", "cif-reference-occupancy-v2")
    monkeypatch.setattr(cache, "_STEM_RECORDING_SCHEMA", "physical-envelope-overlap-raster-independent-tail-v4")
    previous = cache.calculation_signatures(state)
    assert {key for key in current if current[key] != previous[key]} == {
        "request", "wave", "wave_source", "fourdstem_cube",
        "fourdstem_virtual_detectors", "fourdstem_physical_recording",
        "stem", "stem_transport",
    }
