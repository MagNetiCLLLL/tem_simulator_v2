"""HANDOFF v2 first slice; synthetic/software checks, not field validation."""
import json
import subprocess
import sys
from threading import Event

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.artifact_store import ArtifactBundle, ArtifactStore
from temsim.calculation_manifest import capture_calculation_manifest
from temsim.checkpoint_observables import incident_checkpoint_observables
from temsim.immutable_json import json_digest, thaw_json
from temsim.instrument_snapshot import (
    InstrumentSnapshot, capture_instrument_snapshot, decode_instrument, encode_instrument,
)
from temsim.optics.column import default_state
from temsim.physics.core import PropagationCheckpoints
from temsim.physics.illumination import default_illumination_config, illumination_config


@pytest.fixture
def instrument():
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    return state


def test_complete_snapshot_roundtrip_does_not_use_profiles_defaults_or_setters(instrument, monkeypatch):
    state = instrument
    state.lenses[0].percent = 51.234567891234
    state.projector_lens_p1.z_mm += .125
    state.projector_lens_p1.b0_t *= .987
    state.apertures[0].enabled = False
    state.sample.z_mm += .002
    before = json_digest(encode_instrument(state))
    snapshot = capture_instrument_snapshot(state)
    assert json_digest(encode_instrument(state)) == before

    def forbidden(*args, **kwargs):
        raise AssertionError("A working point must not use defaults or profile migration")
    monkeypatch.setattr(type(state), "from_dict", forbidden)
    monkeypatch.setattr(type(state), "to_dict", forbidden)
    restored = InstrumentSnapshot.from_dict(json.loads(json.dumps(snapshot.to_dict()))).restore()
    assert json_digest(encode_instrument(restored)) == before
    assert restored.objective_lens is restored._objective_lens
    assert restored.camera is next(p for p in restored.recording_planes if p.key == "camera")
    assert restored.lenses[0] is not state.lenses[0]
    restored.lenses[0].percent = 10
    assert state.lenses[0].percent == 51.234567891234
    assert snapshot.digest == capture_instrument_snapshot(state).digest


def test_snapshot_data_and_exports_cannot_mutate_saved_controls(instrument):
    snapshot = capture_instrument_snapshot(instrument)
    original = snapshot.digest
    with pytest.raises(TypeError):
        snapshot.graph["nodes"][0]["attributes"]["lenses"] = []
    exported = snapshot.to_dict()
    exported["graph"]["nodes"].clear()
    assert snapshot.digest == original


def test_restored_immutable_model_arrays_remain_immutable(instrument):
    instrument.lenses[0].calibration_samples = np.arange(3, dtype=float)
    instrument.lenses[0].calibration_samples.setflags(write=False)
    restored = capture_instrument_snapshot(instrument).restore()
    with pytest.raises(ValueError):
        restored.lenses[0].calibration_samples.setflags(write=True)


def test_runtime_editor_cannot_publish_an_independent_source(instrument):
    from temsim.runtime_parameters import RuntimeTarget, validate_runtime_assignment
    target = RuntimeTarget("sample", "Sample", instrument.sample)
    with pytest.raises(ValueError, match="historical only"):
        validate_runtime_assignment(target, "wave_illumination", default_illumination_config())
    assert instrument.sample.wave_illumination["model"] == "ray_conditioned_reduced_order"


def test_unsupported_structured_arrays_fail_instead_of_losing_field_types(instrument):
    instrument.lenses[0].calibration_samples = np.array([(1., 2)], dtype=[("field", "f8"), ("id", "i4")])
    with pytest.raises(TypeError, match="unstructured dtype"):
        encode_instrument(instrument)


def test_exact_snapshot_restores_in_another_python_process(instrument):
    instrument.lenses[0].percent = 62.123456789
    snapshot = capture_instrument_snapshot(instrument)
    script = (
        "import sys,json; from temsim.instrument_snapshot import InstrumentSnapshot,encode_instrument; "
        "from temsim.immutable_json import json_digest; "
        "p=InstrumentSnapshot.from_dict(json.load(sys.stdin)); "
        "s=p.restore(); assert s.lenses[0].percent==62.123456789; "
        "print(json_digest(encode_instrument(s)))"
    )
    result = subprocess.run([sys.executable, "-c", script], input=json.dumps(snapshot.to_dict()),
                            text=True, capture_output=True, timeout=45, check=True)
    assert result.stdout.strip() == json_digest(snapshot.graph)


def test_high_accuracy_request_keeps_captured_geometry_and_all_lens_values(instrument):
    from temsim.gui.calculation_request import CapturedCalculationRequest
    from temsim.gui.calculation_controller import CalculationController
    instrument.objective_lens.upper_b0_t += .125
    instrument.projector_lens_p1.z_mm += .01
    captured = CapturedCalculationRequest.capture(instrument, "High accuracy", 9, 2.)
    expected_field = instrument.objective_lens.upper_b0_t
    expected_z = instrument.projector_lens_p1.z_mm
    instrument.objective_lens.upper_b0_t += 10
    prepared = captured.prepare(Event()).snapshot
    assert prepared.objective_lens.upper_b0_t == expected_field
    assert prepared.projector_lens_p1.z_mm == expected_z
    assert prepared.electron_gun.emitter.ray_count == 9
    assert prepared.step_mm == 2.
    sync = CalculationController._calculation_snapshot(prepared, "High accuracy", 9, 2.)
    assert sync.objective_lens.upper_b0_t == expected_field
    assert sync.projector_lens_p1.z_mm == expected_z


def test_snapshot_preserves_new_public_model_inputs_and_audits_schema(instrument):
    instrument.lenses[0].new_calibrated_parameter = {"coefficient": [1., 2.]}
    snapshot = capture_instrument_snapshot(instrument)
    assert snapshot.restore().lenses[0].new_calibrated_parameter == {"coefficient": [1., 2.]}
    graph = thaw_json(snapshot.graph)
    graph["nodes"][0]["fields"].append("unreviewed_field")
    with pytest.raises(ValueError, match="schema changed"):
        decode_instrument(graph)


def test_changed_or_missing_external_bytes_block_restore_but_keep_view(instrument, tmp_path):
    path = tmp_path / "field.dat"
    path.write_bytes(b"original field evidence")
    instrument.lens_field_map_descriptors = {"objective_lens": {"source_path": str(path)}}
    snapshot = capture_instrument_snapshot(instrument)
    record = next(r for r in snapshot.external_inputs if r["role"] == "lens_field_map:objective_lens")
    assert bytes.fromhex(record["content_hex"]) == b"original field evidence"
    path.write_bytes(b"changed field evidence")
    with pytest.raises(ValueError, match="Changed"):
        snapshot.restore()
    path.unlink()
    with pytest.raises(ValueError, match="Missing"):
        snapshot.restore()
    assert snapshot.to_dict()["digest"] == snapshot.digest


def test_manifest_capture_is_read_only_and_retains_complete_working_point(instrument):
    before = json_digest(encode_instrument(instrument))
    first = capture_calculation_manifest(instrument, ray_count=9, step_mm=2.)
    second = capture_calculation_manifest(instrument, ray_count=9, step_mm=2.)
    assert first.digest == second.digest
    assert json_digest(encode_instrument(instrument)) == before
    restored = first.instrument_snapshot.restore()
    assert restored.electron_gun.emitter.ray_count == 9
    assert restored.step_mm == 2.
    assert [(lens.key, lens.percent) for lens in restored.lenses] == [
        (lens.key, lens.percent) for lens in instrument.lenses]
    instrument.projector_lens_p1.b0_t = np.nextafter(instrument.projector_lens_p1.b0_t, np.inf)
    assert capture_instrument_snapshot(instrument).digest != first.instrument_snapshot.digest


def test_failed_manifest_capture_does_not_cancel_existing_work_or_trim_cache(instrument, qtbot, monkeypatch):
    from temsim.gui import calculation_controller as module
    controller = module.CalculationController(persistent_cache_enabled=False)
    controller._running_high_key = "previous request"
    controller._running_high_generation = controller.generation
    previous_event = controller._cancel_event
    previous_generation = controller.generation
    before = json_digest(encode_instrument(instrument))

    def unavailable(*args, **kwargs):
        raise ValueError("Controlled snapshot failure")
    monkeypatch.setattr(module, "capture_calculation_manifest", unavailable)
    monkeypatch.setattr(controller, "_make_room_for_calculation",
                        lambda *_: pytest.fail("A rejected snapshot must not evict completed work"))
    with pytest.raises(ValueError, match="complete working point"):
        controller.submit(instrument, "High accuracy", 9, 2.)
    assert controller.generation == previous_generation
    assert controller._running_high_key == "previous request"
    assert not previous_event.is_set()
    assert json_digest(encode_instrument(instrument)) == before


def checkpoints():
    angles = np.linspace(-.02, .02, 101)[None, :]
    zeros = np.zeros_like(angles)
    return PropagationCheckpoints(np.array([123.]), zeros, angles, zeros, zeros)


def test_checkpoint_and_artifact_buffers_cannot_be_unfrozen_or_aliased():
    original = np.zeros((1, 3))
    cp = PropagationCheckpoints(np.array([1.]), original, original, original, original)
    original[:] = 99
    assert np.all(cp.x_m == 0)
    with pytest.raises(ValueError):
        cp.x_m.setflags(write=True)
    artifact = ArtifactBundle("incident", "s", "m", "c", {"x": cp.x_m}, {"a": [1]})
    with pytest.raises(ValueError):
        artifact.arrays["x"].setflags(write=True)
    with pytest.raises(TypeError):
        artifact.metadata["a"][0] = 2


def test_cached_alpha95_is_read_only_distinct_from_edge_and_live_state(instrument):
    cp = checkpoints()
    records = incident_checkpoint_observables(cp, alive=np.ones(101, bool), weights=np.ones(101))
    before = json_digest(records)
    instrument.lenses[1].percent += 5
    assert records["records"]["alpha95"]["value"] == pytest.approx(np.arctan(.0192), abs=1e-15)
    assert records["records"]["sampled_max_angle"]["value"] == pytest.approx(np.arctan(.02), abs=1e-15)
    assert records["records"]["physical_angular_edge"]["status"] == "UNAVAILABLE"
    assert json_digest(records) == before
    empty = incident_checkpoint_observables(cp, alive=np.zeros(101, bool), weights=np.ones(101))
    assert empty["records"]["source_fraction"]["value"] == 0
    assert empty["records"]["alpha95"]["value"] is None


def test_retired_illumination_cannot_be_enabled_by_labels_or_private_nodes(instrument):
    instrument.sample.wave_illumination = default_illumination_config()
    instrument._wave_source_node = object()
    with pytest.raises(ValueError, match="historical only"):
        illumination_config(instrument)
    del instrument._wave_source_node
    historical = capture_instrument_snapshot(instrument)
    assert historical.to_dict()
    with pytest.raises(ValueError, match="historical only"):
        historical.restore()


def test_historical_illumination_dialog_is_read_only(qtbot):
    from temsim.gui.illumination_dialog import IlluminationDialog
    from PySide6.QtWidgets import QComboBox, QDoubleSpinBox
    dialog = IlluminationDialog(default_illumination_config(), 300.)
    qtbot.addWidget(dialog)
    assert dialog.editor.isReadOnly()
    assert not dialog.findChildren(QComboBox)
    assert not dialog.findChildren(QDoubleSpinBox)
    dialog.accept()
    assert dialog.result() == 0


def test_persisted_seed_has_exact_working_point_and_frozen_diagnostics(instrument, tmp_path):
    from temsim.physics.simulation import run
    instrument.electron_gun.emitter.ray_count = 9
    instrument.step_mm = 5.
    instrument.history_step_mm = 5.
    manifest = capture_calculation_manifest(instrument)
    simulation = run(instrument)
    store = ArtifactStore(tmp_path / "cache", quota_bytes=100_000_000)
    store.put_incident_simulation_seed(manifest, simulation)
    bundle = store.get_array_bundle(manifest, product_key="incident",
        dependency_signature=manifest.calculation_signatures["incident"], codec="incident-simulation-seed-v1")
    snapshot = InstrumentSnapshot.from_dict(bundle.metadata["working_point"])
    assert snapshot.digest == manifest.instrument_snapshot.digest
    before = json_digest(bundle.metadata["beam_observables"])
    instrument.lenses[0].percent += 10
    restored = snapshot.restore()
    assert restored.lenses[0].percent == instrument.lenses[0].percent - 10
    assert json_digest(bundle.metadata["beam_observables"]) == before
    assert bundle.metadata["beam_observables"]["plane_z_mm"] == restored.sample.z_mm
    repeated = run(restored)
    from temsim.physics.beam_statistics import branch_sample_statistics
    assert bundle.metadata["beam_observables"]["records"]["alpha95"]["value"] == pytest.approx(
        branch_sample_statistics(repeated.incident).convergence_95_rad, rel=1e-12, abs=1e-15)
