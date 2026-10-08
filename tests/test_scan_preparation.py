"""Derived scan drives must be the actual fields shared by rays and waves."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QObject, Signal

from temsim.calculation_cache import state_model_signature
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.physics import scan_preparation as preparation


@pytest.fixture(autouse=True)
def empty_preparation_cache():
    preparation._PREPARED.clear()
    yield
    preparation._PREPARED.clear()


def _state():
    state = default_state()
    state.acceleration_backend = "CPU"
    state.acceleration_enabled = False
    state.step_mm = 0.2
    state.descan_deflector.scan_enabled = False
    return state


def _resolved_test_drives(state):
    # Deliberately cross-coupled matrices catch a lost off-diagonal entry;
    # setters also change scalar compatibility readbacks.
    state.ac_deflector.set_pure_shift_coupling(((-1.2, .1), (-.2, -1.3)))
    state.ac_deflector.set_scan_command_matrix_mrad(((.001, .0002), (-.0003, .002)))
    state.descan_deflector.set_image_plane_coupling(((-.7, .3), (.2, -.8)),
        target_key="test_target", target_z_mm=1800.)
    state.descan_deflector.set_scan_command_matrix_mrad(((-.001, -.0002), (.0003, -.002)))
    state.descan_deflector.scan_pixels_x = state.ac_deflector.scan_pixels_x


def test_real_automatic_preparation_is_private_and_reused(monkeypatch):
    import temsim.physics.scan_geometry as geometry
    live = _state()
    before = capture_instrument_snapshot(live)
    working = before.restore()
    real = geometry.calibrate_scan_system
    calls = []

    def calculate(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(geometry, "calibrate_scan_system", calculate)
    preparation.prepare_scan_drives(working, before.digest, observation_stop_z_mm=working.sample.z_mm)
    assert calls == [1]
    assert working.ac_deflector._scan_scale_calibrated
    assert not np.array_equal(working.ac_deflector.scan_command_matrix_mrad,
                              live.ac_deflector.scan_command_matrix_mrad)
    identity = preparation.scan_drive_identity(working)
    second = before.restore()
    preparation.prepare_scan_drives(second, before.digest, observation_stop_z_mm=second.sample.z_mm)
    assert calls == [1]
    assert preparation.scan_drive_identity(second) == identity
    assert capture_instrument_snapshot(live).digest == before.digest
    assert second.electron_gun.to_dict() == live.electron_gun.to_dict()


def test_held_record_is_restored_and_scaled_without_overwriting_live_controls(monkeypatch):
    from temsim.physics.scan_calibration import restore_held
    import temsim.physics.scan_geometry as geometry
    live = _state()
    ac = live.ac_deflector
    record = dict(version=1, ac_ratio=[[-1., 0.], [0., -1.]],
        descan_ratio=[[-.5, 0.], [0., -.5]], command_mrad=[[.001, .0002], [.0003, .002]],
        fov_nm=[ac.scan_field_of_view_x_nm, ac.scan_field_of_view_y_nm],
        target_z_mm=1800., target_key="test", descan_calibrated=False,
        reference=ac.scan_reference)
    ac.calibration_mode = "held"
    ac.calibration_record_json = json.dumps(record)
    ac.scan_pixel_size_nm *= 2
    snapshot = capture_instrument_snapshot(live)
    working = snapshot.restore()
    # Exercise the real persistent-record restoration without spending time
    # on its independent geometric residual measurement.
    monkeypatch.setattr(geometry, "calibrate_scan_system", lambda state, **_: restore_held(state))
    preparation.prepare_scan_drives(working, snapshot.digest, observation_stop_z_mm=working.sample.z_mm)
    np.testing.assert_array_equal(working.ac_deflector.scan_command_matrix_mrad,
                                  np.array(record["command_mrad"])*2)
    assert working.ac_deflector.calibration_record_json == ac.calibration_record_json
    assert working.ac_deflector.calibration_mode == "held"
    assert capture_instrument_snapshot(live).digest == snapshot.digest


def test_upstream_observation_does_not_require_downstream_held_record():
    state = _state()
    state.ac_deflector.calibration_mode = "held"
    snapshot = capture_instrument_snapshot(state)
    preparation.prepare_scan_drives(state, snapshot.digest,
        observation_stop_z_mm=state.ac_deflector.upper_z_mm-state.ac_deflector.effective_thickness_mm)
    assert capture_instrument_snapshot(state).digest == snapshot.digest
    with pytest.raises(ValueError, match="No held scan calibration"):
        preparation.prepare_scan_drives(state, snapshot.digest,
            observation_stop_z_mm=state.ac_deflector.upper_z_mm-1.)


def test_posed_coil_leading_support_prepares_before_unposed_boundary(monkeypatch):
    import temsim.physics.scan_geometry as geometry
    from temsim.physics.instrument_magnetic import column_dipole_fields
    state = _state()
    state._resolved_assembly = replace(state._resolved_assembly, parts=tuple(
        replace(part, data={**part.data, "rotation_y_mrad": 200.})
        if part.key == "ac_deflector" else part for part in state._resolved_assembly.parts))
    unposed = state.ac_deflector.upper_z_mm-state.ac_deflector.effective_thickness_mm*.5
    leading = min(coil.field_support_mm[0] for coil in column_dipole_fields(state)
                  if "ac_deflector" in coil.drive_keys)
    assert leading < unposed
    stop = (leading+unposed)*.5
    calls = []
    monkeypatch.setattr(geometry, "calibrate_scan_system",
        lambda working, **options: calls.append(options["observation_stop_z_mm"]))
    preparation.prepare_scan_drives(state, "posed-request", observation_stop_z_mm=stop)
    assert calls == [stop]


def test_prepare_cache_keeps_dynamic_request_identity(monkeypatch):
    import temsim.physics.scan_geometry as geometry
    calls = []
    monkeypatch.setattr(geometry, "calibrate_scan_system", lambda state, **_: (calls.append(1), _resolved_test_drives(state)))
    state = _state()
    for time in (0., .2):
        state.simulation_time_s = time
        snapshot = capture_instrument_snapshot(state)
        preparation.prepare_scan_drives(snapshot.restore(), snapshot.digest)
    assert calls == [1, 1]


@pytest.mark.parametrize("entry", ("direct", "session"))
def test_wave_entry_prepares_before_column_field_capture(monkeypatch, entry):
    import temsim.physics.tip_wave_pipeline as pipeline
    import temsim.physics.scan_geometry as geometry
    state = _state()
    before = capture_instrument_snapshot(state)
    calls = []
    monkeypatch.setattr(geometry, "calibrate_scan_system", lambda working, **_: (calls.append(1), _resolved_test_drives(working)))

    class ReachedPreparedColumn(Exception):
        pass

    def prepared_column(working, *_args):
        assert working.ac_deflector._scan_scale_calibrated
        assert working.ac_deflector.scan_command_matrix_mrad[0][1] == .0002
        raise ReachedPreparedColumn()

    monkeypatch.setattr(pipeline, "_prepare_column", prepared_column)
    request = replace(pipeline.TipWaveRequest(), stop="plane", observation_z_mm=state.sample.z_mm)
    session = pipeline.TipWaveObservationSession(state, request)
    for index in range(2):
        with pytest.raises(ReachedPreparedColumn):
            if entry == "direct":
                pipeline.simulate_tip_wave(state, request, use_cache=False)
            else:
                session.observe(state.sample.z_mm-index*.01)
    assert calls == [1]
    assert capture_instrument_snapshot(state).digest == before.digest


def test_scan_dwells_keep_one_prepared_request_and_distinct_emission_times(monkeypatch):
    import temsim.physics.tip_wave_scan as scanning
    import temsim.physics.tip_wave_pipeline as pipeline
    import temsim.physics.scan_geometry as geometry
    calls, identities, times = [], [], []
    state = _state()
    state.ac_deflector.scan_pixels_x = state.ac_deflector.scan_lines = 2
    monkeypatch.setattr(geometry, "calibrate_scan_system", lambda working, **_: (calls.append(1), _resolved_test_drives(working)))

    def execute(working, request, snapshot, identity, **kwargs):
        preparation.prepare_scan_drives(working, identity, observation_stop_z_mm=working.sample.z_mm,
                                       session_cache=kwargs["scan_preparation_cache"])
        identities.append(identity)
        times.append(request.tip_time_s)
        # Other background work may evict every global entry during a scan;
        # this captured scan must retain one fixed set of resolved drives.
        with preparation._PREPARED_LOCK:
            preparation._PREPARED.clear()
        return SimpleNamespace()

    monkeypatch.setattr(pipeline, "_execute_tip_wave", execute)
    rows = list(scanning.simulate_tip_scan(state, use_cache=False))
    assert len(rows) == 4 and len(set(times)) == 4
    assert len(set(identities)) == 1 and calls == [1]


class _Calculations(QObject):
    result_ready = Signal(str, object, float)
    failed = Signal(str, str)
    progress_changed = Signal(str, int, int, str)

    def __init__(self):
        super().__init__()
        self.submissions = []
        self.pool = SimpleNamespace(coordinator=SimpleNamespace(register_retained=lambda *_: None))

    def submit_background(self, *args, **kwargs):
        self.submissions.append((args, kwargs))

    def invalidate_pending(self, **kwargs):
        pass


def test_pair_hands_actual_particle_drives_to_wave_and_guards_them(qapp, monkeypatch):
    from temsim.gui import paired_beam_controller as pairing
    import temsim.physics.scan_geometry as geometry
    state = _state()
    before = capture_instrument_snapshot(state)
    controller = pairing.PairedBeamController(lambda: state, calculations=_Calculations())
    monkeypatch.setattr(pairing, "specimen_interactions_active", lambda _: False)
    monkeypatch.setattr(geometry, "calibrate_scan_system", lambda *_a, **_k: pytest.fail("GUI handoff must not recalibrate"))
    ready, errors = [], []
    controller.ready.connect(lambda *args: ready.append(args))
    controller.failed.connect(errors.append)
    controller.start(state, object(), {}, state.electron_gun.emitter.ray_count, state.step_mm)
    executed = before.restore()
    _resolved_test_drives(executed)
    assert state_model_signature(state) == state_model_signature(executed)
    controller._particle_ready("High accuracy", SimpleNamespace(workflow="rays", state_snapshot=executed), .01)
    assert not errors and len(ready) == 1
    prepared, _, _, context, _ = ready[0]
    assert preparation.scan_drive_identity(prepared) == preparation.scan_drive_identity(executed)
    assert capture_instrument_snapshot(prepared).digest == context.instrument_identity
    assert controller.is_current()
    assert capture_instrument_snapshot(state).digest == before.digest
    # Wave preparation recognises this exact resolved execution and does not
    # silently solve again at another particle/wave numerical step.
    preparation.prepare_scan_drives(prepared, context.instrument_identity)
    context.verify_particle()
    wave = SimpleNamespace(instrument_digest=context.instrument_identity,
                           resolved_scan_identity=context.executed_scan_identity)
    context.verify_wave(wave)
    wave.resolved_scan_identity = "different-executed-drive"
    with pytest.raises(ValueError, match="wave scan drives"):
        context.verify_wave(wave)
    wave.resolved_scan_identity = preparation.scan_drive_identity(prepared,
        emission_time_s=getattr(prepared, "simulation_time_s", 0.)+.1)
    with pytest.raises(ValueError, match="wave scan drives"):
        context.verify_wave(wave)
    # Old retained results remain readable, but cannot be claimed as a newly
    # verified matched execution without actual drive provenance.
    with pytest.raises(ValueError, match="wave scan drives"):
        context.verify_wave(SimpleNamespace(instrument_digest=context.instrument_identity))
    executed.ac_deflector.set_scan_command_matrix_mrad(((.002, .0002), (-.0003, .002)))
    assert state_model_signature(state) == state_model_signature(executed)
    with pytest.raises(ValueError, match="scan drives"):
        context.verify_particle()
    state.mini_condenser.percent += 1
    assert not controller.is_current()


def test_scan_drive_identity_detects_physical_host_and_timing_changes():
    state = _state()
    original = preparation.scan_drive_identity(state)
    state.ac_deflector.scan_frame_period_s *= 2
    assert preparation.scan_drive_identity(state) != original
    original = preparation.scan_drive_identity(state)
    state.ac_deflector.effective_thickness_mm /= 2
    assert preparation.scan_drive_identity(state) != original
    original = preparation.scan_drive_identity(state)
    state.simulation_time_s = getattr(state, "simulation_time_s", 0.)+.1
    assert preparation.scan_drive_identity(state) != original


@pytest.mark.parametrize("execution", ("matched", "mismatch", "legacy"))
def test_coherent_worker_only_compares_matching_executed_scan_drives(qapp, monkeypatch, execution):
    """Real Qt worker/guard; its numerical propagation is explicitly a stub."""
    from threading import Event
    import temsim.gui.coherent_beam as gui
    import temsim.gui.beam_plane_data as plane_data
    import temsim.physics.beam_comparison as comparison
    from temsim.gui.paired_beam_controller import BeamPairContext
    from temsim.physics.tip_wave_pipeline import TipWaveRequest, TipWaveResult
    state = _state()
    _resolved_test_drives(state)
    snapshot = capture_instrument_snapshot(state)
    drive_identity = preparation.scan_drive_identity(state)
    context = BeamPairContext("pair", snapshot.physical_digest,
        SimpleNamespace(state_snapshot=state), False, state.sample.z_mm,
        snapshot.digest, state_model_signature(state), state.electron_gun.emitter.ray_count,
        state.step_mm, drive_identity)
    request = TipWaveRequest(stop="plane", observation_z_mm=float(state.sample.z_mm))
    values = {} if execution == "legacy" else {
        "resolved_scan_identity": drive_identity if execution == "matched" else "other-drive"}
    result = TipWaveResult(SimpleNamespace(plane_z_mm=state.sample.z_mm), None,
                           request, False, snapshot.digest, **values)
    monkeypatch.setattr(gui, "simulate_tip_wave", lambda *_a, **_k: result)
    monkeypatch.setattr(gui, "_intensity_preview", lambda *_a, **_k:
        gui._Preview(np.ones((2, 2)), np.zeros((2, 2)), 1., 1, 32))
    monkeypatch.setattr(gui, "_plane_magnetic_gauge", lambda *_: None)
    sampled = []
    monkeypatch.setattr(plane_data, "sample_beam_plane", lambda *_:
        (sampled.append(1), SimpleNamespace())[1])
    monkeypatch.setattr(comparison, "compare_beam_planes", lambda *_: {"guard_fixture": True})
    worker = gui._WaveWorker(1, 1, state, request, Event(), pair_context=context)
    solved, failures = [], []
    worker.signals.solved.connect(lambda *args: solved.append(args))
    worker.signals.failed.connect(lambda *args: failures.append(args))
    worker.run()
    assert not failures and len(solved) == 1
    preview = solved[0][-1]
    if execution == "matched":
        assert sampled == [1] and preview.comparison == {"guard_fixture": True}
        assert preview.comparison_error is None
    else:
        assert not sampled and preview.comparison is None
        assert "wave scan drives" in preview.comparison_error


def test_concurrent_preparation_cache_eviction_keeps_private_outputs(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    import temsim.physics.scan_geometry as geometry
    monkeypatch.setattr(geometry, "calibrate_scan_system", lambda state, **_: _resolved_test_drives(state))
    states = [_state() for _ in range(4)]
    barrier = Barrier(len(states))

    def work(index):
        state = states[index]
        barrier.wait(timeout=10)
        for iteration in range(40):
            key = f"request-{index}-{iteration}"
            preparation.prepare_scan_drives(state, key)
            preparation.prepare_scan_drives(state, key)
            assert state.ac_deflector.scan_command_matrix_mrad[0][1] == .0002
        return preparation.scan_drive_identity(state)

    with ThreadPoolExecutor(max_workers=len(states)) as pool:
        identities = list(pool.map(work, range(len(states))))
    assert len(set(identities)) == 1
    assert len(preparation._PREPARED) <= preparation._MAXIMUM_PREPARED
