"""Scan restart admission; real signatures, no gun or image calculation."""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from temsim.calculation_cache import calculation_signatures, scan_controls_only_incident_change
from temsim.gui.calculation_request import apply_request_numerics
from temsim.instrument_snapshot import decode_instrument, encode_instrument
from temsim.optics.column import default_state
from temsim.simulation_workflow import require_incident_result


@pytest.fixture(scope="module")
def instrument_graph():
    state = default_state()
    state.sample.wave_enabled = state.sample.stem_wave_enabled = False
    state.ac_deflector.scan_enabled = state.descan_deflector.scan_enabled = False
    apply_request_numerics(state, "High accuracy", 3000, .1)
    return encode_instrument(state)


@pytest.fixture
def pair(instrument_graph):
    previous = decode_instrument(instrument_graph)
    current = decode_instrument(instrument_graph)
    signature = calculation_signatures(previous)["incident"]
    current.ac_deflector.scan_enabled = True
    current.ac_deflector.wobble_enabled = False
    current.descan_deflector.scan_enabled = True
    for component in (current.ac_deflector, current.descan_deflector):
        component.scan_pixel_size_nm = .02
        component.scan_pixels_x = component.scan_lines = 32
    return previous, current, signature


def test_real_scan_changes_require_reexecution_without_weakening_incident_identity(pair):
    previous, current, signature = pair
    assert calculation_signatures(current)["incident"] != signature
    assert scan_controls_only_incident_change(previous, current, signature)
    assert calculation_signatures(current)["incident"] != signature
    assert not previous.ac_deflector.scan_enabled
    assert previous.ac_deflector.wobble_enabled


def test_finite_scan_coil_field_width_changes_the_incident_signature(instrument_graph):
    state = decode_instrument(instrument_graph)
    before = calculation_signatures(state)["incident"]
    state.ac_deflector.effective_thickness_mm += .1
    assert calculation_signatures(state)["incident"] != before


def test_sample_structure_and_dose_edits_can_accompany_scan_reexecution(pair):
    previous, current, signature = pair
    current.sample.cif_path = "test-structure-not-loaded-in-this-signature-fixture.cif"
    current.sample.thickness_nm += 1.
    current.sample.inserted = not current.sample.inserted
    current.column_current_limit_percent *= .5
    assert scan_controls_only_incident_change(previous, current, signature)


@pytest.mark.parametrize("component, field, value", [
    ("ac_deflector", "upper_coil_gain", .6),
    ("ac_deflector", "pivot_offset_x", .1),
    ("ac_deflector", "scan_frame_period_s", 2.),
    ("ac_deflector", "scan_reference", "sample_entrance"),
    ("descan_deflector", "descan_target_key", "camera"),
])
def test_scan_drive_adjustments_admit_reexecution(pair, component, field, value):
    previous, current, signature = pair
    setattr(getattr(current, component), field, value)
    assert scan_controls_only_incident_change(previous, current, signature)


def test_held_calibration_record_is_a_drive_dependency_not_a_new_source(pair):
    previous, current, signature = pair
    # An explicit mathematical held-drive record tests input admission only.
    # No claim is made that these matrices were calibrated on this instrument.
    current.ac_deflector.calibration_record_json = json.dumps(dict(
        version=1, ac_ratio=[[-1., 0.], [0., -1.]], descan_ratio=[[1., 0.], [0., 1.]],
        command_mrad=[[.001, 0.], [0., .001]], fov_nm=[.64, .64],
        target_z_mm=float(current.camera.z_mm), target_key="camera",
        descan_calibrated=True, reference="sample_centre"))
    current.ac_deflector.calibration_mode = "held"
    assert scan_controls_only_incident_change(previous, current, signature)


@pytest.mark.parametrize("change", [
    lambda state: setattr(state.lenses[0], "percent", state.lenses[0].percent+1.),
    lambda state: setattr(state.electron_gun.emitter, "ray_count", 5000),
    lambda state: setattr(state, "step_mm", .2),
    lambda state: setattr(state, "history_step_mm", 1.),
    lambda state: setattr(state, "simulation_time_s", .125),
    lambda state: setattr(state.deflectors[0], "upper_x_mrad", state.deflectors[0].upper_x_mrad+.01),
    lambda state: setattr(state.vacuum_map, "enabled", True),
    lambda state: setattr(state.ac_deflector, "z_mm", state.ac_deflector.z_mm+1.),
    lambda state: setattr(state.ac_deflector, "effective_thickness_mm", state.ac_deflector.effective_thickness_mm+.1),
], ids=("lens", "source-sampling", "integration-step", "history-step", "playback-time",
         "other-deflector", "vacuum-physics", "scan-coil-position", "scan-coil-field-width"))
def test_scan_restart_cannot_hide_other_consumed_input_changes(pair, change):
    previous, current, signature = pair
    change(current)
    assert not scan_controls_only_incident_change(previous, current, signature)


def test_matching_or_unbound_previous_state_does_not_admit_scan_restart(pair):
    previous, current, signature = pair
    assert not scan_controls_only_incident_change(previous, previous, signature)
    assert not scan_controls_only_incident_change(None, current, signature)
    assert not scan_controls_only_incident_change(previous, current, "")
    assert not scan_controls_only_incident_change(previous, current, "forged-identity")
    previous.lenses[0].percent += 1.
    assert not scan_controls_only_incident_change(previous, current, signature)


def _prerequisite(state, signature):
    zero = np.zeros((1, 2), dtype=np.float64)
    checkpoint = SimpleNamespace(z_mm=np.array([state.sample.z_mm]),
        x_m=zero, tx_rad=zero, y_m=zero, ty_rad=zero,
        kinetic_energy_ev=np.full((1, 2), 3e5), flight_time_s=np.full((1, 2), 1e-8))
    simulation = SimpleNamespace(incident_checkpoints=checkpoint,
        incident=SimpleNamespace(z=checkpoint.z_mm, x=zero), incident_plan=object(),
        gun_trace=object())
    return SimpleNamespace(signatures={"incident": signature}, simulation=simulation)


def test_complete_old_incident_is_rejected_for_new_scan_without_restart_admission(pair):
    previous, current, signature = pair
    result = _prerequisite(previous, signature)
    assert require_incident_result(previous, result, {"incident": signature}) is result.simulation
    with pytest.raises(ValueError, match="upstream source, optics or numerical settings"):
        require_incident_result(current, result, calculation_signatures(current))
    with pytest.raises(ValueError, match="upstream continuation checkpoint"):
        require_incident_result(current, result, calculation_signatures(current), scan_continuation=True)


@pytest.mark.parametrize("missing", ["incident_plan", "incident_checkpoints", "gun_trace"])
def test_scan_restart_never_admits_incomplete_executed_boundary(pair, missing):
    previous, current, signature = pair
    result = _prerequisite(previous, signature)
    setattr(result.simulation, missing, None)
    with pytest.raises(ValueError, match="completed tip-to-specimen"):
        require_incident_result(current, result, calculation_signatures(current), scan_continuation=True)


@pytest.mark.parametrize("field", ["x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s", "kinetic_energy_ev"])
def test_scan_restart_requires_full_precision_state_at_boundary(pair, field):
    previous, current, signature = pair
    result = _prerequisite(previous, signature)
    checkpoint = result.simulation.incident_checkpoints
    setattr(checkpoint, field, np.asarray(getattr(checkpoint, field), dtype=np.float32))
    with pytest.raises(ValueError, match="full-precision incident checkpoints"):
        require_incident_result(current, result, calculation_signatures(current), scan_continuation=True)
