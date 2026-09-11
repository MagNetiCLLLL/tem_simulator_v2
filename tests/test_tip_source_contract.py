"""Production electrons start at the physical tip, including cached requests."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.optics.electron_gun.effective_source import EffectiveGunSource, gun_binding_digest
from temsim.optics.electron_gun.source_policy import UnsupportedSourceModel


@pytest.fixture
def state():
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    state.electron_gun.emitter.ray_count = 9
    return state


def historical_exit_selection(state):
    gun = state.electron_gun
    gun.effective_source = replace(EffectiveGunSource(1e-9), bound_gun_digest=gun_binding_digest(gun))
    gun.source_representation = "effective_gaussian_schell"


@pytest.mark.parametrize("entry", ["bind", "generate", "trace", "dispatch", "gun_trace", "current", "run", "calculate", "alignment"])
def test_even_a_matching_exit_binding_cannot_bypass_the_gun(state, entry):
    from temsim.optics.electron_gun import effective_source, source
    from temsim.physics.beam_current import effective_source_current_a
    from temsim.physics.simulation import run
    from temsim.simulation_pipeline import calculate
    from temsim.alignment_transaction import AlignmentRequest
    historical_exit_selection(state)
    before = capture_instrument_snapshot(state).digest
    gun = state.electron_gun
    calls = {
        "bind": lambda: effective_source.bind_effective_source(gun, gun.effective_source),
        "generate": lambda: effective_source.generate_gun_emission(gun),
        "trace": lambda: effective_source.trace_effective_source(state),
        "dispatch": lambda: source.trace_source_to_exit(state),
        "gun_trace": gun.trace_to_exit,
        "current": lambda: effective_source_current_a(state),
        "run": lambda: run(state, existing_simulation=object(), optical_only=True),
        "calculate": lambda: calculate(state, existing_result=object()),
        "alignment": lambda: AlignmentRequest.capture(state, "nanoprobe_convergence", 32., revision=0),
    }
    with pytest.raises(UnsupportedSourceModel, match="Custom exit sources are not permitted"):
        calls[entry]()
    assert capture_instrument_snapshot(state).digest == before


@pytest.mark.parametrize("quality", ["Preview", "Medium", "High accuracy"])
def test_all_request_qualities_reject_exit_source_before_reuse(state, quality):
    from temsim.gui.calculation_request import CapturedCalculationRequest
    from temsim.gui.calculation_controller import CalculationController
    historical_exit_selection(state)
    for capture in (lambda: CapturedCalculationRequest.capture(state, quality, 9, .2),
                    lambda: CalculationController._calculation_snapshot(state, quality, 9, .2)):
        with pytest.raises(UnsupportedSourceModel):
            capture()


def test_historical_profile_is_readable_but_cannot_mutate_active_state(state, tmp_path):
    from temsim.profile_io import save_profile, read_profile, apply_profile_values
    historical = default_state()
    historical_exit_selection(historical)
    path = tmp_path / "historical.toml"
    save_profile(path, historical, AssemblyCatalog().default_selection())
    original = path.read_bytes()
    _, values = read_profile(path)
    before = capture_instrument_snapshot(state).digest
    with pytest.raises(UnsupportedSourceModel):
        apply_profile_values(state, values)
    assert capture_instrument_snapshot(state).digest == before
    assert path.read_bytes() == original


@pytest.mark.parametrize("historical", [False, True])
def test_coherent_cache_cannot_substitute_an_exit_source(state, historical):
    from temsim.physics.gun_wave_transport import build_gun_wave_checkpoint
    from temsim.physics.gun_wave_cache import cached_gun_wave_checkpoint
    if historical:
        historical_exit_selection(state)
    class UnreadableStore:
        def get_array_bundle(self, *args, **kwargs):
            pytest.fail("No old exit-source cache may be admitted")
    for call in (lambda: build_gun_wave_checkpoint(state),
                 lambda: cached_gun_wave_checkpoint(state, UnreadableStore())):
        with pytest.raises(UnsupportedSourceModel):
            call()


def test_physical_dispatch_preserves_gun_history_and_cached_upstream_work(state):
    from temsim.optics.electron_gun.source import trace_source_to_exit
    gun = state.electron_gun
    trace = trace_source_to_exit(state)
    assert trace_source_to_exit(state) is trace
    assert trace is gun.trace_to_exit()
    assert trace.z_mm[0] == 0
    assert trace.z_mm[-1] == gun.exit_plane_z_mm
    assert trace.z_mm.size > 2
    assert trace.equal_time_history is not None
    assert trace.dpa_transmitted_current_a is not None
    assert trace.c1_transmitted_current_a is not None
    assert np.all(np.isfinite(trace.x_m))
    gun.c1_aperture.radius_mm = 0.
    blocked = trace_source_to_exit(state)
    assert blocked is not trace
    assert not np.any(blocked.exit_bundle.alive)
    assert blocked.c1_transmitted_current_a == 0.


def test_acceleration_and_gun_focusing_change_the_computed_exit_and_cache(state):
    from temsim.optics.electron_gun.source import trace_source_to_exit
    gun = state.electron_gun
    first = trace_source_to_exit(state)
    first_time = first.equal_time_history.time_s[-1]
    gun.accelerator.high_tension_kv = 200.
    slower = trace_source_to_exit(state)
    assert slower is not first
    assert slower.equal_time_history.time_s[-1] > first_time
    assert not np.array_equal(slower.exit_bundle.x_m, first.exit_bundle.x_m)
    gun.electrostatic_lens.voltage_kv *= 1.1
    refocused = trace_source_to_exit(state)
    assert refocused is not slower
    assert not np.array_equal(refocused.exit_bundle.x_m, slower.exit_bundle.x_m)
    assert trace_source_to_exit(state) is refocused


def test_tip_dialog_changes_only_tip_draft_and_rejects_invalid_inputs(state, qtbot):
    from temsim.gui.gun_source_dialog import GunSourceDialog
    before = capture_instrument_snapshot(state).digest
    dialog = GunSourceDialog(state.electron_gun)
    qtbot.addWidget(dialog)
    assert "reference_current_a" not in dialog.inputs
    dialog.inputs["angular_rms_mrad"].setText("nan")
    dialog.accept()
    assert "finite" in dialog.error.text()
    assert dialog.result() != dialog.DialogCode.Accepted
    dialog.inputs["angular_rms_mrad"].setText("2.5")
    dialog.accept()
    assert dialog.value()["angular_rms_mrad"] == 2.5
    assert capture_instrument_snapshot(state).digest == before


def test_model_inspector_applies_tip_parameters_without_editing_gun_optics(state, qtbot, monkeypatch):
    from temsim.gui.gun_source_dialog import GunSourceDialog
    from temsim.gui.model_inspector import ModelInspectorPage
    page = ModelInspectorPage()
    qtbot.addWidget(page)
    page.set_state(state)
    before = state.electron_gun.to_dict()
    monkeypatch.setattr(GunSourceDialog, "exec", lambda self: self.DialogCode.Accepted)
    monkeypatch.setattr(GunSourceDialog, "value", lambda self: {"angular_rms_mrad": 2.5})
    page._edit_gun_source()
    assert state.electron_gun.emitter.angular_rms_mrad == 2.5
    state.electron_gun.emitter.angular_rms_mrad = before["components"][state.electron_gun.emitter.key]["angular_rms_mrad"]
    assert state.electron_gun.to_dict() == before
