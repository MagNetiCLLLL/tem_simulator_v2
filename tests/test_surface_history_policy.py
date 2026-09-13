"""Storage/reuse wiring checks, not a source-physics or image benchmark."""
from dataclasses import replace

import pytest

from temsim.optics.column import default_state
from temsim.optics.electron_gun.tip_surface import SurfaceCoherence, load_tip_surface_reference
from temsim.physics.tip_wave_pipeline import TipWaveRequest, simulate_tip_wave
from temsim.physics.wave_execution import WaveExecutionOptions


@pytest.mark.parametrize("segmented", [False, True])
@pytest.mark.parametrize("reuse", [False, True])
def test_surface_history_write_store_is_present_independently_of_reuse(tmp_path, monkeypatch, segmented, reuse):
    state = default_state()
    state.electron_gun.emitter.surface_model = replace(load_tip_surface_reference(), coherence=SurfaceCoherence())
    request = TipWaveRequest(stop="gun_exit", execution=WaveExecutionOptions(
        segmented=segmented, cache_directory=str(tmp_path/"executed")))
    def inspect_policy(gun, **kwargs):
        assert kwargs["_energy_cache"] is not None
        assert kwargs["_energy_cache"].root.is_dir()
        assert kwargs["_reuse_energy_cache"] is reuse
        raise InterruptedError("Policy inspected before physical execution")
    monkeypatch.setattr("temsim.physics.surface_gun_wave.build_surface_gun_checkpoint", inspect_policy)
    with pytest.raises(InterruptedError, match="Policy inspected"):
        simulate_tip_wave(state, request, use_cache=reuse)


@pytest.mark.parametrize("value", [None, 0, 1, "false"])
def test_reuse_policy_rejects_ambiguous_non_boolean_values(value):
    from temsim.physics.surface_gun_wave import build_surface_gun_checkpoint
    with pytest.raises(ValueError, match="must be a boolean"):
        build_surface_gun_checkpoint(default_state().electron_gun, _reuse_energy_cache=value)
