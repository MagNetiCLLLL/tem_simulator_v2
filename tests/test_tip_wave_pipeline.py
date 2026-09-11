from dataclasses import replace

import numpy as np
import pytest

from temsim.detector.wave_readout import WaveReadoutOptions
from temsim.optics.column import default_state
from temsim.optics.electron_gun.tip_coherence import TipCoherence, TipWaveNumerics
from temsim.physics.tip_gun_wave import GunWaveNumerics
from temsim.physics.tip_wave_pipeline import TipWaveRequest, simulate_tip_wave
from temsim.physics.wave_execution import WaveExecutionOptions


@pytest.mark.parametrize("segmented", [False, True])
def test_real_gun_with_shared_lens_tail_and_readout_independent_cache(monkeypatch, tmp_path, segmented):
    state = default_state()
    state.electron_gun.emitter.coherence = TipCoherence()
    state.electron_gun.emitter.energy_spread_fwhm_ev = 0.
    request = TipWaveRequest(stop="gun_exit", source=TipWaveNumerics(energy_samples=1),
                             execution=WaveExecutionOptions(segmented=segmented, cache_directory=str(tmp_path/"executed")),
                             gun=GunWaveNumerics(field_step_mm=.1, bore_step_mm=2.))
    result = simulate_tip_wave(state, request)
    assert result.checkpoint.record["shared_column"] is not None
    assert result.checkpoint.beam.total_weight <= 1+1e-10
    assert result.checkpoint.beam.modes[0].energy_kev == pytest.approx(300.)
    reference = result.checkpoint.beam.modes[0].axial_reference
    axial = result.checkpoint.record["energy_transport"][0]
    assert reference.flight_time_s == axial["reference_flight_time_s"] > 0
    assert reference.longitudinal_action_j_s == axial["reference_longitudinal_action_j_s"] > 0
    second = simulate_tip_wave(state, replace(request, readout=WaveReadoutOptions(intensity=False, phase=True)))
    assert second.propagation_cache_hit
    assert second.checkpoint.digest == result.checkpoint.digest
    if not segmented:
        assert second.checkpoint is result.checkpoint
    else:
        assert second.checkpoint.beam.resident_bytes == 0
    assert result.checkpoint.record["shared_column"]["boundary_gauge"].startswith("A=")
    import temsim.physics.tip_wave_pipeline as pipeline
    def recomputed(*args, **kwargs):
        raise RuntimeError("changed source must execute")
    monkeypatch.setattr(pipeline, "build_tip_gun_checkpoint", recomputed)
    state.electron_gun.emitter.emission_current_na *= .5
    with pytest.raises(RuntimeError, match="changed source must execute"):
        simulate_tip_wave(state, request)


def test_cancel_and_invalid_numerics_do_not_start_transport():
    state = default_state()
    with pytest.raises(InterruptedError):
        simulate_tip_wave(state, cancelled=lambda: True)
    for request in (TipWaveRequest(column_step_mm=0), TipWaveRequest(detector_pixels=True),
                    TipWaveRequest(maximum_readout_bytes=-1)):
        with pytest.raises(ValueError):
            request.validate()


def test_unexecuted_equivalent_state_is_not_a_pipeline_option():
    state = default_state()
    state.equivalent_image_lenses_enabled = True
    with pytest.raises(ValueError, match="executed upstream caches"):
        simulate_tip_wave(state, TipWaveRequest(stop="gun_exit"))


def test_installed_energy_filter_is_not_silently_bypassed():
    state = default_state()
    # A selected physical path crossing the real entrance remains unsupported.
    state.energy_filter.entrance_z_mm = min(d.z_mm for d in state.stem_detectors)-1.
    with pytest.raises(ValueError, match="energy filter requires"):
        simulate_tip_wave(state, TipWaveRequest(stop="detector"))


def test_filter_after_selected_detector_does_not_prevent_upstream_calculation(monkeypatch):
    import temsim.physics.tip_wave_pipeline as pipeline
    state = default_state()
    before = state.energy_filter.entrance_z_mm
    assert state.energy_filter_installed
    assert all(d.z_mm < before for d in (*state.stem_detectors, state.camera, state.fluorescent_screen))
    def gun_started(*args, **kwargs):
        raise RuntimeError("physical upstream gun calculation reached")
    monkeypatch.setattr(pipeline, "build_tip_gun_checkpoint", gun_started)
    for key in ("haadf", "bf", "camera"):
        with pytest.raises(RuntimeError, match="upstream gun calculation reached"):
            simulate_tip_wave(state, TipWaveRequest(stop="detector", detector_key=key))
    assert state.energy_filter_installed and state.energy_filter.entrance_z_mm == before


def test_partial_stage_cannot_cross_a_filter_moved_upstream():
    state = default_state()
    state.energy_filter.entrance_z_mm = state.electron_gun.exit_plane_z_mm-1.
    with pytest.raises(ValueError, match="energy filter requires"):
        simulate_tip_wave(state, TipWaveRequest(stop="gun_exit"))


def test_material_multislice_with_actual_atomistic_potential_preserves_branches():
    """An independent specimen operator fixture, not source-chain acceptance."""
    from temsim.physics.specimen_wave_transport import _propagate_specimen
    from temsim.physics.multiplane_wave import PlaneWave
    from temsim.physics.tip_gun_wave import TipGunCheckpoint
    from temsim.physics.wave_flux import BeamState, WaveMode
    from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE
    from temsim.physics.wave_reference import AxialWaveReference
    state = default_state()
    for collection in (state.lenses, state.stigmators, state.corrector_elements, state.deflectors):
        for component in collection:
            component.enabled = False
    state.apertures = []
    state.sample.thickness_nm = .4
    state.sample.wave_slice_thickness_angstrom = 2.
    state.sample.wave_grid_pixels = 128
    state.sample.wave_field_of_view_angstrom = 30.
    state.sample.wave_frozen_phonon_enabled = True
    state.sample.wave_frozen_phonon_configurations = 2
    axis = (np.arange(128)-64)*2e-11
    xx, yy = np.meshgrid(axis, axis)
    amplitude = np.exp(-(xx*xx+yy*yy)/(2*(.2e-9)**2)).astype(complex)
    amplitude /= np.linalg.norm(amplitude)
    reference = AxialWaveReference(2e-8, 1e-22)
    mode = WaveMode(PlaneWave(amplitude, np.eye(2)*2e-11, np.zeros(2)), .4, TIP_REFERENCE, "specimen-fixture", 300., reference)
    checkpoint = TipGunCheckpoint(BeamState((mode,), TIP_REFERENCE), state.sample.z_mm-state.sample.thickness_nm*.5e-6,
        1e-9, {"fixture": "independent physical specimen operator; no source-chain qualification"})
    result = _propagate_specimen(state, checkpoint)
    assert len(result.beam.modes) == 2
    assert 0 < result.beam.total_weight < .4
    assert result.record["potential"]["atomistic_applied"]
    assert result.record["potential"]["frozen_phonon_applied"]
    assert all("phonon:" in m.mode_id for m in result.beam.modes)
    from temsim.physics.tip_gun_wave import _momentum_velocity
    momentum, velocity = _momentum_velocity(300000.)
    for m, row in zip(result.beam.modes, result.record["modes"]):
        assert m.axial_reference.flight_time_s-reference.flight_time_s == pytest.approx(.4e-9/velocity, rel=2e-6, abs=1e-23)
        assert m.axial_reference.longitudinal_action_j_s-reference.longitudinal_action_j_s == pytest.approx(.4e-9*momentum, rel=2e-6, abs=1e-38)
        balance = row["probability_balance"]
        assert balance["accounted_weight"] == pytest.approx(row["input_weight"], abs=1e-13)
        assert balance["first_events"]["real_plasmon"] > 0
    assert all(m.plane.amplitude.imag.max() > 0 for m in result.beam.modes)
    assert result.plane_z_mm == pytest.approx(state.sample.z_mm+state.sample.thickness_nm*.5e-6)
