"""Conditional material instrument and double propagation; independent fixtures."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics.inelastic_wave import _collide_slice, _trajectory_rng, _propagate_inelastic_specimen
from temsim.physics.wave_execution import InelasticWaveNumerics
from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
from temsim.physics.wave_grid import WaveGridNumerics
from temsim.physics.wave_reference import AxialWaveReference
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.tip_gun_wave import TipGunCheckpoint, _momentum_velocity
from temsim.physics.wave_flux import BeamState, WaveMode
from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE, wavelength_m
from test_segmented_tip_wave import quiet_state


def distribution(zero=.5, scatter=.4, absorbed=.1):
    return SimpleNamespace(absorbed_probability=absorbed, channels=(
        SimpleNamespace(key="real_zero_loss", probability=zero, energy_loss_ev=0., characteristic_angle_mrad=0., mean_events=0., approximation="fixture"),
        SimpleNamespace(key="real_plasmon", probability=scatter, energy_loss_ev=100., characteristic_angle_mrad=.02, mean_events=1., approximation="fixture")))


def mode():
    n = 16
    axis = (np.arange(n)-n//2)*2e-11
    xx, yy = np.meshgrid(axis, axis)
    a = np.exp(-(xx*xx+yy*yy)/(4*(4e-11)**2))*(1+.2j)
    a /= np.linalg.norm(a)
    return WaveMode(PlaneWave(a, np.eye(2)*2e-11, np.array((1e-11, -2e-11)), np.eye(2)*20., np.array((1e-5, 2e-5))),
                    .4, TIP_REFERENCE, "material-fixture", 300., AxialWaveReference(1e-8, 1e-22))


def test_inelastic_jump_keeps_full_phase_changes_energy_and_canonical_carrier_units():
    old = mode()
    new, event = _collide_slice(old, np.ones((16, 16), bool), distribution(0., 1., 0.), _trajectory_rng(0, "phase"), 1.)
    assert new.energy_kev == pytest.approx(299.9)
    assert new.weight_per_reference_electron == old.weight_per_reference_electron
    assert new.axial_reference is old.axial_reference
    xy = old.plane.coordinates_m()
    expected = old.plane.full_amplitude(float(wavelength_m(300000)))*np.exp(2j*np.pi*np.einsum("i,iyx->yx", event["kick_rad"], xy)/float(wavelength_m(299900)))
    np.testing.assert_allclose(new.plane.full_amplitude(float(wavelength_m(299900))), expected, atol=5e-14)
    assert new.scattering_history[-1]["kind"] == "real_plasmon"


def test_sampled_instrument_matches_born_probabilities_in_finite_material():
    original = mode()
    inside = original.plane.coordinates_m()[0] > original.plane.origin_m[0]
    material_probability = float(np.sum(abs(original.plane.amplitude[inside])**2))
    expected = np.array([1-material_probability*.5, material_probability*.4, material_probability*.1])
    names = ("real_zero_loss", "real_plasmon", "effective_absorption")
    counts, samples = np.zeros(3), 2500
    rng = _trajectory_rng(937, "population")
    for _ in range(samples):
        result, event = _collide_slice(original, inside, distribution(), rng, 1.)
        counts[names.index(event["kind"])] += 1
        assert result.weight_per_reference_electron+event["absorbed_weight"] == pytest.approx(.4)
    sigma = np.sqrt(expected*(1-expected)/samples)
    assert np.all(abs(counts/samples-expected) < 5*sigma)


def test_counter_rng_is_independent_of_preceding_slices():
    first = _trajectory_rng(3, "tip", 0, 4, 7).random(10)
    _trajectory_rng(3, "tip", 0, 4, 6).random(1000)
    np.testing.assert_array_equal(first, _trajectory_rng(3, "tip", 0, 4, 7).random(10))
    assert not np.array_equal(first, _trajectory_rng(4, "tip", 0, 4, 7).random(10))


def test_outgoing_wave_propagates_remaining_material_at_new_energy_and_resumes(tmp_path, monkeypatch):
    state = quiet_state()
    state.sample.thickness_nm = .4
    state.sample.wave_slice_thickness_angstrom = 2.
    state.sample.wave_grid_pixels = 128
    state.sample.wave_field_of_view_angstrom = 30.
    state.sample.wave_frozen_phonon_enabled = False
    axis = (np.arange(128)-64)*2e-11
    xx, yy = np.meshgrid(axis, axis)
    amplitude = np.exp(-(xx*xx+yy*yy)/(2*(.2e-9)**2)).astype(complex)
    amplitude /= np.linalg.norm(amplitude)
    reference = AxialWaveReference(1e-8, 1e-22)
    beam_mode = WaveMode(PlaneWave(amplitude, np.eye(2)*2e-11, np.zeros(2)), .4, TIP_REFERENCE, "specimen-fixture", 300., reference)
    checkpoint = TipGunCheckpoint(BeamState((beam_mode,), TIP_REFERENCE), state.sample.z_mm-.2e-6, 1e-9, {"fixture": "independent atomistic operator"})
    energies = []
    def material(energy_state):
        energies.append(energy_state.beam_voltage_kv)
        d = distribution(0., 1., 0.)
        d.channels[1].characteristic_angle_mrad = 0.
        return d
    monkeypatch.setattr("temsim.specimen.inelastic.real_inelastic_distribution", material)
    store = ExecutedWaveStore(tmp_path, "actual-specimen-operator-fixture", 1<<29)
    kw = dict(numerics=InelasticWaveNumerics(trajectories_per_mode=2), store=store,
        maximum_step_mm=.5, grid_numerics=WaveGridNumerics(), tip_time_s=0.,
        cancelled=lambda: False, progress_callback=None, verify=lambda: None, use_cache=True)
    result = _propagate_inelastic_specimen(state, checkpoint, **kw)
    assert energies == pytest.approx([300., 299.9, 300., 299.9])
    assert len(result.beam.modes) == 2
    p1, v1 = _momentum_velocity(300000.); p2, v2 = _momentum_velocity(299900.)
    for value in result.beam.modes:
        assert value.energy_kev == pytest.approx(299.8)
        assert len(value.scattering_history) == 2
        assert value.axial_reference.flight_time_s-reference.flight_time_s == pytest.approx(.2e-9/v1+.2e-9/v2, rel=3e-6, abs=1e-23)
        assert value.axial_reference.longitudinal_action_j_s-reference.longitudinal_action_j_s == pytest.approx(.2e-9*(p1+p2), rel=3e-6, abs=1e-38)
    reused = _propagate_inelastic_specimen(state, checkpoint, **kw)
    assert reused.digest == result.digest
    assert len(energies) == 4
    # Interrupted execution must reuse a committed conditional slice and its
    # complete energy/history/RNG state, not restart from an invented pupil.
    interrupted_store = ExecutedWaveStore(tmp_path/"interrupted", "actual-specimen-operator-fixture", 1<<29)
    stop = [False]
    put = interrupted_store.put
    def cancel_after_commit(key, checkpoint):
        value = put(key, checkpoint)
        stop[0] = True
        return value
    monkeypatch.setattr(interrupted_store, "put", cancel_after_commit)
    kw["store"], kw["cancelled"] = interrupted_store, lambda: stop[0]
    with pytest.raises(InterruptedError):
        _propagate_inelastic_specimen(state, checkpoint, **kw)
    assert energies[4:] == [300.]
    stop[0] = False
    monkeypatch.setattr(interrupted_store, "put", put)
    resumed = _propagate_inelastic_specimen(state, checkpoint, **kw)
    assert energies[5:] == pytest.approx([299.9, 300., 299.9])
    for a, b in zip(resumed.beam.modes, result.beam.modes):
        np.testing.assert_allclose(a.plane.amplitude, b.plane.amplitude, rtol=0, atol=0)
        assert a.energy_kev == b.energy_kev
        assert a.scattering_history == b.scattering_history
    assert not list((tmp_path/"interrupted").glob("pending-*"))
