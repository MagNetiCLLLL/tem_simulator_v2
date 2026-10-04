"""Independent discrete moments and physical reference checks."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE
from temsim.physics.beam_comparison import compare_beam_planes, wave_plane_moments, particle_plane_moments
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.tip_gun_wave import TipGunCheckpoint
from temsim.physics.wave_flux import BeamState, WaveMode


def _checkpoint(*, weight=.8, current=2e-9, z=100.):
    wave = PlaneWave(np.ones((2, 2), complex)/2, np.diag((2e-9, 4e-9)), np.array((1e-9, 2e-9)))
    mode = WaveMode(wave, weight, TIP_REFERENCE, "one", 300.)
    return TipGunCheckpoint(BeamState((mode,), TIP_REFERENCE), z, current, {})


def _particles():
    return SimpleNamespace(z_mm=100., x_m=np.array((-1, 1, -1, 1))*1e-9,
        y_m=np.array((-2, -2, 2, 2))*1e-9, source_fraction=np.full(4, .2),
        source_current_pa=2000., weights_valid=True, coordinate_frame="column",
        kinetic_energy_ev=np.full(4, 300000.), ray_count=4, provenance="Incident")


def test_same_discrete_population_uses_absolute_loss_and_same_physical_units():
    result = compare_beam_planes(_particles(), _checkpoint())
    for branch in (result["particle"], result["wave"]):
        assert branch["source_fraction"] == pytest.approx(.8)
        assert branch["current_a"] == pytest.approx(1.6e-9)
        np.testing.assert_allclose(branch["centroid_xy_m"], (0, 0), atol=1e-24)
        np.testing.assert_allclose(branch["rms_xy_m"], (1e-9, 2e-9), rtol=1e-14, atol=0)
        assert branch["radial_rms_m"] == pytest.approx(np.sqrt(5)*1e-9, abs=1e-23)
        assert branch["mean_energy_ev"] == 300000.
        assert branch["rms_energy_ev"] == 0.


def test_incoherent_mode_phase_does_not_cancel_current_and_between_mode_width_is_retained():
    checkpoint = _checkpoint(weight=1.)
    first = checkpoint.beam.modes[0]
    left = replace(first, plane=replace(first.plane, origin_m=np.array((-1e-9, 2e-9))),
                   weight_per_reference_electron=.25, energy_kev=100., mode_id="left")
    right = replace(first, plane=replace(first.plane, amplitude=-first.plane.amplitude,
                    origin_m=np.array((3e-9, 2e-9))), weight_per_reference_electron=.75,
                    energy_kev=300., mode_id="right")
    result = wave_plane_moments(replace(checkpoint, beam=BeamState((left, right), TIP_REFERENCE)))
    assert result["source_fraction"] == pytest.approx(1.)
    np.testing.assert_allclose(result["centroid_xy_m"], (1e-9, 0.), atol=1e-24)
    # Within-mode variance 1 nm² plus weighted between-mode variance 3 nm².
    np.testing.assert_allclose(result["rms_xy_m"], (2e-9, 2e-9), rtol=1e-14, atol=0.)
    assert result["mean_energy_ev"] == 250000.
    assert result["rms_energy_ev"] == pytest.approx(np.sqrt(.25*.75)*200000.)


def test_rotated_sheared_lattice_moments_stay_in_laboratory_coordinates():
    checkpoint = _checkpoint()
    mode = checkpoint.beam.modes[0]
    basis = np.array(((2., 1.), (-1., 3.)))*1e-9
    origin = basis @ np.array((.5, .5))
    plane = replace(mode.plane, basis_m=basis, origin_m=origin)
    result = wave_plane_moments(replace(checkpoint, beam=BeamState((replace(mode, plane=plane),), TIP_REFERENCE)))
    np.testing.assert_allclose(result["centroid_xy_m"], (0., 0.), atol=1e-24)
    np.testing.assert_allclose(result["rms_xy_m"], np.sqrt((5., 10.))*1e-9/2, rtol=1e-14, atol=0.)


def test_unknown_historical_particle_energy_is_not_nominal_voltage():
    particles = _particles()
    particles.kinetic_energy_ev = np.full(4, np.nan)
    result = particle_plane_moments(particles)
    assert result["mean_energy_ev"] is None
    assert result["rms_energy_ev"] is None
    assert result["current_a"] == pytest.approx(1.6e-9)


def test_empty_and_zero_weight_populations_do_not_invent_width_or_energy():
    particles = _particles()
    particles.source_fraction[:] = 0.
    result = compare_beam_planes(particles, _checkpoint(weight=0.))
    for branch in (result["particle"], result["wave"]):
        assert branch["source_fraction"] == 0.
        assert branch["current_a"] == 0.
        assert branch["centroid_xy_m"] is None
        assert branch["rms_xy_m"] is None
        assert branch["mean_energy_ev"] is None


@pytest.mark.parametrize("changes,checkpoint,match", [
    ({"z_mm": 101.}, _checkpoint(), "same Z"),
    ({}, _checkpoint(current=3e-9), "current reference"),
    ({"weights_valid": False}, _checkpoint(), "valid particle"),
    ({"coordinate_frame": "filter"}, _checkpoint(), "laboratory"),
])
def test_comparison_refuses_mismatched_planes_or_ambiguous_particle_metadata(changes, checkpoint, match):
    particles = _particles()
    particles.__dict__.update(changes)
    with pytest.raises(ValueError, match=match):
        compare_beam_planes(particles, checkpoint)


def test_plane_sampler_retains_executed_energy_with_same_column_selection():
    from temsim.gui.beam_plane_data import sample_beam_plane
    branch = SimpleNamespace(z=np.array((1., 3.)), x=np.zeros((2, 2)), y=np.zeros((2, 2)),
        tx=np.zeros((2, 2)), ty=np.zeros((2, 2)), ray_weight=np.array((.25, .75)),
        blocked_z=np.array((np.nan, 1.5)), kinetic_energy_ev=np.array(((10., 20.), (30., 40.))))
    result = SimpleNamespace(simulation=SimpleNamespace(incident=branch,
        metrics={"effective_source_current_pa": 100.}))
    plane = sample_beam_plane(result, 2.)
    np.testing.assert_array_equal(plane.kinetic_energy_ev, (20.,))
    np.testing.assert_array_equal(plane.source_fraction, (.25,))


def test_readout_energy_validation_is_bounded_to_selected_rows(monkeypatch):
    from temsim.physics import particle_energy
    original = particle_energy.validate_kinetic_energy_array
    inspected = []
    def checked(values, shape, *args, **kwargs):
        inspected.append(np.asarray(values).shape)
        return original(values, shape, *args, **kwargs)
    monkeypatch.setattr(particle_energy, "validate_kinetic_energy_array", checked)
    branch = SimpleNamespace(z=np.arange(1000, dtype=float), x=np.zeros((1000, 5)),
                             kinetic_energy_ev=np.ones((1000, 5))*10.)
    sampled = particle_energy.sample_kinetic_energy(branch, 2.5, validate_history=False)
    np.testing.assert_array_equal(sampled, np.full(5, 10.))
    assert inspected == [(2, 5)]
    branch.kinetic_energy_ev[3, 0] = -1.
    with pytest.raises(ValueError, match="positive float64"):
        particle_energy.sample_kinetic_energy(branch, 2.5, validate_history=False)
