"""Independent checks of the diagnostic's trajectory acceptance measurements."""
import numpy as np

from scripts.diagnose_accelerator_mechanism import trajectory_metrics
from scripts.diagnose_planar_gun import CHARGE, MASS, LIGHT


class UniformAcceleration:
    def potential_v_at_global_positions(self, p):
        return 1000*np.asarray(p)[..., 2]

    def field_at_global_positions_v_per_m(self, p):
        e = np.zeros_like(p)
        e[..., 2] = -1000.
        return e


def exact_states(z):
    # Two distinct emission energies propagated in a prescribed uniform field.
    energy = np.array([1., 2.])[None, :] + 1000*z[:, None]
    gminus1 = energy*CHARGE/(MASS*LIGHT**2)
    state = np.zeros((len(z), 2, 6))
    state[..., 4] = np.sqrt(gminus1*(2+gminus1))
    return state


def test_uniform_electrostatic_energy_measured_per_particle():
    z = np.linspace(0., .2, 129)
    state = exact_states(z)
    result = trajectory_metrics(z, state, UniformAcceleration(), state[0])
    assert result['max_energy_invariant_error_ev_all_saved'] < 1e-10
    assert result['minimum_pz_over_mc_all_saved'] > 0
    assert result['nonforward_saved_states'] == 0
    assert result['max_ez_v_per_m_on_saved_paths'] == -1000.


def test_interior_energy_error_cannot_hide_behind_correct_endpoints_or_mean():
    z = np.linspace(0., .2, 129)
    state = exact_states(z)
    # Equal/opposite single-particle energy errors cancel in the mean, and
    # occur away from both endpoints and a 64-state chunk boundary.
    for particle, change in enumerate([.02, -.02]):
        energy = particle+1+1000*z[71]+change
        gm = energy*CHARGE/(MASS*LIGHT**2)
        state[71, particle, 4] = np.sqrt(gm*(gm+2))
    result = trajectory_metrics(z, state, UniformAcceleration(), state[0])
    np.testing.assert_allclose(result['max_energy_invariant_error_ev_all_saved'], .02, atol=1e-10)


def test_energy_conservation_does_not_hide_a_backward_axial_momentum():
    z = np.linspace(0., .2, 129)
    state = exact_states(z)
    state[71, 0, 4] *= -1
    result = trajectory_metrics(z, state, UniformAcceleration(), state[0])
    assert result['max_energy_invariant_error_ev_all_saved'] < 1e-10
    assert result['minimum_pz_over_mc_all_saved'] < 0
    assert result['nonforward_saved_states'] == 1
