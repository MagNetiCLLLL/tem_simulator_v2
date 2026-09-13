"""Actual FEG electric-field contribution; never full source/gun acceptance."""
from dataclasses import asdict

import numpy as np
import pytest
from scipy.constants import e, hbar, m_e

from temsim.optics.column import default_state
from temsim.immutable_json import json_digest
from temsim.instrument_snapshot import encode_instrument
from temsim.physics.electrostatic_wave_channels import potential_matrix_v, sample_feg_electrostatic_channels as sample
from temsim.physics.coupled_low_energy import propagate_coupled_boundary


@pytest.mark.parametrize("shape,quadrature", [((3, 4), (6, 8)), ((3, 3), (9, 9))])
def test_potential_matrix_matches_independent_exponential_quadrature(shape, quadrature):
    phi = np.random.default_rng(18).normal(size=quadrature)
    xy = np.meshgrid(*(np.arange(n)-n//2 for n in quadrature), indexing="ij")
    modes = np.meshgrid(*(np.arange(n)-n//2 for n in shape), indexing="ij")
    phase = sum(x.ravel()[:, None]*k.ravel()[None, :]/n for x, k, n in zip(xy, modes, quadrature))
    r = np.exp(2j*np.pi*phase)/np.sqrt(np.prod(quadrature))
    expected = r.conj().T@(phi.ravel()[:, None]*r)
    actual = potential_matrix_v(phi, shape)
    np.testing.assert_allclose(actual, expected, atol=5e-15)
    np.testing.assert_allclose(actual, actual.conj().T, atol=1e-15)


def test_actual_gun_field_includes_radial_extractor_coupling_and_records_unchanged_inputs():
    gun = default_state().electron_gun
    original = json_digest(encode_instrument(gun))
    shape, basis = (4, 4), np.eye(2)*1e-6
    edges = [.1, .100001, .100002]
    result = sample(gun, edges, .3, shape, basis, np.zeros(2))
    assert result.record["gun_input_identity"] == original == json_digest(encode_instrument(gun))
    assert result.record["gun_source_admission"] == "NOT_A_GUN_CHECKPOINT"
    assert set(result.record["electrostatic_owners"]) == {gun.extractor.key, gun.electrostatic_lens.key, gun.accelerator.key}
    assert result.record["tip_component_energy_ev"] == .3
    assert not result.q_m2.flags.writeable
    # Independent analytic quadratic field from the actual provider.
    z = np.array([(edges[0]+edges[1])/2])
    phi, _, d2, _ = gun.electric_field.axial_potential_v_and_derivatives_per_mm(z)
    x = (np.arange(8)-4)*.5e-6
    xx, yy = np.meshgrid(x, x)
    kinetic = .3+float(phi[0])-.25*(xx*xx+yy*yy)*float(d2[0])*1e6
    v = potential_matrix_v(kinetic, shape)
    ky, kx = np.meshgrid((np.arange(4)-2)*2*np.pi/(4e-6), (np.arange(4)-2)*2*np.pi/(4e-6), indexing="ij")
    expected = 2*m_e*e/hbar**2*v-np.diag((kx*kx+ky*ky).ravel())
    np.testing.assert_allclose(result.q_m2[0], expected, rtol=2e-13, atol=1e4)
    off_diagonal = result.q_m2[0]-np.diag(np.diag(result.q_m2[0]))
    assert np.linalg.norm(off_diagonal) > 1e12  # actual transverse focusing was not discarded
    # Operator fixture at an interior field segment, NOT a replacement source.
    left = np.zeros(16, complex); left[10] = 1.
    propagated = propagate_coupled_boundary(result.q_m2, result.widths_m, result.exit_q_m2, left)
    assert propagated.record["current_balance_residual"] < 1e-12
    assert np.linalg.norm(np.delete(propagated.boundary_amplitude[-1], 10)) > 1e-5
    gun.extractor.voltage_kv *= 1.1
    changed = sample(gun, edges, .3, shape, basis, np.zeros(2))
    assert changed.record["gun_input_identity"] != result.record["gun_input_identity"]
    assert not np.array_equal(changed.q_m2, result.q_m2)


def test_potential_quadrature_refines_independently_at_fixed_physical_domain(record_property):
    gun = default_state().electron_gun
    matrices = [sample(gun, [.1, .100001], .3, (4, 4), np.eye(2)*1e-6, np.zeros(2),
                       quadrature_factor=factor).q_m2[0] for factor in (2, 4, 8, 16)]
    differences = [float(np.linalg.norm(b-a)) for a, b in zip(matrices, matrices[1:])]
    record_property("quadrature_matrix_differences", str(differences))
    assert all(b < a/3. for a, b in zip(differences, differences[1:]))


def test_tip_energy_components_are_used_without_retuning_the_distribution():
    from temsim.optics.electron_gun.tip_coherence import tip_energy_samples
    gun = default_state().electron_gun
    before = asdict(gun.emitter)
    energies = tip_energy_samples(gun.emitter, 9)
    for energy in energies[[0, 4, 8]]:
        prepared = sample(gun, [0., .000001], float(energy), (4, 4), np.eye(2)*2e-9, np.zeros(2))
        assert prepared.record["tip_component_energy_ev"] == energy
        # The existing electric field is zero here: this is not a fabricated
        # new near-tip acceleration law or a propagated-source claim.
        assert prepared.record["local_kinetic_ranges_ev"][0] == (energy, energy)
    assert before == asdict(gun.emitter)
    np.testing.assert_array_equal(tip_energy_samples(gun.emitter, 9), energies)


def test_high_energy_accelerator_cannot_be_skipped_to_use_nonrelativistic_kernel():
    gun = default_state().electron_gun
    with pytest.raises(ValueError, match="nonrelativistic"):
        sample(gun, [gun.exit_plane_z_mm-.001, gun.exit_plane_z_mm], .3,
               (4, 4), np.eye(2)*1e-8, np.zeros(2))


def test_prepare_limits_cancel_and_concurrent_changes_fail_closed(monkeypatch):
    from temsim.optics.electron_gun.electrostatic import FegElectrostaticField
    gun = default_state().electron_gun
    args = (gun, [.1, .100001], .3, (4, 4), np.eye(2)*1e-8, np.zeros(2))
    with pytest.raises(MemoryError):
        sample(*args, maximum_working_bytes=1)
    with pytest.raises(InterruptedError):
        sample(*args, cancelled=lambda: True)
    with pytest.raises(ValueError, match="count"):
        sample(*args, maximum_channels=1)
    original = FegElectrostaticField.potential_v_at_global_positions
    def changes_during_sampling(field, positions):
        gun.extractor.voltage_kv += .001
        return original(field, positions)
    monkeypatch.setattr(FegElectrostaticField, "potential_v_at_global_positions", changes_during_sampling)
    with pytest.raises(ValueError, match="settings changed"):
        sample(*args)


def test_numerical_inputs_are_captured_before_field_sampling(monkeypatch):
    from temsim.optics.electron_gun.electrostatic import FegElectrostaticField
    gun = default_state().electron_gun
    edges, basis, origin = np.array([.1, .100001]), np.eye(2)*1e-8, np.zeros(2)
    expected = sample(gun, edges, .3, (4, 4), basis, origin)
    original = FegElectrostaticField.potential_v_at_global_positions
    def edit_numerics(field, positions):
        edges[:] = [0., 1.]
        basis[:] = np.eye(2)*1e-6
        origin[:] = 2e-6
        return original(field, positions)
    monkeypatch.setattr(FegElectrostaticField, "potential_v_at_global_positions", edit_numerics)
    actual = sample(gun, edges, .3, (4, 4), basis, origin)
    np.testing.assert_array_equal(actual.q_m2, expected.q_m2)
    assert actual.record == expected.record


def test_source_code_change_during_preparation_is_rejected(monkeypatch):
    identities = iter(("before", "after"))
    monkeypatch.setattr("temsim.calculation_manifest.solver_source_identity", lambda: next(identities))
    with pytest.raises(ValueError, match="implementation changed"):
        sample(default_state().electron_gun, [.1, .100001], .3,
               (4, 4), np.eye(2)*1e-8, np.zeros(2))
