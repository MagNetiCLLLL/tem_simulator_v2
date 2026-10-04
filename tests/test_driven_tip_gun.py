"""Physical Tip boundary continuation, including the executed energy origin."""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics.tip_gun_wave import _energy_transport


def test_accelerated_prefix_energy_is_not_added_twice_in_gun_continuation():
    # Translation changes absolute potential, but the same entered *local*
    # kinetic energy and remaining field must produce the same transport.
    count, length_mm, offset_mm = 128, .1, .004
    gain_per_m, local_energy = 2e5, 2.7
    field = SimpleNamespace(potential_v_at_global_positions=lambda p: gain_per_m*p[:, 2])
    gun = SimpleNamespace(electric_field=field)
    rows = []
    for start in (0., offset_mm):
        z = np.linspace(start, start+length_mm, count+1)
        phi = gain_per_m*(z[1:]+z[:-1])*.5e-3
        vector, tensor = np.zeros((count, 2)), np.zeros((count, 2, 2))
        _, row = _energy_transport(gun, z, {float(z[-1])},
            (phi, vector, tensor, vector, tensor), local_energy, lambda: False)
        rows.append(row)
        assert row["exit_axial_energy_ev"] == pytest.approx(local_energy+gain_per_m*length_mm*1e-3)
        assert row["executed_start_z_mm"] == start
    np.testing.assert_allclose(rows[0]["canonical_map"], rows[1]["canonical_map"], rtol=1e-13, atol=1e-15)
    for name in ("reference_flight_time_s", "reference_longitudinal_action_j_s", "momentum_ratio_tip_to_exit"):
        assert rows[0][name] == pytest.approx(rows[1][name], rel=1e-13, abs=0)


def test_driven_gun_requires_captured_instrument_instead_of_assuming_missing_fields():
    from temsim.optics.electron_gun.field_emission import FieldEmissionGun
    from temsim.optics.electron_gun.tip_coherence import TipCoherence
    from temsim.physics.tip_gun_wave import build_tip_gun_checkpoint
    gun = FieldEmissionGun()
    gun.emitter.coherence = TipCoherence(boundary_model="driven_gaussian_schell")
    with pytest.raises(ValueError, match="captured instrument"):
        build_tip_gun_checkpoint(gun)


def test_near_tip_auxiliary_arrays_are_owned_immutable_and_in_checkpoint_identity():
    from dataclasses import replace
    from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE
    from temsim.physics.multiplane_wave import PlaneWave
    from temsim.physics.tip_gun_wave import TipGunCheckpoint
    from temsim.physics.wave_flux import BeamState, WaveMode
    plane = PlaneWave(np.ones((4, 4), complex)/4, np.eye(2)*1e-9, np.zeros(2))
    beam = BeamState((WaveMode(plane, 1., TIP_REFERENCE, "mode", 300.),), TIP_REFERENCE)
    supplied = np.array([1+2j, 3+4j])
    original = TipGunCheckpoint(beam, 450., 1e-5, {}, {"near_tip_m0_spectral_derivative": supplied})
    digest = original.digest
    supplied[0] = 0
    assert original.auxiliary_arrays["near_tip_m0_spectral_derivative"][0] == 1+2j
    assert not original.auxiliary_arrays["near_tip_m0_spectral_derivative"].flags.writeable
    assert original.digest == digest
    assert replace(original, auxiliary_arrays={"near_tip_m0_spectral_derivative": supplied}).digest != digest
