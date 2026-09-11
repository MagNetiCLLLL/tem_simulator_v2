from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm

from temsim.optics.column import default_state
from temsim.physics.column_wave import _linear_factor, _propagate_column
from temsim.physics.canonical_action import CanonicalPath
from temsim.physics.specimen_wave_transport import _regrid_mode, _slice_phase
from test_wave_detector_readout import checkpoint
from temsim.physics.wave_reference import AxialWaveReference


def test_constant_hamiltonian_lift_matches_matrix_exponential():
    g = 1.2
    r = np.array(((0, g), (-g, 0)))
    a = np.block([[r, np.eye(2)], [-np.eye(2)*3., r]])
    path = CanonicalPath(1e-3)
    _linear_factor(path, a, .04)
    np.testing.assert_allclose(path.matrix, expm(a*.04), atol=1e-14)


def test_column_vacuum_transports_actual_shape_energy_and_weight():
    state = default_state()
    for collection in (state.lenses, state.stigmators, state.corrector_elements, state.deflectors):
        for component in collection:
            component.enabled = False
    state.apertures = []
    c = checkpoint(two=True)
    references = (AxialWaveReference(2e-8, 1e-22), AxialWaveReference(3e-8, 2e-22))
    c = replace(c, beam=replace(c.beam, modes=tuple(replace(m, axial_reference=r) for m, r in zip(c.beam.modes, references))))
    result = _propagate_column(state, c, 2001., maximum_step_mm=.1)
    assert result.plane_z_mm == 2001.
    assert result.beam.total_weight == pytest.approx(c.beam.total_weight, rel=1e-10)
    assert [m.mode_id for m in result.beam.modes] == [m.mode_id for m in c.beam.modes]
    assert result.record["upstream_digest"] == c.digest
    from temsim.physics.tip_gun_wave import _momentum_velocity
    momentum, velocity = _momentum_velocity(300000.)
    for old, new in zip(c.beam.modes, result.beam.modes):
        assert new.axial_reference.flight_time_s == pytest.approx(old.axial_reference.flight_time_s+.001/velocity, rel=1e-14)
        assert new.axial_reference.longitudinal_action_j_s == pytest.approx(old.axial_reference.longitudinal_action_j_s+.001*momentum, rel=1e-14, abs=1e-40)
    assert not np.array_equal(result.beam.modes[0].plane.amplitude, c.beam.modes[0].plane.amplitude)


def test_regrid_retains_phase_carriers_without_fitting_or_cropping():
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
    mode = checkpoint().beam.modes[0]
    axis = (np.arange(128)-64)*.51e-6
    result, record = _regrid_mode(mode, axis, axis)
    assert result.weight_per_reference_electron == mode.weight_per_reference_electron
    assert record["interpolation_difference"] < 1e-4
    assert result.plane.probability == pytest.approx(1., abs=1e-12)
    original_phase = np.angle(mode.plane.full_amplitude(float(wavelength_m(300000)))[32, 32])
    new_phase = np.angle(result.plane.full_amplitude(float(wavelength_m(300000)))[64, 64])
    assert new_phase == pytest.approx(original_phase, abs=1e-12)
    with pytest.raises(ValueError, match="crop"):
        _regrid_mode(mode, axis/2, axis/2)


def test_specimen_phase_is_complex_and_not_an_intensity_change():
    mode = checkpoint().beam.modes[0]
    axis = (np.arange(64)-32)*1e-6
    potential = np.ones((64, 64))*3.
    result = _slice_phase(mode, potential, axis, axis, .02, .5)
    np.testing.assert_allclose(result.plane.amplitude, mode.plane.amplitude*np.exp(.03j), atol=1e-14)
    assert result.weight_per_reference_electron == mode.weight_per_reference_electron
    assert result.plane.probability == pytest.approx(1.)
