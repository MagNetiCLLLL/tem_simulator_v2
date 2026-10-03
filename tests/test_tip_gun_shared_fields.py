"""Bounded actual gun transport and independent component-preservation checks.

The source is explicitly declared at the tip; no fitted exit or specimen wave
is supplied. These checks cover a quadratic development operator, not complete
microscope imaging or the accuracy of its chosen physical tip parameters.
"""
import numpy as np
import pytest
from dataclasses import replace
from types import SimpleNamespace
from scipy.constants import c, e, m_e

from temsim.optics.column import default_state
from temsim.optics.electron_gun.tip_coherence import TipCoherence, TipWaveNumerics
from temsim.physics.tip_gun_wave import (
    GunWaveNumerics, _axial_grid, _mask_plane, _column_magnetic_coefficients, build_tip_gun_checkpoint,
    _refine_energy_grid, _energy_transport,
)
from temsim.physics.multiplane_wave import PlaneWave


def _state():
    state = default_state()
    emitter = state.electron_gun.emitter
    emitter.coherence = TipCoherence()
    emitter.virtual_source_fwhm_nm = 50.
    emitter.energy_spread_fwhm_ev = 0.
    return state


def test_exact_extra_aperture_and_field_boundaries_survive_gun_grid():
    gun = _state().electron_gun
    grid, masks = _axial_grid(gun, GunWaveNumerics(field_step_mm=1., bore_step_mm=10.),
                             exact_z_mm=(37.123456789,), extra_mask_planes=(38.123456789,))
    assert 37.123456789 in grid
    assert 38.123456789 in grid
    assert 38.123456789 in masks
    assert np.max(np.diff(grid)) <= 1.+1e-12


def test_column_aperture_inside_gun_intercepts_without_a_second_source():
    from temsim.optics.model import Aperture
    state = _state()
    # Independent geometric fixture: installed production component positions
    # are TOML-owned and must not be overwritten merely to exercise clipping.
    aperture = Aperture("Zero opening", "test_zero_opening", z_mm=100., radius_mm=0., enabled=True)
    wave = PlaneWave(np.ones((32, 32), complex)/32, np.eye(2)*1e-9, np.zeros(2))
    result, rows = _mask_plane(state.electron_gun, wave, 100., .4, (aperture,))
    assert result.probability == 0.
    row = next(row for row in rows if row["component"] == aperture.key)
    assert row["lost_probability"] == pytest.approx(.4)
    assert row["kind"] == "column_aperture"


def test_overlapping_column_stigmator_and_finite_deflector_have_signed_physical_terms():
    from temsim.optics.model import Stigmator, DeflectorPair
    from temsim.physics.core import electron
    state = _state()
    state.lenses = []
    state.corrector_elements = []
    state.stigmators = [Stigmator("Test stigmator", "test_quad", 100.,
                                 max_strength_m2=80., strength_x_percent=25., strength_y_percent=50.)]
    state.deflectors = [DeflectorPair("Test deflector", "test_dipole", 100., 110.,
                                     upper_x_mrad=.2, upper_y_mrad=-.1, thickness_mm=4.)]
    z = np.array((100.,))
    axial, force, hessian = _column_magnetic_coefficients(state, z)
    momentum = electron(state)[1]
    qx, qy, qxy = state.stigmators[0].quadrupole_tensor_m2(z)
    expected = -momentum/e*np.array(((qx[0], qxy[0]), (qxy[0], qy[0])))
    np.testing.assert_allclose(hessian[0], expected, rtol=1e-13, atol=1e-15)
    # Electron force from the captured B reproduces both signed deflection
    # commands, scaled with the physical nominal p and finite coil length.
    np.testing.assert_allclose(force[0]*e/momentum,
                               (.2e-3/.004, -.1e-3/.004), rtol=1e-13, atol=1e-15)
    assert axial[0] == 0.


def test_shared_gun_executes_captured_solved_electric_field_and_real_components():
    """This was rejected by the post-gun guard before any specimen calculation."""
    from temsim.physics.instrument_electric import capture_instrument_electric_field
    state = _state()
    gun = state.electron_gun
    source = TipWaveNumerics(grid_pixels=128, energy_samples=1)
    numeric = GunWaveNumerics(field_step_mm=.2, bore_step_mm=10.)
    result = build_tip_gun_checkpoint(gun, source_numerics=source,
                                     numerics=numeric, _column_state=state, use_cache=True)
    field = capture_instrument_electric_field(state)
    points = np.array(((0., 0., 0.), (0., 0., gun.exit_plane_z_mm*1e-3)))
    potentials = field.potential_rise_v_at_global_positions(points)
    expected_energy = gun.emitter.emission_energy_ev+potentials[1]-potentials[0]
    assert result.beam.modes[0].energy_kev*1000 == pytest.approx(expected_energy, rel=1e-13)
    assert result.record["shared_column"]["electric_field_identity"] == field.numerical_identity
    assert set(result.record["physical_components"]) == {part.key for part in gun.components}
    assert 0 < result.beam.total_weight <= 1.+1e-12
    assert np.all(np.isfinite(result.beam.modes[0].plane.amplitude))
    assert result.beam.modes[0].axial_reference.longitudinal_action_j_s > 0
    assert result.record["tip_emission"]["plane_z_mm"] == 0.
    repeated = build_tip_gun_checkpoint(gun, source_numerics=source,
        numerics=numeric, _column_state=state, use_cache=True)
    assert repeated is result
    refined = build_tip_gun_checkpoint(gun, source_numerics=source,
        numerics=replace(numeric, maximum_fractional_energy_change=.025),
        _column_state=state, use_cache=True)
    assert refined.record["dependency_digest"] != result.record["dependency_digest"]
    assert refined.record["field_steps"] > result.record["field_steps"]
    assert refined.record["shared_column"]["electric_field_identity"] == field.numerical_identity


class _LinearElectric:
    """Independent uniform electric field, rising kinetic energy at the tip."""
    length_m = .05e-3
    rise_v = 32.684617815135046

    def potential_v_at_global_positions(self, positions):
        return 7.+self.rise_v*np.asarray(positions)[:, 2]/self.length_m


def test_energy_refinement_converges_to_exact_linear_potential_drift_and_flight_time():
    electric = _LinearElectric()
    gun = SimpleNamespace(electric_field=electric)
    initial = .3
    # Exact relativistic integrals for K(z)=K0+DeltaK*z/L:
    # int dz/p = c*L/DeltaK * [asinh(p/(m*c))], and
    # int dz/v = L/DeltaK * [p], since dp/dK=1/v.
    energies = np.array((initial, initial+electric.rise_v))*e
    momentum = np.sqrt(energies*(energies+2*m_e*c*c))/c
    exact_drift = (momentum[0]*c*electric.length_m/(energies[1]-energies[0])
        * np.diff(np.arcsinh(momentum/(m_e*c)))[0])
    exact_time = electric.length_m*np.diff(momentum)[0]/np.diff(energies)[0]
    errors = []
    for fraction in (.1, .05, .025):
        numeric = GunWaveNumerics(maximum_fractional_energy_change=fraction)
        z = _refine_energy_grid(np.array((0., electric.length_m*1e3)), electric, initial, numeric)
        points = np.zeros((len(z)-1, 3))
        points[:, 2] = .5*(z[1:]+z[:-1])*1e-3
        phi = electric.potential_v_at_global_positions(points)
        vector, tensor = np.zeros((len(phi), 2)), np.zeros((len(phi), 2, 2))
        _, record = _energy_transport(gun, z, {float(z[-1])},
            (phi, vector, tensor, vector, tensor), initial, lambda: False)
        drift_error = abs(record["canonical_map"][0][2]/exact_drift-1)
        time_error = abs(record["reference_flight_time_s"]/exact_time-1)
        # Independent midpoint remainder bound for 1/sqrt(K), with the
        # relativistic correction included in the exact reference above.
        assert max(drift_error, time_error) < 3*fraction*fraction/32
        errors.append(max(drift_error, time_error))
        kinetic_nodes = initial+electric.rise_v*z/(electric.length_m*1e3)
        assert np.max(abs(np.diff(kinetic_nodes))/np.minimum(kinetic_nodes[:-1], kinetic_nodes[1:])) <= fraction*(1+1e-12)
        assert z[0] == 0. and z[-1] == electric.length_m*1e3
    assert errors[0] > 2.5*errors[1] > 6.25*errors[2]


@pytest.mark.parametrize("value", [0., -1., 1.01, np.inf, np.nan, True])
def test_energy_refinement_tolerance_is_explicit_and_bounded(value):
    with pytest.raises(ValueError, match="maximum_fractional_energy_change"):
        GunWaveNumerics(maximum_fractional_energy_change=value).validate()


def test_energy_refinement_rejects_turning_nodes_and_midpoint_barrier():
    grid = np.array((0., .05))
    falling = SimpleNamespace(potential_v_at_global_positions=lambda p: -20*np.asarray(p)[:, 2]/.05e-3)
    with pytest.raises(ValueError, match="turning/forbidden"):
        _refine_energy_grid(grid, falling, .3, GunWaveNumerics())
    # Positive endpoints must not hide a forbidden interior sampled by the
    # actual electric provider. No linear endpoint fit replaces that query.
    barrier = SimpleNamespace(potential_v_at_global_positions=lambda p:
        -4*(np.asarray(p)[:, 2]/.05e-3)*(1-np.asarray(p)[:, 2]/.05e-3))
    with pytest.raises(ValueError, match="turning/forbidden"):
        _refine_energy_grid(grid, barrier, .3, GunWaveNumerics())


def test_energy_refinement_preserves_nodes_and_respects_step_budget_and_cancellation():
    electric = _LinearElectric()
    grid = np.array((0., .0123456789, .05))
    refined = _refine_energy_grid(grid, electric, .3, GunWaveNumerics())
    assert all(node in refined for node in grid)
    with pytest.raises(ValueError, match="energy refinement exceeds max_steps"):
        _refine_energy_grid(grid, electric, .3, GunWaveNumerics(max_steps=3))
    with pytest.raises(InterruptedError, match="energy refinement cancelled"):
        _refine_energy_grid(grid, electric, .3, GunWaveNumerics(), cancelled=lambda: True)
