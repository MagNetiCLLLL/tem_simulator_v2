"""Independent paraxial Hamiltonian checks, not full microscope qualification."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import e, m_e
from scipy.integrate import quad
from scipy.linalg import expm

from temsim.physics.canonical_action import CanonicalPath
from temsim.physics.closed_gun_field import ClosedGunField
from temsim.physics.column_wave import (
    _column_transports, _component_events, _electric_coefficients, _linear_factor,
    _propagate_column, _slice_prepared,
)
from temsim.physics.tip_gun_wave import _momentum_velocity
from temsim.physics.wave_field_admission import sample_wave_electric
from temsim.physics.wave_reference import AxialWaveReference
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.wave_grid import WaveGridNumerics
from temsim.optics.electron_gun.tip_coherence import wavelength_m
from test_tip_wave_pipeline import _quiet_prepared_column
from test_wave_detector_readout import checkpoint


def prepared(*, gradient=2e6, radial=0., step=.05, magnetic=0., normal=0., skew=0., bx=0., by=0.):
    plan, radii, stops, owners = _quiet_prepared_column(None, 2000., 2001., step)
    r = np.array((0., .01, .02))
    z = np.array((2., 2.0005, 2.001))
    plan.electric_field = ClosedGunField({}, r, z,
        gradient*(z[None, :]-z[0])+radial*r[:, None]**2)
    plan.reference_momentum_kg_m_s = float(_momentum_velocity(300000.)[0])
    plan.midpoint_magnetic_t = np.full(len(plan.step_m), magnetic)
    plan.midpoint_sx_m2 = np.full(len(plan.step_m), normal)
    plan.midpoint_sy_m2 = np.full(len(plan.step_m), -normal)
    plan.midpoint_sxy_m2 = np.full(len(plan.step_m), skew)
    plan.dipole_bx_t = np.full(3*len(plan.step_m), bx)
    plan.dipole_by_t = np.full(3*len(plan.step_m), by)
    return plan, radii, stops, owners


def test_shared_axis_cell_derivatives_read_the_actual_scalar_field():
    plan = prepared(radial=2e7)[0]
    phi, gradient, hessian = sample_wave_electric(plan.electric_field, np.array((2000., 2000.5, 2001.)))
    np.testing.assert_allclose(phi, (0., 1000., 2000.), atol=1e-9)
    np.testing.assert_array_equal(gradient, np.zeros((3, 2)))
    np.testing.assert_allclose(hessian, np.broadcast_to(np.eye(2)*4e7, (3, 2, 2)), rtol=1e-12)


def test_unknown_electric_provider_is_not_silently_linearized():
    with pytest.raises(ValueError, match="resolved wave Hamiltonian"):
        sample_wave_electric(SimpleNamespace(), np.array((2000.,)))


def test_accelerating_map_retains_fixed_canonical_momentum_and_converges():
    p0 = float(_momentum_velocity(300000.)[0])
    exact = quad(lambda z: p0/_momentum_velocity(300000.+2e6*z)[0], 0., .001, epsabs=1e-20)[0]
    errors = []
    for step in (.2, .1, .05):
        combined = CanonicalPath(.001)
        for path in _column_transports(prepared(step=step)[0], 300.):
            combined.append(path.matrix, path.offset, action_m=path.action_m)
        np.testing.assert_allclose(combined.matrix[2:, 2:], np.eye(2), atol=1e-14)
        errors.append(abs(combined.matrix[0, 2]-exact))
    assert errors[1] < errors[0]/3.8
    assert errors[2] < errors[1]/3.8


def test_uniform_finite_dipole_keeps_displacement_and_absolute_weyl_phase():
    generator = np.zeros((4, 4))
    generator[:2, 2:] = np.eye(2)
    force = np.array((0., 0., .07, -.02))
    distance = .03
    path = CanonicalPath(.001)
    _linear_factor(path, generator, distance, force=force)
    np.testing.assert_allclose(path.offset[:2], force[2:]*distance**2/2, atol=1e-18)
    np.testing.assert_allclose(path.offset[2:], force[2:]*distance, atol=1e-18)
    assert path.action_m == pytest.approx(float(force[2:]@force[2:])*distance**3/12, rel=1e-14)
    split = CanonicalPath(.001)
    for _ in range(4):
        _linear_factor(split, generator, distance/4, force=force)
    np.testing.assert_allclose(split.offset, path.offset, atol=1e-18)
    assert split.action_m == pytest.approx(path.action_m, rel=1e-13)


def test_axial_lens_normal_skew_stigmator_and_electric_curvature_share_generator():
    radial, bz, normal, skew = 2e7, .015, .8, -.4
    plan = prepared(gradient=0., radial=radial, magnetic=bz, normal=normal, skew=skew, step=1.)[0]
    p, v = _momentum_velocity(300000.)
    g = -e*bz/(2*p)
    rotation = np.array(((0., g), (-g, 0.)))
    stiffness = np.array(((normal+g*g, skew), (skew, -normal+g*g)))
    electric = np.eye(2)*e*2*radial/(v*p)
    exact = expm(np.block([[rotation, np.eye(2)], [electric-stiffness, rotation]])*.001)
    actual = next(_column_transports(plan, 300.))
    np.testing.assert_allclose(actual.matrix, exact, rtol=1e-13, atol=1e-15)


def test_actual_finite_b_field_is_used_and_not_center_kicked():
    by = .001
    plan = prepared(gradient=0., by=by, step=1.)[0]
    p, _ = _momentum_velocity(300000.)
    force = e*by/p
    actual = next(_column_transports(plan, 300.))
    assert actual.offset[0] == pytest.approx(force*.001**2/2, rel=1e-13)
    assert actual.offset[2] == pytest.approx(force*.001, rel=1e-13)
    assert actual.action_m == pytest.approx(force**2*.001**3/12, rel=1e-13)


def test_component_events_include_a_coil_with_its_centre_outside_the_segment():
    coil = SimpleNamespace(key="finite_deflector", enabled=True, effective_thickness_mm=1.,
        kick_events=lambda **_kw: ((0., .001, -.002),))
    state = SimpleNamespace(deflectors=(coil,), stigmators=(), corrector_elements=(),
        beam_voltage_kv=300., simulation_time_s=0.)
    events, owners = _component_events(state, .2, .4)
    assert events == [(0., .001, -.002)]
    assert owners[0]["finite_field"] is True
    assert owners[0]["z_mm"] < .2


def test_wave_acceleration_preserves_flux_and_converts_carrier_units():
    item = prepared()
    initial = checkpoint()
    mode = replace(initial.beam.modes[0], axial_reference=AxialWaveReference(2e-8, 1e-22))
    initial = replace(initial, beam=replace(initial.beam, modes=(mode,)))
    result = _propagate_column(SimpleNamespace(), initial, 2001., _prepared=item)
    output = result.beam.modes[0]
    assert output.energy_kev == pytest.approx(302., abs=1e-10)
    assert result.beam.total_weight == pytest.approx(initial.beam.total_weight, rel=1e-12)
    plan = item[0]
    nodes, phi, _, _ = _electric_coefficients(plan)
    p0, _ = _momentum_velocity(300000.)
    pend, _ = _momentum_velocity(302000.)
    mapped = CanonicalPath(.001)
    for path in _column_transports(plan, 300.):
        mapped.append(path.matrix)
    expected_tilt = (mapped.matrix[2:, :2]@mode.plane.origin_m + mapped.matrix[2:, 2:]@mode.plane.tilt_rad)*p0/pend
    np.testing.assert_allclose(output.plane.tilt_rad, expected_tilt, atol=1e-15)
    expected_time = quad(lambda z: 1/_momentum_velocity(300000.+2e6*z)[1], 0., .001, epsabs=1e-23)[0]
    expected_action = quad(lambda z: _momentum_velocity(300000.+2e6*z)[0], 0., .001, epsabs=1e-30)[0]
    assert output.axial_reference.flight_time_s-2e-8 == pytest.approx(expected_time, rel=2e-9)
    assert output.axial_reference.longitudinal_action_j_s-1e-22 == pytest.approx(expected_action, rel=2e-9)


def test_segmented_plan_preserves_electric_reference_and_finite_field_support():
    item = prepared(by=.001, step=.1)
    item[3].append({"component": "finite", "z_mm": 2000., "finite_field": True,
                    "effective_thickness_mm": 1.})
    piece = _slice_prepared(item, 2, 4)
    assert piece[0].electric_field is item[0].electric_field
    assert piece[0].reference_momentum_kg_m_s == item[0].reference_momentum_kg_m_s
    np.testing.assert_array_equal(piece[0].dipole_by_t, item[0].dipole_by_t[6:12])
    assert len(piece[3]) == 1  # centre is upstream but the actual coil still overlaps


def _astigmatic_focus_checkpoint(pixels):
    """Independent complex Gaussian on a fixed 192 nm period, no fitted source."""
    period, sigma = 192e-9, 10e-9
    axis = (np.arange(pixels)-pixels//2)*period/pixels
    xx, yy = np.meshgrid(axis, axis)
    amplitude = np.exp(-(xx*xx+yy*yy)/(2*sigma*sigma)).astype(complex)
    amplitude /= np.linalg.norm(amplitude)
    wave = PlaneWave(amplitude, np.eye(2)*period/pixels, np.zeros(2),
        curvature_m1=np.diag((-10000., 0.)), tilt_rad=np.zeros(2))
    initial = checkpoint()
    mode = replace(initial.beam.modes[0], plane=wave)
    return replace(initial, beam=replace(initial.beam, modes=(mode,)))


def test_linear_focus_refines_executed_complex_wave_and_matches_independent_analytic_field():
    # A 0.1 mm drift reaches the x focus while y retains its phase chirp.
    # The coarse y chirp crosses Nyquist, so the sampled Collins chart must
    # refine before applying its operator. The separate fine input is sampled
    # from the same analytic complex Gaussian, not the coarse implementation.
    prepared = _quiet_prepared_column(None, 2000., 2000.1, .1)
    low = _astigmatic_focus_checkpoint(64)
    original = low.beam.modes[0].plane.amplitude.copy()
    automatic = _propagate_column(SimpleNamespace(), low, 2000.1, _prepared=prepared,
        grid_numerics=WaveGridNumerics(maximum_pixels=512))
    fine_input = _astigmatic_focus_checkpoint(512)
    fine = _propagate_column(SimpleNamespace(), fine_input, 2000.1,
        _prepared=prepared, grid_numerics=WaveGridNumerics(automatic_refinement=False, maximum_pixels=512))
    assert automatic.beam.modes[0].plane.amplitude.shape == (512, 512)
    lam, sigma, distance = float(wavelength_m(low.beam.modes[0].energy_kev*1000)), 10e-9, .0001
    precision = np.array((-10000., 0.))+1j*lam/(2*np.pi*sigma*sigma)
    factor = 1+distance*precision
    # Both distinct output lattices are checked against the independently
    # derived complex Gaussian drift, including scalar phase and pixel area.
    # Increasing the INPUT grid alone does not refine a natural Collins
    # output cell; the low-grid execution also needs output quadrature.
    for result, initial, quadrature_pixels in ((automatic, low, 128), (fine, fine_input, 512)):
        output = result.beam.modes[0]
        source = initial.beam.modes[0].plane
        xy = output.plane.coordinates_m()
        area_ratio = abs(np.linalg.det(output.plane.basis_m)/np.linalg.det(source.basis_m))
        coefficient = (source.amplitude[source.amplitude.shape[0]//2, source.amplitude.shape[1]//2]
            /np.sqrt(factor[0]*factor[1])*np.sqrt(area_ratio))
        # The discrete Collins quadrature represents its COMPLETE Fourier
        # period. Poisson images of the chirped input Gaussian therefore
        # contribute at the period edges; they have their own phase and are
        # not a missing tail to discard or fix by a tolerance change. The
        # auto path first resolves its input at 128 samples; the independent
        # path starts with 512. Both retain the same 192 nm source period.
        period = lam*distance/(192e-9/quadrature_pixels)
        chirp_precision = precision+1/distance
        expected = np.zeros(output.plane.amplitude.shape, dtype=complex)
        for x_image in (-1, 0, 1):
            for y_image in (-1, 0, 1):
                phase = (xy[0]**2+xy[1]**2)/distance
                phase = phase-(xy[0]-x_image*period)**2/(distance**2*chirp_precision[0])
                phase = phase-(xy[1]-y_image*period)**2/(distance**2*chirp_precision[1])
                expected += coefficient*np.exp(1j*np.pi/lam*phase)
        np.testing.assert_allclose(output.plane.full_amplitude(lam), expected, rtol=1e-9, atol=1e-12)
        # These exact cell-area amplitudes independently supply the numerical
        # window probability. Both windows cover >8.5 Gaussian intensity RMS;
        # the full-period path also includes all Poisson images above. Thus
        # domain truncation/quadrature cannot explain a >1e-12 flux change.
        window_probability = float(np.vdot(expected, expected).real)
        assert window_probability == pytest.approx(1., rel=1e-12)
        assert output.weight_per_reference_electron/initial.beam.total_weight == pytest.approx(
            window_probability, rel=1e-12)
    np.testing.assert_array_equal(low.beam.modes[0].plane.amplitude, original)
    rows = automatic.record["modes"][0]["grid_refinements"]
    assert rows and rows[0]["operator"] == "distributed_linear"
    assert rows[0]["mode_id"] == low.beam.modes[0].mode_id
    assert rows[0]["z_mm"] == pytest.approx(2000.1)
    assert any(row["method"].startswith("unitary complex Fourier embedding") for row in rows)
    assert any(row["method"].startswith("zero-padded spatial Collins") for row in rows)


@pytest.mark.parametrize("numerics, message", [
    (WaveGridNumerics(maximum_pixels=64), "Wave refinement budget exceeded"),
    (WaveGridNumerics(automatic_refinement=False, maximum_pixels=128), "phase is undersampled"),
])
def test_linear_focus_keeps_unapplied_state_when_refinement_budget_is_unavailable(numerics, message):
    initial = _astigmatic_focus_checkpoint(64)
    before = initial.beam.modes[0].plane.amplitude.copy()
    prepared = _quiet_prepared_column(None, 2000., 2000.1, .1)
    with pytest.raises(ValueError, match=message):
        _propagate_column(SimpleNamespace(), initial, 2000.1, _prepared=prepared, grid_numerics=numerics)
    np.testing.assert_array_equal(initial.beam.modes[0].plane.amplitude, before)
