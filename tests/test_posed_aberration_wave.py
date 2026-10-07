"""Small-inclination canonical Cs screen; independent limits and real GPU."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.constants import e, hbar
from scipy.spatial.transform import Rotation

from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.multipole_wave import apply_multipole_phase
from temsim.physics.posed_aberrations import FrozenLensAberrationKick
from temsim.physics.posed_aberration_wave import apply_posed_spherical
from temsim.physics.tip_gun_wave import _momentum_velocity


P = float(_momentum_velocity(300000.)[0])
WAVELENGTH = 2*np.pi*hbar/P
STRENGTH = 1.95329445e7


def screen(angle=.001, strength=STRENGTH):
    rotation = Rotation.from_rotvec((0., angle, 0.)).as_matrix()
    return FrozenLensAberrationKick("test", 0., strength,
        CoordinateRegistration((0., 0., .5), tuple(map(tuple, rotation))))


def packet(*, broad=False):
    count, pitch, sigma = (256, .25e-6, 5e-6) if broad else (64, 5e-9, 25e-9)
    axis = (np.arange(count)-count//2)*pitch
    xx, yy = np.meshgrid(axis, axis)
    amplitude = np.exp(-(xx*xx+yy*yy)/(2*sigma*sigma)).astype(complex)
    amplitude /= np.linalg.norm(amplitude)
    return PlaneWave(amplitude, np.eye(2)*pitch, np.array((30e-6, 0.)),
                     np.zeros((2, 2)), np.array((.02 if broad else 0., 0.)))


def apply(wave, kick, **kwargs):
    return apply_posed_spherical(wave, WAVELENGTH, kick,
        momentum_kg_m_s=P, reference_momentum_kg_m_s=P, **kwargs)


def test_zero_inclination_reproduces_existing_quartic_operator_without_phase_fitting():
    wave, kick = packet(), screen(0.)
    actual, receipt = apply(wave, kick)
    expected = apply_multipole_phase(wave, WAVELENGTH, spherical_m3=STRENGTH)
    for name in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert receipt["split_steps"] == 0


def test_zero_strength_is_identity_even_outside_nonzero_screen_inclination_domain():
    wave = packet()
    actual, _ = apply(wave, screen(.2, 0.))
    assert actual is wave
    coordinates = (np.array((1e-6,)), np.array((.2,)), np.zeros(1), np.zeros(1))
    rays = screen(.2, 0.).apply_canonical(*coordinates, momentum_kg_m_s=P)
    for before, after in zip(coordinates, rays):
        np.testing.assert_array_equal(after, before)


def test_tilt_changes_nonconstant_relative_phase_with_signed_linear_response():
    from temsim.physics.wave_grid import refine_plane_wave
    # Preserve the original source and physical period, resolving the old
    # quartic operator with Fourier embedding rather than changing the beam.
    wave = refine_plane_wave(packet(broad=True), (512, 512))
    aligned, _ = apply(wave, screen(0.))
    phases = []
    for angle in (.001, -.001, .002):
        result, receipt = apply(wave, screen(angle))
        relative = np.angle(result.amplitude*np.conj(aligned.amplitude))
        difference = relative[256, 296]-relative[256, 216]
        phases.append(difference)
        assert result.probability == pytest.approx(wave.probability, rel=2e-11)
        assert receipt["split_error_estimate"] < 5e-11
    assert abs(phases[0]) > 1e-4  # Not a constant/global phase change.
    assert phases[1] == pytest.approx(-phases[0], rel=2e-3)
    assert phases[2] == pytest.approx(2*phases[0], rel=2e-3)


def test_canonical_ray_has_oblique_position_shift_and_preserves_stopped_nan():
    kick = screen()
    x = np.array((30e-6, np.nan))
    zero = np.zeros(2)
    actual = kick.apply_canonical(x, zero, zero, zero, momentum_kg_m_s=np.full(2, P))
    expected_shift = -np.tan(.001)*STRENGTH*x[0]**4
    assert actual[0][0]-x[0] == pytest.approx(expected_shift, abs=2e-20, rel=2e-5)
    assert actual[1][0] == pytest.approx(-STRENGTH*x[0]**3, rel=2e-5)
    assert np.isnan(actual[0][1])


def test_energy_reference_change_preserves_physical_phase_and_ray_motion():
    wave, kick = packet(), screen()
    ordinary, _ = apply(wave, kick, aligned_bz_t=.05)
    ratio = .8
    changed, _ = apply_posed_spherical(replace(wave, tilt_rad=wave.tilt_rad/ratio),
        WAVELENGTH/ratio, kick, momentum_kg_m_s=P, reference_momentum_kg_m_s=P*ratio,
        aligned_bz_t=.05)
    np.testing.assert_allclose(ordinary.full_amplitude(WAVELENGTH),
        changed.full_amplitude(WAVELENGTH/ratio), rtol=2e-11, atol=2e-13)
    x, zero = np.array((30e-6,)), np.zeros(1)
    original = kick.apply_canonical(x, zero, zero, zero, momentum_kg_m_s=P, aligned_bz_t=.05)
    redefined = kick.apply_canonical(x, zero, zero, zero, momentum_kg_m_s=P,
        reference_momentum_kg_m_s=P*ratio, aligned_bz_t=.05)
    for first, second in zip(original, redefined):
        np.testing.assert_allclose(first, second, rtol=1e-10, atol=2e-18)


def test_unresolved_geometry_budget_and_inclination_fail_before_changing_input():
    wave = packet()
    before = wave.amplitude.copy()
    with pytest.raises(ValueError, match="10 mrad"):
        apply(wave, screen(.02))
    with pytest.raises(ValueError, match="phase remainder"):
        apply(wave, screen(.005), maximum_obliquity_phase_error_rad=1e-12)
    with pytest.raises(InterruptedError):
        apply(wave, screen(), cancelled=lambda: True)
    np.testing.assert_array_equal(wave.amplitude, before)


def test_small_captured_electric_tail_is_budgeted_and_large_variation_rejected():
    wave, kick = packet(), screen()
    ordinary, _ = apply(wave, kick)
    tail, receipt = apply(wave, kick,
        electrostatic_gradient_v_m=(0., 0., -5.82e-4),
        electrostatic_hessian_v_m2=np.diag((-1.09, -1.09)))
    # The field's column transport is executed elsewhere. Here it changes
    # the declared error budget, never adds a duplicate electrostatic step.
    np.testing.assert_array_equal(tail.amplitude, ordinary.amplitude)
    assert 0 < receipt["electric_phase_remainder_bound_rad"] < 1e-10
    assert 0 < receipt["electric_relative_momentum_variation_bound"] < 1e-12
    assert receipt["electric_potential_variation_bound_v"] > 0
    with pytest.raises(ValueError, match="phase remainder"):
        apply(wave, kick, electrostatic_gradient_v_m=(1e8, 0., 0.))


def test_remote_multipole_has_no_cs_effect_but_crossed_support_edge_is_rejected():
    from temsim.physics.posed_column_fields import FrozenPosedMultipole
    field = FrozenPosedMultipole("remote", "remote", CoordinateRegistration(),
        ((1, 0, 1e6),), "uniform", 0., .001, (-1., 1.), .003, P)
    wave, kick = packet(), screen()
    expected, _ = apply(wave, kick)
    actual, _ = apply(wave, kick, fields=(field,))
    np.testing.assert_array_equal(actual.amplitude, expected.amplitude)
    touching = replace(field, registration=CoordinateRegistration((0., 0., .501)))
    with pytest.raises(ValueError, match="support edge"):
        apply(wave, kick, fields=(touching,))


def test_real_gpu_tilted_screen_matches_cpu_absolute_complex_field():
    from test_wave_device import _cuda
    cp = _cuda()
    wave, kick = packet(), screen()
    cpu, cpu_record = apply(wave, kick, aligned_bz_t=.05)
    gpu, gpu_record = apply(replace(wave, amplitude=cp.asarray(wave.amplitude)), kick, aligned_bz_t=.05)
    np.testing.assert_allclose(cp.asnumpy(gpu.amplitude), cpu.amplitude, rtol=2e-10, atol=2e-13)
    assert cpu_record["compute_backend"] == "numpy" and gpu_record["compute_backend"] == "cupy"
    assert gpu.probability == pytest.approx(wave.probability, rel=2e-11)
