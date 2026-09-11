"""T2-13..17: isolated numerical fixtures, not independent production sources."""
from dataclasses import replace
import math

import numpy as np
import pytest

from temsim.physics.canonical_phase import (
    CANONICAL_BASIS, canonical_basis_matrix, mechanical_map_to_canonical,
    validate_canonical_map,
)
from temsim.physics.multiplane_wave import PlaneWave, propagate_plane_wave
from temsim.physics.wave_flux import BeamState, WaveMode


def wave(pixels=128):
    x = (np.arange(pixels)-pixels//2)*5e-10
    a = np.exp(-(x[:, None]**2+x[None, :]**2)/(4*(5e-9)**2)).astype(complex)
    a /= np.linalg.norm(a)
    return PlaneWave(a, np.eye(2)*5e-10, np.zeros(2))


def drift(z):
    m = np.eye(4)
    m[:2, 2:] = np.eye(2)*z
    return m


def test_t213_tilt_materializes_the_full_fourier_centroid():
    w = replace(wave(), tilt_rad=np.array([2e-4, -1e-4]))
    lam = 2e-12
    full = w.full_amplitude(lam)
    spectrum = np.abs(np.fft.fft2(full))**2
    f = np.fft.fftfreq(128, d=5e-10)
    actual = lam*np.array([np.sum(spectrum*f[None, :]), np.sum(spectrum*f[:, None])])/spectrum.sum()
    np.testing.assert_allclose(actual, w.tilt_rad, atol=2e-10, rtol=1e-6)
    assert not np.allclose(full, w.amplitude)


def test_t214_curvature_conversion_preserves_phase_and_rejects_aliasing():
    w = replace(wave(), curvature_m1=np.eye(2)*2000)
    xy = w.coordinates_m()
    expected = w.amplitude*np.exp(1j*np.pi*2000*np.sum(xy**2, axis=0)/2e-12)
    np.testing.assert_allclose(w.full_amplitude(2e-12), expected, rtol=1e-12, atol=1e-14)
    with pytest.raises(ValueError, match="undersampled"):
        replace(w, curvature_m1=np.eye(2)*2e8).full_amplitude(2e-12)


def test_t215_rejects_noncanonical_and_nonsymplectic_maps():
    with pytest.raises(ValueError, match="Non-symplectic"):
        propagate_plane_wave(wave(), 2*np.eye(4), np.zeros(4), 2e-12)
    with pytest.raises(ValueError, match="canonical basis"):
        validate_canonical_map(np.eye(4), basis="mechanical-slopes")
    assert validate_canonical_map(drift(.001), basis=CANONICAL_BASIS) == 0


def test_curvature_symmetry_uses_matrix_scale_without_repair():
    q = np.array(((3000., -2e-10), (1e-10, 3000.)))
    w = replace(wave(), curvature_m1=q)
    np.testing.assert_array_equal(w.curvature_m1, q)
    with pytest.raises(ValueError, match="symmetric"):
        replace(w, curvature_m1=np.array(((3000., 1.), (-1., 3000.))))


def test_full_phase_includes_envelope_nyquist_budget():
    w = wave()
    envelope = w.amplitude*np.exp(1j*2.*np.arange(128)[None, :])
    # Each contribution is below pi, but their sum exceeds pi radians/pixel.
    with pytest.raises(ValueError, match="undersampled"):
        replace(w, amplitude=envelope, tilt_rad=np.array((2e-12/(np.pi*5e-10), 0.))).full_amplitude(2e-12)


def test_t215_mechanical_basis_converts_both_fields_and_reference_momenta():
    kin = canonical_basis_matrix(charge_c=-1.6e-19, bz_t=.4,
                                momentum_kg_m_s=3e-22, reference_momentum_kg_m_s=3e-22)
    kout = canonical_basis_matrix(charge_c=-1.6e-19, bz_t=1.2,
                                 momentum_kg_m_s=4e-22, reference_momentum_kg_m_s=3e-22)
    canonical = drift(.001)
    mechanical = np.linalg.inv(kout) @ canonical @ kin
    with pytest.raises(ValueError, match="Non-symplectic"):
        validate_canonical_map(mechanical)
    np.testing.assert_allclose(mechanical_map_to_canonical(mechanical, input_basis=kin, output_basis=kout),
                               canonical, atol=2e-13, rtol=2e-13)


def test_t216_segment_composition_inverse_and_phase():
    initial = wave()
    lens = np.eye(4)
    lens[2:, :2] = -np.eye(2)*500
    # Stay in a jointly sampled near-field domain; the separate T2-14 case
    # explicitly checks rejection when an expanded carrier exceeds Nyquist.
    m1, m2 = lens @ drift(.000001), drift(.000002)
    first = propagate_plane_wave(initial, m1, np.zeros(4), 2e-12)
    split = propagate_plane_wave(first, m2, np.zeros(4), 2e-12)
    whole = propagate_plane_wave(initial, m2 @ m1, np.zeros(4), 2e-12)
    np.testing.assert_allclose(split.basis_m, whole.basis_m, atol=1e-23, rtol=1e-12)
    np.testing.assert_allclose(split.full_amplitude(2e-12), whole.full_amplitude(2e-12), atol=1e-10, rtol=1e-9)
    back = propagate_plane_wave(split, np.linalg.inv(m2 @ m1), np.zeros(4), 2e-12)
    np.testing.assert_allclose(back.full_amplitude(2e-12), initial.amplitude, atol=1e-10, rtol=1e-9)
    assert back.probability == pytest.approx(initial.probability, abs=1e-12)


def test_t217_wave_density_never_drops_the_carrier_or_restores_lost_weight():
    from types import SimpleNamespace
    from temsim.physics.core import electron
    w = replace(wave(), tilt_rad=np.array([1e-4, 0.]))
    mode = WaveMode(w, .25, "source_reference", "gun:0", 300.)
    lam = electron(SimpleNamespace(beam_voltage_kv=300.))[2]*1e-9
    scale = np.sqrt(.25/abs(np.linalg.det(w.basis_m)))
    np.testing.assert_allclose(mode.weighted_density_amplitude(), w.full_amplitude(lam)*scale)
    assert np.sum(np.abs(mode.weighted_density_amplitude())**2)*abs(np.linalg.det(w.basis_m)) == pytest.approx(.25)
    with pytest.raises(ValueError):
        w.amplitude.setflags(write=True)


def test_beam_modes_do_not_retain_a_mutable_caller_container():
    mode = WaveMode(wave(), .25, "source_reference", "gun:0", 300.)
    modes = [mode]
    beam = BeamState(modes, "source_reference")
    modes.clear()
    assert beam.modes == (mode,) and beam.total_weight == .25
    with pytest.raises(TypeError, match="WaveMode"):
        BeamState([object()], "source_reference")
    with pytest.raises(TypeError, match="PlaneWave"):
        WaveMode(object(), 1., "source_reference", "gun:0", 300.)


@pytest.mark.parametrize("distance", [2e-4, 1e-3, -.001])
def test_fresnel_zoom_keeps_gaussian_size_not_just_norm(distance):
    # A 256-cell source includes >12 sigma to each side. This isolates Fresnel
    # quadrature from the hard input-window tail at 6.4 sigma in the smaller fixture.
    initial = wave(256)
    lam = 2e-12
    out = propagate_plane_wave(initial, drift(distance), np.zeros(4), lam)
    expected_covariance = drift(distance)@initial.canonical_covariance(lam)@drift(distance).T
    # Compare analytic Gaussian marginal covariance, without fitting/renormalising
    # the propagated field. Far-field input/output quadrature is independently sampled.
    xy = out.coordinates_m()
    actual = np.sum(abs(out.amplitude)**2*xy[0]**2)/out.probability
    assert actual == pytest.approx(expected_covariance[0, 0], rel=1e-7)
    assert math.sqrt(actual)/np.linalg.norm(out.basis_m[:, 0]) > 5
