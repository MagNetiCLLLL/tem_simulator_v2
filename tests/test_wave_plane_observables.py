"""Analytical transverse-current and phase tests of the forward-wave contract."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.constants import c, e, m_e

from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE, wavelength_m
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.tip_gun_wave import TipGunCheckpoint
from temsim.physics.wave_flux import BeamState, WaveMode
from temsim.physics.wave_plane_observables import (
    column_mode_observables, canonical_angular_spectrum, interaction_weights)


def mode():
    y, x = np.meshgrid(np.arange(32)-16, np.arange(32)-16, indexing="ij")
    plane = PlaneWave(np.exp(2j*np.pi*(2*x-3*y)/32)/32,
        np.array(((2e-9, .2e-9), (0., 3e-9))), np.array((1e-9, -2e-9)),
        np.array(((100., 20.), (20., -50.))), np.array((.0002, -.0001)))
    return WaveMode(plane, .3, TIP_REFERENCE, "fixture:one", 300.)


def test_phase_only_readout_matches_full_observable_without_fft(monkeypatch):
    from temsim.physics.wave_plane_observables import mode_phase_samples
    original = mode()
    expected = column_mode_observables(original, reference_current_a=1e-9, axial_bz_t=.7).phase_rad
    monkeypatch.setattr(np.fft, "fft2", lambda *a, **kw: pytest.fail("Phase readout must not allocate an FFT"))
    full = mode_phase_samples(original)
    # Enough output space plus one row of temporary buffers.
    chunked = mode_phase_samples(original, maximum_working_bytes=24*32**2+128*32)
    np.testing.assert_array_equal(full, chunked)
    np.testing.assert_allclose(np.exp(1j*full), np.exp(1j*expected), atol=1e-12)
    assert not full.flags.writeable
    assert np.all(np.isnan(mode_phase_samples(replace(original, weight_per_reference_electron=0.))))


def test_phase_only_budget_cannot_change_or_erase_the_wave():
    from temsim.physics.wave_plane_observables import mode_phase_samples
    original = mode()
    before = original.plane.amplitude.copy()
    with pytest.raises(MemoryError):
        mode_phase_samples(original, maximum_working_bytes=1)
    np.testing.assert_array_equal(original.plane.amplitude, before)


def test_2048_phase_view_fits_below_full_current_analysis_budget(monkeypatch):
    from temsim.physics.wave_plane_observables import mode_phase_samples
    original = mode()
    large = replace(original, plane=PlaneWave(np.full((2048, 2048), 1j/2048),
        np.eye(2)*1e-9, np.zeros(2)))
    monkeypatch.setattr(np.fft, "fft2", lambda *a, **kw: pytest.fail("Phase-only view must not run FFT"))
    result = mode_phase_samples(large, maximum_working_bytes=128*1024**2)
    assert result.shape == (2048, 2048)
    np.testing.assert_allclose(result, np.pi/2, atol=0, rtol=0)


@pytest.mark.parametrize("bz", [0., 1., -1.])
def test_affine_spectral_current_includes_carriers_and_magnetic_vector_potential(bz):
    original = mode()
    result = column_mode_observables(original, reference_current_a=100e-9, axial_bz_t=bz)
    xy = original.plane.coordinates_m()
    wave = original.plane
    gamma = 1+300000*e/(m_e*c*c)
    p = m_e*c*np.sqrt(gamma*gamma-1)
    exact = wavelength_m(300000.)*np.linalg.inv(wave.basis_m).T@np.array((2/32, -3/32))
    slopes = (exact+wave.tilt_rad)[:, None, None]+np.einsum("ij,jyx->iyx", wave.curvature_m1,
                                                                        xy-wave.origin_m[:, None, None])
    slopes += e*bz/(2*p)*np.stack((-xy[1], xy[0]))
    expected_axial = 100e-9*.3/32**2/abs(np.linalg.det(wave.basis_m))
    np.testing.assert_allclose(result.axial_current_density_a_per_m2, expected_axial, rtol=1e-13)
    np.testing.assert_allclose(result.transverse_current_density_a_per_m2, expected_axial*slopes, rtol=1e-12, atol=1e-12)
    assert result.integrated_current_a == pytest.approx(30e-9, rel=1e-13)
    np.testing.assert_allclose(np.exp(1j*result.phase_rad),
        original.plane.full_amplitude(wavelength_m(300000.))/abs(original.plane.amplitude), atol=1e-12)


def test_angular_distribution_includes_tilt_and_conserves_source_weight():
    original = mode()
    wave = original.plane
    extra = wavelength_m(300000.)*np.linalg.inv(wave.basis_m).T@np.array((1/32, 2/32))
    original = replace(original, plane=replace(wave, curvature_m1=None, tilt_rad=extra))
    result = canonical_angular_spectrum(original)
    probability = result["probability_per_tip_electron"]
    assert probability.sum() == pytest.approx(.3, rel=1e-13)
    maximum = np.unravel_index(probability.argmax(), probability.shape)
    assert maximum == (15, 19)  # Original (kx,ky)=(2,-3), carrier adds (1,2).
    exact = wavelength_m(300000.)*np.linalg.inv(wave.basis_m).T@np.array((3/32, -1/32))
    np.testing.assert_allclose(result["canonical_angles_rad"][:, maximum[0], maximum[1]], exact)


def test_undersampled_carrier_is_not_replaced_by_envelope_spectrum():
    original = mode()
    original = replace(original, plane=replace(original.plane, tilt_rad=np.ones(2)))
    with pytest.raises(ValueError, match="undersampled"):
        canonical_angular_spectrum(original)


def test_zero_has_no_phase_current_or_angles_with_weight():
    original = mode()
    empty = replace(original, weight_per_reference_electron=0.,
        plane=replace(original.plane, amplitude=np.zeros_like(original.plane.amplitude)))
    result = column_mode_observables(empty, reference_current_a=100e-9, axial_bz_t=1.)
    assert np.isnan(result.phase_rad).all()
    assert not np.any(result.transverse_current_density_a_per_m2)
    assert result.integrated_current_a == 0


def test_interaction_history_grouping_keeps_all_original_weights_and_phase_independent():
    first = mode()
    second = replace(first, mode_id="fixture:two", weight_per_reference_electron=.2,
        scattering_history=({"kind": "plasmon", "loss_ev": 16.},),
        plane=replace(first.plane, amplitude=first.plane.amplitude*1j))
    third = replace(second, mode_id="fixture:three", weight_per_reference_electron=.1)
    checkpoint = TipGunCheckpoint(BeamState((first, second, third), TIP_REFERENCE), 1500., 100e-9,
                                  {"scope": "test fixture"})
    rows = interaction_weights(checkpoint)
    assert len(rows) == 2
    assert sum(row["weight_per_tip_electron"] for row in rows) == pytest.approx(.6)
    assert sum(row["current_a"] for row in rows) == pytest.approx(60e-9)
    assert set(rows[1]["mode_ids"]) == {second.mode_id, third.mode_id}
    assert all("phase" not in row for row in rows)


def test_numerical_limits_and_immutable_arrays():
    with pytest.raises(MemoryError):
        column_mode_observables(mode(), reference_current_a=1e-9, axial_bz_t=0., maximum_working_bytes=1)
    result = column_mode_observables(mode(), reference_current_a=1e-9, axial_bz_t=0.)
    with pytest.raises(ValueError): result.phase_rad[0, 0] = 0.
    with pytest.raises(ValueError):
        column_mode_observables(mode(), reference_current_a=-1., axial_bz_t=0.)
