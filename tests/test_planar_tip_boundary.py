"""Bounded non-paraxial operators and their actual source/field adapter.

These tests do not qualify a full gun or microscope. The actual near-tip
case retains the startup 5 nm, 0.3 eV mean and 0.3 eV spread and all nine
default energy quadrature nodes. It does not establish energy convergence.
"""
from dataclasses import replace
import math

import numpy as np
import pytest

from temsim.physics.accelerating_tip_boundary import (
    AcceleratingTipNumerics, _spectral_to_plane, execute_accelerating_tip,
    scalar_robin_transport,
)
from temsim.physics.planar_tip_boundary import solve_robin_load
from temsim.physics.scattering_load import hermitian_slab, outgoing_load


def test_robin_uniform_medium_matches_propagating_and_evanescent_analytic_solution():
    kappa, depth = 2., .7
    q = np.diag([4., -9.])
    blocks, _ = hermitian_slab(q, np.zeros_like(q), depth, kappa)
    load = outgoing_load([blocks], q, kappa)
    shape = np.ones(2)/math.sqrt(2)
    result = solve_robin_load(load, shape, kappa)
    k = np.array([2., 3j])
    expected_left = 2*kappa/(kappa+k)*shape/math.sqrt(kappa)
    np.testing.assert_allclose(result.amplitude[0], expected_left, atol=2e-15)
    np.testing.assert_allclose(result.amplitude[-1], expected_left*np.exp(1j*k*depth), atol=2e-15)
    np.testing.assert_allclose(result.derivative_per_m[-1], 1j*k*result.amplitude[-1], atol=2e-15)
    assert result.record["injected_current_fraction"] == pytest.approx(1.)
    assert result.record["transmitted_current_fraction"] == pytest.approx(.5)
    assert result.record["reflected_current_fraction"] == pytest.approx(.5)
    assert abs(result.amplitude[-1, 1]) > 0  # No evanescent truncation.
    assert not result.amplitude.flags.writeable


def test_scalar_embedding_matches_independent_coupled_two_way_operator():
    axial, widths, transverse, kappa = np.array([2., 3., 8.]), np.array([.1, .3, .2]), np.array([[0., 4.], [1., 12.]]), 1.5
    blocks = [hermitian_slab(np.diag(value-transverse.ravel()), np.zeros((4, 4)), width, kappa)[0]
              for value, width in zip(axial, widths)]
    load = outgoing_load(blocks, np.diag(10.-transverse.ravel()), kappa)
    shape = np.array([.3, .4j, -.5, .7j]); shape /= np.linalg.norm(shape)
    expected = solve_robin_load(load, shape, kappa)
    logarithm, reflection, k, record = scalar_robin_transport(axial, widths, 10., transverse, kappa)
    amplitude = np.exp(logarithm).ravel()*shape/math.sqrt(kappa)
    np.testing.assert_allclose(amplitude, expected.amplitude[-1], atol=3e-14)
    np.testing.assert_allclose(1j*k.ravel()*amplitude, expected.derivative_per_m[-1], atol=3e-14)
    np.testing.assert_allclose(reflection.ravel()*shape/math.sqrt(kappa), expected.reflected_amplitude, atol=3e-14)
    assert record["layers_with_evanescent_channels"] == 3


def test_cumulative_evanescent_transmission_remains_logarithmic_below_float_range():
    logarithm, reflection, _, record = scalar_robin_transport(
        [1., 1.], [200., 200.], 1., np.array([[9., 0.], [0., 0.]]), 1.)
    assert np.isfinite(logarithm).all() and np.isfinite(reflection).all()
    assert logarithm[0, 0].real < -1000
    assert record["logarithmic_channels_below_float_range"] == 1


def test_inverse_fourier_carrier_retains_large_expansion_offset_and_absolute_phase():
    from temsim.physics.wave_grid import WaveGridNumerics
    # Keep pitch and the physical Gaussian fixed, but include a +/-16-sigma
    # source period. A +/-8-sigma period still has amplitude at its edge
    # (~5.6e-9 per pixel); its periodic Gaussian is not the infinite Gaussian
    # below, despite its tiny omitted *intensity*. Do not relax phase errors
    # to hide this independently known boundary-tail discrepancy.
    size, pitch, sigma, wavelength, length = 256, .25e-9, 2e-9, 1e-9, 1e-6
    origin = np.array([.6e-9, -.4e-9])
    x = (np.arange(size)-size//2)*pitch
    xx, yy = np.meshgrid(x, x)
    initial = np.exp(-(xx*xx+yy*yy)/(4*sigma*sigma))*pitch/math.sqrt(2*np.pi*sigma*sigma)
    frequencies = np.fft.fftshift(np.fft.fftfreq(size, pitch))*2*np.pi
    kxy = np.array(np.meshgrid(frequencies, frequencies))
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(initial), norm="ortho"))
    k0 = 2*np.pi/wavelength
    quadratic = -length/(2*k0)
    spectrum *= np.exp(1j*quadratic*np.sum(kxy*kxy, axis=0))
    output = _spectral_to_plane(spectrum, kxy, np.eye(2)*pitch, origin, wavelength,
        quadratic, WaveGridNumerics(maximum_working_bytes=128*1024**2), lambda: False)
    coords = output.coordinates_m()-origin[:, None, None]
    width = 1+1j*length/(2*k0*sigma*sigma)
    expected = np.exp(-np.sum(coords*coords, axis=0)/(4*sigma*sigma*width))/width
    expected *= math.sqrt(abs(np.linalg.det(output.basis_m)))/math.sqrt(2*np.pi*sigma*sigma)
    carrier = np.exp(1j*np.pi/wavelength*np.einsum("iyx,ij,jyx->yx", coords, output.curvature_m1, coords))
    actual = output.amplitude*carrier
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-7)
    np.testing.assert_allclose(output.origin_m, origin, atol=1e-24)
    assert output.probability == pytest.approx(1., abs=1e-10)
    density = abs(actual)**2
    rms = math.sqrt(float(np.sum(coords[0]**2*density)))
    assert rms == pytest.approx(sigma*abs(width), rel=1e-6)
    assert rms > 10*sigma  # Not wrapped into the original ~32 nm box.


def test_actual_startup_scale_uses_captured_field_and_preserves_source():
    from temsim.immutable_json import json_digest
    from temsim.instrument_snapshot import encode_instrument
    from temsim.optics.column import default_state
    from temsim.optics.electron_gun.tip_coherence import TipCoherence, TipWaveNumerics, generate_tip_boundary_emission
    from temsim.physics.instrument_electric import capture_instrument_electric_field
    state = default_state()
    state.electron_gun.emitter.coherence = TipCoherence()
    source = generate_tip_boundary_emission(state.electron_gun, TipWaveNumerics())
    electric = capture_instrument_electric_field(state)
    before = json_digest(encode_instrument(state))
    result = execute_accelerating_tip(state, source, electric)
    assert before == json_digest(encode_instrument(state))
    emitter = state.electron_gun.emitter
    assert (emitter.virtual_source_fwhm_nm, emitter.emission_energy_ev, emitter.energy_spread_fwhm_ev) == (5., .3, .3)
    assert result.plane_z_mm == pytest.approx(.004)
    assert len(result.modes) == 9
    row = result.record["modes"][0]
    assert row["endpoint_axial_energy_ev"] > row["tip_energy_ev"]
    assert row["balance_residual"] < 1e-8
    assert row["layers_with_evanescent_channels"] > 0
    assert result.record["electric_numerical_identity"] == electric.numerical_identity
    assert result.modes[0].weight_per_reference_electron == pytest.approx(row["transmitted_fraction"]/9, abs=1e-10)
    assert len(set(result.original_energies_ev)) == 9
    assert [row["tip_energy_ev"] for row in result.record["modes"]] == list(result.original_energies_ev)
    assert result.modes[0].axial_reference.flight_time_s > 0
    assert not result.spectral_derivatives_per_m[0].flags.writeable
    assert not result.logarithmic_transmissions[0].flags.writeable
    assert max(r["transverse_phase_error_estimate_rad"] for r in result.record["modes"]) < 1e-3
    assert max(r["outside_checked_domain_fraction"] for r in result.record["modes"]) < 1e-8
    changed = replace(result, modes=(replace(result.modes[0],
        plane=replace(result.modes[0].plane, amplitude=result.modes[0].plane.amplitude*1j)), *result.modes[1:]))
    assert changed.digest != result.digest


def test_boundary_budget_and_cancellation_are_explicit():
    with pytest.raises(ValueError, match="end_z_nm"):
        AcceleratingTipNumerics(end_z_nm=0).validate()
    with pytest.raises(InterruptedError):
        scalar_robin_transport([4.], [.1], 4., np.zeros((2, 2)), 2., cancelled=lambda: True)


def test_experimental_boundary_does_not_silently_enable_unsupported_production_source(monkeypatch):
    from temsim.optics.column import default_state
    from temsim.optics.electron_gun.tip_coherence import TipCoherence
    from temsim.physics.tip_gun_wave import build_tip_gun_checkpoint
    state = default_state(); state.electron_gun.emitter.coherence = TipCoherence()
    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported source must fail before downstream field execution")
    monkeypatch.setattr("temsim.physics.instrument_electric.capture_instrument_electric_field", forbidden)
    with pytest.raises(ValueError, match="non-paraxial"):
        build_tip_gun_checkpoint(state.electron_gun, _column_state=state)


def test_near_tip_rejects_field_identity_from_another_instrument_before_wave_execution(monkeypatch):
    from temsim.optics.column import default_state
    from temsim.optics.electron_gun.tip_coherence import TipCoherence, generate_tip_boundary_emission
    from temsim.physics.instrument_electric import capture_instrument_electric_field
    state = default_state(); state.electron_gun.emitter.coherence = TipCoherence()
    emission = generate_tip_boundary_emission(state.electron_gun)
    electric = capture_instrument_electric_field(state)
    def forbidden(*args, **kwargs):
        pytest.fail("Mismatched field must fail before numerical transport")
    monkeypatch.setattr("temsim.physics.accelerating_tip_boundary.scalar_robin_transport", forbidden)
    with pytest.raises(ValueError, match="does not belong to the captured instrument"):
        execute_accelerating_tip(state, emission, replace(electric, numerical_identity="different numerical field"))
