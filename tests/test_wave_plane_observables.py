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


def _posed_gauge():
    from scipy.spatial.transform import Rotation
    from temsim.physics.lens_field_provider import CoordinateRegistration, FrozenAnalyticField
    from temsim.physics.wave_plane_observables import ColumnPlaneMagneticGauge
    rotation = Rotation.from_rotvec((.002, -.001, .1)).as_matrix()
    field = FrozenAnalyticField("posed_lens", ((.3, 0., .006),),
        CoordinateRegistration((2e-5, -3e-5, 1.5), tuple(map(tuple, rotation))), (-25., 25.), .003)
    return ColumnPlaneMagneticGauge(1502., .2, (field,))


def test_probability_flow_uses_captured_rotated_vector_potential_without_double_count():
    from temsim.physics.tip_gun_wave import _momentum_velocity
    gauge, original = _posed_gauge(), mode()
    # An old scalar metadata value may accompany the new gauge. It must not
    # be added again when a complete captured gauge is present.
    result = column_mode_observables(original, reference_current_a=100e-9,
        axial_bz_t=99., magnetic_gauge=gauge, plane_z_mm=1502.)
    aligned = column_mode_observables(original, reference_current_a=100e-9, axial_bz_t=.2)
    xy = original.plane.coordinates_m()
    field = gauge.fields[0]
    positions = np.moveaxis(np.concatenate((xy, np.full((1, *xy.shape[1:]), 1.502))), 0, -1)
    local = field.registration.positions_global_to_local_m(positions)
    axial = .3*np.exp(-.5*(local[..., 2]/.006)**2)
    native_a = np.stack((-.5*local[..., 1]*axial, .5*local[..., 0]*axial, np.zeros_like(axial)), axis=-1)
    posed_a = np.moveaxis(field.registration.vectors_local_to_global(native_a)[..., :2], -1, 0)
    p = float(_momentum_velocity(300000.)[0])
    expected = aligned.transverse_current_density_a_per_m2+aligned.axial_current_density_a_per_m2*e/p*posed_a
    np.testing.assert_allclose(result.transverse_current_density_a_per_m2, expected, rtol=2e-13, atol=1e-9)
    assert "rigid analytical" in result.gauge
    with pytest.raises(ValueError, match="exact observation plane"):
        column_mode_observables(original, reference_current_a=1e-9, magnetic_gauge=gauge, plane_z_mm=1501.)


def test_gauge_capture_excludes_vector_lenses_and_is_independent_of_later_controls(monkeypatch):
    from types import SimpleNamespace
    from temsim.physics.wave_plane_observables import capture_column_plane_magnetic_gauge
    import temsim.physics.lens_field_provider as providers
    import temsim.physics.core as core
    field = _posed_gauge().fields[0]
    state = SimpleNamespace(field=field, aligned=.2)
    calls = []
    monkeypatch.setattr(providers, "active_vector_providers", lambda current: (current.field,))
    monkeypatch.setattr(providers, "freeze_vector_provider", lambda provider: provider)
    def aligned_field(z, current, *, exclude_mapped_keys):
        calls.append((tuple(z), exclude_mapped_keys))
        return np.array((current.aligned,)), np.zeros(1), np.zeros(1)
    monkeypatch.setattr(core, "fields", aligned_field)
    captured = capture_column_plane_magnetic_gauge(state, 1502.)
    before = captured.vector_potential_xy_t_m(np.zeros((2, 1)))
    digest = captured.fingerprint
    state.aligned = .7
    state.field = replace(field, terms_t_m=((.9, 0., .006),))
    np.testing.assert_array_equal(captured.vector_potential_xy_t_m(np.zeros((2, 1))), before)
    assert captured.fingerprint == digest
    assert captured.fields == (field,)
    assert calls == [((1502.,), ("posed_lens",))]
    assert capture_column_plane_magnetic_gauge(state, 1502.).fingerprint != digest


def test_gauge_capture_ignores_unrelated_finite_map_but_rejects_map_at_plane(monkeypatch):
    from types import SimpleNamespace
    from temsim.physics.wave_plane_observables import capture_column_plane_magnetic_gauge
    import temsim.physics.lens_field_provider as providers
    import temsim.physics.core as core
    field_map = SimpleNamespace(field_support_mm=(1600., 1700.),
                                reference_excitation_percent=100., reference_polarity=1)
    provider = providers.MappedLensFieldProvider("downstream_map", field_map,
        SimpleNamespace(enabled=True, percent=100., polarity=1), None)
    state, excluded, frozen = SimpleNamespace(), [], []
    monkeypatch.setattr(providers, "active_vector_providers", lambda _: (provider,))
    original_freeze = providers.freeze_vector_provider
    def freeze(value):
        frozen.append(value.lens_key)
        return original_freeze(value)
    monkeypatch.setattr(providers, "freeze_vector_provider", freeze)
    def axial_field(z, current, *, exclude_mapped_keys):
        excluded.append(exclude_mapped_keys)
        return np.zeros(1), np.zeros(1), np.zeros(1)
    monkeypatch.setattr(core, "fields", axial_field)
    for z in (1500., 1800.):
        captured = capture_column_plane_magnetic_gauge(state, z)
        assert captured.fields == ()
    assert frozen == []
    assert excluded == [("downstream_map",), ("downstream_map",)]
    with pytest.raises(ValueError, match="vector potential"):
        capture_column_plane_magnetic_gauge(state, 1650.)
    assert frozen == ["downstream_map"]


def test_dynamic_probability_flow_uses_executed_coil_capture_including_material_continuation():
    from dataclasses import asdict
    from temsim.physics.wave_plane_observables import checkpoint_magnetic_gauge
    from test_posed_multipole_wave import field
    static = replace(field(0, "uniform"), dynamic=True, captured_time_s=0.)
    executed = replace(static, polynomial_terms=((1, 0, .05), (0, 1, -.01)), captured_time_s=.3)
    gauge = replace(_posed_gauge(), plane_z_mm=502., fields=(_posed_gauge().fields[0], static))
    checkpoint = TipGunCheckpoint(BeamState((mode(),), TIP_REFERENCE), gauge.plane_z_mm, 1e-9,
        {"schema": "material", "upstream": {"modes": [
            {"mode_id": mode().mode_id, "posed_column_fields": [asdict(executed)]}]}})
    result = checkpoint_magnetic_gauge(gauge, checkpoint)
    assert result.fields == (gauge.fields[0], executed)
    assert gauge.fields[-1] == static
    xy = mode().plane.coordinates_m()
    expected = replace(gauge, fields=(gauge.fields[0], executed))
    np.testing.assert_array_equal(result.vector_potential_xy_t_m(xy), expected.vector_potential_xy_t_m(xy))
    assert np.max(abs(result.vector_potential_xy_t_m(xy)-gauge.vector_potential_xy_t_m(xy))) > 0.


def test_modes_with_different_or_unknown_dynamic_gauges_cannot_claim_a_common_probability_flow():
    from dataclasses import asdict
    from temsim.physics.wave_plane_observables import checkpoint_magnetic_gauge
    from test_posed_multipole_wave import field
    first = replace(field(0, "uniform"), dynamic=True, captured_time_s=.1)
    second = replace(first, polynomial_terms=((1, 0, .07),), captured_time_s=.2)
    gauge = replace(_posed_gauge(), plane_z_mm=502., fields=(first,))
    checkpoint = TipGunCheckpoint(BeamState((mode(),), TIP_REFERENCE), gauge.plane_z_mm, 1e-9,
        {"modes": [{"posed_column_fields": [asdict(value)]} for value in (first, second)]})
    assert checkpoint_magnetic_gauge(gauge, checkpoint) is None
    assert checkpoint_magnetic_gauge(gauge, replace(checkpoint, record={"old": "cache"})) is None
    assert checkpoint_magnetic_gauge(replace(gauge, fields=(replace(first, dynamic=False),)),
                                     replace(checkpoint, record={"old": "cache"})) is not None
