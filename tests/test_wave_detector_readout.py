from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.detector.wave_readout import WaveReadoutOptions, read_wave_detector
from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE, wavelength_m
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.tip_gun_wave import TipGunCheckpoint
from temsim.physics.wave_flux import BeamState, WaveMode


def checkpoint(phase=0., weight=.6, two=False):
    x = (np.arange(64)-32)*1e-6
    xx, yy = np.meshgrid(x, x)
    a = np.exp(-(xx*xx+yy*yy)/(2*(6e-6)**2)).astype(complex)*np.exp(1j*phase)
    a /= np.linalg.norm(a)
    p = PlaneWave(a, np.eye(2)*1e-6, np.zeros(2), np.eye(2)*.01, np.array((1e-7, -2e-7)))
    m = WaveMode(p, weight, TIP_REFERENCE, "energy:0/spatial:0", 300.)
    modes = (m, replace(m, mode_id="energy:0/spatial:1", weight_per_reference_electron=.2,
                        plane=replace(p, amplitude=p.amplitude*1j))) if two else (m,)
    return TipGunCheckpoint(BeamState(modes, TIP_REFERENCE), 2000., 1e-9, {"fixture": "independent detector kernel"})


def detector():
    return SimpleNamespace(z_mm=2000., inserted=True, key="test", pixels=64,
        geometry="square", outer_width_mm=.2, inner_diameter_mm=0.,
        point_spread_model="none", point_spread_sigma_x_mm=0., point_spread_sigma_y_mm=0.,
        point_spread_rotation_deg=0., point_spread_status="provisional_model_parameter",
        point_spread_source="independent fixture")


def test_detector_preserves_carrier_phase_energy_absolute_weight_and_input():
    from temsim.physics.wave_reference import AxialWaveReference
    c = checkpoint(.4, two=True)
    c = replace(c, beam=replace(c.beam, modes=tuple(replace(m, axial_reference=AxialWaveReference(2e-8, 1e-22)) for m in c.beam.modes)))
    r = read_wave_detector(c, detector(), WaveReadoutOptions(phase=True, complex_amplitude=True, covariance=True))
    assert r.optical_probability.sum() == pytest.approx(.8)
    for old, new in zip(c.beam.modes, r.modes):
        assert new.plane is old.plane
        assert new.energy_kev == 300.
        assert new.axial_reference is old.axial_reference
        expected = old.plane.full_amplitude(float(wavelength_m(300000)))*np.sqrt(old.weight_per_reference_electron)
        np.testing.assert_allclose(new.complex_cell_amplitude, expected, atol=1e-14)
        np.testing.assert_allclose(np.exp(1j*new.phase_rad[new.phase_valid]), np.exp(1j*np.angle(expected[new.phase_valid])), atol=1e-14)
        assert new.canonical_covariance.shape == (4, 4)
        assert not new.phase_rad.flags.writeable
    assert "UNDEFINED" in r.record["mixed_total_phase"]
    assert c.beam.total_weight == pytest.approx(.8)


def test_axial_reference_is_part_of_checkpoint_identity_and_survives_unobserved_absorption():
    from temsim.physics.wave_reference import AxialWaveReference
    from temsim.detector.wave_readout import _apply_recording_stop
    c = checkpoint()
    changed = replace(c, beam=replace(c.beam, modes=(replace(c.beam.modes[0], axial_reference=AxialWaveReference(1e-8, 1e-22)),)))
    assert changed.digest != c.digest
    d = detector(); d.readout_enabled = False; d.outer_width_mm = .01
    stopped = _apply_recording_stop(changed, d)
    assert 0 < stopped.beam.total_weight < changed.beam.total_weight
    assert stopped.beam.modes[0].axial_reference is changed.beam.modes[0].axial_reference
    with pytest.raises(ValueError, match="matrix-only"):
        changed.beam.propagate(lambda energy: (np.eye(4), np.zeros(4)))
    for bad in ((-1, 0), (0, np.nan), (True, 0)):
        with pytest.raises(ValueError):
            AxialWaveReference(*bad)


def test_selecting_phase_reuses_full_state_and_does_not_compute_intensity(monkeypatch):
    import temsim.physics.camera_wave as camera
    monkeypatch.setattr(camera, "_deposit_mapped_probability", lambda *a: pytest.fail("unrequested intensity computed"))
    c = checkpoint()
    r = read_wave_detector(c, detector(), WaveReadoutOptions(intensity=False, phase=True))
    assert r.optical_probability is None and r.x_mm is None
    assert r.modes[0].complex_cell_amplitude is None
    assert r.modes[0].plane is c.beam.modes[0].plane


def test_zero_weight_has_no_phase_and_no_counts():
    r = read_wave_detector(checkpoint(weight=0), detector(), WaveReadoutOptions(phase=True))
    assert not r.modes[0].phase_valid.any()
    assert np.isnan(r.modes[0].phase_rad).all()
    assert r.optical_probability.sum() == 0


def test_mixture_intensity_is_invariant_to_independent_mode_phase():
    a = read_wave_detector(checkpoint(0., two=True), detector())
    b = read_wave_detector(checkpoint(1.1, two=True), detector())
    np.testing.assert_allclose(a.optical_probability, b.optical_probability, atol=1e-17)


def test_detector_budget_geometry_and_actual_plane_are_checked():
    with pytest.raises(ValueError, match="bytes"):
        read_wave_detector(checkpoint(), detector(), maximum_bytes=1)
    d = detector(); d.z_mm += 1
    with pytest.raises(ValueError, match="physical detector plane"):
        read_wave_detector(checkpoint(), d)
    with pytest.raises(ValueError, match="at least one"):
        WaveReadoutOptions(intensity=False).validate()


def test_disk_and_psf_preserve_physical_losses():
    d = detector(); d.geometry = "disk"; d.outer_width_mm = .01
    d.point_spread_model = "gaussian"
    d.point_spread_sigma_x_mm = d.point_spread_sigma_y_mm = .001
    r = read_wave_detector(checkpoint(), d)
    assert 0 < r.detected_probability.sum() < r.optical_probability.sum() < .6


def test_phase_is_invalid_outside_sensor_and_stop_absorbs_without_readout():
    from temsim.detector.wave_readout import _apply_recording_stop
    c = checkpoint()
    d = detector(); d.geometry = "annulus"; d.outer_width_mm = .02; d.inner_diameter_mm = .005
    r = read_wave_detector(c, d, WaveReadoutOptions(phase=True, complex_amplitude=True))
    assert not r.modes[0].phase_valid[32, 32]
    assert r.modes[0].complex_cell_amplitude[32, 32] == 0
    assert r.optical_probability.sum() > 0
    d.readout_enabled = False
    transmitted = _apply_recording_stop(c, d)
    assert transmitted.beam.total_weight+r.optical_probability.sum() == pytest.approx(c.beam.total_weight, abs=1e-12)
    with pytest.raises(ValueError, match="readout is disabled"):
        read_wave_detector(c, d)


def test_detector_offset_is_physical_and_not_a_phase_reset():
    c = checkpoint(.2)
    d = detector(); d.outer_width_mm = .01
    centered = read_wave_detector(c, d, WaveReadoutOptions(phase=True))
    d.centre_offset_x_mm = .05
    shifted = read_wave_detector(c, d, WaveReadoutOptions(phase=True))
    assert shifted.optical_probability.sum() < centered.optical_probability.sum()*1e-6
    assert shifted.modes[0].plane is centered.modes[0].plane
