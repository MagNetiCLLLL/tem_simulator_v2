"""Isolated wave diagnostics; not a gun or microscope validation."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.wave_flux import BeamState, WaveMode
from temsim.physics.wave_observables import mixed_wave_observable
from temsim.optics.electron_gun.effective_source import REFERENCE_ID, wavelength_m


def _mode(name="a", weight=.4, origin=(0., 0.), tilt=(0., 0.)):
    n, dx, sigma = 128, .5e-9, 4e-9
    x = (np.arange(n)-n//2)*dx
    values = np.exp(-(x[:, None]**2+x[None, :]**2)/(4*sigma*sigma))
    values /= np.linalg.norm(values)
    return WaveMode(PlaneWave(values, np.eye(2)*dx, np.array(origin), tilt_rad=np.array(tilt)),
                    weight, REFERENCE_ID, name, 300.)


def test_mixed_intensity_not_coherent_mean_and_no_weight_renormalization():
    a = _mode(origin=(-8e-9, 0.))
    b = _mode("b", .2, (8e-9, 0.))
    # An arbitrary mode global phase must not change any mixed intensity.
    b = replace(b, plane=replace(b.plane, amplitude=-b.plane.amplitude))
    beam = BeamState((a, b), REFERENCE_ID)
    assert mixed_wave_observable(beam, "wave_source_fraction")[0] == pytest.approx(.6)
    assert mixed_wave_observable(beam, "wave_centre_x")[0] == pytest.approx(-8e-9/3, abs=1e-20)
    assert mixed_wave_observable(beam, "wave_radius95")[0] > 10e-9


def test_gaussian_contained_radius_and_canonical_angle_have_explicit_sampling_tolerance():
    beam = BeamState((_mode(),), REFERENCE_ID)
    factor = np.sqrt(-2*np.log(.05))
    assert mixed_wave_observable(beam, "wave_radius95")[0] == pytest.approx(4e-9*factor, rel=.025)
    # Fourier cells discretize a small angular Gaussian more coarsely than x.
    theta = wavelength_m(300000)/(4*np.pi*4e-9)
    assert mixed_wave_observable(beam, "canonical_alpha95")[0] == pytest.approx(theta*factor, rel=.10)
    assert "not mechanical" in mixed_wave_observable(beam, "canonical_alpha95")[2]


def test_wave_angle_aliasing_is_unavailable_not_a_zero_angle():
    beam = BeamState((_mode(tilt=(.1, 0.)),), REFERENCE_ID)
    value, status, reason = mixed_wave_observable(beam, "canonical_alpha95")
    assert value is None and status == "OUT_OF_VALIDATED_RANGE" and "undersampled" in reason


def test_wave_diagnostic_budget_is_explicit(monkeypatch):
    monkeypatch.setattr("temsim.physics.wave_observables.DIAGNOSTIC_MEMORY_BYTES", 1)
    beam = BeamState((_mode(),), REFERENCE_ID)
    assert mixed_wave_observable(beam, "wave_radius95")[1] == "OUT_OF_VALIDATED_RANGE"
    assert mixed_wave_observable(beam, "wave_source_fraction")[0] == .4
