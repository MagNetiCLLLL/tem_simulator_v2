"""Explicit test-only finite Fourier pupil for isolated downstream operators.

These supplied fields do not execute extraction/acceleration and cannot admit
any production TEM/STEM request. Public source admission is never patched.
"""
import numpy as np
import pytest
from temsim.physics import stem_wave_imaging as stem
from temsim.optics.aberrations import aberration_phase_rad


def synthetic_probe_spectrum(state, stats, frequencies_x, frequencies_y, wavelength_angstrom):
    """Circular pupil and configured phase, defined solely for operator tests."""
    fx, fy = np.meshgrid(frequencies_x, frequencies_y, indexing="xy")
    step = max(abs(float(frequencies_x[1]-frequencies_x[0])),
               abs(float(frequencies_y[1]-frequencies_y[0])))
    angle = max(float(stats["convergence_semiangle_rad"]), step*wavelength_angstrom)
    cx = np.sin(float(stats["mean_tx_rad"]))/wavelength_angstrom
    cy = np.sin(float(stats["mean_ty_rad"]))/wavelength_angstrom
    pupil = (fx-cx)**2+(fy-cy)**2 <= (np.sin(angle)/wavelength_angstrom)**2
    if not np.any(pupil):
        pupil.flat[np.argmin((fx-cx)**2+(fy-cy)**2)] = True
    coefficients, _ = stem.probe_focus_aberrations(state, stats)
    return pupil.astype(complex)*np.exp(-1j*aberration_phase_rad(fx,fy,wavelength_angstrom,coefficients))


@pytest.fixture(autouse=True)
def supplied_local_probe(monkeypatch):
    monkeypatch.setattr(stem, "_probe_spectrum", synthetic_probe_spectrum)
