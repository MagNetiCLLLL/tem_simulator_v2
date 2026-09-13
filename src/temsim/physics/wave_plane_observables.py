"""Read-only transverse diagnostics of an executed forward column mode.

These are not independent sources or detector counts. A probability-flow
direction is not an individually measured electron trajectory. Canonical
angular spectra are gauge-labelled; in a magnetic field they are not a
joint distribution of the noncommuting kinetic momentum components.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.constants import e

from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE, wavelength_m
from temsim.physics.tip_gun_wave import _momentum_velocity
from temsim.physics.wave_execution import check_available_memory


def _freeze(value):
    value = np.ascontiguousarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def mode_phase_samples(mode, *, maximum_working_bytes=512*1024**2):
    """Read only wrapped point phase; do not allocate FFT/current observables.

    The original affine grid and analytical carriers are retained. This is
    neither an unwrapped reconstruction nor an incoherent-mixture phase.
    Row chunks change memory use, not the evaluated physical coordinates.
    """
    if mode.reference_plane != TIP_REFERENCE:
        raise ValueError("Phase samples require an executed tip-origin mode")
    wave = mode.plane
    ny, nx = wave.amplitude.shape
    output_bytes = 24*wave.amplitude.size
    if type(maximum_working_bytes) is not int or maximum_working_bytes < output_bytes+128*nx:
        raise MemoryError("Phase display memory budget exceeded")
    rows = min(ny, max(1, (maximum_working_bytes-output_bytes)//(128*nx)))
    check_available_memory(output_bytes+128*nx*rows)
    phase = np.empty((ny, nx))
    curvature = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    wavelength = float(wavelength_m(mode.energy_kev*1000))
    for first in range(0, ny, rows):
        last = min(ny, first+rows)
        yy, xx = np.meshgrid(np.arange(first, last)-ny//2, np.arange(nx)-nx//2, indexing="ij")
        delta = np.einsum("ij,jyx->iyx", wave.basis_m, np.stack((xx, yy)))
        carrier = 2*np.pi/wavelength*(.5*np.einsum("iyx,ij,jyx->yx", delta, curvature, delta)
                                     +np.einsum("i,iyx->yx", tilt, delta))
        amplitude = wave.amplitude[first:last]
        phase[first:last] = np.where((amplitude != 0) & (mode.weight_per_reference_electron > 0),
                                    np.angle(amplitude*np.exp(1j*carrier)), np.nan)
    return _freeze(phase)


@dataclass(frozen=True)
class ColumnModeObservables:
    mode_id: str
    energy_kev: float
    coordinates_m: np.ndarray
    cell_probability_per_tip_electron: np.ndarray
    axial_current_density_a_per_m2: np.ndarray
    transverse_current_density_a_per_m2: np.ndarray
    phase_rad: np.ndarray
    integrated_current_a: float
    axial_reference: object
    scattering_history: tuple
    gauge: str = "A=(-Bz*y/2, Bz*x/2, 0), laboratory SI; forward paraxial column state"


def column_mode_observables(mode, *, reference_current_a, axial_bz_t,
                            maximum_working_bytes=512*1024**2):
    """Full envelope-gradient plus analytical carrier and magnetic current.

    The mode is the *forward column* contract, not the two-way low-energy
    gun. The latter requires its saved normal derivative and separate
    radial_mode_observables. Reference current remains the physical tip's.
    """
    if (isinstance(reference_current_a, bool) or not math.isfinite(reference_current_a)
            or reference_current_a < 0 or isinstance(axial_bz_t, bool) or not math.isfinite(axial_bz_t)):
        raise ValueError("Wave diagnostics need a finite tip current and actual axial field")
    if mode.reference_plane != TIP_REFERENCE:
        raise ValueError("Tip-referenced diagnostics require the executed tip-origin mode")
    wave = mode.plane
    required = 256*wave.amplitude.size
    if type(maximum_working_bytes) is not int or maximum_working_bytes < required:
        raise MemoryError("Wave observable memory budget exceeded")
    check_available_memory(required)
    ny, nx = wave.amplitude.shape
    fy, fx = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing="ij")
    frequency = np.einsum("ij,jyx->iyx", np.linalg.inv(wave.basis_m).T, np.stack((fx, fy)))
    wavelength = float(wavelength_m(mode.energy_kev*1000))
    # Fourier derivative of the represented envelope, with the known
    # unwrapped carrier gradient added analytically (no phase differencing).
    momentum_wave = wavelength*np.fft.ifft2(frequency*np.fft.fft2(wave.amplitude), axes=(-2, -1))
    xy = wave.coordinates_m()
    delta = xy-wave.origin_m[:, None, None]
    curvature = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    carrier_gradient = np.einsum("ij,jyx->iyx", curvature, delta)+tilt[:, None, None]
    momentum, _ = _momentum_velocity(mode.energy_kev*1000)
    vector_potential = .5*axial_bz_t*np.stack((-xy[1], xy[0]))
    # p_kinetic = p_canonical - q*A, q=-e for an electron.
    momentum_wave += (carrier_gradient+e*vector_potential/momentum)*wave.amplitude
    area = abs(float(np.linalg.det(wave.basis_m)))
    probability = mode.weight_per_reference_electron*abs(wave.amplitude)**2
    axial = reference_current_a/area*probability
    transverse = reference_current_a*mode.weight_per_reference_electron/area*np.real(wave.amplitude.conj()*momentum_wave)
    carrier = 2*np.pi/wavelength*(.5*np.einsum("iyx,ij,jyx->yx", delta, curvature, delta)
                                 +np.einsum("i,iyx->yx", tilt, delta))
    # Wrapped point phase plus the immutable original carrier, not a claimed
    # resolved phase reconstruction or one phase for a mixed beam.
    phase = np.where(probability > 0, np.angle(wave.amplitude*np.exp(1j*carrier)), np.nan)
    return ColumnModeObservables(mode.mode_id, mode.energy_kev,
        *map(_freeze, (xy, probability, axial, transverse, phase)),
        float(reference_current_a*probability.sum()), mode.axial_reference, mode.scattering_history)


def canonical_angular_spectrum(mode, *, maximum_working_bytes=512*1024**2):
    """Canonical pX/p0,pY/p0 spectrum; materialise/check the complete phase.

    Curvature and tilt are included. If their expansion is undersampled this
    readout refuses; it never substitutes the envelope-only diffraction.
    """
    if mode.reference_plane != TIP_REFERENCE:
        raise ValueError("Tip-referenced diagnostics require the executed tip-origin mode")
    wave = mode.plane
    required = 192*wave.amplitude.size
    if type(maximum_working_bytes) is not int or maximum_working_bytes < required:
        raise MemoryError("Angular wave observable memory budget exceeded")
    check_available_memory(required)
    wavelength = float(wavelength_m(mode.energy_kev*1000))
    full = wave.full_amplitude(wavelength)
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(full), norm="ortho"))
    ny, nx = full.shape
    fy, fx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(ny)),
                         np.fft.fftshift(np.fft.fftfreq(nx)), indexing="ij")
    angles = wavelength*np.einsum("ij,jyx->iyx", np.linalg.inv(wave.basis_m).T, np.stack((fx, fy)))
    probability = mode.weight_per_reference_electron*abs(spectrum)**2
    return {"canonical_angles_rad": _freeze(angles), "probability_per_tip_electron": _freeze(probability),
            "mode_id": mode.mode_id, "energy_kev": mode.energy_kev,
            "scope": "Canonical angle spectrum in the inherited laboratory gauge; not kinetic angles inside a magnetic field"}


def interaction_weights(checkpoint):
    """Exclusive executed branch histories; no invented sites or photons."""
    from temsim.immutable_json import json_digest
    if checkpoint.beam.reference_plane != TIP_REFERENCE:
        raise ValueError("Interaction readout requires the executed tip-origin checkpoint")
    rows = {}
    for mode in checkpoint.beam.modes:
        key = json_digest(mode.scattering_history)
        row = rows.setdefault(key, {"history": mode.scattering_history,
                                   "weight_per_tip_electron": 0., "current_a": 0., "mode_ids": []})
        row["weight_per_tip_electron"] += mode.weight_per_reference_electron
        row["current_a"] += checkpoint.reference_current_a*mode.weight_per_reference_electron
        row["mode_ids"].append(mode.mode_id)
    return tuple(rows.values())
