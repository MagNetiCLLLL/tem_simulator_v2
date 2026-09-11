"""Read-only mixed-mode diagnostics on retained affine wave grids.

Fourier angles are normalized CANONICAL momentum, not mechanical ray slopes
inside a magnetic field. The distinct IDs prevent silently changing DA alpha95.
No pure wave is made by averaging mode amplitudes.
"""
import numpy as np

from temsim.immutable_json import freeze_json

WAVE_DEFINITIONS = freeze_json({
    "wave_source_fraction": {"unit": "1", "definition_id": "wave-probability/gun-exit-electron-v1"},
    "wave_centre_x": {"unit": "m", "definition_id": "mixed-wave-centre-x-v1"},
    "wave_centre_y": {"unit": "m", "definition_id": "mixed-wave-centre-y-v1"},
    "wave_radius95": {"unit": "m", "definition_id": "mixed-wave-current-contained-radius-95-v1"},
    "canonical_alpha95": {"unit": "rad", "definition_id": "mixed-wave-canonical-contained-semiangle-95-v1"},
    "canonical_alpha99": {"unit": "rad", "definition_id": "mixed-wave-canonical-contained-semiangle-99-v1"},
})
DIAGNOSTIC_MEMORY_BYTES = 256*1024**2


def _contained_radius(xy, weights, fraction):
    centre = np.sum(xy*weights[None, :], axis=1)/weights.sum()
    radii = np.linalg.norm(xy-centre[:, None], axis=0)
    order = np.argsort(radii)
    index = np.searchsorted(np.cumsum(weights[order]), fraction*weights.sum())
    return centre, float(radii[order[min(index, len(order)-1)]])


def mixed_wave_observable(beam, observable_id):
    """Return (value, status, reason), using only retained numeric data."""
    if observable_id not in WAVE_DEFINITIONS:
        return None, "UNAVAILABLE", "Not a registered wave observable"
    if observable_id == "wave_source_fraction":
        return beam.total_weight, "AVAILABLE", "Reference: gun exit; truncated mode tails are not renormalized"
    if beam.total_weight <= 0:
        return None, "UNAVAILABLE", "No surviving represented current"
    # Concatenation/sorting/FFT scratch is separate from the retained payload.
    # Refuse oversized interactive diagnostics instead of exhausting the UI.
    cells = sum(mode.plane.amplitude.size for mode in beam.modes if mode.weight_per_reference_electron > 0)
    if cells*160 > DIAGNOSTIC_MEMORY_BYTES:
        return None, "OUT_OF_VALIDATED_RANGE", "Wave diagnostic exceeds the 256 MiB interactive scratch budget"
    angular = observable_id.startswith("canonical_")
    coordinates, probabilities = [], []
    for mode in beam.modes:
        if mode.weight_per_reference_electron == 0:
            continue
        plane = mode.plane
        if angular:
            from temsim.optics.electron_gun.effective_source import wavelength_m
            wavelength = wavelength_m(mode.energy_kev*1000)
            try:
                full = plane.full_amplitude(wavelength)
            except ValueError as error:
                return None, "OUT_OF_VALIDATED_RANGE", str(error)
            ny, nx = full.shape
            fy, fx = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing="ij")
            xy = wavelength*np.einsum("ij,jyx->iyx", np.linalg.inv(plane.basis_m).T, np.stack((fx, fy)))
            mass = abs(np.fft.fft2(full, norm="ortho"))**2
        else:
            xy, mass = plane.coordinates_m(), abs(plane.amplitude)**2
        coordinates.append(xy.reshape(2, -1))
        probabilities.append(mass.ravel()*mode.weight_per_reference_electron)
    xy, weights = np.concatenate(coordinates, axis=1), np.concatenate(probabilities)
    centre, radius = _contained_radius(xy, weights, .99 if observable_id.endswith("99") else .95)
    value = centre[0] if observable_id.endswith("centre_x") else centre[1] if observable_id.endswith("centre_y") else radius
    reason = "Canonical Fourier angles; not mechanical ray alpha95 in a magnetic field" if angular else "Mixed-mode intensity on each captured physical grid"
    return float(value), "AVAILABLE", reason
