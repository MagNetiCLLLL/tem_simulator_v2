"""Read-only transverse diagnostics of an executed forward column mode.

These are not independent sources or detector counts. A probability-flow
direction is not an individually measured electron trajectory. Canonical
angular spectra are gauge-labelled; in a magnetic field they are not a
joint distribution of the noncommuting kinetic momentum components.
"""
from dataclasses import asdict, dataclass, replace
from collections.abc import Mapping
import math

import numpy as np
from scipy.constants import e

from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE, wavelength_m
from temsim.physics.tip_gun_wave import _momentum_velocity
from temsim.physics.wave_execution import check_available_memory


def _freeze(value):
    value = np.ascontiguousarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class ColumnPlaneMagneticGauge:
    """Immutable installed magnetic gauge at one executed global Z plane.

    Fields carry their captured excitation and rigid registration. No readout
    calls back into live instrument controls. The aligned term excludes these
    vector providers, so displaced lenses are not counted twice.
    """
    plane_z_mm: float
    aligned_bz_t: float
    fields: tuple = ()

    def __post_init__(self):
        from temsim.physics.posed_lens_wave import require_analytic_fields
        if (isinstance(self.plane_z_mm, bool) or isinstance(self.aligned_bz_t, bool)
                or not math.isfinite(self.plane_z_mm) or not math.isfinite(self.aligned_bz_t)):
            raise ValueError("Column magnetic gauge requires finite physical Z and aligned field")
        object.__setattr__(self, "plane_z_mm", float(self.plane_z_mm))
        object.__setattr__(self, "aligned_bz_t", float(self.aligned_bz_t))
        object.__setattr__(self, "fields", require_analytic_fields(self.fields))

    @property
    def fingerprint(self):
        from temsim.immutable_json import json_digest
        return json_digest(("column-plane-magnetic-gauge-v1", asdict(self)))

    def require_plane(self, plane_z_mm):
        if (isinstance(plane_z_mm, bool) or plane_z_mm is None or not math.isfinite(plane_z_mm)
                or not math.isclose(self.plane_z_mm, plane_z_mm, rel_tol=0., abs_tol=1e-9)):
            raise ValueError("Probability flow requires the captured magnetic gauge at the exact observation plane")

    def vector_potential_xy_t_m(self, coordinates_m):
        """Evaluate the actual rotated analytical gauge, never an on-axis B fit."""
        xy = np.asarray(coordinates_m, float)
        if xy.ndim < 2 or xy.shape[0] != 2 or not np.isfinite(xy).all():
            raise ValueError("Magnetic gauge coordinates must be finite global XY arrays")
        potential = .5*self.aligned_bz_t*np.stack((-xy[1], xy[0]))
        for field in self.fields:
            evaluate = getattr(field, "vector_potential_at_global_positions_t_m", None)
            if evaluate is not None:
                xyz = np.moveaxis(np.stack((xy[0], xy[1],
                    np.full_like(xy[0], self.plane_z_mm*1e-3))), 0, -1)
                potential += np.moveaxis(evaluate(xyz)[..., :2], -1, 0)
                continue
            rotation = field.registration.rotation_array
            origin = field.registration.origin_array_m
            x, y = xy[0]-origin[0], xy[1]-origin[1]
            z = self.plane_z_mm*1e-3-origin[2]
            local_x = rotation[0, 0]*x+rotation[1, 0]*y+rotation[2, 0]*z
            local_y = rotation[0, 1]*x+rotation[1, 1]*y+rotation[2, 1]*z
            local_z = rotation[0, 2]*x+rotation[1, 2]*y+rotation[2, 2]*z
            axial = np.zeros_like(local_z)
            for amplitude, centre, sigma in field.terms_t_m:
                axial += amplitude*np.exp(-.5*((local_z-centre)/sigma)**2)
            local_ax, local_ay = -.5*local_y*axial, .5*local_x*axial
            for axis in range(2):
                potential[axis] += rotation[axis, 0]*local_ax+rotation[axis, 1]*local_ay
        return potential


def capture_column_plane_magnetic_gauge(state, plane_z_mm):
    """Freeze the worker's executed optical state, never the live GUI state."""
    from temsim.input_io import input_scope
    from temsim.physics.core import fields
    from temsim.physics.lens_field_provider import (
        MappedLensFieldProvider, active_vector_providers, freeze_vector_provider,
    )
    with input_scope(state, inherit=False):
        providers = active_vector_providers(state)
        keys = tuple(provider.lens_key for provider in providers)
        # A finite imported map outside this observation plane does not make
        # an upstream readout unsupported. Keep every vector key excluded from
        # the aligned reduction, even when its map is not sampled here.
        # Native analytical gauges retain their Gaussian tails; applying this
        # finite-map test to them would invent a new exact-Z field truncation.
        posed = tuple(freeze_vector_provider(provider) for provider in providers
                      if not isinstance(provider, MappedLensFieldProvider)
                      or provider.field_support_mm()[0] <= plane_z_mm <= provider.field_support_mm()[1])
        aligned = float(fields(np.array((plane_z_mm,)), state, exclude_mapped_keys=keys)[0][0])
        from temsim.physics.posed_column_fields import capture_posed_column_fields
        from temsim.simulation_modes import is_ideal
        posed += tuple(field for field in capture_posed_column_fields(state, include_hexapole=not is_ideal(state))
                       if field.field_support_mm[0] <= plane_z_mm <= field.field_support_mm[1])
    return ColumnPlaneMagneticGauge(plane_z_mm, aligned, posed)


def checkpoint_magnetic_gauge(gauge, checkpoint):
    """Bind posed drives to the values actually used by the executed modes.

    A driven coil is frozen at each mode's arrival time during propagation,
    which can differ from the instrument's snapshot time. A single common
    current readout is unavailable when those modes require different gauges;
    their intensity and phase readouts remain available. Never recalculate a
    drive from live controls to fill missing execution metadata.
    """
    from temsim.physics.lens_field_provider import CoordinateRegistration
    from temsim.physics.posed_column_fields import FrozenPosedMultipole
    if gauge is None:
        return None
    gauge.require_plane(checkpoint.plane_z_mm)
    captured = tuple(field for field in gauge.fields if isinstance(field, FrozenPosedMultipole))
    if not captured:
        return gauge
    record = checkpoint.record
    while isinstance(record, Mapping):
        rows = record.get("modes", ())
        if rows and all(isinstance(row, Mapping) and "posed_column_fields" in row for row in rows):
            alternatives = []
            for row in rows:
                fields = tuple(FrozenPosedMultipole(**{**dict(value),
                    "registration": CoordinateRegistration(**value["registration"])})
                    for value in row["posed_column_fields"])
                fields = tuple(field for field in fields
                    if field.field_support_mm[0] <= checkpoint.plane_z_mm <= field.field_support_mm[1])
                if {field.lens_key for field in fields} != {field.lens_key for field in captured}:
                    return None
                alternatives.append(tuple(sorted(fields, key=lambda field: field.lens_key)))
            if any(tuple(field.fingerprint for field in fields) !=
                   tuple(field.fingerprint for field in alternatives[0]) for fields in alternatives[1:]):
                return None
            return replace(gauge, fields=tuple(field for field in gauge.fields
                if not isinstance(field, FrozenPosedMultipole))+alternatives[0])
        record = record.get("upstream")
    return None if any(field.dynamic for field in captured) else gauge


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


def column_mode_observables(mode, *, reference_current_a, axial_bz_t=None,
                            magnetic_gauge=None, plane_z_mm=None,
                            maximum_working_bytes=512*1024**2):
    """Full envelope-gradient plus analytical carrier and magnetic current.

    The mode is the *forward column* contract, not the two-way low-energy
    gun. The latter requires its saved normal derivative and separate
    radial_mode_observables. Reference current remains the physical tip's.
    """
    if (isinstance(reference_current_a, bool) or not math.isfinite(reference_current_a)
            or reference_current_a < 0):
        raise ValueError("Wave diagnostics need a finite tip current and actual axial field")
    if magnetic_gauge is not None:
        if not isinstance(magnetic_gauge, ColumnPlaneMagneticGauge):
            raise ValueError("Probability flow needs an immutable captured magnetic gauge")
        magnetic_gauge.require_plane(plane_z_mm)
    elif (axial_bz_t is None or isinstance(axial_bz_t, bool) or not math.isfinite(axial_bz_t)):
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
    # The complete captured gauge is authoritative. axial_bz_t remains an
    # aligned-only compatibility input, never an additional field contribution.
    vector_potential = (magnetic_gauge.vector_potential_xy_t_m(xy) if magnetic_gauge is not None
                        else .5*axial_bz_t*np.stack((-xy[1], xy[0])))
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
        float(reference_current_a*probability.sum()), mode.axial_reference, mode.scattering_history,
        **({"gauge": "Captured sum of aligned symmetric and rigid analytical vector potentials, laboratory SI; forward paraxial column state"}
           if magnetic_gauge is not None else {}))


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
