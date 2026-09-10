"""Shared wave convention and Cartesian polynomial basis (lengths in mm).

W = C_nm theta**(n+1) cos(m(phi-azimuth))/(n+1), chi = 2 pi W/lambda.
Wave transfer is exp(-i chi). Angles use the right-handed laboratory X-Y frame.
This matches abTEM C_nm after mm -> Angstrom and degrees -> radians, not an
unqualified mapping to instrument-vendor knob signs.
"""

from dataclasses import dataclass
import numpy as np

ABERRATION_SCHEMA = "cartesian-c56-holdout-fixed-hardware-cc-v1"
UNIMPLEMENTED_TERMS = ("C41", "C43", "C45", "C52", "C54")


@dataclass(frozen=True)
class WaveTerm:
    name: str
    field: str
    power: int
    harmonic: int
    polar_code: str
    azimuth_field: str | None = None

    @property
    def geometric_order(self):
        return self.power - 1


# Order preserves the original thirteen real fit columns; A5 appends two.
WAVE_TERMS = (
    WaveTerm("C1", "c1_mm", 2, 0, "C10"),
    WaveTerm("A1", "a1_mm", 2, 2, "C12", "a1_azimuth_deg"),
    WaveTerm("B2", "b2_mm", 3, 1, "C21", "b2_azimuth_deg"),
    WaveTerm("A2", "a2_mm", 3, 3, "C23", "a2_azimuth_deg"),
    WaveTerm("C3", "c3_mm", 4, 0, "C30"),
    WaveTerm("S3", "s3_mm", 4, 2, "C32", "s3_azimuth_deg"),
    WaveTerm("A3", "a3_mm", 4, 4, "C34", "a3_azimuth_deg"),
    WaveTerm("C5", "c5_mm", 6, 0, "C50"),
    WaveTerm("A5", "a5_mm", 6, 6, "C56", "a5_azimuth_deg"),
)
CARTESIAN_NAMES = tuple(name for term in WAVE_TERMS for name in
                        ((term.name,) if term.harmonic == 0 else (term.name + "_x", term.name + "_y")))
POWERS = np.array([term.power for term in WAVE_TERMS for _ in range(1 if term.harmonic == 0 else 2)])


def cartesian_coefficients(coefficients):
    """Stable coefficient coordinates, including at zero amplitude, in mm."""
    values = []
    for term in WAVE_TERMS:
        value = float(getattr(coefficients, term.field))
        if term.harmonic:
            angle = term.harmonic * np.deg2rad(getattr(coefficients, term.azimuth_field))
            values.extend((value * np.cos(angle), value * np.sin(angle)))
        else:
            values.append(value)
    return np.asarray(values)


def coefficient_fields(cartesian_mm, *, include_focus=True):
    values = np.asarray(cartesian_mm, float)
    if values.shape != (len(CARTESIAN_NAMES),) or not np.isfinite(values).all():
        raise ValueError("Expected fifteen finite Cartesian aberration coefficients in mm")
    result, index = {}, 0
    for term in WAVE_TERMS:
        if term.harmonic:
            value = complex(*values[index:index+2])
            result[term.field] = abs(value)
            result[term.azimuth_field] = float(np.rad2deg(np.angle(value)) / term.harmonic)
            index += 2
        else:
            if include_focus or term.name != "C1":
                result[term.field] = float(values[index])
            index += 1
    return result


def polynomial_basis(angles, *, gradient=False):
    """Real basis or its analytic x/y gradient, shape (N, 15)/(N, 2, 15)."""
    angles = np.asarray(angles, float)
    if angles.ndim != 2 or angles.shape[1] != 2 or not np.isfinite(angles).all():
        raise ValueError("Ray angles must be a finite N by 2 array in radians")
    x, y = angles.T
    q, r2 = x + 1j*y, x*x + y*y
    columns = []
    for term in WAVE_TERMS:
        k, m, p = (term.power - term.harmonic)//2, term.harmonic, term.power
        if gradient:
            radial = k * r2**(k-1) if k else np.zeros_like(r2)
            azimuthal = m * q**(m-1) if m else np.zeros_like(q)
            value = np.stack(((2*x*radial*q**m + r2**k*azimuthal)/p,
                              (2*y*radial*q**m + r2**k*1j*azimuthal)/p), axis=1)
        else:
            value = r2**k * q**m / p
        columns.append(value.real)
        if m:
            columns.append(value.imag)
    return np.stack(columns, axis=-1)


def predict_displacement_m(angles_rad, cartesian_mm):
    """Ray intercept residual = minus gradient of the wave aberration."""
    return -(polynomial_basis(angles_rad, gradient=True) @ np.asarray(cartesian_mm, float)) * 1e-3
