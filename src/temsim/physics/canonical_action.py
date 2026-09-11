"""Phase of an affine canonical path, not merely its endpoint ray matrix.

Ordering is (x, y, p_x/p0, p_y/p0), in metres and dimensionless momentum.
Weyl displacement is W(q,p) f(x)=exp(i k p.(x-q/2)) f(x-q), k=2*pi/lambda.
An operator is exp(i k action_m) W(offset) U(matrix). The scalar action and
the metaplectic lift are independent: neither is determined by a ray endpoint.

The lift is followed with det(A+i B/length), using a positive imaginary
reference Gaussian. This auxiliary length only selects a coordinate chart;
it is not a physical source or an input to an illumination model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from temsim.physics.canonical_phase import validate_canonical_map


def reference_frame(matrix, length_m):
    if not math.isfinite(length_m) or length_m <= 0:
        raise ValueError("Metaplectic reference length must be finite and positive")
    matrix = np.asarray(matrix, float)
    return matrix[:2, :2] + 1j*matrix[:2, 2:]/length_m


def principal_reference_phase(matrix, length_m):
    """One endpoint lift; production paths must supply their continuous lift."""
    sign, logabs = np.linalg.slogdet(reference_frame(matrix, length_m))
    if sign == 0 or not np.isfinite(logabs):
        raise ValueError("Singular metaplectic reference frame")
    return -.5*float(np.angle(sign))


def validate_reference_phase(matrix, length_m, phase_rad):
    if not math.isfinite(phase_rad):
        raise ValueError("Metaplectic phase must be finite")
    principal = principal_reference_phase(matrix, length_m)
    # Two lifts differ by pi, not by an arbitrary fitted phase.
    if abs(np.exp(2j*(phase_rad-principal))-1) > 1e-8:
        raise ValueError("Metaplectic phase is inconsistent with its canonical endpoint")


def drift_gaussian_phase(drift, precision):
    """Centre phase of a continuous spectral drift of a complex Gaussian.

    precision has positive imaginary part. Each eigenvalue starts at one as
    the real symmetric drift grows from zero; sum individual arguments rather
    than wrapping the determinant's argument and losing a complete turn.
    """
    values = np.linalg.eigvals(np.eye(2) + np.asarray(drift)@precision)
    if not np.all(np.isfinite(values)) or np.any(abs(values) == 0):
        raise ValueError("Unresolved Gaussian reference for the spectral drift")
    return -.5*float(np.sum(np.angle(values)))


def affine_centre_action(matrix, offset, origin, tilt, action_m=0.):
    """Change from Weyl to the local-centre phase convention of PlaneWave."""
    if not math.isfinite(action_m):
        raise ValueError("Affine scalar action must be finite")
    centre = np.r_[origin, tilt]
    mapped = np.asarray(matrix)@centre
    q, p = mapped[:2], mapped[2:]
    dq, dp = np.asarray(offset)[:2], np.asarray(offset)[2:]
    return float(action_m + .5*(p@q-tilt@origin) + dp@q + .5*dp@dq)


@dataclass
class CanonicalPath:
    """Sequential physical maps with a phase lift and Weyl scalar action.

    append() requires resolved increments: an individual reference-frame
    eigenphase must remain below pi/2. Violations require finer physical path
    sampling, not a guessed branch or a repaired endpoint matrix.
    """
    reference_length_m: float
    matrix: np.ndarray = field(default_factory=lambda: np.eye(4), init=False)
    offset: np.ndarray = field(default_factory=lambda: np.zeros(4), init=False)
    action_m: float = field(default=0., init=False)
    reference_phase_rad: float = field(default=0., init=False)

    def __post_init__(self):
        reference_frame(self.matrix, self.reference_length_m)

    def append(self, matrix, offset=None, *, action_m=0.):
        validate_canonical_map(matrix)
        matrix = np.asarray(matrix, float)
        displacement = np.zeros(4) if offset is None else np.asarray(offset, float)
        if (displacement.shape != (4,) or not np.all(np.isfinite(displacement))
                or not math.isfinite(action_m)):
            raise ValueError("Canonical path requires a finite displacement and action")
        combined = matrix@self.matrix
        old_frame = reference_frame(self.matrix, self.reference_length_m)
        new_frame = reference_frame(combined, self.reference_length_m)
        relative = np.linalg.solve(old_frame.T, new_frame.T).T
        arguments = np.angle(np.linalg.eigvals(relative))
        if not np.all(np.isfinite(arguments)) or np.any(abs(arguments) >= np.pi/2):
            raise ValueError("Canonical phase path is undersampled; refine axial steps")
        propagated_offset = matrix@self.offset
        # W(d2) W(M2 d1) has this central Weyl phase. A momentum kick is
        # exp(i k kick.x); its action cannot be recovered from d alone later.
        central = .5*(displacement[2:]@propagated_offset[:2]
                       - displacement[:2]@propagated_offset[2:])
        candidate_action = self.action_m + float(action_m + central)
        candidate_phase = self.reference_phase_rad - .5*float(np.sum(arguments))
        candidate_offset = propagated_offset+displacement
        if not math.isfinite(candidate_action) or not np.all(np.isfinite(candidate_offset)):
            raise ValueError("Canonical path accumulation must remain finite")
        validate_reference_phase(combined, self.reference_length_m, candidate_phase)
        # Commit only after all checks: a rejected increment must not leave a
        # partially advanced path available to the next propagation attempt.
        self.action_m = candidate_action
        self.reference_phase_rad = candidate_phase
        self.offset = candidate_offset
        self.matrix = combined

    def phase_kwargs(self):
        return {"affine_action_m": self.action_m,
                "reference_phase_rad": self.reference_phase_rad,
                "reference_length_m": self.reference_length_m}
