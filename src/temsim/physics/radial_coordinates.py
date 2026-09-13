"""Smooth numerical radial charts, without changing a physical source.

Both complex guide ellipses follow the same consumed optical map. A C2
blend of log width and real curvature redistributes radial resolution. Its
EXACT derivatives enter the covariant wave operator; there is no projected
handoff, re-emission, phase fit, probability repair or discarded wave mode.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RadialCoordinateBlend:
    target_width_over_cap_radius: float
    start_mm: float
    end_mm: float

    def validate(self):
        values = (self.target_width_over_cap_radius, self.start_mm, self.end_mm)
        if any(isinstance(value, bool) or not math.isfinite(value) for value in values):
            raise ValueError("Numerical radial chart inputs must be finite numbers")
        if not .001 <= self.target_width_over_cap_radius <= 1.:
            raise ValueError("Numerical target width/cap radius must be in [0.001, 1]")
        if not 0 < self.start_mm < self.end_mm:
            raise ValueError("Numerical chart transition needs 0 < start < end")
        return self


def radial_chart(q, q_prime, reference_k):
    if not math.isfinite(reference_k) or reference_k <= 0 or q.imag <= 0 or not math.isfinite(abs(q)):
        raise ValueError("The numerical radial wave chart is singular")
    if not math.isfinite(abs(q_prime)):
        raise ValueError("The numerical radial chart derivative is non-finite")
    return (math.sqrt(1/(reference_k*q.imag)), q.real,
            -.5*q_prime.imag/q.imag, q_prime.real)


def blended_radial_chart(q, q_prime, target_q, target_prime, reference_k, z_nm, transition=None):
    """Return b, c, b'/b, c' in nm, nm^-1, nm^-1 and nm^-2.

    q is a numerical ellipse, NOT the physical wave. Replacing its chart
    does not replace the complex coefficients solved by the full wave BVP.
    """
    first = radial_chart(q, q_prime, reference_k)
    if transition is None or z_nm <= transition.start_mm*1e6:
        return first
    second = radial_chart(target_q, target_prime, reference_k)
    if z_nm >= transition.end_mm*1e6:
        return second
    t = (z_nm-transition.start_mm*1e6)/((transition.end_mm-transition.start_mm)*1e6)
    weight = t**3*(10+t*(-15+6*t))
    rate = 30*t*t*(1-t)**2/((transition.end_mm-transition.start_mm)*1e6)
    log_ratio = math.log(second[0]/first[0])
    return (first[0]*math.exp(weight*log_ratio),
            (1-weight)*first[1]+weight*second[1],
            (1-weight)*first[2]+weight*second[2]+rate*log_ratio,
            (1-weight)*first[3]+weight*second[3]+rate*(second[1]-first[1]))
