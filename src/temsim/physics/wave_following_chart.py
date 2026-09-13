"""Numerical radial charts derived from an executed, complete complex wave.

Second moments select coordinates, not a Gaussian replacement for the state.
All Laguerre coefficients, their phase and both propagation directions remain
in the subsequently re-solved boundary-value problem. No pilot current or
field is injected at a downstream plane. These charts are development tools;
neither moment matching nor a small occupation certifies convergence.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.interpolate import CubicSpline, make_smoothing_spline

from temsim.immutable_json import json_digest


def radial_wave_moments(coefficients, width_nm, curvature_per_nm, reference_k):
    """Exact m=0 finite-Laguerre expectations, including the analytic chirp.

    With x=r^2/b^2, X_nm=<n|x|m>, D_nm=<n|1+r*d/dr|m>,
    T_nm=<n|-b^2*Laplacian|m>, R2=b^2<X>, C0=Im<D> and
    P2_intrinsic=(<T>-C0^2/<X>)/b^2. The full covariance is
    C=C0+k*c*R2. Minimising <radial order> gives b=(R2/P2_intrinsic)^1/4
    and c=C/(k*R2). Evaluating intrinsic momentum BEFORE adding the
    analytic chirp avoids catastrophic cancellation for a strongly curved wave.
    """
    a = np.asarray(coefficients, complex)
    if a.ndim != 1 or not len(a) or not np.all(np.isfinite(a)):
        raise ValueError("Radial moments require finite complex coefficients")
    if not all(math.isfinite(v) for v in (width_nm, curvature_per_nm, reference_k)) or min(width_nm, reference_k) <= 0:
        raise ValueError("Radial moment coordinates must be finite with positive width and wave number")
    # Scaling for expectation evaluation is not a change to the transported
    # wave or its current, and avoids under/overflow in very weak modes.
    magnitude = float(np.max(abs(a)))
    if magnitude == 0:
        raise ValueError("A zero field has no defined wave-following chart")
    a = a/magnitude
    norm = float(np.vdot(a, a).real)
    n = np.arange(len(a))
    cross = np.conj(a[1:])*a[:-1]*n[1:]
    diagonal = float(np.dot(2*n+1, abs(a)**2))
    x = (diagonal-2*cross.real.sum())/norm
    t = (diagonal+2*cross.real.sum())/norm
    covariance = float(2*cross.imag.sum()/norm)
    intrinsic = (t-covariance*covariance/x)/(width_nm*width_nm)
    r2 = width_nm*width_nm*x
    if not math.isfinite(intrinsic) or intrinsic <= 0 or r2 <= 0:
        raise ValueError("Radial moments lost positive intrinsic momentum; refine their numerical representation")
    emittance = math.sqrt(r2*intrinsic)
    if emittance < 1-1e-10:
        raise ValueError("Radial moments violate the uncertainty bound")
    return {"radius_squared_nm2": float(r2), "intrinsic_momentum_squared_nm2": float(intrinsic),
        "envelope_covariance": covariance,
        "optimal_width_nm": float((r2/intrinsic)**.25),
        "optimal_curvature_per_nm": float(curvature_per_nm+covariance/(reference_k*r2)),
        "minimum_mean_radial_order": float((emittance-1)/2)}


@dataclass(frozen=True)
class WaveFollowingNumerics:
    iterations: int = 0
    transition_start_nm: float = 10.
    transition_end_nm: float = 100.
    width_multiplier: float = 1.
    phase_order: int = 2
    smoothing_log_z_width: float = 0.

    def validate(self):
        if type(self.phase_order) is not int or self.phase_order not in (2, 4):
            raise ValueError("Numerical radial phase order must be 2 or 4")
        if type(self.iterations) is not int or not 0 <= self.iterations <= 8:
            raise ValueError("Wave-following iterations must be an integer in [0, 8]")
        values = (self.transition_start_nm, self.transition_end_nm, self.width_multiplier, self.smoothing_log_z_width)
        if any(isinstance(v, bool) or not math.isfinite(v) for v in values):
            raise ValueError("Wave-following numerical inputs must be finite")
        if not 0 < self.transition_start_nm < self.transition_end_nm or self.width_multiplier <= 0:
            raise ValueError("Wave-following charts require ordered positive transition planes and width multiplier")
        if not 0 <= self.smoothing_log_z_width <= 1:
            raise ValueError("Numerical chart smoothing width must be in [0, 1] log-Z units")
        return self


class ExecutedWaveChart:
    """Ephemeral coordinates from a solved mode, not a source/cache admission.

    Fit log b and curvature over log z; use the spline's exact derivatives.
    A C2 transition leaves the physical tip matching chart untouched. The
    entire gun and reflected tip load MUST be re-executed with these charts.
    """

    def __init__(self, z_nm, widths_nm, curvatures_per_nm, settings, provenance, *, quartic_per_nm4=None):
        settings.validate()
        z, b, c = (np.array(v, dtype=float, copy=True) for v in (z_nm, widths_nm, curvatures_per_nm))
        if (z.ndim != 1 or len(z) < 3 or b.shape != z.shape or c.shape != z.shape
                or not np.all(np.isfinite((z, b, c))) or np.any(np.diff(z) <= 0)
                or min(z.min(), b.min()) <= 0):
            raise ValueError("Executed chart needs ordered positive planes and widths with finite curvatures")
        if not z[0] < settings.transition_start_nm < settings.transition_end_nm < z[-1]:
            raise ValueError("Wave-following transition must be inside the executed gun")
        self.settings = settings
        self.minimum_z_nm, self.maximum_z_nm = float(z[0]), float(z[-1])
        quartic = np.zeros_like(z) if quartic_per_nm4 is None else np.array(quartic_per_nm4, dtype=float)
        if quartic.shape != z.shape or not np.all(np.isfinite(quartic)):
            raise ValueError("Numerical quartic phase must be finite on every chart plane")
        if settings.phase_order != 4 and np.any(quartic):
            raise ValueError("Quartic phase cannot be silently omitted by a quadratic-only chart")
        s = np.log(z)
        self._scaled_phase = settings.smoothing_log_z_width > 0
        if self._scaled_phase:
            if len(z) < 5:
                raise ValueError("Smoothed numerical charts require at least five physical planes")
            weights = np.r_[(s[1]-s[0])/2, (s[2:]-s[:-2])/2, (s[-1]-s[-2])/2]
            # Fixed numerical length, not an automatically fitted source or
            # current. Minimise integral error + width^4 * integral(f''^2).
            # Scale phase by the local radial width to avoid dimensional and
            # near-tip magnitude biases. The physical wave is NOT smoothed.
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_smoothing_spline.html
            smooth = lambda values: make_smoothing_spline(s, values, w=weights,
                lam=settings.smoothing_log_z_width**4)
            self._log_width = smooth(np.log(b))
            self._curvature = smooth(c*b**2)
            self._quartic = smooth(quartic*b**4)
        else:
            self._log_width = CubicSpline(s, np.log(b), extrapolate=False)
            self._curvature = CubicSpline(s, c, extrapolate=False)
            self._quartic = CubicSpline(s, quartic, extrapolate=False)
        self.digest = json_digest({"z_nm": z.tolist(), "width_nm": b.tolist(),
            "curvature_per_nm": c.tolist(), "provenance": provenance,
            "quartic_per_nm4": quartic.tolist(), "phase_order": settings.phase_order,
            "transition_nm": (settings.transition_start_nm, settings.transition_end_nm),
            "width_multiplier": settings.width_multiplier})
        if self._scaled_phase:
            self.digest = json_digest({"unsmoothed_input": self.digest,
                "smoothing_log_z_width": settings.smoothing_log_z_width,
                "method": "log-width-and-scaled-phase-fixed-smoothing-v1"})

    def _phase_value(self, spline, s, power):
        value, derivative = float(spline(s)), float(spline(s, 1))
        if self._scaled_phase:
            scale = math.exp(-power*float(self._log_width(s)))
            derivative = (derivative-power*value*float(self._log_width(s, 1)))*scale
            value *= scale
        return value, derivative

    def evaluate(self, z_nm, base):
        if not math.isfinite(z_nm) or not self.minimum_z_nm <= z_nm <= self.maximum_z_nm:
            raise ValueError("Wave-following chart cannot extrapolate outside its executed interval")
        start, end = self.settings.transition_start_nm, self.settings.transition_end_nm
        if z_nm <= start:
            return base
        s = math.log(z_nm)
        log_b = float(self._log_width(s))
        c, c_s = self._phase_value(self._curvature, s, 2)
        rate, prime = float(self._log_width(s, 1))/z_nm, c_s/z_nm
        if z_nm >= end:
            return math.exp(log_b), c, rate, prime
        t = (z_nm-start)/(end-start)
        weight, weight_prime = t**3*(10+t*(-15+6*t)), 30*t*t*(1-t)**2/(end-start)
        log_ratio = log_b-math.log(base[0])
        return (base[0]*math.exp(weight*log_ratio), (1-weight)*base[1]+weight*c,
            (1-weight)*base[2]+weight*rate+weight_prime*log_ratio,
            (1-weight)*base[3]+weight*prime+weight_prime*(c-base[1]))

    def quartic_phase(self, z_nm):
        if not math.isfinite(z_nm) or not self.minimum_z_nm <= z_nm <= self.maximum_z_nm:
            raise ValueError("Numerical quartic phase cannot extrapolate outside the executed chart")
        start, end = self.settings.transition_start_nm, self.settings.transition_end_nm
        if z_nm <= start:
            return 0., 0.
        value, derivative = self._phase_value(self._quartic, math.log(z_nm), 4)
        prime = derivative/z_nm
        if z_nm >= end:
            return value, prime
        t = (z_nm-start)/(end-start)
        weight, rate = t**3*(10+t*(-15+6*t)), 30*t*t*(1-t)**2/(end-start)
        return weight*value, weight*prime+rate*value


def chart_from_executed_wave(fields, derivatives, frame, settings):
    """Consume all solved boundary states; keep the right trace at a mask.

    The field/normal derivative/physical provenance digest is recorded. A
    moment does not carry phase and is never reused as a physical beam state.
    """
    count = len(frame["rows"])+1
    if fields.shape != derivatives.shape or fields.shape[0] != count:
        raise ValueError("Wave-following chart requires the complete executed field and derivative history")
    z = np.r_[frame["start_nm"], [row["z_nm"] for row in frame["rows"]]]
    coordinates = [(frame["initial_width_nm"], frame["initial_curvature_per_nm"]), *frame["frames"]]
    quartic = [0., *frame.get("quartic_phases_per_nm4", [0.]*(count-1))]
    if len(quartic) != count:
        raise ValueError("Missing executed quartic phase history")
    if settings.phase_order == 4:
        from temsim.physics.quartic_radial_phase import quartic_wave_moments
        moments = [quartic_wave_moments(a, *coordinate, frame["reference_wave_number_per_nm"], gamma)
            for a, coordinate, gamma in zip(fields, coordinates, quartic)]
    else:
        if any(quartic):
            raise ValueError("Cannot discard an executed quartic phase in quadratic-only adaptation")
        moments = [radial_wave_moments(a, *coordinate, frame["reference_wave_number_per_nm"])
            for a, coordinate in zip(fields, coordinates)]
    indices = np.r_[np.flatnonzero(np.diff(z) > 0), len(z)-1]
    provenance = {"pilot_numerics": frame["numerics"], "previous_chart": frame.get("executed_chart_digest"),
        "pilot_fields": {"real": fields.real.tolist(), "imag": fields.imag.tolist()},
        "pilot_derivatives": {"real": derivatives.real.tolist(), "imag": derivatives.imag.tolist()},
        "coordinates": coordinates, "quartic_phases_per_nm4": quartic,
        "z_nm": z.tolist(), "k_ref": frame["reference_wave_number_per_nm"]}
    chart = ExecutedWaveChart(z[indices], [moments[j]["optimal_width_nm"]*settings.width_multiplier for j in indices],
        [moments[j]["optimal_curvature_per_nm"] for j in indices], settings, provenance,
        quartic_per_nm4=[moments[j].get("optimal_quartic_phase_per_nm4", 0.) for j in indices])
    return chart, {"chart_digest": chart.digest, "boundary_count": count, "unique_planes": len(indices),
        "maximum_minimum_mean_radial_order": max(m["minimum_mean_radial_order"] for m in moments),
        "scope": "Numerical coordinates only; complete reflected tip/gun problem re-executed, no downstream source"}
