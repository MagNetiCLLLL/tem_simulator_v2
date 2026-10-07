"""Quadratic canonical wave Hamiltonians for rigid analytical magnetic lenses.

The gauge of each native first-order lens is A=(-y B0/2, x B0/2, 0),
rigidly transformed with its field. Its curl is exactly the B model consumed
by FrozenAnalyticField (not a higher-order Maxwell lens solution). In global
XY slices the stationary electron Hamiltonian, divided by entrance momentum,
is h=|p0*u+e*A_xy|^2/(2*p*p0)+e*A_z/p0, where u=P_xy/p0 is CANONICAL
momentum, e>0, and p is local relativistic mechanical momentum. Thus the
quadratic approximation must include A_z, affine terms and scalar action.

The cubic/higher spatial remainder is bounded on an executed support box.
That bound certifies this Taylor truncation only, not the pre-existing
paraxial kinetic approximation, field model or axial integration accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.constants import c, e, m_e

from temsim.physics.lens_field_provider import FrozenAnalyticField


_J = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
_CROSS_Z = np.array(((0., -.5, 0.), (.5, 0., 0.), (0., 0., 0.)))


def require_analytic_fields(fields):
    """Admit only immutable fields with the native analytical gauge available."""
    from temsim.physics.posed_column_fields import FrozenPosedMultipole
    result = tuple(fields)
    if any(not isinstance(field, (FrozenAnalyticField, FrozenPosedMultipole)) for field in result):
        raise ValueError("Coherent posed fields require native analytical vector potentials; "
                         "imported 3-D magnetic maps need a consistent vector potential")
    return result


def _axial_derivatives(terms, z):
    value = first = second = 0.
    for amplitude, centre, sigma in terms:
        u = (z-centre)/sigma
        g = amplitude*math.exp(-.5*u*u)
        value += g
        first -= g*u/sigma
        second += g*(u*u-1)/(sigma*sigma)
    return value, first, second


def vector_potential_jet(fields, position_m):
    """Return global A, dA_i/dx_j, d2A_i/dx_j dx_k, in SI units.

    The complete three-dimensional derivatives allow independent curl tests;
    propagation uses their XY restrictions. Every field retains its actual
    rigid registration and Gaussian tail; no nominal axial clipping is added.
    """
    fields = require_analytic_fields(fields)
    position = np.asarray(position_m, float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("Vector potential requires a finite global XYZ position")
    value, gradient, hessian = np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3, 3))
    for field in fields:
        if not isinstance(field, FrozenAnalyticField):
            a, d, dd = field.vector_potential_jet(position)
            value += a
            gradient += d
            hessian += dd
            continue
        rotation = field.registration.rotation_array
        local = field.registration.positions_global_to_local_m(position)
        b, db, ddb = _axial_derivatives(field.terms_t_m, local[2])
        c = _CROSS_Z@local
        local_gradient = _CROSS_Z*b
        local_gradient[:, 2] += c*db
        local_hessian = np.zeros((3, 3, 3))
        local_hessian[:, :, 2] += _CROSS_Z*db
        local_hessian[:, 2, :] += _CROSS_Z*db
        local_hessian[:, 2, 2] += c*ddb
        value += rotation@(c*b)
        gradient += rotation@local_gradient@rotation.T
        hessian += np.einsum("ai,ijk,bj,ck->abc", rotation, local_hessian, rotation, rotation)
    return value, gradient, hessian


def _magnetic_hamiltonian_jet(value, gradient, hessian, momentum, entrance_momentum):
    a, d, dd = value[:2], gradient[:2, :2], hessian[:2, :2, :2]
    coefficient = e*e/(momentum*entrance_momentum)
    constant = .5*coefficient*(a@a)+e*value[2]/entrance_momentum
    linear = np.r_[coefficient*d.T@a+e*gradient[2, :2]/entrance_momentum,
                   e*a/momentum]
    curvature = np.zeros((4, 4))
    curvature[:2, :2] = coefficient*(d.T@d+np.einsum("i,ijk->jk", a, dd))
    curvature[:2, :2] += e*hessian[2, :2, :2]/entrance_momentum
    curvature[:2, 2:] = e*d.T/momentum
    curvature[2:, :2] = curvature[:2, 2:].T
    # The kinetic u^2 drift belongs to the existing column operator, so it is
    # deliberately absent from both full and aligned-only magnetic jets.
    return constant, linear, curvature


def _inverse_momentum_derivatives(energy_ev):
    """1/p and its first three energy derivatives (energy coordinate is eV)."""
    energy = energy_ev*e
    p = math.sqrt(energy*(energy+2*m_e*c*c))/c
    velocity = p*c*c/(energy+m_e*c*c)
    first = -e/(velocity*p*p)
    second = e*e*(2/(velocity*velocity*p**3)+m_e*m_e/p**5)
    third = -e**3*(6/(velocity**3*p**4)+9*m_e*m_e/(velocity*p**6))
    return 1/p, first, second, third


def _inverse_momentum_spatial_jet(momentum, potential_gradient, potential_hessian, energy_ev):
    g = np.zeros(2) if potential_gradient is None else np.asarray(potential_gradient, float)
    h = np.zeros((2, 2)) if potential_hessian is None else np.asarray(potential_hessian, float)
    if (g.shape != (2,) or h.shape != (2, 2) or not np.isfinite(g).all()
            or not np.isfinite(h).all() or not np.allclose(h, h.T, rtol=1e-12, atol=0.)):
        raise ValueError("Scalar-potential derivatives must be finite XY vectors and symmetric Hessians")
    if energy_ev is None:
        # Stable relativistic kinetic energy, including at low momentum.
        rest = m_e*c*c
        energy_ev = (momentum*c)**2/(math.sqrt((momentum*c)**2+rest*rest)+rest)/e
    if not math.isfinite(energy_ev) or energy_ev <= 0.:
        raise ValueError("Posed-lens kinetic energy must be finite and positive")
    inverse, first, second, _ = _inverse_momentum_derivatives(energy_ev)
    if not math.isclose(inverse*momentum, 1., rel_tol=2e-12):
        raise ValueError("Posed-lens kinetic energy and momentum must describe the same electron")
    return g.copy(), h.copy(), float(energy_ev), first*g, first*h+second*np.outer(g, g)


def _gaussian_derivative_bounds(terms, lower, upper):
    """Suprema of |B''| and |B'''| over a finite interval, term by term.

    Extrema are endpoints or roots of the next Hermite polynomial. Summing
    absolute per-Gaussian maxima remains conservative for overlapping terms.
    """
    second = third = 0.
    roots_second = (0., -math.sqrt(3.), math.sqrt(3.))
    roots_third = tuple(sign*math.sqrt(3.+s*math.sqrt(6.))
                        for s in (-1., 1.) for sign in (-1., 1.))
    for amplitude, centre, sigma in terms:
        lo, hi = (lower-centre)/sigma, (upper-centre)/sigma
        candidates2 = (lo, hi, *(x for x in roots_second if lo <= x <= hi))
        candidates3 = (lo, hi, *(x for x in roots_third if lo <= x <= hi))
        second += abs(amplitude)/sigma**2*max(abs((x*x-1)*math.exp(-.5*x*x)) for x in candidates2)
        third += abs(amplitude)/sigma**3*max(abs((3*x-x**3)*math.exp(-.5*x*x)) for x in candidates3)
    return second, third


def _potential_remainder_bound(fields, z_m, xy_bound, centre_xy=(0., 0.)):
    """Euclidean A and scalar A_z remainders about a declared global centre."""
    remainder = axial_remainder = 0.
    for field in fields:
        if not isinstance(field, FrozenAnalyticField):
            from temsim.physics.posed_multipole_wave import multipole_potential_remainder_bound
            bound, axial_bound = multipole_potential_remainder_bound(
                field, np.array((*centre_xy, z_m)), xy_bound)
            remainder += bound
            axial_remainder += axial_bound
            continue
        rotation = field.registration.rotation_array
        local = field.registration.positions_global_to_local_m(np.array((*centre_xy, z_m)))
        local_displacement = abs(rotation[:2, :].T)@xy_bound
        dz = local_displacement[2]
        if dz == 0.:
            continue  # Translated parallel lenses are exactly affine in XY.
        dr = np.linalg.norm(local_displacement[:2])
        radius = np.linalg.norm(abs(local[:2])+local_displacement[:2])
        second, third = _gaussian_derivative_bounds(field.terms_t_m, local[2]-dz, local[2]+dz)
        bound = (radius*third*dz**3+3*dr*second*dz**2)/12.
        remainder += bound
        axial_remainder += np.linalg.norm(rotation[2, :2])*bound
    return remainder, axial_remainder


def _exponential_tail(value, order, xp):
    """exp(x) minus its degree-order Taylor polynomial, stably near zero."""
    # The series is used only on a small interval. Through x^12 its omitted
    # absolute term is below 2e-48 there, well below double roundoff.
    series = xp.zeros_like(value)
    for power in range(12, order, -1):
        series = series*value+1/math.factorial(power)
    series *= value**(order+1)
    direct = xp.expm1(value)-value
    if order == 2:
        direct -= .5*value*value
    return xp.where(abs(value) < 1e-3, series, direct)


def _vector_potential_remainders(fields, z_m, centre_xy, delta_xy, xp):
    """Exact A-A0-A1 and A-A0-A1-A2 on a device lattice.

    Gaussian differences are evaluated with expm1 and short local series.
    Thus an almost parallel lens does not lose its higher spatial orders
    through subtraction of three large vector-potential arrays. Aligned A
    is affine and has identically zero remainder.
    """
    shape = (3, *delta_xy.shape[1:])
    delta_a, remainder_a = xp.zeros(shape, dtype=xp.float64), xp.zeros(shape, dtype=xp.float64)
    for field in fields:
        if not isinstance(field, FrozenAnalyticField):
            from temsim.physics.posed_multipole_wave import multipole_vector_potential_remainders
            second, third = multipole_vector_potential_remainders(
                field, np.array((*centre_xy, z_m)), delta_xy, xp)
            delta_a += second
            remainder_a += third
            continue
        rotation = field.registration.rotation_array
        local_centre = field.registration.positions_global_to_local_m((*centre_xy, z_m))
        local_delta = xp.einsum("ij,jyx->iyx", xp.asarray(rotation[:2, :].T), delta_xy)
        c0 = xp.asarray(_CROSS_Z@local_centre)[:, None, None]
        dc = xp.einsum("ij,jyx->iyx", xp.asarray(_CROSS_Z), local_delta)
        first_difference = xp.zeros_like(local_delta[2])
        second_difference = xp.zeros_like(first_difference)
        third_difference = xp.zeros_like(first_difference)
        for amplitude, centre, sigma in field.terms_t_m:
            u = (local_centre[2]-centre)/sigma
            t = local_delta[2]/sigma
            exponent = -u*t-.5*t*t
            g0 = amplitude*math.exp(-.5*u*u)
            first_difference += g0*xp.expm1(exponent)
            second_difference += g0*(_exponential_tail(exponent, 1, xp)-.5*t*t)
            third_difference += g0*(_exponential_tail(exponent, 2, xp)+.5*u*t**3+.125*t**4)
        delta_a += xp.einsum("ij,jyx->iyx", xp.asarray(rotation),
                            dc*first_difference+c0*second_difference)
        remainder_a += xp.einsum("ij,jyx->iyx", xp.asarray(rotation),
                                dc*second_difference+c0*third_difference)
    return delta_a, remainder_a


@dataclass(frozen=True)
class PosedLensCorrection:
    """Magnetic addition to the existing global canonical generator.

    x'=generator*x+force. The Weyl scalar action rate also contains
    -constant_h; omitting it preserves ray centres but corrupts interference.
    """
    generator: np.ndarray
    force: np.ndarray
    constant_h: float
    fields: tuple
    z_m: float
    momentum: float
    entrance_momentum: float
    vector_potential: np.ndarray
    potential_gradient: np.ndarray
    potential_hessian: np.ndarray
    potential_gradient_v_m: np.ndarray
    potential_hessian_v_m2: np.ndarray
    kinetic_energy_ev: float
    inverse_momentum_gradient: np.ndarray
    inverse_momentum_hessian: np.ndarray
    center_phase_space: np.ndarray
    aligned_b_t: float
    axis_momentum_magnetic: bool = False

    @property
    def supports_magnetic_residual(self):
        """Whether the higher spatial magnetic operator has the same model.

        In the axis-p approximation its electric omission still needs its
        separate budget check; a magnetic residual cannot remove that error.
        """
        return (self.axis_momentum_magnetic or
                not (np.any(self.potential_gradient_v_m) or np.any(self.potential_hessian_v_m2)))

    def reexpanded(self, center_phase_space):
        """Recompute the same magnetic model around another phase-space point.

        A transverse-varying scalar potential retains the existing global
        expansion and its mixed-term bound. This numerical fallback neither
        removes that electric field nor changes its physical parameters.
        """
        centre = (None if not self.axis_momentum_magnetic and
                  (np.any(self.potential_gradient_v_m) or np.any(self.potential_hessian_v_m2))
                  else center_phase_space)
        return posed_lens_correction(self.fields, self.z_m, self.momentum, self.entrance_momentum,
            aligned_b_t=self.aligned_b_t, potential_gradient_v_m=self.potential_gradient_v_m,
            potential_hessian_v_m2=self.potential_hessian_v_m2,
            kinetic_energy_ev=self.kinetic_energy_ev, center_phase_space=centre,
            axis_momentum_magnetic=self.axis_momentum_magnetic)

    def residual_coefficients(self, wave, wavelength, xp=None):
        """Exact native magnetic residual as Hermitian envelope coefficients.

        Returns v[2,ny,nx], W[ny,nx] for
        H_residual = W + 1/2 {v, -i/k grad}, k=2*pi/wavelength.
        Both have dimensionless canonical Hamiltonian units. W is evaluated
        on this wave's retained quadratic phase carrier. This retains all
        spatial orders of the native analytical gauge; it does not extend
        the underlying paraxial kinetic or analytical magnetic field model.
        """
        if not self.supports_magnetic_residual:
            raise ValueError("Higher-order posed magnetic propagation requires a transversely "
                             "constant scalar potential; varying electric-magnetic residuals "
                             "are not supported")
        if not math.isfinite(wavelength) or wavelength <= 0.:
            raise ValueError("Magnetic residual requires a finite positive wavelength")
        if xp is None:
            from temsim.physics.wave_device import array_module
            xp = array_module(wave.amplitude)
        xy = xp.asarray(wave.coordinates_m())
        dq = xy-xp.asarray(self.center_phase_space[:2])[:, None, None]
        curvature = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
        tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
        carrier = (xp.asarray(tilt)[:, None, None]
                   +xp.einsum("ij,jyx->iyx", xp.asarray(curvature),
                               xy-xp.asarray(wave.origin_m)[:, None, None]))
        du = carrier-xp.asarray(self.center_phase_space[2:])[:, None, None]
        delta_a, remainder_a = _vector_potential_remainders(
            self.fields, self.z_m, self.center_phase_space[:2], dq, xp)
        first_a = xp.einsum("ij,jyx->iyx", xp.asarray(self.potential_gradient[:2, :2]), dq)
        constant = xp.asarray(self.entrance_momentum*self.center_phase_space[2:]
                               +e*self.vector_potential[:2])[:, None, None]
        linear = self.entrance_momentum*du+e*first_a
        magnetic_delta = e*delta_a[:2]
        scalar = xp.sum(2*linear*magnetic_delta+magnetic_delta*magnetic_delta
                        +2*constant*e*remainder_a[:2], axis=0)/(2*self.momentum*self.entrance_momentum)
        scalar += e*remainder_a[2]/self.entrance_momentum
        return magnetic_delta/self.momentum, scalar

    def unresolved_electric_hamiltonian_error_bound(self, position_bound_m, canonical_momentum_bound):
        """Bound the magnetic terms omitted by replacing 1/p(phi) with 1/p_axis.

        The captured scalar-potential coefficients stay in laboratory XY.
        The existing pure-electric column operator is neither removed nor
        duplicated. A_z has no 1/p factor, so has no contribution here.
        This error survives application of the full spatial magnetic residual.
        """
        if not self.axis_momentum_magnetic:
            return 0.
        x, u = np.asarray(position_bound_m, float), np.asarray(canonical_momentum_bound, float)
        if (x.shape != (2,) or u.shape != (2,) or not np.isfinite(x).all()
                or not np.isfinite(u).all() or np.any(x < 0) or np.any(u < 0)):
            raise ValueError("Posed-lens support bounds must be finite nonnegative XY vectors")
        global_x = abs(self.center_phase_space[:2])+x
        energy_radius = (abs(self.potential_gradient_v_m)@global_x
                         +.5*global_x@abs(self.potential_hessian_v_m2)@global_x)
        if energy_radius == 0.:
            return 0.
        lower = self.kinetic_energy_ev-energy_radius
        if lower <= 0.:
            raise ValueError("Tilted-lens wave support crosses zero kinetic energy in the captured "
                             "transverse scalar potential; the forward operator was not applied")
        # |d(1/p)/dE| decreases at positive energy. The mean-value bound
        # remains stable even when the default electric tail changes p by
        # much less than one floating-point unit.
        inverse_difference = abs(_inverse_momentum_derivatives(lower)[1])*energy_radius
        remainder, _ = _potential_remainder_bound(self.fields, self.z_m, x, self.center_phase_space[:2])
        a_bound = (np.linalg.norm(abs(self.vector_potential[:2])
                   +abs(self.potential_gradient[:2, :2])@x
                   +.5*np.einsum("ijk,j,k->i", abs(self.potential_hessian[:2, :2, :2]), x, x))
                   +remainder)
        u_bound = np.linalg.norm(abs(self.center_phase_space[2:])+u)
        bound = inverse_difference*(e*a_bound*u_bound
                                    +e*e*a_bound*a_bound/(2*self.entrance_momentum))
        return float(np.nextafter(bound*(1+64*np.finfo(float).eps), np.inf)) if bound else 0.

    def unresolved_electric_phase_error_bound(self, position_bound_m, canonical_momentum_bound,
                                             distance_m, wavelength_m):
        if (not math.isfinite(distance_m) or not math.isfinite(wavelength_m)
                or distance_m < 0 or wavelength_m <= 0):
            raise ValueError("Phase bounds require nonnegative distance and positive wavelength")
        return (self.unresolved_electric_hamiltonian_error_bound(position_bound_m, canonical_momentum_bound)
                *distance_m*2*np.pi/wavelength_m)

    def _inverse_momentum_remainder(self, position_bound):
        """Third-order remainder of 1/p(phi), for the captured scalar jet.

        Its energy derivative magnitudes decrease monotonically for positive
        kinetic energy. Evaluating them at the lowest supported energy is a
        conservative interval bound, including curved scalar potentials.
        """
        linear_energy = abs(self.potential_gradient_v_m)@position_bound
        second_energy = position_bound@abs(self.potential_hessian_v_m2)@position_bound
        energy_radius = linear_energy+.5*second_energy
        if energy_radius == 0.:
            return 0.
        lower = self.kinetic_energy_ev-energy_radius
        if lower <= 0.:
            raise ValueError("Tilted-lens wave support crosses zero kinetic energy in the captured "
                             "transverse scalar potential; the forward operator was not applied")
        _, _, second, third = _inverse_momentum_derivatives(lower)
        first_along_ray = linear_energy+second_energy
        return (abs(third)*first_along_ray**3+3*second*first_along_ray*second_energy)/6.

    def hamiltonian_error_bound(self, position_bound_m, canonical_momentum_bound):
        """Absolute h error over |XY-Xc|<=x, |P_xy/p0-Uc|<=u.

        Bounds must cover the executed wave lattice and its represented phase
        carrier plus residual Fourier support. They are not fitted RMS widths.
        """
        x = np.asarray(position_bound_m, float)
        u = np.asarray(canonical_momentum_bound, float)
        if (x.shape != (2,) or u.shape != (2,) or not np.isfinite(x).all()
                or not np.isfinite(u).all() or np.any(x < 0) or np.any(u < 0)):
            raise ValueError("Posed-lens support bounds must be finite nonnegative XY vectors")
        d = self.potential_gradient[:2, :2]
        dd = self.potential_hessian[:2, :2, :2]
        constant = np.linalg.norm(self.entrance_momentum*self.center_phase_space[2:]
                                  +e*self.vector_potential[:2])
        linear = np.linalg.norm(self.entrance_momentum*u+e*abs(d)@x)
        quadratic = .5*e*np.linalg.norm(np.einsum("ijk,j,k->i", abs(dd), x, x))
        remainder, axial_remainder = _potential_remainder_bound(
            self.fields, self.z_m, x, self.center_phase_space[:2])
        remainder *= e
        numerator = (2*linear*quadratic+quadratic*quadratic
                     +2*(constant+linear+quadratic)*remainder+remainder*remainder)
        bound = numerator/(2*self.momentum*self.entrance_momentum)+e*axial_remainder/self.entrance_momentum
        if self.axis_momentum_magnetic:
            bound += self.unresolved_electric_hamiltonian_error_bound(x, u)
            return float(np.nextafter(bound*(1+64*np.finfo(float).eps), np.inf)) if bound else 0.
        inverse_linear = abs(self.inverse_momentum_gradient)@x
        inverse_quadratic = .5*x@abs(self.inverse_momentum_hessian)@x
        inverse_remainder = self._inverse_momentum_remainder(x)
        # For N=|p0*u+e*A|^2 and f=1/p(phi), retain N0*f1,
        # N1*f1 and N0*f2; the following bounds every remaining product.
        n1 = 2*constant*linear
        n2 = linear*linear+2*constant*quadratic
        mixed = (inverse_linear*(n2+numerator)
                 +inverse_quadratic*(n1+n2+numerator)
                 +inverse_remainder*(constant+linear+quadratic+remainder)**2)
        bound += mixed/(2*self.entrance_momentum)
        return float(np.nextafter(bound*(1+64*np.finfo(float).eps), np.inf)) if bound else 0.

    def phase_error_bound(self, position_bound_m, canonical_momentum_bound, distance_m, wavelength_m):
        """Conservative local discarded-Hamiltonian phase, in radians.

        Axial integration needs its own step convergence check. A per-metre
        policy applied to this bound is invariant under checkpoint grouping.
        """
        if (not math.isfinite(distance_m) or not math.isfinite(wavelength_m)
                or distance_m < 0 or wavelength_m <= 0):
            raise ValueError("Phase bounds require nonnegative distance and positive wavelength")
        return self.hamiltonian_error_bound(position_bound_m, canonical_momentum_bound)*distance_m*2*np.pi/wavelength_m

    def support_bounds(self, position_bound_m, canonical_momentum_bound):
        """Conservative global/local paraxial direction bounds on the same box.

        The mechanical transverse momentum is p0*u+e*A. Its norm divided by
        the smallest supported total p is rho; the actual slope bound is
        rho/sqrt(1-rho^2). The
        kinetic expansion's omitted/retained transverse term ratio is exactly
        rho^2/(1+sqrt(1-rho^2))^2. This is not a total propagated phase error.
        Native radial_support_m bounds the provider's geometric support, not
        a fitted field-validity radius, so it is reported but not enforced here.
        """
        x, u = np.asarray(position_bound_m, float), np.asarray(canonical_momentum_bound, float)
        if (x.shape != (2,) or u.shape != (2,) or not np.isfinite(x).all()
                or not np.isfinite(u).all() or np.any(x < 0) or np.any(u < 0)):
            raise ValueError("Posed-lens support bounds must be finite nonnegative XY vectors")
        remainder, _ = _potential_remainder_bound(self.fields, self.z_m, x, self.center_phase_space[:2])
        a_delta_bound = (abs(self.potential_gradient[:2, :2])@x
                   +.5*np.einsum("ijk,j,k->i", abs(self.potential_hessian[:2, :2, :2]), x, x)
                   +remainder)
        global_x = abs(self.center_phase_space[:2])+x
        energy_radius = (abs(self.potential_gradient_v_m)@global_x
                         +.5*global_x@abs(self.potential_hessian_v_m2)@global_x)
        if self.kinetic_energy_ev <= energy_radius:
            raise ValueError("Tilted-lens wave support crosses zero kinetic energy in the captured "
                             "transverse scalar potential; the forward operator was not applied")
        minimum_p = (1/_inverse_momentum_derivatives(self.kinetic_energy_ev-energy_radius)[0]
                     if energy_radius else self.momentum)
        mechanical = (abs(self.entrance_momentum*self.center_phase_space[2:]+e*self.vector_potential[:2])
                      +self.entrance_momentum*u+e*a_delta_bound)/minimum_p
        rho = float(np.linalg.norm(mechanical))
        denominator = math.sqrt(1-rho*rho) if rho < 1. else 0.
        slopes = mechanical/denominator if denominator else np.full(2, np.inf)
        local = []
        for field in self.fields:
            if not field.scale:
                continue
            rotation = field.registration.rotation_array
            direction_z_lower = (float(rotation[2, 2]-abs(rotation[:2, 2])@slopes)
                                 if denominator else -math.inf)
            transverse = (abs(rotation[2, :2])+abs(rotation[:2, :2].T)@slopes
                          if denominator else np.full(2, np.inf))
            local_position = field.registration.positions_global_to_local_m((*self.center_phase_space[:2], self.z_m))
            radius = float(np.linalg.norm(abs(local_position[:2])+abs(rotation[:2, :2].T)@x))
            local.append({"lens_key": field.lens_key,
                          "forward_direction_lower_bound": direction_z_lower,
                          "transverse_over_longitudinal_bound": (
                              float(np.linalg.norm(transverse))/direction_z_lower
                              if direction_z_lower > 0 else math.inf),
                          "radius_bound_m": radius,
                          "provider_radial_support_m": field.radial_support_m})
        return {"global_transverse_over_longitudinal_bound": rho/denominator if denominator else math.inf,
                "kinetic_relative_remainder_bound": rho*rho/(1+denominator)**2 if denominator else math.inf,
                "local": tuple(local)}


def posed_lens_correction(fields, z_m, momentum, entrance_momentum, *, aligned_b_t=0.,
                          potential_gradient_v_m=None, potential_hessian_v_m2=None,
                          kinetic_energy_ev=None, center_phase_space=None,
                          axis_momentum_magnetic=False):
    """Full magnetic jet minus the aligned-only jet already in column_wave.

    The sum of vector potentials is squared BEFORE subtracting the baseline,
    retaining interference/cross terms between posed and aligned lens fields.
    """
    fields = require_analytic_fields(fields)
    if (not np.isfinite((z_m, momentum, entrance_momentum, aligned_b_t)).all()
            or min(momentum, entrance_momentum) <= 0):
        raise ValueError("Posed lens Hamiltonian needs finite Z/field and positive momenta")
    centre = np.zeros(4) if center_phase_space is None else np.array(center_phase_space, float, copy=True)
    if centre.shape != (4,) or not np.isfinite(centre).all():
        raise ValueError("Magnetic expansion centre must be finite global XY and canonical momentum")
    eg, eh, energy, inverse_gradient, inverse_hessian = _inverse_momentum_spatial_jet(
        momentum, potential_gradient_v_m, potential_hessian_v_m2, kinetic_energy_ev)
    if np.any(centre) and (np.any(eg) or np.any(eh)) and not axis_momentum_magnetic:
        raise ValueError("Moving magnetic expansion centre requires transversely constant scalar potential; "
                         "retain the global electric-magnetic expansion for this interval")
    if axis_momentum_magnetic:
        # Preserve eg/eh as immutable captured inputs for the separate
        # omission budget. Only the magnetic 1/p Taylor jet is disabled;
        # the caller's electric transport is unchanged.
        inverse_gradient = np.zeros(2)
        inverse_hessian = np.zeros((2, 2))
    value, gradient, hessian = vector_potential_jet(fields, (*centre[:2], z_m))
    aligned_gradient = _CROSS_Z*aligned_b_t
    value += aligned_gradient@np.r_[centre[:2], 0.]
    gradient += aligned_gradient
    constant, linear, curvature = _magnetic_hamiltonian_jet(value, gradient, hessian, momentum, entrance_momentum)
    # The magnetic h is linear in canonical u. A nonzero centre momentum
    # contributes to the position Hessian through u_c dot d2A/dq2.
    constant += e/momentum*(value[:2]@centre[2:])
    linear[:2] += e/momentum*gradient[:2, :2].T@centre[2:]
    curvature[:2, :2] += e/momentum*np.einsum("i,ijk->jk", centre[2:], hessian[:2, :2, :2])
    # A translated/tilted field has A_xy(axis)!=0. In that case spatial
    # variation of 1/p(phi) contributes at first and second order too, rather
    # than only the higher-order terms of an axially centred field.
    a = value[:2]
    d_a = gradient[:2, :2].T@a
    coefficient = e*e/entrance_momentum
    linear[:2] += .5*coefficient*(a@a)*inverse_gradient
    curvature[:2, :2] += coefficient*(np.outer(d_a, inverse_gradient)
        +np.outer(inverse_gradient, d_a)+.5*(a@a)*inverse_hessian)
    mixed = e*np.outer(inverse_gradient, a)
    curvature[:2, 2:] += mixed
    curvature[2:, :2] += mixed.T
    # Convert the polynomial in (x-centre) back to the unchanged laboratory
    # canonical variables before subtracting the already executed baseline.
    constant += -linear@centre+.5*centre@curvature@centre
    linear -= curvature@centre
    _, _, baseline = _magnetic_hamiltonian_jet(np.zeros(3), aligned_gradient,
                                              np.zeros((3, 3, 3)), momentum, entrance_momentum)
    generator, force = _J@(curvature-baseline), _J@linear
    for array in (generator, force, value, gradient, hessian, eg, eh, inverse_gradient, inverse_hessian, centre):
        array.setflags(write=False)
    return PosedLensCorrection(generator, force, float(constant), fields, float(z_m),
        float(momentum), float(entrance_momentum), value, gradient, hessian,
        eg, eh, energy, inverse_gradient, inverse_hessian, centre, float(aligned_b_t),
        bool(axis_momentum_magnetic))
