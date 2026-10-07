"""Small-inclination Cs screen on the common laboratory-Z wave lattice.

For relative position q, plane depth d=-b.q and S0=-K|q|**4/4,
the first-inclination canonical action is
    S = (p/p_ref) S0 + (d grad S0).(u + e A_xy/p_ref).
Its momentum term is Weyl ordered. Thus the screen moves position as well
as phase; no image rotation/interpolation or extra magnetic propagation is
performed. Positive phase exp(+i k S) means dq/ds=-S_u, du/ds=S_q.
The geometry remainder is a principal-symbol estimate, not a certification
of the full quantum model, material Cs calibration or longitudinal splitting.
"""
from dataclasses import replace
import math

import numpy as np
from scipy.constants import c, e, hbar, m_e

from temsim.physics.lens_field_provider import FrozenAnalyticField
from temsim.physics.posed_column_fields import FrozenPosedMultipole
from temsim.physics.wave_device import array_module


MODEL_VERSION = "canonical-first-inclination-cs-v1"
MAXIMUM_INCLINATION_RAD = .01
MAXIMUM_MECHANICAL_SLOPE = .1


def _geometry(kick):
    rotation = kick.registration.rotation_array
    normal = rotation[:, 2]
    inclination = math.atan2(float(np.linalg.norm(normal[:2])), float(normal[2]))
    if inclination > MAXIMUM_INCLINATION_RAD:
        raise ValueError("Tilted Cs screen exceeds the 10 mrad small-inclination model domain")
    centre = rotation@np.array((0., 0., kick.local_z_m))+kick.registration.origin_array_m
    return centre[:2], normal[:2]/normal[2], inclination


def _supported(fields):
    return all(isinstance(field, (FrozenAnalyticField, FrozenPosedMultipole)) for field in fields)


def _potential(points, fields, aligned_bz_t, *, gradient=False, xp=np):
    """Actual captured gauge; array-valued analytic derivatives for ray flow."""
    points = xp.asarray(points, dtype=xp.float64)
    a = .5*aligned_bz_t*xp.stack((-points[..., 1], points[..., 0], xp.zeros_like(points[..., 0])), axis=-1)
    if gradient:
        derivative = np.zeros((*points.shape[:-1], 3, 3))
        derivative[..., 0, 1], derivative[..., 1, 0] = -.5*aligned_bz_t, .5*aligned_bz_t
    for field in fields:
        if isinstance(field, FrozenPosedMultipole):
            if gradient:
                value, first, _ = field.vector_potential_jet(points)
                a += value
                derivative += first
            else:
                a += field.vector_potential_at_global_positions_t_m(points, xp=xp)
            continue
        if not isinstance(field, FrozenAnalyticField):
            raise ValueError("Tilted Cs coherent screen requires the captured analytic magnetic gauge")
        rotation = xp.asarray(field.registration.rotation_array)
        local = (points-xp.asarray(field.registration.origin_array_m))@rotation
        magnetic = xp.zeros_like(local[..., 0])
        first_b = xp.zeros_like(magnetic)
        for amplitude, centre, sigma in field.terms_t_m:
            t = (local[..., 2]-centre)/sigma
            term = amplitude*xp.exp(-.5*t*t)
            magnetic += term
            if gradient:
                first_b -= t*term/sigma
        a += .5*xp.stack((-local[..., 1]*magnetic, local[..., 0]*magnetic,
                          xp.zeros_like(magnetic)), axis=-1)@rotation.T
        if gradient:
            local_d = np.zeros((*local.shape[:-1], 3, 3))
            local_d[..., 0, 1], local_d[..., 1, 0] = -.5*magnetic, .5*magnetic
            local_d[..., 0, 2], local_d[..., 1, 2] = -.5*local[..., 1]*first_b, .5*local[..., 0]*first_b
            derivative += np.einsum("ai,...ij,bj->...ab", rotation, local_d, rotation)
    return (a, derivative) if gradient else a


def _field_bounds(fields, aligned_bz_t, global_xy_bound, z_m, depth):
    """Conservative whole-box |A|, |B| bounds for the admitted native laws."""
    radius = float(np.linalg.norm(global_xy_bound))
    a_bound, b_bound = abs(aligned_bz_t)*radius/2, abs(aligned_bz_t)
    for field in fields:
        r = field.registration.rotation_array
        local_centre = (np.array((0., 0., z_m))-field.registration.origin_array_m)@r
        local_extent = abs(r).T@np.r_[global_xy_bound, depth]
        xy = abs(local_centre[:2])+local_extent[:2]
        if isinstance(field, FrozenPosedMultipole):
            zlow, zhigh = np.asarray(field.native_support_mm)*1e-3
            near, far = local_centre[2]-local_extent[2], local_centre[2]+local_extent[2]
            if not field.scale or far < zlow or near > zhigh:
                continue
            if field.scale and (near < zlow < far or near < zhigh < far):
                raise ValueError("Tilted Cs bridge crosses a native magnetic support edge; "
                                 "a resolved oblique interface operator is required")
            for px, py, value in field.polynomial_terms:
                a_bound += abs(value)*xy[0]**px*xy[1]**py
                if px:
                    b_bound += px*abs(value)*xy[0]**(px-1)*xy[1]**py
                if py:
                    b_bound += py*abs(value)*xy[0]**px*xy[1]**(py-1)
        else:
            local_radius = float(np.linalg.norm(xy))
            peak = sum(abs(value) for value, _, _ in field.terms_t_m)
            derivative = sum(abs(value)/(sigma*math.sqrt(math.e)) for value, _, sigma in field.terms_t_m)
            a_bound += local_radius*peak/2
            b_bound += peak+local_radius*derivative/2
    return a_bound, b_bound


def _flow_radius(kick, radius, b_norm):
    denominator = 1-3*b_norm*abs(kick.strength_m3)*radius**3
    if denominator <= 0:
        raise ValueError("Tilted Cs screen has unbounded transverse action within its event")
    # dr/ds <= |b K| r**4, including the outward-moving side of the screen.
    return radius/denominator**(1/3)


def _support_expansion(kick, radius, b_norm):
    advanced_radius = _flow_radius(kick, radius, b_norm)
    # Use the admitted maximum slope here, before the field-dependent slope
    # is known, to cover both the canonical motion and the oblique bridge.
    depth = b_norm*advanced_radius/(1-b_norm*MAXIMUM_MECHANICAL_SLOPE)
    return advanced_radius-radius+depth*MAXIMUM_MECHANICAL_SLOPE, depth


def _remainder(kick, radius, theta, wavelength, p, p_ref, b_norm, magnetic_bound):
    """Bound the omitted straight-plane Cs action and a magnetic bridge term.

    Exact straight intersection: tau=d/(1+b.theta),
    r_local**2=|q+tau theta|**2+tau**2. Compare its quartic action
    with r**4+4*r**2*d*q.theta. The extra magnetic term bounds the
    positional bending neglected by this thin straight-bridge model.
    """
    strength = abs(kick.strength_m3)
    radius = _flow_radius(kick, radius, b_norm)
    incoming_product = b_norm*theta
    if not math.isfinite(theta) or theta > MAXIMUM_MECHANICAL_SLOPE or incoming_product >= .05:
        raise ValueError("Tilted Cs screen leaves the 0.1 mechanical-slope model domain")
    incoming_tau = b_norm*radius/(1-incoming_product)
    # A sloping incident trajectory can reach a larger radius than the
    # perpendicular projection alone. Bound its exact plane intersection.
    hit_radius = math.hypot(radius+incoming_tau*theta, incoming_tau)
    theta += strength*hit_radius**3
    if not math.isfinite(theta) or theta > MAXIMUM_MECHANICAL_SLOPE:
        raise ValueError("Tilted Cs screen leaves the 0.1 mechanical-slope model domain")
    product = b_norm*theta
    if product >= .05:
        raise ValueError("Tilted Cs screen cannot establish forward plane intersection")
    depth = b_norm*radius
    tau = depth/(1-product)
    delta = 2*tau*radius*theta+tau*tau*(1+theta*theta)
    quartic = (4*radius**3*theta*depth*product/(1-product)
               +2*radius*radius*tau*tau*(1+theta*theta)+delta*delta)
    symbol_action = (p/p_ref)*strength*quartic/4
    # Full-field Lorentz acceleration bound intentionally overestimates an
    # axial solenoid; it does not assume Bx/By vanish on the displaced beam.
    acceleration = e*magnetic_bound/p*(1+theta*theta)**1.5
    displacement = .5*acceleration*tau*tau
    bridge_action = (p/p_ref)*strength*(radius+tau*theta+displacement)**3*displacement
    return {"geometric_phase_remainder_bound_rad": 2*math.pi*symbol_action/wavelength,
            "magnetic_bridge_phase_bound_rad": 2*math.pi*bridge_action/wavelength,
            "maximum_mechanical_slope_bound": theta, "maximum_bridge_depth_m": tau,
            "maximum_intersection_radius_m": math.hypot(radius+tau*theta, tau)}


def _electric_remainder(kick, wavelength, p, p_ref, xy_bound, depth, bounds,
                        gradient_v_m, hessian_v_m2):
    """Budget a captured local scalar-potential expansion, without E transport.

    The coefficients belong to the same on-axis node as p. This is a bound
    for that local polynomial, not an assertion about unresolved higher
    derivatives or a bridge crossing an unmodelled electric interface.
    """
    gradient = np.zeros(3) if gradient_v_m is None else np.asarray(gradient_v_m, float)
    hessian = np.zeros((3, 3)) if hessian_v_m2 is None else np.asarray(hessian_v_m2, float)
    if hessian.shape == (2, 2):
        expanded = np.zeros((3, 3))
        expanded[:2, :2] = hessian
        hessian = expanded
    if gradient.shape != (3,) or hessian.shape != (3, 3) or not np.isfinite(gradient).all() or not np.isfinite(hessian).all():
        raise ValueError("Tilted Cs electric bounds require finite local potential derivatives")
    if not np.allclose(hessian, hessian.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Tilted Cs electric potential Hessian must be symmetric")
    extent = np.r_[xy_bound, depth]
    delta_v = float(abs(gradient)@extent+.5*extent@abs(hessian)@extent)
    field_bound = float(np.linalg.norm(abs(gradient)+abs(hessian)@extent))
    delta_energy = e*delta_v
    energy = math.hypot(p*c, m_e*c*c)
    # Stable relativistic kinetic energy and relative momentum differences,
    # including default numerical tails far below double precision in p.
    kinetic = (p*c)**2/(energy+m_e*c*c)
    if delta_energy >= kinetic:
        raise ValueError("Tilted Cs electric variation cannot establish positive kinetic energy")
    plus = delta_energy*(2*energy+delta_energy)/(p*c)**2
    minus = delta_energy*(2*energy-delta_energy)/(p*c)**2
    if minus >= 1:
        raise ValueError("Tilted Cs electric variation cannot establish forward momentum")
    epsilon = max(plus/(math.sqrt(1+plus)+1), minus/(1+math.sqrt(1-minus)))
    p_min = p*math.sqrt(1-minus)
    velocity_min = p_min*c*c/(energy-delta_energy)
    theta, tau = bounds["maximum_mechanical_slope_bound"], bounds["maximum_bridge_depth_m"]
    acceleration = e*field_bound/(p_min*velocity_min)*(1+theta*theta)**1.5
    slope_variation = theta*epsilon/(1-epsilon)+acceleration*tau
    if theta+slope_variation > MAXIMUM_MECHANICAL_SLOPE:
        raise ValueError("Tilted Cs electric bridge leaves the 0.1 mechanical-slope model domain")
    displacement = tau*theta*epsilon/(1-epsilon)+.5*acceleration*tau*tau
    radius = bounds["maximum_intersection_radius_m"]
    scale = (2*math.pi/wavelength)*(p/p_ref)*abs(kick.strength_m3)
    momentum_phase = scale*epsilon*(radius+displacement)**4/4
    bridge_phase = scale*(1+epsilon)*(radius+displacement)**3*displacement
    return {"electric_potential_variation_bound_v": delta_v,
            "electric_field_bound_v_m": field_bound,
            "electric_relative_momentum_variation_bound": epsilon,
            "electric_momentum_phase_bound_rad": momentum_phase,
            "electric_bridge_phase_bound_rad": bridge_phase,
            "electric_phase_remainder_bound_rad": momentum_phase+bridge_phase,
            "maximum_mechanical_slope_bound": theta+slope_variation,
            "electric_bound_scope": "captured local scalar-potential polynomial; no additional electric propagation"}


def _phase(wave, wavelength, kick, centre, scale):
    from temsim.physics.multipole_wave import apply_multipole_phase
    shifted = replace(wave, origin_m=wave.origin_m-centre)
    result = apply_multipole_phase(shifted, wavelength, spherical_m3=kick.strength_m3*scale)
    return replace(result, origin_m=wave.origin_m)


def apply_posed_spherical(wave, wavelength, kick, *, momentum_kg_m_s,
        reference_momentum_kg_m_s, fields=(), aligned_bz_t=0., error_budget=1e-10,
        maximum_obliquity_phase_error_rad=.01, electrostatic_gradient_v_m=None,
        electrostatic_hessian_v_m2=None, cancelled=lambda: False):
    """Apply one captured Cs event, returning (complex wave, execution record).

    Phase/model and numerical-exponential budgets are separate. The reported
    split error is a step-doubling estimate, not a rigorous total-error bound.
    All grid samples and the full envelope Nyquist box enter the domain check;
    no beam tail, phase, intensity or probability is silently removed/refitted.
    """
    from temsim.physics.wave_magnetic_residual import apply_magnetic_residual
    p, p_ref = float(momentum_kg_m_s), float(reference_momentum_kg_m_s)
    if not np.isfinite((p, p_ref, wavelength, aligned_bz_t, error_budget, maximum_obliquity_phase_error_rad)).all() or min(p, p_ref, wavelength, error_budget, maximum_obliquity_phase_error_rad) <= 0:
        raise ValueError("Tilted Cs screen requires finite positive momenta, wavelength and error budgets")
    if cancelled():
        raise InterruptedError("Tilted Cs screen cancelled")
    if kick.strength_m3 == 0:
        return wave, {"model": MODEL_VERSION, "lens_key": kick.lens_key, "method": "zero Cs identity",
                      "input_probability": wave.probability, "output_probability": wave.probability,
                      "series_error_bound": 0., "split_error_estimate": 0., "split_steps": 0}
    centre, b, inclination = _geometry(kick)
    record = {"model": MODEL_VERSION, "lens_key": kick.lens_key, "z_mm": kick.z_mm,
              "inclination_rad": inclination, "strength_m3": kick.strength_m3,
              "input_probability": wave.probability,
              "model_scope": "first-inclination canonical Cs thin screen; geometric principal-symbol, magnetic bridge and captured local electric bounds; not a full oblique wave solution"}
    if cancelled():
        raise InterruptedError("Tilted Cs screen cancelled")
    if not np.any(b) or kick.strength_m3 == 0:
        result = _phase(wave, wavelength, kick, centre, p/p_ref)
        return result, {**record, "method": "aligned radial quartic phase", "output_probability": result.probability,
                        "geometric_phase_remainder_bound_rad": 0., "magnetic_bridge_phase_bound_rad": 0.,
                        "series_error_bound": 0., "split_error_estimate": 0., "split_steps": 0}
    if not _supported(fields):
        raise ValueError("Tilted Cs coherent screen requires analytic vector potentials")
    xp = array_module(wave.amplitude)
    ny, nx = wave.amplitude.shape
    extent = abs(wave.basis_m)@np.array((nx//2, ny//2))
    radial_bound = float(np.linalg.norm(abs(wave.origin_m-centre)+extent))
    xy_bound = abs(wave.origin_m)+extent
    curvature = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    canonical_bound = abs(tilt)+abs(curvature)@extent+.5*wavelength*abs(np.linalg.inv(wave.basis_m).T)@np.ones(2)
    b_norm = float(np.linalg.norm(b))
    expansion, maximum_depth = _support_expansion(kick, radial_bound, b_norm)
    a_bound, b_bound = _field_bounds(fields, aligned_bz_t, xy_bound+expansion,
                                    kick.z_mm*1e-3, maximum_depth)
    theta = (p_ref*float(np.linalg.norm(canonical_bound))+e*a_bound)/p
    bounds = _remainder(kick, radial_bound, theta, wavelength, p, p_ref, b_norm, b_bound)
    bounds.update(_electric_remainder(kick, wavelength, p, p_ref, xy_bound+expansion, maximum_depth,
        bounds, electrostatic_gradient_v_m, electrostatic_hessian_v_m2))
    if (bounds["geometric_phase_remainder_bound_rad"]+bounds["magnetic_bridge_phase_bound_rad"]
            +bounds["electric_phase_remainder_bound_rad"]) > maximum_obliquity_phase_error_rad:
        raise ValueError("Tilted Cs first-inclination phase remainder exceeds its declared event budget; "
                         "a higher-order oblique operator is required")
    xy = wave.coordinates_m()
    q = xy-xp.asarray(centre)[:, None, None]
    radius2 = xp.sum(q*q, axis=0)
    gradient = -kick.strength_m3*radius2[None]*q
    depth = -xp.einsum("i,iyx->yx", xp.asarray(b), q)
    velocity = depth[None]*gradient
    lattice_displacement = float(xp.max(abs(xp.einsum("ij,jyx->iyx", xp.asarray(np.linalg.inv(wave.basis_m)), velocity))))
    if lattice_displacement > .125*min(wave.amplitude.shape):
        raise ValueError("Tilted Cs displacement exceeds the physical wave-grid domain; a wider domain is required")
    points = xp.moveaxis(xp.concatenate((xy, xp.full((1, ny, nx), kick.z_mm*1e-3)), axis=0), 0, -1)
    magnetic_a = xp.moveaxis(_potential(points, fields, aligned_bz_t, xp=xp)[..., :2], -1, 0)
    local = xy-xp.asarray(wave.origin_m)[:, None, None]
    def evolve(steps):
        value, records = wave, []
        for _ in range(steps):
            if cancelled():
                raise InterruptedError("Tilted Cs screen cancelled")
            value = _phase(value, wavelength, kick, centre, p/p_ref/(2*steps))
            current_tilt = np.zeros(2) if value.tilt_rad is None else value.tilt_rad
            current_curve = np.zeros((2, 2)) if value.curvature_m1 is None else value.curvature_m1
            carrier = xp.asarray(current_tilt)[:, None, None]+xp.einsum("ij,jyx->iyx", xp.asarray(current_curve), local)
            scalar = xp.sum(velocity*(carrier+e/p_ref*magnetic_a), axis=0)
            phase = (2*math.pi/wavelength/steps)*scalar
            increment = max(float(xp.max(abs(xp.diff(phase, axis=axis)))) for axis in (0, 1))
            if increment > .2*math.pi:
                from temsim.physics.wave_grid import WaveSamplingError
                raise WaveSamplingError("Tilted Cs inclination action exceeds the phase sampling guard",
                                        increment/(.2*math.pi))
            # Residual API multiplies -ik*distance*H. Here distance=1 metre
            # is an algebraic unit, H=-R/(1 metre), not an extra column drift.
            value, receipt = apply_magnetic_residual(value, -velocity, -scalar, wavelength, 1./steps,
                error_budget=error_budget/(16*steps), cancelled=cancelled)
            receipt["maximum_phase_increment_rad"] = increment
            records.append(receipt)
            value = _phase(value, wavelength, kick, centre, p/p_ref/(2*steps))
        return value, records
    previous, _ = evolve(1)
    split_error = math.inf
    for steps in (2, 4, 8, 16, 32):
        current, receipts = evolve(steps)
        # Both branches have the identical accumulated analytic Cs carrier.
        # Compare absolute complex amplitudes; no best-fit/global-phase removal.
        difference = float(xp.linalg.norm(current.amplitude-previous.amplitude))
        split_error = difference/max(math.sqrt(wave.probability), np.finfo(float).tiny)/3
        if split_error <= error_budget*.5:
            return current, {**record, **bounds, "compute_backend": "numpy" if xp is np else "cupy",
                "method": "quartic phase / Hermitian inclination action / quartic phase",
                "maximum_obliquity_phase_error_rad": maximum_obliquity_phase_error_rad,
                "output_probability": current.probability, "split_steps": steps,
                "split_error_estimate": split_error, "requested_numerical_error_budget": error_budget,
                "series_error_bound": sum(row["series_error_bound"] for row in receipts),
                "maximum_lattice_displacement": lattice_displacement,
                "maximum_phase_increment_rad": max(row["maximum_phase_increment_rad"] for row in receipts),
                "operator_applications": sum(row["operator_applications"] for row in receipts)}
        previous = current
    raise ValueError("Tilted Cs screen split refinement did not satisfy its numerical budget")


def apply_canonical_ray_kick(kick, x, tx, y, ty, *, momentum_kg_m_s,
        reference_momentum_kg_m_s=None, fields=(), aligned_bz_t=0.,
        maximum_obliquity_phase_error_rad=.01, electrostatic_gradient_v_m=None,
        electrostatic_hessian_v_m2=None):
    """The same first-inclination action, evaluated as Hamiltonian rays.

    Unknown imported maps retain the historical mechanical thin-screen law;
    coherent execution rejects those maps rather than claiming a shared gauge.
    """
    if not kick.strength_m3:
        return x, tx, y, ty
    if not _supported(fields):
        return kick.apply(x, tx, y, ty)
    centre, b, _ = _geometry(kick)
    if not np.any(b):
        return kick.apply(x, tx, y, ty)
    arrays = np.broadcast_arrays(*(np.asarray(value, float) for value in (x, tx, y, ty, momentum_kg_m_s)))
    x, tx, y, ty, p = (a.copy() for a in arrays)
    valid = np.isfinite(x)&np.isfinite(tx)&np.isfinite(y)&np.isfinite(ty)&np.isfinite(p)&(p > 0)
    if not valid.any():
        return x, tx, y, ty
    p = p[valid]
    pref = p if reference_momentum_kg_m_s is None else np.broadcast_to(reference_momentum_kg_m_s, x.shape)[valid]
    if not np.isfinite(pref).all() or np.any(pref <= 0):
        raise ValueError("Tilted Cs ray reference momentum must be positive and finite")
    q = np.column_stack((x[valid], y[valid]))
    radius = float(np.max(np.linalg.norm(q-centre, axis=1)))
    slope = float(np.max(np.hypot(tx[valid], ty[valid])))
    b_norm = float(np.linalg.norm(b))
    expansion, maximum_depth = _support_expansion(kick, radius, b_norm)
    _, magnetic_bound = _field_bounds(fields, aligned_bz_t, np.max(abs(q), axis=0)+expansion,
        kick.z_mm*1e-3, maximum_depth)
    pmax = float(np.max(p))
    bounds = _remainder(kick, radius, slope, 2*math.pi*hbar/pmax, pmax, pmax,
        b_norm, magnetic_bound*pmax/float(np.min(p)))
    # For an ensemble, the lowest momentum bounds the relative electric
    # perturbation; scaling its phase by pmax/pmin covers the high-energy ray.
    pmin = float(np.min(p))
    electric_bounds = _electric_remainder(kick, 2*math.pi*hbar/pmax, pmin, pmin,
        np.max(abs(q), axis=0)+expansion, maximum_depth, bounds,
        electrostatic_gradient_v_m, electrostatic_hessian_v_m2)
    bounds.update(electric_bounds)
    if (bounds["geometric_phase_remainder_bound_rad"]+bounds["magnetic_bridge_phase_bound_rad"]
            +bounds["electric_phase_remainder_bound_rad"]) > maximum_obliquity_phase_error_rad:
        raise ValueError("Tilted Cs ray first-inclination phase remainder exceeds its declared event budget")
    points = lambda pos: np.column_stack((pos, np.full(len(pos), kick.z_mm*1e-3)))
    a = _potential(points(q), fields, aligned_bz_t)[:, :2]
    u = (p[:, None]*np.column_stack((tx[valid], ty[valid]))-e*a)/pref[:, None]
    def flow(pos, momentum):
        relative = pos-centre
        r2 = np.sum(relative*relative, axis=1)
        g = -kick.strength_m3*r2[:, None]*relative
        h = -kick.strength_m3*(r2[:, None, None]*np.eye(2)+2*relative[:, :, None]*relative[:, None, :])
        depth = -relative@b
        v = depth[:, None]*g
        dv = depth[:, None, None]*h-g[:, :, None]*b[None, None, :]
        a, da = _potential(points(pos), fields, aligned_bz_t, gradient=True)
        kinetic = momentum+e/pref[:, None]*a[:, :2]
        du = ((p/pref)[:, None]*g+np.einsum("nij,ni->nj", dv, kinetic)
              +e/pref[:, None]*np.einsum("ni,nij->nj", v, da[:, :2, :2]))
        return -v, du
    # The domain restricts fractional obliquity displacement to <=1e-3;
    # four RK4 steps resolve this shared first-inclination flow cheaply.
    for _ in range(4):
        h = .25
        a1, b1 = flow(q, u)
        a2, b2 = flow(q+h*a1/2, u+h*b1/2)
        a3, b3 = flow(q+h*a2/2, u+h*b2/2)
        a4, b4 = flow(q+h*a3, u+h*b3)
        q += h*(a1+2*a2+2*a3+a4)/6
        u += h*(b1+2*b2+2*b3+b4)/6
    a = _potential(points(q), fields, aligned_bz_t)[:, :2]
    mechanical = (pref[:, None]*u+e*a)/p[:, None]
    x[valid], y[valid] = q.T
    tx[valid], ty[valid] = mechanical.T
    return x, tx, y, ty
