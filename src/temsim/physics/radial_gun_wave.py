"""Two-way scalar orbital wave operator for the grounded, round FEG.

The moving Laguerre basis is a numerical coordinate frame, not an electron
source. Its reference ellipse follows a quadratic optical map, while the
stationary wave operator samples the actual electrode potential, including
its nonquadratic terms. Every installed gun bore/aperture remains an
absorbing operation. Non-axisymmetric gun fields require the corresponding
2-D basis and are rejected here rather than switched off.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.constants import c, e, hbar, m_e
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.special import eval_laguerre

from temsim.physics.scattering_load import hermitian_slab, outgoing_load
from temsim.physics.surface_wave import KINETIC_NM2_PER_EV
from temsim.physics.liouville_wave import coordinate_terms, derivative_jump, PhysicalCoordinateLoad
from temsim.physics.magnus_scattering import GAUSS_FRACTIONS, cf4_slab
from temsim.physics.adaptive_scattering import adaptive_cf4, AxialRefinement
from temsim.physics.radial_coordinates import RadialCoordinateBlend, blended_radial_chart
from temsim.physics.wave_following_chart import WaveFollowingNumerics
from temsim.physics.quartic_radial_phase import quartic_operators
from temsim.physics.occupied_axial_refinement import OccupiedAxialRefinement

REST_EV = m_e*c*c/e


@dataclass(frozen=True)
class RadialGunNumerics:
    radial_modes: int = 16
    potential_quadrature: int = 96
    relative_axial_step: float = .02
    field_step_mm: float = .1
    bore_step_mm: float = 1.
    maximum_steps: int = 100_000
    maximum_working_bytes: int = 16*1024**3
    coordinate_width_over_cap_radius: float = .2
    axial_integrator: str = "midpoint"
    axial_refinement: AxialRefinement = AxialRefinement()
    adaptive_until_mm: float | None = None
    diagnostic_planes_mm: tuple[float, ...] = ()
    coordinate_blend: RadialCoordinateBlend | None = None
    wave_following: WaveFollowingNumerics = WaveFollowingNumerics()
    occupied_refinement: OccupiedAxialRefinement = OccupiedAxialRefinement()

    def validate(self):
        if self.axial_integrator not in ("midpoint", "cf4", "adaptive_cf4"):
            raise ValueError("Radial gun axial integrator must be midpoint, cf4 or adaptive_cf4")
        self.axial_refinement.validate()
        self.wave_following.validate()
        self.occupied_refinement.validate()
        if self.coordinate_blend is not None:
            self.coordinate_blend.validate()
        if self.adaptive_until_mm is not None and (not math.isfinite(self.adaptive_until_mm) or self.adaptive_until_mm <= 0):
            raise ValueError("Adaptive-region end must be positive and finite")
        if not isinstance(self.diagnostic_planes_mm, tuple) or any(not math.isfinite(z) or z <= 0 for z in self.diagnostic_planes_mm):
            raise ValueError("Diagnostic gun planes must be a tuple of positive finite millimetres")
        for name, low, high in (("radial_modes", 2, 256), ("potential_quadrature", 8, 1024),
                                ("maximum_steps", 2, 1_000_000)):
            if type(getattr(self, name)) is not int or not low <= getattr(self, name) <= high:
                raise ValueError(f"Invalid radial gun {name}")
        if self.potential_quadrature < 2*self.radial_modes:
            raise ValueError("Potential quadrature must resolve at least twice the radial modes")
        for name in ("relative_axial_step", "field_step_mm", "bore_step_mm"):
            if not math.isfinite(getattr(self, name)) or not 0 < getattr(self, name):
                raise ValueError(f"Invalid radial gun {name}")
        if self.relative_axial_step > .25:
            raise ValueError("Radial gun relative axial step must not exceed 0.25")
        if type(self.maximum_working_bytes) is not int or self.maximum_working_bytes <= 0:
            raise ValueError("Radial gun memory budget must be positive")
        width = self.coordinate_width_over_cap_radius
        if isinstance(width, bool) or not math.isfinite(width) or not .001 <= width <= 1.:
            raise ValueError("The numerical Laguerre coordinate width/cap radius must be in [0.001, 1]")
        return self


def squared_wave_number(kinetic_ev):
    """Scalar relativistic orbital dispersion, in inverse square nm."""
    kinetic = np.asarray(kinetic_ev, float)
    return KINETIC_NM2_PER_EV*kinetic*(1+kinetic/(2*REST_EV))


def laguerre_operators(count):
    n = np.arange(count)
    x = np.diag(2*n+1.)-np.diag(n[1:], 1)-np.diag(n[1:], -1)
    dilation = np.diag(n[1:], -1)-np.diag(n[1:], 1)
    kinetic = np.diag(4*n+2.)-x
    return x, dilation, kinetic


def basis_values(radius_nm, width_nm, curvature_per_nm, reference_k, count):
    radius = np.asarray(radius_nm, float)
    x = (radius/width_nm)**2
    common = np.exp(-x/2+.5j*reference_k*curvature_per_nm*radius**2)/(math.sqrt(math.pi)*width_nm)
    return np.stack([eval_laguerre(n, x)*common for n in range(count)], axis=-1)


def aperture_projection(radius_nm, width_nm, count, quadrature):
    """Analytic Laguerre disk integral; quadrature is a compatibility argument.

    The Sturm-Liouville boundary identity gives every off-diagonal entry.
    L_n'=-sum_{j<n} L_j then gives each diagonal. No fitted transmission,
    sampled hard edge or numerical eigenspace truncation is introduced.
    """
    if radius_nm <= 0:
        return np.zeros((count, count))
    limit = (radius_nm/width_nm)**2
    if not math.isfinite(limit):
        raise ValueError("Aperture/basis ratio must be finite")
    # Scale the polynomial recurrence, not the returned wave/probability.
    # This avoids 0*inf for very wide apertures and high polynomial orders.
    values = np.zeros(count)
    previous, current, log_scale = 0., 1., 0.
    for n in range(count):
        if current:
            exponent = math.log(abs(current))+log_scale-limit/2
            values[n] = math.copysign(math.exp(exponent), current)
        following = ((2*n+1-limit)*current-n*previous)/(n+1)
        previous, current = current, following
        size = max(abs(previous), abs(current))
        if size > 1e100:
            previous /= size; current /= size; log_scale += math.log(size)
    derivative = np.r_[0., -np.cumsum(values[:-1])]
    order = np.arange(count)
    difference = order[None, :]-order[:, None]
    matrix = np.divide(limit*(derivative[:, None]*values[None, :]-values[:, None]*derivative[None, :]),
        difference, out=np.zeros((count, count)), where=difference != 0)
    diagonal = -np.expm1(-limit)*np.ones(count)
    # expm1 preserves the tiny n=0 disk probability. Higher diagonals follow
    # the exact derivative identity; no eigenvalue clipping or renormalising.
    diagonal[1:] = 1-values[1:]**2-2*np.tril(matrix, -1).sum(axis=1)[1:]
    np.fill_diagonal(matrix, diagonal)
    return matrix


def _multiply_laguerre_x(matrix):
    """Right-multiply by the exact tridiagonal coordinate operator, O(N^2)."""
    n = np.arange(len(matrix))
    result = matrix*(2*n+1.)[None, :]
    result[:, 1:] -= matrix[:, :-1]*n[1:][None, :]
    result[:, :-1] -= matrix[:, 1:]*n[1:][None, :]
    return result


def piecewise_radial_potential(width_nm, count, radii_nm, kinetic_ev, axis_kinetic_ev):
    """Exact finite-basis integral for a potential linear in r^2 per knot.

    Relativistic k^2 is quadratic in energy, hence quadratic in x=(r/b)^2
    on each piece. Two extra Laguerre modes retain BOTH coordinate products
    before projection. No quadrature loop, sampled potential phase or removed
    high-order coupling is used.
    """
    from temsim.physics.radial_potential_batch import piecewise_potential
    return piecewise_potential(width_nm, count, radii_nm, kinetic_ev, axis_kinetic_ev,
                               KINETIC_NM2_PER_EV, REST_EV)


def resolved_radial_quadrature(width_nm, count, field_r_nm, order):
    """Positive radial quadrature split at every consumed potential knot.

    The electrode field is piecewise linear in squared radius. Global Laguerre
    quadrature does not resolve those derivative jumps, even if it resolves
    the basis Gram matrix. Integrate each smooth piece separately. The
    exponential basis tail beyond 4*N+160 is independently negligible; no
    wave coefficient is removed or renormalised.
    """
    limit = math.sqrt(4*count+160)
    knots = np.asarray(field_r_nm)/width_nm
    knots = np.unique(np.r_[0., knots[(knots > 0) & (knots < limit)], limit,
                             np.linspace(0., limit, math.ceil(limit/.5)+1)])
    nodes, weights = np.polynomial.legendre.leggauss(order)
    half = np.diff(knots)/2
    scaled_r = (.5*(knots[:-1]+knots[1:]))[:, None]+half[:, None]*nodes
    weights = (half[:, None]*weights*2*scaled_r).ravel()
    scaled_r = scaled_r.ravel()
    x = scaled_r**2
    basis = np.array([eval_laguerre(n, x) for n in range(count)])*np.exp(-x/2)
    weighted = basis*np.sqrt(weights)
    error = float(np.max(abs(weighted@weighted.T-np.eye(count))))
    if not np.isfinite(error) or error > 1e-11:
        raise ValueError(f"Resolved electrode quadrature Gram error {error:.6g}; increase quadrature order")
    return scaled_r*width_nm, weighted


def gun_grid(gun, start_nm, numerics):
    from temsim.physics.tip_gun_wave import _axial_grid, GunWaveNumerics
    coarse, masks = _axial_grid(gun, GunWaveNumerics(field_step_mm=numerics.field_step_mm,
        bore_step_mm=numerics.bore_step_mm, max_steps=numerics.maximum_steps))
    start = start_nm*1e-6
    near = [start]
    while near[-1] < min(1., gun.exit_plane_z_mm):
        near.append(near[-1]*(1+numerics.relative_axial_step))
        if len(near) > numerics.maximum_steps:
            raise ValueError("Near-tip radial wave grid exceeds its step budget")
    z = np.unique(np.r_[near, coarse[coarse > start]])
    z = z[z <= gun.exit_plane_z_mm]
    if len(z)-1 > numerics.maximum_steps:
        raise ValueError("Radial gun grid exceeds its step budget")
    return z*1e6, masks


def merge_gun_nodes(nodes_nm, mask_planes_mm):
    """Merge roundoff-equivalent nodes and schedule each physical plane once.

    Electrode knots use metres, bore events use millimetres and this solver
    uses nanometres. Exact float uniqueness is insufficient after conversion.
    The eight-ULP rule merges representation roundoff, not a finite physical
    interval. Coincident components are still all applied by _mask_events.
    """
    nodes = np.asarray(nodes_nm, float)
    masks = np.asarray(sorted(mask_planes_mm), float)*1e6
    if (nodes.ndim != 1 or not len(nodes) or not np.all(np.isfinite(nodes))
            or not np.all(np.isfinite(masks))):
        raise ValueError("Gun axial nodes must be finite")
    first, last = float(nodes.min()), float(nodes.max())
    masks = masks[(masks > first) & (masks <= last)]
    entries = sorted([(float(z), False) for z in nodes]+[(float(z), True) for z in masks])
    merged, schedule = [], {}
    cluster = []
    def flush():
        marked = [value for value, mask in cluster if mask]
        value = marked[0] if marked else cluster[0][0]
        if any(z == first for z, _ in cluster):
            value = first
        elif any(z == last for z, _ in cluster):
            value = last
        if marked:
            schedule[len(merged)] = value
        merged.append(value)
    for entry in entries:
        if cluster and entry[0]-cluster[0][0] > 8*abs(np.spacing(max(abs(entry[0]), abs(cluster[0][0])))):
            flush()
            cluster = []
        cluster.append(entry)
    flush()
    return np.asarray(merged), schedule


def _mask_events(gun, z_nm):
    z_mm = z_nm*1e-6
    events = []
    for component in gun.bore_components:
        if abs(z_mm-component.mechanical_center_from_tip_mm) <= .5*component.mechanical_length_mm+1e-10:
            events.append((component.key, component.mechanical_clear_bore_diameter_mm*.5e6, "body_bore"))
    for aperture in (gun.dpa_aperture, gun.c1_aperture):
        if aperture.enabled and abs(z_mm-aperture.z_mm) < 1e-10:
            if aperture.kind == "energy_selection_slit":
                raise ValueError("A gun slit requires the non-axisymmetric coherent basis; it cannot be skipped")
            events.append((aperture.key, aperture.radius_mm*1e6, aperture.interaction_kind))
    return events


def prepare_round_gun(gun, start_nm, energy_ev, numerics, *, column_state=None, executed_chart=None, refinement_plan=None,
                      cancelled=lambda: False, progress_callback=None):
    """Execute all far-gun operators and their reflected load at the near field.

    Scalar Klein-Gordon orbital optics excludes spin and space charge. This
    is a finite transverse Galerkin calculation; basis and axial convergence
    are distinct from the exact scattering-matrix current identity.
    """
    from temsim.physics.grounded_tip_field import grounded_field
    from temsim.physics.wave_execution import check_available_memory
    numerics.validate()
    if gun.monochromator_installed:
        raise ValueError("Installed Wien/slit optics require the non-axisymmetric coherent gun operator")
    for aperture in (gun.dpa_aperture, gun.c1_aperture):
        if aperture.enabled and (aperture.offset_x_mm or aperture.offset_y_mm):
            raise ValueError("An offset gun aperture needs the non-axisymmetric coherent operator; it cannot be recentered")
    z, mask_planes = gun_grid(gun, start_nm, numerics)
    field = grounded_field(gun)
    # Split at every consumed electrode-potential knot. The axial action
    # quadrature then sees the same piecewise-linear potential as the solver.
    knots = field.z*1e9
    diagnostic_planes = np.asarray(numerics.diagnostic_planes_mm)*1e6
    diagnostic_planes = diagnostic_planes[(diagnostic_planes > z[0]) & (diagnostic_planes <= z[-1])]
    extra = [] if numerics.adaptive_until_mm is None else [numerics.adaptive_until_mm*1e6]
    if numerics.wave_following.iterations:
        settings = numerics.wave_following
        if not start_nm < settings.transition_start_nm < settings.transition_end_nm < z[-1]:
            raise ValueError("Wave-following transition must be inside the executed gun interval")
        extra.extend((settings.transition_start_nm, settings.transition_end_nm))
    if numerics.coordinate_blend is not None:
        transition = numerics.coordinate_blend
        if transition.start_mm*1e6 <= start_nm or transition.end_mm > gun.exit_plane_z_mm:
            raise ValueError("Numerical radial chart transition must lie inside the executed gun interval")
        extra.extend((transition.start_mm*1e6, transition.end_mm*1e6))
    extra = [point for point in extra if z[0] < point < z[-1]]
    z, mask_schedule = merge_gun_nodes(np.r_[z, knots[(knots > z[0]) & (knots < z[-1])], diagnostic_planes, extra], mask_planes)
    diagnostic_indices = {int(np.argmin(abs(z-point))) for point in diagnostic_planes}
    if len(z)-1 > numerics.maximum_steps:
        raise ValueError("Resolved electrode knots exceed the radial gun step budget")
    count = numerics.radial_modes
    estimated = 16*(12*len(z)+40)*count**2
    if estimated > numerics.maximum_working_bytes:
        raise MemoryError(f"Radial gun scattering needs approximately {estimated} bytes")
    check_available_memory(estimated)
    positions = np.zeros((len(z), 3)); positions[:, 2] = z*1e-9
    phi = field.potential_rise_v_at_global_positions(positions)
    if np.any(energy_ev+phi <= 0):
        raise ValueError("The numerical round-gun frame needs a positive axial kinetic reference")
    # This curvature ONLY guides the numerical basis. The wave potential
    # below is independently sampled across its complete quadrature domain.
    radial_step_nm = np.maximum(field.r[1]*1e9, np.minimum(1e5, z*.001))
    off = positions.copy(); off[:, 0] = radial_step_nm*1e-9
    hessian = 2*(field.potential_rise_v_at_global_positions(off)-phi)/radial_step_nm**2
    gauss_x, gauss_w = np.polynomial.legendre.leggauss(8)
    kinetic_quadrature = energy_ev + (.5*(phi[:-1]+phi[1:]))[:, None] + .5*np.diff(phi)[:, None]*gauss_x
    average_k = np.sqrt(squared_wave_number(kinetic_quadrature))@(gauss_w/2)
    inverse_velocity = m_e*(1+kinetic_quadrature/REST_EV)/(hbar*np.sqrt(squared_wave_number(kinetic_quadrature))*1e9)
    average_inverse_v = inverse_velocity@(gauss_w/2)
    magnetic = gun.magnetic_field
    for radial in (0., 1e-6, 1e-4):
        for azimuth in (0., math.pi/2):
            point = positions.copy()
            point[:, 0], point[:, 1] = radial*math.cos(azimuth), radial*math.sin(azimuth)
            sampled = magnetic.field_at_global_positions_t(point)
            if np.any(sampled != 0):
                raise ValueError("Nonzero gun magnetic fields need the non-axisymmetric coherent operator; no field was omitted")
    axial_b = np.zeros(len(z))
    column_signature = None
    if column_state is not None:
        from temsim.physics.column_wave import _prepare_column
        from temsim.physics.core import fields
        plan, radii, stops, owners = _prepare_column(column_state, start_nm*1e-6,
            gun.exit_plane_z_mm, numerics.field_step_mm)
        names = ("sx_m2", "sy_m2", "midpoint_sx_m2", "midpoint_sy_m2", "hex_normal_m3", "hex_skew_m3",
                 "midpoint_hex_normal_m3", "midpoint_hex_skew_m3", "cs_kick_m3", "thin_power_m1",
                 "thin_rotation_rad", "kick_x_rad", "kick_y_rad")
        if plan.mapped_fields or stops or owners or any(np.any(getattr(plan, name)) for name in names):
            raise ValueError("A non-axisymmetric column operation intersects the gun and needs its joint wave operator")
        axial_b = fields(z*1e-6, column_state)[0]
        column_signature = plan.signature
    x, dilation, transverse = laguerre_operators(count)
    model = gun.emitter.surface_model
    radius = model.geometry.apex_radius_nm
    patch = radius*math.sin(math.radians(model.emission.cap_half_angle_deg))
    width0 = patch*numerics.coordinate_width_over_cap_radius
    # Numerical coordinate width ONLY. Changing it requires new execution
    # and convergence checks; it never changes the physical reservoir.
    k_ref = math.sqrt(float(squared_wave_number(energy_ev+phi[0])))
    q = 1/radius+1j/(k_ref*width0**2)
    target_width = width0 if numerics.coordinate_blend is None else patch*numerics.coordinate_blend.target_width_over_cap_radius
    target_q = 1/radius+1j/(k_ref*target_width**2)
    operators, rows, frames, samplers = [], [], [], {}
    chart_derivatives = []
    potential_slopes = np.diff(phi)/np.diff(z)
    def longitudinal(kinetic, slope):
        return coordinate_terms(squared_wave_number(kinetic),
            KINETIC_NM2_PER_EV*slope*(1+kinetic/REST_EV),
            KINETIC_NM2_PER_EV*slope*slope/REST_EV, k_ref)
    initial_alpha, initial_log, _ = longitudinal(energy_ev+phi[0], potential_slopes[0])
    boundary_alpha, boundary_log = [float(initial_alpha)], [float(initial_log)]
    flight, action = 0., 0.
    carrier_phasor = 1.+0j
    end_q = None
    identity = np.eye(count)
    def momentum_v(kinetic):
        k = math.sqrt(float(squared_wave_number(kinetic)))
        velocity = hbar*k*1e9/(m_e*(1+kinetic/REST_EV))
        return k, velocity
    def frame(value, target, zz):
        # Coordinates alone, independent of which side supplies the local
        # optical guide derivative at a potential knot.
        base = blended_radial_chart(value, 0j, target, 0j, k_ref, zz, numerics.coordinate_blend)
        return (base if executed_chart is None else executed_chart.evaluate(zz, base))[:2]
    def phase(zz):
        return (0., 0.) if executed_chart is None else executed_chart.quartic_phase(zz)
    def coordinate_rates(value, target, zz, strength, drift):
        coordinates = blended_radial_chart(value, strength-drift*value*value,
            target, strength-drift*target*target, k_ref, zz, numerics.coordinate_blend)
        if executed_chart is not None:
            coordinates = executed_chart.evaluate(zz, coordinates)
        return (*coordinates[2:], phase(zz)[1])
    for index, dz in enumerate(np.diff(z)):
        if cancelled():
            raise InterruptedError("Round gun wave propagation cancelled")
        zm = .5*(z[index]+z[index+1])
        kinetic0 = energy_ev+.5*(phi[index]+phi[index+1])
        k, velocity = momentum_v(kinetic0)
        alpha, _, longitudinal_correction = longitudinal(kinetic0, potential_slopes[index])
        left_alpha, left_log, _ = longitudinal(energy_ev+phi[index], potential_slopes[index])
        if index and left_log != boundary_log[-1]:
            jump = float((boundary_log[-1]-left_log)/left_alpha)
            operators.append(derivative_jump(jump, k_ref, count))
            rows.append({"kind": "vacuum_fields", "z_nm": float(z[index]),
                         "liouville_derivative_jump": jump})
            frames.append(frame(q, target_q, z[index]))
            boundary_alpha.append(float(left_alpha)); boundary_log.append(float(left_log))
        bz = .5*(axial_b[index]+axial_b[index+1])
        magnetic_k = e*bz/(2*hbar)*1e-18  # nm^-2
        strength = KINETIC_NM2_PER_EV*(1+kinetic0/REST_EV)*.5*(hessian[index]+hessian[index+1])/(2*k*k_ref)
        strength -= magnetic_k**2/(k*k_ref)
        drift = k_ref/k
        half_map = expm(np.array(((0., drift), (strength, 0.)))*(.5*dz))
        def advance(value):
            return (half_map[1, 0]+half_map[1, 1]*value)/(half_map[0, 0]+half_map[0, 1]*value)
        def sample_operator(fraction, value, target, *, index=index, dz=dz, strength=strength, drift=drift):
            # This exact derivative belongs to the frozen numerical guide,
            # even when physical fields below are sampled at a Gauss node.
            q_prime = strength-drift*value*value
            target_prime = strength-drift*target*target
            zz = z[index]+fraction*dz
            width, curvature, width_ratio_prime, curvature_prime = blended_radial_chart(
                value, q_prime, target, target_prime, k_ref, zz, numerics.coordinate_blend)
            if executed_chart is not None:
                width, curvature, width_ratio_prime, curvature_prime = executed_chart.evaluate(zz,
                    (width, curvature, width_ratio_prime, curvature_prime))
            connection = .5*k_ref*curvature_prime*width**2*x + 1j*width_ratio_prime*dilation
            limit = width*math.sqrt(4*count+160)
            field_knots = field.r*1e9
            if limit > field_knots[-1]:
                raise ValueError("The retained radial basis exceeds the solved electrode domain")
            r = np.r_[field_knots[field_knots < limit], limit]
            points = np.column_stack((r, np.zeros(len(r)), np.full(len(r), zz)))*1e-9
            local = energy_ev+field.potential_rise_v_at_global_positions(points)
            if np.any(local <= 0):
                raise ValueError("Transverse basis reaches the metal/negative-energy region; refine its physical domain")
            axis_kinetic = energy_ev+phi[index]+fraction*(phi[index+1]-phi[index])
            potential = piecewise_radial_potential(width, count, r, local, axis_kinetic)
            kinetic = transverse/width**2 + (k_ref*curvature*width)**2*x-2j*k_ref*curvature*dilation
            # Retain ||Phi' f||^2 outside the truncated moving subspace.
            closure = np.zeros_like(kinetic)
            closure[-1, -1] = count**2*((.5*k_ref*curvature_prime*width**2)**2+width_ratio_prime**2)
            quartic, quartic_prime = phase(zz)
            if quartic or quartic_prime:
                kinetic, connection, closure = quartic_operators(count, width, curvature, k_ref,
                    width_ratio_prime, curvature_prime, quartic, quartic_prime)
            local_b = axial_b[index]+fraction*(axial_b[index+1]-axial_b[index])
            local_magnetic_k = e*local_b/(2*hbar)*1e-18
            residual = potential-kinetic-(local_magnetic_k*width)**2*x-closure
            local_alpha, _, correction = longitudinal(axis_kinetic, potential_slopes[index])
            return (residual+correction*identity)/local_alpha**2, connection/local_alpha

        q_mid = advance(q)
        target_mid = advance(target_q)
        # Save the one-sided exact numerical-frame rates with each executed
        # boundary. A projected covariant derivative alone is insufficient
        # for reconstructing the local real-space current density.
        left_rates = coordinate_rates(q, target_q, z[index], strength, drift)
        chart_derivatives.extend([left_rates]*(len(frames)+1-len(chart_derivatives)))
        right_rates = coordinate_rates(advance(q_mid), advance(target_mid), z[index+1], strength, drift)
        action_width = float(average_k[index]*dz/k_ref)
        # Freeze every per-interval value before retaining the sampler. A late
        # bound loop closure would silently re-evaluate an earlier step with
        # the last interval's fields/guide instead of its executed settings.
        guide = np.array(((0., drift), (strength, 0.)))
        def sample_at_action(target, *, index=index, dz=dz, guide=guide,
                             q=q, target_q=target_q, sample_operator=sample_operator):
            def action_fraction(fraction):
                energies = energy_ev+phi[index]+fraction*(phi[index+1]-phi[index])*(gauss_x+1)/2
                return fraction*float(np.sqrt(squared_wave_number(energies))@(gauss_w/2))/average_k[index]
            fraction = brentq(lambda f: action_fraction(f)-target, 0., 1., xtol=1e-14)
            mapping = expm(guide*(fraction*dz))
            value = (mapping[1, 0]+mapping[1, 1]*q)/(mapping[0, 0]+mapping[0, 1]*q)
            target = (mapping[1, 0]+mapping[1, 1]*target_q)/(mapping[0, 0]+mapping[0, 1]*target_q)
            return sample_operator(fraction, value, target)
        adaptive_region = numerics.adaptive_until_mm is None or z[index] < numerics.adaptive_until_mm*1e6
        method = numerics.axial_integrator
        if method == "adaptive_cf4" and not adaptive_region:
            method = "midpoint"
        if method == "midpoint":
            residual, connection = sample_operator(.5, q_mid, target_mid)
            try:
                scatter, diagnostic = hermitian_slab(residual, connection, action_width,
                    k_ref, carrier_k=k_ref)
            except ValueError as error:
                raise ValueError(f"Gun interval [{z[index]:.9g}, {z[index+1]:.9g}] nm "
                                 f"at {energy_ev:.9g} eV: {error}") from error
        else:
            # CF4 nodes are Gauss nodes in s=int(k/k_ref dz), not in z.
            # Potential knots split every interval, so energy is linear here.
            if method == "adaptive_cf4":
                def refinement_progress(done, total, message):
                    if progress_callback is not None and done % 32 == 0:
                        progress_callback(done, total, f"{message} at {z[index]*1e-6:.6g} mm")
                try:
                    scatter, diagnostic = adaptive_cf4(sample_at_action, action_width, k_ref,
                        numerics.axial_refinement, cancelled=cancelled, progress_callback=refinement_progress)
                except ValueError as error:
                    raise ValueError(f"Gun interval [{z[index]:.9g}, {z[index+1]:.9g}] nm: {error}") from error
            else:
                scatter, diagnostic = cf4_slab(*(sample_at_action(f) for f in GAUSS_FRACTIONS),
                    action_width, k_ref, cancelled=cancelled)
        if refinement_plan is not None:
            samplers[len(operators)] = (sample_at_action, action_width)
        operators.append(scatter)
        rows.append({"kind": "vacuum_fields", "z_nm": float(z[index+1]),
                     "diagnostic_checkpoint": index+1 in diagnostic_indices, **diagnostic})
        q = advance(q_mid)
        target_q = advance(target_mid)
        frames.append(frame(q, target_q, z[index+1]))
        right_alpha, right_log, _ = longitudinal(energy_ev+phi[index+1], potential_slopes[index])
        boundary_alpha.append(float(right_alpha)); boundary_log.append(float(right_log))
        flight += dz*1e-9*average_inverse_v[index]
        action += hbar*average_k[index]*dz
        carrier_phasor *= np.exp(1j*float(average_k[index]*dz))
        if index+1 in mask_schedule:
            width_end, _ = frame(q, target_q, z[index+1])
            for key, radius_nm, kind in _mask_events(gun, mask_schedule[index+1]):
                projection = aperture_projection(radius_nm, width_end, count, numerics.potential_quadrature)
                zero = np.zeros_like(projection)
                operators.append((zero, projection, projection, zero))
                rows.append({"kind": kind, "component": key, "radius_nm": radius_nm, "z_nm": float(z[index+1]),
                    "projection_closure_defect_norm": float(np.linalg.norm(projection-projection@projection, ord=2)),
                    "closure_scope": "A nonzero defect includes unresolved masked modes; losses are not certified physical absorption"})
                frames.append(frame(q, target_q, z[index+1]))
                boundary_alpha.append(float(right_alpha)); boundary_log.append(float(right_log))
        chart_derivatives.extend([right_rates]*(len(frames)+1-len(chart_derivatives)))
        if progress_callback is not None and index % 64 == 0:
            progress_callback(index, len(z)-1, "Coherent extraction, acceleration, focusing and apertures")
    width, curvature = frame(q, target_q, z[-1])
    kinetic_exit = energy_ev+float(phi[-1])
    k_exit = math.sqrt(float(squared_wave_number(kinetic_exit)))
    kinetic = transverse/width**2+(k_ref*curvature*width)**2*x-2j*k_ref*curvature*dilation
    exit_quartic, _ = phase(z[-1])
    if exit_quartic:
        kinetic, _, _ = quartic_operators(count, width, curvature, k_ref, quartic=exit_quartic)
    exit_magnetic_k = e*axial_b[-1]/(2*hbar)*1e-18
    exit_alpha = k_exit/k_ref
    if boundary_log[-1] != 0:
        jump = float(boundary_log[-1]/exit_alpha)
        operators.append(derivative_jump(jump, k_ref, count))
        rows.append({"kind": "vacuum_fields", "z_nm": float(z[-1]), "liouville_derivative_jump": jump})
        frames.append(frame(q, target_q, z[-1]))
        boundary_alpha.append(exit_alpha); boundary_log.append(0.)
    end_q = k_ref*k_ref*identity-(kinetic+(exit_magnetic_k*width)**2*x)/exit_alpha**2
    transformed = outgoing_load(operators, end_q, k_ref, cancelled=cancelled, progress_callback=progress_callback)
    load = PhysicalCoordinateLoad(transformed, np.asarray(boundary_alpha), np.asarray(boundary_log))
    if refinement_plan is not None:
        refinement_plan.update(operators=operators, samplers=samplers, exit_q=end_q, kappa=k_ref,
                               alpha=np.asarray(boundary_alpha), log_derivative=np.asarray(boundary_log))
    chart_derivatives.extend([chart_derivatives[-1]]*(len(frames)+1-len(chart_derivatives)))
    return load, {"numerics": asdict(numerics), "estimated_working_bytes": estimated, "start_nm": start_nm,
        "executed_chart_digest": None if executed_chart is None else executed_chart.digest,
        "frames": frames, "rows": rows, "initial_width_nm": width0,
        "chart_derivatives": chart_derivatives,
        "chart_derivative_columns": ["width_log_rate_per_nm", "curvature_prime_per_nm2", "quartic_prime_per_nm5"],
        "quartic_phases_per_nm4": [phase(row["z_nm"])[0] for row in rows],
        "exit_quartic_phase_per_nm4": exit_quartic,
        "initial_curvature_per_nm": 1/radius, "reference_wave_number_per_nm": k_ref,
        "exit_width_nm": width, "exit_curvature_per_nm": curvature,
        "exit_energy_ev": kinetic_exit, "reference_flight_time_s": flight,
        "reference_longitudinal_action_j_s": action, "column_signature": column_signature,
        "reference_carrier_phase_mod_rad": float(np.angle(carrier_phasor)),
        "exit_axial_field_t": float(axial_b[-1]),
        "axial_coordinate": "Exact Liouville action coordinate; scalar correction, derivative jumps and both directions retained",
        "radial_integration": "Analytic Laguerre aperture moments and piecewise r^2-potential moments; two extra coordinate-product modes retained",
        "equation": "Scalar stationary Klein-Gordon orbital equation; moving m=0 Laguerre-Galerkin basis",
        "limitations": ["No spin, space charge or material tunnelling prediction",
            "Axisymmetric gun only; nonzero gun multipoles/Wien/slit are rejected",
            "Covariant scattering slabs; independent axial/basis/potential convergence required for the selected integrator",
            "Gun apertures and bores retain the existing absorbing-projection model"]}
