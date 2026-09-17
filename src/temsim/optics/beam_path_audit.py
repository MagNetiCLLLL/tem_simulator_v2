"""Current-weighted, tip-origin spot and crossover diagnostics.

These observations do not supply a downstream source or change any optics.
Gun-plane interpolation is diagnostic, not a nanometre-focus acceptance test.
Column planes are full-precision integration checkpoints. All sizes are radial
RMS or 95%-current DIAMETER, never an unlabeled mixture of radius and diameter.
"""
from dataclasses import asdict, dataclass, replace
import math
from temsim import input_io

import numpy as np

from temsim.physics.beam_statistics import transverse_beam_statistics


@dataclass(frozen=True)
class SpotMeasurement:
    name: str
    z_mm: float | None
    rays: int
    source_current_fraction: float
    rms_radius_nm: float
    diameter95_nm: float
    alpha95_mrad: float
    covariance_waist_offset_mm: float
    projected_emittance_x_m_rad: float
    projected_emittance_y_m_rad: float
    population: str
    precision: str


def uniform_cap_footprint(model):
    """Exact projected sizes of the configured uniform-area spherical cap.

    This is the physical emitting footprint, NOT a fitted virtual source or
    an image-plane beam. It remains exact when a numerical rule concentrates
    samples on a small optical acceptance and coarsely samples the outer cap.
    """
    model.validate()
    theta = math.radians(model.emission.cap_half_angle_deg)
    h = 2*math.sin(theta/2)**2
    radius = model.geometry.apex_radius_nm
    return dict(population="entire prescribed uniform-area emitting cap",
        precision="analytic geometry; not a propagated waist",
        edge_diameter_nm=2*radius*math.sin(theta),
        diameter95_nm=2*radius*math.sqrt(2*.95*h-(.95*h)**2),
        rms_radius_nm=radius*math.sqrt(h-h*h/3),
        cap_depth_nm=radius*h)


def spot_measurement(name, z_mm, coordinates, weights, alive, *,
                     population="all surviving current", precision="full precision"):
    """Measure a specified population without renormalising source current.

Projected geometric RMS emittance is sqrt(det Cov(x,x')); it is not a
conserved normalised emittance during acceleration, in a magnetic field, or
under nonlinear transport. It must not be advertised as a brightness bound.
"""
    x, y, tx, ty = (np.asarray(v, float) for v in coordinates)
    w = np.asarray(weights, float)
    mask = np.asarray(alive, bool).copy()
    if any(v.shape != w.shape for v in (x, y, tx, ty, mask)) or w.ndim != 1:
        raise ValueError("Spot arrays must have the same one-dimensional shape")
    if np.any(~np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Spot weights must be finite, nonnegative and nonempty")
    mask &= w > 0
    if any(np.any(~np.isfinite(v[mask])) for v in (x, y, tx, ty)):
        raise ValueError("Nonfinite current-carrying spot coordinates")
    if not mask.any():
        return SpotMeasurement(name, z_mm, 0, 0., *(math.nan,)*6, population, precision)
    stats = transverse_beam_statistics(x, y, tx, ty, weights=w, alive=mask)
    selected_w = w[mask]/w[mask].sum()

    def emittance(position, slope):
        # Weighted regression residual avoids subtracting nearly equal
        # covariance determinants for a strongly correlated expanding beam.
        p, a = position[mask], slope[mask]
        p = p-np.sum(selected_w*p)
        a = a-np.sum(selected_w*a)
        variance = np.sum(selected_w*p*p)
        residual = a-p*np.sum(selected_w*p*a)/variance if variance > 0 else a
        return float(np.sqrt(variance*np.sum(selected_w*residual*residual)))

    return SpotMeasurement(name, z_mm, stats.surviving_rays, stats.surviving_fraction,
        stats.radius_rms_m*1e9, 2*stats.radius_95_m*1e9, stats.convergence_95_mrad,
        stats.waist_offset_m*1e3, emittance(x, tx), emittance(y, ty), population, precision)


def emission_measurement(bundle, *, alive=None):
    """Physical cap footprint; a curved emitting surface is not an image plane.

Use full 3-D directions, including negative dz, rather than the ambiguous slope
representation at emission. Do not infer a field-free waist inside the metal.
"""
    w = np.asarray(bundle.weight, float)
    live = w > 0
    if alive is not None:
        alive = np.asarray(alive,bool)
        if alive.shape != live.shape:
            raise ValueError("Emission population must match the executed ray IDs")
        live &= alive
    zero = np.zeros_like(w)
    row = spot_measurement("Tip emission footprint", None,
        (bundle.x_m, bundle.y_m, zero, zero), w, live)
    if not live.any():
        return row
    direction = getattr(bundle, "surface_direction", None)
    if direction is None:
        direction = np.stack((bundle.tx_rad, bundle.ty_rad, np.ones_like(w)), axis=1)
    d = np.asarray(direction, float)[live]
    if d.shape != (int(live.sum()), 3) or np.any(~np.isfinite(d)) or np.any(np.linalg.norm(d, axis=1) == 0):
        raise ValueError("Invalid physical emission direction")
    d = d/np.linalg.norm(d, axis=1)[:, None]
    selected_w = w[live]/w[live].sum()
    chief = selected_w@d
    if np.linalg.norm(chief) < 1e-12:
        alpha = math.nan  # Isotropic population has no unique chief direction.
    else:
        chief /= np.linalg.norm(chief)
        angles = np.arctan2(np.linalg.norm(np.cross(d, chief), axis=1), np.clip(d@chief, -1, 1))
        order = np.argsort(angles)
        alpha = angles[order[min(np.searchsorted(np.cumsum(selected_w[order]), .95), len(order)-1)]]*1e3
    return replace(row, alpha95_mrad=float(alpha), covariance_waist_offset_mm=math.nan,
                   projected_emittance_x_m_rad=math.nan, projected_emittance_y_m_rad=math.nan,
                   population=("physical emitting surface; not a common transverse plane"
                               if getattr(bundle, "surface_direction", None) is not None
                               else "historical tip emission reference plane"))


def gun_plane_coordinates(trace, z_mm):
    """First forward crossing in executed time histories, excluding prior stops.

The result has the original ray-ID ordering; missing/stopped rays are masked,
never replaced by central rays. A later stop must not erase an earlier plane.
"""
    coordinates, masks = gun_planes_coordinates(trace, [float(z_mm)])
    return coordinates[0], masks[0]


def gun_planes_coordinates(trace, planes_mm):
    """Batch history observations, preserving first forward crossings and IDs.

    Search the running maximum once per ray rather than scanning its entire
    history again for every requested plane. This is only faster observation
    of an executed trace: no propagation or physical stop is bypassed.
    Output shapes are (planes, 4, rays) and (planes, rays).
    """
    z = np.asarray(planes_mm, float)
    history = trace.equal_time_history
    if history is None or z.ndim != 1 or not np.all(np.isfinite(z)):
        raise ValueError("Finite gun plane and executed time histories are required")
    weights = np.asarray(trace.exit_bundle.weight, float)
    mask = (weights > 0)[None, :] & (np.isnan(trace.blocked_z_mm)[None, :]
                                   | (trace.blocked_z_mm[None, :] > z[:, None]))
    at_exit = abs(z-float(trace.z_mm[-1])) <= 8*np.spacing(np.maximum(1., abs(z)))
    result = np.full((len(z), 4, weights.size), np.nan)
    observed = mask & ~at_exit[:, None]
    for ray in np.flatnonzero(observed.any(axis=0)):
        selected = np.flatnonzero(observed[:, ray])
        planes = z[selected]
        rz = history.z_mm[:, ray]
        height = np.maximum.accumulate(np.where(np.isfinite(rz), rz, -np.inf))
        upper = np.searchsorted(height, planes, side="left")
        if np.any(upper == len(rz)):
            missing = planes[upper == len(rz)][0]
            raise ValueError(f"Missing recorded forward crossing for ray {ray} at Z={missing} mm")
        lower = np.maximum(0, upper-1)
        if np.any(planes < rz[lower]):
            raise ValueError("Requested plane precedes an emitting surface point")
        if not np.isfinite(rz[lower]).all():
            raise ValueError("Nonfinite history at a forward crossing")
        fraction = np.divide(planes-rz[lower], rz[upper]-rz[lower], out=np.zeros_like(planes),
                             where=rz[upper] != rz[lower])
        for index, values in enumerate((history.x_m, history.y_m, history.tx_rad, history.ty_rad)):
            result[selected, index, ray] = values[lower, ray] + fraction*(values[upper, ray]-values[lower, ray])
    if np.any(at_exit):
        e = trace.exit_bundle
        result[at_exit] = np.array([e.x_m, e.y_m, e.tx_rad, e.ty_rad])
        mask[at_exit] = np.asarray(e.alive, bool) & (e.weight > 0)
    return result, mask


@input_io.using_state_inputs
def incident_checkpoints(state, planes_mm, *, step_mm=.05):
    """Execute/reuse this state's gun and propagate through all incident optics.

No caller-supplied exit bundle is accepted. Physical clipping determines the
mask at each requested plane; eventual specimen losses do not erase C1 data.
"""
    from temsim.optics.electron_gun.source import trace_source_to_exit
    from temsim.optics.direct_alignment import _pre_sample_kick_events
    from temsim.physics.core import propagate
    from temsim.physics.aperture_clipping import clip_segment
    from temsim.physics.column_wall import clip_column_wall
    start = float(state.electron_gun.exit_plane_z_mm)
    planes = np.asarray(planes_mm, float)
    if (planes.ndim != 1 or not planes.size or np.any(~np.isfinite(planes))
            or np.any(np.diff(planes) <= 0) or planes[0] < start
            or planes[-1] > state.sample.upper_surface_z_mm or not math.isfinite(step_mm) or step_mm <= 0):
        raise ValueError("Ordered audit planes must lie between gun exit and specimen entrance")
    if bool(getattr(getattr(state, "vacuum_map", None), "enabled", False)):
        raise ValueError("This spot audit does not yet qualify active residual-medium transport")
    gun = trace_source_to_exit(state)
    e = gun.exit_bundle
    apertures = tuple(a.z_mm for a in state.apertures if start < a.z_mm <= planes[-1])
    z, x, tx, y, ty, cp = propagate(state, start, float(planes[-1]),
        e.x_m, e.tx_rad, e.y_m, e.ty_rad, energy_offset_ev=e.energy_offset_ev,
        events=_pre_sample_kick_events(state), save_z_mm=tuple(planes)+apertures,
        checkpoint_z_mm=tuple(planes), return_checkpoints=True, maximum_step_mm=step_mm)
    alive = np.asarray(e.alive, bool).copy()
    stops = np.asarray(gun.blocked_z_mm, float).copy()
    keys = list(gun.blocked_key)
    alive, stops, keys = clip_segment(state, z, x, y, alive, stops, keys)
    alive, stops, keys = clip_column_wall(state, z, x, y, alive, stops, keys)
    mask = ((e.weight > 0)[None, :] & np.asarray(e.alive, bool)[None, :]
            & (np.isnan(stops)[None, :] | (stops[None, :] > cp.z_mm[:, None])))
    if len(cp.z_mm) != len(planes) or not np.allclose(cp.z_mm, planes, rtol=0, atol=1e-10):
        raise ValueError("The audit requires exact full-precision physical planes")
    return gun, cp, mask


def crossover_intervals(crossovers_mm, component_planes, *, boundary_tolerance_mm=1e-6):
    """Ordered topology including multiplicity, not just the nearest lens name.

All co-located component keys form one boundary. A crossover on a component is
distinct from one in either neighbouring gap. Use identical physical boundaries
for the reference and candidate; this function never moves a target into a gap.
"""
    positions = np.asarray(crossovers_mm, float)
    if (positions.ndim != 1 or np.any(~np.isfinite(positions))
            or np.any(np.diff(positions) <= 0)):
        raise ValueError("Crossover positions must be finite and strictly ordered")
    groups = {}
    for key, value in component_planes:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Component planes must be finite")
        groups.setdefault(value, []).append(str(key))
    ordered = sorted((z, tuple(sorted(keys))) for z, keys in groups.items())
    if len(ordered) < 2:
        raise ValueError("At least two component boundaries are required")
    if not math.isfinite(boundary_tolerance_mm) or boundary_tolerance_mm < 0:
        raise ValueError("Invalid boundary tolerance")
    result = []
    for z in positions:
        on = [keys for plane, keys in ordered if abs(plane-z) <= boundary_tolerance_mm]
        if on:
            result.append(("on", tuple(k for keys in on for k in keys)))
            continue
        before = [keys for plane, keys in ordered if plane < z]
        after = [keys for plane, keys in ordered if plane > z]
        if not before or not after:
            raise ValueError("Crossover is outside the declared optical path")
        result.append(("between", before[-1], after[0]))
    return tuple(result)


def optical_component_planes(state, *, full_path=False):
    """Installed incident optical centres, not schematic pole/body coordinates.

An installed but disabled lens is still a boundary in the assembly order.
Derived image planes, stage solids and decorative pole parts are not new
optical components. Paired devices use their canonical device reference plane.
With full_path, include gun apertures and post-specimen devices. Remaining
inside the accelerator body alone does not prove a waist stayed before DPA.
"""
    kinds = {"aperture", "continuous_aperture", "deflector", "paired_deflector",
             "finite_paired_deflector", "finite_quadrupole_stigmator", "hexapole",
             "magnetic_lens", "quadrupole", "round_lens", "sample", "stigmator",
             "electrostatic_lens", "extractor_electrode", "multistage_accelerator",
             "detector", "cold_field_emitter"}
    start = float(state.electron_gun.exit_plane_z_mm)
    lo, end = (-np.inf, np.inf) if full_path else (start, float(state.sample.z_mm))
    result = [(c.key, float(c.optical_reference_plane_z_mm))
              for c in state._resolved_optics_layout if c.kind in kinds
              and c.optical_reference_plane_z_mm is not None
              and lo <= c.optical_reference_plane_z_mm <= end]
    return sorted(result+[("gun_exit", start)], key=lambda item: (item[1], item[0]))


def require_same_topology(reference_mm, candidate_mm, component_planes, *, candidate_component_planes=None):
    reference = crossover_intervals(reference_mm, component_planes)
    candidate = crossover_intervals(candidate_mm,
        component_planes if candidate_component_planes is None else candidate_component_planes)
    if reference != candidate:
        raise ValueError(f"Crossover topology changed: reference={reference}; candidate={candidate}")
    return reference


def partition_surface_crossovers(crossovers, upper_surface_mm, *, tolerance_nm=1.):
    """Separate a requested terminal focus from intermediate optical waists.

    Apply the same declared surface-focus tolerance to reference and candidate.
    Keep every raw root in the report. No other crossover may be discarded or
    merged, including a near-surface root outside the physical focus tolerance.
    This classification does not establish either focus or convergence.
    """
    if (not math.isfinite(upper_surface_mm) or not math.isfinite(tolerance_nm)
            or tolerance_nm <= 0):
        raise ValueError("A finite surface and positive focus tolerance are required")
    intermediate, terminal = [], []
    for row in crossovers:
        z = float(row["z_mm"])
        if not math.isfinite(z):
            raise ValueError("Crossover positions must be finite")
        (terminal if abs(z-upper_surface_mm)*1e6 <= tolerance_nm else intermediate).append(row)
    return intermediate, terminal


def probe_qualification(measurement, *, target_mrad, maximum_diameter95_nm,
                        reference_crossovers_mm, candidate_crossovers_mm,
                        component_planes, sampling_converged):
    """An angle/focus fit alone must never pass as a qualified small probe.

The caller must supply an executed, documented topology reference and an
independent sampling verdict. This helper is a gate, not a physical solver.
"""
    if not math.isfinite(maximum_diameter95_nm) or maximum_diameter95_nm <= 0:
        raise ValueError("A positive physical probe-diameter limit is required")
    if type(sampling_converged) is not bool:
        raise ValueError("Sampling convergence must be an explicit boolean verdict")
    reasons = []
    if not measurement.accepts(target_mrad):
        reasons.append("angle_or_surface_focus")
    diameter = 2*measurement.statistics.radius_95_m*1e9
    if not math.isfinite(diameter) or diameter > maximum_diameter95_nm:
        reasons.append("probe_diameter")
    if not len(reference_crossovers_mm):
        reasons.append("missing_crossover_reference")
    else:
        try:
            require_same_topology(reference_crossovers_mm, candidate_crossovers_mm, component_planes)
        except ValueError:
            reasons.append("crossover_topology")
    if not sampling_converged:
        reasons.append("sampling_convergence")
    return dict(status="PASS" if not reasons else "FAIL", failures=reasons,
                diameter95_nm=float(diameter), diameter95_limit_nm=float(maximum_diameter95_nm))


def crossover_candidates(checkpoints, masks, weights):
    """Bracket current-weighted converging-to-diverging variance crossings.

Unlike plot-node minima, this can locate a narrow waist between samples.
Hermite coordinates use executed slopes. The interpolated Z is a proposal:
re-execute nearby exact planes and refine the axial sampling before acceptance.
The same surviving population is used on BOTH sides of each bracket, so a
loss of outer rays cannot manufacture a waist. Zero-current probes are ignored.
"""
    from scipy.optimize import brentq
    z = np.asarray(checkpoints.z_mm, float)
    arrays = [np.asarray(getattr(checkpoints, key), float) for key in
              ("x_m", "y_m", "tx_rad", "ty_rad")]
    masks, weights = np.asarray(masks, bool), np.asarray(weights, float)
    if (z.ndim != 1 or len(z) < 2 or np.any(~np.isfinite(z)) or np.any(np.diff(z) <= 0)
            or any(v.shape != masks.shape for v in arrays)
            or masks.shape != (len(z), len(weights)) or weights.ndim != 1
            or np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("Invalid crossover checkpoint arrays")
    roots = []
    for j in range(len(z)-1):
        live = masks[j] & masks[j+1] & (weights > 0)
        if live.sum() < 5:
            continue
        w = weights[live]/weights[live].sum()
        if any(np.any(~np.isfinite(v[j:j+2, live])) for v in arrays):
            raise ValueError("Nonfinite current-carrying crossover bracket")
        x, y, tx, ty = [v[j:j+2, live] for v in arrays]
        x = x-(x@w)[:, None]
        y = y-(y@w)[:, None]
        tx = tx-(tx@w)[:, None]
        ty = ty-(ty@w)[:, None]
        corr = ((x*tx+y*ty)@w)
        if not corr[0] < 0 <= corr[1]:
            continue
        h = (z[j+1]-z[j])*1e-3

        def interpolate(p, slope, t):
            point = ((2*t**3-3*t**2+1)*p[0] + (t**3-2*t**2+t)*h*slope[0]
                     + (-2*t**3+3*t**2)*p[1] + (t**3-t**2)*h*slope[1])
            derivative = ((6*t*t-6*t)*p[0]/h + (3*t*t-4*t+1)*slope[0]
                          + (-6*t*t+6*t)*p[1]/h + (3*t*t-2*t)*slope[1])
            return point, derivative

        def radial_derivative(t):
            px, vx = interpolate(x, tx, t)
            py, vy = interpolate(y, ty, t)
            return float(w@(px*vx+py*vy))

        t = brentq(radial_derivative, 0., 1., xtol=1e-12)
        px, _ = interpolate(x, tx, t)
        py, _ = interpolate(y, ty, t)
        roots.append(dict(z_mm=float(z[j]+t*(z[j+1]-z[j])),
            rms_radius_nm=float(np.sqrt(w@(px*px+py*py))*1e9),
            bracket_mm=(float(z[j]), float(z[j+1])), rays=int(live.sum()),
            status="INTERPOLATED_CANDIDATE_NOT_FOCUS_ACCEPTANCE"))
    return roots


def refine_crossover_candidate(state, candidate, *, step_mm=.05):
    """Re-execute a bracketed waist at physical planes; never trust plot minima.

Return scalar evidence only. Step/sampling convergence and the full topology
are deliberately separate qualifications.
"""
    from scipy.optimize import brentq
    left, right = map(float, candidate["bracket_mm"])
    if not np.isfinite([left, right]).all() or left >= right:
        raise ValueError("A strictly ordered crossover bracket is required")
    gun, cp, masks = incident_checkpoints(state, [left, right], step_mm=step_mm)
    population = masks[0] & masks[1]
    if population.sum() < 5:
        raise ValueError("Insufficient shared current-carrying crossover samples")
    weights = gun.exit_bundle.weight
    measured = {}

    def remember(z, cp, j, live):
        if np.any(population & ~live):
            raise ValueError("Crossover population changed inside its bracket")
        values = (cp.x_m[j], cp.y_m[j], cp.tx_rad[j], cp.ty_rad[j])
        stats = transverse_beam_statistics(*values, weights=weights, alive=population)
        measured[float(z)] = (stats.radial_position_angle_covariance_m_rad,
            spot_measurement("Crossover", float(z), values, weights, population,
                population="same current-carrying rays throughout crossover bracket"))

    remember(left, cp, 0, masks[0])
    remember(right, cp, 1, masks[1])

    def correlation(z):
        if float(z) not in measured:
            _, p, m = incident_checkpoints(state, [float(z)], step_mm=step_mm)
            remember(z, p, 0, m[0])
        return measured[float(z)][0]

    if not correlation(left) < 0 <= correlation(right):
        raise ValueError("Executed bracket does not contain a converging-to-diverging waist")
    root = float(brentq(correlation, left, right, xtol=1e-9, rtol=1e-13, maxiter=24))
    correlation(root)
    return dict(status="EXECUTED_AT_FIXED_SAMPLING", z_mm=root, bracket_mm=[left, right],
        step_mm=step_mm, evaluations=len(measured), measurement=asdict(measured[root][1]))
