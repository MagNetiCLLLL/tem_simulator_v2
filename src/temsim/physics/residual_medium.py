"""Classical residual-medium transport using independent-atom screened Coulomb events.

The Wentzel–Moliere approximation neglects spin, molecular bonding, nuclear
recoil and inelastic excitation. Its low-energy extrapolation is deliberately
reported, not represented as calibrated molecular cross-section data. Elastic
events conserve particle weight and kinetic energy. Only explicitly supplied
removal cross sections kill particles; exp(-n sigma L) is uncollided survival,
not a second absorption applied to elastically scattered electrons.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import numpy as np
from scipy.constants import alpha, c, e, epsilon_0, hbar, m_e, physical_constants

from temsim.vacuum import ResolvedMedium, resolve_regions

MODEL = "independent-atom-Wentzel-Moliere-elastic-v2-linear-gaps"
MODEL_SCOPE = ("Independent-atom screened elastic approximation; low-energy extrapolation, "
               "no molecular bonding, recoil, ionisation, stopping power or cell windows. "
               "Liquid is an elastic density surrogate. Scattered electrons retain weight; "
               "removal requires a separate supplied cross section. Column uses forward-Z optics.")


def atomic_cross_section(energy_ev, atomic_number):
    """Return total elastic sigma (m²), screening A; dσ/dΩ=C/(A+sin²(θ/2))².

    Geant4 Physics Reference Manual, Electron Screened Single Scattering,
    equations 95–96; integrate dΩ=4π d(sin²(θ/2)) over the full sphere.
    """
    energy = np.asarray(energy_ev, dtype=float)
    if np.any(~np.isfinite(energy)) or np.any(energy <= 0):
        raise ValueError("Residual-medium transport needs positive kinetic energy")
    rest = m_e*c*c
    kinetic = energy*e
    momentum = np.sqrt(kinetic*(kinetic+2*rest))/c
    beta = momentum*c/(kinetic+rest)
    a_tf = .88534*physical_constants["Bohr radius"][0]/atomic_number**(1/3)
    screening = (hbar/(2*momentum*a_tf))**2*(1.13+3.76*(alpha*atomic_number/beta)**2)
    coulomb = (atomic_number*e*e/(4*np.pi*epsilon_0)/(2*momentum*beta*c))**2
    sigma = 4*np.pi*coulomb/(screening*(screening+1))
    return sigma, screening


def medium_coefficients(medium, energy_ev):
    from ase.formula import Formula
    from ase.data import atomic_numbers
    atoms = Formula(medium.formula).count()
    channels = []
    total = np.zeros_like(np.asarray(energy_ev, dtype=float))
    density = medium.number_density_m3()
    for symbol, count in atoms.items():
        sigma, screening = atomic_cross_section(energy_ev, atomic_numbers[symbol])
        mu = density*count*sigma
        total = total+mu
        channels.append((mu, screening))
    return total, density*medium.removal_cross_section_m2, channels


def region_rate_bound(region, energy_ev):
    """Maximum endpoint hazard bounds every point of a linear transition."""
    a, removal, _ = medium_coefficients(region.medium, energy_ev)
    rate = a+removal
    if region.end_medium is not None:
        b, removal, _ = medium_coefficients(region.end_medium, energy_ev)
        rate = np.maximum(rate, b+removal)
    return rate


def linear_hazard_fraction(total_hazard, start_rate, end_rate, distance):
    """Invert ∫[0,tL] μ(l)dl, including decreasing and zero-start rates."""
    h = np.maximum(np.asarray(total_hazard), 0.)
    a, b = start_rate*distance, end_rate*distance
    root = np.sqrt(np.maximum(a*a+2*(b-a)*h, 0.))
    return np.clip(np.divide(2*h, a+root, out=np.zeros_like(h), where=(a+root)>0), 0., 1.)


def rotate_direction(direction, theta, phi):
    direction = np.asarray(direction, dtype=float)
    direction = direction/np.linalg.norm(direction, axis=-1, keepdims=True)
    axis = np.zeros_like(direction)
    axis[:, 2] = 1
    axis[np.abs(direction[:, 2]) > .9] = (1, 0, 0)
    u = np.cross(direction, axis)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(direction, u)
    return (direction*np.cos(theta)[:, None] + np.sin(theta)[:, None]
            *(u*np.cos(phi)[:, None]+v*np.sin(phi)[:, None]))


def segment_fraction(region, start_m, end_m):
    """Exact straight-segment clipping to a slab or finite cylindrical cell."""
    start, delta = np.asarray(start_m)*1000, (np.asarray(end_m)-start_m)*1000
    n = len(start)
    low, high = np.zeros(n), np.ones(n)
    dz = delta[:, 2]
    moving = np.abs(dz) > 1e-30
    a = np.divide(region.start_z_mm-start[:, 2], dz, out=np.zeros(n), where=moving)
    b = np.divide(region.end_z_mm-start[:, 2], dz, out=np.ones(n), where=moving)
    low[moving] = np.maximum(0, np.minimum(a, b)[moving])
    high[moving] = np.minimum(1, np.maximum(a, b)[moving])
    outside = (~moving) & ((start[:, 2] < region.start_z_mm) | (start[:, 2] >= region.end_z_mm))
    high[outside] = 0
    if region.radius_mm is not None:
        xy = start[:, :2]-np.array((region.center_x_mm, region.center_y_mm))
        dxy = delta[:, :2]
        aa = np.sum(dxy*dxy, axis=1)
        bb = 2*np.sum(xy*dxy, axis=1)
        cc = np.sum(xy*xy, axis=1)-region.radius_mm**2
        disc = bb*bb-4*aa*cc
        moving_xy = aa > 1e-30
        root = np.sqrt(np.maximum(disc, 0))
        enter = np.divide(-bb-root, 2*aa, out=np.zeros(n), where=moving_xy)
        leave = np.divide(-bb+root, 2*aa, out=np.ones(n), where=moving_xy)
        low[moving_xy] = np.maximum(low, enter)[moving_xy]
        high[moving_xy] = np.minimum(high, leave)[moving_xy]
        high[(disc < 0) | ((~moving_xy) & (cc > 0))] = 0
    return low, np.maximum(low, high)


class MediumTransport:
    """One executed segment's particle identities, hazards and path accounting."""
    def __init__(self, regions, count, seed, *, stream=0, alive=None, max_step_tau=.02):
        self.regions = tuple(regions)
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
        self.alive = np.ones(count, bool) if alive is None else np.asarray(alive, bool).copy()
        self.blocked_z = np.full(count, np.nan)
        self.blocked_key = [""]*count
        self.path_m = {r.key: np.zeros(count) for r in regions}
        self.tau = {r.key: np.zeros(count) for r in regions}
        self.events = {r.key: np.zeros(count, np.int64) for r in regions}
        self.elastic_clock = self.rng.exponential(size=count)
        self.removal_clock = self.rng.exponential(size=count)
        self.low_energy_path_m = np.zeros(count)
        self.max_step_tau = max_step_tau
        self._coefficients = {}
        self.solid_specimen = None

    def _coeff(self, region, energies):
        # Column energy is constant between elastic collisions; gun energy is
        # evaluated afresh on every actual accelerating step.
        key = (region.key, energies.tobytes())
        if key not in self._coefficients:
            first = medium_coefficients(region.medium, energies)
            last = first if region.end_medium is None else medium_coefficients(region.end_medium, energies)
            self._coefficients = {key: (first, last)}
        return self._coefficients[key]

    def advance(self, start, end, direction, energy_ev, *, forward_only=False):
        start, end = np.asarray(start), np.asarray(end)
        output = np.array(direction, dtype=float, copy=True)
        count = len(start)
        energies = np.broadcast_to(np.asarray(energy_ev, float), (count,))
        valid = self.alive & np.all(np.isfinite(start), axis=1) & np.all(np.isfinite(end), axis=1)
        length = np.zeros(count)
        length[valid] = np.linalg.norm(end[valid]-start[valid], axis=1)
        if not np.any(length):
            return output
        z_min = min(float(np.min(start[valid, 2])), float(np.min(end[valid, 2])))*1000
        z_max = max(float(np.max(start[valid, 2])), float(np.max(end[valid, 2])))*1000
        cell = next((r for r in self.regions if r.key == "specimen_cell"), None)
        if cell is not None and (cell.end_z_mm <= z_min or cell.start_z_mm >= z_max):
            cell = None
        cell_interval = segment_fraction(cell, start, end) if cell is not None else None
        intervals = []
        for r in self.regions:
            if r.end_z_mm <= z_min or r.start_z_mm >= z_max:
                continue
            if r.medium.number_density_m3() == 0 and (r.end_medium is None or r.end_medium.number_density_m3() == 0):
                continue
            lo, hi = segment_fraction(r, start, end)
            if r.radius_mm is None and cell_interval is not None:
                cl, ch = cell_interval
                # Replace, never add, the cell medium to the ambient column.
                intervals.extend(((r, lo, np.minimum(hi, np.maximum(lo, cl))),
                                  (r, np.maximum(lo, np.minimum(hi, ch)), hi)))
            else:
                intervals.append((r, lo, hi))
        solid_overlap = (self.solid_specimen is not None
                         and self.solid_specimen.z_mm-self.solid_specimen.thickness_nm*.5e-6 < z_max
                         and self.solid_specimen.z_mm+self.solid_specimen.thickness_nm*.5e-6 > z_min)
        if solid_overlap:
            sl, sh = specimen_interval(self.solid_specimen, start, end)
            trimmed = []
            for r, lo, hi in intervals:
                trimmed.extend(((r, lo, np.minimum(hi, np.maximum(lo, sl))),
                                (r, np.maximum(lo, np.minimum(hi, sh)), hi)))
            intervals = trimmed
        # The axial plan resolves all slab boundaries, but this also handles a
        # gun step straddling one boundary or a radial cell entry/exit.
        if len(intervals) > 1:
            # Radial crossings occur at different fractions for different
            # electrons. Sort separately per ray to preserve competing hazards.
            order = np.argsort(np.stack([v[1] for v in intervals]), axis=0, kind="stable")
            ordered = []
            for rank in range(len(intervals)):
                for k, (r, lo, hi) in enumerate(intervals):
                    selected = order[rank] == k
                    if np.any(selected & (hi > lo)):
                        ordered.append((r, lo, np.where(selected, hi, lo)))
            intervals = ordered
        else:
            intervals.sort(key=lambda item: item[0].start_z_mm)
        for r, lo, hi in intervals:
            fraction = np.maximum(hi-lo, 0)
            active = valid & self.alive & (fraction > 0) & (length > 0)
            if not np.any(active):
                continue
            safe_energy = np.where(np.isfinite(energies) & (energies > 0), energies, 1.)
            first, last = self._coeff(r, safe_energy)
            distance = np.where(active, length*fraction, 0)
            dz = (end[:, 2]-start[:, 2])*1000
            zstart = start[:, 2]*1000
            f0 = np.clip((zstart+lo*dz-r.start_z_mm)/(r.end_z_mm-r.start_z_mm), 0, 1)
            f1 = np.clip((zstart+hi*dz-r.start_z_mm)/(r.end_z_mm-r.start_z_mm), 0, 1)
            mu0, mu1 = first[0]*(1-f0)+last[0]*f0, first[0]*(1-f1)+last[0]*f1
            loss0, loss1 = first[1]*(1-f0)+last[1]*f0, first[1]*(1-f1)+last[1]*f1
            # Loss is an independent supplied process. Stop at its sampled
            # path coordinate so upstream readouts retain their correct flux.
            hit = active & ((loss0+loss1)*.5*distance >= self.removal_clock)
            traversed = np.ones(count)
            if np.any(hit):
                traversed[hit] = linear_hazard_fraction(self.removal_clock[hit], loss0[hit], loss1[hit], distance[hit])
                distance *= traversed
                t = lo[hit]+distance[hit]/length[hit]
                ids = np.flatnonzero(hit)
                self.blocked_z[hit] = (start[hit, 2]+t*(end[hit, 2]-start[hit, 2]))*1000
                for i in ids:
                    self.blocked_key[i] = f"medium_removal:{r.key}"
            mu1 = mu0+(mu1-mu0)*traversed
            loss1 = loss0+(loss1-loss0)*traversed
            f1 = f0+(f1-f0)*traversed
            tau = .5*(mu0+mu1)*distance
            if np.max(tau, initial=0) > max(.05, 2.5*self.max_step_tau):
                raise ValueError("Residual-medium step is too optically thick; reduce particle integration step")
            self.path_m[r.key] += distance
            self.tau[r.key] += tau
            self.low_energy_path_m += np.where(safe_energy < 50, distance, 0)
            self.elastic_clock -= tau
            self.removal_clock -= .5*(loss0+loss1)*distance
            for _ in range(1024):
                scatter = active & (self.elastic_clock <= 0)
                if not np.any(scatter):
                    break
                ids = np.flatnonzero(scatter)
                location = linear_hazard_fraction(tau[ids]+self.elastic_clock[ids], mu0[ids], mu1[ids], distance[ids])
                mixture = f0[ids]+(f1[ids]-f0[ids])*location
                local_mu = first[0][ids]*(1-mixture)+last[0][ids]*mixture
                choice = self.rng.random(len(ids))*local_mu
                cumulative = np.zeros(len(ids))
                screening = np.zeros(len(ids))
                for channels, mix in ((first[2], 1-mixture), (last[2], mixture)):
                    for channel_mu, aa in channels:
                        hazard = channel_mu[ids]*mix
                        selected = (choice >= cumulative) & (choice < cumulative+hazard)
                        screening[selected] = aa[ids][selected]
                        cumulative += hazard
                u = self.rng.random(len(ids))
                # Inverse CDF on s=sin²(theta/2): F=s(A+1)/(A+s).
                sine2 = screening*u/(screening+1-u)
                theta = 2*np.arcsin(np.sqrt(np.clip(sine2, 0, 1)))
                phi = self.rng.uniform(0, 2*np.pi, len(ids))
                output[ids] = rotate_direction(output[ids], theta, phi)
                self.events[r.key][ids] += 1
                self.elastic_clock[ids] += self.rng.exponential(size=len(ids))
                if forward_only:
                    reverse = ids[output[ids, 2] <= 0]
                    self.alive[reverse] = False
                    self.blocked_z[reverse] = end[reverse, 2]*1000
                    for i in reverse:
                        self.blocked_key[i] = f"medium_backscatter:{r.key}"
                    active[reverse] = False
            else:
                raise ValueError("Residual-medium collision budget exceeded; reduce integration step")
            self.alive[hit] = False
        return output

    def merge_stops(self, alive, blocked, keys, indices=None):
        sel = np.arange(len(self.alive)) if indices is None else np.arange(len(self.alive))[indices]
        for j, i in enumerate(sel):
            z = self.blocked_z[i]
            if np.isfinite(z) and (not np.isfinite(blocked[j]) or z < blocked[j]):
                alive[j], blocked[j], keys[j] = False, z, self.blocked_key[i]
        return alive, blocked, keys

    def report(self, indices=None):
        selected = slice(None) if indices is None else indices
        count = len(self.alive[selected])
        stop_keys = np.asarray(self.blocked_key, dtype=object)[selected].tolist()
        return {"model": MODEL, "scope": MODEL_SCOPE, "regions": [
            {"key": r.key, "name": r.name, "medium": asdict(r.medium),
             "end_medium": asdict(r.end_medium) if r.end_medium is not None else None,
             "mean_path_m": float(np.mean(self.path_m[r.key][selected])) if count else 0.,
             "mean_elastic_optical_depth": float(np.mean(self.tau[r.key][selected])) if count else 0.,
             "mean_uncollided_fraction": float(np.mean(np.exp(-self.tau[r.key][selected]))) if count else 1.,
             "elastic_events": int(np.sum(self.events[r.key][selected]))}
            for r in self.regions],
            "low_energy_extrapolation_path_m": float(np.sum(self.low_energy_path_m[selected])),
            "removal_events": sum(k.startswith("medium_removal:") for k in stop_keys),
            "backscatter_exits": sum(k.startswith("medium_backscatter:") for k in stop_keys)}


def medium_grid_nodes(state, z0, z1, energies=None):
    """Resolve cell/slab boundaries and optically thin steps without truncation."""
    if not getattr(state, "vacuum_map", None) or not state.vacuum_map.enabled:
        return []
    energies = np.array([state.beam_voltage_kv*1000]) if energies is None else np.asarray(energies)
    nodes = []
    for r in resolve_regions(state):
        a, b = max(z0, r.start_z_mm), min(z1, r.end_z_mm)
        if b <= a:
            continue
        rate = float(np.max(region_rate_bound(r, energies), initial=0))
        # Factor two allows modest off-axis path elongation. Actual step tau
        # is checked again at the executed trajectory, never silently capped.
        step = state.vacuum_map.max_optical_depth_per_step/max(rate*2, 1e-300)*1000
        if r.radius_mm is not None:
            step = min(step, (b-a)/16, r.radius_mm/4)
        count = max(1, math.ceil((b-a)/step))
        if len(nodes)+count+1 > state.vacuum_map.max_transport_nodes:
            raise ValueError("Vacuum medium requires more integration nodes than the configured budget")
        nodes.extend(np.linspace(a, b, count+1))
    return nodes


class ColumnMediumTransport(MediumTransport):
    """Step-end collision operator between the existing field integrations.

    Physical stops bound every integration segment BEFORE path accounting.
    No stochastic operator is used for transfer matrices / calibration rays.
    """
    def __init__(self, state, plan, count, energies, *, alive=None, stream=1):
        regions = tuple(r for r in resolve_regions(state)
                        if r.start_z_mm < plan.z_mm[-1] and r.end_z_mm > plan.z_mm[0])
        super().__init__(regions, count, state.vacuum_map.seed, alive=alive, stream=stream,
                         max_step_tau=state.vacuum_map.max_optical_depth_per_step)
        self.state, self.z, self.energies = state, plan.z_mm, energies
        from temsim.specimen.source import specimen_is_vacuum
        if (getattr(state.sample, "inserted", True) and not specimen_is_vacuum(state.sample)
                and state.sample.thickness_nm > 0):
            self.solid_specimen = state.sample
        self.node_radius = np.full(len(self.z), np.inf)
        self.interval_radius = np.full(len(self.z)-1, np.inf)
        from temsim.physics.column_wall import _vacuum_segments
        for segment in _vacuum_segments(state, self.z):
            lo, hi = segment.start_z_mm, segment.end_z_mm
            radius = segment.inner_diameter_mm*.5*1e-3
            mask = (self.z >= lo-1e-9) & (self.z <= hi+1e-9)
            self.node_radius[mask] = np.minimum(self.node_radius[mask], radius)
            midpoint = (self.z[:-1]+self.z[1:])/2
            mask = (midpoint >= lo) & (midpoint <= hi)
            self.interval_radius[mask] = np.minimum(self.interval_radius[mask], radius)
        planes = [float(a.z_mm) for a in state.apertures
                  if getattr(a, "enabled", True) and getattr(a, "installed", True)]
        for attr in ("camera", "fluorescent_screen", "haadf_detector", "df_detector", "bf_detector"):
            device = getattr(state, attr, None)
            if device is not None:
                planes.append(float(device.z_mm))
        planes.extend(float(p.z_mm) for p in getattr(state, "recording_planes", ()))
        self.recording_keys = {p.key for p in getattr(state, "recording_planes", ())}
        self.plane_steps = {max(0, int(np.searchsorted(self.z, p, side="left"))-1)
                            for p in planes if self.z[0] <= p <= self.z[-1]}

    def __call__(self, j, before, after):
        x0, tx0, y0, ty0 = before
        x, tx, y, ty = after
        was_alive = self.alive.copy()
        starts = np.column_stack((x0, y0, np.full(len(x), self.z[j]*1e-3)))
        ends = np.column_stack((x, y, np.full(len(x), self.z[j+1]*1e-3)))
        stop_z = np.full(len(x), np.nan)
        stop_keys = [""]*len(x)
        # Resolve a circular wall and a possible diameter shoulder at this
        # exact saved/integration boundary, including inward-then-outward rays.
        radius = self.interval_radius[j]
        old_rad2, rad2 = x0*x0+y0*y0, x*x+y*y
        outside0 = was_alive & (old_rad2 >= self.node_radius[j]**2)
        outer = was_alive & (rad2 >= radius*radius) & ~outside0
        shoulder = was_alive & (rad2 >= self.node_radius[j+1]**2) & ~outside0 & ~outer
        stop_z[outside0] = self.z[j]
        stop_z[shoulder] = self.z[j+1]
        if np.any(outer):
            dx, dy = x-x0, y-y0
            a, b = dx*dx+dy*dy, 2*(x0*dx+y0*dy)
            cc = old_rad2-radius*radius
            disc = np.maximum(b*b-4*a*cc, 0)
            frac = np.divide(-b+np.sqrt(disc), 2*a, out=np.ones(len(x)), where=a > 0)
            stop_z[outer] = self.z[j]+np.clip(frac[outer], 0, 1)*(self.z[j+1]-self.z[j])
        for i in np.flatnonzero(outside0 | outer | shoulder):
            stop_keys[i] = "column_wall"
        if j in self.plane_steps:
            zz = self.z[j:j+2]
            xx, yy = np.stack((x0, x)), np.stack((y0, y))
            al = was_alive & ~np.isfinite(stop_z)
            if self.z[j+1] <= self.state.sample.z_mm:
                from temsim.physics.simulation import _clip_aperture_segment
                al, stop_z, stop_keys = _clip_aperture_segment(self.state, zz, xx, yy, al, stop_z, stop_keys)
            else:
                from temsim.physics.recording_clipping import clip_recording_planes
                al, stop_z, stop_keys = clip_recording_planes(self.state, zz, xx, yy, al, stop_z, stop_keys)
        fraction = np.where(np.isfinite(stop_z), np.clip((stop_z-self.z[j])/(self.z[j+1]-self.z[j]), 0, 1), 1)
        physical_end = starts+(ends-starts)*fraction[:, None]
        directions = np.column_stack((tx, ty, np.ones(len(x))))
        directions = self.advance(starts, physical_end, directions, self.energies, forward_only=True)
        physical = was_alive & np.isfinite(stop_z) & (~np.isfinite(self.blocked_z) | (stop_z < self.blocked_z))
        self.blocked_z[physical] = stop_z[physical]
        for i in np.flatnonzero(physical):
            self.blocked_key[i] = stop_keys[i]
        self.alive[physical] = False
        forward = self.alive & np.isfinite(directions[:, 2]) & (directions[:, 2] > 0)
        tx[forward] = directions[forward, 0]/directions[forward, 2]
        ty[forward] = directions[forward, 1]/directions[forward, 2]
        # Existing affine raster previews require optical continuations beyond
        # an intercepted recording plane. Those are diagnostic ghost paths,
        # never live medium transport: self.alive remains false, so neither
        # gas path nor collisions accrue after absorption. Other stopped rays
        # can be frozen and cannot be rescued by this geometric preview.
        continuation = np.array([key in self.recording_keys for key in self.blocked_key])
        stopped = ~self.alive & ~continuation
        x[stopped], y[stopped] = physical_end[stopped, 0], physical_end[stopped, 1]
        frozen = ~was_alive & ~continuation
        x[frozen], y[frozen] = x0[frozen], y0[frozen]
        tx[stopped], ty[stopped] = 0., 0.
        return x, tx, y, ty


def specimen_interval(sample, start, end):
    """Exclude the finite solid specimen from ambient gas / cell liquid."""
    from temsim.vacuum import Medium
    half = sample.thickness_nm*.5e-6
    slab = ResolvedMedium("solid", "solid", sample.z_mm-half, sample.z_mm+half, Medium())
    lo, hi = segment_fraction(slab, start, end)
    scales = np.array((sample.size_x_nm, sample.size_y_nm))*.5e-9
    centre = np.array((sample.centre_x_nm, sample.centre_y_nm))*1e-9
    p = (start[:, :2]-centre)/scales
    d = (end[:, :2]-start[:, :2])/scales
    if sample.envelope_shape == "disk":
        aa, bb, cc = np.sum(d*d, axis=1), 2*np.sum(p*d, axis=1), np.sum(p*p, axis=1)-1
        disc = bb*bb-4*aa*cc
        moving = aa > 1e-30
        root = np.sqrt(np.maximum(disc, 0))
        a = np.divide(-bb-root, 2*aa, out=np.zeros(len(p)), where=moving)
        b = np.divide(-bb+root, 2*aa, out=np.ones(len(p)), where=moving)
        lo, hi = np.maximum(lo, a), np.minimum(hi, b)
        hi[(disc < 0) | ((~moving) & (cc > 0))] = 0
    else:
        for axis in (0, 1):
            moving = np.abs(d[:, axis]) > 1e-30
            a = np.divide(-1-p[:, axis], d[:, axis], out=np.zeros(len(p)), where=moving)
            b = np.divide(1-p[:, axis], d[:, axis], out=np.ones(len(p)), where=moving)
            lo = np.maximum(lo, np.minimum(a, b))
            hi = np.minimum(hi, np.maximum(a, b))
            hi[(~moving) & (np.abs(p[:, axis]) > 1)] = 0
    return lo, np.maximum(lo, hi)
