"""Measure a tip-origin incident particle probe at the specimen upper surface.

Z increases downstream. Sample Z is the centre; the entrance is Z-t/2.
Full-precision transport checkpoints, not plotted coordinates, define focus.
The three focus planes are all upstream of (or exactly on) the material.
This module does not calculate diffraction or infer a wave-optical probe size.
"""
from dataclasses import dataclass
import math

import numpy as np

from temsim.optics.direct_alignment import _pre_sample_kick_events
from temsim.physics.aperture_clipping import clip_segment
from temsim.physics.beam_statistics import TransverseBeamStatistics, transverse_beam_statistics
from temsim.physics.column_wall import clip_column_wall
from temsim.physics.core import propagate


@dataclass(frozen=True)
class SurfaceFocusMeasurement:
    surface_z_mm: float
    statistics: TransverseBeamStatistics
    local_waist_offset_nm: float
    variance_second_derivative: float
    upstream_rms_radii_m: tuple[float, float, float]
    step_mm: float
    effective_rays: float | None = None

    def accepts(self, target_mrad: float, *, angle_relative_tolerance=.01,
                focus_tolerance_nm=1., minimum_current_fraction=0.,
                minimum_rays=16) -> bool:
        """Do not confuse preserved crossover count with preserved current.

        Real pupils may accept far less than 1% of source current. Require positive
        current and resolved statistical support; an optional minimum-current target
        belongs to the experiment, not an implicit source-transmission constraint.
        """
        s=self.statistics
        effective=self.effective_rays if self.effective_rays is not None else s.surviving_rays
        return bool(
            math.isfinite(target_mrad) and target_mrad>0
            and math.isfinite(minimum_current_fraction) and 0<=minimum_current_fraction<=1
            and math.isfinite(self.variance_second_derivative)
            and self.variance_second_derivative>0
            and math.isfinite(self.local_waist_offset_nm)
            and abs(self.local_waist_offset_nm)<=focus_tolerance_nm
            and abs(s.waist_offset_m*1e9)<=focus_tolerance_nm
            and abs(s.convergence_95_mrad/target_mrad-1)<=angle_relative_tolerance
            and s.surviving_rays>=minimum_rays
            and math.isfinite(effective)
            and effective>=minimum_rays*(1-32*np.finfo(float).eps)
            and math.isfinite(s.surviving_fraction) and s.surviving_fraction>0
            and s.surviving_fraction>=minimum_current_fraction)


def surface_z_mm(state) -> float:
    z=float(state.sample.z_mm)
    thickness=float(state.sample.thickness_nm)
    if not math.isfinite(z) or not math.isfinite(thickness) or thickness<0:
        raise ValueError('Surface focus requires finite sample Z and non-negative thickness')
    return z-thickness*.5e-6


def local_waist_from_radii(radii_m, spacing_m: float) -> tuple[float,float]:
    """One-sided variance derivative at the last of three upstream planes.

Returns (estimated local waist offset in nm, variance second derivative).
A positive second derivative distinguishes a local minimum from a maximum.
No field-free extrapolation across the specimen is needed.
"""
    r=np.asarray(radii_m,dtype=float)
    h=float(spacing_m)
    if r.shape!=(3,) or not np.all(np.isfinite(r)) or np.any(r<0) or not math.isfinite(h) or h<=0:
        raise ValueError('Focus derivative requires three finite radii and positive plane spacing')
    v=r*r
    # Difference before combining to reduce subtraction of the common radius.
    d0,d1=v[2]-v[1],v[1]-v[0]
    first=(3*d0-d1)/(2*h)
    second=(d0-d1)/(h*h)
    offset=-first/second*1e9 if second>0 else math.nan
    return float(offset),float(second)


def measure_surface_focus(state, *, step_mm=.05, upstream_spacing_nm=100.) -> SurfaceFocusMeasurement:
    """Execute the physical gun (or reuse its exact cached trace), then focus."""
    from temsim.optics.electron_gun.source import trace_source_to_exit
    from temsim.physics.beam_current import effective_source_current_a
    if state.electron_gun.source_representation!='classical_particles':
        raise ValueError('Surface particle focus requires the classical tip-origin source')
    if state.vacuum_map.enabled:
        raise ValueError('Surface focus with active residual-medium transport is not yet qualified')
    if effective_source_current_a(state)<=0:
        raise ValueError('Surface focus requires a nonzero physical source current')
    end=surface_z_mm(state)
    start=float(state.electron_gun.exit_plane_z_mm)
    h=float(upstream_spacing_nm)*1e-6
    if not math.isfinite(step_mm) or step_mm<=0 or not math.isfinite(h) or h<=0 or end-2*h<=start:
        raise ValueError('Focus planes must be downstream of the gun, with positive finite steps')
    planes=(end-2*h,end-h,end)
    gun=trace_source_to_exit(state)
    e=gun.exit_bundle
    apertures=tuple(a.z_mm for a in state.apertures if start<a.z_mm<end)
    z,x,tx,y,ty,checkpoints=propagate(state,start,end,e.x_m,e.tx_rad,e.y_m,e.ty_rad,
        events=_pre_sample_kick_events(state),energy_offset_ev=e.energy_offset_ev,
        # Checkpoint requests alone select the nearest existing grid node.
        # Explicit save planes also split integration at the exact surfaces.
        save_z_mm=apertures+planes,checkpoint_z_mm=planes,return_checkpoints=True,
        maximum_step_mm=step_mm)
    alive=np.asarray(e.alive,bool).copy()
    stops=np.asarray(gun.blocked_z_mm,float).copy()
    keys=list(gun.blocked_key)
    alive,stops,keys=clip_segment(state,z,x,y,alive,stops,keys)
    alive,stops,keys=clip_column_wall(state,z,x,y,alive,stops,keys)
    alive &= np.asarray(e.weight)>0
    rows=[]
    for plane in planes:
        indices=np.flatnonzero(np.abs(checkpoints.z_mm-plane)<1e-10)
        if len(indices)!=1:
            raise ValueError('Missing exact surface-focus checkpoint')
        row=int(indices[0])
        values=(checkpoints.x_m[row],checkpoints.y_m[row],checkpoints.tx_rad[row],checkpoints.ty_rad[row])
        if any(not np.all(np.isfinite(v[alive])) for v in values):
            raise ValueError('Nonfinite current-carrying surface-focus trajectory')
        rows.append(transverse_beam_statistics(*values,alive=alive,weights=e.weight))
    radii=tuple(row.radius_rms_m for row in rows)
    offset,second=local_waist_from_radii(radii,h*1e-3)
    weights=np.asarray(e.weight,float)[alive]
    weights=weights/np.sum(weights)
    effective_rays=float(1/np.sum(weights*weights))
    return SurfaceFocusMeasurement(end,rows[-1],offset,second,radii,float(step_mm),effective_rays)


def refine_surface_focus(state, target_mrad=30., *, initial=None, maximum_evaluations=50):
    """Return a detached, step-validated C3/objective candidate and its evidence.

No source, C1/C2, aperture, geometry or input state is changed. Acceptance here
is at the supplied particle budget, NOT a certification of sampling convergence.
"""
    from scipy.optimize import least_squares
    from temsim.instrument_snapshot import capture_instrument_snapshot
    keys=('condenser_lens_3','objective_lens')
    if not math.isfinite(target_mrad) or target_mrad<=0 or maximum_evaluations<1:
        raise ValueError('Focus target and evaluation budget must be positive')
    candidate=capture_instrument_snapshot(state).restore()
    lenses={lens.key:lens for lens in candidate.lenses}
    if any(key not in lenses or not lenses[key].enabled for key in keys):
        raise ValueError('Surface focus requires enabled C3 and objective lenses')
    upper=np.array([lenses[key].max_percent for key in keys],float)
    initial=np.array([lenses[key].percent for key in keys] if initial is None else initial,float)
    if initial.shape!=(2,) or not np.all(np.isfinite(initial)) or np.any(initial<0) or np.any(initial>upper):
        raise ValueError('Initial C3/objective settings must be inside their physical limits')
    measurements={}

    def set_controls(values):
        for key,value in zip(keys,values):
            lenses[key].percent=float(value)

    def measure(values):
        identity=tuple(float(v) for v in values)
        if identity not in measurements:
            set_controls(values)
            measurements[identity]=measure_surface_focus(candidate,step_mm=.025)
        return measurements[identity]

    def residual(values):
        s=measure(values).statistics
        return np.array([math.log(s.convergence_95_mrad/target_mrad),s.waist_offset_m/1e-6])

    fit=least_squares(residual,initial,bounds=(np.zeros(2),upper),
        max_nfev=int(maximum_evaluations),diff_step=1e-6,x_scale='jac',
        ftol=1e-11,xtol=1e-11,gtol=1e-11)
    refined=measure(fit.x)
    set_controls(fit.x)
    reference=measure_surface_focus(candidate,step_mm=.05)
    passed=bool(fit.success and reference.accepts(target_mrad) and refined.accepts(target_mrad))
    evidence=dict(status='PASS_AT_FIXED_PARTICLE_BUDGET' if passed else 'FAIL',
        sampling_convergence='NOT_TESTED',target_alpha95_mrad=float(target_mrad),
        probe_qualification='NOT_QUALIFIED: spot size, crossover topology and sampling require separate acceptance',
        particle_count=int(candidate.electron_gun.emitter.ray_count),
        strengths=dict(zip(keys,map(float,fit.x))),evaluations=len(measurements),
        optimizer_success=bool(fit.success),optimizer_message=str(fit.message),
        reference=reference,refined=refined)
    return candidate if passed else None,evidence
