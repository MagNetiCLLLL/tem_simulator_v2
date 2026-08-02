from temsim.physics.crossovers import first_crossover_after_lens

from temsim.physics.column_wall import clip_column_wall

from temsim.physics.aperture_clipping import clip_segment as _clip_aperture_segment

from dataclasses import dataclass

import numpy as np, math

from temsim.physics.chromatic import objective_chromatic_kick

from temsim.physics.core import propagate,transfer,electron

from temsim.physics.beam_waist import detect_beam_waist


from temsim.physics.corrector_crossovers import detect_corrector_crossovers

from temsim.physics.recording_stop import determine_tem_stop_z
from temsim.physics.recording_clipping import clip_recording_planes
from temsim.component_keys import CONDENSER_LENS_2, CONDENSER_LENS_3

COLOURS={'000':(1.,.9,.2),'+g':(1.,.15,.1),'-g':(.1,.7,1.)}

@dataclass

class Branch:
    name:str; colour:tuple; z:np.ndarray; x:np.ndarray; y:np.ndarray; tx:np.ndarray; ty:np.ndarray; alive:np.ndarray; blocked_z:np.ndarray; blocked_key:list; weight:float; energy_offset_ev:np.ndarray; ray_weight:np.ndarray|None=None

@dataclass

class Simulation:

    incident:Branch; branches:dict; metrics:dict; gun_waist:dict|None=None; c2c3_crossover:dict|None=None; corrector_crossovers:list|None=None; gun_trace:object|None=None


def _legacy_clip_unused(s,z,X,Y):

    n=X.shape[1]; alive=np.ones(n,bool); blocked=np.full(n,np.nan); keys=['']*n

    for ap in sorted(s.apertures,key=lambda q:q.z_mm):

        if not ap.enabled: continue

        idx=int(np.argmin(abs(z-ap.z_mm)))

        passed=np.hypot(X[idx]*1e3-ap.offset_x_mm,Y[idx]*1e3-ap.offset_y_mm)<=ap.radius_mm

        new=alive&~passed;blocked[new]=ap.z_mm

        for j in np.flatnonzero(new):keys[j]=ap.key

        alive &= passed

    return alive,blocked,keys


def run(s, *, resolved_layout=None):
    # The low-level entry point is also public and is used directly by tests
    # and scripts, so it must enforce the same TOML-owned geometry contract as
    # the application-facing calculation pipeline.
    if resolved_layout is None:
        from temsim.column.state_layout import apply_physical_layout_to_state
        resolved_layout = apply_physical_layout_to_state(s)

    gun=s.electron_gun.validate()
    gun_trace=gun.trace_to_exit()
    emitted=gun_trace.exit_bundle
    x,y=emitted.x_m,emitted.y_m
    tx,ty=emitted.tx_rad,emitted.ty_rad
    dE=emitted.energy_offset_ev
    n=x.size
    phi=np.arctan2(y,x)

    pre_events=[]

    post_events=[]

    for d in s.deflectors:

        if not d.enabled:

            continue

        if hasattr(d, "kick_events"):
            pair = d.kick_events()
        else:
            pair=[(d.upper_z_mm,d.upper_x_mrad*1e-3,d.upper_y_mrad*1e-3),(d.lower_z_mm,d.lower_x_mrad*1e-3,d.lower_y_mrad*1e-3)]

        for event in pair:

            (pre_events if event[0] <= s.sample.z_mm else post_events).append(event)

    for component in getattr(s, "corrector_elements", []):
        if not getattr(component, "enabled", False):
            continue
        if not hasattr(component, "kick_events"):
            continue
        try:
            events = component.kick_events(
                time_s=float(getattr(s, "simulation_time_s", 0.0))
            )
        except TypeError:
            events = component.kick_events()
        for event in events:
            (
                pre_events
                if event[0] <= s.sample.z_mm
                else post_events
            ).append(event)

    z_column,X_column,TX_column,Y_column,TY_column=propagate(
        s,gun.exit_plane_z_mm,s.sample.z_mm,
        x,tx,y,ty,pre_events,dE
    )
    z=np.r_[gun_trace.z_mm,z_column[1:]]
    X=np.vstack((gun_trace.x_m,X_column[1:]))
    TX=np.vstack((gun_trace.tx_rad,TX_column[1:]))
    Y=np.vstack((gun_trace.y_m,Y_column[1:]))
    TY=np.vstack((gun_trace.ty_rad,TY_column[1:]))
    alive=emitted.alive.copy()
    blocked=gun_trace.blocked_z_mm.copy()
    keys=list(gun_trace.blocked_key)
    alive,blocked,keys=_clip_aperture_segment(
        s,z,X,Y,alive,blocked,keys
    )

    alive,blocked,keys=clip_column_wall(s,z,X,Y,alive,blocked,keys)

    incident=Branch('incident',(.2,.4,1.),z,X,Y,TX,TY,alive,blocked,keys,1.,dE,emitted.weight)

    gun_field_end,gun_diagnostic_end=gun.diagnostic_waist_region_mm
    gun_waist=detect_beam_waist(
        incident,gun_field_end,gun_diagnostic_end
    )
    s.last_gun_waist_mm=float('nan') if gun_waist is None else gun_waist['z_mm']

    _,_,lam=electron(s);theta=lam*s.sample.g_inv_nm;se=s.sample.excitation_error_inv_nm;sig=max(1e-9,s.sample.rocking_width_inv_nm);weight=math.exp(-.5*(se/sig)**2);diffuse=s.sample.diffuse_broadening_mrad*1e-3*min(1.,abs(se)/(3*sig));jitter=diffuse*np.sin(phi*1.73)

    branches={}

    chromatic_tx=np.zeros(n);chromatic_ty=np.zeros(n)

    if getattr(s,'chromatic_aberration_enabled',False):


        try:

            from temsim.optics.lens_focal_length import focal_length_mm

            obj=s.objective_lens

            fobj=focal_length_mm(obj,s.beam_voltage_kv)

        except Exception:

            fobj=2.5

        chromatic_tx,chromatic_ty=objective_chromatic_kick(
            X[-1],Y[-1],dE,s.beam_voltage_kv*1000.0,
            float(getattr(s.objective_lens,'cc_mm',2.0) or 2.0),fobj
        )

    branch_specs=[('000',0.,1.)]+([('+g',theta,weight),('-g',-theta,weight)] if getattr(s.sample,'diffraction_enabled',True) else [])

    for name,kick,w in branch_specs:

        zp,XP,TP,YP,TYP=propagate(s,s.sample.z_mm,determine_tem_stop_z(s),X[-1],TX[-1]+kick+jitter+chromatic_tx,Y[-1],TY[-1]+chromatic_ty,post_events,dE)

        # Post-sample apertures and recording planes are resolved together
        # below so upstream stops always win over downstream stops.

        al=alive.copy();bl=blocked.copy();ks=list(keys)

        # Resolve detector and wall candidates, then keep the earliest axial
        # intercept.  Column-wall clipping compares against an existing stop,
        # so a later wall cannot hide an earlier detector and vice versa.
        al,bl,ks=clip_recording_planes(s,zp,XP,YP,al,bl,ks)

        al,bl,ks=clip_column_wall(s,zp,XP,YP,al,bl,ks)

        branches[name]=Branch(name,COLOURS[name],zp,XP,YP,TP,TYP,al,bl,ks,w,dE,emitted.weight)

    post=transfer(s,s.sample.z_mm,determine_tem_stop_z(s));half=s.camera.width_mm/2;M=post[0,0];B=post[0,1]

    if s.projector_mode=='image':

        metrics={'mode':'image','magnification':abs(M),'object_full_m':s.camera.width_mm*1e-3/max(abs(M),1e-15),'relay_error':abs(B)}

    else:

        L=max(abs(B),1e-15);mrad_half=half/L;metrics={'mode':'diffraction','effective_camera_length_m':L,'mrad_half':mrad_half,'g_half_inv_nm':mrad_half*1e-3/lam,'relay_error':abs(post[0,0])}

    metrics.update({'lambda_nm':lam,'theta_g_mrad':theta*1e3,'diffraction_weight':weight})

    crossovers=detect_corrector_crossovers(incident,getattr(s,"corrector_crossover_targets_mm",[810.0,853.0,963.0]))

    lens_map={lens.key:lens for lens in s.lenses}

    c2c3=None

    condenser_lens_3 = lens_map.get(CONDENSER_LENS_3)
    condenser_lens_3_enabled = (
        condenser_lens_3 is not None
        and getattr(condenser_lens_3, "enabled", True)
    )
    if (
        getattr(s,"column_mode","three_lens")=="three_lens"
        and condenser_lens_3_enabled
        and CONDENSER_LENS_2 in lens_map
    ):
        c2c3=first_crossover_after_lens(
            incident,lens_map[CONDENSER_LENS_2].z_mm,
            "C2-C3 intermediate image crossover",
            stop_z_mm=condenser_lens_3.z_mm)

    result=Simulation(
        incident=incident, branches=branches, metrics=metrics, gun_waist=gun_waist,
        c2c3_crossover=c2c3, corrector_crossovers=crossovers,
        gun_trace=gun_trace,
    )

    return result
