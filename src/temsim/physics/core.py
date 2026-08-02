from temsim.physics.acceleration import momentum_profile
from temsim.component_keys import CONDENSER_LENS_KEYS
"""Parallel ray propagator with downsampled history.

Uses Numba across rays when available. The three-Gaussian fields and physical-axis
stigmators are unchanged. Full integration uses state.step_mm; only stored plotting
history is downsampled to reduce memory pressure.
"""
import math
import numpy as np

from temsim.optics.lens_focal_length import focal_length_mm

E=1.602176634e-19
M=9.1093837015e-31
C=299792458.0
H=6.62607015e-34

try:
    from numba import njit, prange
    NUMBA_AVAILABLE=True
except Exception:
    NUMBA_AVAILABLE=False
    def njit(*args,**kwargs):
        def wrapper(func): return func
        return wrapper
    range_parallel=range
    prange=range

def electron(state):
    kinetic=E*state.beam_voltage_kv*1000.0
    rest=M*C*C
    momentum=math.sqrt(kinetic*kinetic+2.0*kinetic*rest)/C
    # Electron charge is signed.  Keeping it signed is essential for magnetic
    # image rotation; round-lens focusing depends on q**2, while the Larmor
    # angle changes sign with q*Bz.
    return -E,momentum,H/momentum*1.0e9

def fields(z,state):
    if hasattr(state,"sync_objective"): state.sync_objective()
    z=np.asarray(z,float)
    magnetic=np.zeros_like(z)
    sx=np.zeros_like(z)
    sy=np.zeros_like(z)
    for lens in state.lenses:
        if not getattr(lens,"enabled",True): continue
        polarity = int(getattr(lens, "polarity", 1))
        if polarity not in (-1, 1):
            raise ValueError(
                f"{getattr(lens, 'name', 'Round lens')} polarity must be +1 or -1."
            )
        if lens.key in CONDENSER_LENS_KEYS:
            magnetic += state.condenser_system[lens.key].magnetic_field_t(z)
            continue
        if hasattr(lens, "magnetic_field_t"):
            magnetic += lens.magnetic_field_t(z)
            continue
        for g in lens.gaussian:
            magnetic += lens.scale()*g.amplitude*np.exp(-0.5*((z-(lens.z_mm+g.offset*lens.a_mm))/(g.sigma*lens.a_mm))**2)
    for stig in state.stigmators:
        if not stig.enabled: continue
        if hasattr(stig, "quadrupole_strengths_m2"):
            qx, qy = stig.quadrupole_strengths_m2(z)
            sx += qx
            sy += qy
            continue
        envelope=np.exp(-0.5*((z-stig.z_mm)/max(1e-12,stig.length_mm/2.355))**2)
        xset=stig.max_strength_m2*stig.strength_x_percent/100.0
        yset=stig.max_strength_m2*stig.strength_y_percent/100.0
        q=0.5*(xset-yset)*envelope
        sx += q
        sy -= q
    for component in getattr(state, "corrector_elements", []):
        if not getattr(component, "enabled", False):
            continue
        if not hasattr(component, "quadrupole_strength_m2"):
            continue
        q = component.quadrupole_strength_m2(z)
        sx += q
        sy -= q
    return magnetic,sx,sy

def bz(z,state): return fields(z,state)[0]


def hexapole_field(z, state):
    """Sum continuous nonlinear hexapole coefficients on the optical axis."""

    z = np.asarray(z, float)
    strength = np.zeros_like(z)
    for component in getattr(state, "corrector_elements", []):
        if not getattr(component, "enabled", False):
            continue
        if not hasattr(component, "hexapole_strength_m3"):
            continue
        strength += component.hexapole_strength_m3(z)
    return strength


def larmor_coefficients_m1(magnetic_t, momentum, z_mm):
    """Return signed Larmor rate and its axial derivative.

    The coordinates use a right-handed frame with electrons travelling along
    +Z.  ``polarity=+1`` means +Bz and positive rotation follows the right-hand
    rule about +Z.  The rate is ``-q Bz / (2 pz)``; the coupled laboratory-frame
    equations below use ``g = q Bz / (2 pz) = -d(phi)/dz``.
    """

    magnetic = np.asarray(magnetic_t, dtype=float)
    momentum = np.asarray(momentum, dtype=float)
    z_m = np.asarray(z_mm, dtype=float) * 1.0e-3
    g = (-E) * magnetic[:, None] / (2.0 * momentum)
    if g.ndim == 1:
        g = g[:, None]
    if len(z_m) < 2:
        gradient = np.zeros_like(g)
    else:
        gradient = np.gradient(g, z_m, axis=0, edge_order=1)
    return (
        np.ascontiguousarray(g, dtype=np.float64),
        np.ascontiguousarray(gradient, dtype=np.float64),
    )


def spherical_aberration_kick_m3(z_mm, state):
    """Return thin, rotationally symmetric third-order lens-kick strengths.

    A positive conventional ``Cs`` increases the inward deflection of marginal
    rays.  For an equivalent focal length ``f`` the kick is
    ``delta slope = -(Cs/f**4) * r**2 * r_vector``.  It therefore reproduces
    the standard transverse blur magnitude ``Cs * alpha**3`` at the paraxial
    focal plane for a collimated bundle.  Field-derived aberration must not be
    combined with this calibrated preview model.
    """

    z = np.asarray(z_mm, dtype=float)
    result = np.zeros(z.size, dtype=np.float64)
    if z.size == 0:
        return result
    half_step = math.inf if z.size < 2 else 0.5 * abs(float(z[1] - z[0]))
    for lens in getattr(state, "lenses", ()):
        if not bool(getattr(lens, "enabled", True)):
            continue
        cs_mm = getattr(lens, "cs_mm", None)
        if cs_mm is None or float(cs_mm) == 0.0:
            continue
        if not math.isfinite(float(cs_mm)):
            raise ValueError(
                f"{getattr(lens, 'name', 'Round lens')} Cs must be finite."
            )
        lens_z = float(getattr(lens, "z_mm"))
        if lens_z < float(z[0]) - half_step or lens_z > float(z[-1]) + half_step:
            continue
        try:
            focal_mm = float(focal_length_mm(lens, state.beam_voltage_kv))
        except Exception:
            continue
        if not math.isfinite(focal_mm) or focal_mm <= 0.0:
            continue
        index = int(np.argmin(np.abs(z - lens_z)))
        cs_m = float(cs_mm) * 1.0e-3
        focal_m = focal_mm * 1.0e-3
        result[index] += cs_m / focal_m**4
    return result


@njit(cache=True,parallel=True,fastmath=True)
def _parallel_rk4(
    kx, ky, hex_strength, larmor_g, larmor_gradient, cs_kick, h,
    x0, tx0, y0, ty0,
    kickx, kicky, save_index, es_alpha, es_beta,
    gun_index, gun_focal_m,
):
    nr=x0.size
    ns=save_index.size
    X=np.empty((ns,nr),np.float32)
    TX=np.empty((ns,nr),np.float32)
    Y=np.empty((ns,nr),np.float32)
    TY=np.empty((ns,nr),np.float32)
    for ray in prange(nr):
        x=x0[ray];tx=tx0[ray];y=y0[ray];ty=ty0[ray];s=0
        for j in range(kx.shape[0]):
            tx += kickx[j];ty += kicky[j]
            if cs_kick[j] != 0.0:
                radius_sq=x*x+y*y
                tx -= cs_kick[j]*radius_sq*x
                ty -= cs_kick[j]*radius_sq*y
            if s<ns and j==save_index[s]:
                X[s,ray]=x;TX[s,ray]=tx;Y[s,ray]=y;TY[s,ray]=ty;s+=1
            if j>=kx.shape[0]-1: continue
            kx0=kx[j,ray];ky0=ky[j,ray];kxm=0.5*(kx[j,ray]+kx[j+1,ray]);kym=0.5*(ky[j,ray]+ky[j+1,ray]);kx1=kx[j+1,ray];ky1=ky[j+1,ray]
            g0=larmor_g[j,ray];gm=0.5*(larmor_g[j,ray]+larmor_g[j+1,ray]);g1=larmor_g[j+1,ray]
            dg0=larmor_gradient[j,ray];dgm=0.5*(larmor_gradient[j,ray]+larmor_gradient[j+1,ray]);dg1=larmor_gradient[j+1,ray]
            h0=hex_strength[j];hm=0.5*(hex_strength[j]+hex_strength[j+1]);h1=hex_strength[j+1]
            ax1=tx;ay1=ty
            ax2=-es_alpha[j]*tx-(kx0+es_beta[j])*x-h0*(x*x-y*y)+2.0*g0*ty+dg0*y
            ay2=-es_alpha[j]*ty-(ky0+es_beta[j])*y+2.0*h0*x*y-2.0*g0*tx-dg0*x
            bx= x+0.5*h*ax1; btx=tx+0.5*h*ax2
            by= y+0.5*h*ay1; bty=ty+0.5*h*ay2
            bx1=btx;by1=bty
            bx2=-0.5*(es_alpha[j]+es_alpha[j+1])*btx-(kxm+0.5*(es_beta[j]+es_beta[j+1]))*bx-hm*(bx*bx-by*by)+2.0*gm*bty+dgm*by
            by2=-0.5*(es_alpha[j]+es_alpha[j+1])*bty-(kym+0.5*(es_beta[j]+es_beta[j+1]))*by+2.0*hm*bx*by-2.0*gm*btx-dgm*bx
            cx=x+0.5*h*bx1;ctx=tx+0.5*h*bx2
            cy=y+0.5*h*by1;cty=ty+0.5*h*by2
            cx1=ctx;cy1=cty
            cx2=-0.5*(es_alpha[j]+es_alpha[j+1])*ctx-(kxm+0.5*(es_beta[j]+es_beta[j+1]))*cx-hm*(cx*cx-cy*cy)+2.0*gm*cty+dgm*cy
            cy2=-0.5*(es_alpha[j]+es_alpha[j+1])*cty-(kym+0.5*(es_beta[j]+es_beta[j+1]))*cy+2.0*hm*cx*cy-2.0*gm*ctx-dgm*cx
            dx=x+h*cx1;dtx=tx+h*cx2
            dy=y+h*cy1;dty=ty+h*cy2
            dx1=dtx;dy1=dty
            dx2=-es_alpha[j+1]*dtx-(kx1+es_beta[j+1])*dx-h1*(dx*dx-dy*dy)+2.0*g1*dty+dg1*dy
            dy2=-es_alpha[j+1]*dty-(ky1+es_beta[j+1])*dy+2.0*h1*dx*dy-2.0*g1*dtx-dg1*dx
            x += h*(ax1+2*bx1+2*cx1+dx1)/6.0
            tx += h*(ax2+2*bx2+2*cx2+dx2)/6.0
            y += h*(ay1+2*by1+2*cy1+dy1)/6.0
            ty += h*(ay2+2*by2+2*cy2+dy2)/6.0
    return X,TX,Y,TY

def _vectorised_rk4(
    kx, ky, hex_strength, larmor_g, larmor_gradient, cs_kick, h,
    x, tx, y, ty,
    kickx, kicky, save_index, es_alpha, es_beta,
    gun_index, gun_focal_m,
):
    nr=x.size;ns=save_index.size
    X=np.empty((ns,nr),np.float32);TX=np.empty_like(X);Y=np.empty_like(X);TY=np.empty_like(X);s=0
    for j in range(kx.shape[0]):
        tx += kickx[j];ty += kicky[j]
        if cs_kick[j] != 0.0:
            radius_sq=x*x+y*y
            tx -= cs_kick[j]*radius_sq*x
            ty -= cs_kick[j]*radius_sq*y
        if s<ns and j==save_index[s]:
            X[s]=x;TX[s]=tx;Y[s]=y;TY[s]=ty;s+=1
        if j>=kx.shape[0]-1: continue
        kx0=kx[j];ky0=ky[j];kxm=.5*(kx[j]+kx[j+1]);kym=.5*(ky[j]+ky[j+1]);kx1=kx[j+1];ky1=ky[j+1]
        g0=larmor_g[j];gm=.5*(larmor_g[j]+larmor_g[j+1]);g1=larmor_g[j+1]
        dg0=larmor_gradient[j];dgm=.5*(larmor_gradient[j]+larmor_gradient[j+1]);dg1=larmor_gradient[j+1]
        h0=hex_strength[j];hm=.5*(hex_strength[j]+hex_strength[j+1]);h1=hex_strength[j+1]
        ax1=tx;ay1=ty
        ax2=-es_alpha[j]*tx-(kx0+es_beta[j])*x-h0*(x*x-y*y)+2.0*g0*ty+dg0*y
        ay2=-es_alpha[j]*ty-(ky0+es_beta[j])*y+2.0*h0*x*y-2.0*g0*tx-dg0*x
        bx=x+.5*h*ax1;btx=tx+.5*h*ax2
        by=y+.5*h*ay1;bty=ty+.5*h*ay2
        bx1=btx;by1=bty
        bx2=-.5*(es_alpha[j]+es_alpha[j+1])*btx-(kxm+.5*(es_beta[j]+es_beta[j+1]))*bx-hm*(bx*bx-by*by)+2.0*gm*bty+dgm*by
        by2=-.5*(es_alpha[j]+es_alpha[j+1])*bty-(kym+.5*(es_beta[j]+es_beta[j+1]))*by+2.0*hm*bx*by-2.0*gm*btx-dgm*bx
        cx=x+.5*h*bx1;ctx=tx+.5*h*bx2
        cy=y+.5*h*by1;cty=ty+.5*h*by2
        cx1=ctx;cy1=cty
        cx2=-.5*(es_alpha[j]+es_alpha[j+1])*ctx-(kxm+.5*(es_beta[j]+es_beta[j+1]))*cx-hm*(cx*cx-cy*cy)+2.0*gm*cty+dgm*cy
        cy2=-.5*(es_alpha[j]+es_alpha[j+1])*cty-(kym+.5*(es_beta[j]+es_beta[j+1]))*cy+2.0*hm*cx*cy-2.0*gm*ctx-dgm*cx
        dx=x+h*cx1;dtx=tx+h*cx2
        dy=y+h*cy1;dty=ty+h*cy2
        dx1=dtx;dy1=dty
        dx2=-es_alpha[j+1]*dtx-(kx1+es_beta[j+1])*dx-h1*(dx*dx-dy*dy)+2.0*g1*dty+dg1*dy
        dy2=-es_alpha[j+1]*dty-(ky1+es_beta[j+1])*dy+2.0*h1*dx*dy-2.0*g1*dtx-dg1*dx
        x=x+h*(ax1+2*bx1+2*cx1+dx1)/6;tx=tx+h*(ax2+2*bx2+2*cx2+dx2)/6;y=y+h*(ay1+2*by1+2*cy1+dy1)/6;ty=ty+h*(ay2+2*by2+2*cy2+dy2)/6
    return X,TX,Y,TY

def propagate(
    state,z0,z1,x,tx,y,ty,events=(),energy_offset_ev=None,
    *,include_spherical_aberration=True,
):
    step=float(state.step_mm)
    zfull=np.arange(z0,z1+step/2,step)
    magnetic,sx,sy=fields(zfull,state)
    hex_strength=np.ascontiguousarray(hexapole_field(zfull,state),np.float64)
    arrays=[np.ascontiguousarray(a,np.float64) for a in (x,tx,y,ty)]
    momentum=momentum_profile(state,zfull,energy_offset_ev)
    if momentum.ndim==1:
        momentum=np.broadcast_to(momentum[:,None],(len(zfull),arrays[0].size))
    larmor_g,larmor_gradient=larmor_coefficients_m1(magnetic,momentum,zfull)
    kx=np.ascontiguousarray(np.broadcast_to(sx[:,None],larmor_g.shape),np.float64)
    ky=np.ascontiguousarray(np.broadcast_to(sy[:,None],larmor_g.shape),np.float64)
    cs_kick=np.ascontiguousarray(
        spherical_aberration_kick_m3(zfull,state)
        if include_spherical_aberration else np.zeros(len(zfull)),
        np.float64,
    )
    kickx=np.zeros(len(zfull),np.float64);kicky=np.zeros(len(zfull),np.float64)
    for ze,dx,dy in sorted(events):
        idx=int(np.clip(round((ze-z0)/step),0,len(zfull)-1));kickx[idx]+=dx;kicky[idx]+=dy
    history_step=max(step,float(getattr(state,"history_step_mm",2.0)))
    stride=max(1,int(round(history_step/step)))
    save=np.arange(0,len(zfull),stride,dtype=np.int64)
    observation_z = getattr(state, "observation_plane_z_mm", None)
    if observation_z is not None and z0 <= float(observation_z) <= z1:
        observation_index = np.int64(np.clip(
            round((float(observation_z) - z0) / step),
            0,
            len(zfull) - 1,
        ))
        save = np.unique(np.r_[save, observation_index])
    if save[-1] != len(zfull)-1: save=np.r_[save,np.int64(len(zfull)-1)]
    gun_index=np.int64(-1);gun_focal_m=np.float64(-1.0)
    es_alpha=np.zeros(len(zfull),np.float64)
    es_beta=np.zeros(len(zfull),np.float64)
    requested=str(getattr(state,"acceleration_backend","Auto"));use_numba=bool(getattr(state,"acceleration_enabled",False)) and requested!="CPU" and NUMBA_AVAILABLE and arrays[0].size>=256
    state.active_backend="Numba CPU" if use_numba else "CPU"
    if use_numba:
        X,TX,Y,TY=_parallel_rk4(kx,ky,hex_strength,larmor_g,larmor_gradient,cs_kick,step*1e-3,*arrays,kickx,kicky,save,es_alpha,es_beta,gun_index,gun_focal_m)
    else:
        X,TX,Y,TY=_vectorised_rk4(kx,ky,hex_strength,larmor_g,larmor_gradient,cs_kick,step*1e-3,*arrays,kickx,kicky,save,es_alpha,es_beta,gun_index,gun_focal_m)
    return zfull[save],X,TX,Y,TY

def transfer(state,z0,z1):
    _,x,tx,_,_=propagate(
        state,z0,z1,
        np.array([1.,0.]),np.array([0.,1.]),np.zeros(2),np.zeros(2),
        include_spherical_aberration=False,
    )
    return np.array([[x[-1,0],x[-1,1]],[tx[-1,0],tx[-1,1]]],float)
