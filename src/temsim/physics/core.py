from temsim.physics.acceleration import momentum_profile
from temsim.component_keys import CONDENSER_LENS_KEYS
from temsim.optics.equivalent_image_lenses import (
    IMAGE_LENS_KEYS,
    equivalent_image_events,
    equivalent_image_lenses_enabled,
)
"""Parallel ray propagator with downsampled history.

Uses Numba across rays when available. The three-Gaussian fields and physical-axis
stigmators are unchanged. Full integration uses state.step_mm; only stored plotting
history is downsampled to reduce memory pressure.
"""
import math
from dataclasses import dataclass
import hashlib
import numpy as np

from temsim.optics.lens_focal_length import focal_length_mm
from temsim.optics.magnetic_lens_aberration import spherical_aberration_mm
from temsim.physics.compute_backend import (
    BACKEND_CPU,
    BACKEND_CUDA,
    BACKEND_NUMBA,
    choose_ray_backend,
)

E=1.602176634e-19
M=9.1093837015e-31
C=299792458.0
H=6.62607015e-34
# This is now a solver boundary, not only a drawing range.  At 7 sigma a
# Gaussian amplitude is 2.29e-11 of its peak and its omitted two-sided
# integral is about 2.56e-12.  The explicit boundary makes upstream cache
# independence testable while keeping the truncation below ray-solver error.
FIELD_SIGMA_CUTOFF = 7.0


@dataclass(frozen=True, slots=True)
class PropagationCheckpoints:
    """Full-precision ray phase space at selected axial planes.

    Arrays use metres for X/Y, radians for TX/TY and have shape
    ``(checkpoint, ray)``.  They are kept separate from the float32 plotting
    history so resuming a high-accuracy integration does not add an extra
    history-quantisation error.
    """

    z_mm: np.ndarray
    x_m: np.ndarray
    tx_rad: np.ndarray
    y_m: np.ndarray
    ty_rad: np.ndarray


@dataclass(frozen=True, slots=True)
class AxialPropagationPlan:
    """Immutable, full-axis optical coefficients for restartable propagation."""

    z_mm: np.ndarray
    step_m: np.ndarray
    magnetic_t: np.ndarray
    sx_m2: np.ndarray
    sy_m2: np.ndarray
    hex_normal_m3: np.ndarray
    hex_skew_m3: np.ndarray
    cs_kick_m3: np.ndarray
    thin_power_m1: np.ndarray
    thin_rotation_rad: np.ndarray
    kick_x_rad: np.ndarray
    kick_y_rad: np.ndarray
    save_index: np.ndarray
    checkpoint_index: np.ndarray
    solver_signature: str
    signature: str


def _frozen_array(values, dtype=np.float64):
    result = np.ascontiguousarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


def _support_mask(z_mm, provider):
    """Return the solver's explicit finite-support mask for one field."""

    if hasattr(provider, "field_support_mm"):
        try:
            lower, upper = provider.field_support_mm(FIELD_SIGMA_CUTOFF)
        except TypeError:
            lower, upper = provider.field_support_mm()
    else:
        if not (
            hasattr(provider, "effective_length_mm")
            or hasattr(provider, "length_mm")
        ):
            return np.ones_like(np.asarray(z_mm), dtype=bool)
        centre = float(getattr(provider, "z_mm"))
        length = float(
            getattr(
                provider,
                "effective_length_mm",
                getattr(provider, "length_mm", 0.0),
            )
        )
        sigma = max(abs(length) / 2.355, 1e-12)
        lower = centre - FIELD_SIGMA_CUTOFF * sigma
        upper = centre + FIELD_SIGMA_CUTOFF * sigma
    z = np.asarray(z_mm, dtype=float)
    return (z >= float(lower)) & (z <= float(upper))

try:
    from numba import cuda, njit, prange
    NUMBA_AVAILABLE=True
except Exception:
    NUMBA_AVAILABLE=False
    cuda=None
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
    equivalent_image = bool(
        getattr(state, "_using_equivalent_image_propagation", False)
    )
    post_sample_only = (
        equivalent_image
        and z.size > 0
        and float(z[0]) >= float(state.sample.z_mm)
    )
    for lens in state.lenses:
        if not getattr(lens,"enabled",True): continue
        polarity = int(getattr(lens, "polarity", 1))
        if polarity not in (-1, 1):
            raise ValueError(
                f"{getattr(lens, 'name', 'Round lens')} polarity must be +1 or -1."
            )
        if lens.key in CONDENSER_LENS_KEYS:
            provider = state.condenser_system[lens.key]
            contribution = provider.magnetic_field_t(z)
            magnetic += np.where(_support_mask(z, provider), contribution, 0.0)
            continue
        if hasattr(lens, "magnetic_field_t"):
            contribution = lens.magnetic_field_t(z)
            if equivalent_image and lens.key in IMAGE_LENS_KEYS:
                if post_sample_only:
                    continue
                contribution = np.where(
                    z < float(state.sample.z_mm), contribution, 0.0
                )
            magnetic += np.where(_support_mask(z, lens), contribution, 0.0)
            continue
        mask = _support_mask(z, lens)
        for g in lens.gaussian:
            contribution = lens.scale()*g.amplitude*np.exp(-0.5*((z-(lens.z_mm+g.offset*lens.a_mm))/(g.sigma*lens.a_mm))**2)
            magnetic += np.where(mask, contribution, 0.0)
    for stig in state.stigmators:
        if not stig.enabled: continue
        if hasattr(stig, "quadrupole_strengths_m2"):
            qx, qy = stig.quadrupole_strengths_m2(z)
            mask = _support_mask(z, stig)
            sx += np.where(mask, qx, 0.0)
            sy += np.where(mask, qy, 0.0)
            continue
        envelope=np.exp(-0.5*((z-stig.z_mm)/max(1e-12,stig.length_mm/2.355))**2)
        xset=stig.max_strength_m2*stig.strength_x_percent/100.0
        yset=stig.max_strength_m2*stig.strength_y_percent/100.0
        q=0.5*(xset-yset)*envelope
        mask = _support_mask(z, stig)
        sx += np.where(mask, q, 0.0)
        sy -= np.where(mask, q, 0.0)
    for component in getattr(state, "corrector_elements", []):
        if not getattr(component, "enabled", False):
            continue
        if not hasattr(component, "quadrupole_strength_m2"):
            continue
        q = component.quadrupole_strength_m2(z)
        mask = _support_mask(z, component)
        sx += np.where(mask, q, 0.0)
        sy -= np.where(mask, q, 0.0)
    return magnetic,sx,sy

def bz(z,state): return fields(z,state)[0]


def hexapole_field(z, state):
    """Return the summed normal hexapole coefficient for compatibility."""

    return hexapole_field_components(z, state)[0]


def hexapole_field_components(z, state):
    """Sum continuous normal and skew hexapole coefficients."""

    z = np.asarray(z, float)
    normal = np.zeros_like(z)
    skew = np.zeros_like(z)
    for component in getattr(state, "corrector_elements", []):
        if not getattr(component, "enabled", False):
            continue
        if hasattr(component, "hexapole_strength_components_m3"):
            component_normal, component_skew = (
                component.hexapole_strength_components_m3(z)
            )
            mask = _support_mask(z, component)
            normal += np.where(mask, component_normal, 0.0)
            skew += np.where(mask, component_skew, 0.0)
            continue
        if hasattr(component, "hexapole_strength_m3"):
            contribution = component.hexapole_strength_m3(z)
            normal += np.where(
                _support_mask(z, component), contribution, 0.0
            )
    return normal, skew


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
        cs_mm = spherical_aberration_mm(lens, state.beam_voltage_kv)
        if cs_mm is None or float(cs_mm) == 0.0:
            continue
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
    kx, ky, hex_normal, hex_skew, larmor_g, larmor_gradient, cs_kick,
    thin_power, thin_rotation, step_m,
    x0, tx0, y0, ty0,
    kickx, kicky, save_index, checkpoint_index, es_alpha, es_beta,
    gun_index, gun_focal_m,
):
    nr=x0.size
    ns=save_index.size
    X=np.empty((ns,nr),np.float32)
    TX=np.empty((ns,nr),np.float32)
    Y=np.empty((ns,nr),np.float32)
    TY=np.empty((ns,nr),np.float32)
    nc=checkpoint_index.size
    CX=np.empty((nc,nr),np.float64)
    CTX=np.empty((nc,nr),np.float64)
    CY=np.empty((nc,nr),np.float64)
    CTY=np.empty((nc,nr),np.float64)
    for ray in prange(nr):
        x=x0[ray];tx=tx0[ray];y=y0[ray];ty=ty0[ray];s=0;c=0
        for j in range(kx.shape[0]):
            if thin_power[j] != 0.0:
                tx -= thin_power[j]*x
                ty -= thin_power[j]*y
            if thin_rotation[j] != 0.0:
                cosine=math.cos(thin_rotation[j])
                sine=math.sin(thin_rotation[j])
                rotated_x=cosine*x-sine*y
                rotated_y=sine*x+cosine*y
                rotated_tx=cosine*tx-sine*ty
                rotated_ty=sine*tx+cosine*ty
                x=rotated_x;y=rotated_y;tx=rotated_tx;ty=rotated_ty
            tx += kickx[j];ty += kicky[j]
            if cs_kick[j] != 0.0:
                radius_sq=x*x+y*y
                tx -= cs_kick[j]*radius_sq*x
                ty -= cs_kick[j]*radius_sq*y
            if s<ns and j==save_index[s]:
                X[s,ray]=x;TX[s,ray]=tx;Y[s,ray]=y;TY[s,ray]=ty;s+=1
            if c<nc and j==checkpoint_index[c]:
                CX[c,ray]=x;CTX[c,ray]=tx;CY[c,ray]=y;CTY[c,ray]=ty;c+=1
            if j>=kx.shape[0]-1: continue
            h=step_m[j]
            kx0=kx[j,ray];ky0=ky[j,ray];kxm=0.5*(kx[j,ray]+kx[j+1,ray]);kym=0.5*(ky[j,ray]+ky[j+1,ray]);kx1=kx[j+1,ray];ky1=ky[j+1,ray]
            g0=larmor_g[j,ray];gm=0.5*(larmor_g[j,ray]+larmor_g[j+1,ray]);g1=larmor_g[j+1,ray]
            dg0=larmor_gradient[j,ray];dgm=0.5*(larmor_gradient[j,ray]+larmor_gradient[j+1,ray]);dg1=larmor_gradient[j+1,ray]
            hn0=hex_normal[j];hnm=0.5*(hex_normal[j]+hex_normal[j+1]);hn1=hex_normal[j+1]
            hs0=hex_skew[j];hsm=0.5*(hex_skew[j]+hex_skew[j+1]);hs1=hex_skew[j+1]
            ax1=tx;ay1=ty
            hu=x*x-y*y;hv=2.0*x*y
            ax2=-es_alpha[j]*tx-(kx0+es_beta[j])*x-hn0*hu-hs0*hv+2.0*g0*ty+dg0*y
            ay2=-es_alpha[j]*ty-(ky0+es_beta[j])*y+hn0*hv-hs0*hu-2.0*g0*tx-dg0*x
            bx= x+0.5*h*ax1; btx=tx+0.5*h*ax2
            by= y+0.5*h*ay1; bty=ty+0.5*h*ay2
            bx1=btx;by1=bty
            bhu=bx*bx-by*by;bhv=2.0*bx*by
            bx2=-0.5*(es_alpha[j]+es_alpha[j+1])*btx-(kxm+0.5*(es_beta[j]+es_beta[j+1]))*bx-hnm*bhu-hsm*bhv+2.0*gm*bty+dgm*by
            by2=-0.5*(es_alpha[j]+es_alpha[j+1])*bty-(kym+0.5*(es_beta[j]+es_beta[j+1]))*by+hnm*bhv-hsm*bhu-2.0*gm*btx-dgm*bx
            cx=x+0.5*h*bx1;ctx=tx+0.5*h*bx2
            cy=y+0.5*h*by1;cty=ty+0.5*h*by2
            cx1=ctx;cy1=cty
            chu=cx*cx-cy*cy;chv=2.0*cx*cy
            cx2=-0.5*(es_alpha[j]+es_alpha[j+1])*ctx-(kxm+0.5*(es_beta[j]+es_beta[j+1]))*cx-hnm*chu-hsm*chv+2.0*gm*cty+dgm*cy
            cy2=-0.5*(es_alpha[j]+es_alpha[j+1])*cty-(kym+0.5*(es_beta[j]+es_beta[j+1]))*cy+hnm*chv-hsm*chu-2.0*gm*ctx-dgm*cx
            dx=x+h*cx1;dtx=tx+h*cx2
            dy=y+h*cy1;dty=ty+h*cy2
            dx1=dtx;dy1=dty
            dhu=dx*dx-dy*dy;dhv=2.0*dx*dy
            dx2=-es_alpha[j+1]*dtx-(kx1+es_beta[j+1])*dx-hn1*dhu-hs1*dhv+2.0*g1*dty+dg1*dy
            dy2=-es_alpha[j+1]*dty-(ky1+es_beta[j+1])*dy+hn1*dhv-hs1*dhu-2.0*g1*dtx-dg1*dx
            x += h*(ax1+2*bx1+2*cx1+dx1)/6.0
            tx += h*(ax2+2*bx2+2*cx2+dx2)/6.0
            y += h*(ay1+2*by1+2*cy1+dy1)/6.0
            ty += h*(ay2+2*by2+2*cy2+dy2)/6.0
    return X,TX,Y,TY,CX,CTX,CY,CTY

def _vectorised_rk4(
    kx, ky, hex_normal, hex_skew, larmor_g, larmor_gradient, cs_kick,
    thin_power, thin_rotation, step_m,
    x, tx, y, ty,
    kickx, kicky, save_index, checkpoint_index, es_alpha, es_beta,
    gun_index, gun_focal_m,
):
    nr=x.size;ns=save_index.size
    X=np.empty((ns,nr),np.float32);TX=np.empty_like(X);Y=np.empty_like(X);TY=np.empty_like(X);s=0
    nc=checkpoint_index.size;c=0
    CX=np.empty((nc,nr),np.float64);CTX=np.empty_like(CX);CY=np.empty_like(CX);CTY=np.empty_like(CX)
    for j in range(kx.shape[0]):
        if thin_power[j] != 0.0:
            tx = tx-thin_power[j]*x
            ty = ty-thin_power[j]*y
        if thin_rotation[j] != 0.0:
            cosine=math.cos(thin_rotation[j]);sine=math.sin(thin_rotation[j])
            rotated_x=cosine*x-sine*y
            rotated_y=sine*x+cosine*y
            rotated_tx=cosine*tx-sine*ty
            rotated_ty=sine*tx+cosine*ty
            x=rotated_x;y=rotated_y;tx=rotated_tx;ty=rotated_ty
        tx += kickx[j];ty += kicky[j]
        if cs_kick[j] != 0.0:
            radius_sq=x*x+y*y
            tx -= cs_kick[j]*radius_sq*x
            ty -= cs_kick[j]*radius_sq*y
        if s<ns and j==save_index[s]:
            X[s]=x;TX[s]=tx;Y[s]=y;TY[s]=ty;s+=1
        if c<nc and j==checkpoint_index[c]:
            CX[c]=x;CTX[c]=tx;CY[c]=y;CTY[c]=ty;c+=1
        if j>=kx.shape[0]-1: continue
        h=step_m[j]
        kx0=kx[j];ky0=ky[j];kxm=.5*(kx[j]+kx[j+1]);kym=.5*(ky[j]+ky[j+1]);kx1=kx[j+1];ky1=ky[j+1]
        g0=larmor_g[j];gm=.5*(larmor_g[j]+larmor_g[j+1]);g1=larmor_g[j+1]
        dg0=larmor_gradient[j];dgm=.5*(larmor_gradient[j]+larmor_gradient[j+1]);dg1=larmor_gradient[j+1]
        hn0=hex_normal[j];hnm=.5*(hex_normal[j]+hex_normal[j+1]);hn1=hex_normal[j+1]
        hs0=hex_skew[j];hsm=.5*(hex_skew[j]+hex_skew[j+1]);hs1=hex_skew[j+1]
        ax1=tx;ay1=ty
        hu=x*x-y*y;hv=2.0*x*y
        ax2=-es_alpha[j]*tx-(kx0+es_beta[j])*x-hn0*hu-hs0*hv+2.0*g0*ty+dg0*y
        ay2=-es_alpha[j]*ty-(ky0+es_beta[j])*y+hn0*hv-hs0*hu-2.0*g0*tx-dg0*x
        bx=x+.5*h*ax1;btx=tx+.5*h*ax2
        by=y+.5*h*ay1;bty=ty+.5*h*ay2
        bx1=btx;by1=bty
        bhu=bx*bx-by*by;bhv=2.0*bx*by
        bx2=-.5*(es_alpha[j]+es_alpha[j+1])*btx-(kxm+.5*(es_beta[j]+es_beta[j+1]))*bx-hnm*bhu-hsm*bhv+2.0*gm*bty+dgm*by
        by2=-.5*(es_alpha[j]+es_alpha[j+1])*bty-(kym+.5*(es_beta[j]+es_beta[j+1]))*by+hnm*bhv-hsm*bhu-2.0*gm*btx-dgm*bx
        cx=x+.5*h*bx1;ctx=tx+.5*h*bx2
        cy=y+.5*h*by1;cty=ty+.5*h*by2
        cx1=ctx;cy1=cty
        chu=cx*cx-cy*cy;chv=2.0*cx*cy
        cx2=-.5*(es_alpha[j]+es_alpha[j+1])*ctx-(kxm+.5*(es_beta[j]+es_beta[j+1]))*cx-hnm*chu-hsm*chv+2.0*gm*cty+dgm*cy
        cy2=-.5*(es_alpha[j]+es_alpha[j+1])*cty-(kym+.5*(es_beta[j]+es_beta[j+1]))*cy+hnm*chv-hsm*chu-2.0*gm*ctx-dgm*cx
        dx=x+h*cx1;dtx=tx+h*cx2
        dy=y+h*cy1;dty=ty+h*cy2
        dx1=dtx;dy1=dty
        dhu=dx*dx-dy*dy;dhv=2.0*dx*dy
        dx2=-es_alpha[j+1]*dtx-(kx1+es_beta[j+1])*dx-hn1*dhu-hs1*dhv+2.0*g1*dty+dg1*dy
        dy2=-es_alpha[j+1]*dty-(ky1+es_beta[j+1])*dy+hn1*dhv-hs1*dhu-2.0*g1*dtx-dg1*dx
        x=x+h*(ax1+2*bx1+2*cx1+dx1)/6;tx=tx+h*(ax2+2*bx2+2*cx2+dx2)/6;y=y+h*(ay1+2*by1+2*cy1+dy1)/6;ty=ty+h*(ay2+2*by2+2*cy2+dy2)/6
    return X,TX,Y,TY,CX,CTX,CY,CTY


if NUMBA_AVAILABLE:
    @cuda.jit
    def _cuda_rk4_kernel(
        kx_axis, ky_axis, hex_normal, hex_skew, larmor_axis,
        larmor_gradient_axis, inverse_momentum, cs_kick,
        thin_power, thin_rotation, step_m,
        x0, tx0, y0, ty0, kickx, kicky, save_index, checkpoint_index,
        es_alpha, es_beta, X, TX, Y, TY, CX, CTX, CY, CTY,
    ):
        ray = cuda.grid(1)
        if ray >= x0.size:
            return
        x = x0[ray]
        tx = tx0[ray]
        y = y0[ray]
        ty = ty0[ray]
        saved = 0
        checkpointed = 0
        save_count = save_index.size
        checkpoint_count = checkpoint_index.size
        step_count = kx_axis.size
        for j in range(step_count):
            if thin_power[j] != 0.0:
                tx -= thin_power[j] * x
                ty -= thin_power[j] * y
            if thin_rotation[j] != 0.0:
                cosine = math.cos(thin_rotation[j])
                sine = math.sin(thin_rotation[j])
                rotated_x = cosine * x - sine * y
                rotated_y = sine * x + cosine * y
                rotated_tx = cosine * tx - sine * ty
                rotated_ty = sine * tx + cosine * ty
                x = rotated_x
                y = rotated_y
                tx = rotated_tx
                ty = rotated_ty
            tx += kickx[j]
            ty += kicky[j]
            if cs_kick[j] != 0.0:
                radius_sq = x * x + y * y
                tx -= cs_kick[j] * radius_sq * x
                ty -= cs_kick[j] * radius_sq * y
            if saved < save_count and j == save_index[saved]:
                X[saved, ray] = x
                TX[saved, ray] = tx
                Y[saved, ray] = y
                TY[saved, ray] = ty
                saved += 1
            if (
                checkpointed < checkpoint_count
                and j == checkpoint_index[checkpointed]
            ):
                CX[checkpointed, ray] = x
                CTX[checkpointed, ray] = tx
                CY[checkpointed, ray] = y
                CTY[checkpointed, ray] = ty
                checkpointed += 1
            if j >= step_count - 1:
                continue

            h = step_m[j]

            kx0 = kx_axis[j]
            ky0 = ky_axis[j]
            kxm = 0.5 * (kx_axis[j] + kx_axis[j + 1])
            kym = 0.5 * (ky_axis[j] + ky_axis[j + 1])
            kx1 = kx_axis[j + 1]
            ky1 = ky_axis[j + 1]
            inverse_p = inverse_momentum[ray]
            g0 = larmor_axis[j] * inverse_p
            gm = 0.5 * (
                larmor_axis[j] + larmor_axis[j + 1]
            ) * inverse_p
            g1 = larmor_axis[j + 1] * inverse_p
            dg0 = larmor_gradient_axis[j] * inverse_p
            dgm = 0.5 * (
                larmor_gradient_axis[j] + larmor_gradient_axis[j + 1]
            ) * inverse_p
            dg1 = larmor_gradient_axis[j + 1] * inverse_p
            hn0 = hex_normal[j]
            hnm = 0.5 * (hex_normal[j] + hex_normal[j + 1])
            hn1 = hex_normal[j + 1]
            hs0 = hex_skew[j]
            hsm = 0.5 * (hex_skew[j] + hex_skew[j + 1])
            hs1 = hex_skew[j + 1]

            ax1 = tx
            ay1 = ty
            hu = x * x - y * y
            hv = 2.0 * x * y
            ax2 = (
                -es_alpha[j] * tx - (kx0 + es_beta[j]) * x
                - hn0 * hu - hs0 * hv + 2.0 * g0 * ty + dg0 * y
            )
            ay2 = (
                -es_alpha[j] * ty - (ky0 + es_beta[j]) * y
                + hn0 * hv - hs0 * hu - 2.0 * g0 * tx - dg0 * x
            )
            bx = x + 0.5 * h * ax1
            btx = tx + 0.5 * h * ax2
            by = y + 0.5 * h * ay1
            bty = ty + 0.5 * h * ay2
            bx1 = btx
            by1 = bty
            bhu = bx * bx - by * by
            bhv = 2.0 * bx * by
            alpha_mid = 0.5 * (es_alpha[j] + es_alpha[j + 1])
            beta_mid = 0.5 * (es_beta[j] + es_beta[j + 1])
            bx2 = (
                -alpha_mid * btx - (kxm + beta_mid) * bx
                - hnm * bhu - hsm * bhv + 2.0 * gm * bty + dgm * by
            )
            by2 = (
                -alpha_mid * bty - (kym + beta_mid) * by
                + hnm * bhv - hsm * bhu - 2.0 * gm * btx - dgm * bx
            )
            cx = x + 0.5 * h * bx1
            ctx = tx + 0.5 * h * bx2
            cy = y + 0.5 * h * by1
            cty = ty + 0.5 * h * by2
            cx1 = ctx
            cy1 = cty
            chu = cx * cx - cy * cy
            chv = 2.0 * cx * cy
            cx2 = (
                -alpha_mid * ctx - (kxm + beta_mid) * cx
                - hnm * chu - hsm * chv + 2.0 * gm * cty + dgm * cy
            )
            cy2 = (
                -alpha_mid * cty - (kym + beta_mid) * cy
                + hnm * chv - hsm * chu - 2.0 * gm * ctx - dgm * cx
            )
            dx = x + h * cx1
            dtx = tx + h * cx2
            dy = y + h * cy1
            dty = ty + h * cy2
            dx1 = dtx
            dy1 = dty
            dhu = dx * dx - dy * dy
            dhv = 2.0 * dx * dy
            dx2 = (
                -es_alpha[j + 1] * dtx - (kx1 + es_beta[j + 1]) * dx
                - hn1 * dhu - hs1 * dhv + 2.0 * g1 * dty + dg1 * dy
            )
            dy2 = (
                -es_alpha[j + 1] * dty - (ky1 + es_beta[j + 1]) * dy
                + hn1 * dhv - hs1 * dhu - 2.0 * g1 * dtx - dg1 * dx
            )
            x += h * (ax1 + 2.0 * bx1 + 2.0 * cx1 + dx1) / 6.0
            tx += h * (ax2 + 2.0 * bx2 + 2.0 * cx2 + dx2) / 6.0
            y += h * (ay1 + 2.0 * by1 + 2.0 * cy1 + dy1) / 6.0
            ty += h * (ay2 + 2.0 * by2 + 2.0 * cy2 + dy2) / 6.0
else:
    _cuda_rk4_kernel = None


def _cuda_rk4(
    kx_axis, ky_axis, hex_normal, hex_skew, larmor_axis,
    larmor_gradient_axis, inverse_momentum, cs_kick,
    thin_power, thin_rotation, step_m,
    x0, tx0, y0, ty0, kickx, kicky, save_index, checkpoint_index,
    es_alpha, es_beta,
):
    """Run the independent-ray RK4 integration on a CUDA device."""
    if cuda is None or _cuda_rk4_kernel is None:
        raise RuntimeError("Numba CUDA support is not importable")
    device_inputs = [
        cuda.to_device(value) for value in (
            kx_axis, ky_axis, hex_normal, hex_skew, larmor_axis,
            larmor_gradient_axis, inverse_momentum, cs_kick,
            thin_power, thin_rotation, step_m,
            x0, tx0, y0, ty0, kickx, kicky, save_index, checkpoint_index,
            es_alpha, es_beta,
        )
    ]
    output_shape = (save_index.size, x0.size)
    device_outputs = [
        cuda.device_array(output_shape, dtype=np.float32) for _ in range(4)
    ]
    checkpoint_shape = (checkpoint_index.size, x0.size)
    device_outputs.extend(
        cuda.device_array(checkpoint_shape, dtype=np.float64) for _ in range(4)
    )
    threads = 128
    blocks = (x0.size + threads - 1) // threads
    _cuda_rk4_kernel[blocks, threads](*device_inputs, *device_outputs)
    cuda.synchronize()
    return tuple(output.copy_to_host() for output in device_outputs)


def _record_active_backend(state, backend, fallback_reason=None):
    """Accumulate backends without letting two-ray diagnostics hide CUDA."""
    used = getattr(state, "_active_backends_used", None)
    if used is None:
        used = set()
        state._active_backends_used = used
    used.add(str(backend))
    priority = (BACKEND_CUDA, BACKEND_NUMBA, BACKEND_CPU)
    primary = next(name for name in priority if name in used)
    auxiliaries = sorted(name for name in used if name != primary)
    label = primary
    if auxiliaries:
        label += f" (+ {', '.join(auxiliaries)} auxiliaries)"
    if fallback_reason:
        label += f" (fallback: {fallback_reason})"
    state.active_backend = label


def _endpoint_exact_axial_grid(z0, z1, maximum_step_mm):
    """Return a forward grid whose endpoints are physically exact.

    Full intervals retain ``maximum_step_mm`` so calibrated thin kicks and
    continuous corrector fields keep their established sampling phase.  Only
    the final interval is shortened when necessary.  RK4 therefore reaches
    interfaces such as the specimen plane without crossing or stopping short.
    """

    start = float(z0)
    stop = float(z1)
    maximum_step = float(maximum_step_mm)
    if not math.isfinite(start) or not math.isfinite(stop):
        raise ValueError("Propagation Z limits must be finite")
    if not math.isfinite(maximum_step) or maximum_step <= 0.0:
        raise ValueError("Propagation step must be finite and positive")
    span = stop - start
    if span < 0.0:
        raise ValueError("Propagation stop Z must not precede start Z")
    if span == 0.0:
        return (
            np.array([start], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    quotient = span / maximum_step
    nearest_integer = int(round(quotient))
    # Subtracting two large absolute Z coordinates can leave an exact
    # step-multiple a few ulps away from its integer quotient.  Scale the
    # tolerance with the absolute coordinate magnitudes as well as the span;
    # otherwise a spurious ~1e-13 mm terminal interval makes np.gradient
    # numerically singular at an exact optical plane.
    quotient_tolerance = 64.0 * np.finfo(np.float64).eps * max(
        1.0,
        abs(quotient),
        abs(start) / maximum_step,
        abs(stop) / maximum_step,
    )
    if (
        nearest_integer >= 1
        and abs(quotient - nearest_integer) <= quotient_tolerance
    ):
        full_interval_count = nearest_integer
        grid = start + maximum_step * np.arange(
            full_interval_count + 1, dtype=np.float64
        )
    else:
        full_interval_count = int(math.floor(quotient))
        grid = start + maximum_step * np.arange(
            full_interval_count + 1, dtype=np.float64
        )
        grid = np.append(grid, stop)

    grid[0] = start
    grid[-1] = stop
    intervals = np.diff(grid)
    step_tolerance = 32.0 * np.finfo(np.float64).eps * max(
        1.0, abs(start), abs(stop), abs(maximum_step)
    )
    if (
        np.any(intervals <= 0.0)
        or np.any(intervals > maximum_step + step_tolerance)
    ):
        raise RuntimeError("Failed to construct an endpoint-exact axial grid")
    return grid, intervals


def _nearest_axial_grid_index(value, grid):
    """Return the nearest valid index on a monotonic axial grid."""

    size = len(grid)
    if size <= 1:
        return np.int64(0)
    requested = float(value)
    upper = int(np.searchsorted(grid, requested, side="left"))
    if upper <= 0:
        return np.int64(0)
    if upper >= size:
        return np.int64(size - 1)
    lower = upper - 1
    if requested - float(grid[lower]) <= float(grid[upper]) - requested:
        return np.int64(lower)
    return np.int64(upper)


def _piecewise_endpoint_exact_axial_grid(
    z0, z1, maximum_step_mm, interior_z_mm=()
):
    """Build a step-bounded grid containing every optical event plane."""

    start = float(z0)
    stop = float(z1)
    boundaries = [
        start,
        *sorted({
            float(value)
            for value in interior_z_mm
            if start < float(value) < stop
        }),
        stop,
    ]
    segments = [
        _endpoint_exact_axial_grid(left, right, maximum_step_mm)[0]
        for left, right in zip(boundaries[:-1], boundaries[1:])
    ]
    grid = np.concatenate([
        segment if index == 0 else segment[1:]
        for index, segment in enumerate(segments)
    ])
    return grid, np.diff(grid)

def build_propagation_plan(
    state, z0, z1, events=(), *, include_spherical_aberration=True,
    include_hexapole=True, save_z_mm=(), checkpoint_z_mm=(),
    maximum_step_mm=None,
):
    """Build the single global axial plan used by full and resumed traces."""

    requested_step=float(state.step_mm)
    if maximum_step_mm is not None:
        requested_step=min(requested_step,float(maximum_step_mm))
    image_lens_events=equivalent_image_events(state,float(z0),float(z1))
    exact_z_mm = [event.z_mm for event in image_lens_events]
    exact_z_mm.extend(float(value) for value in save_z_mm)
    zfull,step_mm=_piecewise_endpoint_exact_axial_grid(
        z0,z1,requested_step,
        exact_z_mm,
    )
    step_m=np.ascontiguousarray(step_mm*1e-3,np.float64)
    grid_start=float(zfull[0])
    had_equivalent_propagation_flag = hasattr(
        state, "_using_equivalent_image_propagation"
    )
    previous_equivalent_propagation_flag = bool(
        getattr(state, "_using_equivalent_image_propagation", False)
    )
    state._using_equivalent_image_propagation = (
        equivalent_image_lenses_enabled(state)
        and float(zfull[0]) >= float(state.sample.z_mm)
    )
    try:
        magnetic,sx,sy=fields(zfull,state)
    finally:
        if had_equivalent_propagation_flag:
            state._using_equivalent_image_propagation = (
                previous_equivalent_propagation_flag
            )
        else:
            delattr(state, "_using_equivalent_image_propagation")
    if include_hexapole:
        hex_normal, hex_skew = hexapole_field_components(zfull, state)
    else:
        hex_normal = np.zeros(len(zfull), np.float64)
        hex_skew = np.zeros(len(zfull), np.float64)
    hex_normal=np.ascontiguousarray(hex_normal,np.float64)
    hex_skew=np.ascontiguousarray(hex_skew,np.float64)
    cs_kick=np.ascontiguousarray(
        spherical_aberration_kick_m3(zfull,state)
        if include_spherical_aberration else np.zeros(len(zfull)),
        np.float64,
    )
    thin_power=np.zeros(len(zfull),np.float64)
    thin_rotation=np.zeros(len(zfull),np.float64)
    for lens_event in image_lens_events:
        index=_nearest_axial_grid_index(lens_event.z_mm,zfull)
        thin_power[index]+=float(lens_event.power_m1)
        thin_rotation[index]+=float(lens_event.rotation_rad)
    thin_power=np.ascontiguousarray(thin_power,np.float64)
    thin_rotation=np.ascontiguousarray(thin_rotation,np.float64)
    kickx=np.zeros(len(zfull),np.float64);kicky=np.zeros(len(zfull),np.float64)
    for ze,dx,dy in sorted(events):
        idx=_nearest_axial_grid_index(ze,zfull);kickx[idx]+=dx;kicky[idx]+=dy
    history_step=max(requested_step,float(getattr(state,"history_step_mm",2.0)))
    stride=max(1,int(round(history_step/requested_step)))
    save=np.arange(0,len(zfull),stride,dtype=np.int64)
    observation_z = getattr(state, "observation_plane_z_mm", None)
    if observation_z is not None and grid_start <= float(observation_z) <= float(zfull[-1]):
        observation_index = _nearest_axial_grid_index(observation_z, zfull)
        save = np.unique(np.r_[save, observation_index])
    requested_save_indices = [
        _nearest_axial_grid_index(value, zfull)
        for value in save_z_mm
        if grid_start <= float(value) <= float(zfull[-1])
    ]
    if requested_save_indices:
        save = np.unique(np.r_[save, requested_save_indices])
    checkpoint_indices = np.unique(np.asarray([
        _nearest_axial_grid_index(value, zfull)
        for value in checkpoint_z_mm
        if grid_start <= float(value) <= float(zfull[-1])
    ], dtype=np.int64))
    if save[-1] != len(zfull)-1: save=np.r_[save,np.int64(len(zfull)-1)]
    digest = hashlib.sha256()
    digest.update(f"field-cutoff={FIELD_SIGMA_CUTOFF:.17g}".encode("ascii"))
    solver_signature = repr((
        float(getattr(state, "beam_voltage_kv")),
        float(requested_step),
        float(getattr(state, "history_step_mm", requested_step)),
        bool(getattr(state, "acceleration_enabled", False)),
        str(getattr(state, "acceleration_backend", "Auto")),
        FIELD_SIGMA_CUTOFF,
    ))
    digest.update(solver_signature.encode("utf-8"))
    for values in (
        zfull, step_m, magnetic, sx, sy, hex_normal, hex_skew,
        cs_kick, thin_power, thin_rotation, kickx, kicky, save,
        checkpoint_indices,
    ):
        contiguous = np.ascontiguousarray(values, dtype=np.float64)
        digest.update(contiguous.shape.__repr__().encode("ascii"))
        digest.update(contiguous.tobytes())
    return AxialPropagationPlan(
        z_mm=_frozen_array(zfull),
        step_m=_frozen_array(step_m),
        magnetic_t=_frozen_array(magnetic),
        sx_m2=_frozen_array(sx),
        sy_m2=_frozen_array(sy),
        hex_normal_m3=_frozen_array(hex_normal),
        hex_skew_m3=_frozen_array(hex_skew),
        cs_kick_m3=_frozen_array(cs_kick),
        thin_power_m1=_frozen_array(thin_power),
        thin_rotation_rad=_frozen_array(thin_rotation),
        kick_x_rad=_frozen_array(kickx),
        kick_y_rad=_frozen_array(kicky),
        save_index=_frozen_array(save, np.int64),
        checkpoint_index=_frozen_array(checkpoint_indices, np.int64),
        solver_signature=solver_signature,
        signature=digest.hexdigest(),
    )


def propagation_plan_common_prefix_nodes(previous, current):
    """Return count of leading nodes whose actual optical actions are equal."""

    if previous is None or current is None:
        return 0
    if previous.solver_signature != current.solver_signature:
        return 0
    old_z = np.asarray(previous.z_mm)
    new_z = np.asarray(current.z_mm)
    count = min(old_z.size, new_z.size)
    if count == 0:
        return 0
    arrays = (
        (old_z, new_z),
        (previous.magnetic_t, current.magnetic_t),
        (previous.sx_m2, current.sx_m2),
        (previous.sy_m2, current.sy_m2),
        (previous.hex_normal_m3, current.hex_normal_m3),
        (previous.hex_skew_m3, current.hex_skew_m3),
        (previous.cs_kick_m3, current.cs_kick_m3),
        (previous.thin_power_m1, current.thin_power_m1),
        (previous.thin_rotation_rad, current.thin_rotation_rad),
        (previous.kick_x_rad, current.kick_x_rad),
        (previous.kick_y_rad, current.kick_y_rad),
    )
    equal = np.ones(count, dtype=bool)
    for old, new in arrays:
        equal &= np.asarray(old[:count]) == np.asarray(new[:count])
    changed = np.flatnonzero(~equal)
    if changed.size:
        # Bz at node i affects the numerical derivative and the RK4 interval
        # ending at i.  Keep a two-node safety halo before the first change.
        return max(0, int(changed[0]) - 2)
    if old_z.size != new_z.size:
        return max(0, count - 2)
    return count


def execute_propagation_plan(
    state, plan, x, tx, y, ty, energy_offset_ev=None, *, start_index=0,
    include_initial_plane_kicks=True,
):
    """Execute a complete plan or resume it from an after-action checkpoint."""

    start_index = int(start_index)
    if not 0 <= start_index < len(plan.z_mm):
        raise ValueError("Propagation-plan start index is out of range")
    zfull = np.asarray(plan.z_mm[start_index:], dtype=np.float64)
    step_m = np.ascontiguousarray(plan.step_m[start_index:], np.float64)
    magnetic = np.asarray(plan.magnetic_t[start_index:], dtype=np.float64)
    sx = np.asarray(plan.sx_m2[start_index:], dtype=np.float64)
    sy = np.asarray(plan.sy_m2[start_index:], dtype=np.float64)
    hex_normal = np.ascontiguousarray(
        plan.hex_normal_m3[start_index:], np.float64
    )
    hex_skew = np.ascontiguousarray(
        plan.hex_skew_m3[start_index:], np.float64
    )
    cs_kick = np.array(plan.cs_kick_m3[start_index:], dtype=np.float64)
    thin_power = np.array(plan.thin_power_m1[start_index:], dtype=np.float64)
    thin_rotation = np.array(
        plan.thin_rotation_rad[start_index:], dtype=np.float64
    )
    kickx = np.array(plan.kick_x_rad[start_index:], dtype=np.float64)
    kicky = np.array(plan.kick_y_rad[start_index:], dtype=np.float64)
    if not bool(include_initial_plane_kicks):
        cs_kick[0]=0.0
        thin_power[0]=0.0
        thin_rotation[0]=0.0
        kickx[0]=0.0
        kicky[0]=0.0
    arrays=[np.array(a,dtype=np.float64,order="C",copy=True) for a in (x,tx,y,ty)]
    global_save = np.asarray(plan.save_index, dtype=np.int64)
    save = np.ascontiguousarray(
        global_save[global_save >= start_index] - start_index, np.int64
    )
    global_checkpoints = np.asarray(plan.checkpoint_index, dtype=np.int64)
    checkpoint_index = np.ascontiguousarray(
        global_checkpoints[global_checkpoints >= start_index] - start_index,
        np.int64,
    )
    gun_index=np.int64(-1);gun_focal_m=np.float64(-1.0)
    es_alpha=np.zeros(len(zfull),np.float64)
    es_beta=np.zeros(len(zfull),np.float64)
    backend, fallback_reason = choose_ray_backend(
        getattr(state, "acceleration_backend", "Auto"),
        acceleration_enabled=bool(
            getattr(state, "acceleration_enabled", False)
        ),
        ray_count=arrays[0].size,
    )

    def cpu_coefficients():
        halo_start = max(0, start_index - 1)
        coefficient_z = np.asarray(plan.z_mm[halo_start:], dtype=np.float64)
        coefficient_magnetic = np.asarray(
            plan.magnetic_t[halo_start:], dtype=np.float64
        )
        momentum = momentum_profile(state, coefficient_z, energy_offset_ev)
        if momentum.ndim == 1:
            momentum = np.broadcast_to(
                momentum[:, None], (len(coefficient_z), arrays[0].size)
            )
        larmor_g, larmor_gradient = larmor_coefficients_m1(
            coefficient_magnetic, momentum, coefficient_z
        )
        offset = start_index - halo_start
        larmor_g = np.ascontiguousarray(larmor_g[offset:], np.float64)
        larmor_gradient = np.ascontiguousarray(
            larmor_gradient[offset:], np.float64
        )
        coefficient_shape = larmor_g.shape
        kx = np.ascontiguousarray(
            np.broadcast_to(sx[:, None], coefficient_shape), np.float64
        )
        ky = np.ascontiguousarray(
            np.broadcast_to(sy[:, None], coefficient_shape), np.float64
        )
        return kx, ky, larmor_g, larmor_gradient

    if backend == BACKEND_CUDA:
        try:
            # Beyond the gun exit the accelerating potential is constant, so
            # g(z, ray) = -e Bz(z) / (2 p(ray)) is exactly separable.  Keeping
            # the axial factor and per-ray inverse momentum separate avoids
            # allocating and transferring several enormous (Z, ray) arrays.
            momentum_at_start = momentum_profile(
                state, zfull[:1], energy_offset_ev
            )
            if momentum_at_start.ndim == 1:
                inverse_momentum = np.full(
                    arrays[0].size,
                    1.0 / float(momentum_at_start[0]),
                    dtype=np.float64,
                )
            else:
                inverse_momentum = np.ascontiguousarray(
                    1.0 / momentum_at_start[0], dtype=np.float64
                )
            halo_start = max(0, start_index - 1)
            coefficient_z = np.asarray(plan.z_mm[halo_start:], dtype=float)
            coefficient_magnetic = np.asarray(
                plan.magnetic_t[halo_start:], dtype=float
            )
            full_larmor_axis = np.ascontiguousarray(
                (-E) * coefficient_magnetic / 2.0, dtype=np.float64
            )
            z_m = coefficient_z * 1.0e-3
            offset = start_index - halo_start
            larmor_axis = np.ascontiguousarray(
                full_larmor_axis[offset:], dtype=np.float64
            )
            larmor_gradient_axis = np.ascontiguousarray(
                (
                    np.gradient(full_larmor_axis, z_m, edge_order=1)[offset:]
                    if len(z_m) >= 2 else np.zeros_like(larmor_axis)
                ),
                dtype=np.float64,
            )
            outputs=_cuda_rk4(
                np.ascontiguousarray(sx, np.float64),
                np.ascontiguousarray(sy, np.float64),
                hex_normal,hex_skew,larmor_axis,larmor_gradient_axis,
                inverse_momentum,cs_kick,thin_power,thin_rotation,
                step_m,*arrays,kickx,kicky,
                save,checkpoint_index,es_alpha,es_beta,
            )
            _record_active_backend(state, backend, fallback_reason)
        except Exception as exc:
            # CUDA is optional.  A driver/JIT/memory failure must not discard
            # the user's calculation; continue with the best CPU backend.
            backend = BACKEND_NUMBA if NUMBA_AVAILABLE else BACKEND_CPU
            _record_active_backend(state, backend, f"CUDA error: {exc}")
            kx, ky, larmor_g, larmor_gradient = cpu_coefficients()
            if backend == BACKEND_NUMBA:
                outputs=_parallel_rk4(kx,ky,hex_normal,hex_skew,larmor_g,larmor_gradient,cs_kick,thin_power,thin_rotation,step_m,*arrays,kickx,kicky,save,checkpoint_index,es_alpha,es_beta,gun_index,gun_focal_m)
            else:
                outputs=_vectorised_rk4(kx,ky,hex_normal,hex_skew,larmor_g,larmor_gradient,cs_kick,thin_power,thin_rotation,step_m,*arrays,kickx,kicky,save,checkpoint_index,es_alpha,es_beta,gun_index,gun_focal_m)
    elif backend == BACKEND_NUMBA:
        kx, ky, larmor_g, larmor_gradient = cpu_coefficients()
        outputs=_parallel_rk4(kx,ky,hex_normal,hex_skew,larmor_g,larmor_gradient,cs_kick,thin_power,thin_rotation,step_m,*arrays,kickx,kicky,save,checkpoint_index,es_alpha,es_beta,gun_index,gun_focal_m)
        _record_active_backend(state, backend, fallback_reason)
    else:
        kx, ky, larmor_g, larmor_gradient = cpu_coefficients()
        outputs=_vectorised_rk4(kx,ky,hex_normal,hex_skew,larmor_g,larmor_gradient,cs_kick,thin_power,thin_rotation,step_m,*arrays,kickx,kicky,save,checkpoint_index,es_alpha,es_beta,gun_index,gun_focal_m)
        _record_active_backend(state, backend, fallback_reason)
    X,TX,Y,TY,CX,CTX,CY,CTY=outputs
    checkpoints = PropagationCheckpoints(
        z_mm=_frozen_array(zfull[checkpoint_index]),
        x_m=_frozen_array(CX),
        tx_rad=_frozen_array(CTX),
        y_m=_frozen_array(CY),
        ty_rad=_frozen_array(CTY),
    )
    return zfull[save],X,TX,Y,TY,checkpoints


def propagate(
    state,z0,z1,x,tx,y,ty,events=(),energy_offset_ev=None,
    *,include_spherical_aberration=True,include_hexapole=True,
    save_z_mm=(),include_initial_plane_kicks=True,
    checkpoint_z_mm=(),return_checkpoints=False,maximum_step_mm=None,
):
    plan = build_propagation_plan(
        state,z0,z1,events,
        include_spherical_aberration=include_spherical_aberration,
        include_hexapole=include_hexapole,
        save_z_mm=save_z_mm,
        checkpoint_z_mm=checkpoint_z_mm,
        maximum_step_mm=maximum_step_mm,
    )
    result = execute_propagation_plan(
        state,plan,x,tx,y,ty,energy_offset_ev,
        include_initial_plane_kicks=include_initial_plane_kicks,
    )
    return result if return_checkpoints else result[:5]

def transfer(state,z0,z1):
    _,x,tx,_,_=propagate(
        state,z0,z1,
        np.array([1.,0.]),np.array([0.,1.]),np.zeros(2),np.zeros(2),
        include_spherical_aberration=False,include_hexapole=False,
    )
    return np.array([[x[-1,0],x[-1,1]],[tx[-1,0],tx[-1,1]]],float)


def complex_transfer(state, z0, z1):
    """Return the first-order, Larmor-coupled transfer in complex form."""

    _, x, tx, y, ty = propagate(
        state, z0, z1,
        np.array([1.0, 0.0]), np.array([0.0, 1.0]),
        np.zeros(2), np.zeros(2),
        include_spherical_aberration=False,
        include_hexapole=False,
    )
    return np.array(
        [
            [x[-1, 0] + 1j * y[-1, 0], x[-1, 1] + 1j * y[-1, 1]],
            [
                tx[-1, 0] + 1j * ty[-1, 0],
                tx[-1, 1] + 1j * ty[-1, 1],
            ],
        ],
        dtype=np.complex128,
    )
