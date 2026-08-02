import numpy as np

def _hit_mask(plane,x_mm,y_mm):
    if hasattr(plane, "hit_mask"):
        return np.asarray(plane.hit_mask(x_mm, y_mm), dtype=bool)
    outer=max(float(plane.outer_width_mm),0.)/2.;inner=max(float(plane.inner_diameter_mm),0.)/2.
    r=np.hypot(x_mm,y_mm);g=str(plane.geometry).lower()
    if g in {'annulus','annular','ring'}:return (r>=inner)&(r<=outer)
    if g in {'square','rectangle','camera'}:return (np.abs(x_mm)<=outer)&(np.abs(y_mm)<=outer)
    return r<=outer

def clip_recording_planes(state,z,x,y,alive,blocked_z,blocked_key):
    """Clip post-sample apertures and recording planes in physical order."""

    z=np.asarray(z,float);alive=np.asarray(alive,bool).copy();blocked_z=np.asarray(blocked_z,float).copy();blocked_key=list(blocked_key)
    x=np.asarray(x,float);y=np.asarray(y,float)
    candidates = [
        (float(aperture.z_mm), 0, aperture)
        for aperture in getattr(state, "apertures", [])
        if (
            bool(getattr(aperture, "enabled", False))
            and bool(getattr(aperture, "installed", True))
            and float(aperture.z_mm) >= float(state.sample.z_mm)
        )
    ]
    candidates.extend(
        (
            float(plane.z_mm),
            1,
            plane,
        )
        for plane in getattr(state, "recording_planes", [])
        if bool(getattr(plane, "inserted", False))
    )
    for plane_z_mm, kind_order, plane in sorted(candidates):
        if plane_z_mm<z[0]-1e-9 or plane_z_mm>z[-1]+1e-9:continue
        hi=int(np.searchsorted(z,plane_z_mm,'left'));hi=min(max(hi,1),len(z)-1);lo=hi-1
        f=(plane_z_mm-z[lo])/max(z[hi]-z[lo],1e-12)
        xx=(x[lo]+f*(x[hi]-x[lo]))*1e3;yy=(y[lo]+f*(y[hi]-y[lo]))*1e3
        reaches = alive | (
            np.isfinite(blocked_z)
            & (blocked_z > plane_z_mm + 1.0e-9)
        )
        if kind_order == 0:
            if hasattr(plane, "transmission_mask"):
                passes = np.asarray(
                    plane.transmission_mask(xx, yy),
                    dtype=bool,
                )
            else:
                radius = max(float(plane.radius_mm), 0.0)
                passes = (
                    np.hypot(
                        xx - float(plane.offset_x_mm),
                        yy - float(plane.offset_y_mm),
                    )
                    <= radius
                )
            hit = reaches & ~passes
        else:
            hit=reaches&_hit_mask(plane,xx,yy)
        alive[hit]=False;blocked_z[hit]=plane_z_mm
        for i in np.flatnonzero(hit):blocked_key[i]=str(plane.key)
    alive = np.isnan(blocked_z)
    return alive,blocked_z,blocked_key
