from __future__ import annotations
import numpy as np

def detect_beam_waist(branch,z_min_mm,z_max_mm,min_rays=5):
    z=np.asarray(branch.z,float);x=np.asarray(branch.x,float);y=np.asarray(branch.y,float);tx=np.asarray(branch.tx,float);ty=np.asarray(branch.ty,float);blocked=np.asarray(branch.blocked_z,float)
    candidate=np.flatnonzero((z>float(z_min_mm))&(z<float(z_max_mm)))
    rms=np.full(z.size,np.nan);corr=np.full(z.size,np.nan);count=np.zeros(z.size,int)
    for j in candidate:
        valid=np.isfinite(x[j])&np.isfinite(y[j])&np.isfinite(tx[j])&np.isfinite(ty[j])&(np.isnan(blocked)|(blocked>=z[j]))
        if valid.sum()<min_rays:continue
        dx=x[j,valid]-x[j,valid].mean();dy=y[j,valid]-y[j,valid].mean();dtx=tx[j,valid]-tx[j,valid].mean();dty=ty[j,valid]-ty[j,valid].mean()
        rms[j]=np.sqrt(np.mean(dx*dx+dy*dy));corr[j]=np.mean(dx*dtx+dy*dty);count[j]=valid.sum()
    valid=candidate[np.isfinite(rms[candidate])]
    minima=[j for j in valid[1:-1] if np.isfinite(rms[j-1]) and np.isfinite(rms[j+1]) and rms[j]<=rms[j-1] and rms[j]<=rms[j+1]]
    physical=[j for j in minima if corr[j-1]<=0 and corr[j+1]>=0]
    if not (physical or minima):return None
    j=min(physical or minima,key=lambda k:rms[k])
    return {'z_mm':float(z[j]),'rms_radius_mm':float(rms[j]*1e3),'correlation':float(corr[j]),'ray_count':int(count[j]),'kind':'ensemble RMS beam waist'}
