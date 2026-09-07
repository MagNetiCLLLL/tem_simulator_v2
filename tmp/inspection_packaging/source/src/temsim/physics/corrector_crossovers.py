import numpy as np

def detect_corrector_crossovers(branch, targets, half_window_mm=14.0):
    z=np.asarray(branch.z,float);x=np.asarray(branch.x,float);y=np.asarray(branch.y,float)
    tx=np.asarray(branch.tx,float);ty=np.asarray(branch.ty,float);blocked=np.asarray(branch.blocked_z,float)
    out=[]
    labels=("Before corrector","Inside corrector","Post-scan crossover")
    for label,target in zip(labels,targets):
        idx=np.flatnonzero((z>=target-half_window_mm)&(z<=target+half_window_mm))
        candidates=[]
        for j in idx:
            if j<=0 or j>=z.size-1: continue
            valid=np.isfinite(x[j])&np.isfinite(y[j])&np.isfinite(tx[j])&np.isfinite(ty[j])&(np.isnan(blocked)|(blocked>=z[j]))
            if valid.sum()<5: continue
            def values(k):
                dx=x[k,valid]-x[k,valid].mean();dy=y[k,valid]-y[k,valid].mean()
                dtx=tx[k,valid]-tx[k,valid].mean();dty=ty[k,valid]-ty[k,valid].mean()
                return np.sqrt(np.mean(dx*dx+dy*dy)),np.mean(dx*dtx+dy*dty)
            rm,cm=values(j);rl,cl=values(j-1);rr,cr=values(j+1)
            if rm<=rl and rm<=rr and cl<0.0 and cr>0.0:
                candidates.append((rm,j,cm))
        if candidates:
            rm,j,cm=min(candidates,key=lambda q:q[0])
            out.append({"name":label,"z_mm":float(z[j]),"target_z_mm":float(target),"rms_radius_mm":float(rm*1e3),"verified":True})
    return out
