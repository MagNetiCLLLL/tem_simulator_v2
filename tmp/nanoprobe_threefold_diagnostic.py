import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from temsim.optics.column import default_state
from temsim.assembly_catalog import AssemblyCatalog
from temsim.operating_modes import apply_operating_mode_pair, direct_alignment_by_key
import temsim.optics.direct_alignment as da

out=Path("tmp/nanoprobe_threefold_diagnostic")
out.mkdir(exist_ok=True)
s=default_state()
c=AssemblyCatalog()
c.apply(s,c.default_selection())
apply_operating_mode_pair(s,"nano_probe","diffraction")
s.electron_gun.emitter.ray_count=2000
v=np.array([next(x for x in s.lenses if x.key==k).percent for k in ("condenser_lens_2","condenser_lens_3")])
original_propagate=da.propagate
original_statistics=da.transverse_beam_statistics
rows=[]
fig,axes=plt.subplots(1,4,figsize=(16,4),constrained_layout=True)
for ax,(label,cs,hp) in zip(axes,[("full",True,True),("no_hexapoles",True,False),("no_spherical",False,True),("linear",False,False)]):
    def propagate(*args,**kwargs):
        kwargs.update(include_spherical_aberration=cs,include_hexapole=hp)
        return original_propagate(*args,**kwargs)
    def statistics(x,y,tx,ty,**kwargs):
        result=original_statistics(x,y,tx,ty,**kwargs)
        mask=np.asarray(kwargs["alive"],dtype=bool)
        weights=np.asarray(kwargs["weights"])[mask]
        weights=weights/weights.sum()
        z=(np.asarray(x)[mask]-result.mean_x_m)+1j*(np.asarray(y)[mask]-result.mean_y_m)
        radius=np.abs(z)
        theta=np.angle(z)
        row=dict(case=label,sample_z_mm=s.sample.z_mm,n=result.surviving_rays,rms_radius_nm=result.radius_rms_m*1e9,D95_nm=2*result.radius_95_m*1e9,D99_nm=2*result.radius_99_m*1e9,waist_offset_nm=result.waist_offset_m*1e9,angular_harmonic3=float(abs(np.sum(weights*np.exp(3j*theta)))),normalized_cubic3=float(abs(np.sum(weights*z**3))/np.sum(weights*radius**3)),alpha99_mrad=result.convergence_99_mrad)
        rows.append(row)
        np.savez(out/(label+".npz"),x_nm=z.real*1e9,y_nm=z.imag*1e9,weights=weights)
        ax.scatter(z.real*1e9,z.imag*1e9,s=2)
        ax.set_aspect("equal")
        ax.set_title(label+"\nRMS %.3f nm; m3 %.3f"%(row["rms_radius_nm"],row["normalized_cubic3"]))
        ax.set_xlabel("X (nm)")
        ax.set_ylabel("Y (nm)")
        print(json.dumps(row),flush=True)
        return result
    da.propagate=propagate
    da.transverse_beam_statistics=statistics
    da._validate_condenser_production(s,direct_alignment_by_key("nanoprobe_convergence"),v,step_mm=0.1)
(out/"results.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
fig.savefig(out/"comparison.png",dpi=160)
print("saved",out.resolve(),flush=True)
