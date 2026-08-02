import numpy as np
from temsim.component_keys import (
    ADAPTER_LENS,
    CONDENSER_LENS_1,
    CONDENSER_LENS_2,
    CONDENSER_LENS_3,
    DIFFRACTION_LENS,
    INTERMEDIATE_LENS,
    PROJECTOR_LENS_1,
    PROJECTOR_LENS_2,
    MINI_CONDENSER,
    OBJECTIVE_LENS,
    PROBE_TL12_LENS,
    PROBE_TL21_LENS,
    PROBE_TL22_LENS,
)
from temsim.component_names import LENS_SHORT_NAMES


LENS_CROSSOVER_NAMES = {

    "gun": "Gun L crossover",

    CONDENSER_LENS_1: f"{LENS_SHORT_NAMES[CONDENSER_LENS_1]} crossover",

    CONDENSER_LENS_2: f"{LENS_SHORT_NAMES[CONDENSER_LENS_2]} crossover",

    CONDENSER_LENS_3: f"{LENS_SHORT_NAMES[CONDENSER_LENS_3]} crossover",

    ADAPTER_LENS: "ADL crossover",

    PROBE_TL22_LENS: "TL22 crossover",

    PROBE_TL21_LENS: "TL21 crossover",

    PROBE_TL12_LENS: "TL12 crossover",

    MINI_CONDENSER: "MC/TL11 crossover",

    OBJECTIVE_LENS: "Objective L crossover",

    DIFFRACTION_LENS: "Diff L crossover",

    INTERMEDIATE_LENS: "Int L crossover",

    PROJECTOR_LENS_1: "P1 crossover",

    PROJECTOR_LENS_2: "P2 crossover",

}



def _branch_waists(branch, minimum_rays=5):

    z=np.asarray(branch.z,float);x=np.asarray(branch.x,float);y=np.asarray(branch.y,float)

    tx=np.asarray(branch.tx,float);ty=np.asarray(branch.ty,float)

    blocked=np.asarray(branch.blocked_z,float)

    waists=[]

    for j in range(1,len(z)-1):

        valid=np.isfinite(x[j])&np.isfinite(y[j])&np.isfinite(tx[j])&np.isfinite(ty[j])&(np.isnan(blocked)|(blocked>=z[j]))

        if int(valid.sum())<minimum_rays:continue

        def rc(k):

            dx=x[k,valid]-x[k,valid].mean();dy=y[k,valid]-y[k,valid].mean()

            dtx=tx[k,valid]-tx[k,valid].mean();dty=ty[k,valid]-ty[k,valid].mean()

            return float(np.sqrt(np.mean(dx*dx+dy*dy))),float(np.mean(dx*dtx+dy*dty))

        rl,cl=rc(j-1);rm,cm=rc(j);rr,cr=rc(j+1)

        if rm<=rl and rm<=rr and cl<0.0<cr:

            waists.append((float(z[j]),rm*1000.0))

    return waists



def detect_all_lens_crossovers(branches, lenses, tolerance_mm=1.0):

    """Detect real waists and assign each to its nearest upstream lens.


    Each physical waist is labelled once. Assignment to the nearest upstream

    lens prevents the same waist being repeatedly claimed by every earlier lens.

    This also allows P1/P2 crossovers to be found on post-specimen branches.

    """

    lens_list=[x for x in lenses if bool(getattr(x,"enabled",True)) and str(getattr(x,"key",""))!="gun"]

    lens_list=sorted(lens_list,key=lambda x:float(x.z_mm))

    all_waists=[]

    for branch in branches:

        for z,rms in _branch_waists(branch):

            if not any(abs(z-old[0])<=tolerance_mm for old in all_waists):

                all_waists.append((z,rms))

    results=[]

    for z,rms in sorted(all_waists):

        upstream=[lens for lens in lens_list if float(lens.z_mm)<z]

        if not upstream:continue

        lens=max(upstream,key=lambda x:float(x.z_mm))

        key=str(lens.key)

        results.append({

            "name":LENS_CROSSOVER_NAMES.get(key,f"{lens.name} crossover"),

            "z_mm":z,

            "rms_radius_mm":rms,

            "source_lens_key":key,

            "source_lens_name":lens.name,

            "verified":True,

        })

    return results
