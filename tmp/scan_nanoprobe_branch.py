import json, time
from dataclasses import asdict
import numpy as np
from scipy.optimize import brentq
from temsim.assembly_catalog import AssemblyCatalog
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.optics.direct_alignment import _CondenserMeasurementModel

state=default_state()
catalog=AssemblyCatalog(); catalog.apply(state,catalog.default_selection())
apply_operating_mode_pair(state,'nano_probe','diffraction')
state.electron_gun.emitter.ray_count=1000
model=_CondenserMeasurementModel(state,step_mm=.1)
print('ready', flush=True)
print(asdict(model.measure((25.2577,21.3027))),flush=True)
rows=[]
for c2 in np.arange(0.,101.,2.):
    c3s=np.linspace(0.,100.,101)
    vals=[model.measure((c2,c3)).waist_offset_m for c3 in c3s]
    found=[]
    for j in range(len(c3s)-1):
        if vals[j]*vals[j+1] >= 0: continue
        c3=brentq(lambda x:model.measure((c2,x)).waist_offset_m,c3s[j],c3s[j+1],xtol=1e-11)
        stats=model.measure((c2,c3))
        if abs(stats.waist_offset_m)>1e-7: continue
        row={'c2':c2,'c3':c3,**asdict(stats)};rows.append(row)
        found.append((c3,stats.convergence_95_mrad,stats.radius_rms_m*1e9))
    print(c2,found,flush=True)
open('tmp/nanoprobe_full_scan.json','w').write(json.dumps(rows,indent=2))
