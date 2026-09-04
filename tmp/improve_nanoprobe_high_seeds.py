import json
import numpy as np
from scipy.optimize import brentq
from temsim.assembly_catalog import AssemblyCatalog
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.optics.direct_alignment import _CondenserMeasurementModel
state=default_state();catalog=AssemblyCatalog();catalog.apply(state,catalog.default_selection())
apply_operating_mode_pair(state,'nano_probe','diffraction');state.electron_gun.emitter.ray_count=1000
prior=json.load(open('tmp/nanoprobe_full_scan.json'))
output=[]
for diameter in (144,168,192,216,240):
 state.condenser_aperture_2.diameter_um=diameter
 model=_CondenserMeasurementModel(state,step_mm=.1)
 target=diameter*.25
 candidates=[]
 for r in prior:
  if r['radius_rms_m']>1e-9:continue
  c2=r['c2'];c3=brentq(lambda x:model.measure((c2,x)).waist_offset_m,r['c3']-.03,r['c3']+.03,xtol=1e-10)
  st=model.measure((c2,c3));candidates.append((c2,c3,st.convergence_95_mrad))
 best=min(candidates,key=lambda r:abs(r[2]-target))
 # refine local candidate C2 on the same narrow source-image branch.
 anchor=best
 for c2 in np.linspace(max(0,anchor[0]-1.99),min(100,anchor[0]+1.99),81):
  c3=brentq(lambda x:model.measure((c2,x)).waist_offset_m,anchor[1]-2,anchor[1]+2,xtol=1e-10)
  st=model.measure((c2,c3));candidates.append((c2,c3,st.convergence_95_mrad))
 best=min(candidates,key=lambda r:abs(r[2]-target))
 row={'diameter_um':diameter,'target':target,'best':best,'max':max(candidates,key=lambda r:r[2])}
 output.append(row);print(row,flush=True)
open('tmp/nanoprobe_high_seeds.json','w').write(json.dumps(output,indent=2))
