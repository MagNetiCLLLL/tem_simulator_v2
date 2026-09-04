import json
import numpy as np
from scipy.optimize import brentq
from temsim.assembly_catalog import AssemblyCatalog
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.optics.direct_alignment import _CondenserMeasurementModel
state=default_state(); catalog=AssemblyCatalog();catalog.apply(state,catalog.default_selection())
apply_operating_mode_pair(state,'nano_probe','diffraction')
state.electron_gun.emitter.ray_count=1000
model=_CondenserMeasurementModel(state,step_mm=.1)
rows=[]
for c2 in np.arange(12.,20.01,.1):
    c3=brentq(lambda x:model.measure((c2,x)).waist_offset_m,21.,30.,xtol=1e-11)
    s=model.measure((c2,c3));rows.append((c2,c3,s.convergence_95_mrad,s.convergence_edge_rad*1e3,s.radius_rms_m*1e9,s.surviving_fraction))
print('best max95',max(rows,key=lambda x:x[2]),flush=True)
print('best edge',max(rows,key=lambda x:x[3]),flush=True)
for diameter in (110.,115.,120.,125.):
    state.condenser_aperture_2.diameter_mm=diameter/1000.
    model=_CondenserMeasurementModel(state,step_mm=.1)
    for c2 in (15.,16.,20.,25.25772717253661):
        c3=brentq(lambda x:model.measure((c2,x)).waist_offset_m,21.,30.,xtol=1e-11)
        s=model.measure((c2,c3))
        print('aperture',diameter,'c2,c3',c2,c3,'alpha95',s.convergence_95_mrad,flush=True)
open('tmp/nanoprobe_fine_scan.json','w').write(json.dumps(rows,indent=2))
