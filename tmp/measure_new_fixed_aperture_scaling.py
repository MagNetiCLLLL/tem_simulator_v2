import json
import numpy as np
from temsim.assembly_catalog import AssemblyCatalog
from temsim.operating_modes import apply_operating_mode_pair,direct_alignment_by_key
from temsim.optics.column import default_state
from temsim.optics.direct_alignment import _validate_condenser_production
state=default_state();catalog=AssemblyCatalog();catalog.apply(state,catalog.default_selection())
apply_operating_mode_pair(state,'nano_probe','imaging')
state.electron_gun.emitter.ray_count=1000
definition=direct_alignment_by_key('nanoprobe_convergence')
lenses={lens.key:lens for lens in state.lenses}
vector=np.asarray([lenses[key].percent for key in ('condenser_lens_2','condenser_lens_3')])
warm=json.load(open('tmp/nanoprobe_warmstart_table.json'))
diameters=warm['precalculation_aperture_diameters_um']
angles=[]
print('vector',vector,flush=True)
for diameter in diameters:
 state.condenser_aperture_2.diameter_um=diameter
 m=_validate_condenser_production(state,definition,vector,step_mm=.1)
 angles.append(m.value);print(diameter,m.value,m.constraint_value,flush=True)
errors=np.asarray(angles)-.25*np.asarray(diameters)
result={'lens_vector':vector.tolist(),'diameters_um':diameters,'angles_mrad':angles,'slope_mrad_per_um':.25,'rms_error_mrad':float(np.sqrt(np.mean(errors**2))),'maximum_absolute_error_mrad':float(np.max(np.abs(errors))),'monotonic':bool(np.all(np.diff(angles)>0))}
open('tmp/nanoprobe_fixed_aperture_scaling.json','w').write(json.dumps(result,indent=2))
print(result,flush=True)
