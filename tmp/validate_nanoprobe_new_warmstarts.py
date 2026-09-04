import json
from dataclasses import asdict,replace
from temsim.assembly_catalog import AssemblyCatalog
from temsim.operating_modes import apply_operating_mode_pair,direct_alignment_by_key
from temsim.optics.column import default_state
from temsim.optics.direct_alignment import apply_direct_alignment
from numba import set_num_threads
set_num_threads(4)
rows=json.load(open('tmp/nanoprobe_new_warmstarts.json'))
warm=[r for r in rows if r['diameter_um'] not in (20,60,140)]
definition=direct_alignment_by_key('nanoprobe_convergence')
definition=replace(definition,targets={**definition.targets,
 'precalculation_aperture_diameters_um':[r['diameter_um'] for r in warm],
 'precalculation_convergence_mrad':[r['target_mrad'] for r in warm],
 'precalculation_vectors':[r['vector'] for r in warm],
 'aperture_scaling_reference_mrad_per_um':.25,
})
results=[]
for diameter in (240,12,120,20,60,140):
 state=default_state();catalog=AssemblyCatalog();catalog.apply(state,catalog.default_selection())
 apply_operating_mode_pair(state,'nano_probe','imaging')
 state.electron_gun.emitter.ray_count=1000
 state.hp2_hexapole.strength_m3=626910.5744;state.hp1_hexapole.strength_m3=374822.5253;state.hp1_hexapole.orientation_rad=-.04231957336
 next(l for l in state.lenses if l.key=='condenser_lens_3').percent=21.2736229083
 state.condenser_aperture_2.diameter_um=diameter
 print('BEGIN',diameter,flush=True)
 result=apply_direct_alignment(state,'nanoprobe_convergence',diameter*.25,definition=definition)
 results.append({'diameter_um':diameter,'result':asdict(result)})
 print('RESULT',diameter,asdict(result),flush=True)
 open('tmp/nanoprobe_new_warmstart_validation.json','w').write(json.dumps(results,indent=2))
