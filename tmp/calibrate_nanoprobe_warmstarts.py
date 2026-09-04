import json, math
from dataclasses import asdict
import numpy as np
from scipy.optimize import brentq
from temsim.assembly_catalog import AssemblyCatalog
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.optics.direct_alignment import _CondenserMeasurementModel
from temsim.optics.probe_calibration import IncidentProbeModel
from numba import set_num_threads
set_num_threads(4)
state=default_state();catalog=AssemblyCatalog();catalog.apply(state,catalog.default_selection())
apply_operating_mode_pair(state,'nano_probe','diffraction')
state.electron_gun.emitter.ray_count=1000
state.step_mm=.1;state.acceleration_enabled=True;state.acceleration_backend='Numba CPU'
state.hp2_hexapole.strength_m3=626910.5744
state.hp1_hexapole.strength_m3=374822.5253
state.hp1_hexapole.orientation_rad=-.04231957336
lenses={lens.key:lens for lens in state.lenses}
c2l=lenses['condenser_lens_2'];c3l=lenses['condenser_lens_3']
basec2=25.25772717253661
rows=[]
diameters=(12,20,24,48,60,72,96,120,140,144,168,192,216,240)
for diameter in diameters:
    state.condenser_aperture_2.diameter_um=float(diameter)
    model=_CondenserMeasurementModel(state,step_mm=.1)
    target=diameter*.25
    cache={}
    def focus(c2):
        if c2 not in cache:
            c3=brentq(lambda x:model.measure((c2,x)).waist_offset_m,21.0 if c2<29 else 19.0,30.0,xtol=1e-10)
            stat=model.measure((c2,c3))
            cache[c2]=(c3,stat)
        return cache[c2]
    initial=focus(basec2)
    candidates=[(basec2,*initial)]
    if abs(initial[1].convergence_95_mrad/target-1)>.015:
        grid=sorted(set([*np.arange(12.,51.,1.),basec2]))
        for c2 in grid:
            candidates.append((c2,*focus(c2)))
        for low,high in zip(grid[:-1],grid[1:]):
            if (focus(low)[1].convergence_95_mrad-target)*(focus(high)[1].convergence_95_mrad-target)<0:
                found=brentq(lambda v:focus(v)[1].convergence_95_mrad-target,low,high,xtol=1e-7)
                candidates.append((found,*focus(found)))
    best=min(candidates,key=lambda item:(max(0,abs(item[2].convergence_95_mrad/target-1)-.005),abs(item[0]-basec2)))
    c2,c3,stats=best
    row={'diameter_um':diameter,'target_mrad':target,'vector':[c2,c3],'linear':asdict(stats)}
    print('LINEAR',diameter,target,c2,c3,stats.convergence_95_mrad,flush=True)
    c2l.percent=c2;c3l.percent=c3
    if diameter in (12,20,60,120,140,240):
        production=IncidentProbeModel(state)
        before=production.trace().statistics
        centre=c3
        def waist(c3):
            c3l.percent=c3
            return production.trace().statistics.waist_offset_m
        for width in (.0005,.005,.05):
            if waist(centre-width)*waist(centre+width)<0:
                c3=brentq(waist,centre-width,centre+width,xtol=1e-11)
                break
        c3l.percent=c3
        after=production.trace().statistics
        refined=production.trace(step_mm=.05).statistics
        row['production_vector']=[c2,c3]
        row['production']=asdict(after);row['refined']=asdict(refined)
        row['before_focus']=asdict(before)
        print('PRODUCTION',diameter,target,c2,c3,after.convergence_95_mrad,'waist nm',after.waist_offset_m*1e9,
              'refined',refined.convergence_95_mrad,refined.waist_offset_m*1e9,flush=True)
    rows.append(row)
    open('tmp/nanoprobe_new_warmstarts.json','w').write(json.dumps(rows,indent=2))
