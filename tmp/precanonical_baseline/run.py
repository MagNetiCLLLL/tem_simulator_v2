import importlib.abc
import importlib.util
import sys
import json
import time
from pathlib import Path
base=Path(__file__).resolve().parent
class BaselineModules(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        relative={"temsim.physics.core":"core.py","temsim.optics.direct_alignment":"direct_alignment.py"}.get(fullname)
        if relative:
            return importlib.util.spec_from_file_location(fullname,base/relative)
        return None
sys.meta_path.insert(0,BaselineModules())
from temsim.optics.column import default_state
from temsim.assembly_catalog import AssemblyCatalog
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.direct_alignment import apply_direct_alignment
import temsim.physics.core as core
import temsim.optics.direct_alignment as da
print("old_modules",core.__file__,da.__file__,flush=True)
s=default_state()
c=AssemblyCatalog()
c.apply(s,c.default_selection())
apply_operating_mode_pair(s,"nano_probe","imaging")
s.electron_gun.emitter.ray_count=1000
print("starting",s.sample.z_mm,s.condenser_aperture_2.diameter_um,flush=True)
t=time.perf_counter()
r=apply_direct_alignment(s,"nanoprobe_convergence",30.0)
print("elapsed",time.perf_counter()-t,flush=True)
print(r,flush=True)
(base/"result.txt").write_text(repr(r),encoding="utf-8")
