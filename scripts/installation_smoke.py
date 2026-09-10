"""Run with an isolated installed interpreter (-I), outside the source tree."""
import argparse
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    import temsim
    from temsim.optics.column import default_state
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.calculation_manifest import solver_source_identity
    from temsim.physics.stem_wave_imaging import AngularDetector,simulate_angle_resolved_stem
    from temsim.physics.compute_backend import cupy_capability
    from PySide6.QtWidgets import QApplication
    from temsim.gui.model_inspector import ModelInspectorPage
    state=default_state()
    catalog=AssemblyCatalog(); catalog.apply(state,catalog.default_selection())
    state.acceleration_enabled=False; state.acceleration_backend="CPU"
    state.sample.specimen_mode="reference"; state.sample.reference_sample_key="si_110"
    state.sample.thickness_nm=.4; state.sample.wave_grid_pixels=32
    state.sample.wave_field_of_view_angstrom=8.; state.sample.wave_atomistic_enabled=True
    state.sample.wave_multislice_enabled=True; state.sample.wave_frozen_phonon_enabled=False
    incident=SimpleNamespace(alive=np.ones(5,bool),ray_weight=np.array([.6,.1,.1,.1,.1]),
        x=np.zeros((1,5)),y=np.zeros((1,5)),tx=np.array([[0,.002,-.002,0,0]]),ty=np.array([[0,0,0,.002,-.002]]))
    xy=np.array([[-5e-5,5e-5],[-5e-5,5e-5]])
    result=simulate_angle_resolved_stem(state,SimpleNamespace(incident=incident),
        (AngularDetector("bf",0,10),AngularDetector("df",10,24)),xy,xy.T)
    assert result.metrics["wave_compute_backend"]=="NumPy CPU"
    assert all(np.all(np.isfinite(a)) for a in result.fractions.values())
    assert float(np.mean(result.fractions["bf"])) > 0
    app=QApplication.instance() or QApplication([])
    inspector=ModelInspectorPage(); inspector.set_state(state); inspector.resize(1000,650)
    inspector.show(); app.processEvents(); inspector.close()
    gpu=cupy_capability()
    receipt={"status":"PASS","utc":datetime.now(timezone.utc).isoformat(),"installed_module":str(Path(temsim.__file__).resolve()),
        "solver_source_sha256":solver_source_identity(),"cpu_case":"Si [110] atomistic multislice, 0.4 nm, 32 grid, 2x2 scan", 
        "fractions":{k:v.tolist() for k,v in result.fractions.items()},"gui":"offscreen model inspector opened",
        "gpu_validation":"NOT_RUN" if not gpu.available else "NOT_RUN: this script validates CPU installation",
        "gpu_capability":gpu.detail,"versions":{n:metadata.version(n) for n in ("numpy","scipy","abtem","PySide6","tem-simulator-v2")}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    print(json.dumps(receipt))


if __name__=="__main__": main()
