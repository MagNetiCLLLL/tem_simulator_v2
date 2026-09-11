"""Check an installed package with -I outside the source tree.

Imports, particle transport, an independent Si multislice kernel and the GUI
are installation checks. They do not qualify production TEM/STEM images.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def installation_checks():
    import numpy as np
    import temsim
    from temsim.optics.column import default_state
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.calculation_manifest import solver_source_identity
    from temsim.physics.compute_backend import cupy_capability
    from temsim.physics.core import electron
    from temsim.physics.multislice import propagate_multislice
    from temsim.physics.simulation import run
    from temsim.physics.source_admission import UnsupportedWaveSource, require_gun_wave_source
    from temsim.physics.wave_imaging import prepare_specimen_potentials, interaction_constant_rad_per_v_angstrom
    from temsim.specimen.presets import load_specimen_preset
    from PySide6.QtWidgets import QApplication
    from temsim.gui.model_inspector import ModelInspectorPage
    state=default_state()
    catalog=AssemblyCatalog(); catalog.apply(state,catalog.default_selection())
    state.acceleration_enabled=False; state.acceleration_backend="CPU"
    state.electron_gun.emitter.ray_count=9
    state.step_mm=5.; state.history_step_mm=5.
    simulation=run(state,optical_only=True)
    assert simulation.gun_trace is not None
    assert simulation.incident.x.shape[1] == 9
    assert all(np.all(np.isfinite(getattr(simulation.incident, key))) for key in ("x","y","tx","ty"))
    state.sample.specimen_mode="reference"; state.sample.reference_sample_key="si_110"
    state.sample.thickness_nm=.4; state.sample.wave_grid_pixels=32
    state.sample.wave_field_of_view_angstrom=8.; state.sample.wave_atomistic_enabled=True
    state.sample.wave_multislice_enabled=True; state.sample.wave_frozen_phonon_enabled=False
    prepared=prepare_specimen_potentials(state,load_specimen_preset("si_110"))
    assert prepared.metrics["atomistic_applied"]
    potential=prepared.potential_configurations_v_angstrom[0]
    # A numerical kernel fixture, not an inferred wave from the gun or its rays.
    plane=np.ones(potential.shape[-2:],dtype=complex)
    plane/=np.sqrt(np.sum(np.abs(plane)**2))
    exit_wave, diagnostics=propagate_multislice(plane,potential,
        pixel_size_angstrom=(float(np.diff(prepared.y_angstrom)[0]),float(np.diff(prepared.x_angstrom)[0])),
        wavelength_angstrom=electron(state)[2]*10.,
        interaction_constant_rad_per_v_angstrom=interaction_constant_rad_per_v_angstrom(state.beam_voltage_kv),
        slice_thicknesses_angstrom=prepared.slice_thicknesses_angstrom)
    assert diagnostics.compute_backend=="NumPy CPU"
    assert np.all(np.isfinite(exit_wave))
    exit_norm=float(np.sum(np.abs(exit_wave)**2))
    assert 0 < exit_norm <= 1.+1e-10
    try:
        require_gun_wave_source(state,product="Installation wave-image check")
    except UnsupportedWaveSource as error:
        wave_images={"status":"UNAVAILABLE","reason":str(error)}
    else:
        wave_images={"status":"NOT_RUN","reason":"Admission alone does not execute or validate an image"}
    app=QApplication.instance() or QApplication([])
    inspector=ModelInspectorPage(); inspector.set_state(state); inspector.resize(1000,650)
    inspector.show(); app.processEvents(); inspector.close()
    gpu=cupy_capability()
    return {"schema":"installation-smoke-v2","status":"PASS",
        "scope":"Package imports, particle column, independent CIF/multislice kernel and offscreen GUI; excludes production TEM/STEM images",
        "utc":datetime.now(timezone.utc).isoformat(),"installed_module":str(Path(temsim.__file__).resolve()),
        "solver_source_sha256":solver_source_identity(),
        "particle_column":{"status":"PASS","rays":int(simulation.incident.x.shape[1])},
        "specimen_kernel":{"status":"PASS","case":"Si [110], 0.4 nm, 32 grid, normalized plane-wave fixture",
            "grid_shape":list(exit_wave.shape),"exit_norm":exit_norm,"backend":diagnostics.compute_backend},
        "wave_images":wave_images,"gui":"offscreen model inspector opened",
        "gpu_validation":"NOT_RUN" if not gpu.available else "NOT_RUN: this script validates CPU installation",
        "gpu_capability":gpu.detail,"versions":{n:metadata.version(n) for n in ("numpy","scipy","abtem","PySide6","tem-simulator-v2")}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    receipt=installation_checks()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    print(json.dumps(receipt))
    return 0


if __name__=="__main__": raise SystemExit(main())
