"""Export an explicit grounded-tip profile and reproducible CPU diagnostics.

Never changes an open application's settings or generates a substitute image.
The output directory must be new. Optional Qt rendering is offscreen only.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np


def render_editor(gun, path):
    """Offline UI evidence, with an installed font rather than Qt fallback."""
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QApplication
    from temsim.gui.gun_source_dialog import GunSourceDialog
    application = QApplication.instance() or QApplication([])
    for filename in ("segoeui.ttf", "segoeuib.ttf"):
        font = Path(os.environ.get("WINDIR", "C:/Windows"))/"Fonts"/filename
        if font.is_file():
            QFontDatabase.addApplicationFont(str(font))
    application.setFont(QFont("Segoe UI", 10))
    dialog = GunSourceDialog(gun)
    dialog.resize(780, 700)
    dialog.show()
    application.processEvents()
    dialog.grab().save(str(path))
    dialog.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trace-rays", type=int, default=9)
    parser.add_argument("--render-editor", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from temsim.optics.column import default_state
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.electron_gun.tip_surface import load_tip_surface_reference, reference_path_for_gun
    from temsim.physics.grounded_tip_field import field_request
    from temsim.profile_io import save_profile
    state = default_state()
    gun = state.electron_gun
    path = reference_path_for_gun(gun)
    model = load_tip_surface_reference(path)
    gun.emitter.coherence = None
    gun.emitter.surface_model = model
    gun.emitter.ray_count = args.trace_rays
    save_profile(args.output/"grounded-tip-profile.toml", state, AssemblyCatalog().default_selection())
    evidence = {"schema": "grounded-tip-reference-evidence-v1", "reference_sha256": sha256(path.read_bytes()).hexdigest(),
                "model": model.to_dict(), "field_request": field_request(gun),
                "scope": "CPU reference fields and classical gun trajectories only", "coherent_image_generated": False}
    points = np.array([[0., 0., z] for z in (0., 1e-9, 1e-8, 1e-7, 1e-6, 1e-3, gun.exit_plane_z_mm*1e-3)])
    refinement = []
    for factor in (1, 2, 4):
        gun.emitter.surface_model = replace(model, field_numerics=replace(model.field_numerics,
            radial_nodes=model.field_numerics.radial_nodes*factor,
            axial_nodes=model.field_numerics.axial_nodes*factor,
            apex_cells_per_radius=model.field_numerics.apex_cells_per_radius*factor))
        field = gun.electric_field
        row = {"factor": factor, "report": field.report, "z_m": points[:, 2].tolist(),
               "potential_v": field.potential_v_at_global_positions(points).tolist(),
               "field_z_v_per_m": field.field_at_global_positions_v_per_m(points)[:, 2].tolist()}
        refinement.append(row)
        print(f"Field refinement {factor}: {row['report']['grid_shape']}", flush=True)
    evidence["field_refinement"] = refinement
    gun.emitter.surface_model = model
    started = time.perf_counter()
    try:
        result = gun.trace_to_exit(args.trace_rays)
        evidence["trace"] = {**result.surface_model_report, "ray_count": args.trace_rays,
            "surviving_count": int(result.exit_bundle.alive.sum()), "elapsed_s": time.perf_counter()-started,
            "emitted_current_a": result.emitted_current_a, "exit_current_a": result.c1_transmitted_current_a}
        np.savez_compressed(args.output/"gun-rays.npz", z_mm=result.z_mm, x_m=result.x_m, y_m=result.y_m,
                            alive=result.exit_bundle.alive, energy_offset_ev=result.exit_bundle.energy_offset_ev,
                            source_current_weight=result.exit_bundle.weight)
        print(json.dumps(evidence["trace"], indent=2), flush=True)
    except Exception as error:
        evidence["trace"] = {"status": "FAILED", "error_type": type(error).__name__, "error": str(error)}
        raise
    finally:
        (args.output/"report.json").write_text(json.dumps(evidence, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    if args.render_editor:
        render_editor(gun, args.output/"tip-editor.png")


if __name__ == "__main__":
    main()
