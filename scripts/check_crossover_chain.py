"""Detached spot/crossover audit, optionally using an extracted historical tree.

Historical execution is a comparison only. It never replaces the active source,
restores a prohibited downstream source, or writes calculated particle arrays.
"""
import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", type=Path,
                        help="Extracted src/configs tree; isolated comparison, not active settings")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output-report", type=Path, help="Lightweight JSON only, never particle arrays")
    parser.add_argument("--reference-report", type=Path, help="Executed incident-crossover report to compare")
    parser.add_argument("--refine-crossovers", action="store_true", help="Re-execute each bracket at its physical waist")
    parser.add_argument("--rays", type=int, default=769)
    parser.add_argument("--axial-spacing-mm", type=float, default=.5)
    parser.add_argument("--step-mm", type=float, default=.05)
    parser.add_argument("--sampling", choices=("uniform_area", "apex_stratified_v1"),
                        help="Numerical cap quadrature only; physical emission stays unchanged")
    args = parser.parse_args()
    if args.output_report and args.output_report.exists():
        raise FileExistsError("Report already exists; choose a new path to preserve prior evidence")
    if args.historical_root:
        import os
        root = args.historical_root.resolve(strict=True)
        sys.path.insert(0, str(root/"src"))
        os.environ["TEMSIM_PROJECT_ROOT"] = str(root)
    from temsim.optics.column import default_state
    from temsim.operating_modes import apply_operating_mode_pair
    import numpy as np
    # The auditor itself is current code; transport/configuration are from the
    # explicitly selected tree. Report both rather than presenting an old run
    # as an active calibration.
    path = Path(__file__).resolve().parents[1]/"src/temsim/optics/beam_path_audit.py"
    spec = importlib.util.spec_from_file_location("spot_auditor", path)
    audit = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = audit
    spec.loader.exec_module(audit)
    state = default_state()
    apply_operating_mode_pair(state, "nano_probe", "diffraction")
    if args.profile:
        from temsim.profile_io import read_profile, apply_profile_values
        from temsim.assembly_catalog import AssemblyCatalog
        from temsim.column.state_layout import apply_physical_layout_to_state
        selection, values = read_profile(args.profile)
        AssemblyCatalog().apply(state, selection)
        skipped = apply_profile_values(state, values)
        apply_physical_layout_to_state(state, preserve_operating_parameters=True)
        print("Profile skipped fields:", skipped, flush=True)
    if hasattr(state, "vacuum_map"):
        state.vacuum_map.enabled = False  # Declared vacuum-free optical comparison.
    state._optical_tuning = True
    state.electron_gun.emitter.ray_count = args.rays
    if args.sampling:
        from dataclasses import replace
        model = state.electron_gun.emitter.surface_model
        if model is None:
            raise ValueError("Cap quadrature requires a physical curved emitting surface")
        state.electron_gun.emitter.surface_model = replace(model,
            emission=replace(model.emission, spatial_sampling=args.sampling))
    state.step_mm = args.step_mm
    if not np.isfinite(args.axial_spacing_mm) or args.axial_spacing_mm <= 0:
        raise ValueError("Axial spacing must be positive and finite")
    end = float(state.sample.upper_surface_z_mm)
    c1_z = float(next(lens.z_mm for lens in state.lenses if lens.key == "condenser_lens_1"))
    planes = np.unique(np.r_[np.arange(state.electron_gun.exit_plane_z_mm,
                                      end, args.axial_spacing_mm), c1_z, end])
    gun, cp, mask = audit.incident_checkpoints(state, planes, step_mm=args.step_mm)
    e = state.electron_gun.emit(args.rays)
    emission = audit.emission_measurement(e)
    rows = [emission]
    for name, z in [("Gun exit", state.electron_gun.exit_plane_z_mm),
                    ("C1 centre", c1_z), ("Specimen upper surface", end)]:
        indices = np.flatnonzero(abs(cp.z_mm-z) < 1e-10)
        if len(indices) != 1:
            raise ValueError(f"Missing exact diagnostic plane: {name}")
        j = int(indices[0])
        rows.append(audit.spot_measurement(name, float(cp.z_mm[j]),
            (cp.x_m[j], cp.y_m[j], cp.tx_rad[j], cp.ty_rad[j]), e.weight, mask[j]))
    crossovers = audit.crossover_candidates(cp, mask, e.weight)
    if args.refine_crossovers:
        refined = []
        for crossover in crossovers:
            item = audit.refine_crossover_candidate(state, crossover, step_mm=args.step_mm)
            item["rms_radius_nm"] = item["measurement"]["rms_radius_nm"]
            refined.append(item)
        crossovers = refined
    component_planes = audit.optical_component_planes(state)
    topology = "NOT_COMPARED"
    if args.reference_report:
        reference = json.loads(args.reference_report.read_text(encoding="utf-8"))
        try:
            audit.require_same_topology([r["z_mm"] for r in reference["crossovers"]],
                [r["z_mm"] for r in crossovers], reference["component_planes"],
                candidate_component_planes=component_planes)
            topology = "SAME_INCIDENT_INTERVALS; interpolation/step convergence still required"
        except ValueError as error:
            topology = "FAIL: "+str(error)
    report = dict(scope="historical-tree comparison" if args.historical_root else "current-tree audit",
        historical_root=str(args.historical_root) if args.historical_root else None,
        scope_z_mm=[float(cp.z_mm[0]), float(cp.z_mm[-1])],
        vacuum_transport=False, rays=args.rays, step_mm=args.step_mm,
        axial_spacing_mm=args.axial_spacing_mm,
        source_model="curved surface" if state.electron_gun.emitter.surface_model else "historical tip Gaussian",
        lens_percent={lens.key: lens.percent for lens in state.lenses if lens.enabled},
        spots=[asdict(row) for row in rows],
        crossovers=crossovers, component_planes=component_planes, topology_comparison=topology,
        qualification="DIAGNOSTIC_ONLY; gun and post-specimen topology are separate checks")
    def strict_json(value):
        if isinstance(value, dict):
            return {key: strict_json(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [strict_json(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    payload = json.dumps(strict_json(report), indent=2, allow_nan=False)
    print(payload, flush=True)
    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        with args.output_report.open("x", encoding="utf-8") as stream:
            stream.write(payload+"\n")


if __name__ == "__main__":
    main()
