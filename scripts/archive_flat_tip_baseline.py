"""Archive/recompute the default flat-tip optical trajectory, never an exit source.

Numerical arrays and source archives stay local. A comparison executes the full
upstream calculation afresh; archived particles are never injected into it.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import time
import zipfile
import tomllib
from copy import deepcopy

import numpy as np


def build_state():
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.column import default_state
    from temsim.operating_modes import apply_operating_mode_pair
    from temsim.physics.optical_tuning import prepare_tuning_snapshot
    from temsim.column.state_layout import apply_physical_layout_to_state
    state = default_state()
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    catalog.apply(state, selection)
    apply_operating_mode_pair(state, "nano_probe", "diffraction",
        column_name=selection.column, recording_name=selection.recording)
    emitter = state.electron_gun.emitter
    assert emitter.surface_model is None and emitter.coherence is None
    assert emitter.curvature_nm_inv == 0
    emitter.ray_count = 49
    state.step_mm = state.history_step_mm = 1.0
    prepare_tuning_snapshot(state, "Preview")
    layout = apply_physical_layout_to_state(state)
    return state, layout


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def pack(value, arrays, name="result"):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return pack(value.tolist(), arrays, name)
        arrays[name] = value
        return {"array": name, "shape": list(value.shape), "dtype": value.dtype.str}
    if is_dataclass(value):
        return {f.name: pack(getattr(value, f.name), arrays, name+"/"+f.name) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): pack(v, arrays, name+"/"+str(k)) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [pack(v, arrays, name+"/"+str(i)) for i, v in enumerate(value)]
    if isinstance(value, np.generic):
        return pack(value.item(), arrays, name)
    if isinstance(value, float) and not np.isfinite(value):
        return {"float": value.hex()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unarchived result type: {type(value)} at {name}")


def render(simulation, state, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    stages = [simulation.incident, *simulation.branches.values()]
    for ax, direction in zip(axes, ("x", "y")):
        for stage in stages:
            for i in range(stage.x.shape[1]):
                live = ~np.isfinite(stage.blocked_z[i]) | (stage.z <= stage.blocked_z[i])
                ax.plot(stage.z[live], getattr(stage, direction)[live, i]*1e3,
                        color=plt.cm.hsv(i/stage.x.shape[1]), lw=.65, alpha=.8)
        ax.axvline(state.sample.z_mm, color="black", ls=":", lw=1)
        ax.set_ylabel(f"{direction.upper()} (mm)")
        ax.grid(alpha=.15)
    axes[-1].set_xlabel("Axial Z (mm)")
    fig.suptitle("Flat tip baseline | FEG / C3 + Probe Corrector | Nanoprobe / Diffraction\n"
                 "49 rays, 1 mm column step | physical coordinates | optical reference, no specimen signals")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compare_arrays(previous, current):
    rows = {}
    with np.load(previous, allow_pickle=False) as stored:
        if set(stored.files) != set(current):
            raise ValueError("Trajectory array inventory changed")
        for key, array in current.items():
            before = stored[key]
            equal = before.dtype == array.dtype and before.shape == array.shape
            if equal:
                equal = np.array_equal(before, array, equal_nan=True) if array.dtype.kind in "fc" else np.array_equal(before, array)
            row = {"equal": bool(equal)}
            if before.shape == array.shape and before.dtype.kind in "fci" and array.dtype.kind in "fci":
                finite = np.isfinite(before) & np.isfinite(array)
                row["maximum_absolute_difference"] = float(np.max(np.abs(before[finite]-array[finite]), initial=0))
            rows[key] = row
    return rows


def compare_repacked_inputs(previous, current, modules, root):
    """Allow only named storage repacks with exactly identical resolved data."""
    from temsim.module_manifest import read_document
    from temsim.subassemblies import dependencies
    old = {(row["role"], row["path"]): row for row in previous}
    new = {(row["role"], row["path"]): row for row in current}
    evidence = []
    for relative in modules:
        path = (root / relative).resolve()
        matches = [key for key in old if Path(key[1]).resolve() == path]
        if len(matches) != 1 or matches[0] not in new:
            raise ValueError("Repacked module must be an existing consumed input: " + str(path))
        key = matches[0]
        before = tomllib.loads(bytes.fromhex(old[key]["content_hex"]).decode("utf-8-sig"))
        after = read_document(path)
        after.pop("subassemblies", None)
        if before != after:
            raise ValueError("Repacked module changed resolved physical inputs: " + str(path))
        # Bind each new dependency to the captured bytes used by this run.
        raw = tomllib.loads(bytes.fromhex(new[key]["content_hex"]).decode("utf-8-sig"))
        if path.read_bytes().hex() != new[key]["content_hex"]:
            raise ValueError("Repacked module changed during calculation")
        for source, data in dependencies(raw, path).items():
            dep_key = ("assembly:subassembly", str(source))
            if dep_key not in new or new[dep_key]["content_hex"] != data.hex():
                raise ValueError("Subassembly changed during calculation or was not captured")
            new.pop(dep_key)
        evidence.append({"module": str(path), "old_sha256": old[key]["sha256"],
                         "new_sha256": new[key]["sha256"], "resolved_document_equal": True})
        old.pop(key)
        new.pop(key)
    if old != new:
        raise ValueError("External inputs beyond the declared storage repack changed")
    return evidence


def compare_navigation_graph(previous, current, root):
    """Permit only newly captured section metadata, verified against its files.

    Do not suppress geometry, vacuum settings, physical inputs or object graph
    differences. This allowance is explicitly requested for the stage-3 archive.
    """
    from temsim.module_manifest import read_document

    def plain(value):
        if not isinstance(value, dict):
            return value
        if "mapping" in value:
            return {plain(k): plain(v) for k, v in value["mapping"]}
        if "tuple" in value:
            return [plain(v) for v in value["tuple"]]
        raise ValueError("Unexpected navigation metadata encoding")

    candidate = deepcopy(current)
    evidence = []
    for index, node in enumerate(candidate["nodes"]):
        if node["type"] != "temsim.column.module_assembly:ModuleDefinition":
            continue
        geometry = node["attributes"]["geometry"]["mapping"]
        rows = [value for key, value in geometry if key == "navigation_subassemblies"]
        if not rows:
            continue
        original = previous["nodes"][index] if index < len(previous["nodes"]) else {}
        old_geometry = original.get("attributes", {}).get("geometry", {}).get("mapping", [])
        if any(key == "navigation_subassemblies" for key, value in old_geometry):
            continue  # Existing metadata must compare exactly, like every input.
        source = node["attributes"]["source_file"]
        expected = read_document(root/"configs/instruments"/source, capture_navigation=True).get("_navigation_subassemblies")
        if len(rows) != 1 or plain(rows[0]) != expected:
            raise ValueError(f"Captured assembly navigation differs from its definition: {source}")
        node["attributes"]["geometry"]["mapping"] = [[k, v] for k, v in geometry if k != "navigation_subassemblies"]
        evidence.append({"module": source, "subassemblies": [r["key"] for r in expected]})
    return {"physical_input_graph_equal": previous == candidate, "verified_navigation_metadata": evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--repacked-module", action="append", default=[],
                        help="Repository-relative module repacked without physical changes; requires --compare")
    parser.add_argument("--verify-navigation-metadata", action="store_true",
                        help="Verify new captured subassembly metadata against definitions; compare every other input exactly")
    args = parser.parse_args()
    if args.repacked_module and not args.compare:
        parser.error("--repacked-module requires --compare")
    if args.verify_navigation_metadata and not args.compare:
        parser.error("--verify-navigation-metadata requires --compare")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.physics.simulation import run
    from temsim.optics.energy_filter_raytrace import simulate_energy_filter
    root = Path(__file__).resolve().parents[1]
    started = time.perf_counter()
    state, layout = build_state()
    snapshot = capture_instrument_snapshot(state).to_dict()
    write_json(output/"working_point.json", snapshot)
    # Archive exact source/config inputs, including local uncommitted work.
    source_hashes = {}
    with zipfile.ZipFile(output/"source_and_inputs.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in (root/"src", root/"configs"):
            for file in sorted(folder.rglob("*")):
                if file.is_file() and file.suffix in {".py", ".toml", ".json"}:
                    relative = file.relative_to(root).as_posix()
                    data = file.read_bytes()
                    source_hashes[relative] = sha256(data).hexdigest()
                    archive.writestr(relative, data)
        archive.write(Path(__file__), "scripts/"+Path(__file__).name)
    write_json(output/"input_inventory.json", source_hashes)
    before_graph = snapshot["graph"]
    print("Inputs archived; executing tip, extraction, acceleration and column", flush=True)
    simulation = run(state, resolved_layout=layout, optical_only=True)
    # Preserve the normal Preview trajectory, and separately execute the actual
    # downstream filter from these same transported particles when enabled.
    print("Column finished; tracing installed energy filter", flush=True)
    energy_filter = simulate_energy_filter(state, simulation)
    arrays = {}
    metadata = pack({"gun": simulation.gun_trace, "incident": simulation.incident,
                     "branches": simulation.branches, "energy_filter": energy_filter}, arrays)
    write_json(output/"trajectories.json", metadata)
    np.savez_compressed(output/"trajectories.npz", **arrays)
    render(simulation, state, output/"ray_diagram.png")
    result = {
        "schema": "flat-tip-assembly-baseline-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "python": platform.python_version(), "numpy": np.__version__, "solver_identity": snapshot["implementation"],
        "working_point_digest": snapshot["digest"], "seconds": time.perf_counter()-started,
        "scope": "Preview optical reference plus installed energy-filter particle transport; no specimen signals or waves",
        "source": "tip-origin classical flat emission; no downstream source or cached upstream injection",
        "recipe": {"assembly": "FEG / C3 + Probe Corrector / Energy Filter", "illumination": "nano_probe",
                   "projection": "diffraction", "curvature_nm_inv": 0., "rays": 49, "column_step_mm": 1.,
                   "vacuum_enabled": state.vacuum_map.enabled},
        "part_count": len(state._resolved_assembly.parts), "array_count": len(arrays),
        "sample_z_mm": state.sample.z_mm,
        "sample_survivors": int(np.count_nonzero(simulation.incident.alive)),
        "branch_stops": {key: dict(Counter(branch.blocked_key)) for key, branch in simulation.branches.items()},
        "archives": {name: sha256((output/name).read_bytes()).hexdigest()
                     for name in ("working_point.json", "trajectories.json", "trajectories.npz", "source_and_inputs.zip")},
    }
    if args.compare:
        baseline = args.compare.resolve()
        previous = json.loads((baseline/"report.json").read_text(encoding="utf-8"))
        for name, checksum in previous["archives"].items():
            if sha256((baseline/name).read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Baseline archive checksum mismatch: {name}")
        original = json.loads((baseline/"working_point.json").read_text(encoding="utf-8"))
        comparison = compare_arrays(baseline/"trajectories.npz", arrays)
        result["comparison"] = {"baseline": str(baseline), "input_graph_equal": original["graph"] == before_graph,
            "external_inputs_equal": original["external_inputs"] == snapshot["external_inputs"],
            "result_metadata_equal": json.loads((baseline/"trajectories.json").read_text(encoding="utf-8")) == metadata,
            "arrays": comparison}
        repack = []
        if args.repacked_module:
            repack = compare_repacked_inputs(original["external_inputs"], snapshot["external_inputs"], args.repacked_module, root)
            result["comparison"]["verified_storage_repack"] = repack
        graph_equal = result["comparison"]["input_graph_equal"]
        if args.verify_navigation_metadata:
            navigation = compare_navigation_graph(original["graph"], before_graph, root)
            result["comparison"].update(navigation)
            graph_equal = navigation["physical_input_graph_equal"]
        result["comparison"]["passed"] = all((graph_equal,
            result["comparison"]["external_inputs_equal"] or bool(repack), result["comparison"]["result_metadata_equal"],
            all(row["equal"] for row in comparison.values())))
    write_json(output/"report.json", result)
    print(json.dumps({k:v for k,v in result.items() if k not in {"archives", "comparison"}}, indent=2), flush=True)
    if args.compare:
        print("Comparison passed:", result["comparison"]["passed"], flush=True)
        if not result["comparison"]["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
