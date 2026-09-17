"""Scoped topology targets and comparisons, never a passed model certificate."""
from hashlib import sha256
from pathlib import Path
from temsim import input_io
import tomllib

from temsim.immutable_json import freeze_json, thaw_json


@input_io.using_state_inputs
def topology_reference(snapshot):
    """Inspect captured labels without restoring state or executing physics."""
    from temsim.paths import OPERATING_MODE_CONFIG_ROOT
    graph = snapshot.graph
    nodes = graph["nodes"]
    root = nodes[graph["root"]["ref"]]["attributes"]
    assembly_ref = root.get("_resolved_assembly", {})
    assembly = nodes[assembly_ref["ref"]]["attributes"] if "ref" in assembly_ref else {}
    selected = {row["tuple"][0]: str(row["tuple"][1]).replace("\\", "/")
                for row in assembly.get("selected_module_paths", {}).get("tuple", ())}
    mode = root.get("illumination_mode")
    scope = dict(snapshot_id=snapshot.digest, implementation=snapshot.implementation,
        selected_modules=selected, mode=mode, model=root.get("simulation_mode"),
        qualification="NOT_ESTABLISHED")
    if (selected.get("gun") != "gun/FEG.toml"
            or selected.get("column") != "column/C3_ProbeCorrector.toml"
            or "beam_blanker" in selected or mode not in {"TEM", "STEM"}):
        return freeze_json(dict(scope, status="NO_SCOPED_REFERENCE",
            reason="No approved topology reference for this captured gun, assembly and mode"))
    if mode == "TEM":
        path = Path(OPERATING_MODE_CONFIG_ROOT)/"illumination_targets.toml"
        content = input_io.read_bytes(path)
        reference = tomllib.loads(content.decode("utf-8"))["crossover_baselines"]["feg_c3_probe_corrector_microprobe"]
        if reference.get("authorised") is not True:
            return freeze_json(dict(scope, status="REFERENCE_NOT_AUTHORISED"))
        interpretation = "User-approved Microprobe topology target, distinct from the historical Nanoprobe target"
    else:
        path = Path(OPERATING_MODE_CONFIG_ROOT)/"topology_references.toml"
        content = input_io.read_bytes(path)
        reference = tomllib.loads(content.decode("utf-8"))["feg_c3_probe_corrector_nanoprobe"]
        interpretation = reference["interpretation"]
    if len(reference["intervals"]) != reference["count"]:
        raise ValueError("Topology reference count and ordered intervals disagree")
    return freeze_json(dict(scope, status="TARGET_ONLY", reference=reference,
        reference_file=str(path), reference_sha256=sha256(content).hexdigest(),
        interpretation=interpretation))


def topology_run(roots, components, *, surface_mm, focus_tolerance_nm, reference):
    """Keep every raw root and distinguish the separately constrained surface focus."""
    from temsim.optics.beam_path_audit import crossover_intervals, partition_surface_crossovers
    intermediate, terminal = partition_surface_crossovers(roots, surface_mm, tolerance_nm=focus_tolerance_nm)
    intervals = thaw_json(freeze_json(crossover_intervals([r["z_mm"] for r in intermediate], components)))
    expected = reference.get("reference") if reference.get("status") == "TARGET_ONLY" else None
    return dict(roots=roots, intermediate_count=len(intermediate), terminal_roots=terminal,
        ordered_intervals=intervals, component_planes=components,
        surface_mm=surface_mm, focus_tolerance_nm=focus_tolerance_nm,
        reference_match=None if expected is None else
            len(intermediate) == expected["count"] and intervals == thaw_json(expected["intervals"]),
        status="EXECUTED_AT_FIXED_SAMPLING", qualification="NOT_ESTABLISHED")


def compare_topology(left, right):
    if left is None or right is None:
        return dict(status="NOT_REQUESTED")
    if any(row["status"] != "EXECUTED_AT_FIXED_SAMPLING" for row in (left, right)):
        return dict(status="UNRESOLVED", reason="At least one proposed crossover was not executed successfully")
    same = (left["ordered_intervals"] == right["ordered_intervals"]
            and left["intermediate_count"] == right["intermediate_count"]
            and len(left["terminal_roots"]) == len(right["terminal_roots"]))
    target_failed = any(row["reference_match"] is False for row in (left, right))
    return dict(status="MATCH_FOR_CHECKED_AXIS" if same and not target_failed else "UNRESOLVED",
        same_ordered_intervals=same, scoped_reference_failed=target_failed,
        reason="Fixed-sampling root execution and this numerical axis only; axial bracket and other axes still need independent checks")
