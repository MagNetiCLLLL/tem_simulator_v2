"""Run one explicit high-accuracy request and verify the application's disk seed.

This uses the same snapshot, pipeline and incident-seed codec as the desktop
controller. It does not solve lens presets or change an open GUI's state.
Only incident propagation is currently restartable from disk; full signals
are not serialized by this utility. No pickle or arbitrary object codec is used.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from temsim.artifact_store import ArtifactStore
from temsim.assembly_catalog import AssemblyCatalog
from temsim.cache_preferences import load_cache_preferences
from temsim.calculation_manifest import (
    CalculationManifest, ExternalInputIdentity, SolverIdentity, capture_calculation_manifest,
)
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.gui.calculation_controller import (
    CalculationController,
    HIGH_ACCURACY_MEMORY_BUDGET_BYTES,
    default_artifact_cache_root,
    estimate_calculation_memory_bytes,
)
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.profile_io import apply_profile_values, read_profile, save_profile
from temsim.simulation_modes import switch_mode
from temsim.simulation_pipeline import calculate


def verify_incident_seed(original, restored) -> list[str]:
    """Compare the retained phase space, current weights and propagation plan."""
    if restored is None:
        raise RuntimeError("Calculation finished, but no restart seed could be read back")
    compared = []
    groups = (
        ("incident", original.incident, restored.incident,
         ("z", "x", "y", "tx", "ty", "alive", "blocked_z", "energy_offset_ev", "ray_weight")),
        ("checkpoint", original.incident_checkpoints, restored.incident_checkpoints,
         ("z_mm", "x_m", "tx_rad", "y_m", "ty_rad")),
        ("gun_exit", original.gun_trace.exit_bundle, restored.gun_trace.exit_bundle,
         ("x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev", "weight", "ray_id", "alive")),
        ("plan", original.incident_plan, restored.incident_plan,
         tuple(field.name for field in fields(original.incident_plan)
               if isinstance(getattr(original.incident_plan, field.name), np.ndarray))),
    )
    for label, source, loaded, names in groups:
        for name in names:
            before, after = getattr(source, name), getattr(loaded, name)
            if before is None and after is None:
                continue
            if before is None or after is None or not np.array_equal(before, after, equal_nan=True):
                raise RuntimeError(f"Persisted {label}.{name} differs from the computed result")
            compared.append(f"{label}.{name}")
    if original.incident_plan.signature != restored.incident_plan.signature:
        raise RuntimeError("Persisted propagation plan does not match the calculation")
    return compared


def read_manifest(path: Path) -> CalculationManifest:
    """Restore only the explicit JSON manifest schema, never Python objects."""
    document = json.loads(path.read_text(encoding="utf-8"))
    values = {field.name: document[field.name] for field in fields(CalculationManifest)}
    values["solver"] = SolverIdentity(**document["solver"])
    values["external_inputs"] = tuple(ExternalInputIdentity(**row) for row in document["external_inputs"])
    result = CalculationManifest(**values)
    if result.digest != document["digest"]:
        raise ValueError("Saved calculation manifest digest does not match its contents")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--profile", type=Path, help="Saved operating profile")
    source.add_argument("--startup-defaults", action="store_true",
                        help="Use the desktop's default assembly and stored operating values")
    parser.add_argument("--rays", type=int, default=15000)
    parser.add_argument("--step-mm", type=float, default=0.1)
    parser.add_argument("--describe", action="store_true", help="Inspect without computing or writing")
    parser.add_argument("--verify-existing", action="store_true",
                        help="Verify a saved request's disk seed without repeating any physics")
    parser.add_argument("--cache-root", type=Path, help="Explicit cache store; default: application user cache")
    parser.add_argument("--output", type=Path,
                        help="New directory for the request profile, manifest and verification report")
    args = parser.parse_args(argv)
    if not args.describe and args.output is None:
        parser.error("--output is required for a reproducible calculation")
    if args.verify_existing and (args.describe or args.profile or args.startup_defaults):
        parser.error("--verify-existing uses the saved manifest, not new input settings")
    if not args.verify_existing and not (args.profile or args.startup_defaults):
        parser.error("Choose --profile or --startup-defaults")
    if not args.verify_existing and args.output is not None and args.output.exists():
        parser.error("The output already exists; choose a new directory")
    from PySide6.QtCore import QSettings
    preferences = load_cache_preferences(QSettings("TEM Simulator", "TEM Simulator v2"))
    cache_root = args.cache_root or default_artifact_cache_root()
    if args.verify_existing:
        manifest = read_manifest(args.output / "manifest.json")
        store = ArtifactStore(cache_root, quota_bytes=preferences.disk_cache_budget_bytes)
        seed = CalculationController.load_persisted_incident_seed(store, manifest)
        if seed is None:
            raise RuntimeError("No matching incident seed is present; no physics was run")
        completion = args.output / "calculation.json"
        report = json.loads(completion.read_text(encoding="utf-8")) if completion.exists() else {}
        report.update({
            "verified_utc": datetime.now(timezone.utc).isoformat(),
            "disk_cache_root": str(store.root), "cache_readback_verified": True,
            "verification": "Stored array checksums, logical content digest and exact request dependencies",
            "physics_repeated": False, "manifest_digest": manifest.digest,
            "request_signature": manifest.calculation_signatures["request"],
            "incident_shape": list(seed.incident.x.shape),
            "surviving_sample_rays": int(np.count_nonzero(seed.incident.alive)),
            "persistent_scope": "incident propagation seed only; not complete TEM/STEM/EDS products",
        })
        (args.output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        return 0

    catalog = AssemblyCatalog()
    state = default_state()
    if args.profile:
        selection, values = read_profile(args.profile)
        catalog.apply(state, selection)
        skipped = apply_profile_values(state, values)
        if skipped:
            raise ValueError(f"Unsupported profile fields: {skipped}")
    else:
        selection = catalog.default_selection()
        catalog.apply(state, selection)
        if state.nanopulser.installed:
            raise ValueError("Use an explicit profile for an installed NanoPulser; no preset solve is allowed")
        apply_operating_mode_pair(
            state, "micro_probe" if state.illumination_mode.upper() == "TEM" else "nano_probe",
            "imaging" if state.projector_mode.lower() == "image" else "diffraction",
            column_name=selection.column, recording_name=selection.recording,
        )
        switch_mode(state, "ideal")
    apply_physical_layout_to_state(state, preserve_operating_parameters=True)
    snapshot = CalculationController._calculation_snapshot(state, "High accuracy", args.rays, args.step_mm)
    estimate = estimate_calculation_memory_bytes(snapshot, "High accuracy", args.rays, args.step_mm)
    if estimate > HIGH_ACCURACY_MEMORY_BUDGET_BYTES:
        raise ValueError(f"Estimated peak {estimate} exceeds the application's memory allowance")
    description = {
        "source": str(args.profile.resolve()) if args.profile else "desktop startup defaults",
        "assembly": {name: getattr(selection, name) for name in ("gun", "column", "recording")},
        "quality": "High accuracy", "rays": args.rays, "step_mm": args.step_mm,
        "voltage_kv": snapshot.beam_voltage_kv, "simulation_mode": snapshot.simulation_mode,
        "specimen_mode": snapshot.sample.specimen_mode,
        "specimen_preset": snapshot.sample.specimen_preset_key,
        "tem_wave_enabled": snapshot.sample.wave_enabled,
        "stem_wave_enabled": snapshot.sample.stem_wave_enabled,
        "estimated_peak_bytes": estimate, "disk_cache_root": str(cache_root),
        "disk_cache_quota_bytes": preferences.disk_cache_budget_bytes,
        "persistent_scope": "incident propagation seed only; not complete TEM/STEM/EDS products",
    }
    print(json.dumps(description, indent=2), flush=True)
    if args.describe:
        return 0

    # Fail before expensive work if the real application cache cannot be opened.
    store = ArtifactStore(cache_root, quota_bytes=preferences.disk_cache_budget_bytes)
    manifest = capture_calculation_manifest(snapshot, ray_count=args.rays, step_mm=args.step_mm,
                                            selection=selection)
    serialized_manifest = json.dumps(manifest.to_dict(), indent=2)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    save_profile(output / "request.toml", snapshot, selection)
    (output / "manifest.json").write_text(serialized_manifest, encoding="utf-8")
    last_progress = [-float("inf"), ""]

    def progress(completed, total, stage):
        now = perf_counter()
        if stage != last_progress[1] or now - last_progress[0] >= 5 or completed == total:
            print(f"{completed}/{total}: {stage}", flush=True)
            last_progress[:] = [now, stage]

    started = perf_counter()
    result = calculate(snapshot, progress_callback=progress)
    calculation_seconds = perf_counter() - started
    report = {
        **description, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "calculation_seconds": calculation_seconds,
        "request_signature": result.signatures["request"],
        "calculated_products": sorted(result.calculated_products),
        "reused_products": sorted(result.reused_products),
        "incident_shape": list(result.simulation.incident.x.shape),
        "backend": str(getattr(result.state_snapshot, "active_backend", "unknown")),
        "cache_readback_verified": False,
    }
    # Retain completion evidence even if the OS rejects the later cache write.
    (output / "calculation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    identity = CalculationController.persist_incident_seed(store, manifest, result)
    # A new store is the restart boundary. Readback verifies both the content
    # checksums (inside the codec) and the retained arrays against this result.
    reopened = ArtifactStore(store.root, quota_bytes=preferences.disk_cache_budget_bytes)
    restored = CalculationController.load_persisted_incident_seed(reopened, manifest)
    compared = verify_incident_seed(result.simulation, restored)
    report.update({
        "artifact_identity": identity,
        "cache_readback_verified": True, "compared_arrays": compared,
    })
    (output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
