"""Explicit export scope; removing assets never creates a restorable state."""
from collections.abc import Mapping
from hashlib import sha256

from temsim.immutable_json import thaw_json
from temsim.instrument_snapshot import InstrumentSnapshot, SNAPSHOT_SCHEMA

METADATA_SCHEMA = "working-point-metadata-only-v1"
EXPORT_MODES = ("inputs_and_results", "inputs", "metadata")


def _summary_value(value):
    if isinstance(value, Mapping):
        if "array" in value and "dtype" in value and "shape" in value:
            return {"omitted_array": {"dtype": value["dtype"], "shape": list(value["shape"]),
                    "sha256": sha256(bytes.fromhex(value["array"])).hexdigest()}}
        if "asset" in value and "dtype" in value and "shape" in value:
            return {"omitted_array": {"dtype": value["dtype"], "shape": list(value["shape"]),
                    "sha256": value["asset"]}}
        return {key: (None if key == "content_hex" else _summary_value(item))
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summary_value(item) for item in value]
    return value


def export_checkpoint(checkpoint, mode):
    """Derive an honest package identity without restoring or running optics.

    Complete input exports include the existing declared dependencies. Exact
    restoration still verifies original files until the archive resolver covers
    all implicit loaders; this function does not claim offline reproducibility.
    """
    from temsim.sampling_diagnostics import checkpoint_sampling_summary
    from temsim.working_point import WorkingPointCheckpoint
    from temsim.working_point_archive import WorkingPointArchiveIndex

    if mode not in EXPORT_MODES:
        raise ValueError("Unknown working-point export content")
    if mode != "metadata":
        if checkpoint.is_metadata_only or checkpoint.snapshot.graph.get("schema") != SNAPSHOT_SCHEMA:
            raise ValueError("This record has no complete input payload; export metadata only")
        if any(row["content_hex"] is None for row in checkpoint.snapshot.external_inputs):
            raise ValueError("A declared external input is missing; export metadata only")
    if mode == "inputs_and_results":
        if not checkpoint.has_retained_payload:
            raise ValueError("No retained results; choose complete inputs or metadata")
        if isinstance(checkpoint, WorkingPointArchiveIndex):
            raise ValueError("Load retained data before exporting inputs and results")
        return checkpoint
    if mode == "inputs" and checkpoint.is_input_design:
        return WorkingPointCheckpoint(checkpoint.snapshot, {}, checkpoint.plane_z_mm,
            checkpoint.stage_signature, checkpoint.metadata, checkpoint.parent_id)
    snapshot = checkpoint.snapshot
    if mode == "metadata":
        if checkpoint.is_metadata_only:
            return WorkingPointCheckpoint(snapshot, {}, checkpoint.plane_z_mm,
                checkpoint.stage_signature, checkpoint.metadata, checkpoint.parent_id)
        graph = _summary_value(snapshot.graph)
        graph["schema"] = METADATA_SCHEMA
        snapshot = InstrumentSnapshot(graph, tuple(_summary_value(snapshot.external_inputs)), snapshot.implementation)
    metadata = {
        "package_kind": "METADATA_ONLY" if mode == "metadata" else "INSTRUMENT_INPUTS_ONLY",
        "exported_from_id": checkpoint.digest,
        "original_snapshot_id": checkpoint.snapshot.digest,
        "validation_status": "NOT_RUN",
        "source_representation": checkpoint.metadata.get("source_representation", "captured-inputs-only"),
        "created_at_utc": checkpoint.metadata.get("created_at_utc", "Unrecorded"),
        "offline_reproducible": mode != "metadata" and "archived_inputs" in snapshot.graph,
        "restoration_policy": ("READ_ONLY_METADATA" if mode == "metadata" else
                               "READ_ONLY_ARCHIVE" if "archived_inputs" in snapshot.graph else "VERIFY_ORIGINAL_DEPENDENCIES"),
    }
    if mode == "metadata":
        metadata["archived_record_metadata"] = _summary_value(checkpoint.metadata)
        metadata["unverified_scalar_summary"] = thaw_json(checkpoint_sampling_summary(checkpoint))
    return WorkingPointCheckpoint(snapshot, {}, checkpoint.plane_z_mm,
        snapshot.physical_digest, metadata, checkpoint.digest)


def make_portable_inputs(checkpoint):
    """Explicitly capture a new input design, leaving old results untouched."""
    from datetime import datetime, timezone
    from temsim import input_io
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.working_point import WorkingPointCheckpoint
    from temsim.optics.electron_gun.source_policy import require_physical_gun_source
    if checkpoint.is_metadata_only:
        raise ValueError("A metadata-only record has no inputs to make portable")
    state = checkpoint.snapshot.restore()
    require_physical_gun_source(state.electron_gun)
    payload = input_io.capture_input_archive(state)
    input_io.bind_archive(state, payload)
    snapshot = capture_instrument_snapshot(state)
    return WorkingPointCheckpoint(snapshot, {}, checkpoint.plane_z_mm, snapshot.physical_digest,
        dict(package_kind="INSTRUMENT_INPUTS_ONLY", source_representation="captured-inputs-only",
             validation_status="NOT_RUN", offline_reproducible=True, restoration_policy="READ_ONLY_ARCHIVE",
             created_at_utc=datetime.now(timezone.utc).isoformat(),
             archive_scope="Captured configuration tree and declared external inputs; compatible installed solver required"),
        checkpoint.digest)
