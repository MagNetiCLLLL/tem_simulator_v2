"""Gun-wave payload codec for the existing quota/checksum ArtifactStore.

Exact upstream operator inputs may be shared by different complete snapshots.
Reuse creates a new checkpoint with explicit parentage; the captured historical
snapshot and wave are never edited or relabelled in place. Nearby optical
settings and old ray-conditioned pupils are not coherent seeds.
"""
from dataclasses import replace

from temsim.artifact_store import ArtifactIntegrityError
from temsim.calculation_manifest import capture_calculation_manifest
from temsim.instrument_snapshot import InstrumentSnapshot
from temsim.physics.gun_wave_transport import (
    GunWaveCheckpoint, _prepare_gun_wave_plan, _execute_gun_wave_plan,
)
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.wave_flux import BeamState, WaveMode
from temsim.optics.electron_gun.effective_source import _reconstruct_historical_emission

PRODUCT = "gun_wave_incident"
CODEC = "gun-wave-modes-v1"
FIELDS = ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad")


def gun_wave_manifest(state):
    return _capture_gun_wave_request(state)[0]


def _capture_gun_wave_request(state):
    manifest = capture_calculation_manifest(state)
    prepared = _prepare_gun_wave_plan(manifest.instrument_snapshot)
    manifest = replace(manifest, calculation_signatures={**manifest.calculation_signatures,
                       PRODUCT: prepared.stage_signature})
    return manifest, prepared


def gun_wave_payload(checkpoint):
    """One numeric codec for disk artifacts and portable working points."""
    if not isinstance(checkpoint, GunWaveCheckpoint):
        raise TypeError("Only a gun-wave checkpoint can be stored with this codec")
    snapshot = checkpoint.snapshot
    arrays, modes = {}, []
    for i, mode in enumerate(checkpoint.beam.modes):
        if mode.axial_reference is not None:
            raise ValueError("The historical exit-wave codec cannot discard a physical-tip axial reference")
        if mode.scattering_history:
            raise ValueError("The historical exit-wave codec cannot discard a scattering history")
        row = {"mode_id": mode.mode_id, "energy_kev": mode.energy_kev,
               "weight": mode.weight_per_reference_electron, "fields": []}
        for field in FIELDS:
            value = getattr(mode.plane, field)
            if value is not None:
                arrays[f"m{i}.{field}"] = value
                row["fields"].append(field)
        modes.append(row)
    return arrays, {"codec": CODEC, "snapshot": snapshot.to_dict(), "execution": checkpoint.execution,
            "plane_z_mm": checkpoint.plane_z_mm, "source_id": checkpoint.emission.digest,
            "reference_plane": checkpoint.beam.reference_plane, "modes": modes,
            "checkpoint_digest": checkpoint.digest}


def save_gun_wave_checkpoint(store, manifest, checkpoint):
    arrays, metadata = gun_wave_payload(checkpoint)
    snapshot = manifest.instrument_snapshot
    if snapshot is None or snapshot.digest != checkpoint.snapshot.digest:
        raise ValueError("Gun-wave payload does not match the captured calculation")
    if checkpoint.execution.get("stage_signature") != manifest.calculation_signatures[PRODUCT]:
        raise ValueError("Gun-wave execution does not match the captured incident operator")
    return store.put_array_bundle(manifest, product_key=PRODUCT,
        dependency_signature=manifest.calculation_signatures[PRODUCT], codec=CODEC,
        arrays=arrays, metadata=metadata)


def beam_from_gun_wave_payload(data, arrays):
    """Read numeric modes without restoring/evaluating any live instrument."""
    if data.get("codec") != CODEC:
        raise ValueError("Unsupported gun-wave numeric codec")
    from temsim.optics.electron_gun.effective_source import REFERENCE_ID
    if data.get("reference_plane") != REFERENCE_ID:
        raise ValueError("Gun-wave modes must retain their gun-exit electron reference")
    modes, expected_arrays = [], set()
    for i, row in enumerate(data["modes"]):
        if not set(row["fields"]) <= set(FIELDS):
            raise ValueError("Unknown gun-wave array field")
        values = {name: arrays[f"m{i}.{name}"] for name in row["fields"]}
        expected_arrays.update(f"m{i}.{name}" for name in values)
        modes.append(WaveMode(PlaneWave(**values), row["weight"], data["reference_plane"],
                              row["mode_id"], row["energy_kev"]))
    if expected_arrays != set(arrays):
        raise ValueError("Unreferenced or missing gun-wave array payload")
    return BeamState(tuple(modes), data["reference_plane"])


def gun_wave_from_payload(data, arrays, *, _compatible_emission=None):
    from temsim.instrument_snapshot import decode_instrument
    snapshot = InstrumentSnapshot.from_dict(data["snapshot"])
    if _compatible_emission is None:
        emission = _reconstruct_historical_emission(decode_instrument(snapshot.graph).electron_gun)
    else:
        # Only the cached gun graph is evaluated here, not a former specimen
        # CIF or the user's now-changed live optical controls. Compatibility
        # was established from the current exact operator dependencies below.
        from temsim.instrument_snapshot import decode_instrument
        emission = _reconstruct_historical_emission(decode_instrument(snapshot.graph).electron_gun)
        if emission.digest != _compatible_emission.digest:
            raise ValueError("Cached and requested gun emissions differ")
    if emission.digest != data["source_id"]:
        raise ValueError("Gun emission content differs from the cached execution")
    checkpoint = GunWaveCheckpoint(snapshot, emission, beam_from_gun_wave_payload(data, arrays),
                                   data["plane_z_mm"], data["execution"])
    if checkpoint.digest != data["checkpoint_digest"]:
        raise ValueError("Gun-wave checkpoint content identity changed")
    return checkpoint


def load_gun_wave_checkpoint(store, manifest, *, _prepared=None):
    """Read an exact compatible product and bind a new result if needed."""
    from temsim.instrument_snapshot import decode_instrument
    from temsim.optics.electron_gun.source_policy import require_tip_coherent_source
    require_tip_coherent_source(decode_instrument(manifest.instrument_snapshot.graph).electron_gun)
    prepared = _prepared if _prepared is not None else _prepare_gun_wave_plan(manifest.instrument_snapshot)
    if (prepared.snapshot.digest != manifest.instrument_snapshot.digest
            or prepared.stage_signature != manifest.calculation_signatures[PRODUCT]):
        raise ValueError("Requested gun-wave dependency identity is not current")
    bundle = store.get_array_bundle(manifest, product_key=PRODUCT,
        dependency_signature=manifest.calculation_signatures[PRODUCT], codec=CODEC)
    if bundle is None:
        return None
    data = bundle.metadata
    try:
        if data["execution"].get("stage_signature") != prepared.stage_signature:
            raise ValueError("Cached incident operator differs from its dependency reference")
        parent = gun_wave_from_payload(data, bundle.arrays, _compatible_emission=prepared.emission)
        if parent.snapshot.physical_digest == prepared.snapshot.physical_digest:
            return parent
        if parent.plane_z_mm != float(prepared.plan.z_mm[-1]):
            raise ValueError("A changed incident plane cannot be renamed as a cache hit")
        # New ownership and explicit causal links; the old payload remains
        # byte-for-byte immutable. Its arrays are safe to share read-only.
        execution = dict(parent.execution)
        execution.update({"source_snapshot_id": prepared.snapshot.digest,
            "specimen_centre_z_mm": float(prepared.working.sample.z_mm),
            "specimen_thickness_nm": float(prepared.working.sample.thickness_nm),
            "executed_snapshot_id": parent.execution.get("executed_snapshot_id", parent.snapshot.digest),
            "reuse": {"kind": "exact-incident-operator", "parent_checkpoint_id": parent.digest,
                      "parent_snapshot_id": parent.snapshot.digest,
                      "stage_signature": prepared.stage_signature}})
        return GunWaveCheckpoint(prepared.snapshot, prepared.emission, parent.beam,
                                  parent.plane_z_mm, execution)
    except (ValueError, KeyError, TypeError) as error:
        raise ArtifactIntegrityError(f"Invalid gun-wave cache: {error}") from error


def cached_gun_wave_checkpoint(state, store, *, cancelled=lambda: False, progress_callback=None):
    """Return (checkpoint, reused); cancellation never publishes a partial result."""
    if cancelled():
        raise InterruptedError("Gun-wave request cancelled")
    manifest, prepared = _capture_gun_wave_request(state)
    checkpoint = load_gun_wave_checkpoint(store, manifest, _prepared=prepared)
    if cancelled():
        raise InterruptedError("Gun-wave request cancelled during cache loading")
    if checkpoint is not None:
        if progress_callback is not None:
            progress_callback(1, 1, "Gun coherent checkpoint: exact incident-operator cache hit")
        return checkpoint, True
    # Restore captured inputs, so subsequent live edits cannot affect this job.
    checkpoint = _execute_gun_wave_plan(prepared, cancelled=cancelled, progress_callback=progress_callback)
    if cancelled():
        raise InterruptedError("Gun-wave request cancelled before cache publication")
    save_gun_wave_checkpoint(store, manifest, checkpoint)
    return checkpoint, False
