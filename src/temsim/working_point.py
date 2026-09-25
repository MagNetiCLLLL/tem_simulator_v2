"""Immutable, inspectable working points over the existing artifact store.

Viewing and diagnostic evaluation do not restore a State or evaluate a lens.
Packages contain plain JSON and numeric NPY arrays, never executable pickle.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import Mapping
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from temsim.immutable_json import freeze_json, json_digest, thaw_json
from temsim.instrument_snapshot import InstrumentSnapshot
from temsim.physics.wave_observables import WAVE_DEFINITIONS

PACKAGE_SCHEMA = "working-point-package-v1"
OBSERVABLE_DEFINITIONS = freeze_json({
    **WAVE_DEFINITIONS,
    "alpha95": {"unit": "rad", "definition_id": "chief-ray-current-contained-semiangle-95-v1"},
    "alpha99": {"unit": "rad", "definition_id": "chief-ray-current-contained-semiangle-99-v1"},
    "physical_angular_edge": {"unit": "rad", "definition_id": "physical-pupil-support-edge-v1"},
    "sampled_max_angle": {"unit": "rad", "definition_id": "chief-ray-finite-sample-maximum-v1"},
    "radius95": {"unit": "m", "definition_id": "chief-ray-current-contained-radius-95-v1"},
    "centre_x": {"unit": "m", "definition_id": "current-weighted-centre-x-v1"},
    "centre_y": {"unit": "m", "definition_id": "current-weighted-centre-y-v1"},
    "source_fraction": {"unit": "1", "definition_id": "incident-surviving-current/source-current-v1"},
})


def _frozen_array(value):
    a = np.asarray(value)
    if a.dtype.kind not in "biufc" or a.dtype.fields is not None:
        raise ValueError("Working-point payloads require plain numeric arrays")
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _array_identity(a):
    # Hash the contiguous buffer directly; avoid a whole-history bytes copy.
    return {"shape": list(a.shape), "dtype": a.dtype.str,
            "sha256": sha256(np.ascontiguousarray(a)).hexdigest()}


@dataclass(frozen=True)
class ObservableRecord:
    observable_id: str
    value: object
    unit: str
    definition_id: str
    plane_id: str
    checkpoint_id: str
    status: str
    estimator_version: str = "weighted-incident-observables-v1"
    reason: str = ""

    def __post_init__(self):
        object.__setattr__(self, "value", freeze_json(self.value))
        if self.status not in {"AVAILABLE", "UNAVAILABLE", "NOT_COMPUTED", "OUT_OF_VALIDATED_RANGE"}:
            raise ValueError("Unknown observable availability")


class CheckpointObservables:
    """Lazy pure derived products; the checkpoint and its arrays never change."""
    def __init__(self, checkpoint):
        self._checkpoint = checkpoint
        self._wave_beam = None
        self._ray_records = None

    def get(self, observable_id: str) -> ObservableRecord:
        cp = self._checkpoint
        definition = OBSERVABLE_DEFINITIONS.get(observable_id)
        if definition is None:
            return ObservableRecord(observable_id, None, "", "unregistered", cp.plane_id,
                                    cp.digest, "UNAVAILABLE", reason="Observable is not registered")
        if observable_id in WAVE_DEFINITIONS:
            payload = cp.metadata.get("gun_wave")
            if payload is None:
                value, status, reason = None, "NOT_COMPUTED", "No retained coherent-mode payload"
            else:
                from temsim.physics.gun_wave_cache import beam_from_gun_wave_payload
                from temsim.physics.wave_observables import mixed_wave_observable
                if self._wave_beam is None:
                    self._wave_beam = beam_from_gun_wave_payload(payload, cp.arrays)
                value, status, reason = mixed_wave_observable(self._wave_beam, observable_id)
            return ObservableRecord(observable_id, value, definition["unit"], definition["definition_id"],
                                    cp.plane_id, cp.digest, status, "mixed-wave-observables-v1", reason)
        required = {"x_m", "y_m", "tx_rad", "ty_rad", "weight", "alive"}
        if not required <= cp.arrays.keys():
            return ObservableRecord(observable_id, None, definition["unit"], definition["definition_id"],
                                    cp.plane_id, cp.digest, "NOT_COMPUTED", reason="No retained ray payload")
        from temsim.physics.core import PropagationCheckpoints
        from temsim.checkpoint_observables import incident_checkpoint_observables
        a = cp.arrays
        planes = PropagationCheckpoints(np.array([cp.plane_z_mm]),
            a["x_m"][None], a["tx_rad"][None], a["y_m"][None], a["ty_rad"][None])
        if self._ray_records is None:
            self._ray_records = incident_checkpoint_observables(planes, alive=a["alive"], weights=a["weight"])["records"]
        row = self._ray_records[observable_id]
        if row["definition_id"] != definition["definition_id"]:
            raise ValueError("Observable estimator definition does not match the registry")
        return ObservableRecord(observable_id, row["value"], row["unit"], row["definition_id"],
                                cp.plane_id, cp.digest, row["status"], reason=row["reason"])


@dataclass(frozen=True)
class WorkingPointCheckpoint:
    snapshot: InstrumentSnapshot
    arrays: Mapping
    plane_z_mm: float
    stage_signature: str
    metadata: Mapping
    parent_id: str | None = None

    def __post_init__(self):
        if not np.isfinite(self.plane_z_mm) or not self.stage_signature:
            raise ValueError("Checkpoint needs a physical reference plane and local dependency signature")
        object.__setattr__(self, "arrays", MappingProxyType({str(k): _frozen_array(v) for k, v in self.arrays.items()}))
        object.__setattr__(self, "metadata", freeze_json(self.metadata))
        rays = {"x_m", "y_m", "tx_rad", "ty_rad", "weight", "alive"}
        if rays & self.arrays.keys():
            if not rays <= self.arrays.keys():
                raise ValueError("Incomplete retained ray population")
            shape = self.arrays["x_m"].shape
            if len(shape) != 1 or any(self.arrays[key].shape != shape for key in rays):
                raise ValueError("Retained ray arrays must have matching one-dimensional shapes")
            if self.arrays["alive"].dtype.kind != "b":
                raise ValueError("Ray survival must be an explicit boolean array")

    @property
    def plane_id(self):
        return "column-z-mm:" + float(self.plane_z_mm).hex()

    @cached_property
    def _array_identities(self):
        # Arrays own immutable bytes, so content identities can safely serve
        # both the package digest and its manifest without another full scan.
        return freeze_json({k: _array_identity(v) for k, v in self.arrays.items()})

    @cached_property
    def payload_hash(self):
        return json_digest(self._array_identities)

    @cached_property
    def digest(self):
        return json_digest({"snapshot": self.snapshot.digest, "plane": self.plane_id,
                            "signature": self.stage_signature, "payload": self.payload_hash,
                            "metadata": self.metadata, "parent": self.parent_id})

    @property
    def observables(self):
        return CheckpointObservables(self)

    @classmethod
    def from_result(cls, result, *, parent_id=None):
        manifest = getattr(result, "calculation_manifest", None)
        if manifest is None or manifest.instrument_snapshot is None:
            raise ValueError("Historical result has no complete working point; it cannot be upgraded implicitly")
        simulation = result.simulation
        incident = simulation.incident
        gun = getattr(simulation, "gun_trace", None)
        if gun is None:
            raise ValueError("No retained gun execution accompanies this result")
        from temsim.instrument_snapshot import decode_instrument
        from temsim.physics.beam_current import effective_source_current_a
        captured = decode_instrument(manifest.instrument_snapshot.graph)
        exact = getattr(simulation, "incident_checkpoints", None)
        plane = float(captured.sample.z_mm)
        matches = np.flatnonzero(np.asarray(exact.z_mm) == plane) if exact is not None else []
        if len(matches) != 1:
            raise ValueError("No unique full-precision incident checkpoint at the result plane; display history is not a scientific checkpoint")
        arrays = {name: np.asarray(getattr(exact, name)[matches[0]])
                  for name in ("x_m", "y_m", "tx_rad", "ty_rad")}
        arrays.update(weight=incident.ray_weight, alive=incident.alive,
                      energy_offset_ev=incident.energy_offset_ev,
                      gun_ray_id=gun.exit_bundle.ray_id)
        return cls(manifest.instrument_snapshot, arrays, plane,
                   str(result.signatures["incident"]),
                   {"source_representation": "gun-derived-particles",
                    "quality": getattr(captured, "_tuning_quality", "Configured numerical settings"),
                    "source_current_a": effective_source_current_a(captured),
                    "created_at_utc": manifest.created_at_utc,
                    "coordinate_precision": "exact-integration-checkpoint",
                    "phase_status": "NOT_COMPUTED", "validation_status": "NOT_RUN",
                    "manifest_id": manifest.digest, "solver": manifest.solver.source_digest}, parent_id)

    @classmethod
    def from_gun_wave(cls, checkpoint, *, parent_id=None):
        from temsim.physics.gun_wave_cache import gun_wave_payload
        arrays, payload = gun_wave_payload(checkpoint)
        payload.pop("snapshot")  # The package already owns this exact snapshot.
        # The complete snapshot identifies ownership; the stage signature
        # identifies the actually consumed operator. They are not synonyms.
        # Keep historical payloads without a local signature conservative.
        signature = checkpoint.execution.get("stage_signature", checkpoint.snapshot.physical_digest)
        return cls(checkpoint.snapshot, arrays, checkpoint.plane_z_mm, signature,
            {"source_representation": "gun-derived-coherent-modes", "phase_status": "RETAINED",
             "validation_status": checkpoint.execution["validation_status"],
             "gun_wave": payload}, parent_id)

    def gun_wave_checkpoint(self):
        """Explicit compatible continuation; not used by read-only browsing."""
        from temsim.physics.gun_wave_cache import gun_wave_from_payload
        payload = self.metadata.get("gun_wave")
        if payload is None:
            raise ValueError("This working point has no retained gun-wave modes")
        checkpoint = gun_wave_from_payload({**payload, "snapshot": self.snapshot.to_dict()}, self.arrays)
        if checkpoint.snapshot.digest != self.snapshot.digest or checkpoint.plane_z_mm != self.plane_z_mm:
            raise ValueError("Gun-wave payload and working-point identity differ")
        return checkpoint

    def compatible_state(self):
        """Explicit full restore; never called for table/observable display."""
        if self.is_metadata_only:
            raise ValueError("Metadata-only record: input assets and results are absent; restoration is unavailable")
        if self.is_input_design:
            # No old transport is accepted. Every observable requires execution
            # with the current solver; the archived identity is not relabelled.
            from temsim.input_design import restore_input_design
            return restore_input_design(self.snapshot)
        return self.snapshot.restore()

    @property
    def is_input_design(self):
        return self.metadata.get("package_kind") == "INSTRUMENT_INPUTS_ONLY" and not self.arrays

    @property
    def is_metadata_only(self):
        from temsim.working_point_export import METADATA_SCHEMA
        return self.metadata.get("package_kind") == "METADATA_ONLY" or self.snapshot.graph.get("schema") == METADATA_SCHEMA

    @property
    def has_retained_payload(self):
        return bool(self.arrays)

    def write_package(self, path, *, overwrite=False, evidence=(), mode=None,
                      compression=ZIP_DEFLATED, compresslevel=None, maximum_unpacked_bytes=8*1024**3):
        if mode is not None:
            from temsim.working_point_export import export_checkpoint
            self = export_checkpoint(self, mode)
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError("Working-point export already exists")
        document = {"schema": PACKAGE_SCHEMA, "snapshot": self.snapshot.to_dict(),
                    "plane_z_mm": self.plane_z_mm, "stage_signature": self.stage_signature,
                    "metadata": thaw_json(self.metadata), "parent_id": self.parent_id,
                    "arrays": {}, "digest": self.digest}
        from temsim.sampling_diagnostics import checkpoint_sampling_summary
        document["index_summary"] = thaw_json(checkpoint_sampling_summary(self))
        document["evidence"] = thaw_json(freeze_json(evidence))
        for index, (key, value) in enumerate(sorted(self.arrays.items())):
            document["arrays"][key] = {"entry": f"arrays/{index}.npy", **thaw_json(self._array_identities[key])}
        from temsim.working_point_archive import prepare_manifest
        manifest_bytes = prepare_manifest(document, self.arrays,
                                         maximum_unpacked_bytes=maximum_unpacked_bytes)
        with NamedTemporaryFile(dir=path.parent, prefix=".working-point-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            with ZipFile(temporary, "w", compression=compression, compresslevel=compresslevel) as archive:
                for index, (key, value) in enumerate(sorted(self.arrays.items())):
                    entry = f"arrays/{index}.npy"
                    # Stream NPY chunks into the atomic package. Large particle
                    # histories must not acquire a second whole-array BytesIO
                    # allocation before being written to disk.
                    with archive.open(entry, "w", force_zip64=True) as stream:
                        np.save(stream, value, allow_pickle=False)
                archive.writestr("manifest.json", manifest_bytes)
            if path.exists() and not overwrite:
                raise FileExistsError("Working-point export already exists")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def read_package(cls, path, *, maximum_unpacked_bytes=8*1024**3):
        from temsim.working_point_archive import read_manifest, read_numeric_entry
        with ZipFile(path) as archive:
            data, snapshot = read_manifest(archive, maximum_unpacked_bytes=maximum_unpacked_bytes)
            arrays = {}
            for key, record in data["arrays"].items():
                arrays[key] = read_numeric_entry(archive, record, maximum_unpacked_bytes=maximum_unpacked_bytes)
        result = cls(snapshot, arrays, data["plane_z_mm"],
                     data["stage_signature"], data["metadata"], data["parent_id"])
        # Check the actual immutable payload, never seed this cache from the
        # manifest. The same verified identities supply the package digest.
        for key, record in data["arrays"].items():
            if thaw_json(result._array_identities[key]) != {k: record[k] for k in ("shape", "dtype", "sha256")}:
                raise ValueError("Working-point numeric content checksum mismatch")
        if result.digest != data["digest"]:
            raise ValueError("Working-point package checksum mismatch")
        return result



# This adapter uses the existing package manifest and deferred NPY products.
from temsim.working_point_archive import WorkingPointArchiveIndex


def component_rows(snapshot: InstrumentSnapshot):
    """Raw captured parameters for a read-only tree; never call a live getter."""
    for node in snapshot.graph["nodes"]:
        attrs = node.get("attributes", {})
        key = attrs.get("key")
        if isinstance(key, str):
            yield key, node["type"].split(":")[-1], attrs


def snapshot_changes(before: InstrumentSnapshot, after: InstrumentSnapshot):
    """Exact graph differences, including disabled components and model data."""
    changes = []
    def visit(left, right, path):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(left.keys() | right.keys()):
                if key not in left or key not in right:
                    changes.append((path + "/" + key, left.get(key), right.get(key)))
                else:
                    visit(left[key], right[key], path + "/" + key)
        elif isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
            for index, (old, new) in enumerate(zip(left, right)):
                visit(old, new, path + "/" + str(index))
        elif left != right:
            changes.append((path, left, right))
    visit(thaw_json(freeze_json(before.identity_payload())),
          thaw_json(freeze_json(after.identity_payload())), "")
    return tuple(changes)
