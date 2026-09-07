"""Checksum-verified persistent storage for numeric calculation artifacts.

The store never deserializes Python objects.  Artifacts are restricted to
JSON metadata plus non-object NumPy arrays loaded with ``allow_pickle=False``.
Logical content is addressed by SHA-256; a separate atomic reference maps one
dependency/solver identity to that immutable content object.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import errno
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from threading import Lock, RLock
import time
from types import MappingProxyType

import numpy as np

from temsim.calculation_manifest import (
    CalculationManifest,
    assert_external_inputs_unchanged,
)
from temsim.immutable_json import (
    canonical_json_bytes,
    freeze_json,
    json_digest,
    thaw_json,
)
from temsim.physics.core import PropagationCheckpoints
from temsim.physics.core import AxialPropagationPlan
from temsim.physics.lens_field_provider import (
    CoordinateRegistration, FieldMapProvenance, FrozenMappedField, MagneticFieldMap,
)
from temsim.physics.simulation import Branch, Simulation
from temsim.optics.electron_gun.base import GunExitBundle, GunTraceResult


ARTIFACT_STORE_SCHEMA_VERSION = 1
_ARRAY_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,95}$")
_ARRAY_FILENAME = re.compile(r"^array-[0-9]{4}\.npy$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STORE_LOCKS_GUARD = Lock()
_STORE_LOCKS: dict[str, "_StoreMutex"] = {}
_INCIDENT_PLAN_ARRAY_FIELDS = (
    "z_mm",
    "step_m",
    "magnetic_t",
    "sx_m2",
    "sy_m2",
    "hex_normal_m3",
    "hex_skew_m3",
    "midpoint_magnetic_t",
    "midpoint_sx_m2",
    "midpoint_sy_m2",
    "midpoint_hex_normal_m3",
    "midpoint_hex_skew_m3",
    "cs_kick_m3",
    "thin_power_m1",
    "thin_rotation_rad",
    "kick_x_rad",
    "kick_y_rad",
    "save_index",
    "checkpoint_index",
)
_CHECKPOINT_ARRAY_FIELDS = ("z_mm", "x_m", "tx_rad", "y_m", "ty_rad")
_GUN_TRACE_ARRAY_FIELDS = (
    "z_mm",
    "x_m",
    "y_m",
    "tx_rad",
    "ty_rad",
    "blocked_z_mm",
)
_GUN_EXIT_ARRAY_FIELDS = (
    "x_m",
    "y_m",
    "tx_rad",
    "ty_rad",
    "energy_offset_ev",
    "weight",
    "ray_id",
    "alive",
)
_INCIDENT_BRANCH_ARRAY_FIELDS = (
    "z",
    "x",
    "y",
    "tx",
    "ty",
    "alive",
    "blocked_z",
    "energy_offset_ev",
    "ray_weight",
)


class ArtifactStoreError(RuntimeError):
    """Base class for persistent-artifact failures."""


class ArtifactIntegrityError(ArtifactStoreError):
    """Raised when stored bytes do not match their immutable manifest."""


class ArtifactTooLargeError(ArtifactStoreError):
    """Raised when one artifact cannot fit within the configured quota."""


class _StoreMutex:
    """Re-entrant thread mutex backed by one cross-process file lock."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._thread_lock = RLock()
        self._depth = 0
        self._stream = None

    @staticmethod
    def _acquire_os_lock(stream) -> None:
        if os.name == "nt":
            import msvcrt

            stream.seek(0, os.SEEK_END)
            if stream.tell() < 1:
                stream.write(b"\0")
                stream.flush()
                os.fsync(stream.fileno())
            while True:
                try:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    return
                except OSError as exc:
                    if exc.errno not in {
                        errno.EACCES,
                        errno.EAGAIN,
                        errno.EDEADLK,
                    }:
                        raise
                    time.sleep(0.05)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)

    @staticmethod
    def _release_os_lock(stream) -> None:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def held(self):
        self._thread_lock.acquire()
        try:
            if self._depth == 0:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                stream = self.path.open("a+b")
                try:
                    self._acquire_os_lock(stream)
                except BaseException:
                    stream.close()
                    raise
                self._stream = stream
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
                if self._depth == 0:
                    stream = self._stream
                    self._stream = None
                    if stream is not None:
                        try:
                            self._release_os_lock(stream)
                        finally:
                            stream.close()
        finally:
            self._thread_lock.release()


def _shared_store_mutex(root: Path) -> _StoreMutex:
    key = os.path.normcase(str(root))
    with _STORE_LOCKS_GUARD:
        mutex = _STORE_LOCKS.get(key)
        if mutex is None:
            mutex = _StoreMutex(root / ".artifact-store.lock")
            _STORE_LOCKS[key] = mutex
        return mutex


@dataclass(frozen=True, slots=True)
class ArtifactBundle:
    product_key: str
    dependency_signature: str
    manifest_digest: str
    content_digest: str
    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        frozen_arrays: dict[str, np.ndarray] = {}
        for name, values in self.arrays.items():
            array = np.ascontiguousarray(values).copy()
            if array.dtype.hasobject:
                raise TypeError("Persistent artifacts cannot contain objects")
            array.setflags(write=False)
            frozen_arrays[str(name)] = array
        object.__setattr__(
            self, "arrays", MappingProxyType(frozen_arrays)
        )
        object.__setattr__(self, "metadata", freeze_json(self.metadata))


def _prepared_arrays(
    arrays: Mapping[str, object],
) -> dict[str, np.ndarray]:
    if not isinstance(arrays, Mapping) or not arrays:
        raise ValueError("An artifact must contain at least one numeric array")
    prepared: dict[str, np.ndarray] = {}
    for raw_name, values in sorted(arrays.items(), key=lambda item: str(item[0])):
        name = str(raw_name)
        if not _ARRAY_NAME.fullmatch(name) or name.startswith("__"):
            raise ValueError(f"Unsafe artifact array name: {name!r}")
        array = np.ascontiguousarray(values)
        if array.dtype.hasobject:
            raise TypeError(
                f"Artifact array {name!r} has an object-containing dtype"
            )
        prepared[name] = array
    return prepared


def _logical_content_digest(
    *,
    codec: str,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, object],
) -> str:
    digest = sha256()
    digest.update(f"artifact-store-v{ARTIFACT_STORE_SCHEMA_VERSION}".encode())
    digest.update(b"\0")
    digest.update(str(codec).encode("utf-8"))
    digest.update(b"\0")
    digest.update(canonical_json_bytes(metadata))
    for name, array in sorted(arrays.items()):
        digest.update(b"\0array\0")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json_bytes(list(array.shape)))
        digest.update(b"\0")
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class ArtifactStore:
    """Persistent content store with an LRU byte quota.

    The first supported expensive product is the incident propagation
    checkpoint bundle.  ``put_array_bundle`` remains intentionally generic for
    future wave/checkpoint codecs, while still refusing arbitrary objects.
    """

    def __init__(self, root: str | Path, *, quota_bytes: int) -> None:
        self.root = Path(root).expanduser().resolve()
        self.quota_bytes = int(quota_bytes)
        if self.quota_bytes <= 0:
            raise ValueError("Artifact-store quota must be positive")
        self.objects_root = self.root / "objects"
        self.references_root = self.root / "refs"
        self.root.mkdir(parents=True, exist_ok=True)
        self._mutex = _shared_store_mutex(self.root)
        with self._locked():
            self.objects_root.mkdir(parents=True, exist_ok=True)
            self.references_root.mkdir(parents=True, exist_ok=True)
            for pending in self.objects_root.glob(".pending-*"):
                if pending.is_dir():
                    self._managed_remove(pending)
            for pending in self.references_root.glob(".*.tmp"):
                if pending.is_file():
                    self._managed_remove(pending)
            # A process may exit after publishing an object but before its
            # reference. Recover that orphan before the next quota decision.
            self._garbage_collect_objects()

    @contextmanager
    def _locked(self):
        with self._mutex.held():
            yield

    def set_quota_bytes(self, quota_bytes: int) -> None:
        """Set the next-write quota without synchronous scanning or deletion.

        Updating this integer does not touch any reference or content object.
        Writers snapshot the current quota before admitting a new reference.
        A change after that point applies to the next write. Already loaded
        artifacts remain valid.
        """

        if isinstance(quota_bytes, bool) or not isinstance(quota_bytes, int) or quota_bytes <= 0:
            raise ValueError("Artifact-store quota must be a positive integer")
        self.quota_bytes = quota_bytes

    @staticmethod
    def _identity(
        manifest: CalculationManifest,
        product_key: str,
        dependency_signature: str,
        codec: str,
    ) -> str:
        return json_digest({
            "schema_version": ARTIFACT_STORE_SCHEMA_VERSION,
            "product_key": str(product_key),
            "dependency_signature": str(dependency_signature),
            "solver_identity": manifest.solver.digest,
            "codec": str(codec),
        })

    def _object_path(self, content_digest: str) -> Path:
        if not _SHA256.fullmatch(str(content_digest)):
            raise ArtifactIntegrityError("Invalid artifact content digest")
        return self.objects_root / content_digest[:2] / content_digest

    def _reference_path(self, identity: str) -> Path:
        if not _SHA256.fullmatch(str(identity)):
            raise ArtifactIntegrityError("Invalid artifact reference identity")
        return self.references_root / f"{identity}.json"

    def _assert_managed_path(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ArtifactStoreError("Artifact path escapes the store root")
        return resolved

    @staticmethod
    def _write_file_atomic(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    def put_array_bundle(
        self,
        manifest: CalculationManifest,
        *,
        product_key: str,
        dependency_signature: str,
        arrays: Mapping[str, object],
        metadata: Mapping[str, object] | None = None,
        codec: str = "numeric-array-bundle-v1",
    ) -> str:
        """Store one immutable numeric bundle and return its cache identity."""

        with self._locked():
            return self._put_array_bundle_unlocked(
                manifest,
                product_key=product_key,
                dependency_signature=dependency_signature,
                arrays=arrays,
                metadata=metadata,
                codec=codec,
            )

    def _put_array_bundle_unlocked(
        self,
        manifest: CalculationManifest,
        *,
        product_key: str,
        dependency_signature: str,
        arrays: Mapping[str, object],
        metadata: Mapping[str, object] | None = None,
        codec: str = "numeric-array-bundle-v1",
    ) -> str:

        product = str(product_key)
        signature = str(dependency_signature)
        expected = str(manifest.calculation_signatures.get(product, ""))
        if not expected or signature != expected:
            raise ValueError(
                f"Artifact signature does not match manifest product {product!r}"
            )
        assert_external_inputs_unchanged(manifest)
        prepared = _prepared_arrays(arrays)
        frozen_metadata = freeze_json(metadata or {})
        content_digest = _logical_content_digest(
            codec=codec,
            arrays=prepared,
            metadata=frozen_metadata,
        )
        identity = self._identity(manifest, product, signature, codec)
        target = self._object_path(content_digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        created_object = False
        if not target.exists():
            temporary = Path(tempfile.mkdtemp(
                prefix=".pending-", dir=self.objects_root
            ))
            try:
                array_rows: dict[str, object] = {}
                for index, (name, array) in enumerate(sorted(prepared.items())):
                    filename = f"array-{index:04d}.npy"
                    array_path = temporary / filename
                    with array_path.open("wb") as stream:
                        np.save(stream, array, allow_pickle=False)
                        stream.flush()
                        os.fsync(stream.fileno())
                    array_rows[name] = {
                        "file": filename,
                        "dtype": array.dtype.str,
                        "shape": list(array.shape),
                        "sha256": _sha256_file(array_path),
                    }
                object_manifest = {
                    "schema_version": ARTIFACT_STORE_SCHEMA_VERSION,
                    "codec": codec,
                    "content_digest": content_digest,
                    "metadata": thaw_json(frozen_metadata),
                    "arrays": array_rows,
                }
                manifest_path = temporary / "manifest.json"
                with manifest_path.open("wb") as stream:
                    stream.write(canonical_json_bytes(object_manifest))
                    stream.flush()
                    os.fsync(stream.fileno())
                if target.exists():
                    shutil.rmtree(temporary)
                else:
                    os.replace(temporary, target)
                    created_object = True
            except Exception:
                if temporary.exists():
                    shutil.rmtree(temporary)
                raise

        reference = {
            "schema_version": ARTIFACT_STORE_SCHEMA_VERSION,
            "identity": identity,
            "product_key": product,
            "dependency_signature": signature,
            "solver_identity": manifest.solver.digest,
            "geometry_fingerprint": manifest.geometry_fingerprint,
            "manifest_digest": manifest.digest,
            "codec": codec,
            "content_digest": content_digest,
        }
        reference_path = self._reference_path(identity)
        reference_bytes = canonical_json_bytes(reference)
        try:
            # Reject a bundle that cannot fit even after evicting every other
            # reference. Do this before replacing an existing identity: a
            # failed replacement must retain its previous content as well.
            quota_bytes = self.quota_bytes
            if self._minimum_store_size(target, len(reference_bytes)) > quota_bytes:
                raise ArtifactTooLargeError(
                    "Artifact is larger than the persistent cache quota"
                )
            self._write_file_atomic(reference_path, reference_bytes)
        except Exception:
            if created_object:
                self._managed_remove(target)
            raise
        self._prune_to_quota(keep_identity=identity, quota_bytes=quota_bytes)
        if not reference_path.exists():
            raise ArtifactTooLargeError(
                "Artifact is larger than the persistent cache quota"
            )
        return identity

    def get_array_bundle(
        self,
        manifest: CalculationManifest,
        *,
        product_key: str,
        dependency_signature: str,
        codec: str = "numeric-array-bundle-v1",
    ) -> ArtifactBundle | None:
        """Load and checksum one bundle, or return ``None`` on a cache miss."""

        with self._locked():
            return self._get_array_bundle_unlocked(
                manifest,
                product_key=product_key,
                dependency_signature=dependency_signature,
                codec=codec,
            )

    def _get_array_bundle_unlocked(
        self,
        manifest: CalculationManifest,
        *,
        product_key: str,
        dependency_signature: str,
        codec: str = "numeric-array-bundle-v1",
    ) -> ArtifactBundle | None:

        assert_external_inputs_unchanged(manifest)
        identity = self._identity(
            manifest, product_key, dependency_signature, codec
        )
        reference_path = self._reference_path(identity)
        if not reference_path.is_file():
            return None
        try:
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError("Invalid artifact reference") from exc
        if (
            reference.get("identity") != identity
            or reference.get("product_key") != str(product_key)
            or reference.get("dependency_signature")
            != str(dependency_signature)
            or reference.get("solver_identity") != manifest.solver.digest
            or reference.get("codec") != codec
        ):
            raise ArtifactIntegrityError("Artifact reference identity mismatch")
        content_digest = str(reference.get("content_digest", ""))
        object_path = self._object_path(content_digest)
        object_manifest_path = object_path / "manifest.json"
        try:
            object_manifest = json.loads(
                object_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError("Missing artifact object manifest") from exc
        if (
            object_manifest.get("content_digest") != content_digest
            or object_manifest.get("codec") != codec
            or int(object_manifest.get("schema_version", -1))
            != ARTIFACT_STORE_SCHEMA_VERSION
        ):
            raise ArtifactIntegrityError("Artifact object identity mismatch")
        arrays: dict[str, np.ndarray] = {}
        rows = object_manifest.get("arrays")
        if not isinstance(rows, dict) or not rows:
            raise ArtifactIntegrityError("Artifact object has no arrays")
        for name, row in sorted(rows.items()):
            if not isinstance(row, dict) or not _ARRAY_NAME.fullmatch(name):
                raise ArtifactIntegrityError("Invalid artifact array manifest")
            filename = str(row.get("file", ""))
            if not _ARRAY_FILENAME.fullmatch(filename):
                raise ArtifactIntegrityError("Invalid artifact array filename")
            array_path = self._assert_managed_path(object_path / filename)
            if not array_path.is_file() or _sha256_file(array_path) != row.get("sha256"):
                raise ArtifactIntegrityError(
                    f"Artifact array checksum failed: {name}"
                )
            try:
                with array_path.open("rb") as stream:
                    array = np.load(stream, allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise ArtifactIntegrityError(
                    f"Artifact array could not be loaded: {name}"
                ) from exc
            if (
                array.dtype.hasobject
                or array.dtype.str != row.get("dtype")
                or list(array.shape) != row.get("shape")
            ):
                raise ArtifactIntegrityError(
                    f"Artifact array metadata mismatch: {name}"
                )
            arrays[name] = np.ascontiguousarray(array)
        metadata = object_manifest.get("metadata", {})
        actual_digest = _logical_content_digest(
            codec=codec,
            arrays=arrays,
            metadata=metadata,
        )
        if actual_digest != content_digest:
            raise ArtifactIntegrityError("Artifact logical checksum failed")
        os.utime(reference_path, None)
        return ArtifactBundle(
            product_key=str(product_key),
            dependency_signature=str(dependency_signature),
            manifest_digest=str(reference.get("manifest_digest", "")),
            content_digest=content_digest,
            arrays=arrays,
            metadata=metadata,
        )

    def put_propagation_checkpoints(
        self,
        manifest: CalculationManifest,
        checkpoints: PropagationCheckpoints,
    ) -> str:
        return self.put_array_bundle(
            manifest,
            product_key="incident",
            dependency_signature=str(
                manifest.calculation_signatures["incident"]
            ),
            codec="incident-propagation-checkpoints-v1",
            arrays={
                "z_mm": checkpoints.z_mm,
                "x_m": checkpoints.x_m,
                "tx_rad": checkpoints.tx_rad,
                "y_m": checkpoints.y_m,
                "ty_rad": checkpoints.ty_rad,
            },
            metadata={
                "coordinate_system": "column-z-downstream",
                "units": {
                    "z_mm": "mm",
                    "x_m": "m",
                    "tx_rad": "rad",
                    "y_m": "m",
                    "ty_rad": "rad",
                },
            },
        )

    def get_propagation_checkpoints(
        self, manifest: CalculationManifest
    ) -> PropagationCheckpoints | None:
        bundle = self.get_array_bundle(
            manifest,
            product_key="incident",
            dependency_signature=str(
                manifest.calculation_signatures["incident"]
            ),
            codec="incident-propagation-checkpoints-v1",
        )
        if bundle is None:
            return None
        required = {"z_mm", "x_m", "tx_rad", "y_m", "ty_rad"}
        if set(bundle.arrays) != required:
            raise ArtifactIntegrityError(
                "Incident checkpoint artifact has unexpected arrays"
            )
        z = bundle.arrays["z_mm"]
        shape = bundle.arrays["x_m"].shape
        if z.ndim != 1 or len(shape) != 2 or shape[0] != z.size:
            raise ArtifactIntegrityError("Incident checkpoint shape is invalid")
        if any(bundle.arrays[name].shape != shape for name in required - {"z_mm"}):
            raise ArtifactIntegrityError(
                "Incident checkpoint phase-space arrays do not align"
            )
        return PropagationCheckpoints(
            z_mm=z,
            x_m=bundle.arrays["x_m"],
            tx_rad=bundle.arrays["tx_rad"],
            y_m=bundle.arrays["y_m"],
            ty_rad=bundle.arrays["ty_rad"],
        )

    def put_incident_simulation_seed(
        self,
        manifest: CalculationManifest,
        simulation: Simulation,
    ) -> str:
        """Persist the complete numeric seed required by ``simulation.run``.

        Checkpoints alone cannot restart the current solver: safe reuse also
        requires the propagation plan, gun exit phase space and the retained
        incident history. This codec stores precisely those values and no
        arbitrary Python object graph.
        """

        plan = getattr(simulation, "incident_plan", None)
        checkpoints = getattr(simulation, "incident_checkpoints", None)
        gun_trace = getattr(simulation, "gun_trace", None)
        incident = getattr(simulation, "incident", None)
        if any(
            item is None
            for item in (plan, checkpoints, gun_trace, incident)
        ):
            raise ValueError("Simulation has no complete incident restart seed")
        exit_bundle = getattr(gun_trace, "exit_bundle", None)
        if exit_bundle is None:
            raise ValueError("Incident restart seed has no gun exit bundle")
        arrays: dict[str, object] = {}
        map_metadata = []
        for index, item in enumerate(plan.mapped_fields):
            field_map = item.field_map
            for number, value in enumerate(field_map.axes_m):
                arrays[f"map{index}.axis{number}"] = value
            for number, value in enumerate(field_map.components_t):
                arrays[f"map{index}.component{number}"] = value
            map_metadata.append({
                "lens_key": item.lens_key, "scale": item.scale,
                "map_type": field_map.map_type,
                "geometry_fingerprint": field_map.geometry_fingerprint,
                "reference_excitation_percent": field_map.reference_excitation_percent,
                "reference_polarity": field_map.reference_polarity,
                "origin_global_m": field_map.registration.origin_global_m,
                "rotation_local_to_global": field_map.registration.rotation_local_to_global,
                "divergence_tolerance": field_map.divergence_tolerance,
                "provenance": {name: getattr(field_map.provenance, name)
                               for name in ("kind", "source_path", "source_sha256", "source_note")},
            })
        for field in _INCIDENT_PLAN_ARRAY_FIELDS:
            arrays[f"plan.{field}"] = getattr(plan, field)
        for field in _CHECKPOINT_ARRAY_FIELDS:
            arrays[f"checkpoint.{field}"] = getattr(checkpoints, field)
        for field in _GUN_TRACE_ARRAY_FIELDS:
            arrays[f"gun.{field}"] = getattr(gun_trace, field)
        for field in _GUN_EXIT_ARRAY_FIELDS:
            arrays[f"gun_exit.{field}"] = getattr(exit_bundle, field)
        for field in _INCIDENT_BRANCH_ARRAY_FIELDS:
            value = getattr(incident, field, None)
            if value is not None:
                arrays[f"incident.{field}"] = value
        return self.put_array_bundle(
            manifest,
            product_key="incident",
            dependency_signature=str(
                manifest.calculation_signatures["incident"]
            ),
            codec="incident-simulation-seed-v1",
            arrays=arrays,
            metadata={
                "coordinate_system": "column-z-downstream",
                "plan_solver_signature": str(plan.solver_signature),
                "plan_signature": str(plan.signature),
                "plan_mapped_fields": map_metadata,
                "gun_blocked_key": list(gun_trace.blocked_key),
                "gun_scalars": {
                    name: getattr(gun_trace, name)
                    for name in (
                        "emitted_current_a",
                        "dpa_transmitted_current_a",
                        "c1_transmitted_current_a",
                        "monochromator_transmitted_current_a",
                        "output_energy_fwhm_ev",
                        "slit_dispersion_um_per_ev",
                    )
                },
                "incident_name": str(incident.name),
                "incident_colour": list(incident.colour),
                "incident_blocked_key": list(incident.blocked_key),
                "incident_weight": float(incident.weight),
                "incident_interaction_kind": str(
                    incident.interaction_kind
                ),
            },
        )

    def get_incident_simulation_seed(
        self,
        manifest: CalculationManifest,
    ) -> Simulation | None:
        """Restore a checksum-verified incident seed for ``simulation.run``."""

        bundle = self.get_array_bundle(
            manifest,
            product_key="incident",
            dependency_signature=str(
                manifest.calculation_signatures["incident"]
            ),
            codec="incident-simulation-seed-v1",
        )
        if bundle is None:
            return None
        arrays = bundle.arrays

        def fields(prefix: str, names: tuple[str, ...]) -> dict[str, np.ndarray]:
            try:
                return {
                    name: arrays[f"{prefix}.{name}"] for name in names
                }
            except KeyError as exc:
                raise ArtifactIntegrityError(
                    f"Incident seed is missing {exc.args[0]}"
                ) from exc

        metadata = thaw_json(bundle.metadata)
        try:
            mapped_fields = []
            for index, row in enumerate(metadata.get("plan_mapped_fields", ())):
                dimensions = 2 if row["map_type"] == "axisymmetric_rz" else 3
                field_map = MagneticFieldMap(
                    map_type=row["map_type"],
                    axes_m=tuple(arrays[f"map{index}.axis{n}"] for n in range(dimensions)),
                    components_t=tuple(arrays[f"map{index}.component{n}"] for n in range(dimensions)),
                    registration=CoordinateRegistration(
                        origin_global_m=tuple(row["origin_global_m"]),
                        rotation_local_to_global=tuple(tuple(v) for v in row["rotation_local_to_global"])),
                    geometry_fingerprint=row["geometry_fingerprint"],
                    reference_excitation_percent=row["reference_excitation_percent"],
                    reference_polarity=row["reference_polarity"],
                    divergence_tolerance=row["divergence_tolerance"],
                    provenance=FieldMapProvenance(**row["provenance"]),
                )
                mapped_fields.append(FrozenMappedField(row["lens_key"], field_map, row["scale"]))
            plan = AxialPropagationPlan(
                **fields("plan", _INCIDENT_PLAN_ARRAY_FIELDS),
                solver_signature=str(metadata["plan_solver_signature"]),
                signature=str(metadata["plan_signature"]),
                mapped_fields=tuple(mapped_fields),
            )
            checkpoints = PropagationCheckpoints(
                **fields("checkpoint", _CHECKPOINT_ARRAY_FIELDS)
            )
            exit_bundle = GunExitBundle(
                **fields("gun_exit", _GUN_EXIT_ARRAY_FIELDS)
            )
            gun_scalars = dict(metadata["gun_scalars"])
            gun_trace = GunTraceResult(
                **fields("gun", _GUN_TRACE_ARRAY_FIELDS),
                exit_bundle=exit_bundle,
                blocked_key=tuple(metadata["gun_blocked_key"]),
                **gun_scalars,
            )
            incident_arrays = fields(
                "incident",
                tuple(
                    name
                    for name in _INCIDENT_BRANCH_ARRAY_FIELDS
                    if f"incident.{name}" in arrays
                ),
            )
            incident = Branch(
                name=str(metadata["incident_name"]),
                colour=tuple(metadata["incident_colour"]),
                blocked_key=list(metadata["incident_blocked_key"]),
                weight=float(metadata["incident_weight"]),
                interaction_kind=str(
                    metadata["incident_interaction_kind"]
                ),
                **incident_arrays,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "Incident seed metadata is invalid"
            ) from exc
        return Simulation(
            incident=incident,
            branches={},
            metrics={"persistent_incident_seed": True},
            gun_trace=gun_trace,
            incident_plan=plan,
            incident_checkpoints=checkpoints,
        )

    def _managed_remove(self, path: Path) -> None:
        target = self._assert_managed_path(path)
        if target.is_dir():
            shutil.rmtree(target)
        else:
            try:
                target.unlink()
            except FileNotFoundError:
                pass

    def _reference_rows(self) -> list[tuple[int, Path, str]]:
        with self._locked():
            rows = []
            for path in self.references_root.glob("*.json"):
                try:
                    document = json.loads(path.read_text(encoding="utf-8"))
                    content = str(document["content_digest"])
                    if not _SHA256.fullmatch(content):
                        raise ValueError("invalid content digest")
                    stamp = path.stat().st_mtime_ns
                except (
                    OSError,
                    KeyError,
                    UnicodeError,
                    ValueError,
                    json.JSONDecodeError,
                ):
                    self._managed_remove(path)
                    continue
                rows.append((stamp, path, content))
            return sorted(rows, key=lambda row: (row[0], row[1].name))

    def _store_size(self) -> int:
        return sum(
            path.stat().st_size
            for path in self.root.rglob("*")
            if path.is_file()
        )

    def _minimum_store_size(self, keep_object: Path, reference_bytes: int) -> int:
        """Size after keeping just the proposed reference and its content.

        Count actual array headers, object/reference manifests and fixed
        overhead such as the OS lock file. Subtract only files the normal
        eviction/garbage collection can remove; shared content is counted
        once, even when replacing an existing reference to that same object.
        This is a read-only admission check, including for a reduced quota.
        """

        size = self._store_size() + reference_bytes
        for path in self.references_root.glob("*.json"):
            if path.is_file():
                size -= path.stat().st_size
        for prefix in self.objects_root.iterdir():
            if not prefix.is_dir() or prefix.name.startswith(".pending-"):
                continue
            for object_path in prefix.iterdir():
                if object_path.is_dir() and object_path != keep_object:
                    size -= sum(
                        path.stat().st_size
                        for path in object_path.rglob("*")
                        if path.is_file()
                    )
        return size

    def _garbage_collect_objects(self) -> None:
        with self._locked():
            referenced = {row[2] for row in self._reference_rows()}
            for prefix in self.objects_root.iterdir():
                if not prefix.is_dir() or prefix.name.startswith(".pending-"):
                    continue
                for object_path in prefix.iterdir():
                    if (
                        object_path.is_dir()
                        and object_path.name not in referenced
                    ):
                        self._managed_remove(object_path)
                if prefix.exists() and not any(prefix.iterdir()):
                    prefix.rmdir()

    def _prune_to_quota(self, *, keep_identity: str, quota_bytes: int) -> None:
        with self._locked():
            # Reclaim objects left between the object and reference commits
            # before deciding whether the newly committed artifact fits.
            self._garbage_collect_objects()
            rows = self._reference_rows()
            keep_path = self._reference_path(keep_identity)
            for _stamp, path, _content in rows:
                if self._store_size() <= quota_bytes:
                    break
                if path != keep_path:
                    self._managed_remove(path)
                    self._garbage_collect_objects()
            if self._store_size() > quota_bytes:
                self._managed_remove(keep_path)
                self._garbage_collect_objects()


__all__ = (
    "ARTIFACT_STORE_SCHEMA_VERSION",
    "ArtifactBundle",
    "ArtifactIntegrityError",
    "ArtifactStore",
    "ArtifactStoreError",
    "ArtifactTooLargeError",
)
