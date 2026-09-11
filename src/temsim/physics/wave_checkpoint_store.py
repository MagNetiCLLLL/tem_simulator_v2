"""Private executed-checkpoint store with streaming arrays and bound dependencies.

Only the pipeline constructs lookup keys from captured physical inputs. This
is not an import API for a user-supplied beam. No pickle or executable codec.
Incomplete writes are never indexed; immutable mode buffers load one at a time.
"""
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np

from temsim.immutable_json import freeze_json, thaw_json, json_digest
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.wave_flux import WaveMode
from temsim.physics.wave_reference import AxialWaveReference


def _hash_file(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class _StoredModes(Sequence):
    root: Path
    rows: tuple
    reference: str

    def __post_init__(self):
        object.__setattr__(self, "rows", freeze_json(self.rows))

    def __len__(self):
        return len(self.rows)

    def _array(self, item):
        path = self.root/item["file"]
        if path.parent != self.root or path.is_symlink() or _hash_file(path) != item["sha256"]:
            raise ValueError("Executed wave cache array checksum/path changed")
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def geometries(self):
        """Plan a covering domain without reading every large amplitude."""
        for row in self.rows:
            arrays = row["arrays"]
            yield tuple(arrays["amplitude"]["shape"]), self._array(arrays["basis_m"]), self._array(arrays["origin_m"])

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise TypeError("Read stored modes one at a time, without materialising a slice")
        row = self.rows[index]
        values = {}
        for field, item in row["arrays"].items():
            # mmap avoids a second heap-sized input; PlaneWave owns the final
            # immutable bytes so later external writes cannot mutate a mode.
            values[field] = self._array(item)
        reference = None if row["axial_reference"] is None else AxialWaveReference(**row["axial_reference"])
        return WaveMode(PlaneWave(**values), row["weight"], self.reference, row["mode_id"], row["energy_kev"],
                        reference, tuple(row.get("scattering_history", ())))


@dataclass(frozen=True)
class _StoredBeam:
    modes: _StoredModes
    reference_plane: str
    content_identity: str

    @property
    def total_weight(self):
        return sum(r["weight"] for r in self.modes.rows)

    @property
    def resident_bytes(self):
        return 0


class _ModeWriter:
    def __init__(self, store, key, reference):
        self.store, self.key, self.reference = store, key, reference
        self.directory = store.root/("pending-"+uuid4().hex)
        self.directory.mkdir()
        self.rows, self.bytes = [], 0
        self.mode_ids = set()

    def append(self, mode):
        if mode.reference_plane != self.reference:
            raise ValueError("Stored modes must retain their original electron reference")
        if mode.mode_id in self.mode_ids:
            raise ValueError("Duplicate stored wave mode")
        required = sum(getattr(mode.plane, n).nbytes+256 for n in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad")
                       if getattr(mode.plane, n) is not None)
        self.store.check_space(required)
        row = {"mode_id": mode.mode_id, "weight": mode.weight_per_reference_electron, "energy_kev": mode.energy_kev,
               "axial_reference": None if mode.axial_reference is None else asdict(mode.axial_reference),
               "scattering_history": thaw_json(mode.scattering_history), "arrays": {}}
        for name in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
            value = getattr(mode.plane, name)
            if value is None:
                continue
            path = self.directory/f"m{len(self.rows)}_{name}.npy"
            np.save(path, value, allow_pickle=False)
            row["arrays"][name] = {"file": path.name, "sha256": _hash_file(path), "shape": value.shape,
                                   "nbytes": value.nbytes}
            self.bytes += path.stat().st_size
            self.store._used_bytes += path.stat().st_size
        self.rows.append(row)
        self.mode_ids.add(mode.mode_id)

    def finish(self, z_mm, current_a, record):
        if not self.rows:
            raise ValueError("An executed checkpoint needs at least one mode")
        data = {"schema": "executed-tip-wave-disk-v1", "dependency": self.store.dependency, "key": self.key,
                "reference_plane": self.reference, "plane_z_mm": float(z_mm), "current_a": current_a,
                "modes": self.rows, "record": thaw_json(freeze_json(record))}
        data["manifest_digest"] = json_digest(data)
        path = self.directory/"manifest.json"
        payload = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.store.check_space(len(payload)+512)  # manifest and atomic index
        path.write_bytes(payload)
        self.store._used_bytes += len(payload)
        destination = self.store.root/(self.key+"-"+uuid4().hex)
        self.directory.rename(destination)
        # Index commit is atomic; readers never see a partially written mode.
        index = self.store.root/(self.key+".json")
        temporary = index.with_name("index-"+uuid4().hex+".json")
        temporary.write_text(json.dumps({"directory": destination.name, "digest": data["manifest_digest"]}), encoding="utf-8")
        self.store._used_bytes += temporary.stat().st_size-(index.stat().st_size if index.exists() else 0)
        os.replace(temporary, index)
        return self.store.get(self.key)

    def abort(self):
        # Delete only this writer's verified private pending directory.
        if self.directory.parent.resolve() != self.store.root.resolve() or not self.directory.name.startswith("pending-"):
            raise ValueError("Invalid private cache cleanup path")
        if self.directory.exists():
            shutil.rmtree(self.directory)
            # A failed np.save may have left bytes before its accounting update.
            self.store._used_bytes = sum(p.stat().st_size for p in self.store.root.rglob("*") if p.is_file())


class ExecutedWaveStore:
    def __init__(self, directory, dependency, maximum_bytes):
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.dependency, self.maximum_bytes = dependency, maximum_bytes
        self._used_bytes = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

    def key(self, *parts):
        return json_digest(("executed-tip-wave-disk-v1", self.dependency, parts))

    def check_space(self, required):
        if self._used_bytes+required > self.maximum_bytes or shutil.disk_usage(self.root).free < required+64*1024**2:
            raise ValueError("Executed wave disk cache budget/free space exhausted; committed upstream checkpoints remain reusable")

    def writer(self, key, reference):
        return _ModeWriter(self, key, reference)

    def get(self, key):
        from temsim.physics.tip_gun_wave import TipGunCheckpoint
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Executed cache keys must be calculated dependency digests")
        index = self.root/(key+".json")
        if not index.exists():
            return None
        pointer = json.loads(index.read_text(encoding="utf-8"))
        directory = self.root/pointer["directory"]
        if directory.parent != self.root or directory.is_symlink():
            raise ValueError("Invalid executed cache directory")
        data = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
        identity = data.pop("manifest_digest")
        if identity != pointer["digest"] or identity != json_digest(data) or data["dependency"] != self.dependency or data["key"] != key:
            raise ValueError("Executed wave cache identity/dependency changed")
        if data["schema"] != "executed-tip-wave-disk-v1":
            raise ValueError("Unsupported executed wave cache schema")
        beam = _StoredBeam(_StoredModes(directory, data["modes"], data["reference_plane"]), data["reference_plane"], identity)
        return TipGunCheckpoint(beam, data["plane_z_mm"], data["current_a"],
            {**data["record"], "storage": {"manifest_digest": identity, "key": key,
                                          "dependency": self.dependency, "directory": str(directory)}})

    def put(self, key, checkpoint):
        writer = self.writer(key, checkpoint.beam.reference_plane)
        try:
            for mode in checkpoint.beam.modes:
                writer.append(mode)
            return writer.finish(checkpoint.plane_z_mm, checkpoint.reference_current_a, checkpoint.record)
        finally:
            writer.abort()


def resident_wave_bytes(beam):
    return beam.resident_bytes if isinstance(beam, _StoredBeam) else sum(m.plane.amplitude.nbytes for m in beam.modes)
