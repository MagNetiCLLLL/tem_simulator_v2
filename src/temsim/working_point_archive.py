"""Manifest-only index and bounded readers for existing working-point archives."""
from dataclasses import dataclass
from io import BytesIO
import json
import math
from pathlib import Path
import re
import stat
from zipfile import ZipFile

import numpy as np

from temsim.immutable_json import freeze_json, json_digest
from temsim.instrument_snapshot import InstrumentSnapshot

DEFAULT_MAXIMUM_UNPACKED_BYTES = 8 * 1024**3
MAXIMUM_MANIFEST_BYTES = 256 * 1024**2
MAXIMUM_PACKAGE_ENTRIES = 10000


def _validate_budget(maximum):
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("Working-point memory budget must be a positive integer number of bytes")
    return maximum


def prepare_manifest(document, arrays, *, maximum_unpacked_bytes):
    """Validate the same bounds as the reader before creating an output file."""
    maximum = _validate_budget(maximum_unpacked_bytes)
    if len(arrays) + 1 > MAXIMUM_PACKAGE_ENTRIES:
        raise ValueError("Working-point package exceeds the entry limit")
    total = 0
    for value in arrays.values():
        total += _numeric_size(value.shape, value.dtype, maximum)
        header = BytesIO()
        np.lib.format.write_array_header_1_0(header, np.lib.format.header_data_from_array_1_0(value))
        total += header.tell()
    limit = min(maximum, MAXIMUM_MANIFEST_BYTES)
    output, size = BytesIO(), 0
    for chunk in json.JSONEncoder(allow_nan=False, separators=(",", ":")).iterencode(document):
        encoded = chunk.encode("utf-8")
        size += len(encoded)
        if size > limit:
            raise ValueError("Missing or oversized working-point manifest")
        output.write(encoded)
    if total + size > maximum:
        raise ValueError("Working-point package exceeds the declared memory budget")
    return output.getvalue()


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key in working-point package")
        result[key] = value
    return result


def read_manifest(archive, *, maximum_unpacked_bytes):
    from temsim.working_point import PACKAGE_SCHEMA
    _validate_budget(maximum_unpacked_bytes)
    entries = archive.infolist()
    names = {entry.filename for entry in entries}
    if len(names) != len(entries):
        raise ValueError("Duplicate package entries")
    if any(stat.S_ISLNK(entry.external_attr >> 16) for entry in entries):
        raise ValueError("Working-point package links are not supported")
    if len(entries) > MAXIMUM_PACKAGE_ENTRIES or sum(entry.file_size for entry in entries) > maximum_unpacked_bytes:
        raise ValueError("Working-point package exceeds the declared memory budget")
    if "manifest.json" not in names or archive.getinfo("manifest.json").file_size > min(maximum_unpacked_bytes, MAXIMUM_MANIFEST_BYTES):
        raise ValueError("Missing or oversized working-point manifest")
    def nonfinite(value):
        raise ValueError("Nonfinite JSON value in working-point manifest")
    data = json.loads(archive.read("manifest.json"), object_pairs_hook=_unique_pairs, parse_constant=nonfinite)
    if data["schema"] != PACKAGE_SCHEMA:
        raise ValueError("Unsupported working-point package")
    records = data["arrays"]
    if not isinstance(records, dict):
        raise ValueError("Array inventory must be a mapping")
    declared = set()
    identities = {}
    for key, record in records.items():
        entry = record["entry"]
        if not re.fullmatch(r"arrays/[0-9]+\.npy", entry) or entry in declared:
            raise ValueError("Unsafe or aliased numeric package entry")
        declared.add(entry)
        identities[key] = {k: record[k] for k in ("shape", "dtype", "sha256")}
        _numeric_size(record["shape"], record["dtype"], maximum_unpacked_bytes)
    if names != {"manifest.json", *declared}:
        raise ValueError("Missing or undeclared package entry")
    for record in records.values():
        validate_numeric_entry(archive, record, maximum_unpacked_bytes=maximum_unpacked_bytes)
    snapshot = InstrumentSnapshot.from_dict(data["snapshot"])
    plane = data["plane_z_mm"]
    if not isinstance(plane, (int, float)) or not math.isfinite(plane) or not data["stage_signature"]:
        raise ValueError("Invalid checkpoint plane or signature")
    expected = json_digest(dict(snapshot=snapshot.digest, plane="column-z-mm:"+float(plane).hex(),
        signature=data["stage_signature"], payload=json_digest(identities), metadata=data["metadata"], parent=data["parent_id"]))
    if expected != data["digest"]:
        raise ValueError("Working-point package checksum mismatch")
    return data, snapshot


def _numeric_size(shape, dtype, maximum):
    if not isinstance(shape, (list, tuple)) or len(shape) > 32 or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("Invalid numeric array dimensions")
    dtype = np.dtype(dtype)
    if dtype.kind not in "biufc" or dtype.fields is not None:
        raise ValueError("Package arrays require plain numeric dtypes")
    count = math.prod(shape)
    size = count * dtype.itemsize
    if size > maximum or count > np.iinfo(np.intp).max:
        raise ValueError("Numeric array exceeds the declared memory budget")
    return size


def validate_numeric_entry(archive, record, *, maximum_unpacked_bytes):
    # Check header dimensions before np.load is allowed to allocate a buffer.
    with archive.open(record["entry"]) as stream:
        version = np.lib.format.read_magic(stream)
        reader = {(1, 0): np.lib.format.read_array_header_1_0,
                  (2, 0): np.lib.format.read_array_header_2_0}.get(version)
        if reader is None:
            raise ValueError("Unsupported numeric NPY header version")
        shape, fortran_order, dtype = reader(stream, max_header_size=16384)
        size = _numeric_size(shape, dtype, maximum_unpacked_bytes)
        if list(shape) != record["shape"] or dtype.str != record["dtype"]:
            raise ValueError("Numeric header does not match the manifest")
        if archive.getinfo(record["entry"]).file_size != stream.tell() + size:
            raise ValueError("Numeric payload length does not match its header")


def read_numeric_entry(archive, record, *, maximum_unpacked_bytes):
    validate_numeric_entry(archive, record, maximum_unpacked_bytes=maximum_unpacked_bytes)
    with archive.open(record["entry"]) as stream:
        return np.load(stream, allow_pickle=False, max_header_size=16384)


@dataclass(frozen=True)
class WorkingPointArchiveIndex:
    path: Path
    snapshot: object
    plane_z_mm: float
    stage_signature: str
    metadata: object
    parent_id: str | None
    digest: str
    array_records: object
    index_summary: object
    evidence: object
    maximum_unpacked_bytes: int
    unpacked_size_bytes: int

    @classmethod
    def read(cls, path, *, maximum_unpacked_bytes=8*1024**3):
        with ZipFile(path) as archive:
            data, snapshot = read_manifest(archive, maximum_unpacked_bytes=maximum_unpacked_bytes)
            unpacked_size_bytes = sum(item.file_size for item in archive.infolist())
        return cls(Path(path).resolve(), snapshot, data["plane_z_mm"], data["stage_signature"],
            freeze_json(data["metadata"]), data["parent_id"], data["digest"], freeze_json(data["arrays"]),
            freeze_json(data.get("index_summary", {})), freeze_json(data.get("evidence", [])),
            maximum_unpacked_bytes, unpacked_size_bytes)

    @property
    def plane_id(self):
        return "column-z-mm:" + float(self.plane_z_mm).hex()

    @property
    def has_retained_payload(self):
        return bool(self.array_records)

    @property
    def is_input_design(self):
        return self.metadata.get("package_kind") == "INSTRUMENT_INPUTS_ONLY" and not self.array_records

    @property
    def is_metadata_only(self):
        from temsim.working_point_export import METADATA_SCHEMA
        return self.metadata.get("package_kind") == "METADATA_ONLY" or self.snapshot.graph.get("schema") == METADATA_SCHEMA

    def load(self):
        from temsim.working_point import WorkingPointCheckpoint
        point = WorkingPointCheckpoint.read_package(self.path, maximum_unpacked_bytes=self.maximum_unpacked_bytes)
        if point.digest != self.digest:
            raise ValueError("Archive changed after indexing; re-import the new record explicitly")
        return point

    def compatible_state(self):
        return self.load().compatible_state()

    def write_package(self, path, **kwargs):
        if kwargs.get("mode") in {"inputs", "metadata"}:
            from temsim.working_point_export import export_checkpoint
            point = export_checkpoint(self, kwargs.pop("mode"))
            return point.write_package(path, **kwargs)
        return self.load().write_package(path, **kwargs)
