"""Bounded, dependency-based virtual-electron history archives (.temdiag).

These files are diagnostic histories, never microscope sources, accepted main
results or transport checkpoints. Loading does not execute or restore a field.
An explicit new execution needs the corresponding captured result and assets.
JSON metadata and float64 NPY arrays are checksummed; no pickle is admitted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile
import zlib
from zipfile import BadZipFile, ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np

from temsim import __version__
from temsim.magnetic_test_particle import TestElectronSettings, TestElectronTrajectory

SCHEMA = "temsim.virtual-electron-diagnostic"
SCHEMA_VERSION = 1
EXTENSION = ".temdiag"
_ARRAYS = ("positions_m", "directions", "time_s", "path_length_m", "kinetic_energy_ev",
           "electrostatic_potential_v", "speed_m_per_s", "momentum_kg_m_per_s")
_VECTORS = {"positions_m", "directions", "momentum_kg_m_per_s"}
_FAILURES = {"initial_outside_domain", "step_limit", "cancelled", "numerical_limit", "in_progress"}
_STATES = {"uncomputed", "completed", "incomplete", "previous", "failed"}
_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_COLOUR = re.compile(r"#[0-9a-fA-F]{6}\Z")


@dataclass(frozen=True)
class DiagnosticSessionLimits:
    maximum_archive_bytes: int = 128 * 1024**2
    maximum_unpacked_bytes: int = 256 * 1024**2
    maximum_manifest_bytes: int = 2 * 1024**2
    maximum_memory_bytes: int = 512 * 1024**2
    maximum_records: int = 64
    maximum_points_per_trajectory: int = 200_001

    def __post_init__(self):
        for item in fields(self):
            if type(getattr(self, item.name)) is not int or getattr(self, item.name) < 1:
                raise ValueError(f"{item.name} must be a positive integer")


DEFAULT_LIMITS = DiagnosticSessionLimits()


@dataclass(frozen=True)
class DiagnosticDependency:
    result_reference: str | None = None
    physical_identity: str | None = None
    numerical_identity: str | None = None
    transport_identity: str | None = None
    provider_notes: tuple[str, ...] = ()
    bounds_m: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None

    def __post_init__(self):
        for name in ("result_reference", "physical_identity", "numerical_identity", "transport_identity"):
            _text(getattr(self, name), name, optional=True)
        object.__setattr__(self, "provider_notes", _notes(self.provider_notes))
        if self.bounds_m is not None:
            array = np.asarray(self.bounds_m)
            if array.shape != (2, 3) or array.dtype.kind not in "fi" or not np.isfinite(array).all():
                raise ValueError("Dependency bounds must contain two finite XYZ points in metres")
            if np.any(array[0] > array[1]):
                raise ValueError("Dependency lower bounds exceed upper bounds")
            object.__setattr__(self, "bounds_m", tuple(tuple(float(x) for x in row) for row in array))

    def matches(self, *, physical_identity=None, numerical_identity=None, transport_identity=None):
        """Only compare known identities; a match never authorizes cache adoption."""
        return bool(self.physical_identity and self.numerical_identity
                    and self.physical_identity == physical_identity
                    and self.numerical_identity == numerical_identity
                    and (self.transport_identity is None or self.transport_identity == transport_identity))


@dataclass(frozen=True)
class DiagnosticDisplayState:
    projection_degrees: float = 0.
    axial_limits_mm: tuple[float, float] | None = None
    transverse_limits_mm: tuple[float, float] | None = None
    overlay: bool = False
    selected_key: str | None = None
    show_background: bool = True

    def __post_init__(self):
        object.__setattr__(self, "projection_degrees", _finite(self.projection_degrees, "projection_degrees"))
        _boolean(self.overlay, "overlay")
        _boolean(self.show_background, "show_background")
        if self.selected_key is not None:
            _identifier(self.selected_key)
        for name in ("axial_limits_mm", "transverse_limits_mm"):
            value = getattr(self, name)
            if value is not None:
                if len(value) != 2:
                    raise ValueError(f"{name} needs two finite ordered values")
                low, high = (_finite(x, name) for x in value)
                if low >= high:
                    raise ValueError(f"{name} needs increasing limits")
                object.__setattr__(self, name, (low, high))


@dataclass(frozen=True)
class DiagnosticElectronRecord:
    key: str
    label: str
    colour: str
    settings: TestElectronSettings
    checked: bool = True
    trajectory: TestElectronTrajectory | None = None
    trajectory_settings: TestElectronSettings | None = None
    revision: int = 0
    state: str = "uncomputed"
    error: str | None = None
    dependency: DiagnosticDependency | None = None

    def __post_init__(self):
        _identifier(self.key)
        _text(self.label, "electron label", maximum=256)
        if not isinstance(self.colour, str) or not _COLOUR.fullmatch(self.colour):
            raise ValueError("Electron colour must be #RRGGBB")
        _boolean(self.checked, "checked")
        if type(self.revision) is not int or not 0 <= self.revision <= 2**53:
            raise ValueError("Electron revision must be a nonnegative integer")
        _text(self.error, "error", optional=True)
        if self.state not in _STATES:
            raise ValueError("Unknown diagnostic electron state")
        if not isinstance(self.settings, TestElectronSettings):
            raise ValueError("Diagnostic settings must describe one test electron")
        _settings(asdict(self.settings))
        if self.dependency is not None and not isinstance(self.dependency, DiagnosticDependency):
            raise ValueError("Invalid diagnostic dependency")
        if self.trajectory is None:
            if self.trajectory_settings is not None or self.state not in {"uncomputed", "failed"}:
                raise ValueError("An absent trajectory cannot be presented as an executed result")
        else:
            if not isinstance(self.trajectory_settings, TestElectronSettings):
                raise ValueError("Stored trajectories require their actual executed settings")
            _settings(asdict(self.trajectory_settings))
            trajectory = _freeze_trajectory(self.trajectory, self.trajectory_settings)
            object.__setattr__(self, "trajectory", trajectory)
            if self.state == "uncomputed":
                raise ValueError("An uncomputed record cannot contain a trajectory")
            if self.state == "completed" and (not trajectory.completed or self.settings != self.trajectory_settings):
                raise ValueError("Completed record does not match its executed trajectory and settings")
            if self.state == "incomplete" and trajectory.completed:
                raise ValueError("Incomplete record contains a completed trajectory")


@dataclass(frozen=True)
class DiagnosticSession:
    records: tuple[DiagnosticElectronRecord, ...]
    dependency: DiagnosticDependency = field(default_factory=DiagnosticDependency)
    display: DiagnosticDisplayState = field(default_factory=DiagnosticDisplayState)
    app_version: str = __version__
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        object.__setattr__(self, "records", tuple(self.records))
        if not isinstance(self.dependency, DiagnosticDependency) or not isinstance(self.display, DiagnosticDisplayState):
            raise ValueError("Invalid diagnostic dependency or display state")
        _text(self.app_version, "app_version", maximum=128)
        _text(self.created_at_utc, "created_at_utc", maximum=128)
        try:
            timestamp = datetime.fromisoformat(self.created_at_utc)
            if timestamp.utcoffset() is None or timestamp.utcoffset().total_seconds() != 0:
                raise ValueError("Timestamp must use UTC")
        except (TypeError, ValueError) as exc:
            raise ValueError("created_at_utc must be an ISO 8601 UTC timestamp") from exc
        ids = set()
        for record in self.records:
            if not isinstance(record, DiagnosticElectronRecord):
                raise ValueError("Session records must be diagnostic electron records")
            if record.key in ids:
                raise ValueError("Duplicate diagnostic electron ID")
            ids.add(record.key)
            dependency = record.dependency or self.dependency
            if record.trajectory is not None:
                for name, actual in (("physical_identity", record.trajectory.physical_field_identity),
                                     ("numerical_identity", record.trajectory.numerical_field_identity)):
                    declared = getattr(dependency, name)
                    if declared is not None and actual is not None and declared != actual:
                        raise ValueError("Trajectory does not belong to its declared field dependency")
        if self.display.selected_key is not None and self.display.selected_key not in ids:
            raise ValueError("Selected electron is not present in the session")


def _text(value, name, *, maximum=8192, optional=False):
    if optional and value is None:
        return
    if not isinstance(value, str) or not value or len(value) > maximum or "\x00" in value:
        raise ValueError(f"Invalid {name}")


def _identifier(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("Invalid diagnostic electron ID")


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be a JSON boolean")


def _finite(value, name):
    if isinstance(value, (bool, str)) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _notes(value):
    if not isinstance(value, (list, tuple)) or len(value) > 256:
        raise ValueError("Invalid provider/trajectory notes")
    for item in value:
        _text(item, "note")
    return tuple(value)


def _exact_dict(value, names, label):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f"Missing or unknown {label} fields")


def _settings(value):
    _exact_dict(value, (item.name for item in fields(TestElectronSettings)), "electron settings")
    for name, item in value.items():
        if name == "position_m":
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise ValueError("position_m must contain XYZ metres")
            for coordinate in item:
                _finite(coordinate, name)
        else:
            _finite(item, name)
    if type(value["max_steps"]) is not int:
        raise ValueError("max_steps must be an integer")
    return TestElectronSettings(**value)


def _freeze_trajectory(value, settings):
    if not isinstance(value, TestElectronTrajectory):
        raise ValueError("Invalid diagnostic trajectory")
    _boolean(value.completed, "trajectory.completed")
    _text(value.reason, "trajectory.reason", maximum=256)
    if type(value.steps) is not int or not 0 <= value.steps <= min(settings.max_steps, 200_000):
        raise ValueError("Invalid trajectory step count")
    if value.completed and (value.reason in _FAILURES or value.reason.startswith("unsupported_field:")):
        raise ValueError("An unfinished or unsupported path cannot be marked completed")
    for name in ("physical_field_identity", "numerical_field_identity", "execution_identity"):
        _text(getattr(value, name), name, optional=True)
    changes = {"notes": _notes(value.notes)}
    count = value.steps + 1
    for name in _ARRAYS:
        array = np.asarray(getattr(value, name))
        shape = (count, 3) if name in _VECTORS else (count,)
        if array.shape != shape or array.dtype != np.dtype("float64"):
            raise ValueError(f"{name} must be float64 with shape {shape}")
        if name == "electrostatic_potential_v":
            if np.isinf(array).any():
                raise ValueError("Infinite electric potential is invalid")
        elif not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
        # The bytes owner prevents callers from re-enabling writes after load.
        changes[name] = np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(shape)
    for name in ("time_s", "path_length_m"):
        array = changes[name]
        if array[0] != 0 or np.any(np.diff(array) < 0):
            raise ValueError(f"{name} must start at zero and be nondecreasing")
    for name in ("kinetic_energy_ev", "speed_m_per_s"):
        if np.any(changes[name] < 0):
            raise ValueError(f"{name} must be nonnegative")
    if np.any(changes["speed_m_per_s"] >= 299_792_458.):
        raise ValueError("Electron speed must remain below the speed of light")
    unknown_potential = np.isnan(changes["electrostatic_potential_v"]).any()
    for name in ("energy_invariant_error_ev", "energy_invariant_relative_error"):
        error = getattr(value, name)
        if unknown_potential and isinstance(error, (float, np.floating)) and np.isnan(error):
            continue
        if _finite(error, name) < 0:
            raise ValueError(f"{name} must be nonnegative or explicitly unavailable")
    return replace(value, **changes)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key in diagnostic session")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("Nonfinite JSON number in diagnostic session")


def _session_payload(session, limits):
    if not isinstance(session, DiagnosticSession):
        raise ValueError("Only a diagnostic session can be saved here")
    if len(session.records) > limits.maximum_records:
        raise ValueError("Diagnostic session exceeds the electron-count limit")
    arrays, records, total = {}, [], 0
    for index, record in enumerate(session.records):
        item = {name: getattr(record, name) for name in ("key", "label", "colour", "checked", "revision", "state", "error")}
        item.update(settings=asdict(record.settings), trajectory_settings=None, trajectory=None,
                    dependency=asdict(record.dependency) if record.dependency is not None else None)
        if record.trajectory is not None:
            trace = record.trajectory
            if trace.steps + 1 > limits.maximum_points_per_trajectory:
                raise ValueError("Diagnostic trajectory exceeds the point-count limit")
            descriptors = {}
            for name in _ARRAYS:
                array = getattr(trace, name)
                total += array.nbytes + 256
                if total > limits.maximum_unpacked_bytes:
                    raise ValueError("Diagnostic session exceeds the unpacked-size limit")
                stream = BytesIO()
                np.save(stream, array, allow_pickle=False)
                data = stream.getvalue()
                path = f"arrays/{index:04d}-{name}.npy"
                arrays[path] = data
                descriptors[name] = {"path": path, "shape": list(array.shape), "dtype": "<f8", "sha256": sha256(data).hexdigest()}
            trace_meta = {name: getattr(trace, name) for name in ("reason", "completed", "steps", "notes", "physical_field_identity", "numerical_field_identity", "execution_identity")}
            for name in ("energy_invariant_error_ev", "energy_invariant_relative_error"):
                number = getattr(trace, name)
                trace_meta[name] = None if np.isnan(number) else number
            trace_meta["arrays"] = descriptors
            item.update(trajectory_settings=asdict(record.trajectory_settings), trajectory=trace_meta)
        records.append(item)
    payload = {"schema": SCHEMA, "version": SCHEMA_VERSION, "mode": "captured-result-dependency",
               "app_version": session.app_version, "created_at_utc": session.created_at_utc,
               "dependency": asdict(session.dependency), "display": asdict(session.display), "records": records}
    manifest = _canonical({"payload": payload, "sha256": sha256(_canonical(payload)).hexdigest()})
    if len(manifest) > limits.maximum_manifest_bytes or len(manifest) + sum(map(len, arrays.values())) > limits.maximum_unpacked_bytes:
        raise ValueError("Diagnostic session exceeds its metadata or unpacked-size limit")
    if 2 * sum(map(len, arrays.values())) + 8 * len(manifest) > limits.maximum_memory_bytes:
        raise ValueError("Diagnostic session exceeds its memory limit")
    return manifest, arrays


def save_diagnostic_session(session, path, *, limits=DEFAULT_LIMITS, overwrite=True):
    """Atomically save one consistent immutable snapshot, preserving old files on error."""
    manifest, arrays = _session_payload(session, limits)
    target = Path(path)
    if target.suffix.lower() != EXTENSION:
        raise ValueError(f"Diagnostic sessions must use the independent {EXTENSION} extension")
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            with ZipFile(handle, "w", compression=ZIP_DEFLATED, compresslevel=3) as archive:
                archive.writestr("manifest.json", manifest)
                for name, data in arrays.items():
                    archive.writestr(name, data)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.stat().st_size > limits.maximum_archive_bytes:
            raise ValueError("Diagnostic session exceeds the archive-size limit")
        if overwrite:
            os.replace(temporary, target)
        else:
            # Hard-link creation is atomic and refuses an existing destination.
            os.link(temporary, target)
            temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def _read_array(archive, descriptor, *, expected_path, count, vector, limits):
    _exact_dict(descriptor, ("path", "shape", "dtype", "sha256"), "array descriptor")
    expected_shape = [count, 3] if vector else [count]
    if (descriptor["path"] != expected_path or not isinstance(descriptor["shape"], list)
            or any(type(value) is not int for value in descriptor["shape"]) or descriptor["shape"] != expected_shape
            or descriptor["dtype"] != "<f8"):
        raise ValueError("Invalid diagnostic array path, shape or dtype")
    expected_bytes = count * (3 if vector else 1) * 8
    if archive.getinfo(expected_path).file_size > expected_bytes + 4096:
        raise ValueError("Diagnostic NPY byte count exceeds its bounded shape/header")
    data = archive.read(expected_path)
    if sha256(data).hexdigest() != descriptor["sha256"]:
        raise ValueError("Diagnostic array checksum mismatch")
    stream = BytesIO(data)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=4096)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=4096)
    else:
        raise ValueError("Unsupported diagnostic NPY version")
    if list(shape) != expected_shape or dtype != np.dtype("<f8") or fortran:
        raise ValueError("Diagnostic NPY shape/dtype does not match its descriptor")
    if len(data) - stream.tell() != expected_bytes or expected_bytes > limits.maximum_unpacked_bytes:
        raise ValueError("Diagnostic NPY byte count does not match its shape")
    return np.frombuffer(data, dtype="<f8", offset=stream.tell()).reshape(shape)


def _dataclass_from_dict(cls, value):
    _exact_dict(value, (item.name for item in fields(cls)), cls.__name__)
    return cls(**value)


def _load(archive, limits):
    infos = archive.infolist()
    if len(infos) > 1 + 8 * limits.maximum_records or len({item.filename for item in infos}) != len(infos):
        raise ValueError("Too many or duplicate diagnostic archive members")
    if any(item.is_dir() or item.flag_bits & 1 or item.compress_type not in {ZIP_STORED, ZIP_DEFLATED} for item in infos):
        raise ValueError("Unsupported diagnostic archive member")
    if sum(item.file_size for item in infos) > limits.maximum_unpacked_bytes:
        raise ValueError("Diagnostic session exceeds the unpacked-size limit")
    index = {item.filename: item for item in infos}
    if "manifest.json" not in index or index["manifest.json"].file_size > limits.maximum_manifest_bytes:
        raise ValueError("Missing or oversized diagnostic manifest")
    if 2 * sum(item.file_size for item in infos) + 8 * index["manifest.json"].file_size > limits.maximum_memory_bytes:
        raise ValueError("Diagnostic session exceeds its memory limit")
    document = json.loads(archive.read("manifest.json"), object_pairs_hook=_unique_pairs, parse_constant=_nonfinite)
    _exact_dict(document, ("payload", "sha256"), "diagnostic manifest")
    payload = document["payload"]
    if sha256(_canonical(payload)).hexdigest() != document["sha256"]:
        raise ValueError("Diagnostic manifest checksum mismatch")
    _exact_dict(payload, ("schema", "version", "mode", "app_version", "created_at_utc", "dependency", "display", "records"), "session")
    if payload["schema"] != SCHEMA or type(payload["version"]) is not int or payload["version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported diagnostic session schema/version")
    if payload["mode"] != "captured-result-dependency":
        raise ValueError("Unsupported diagnostic session mode")
    items = payload["records"]
    if not isinstance(items, list) or len(items) > limits.maximum_records:
        raise ValueError("Invalid or excessive diagnostic electron count")
    dependency = _dataclass_from_dict(DiagnosticDependency, payload["dependency"])
    display = _dataclass_from_dict(DiagnosticDisplayState, payload["display"])
    names, records = {"manifest.json"}, []
    for index, item in enumerate(items):
        _exact_dict(item, (field.name for field in fields(DiagnosticElectronRecord)), "diagnostic record")
        values = dict(item)
        values["settings"] = _settings(values["settings"])
        if values["dependency"] is not None:
            values["dependency"] = _dataclass_from_dict(DiagnosticDependency, values["dependency"])
        if values["trajectory_settings"] is not None:
            values["trajectory_settings"] = _settings(values["trajectory_settings"])
        trace = item["trajectory"]
        if trace is not None:
            expected = {field.name for field in fields(TestElectronTrajectory)} - set(_ARRAYS) | {"arrays"}
            _exact_dict(trace, expected, "diagnostic trajectory")
            if type(trace["steps"]) is not int or not 0 <= trace["steps"] < limits.maximum_points_per_trajectory:
                raise ValueError("Diagnostic trajectory exceeds the point-count limit")
            _exact_dict(trace["arrays"], _ARRAYS, "trajectory arrays")
            data = {name: value for name, value in trace.items() if name != "arrays"}
            for name in _ARRAYS:
                member = f"arrays/{index:04d}-{name}.npy"
                names.add(member)
                data[name] = _read_array(archive, trace["arrays"][name], expected_path=member,
                                         count=trace["steps"] + 1, vector=name in _VECTORS, limits=limits)
            for name in ("energy_invariant_error_ev", "energy_invariant_relative_error"):
                if data[name] is None:
                    data[name] = float("nan")
            values["trajectory"] = TestElectronTrajectory(**data)
        records.append(DiagnosticElectronRecord(**values))
    if names != set(archive.namelist()):
        raise ValueError("Unknown, missing or unsafe diagnostic archive path")
    return DiagnosticSession(tuple(records), dependency, display, payload["app_version"], payload["created_at_utc"])


def _preflight_zip(handle, limits):
    """Bound ZIP indexing before ZipFile allocates its central-directory graph.

    The diagnostic writer needs neither ZIP64 nor multidisk archives. Scan
    actual central-directory headers: the EOCD entry count is untrusted and
    Python's ZIP reader otherwise indexes all entries regardless of that count.
    Only a bounded tail, one fixed header and one member's metadata are held.
    """
    size = os.fstat(handle.fileno()).st_size
    if size > limits.maximum_archive_bytes:
        raise ValueError("Diagnostic session exceeds the archive-size limit")
    tail_size = min(size, 22 + 65535)
    handle.seek(size - tail_size)
    tail = handle.read(tail_size)
    end = tail.rfind(b"PK\x05\x06")
    if end < 0 or end + 22 > len(tail):
        raise BadZipFile("Missing diagnostic ZIP end record")
    (_, disk, directory_disk, disk_count, declared_count, directory_size,
     directory_offset, comment_size) = struct.unpack("<4s4H2IH", tail[end:end+22])
    end_offset = size - tail_size + end
    if end + 22 + comment_size != len(tail):
        raise BadZipFile("Invalid diagnostic ZIP end record or trailing data")
    if (disk_count == 0xffff or declared_count == 0xffff
            or directory_size == 0xffffffff or directory_offset == 0xffffffff):
        raise ValueError("ZIP64 diagnostic archives are unsupported")
    if disk or directory_disk or disk_count != declared_count:
        raise ValueError("Multidisk diagnostic archives are unsupported")
    if end_offset >= 20:
        handle.seek(end_offset - 20)
        if handle.read(4) == b"PK\x06\x07":
            raise ValueError("ZIP64 diagnostic archives are unsupported")
    entry_limit = 1 + 8 * limits.maximum_records
    if declared_count > entry_limit:
        raise ValueError("Too many diagnostic archive members")
    if (directory_offset > end_offset or directory_size > end_offset
            or directory_offset + directory_size != end_offset):
        raise BadZipFile("Invalid diagnostic ZIP central-directory bounds")
    # ZipFile reads the entire directory before creating entry objects. Bound
    # both that byte buffer and a conservative metadata expansion in advance.
    if 8 * directory_size > limits.maximum_memory_bytes:
        raise ValueError("Diagnostic session exceeds its central-directory memory limit")
    handle.seek(directory_offset)
    count = unpacked_size = manifest_size = 0
    while handle.tell() < end_offset:
        if count >= entry_limit:
            raise ValueError("Too many actual diagnostic archive members")
        if end_offset - handle.tell() < 46:
            raise BadZipFile("Truncated diagnostic ZIP central header")
        header = handle.read(46)
        if len(header) != 46:
            raise BadZipFile("Truncated diagnostic ZIP central header")
        (signature, _made_by, needed, flags, method, _time, _date, _crc,
         compressed_size, expanded_size, name_size, extra_size, member_comment_size,
         member_disk, _internal, _external, local_offset) = struct.unpack("<4s6H3I5H2I", header)
        if signature != b"PK\x01\x02":
            raise BadZipFile("Invalid diagnostic ZIP central header")
        if needed == 45 or any(value == 0xffffffff for value in
                               (compressed_size, expanded_size, local_offset)):
            raise ValueError("ZIP64 diagnostic archives are unsupported")
        if needed not in (10, 20) or flags & ~(8 | 2048) or method not in (ZIP_STORED, ZIP_DEFLATED):
            raise ValueError("Unsupported diagnostic ZIP version or member encoding")
        if member_disk:
            raise ValueError("Multidisk diagnostic archives are unsupported")
        variable_size = name_size + extra_size + member_comment_size
        if not name_size or variable_size > end_offset - handle.tell():
            raise BadZipFile("Invalid diagnostic ZIP member metadata bounds")
        if local_offset + 30 + compressed_size > directory_offset:
            raise BadZipFile("Invalid diagnostic ZIP member data bounds")
        name = handle.read(name_size)
        extra = handle.read(extra_size)
        extra_offset = 0
        while extra_offset < len(extra):
            if len(extra) - extra_offset < 4:
                raise BadZipFile("Truncated diagnostic ZIP extra-field header")
            kind, length = struct.unpack_from("<HH", extra, extra_offset)
            extra_offset += 4
            if kind == 1:
                raise ValueError("ZIP64 diagnostic archives are unsupported")
            if length > len(extra) - extra_offset:
                raise BadZipFile("Truncated diagnostic ZIP extra field")
            extra_offset += length
        handle.seek(member_comment_size, os.SEEK_CUR)
        count += 1
        unpacked_size += expanded_size
        if name == b"manifest.json":
            manifest_size = expanded_size
        if unpacked_size > limits.maximum_unpacked_bytes:
            raise ValueError("Diagnostic session exceeds the unpacked-size limit")
        if manifest_size > limits.maximum_manifest_bytes:
            raise ValueError("Missing or oversized diagnostic manifest")
        estimated_memory = (2 * unpacked_size + 8 * manifest_size
                            + 8 * directory_size + 1024 * count)
        if estimated_memory > limits.maximum_memory_bytes:
            raise ValueError("Diagnostic session exceeds its memory limit")
    if count != declared_count or not count:
        raise BadZipFile("Diagnostic ZIP declared count differs from actual entries")
    handle.seek(0)


def load_diagnostic_session(path, *, limits=DEFAULT_LIMITS):
    """Validate fully before returning detached history; never mutate a live state."""
    try:
        with Path(path).open("rb") as handle:
            _preflight_zip(handle, limits)
            with ZipFile(handle) as archive:
                return _load(archive, limits)
    except (BadZipFile, KeyError, TypeError, AttributeError, UnicodeError, RecursionError,
            OverflowError, EOFError, NotImplementedError, zlib.error) as exc:
        raise ValueError(f"Invalid diagnostic session: {exc}") from exc
