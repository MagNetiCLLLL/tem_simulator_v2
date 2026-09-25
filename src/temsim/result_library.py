"""Small navigation index for exported, executed classical calculation results.

This index supplies no physical source or calculation state. The result codec
remains responsible for validating and loading the referenced packages.
"""
from datetime import datetime, timedelta
from contextlib import contextmanager
import errno
import json
import math
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from threading import RLock
from time import monotonic, sleep
from uuid import uuid4


LIBRARY_SCHEMA = "temsim-result-library-v1"
MAX_INDEX_BYTES = 4 * 1024**2
MAX_ENTRIES = 1000
LOCK_TIMEOUT_SECONDS = 5.0
_ENTRY_FIELDS = frozenset({
    "id", "name", "path", "result_identity", "package_digest", "quality",
    "target_z_mm", "resumable_through_z_mm", "saved_at_utc", "size_bytes",
})


def _text(value, label, maximum):
    if (not isinstance(value, str) or not value.strip() or len(value) > maximum
            or not value.isprintable()):
        raise ValueError(f"Result library {label} must be nonempty printable text (maximum {maximum} characters)")
    return value.strip()


def _entry_id(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32}", value) is None:
        raise ValueError("Result library entry ID is invalid")
    return value


def _validate_entry(entry):
    if not isinstance(entry, dict) or set(entry) != _ENTRY_FIELDS:
        raise ValueError("Result library entry fields do not match the current schema")
    _entry_id(entry["id"])
    for key, maximum in (("name", 200), ("quality", 128)):
        if _text(entry[key], key, maximum) != entry[key]:
            raise ValueError(f"Result library {key} contains surrounding whitespace")
    path = entry["path"]
    if (not isinstance(path, str) or not path or len(path) > 32767
            or "\0" in path or not Path(path).is_absolute()):
        raise ValueError("Result library path must be absolute")
    for key in ("result_identity", "package_digest"):
        if not isinstance(entry[key], str) or re.fullmatch(r"[0-9a-f]{64}", entry[key]) is None:
            raise ValueError(f"Result library {key} must be a SHA-256 identity")
    for key in ("target_z_mm", "resumable_through_z_mm"):
        value = entry[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Result library {key} must be a finite axial coordinate in mm")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"Result library {key} must be a finite axial coordinate in mm")
    timestamp = entry["saved_at_utc"]
    if not isinstance(timestamp, str) or len(timestamp) > 64:
        raise ValueError("Result library save time must be an ISO UTC timestamp")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Result library save time must be an ISO UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("Result library save time must include the UTC timezone")
    size = entry["size_bytes"]
    if type(size) is not int or not 0 < size < 2**63:
        raise ValueError("Result library package size must be a positive integer")


def _validate_index(document):
    if (not isinstance(document, dict)
            or set(document) != {"schema", "entries", "startup_entry_id"}
            or document["schema"] != LIBRARY_SCHEMA):
        raise ValueError("Unsupported result library index schema")
    entries = document["entries"]
    if not isinstance(entries, list) or len(entries) > MAX_ENTRIES:
        raise ValueError("Result library entry list is invalid or exceeds its limit")
    ids, names = set(), set()
    for entry in entries:
        _validate_entry(entry)
        if entry["id"] in ids or entry["name"] in names:
            raise ValueError("Result library contains duplicate entry IDs or names")
        ids.add(entry["id"])
        names.add(entry["name"])
    startup = document["startup_entry_id"]
    if startup is not None and _entry_id(startup) not in ids:
        raise ValueError("Result library startup entry does not exist")
    return document


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key in result library index")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"Nonfinite JSON value in result library index: {value}")


class ResultLibrary:
    """Persist favourites and one optional startup selection without package I/O.

    Reads return detached scalar metadata, including entries whose exported
    files have since moved. A later codec load must check their actual identity.
    Every read-modify-write transaction holds an operating-system file lock,
    including between independent instances and processes. Atomic replacement
    keeps failures from damaging the old document. Lock waits are bounded.
    """

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        if self.root.exists() and not self.root.is_dir():
            raise ValueError("Result library root must be a directory")
        self.index_path = self.root / "index.json"
        self._lock = RLock()
        with self._transaction():
            self._read_index()

    @contextmanager
    def _transaction(self, *, create=False):
        deadline = monotonic() + LOCK_TIMEOUT_SECONDS
        message = f"Timed out waiting for result library index: {self.index_path}"
        if not self._lock.acquire(timeout=LOCK_TIMEOUT_SECONDS):
            raise TimeoutError(message)
        descriptor = None
        try:
            if create:
                self.root.mkdir(parents=True, exist_ok=True)
            elif not self.root.exists():
                # An absent index is a valid empty view; reads need not create
                # a directory solely to coordinate a future writer.
                yield
                return
            lock_path = self.root / ".index.lock"
            if lock_path.is_symlink():
                raise ValueError("Result library lock must be an owned file")
            descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            if os.name == "nt":
                import msvcrt

                def acquire():
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    # Windows byte locks can cover an empty file. Never write
                    # an initial byte that could race another lock owner.
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                def acquire():
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            while True:
                try:
                    acquire()
                    break
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                        raise
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        raise TimeoutError(message) from exc
                    sleep(min(0.02, remaining))
            yield
        finally:
            try:
                if descriptor is not None:
                    # Closing releases the OS lock even after exceptions; the
                    # OS also releases it if a process exits unexpectedly.
                    # Keep the lock file, so another process cannot lock a new
                    # inode while a former owner still holds the old one.
                    os.close(descriptor)
            finally:
                self._lock.release()

    def _read_index(self):
        if self.index_path.is_symlink():
            raise ValueError("Result library index must be an owned JSON file, not a symbolic link")
        try:
            with self.index_path.open("rb") as stream:
                payload = stream.read(MAX_INDEX_BYTES + 1)
        except FileNotFoundError:
            return {"schema": LIBRARY_SCHEMA, "entries": [], "startup_entry_id": None}
        except OSError as exc:
            if self.index_path.is_dir():
                raise ValueError("Result library index must be a JSON file") from exc
            raise
        if len(payload) > MAX_INDEX_BYTES:
            raise ValueError("Result library index exceeds its size limit")
        try:
            document = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_pairs,
                                  parse_constant=_nonfinite)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("Malformed result library index") from exc
        return _validate_index(document)

    def _write_index(self, document):
        _validate_index(document)
        payload = (json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
        if len(payload) > MAX_INDEX_BYTES:
            raise ValueError("Result library index exceeds its size limit")
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(dir=self.root, prefix=".result-library-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.index_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def allocate_path(self) -> Path:
        """Return a new package path under the owned results directory."""
        with self._transaction(create=True):
            self._read_index()
            directory = self.root / "results"
            if directory.resolve() != directory:
                raise ValueError("Result library results directory must stay inside its owned root")
            if directory.exists() and not directory.is_dir():
                raise ValueError("Result library results path must be a directory")
            directory.mkdir(parents=True, exist_ok=True)
            while True:
                path = directory / f"{uuid4().hex}.temresult"
                if not path.exists():
                    return path

    def remember(self, name: str, info: dict) -> dict:
        """Index an already exported result; a same-name save replaces metadata."""
        with self._transaction(create=True):
            document = self._read_index()
            name = _text(name, "name", 200)
            required = {"path", "identity", "_package_digest", "quality", "target_z_mm",
                        "resumable_through_z_mm", "saved_at_utc"}
            if not isinstance(info, dict) or not required <= set(info):
                raise ValueError("Result library requires completed export metadata")
            if not isinstance(info["path"], (str, Path)) or not str(info["path"]):
                raise ValueError("Result library requires an exported package path")
            path = Path(info["path"]).expanduser().resolve()
            if not path.is_file():
                raise ValueError("Result library exported package does not exist")
            previous = next((row for row in document["entries"] if row["name"] == name), None)
            entry = {
                "id": previous["id"] if previous is not None else uuid4().hex,
                "name": name, "path": str(path), "result_identity": info["identity"],
                "package_digest": info["_package_digest"], "quality": _text(info["quality"], "quality", 128),
                "target_z_mm": info["target_z_mm"],
                "resumable_through_z_mm": info["resumable_through_z_mm"],
                "saved_at_utc": info["saved_at_utc"], "size_bytes": path.stat().st_size,
            }
            _validate_entry(entry)
            if previous is None:
                document["entries"].append(entry)
            else:
                document["entries"][document["entries"].index(previous)] = entry
            self._write_index(document)
            return dict(entry)

    def entries(self) -> tuple[dict, ...]:
        with self._transaction():
            return tuple(dict(row) for row in self._read_index()["entries"])

    def set_startup(self, entry_id: str | None) -> None:
        with self._transaction(create=True):
            document = self._read_index()
            document["startup_entry_id"] = entry_id
            self._write_index(document)

    def startup_entry(self) -> dict | None:
        with self._transaction():
            document = self._read_index()
            return next((dict(row) for row in document["entries"]
                         if row["id"] == document["startup_entry_id"]), None)
