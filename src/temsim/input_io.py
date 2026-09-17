"""Read-only, request-local resolution of explicitly captured model inputs.

No global Path monkeypatch, extraction to original paths or archive-to-live
fallback occurs. Output/cache writers do not use this model-input interface.
"""
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache, wraps
from hashlib import sha256
from inspect import signature
from importlib.metadata import version, PackageNotFoundError
from io import BytesIO, StringIO
from pathlib import Path, PurePosixPath
import platform
from threading import RLock
from types import MappingProxyType
from weakref import WeakValueDictionary

from temsim.immutable_json import freeze_json, json_digest, thaw_json

ARCHIVE_SCHEMA = "model-input-archive-v1"
MAX_INPUT_BYTES = 96 * 1024**2
_ACTIVE = ContextVar("temsim_model_inputs", default=None)
_RESOLVERS = WeakValueDictionary()
_LOCK = RLock()


@lru_cache(maxsize=1)
def runtime_identity():
    libraries = {}
    for name in ("numpy", "scipy", "numba", "ase", "abtem", "xraylib"):
        try:
            libraries[name] = version(name)
        except PackageNotFoundError:
            libraries[name] = None
    return freeze_json(dict(python=platform.python_version(), libraries=libraries))


class InputArchive:
    """Verified immutable file content; original names are provenance only."""

    def __init__(self, payload):
        if payload.get("schema") != ARCHIVE_SCHEMA:
            raise ValueError("Unsupported model-input archive schema")
        if payload.get("digest") != json_digest({k: v for k, v in payload.items() if k != "digest"}):
            raise ValueError("Model-input archive checksum mismatch")
        self.payload = freeze_json(payload)
        self.identity = payload["digest"]
        self.config_root = self._absolute(payload["config_root"])
        rows = payload["files"]
        if not isinstance(rows, (tuple, list)) or len(rows) > 10000:
            raise ValueError("Unreasonable archived model-input inventory")
        files, relatives, total = {}, {}, 0
        for row in rows:
            path = self._absolute(row["path"])
            if path in files:
                raise ValueError("Duplicate archived model-input path")
            text = row["content_hex"]
            if not isinstance(text, str) or len(text) > 2 * MAX_INPUT_BYTES:
                raise ValueError("Archived model input exceeds the bounded input budget")
            total += len(text) // 2
            if total > MAX_INPUT_BYTES:
                raise ValueError("Archived model inputs exceed the bounded input budget")
            content = bytes.fromhex(text)
            if sha256(content).hexdigest() != row["sha256"]:
                raise ValueError("Archived model-input content checksum mismatch")
            from temsim.input_array_validation import validate_input_arrays
            validate_input_arrays(path, content, MAX_INPUT_BYTES)
            relative = row.get("config_relative")
            if relative is not None:
                portable = PurePosixPath(relative)
                if (not relative or portable.is_absolute() or ".." in portable.parts
                        or "\\" in relative or ":" in relative or portable.as_posix() != relative):
                    raise ValueError("Unsafe configuration archive entry")
                if relative in relatives or path != self.config_root / Path(relative):
                    raise ValueError("Ambiguous archived configuration path")
                relatives[relative] = path
            files[path] = content
        self._files = MappingProxyType(files)
        self._relatives = MappingProxyType(relatives)

    def __deepcopy__(self, memo):
        return self

    @staticmethod
    def _absolute(value):
        if not isinstance(value, str) or "\0" in value:
            raise ValueError("Invalid archived model-input path")
        path = Path(value)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("Archived model-input provenance must be absolute")
        return path

    def original_path(self, path):
        from temsim.paths import CONFIG_ROOT
        path = Path(path).expanduser().absolute()
        if path in self._files or path.is_relative_to(self.config_root):
            return path
        current_root = Path(CONFIG_ROOT).absolute()
        if path.is_relative_to(current_root):
            return self.config_root / path.relative_to(current_root)
        return path

    def read(self, path):
        original = self.original_path(path)
        try:
            return self._files[original]
        except KeyError as exc:
            raise FileNotFoundError(f"Input is absent from the selected archive: {original}") from exc

    def assert_current_runtime(self):
        if self.payload.get("runtime") != runtime_identity():
            raise ValueError("Archive runtime dependencies changed; migrate inputs explicitly before recalculating")

    def contains(self, path):
        return self.original_path(path) in self._files

    def paths(self, directory, pattern, *, recursive=False):
        directory = self.original_path(directory)
        return tuple(sorted(path for path in self._files
            if path.is_relative_to(directory)
            and (recursive or len(path.relative_to(directory).parts) == len(PurePosixPath(pattern).parts))
            and path.relative_to(directory).match(pattern)))


def archive_for(payload):
    """Small bounded resolver reuse; never caches mutable caller payloads."""
    if payload is None:
        return None
    identity = json_digest(payload)
    with _LOCK:
        existing = _RESOLVERS.get(identity)
        if existing is not None:
            return existing
    resolver = InputArchive(payload)
    with _LOCK:
        _RESOLVERS[identity] = resolver
    return resolver


def active_archive():
    return _ACTIVE.get()


def archive_payload(state):
    payload = getattr(state, "_archive_inputs", None)
    if payload is None and isinstance(getattr(state, "_archive_resolver", None), InputArchive):
        payload = state._archive_resolver.payload
    if payload is None and isinstance(state, Mapping):
        payload = state.get("archive_inputs")
    if payload is None:
        graph = getattr(state, "graph", None)
        if isinstance(graph, Mapping):
            payload = graph.get("archived_inputs")
    if payload is None:
        for name in ("checkpoint", "snapshot", "start_snapshot", "instrument_snapshot", "recipe", "payload", "state_payload", "source_state"):
            nested = getattr(state, name, None)
            if nested is not None and nested is not state:
                payload = archive_payload(nested)
                if payload is not None:
                    break
    return payload


def bind_archive(state, payload):
    resolver = archive_for(payload)
    state._archive_inputs, state._archive_resolver = resolver.payload, resolver
    sample = getattr(state, "sample", None)
    if sample is not None:
        sample._archive_resolver = resolver
    return resolver


@contextmanager
def input_scope(state, *, inherit=True):
    payload = archive_payload(state)
    cached = getattr(state, "_archive_resolver", None)
    resolver = (cached if cached is not None and cached.payload is payload else archive_for(payload)) if payload is not None else (_ACTIVE.get() if inherit else None)
    if (payload is not None and hasattr(state, "__dict__")
            and not getattr(getattr(type(state), "__dataclass_params__", None), "frozen", False)):
        if hasattr(state, "_archive_inputs") or hasattr(state, "electron_gun"):
            state._archive_inputs = resolver.payload
        state._archive_resolver = resolver
    token = _ACTIVE.set(resolver)
    try:
        yield resolver
    finally:
        _ACTIVE.reset(token)


def using_state_inputs(function):
    """Bind existing public operations to their captured state or owner."""
    parameters = signature(function)
    @wraps(function)
    def wrapped(*args, **kwargs):
        values = parameters.bind_partial(*args, **kwargs).arguments
        state = next((values[key] for key in ("state", "s", "sample", "snapshot", "request", "checkpoint", "recipe", "bank", "manifest")
                      if key in values), None)
        if state is None:
            owner = values.get("self")
            state = getattr(owner, "state", getattr(owner, "_state", getattr(owner, "_model_state", owner)))
        # A component supplied by an enclosing state operation inherits that
        # operation's resolver. An explicit live State always clears it.
        with input_scope(state, inherit="sample" in values) as resolver:
            if resolver is not None:
                resolver.assert_current_runtime()
            return function(*args, **kwargs)
    return wrapped


def scoped_lru_cache(*, maxsize=128):
    """Parsed live inputs and each archived input set have separate cache keys."""
    def decorate(function):
        @lru_cache(maxsize=maxsize)
        def cached(identity, args, kwargs):
            return function(*args, **dict(kwargs))
        @wraps(function)
        def wrapped(*args, **kwargs):
            resolver = _ACTIVE.get()
            return cached(resolver.identity if resolver is not None else "live", args, tuple(sorted(kwargs.items())))
        wrapped.cache_clear = cached.cache_clear
        wrapped.cache_info = cached.cache_info
        return wrapped
    return decorate


def read_bytes(path):
    resolver = _ACTIVE.get()
    return resolver.read(path) if resolver is not None else Path(path).read_bytes()


def read_text(path, encoding="utf-8"):
    return read_bytes(path).decode(encoding)


def open_input(path, mode="rb", *, encoding=None):
    if mode not in {"r", "rb"}:
        raise ValueError("Model-input streams are read-only")
    if _ACTIVE.get() is None:
        return Path(path).open(mode, encoding=encoding)
    data = read_bytes(path)
    return BytesIO(data) if mode == "rb" else StringIO(data.decode(encoding or "utf-8"))


def is_file(path):
    resolver = _ACTIVE.get()
    return resolver.contains(path) if resolver is not None else Path(path).is_file()


def input_paths(directory, pattern="*", *, recursive=False):
    resolver = _ACTIVE.get()
    if resolver is not None:
        return resolver.paths(directory, pattern, recursive=recursive)
    return tuple(Path(directory).rglob(pattern) if recursive else Path(directory).glob(pattern))


def capture_input_archive(state):
    """Explicit portable-input capture; defaults and existing results unchanged."""
    from temsim.calculation_manifest import capture_external_input_identities
    from temsim.paths import CONFIG_ROOT
    existing = archive_payload(state)
    if existing is not None:
        return archive_for(existing).payload
    paths = set(Path(CONFIG_ROOT).rglob("*"))
    assembly = getattr(state, "_resolved_assembly", None)
    if assembly is not None:
        paths.update(Path(assembly.root).rglob("*.toml"))
    identities = capture_external_input_identities(state)
    for identity in identities:
        if not identity.available:
            raise ValueError(f"Cannot capture absent model input: {identity.role}")
        paths.add(Path(identity.path))
    rows, total = [], 0
    root = Path(CONFIG_ROOT).resolve()
    for path in sorted(paths):
        if path.is_symlink() or path.is_junction():
            raise ValueError("Portable input capture rejects symbolic links and junctions")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Portable input capture requires regular files without links")
        size = path.stat().st_size
        total += size
        if total > MAX_INPUT_BYTES or len(rows) >= 10000:
            raise ValueError("Portable input capture exceeds the 96 MiB / 10000-file input budget")
        data = path.read_bytes()
        if len(data) != size:
            raise ValueError("Model input changed while capturing the archive")
        path = path.resolve()
        rows.append(dict(path=str(path), config_relative=path.relative_to(root).as_posix() if path.is_relative_to(root) else None,
                         content_hex=data.hex(), sha256=sha256(data).hexdigest()))
    by_path = {row["path"]: row for row in rows}
    for identity in identities:
        if by_path[identity.path]["sha256"] != identity.sha256:
            raise ValueError("External model changed while capturing portable inputs")
    payload = dict(schema=ARCHIVE_SCHEMA, config_root=str(root), files=rows, runtime=runtime_identity())
    payload["digest"] = json_digest(payload)
    return InputArchive(payload).payload


def restore_profile_inputs(function):
    """Retain archive ownership across the existing profile reconstruction path."""
    @wraps(function)
    def wrapped(payload):
        with input_scope(payload, inherit=False):
            result = function(payload)
        archive = archive_payload(payload)
        if archive is not None:
            bind_archive(result, archive)
        return result
    return wrapped
