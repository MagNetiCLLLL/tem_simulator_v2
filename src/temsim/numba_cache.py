"""Select native-code caches for the complete installed Python implementation.

Numba's individual-function cache does not invalidate all imported dependencies.
Keeping each package revision in its own directory avoids mixing those cached
callers with another revision's callees. This is startup configuration only;
already imported Numba and an explicit user cache selection are left alone.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import tempfile


def _numba_imported() -> bool:
    return "numba" in sys.modules or "numba.core.config" in sys.modules


def _source_digest(source_root: Path) -> str:
    """Include every Python dependency inside the installed temsim package."""
    source_root = Path(source_root)
    paths = sorted(source_root.rglob("*.py"), key=lambda path: path.relative_to(source_root).as_posix())
    if not paths:
        raise OSError(f"No Python sources available for native cache identity: {source_root}")
    digest = hashlib.sha256(b"temsim-native-cache-v1\0")
    for path in paths:
        relative = path.relative_to(source_root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _candidate_bases():
    local = os.environ.get("LOCALAPPDATA")
    if local:
        yield Path(local) / "temsim" / "numba"
    yield Path(tempfile.gettempdir()) / "temsim" / "numba"


def _writable_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    # Test write access without retaining a probe file or touching older caches.
    with tempfile.TemporaryFile(dir=path):
        pass
    return path


def configure_numba_cache(*, source_root: Path | None = None) -> str | None:
    """Return the selected path, or None when Numba was already imported.

    Explicit NUMBA_CACHE_DIR (even an empty value) is never replaced. Late
    callers cannot safely change an active compiler's cache configuration; they
    retain its existing configuration and need a fresh process for migration.
    No existing cache, calculation, or acquisition file is removed.
    """
    if "NUMBA_CACHE_DIR" in os.environ:
        return os.environ["NUMBA_CACHE_DIR"]
    if _numba_imported():
        return None
    digest = _source_digest(Path(__file__).parent if source_root is None else source_root)
    error = None
    for base in _candidate_bases():
        try:
            selected = str(_writable_directory(base / digest))
        except OSError as exc:
            error = exc
            continue
        os.environ["NUMBA_CACHE_DIR"] = selected
        return selected
    raise OSError("Cannot create a writable, code-versioned Numba cache directory") from error
