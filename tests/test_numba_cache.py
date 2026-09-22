"""Native caches follow package code, while explicit/active settings survive."""
from pathlib import Path

import pytest

from temsim import numba_cache as cache


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    root = tmp_path / "sources"
    root.mkdir()
    (root / "__init__.py").write_text("# package\n", encoding="utf-8")
    (root / "submodule").mkdir()
    (root / "submodule" / "dependency.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setattr(cache, "_numba_imported", lambda: False)
    return root


def test_same_source_reuses_directory_and_dependency_changes_select_new_one(source_tree, monkeypatch):
    first = Path(cache.configure_numba_cache(source_root=source_tree))
    sentinel = first / "retained-cache"
    sentinel.write_bytes(b"keep")
    monkeypatch.delenv("NUMBA_CACHE_DIR")
    assert Path(cache.configure_numba_cache(source_root=source_tree)) == first
    # Same file length and timestamp are deliberately insufficient identities.
    dependency = source_tree / "submodule" / "dependency.py"
    timestamp = dependency.stat().st_mtime_ns
    dependency.write_text("VALUE = 2\n", encoding="utf-8")
    cache.os.utime(dependency, ns=(timestamp, timestamp))
    monkeypatch.delenv("NUMBA_CACHE_DIR")
    second = Path(cache.configure_numba_cache(source_root=source_tree))
    assert first != second and first.parent == second.parent
    assert sentinel.read_bytes() == b"keep"


def test_source_identity_includes_paths_additions_removals_but_ignores_outputs(source_tree):
    original = cache._source_digest(source_tree)
    (source_tree / "output.npy").write_bytes(b"not source")
    assert cache._source_digest(source_tree) == original
    added = source_tree / "extra.py"
    added.write_text("pass\n", encoding="utf-8")
    with_added = cache._source_digest(source_tree)
    assert with_added != original
    renamed = source_tree / "renamed.py"
    added.rename(renamed)
    assert cache._source_digest(source_tree) != with_added
    renamed.unlink()
    assert cache._source_digest(source_tree) == original


@pytest.mark.parametrize("explicit", ["", "custom/cache"])
def test_explicit_cache_selection_is_preserved_even_when_numba_is_active(source_tree, monkeypatch, explicit):
    monkeypatch.setenv("NUMBA_CACHE_DIR", explicit)
    monkeypatch.setattr(cache, "_numba_imported", lambda: True)
    monkeypatch.setattr(cache, "_source_digest", lambda _: pytest.fail("Do not inspect explicitly selected caches"))
    assert cache.configure_numba_cache(source_root=source_tree) == explicit
    assert cache.os.environ["NUMBA_CACHE_DIR"] == explicit


def test_late_initialization_does_not_mutate_active_numba_configuration(source_tree, monkeypatch):
    monkeypatch.setattr(cache, "_numba_imported", lambda: True)
    monkeypatch.setattr(cache, "_source_digest", lambda _: pytest.fail("No late cache selection"))
    assert cache.configure_numba_cache(source_root=source_tree) is None
    assert "NUMBA_CACHE_DIR" not in cache.os.environ


def test_unwritable_local_directory_uses_temp_fallback(source_tree, monkeypatch, tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_bytes(b"file instead of a directory")
    monkeypatch.setenv("LOCALAPPDATA", str(blocked))
    monkeypatch.setattr(cache.tempfile, "gettempdir", lambda: str(tmp_path / "temporary"))
    selected = Path(cache.configure_numba_cache(source_root=source_tree))
    assert selected.parent == tmp_path / "temporary" / "temsim" / "numba"
    assert selected.is_dir() and blocked.read_bytes() == b"file instead of a directory"


def test_unknown_source_or_unwritable_cache_does_not_fall_back_to_stale_code(source_tree, monkeypatch, tmp_path):
    with pytest.raises(OSError, match="No Python sources"):
        cache.configure_numba_cache(source_root=tmp_path / "absent")
    def deny(_path):
        raise PermissionError("controlled write denial")
    monkeypatch.setattr(cache, "_writable_directory", deny)
    with pytest.raises(OSError, match="code-versioned"):
        cache.configure_numba_cache(source_root=source_tree)
    assert "NUMBA_CACHE_DIR" not in cache.os.environ
