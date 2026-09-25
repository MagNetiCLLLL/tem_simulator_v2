"""Scalar navigation metadata only; result payload validation belongs to its codec."""
import json
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from pathlib import Path
from threading import Event
from time import monotonic

import pytest

from temsim import result_library
from temsim.result_library import ResultLibrary


def _process_writer(root, name, info, start, read_ready, release, completed, output, pause_write):
    """Spawn-safe writer; no Qt, calculation work or array data is involved."""
    try:
        library = ResultLibrary(root)
        if pause_write:
            original_write = library._write_index

            def write_after_other_writer_starts(document):
                read_ready.set()
                if not release.wait(10):
                    raise TimeoutError("Test writer was not released")
                original_write(document)

            library._write_index = write_after_other_writer_starts
        output.put(("ready", name))
        if not start.wait(10):
            raise TimeoutError("Test writer was not started")
        library.remember(name, info)
        output.put(("saved", name))
    except BaseException as exc:
        output.put(("error", name, repr(exc)))
    finally:
        completed.set()


def _process_exit_with_lock(root, ready):
    import os

    library = ResultLibrary(root)
    with library._transaction(create=True):
        ready.set()
        # Simulate a stopped application without context-manager cleanup.
        os._exit(0)


def export_info(tmp_path, name="export.temresult", **changes):
    package = tmp_path / name
    package.parent.mkdir(parents=True, exist_ok=True)
    package.write_bytes(b"opaque exported package; the index never decodes this")
    info = {
        "path": str(package), "identity": "a" * 64, "_package_digest": "b" * 64,
        "quality": "High accuracy", "target_z_mm": 1599.25,
        "resumable_through_z_mm": 1599.25,
        "saved_at_utc": "2026-09-22T12:34:56.123456+00:00",
    }
    return dict(info, **changes)


def test_empty_library_and_allocations_are_owned_without_creating_packages(tmp_path):
    library = ResultLibrary(tmp_path / "library")
    assert library.entries() == ()
    assert library.startup_entry() is None
    assert not library.root.exists()
    first, second = library.allocate_path(), library.allocate_path()
    assert first != second
    assert first.parent == second.parent == library.root / "results"
    assert first.suffix == second.suffix == ".temresult"
    assert first.parent.is_dir()
    assert not first.exists() and not second.exists()
    assert library.entries() == ()


def test_external_export_roundtrip_copies_only_detached_navigation_metadata(tmp_path):
    library = ResultLibrary(tmp_path / "library")
    info = export_info(tmp_path / "external")
    info.update(arrays=object(), executed_state=object(), size_bytes=999)
    entry = library.remember("  常用样品截面  ", info)
    assert entry["name"] == "常用样品截面"
    assert entry["path"] == str(Path(info["path"]).resolve())
    assert entry["size_bytes"] == Path(info["path"]).stat().st_size
    assert entry["result_identity"] == info["identity"]
    assert entry["package_digest"] == info["_package_digest"]
    assert set(entry) == {"id", "name", "path", "result_identity", "package_digest",
                          "quality", "target_z_mm", "resumable_through_z_mm",
                          "saved_at_utc", "size_bytes"}
    copy = dict(entry)
    entry["name"] = "changed by caller"
    info["identity"] = "c" * 64
    assert library.entries() == (copy,)
    assert ResultLibrary(library.root).entries() == (copy,)


def test_same_name_updates_stable_entry_and_startup_without_deleting_old_file(tmp_path):
    library = ResultLibrary(tmp_path / "library")
    first_info = export_info(tmp_path, "first.temresult")
    first = library.remember("saved", first_info)
    other = library.remember("other", export_info(tmp_path, "other.temresult"))
    library.set_startup(first["id"])
    replacement = library.remember("saved", export_info(
        tmp_path, "replacement.temresult", identity="c" * 64, _package_digest="d" * 64,
        target_z_mm=2500.0, resumable_through_z_mm=2490.0))
    assert replacement["id"] == first["id"]
    assert replacement["result_identity"] != first["result_identity"]
    assert library.entries() == (replacement, other)
    assert ResultLibrary(library.root).startup_entry() == replacement
    assert Path(first_info["path"]).exists()
    library.set_startup(None)
    assert ResultLibrary(library.root).startup_entry() is None


def test_separate_instances_see_each_successful_mutation(tmp_path):
    first, second = ResultLibrary(tmp_path / "library"), ResultLibrary(tmp_path / "library")
    a = first.remember("first", export_info(tmp_path, "first.temresult"))
    b = second.remember("second", export_info(tmp_path, "second.temresult"))
    first.set_startup(b["id"])
    assert second.entries() == (a, b)
    assert second.startup_entry() == b


def test_missing_export_is_retained_in_index_and_reads_never_open_packages(tmp_path, monkeypatch):
    library = ResultLibrary(tmp_path / "library")
    info = export_info(tmp_path)
    entry = library.remember("saved", info)
    library.set_startup(entry["id"])
    Path(info["path"]).unlink()
    original_open = Path.open

    def index_only(path, *args, **kwargs):
        assert path == library.index_path, "Navigation must not decode result packages"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", index_only)
    assert ResultLibrary(library.root).startup_entry() == entry
    assert library.entries() == (entry,)


@pytest.mark.parametrize("name", ["", "   ", "a\nb", "a\x00b", "x" * 201, None, 10])
def test_invalid_names_never_create_index(tmp_path, name):
    library = ResultLibrary(tmp_path / "library")
    with pytest.raises(ValueError, match="name"):
        library.remember(name, export_info(tmp_path))
    assert not library.index_path.exists()


@pytest.mark.parametrize("change", [
    {"identity": "unverified"}, {"_package_digest": "a" * 63},
    {"quality": ""}, {"target_z_mm": None}, {"target_z_mm": True},
    {"target_z_mm": float("nan")}, {"resumable_through_z_mm": float("inf")},
    {"resumable_through_z_mm": "1599.25"}, {"saved_at_utc": None},
    {"saved_at_utc": "2026-09-22T12:34:56"},
    {"saved_at_utc": "2026-09-22T12:34:56+01:00"},
])
def test_invalid_export_metadata_never_changes_existing_index(tmp_path, change):
    library = ResultLibrary(tmp_path / "library")
    first = library.remember("saved", export_info(tmp_path))
    original = library.index_path.read_bytes()
    with pytest.raises(ValueError):
        library.remember("saved", export_info(tmp_path, "invalid.temresult", **change))
    assert library.entries() == (first,)
    assert library.index_path.read_bytes() == original


def test_missing_empty_or_directory_export_and_incomplete_metadata_are_rejected(tmp_path):
    library = ResultLibrary(tmp_path / "library")
    info = export_info(tmp_path)
    Path(info["path"]).unlink()
    for path in (info["path"], tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            library.remember("saved", dict(info, path=path))
    Path(info["path"]).touch()
    with pytest.raises(ValueError, match="size"):
        library.remember("saved", info)
    for invalid in ({}, None, {"path": info["path"]}):
        with pytest.raises(ValueError, match="metadata"):
            library.remember("saved", invalid)
    assert library.entries() == ()


@pytest.mark.parametrize("corrupt", [
    b"{", b"\xff", b"[]", b'{"schema":"unknown"}',
    b'{"schema":"temsim-result-library-v1","schema":"temsim-result-library-v1",'
    b'"entries":[],"startup_entry_id":null}',
    b'{"schema":"temsim-result-library-v1","entries":NaN,"startup_entry_id":null}',
])
def test_bad_index_is_reported_and_never_overwritten(tmp_path, corrupt):
    library = ResultLibrary(tmp_path / "library")
    library.root.mkdir()
    library.index_path.write_bytes(corrupt)
    with pytest.raises(ValueError):
        ResultLibrary(library.root)
    info = export_info(tmp_path)
    for action in (library.entries, library.startup_entry, library.allocate_path,
                   lambda: library.set_startup(None), lambda: library.remember("saved", info)):
        with pytest.raises(ValueError):
            action()
        assert library.index_path.read_bytes() == corrupt


@pytest.mark.parametrize("mutation", [
    lambda doc: doc.update(extra="unsupported"),
    lambda doc: doc.update(entries={}),
    lambda doc: doc.update(startup_entry_id="e" * 32),
    lambda doc: doc["entries"].append(dict(doc["entries"][0])),
    lambda doc: doc["entries"][0].update(extra="unsupported"),
    lambda doc: doc["entries"][0].update(path="relative.temresult"),
    lambda doc: doc["entries"][0].update(size_bytes=True),
])
def test_strict_structure_rejects_unknown_fields_duplicates_and_dangling_startup(tmp_path, mutation):
    library = ResultLibrary(tmp_path / "library")
    library.remember("saved", export_info(tmp_path))
    document = json.loads(library.index_path.read_text(encoding="utf-8"))
    mutation(document)
    payload = json.dumps(document).encode("utf-8")
    library.index_path.write_bytes(payload)
    with pytest.raises(ValueError):
        library.set_startup(None)
    assert library.index_path.read_bytes() == payload


def test_size_and_entry_limits_do_not_overwrite_previous_index(tmp_path, monkeypatch):
    library = ResultLibrary(tmp_path / "library")
    first = library.remember("saved", export_info(tmp_path))
    original = library.index_path.read_bytes()
    monkeypatch.setattr(result_library, "MAX_ENTRIES", 1)
    with pytest.raises(ValueError, match="limit"):
        library.remember("extra", export_info(tmp_path, "second.temresult"))
    assert library.entries() == (first,)
    monkeypatch.setattr(result_library, "MAX_ENTRIES", 1000)
    monkeypatch.setattr(result_library, "MAX_INDEX_BYTES", len(original) + 50)
    with pytest.raises(ValueError, match="size limit"):
        library.remember("extra", export_info(tmp_path, "second.temresult"))
    assert library.index_path.read_bytes() == original
    monkeypatch.setattr(result_library, "MAX_INDEX_BYTES", len(original) - 1)
    with pytest.raises(ValueError, match="size limit"):
        library.entries()
    assert library.index_path.read_bytes() == original


@pytest.mark.parametrize("failure", ["fsync", "replace"])
def test_atomic_write_failure_preserves_index_and_default_selection(tmp_path, monkeypatch, failure):
    library = ResultLibrary(tmp_path / "library")
    first = library.remember("saved", export_info(tmp_path))
    library.set_startup(first["id"])
    original = library.index_path.read_bytes()

    def fail(*_args):
        raise OSError("disk write failed")

    monkeypatch.setattr(result_library.os, failure, fail)
    with pytest.raises(OSError, match="disk write failed"):
        library.remember("saved", export_info(tmp_path, "second.temresult", identity="c" * 64))
    with pytest.raises(OSError, match="disk write failed"):
        library.set_startup(None)
    assert library.index_path.read_bytes() == original
    assert library.startup_entry() == first
    assert list(library.root.glob(".result-library-*.tmp")) == []


def test_unknown_startup_id_does_not_change_selection(tmp_path):
    library = ResultLibrary(tmp_path / "library")
    first = library.remember("saved", export_info(tmp_path))
    library.set_startup(first["id"])
    original = library.index_path.read_bytes()
    with pytest.raises(ValueError, match="does not exist"):
        library.set_startup("f" * 32)
    assert library.startup_entry() == first
    assert library.index_path.read_bytes() == original


def test_owned_paths_reject_files_in_place_of_directories(tmp_path):
    file = tmp_path / "file"
    file.write_bytes(b"existing data")
    with pytest.raises(ValueError, match="directory"):
        ResultLibrary(file)
    library = ResultLibrary(tmp_path / "library")
    library.root.mkdir()
    results = library.root / "results"
    results.write_bytes(b"existing data")
    with pytest.raises(ValueError, match="directory"):
        library.allocate_path()
    assert results.read_bytes() == b"existing data"
    library.index_path.mkdir()
    with pytest.raises(ValueError, match="JSON file"):
        library.entries()


def test_independent_instances_concurrently_preserve_both_updates(tmp_path, monkeypatch):
    root = tmp_path / "library"
    first, second = ResultLibrary(root), ResultLibrary(root)
    original_write = first._write_index
    read_ready, release, second_done = Event(), Event(), Event()
    first_info = export_info(tmp_path, "first.temresult")
    second_info = export_info(tmp_path, "second.temresult")

    def delayed_write(document):
        read_ready.set()
        assert release.wait(10)
        original_write(document)

    def other_writer():
        try:
            return second.remember("second", second_info)
        finally:
            second_done.set()

    monkeypatch.setattr(first, "_write_index", delayed_write)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_write = pool.submit(first.remember, "first", first_info)
        try:
            assert read_ready.wait(5)
            second_write = pool.submit(other_writer)
            # Before the fix, the second writer completed against the same old
            # index and its entry was then lost when the first writer resumed.
            second_done.wait(0.3)
        finally:
            release.set()
        first_write.result(timeout=10)
        second_write.result(timeout=10)
    assert {row["name"] for row in ResultLibrary(root).entries()} == {"first", "second"}


def test_independent_processes_concurrently_preserve_both_updates(tmp_path):
    context = multiprocessing.get_context("spawn")
    root = tmp_path / "library"
    starts = [context.Event(), context.Event()]
    done = [context.Event(), context.Event()]
    read_ready, release, output = context.Event(), context.Event(), context.Queue()
    children = [context.Process(target=_process_writer, args=(
        root, name, export_info(tmp_path, name + ".temresult"), starts[index],
        read_ready, release, done[index], output, index == 0,
    )) for index, name in enumerate(("first", "second"))]
    try:
        for child in children:
            child.start()
        assert {output.get(timeout=15), output.get(timeout=15)} == {
            ("ready", "first"), ("ready", "second")}
        starts[0].set()
        assert read_ready.wait(5)
        starts[1].set()
        done[1].wait(0.3)
        release.set()
        assert {output.get(timeout=15), output.get(timeout=15)} == {
            ("saved", "first"), ("saved", "second")}
    finally:
        for start in starts:
            start.set()
        release.set()
        for child in children:
            if child.pid is not None:
                child.join(timeout=15)
                assert not child.is_alive()
                assert child.exitcode == 0
        output.close()
        output.join_thread()
    assert {row["name"] for row in ResultLibrary(root).entries()} == {"first", "second"}


def test_contended_transaction_times_out_without_changes_then_can_retry(tmp_path, monkeypatch):
    first, second = ResultLibrary(tmp_path / "library"), ResultLibrary(tmp_path / "library")
    info = export_info(tmp_path)
    saved = first.remember("saved", info)
    first.set_startup(saved["id"])
    original = first.index_path.read_bytes()
    monkeypatch.setattr(result_library, "LOCK_TIMEOUT_SECONDS", 0.05)
    with first._transaction(create=True):
        started = monotonic()
        with pytest.raises(TimeoutError, match="result library"):
            second.remember("second", info)
        assert monotonic() - started < 2.0
        with pytest.raises(TimeoutError, match="result library"):
            second.entries()
        assert first.index_path.read_bytes() == original
    second.remember("second", info)
    assert len(first.entries()) == 2
    assert first.startup_entry() == saved


def test_process_exit_releases_lock_without_deleting_the_lock_file(tmp_path):
    context = multiprocessing.get_context("spawn")
    library = ResultLibrary(tmp_path / "library")
    first = library.remember("first", export_info(tmp_path, "first.temresult"))
    ready = context.Event()
    child = context.Process(target=_process_exit_with_lock, args=(library.root, ready))
    child.start()
    try:
        assert ready.wait(10)
    finally:
        child.join(timeout=15)
        assert not child.is_alive()
        assert child.exitcode == 0
    assert (library.root / ".index.lock").is_file()
    second = ResultLibrary(library.root).remember("second", export_info(tmp_path, "second.temresult"))
    assert library.entries() == (first, second)
