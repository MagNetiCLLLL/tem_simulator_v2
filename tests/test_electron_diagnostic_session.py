"""Independent diagnostic histories: safe round trips and source boundaries."""
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

import temsim.electron_diagnostic_session as io
from temsim.magnetic_test_particle import TestElectronSettings, TestElectronTrajectory, electron_momentum_and_speed


def _trajectory(settings, *, reason="path_limit", completed=True, physical="physical", numerical="numerical"):
    """Analytical straight-line fixture; no field solver or subprocess is run."""
    momentum, speed = electron_momentum_and_speed(settings.kinetic_energy_ev)
    path = np.linspace(0., settings.max_path_length_m, 4)
    positions = np.tile(settings.position_m, (4, 1))
    positions[:, 2] += path
    direction = np.tile([0., 0., 1.], (4, 1))
    return TestElectronTrajectory(positions, direction, path / speed, path, 0., 0., reason, completed,
        3, np.full(4, settings.kinetic_energy_ev), np.zeros(4), np.full(4, speed),
        direction * momentum, ("Analytical straight-line fixture, not microscope qualification",),
        physical, numerical, "execution")


def _session(*, reason="path_limit", completed=True):
    settings = TestElectronSettings(max_path_length_m=.002)
    trace = _trajectory(settings, reason=reason, completed=completed)
    record = io.DiagnosticElectronRecord("electron-1", "Diagnostic α", "#ffd166", settings,
        trajectory=trace, trajectory_settings=settings, state="completed" if completed else "incomplete")
    return io.DiagnosticSession((record,), io.DiagnosticDependency("saved-result-7", "physical", "numerical",
        "transport", ("Finite synthetic fixture",), ((-.01, -.01, 0.), (.01, .01, .1))),
        io.DiagnosticDisplayState(72., (-1., 101.), (-3., 3.), True, record.key, False))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _rewrite(path, *, mutate=None, transform_members=None):
    with ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    document = json.loads(members["manifest.json"])
    if mutate is not None:
        mutate(document["payload"])
    if transform_members is not None:
        transform_members(members, document["payload"])
    document["sha256"] = sha256(_canonical(document["payload"])).hexdigest()
    members["manifest.json"] = _canonical(document)
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def test_roundtrip_preserves_complete_history_identity_display_and_exact_arrays(tmp_path):
    original = _session()
    path = io.save_diagnostic_session(original, tmp_path / "lesson.temdiag")
    restored = io.load_diagnostic_session(path)
    assert restored.dependency == original.dependency
    assert restored.display == original.display
    assert restored.created_at_utc == original.created_at_utc
    assert restored.app_version == original.app_version
    assert restored.records[0].settings == original.records[0].settings
    assert restored.records[0].trajectory_settings == original.records[0].trajectory_settings
    for name in io._ARRAYS:
        actual = getattr(restored.records[0].trajectory, name)
        np.testing.assert_array_equal(actual, getattr(original.records[0].trajectory, name))
        assert not actual.flags.writeable
        with pytest.raises(ValueError):
            actual.setflags(write=True)
    assert restored.records[0].trajectory.execution_identity == "execution"
    assert path.stat().st_size < 15_000


@pytest.mark.parametrize("reason,complete", [("electrode:anode", True), ("domain_exit", True),
    ("cancelled", False), ("in_progress", False), ("step_limit", False)])
def test_termination_completion_and_partial_history_are_not_changed(tmp_path, reason, complete):
    path = io.save_diagnostic_session(_session(reason=reason, completed=complete), tmp_path / "history.temdiag")
    record = io.load_diagnostic_session(path).records[0]
    assert record.trajectory.reason == reason
    assert record.trajectory.completed is complete
    assert record.state == ("completed" if complete else "incomplete")


def test_uncomputed_session_is_readable_without_any_captured_fields(tmp_path):
    record = io.DiagnosticElectronRecord("new", "Electron", "#abcdef", TestElectronSettings())
    restored = io.load_diagnostic_session(io.save_diagnostic_session(io.DiagnosticSession((record,)), tmp_path / "new.temdiag"))
    assert restored.records == (record,)
    assert restored.records[0].trajectory is None
    assert not restored.dependency.matches()


def test_unknown_historical_identities_are_not_filled_from_dependency(tmp_path):
    original = _session()
    record = replace(original.records[0], trajectory=replace(original.records[0].trajectory,
        physical_field_identity=None, numerical_field_identity=None, execution_identity=None), state="previous")
    saved = replace(original, records=(record,), dependency=io.DiagnosticDependency())
    restored = io.load_diagnostic_session(io.save_diagnostic_session(saved, tmp_path / "history.temdiag"))
    assert restored.records[0].trajectory.physical_field_identity is None
    assert not restored.dependency.matches(physical_identity="current", numerical_identity="current")


def test_changed_controls_keep_separate_executed_settings_and_old_identity(tmp_path):
    original = _session()
    prior = original.records[0]
    changed = replace(prior, settings=replace(prior.settings, polar_angle_deg=1.), revision=1, state="previous")
    restored = io.load_diagnostic_session(io.save_diagnostic_session(replace(original, records=(changed,)), tmp_path / "edited.temdiag"))
    record = restored.records[0]
    assert record.settings.polar_angle_deg == 1.
    assert record.trajectory_settings.polar_angle_deg == 0.
    assert record.state == "previous"
    assert record.trajectory.physical_field_identity == "physical"
    with pytest.raises(ValueError, match="Completed"):
        replace(changed, state="completed")


def test_independent_records_keep_their_original_dependencies_after_mixed_recalculation(tmp_path):
    original = _session()
    record = original.records[0]
    newer = replace(record, key="electron-2", trajectory=replace(record.trajectory,
        physical_field_identity="physical-new", numerical_field_identity="numerical-new"),
        dependency=io.DiagnosticDependency(physical_identity="physical-new", numerical_identity="numerical-new"))
    saved = replace(original, records=(record, newer))
    loaded = io.load_diagnostic_session(io.save_diagnostic_session(saved, tmp_path / "mixed.temdiag"))
    assert loaded.records[0].dependency is None
    assert loaded.records[1].dependency.physical_identity == "physical-new"
    assert loaded.dependency.physical_identity == "physical"
    assert not loaded.dependency.matches(physical_identity="physical-new", numerical_identity="numerical-new")
    with pytest.raises(ValueError, match="declared field"):
        replace(saved, records=(replace(newer, dependency=None),))


def test_unknown_electric_potential_is_explicit_and_never_replaced_by_zero(tmp_path):
    original = _session()
    unknown = replace(original.records[0].trajectory, electrostatic_potential_v=np.full(4, np.nan),
        energy_invariant_error_ev=np.nan, energy_invariant_relative_error=np.nan)
    saved = replace(original, records=(replace(original.records[0], trajectory=unknown),))
    path = io.save_diagnostic_session(saved, tmp_path / "unknown.temdiag")
    with ZipFile(path) as archive:
        raw = archive.read("manifest.json")
        assert b'"energy_invariant_error_ev":null' in raw
        assert b"NaN" not in raw
    loaded = io.load_diagnostic_session(path).records[0].trajectory
    assert np.isnan(loaded.electrostatic_potential_v).all()
    assert np.isnan(loaded.energy_invariant_error_ev)


@pytest.mark.parametrize("mutation,expected", [
    (lambda p: p.update(version=999), "schema/version"),
    (lambda p: p.update(version=True), "schema/version"),
    (lambda p: p.pop("dependency"), "fields"),
    (lambda p: p["records"].append(p["records"][0]), "array|archive|Duplicate"),
    (lambda p: p["records"][0]["settings"].update(kinetic_energy_ev="nan"), "finite"),
    (lambda p: p["records"][0].update(checked="false"), "boolean"),
    (lambda p: p["display"].update(overlay="false"), "boolean"),
    (lambda p: p["records"][0]["trajectory"].update(completed=True, reason="cancelled"), "unfinished"),
    (lambda p: p["records"][0]["settings"].update(polar_angle_deg=2.), "Completed"),
    (lambda p: p["display"].update(selected_key="absent"), "Selected"),
    (lambda p: p["records"][0]["trajectory"]["arrays"]["positions_m"].update(path="../escape.npy"), "path"),
    (lambda p: p["records"][0]["trajectory"]["arrays"]["positions_m"].update(dtype="object"), "dtype"),
    (lambda p: p["records"][0]["trajectory"].update(steps=10**12), "point-count"),
])
def test_corrupt_metadata_is_rejected_before_returning_session(tmp_path, mutation, expected):
    path = io.save_diagnostic_session(_session(), tmp_path / "corrupt.temdiag")
    _rewrite(path, mutate=mutation)
    with pytest.raises(ValueError, match=expected):
        io.load_diagnostic_session(path)


def test_manifest_and_array_checksum_failures(tmp_path):
    path = io.save_diagnostic_session(_session(), tmp_path / "corrupt.temdiag")
    with ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members["manifest.json"] = members["manifest.json"].replace(b"Diagnostic", b"diagnostic", 1)
    with ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError, match="manifest checksum"):
        io.load_diagnostic_session(path)
    io.save_diagnostic_session(_session(), path)
    def corrupt(members, payload):
        name = payload["records"][0]["trajectory"]["arrays"]["positions_m"]["path"]
        members[name] = members[name][:-1] + bytes([members[name][-1] ^ 1])
    _rewrite(path, transform_members=corrupt)
    with pytest.raises(ValueError, match="array checksum"):
        io.load_diagnostic_session(path)


@pytest.mark.parametrize("case", ["object", "huge_shape", "trailing_bytes", "infinite_positions"])
def test_malicious_npy_is_validated_without_pickle_or_unbounded_allocation(tmp_path, monkeypatch, case):
    path = io.save_diagnostic_session(_session(), tmp_path / "bad-array.temdiag")
    def corrupt(members, payload):
        descriptor = payload["records"][0]["trajectory"]["arrays"]["positions_m"]
        stream = BytesIO()
        if case == "object":
            np.save(stream, np.array([["bad"] * 3] * 4, dtype=object), allow_pickle=True)
        elif case == "huge_shape":
            np.lib.format.write_array_header_1_0(stream, {"descr": "<f8", "fortran_order": False, "shape": (10**12, 3)})
        elif case == "trailing_bytes":
            stream.write(members[descriptor["path"]] + b"unwanted")
        else:
            np.save(stream, np.full((4, 3), np.inf), allow_pickle=False)
        data = stream.getvalue()
        descriptor["sha256"] = sha256(data).hexdigest()
        members[descriptor["path"]] = data
    _rewrite(path, transform_members=corrupt)
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("Generic NPY/pickle loader must not be used"))
    with pytest.raises(ValueError, match="shape/dtype|byte count|finite"):
        io.load_diagnostic_session(path)


@pytest.mark.parametrize("kind", ["archive", "unpacked", "manifest", "memory", "records", "points"])
def test_independent_read_budgets(tmp_path, kind):
    original = _session()
    if kind == "records":
        original = replace(original, records=(original.records[0], replace(original.records[0], key="electron-2")))
    path = io.save_diagnostic_session(original, tmp_path / "bounded.temdiag")
    options = {"archive": {"maximum_archive_bytes": 100}, "unpacked": {"maximum_unpacked_bytes": 100},
        "manifest": {"maximum_manifest_bytes": 100}, "memory": {"maximum_memory_bytes": 100}, "records": {"maximum_records": 1},
        "points": {"maximum_points_per_trajectory": 3}}
    with pytest.raises(ValueError, match="limit|oversized|excessive|Too many"):
        io.load_diagnostic_session(path, limits=replace(io.DEFAULT_LIMITS, **options[kind]))


def test_extra_path_and_duplicate_zip_entry_rejected_without_extraction(tmp_path):
    path = io.save_diagnostic_session(_session(), tmp_path / "unsafe.temdiag")
    with ZipFile(path, "a") as archive:
        archive.writestr("../escaped.txt", "never extract")
    with pytest.raises(ValueError, match="unsafe"):
        io.load_diagnostic_session(path)
    assert not (tmp_path.parent / "escaped.txt").exists()
    io.save_diagnostic_session(_session(), path)
    with ZipFile(path, "a") as archive, pytest.warns(UserWarning, match="Duplicate"):
        archive.writestr("manifest.json", "{}")
    with pytest.raises(ValueError, match="duplicate"):
        io.load_diagnostic_session(path)


def test_atomic_failure_preserves_existing_file_and_cleans_temporary(tmp_path, monkeypatch):
    path = io.save_diagnostic_session(_session(), tmp_path / "safe.temdiag")
    before = path.read_bytes()
    def denied(*args):
        raise PermissionError("injected replacement failure")
    monkeypatch.setattr(io.os, "replace", denied)
    with pytest.raises(PermissionError, match="injected"):
        io.save_diagnostic_session(_session(), path)
    assert path.read_bytes() == before
    assert set(tmp_path.iterdir()) == {path}


def test_save_limit_failure_and_no_overwrite_preserve_existing_file(tmp_path):
    path = io.save_diagnostic_session(_session(), tmp_path / "safe.temdiag")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="limit"):
        io.save_diagnostic_session(_session(), path, limits=replace(io.DEFAULT_LIMITS, maximum_unpacked_bytes=20))
    with pytest.raises(FileExistsError):
        io.save_diagnostic_session(_session(), path, overwrite=False)
    assert path.read_bytes() == before
    other = io.save_diagnostic_session(_session(), tmp_path / "new.temdiag", overwrite=False)
    assert io.load_diagnostic_session(other).records[0].key == "electron-1"


def test_main_result_profile_source_and_continuation_boundaries_reject_diagnostics(tmp_path):
    from temsim.particle_section_io import load_section_result
    from temsim.working_point_archive import WorkingPointArchiveIndex
    from temsim.profile_io import read_profile
    from temsim.physics.particle_sections import validate_section_checkpoint
    from temsim.physics.illumination import state_for_source_node
    session = _session()
    path = io.save_diagnostic_session(session, tmp_path / "not-a-source.temdiag")
    with pytest.raises(ValueError):
        load_section_result(path)
    with pytest.raises((ValueError, KeyError)):
        WorkingPointArchiveIndex.read(path)
    with pytest.raises(ValueError):
        read_profile(path)
    for candidate in (session, session.records[0].settings, session.records[0].trajectory):
        with pytest.raises(ValueError, match="checkpoint"):
            validate_section_checkpoint(candidate)
        with pytest.raises(ValueError, match="Independent source"):
            state_for_source_node(None, candidate)


def test_duplicate_electron_ids_rejected_without_executed_arrays(tmp_path):
    record = io.DiagnosticElectronRecord("same", "Electron", "#abcdef", TestElectronSettings())
    session = io.DiagnosticSession((record,))
    path = io.save_diagnostic_session(session, tmp_path / "duplicate.temdiag")
    _rewrite(path, mutate=lambda p: p["records"].append(p["records"][0]))
    with pytest.raises(ValueError, match="Duplicate diagnostic electron"):
        io.load_diagnostic_session(path)


@pytest.mark.parametrize("value", [True, "1", float("nan"), float("inf")])
def test_nonfinite_or_ambiguous_display_inputs_are_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        io.DiagnosticDisplayState(projection_degrees=value)


def test_missing_file_and_truncated_archive_do_not_return_partial_session(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.load_diagnostic_session(tmp_path / "missing.temdiag")
    path = tmp_path / "broken.temdiag"
    path.write_bytes(b"PK\x03\x04truncated")
    with pytest.raises(ValueError, match="Invalid diagnostic session"):
        io.load_diagnostic_session(path)


def _edit_end_record(path, **changes):
    """Mutate little-endian classic EOCD fields without allocating a ZIP index."""
    import struct
    data = bytearray(path.read_bytes())
    offset = data.rfind(b"PK\x05\x06")
    offsets = {"disk": (4, "H"), "directory_disk": (6, "H"),
               "disk_count": (8, "H"), "count": (10, "H"),
               "directory_size": (12, "I"), "directory_offset": (16, "I"),
               "comment_size": (20, "H")}
    for name, value in changes.items():
        relative, kind = offsets[name]
        struct.pack_into("<" + kind, data, offset + relative, value)
    path.write_bytes(data)


def _forbid_zip_index(monkeypatch):
    monkeypatch.setattr(io, "ZipFile", lambda *a, **k: pytest.fail(
        "Malformed archive must be rejected before ZipFile allocates its index"))


def test_forged_small_entry_count_cannot_bypass_actual_entry_limit_before_zip_index(tmp_path, monkeypatch):
    path = io.save_diagnostic_session(_session(), tmp_path / "forged-count.temdiag")
    with ZipFile(path, "a") as archive:
        for index in range(6):
            archive.writestr(f"extra-{index}.txt", b"")
    # The header claims one member; there are fifteen actual central entries.
    _edit_end_record(path, disk_count=1, count=1)
    _forbid_zip_index(monkeypatch)
    with pytest.raises(ValueError, match="Too many actual"):
        io.load_diagnostic_session(path, limits=replace(io.DEFAULT_LIMITS, maximum_records=1))


def test_forged_count_below_limit_still_fails_before_zip_index(tmp_path, monkeypatch):
    path = io.save_diagnostic_session(_session(), tmp_path / "lying-count.temdiag")
    _edit_end_record(path, disk_count=1, count=1)
    _forbid_zip_index(monkeypatch)
    with pytest.raises(ValueError, match="declared count"):
        io.load_diagnostic_session(path)


@pytest.mark.parametrize("changes,expected", [
    ({"disk": 1}, "Multidisk"), ({"directory_disk": 1}, "Multidisk"),
    ({"disk_count": 0xffff, "count": 0xffff}, "ZIP64"),
    ({"directory_size": 0xffffffff}, "ZIP64"),
    ({"directory_offset": 0xffffffff}, "ZIP64"),
    ({"directory_offset": 1}, "central-directory bounds"),
    ({"directory_size": 1}, "central-directory bounds"),
    ({"comment_size": 10}, "end record"),
])
def test_invalid_end_records_rejected_before_zip_index(tmp_path, monkeypatch, changes, expected):
    path = io.save_diagnostic_session(_session(), tmp_path / "bad-end.temdiag")
    _edit_end_record(path, **changes)
    _forbid_zip_index(monkeypatch)
    with pytest.raises(ValueError, match=expected):
        io.load_diagnostic_session(path)


@pytest.mark.parametrize("kind,expected", [("signature", "central header"),
    ("metadata_size", "metadata bounds"), ("needed_version", "version"),
    ("local_offset", "member data bounds"), ("expanded_size", "unpacked-size")])
def test_invalid_central_header_rejected_before_zip_index(tmp_path, monkeypatch, kind, expected):
    import struct
    path = io.save_diagnostic_session(_session(), tmp_path / "bad-central.temdiag")
    data = bytearray(path.read_bytes())
    end = data.rfind(b"PK\x05\x06")
    directory = struct.unpack_from("<I", data, end + 16)[0]
    if kind == "signature":
        data[directory:directory+4] = b"bad!"
    elif kind == "metadata_size":
        struct.pack_into("<H", data, directory + 28, 0xffff)
    elif kind == "needed_version":
        struct.pack_into("<H", data, directory + 6, 21)
    elif kind == "local_offset":
        struct.pack_into("<I", data, directory + 42, directory)
    else:
        struct.pack_into("<I", data, directory + 24, io.DEFAULT_LIMITS.maximum_unpacked_bytes + 1)
    path.write_bytes(data)
    _forbid_zip_index(monkeypatch)
    with pytest.raises(ValueError, match=expected):
        io.load_diagnostic_session(path)


def test_real_zip64_extra_field_is_rejected_before_zip_index(tmp_path, monkeypatch):
    from zipfile import ZipInfo
    import struct
    path = tmp_path / "zip64-extra.temdiag"
    entry = ZipInfo("manifest.json")
    entry.extra = struct.pack("<HHQQ", 1, 16, 2, 2)
    with ZipFile(path, "w") as archive:
        archive.writestr(entry, b"{}")
    _forbid_zip_index(monkeypatch)
    with pytest.raises(ValueError, match="ZIP64"):
        io.load_diagnostic_session(path)


def test_preflight_accepts_writer_then_constructs_index_once(tmp_path, monkeypatch):
    path = io.save_diagnostic_session(_session(), tmp_path / "valid.temdiag")
    original, calls = io.ZipFile, []
    def indexed(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(io, "ZipFile", indexed)
    loaded = io.load_diagnostic_session(path)
    assert loaded.records[0].trajectory.completed
    assert calls == [1]


def test_memory_budget_prevents_zip_directory_index_allocation(tmp_path, monkeypatch):
    path = io.save_diagnostic_session(_session(), tmp_path / "low-memory.temdiag")
    _forbid_zip_index(monkeypatch)
    with pytest.raises(ValueError, match="memory limit"):
        io.load_diagnostic_session(path, limits=replace(io.DEFAULT_LIMITS, maximum_memory_bytes=100))
