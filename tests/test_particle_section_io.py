"""Executed section persistence and request isolation, including real tip transport."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.particle_section_io import (
    normalise_section_request, section_request_signatures,
    save_section_result, load_section_result, _unpack,
)


@pytest.mark.parametrize("section_request", [
    {"target_z_mm": float("nan"), "component_keys": ("objective_lens",)},
    {"target_z_mm": 1000, "component_keys": "objective_lens"},
    {"target_z_mm": 1000, "component_keys": (None,)},
])
def test_section_request_rejects_incomplete_or_ambiguous_inputs(section_request):
    with pytest.raises(ValueError):
        normalise_section_request(section_request)


def test_plane_and_participant_identity_are_separate_from_full_calculations():
    base = {"request": "full-request", "incident": "upstream"}
    request = {"target_z_mm": 1000., "component_keys": ("b", "a")}
    first = section_request_signatures(base, request)
    assert first["request"] != base["request"]
    assert first == section_request_signatures(base, dict(request, component_keys=("a", "b")))
    assert first != section_request_signatures(base, dict(request, target_z_mm=1001.))
    assert first != section_request_signatures(base, dict(request, component_keys=("a",)))
    assert base == {"request": "full-request", "incident": "upstream"}
    assert normalise_section_request(request, quality="High accuracy") == dict(request, component_keys=("a", "b"))
    assert normalise_section_request(dict(request, component_keys=()), quality="High accuracy")["component_keys"] == ()
    with pytest.raises(ValueError, match="quality"):
        normalise_section_request(request, quality="Unknown")


def test_section_decoder_does_not_import_file_supplied_classes():
    with pytest.raises(ValueError, match="Unknown"):
        _unpack({"type": "os.system", "fields": {}}, {}, {})


@pytest.fixture(scope="module")
def executed_section():
    from temsim.optics.column import default_state
    from temsim.gui.calculation_controller import CalculationController, CalculationWorker
    from temsim.calculation_cache import calculation_signatures, state_model_signature
    source = default_state()
    source.sample.inserted = False  # This fixture exercises vacuum column continuation.
    source.sample.wave_enabled = source.sample.stem_wave_enabled = False
    snapshot = CalculationController._calculation_snapshot(source, "Preview", 49, 2.)
    request = {"target_z_mm": float(snapshot.sample.z_mm) + 30.,
               "component_keys": ("objective_lens",)}
    worker = CalculationWorker(1, "Preview", snapshot, section_request=request,
        model_signature=state_model_signature(source),
        request_signatures=section_request_signatures(calculation_signatures(snapshot), request))
    results, errors = [], []
    worker.signals.result.connect(lambda _g, _q, r, _t: results.append(r))
    worker.signals.error.connect(lambda *args: errors.append(args))
    worker.run()
    assert not errors, errors
    assert len(results) == 1
    return results[0]


def test_saved_section_round_trip_retains_full_state_and_can_extend(executed_section, tmp_path, monkeypatch):
    from temsim.physics.particle_sections import run_particle_section, validate_section_checkpoint
    from temsim.optics.electron_gun import source
    path = tmp_path / "saved.temsection"
    save_section_result(executed_section, path)
    restored = load_section_result(path)
    old = executed_section.simulation.section_checkpoint
    new = restored.simulation.section_checkpoint
    validate_section_checkpoint(new)
    assert restored.layout is not None and restored.assembly is not None
    assert new.gun_dependency_signature == old.gun_dependency_signature
    for left, right in zip(old.segments, new.segments):
        for field in ("x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s"):
            np.testing.assert_array_equal(getattr(left.checkpoints, field), getattr(right.checkpoints, field))
            assert getattr(right.checkpoints, field).dtype == np.float64
        for field in ("source_ray_id", "alive", "blocked_z", "energy_offset_ev", "ray_weight"):
            np.testing.assert_array_equal(getattr(left.branch, field), getattr(right.branch, field))
    monkeypatch.setattr(source, "trace_source_to_exit", lambda *a, **k: pytest.fail("Matching saved gun must be reused"))
    target = restored.simulation.metrics["section_target_z_mm"] + 25.
    # Move the editable region downstream. The saved completed upstream section
    # becomes a cache; it never replaces the physical source definition.
    result = run_particle_section(restored.state_snapshot, observation_stop_z_mm=target,
        tuning_component_keys=("projector_lens_1",), existing_simulation=restored.simulation)
    assert result.metrics["section_gun_reused"]
    assert result.metrics["section_reused_prefix"]
    assert result.branches["000"].z[-1] == target
    assert result.metrics["section_resume_z_mm"] > restored.state_snapshot.sample.z_mm


def test_save_refuses_a_display_only_result(executed_section, tmp_path):
    old = executed_section.simulation
    sim = replace(old, section_checkpoint=None)
    with pytest.raises(ValueError, match="Calculate a section"):
        save_section_result(replace(executed_section, simulation=sim), tmp_path / "bad.temsection")


def test_retained_loaded_section_obeys_controller_cache_budget(executed_section):
    from temsim.gui.calculation_controller import CalculationController
    controller = CalculationController(persistent_cache_enabled=False, tuning_cache_budget_bytes=0)
    with pytest.raises(ValueError, match="budget"):
        controller.retain_section_seed(executed_section)
    assert controller.cache_statistics()["tuning_entries"] == 0


def test_material_section_round_trip_reuses_executed_scattering(executed_section, tmp_path, monkeypatch):
    from temsim import simulation_pipeline as pipeline
    from temsim.physics.optical_tuning import prepare_tuning_snapshot
    from temsim.instrument_snapshot import decode_instrument, encode_instrument
    state = decode_instrument(encode_instrument(executed_section.state_snapshot))
    state.sample.inserted = True
    state.sample.specimen_mode = "reference"
    state.sample.size_x_nm = state.sample.size_y_nm = 100_000.
    state.sample.thickness_nm = 5.
    state.sample.eds_enabled = False
    state.ac_deflector.scan_enabled = False
    state.energy_filter.enabled = False
    prepare_tuning_snapshot(state, "Preview", particle_signals=True)
    request = {"target_z_mm": state.sample.z_mm+30., "component_keys": ("objective_lens",)}
    first = pipeline.calculate_particle_section(state, **request, existing_result=executed_section)
    assert first.specimen_exit is not None
    assert hasattr(first.simulation, "material_section_cache")
    path = tmp_path / "scattered.temsection"
    save_section_result(first, path)
    restored = load_section_result(path)
    before = first.simulation.material_section_cache
    after = restored.simulation.material_section_cache
    assert before.specimen_exit.segments and after.specimen_exit.segments
    for left, right in zip(before.specimen_exit.segments, after.specimen_exit.segments, strict=True):
        for name in ("x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s"):
            np.testing.assert_array_equal(getattr(left.checkpoints, name), getattr(right.checkpoints, name))
    for name in ("source_ray_index", "position_nm", "direction", "weight",
                 "kinetic_energy_ev", "reference_time_offset_s", "material_path_nm"):
        np.testing.assert_array_equal(getattr(before.elastic_transport.terminal_electrons, name),
                                      getattr(after.elastic_transport.terminal_electrons, name))
    assert restored.state_snapshot._particle_tuning
    assert not restored.state_snapshot._optical_tuning
    original = pipeline.run_specimen_interactions
    def verify_reuse(*args, **kwargs):
        previous = kwargs.get("existing_result")
        assert previous is not None
        assert previous.elastic_transport is after.elastic_transport
        result = original(*args, **kwargs)
        assert result.elastic_transport is after.elastic_transport
        return result
    monkeypatch.setattr(pipeline, "run_specimen_interactions", verify_reuse)
    extended = pipeline.calculate_particle_section(restored.state_snapshot,
        request["target_z_mm"]+10., ("projector_lens_1",), existing_result=restored)
    assert {"elastic", "inelastic"} <= extended.reused_products
    assert extended.simulation.metrics["section_gun_reused"]
    assert extended.specimen_exit.metrics["material_section_reused_prefix"]
    assert all(row["resume_z_mm"] == request["target_z_mm"]
               for row in extended.specimen_exit.metrics["material_section_resume"])


def test_invalid_section_archive_reports_value_error(tmp_path):
    path = tmp_path / "invalid.temsection"
    path.write_bytes(b"not a section archive")
    with pytest.raises(ValueError, match="checkpoint"):
        load_section_result(path)


def test_automatic_archive_is_atomic_deduplicated_and_self_describing(executed_section, tmp_path):
    from pathlib import Path
    from zipfile import ZIP_STORED, ZipFile
    from temsim.particle_section_io import archive_section_result, section_archive_identity
    first = archive_section_result(executed_section, tmp_path)
    path = Path(first["path"])
    timestamp = path.stat().st_mtime_ns
    restored = load_section_result(path)
    assert restored.section_archive_info["identity"] == section_archive_identity(executed_section)
    assert restored.section_archive_info["target_z_mm"] == executed_section.simulation.metrics["section_target_z_mm"]
    assert restored.section_archive_info["particle_count"] == 49
    assert restored.section_archive_info["quality"] == "Preview"
    assert restored.section_archive_info["saved_at_utc"]
    assert restored.section_archive_info["resumable_through_z_mm"] >= restored.state_snapshot.sample.z_mm
    with ZipFile(path) as archive:
        assert all(item.compress_type == ZIP_STORED for item in archive.infolist())
    second = archive_section_result(executed_section, tmp_path)
    assert second["reused"]
    assert path.stat().st_mtime_ns == timestamp
    assert len(list(tmp_path.glob("*.temsection"))) == 1


def test_failed_archive_write_preserves_previous_file_and_calculation(executed_section, tmp_path, monkeypatch):
    from temsim import working_point
    path = tmp_path / "manual.temsection"
    path.write_bytes(b"previous user archive")
    checkpoint = executed_section.simulation.section_checkpoint
    monkeypatch.setattr(working_point.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("Disk unavailable")))
    with pytest.raises(OSError, match="Disk unavailable"):
        save_section_result(executed_section, path, overwrite=True)
    assert path.read_bytes() == b"previous user archive"
    assert not tuple(tmp_path.glob(".working-point-*.tmp"))
    assert executed_section.simulation.section_checkpoint is checkpoint
