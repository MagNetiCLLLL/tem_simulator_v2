"""Page-workflow restart contracts using small storage-only particle records."""
from copy import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from test_shared_field_archives import storage_result
from temsim.particle_section_io import (
    _pack_record_graph, _result_records, archive_section_result,
    load_section_result, save_section_result, section_archive_identity,
    section_archive_summary,
)
from temsim.physics.completed_particle_section import capture_completed_particle_section
from temsim.working_point import WorkingPointCheckpoint
from temsim.working_point_archive import WorkingPointArchiveIndex


@pytest.fixture
def optical_result(storage_result, monkeypatch):
    """The 3 mm incident checkpoint is exact; 5 mm is an optical display only."""
    state = copy(storage_result.state_snapshot)
    state.sample = copy(state.sample)
    state.sample.inserted = True
    original = storage_result.simulation
    display = replace(original.incident, z=np.array((3., 4., 5.)))
    simulation = replace(original, branches={"000": display},
        metrics=dict(original.metrics, optical_tuning=True), section_checkpoint=None)
    monkeypatch.setattr("temsim.physics.particle_sections.gun_dependency_signature",
                        lambda _state: "synthetic gun identity")
    return replace(storage_result, simulation=simulation, state_snapshot=state, workflow="rays")


def test_rays_archive_captures_only_executed_incident_checkpoint(optical_result):
    result = optical_result
    simulation = result.simulation
    assert capture_completed_particle_section(result)
    assert tuple(s.name for s in simulation.section_checkpoint.segments) == ("incident",)
    assert simulation.metrics["section_target_z_mm"] == 5.
    assert simulation.metrics["section_resumable_through_z_mm"] == 3.
    assert simulation.metrics["section_physics_scope"] == "optical_reference_without_specimen_interactions"
    assert simulation.metrics["section_full_path"]
    assert not simulation.metrics["specimen_transport_completed"]
    assert simulation.material_section_cache is None
    assert result.state_snapshot.sample.inserted  # No physical input is disabled.
    retained = simulation.section_checkpoint.segments[0].checkpoints
    for name in ("x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s", "kinetic_energy_ev"):
        assert getattr(retained, name).tobytes() == getattr(simulation.incident_checkpoints, name).tobytes()
    summary = section_archive_summary(result)
    assert summary["workflow"] == "rays"
    assert not summary["specimen_transport_completed"]
    assert "optical reference" in summary["resume_details"]
    assert "3 mm" in summary["resume_details"] and "5 mm" in summary["resume_details"]


def test_rays_never_promotes_retained_material_or_post_records_to_restart(optical_result):
    result = optical_result
    old = object()
    result.simulation.material_section_cache = old
    result.simulation.completed_post_sections = (old,)
    result.specimen_interactions = SimpleNamespace(
        elastic_transport=old, inelastic_distribution=old, wave_imaging=None)
    result.specimen_exit = old
    assert capture_completed_particle_section(result)
    assert len(result.simulation.section_checkpoint.segments) == 1
    assert result.simulation.material_section_cache is None
    assert result.specimen_exit is old  # Existing view products are retained.


@pytest.mark.parametrize("quality", ["Preview", "Medium", "High accuracy"])
def test_capture_preserves_actual_calculation_quality(optical_result, quality):
    optical_result.simulation.metrics["tuning_quality"] = quality
    assert capture_completed_particle_section(optical_result)
    assert section_archive_summary(optical_result)["quality"] == quality


def test_paused_wave_controls_are_not_executed_wave_products(optical_result, tmp_path):
    optical_result.state_snapshot.sample.wave_enabled = True
    optical_result.state_snapshot.sample.stem_wave_enabled = True
    assert capture_completed_particle_section(optical_result)
    path = tmp_path / "rays-with-paused-wave-settings.temsection"
    save_section_result(optical_result, path)
    restored = load_section_result(path)
    assert restored.workflow == "rays" and restored.wave_imaging is None
    assert restored.state_snapshot.sample.wave_enabled
    assert restored.state_snapshot.sample.stem_wave_enabled
    assert restored.simulation.metrics["section_resumable_through_z_mm"] == 3.


@pytest.mark.parametrize("owner", ["result", "interactions"])
def test_actual_coherent_products_do_not_become_particle_checkpoints(optical_result, owner):
    if owner == "result":
        optical_result.wave_imaging = object()
    else:
        optical_result.specimen_interactions = SimpleNamespace(wave_imaging=object())
    assert not capture_completed_particle_section(optical_result)
    assert optical_result.simulation.section_checkpoint is None


def test_vacuum_result_does_not_claim_specimen_interactions(optical_result):
    optical_result.workflow = "full"
    optical_result.state_snapshot.sample.inserted = False
    optical_result.simulation.metrics["optical_tuning"] = False
    assert capture_completed_particle_section(optical_result)
    summary = section_archive_summary(optical_result)
    assert summary["physics_scope"] == "classical_particles_without_specimen_interactions"
    assert not summary["specimen_transport_completed"]


def test_actual_material_products_create_material_restart_and_scope(optical_result, monkeypatch):
    result = optical_result
    result.workflow = "sample"
    result.simulation.metrics["optical_tuning"] = False
    result.specimen_interactions = SimpleNamespace(
        elastic_transport=object(), inelastic_distribution=object(), eds_spectrum=None,
        wave_imaging=None, incident_bundle=SimpleNamespace(target_centroid_nm=(0., 0.)))
    result.specimen_exit = SimpleNamespace(segments=(
        SimpleNamespace(branch=SimpleNamespace(z=np.array((3., 5.)))),))
    monkeypatch.setattr("temsim.calculation_cache.calculation_signatures",
                        lambda _state: {"elastic": "fixture"})
    assert capture_completed_particle_section(result)
    summary = section_archive_summary(result)
    assert summary["specimen_transport_completed"]
    assert summary["physics_scope"] == "classical_particles_with_specimen_interactions"
    assert summary["resumable_through_z_mm"] == 5.


def test_workflow_roundtrip_and_automatic_archive_keep_scope(optical_result, tmp_path):
    assert capture_completed_particle_section(optical_result)
    info = archive_section_result(optical_result, tmp_path)
    restored = load_section_result(info["path"])
    assert restored.workflow == "rays"
    assert restored.state_snapshot.sample.inserted
    assert not restored.section_archive_info["specimen_transport_completed"]
    assert restored.section_archive_info["physics_scope"] == info["physics_scope"]
    assert restored.section_archive_info["target_z_mm"] == 5.
    assert restored.section_archive_info["resumable_through_z_mm"] == 3.
    assert archive_section_result(optical_result, tmp_path)["reused"]
    assert section_archive_identity(replace(optical_result, workflow="sample")) != info["identity"]
    assert _result_records(restored)["workflow"] == "rays"


def test_historical_archive_without_workflow_remains_readable_not_resumable(storage_result, tmp_path):
    current = save_section_result(storage_result, tmp_path / "current.temsection")
    records = _result_records(storage_result)
    records.pop("workflow")
    arrays = {}
    arrays[current.metadata["record_graph"]] = _pack_record_graph(records, arrays)
    historical = WorkingPointCheckpoint(replace(current.snapshot, implementation="0" * 64),
        arrays, current.plane_z_mm, current.stage_signature, current.metadata)
    path = tmp_path / "historical.temsection"
    historical.write_package(path)
    restored = WorkingPointArchiveIndex.read(path).load()
    assert restored.digest == historical.digest
    assert all(not value.flags.writeable for value in restored.arrays.values())
    with pytest.raises(ValueError, match="historical viewing only"):
        load_section_result(path)
