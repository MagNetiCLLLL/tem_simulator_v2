"""Bounded optical section fixtures; not full-source/material qualification."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.optics.electron_gun.base import GunExitBundle, GunTraceResult
from temsim.physics import particle_sections as sections
from temsim.physics.simulation import run


@pytest.fixture
def fixture(monkeypatch):
    state = default_state()
    state.vacuum_map.enabled = False
    state.acceleration_enabled = False
    state.acceleration_backend = "CPU"
    state.step_mm = .5
    state.history_step_mm = 1.
    state.sample.inserted = False
    state.sample.z_mm = 454.
    for collection in ("lenses", "deflectors", "stigmators", "corrector_elements", "apertures"):
        for component in getattr(state, collection, ()):
            component.enabled = False
    for component in state.recording_planes:
        component.inserted = False
    state.nanopulser.installed = False
    x = np.array([0., 1e-7, -1e-7])
    tx = np.array([0., 1e-6, -1e-6])
    zero = np.zeros(3)
    times = np.full(3, 1e-8)
    emitted = GunExitBundle(x, zero, tx, zero, zero, np.full(3, 1/3),
                            np.array([1, 3, 8]), np.ones(3, bool), times)
    history = np.vstack((x, x))
    gun = GunTraceResult(np.array([449., 450.]), history, np.zeros_like(history),
        np.vstack((tx, tx)), np.zeros_like(history), emitted, np.full(3, np.nan),
        ("", "", ""), 1e-9, 1e-9, 1e-9, flight_time_s=np.vstack((times-.01e-8, times)),
        emission_reference={"ray_id": emitted.ray_id,
                            "position_m": np.column_stack((x, zero, zero)),
                            "direction": np.tile([0., 0., 1.], (3, 1)),
                            "normal": np.tile([0., 0., 1.], (3, 1))})
    calls = []
    def trace(_state):
        calls.append(True)
        return gun
    monkeypatch.setattr("temsim.optics.electron_gun.source.trace_source_to_exit", trace)
    # A short, explicitly artificial drift geometry keeps real column kernels
    # under test without restarting the physical high-resolution gun.
    monkeypatch.setattr("temsim.physics.column_wall.clip_column_wall", lambda s,z,x,y,a,b,k: (a,b,k))
    monkeypatch.setattr(sections, "gun_dependency_signature", lambda s: str(s.electron_gun.emitter.ray_count))
    return state, gun, calls


def trace(state, target, previous=None, keys=()):
    return run(state, optical_only=True, resolved_layout=object(),
               observation_stop_z_mm=target, tuning_component_keys=keys,
               existing_simulation=previous)


def test_drift_section_has_exact_phase_and_original_clock(fixture):
    state, gun, _ = fixture
    result = trace(state, 452.25)
    endpoint = result.section_checkpoint.segments[0].checkpoints
    np.testing.assert_allclose(endpoint.x_m[-1], gun.exit_bundle.x_m + gun.exit_bundle.tx_rad*.00225,
                               atol=2e-22, rtol=1e-14)
    assert result.incident.z[-1] == 452.25
    assert not result.branches
    assert np.all(result.incident.flight_time_s[-1] > gun.exit_bundle.flight_time_s)
    np.testing.assert_array_equal(result.incident.source_ray_id, [1,3,8])
    assert sections.validate_section_checkpoint(result.section_checkpoint) is result.section_checkpoint
    assert result.metrics["sample_statistics_status"] == "NOT_REACHED"
    assert not result.metrics["sample_illumination_evaluated"]
    assert result.metrics["sample_beam_surviving_rays"] == 0
    assert np.isnan(result.metrics["sample_beam_surviving_fraction"])
    assert np.isnan(result.metrics["sample_waist_offset_mm"])
    assert result.metrics["section_beam_surviving_rays"] == 3


def test_extension_uses_saved_endpoint_and_reselection_can_move_upstream(fixture):
    state, _, calls = fixture
    first = trace(state, 452.25)
    second = trace(state, 453.75, first)
    assert second.metrics["section_resume_z_mm"] == 452.25
    assert second.metrics["section_gun_reused"]
    assert len(calls) == 1
    sections.validate_section_checkpoint(second.section_checkpoint)
    shorter = trace(state, 451.5, second)
    assert shorter.incident.z[-1] == 451.5
    sections.validate_section_checkpoint(shorter.section_checkpoint)
    same = trace(state, 451.5, shorter)
    assert same.metrics["section_resume_z_mm"] == 451.5
    np.testing.assert_array_equal(same.incident.x, shorter.incident.x)


def test_zero_length_gun_exit_section_and_boundary_validation(fixture):
    state, gun, calls = fixture
    result = trace(state, 450.)
    np.testing.assert_array_equal(result.incident.x[-1], gun.exit_bundle.x_m)
    sections.validate_section_checkpoint(result.section_checkpoint)
    again = trace(state, 450., result)
    assert again.metrics["section_gun_reused"]
    before = len(calls)
    for target in (449., np.nan, sections.section_limits(state)[1]+1):
        with pytest.raises(ValueError, match="section must lie"):
            trace(state, target)
    assert len(calls) == before


def test_downstream_deflector_kick_is_not_moved_to_section_endpoint(fixture, monkeypatch):
    state, gun, _ = fixture
    monkeypatch.setattr(sections, "_events", lambda s: ((453., .001, 0.),))
    before = trace(state, 452.9)
    np.testing.assert_array_equal(before.incident.tx[-1], gun.exit_bundle.tx_rad)
    reached = trace(state, 453., before)
    np.testing.assert_allclose(reached.incident.tx[-1], gun.exit_bundle.tx_rad+.001, atol=1e-16)
    repeated = trace(state, 453., reached)
    np.testing.assert_array_equal(repeated.incident.tx[-1], reached.incident.tx[-1])


def test_pre_specimen_stop_does_not_apply_chromatic_reference_kick(fixture, monkeypatch):
    state, _, _ = fixture
    calls = []
    def kick(s,x,y,energy,**kwargs):
        calls.append(True)
        return np.full_like(x, .002), np.full_like(y, -.001)
    monkeypatch.setattr("temsim.physics.chromatic.objective_chromatic_kick_from_state", kick)
    first = trace(state, 453.)
    assert not calls and not first.branches
    second = trace(state, 456., first)
    assert len(calls) == 1
    np.testing.assert_allclose(second.branches["000"].tx[0], second.incident.tx[-1]+.002)
    assert second.branches["000"].z[-1] == 456.
    assert second.metrics["sample_scattering_applied"] is False
    assert second.metrics["sample_illumination_evaluated"]
    assert second.metrics["sample_statistics_status"] != "NOT_REACHED"
    sections.validate_section_checkpoint(second.section_checkpoint)


def test_absorbing_detector_survives_extension_without_resurrection(fixture):
    state, _, _ = fixture
    state.camera.z_mm = 455.
    state.camera.inserted = True
    state.camera.outer_width_mm = 100.
    first = trace(state, 456.)
    assert not np.any(first.branches["000"].alive)
    np.testing.assert_array_equal(first.branches["000"].blocked_z, np.full(3,455.))
    later = trace(state, 458., first)
    assert not np.any(later.branches["000"].alive)
    assert set(later.branches["000"].blocked_key) == {state.camera.key}


def test_selected_component_support_limits_reuse_not_its_centre(fixture):
    state, _, _ = fixture
    state.sample.z_mm = 800.
    lens = next(l for l in state.lenses if l.key == "condenser_lens_2")
    first = trace(state, 700.)
    boundary = state.condenser_system[lens.key].field_support_mm()[0]
    second = trace(state, 700., first, (lens.key,))
    assert second.metrics["section_resume_z_mm"] < boundary < lens.z_mm
    # A changed upstream coefficient must never be hidden by a selected range.
    lens.enabled = True
    lens.percent += .2
    changed = trace(state, 700., second, (lens.key,))
    fresh = trace(state, 700., keys=(lens.key,))
    np.testing.assert_allclose(changed.incident.x[-1], fresh.incident.x[-1], rtol=2e-5, atol=1e-14)
    assert changed.metrics["section_resume_z_mm"] < boundary


def test_source_change_reexecutes_gun_and_does_not_reuse_column(fixture):
    state, _, calls = fixture
    first = trace(state, 453.)
    state.electron_gun.emitter.ray_count += 1
    changed = trace(state, 453., first)
    assert len(calls) == 2
    assert not changed.metrics["section_gun_reused"]
    assert not changed.metrics["section_reused_prefix"]
    with pytest.raises(ValueError, match="Unknown section tuning"):
        trace(state, 453., keys=("not_a_component",))


def test_corrupt_section_checkpoint_is_not_reused(fixture):
    state, _, calls = fixture
    first = trace(state, 453.)
    segment = first.section_checkpoint.segments[0]
    malformed = replace(segment, branch=replace(segment.branch, source_ray_id=np.array([1])))
    first.section_checkpoint = replace(first.section_checkpoint, segments=(malformed,))
    with pytest.raises(ValueError, match="ancestry"):
        sections.validate_section_checkpoint(first.section_checkpoint)
    second = trace(state, 453., first)
    assert len(calls) == 2 and not second.metrics["section_reused_prefix"]


def test_spherical_kick_never_leaks_from_outside_section(fixture, monkeypatch):
    from temsim.physics import core
    state, _, _ = fixture
    lens = state.lenses[0]
    lens.enabled = True
    monkeypatch.setattr(core, "spherical_aberration_mm", lambda lens, voltage: 1.)
    monkeypatch.setattr(core, "focal_length_mm", lambda lens, voltage: 1.)
    for grid in (np.array([lens.z_mm-.01]), np.array([lens.z_mm-1.,lens.z_mm-.01])):
        np.testing.assert_array_equal(core.spherical_aberration_kick_m3(grid,state), np.zeros(len(grid)))
    assert core.spherical_aberration_kick_m3(np.array([lens.z_mm]),state)[0] > 0


def test_vacuum_participation_reexecutes_transport_instead_of_resetting_rng(fixture):
    state, _, _ = fixture
    state.sample.z_mm = 1599.2  # Restore installed vacuum-region anchor ordering.
    state.vacuum_map.enabled = True
    first = trace(state, 451.)
    second = trace(state, 452., first)
    assert second.metrics["section_vacuum_restart"] == "recomputed"
    assert not second.metrics["section_reused_prefix"]
    assert second.metrics["section_resume_z_mm"] == 450.
    assert second.incident.vacuum_report is not None


@pytest.mark.parametrize("fault", ["branch_time", "checkpoint_time", "indices"])
def test_missing_clocks_or_invalid_plan_indices_are_not_loadable(fixture, fault):
    state, _, _ = fixture
    value = trace(state, 453.).section_checkpoint
    segment = value.segments[0]
    if fault == "branch_time":
        segment = replace(segment, branch=replace(segment.branch, flight_time_s=None))
    elif fault == "checkpoint_time":
        segment = replace(segment, checkpoints=replace(segment.checkpoints, flight_time_s=None))
    else:
        segment = replace(segment, plan=replace(segment.plan, checkpoint_index=np.array([1e9])))
    with pytest.raises(ValueError):
        sections.validate_section_checkpoint(replace(value, segments=(segment,)))
