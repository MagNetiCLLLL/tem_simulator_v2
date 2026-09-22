"""Small executed-material fixtures for post-specimen checkpoint continuation."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from test_material_particle_sections import material_case
from test_particle_sections import fixture as optical_fixture
from temsim import simulation_pipeline as pipeline
from temsim.physics import particle_sections as sections
from temsim.specimen import downstream_transport as downstream


@pytest.fixture
def case(material_case):
    first = pipeline.calculate_particle_section(material_case.state, 456.)
    return material_case, first


def build(case, stop=458., previous=None, keys=(), *, transport=None, simulation=None):
    fixture, first = case
    return downstream.build_geometric_specimen_exit(
        fixture.state, first.simulation if simulation is None else simulation,
        first.specimen_interactions.elastic_transport if transport is None else transport,
        first.specimen_interactions.inelastic_distribution,
        save_z_mm=(stop,), stop_z_mm=stop, existing_exit=previous,
        tuning_component_keys=keys,
    )


def assert_same_endpoint(left, right):
    assert len(left.branches) == len(right.branches)
    for a, b in zip(left.branches, right.branches, strict=True):
        assert a.name == b.name and a.weight == b.weight
        for key in ("alive", "blocked_z", "energy_offset_ev", "ray_weight", "source_ray_id"):
            np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
        assert a.blocked_key == b.blocked_key
        for key in ("x", "tx", "y", "ty", "flight_time_s"):
            np.testing.assert_allclose(getattr(a, key)[-1], getattr(b, key)[-1],
                                       rtol=2e-13, atol=1e-22, equal_nan=True)


def test_extension_resumes_actual_material_checkpoints_with_original_clocks(case, monkeypatch):
    _, first = case
    starts = []
    original = sections.execute_propagation_plan
    def execute(state, plan, *args, **kwargs):
        starts.append(float(plan.z_mm[kwargs.get("start_index", 0)]))
        return original(state, plan, *args, **kwargs)
    monkeypatch.setattr(sections, "execute_propagation_plan", execute)
    resumed = build(case, previous=first.specimen_exit)
    assert starts and set(starts) == {456.}
    assert resumed.metrics["material_section_reused_prefix"]
    assert len(resumed.checkpoints) == len(resumed.segments) == len(resumed.branches)
    cold = build(case)
    assert_same_endpoint(resumed, cold)
    for branch in resumed.branches:
        assert np.all(np.isnan(branch.flight_time_s[0]))  # virtual reference row
        if np.any(branch.energy_offset_ev < 0.):
            assert np.all(np.isnan(branch.flight_time_s))  # aggregate loss lacks event depth
        else:
            assert np.all(branch.flight_time_s[-1] > 1e-8)


def test_post_lens_edit_reuses_only_checkpoint_before_its_field(case):
    fixture, _ = case
    lens = next(l for l in fixture.state.lenses if l.key == "projector_lens_2")
    lens.z_mm, lens.a_mm, lens.enabled, lens.percent = 457., .08, True, 10.
    first = build(case, keys=(lens.key,))
    boundary = lens.field_support_mm()[0]
    lens.percent = 10.1
    changed = build(case, previous=first, keys=(lens.key,))
    records = changed.metrics["material_section_resume"]
    assert records and all(r["hit"] and 454. < r["resume_z_mm"] < boundary for r in records)
    assert_same_endpoint(changed, build(case, keys=(lens.key,)))


@pytest.mark.parametrize("changed", ["phase", "energy", "time", "weight", "ids", "physics"])
def test_changed_material_identity_never_reuses_stale_branch(case, changed):
    fixture, first = case
    transport = first.specimen_interactions.elastic_transport
    terminal = transport.terminal_electrons
    simulation = first.simulation
    if changed == "phase":
        positions = terminal.position_nm.copy()
        positions[:, 0] += 1.
        terminal = replace(terminal, position_nm=positions)
    elif changed == "energy":
        terminal = replace(terminal, kinetic_energy_ev=terminal.kinetic_energy_ev - .5)
    elif changed == "time":
        terminal = replace(terminal, reference_time_offset_s=terminal.reference_time_offset_s + 1e-13)
    elif changed == "weight":
        weights = terminal.weight.copy()
        weights[0] += .01
        weights[1] -= .01
        terminal = replace(terminal, weight=weights)
    elif changed == "ids":
        simulation = SimpleNamespace(incident=replace(simulation.incident,
            source_ray_id=np.asarray(simulation.incident.source_ray_id) + 100),
            gun_trace=simulation.gun_trace)
    elif changed == "physics":
        fixture.state.sample.eds_elastic_seed += 1
    changed_exit = build(case, previous=first.specimen_exit,
        transport=replace(transport, terminal_electrons=terminal), simulation=simulation)
    assert not changed_exit.metrics["material_section_reused_prefix"]
    assert all(r["resume_z_mm"] == 454. for r in changed_exit.metrics["material_section_resume"])


def test_absorbed_rays_are_not_revived_by_material_extension(case):
    fixture, _ = case
    fixture.state.camera.set_optical_reference_z_mm(fixture.state.selected_area_aperture.z_mm, 455.)
    fixture.state.camera.inserted = True
    fixture.state.camera.outer_width_mm = 100.
    first = build(case, stop=456.)
    assert all(not np.any(b.alive) for b in first.branches)
    extended = build(case, previous=first)
    assert extended.metrics["material_section_reused_prefix"]
    assert all(not np.any(b.alive) for b in extended.branches)
    assert all(np.all(np.isnan(b.flight_time_s[-1])) for b in extended.branches)
    assert_same_endpoint(extended, build(case))


def test_corrupt_checkpoint_clock_is_recomputed(case):
    _, first = case
    segments = []
    for segment in first.specimen_exit.segments:
        cp = segment.checkpoints
        broken = SimpleNamespace(**{key: getattr(cp, key) for key in cp.__dataclass_fields__})
        broken.flight_time_s = cp.flight_time_s[:, :0]
        segments.append(replace(segment, checkpoints=broken))
    broken_exit = replace(first.specimen_exit, segments=tuple(segments))
    result = build(case, previous=broken_exit)
    assert not result.metrics["material_section_reused_prefix"]
    assert_same_endpoint(result, build(case))


def test_participating_vacuum_uses_full_original_transport(case, monkeypatch):
    fixture, first = case
    fixture.state.vacuum_map.enabled = True
    # The artificial 450--458 mm column does not use the full instrument's
    # vacuum anchors. Empty participating medium here isolates restart policy.
    monkeypatch.setattr("temsim.vacuum.resolve_regions", lambda *a, **k: ())
    monkeypatch.setattr("temsim.physics.residual_medium.resolve_regions", lambda *a, **k: ())
    calls = []
    original = downstream.propagate
    def propagate(state, start, stop, *args, **kwargs):
        calls.append(start)
        return original(state, start, stop, *args, **kwargs)
    monkeypatch.setattr(downstream, "propagate", propagate)
    result = build(case, previous=first.specimen_exit)
    assert calls and set(calls) == {454.}
    assert not result.segments and not result.metrics["material_section_reused_prefix"]
    assert result.metrics["material_section_vacuum_restart"] == "recomputed"
