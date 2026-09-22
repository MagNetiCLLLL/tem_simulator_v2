"""Actual short-column material fixtures bridge archived and complete runs."""
from dataclasses import replace

import numpy as np
import pytest

from test_material_particle_sections import material_case
from test_particle_sections import fixture as optical_fixture
from temsim import simulation_pipeline as pipeline
from temsim.physics.completed_particle_section import restore_material_interactions
from temsim.physics.particle_sections import validate_section_checkpoint


def test_material_envelope_requires_actual_incident_identity(material_case):
    case = material_case
    first = pipeline.calculate_particle_section(case.state, 456.)
    restored, cache = restore_material_interactions(case.state, first.simulation, first)
    assert cache is first.simulation.material_section_cache
    assert restored.elastic_transport is first.specimen_interactions.elastic_transport
    changed_x = first.simulation.incident.x.copy()
    changed_x[-1, 0] += 1e-9
    changed = replace(first.simulation, incident=replace(first.simulation.incident, x=changed_x))
    assert restore_material_interactions(case.state, changed, first) == (None, None)
    case.state.sample.thickness_nm += 1.
    assert restore_material_interactions(case.state, first.simulation, first) == (None, None)


def test_normal_pipeline_consumes_saved_material_without_repeating_collisions(material_case, monkeypatch):
    case = material_case
    first = pipeline.calculate_particle_section(case.state, 456.)
    seed = replace(first, specimen_interactions=None)
    seed.loaded_section_only = True
    monkeypatch.setattr(pipeline, "run", lambda *a, **k: replace(first.simulation))
    monkeypatch.setattr("temsim.specimen.elastic_transport.simulate_elastic_point_transport",
                        lambda *a, **k: pytest.fail("Compatible material collisions must not run again"))
    observed = []
    original = pipeline.build_geometric_specimen_exit
    def downstream(*args, **kwargs):
        observed.append(kwargs.get("existing_exit"))
        return original(*args, **kwargs)
    monkeypatch.setattr(pipeline, "build_geometric_specimen_exit", downstream)
    result = pipeline.calculate(case.state, existing_result=seed)
    assert result.specimen_interactions.elastic_transport is first.specimen_interactions.elastic_transport
    assert "elastic" in result.reused_products
    assert observed == [first.specimen_exit]
    records = result.specimen_exit.metrics["material_section_resume"]
    assert records and all(row["hit"] and row["resume_z_mm"] == 456. for row in records)
    for branch in result.specimen_exit.branches:
        if np.any(branch.energy_offset_ev < 0.):
            assert np.all(np.isnan(branch.flight_time_s))


@pytest.mark.parametrize("field", ["x_m", "y_m", "tx_rad", "ty_rad"])
def test_partial_precision_checkpoint_is_not_admitted(optical_fixture, field):
    from temsim.physics.particle_sections import run_particle_section
    state, _, _ = optical_fixture
    result = run_particle_section(state, observation_stop_z_mm=453., resolved_layout=object())
    original = result.section_checkpoint
    segment = original.segments[0]
    bad = replace(segment.checkpoints)
    # Deliberately corrupt a runtime object after the normal constructor has
    # enforced its dtype contract; it must not be admitted as executed state.
    object.__setattr__(bad, field, getattr(segment.checkpoints, field).astype(np.float32))
    with pytest.raises(ValueError, match="full-precision"):
        validate_section_checkpoint(replace(original, segments=(replace(segment, checkpoints=bad),)))


def test_complete_axial_transport_stops_at_filter_entrance_without_later_kick(optical_fixture, monkeypatch):
    from temsim.physics.simulation import run
    state, gun, _ = optical_fixture
    state.energy_filter_installed = True
    state.energy_filter_mode = "energy_filter"
    state.energy_filter.enabled = True
    state.energy_filter.entrance_z_mm = 456.
    deflector = state.deflectors[0]
    deflector.enabled = True
    monkeypatch.setattr(type(deflector), "kick_events",
                        lambda *a, **k: ((455., .001, 0.), (457., .1, 0.)))
    result = run(state, resolved_layout=object())
    outgoing = result.branches["000"]
    assert outgoing.z[-1] == 456.
    np.testing.assert_allclose(outgoing.tx[-1], gun.exit_bundle.tx_rad+.001, rtol=1e-12, atol=1e-16)
