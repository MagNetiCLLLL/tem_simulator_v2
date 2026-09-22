"""Pipeline ownership and provenance, with expensive solvers replaced by fixtures."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from temsim import simulation_pipeline as pipeline
from temsim.assembly_catalog import AssemblyCatalog
from temsim.specimen.interaction_types import ConservationCheck
from temsim.optics.column import default_state
from temsim.physics.simulation import Simulation
from temsim.specimen.downstream_transport import GeometricSpecimenExit


@pytest.fixture
def pipeline_case(monkeypatch):
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, replace(catalog.default_selection(), recording="Energy Filter"))
    state.sample.wave_enabled = state.sample.stem_wave_enabled = False
    state.sample.eds_enabled = False
    state.sample.inserted = False
    state.ac_deflector.enabled = state.ac_deflector.scan_enabled = True
    state.ac_deflector.wobble_enabled = False
    state.descan_deflector.scan_enabled = False
    state.sample.stem_image_enabled = True
    original_branch = SimpleNamespace(name="reference", weight=1.)
    detailed_branch = SimpleNamespace(name="finite_exit", weight=.4)
    simulation = Simulation(incident=SimpleNamespace(), branches={"reference": original_branch},
                            metrics={"sample_beam_surviving_fraction": 1.}, gun_trace=object())
    elastic = object()
    interactions = SimpleNamespace(elastic_transport=elastic, eds_spectrum=None,
        inelastic_distribution=None, completed_observables=frozenset(), metrics={},
        conservation=(ConservationCheck("fixture population", "probability", "1", 1., (("retained", 1.),)),))
    order, seen, progress = [], [], []
    monkeypatch.setattr(pipeline, "_geometric_specimen_transport_requested", lambda state: True)
    monkeypatch.setattr(pipeline, "sample_illumination_absent", lambda *args: False)
    monkeypatch.setattr(pipeline, "run", lambda *args, **kwargs: simulation)
    monkeypatch.setattr(pipeline, "retain_specimen_observables", lambda *args: interactions)
    monkeypatch.setattr(pipeline, "run_specimen_interactions", lambda *args, **kwargs: interactions)
    monkeypatch.setattr(pipeline, "calculate_scan_geometry", lambda *args: object())
    monkeypatch.setattr(pipeline, "calculate_scan_ray_paths", lambda *args: object())
    monkeypatch.setattr(pipeline, "detect_all_lens_crossovers", lambda *args: ())
    monkeypatch.setattr(pipeline, "aperture_stop_records", lambda *args: ())

    def build(state, sim, passed_elastic, *args, **kwargs):
        assert sim is simulation and passed_elastic is elastic
        order.append("specimen_exit")
        return GeometricSpecimenExit((detailed_branch,),
            {"tracked_downstream_source_probability": .4,
             "inelastic_absorbed_source_probability": .1},
            dependency_signature=kwargs["dependency_signature"])

    def stem(state, sim, **kwargs):
        order.append("stem")
        assert kwargs["geometric_specimen_exit"].branches == (detailed_branch,)
        return object()

    def energy_filter(state, sim, *args, **kwargs):
        order.append("energy_filter")
        seen.append(sim)
        return SimpleNamespace(
            entrance_provenance=sim.metrics["energy_filter_entrance_provenance"],
            entrance_dependency_signature=sim.metrics["energy_filter_entrance_dependency_signature"])

    monkeypatch.setattr(pipeline, "build_geometric_specimen_exit", build)
    monkeypatch.setattr(pipeline, "calculate_stem_scan_frame", stem)
    monkeypatch.setattr(pipeline, "simulate_energy_filter", energy_filter)
    return SimpleNamespace(state=state, simulation=simulation, branch=detailed_branch,
                           order=order, seen=seen, progress=progress)


def test_filter_runs_once_after_specimen_exit_and_stem_and_retains_real_source(pipeline_case):
    c = pipeline_case
    result = pipeline.calculate(c.state, progress_callback=lambda *event: c.progress.append(event))
    assert c.order == ["specimen_exit", "stem", "energy_filter"]
    assert len(c.seen) == 1
    source = c.seen[0]
    assert tuple(source.branches.values()) == (c.branch,)
    assert source.incident is c.simulation.incident
    assert source.gun_trace is c.simulation.gun_trace
    assert source.metrics["branch_weights_are_absolute"] is True
    assert list(c.simulation.branches) == ["reference"]
    assert "energy_filter_entrance_provenance" not in c.simulation.metrics
    assert result.energy_filter.entrance_provenance == "validated_specimen_exit"
    assert result.energy_filter.entrance_dependency_signature == result.signatures["sample_downstream"]
    stages = [row["stage"] for row in result.performance["stages"]]
    assert stages.index("Tracing the energy filter") > stages.index("Calculating the STEM detector frame")
    assert stages.index("Tracing the energy filter") > stages.index("Propagating specimen-exit electrons downstream")
    assert c.progress[-1][2] == "Complete"
    positions = [done / total for done, total, _ in c.progress]
    assert positions == sorted(positions)


def test_matching_filter_cache_requires_the_same_validated_entrance(pipeline_case):
    c = pipeline_case
    first = pipeline.calculate(c.state)
    c.order.clear()
    second = pipeline.calculate(c.state, existing_result=first)
    assert c.order == []
    assert second.energy_filter is first.energy_filter
    assert "energy_filter" in second.reused_products
    assert second.specimen_exit is first.specimen_exit

    old_reference = SimpleNamespace(entrance_provenance="optical_column_reference",
        entrance_dependency_signature=first.signatures["column"])
    prior = replace(first, energy_filter=old_reference)
    third = pipeline.calculate(c.state, existing_result=prior)
    assert c.order == ["energy_filter"]
    assert "energy_filter" in third.calculated_products
    assert "sample_downstream" in third.reused_products
    assert tuple(c.seen[-1].branches.values()) == (c.branch,)


def test_filter_view_without_detailed_exit_is_explicit_and_does_not_mutate_column():
    original = Simulation(incident=object(), branches={"reference": object()}, metrics={"kept": True})
    view = pipeline._energy_filter_source_view(original, None,
        {"column": "column-version", "sample_downstream": "detailed-version"})
    assert view.branches is original.branches
    assert view.metrics["energy_filter_entrance_provenance"] == "optical_column_reference"
    assert view.metrics["energy_filter_entrance_dependency_signature"] == "column-version"
    assert original.metrics == {"kept": True}


def test_filter_view_refuses_wrong_specimen_exit_signature():
    original = Simulation(incident=object(), branches={}, metrics={})
    wrong = GeometricSpecimenExit((), {"tracked_downstream_source_probability": 0.,
        "inelastic_absorbed_source_probability": 0.}, dependency_signature="old")
    with pytest.raises(ValueError, match="validated specimen-exit"):
        pipeline._energy_filter_source_view(original, wrong,
            {"column": "column-version", "sample_downstream": "current"})


def test_no_illumination_has_no_filter_population_and_explicit_provenance():
    original = Simulation(incident=object(), branches={"reference": object()}, metrics={})
    view = pipeline._energy_filter_source_view(original, None,
        {"column": "column-version", "sample_downstream": "detailed-version"},
        no_illumination=True)
    assert view.branches == {}
    assert view.metrics["energy_filter_entrance_provenance"] == "no_incident_illumination"


def test_no_illumination_filter_result_remains_reusable(pipeline_case, monkeypatch):
    c = pipeline_case
    c.state.ac_deflector.scan_enabled = False
    c.state.sample.stem_image_enabled = False
    monkeypatch.setattr(pipeline, "sample_illumination_absent", lambda *args: True)
    first = pipeline.calculate(c.state)
    assert c.order == ["energy_filter"]
    assert first.energy_filter.entrance_provenance == "no_incident_illumination"
    c.order.clear()
    second = pipeline.calculate(c.state, existing_result=first)
    assert c.order == []
    assert second.energy_filter is first.energy_filter
    assert "energy_filter" in second.reused_products


def test_uninstalled_filter_does_not_build_an_entrance_or_report_filter_tracing(
    pipeline_case, monkeypatch,
):
    c = pipeline_case
    catalog = AssemblyCatalog()
    catalog.apply(c.state, catalog.default_selection(), preserve_operating_parameters=True)

    def forbidden(*args, **kwargs):
        pytest.fail("An inactive energy filter must not request an entrance or trace")

    monkeypatch.setattr(pipeline, "_energy_filter_source_view", forbidden)
    monkeypatch.setattr(pipeline, "simulate_energy_filter", forbidden)
    result = pipeline.calculate(c.state, progress_callback=lambda *event: c.progress.append(event))
    assert result.energy_filter is None
    assert c.order == ["specimen_exit", "stem"]
    assert result.specimen_exit.branches == (c.branch,)
    assert all("energy filter" not in row["stage"].lower()
               for row in result.performance["stages"])
    assert c.progress[-1][2] == "Complete"
    fractions = [done / total for done, total, _ in c.progress]
    assert fractions == sorted(fractions)
