import json
from dataclasses import replace

import pytest

from temsim.design_experiments import (
    MetricObservation,
    ParameterEdit,
    StateHistory,
    SweepAxis,
    ToleranceRule,
    estimate_sensitivities,
    evaluate_tolerances,
    load_recipe,
    plan_parameter_sweep,
    recipe_from_snapshot,
    replace_parameter,
    save_recipe,
)
from temsim.design_explorer import (
    HighAccuracyRequest,
    capture_design_snapshot,
)
from temsim.calculation_manifest import ExternalInputIdentity
from temsim.gui.design_explorer import DesignExplorerPage


def _snapshot(slot="A"):
    return capture_design_snapshot(
        {
            "lenses": [
                {"key": "c1", "percent": 20.0},
                {"key": "c2", "percent": 40.0},
            ],
            "sample": {"thickness_nm": 10.0},
        },
        {
            "gun": "FEG",
            "column": "C3 + Probe Corrector",
            "recording": "Energy Filter",
            "beam_blanker": "None",
        },
        slot=slot,
        request=HighAccuracyRequest(100, 0.1),
        model_signature="model",
        external_signature="external",
        external_inputs=(ExternalInputIdentity(
            role="specimen:cif",
            path="C:/captured/specimen.cif",
            available=True,
            size_bytes=12,
            sha256="a" * 64,
        ),),
        geometry_fingerprint="geometry",
        request_signatures={"request": "request", "incident": "incident"},
        captured_at_utc="2026-09-05T12:00:00+00:00",
    )


def test_replace_parameter_addresses_components_by_stable_key():
    replaced = replace_parameter(
        _snapshot().state_payload,
        "lenses[c2].percent",
        55.0,
    )

    assert replaced["lenses"][0]["percent"] == pytest.approx(20.0)
    assert replaced["lenses"][1]["percent"] == pytest.approx(55.0)
    with pytest.raises(KeyError, match="missing"):
        replace_parameter(replaced, "lenses[missing].percent", 1.0)


def test_recipe_round_trip_history_and_branching(tmp_path):
    base = recipe_from_snapshot(_snapshot(), name="baseline")
    adjusted = base.with_edits(
        "C2 adjusted", (ParameterEdit("lenses[c2].percent", 50.0),)
    )
    path = tmp_path / "design-recipe.json"
    save_recipe(path, adjusted)

    restored = load_recipe(path)
    assert restored.digest == adjusted.digest
    assert restored.state_payload["lenses"][1]["percent"] == pytest.approx(50.0)
    assert restored.geometry_fingerprint == "geometry"
    assert restored.external_model_signature == "external"
    assert restored.external_inputs == adjusted.external_inputs

    history = StateHistory(limit=3)
    assert history.record(base)
    assert not history.record(base)
    assert history.record(adjusted)
    assert history.undo().digest == base.digest
    branch = base.with_edits(
        "C1 branch", (ParameterEdit("lenses[c1].percent", 25.0),)
    )
    assert history.record(branch)
    assert not history.can_redo

    document = json.loads(path.read_text(encoding="utf-8"))
    document["base_state_payload"]["sample"]["thickness_nm"] = 12.0
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_recipe(path)


def test_legacy_recipe_schema_loads_but_keeps_missing_provenance_explicit(
    tmp_path,
):
    current = recipe_from_snapshot(_snapshot(), name="current")
    legacy = replace(
        current,
        schema_version=1,
        external_model_signature="",
        external_inputs=(),
    )
    path = tmp_path / "legacy-design-recipe.json"
    save_recipe(path, legacy)

    restored = load_recipe(path)

    assert restored.schema_version == 1
    assert restored.external_model_signature == ""
    assert restored.external_inputs == ()
    assert restored.digest == legacy.digest


def test_sweep_sensitivity_and_tolerance_are_quantitative():
    recipe = recipe_from_snapshot(_snapshot(), name="sensitivity")
    sweep = plan_parameter_sweep(
        recipe,
        (
            SweepAxis("lenses[c1].percent", (10.0, 20.0, 30.0), "%"),
            SweepAxis("lenses[c2].percent", (30.0, 50.0), "%"),
        ),
    )
    observations = []
    for point in sweep.points:
        c1 = point.coordinates["lenses[c1].percent"]
        c2 = point.coordinates["lenses[c2].percent"]
        observations.append(MetricObservation(
            point.index,
            {"probe_nm": 2.0 * c1 + 3.0 * c2 + 5.0},
        ))

    estimates = {
        row.parameter_path: row
        for row in estimate_sensitivities(sweep, observations)
    }
    assert estimates["lenses[c1].percent"].derivative == pytest.approx(2.0)
    assert estimates["lenses[c2].percent"].derivative == pytest.approx(3.0)
    assert estimates["lenses[c1].percent"].r_squared == pytest.approx(1.0)

    results = evaluate_tolerances(
        {"probe_nm": 0.18, "current_pa": 10.0},
        (
            ToleranceRule("probe_nm", maximum=0.2),
            ToleranceRule("current_pa", minimum=20.0),
            ToleranceRule("missing", maximum=1.0),
        ),
    )
    assert [row.passed for row in results] == [True, False, False]


def test_sweep_is_bounded_and_does_not_mutate_recipe():
    recipe = recipe_from_snapshot(_snapshot(), name="bounded")
    original = recipe.state_payload["lenses"][0]["percent"]
    with pytest.raises(ValueError, match="limit"):
        plan_parameter_sweep(
            recipe,
            (SweepAxis("lenses[c1].percent", tuple(range(20))),),
            maximum_points=10,
        )
    assert recipe.state_payload["lenses"][0]["percent"] == original


def test_design_explorer_capture_exposes_recipe_history_and_sweep(qtbot):
    page = DesignExplorerPage()
    qtbot.addWidget(page)
    page.set_capture(_snapshot())

    assert page.state_history.current is not None
    assert page.recipe_for_slot("A").geometry_fingerprint == "geometry"
    sweep = page.plan_sweep(
        "A", (SweepAxis("lenses[c1].percent", (10.0, 20.0)),)
    )
    assert len(sweep.points) == 2
    observations = tuple(
        MetricObservation(
            point.index,
            {"probe_nm": 2.0 * point.coordinates["lenses[c1].percent"]},
        )
        for point in sweep.points
    )
    assert page.analyse_sweep(sweep, observations)[0].derivative == pytest.approx(2.0)
    assert page.check_tolerances(
        {"probe_nm": 0.2}, (ToleranceRule("probe_nm", maximum=0.3),)
    )[0].passed
