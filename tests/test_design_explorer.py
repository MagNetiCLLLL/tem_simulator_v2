import gc
from types import SimpleNamespace
import weakref

import pytest
from PySide6.QtCore import Qt

from temsim.design_explorer import (
    ArtifactState,
    PRODUCT_STAGES,
    HighAccuracyRequest,
    ProductStatus,
    ResultProductSummary,
    capture_design_snapshot,
    diff_design_snapshots,
    evaluate_product_statuses,
    summarise_calculation_result,
)
from temsim.calculation_cache import (
    calculation_signatures,
    calculation_signatures_for_request,
)
from temsim.gui.calculation_controller import CalculationController
from temsim.gui.main_window import MainWindow
from temsim.gui.visualization import VisualizationWorkspace
from temsim.optics.column import default_state
from temsim.simulation_pipeline import CalculationResult
from temsim.specimen.downstream_transport import GeometricSpecimenExit
from temsim.specimen.sample_region import SampleRegionResult


def _signatures(request_key="request", *, suffix="same"):
    return {
        "request": request_key,
        **{
            stage.signature_key: f"{stage.signature_key}-{suffix}"
            for stage in PRODUCT_STAGES
        },
    }


def _capture(
    payload,
    *,
    slot,
    selection=None,
    ray_count=100,
    step_mm=0.1,
    model_signature="model",
    external_signature="",
    request_key=None,
):
    return capture_design_snapshot(
        payload,
        selection
        or {
            "gun": "FEG",
            "column": "C3 + Probe Corrector",
            "recording": "Energy Filter",
            "beam_blanker": "None",
        },
        slot=slot,
        request=HighAccuracyRequest(ray_count, step_mm),
        model_signature=model_signature,
        external_signature=external_signature,
        request_signatures=_signatures(request_key or f"request-{slot}"),
        captured_at_utc="2026-09-04T12:00:00+00:00",
    )


def _statuses_by_key(*args, **kwargs):
    return {
        item.stage.key: item
        for item in evaluate_product_statuses(*args, **kwargs)
    }


def test_snapshot_is_recursively_detached_from_mutable_inputs():
    payload = {
        "lenses": [
            {"key": "c1", "percent": 20.0},
            {"key": "c2", "percent": 40.0},
        ],
        "sample": {
            "virtual_regions": [
                {
                    "key": "region-1",
                    "weights": [1.0, {"density": 0.5}],
                }
            ],
            "frozen_phonon_sigma_by_element_angstrom": {"Si": 0.075},
        },
    }
    selection = {
        "gun": "FEG",
        "column": "C3 + Probe Corrector",
        "recording": "Energy Filter",
        "beam_blanker": "None",
    }

    snapshot = _capture(payload, slot="A", selection=selection)
    payload["lenses"][0]["percent"] = 99.0
    payload["sample"]["virtual_regions"][0]["weights"][1][
        "density"
    ] = 0.9
    payload["sample"]["frozen_phonon_sigma_by_element_angstrom"][
        "Si"
    ] = 0.12
    selection["gun"] = "Thermionic"

    assert snapshot.state_payload["lenses"][0]["percent"] == 20.0
    region = snapshot.state_payload["sample"]["virtual_regions"][0]
    assert region["weights"][1]["density"] == 0.5
    assert snapshot.state_payload["sample"][
        "frozen_phonon_sigma_by_element_angstrom"
    ]["Si"] == pytest.approx(0.075)
    assert snapshot.selection["gun"] == "FEG"
    with pytest.raises(TypeError):
        snapshot.state_payload["sample"]["new_value"] = 1
    with pytest.raises(TypeError):
        region["weights"][1]["density"] = 1.0


def test_keyed_component_list_diff_is_stable_across_reordering():
    snapshot_a = _capture(
        {
            "lenses": [
                {"key": "c1", "percent": 10.0},
                {"key": "c2", "percent": 20.0},
            ]
        },
        slot="A",
        model_signature="model-a",
    )
    snapshot_b = _capture(
        {
            "lenses": [
                {"key": "c2", "percent": 30.0},
                {"key": "c1", "percent": 10.0},
            ]
        },
        slot="B",
        model_signature="model-b",
    )

    differences = diff_design_snapshots(snapshot_a, snapshot_b)

    assert [difference.path for difference in differences] == [
        "state.lenses[c2].percent"
    ]
    assert differences[0].value_a == 20.0
    assert differences[0].value_b == 30.0


def test_external_model_change_is_reported_alongside_setting_changes():
    snapshot_a = _capture(
        {"lenses": [{"key": "c1", "percent": 10.0}]},
        slot="A",
        external_signature="external-a",
    )
    snapshot_b = _capture(
        {"lenses": [{"key": "c1", "percent": 20.0}]},
        slot="B",
        external_signature="external-b",
    )

    differences = diff_design_snapshots(snapshot_a, snapshot_b)

    assert [difference.path for difference in differences] == [
        "identity.external_model_toml_or_cif",
        "state.lenses[c1].percent",
    ]


def test_diff_reports_selection_and_high_accuracy_request_controls():
    payload = {"sample": {"specimen_mode": "virtual"}}
    snapshot_a = _capture(
        payload,
        slot="A",
        selection={
            "gun": "FEG",
            "column": "C3 + Probe Corrector",
            "recording": "Energy Filter",
            "beam_blanker": "None",
        },
        ray_count=100,
        step_mm=0.1,
        request_key="request-a",
    )
    snapshot_b = _capture(
        payload,
        slot="B",
        selection={
            "gun": "FEG + Mono",
            "column": "C3 + Probe Corrector",
            "recording": "Energy Filter",
            "beam_blanker": "None",
        },
        ray_count=250,
        step_mm=0.05,
        request_key="request-b",
    )

    differences = diff_design_snapshots(snapshot_a, snapshot_b)

    assert [difference.path for difference in differences] == [
        "assembly.gun",
        "calculation.high_accuracy_ray_count",
        "calculation.high_accuracy_step_mm",
    ]
    assert differences[0].value_a == "FEG"
    assert differences[0].value_b == "FEG + Mono"
    assert differences[1].value_a == 100
    assert differences[1].value_b == 250
    assert differences[2].value_a == pytest.approx(0.1)
    assert differences[2].value_b == pytest.approx(0.05)


def test_result_summary_copies_only_small_metadata_and_does_not_pin_result():
    class Payload:
        pass

    result = Payload()
    simulation = Payload()
    incident = Payload()
    bulky_product = Payload()
    simulation.incident = incident
    result.simulation = simulation
    result.energy_filter = bulky_product
    result.signatures = {
        "request": "request-a",
        "column": "column-a",
        "incident": "incident-a",
    }
    result.model_signature = "model-a"
    result.calculated_products = {"column"}
    result.reused_products = {"incident"}
    result_ref = weakref.ref(result)
    simulation_ref = weakref.ref(simulation)
    product_ref = weakref.ref(bulky_product)

    summary = summarise_calculation_result(result)
    result.signatures["request"] = "mutated-after-summary"
    del result, simulation, incident, bulky_product
    gc.collect()

    assert result_ref() is None
    assert simulation_ref() is None
    assert product_ref() is None
    assert summary.request_signature == "request-a"
    assert summary.signatures["request"] == "request-a"
    assert summary.available_stage_keys == {
        "column",
        "incident",
        "energy_filter",
    }
    assert summary.calculated_products == {"column"}
    assert summary.reused_products == {"incident"}


def test_product_statuses_cover_ready_reusable_recalculate_and_off():
    source_signatures = _signatures("request-a", suffix="source")
    exact_summary = ResultProductSummary(
        model_signature="model-a",
        request_signature="request-a",
        signatures=source_signatures,
        available_stage_keys={"incident", "column"},
        calculated_products={"column"},
        reused_products={"incident"},
    )
    exact = _statuses_by_key(
        source_signatures,
        exact_summary,
        requested_stage_keys={"incident", "column"},
    )

    assert exact["incident"].status is ProductStatus.READY
    assert exact["column"].status is ProductStatus.READY
    assert exact["wave"].status is ProductStatus.OFF
    assert exact["incident"].artifact_state is ArtifactState.REUSED
    assert exact["column"].artifact_state is ArtifactState.CALCULATED
    assert exact["wave"].artifact_state is ArtifactState.MISSING
    assert exact["wave"].requested is False

    current_signatures = dict(source_signatures)
    current_signatures["request"] = "request-b"
    current_signatures["column"] = "column-new"
    changed = _statuses_by_key(
        current_signatures,
        exact_summary,
        requested_stage_keys={"incident", "column"},
    )

    assert changed["incident"].status is ProductStatus.REUSABLE
    assert changed["column"].status is ProductStatus.RECALCULATE
    assert changed["wave"].status is ProductStatus.OFF
    assert changed["incident"].artifact_state is ArtifactState.REUSED
    assert changed["column"].artifact_state is ArtifactState.STALE


@pytest.mark.parametrize("current_request", ("request-a", "request-b"))
def test_matching_signature_without_artifact_is_never_reported_reusable(
    current_request,
):
    source_signatures = _signatures("request-a", suffix="source")
    summary = ResultProductSummary(
        model_signature="model-a",
        request_signature="request-a",
        signatures=source_signatures,
        available_stage_keys=frozenset(),
        calculated_products={"incident"},
        reused_products={"incident"},
    )
    current_signatures = dict(source_signatures)
    current_signatures["request"] = current_request

    statuses = _statuses_by_key(
        current_signatures,
        summary,
        requested_stage_keys={"incident"},
    )

    assert statuses["incident"].status is ProductStatus.RECALCULATE
    assert statuses["incident"].reason == "No cached product is available"
    assert statuses["incident"].artifact_state is ArtifactState.MISSING


def test_no_illumination_terminal_state_reuses_when_dependencies_match():
    signatures = _signatures("request-dark", suffix="dark")
    result = CalculationResult(
        simulation=SimpleNamespace(
            incident=object(),
            metrics={"sample_illumination_status": "no incident current"},
        ),
        energy_filter=None,
        signatures=signatures,
        calculated_products=frozenset({"column"}),
    )
    summary = summarise_calculation_result(result)

    exact = _statuses_by_key(
        signatures,
        summary,
        requested_stage_keys={
            "elastic",
            "eds",
            "wave",
            "wave_source",
            "sample_downstream",
        },
    )
    assert {
        exact[key].artifact_state
        for key in (
            "elastic",
            "eds",
            "wave",
            "wave_source",
            "sample_downstream",
        )
    } == {ArtifactState.CALCULATED}

    changed = dict(signatures)
    changed["request"] = "request-lit"
    nonexact = _statuses_by_key(
        changed,
        summary,
        requested_stage_keys={
            "elastic",
            "eds",
            "wave",
            "wave_source",
            "sample_downstream",
        },
    )
    assert {
        nonexact[key].artifact_state
        for key in (
            "elastic",
            "eds",
            "wave",
            "wave_source",
            "sample_downstream",
        )
    } == {ArtifactState.REUSED}

    changed["elastic"] = "new-elastic-dependencies"
    invalidated = _statuses_by_key(
        changed,
        summary,
        requested_stage_keys={"elastic"},
    )
    assert invalidated["elastic"].artifact_state is ArtifactState.MISSING

    changed["sample_downstream"] = "new-downstream-dependencies"
    invalidated = _statuses_by_key(
        changed,
        summary,
        requested_stage_keys={"sample_downstream"},
    )
    assert (
        invalidated["sample_downstream"].artifact_state
        is ArtifactState.MISSING
    )


def test_describe_high_accuracy_reuse_does_not_change_lru_order():
    controller = CalculationController(
        high_cache_limit=4,
        high_cache_budget_bytes=1024**2,
    )
    state = default_state()
    request = controller.describe_high_accuracy_reuse(
        state,
        ray_count=1_000,
        step_mm=0.5,
    )
    signatures = request.request_signatures
    cached_results = (
        CalculationResult(
            simulation=SimpleNamespace(),
            energy_filter=None,
            signatures={
                "request": "oldest",
                "incident": signatures["incident"],
            },
        ),
        CalculationResult(
            simulation=SimpleNamespace(),
            energy_filter=None,
            signatures={
                "request": "middle",
                "column": signatures["column"],
            },
        ),
        CalculationResult(
            simulation=SimpleNamespace(),
            energy_filter=None,
            signatures={
                "request": "newest",
                "scan_geometry": signatures["scan_geometry"],
            },
        ),
    )
    for result in cached_results:
        controller._cache_result(result)
    order_before = tuple(controller._high_cache)

    controller.describe_high_accuracy_reuse(
        state,
        ray_count=1_000,
        step_mm=0.5,
    )

    assert order_before == ("oldest", "middle", "newest")
    assert tuple(controller._high_cache) == order_before


def test_fast_request_signatures_match_the_solver_snapshot():
    state = default_state()
    state.simulation_time_s = 0.125
    snapshot = CalculationController._calculation_snapshot(
        state,
        "High accuracy",
        15_000,
        0.1,
    )

    assert calculation_signatures_for_request(
        state,
        ray_count=15_000,
        step_mm=0.1,
    ) == calculation_signatures(snapshot)


def test_displayed_exact_result_is_ready_without_controller_cache_owner():
    controller = CalculationController(
        high_cache_limit=4,
        high_cache_budget_bytes=1,
    )
    state = default_state()
    initial = controller.describe_high_accuracy_reuse(
        state,
        ray_count=1_000,
        step_mm=0.5,
    )
    displayed = CalculationResult(
        simulation=SimpleNamespace(incident=object(), metrics={}),
        energy_filter=None,
        signatures=dict(initial.request_signatures),
        calculated_products=frozenset({"column", "diagnostics"}),
    )
    summary = summarise_calculation_result(displayed)

    described = controller.describe_high_accuracy_reuse(
        state,
        ray_count=1_000,
        step_mm=0.5,
        completed_summary=summary,
    )
    statuses = {
        item.stage.key: item.artifact_state
        for item in described.product_statuses
    }

    assert controller._high_cache == {}
    assert statuses["column"] is ArtifactState.CALCULATED
    assert statuses["incident"] is ArtifactState.CALCULATED
    assert statuses["diagnostics"] is ArtifactState.CALCULATED


def test_migrated_vacuum_does_not_request_specimen_transport_products():
    controller = CalculationController()
    state = default_state()
    state.sample.specimen_mode = "virtual"
    state.sample.specimen_preset_key = "vacuum"
    state.sample.inserted = True
    state.sample.eds_enabled = True
    # A legacy vacuum preset becomes a retracted real-structure source when
    # loaded; retired modes must not be sent directly to the solver.
    payload = state.to_dict()
    payload["schema_version"] = 76
    state = type(state).from_dict(payload)
    assert state.sample.specimen_mode == "reference"
    assert state.sample.inserted is False

    described = controller.describe_high_accuracy_reuse(
        state,
        ray_count=1_000,
        step_mm=0.5,
    )
    statuses = {item.stage.key: item for item in described.product_statuses}

    assert statuses["elastic"].requested is False
    assert statuses["eds"].requested is False
    assert statuses["sample_region"].requested is False
    assert statuses["sample_downstream"].requested is False


def test_disabled_scan_is_reported_off_instead_of_permanently_missing():
    controller = CalculationController()
    state = default_state()
    state.ac_deflector.scan_enabled = False
    state.descan_deflector.scan_enabled = False

    described = controller.describe_high_accuracy_reuse(
        state,
        ray_count=1_000,
        step_mm=0.5,
    )
    statuses = {item.stage.key: item for item in described.product_statuses}

    assert statuses["scan_geometry"].requested is False
    assert statuses["scan_ray_paths"].requested is False
    assert statuses["stem"].requested is False
    assert statuses["scan_geometry"].status is ProductStatus.OFF
    assert statuses["scan_ray_paths"].status is ProductStatus.OFF


def test_descan_only_requests_geometry_but_not_ray_playback_or_stem():
    controller = CalculationController()
    state = default_state()
    state.ac_deflector.scan_enabled = False
    state.descan_deflector.enabled = True
    state.descan_deflector.scan_enabled = True

    described = controller.describe_high_accuracy_reuse(
        state,
        ray_count=1_000,
        step_mm=0.5,
    )
    statuses = {item.stage.key: item for item in described.product_statuses}

    assert statuses["scan_geometry"].requested is True
    assert statuses["scan_ray_paths"].requested is False
    assert statuses["stem"].requested is False


def test_geometric_real_stem_requests_first_class_specimen_exit():
    controller = CalculationController()
    state = default_state()
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = "configured.cif"
    state.sample.stem_wave_enabled = False
    state.ac_deflector.enabled = True
    state.ac_deflector.scan_enabled = True

    described = controller.describe_high_accuracy_reuse(
        state,
        ray_count=1_000,
        step_mm=0.5,
    )
    statuses = {item.stage.key: item for item in described.product_statuses}

    assert statuses["elastic"].requested is True
    assert statuses["sample_downstream"].requested is True
    assert statuses["stem"].requested is True


def test_post_sample_lens_edit_keeps_specimen_products_reusable():
    controller = CalculationController(
        high_cache_limit=4,
        high_cache_budget_bytes=1024**2,
    )
    state = default_state()
    state.ac_deflector.enabled = True
    state.ac_deflector.scan_enabled = True
    initial = controller.describe_high_accuracy_reuse(
        state,
        ray_count=1_000,
        step_mm=0.5,
    )
    interactions = SimpleNamespace(
        elastic_transport=object(),
    )
    interactions.eds_spectrum = SimpleNamespace(
        elastic_transport=interactions.elastic_transport,
    )
    interactions.metrics = {
        "dependency_signatures": dict(initial.request_signatures),
    }
    checkpoint = GeometricSpecimenExit(
        (),
        {
            "tracked_downstream_source_probability": 0.0,
            "inelastic_absorbed_source_probability": 0.0,
        },
        dependency_signature=initial.request_signatures[
            "sample_downstream"
        ],
    )
    sample_region = SampleRegionResult(
        entry_z_mm=1.0,
        exit_z_mm=2.0,
        electron_paths=(),
        photon_paths=(),
        downstream_branches=(),
        spectrum=interactions.eds_spectrum,
        interactions=interactions,
        metrics={
            "sample_region_signature": initial.request_signatures[
                "sample_region"
            ],
            "sample_downstream_signature": initial.request_signatures[
                "sample_downstream"
            ],
            "tracked_downstream_source_probability": 0.0,
            "inelastic_absorbed_source_probability": 0.0,
        },
        specimen_exit=checkpoint,
    )
    result = CalculationResult(
        simulation=SimpleNamespace(incident=object()),
        energy_filter=None,
        specimen_interactions=interactions,
        scan_geometry=object(),
        scan_ray_paths=object(),
        specimen_exit=checkpoint,
        sample_region=sample_region,
        signatures=dict(initial.request_signatures),
        calculated_products=frozenset({
            "column",
            "elastic",
            "eds",
            "scan_geometry",
            "scan_ray_paths",
        }),
    )
    controller._cache_result(result)
    projector = next(
        lens for lens in state.lenses if lens.key == "projector_lens_1"
    )
    projector.percent += 0.25

    changed = controller.describe_high_accuracy_reuse(
        state,
        ray_count=1_000,
        step_mm=0.5,
    )
    statuses = {
        item.stage.key: item.status
        for item in changed.product_statuses
    }

    assert statuses["incident"] is ProductStatus.REUSABLE
    assert statuses["elastic"] is ProductStatus.REUSABLE
    assert statuses["eds"] is ProductStatus.REUSABLE
    assert statuses["sample_region"] is ProductStatus.REUSABLE
    assert statuses["sample_downstream"] is ProductStatus.RECALCULATE
    assert statuses["column"] is ProductStatus.RECALCULATE
    assert statuses["scan_geometry"] is ProductStatus.RECALCULATE
    assert statuses["scan_ray_paths"] is ProductStatus.RECALCULATE


def test_result_summary_rejects_unsigned_sample_cache_placeholders():
    signatures = _signatures("request-a", suffix="sample")
    result = CalculationResult(
        simulation=SimpleNamespace(incident=object(), metrics={}),
        energy_filter=None,
        sample_region=object(),
        specimen_exit=GeometricSpecimenExit((), {}),
        signatures=signatures,
    )

    summary = summarise_calculation_result(result)

    assert "sample_region" not in summary.available_stage_keys
    assert "sample_downstream" not in summary.available_stage_keys


def test_capture_buttons_compare_designs_without_submitting_solver(
    qtbot,
    monkeypatch,
):
    window = MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    submissions = []
    monkeypatch.setattr(
        window.calculations,
        "submit",
        lambda *args, **kwargs: submissions.append((args, kwargs)),
    )
    page = window.workspace.design_explorer

    with qtbot.waitSignal(page.capture_requested) as captured_a:
        qtbot.mouseClick(page.capture_a, Qt.MouseButton.LeftButton)
    assert captured_a.args == ["A"]
    assert page.snapshot_a is not None

    window.high_rays.setValue(window.high_rays.value() + 1_000)
    with qtbot.waitSignal(page.capture_requested) as captured_b:
        qtbot.mouseClick(page.capture_b, Qt.MouseButton.LeftButton)
    assert captured_b.args == ["B"]
    assert page.snapshot_b is not None

    displayed_paths = {
        page.diff_table.item(row, 0).text()
        for row in range(page.diff_table.rowCount())
    }
    assert "calculation › high_accuracy_ray_count" in displayed_paths
    assert page.compare_summary.text() == "1 changed settings"
    assert submissions == []


def test_visualization_workspace_places_design_explorer_last(qtbot):
    workspace = VisualizationWorkspace()
    qtbot.addWidget(workspace)

    last_index = workspace.tabs.count() - 1

    assert workspace.tabs.widget(last_index) is workspace.design_explorer
    assert workspace.tabs.tabText(last_index) == "Design Explorer"


def test_preview_display_preserves_captured_designs(qtbot, monkeypatch):
    workspace = VisualizationWorkspace()
    qtbot.addWidget(workspace)
    snapshot_a = _capture(
        {"sample": {"size_x_nm": 1_000.0}},
        slot="A",
    )
    snapshot_b = _capture(
        {"sample": {"size_x_nm": 2_000.0}},
        slot="B",
    )
    workspace.design_explorer.set_capture(snapshot_a)
    workspace.design_explorer.set_capture(snapshot_b)

    monkeypatch.setattr(
        "temsim.gui.visualization.sample_illumination_absent",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        workspace,
        "_prepare_scan_ray_playback",
        lambda _result: None,
    )
    monkeypatch.setattr(
        workspace,
        "_draw_ray_diagram",
        lambda *_args, **_kwargs: None,
    )
    for view in (
        workspace.physical_layout,
        workspace.magnetic_field,
        workspace.probe_aberrations,
        workspace.image_aberrations,
        workspace.optical_transfer,
        workspace.transverse_beam,
        workspace.sample_page,
        workspace.scan_control,
    ):
        monkeypatch.setattr(
            view,
            "display_result",
            lambda *_args, **_kwargs: None,
        )

    preview = SimpleNamespace(
        model_signature="preview-model",
        simulation=object(),
        state_snapshot=object(),
        scan_geometry=None,
        stem_scan=None,
    )

    workspace.display_result(preview, "Preview")

    assert workspace.design_explorer.snapshot_a is snapshot_a
    assert workspace.design_explorer.snapshot_b is snapshot_b
    assert workspace.design_explorer.diff_table.rowCount() == 1
