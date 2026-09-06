from temsim.gui.model_inspector import ModelInspectorPage
from temsim.optics.column import default_state
from temsim.optics.model import State


def test_model_controls_are_explicit_serializable_and_do_not_solve(qtbot, monkeypatch):
    from temsim.physics import axisymmetric_magnetostatics
    monkeypatch.setattr(axisymmetric_magnetostatics, "solve_geometry_field_map", lambda *args: (_ for _ in ()).throw(AssertionError("UI must not solve")))
    state = default_state()
    original_strengths = [lens.percent for lens in state.lenses]
    page = ModelInspectorPage()
    qtbot.addWidget(page)
    page.set_state(state)
    page.permeability.setText("1000")
    page.ampere_turns.setText("250")
    with qtbot.waitSignal(page.changed):
        page._apply_field()
    key = page.lens.currentData()
    assert state.lens_field_map_descriptors[key]["solver"] == "axisymmetric_linear_fem"
    assert original_strengths == [lens.percent for lens in state.lenses]
    page.aberration_mode.setCurrentIndex(page.aberration_mode.findData("field_derived"))
    with qtbot.waitSignal(page.changed):
        page._apply_mode()
    restored = State.from_dict(state.to_dict())
    assert restored.probe_aberrations["mode"] == "field_derived"
    assert restored.lens_field_map_descriptors == state.lens_field_map_descriptors
    assert all(page.table.item(row, 1).text() != "Validated" for row in range(page.table.rowCount()))


def test_field_fit_is_reused_by_detached_snapshot_and_display(qtbot, monkeypatch):
    from temsim.optics.column import default_state
    from temsim.optics import aberrations, field_aberrations
    from temsim.physics import lens_field_provider
    from temsim.gui.aberration_view import AberrationComparisonView

    state = default_state()
    # Production workers start from the normalised serialization boundary.
    state = type(state).from_dict(state.to_dict())
    state.image_aberrations = {"mode": "field_derived"}
    coefficient = aberrations.EffectiveAberrationSet("objective image", "field-derived", c3_mm=.42)
    diagnostics = {
        "fit_rms_m": 1e-12, "fit_condition_number": 5,
        "source": "Test-only fit", "diagnostic_scope": "UI cache contract",
        "unmeasured_coefficients": (),
        "coefficient_status": {term: "field ray fit" for term, _, _ in aberrations.SYSTEM_COEFFICIENT_ROWS},
        "correction_comparison_available": False,
    }
    calls = []
    def fit(state, system):
        calls.append(system)
        return coefficient, coefficient, diagnostics
    monkeypatch.setattr(field_aberrations, "derive_field_aberrations", fit)
    monkeypatch.setattr(lens_field_provider, "active_mapped_providers", lambda state: ())
    aberrations.prepare_field_aberration_diagnostics(state)
    snapshot = type(state).from_dict(state.to_dict())
    aberrations.prepare_field_aberration_diagnostics(snapshot, state)
    assert calls == ["image"]
    assert snapshot._effective_aberration_cache is not state._effective_aberration_cache

    view = AberrationComparisonView(fixed_system="image")
    qtbot.addWidget(view)
    view._state = snapshot
    view._refresh()
    view._refresh()
    assert calls == ["image"]
    assert "correction comparison not calculated" in view.summary.text()
    assert view.table.isColumnHidden(2)
    assert view.table.isColumnHidden(4)
    assert view.table.rowCount() == len(aberrations.SYSTEM_COEFFICIENT_ROWS)
