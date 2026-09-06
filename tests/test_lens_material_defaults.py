"""Reference/default contracts only; synthetic currents are not coil ratings."""

from copy import deepcopy
from pathlib import Path
import tomllib

import numpy as np
import pytest

from temsim.magnetic_materials import BHCurve, lens_material_defaults


def test_default_uses_pinned_source_values_without_refitting():
    defaults = lens_material_defaults()
    material = defaults["bh_material"]
    reference = defaults["linear_material_reference"]
    assert material["key"] == reference["material_key"] == "femm_pure_iron"
    assert reference["relative_permeability"] == 14872.0
    assert reference["source_fields"] == ["Mu_x", "Mu_y"]
    assert reference["source_sha256"] == material["source_sha256"]
    assert reference["source_url"] == material["source_url"]
    assert reference["source_block"] == "Pure Iron"
    assert len(material["b_t"]) == 21
    assert material["b_t"][1] == .227065
    assert material["h_a_per_m"][1] == 13.8984
    values, _ = BHCurve(material).evaluate(material["b_t"])
    np.testing.assert_allclose(values, material["h_a_per_m"], rtol=1e-13, atol=1e-12)
    with pytest.raises(ValueError, match="range exceeded"):
        BHCurve(material).evaluate([2.57])
    assert "ampere_turns" not in defaults


def test_defaults_return_independent_material_snapshots():
    first = lens_material_defaults()
    original = deepcopy(first)
    first["bh_material"]["h_a_per_m"][1] *= 2
    first["linear_material_reference"]["relative_permeability"] = 10
    first["linear_material_reference"]["selection_source_urls"].clear()
    assert lens_material_defaults() == original


@pytest.mark.parametrize("change, message", [
    ({"schema_version": 2}, "schema"),
    ({"material_key": "missing"}, "exactly one"),
    ({"linear_relative_permeability": -1}, "finite and positive"),
    ({"linear_relative_permeability": float("nan")}, "finite and positive"),
    ({"linear_relative_permeability": True}, "finite and positive"),
    ({"linear_source_fields": ["invented"]}, "source|cited"),
])
def test_invalid_defaults_fail_explicitly(monkeypatch, change, message):
    from temsim import magnetic_materials
    real_loads = tomllib.loads

    def altered(text):
        row = real_loads(text)
        if "schema_version" in row:
            row.update(change)
        return row

    monkeypatch.setattr(magnetic_materials.tomllib, "loads", altered)
    with pytest.raises(ValueError, match=message):
        lens_material_defaults()


def test_material_defaults_are_in_wheel_data_files():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    data_files = project["tool"]["setuptools"]["data-files"]
    assert "configs/materials/lens_defaults.toml" in data_files["configs/materials"]
    assert "configs/materials/magnetic/*.toml" in data_files["configs/materials/magnetic"]


def test_gui_prefills_material_only_and_retains_explicit_inputs(qtbot, monkeypatch):
    from temsim.gui.model_inspector import ModelInspectorPage
    from temsim.optics.column import default_state
    from temsim.optics.model import State
    from temsim.physics import axisymmetric_magnetostatics

    monkeypatch.setattr(axisymmetric_magnetostatics, "solve_geometry_field_map",
                        lambda *a: pytest.fail("Material editing must not solve a field"))
    state = default_state()
    before = state.to_dict()
    page = ModelInspectorPage()
    qtbot.addWidget(page)
    page.set_state(state)
    assert float(page.permeability.text()) == 14872
    assert page.bh_material.currentData()["key"] == "femm_pure_iron"
    assert page.ampere_turns.text() == ""
    assert state.to_dict() == before
    with qtbot.waitSignal(page.error) as error:
        page._apply_field()
    assert "ampere-turns" in error.args[0]
    assert state.to_dict() == before
    # Synthetic test input, not an assigned default or instrument calibration.
    page.ampere_turns.setText("250")
    page._apply_field()
    key = page.lens.currentData()
    recipe = state.lens_field_map_descriptors[key]
    assert recipe["linear_material_reference"] == lens_material_defaults()["linear_material_reference"]
    restored = State.from_dict(state.to_dict())
    assert restored.lens_field_map_descriptors == state.lens_field_map_descriptors
    assert [lens.percent for lens in state.lenses] == [lens.percent for lens in default_state().lenses]
    page.permeability.setText("700")
    assert "user-defined" in page.material_hint.text()
    page._apply_field()
    assert "linear_material_reference" not in state.lens_field_map_descriptors[key]
    page.set_state(state)
    assert float(page.permeability.text()) == 700
    assert float(page.ampere_turns.text()) == 250


def test_gui_does_not_copy_previous_custom_material_to_new_lens(qtbot):
    from temsim.gui.model_inspector import ModelInspectorPage
    from temsim.optics.column import default_state

    state = default_state()
    first, second = state.lenses[:2]
    custom = lens_material_defaults()["bh_material"]
    custom["label"] = "User-supplied test material"
    custom["h_a_per_m"][1] *= 1.1
    state.lens_field_map_descriptors[first.key] = dict(
        solver="axisymmetric_nonlinear_fem", ampere_turns=250, bh_material=custom,
        material_bh_overrides={"custom_iron": deepcopy(custom)})
    before = deepcopy(state.lens_field_map_descriptors)
    page = ModelInspectorPage()
    qtbot.addWidget(page)
    page.set_state(state)
    page.lens.setCurrentIndex(page.lens.findData(first.key))
    assert page.bh_material.currentData() == custom
    page.lens.setCurrentIndex(page.lens.findData(second.key))
    assert page.bh_material.currentData() == lens_material_defaults()["bh_material"]
    page.lens.setCurrentIndex(page.lens.findData(first.key))
    assert page.bh_material.currentData() == custom
    page.default_material.click()
    assert page.bh_material.currentData() == lens_material_defaults()["bh_material"]
    assert float(page.ampere_turns.text()) == 250
    assert state.lens_field_map_descriptors == before  # Draft only until applied.
    page._apply_field()
    assert state.lens_field_map_descriptors[first.key]["bh_material"] == lens_material_defaults()["bh_material"]
    assert state.lens_field_map_descriptors[first.key]["material_bh_overrides"] == before[first.key]["material_bh_overrides"]


@pytest.mark.parametrize("solver_index", [0, 1])
def test_all_lens_default_recipes_pass_readiness_without_changing_excitation(qtbot, solver_index):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.gui.model_inspector import ModelInspectorPage
    from temsim.optics.column import default_state
    from temsim.simulation_modes import linear_mode_issues, nonlinear_mode_issues

    catalog, state = AssemblyCatalog(), default_state()
    catalog.apply(state, catalog.default_selection())
    strengths = [(lens.percent, lens.polarity) for lens in state.lenses]
    assembly = state._resolved_assembly
    page = ModelInspectorPage()
    qtbot.addWidget(page)
    page.set_state(state)
    for lens in state.lenses:
        if lens.enabled:
            page.lens.setCurrentIndex(page.lens.findData(lens.key))
            page.field_solver.setCurrentIndex(solver_index)
            page.ampere_turns.setText("250")  # Synthetic, deliberately NOT a UI default.
            page._apply_field()
    issues = (linear_mode_issues, nonlinear_mode_issues)[solver_index](state)
    assert issues == ()
    assert state._resolved_assembly is assembly
    assert [(lens.percent, lens.polarity) for lens in state.lenses] == strengths
