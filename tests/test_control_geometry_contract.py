"""WP-05: sourced current, explicit geometry scope and compatible state."""

from dataclasses import replace
import json
from types import SimpleNamespace
import numpy as np
import pytest

from temsim.excitation_calibration import ExcitationCalibration, calibration_from_recipe, recipe_with_calibration
from temsim.geometry_effects import field_geometry_admission, geometry_effects


@pytest.mark.parametrize("percent, polarity", [(0, 1), (25, 1), (80, -1), (100, 1)])
def test_known_turns_current_ni_control_roundtrip(percent, polarity):
    cal = ExcitationCalibration(400., 80., 200, "measured", "Synthetic electrical acceptance fixture", "2026-09-10", .01, (0, 3)).validate()
    point = cal.at_control(percent, polarity)
    assert point["ampere_turns"] == pytest.approx(percent*5*polarity)
    assert point["current_A"] == pytest.approx(percent*.025*polarity)
    assert cal.control_for_current(point["current_A"], zero_polarity=polarity) == pytest.approx((percent, polarity))
    assert calibration_from_recipe(recipe_with_calibration({"solver": "axisymmetric_linear_fem"}, cal)) == cal


def test_unknown_turns_and_conflicting_reference_do_not_fabricate_current():
    cal = calibration_from_recipe({"ampere_turns": 250})
    assert cal.at_control(80)["current_A"] is None
    with pytest.raises(ValueError, match="unavailable"):
        cal.control_for_current(1)
    with pytest.raises(ValueError, match="disagrees"):
        calibration_from_recipe({"ampere_turns": 400, "excitation_calibration": cal.to_dict()})
    with pytest.raises(ValueError, match="source"):
        ExcitationCalibration(250, turns=200, source="").validate()
    with pytest.raises(ValueError, match="range"):
        ExcitationCalibration(250, turns=200, valid_current_range_A=(0, .5)).control_for_current(1)


def _binding():
    return SimpleNamespace(canonical_geometry_json=json.dumps({"lens_assembly": {"parts": [{"key": "coil"}], "magnetostatic_neighbours": []}}))


def test_geometry_requires_explicit_approximation_and_legacy_is_named():
    state = SimpleNamespace(_resolved_assembly=SimpleNamespace(parts=[SimpleNamespace(key="coil", data={"model_3d": {"features": [{"kind": "hole"}]}})]))
    with pytest.raises(ValueError, match="cannot consume"):
        field_geometry_admission(state, _binding(), {"geometry_policy": "require_full_geometry"})
    report = field_geometry_admission(state, _binding(), {"geometry_policy": "authoritative_dimensions"})
    assert report["ignored_model_3d_parts"] == ["coil"] and report["status"] == "explicit_approximation"
    assert field_geometry_admission(state, _binding(), {})["legacy_policy"]


def test_geometry_effects_separate_cad_material_vacuum_and_optical_dimensions():
    row = {"key": "coil", "parent_key": "lens", "mechanical_profile": "magnetic_excitation_coil",
           "mechanical_outer_diameter_mm": 40, "mechanical_inner_diameter_mm": 20,
           "length_mm": 10, "local_start_z_mm": 0, "local_end_z_mm": 10, "vacuum_inner_diameter_mm": 5,
           "model_3d": {"features": [{"kind": "hole", "diameter_mm": 1}]}}
    report = geometry_effects(row, by_key={"lens": {"key": "lens", "mechanical_profile": "magnetic_lens_assembly"}}, simulation_mode="custom",
                              descriptors={"lens": {"solver": "axisymmetric_linear_fem", "ampere_turns": 250, "relative_permeability": 1000}})
    fields = {item["path"][2]: item for item in report["parameters"]}
    assert fields["model_3d"]["display"] and not fields["model_3d"]["scientific_input"]
    assert fields["mechanical_outer_diameter_mm"]["field"]
    assert fields["vacuum_inner_diameter_mm"]["vacuum_or_stop"]


def test_geometry_admission_precedes_provider_cache_and_solver(monkeypatch):
    from temsim.optics.column import default_state
    from temsim.physics import lens_field_provider
    import temsim.geometry_effects as module
    state = default_state()
    lens = state.lenses[0]
    state.lens_field_map_descriptors[lens.key] = {"solver": "axisymmetric_linear_fem", "ampere_turns": 250, "relative_permeability": 1000}
    def rejected(*args):
        raise ValueError("Unrepresented geometry admission guard")
    monkeypatch.setattr(module, "field_geometry_admission", rejected)
    state._runtime_lens_field_provider_cache = {lens.key: (None, object())}
    with pytest.raises(ValueError, match="admission guard"):
        lens_field_provider.resolve_runtime_lens_field_provider(state, lens.key, lens)


def test_calibration_state_profile_roundtrip_preserves_operating_point(tmp_path):
    from temsim.optics.column import default_state
    from temsim.optics.model import State
    from temsim.profile_io import save_profile, read_profile, apply_profile_values
    from temsim.assembly_catalog import AssemblyCatalog
    state = default_state()
    lens = state.lenses[0]
    calibration = ExcitationCalibration(250., turns=200, source="Explicit test fixture")
    state.lens_field_map_descriptors[lens.key] = recipe_with_calibration(
        {"solver": "axisymmetric_linear_fem", "relative_permeability": 1000}, calibration)
    controls = [(l.percent, l.polarity) for l in state.lenses]
    restored = State.from_dict(state.to_dict())
    assert [(l.percent, l.polarity) for l in restored.lenses] == controls
    path = tmp_path / "current.toml"
    save_profile(path, state, AssemblyCatalog().default_selection())
    _, values = read_profile(path)
    assert apply_profile_values(restored, values) == []
    assert calibration_from_recipe(restored.lens_field_map_descriptors[lens.key]).turns == 200


def test_calibration_dialog_changes_current_only_when_requested(qtbot):
    from temsim.gui.excitation_dialog import ExcitationDialog
    lens = SimpleNamespace(percent=50, polarity=1, max_percent=100, enabled=True)
    dialog = ExcitationDialog(ExcitationCalibration(250), lens)
    qtbot.addWidget(dialog)
    dialog.apply_current.setChecked(True)
    dialog.current.setText("-1")
    dialog.accept()
    assert dialog.control is None and "unavailable" in dialog.error.text()
    dialog.turns.setText("200")
    dialog.source.setText("Known test winding")
    dialog.accept()
    assert dialog.control == (80., -1)
    assert lens.percent == 50 and lens.polarity == 1
