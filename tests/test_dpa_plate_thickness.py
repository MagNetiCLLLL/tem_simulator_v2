"""Declared DPA defaults and unspecified-thickness legacy compatibility."""
from types import SimpleNamespace
import tomllib

import numpy as np
import pytest
import tomli_w

from temsim import module_manifest
from temsim.manifest_editor import ManifestEditor, ManifestTarget
from temsim.part_model_3d import part_dimension_specs, part_model_from_document
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.posed_wave_aperture import PosedWaveAperture


MODULE = "project_and_recording_system/NoEnergyFilter.toml"
KEY = "projection_chamber_dpa_aperture"
PATH = ("parts", KEY, "plate_thickness_mm")


@pytest.fixture
def legacy_module(tmp_path):
    document = tomllib.loads(PartModelDocument(INSTRUMENT_CONFIG_ROOT / MODULE)._editable_text)
    next(part for part in document["parts"] if part["key"] == KEY).pop(PATH[2])
    path = tmp_path / "legacy-dpa.toml"
    path.write_text(tomli_w.dumps(document), encoding="utf-8")
    return path


@pytest.mark.parametrize("module", [MODULE, "project_and_recording_system/EnergyFilter.toml"])
def test_both_assemblies_declare_same_point_two_mm_plate(module):
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT / module)
    assert draft.part(KEY)[PATH[2]] == .2
    dimension = next(item for item in part_dimension_specs(draft.document, KEY) if item.path == PATH)
    field = next(item for item in ManifestEditor().fields(ManifestTarget(module, KEY)) if item.path == PATH)
    assert dimension.editable and field.editable and dimension.value == field.value == .2
    assert not draft.dirty


def test_zero_unspecified_entry_exists_in_both_existing_editors(legacy_module):
    draft = PartModelDocument(legacy_module)
    assert "plate_thickness_mm" not in draft.part(KEY)
    dimension = next(item for item in part_dimension_specs(draft.document, KEY) if item.path == PATH)
    fields = ManifestEditor(root=legacy_module.parent).fields(ManifestTarget(legacy_module.name, KEY))
    field = next(item for item in fields if item.path == PATH)
    assert dimension.editable and field.editable and dimension.value == field.value == 0.
    for description in (dimension.reason, field.meaning.description):
        assert "0 means unspecified" in description
        assert "positive actual plate thickness" in description
    assert not draft.dirty


def test_gui_stages_positive_thickness_then_model_persists_and_wave_consumes_it(qtbot, tmp_path):
    from temsim.gui.parameter_panel import ParameterPanel
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT / MODULE)
    panel = ParameterPanel()
    qtbot.addWidget(panel)
    target = ManifestTarget(MODULE, KEY)
    panel.set_context(draft.part(KEY)["name"], None, target, ManifestEditor().fields(target), None)
    item = next(panel.manifest_table.item(row, 1) for row in range(panel.manifest_table.rowCount())
                if panel.manifest_table.item(row, 0).text() == PATH[2])
    assert "positive actual plate thickness" in item.toolTip()
    received = []
    panel.manifest_save_requested.connect(lambda chosen, updates: received.append((chosen, updates)))
    item.setText("0.08")
    panel._save_manifest()
    assert received == [(target, {PATH: .08})]
    draft.set_dimension(PATH, received[0][1][PATH])
    meshes = part_model_from_document(draft.document, KEY, include_children=False).meshes
    assert meshes
    vertices = np.concatenate([mesh.vertices for mesh in meshes])
    assert np.ptp(vertices[:, 2]) == pytest.approx(.08)
    # Save a detached copy: the authoritative shipped instrument is untouched.
    saved = tmp_path / "dpa.toml"
    draft.save_copy(saved)
    reopened = PartModelDocument(saved)
    assert reopened.part(KEY)[PATH[2]] == .08
    part = SimpleNamespace(key=KEY, data=reopened.part(KEY))
    state = SimpleNamespace(_resolved_assembly=SimpleNamespace(parts=[part]))
    aperture = SimpleNamespace(key=KEY, z_mm=part.data["local_center_z_mm"],
                               radius_mm=.5*part.data["mechanical_bore_diameter_mm"])
    plate = PosedWaveAperture.from_component(state, aperture, CoordinateRegistration())
    assert plate.local_end_z_mm-plate.local_start_z_mm == pytest.approx(.08)


@pytest.mark.parametrize("value", [-.1, True, float("nan")])
def test_invalid_optional_thickness_rejected_by_draft_and_manifest(value):
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT / MODULE)
    with pytest.raises(ValueError):
        draft.set_dimension(PATH, value)
    assert not draft.dirty
    draft.part(KEY)["plate_thickness_mm"] = value
    with pytest.raises(ValueError, match="plate_thickness_mm"):
        module_manifest.validate_document(draft.document)


def test_optional_field_insertion_is_limited_to_dpa(legacy_module):
    source = legacy_module.read_text(encoding="utf-8")
    staged = module_manifest.stage_manifest_text(source, {PATH: .04})
    row = next(part for part in tomllib.loads(staged)["parts"] if part["key"] == KEY)
    assert row["plate_thickness_mm"] == .04
    with pytest.raises(ValueError, match="Missing TOML field"):
        module_manifest.stage_manifest_text(source, {("parts", "camera", "plate_thickness_mm"): .04})
