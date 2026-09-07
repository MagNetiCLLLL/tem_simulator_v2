"""Part length editing preserves placement and the validated TOML authority."""

from copy import deepcopy
from pathlib import Path
import shutil
import tomllib

import pytest

from temsim import module_manifest
from temsim.assembly_catalog import AssemblyCatalog
from temsim.column.module_assembly import resolve_module_assembly
from temsim.column.state_layout import layout_configuration_from_state
from temsim.manifest_editor import (
    ManifestEditor, ManifestTarget, resized_part_axial_coordinates,
)
from temsim.optics.column import default_state
from temsim.paths import INSTRUMENT_CONFIG_ROOT


@pytest.fixture
def editing_context(tmp_path):
    root = tmp_path / "instruments"
    shutil.copytree(INSTRUMENT_CONFIG_ROOT, root)
    state = default_state()
    return root, ManifestEditor(root), state, layout_configuration_from_state(state)


@pytest.mark.parametrize("key,length,expected", [
    ("intermediate_lens_housing", 225.0, (140.0, 365.0)),
    ("intermediate_lens_excitation_coil", 168.0, (168.5, 336.5)),
])
def test_length_save_reloads_resolved_geometry_without_moving_other_parts(editing_context, key, length, expected):
    root, editor, state, configuration = editing_context
    target = ManifestTarget("project_and_recording_system/EnergyFilter.toml", key)
    path = root / target.module_path
    before = tomllib.loads(path.read_text(encoding="utf-8"))
    old_assembly = resolve_module_assembly(configuration, root=root)
    updates = {("parts", key, "length_mm"): length}
    original_updates = dict(updates)
    originals = editor.save(target, updates, configuration)
    after = tomllib.loads(path.read_text(encoding="utf-8"))
    original_part = next(part for part in before["parts"] if part["key"] == key)
    expected_part = dict(original_part, length_mm=length,
                         local_start_z_mm=expected[0], local_end_z_mm=expected[1])
    expected_document = deepcopy(before)
    next(part for part in expected_document["parts"] if part["key"] == key).update(expected_part)
    assert after == expected_document
    assert updates == original_updates
    assert tomllib.loads(originals[target.module_path]) == before

    # Exercise the same catalog reload used after the GUI saves, including
    # physical layout, instead of checking only the serialized coordinates.
    catalog = AssemblyCatalog(root)
    new_assembly = catalog.apply(state, catalog.default_selection(), preserve_operating_parameters=True)
    for part in new_assembly.parts:
        old = old_assembly.part(part.key)
        assert part.center_z_mm == old.center_z_mm
        if part.key == key:
            assert part.length_mm == length
            assert part.end_z_mm - part.start_z_mm == pytest.approx(length)
        else:
            assert (part.start_z_mm, part.end_z_mm, part.length_mm) == (
                old.start_z_mm, old.end_z_mm, old.length_mm,
            )
    assert next(field.value for field in editor.fields(target) if field.label == "length_mm") == length


@pytest.mark.parametrize("length", [-1.0, float("nan"), float("inf"), float("-inf"), True, "225"])
@pytest.mark.parametrize("explicit_endpoints", [False, True])
def test_invalid_length_preserves_original_file(editing_context, length, explicit_endpoints):
    root, editor, _state, configuration = editing_context
    target = ManifestTarget("project_and_recording_system/EnergyFilter.toml", "intermediate_lens_housing")
    path = root / target.module_path
    original = path.read_bytes()
    updates = {("parts", target.part_key, "length_mm"): length}
    if explicit_endpoints:
        # A valid UI edit followed by a quoted-number edit leaves the already
        # linked endpoints in the table. They must not bypass type validation.
        updates.update({
            ("parts", target.part_key, "local_start_z_mm"): 140.0,
            ("parts", target.part_key, "local_end_z_mm"): 365.0,
        })
    with pytest.raises(ValueError, match="finite non-negative"):
        editor.save(target, updates, configuration)
    assert path.read_bytes() == original


def test_conflicting_explicit_endpoints_are_rejected_without_rewriting(editing_context):
    root, editor, _state, configuration = editing_context
    target = ManifestTarget("project_and_recording_system/EnergyFilter.toml", "intermediate_lens_housing")
    path = root / target.module_path
    original = path.read_bytes()
    with pytest.raises(ValueError, match="length mismatch"):
        editor.save(target, {
            ("parts", target.part_key, "length_mm"): 225.0,
            ("parts", target.part_key, "local_start_z_mm"): 140.0,
            ("parts", target.part_key, "local_end_z_mm"): 367.5,
        }, configuration)
    assert path.read_bytes() == original


def test_explicit_offcenter_endpoints_are_preserved(editing_context):
    root, editor, _state, configuration = editing_context
    target = ManifestTarget("project_and_recording_system/EnergyFilter.toml", "intermediate_lens_excitation_coil")
    editor.save(target, {
        ("parts", target.part_key, "length_mm"): 168.0,
        ("parts", target.part_key, "local_start_z_mm"): 171.5,
        ("parts", target.part_key, "local_end_z_mm"): 339.5,
    }, configuration)
    values = {field.label: field.value for field in editor.fields(target)}
    assert (values["local_start_z_mm"], values["local_center_z_mm"], values["local_end_z_mm"]) == (
        171.5, 252.5, 339.5,
    )


def test_impossible_envelope_retains_collision_validation(editing_context):
    root, editor, _state, configuration = editing_context
    target = ManifestTarget("project_and_recording_system/EnergyFilter.toml", "intermediate_lens_housing")
    path = root / target.module_path
    original = path.read_bytes()
    with pytest.raises(ValueError, match="[Oo]verlap|clearance"):
        editor.save(target, {("parts", target.part_key, "length_mm"): 241.0}, configuration)
    assert path.read_bytes() == original


def test_noncentral_and_zero_length_reference_resize():
    part = {"local_start_z_mm": 10.0, "local_center_z_mm": 12.0, "local_end_z_mm": 20.0}
    assert resized_part_axial_coordinates(part, 20.0) == {
        "local_start_z_mm": 8.0, "local_end_z_mm": 28.0,
    }
    assert resized_part_axial_coordinates(part, 20.0, center_z_mm=32.0) == {
        "local_start_z_mm": 28.0, "local_end_z_mm": 48.0,
    }
    point = {"local_start_z_mm": 5.0, "local_center_z_mm": 5.0, "local_end_z_mm": 5.0}
    assert resized_part_axial_coordinates(point, 2.0) == {
        "local_start_z_mm": 4.0, "local_end_z_mm": 6.0,
    }
    assert resized_part_axial_coordinates(part, 0.0) == {
        "local_start_z_mm": 12.0, "local_end_z_mm": 12.0,
    }


def test_raw_toml_staging_remains_strict(editing_context):
    root, _editor, _state, _configuration = editing_context
    text = (root / "project_and_recording_system/EnergyFilter.toml").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="length mismatch"):
        module_manifest.stage_manifest_text(text, {("parts", "intermediate_lens_housing", "length_mm"): 225.0})
