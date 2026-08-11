from copy import deepcopy
from pathlib import Path
import shutil
import tomllib

import pytest

from temsim import module_manifest
from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.column.module_assembly import (
    STRUCTURAL_FIELD_SOURCES,
    _state_targets,
)
from temsim.manifest_editor import ManifestEditor
from temsim.optics.column import default_state


def test_scan_and_descan_are_mirrored_about_sample_in_every_column_toml():
    root = Path(__file__).parents[1] / "configs" / "instruments" / "column"
    checked = 0
    for path in sorted(root.glob("*.toml")):
        with path.open("rb") as handle:
            values = tomllib.load(handle)
        parts = {part["key"]: part for part in values.get("parts", ())}
        if not {"sample", "ac_deflector", "descan_deflector"} <= set(parts):
            continue
        sample_z = float(parts["sample"]["local_center_z_mm"])
        ac_z = float(parts["ac_deflector"]["local_center_z_mm"])
        descan_z = float(parts["descan_deflector"]["local_center_z_mm"])
        assert sample_z - ac_z == pytest.approx(descan_z - sample_z)
        assert (
            float(parts["objective_stigmator"]["local_start_z_mm"])
            - float(parts["descan_deflector"]["local_end_z_mm"])
        ) == pytest.approx(5.0)
        assert (
            float(parts["image_diffraction_deflector"]["local_start_z_mm"])
            - float(parts["objective_stigmator"]["local_end_z_mm"])
        ) == pytest.approx(5.0)
        checked += 1
    assert checked == 5


def test_catalog_reports_variant_scope_and_unique_active_authorities():
    audit = ManifestEditor().validate_catalog()

    assert audit.module_count == 10
    assert audit.part_definition_count == 466
    assert audit.logical_part_key_count == 192
    assert audit.variant_scoped_duplicate_count == 274
    assert audit.assembly_count == 30
    assert audit.resolved_part_authority_count == 3638


def test_selected_runtime_components_record_their_one_toml_authority():
    catalog = AssemblyCatalog()
    state = default_state()
    assembly = catalog.apply(state, catalog.default_selection())

    assert len(assembly.part_authorities) == len(assembly.parts)
    assert len(set(assembly.part_authorities.values())) == len(assembly.parts)
    for part in assembly.parts:
        assert assembly.part_authorities[part.key] == (
            f"{part.source_file}::parts[{part.key}]"
        )

    for component in state.electron_gun.components:
        part = assembly.part(component.key)
        assert component._manifest_definition_id == part.definition_id
    for key, (_, component) in _state_targets(state).items():
        if key not in assembly.part_authorities:
            continue
        assert component._manifest_definition_id == (
            assembly.part_authorities[key]
        )
    assert state.energy_filter._manifest_definition_id == (
        assembly.part_authorities["energy_filter"]
    )


def test_custom_catalog_root_is_the_final_geometry_and_gun_exit_authority(
    tmp_path: Path,
):
    root = tmp_path / "instruments"
    shutil.copytree(module_manifest.MODULE_ROOT, root)
    path = root / "gun" / "FEG.toml"
    text = path.read_text(encoding="utf-8")
    staged = module_manifest.stage_manifest_text(text, {
        ("ports", "exit", "local_z_mm"): 451.0,
        ("geometry", "length_mm"): 451.0,
        ("parts", "feg_electrostatic_lens", "local_start_z_mm"): 21.0,
        ("parts", "feg_electrostatic_lens", "local_center_z_mm"): 25.0,
        ("parts", "feg_electrostatic_lens", "local_end_z_mm"): 29.0,
        (
            "parts",
            "feg_electrostatic_lens",
            "optical_reference_local_z_mm",
        ): 25.0,
    })
    path.write_text(staged, encoding="utf-8")

    recording_path = (
        root
        / "project_and_recording_system"
        / "EnergyFilter.toml"
    )
    recording_text = recording_path.read_text(encoding="utf-8")
    recording_staged = module_manifest.stage_manifest_text(
        recording_text,
        {
            (
                "parts",
                "selected_area_aperture",
                "mechanical_outer_diameter_mm",
            ): 82.0,
            (
                "parts",
                "energy_filter_slit",
                "clear_height_mm",
            ): 13.25,
        },
    )
    recording_path.write_text(recording_staged, encoding="utf-8")

    catalog = AssemblyCatalog(root)
    state = default_state()
    assembly = catalog.apply(state, catalog.default_selection())
    lens = state.electron_gun.electrostatic_lens

    assert assembly.root == root.resolve()
    assert assembly.selected_path("gun") == "gun/FEG.toml"
    assert assembly.part("feg_electrostatic_lens").center_z_mm == 25.0
    assert lens.mechanical_center_from_tip_mm == 25.0
    assert lens.optical_reference_from_tip_mm == 25.0
    assert lens._manifest_source_file == "gun/FEG.toml"
    assert state.electron_gun.exit_plane_z_mm == 451.0
    assert state.selected_area_aperture.mechanical_outer_diameter_mm == 82.0
    assert state.energy_filter.energy_slit.clear_height_m == pytest.approx(
        13.25e-3
    )


def test_module_manifest_rejects_duplicate_part_key_and_order():
    path = module_manifest.MODULE_ROOT / "gun" / "FEG.toml"
    document = module_manifest.read_document(path)

    duplicate_key = deepcopy(document)
    duplicate_key["parts"].append(deepcopy(duplicate_key["parts"][0]))
    duplicate_key["parts"][-1]["order"] = 99
    with pytest.raises(ValueError, match="Duplicate part key"):
        module_manifest.validate_document(duplicate_key)

    duplicate_order = deepcopy(document)
    duplicate_order["parts"][1]["order"] = duplicate_order["parts"][0][
        "order"
    ]
    with pytest.raises(ValueError, match="Duplicate part order"):
        module_manifest.validate_document(duplicate_order)


def test_catalog_rejects_a_module_file_listed_twice(tmp_path: Path):
    root = tmp_path / "instruments"
    shutil.copytree(module_manifest.MODULE_ROOT, root)
    path = root / "catalog.toml"
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    duplicate = deepcopy(document["gun_variants"][0])
    duplicate["name"] = "Duplicate FEG"
    duplicate["electron_gun"] = "Duplicate"
    document["gun_variants"].append(duplicate)

    # A small textual insertion preserves the project's editable TOML style.
    text = path.read_text(encoding="utf-8")
    marker = "[[column_variants]]"
    duplicate_text = (
        "[[gun_variants]]\n"
        'name = "Duplicate FEG"\n'
        'file = "gun/FEG.toml"\n'
        'electron_gun = "Duplicate"\n'
        "monochromator = false\n\n"
    )
    path.write_text(
        text.replace(marker, duplicate_text + marker, 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate file"):
        AssemblyCatalog(root)


def test_duplicate_runtime_keys_cannot_silently_override_each_other():
    state = default_state()
    state.lenses.append(deepcopy(state.lenses[0]))

    with pytest.raises(ValueError, match="Duplicate runtime component key"):
        _state_targets(state)


def test_missing_toml_structure_cannot_fall_back_to_python(
    tmp_path: Path,
):
    root = tmp_path / "instruments"
    shutil.copytree(module_manifest.MODULE_ROOT, root)
    path = root / "project_and_recording_system" / "EnergyFilter.toml"
    text = path.read_text(encoding="utf-8")
    field = "mechanical_outer_diameter_mm = 80.0\n"
    assert text.count(field) >= 1
    path.write_text(text.replace(field, "", 1), encoding="utf-8")

    with pytest.raises(ValueError, match="mechanical_outer_diameter_mm"):
        catalog = AssemblyCatalog(root)
        catalog.apply(default_state(), catalog.default_selection())


def test_all_catalog_assemblies_apply_one_authority_per_runtime_part():
    catalog = AssemblyCatalog()
    assembly_count = 0
    for gun in catalog.guns:
        for column in catalog.columns:
            for recording in catalog.recording_systems:
                selection = AssemblySelection(
                    gun=gun.name,
                    column=column.name,
                    recording=recording.name,
                )
                state = default_state()
                assembly = catalog.apply(state, selection)
                authorities = assembly.part_authorities

                assert len(authorities) == len(assembly.parts)
                assert len(set(authorities.values())) == len(assembly.parts)
                for component in state.electron_gun.components:
                    assert component._manifest_definition_id == (
                        authorities[component.key]
                    )
                for key, (_, component) in _state_targets(state).items():
                    if key in authorities:
                        assert component._manifest_definition_id == (
                            authorities[key]
                        )
                        if hasattr(component, "name"):
                            assert component.name == assembly.part(key).name
                assembly_count += 1

    assert assembly_count == 30


def test_saved_state_omits_every_manifest_owned_structural_attribute():
    state = default_state()
    payload = state.to_dict()

    for collection in (
        "lenses",
        "apertures",
        "stigmators",
        "deflectors",
        "corrector_elements",
        "recording_planes",
    ):
        for component in payload[collection]:
            for attribute in STRUCTURAL_FIELD_SOURCES:
                assert attribute not in component
