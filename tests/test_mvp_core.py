from pathlib import Path
import tomllib

import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.manifest_editor import ManifestEditor
from temsim.module_manifest import validate_document
from temsim.operating_modes import load_operating_mode_catalog
from temsim.optics.column import default_state
from temsim.physics.simulation import run
from temsim.profile_io import apply_profile_values, read_profile, save_profile
from temsim import presets


PROJECTOR_RECONSTRUCTION = {
    "diffraction_lens": {
        "center": 82.5,
        "length": 100.0,
        "envelope_length": 100.0,
        "housing_od": 180.0,
        "yoke_od": 170.0,
        "yoke_id": 99.5,
        "coil_id": 75.0,
        "coil_od": 91.5,
        "coil_length": 90.0,
        "pole_bore": 21.5,
        "clear_bore": 20.0,
        "pole_gap": 4.0,
        "pole_shoulder_od": 52.0,
        "pole_nose": 12.0,
    },
    "intermediate_lens": {
        "center": 252.5,
        "length": 230.0,
        "envelope_length": 230.0,
        "housing_od": 180.0,
        "yoke_od": 170.0,
        "yoke_id": 99.98322635,
        "coil_id": 75.0,
        "coil_od": 91.98322635,
        "coil_length": 162.0,
        "pole_bore": 21.5,
        "clear_bore": 20.0,
        "pole_gap": 6.0,
        "pole_shoulder_od": 52.0,
        "pole_nose": 12.0,
    },
    "projector_lens_1": {
        "center": 432.5,
        "length": 120.0,
        "envelope_length": 120.0,
        "housing_od": 180.0,
        "yoke_od": 170.0,
        "yoke_id": 97.52,
        "coil_id": 75.0,
        "coil_od": 89.52,
        "coil_length": 108.0,
        "pole_bore": 21.5,
        "clear_bore": 20.0,
        "pole_gap": 5.0,
        "pole_shoulder_od": 52.0,
        "pole_nose": 13.0,
    },
    "projector_lens_2": {
        "center": 635.0,
        "length": 275.0,
        "envelope_length": 275.0,
        "housing_od": 180.0,
        "yoke_od": 170.0,
        "yoke_id": 97.16,
        "coil_id": 75.0,
        "coil_od": 89.16,
        "coil_length": 162.0,
        "pole_bore": 21.5,
        "clear_bore": 20.0,
        "pole_gap": 7.0,
        "pole_shoulder_od": 52.0,
        "pole_nose": 14.0,
    },
}


def test_every_catalog_option_resolves_to_the_requested_toml():
    catalog = AssemblyCatalog()
    state = default_state()
    for column in catalog.columns:
        selection = catalog.default_selection().__class__(
            gun="FEG",
            column=column.name,
            recording="Energy Filter",
        )
        assembly = catalog.apply(state, selection)
        assert dict(assembly.selected_module_paths)["column"] == column.file


def test_every_lens_and_preset_uses_at_most_one_hundred_percent():
    catalog = AssemblyCatalog()
    for gun in catalog.guns:
        for column in catalog.columns:
            for recording in catalog.recording_systems:
                state = default_state()
                catalog.apply(
                    state,
                    catalog.default_selection().__class__(
                        gun.name, column.name, recording.name
                    ),
                )
                for lens in state.lenses:
                    assert lens.max_percent == pytest.approx(100.0)
                    assert 0.0 <= lens.percent <= 100.0

    state = default_state()
    catalog.apply(state, catalog.default_selection())
    for preset_name in presets.P:
        candidate = type(state).from_dict(state.to_dict())
        presets.apply(candidate, preset_name)
        assert all(
            0.0 <= lens.percent <= 100.0
            for lens in candidate.lenses
        )

    operating_catalog = load_operating_mode_catalog()
    assert all(
        float(values.get("percent", 0.0)) < 100.0
        for mode in operating_catalog.modes
        for values in mode.devices.values()
    )


def test_rebased_lens_percentages_preserve_reference_fields():
    catalog = AssemblyCatalog()
    state = default_state()
    catalog.apply(state, catalog.default_selection())
    by_key = {lens.key: lens for lens in state.lenses}

    assert by_key["probe_tl12_lens"].scale() == pytest.approx(0.33)
    objective = by_key["objective_lens"]
    assert (
        objective.upper_b0_t * objective.percent / 100.0
    ) == pytest.approx(0.26910326119741307)

    catalog.apply(
        state,
        catalog.default_selection().__class__(
            "FEG", "C3 + Probe Corrector + Image Corrector", "Energy Filter"
        ),
    )
    by_key = {lens.key: lens for lens in state.lenses}
    expected_scales = {
        "image_ol_post_lens": 1.82184515,
        "image_tl22_lens": 1.27567878,
        "image_adapter_lens": 0.25151499,
    }
    for key, expected in expected_scales.items():
        assert by_key[key].scale() == pytest.approx(expected)
        assert by_key[key].percent == pytest.approx(60.0)


def test_every_active_part_has_a_confirmed_assembly_anchor():
    catalog = AssemblyCatalog()
    state = default_state()
    assembly = catalog.apply(state, catalog.default_selection())
    records = ManifestEditor.anchor_records(assembly)

    assert len(records) == len(assembly.parts)
    assert all(record.anchor for record in records)
    assert {record.part_key for record in records} == {
        part.key for part in assembly.parts
    }
    by_key = {record.part_key: record for record in records}
    assert by_key["condenser_lens_2"].anchor == "condenser_lens_1"
    assert by_key["objective_upper_pole"].anchor == "objective_lens"


def test_complete_catalog_and_every_assembly_combination_validate():
    audit = ManifestEditor().validate_catalog()

    assert audit.module_count == 11
    assert audit.part_definition_count == 482
    assert audit.assembly_count == 30


def test_all_apertures_declare_photo_informed_pt_strip_and_single_rod():
    root = Path(__file__).parents[1] / "configs" / "instruments"
    aperture_keys = {
        "feg_dpa_aperture",
        "feg_c1_aperture",
        "thermionic_anode_aperture",
        "thermionic_c1_aperture",
        "condenser_aperture_2",
        "condenser_aperture_3",
        "objective_aperture",
        "selected_area_aperture",
        "energy_filter_entrance_aperture",
    }
    seen = set()
    for path in root.rglob("*.toml"):
        document = tomllib.loads(path.read_text(encoding="utf-8"))
        if "module" not in document:
            continue
        validate_document(document)
        for part in document["parts"]:
            if part["key"] not in aperture_keys:
                continue
            seen.add(part["key"])
            assert part["aperture_plate_material"] == (
                "platinum_user_identified_unverified"
            )
            assert part["aperture_plate_form"] == "perforated_strip"
            assert part["aperture_plate_attachment"] == (
                "screw_to_single_connection_rod"
            )
            assert part["aperture_mechanism_evidence_status"] == (
                "user_identified_photo_topology_not_dimensionally_calibrated"
            )
            assert "without OEM dimensions" in (
                part["aperture_mechanism_evidence_source"]
            )
    assert seen == aperture_keys

    document = tomllib.loads(
        (root / "column" / "C3.toml").read_text(encoding="utf-8")
    )
    by_key = {part["key"]: part for part in document["parts"]}
    del by_key["objective_aperture"]["aperture_plate_attachment"]
    with pytest.raises(
        ValueError,
        match="Missing objective_aperture aperture mechanism metadata",
    ):
        validate_document(document)


@pytest.mark.parametrize(
    ("gun_file", "accelerator_key"),
    (
        ("FEG.toml", "feg_accelerator"),
        ("FEG_Mono.toml", "feg_accelerator"),
        ("Thermionic.toml", "thermionic_accelerator"),
    ),
)
def test_accelerator_ring_stack_topology_has_photo_provenance(
    gun_file, accelerator_key
):
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "instruments"
        / "gun"
        / gun_file
    )
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    validate_document(document)
    part = next(
        item for item in document["parts"]
        if item["key"] == accelerator_key
    )

    assert part["accelerator_electrode_stack_form"] == (
        "repeated_annular_electrode_stages"
    )
    assert part["accelerator_electrode_stack_evidence_status"] == (
        "user_supplied_side_view_topology_not_dimensionally_calibrated"
    )
    assert "without an OEM scale" in (
        part["accelerator_electrode_stack_evidence_source"]
    )
    centers = tuple(float(value) for value in part["stage_centers_z_mm"])
    assert len(centers) == 10
    assert all(
        downstream > upstream
        for upstream, downstream in zip(centers, centers[1:])
    )
    assert all(
        float(part["local_start_z_mm"])
        <= center
        <= float(part["local_end_z_mm"])
        for center in centers
    )

    del part["accelerator_electrode_stack_form"]
    with pytest.raises(
        ValueError,
        match=f"Missing {accelerator_key} accelerator-stack metadata",
    ):
        validate_document(document)


def test_energy_filter_is_the_only_selectable_recording_system():
    catalog = AssemblyCatalog()

    assert [option.name for option in catalog.recording_systems] == [
        "Energy Filter"
    ]

    legacy_selection = AssemblySelection(
        "FEG", "C3", "No Energy Filter"
    )
    state = default_state()
    assembly = catalog.apply(state, legacy_selection)

    assert dict(assembly.selected_module_paths)[
        "project_and_recording_system"
    ].endswith("EnergyFilter.toml")
    assert state.energy_filter_installed is True
    assert state.energy_filter_mode == "energy_filter"
    assert state.energy_filter.enabled is True


def test_magnetic_lens_mechanical_layers_are_required_and_radially_nested():
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "instruments"
        / "column"
        / "C3.toml"
    )
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    document["parts"] = [
        part for part in document["parts"]
        if part["key"] != "condenser_lens_3_yoke"
    ]
    with pytest.raises(ValueError, match="Missing magnetic-lens mechanical parts"):
        validate_document(document)

    document = tomllib.loads(path.read_text(encoding="utf-8"))
    by_key = {part["key"]: part for part in document["parts"]}
    by_key["condenser_lens_3_yoke"]["mechanical_outer_diameter_mm"] = (
        by_key["condenser_lens_3_housing"][
            "mechanical_inner_diameter_mm"
        ] + 1.0
    )
    with pytest.raises(ValueError, match="radial layers overlap"):
        validate_document(document)

    document = tomllib.loads(path.read_text(encoding="utf-8"))
    by_key = {part["key"]: part for part in document["parts"]}
    by_key["mini_condenser"]["mechanical_outer_diameter_mm"] = 101.0
    by_key["mini_condenser_housing"][
        "mechanical_outer_diameter_mm"
    ] = 101.0
    with pytest.raises(
        ValueError,
        match="housing must fit radially inside.*excitation-coil bore",
    ):
        validate_document(document)

    document = tomllib.loads(path.read_text(encoding="utf-8"))
    by_key = {part["key"]: part for part in document["parts"]}
    by_key["objective_upper_pole"]["pole_stem_outer_diameter_mm"] = 59.0
    by_key["objective_lower_pole"]["pole_stem_outer_diameter_mm"] = 59.0
    with pytest.raises(
        ValueError,
        match="mounting-shank OD must equal the excitation-coil ID",
    ):
        validate_document(document)

    document = tomllib.loads(path.read_text(encoding="utf-8"))
    by_key = {part["key"]: part for part in document["parts"]}
    by_key["condenser_lens_3_housing"][
        "mechanical_outer_diameter_mm"
    ] -= 1.0
    with pytest.raises(
        ValueError,
        match="outer diameter must equal its parent lens envelope",
    ):
        validate_document(document)

    document = tomllib.loads(path.read_text(encoding="utf-8"))
    by_key = {part["key"]: part for part in document["parts"]}
    by_key["condenser_lens_3_excitation_coil"][
        "mechanical_outer_diameter_mm"
    ] += 1.0
    with pytest.raises(
        ValueError,
        match="radial thickness must follow.*per-lens reconstruction",
    ):
        validate_document(document)

    document = tomllib.loads(path.read_text(encoding="utf-8"))
    by_key = {part["key"]: part for part in document["parts"]}
    by_key["condenser_lens_3_excitation_coil"][
        "mechanical_inner_diameter_mm"
    ] = 90.0
    by_key["condenser_lens_3_excitation_coil"][
        "mechanical_outer_diameter_mm"
    ] = 150.0
    with pytest.raises(
        ValueError,
        match="Excitation-coil material overlap.*condenser_lens_3_upper_pole",
    ):
        validate_document(document)

    document = tomllib.loads(path.read_text(encoding="utf-8"))
    by_key = {part["key"]: part for part in document["parts"]}
    by_key["condenser_lens_1_lower_pole"][
        "mechanical_outer_diameter_mm"
    ] -= 1.0
    with pytest.raises(
        ValueError,
        match="C1/C2 interface poles must match",
    ):
        validate_document(document)


def test_c1_c2_objective_vacuum_tube_ratio_and_interface_are_required():
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "instruments"
        / "column"
        / "C3.toml"
    )
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    document["geometry"][
        "c1_c2_objective_vacuum_tube_inner_diameter_mm"
    ] = 6.0
    with pytest.raises(ValueError, match="ID must be 30% of its OD"):
        validate_document(document)

    document = tomllib.loads(path.read_text(encoding="utf-8"))
    by_key = {part["key"]: part for part in document["parts"]}
    by_key["objective_upper_pole"][
        "pole_vacuum_connector_outer_diameter_mm"
    ] = 18.0
    with pytest.raises(
        ValueError,
        match="Objective pole vacuum connectors must match",
    ):
        validate_document(document)

    document = tomllib.loads(path.read_text(encoding="utf-8"))
    document["geometry"][
        "c2_aperture_service_clearance_after_cartridge_mm"
    ] = 4.0
    with pytest.raises(
        ValueError,
        match="C2 aperture must retain its declared service clearance",
    ):
        validate_document(document)

    document = tomllib.loads(path.read_text(encoding="utf-8"))
    by_key = {part["key"]: part for part in document["parts"]}
    by_key["condenser_aperture_2"]["parent_key"] = "condenser_lens_2"
    with pytest.raises(
        ValueError,
        match="C2 aperture must be a standalone mechanism",
    ):
        validate_document(document)


@pytest.mark.parametrize(
    "column_file",
    (
        "C2.toml",
        "C3.toml",
        "C3_ImageCorrector.toml",
        "C3_ProbeCorrector.toml",
        "C3_ProbeCorrector_ImageCorrector.toml",
    ),
)
def test_objective_nested_accessories_clear_the_split_coil_bore(column_file):
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "instruments"
        / "column"
        / column_file
    )
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    by_key = {part["key"]: part for part in document["parts"]}
    coil_inner = by_key["objective_lens_excitation_coil"][
        "mechanical_inner_diameter_mm"
    ]
    expected_outer = {
        "condenser_stigmator": 56.0,
        "ac_deflector": 54.0,
        "descan_deflector": 54.0,
        "objective_stigmator": 56.0,
        "image_diffraction_deflector": 54.0,
    }
    for key, outer in expected_outer.items():
        assert by_key[key]["mechanical_outer_diameter_mm"] == pytest.approx(
            outer
        )
        assert 0.5 * (coil_inner - outer) >= 2.0 - 1.0e-9


def test_objective_split_coil_rejects_nested_accessory_material_overlap():
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "instruments"
        / "column"
        / "C3.toml"
    )
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    by_key = {part["key"]: part for part in document["parts"]}
    by_key["ac_deflector"]["mechanical_outer_diameter_mm"] = 61.0
    with pytest.raises(
        ValueError,
        match="Excitation-coil material overlap.*ac_deflector",
    ):
        validate_document(document)


@pytest.mark.parametrize(
    "column",
    (
        "C2",
        "C3",
        "C3 + Probe Corrector",
        "C3 + Image Corrector",
        "C3 + Probe Corrector + Image Corrector",
    ),
)
def test_objective_aperture_stop_is_co_located_in_the_pole_gap(column):
    state = default_state()
    assembly = AssemblyCatalog().apply(
        state,
        AssemblySelection("FEG", column, "Energy Filter"),
    )
    part = assembly.part("objective_aperture")

    assert state.objective_aperture.z_mm == pytest.approx(part.center_z_mm)
    assert state.objective_aperture.z_mm == pytest.approx(
        state.sample.z_mm
        + state.objective_aperture.mechanical_center_below_sample_mm
    )
    state.objective_aperture.validate_co_located_with_mechanics(
        state.sample.z_mm
    )
    state.objective_aperture.validate_between_poles(state.objective_lens)


@pytest.mark.parametrize(
    "column", ("C2", "C3", "C3 + Probe Corrector")
)
def test_standalone_selected_area_aperture_is_between_deflector_and_stigmator(
    column,
):
    state = default_state()
    assembly = AssemblyCatalog().apply(
        state,
        AssemblySelection("FEG", column, "Energy Filter"),
    )
    image_deflector = assembly.part("image_diffraction_deflector")
    aperture = assembly.part("selected_area_aperture")
    diffraction_stigmator = assembly.part("diffraction_stigmator")

    assert image_deflector.end_z_mm <= aperture.start_z_mm
    assert aperture.end_z_mm <= diffraction_stigmator.start_z_mm


@pytest.mark.parametrize(
    "column",
    (
        "C2",
        "C3",
        "C3 + Probe Corrector",
        "C3 + Image Corrector",
        "C3 + Probe Corrector + Image Corrector",
    ),
)
def test_c1_c2_use_contiguous_sections_of_one_shared_housing(column):
    state = default_state()
    assembly = AssemblyCatalog().apply(
        state,
        AssemblySelection("FEG", column, "Energy Filter"),
    )
    c1 = assembly.part("condenser_lens_1_housing")
    c2 = assembly.part("condenser_lens_2_housing")
    c1_lens = assembly.part("condenser_lens_1")
    c2_lens = assembly.part("condenser_lens_2")
    c1_pole = assembly.part("condenser_lens_1_lower_pole")
    c2_pole = assembly.part("condenser_lens_2_upper_pole")
    cartridge = assembly.part("c1_c2_pole_piece_cartridge")
    c2_aperture = assembly.part("condenser_aperture_2")
    downstream = assembly.part(
        "beam_deflector" if column == "C2" else "condenser_deflector"
    )
    c1_coil = assembly.part("condenser_lens_1_excitation_coil")
    c2_coil = assembly.part("condenser_lens_2_excitation_coil")
    objective_upper_pole = assembly.part("objective_upper_pole")
    continuous_tube = next(
        segment
        for segment in assembly.vacuum_liner_segments
        if segment.key == "@vacuum_liner:c1_c2_to_upper_objective"
    )

    assert c1.data["shared_housing_key"] == (
        "condenser_c1_c2_shared_housing"
    )
    assert c2.data["shared_housing_key"] == c1.data["shared_housing_key"]
    assert c1.end_z_mm == pytest.approx(c2.start_z_mm)
    assert c1_lens.length_mm == pytest.approx(100.0)
    assert c2_lens.length_mm == pytest.approx(200.0)
    assert c2_lens.length_mm == pytest.approx(2.0 * c1_lens.length_mm)
    assert c1_lens.length_mm + c2_lens.length_mm == pytest.approx(300.0)
    assert c1.length_mm == pytest.approx(c1_lens.length_mm)
    assert c2.length_mm == pytest.approx(c2_lens.length_mm)
    assert c1_coil.length_mm == pytest.approx(90.0)
    assert c2_coil.length_mm == pytest.approx(162.0)
    assert cartridge.start_z_mm == pytest.approx(c1.start_z_mm)
    assert cartridge.end_z_mm == pytest.approx(c2.end_z_mm)
    assert cartridge.data["mechanical_inner_diameter_mm"] == pytest.approx(
        60.0
    )
    assert cartridge.data["mechanical_outer_diameter_mm"] == pytest.approx(
        90.75
    )
    assert cartridge.data["mechanical_only"] is True
    assert c2_aperture.start_z_mm - cartridge.end_z_mm == pytest.approx(5.0)
    assert c2_aperture.length_mm == pytest.approx(20.0)
    assert c2_aperture.center_z_mm == pytest.approx(
        cartridge.end_z_mm + 15.0
    )
    assert downstream.start_z_mm - c2_aperture.end_z_mm == pytest.approx(
        0.0 if column == "C2" else 15.0
    )
    assert c2_aperture.parent_key is None
    assert "mechanical_overlap_group" not in c2_aperture.data
    if column == "C2":
        objective = assembly.part("objective_lens")
        assert downstream.end_z_mm == pytest.approx(objective.start_z_mm)
        assert state.beam_deflector.z_mm == pytest.approx(
            downstream.center_z_mm
        )
    assert continuous_tube.start_z_mm == pytest.approx(cartridge.start_z_mm)
    assert continuous_tube.end_z_mm == pytest.approx(
        objective_upper_pole.start_z_mm
    )
    assert continuous_tube.outer_diameter_mm == pytest.approx(19.2)
    assert continuous_tube.inner_diameter_mm == pytest.approx(5.76)
    assert (
        continuous_tube.inner_diameter_mm
        / continuous_tube.outer_diameter_mm
    ) == pytest.approx(0.30)
    assert objective_upper_pole.data[
        "pole_vacuum_connector_outer_diameter_mm"
    ] == pytest.approx(continuous_tube.outer_diameter_mm)
    for field in (
        "mechanical_bore_diameter_mm",
        "mechanical_tip_diameter_mm",
        "mechanical_outer_diameter_mm",
    ):
        assert c1_pole.data[field] == pytest.approx(c2_pole.data[field])
    assert c1_pole.data["mechanical_container_key"] == cartridge.key
    assert c2_pole.data["mechanical_container_key"] == cartridge.key
    pole_gap_midpoint = 0.5 * (c1_pole.end_z_mm + c2_pole.start_z_mm)
    assert c2_pole.start_z_mm - c1_pole.end_z_mm == pytest.approx(20.0)
    assert pole_gap_midpoint == pytest.approx(c1_lens.end_z_mm)
    for field in (
        "mechanical_inner_diameter_mm",
        "mechanical_outer_diameter_mm",
    ):
        assert c1_coil.data[field] == pytest.approx(c2_coil.data[field])


def test_monochromator_slit_has_a_separate_colocated_mechanical_envelope():
    state = default_state()
    assembly = AssemblyCatalog().apply(
        state,
        AssemblySelection("FEG + Mono", "C3", "Energy Filter"),
    )
    c1 = assembly.part("feg_c1_aperture")
    slit = assembly.part("feg_monochromator_slit")

    assert slit.center_z_mm == pytest.approx(c1.center_z_mm)
    assert slit.parent_key == c1.key
    assert slit.data["mechanical_only"] is True


def test_recording_surfaces_are_thin_interaction_planes():
    state = default_state()
    assembly = AssemblyCatalog().apply(
        state,
        AssemblySelection("FEG", "C3", "Energy Filter"),
    )
    for key in ("haadf", "flu_screen", "df", "bf", "camera"):
        part = assembly.part(key)
        assert part.length_mm == pytest.approx(0.5)
        assert part.data["mechanical_part_role"] == "interaction_plane"


def test_energy_filter_uses_a_colocated_curvilinear_branch_interface():
    state = default_state()
    assembly = AssemblyCatalog().apply(
        state,
        AssemblySelection("FEG", "C3", "Energy Filter"),
    )
    interface = assembly.part("energy_filter")
    aperture = assembly.part("energy_filter_entrance_aperture")

    assert interface.length_mm == pytest.approx(0.0)
    assert interface.center_z_mm == pytest.approx(aperture.center_z_mm)
    assert interface.data["path_coordinate"] == "curvilinear_s_mm"


def test_projector_lenses_form_a_compact_uniform_gap_stack():
    catalog = AssemblyCatalog()
    state = default_state()
    assembly = catalog.apply(
        state,
        AssemblySelection("FEG", "C3 + Probe Corrector", "Energy Filter"),
    )
    keys = (
        "diffraction_lens",
        "intermediate_lens",
        "projector_lens_1",
        "projector_lens_2",
    )
    parts = [assembly.part(key) for key in keys]
    envelopes = [assembly.part(f"{key}_housing") for key in keys]
    clearances = [
        downstream.start_z_mm - upstream.end_z_mm
        for upstream, downstream in zip(envelopes, envelopes[1:])
    ]

    assert clearances == pytest.approx((5.0, 5.0, 5.0))
    lens_by_key = {lens.key: lens for lens in state.lenses}
    for part in parts:
        assert lens_by_key[part.key].z_mm == pytest.approx(part.center_z_mm)
        upper = assembly.part(f"{part.key}_upper_pole")
        lower = assembly.part(f"{part.key}_lower_pole")
        housing = assembly.part(f"{part.key}_housing")
        assert upper.parent_key == part.key
        assert lower.parent_key == part.key
        assert housing.start_z_mm <= part.start_z_mm
        assert part.end_z_mm <= housing.end_z_mm
        assert lower.start_z_mm - upper.end_z_mm == pytest.approx(
            PROJECTOR_RECONSTRUCTION[part.key]["pole_gap"]
        )


@pytest.mark.parametrize("manifest_name", (
    "NoEnergyFilter.toml",
    "EnergyFilter.toml",
))
def test_projector_manifests_use_compact_user_defined_envelopes(
    manifest_name,
):
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "instruments"
        / "project_and_recording_system"
        / manifest_name
    )
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    validate_document(document)
    assert document["geometry"]["vacuum_liner_wall_thickness_mm"] == (
        pytest.approx(0.75)
    )
    assert document["geometry"]["projector_stack_inter_lens_gap_mm"] == (
        pytest.approx(5.0)
    )
    assert document["geometry"][
        "projector_stack_vacuum_inner_diameter_mm"
    ] == pytest.approx(20.0)
    assert document["geometry"]["projector_stack_geometry_status"] == (
        "user_defined_non_oem_principle_model"
    )
    by_key = {part["key"]: part for part in document["parts"]}

    for key, expected in PROJECTOR_RECONSTRUCTION.items():
        lens = by_key[key]
        housing = by_key[f"{key}_housing"]
        yoke = by_key[f"{key}_yoke"]
        coil = by_key[f"{key}_excitation_coil"]
        poles = (
            by_key[f"{key}_upper_pole"],
            by_key[f"{key}_lower_pole"],
        )

        for stack_part in (lens, housing, yoke, coil, *poles):
            assert stack_part["vacuum_inner_diameter_mm"] == pytest.approx(
                20.0
            )

        assert lens["local_center_z_mm"] == pytest.approx(expected["center"])
        assert lens["optical_reference_local_z_mm"] == pytest.approx(
            expected["center"]
        )
        assert lens["length_mm"] == pytest.approx(expected["length"])
        assert lens["local_start_z_mm"] == pytest.approx(
            expected["center"] - 0.5 * expected["length"]
        )
        assert lens["local_end_z_mm"] == pytest.approx(
            expected["center"] + 0.5 * expected["length"]
        )
        assert housing["length_mm"] == pytest.approx(
            expected["envelope_length"]
        )
        assert yoke["length_mm"] == pytest.approx(
            expected["envelope_length"]
        )
        assert lens["mechanical_outer_diameter_mm"] == pytest.approx(
            expected["housing_od"]
        )
        assert lens["mechanical_clear_bore_diameter_mm"] == pytest.approx(
            expected["clear_bore"]
        )
        assert lens["pole_gap_mm"] == pytest.approx(expected["pole_gap"])
        assert lens["mechanical_geometry_status"] == (
            "engineering_reconstruction_not_oem"
        )
        # The structured status declares the non-OEM model; provenance prose
        # may be clarified without changing the geometry or its authority.
        assert lens["mechanical_geometry_source"].strip()

        assert housing["mechanical_outer_diameter_mm"] == pytest.approx(
            expected["housing_od"]
        )
        assert yoke["mechanical_outer_diameter_mm"] == pytest.approx(
            expected["yoke_od"]
        )
        assert yoke["mechanical_inner_diameter_mm"] == pytest.approx(
            expected["yoke_id"]
        )
        # Coil diameters are user-editable, like its axial envelope. Keep
        # checking material thickness, vacuum clearance and radial nesting
        # without replacing a valid custom diameter with the reconstruction.
        assert lens["mechanical_clear_bore_diameter_mm"] <= coil["mechanical_inner_diameter_mm"]
        assert coil["mechanical_inner_diameter_mm"] < coil["mechanical_outer_diameter_mm"]
        assert coil["mechanical_outer_diameter_mm"] <= yoke["mechanical_inner_diameter_mm"]
        assert coil["parent_key"] == key
        # Coil length is independently editable. Its persisted material
        # envelope must remain centred and inside the surrounding hardware;
        # it need not retain the initial reconstruction's axial fraction.
        assert coil["length_mm"] > 0
        assert coil["local_center_z_mm"] == pytest.approx(lens["local_center_z_mm"])
        assert coil["local_start_z_mm"] == pytest.approx(
            coil["local_center_z_mm"] - coil["length_mm"] / 2
        )
        assert coil["local_end_z_mm"] == pytest.approx(
            coil["local_center_z_mm"] + coil["length_mm"] / 2
        )
        assert housing["local_start_z_mm"] <= coil["local_start_z_mm"]
        assert coil["local_end_z_mm"] <= housing["local_end_z_mm"]

        for pole in poles:
            assert pole["mechanical_outer_diameter_mm"] == pytest.approx(
                expected["pole_shoulder_od"]
            )
            assert pole["mechanical_bore_diameter_mm"] == pytest.approx(
                expected["pole_bore"]
            )
            assert pole["vacuum_inner_diameter_mm"] == pytest.approx(
                expected["clear_bore"]
            )
            assert pole["pole_nose_axial_length_mm"] == pytest.approx(
                expected["pole_nose"]
            )
            assert pole["pole_cone_angle_to_axis_deg"] == pytest.approx(63.0)
            assert pole["pole_face_land_axial_thickness_mm"] == pytest.approx(
                3.0
            )
            assert pole["pole_root_fillet_radius_range_mm"] == pytest.approx(
                (2.0, 4.0)
            )


def test_recording_manifest_accepts_nonuniform_nonoverlapping_projector_lens_gap():
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "instruments"
        / "project_and_recording_system"
        / "EnergyFilter.toml"
    )
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    intermediate = next(
        part
        for part in document["parts"]
        if part["key"] == "intermediate_lens_housing"
    )
    intermediate.update({
        "local_start_z_mm": 136.5,
        "local_center_z_mm": 251.5,
        "local_end_z_mm": 366.5,
    })

    validate_document(document)
    assert intermediate["local_center_z_mm"] == 251.5


def test_recording_manifest_requires_projector_geometry_provenance():
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "instruments"
        / "project_and_recording_system"
        / "EnergyFilter.toml"
    )
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    diffraction = next(
        part
        for part in document["parts"]
        if part["key"] == "diffraction_lens"
    )
    diffraction.pop("mechanical_geometry_source")

    with pytest.raises(
        ValueError,
        match="Missing mechanical_geometry_source for projector lens",
    ):
        validate_document(document)


def test_operating_mode_storage_tracks_calculated_and_retained_values():
    catalog = load_operating_mode_catalog()
    by_key = {mode.key: mode for mode in catalog.modes}

    assert set(by_key) == {
        "micro_probe", "nano_probe", "imaging", "diffraction"
    }
    assert {mode.family for mode in catalog.modes} == {
        "condenser", "projector"
    }
    assert all(mode.devices for mode in catalog.modes)
    assert all(
        mode.calibration_status.startswith(
            ("calibrated_", "computed_", "retained_not_recomputed_")
        )
        for mode in catalog.modes
    )
    assert by_key["micro_probe"].calibration_status.startswith(
        "retained_not_recomputed_"
    )
    assert by_key["nano_probe"].calibration_status.startswith(
        "computed_300kv_non_oem_source_probe_corrector"
    )
    assert "non_oem" in by_key["diffraction"].calibration_status
    assert by_key["micro_probe"].targets[
        "achieved_convergence_sem_angle_mrad"
    ] < 0.5
    assert 20.0 <= by_key["nano_probe"].targets[
        "achieved_convergence_sem_angle_mrad"
    ] <= 40.0
    assert by_key["micro_probe"].apertures[
        "condenser_aperture_2"
    ]["diameter_mm"] == pytest.approx(0.10)
    assert by_key["nano_probe"].apertures[
        "condenser_aperture_2"
    ]["diameter_mm"] == pytest.approx(0.10)
    assert all(
        "radius_mm" not in values
        for mode in by_key.values()
        for values in mode.apertures.values()
    )
    for mode_key in ("micro_probe", "nano_probe"):
        assert by_key[mode_key].devices["probe_tl22_lens"][
            "percent"
        ] == pytest.approx(60.0)
        assert by_key[mode_key].devices["probe_tl21_lens"][
            "percent"
        ] == pytest.approx(60.0)
        assert by_key[mode_key].devices["probe_tl12_lens"][
            "percent"
        ] == pytest.approx(60.0)
    assert by_key["imaging"].targets["conjugate_plane"] == (
        "objective_image_plane"
    )
    assert by_key["diffraction"].targets["conjugate_plane"] == (
        "mode_dependent_diffraction_reference_plane"
    )
    assert by_key["diffraction"].targets["reference_surface"] == (
        "stem_main_screen_or_tem_active_recording_stop"
    )
    assert by_key["imaging"].targets[
        "achieved_relay_error_um"
    ] < 0.01
    assert by_key["imaging"].targets[
        "validation_step_mm"
    ] == pytest.approx(0.00625)
    assert by_key["diffraction"].targets[
        "achieved_effective_camera_length_m"
    ] == pytest.approx(0.05, rel=3.0e-2)
    constraint = next(
        item
        for item in catalog.crossover_constraints
        if item.key == "c1_c2_interlens"
    )
    assert constraint.upstream_lens == "condenser_lens_1"
    assert constraint.downstream_lens == "condenser_lens_2"
    assert constraint.target_z_source == "pole_gap_midpoint"
    assert constraint.status == "confirmed"


def test_every_part_owns_vacuum_geometry_and_condenser_poles_are_nested():
    catalog = AssemblyCatalog()
    state = default_state()
    assembly = catalog.apply(state, catalog.default_selection())

    assert all(
        float(part.data["vacuum_inner_diameter_mm"]) > 0.0
        for part in assembly.parts
    )
    by_key = {part.key: part for part in assembly.parts}
    condenser_poles = {
        "condenser_lens_1": ("condenser_lens_1_lower_pole",),
        "condenser_lens_2": ("condenser_lens_2_upper_pole",),
        "condenser_lens_3": (
            "condenser_lens_3_upper_pole",
            "condenser_lens_3_lower_pole",
        ),
    }
    for lens_key, pole_keys in condenser_poles.items():
        for pole_key in pole_keys:
            pole = by_key[pole_key]
            assert pole.parent_key == lens_key
            assert pole.start_z_mm >= by_key[lens_key].start_z_mm
            assert pole.end_z_mm <= by_key[lens_key].end_z_mm
    assert assembly.vacuum_bore_segments[0].start_z_mm == 0.0
    assert assembly.vacuum_bore_segments[-1].end_z_mm == assembly.exit_z_mm


def test_operating_profile_round_trip_uses_toml(tmp_path: Path):
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    state = default_state()
    state.objective_lens.percent = 87.25
    state.objective_lens.cs_mm = 0.85
    state.objective_lens.polarity = -1
    state.sample.wave_multislice_enabled = False
    state.sample.wave_slice_thickness_angstrom = 1.25
    state.sample.wave_atomistic_enabled = True
    state.sample.wave_frozen_phonon_enabled = True
    state.sample.wave_frozen_phonon_configurations = 9
    state.sample.wave_frozen_phonon_sigma_angstrom = 0.072
    state.sample.wave_frozen_phonon_seed = 12345
    state.sample.real_inelastic_enabled = True
    state.sample.real_plasmon_mean_free_path_nm = 177.0
    state.sample.real_ionisation_mean_free_path_nm = 999.0
    state.sample.real_absorption_mean_free_path_nm = 2500.0
    path = tmp_path / "operating-profile.toml"

    save_profile(path, state, selection)
    loaded_selection, values = read_profile(path)
    restored = default_state()
    skipped = apply_profile_values(restored, values)

    assert loaded_selection == selection
    assert skipped == []
    assert restored.objective_lens.percent == 87.25
    assert restored.objective_lens.cs_mm == 0.85
    assert restored.objective_lens.polarity == -1
    assert restored.sample.wave_multislice_enabled is False
    assert restored.sample.wave_slice_thickness_angstrom == pytest.approx(1.25)
    assert restored.sample.wave_atomistic_enabled is True
    assert restored.sample.wave_frozen_phonon_enabled is True
    assert restored.sample.wave_frozen_phonon_configurations == 9
    assert restored.sample.wave_frozen_phonon_sigma_angstrom == pytest.approx(
        0.072
    )
    assert restored.sample.wave_frozen_phonon_seed == 12345
    assert restored.sample.real_inelastic_enabled is True
    assert restored.sample.real_plasmon_mean_free_path_nm == pytest.approx(177.0)
    assert restored.sample.real_ionisation_mean_free_path_nm == pytest.approx(999.0)
    assert restored.sample.real_absorption_mean_free_path_nm == pytest.approx(2500.0)


@pytest.mark.parametrize("gun", ("FEG", "FEG + Mono", "Thermionic"))
def test_every_c2_assembly_can_propagate_beyond_its_last_wall(gun):
    catalog = AssemblyCatalog()
    state = default_state()
    assembly = catalog.apply(
        state, AssemblySelection(gun, "C2", "Energy Filter")
    )
    state.step_mm = 5.0
    state.history_step_mm = 5.0
    state.sample.diffraction_enabled = False
    emitter = getattr(state.electron_gun, "emitter", None)
    if emitter is not None:
        emitter.ray_count = 9
    else:
        state.electron_gun.ray_count = 9

    result = run(state)

    assert result.branches["000"].z[-1] > assembly.exit_z_mm


def test_profile_assignments_are_transactional_and_domain_checked():
    state = default_state()
    original_percent = state.objective_lens.percent

    with pytest.raises(ValueError, match="step_mm must be positive"):
        apply_profile_values(state, {
            "objective_lens": {"percent": 87.0},
            "simulation": {"step_mm": 0.0},
        })

    assert state.objective_lens.percent == original_percent
    assert state.step_mm == 0.5

    with pytest.raises(ValueError, match="wave_slice_thickness_angstrom"):
        apply_profile_values(
            state,
            {"sample": {"wave_slice_thickness_angstrom": 0.0}},
        )

    with pytest.raises(ValueError, match="wave_frozen_phonon_configurations"):
        apply_profile_values(
            state,
            {"sample": {"wave_frozen_phonon_configurations": 65}},
        )

    with pytest.raises(ValueError, match="real_plasmon_mean_free_path_nm"):
        apply_profile_values(
            state,
            {"sample": {"real_plasmon_mean_free_path_nm": -1.0}},
        )


def test_profile_cannot_override_catalog_owned_topology():
    state = default_state()
    skipped = apply_profile_values(
        state, {"simulation": {"corrector_mode": "no_corrector"}}
    )

    assert skipped == ["simulation.corrector_mode"]
    assert state.corrector_mode == "probe_corrector"


def test_profile_operating_values_survive_layout_revalidation():
    catalog = AssemblyCatalog()
    state = default_state()
    catalog.apply(state, catalog.default_selection())
    apply_profile_values(state, {
        "objective_lens": {
            "percent": 87.25,
            "cs_mm": 0.85,
            "polarity": -1,
        }
    })

    apply_physical_layout_to_state(
        state, preserve_operating_parameters=True
    )

    assert state.objective_lens.percent == 87.25
    assert state.objective_lens.cs_mm == 0.85
    assert state.objective_lens.polarity == -1
