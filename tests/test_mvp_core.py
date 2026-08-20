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
        "length": 68.0,
        "housing_od": 168.0,
        "yoke_od": 162.0,
        "yoke_id": 143.0,
        "coil_id": 56.0,
        "coil_od": 140.0,
        "coil_length": 44.0,
        "pole_bore": 10.0,
        "clear_bore": 8.5,
        "pole_gap": 4.0,
        "pole_shoulder_od": 54.0,
        "pole_nose": 12.0,
    },
    "intermediate_lens": {
        "center": 252.5,
        "length": 60.0,
        "housing_od": 164.0,
        "yoke_od": 158.0,
        "yoke_id": 141.0,
        "coil_id": 60.0,
        "coil_od": 138.0,
        "coil_length": 38.0,
        "pole_bore": 12.0,
        "clear_bore": 10.5,
        "pole_gap": 6.0,
        "pole_shoulder_od": 55.0,
        "pole_nose": 12.0,
    },
    "projector_lens_1": {
        "center": 432.5,
        "length": 62.0,
        "housing_od": 164.0,
        "yoke_od": 158.0,
        "yoke_id": 141.0,
        "coil_id": 58.0,
        "coil_od": 138.0,
        "coil_length": 40.0,
        "pole_bore": 10.0,
        "clear_bore": 8.5,
        "pole_gap": 5.0,
        "pole_shoulder_od": 54.0,
        "pole_nose": 13.0,
    },
    "projector_lens_2": {
        "center": 635.0,
        "length": 70.0,
        "housing_od": 171.0,
        "yoke_od": 165.0,
        "yoke_id": 148.0,
        "coil_id": 66.0,
        "coil_od": 145.0,
        "coil_length": 45.0,
        "pole_bore": 15.0,
        "clear_bore": 13.5,
        "pole_gap": 7.0,
        "pole_shoulder_od": 62.0,
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

    assert audit.module_count == 10
    assert audit.part_definition_count == 466
    assert audit.assembly_count == 15


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

    assert c1.data["shared_housing_key"] == (
        "condenser_c1_c2_shared_housing"
    )
    assert c2.data["shared_housing_key"] == c1.data["shared_housing_key"]
    assert c1.end_z_mm == pytest.approx(c2.start_z_mm)


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


def test_projector_lenses_have_non_overlapping_mechanical_envelopes():
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
    clearances = [
        downstream.start_z_mm - upstream.end_z_mm
        for upstream, downstream in zip(parts, parts[1:])
    ]

    assert clearances == pytest.approx((106.0, 119.0, 136.5))
    lens_by_key = {lens.key: lens for lens in state.lenses}
    for part in parts:
        assert lens_by_key[part.key].z_mm == pytest.approx(part.center_z_mm)
        upper = assembly.part(f"{part.key}_upper_pole")
        lower = assembly.part(f"{part.key}_lower_pole")
        assert upper.parent_key == part.key
        assert lower.parent_key == part.key
        assert upper.start_z_mm == pytest.approx(part.start_z_mm)
        assert lower.end_z_mm == pytest.approx(part.end_z_mm)
        assert lower.start_z_mm - upper.end_z_mm == pytest.approx(
            PROJECTOR_RECONSTRUCTION[part.key]["pole_gap"]
        )


@pytest.mark.parametrize("manifest_name", (
    "NoEnergyFilter.toml",
    "EnergyFilter.toml",
))
def test_projector_manifests_use_public_reference_engineering_dimensions(
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
        assert "not an OEM production drawing" in (
            lens["mechanical_geometry_source"]
        )

        assert housing["mechanical_outer_diameter_mm"] == pytest.approx(
            expected["housing_od"]
        )
        assert yoke["mechanical_outer_diameter_mm"] == pytest.approx(
            expected["yoke_od"]
        )
        assert yoke["mechanical_inner_diameter_mm"] == pytest.approx(
            expected["yoke_id"]
        )
        assert coil["mechanical_inner_diameter_mm"] == pytest.approx(
            expected["coil_id"]
        )
        assert coil["mechanical_outer_diameter_mm"] == pytest.approx(
            expected["coil_od"]
        )
        assert coil["length_mm"] == pytest.approx(expected["coil_length"])

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


def test_recording_manifest_rejects_future_projector_lens_overlap():
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
        if part["key"] == "intermediate_lens"
    )
    intermediate.update({
        "local_start_z_mm": 120.0,
        "local_center_z_mm": 150.0,
        "local_end_z_mm": 180.0,
        "optical_reference_local_z_mm": 150.0,
    })

    with pytest.raises(ValueError, match="requires at least 5 mm"):
        validate_document(document)


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


def test_operating_mode_storage_contains_calculated_optical_values():
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
        mode.calibration_status.startswith("calibrated_")
        for mode in catalog.modes
    )
    assert by_key["micro_probe"].targets[
        "achieved_convergence_sem_angle_mrad"
    ] < 0.5
    assert 20.0 <= by_key["nano_probe"].targets[
        "achieved_convergence_sem_angle_mrad"
    ] <= 40.0
    assert by_key["micro_probe"].apertures[
        "condenser_aperture_2"
    ]["radius_mm"] == pytest.approx(0.05)
    assert by_key["nano_probe"].apertures[
        "condenser_aperture_2"
    ]["radius_mm"] == pytest.approx(0.10)
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
        "objective_back_focal_plane"
    )
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
