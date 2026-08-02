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


def test_every_catalog_option_resolves_to_the_requested_toml():
    catalog = AssemblyCatalog()
    state = default_state()
    for column in catalog.columns:
        selection = catalog.default_selection().__class__(
            gun="FEG",
            column=column.name,
            recording="No Energy Filter",
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


def test_rebased_lens_percentages_preserve_reference_fields():
    catalog = AssemblyCatalog()
    state = default_state()
    catalog.apply(state, catalog.default_selection())
    by_key = {lens.key: lens for lens in state.lenses}

    assert by_key["probe_tl12_lens"].scale() == pytest.approx(0.33)
    objective = by_key["objective_lens"]
    assert (
        objective.upper_b0_t * objective.percent / 100.0
    ) == pytest.approx(0.2690863306357239)

    catalog.apply(
        state,
        catalog.default_selection().__class__(
            "FEG", "C3 + Probe Corrector + Image Corrector", "Energy Filter"
        ),
    )
    by_key = {lens.key: lens for lens in state.lenses}
    expected_scales = {
        "image_ol_post_lens": 1.690720840008,
        "image_tl22_lens": 0.196885770873,
        "image_adapter_lens": 0.332974454964,
    }
    for key, expected in expected_scales.items():
        assert by_key[key].scale() == pytest.approx(expected)


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
    assert audit.part_definition_count == 441
    assert audit.assembly_count == 30


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


@pytest.mark.parametrize("recording", ("No Energy Filter", "Energy Filter"))
def test_projector_lenses_have_non_overlapping_mechanical_envelopes(recording):
    catalog = AssemblyCatalog()
    state = default_state()
    assembly = catalog.apply(
        state, AssemblySelection("FEG", "C3 + Probe Corrector", recording)
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

    assert clearances[0] == pytest.approx(5.0)
    assert all(clearance >= 0.0 for clearance in clearances)
    lens_by_key = {lens.key: lens for lens in state.lenses}
    for part in parts:
        assert lens_by_key[part.key].z_mm == pytest.approx(part.center_z_mm)
        upper = assembly.part(f"{part.key}_upper_pole")
        lower = assembly.part(f"{part.key}_lower_pole")
        assert upper.parent_key == part.key
        assert lower.parent_key == part.key
        assert upper.start_z_mm == pytest.approx(part.start_z_mm)
        assert lower.end_z_mm == pytest.approx(part.end_z_mm)
        assert lower.start_z_mm - upper.end_z_mm == pytest.approx(20.0)


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
        "local_start_z_mm": 150.0,
        "local_center_z_mm": 237.5,
        "local_end_z_mm": 325.0,
        "optical_reference_local_z_mm": 237.5,
    })

    with pytest.raises(ValueError, match="requires at least 5 mm"):
        validate_document(document)


def test_operating_mode_storage_waits_for_confirmed_calibration_values():
    catalog = load_operating_mode_catalog()
    by_key = {mode.key: mode for mode in catalog.modes}

    assert set(by_key) == {
        "micro_probe", "nano_probe", "imaging", "diffraction"
    }
    assert {mode.family for mode in catalog.modes} == {
        "condenser", "projector"
    }
    assert all(not mode.devices and not mode.apertures for mode in catalog.modes)
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


@pytest.mark.parametrize("gun", ("FEG", "FEG + Mono", "Thermionic"))
@pytest.mark.parametrize("recording", ("No Energy Filter", "Energy Filter"))
def test_every_c2_assembly_can_propagate_beyond_its_last_wall(gun, recording):
    catalog = AssemblyCatalog()
    state = default_state()
    assembly = catalog.apply(
        state, AssemblySelection(gun, "C2", recording)
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
