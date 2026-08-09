from copy import deepcopy

import pytest

from temsim import module_manifest
from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.component_keys import (
    DIFFRACTION_LENS,
    IMAGE_CORRECTOR_TL11_LENS,
    IMAGE_CORRECTOR_TL12_LENS,
    IMAGE_CORRECTOR_TL21_LENS,
    IMAGE_CORRECTOR_TL22_LENS,
    INTERMEDIATE_LENS,
    MINI_CONDENSER,
    OBJECTIVE_LENS,
    PROBE_TL12_LENS,
    PROBE_TL21_LENS,
    PROBE_TL22_LENS,
    PROJECTOR_LENS_1,
    PROJECTOR_LENS_2,
)
from temsim.optics.column import default_state
from temsim.presets import apply as apply_preset


def _lens_by_key(state):
    return {lens.key: lens for lens in state.lenses}


def _configured_polarities(state):
    return {
        key: int(part.data["field_polarity"])
        for key, part in state._module_manifest_parts.items()
        if module_manifest.part_requires_field_polarity(part.data)
    }


@pytest.mark.parametrize(
    "selection",
    (
        AssemblyCatalog().default_selection(),
        AssemblySelection(
            "FEG",
            "C3 + Probe Corrector + Image Corrector",
            "Energy Filter",
        ),
    ),
)
def test_selected_manifest_owns_every_magnetic_lens_polarity(selection):
    state = default_state()
    AssemblyCatalog().apply(state, selection)
    lenses = _lens_by_key(state)
    configured = _configured_polarities(state)

    assert configured
    assert set(configured) <= lenses.keys()
    for key, field_polarity in configured.items():
        lens = lenses[key]
        part = state._module_manifest_parts[key]
        assert lens.polarity == field_polarity
        assert lens.field_polarity_status == (
            part.data["field_polarity_status"]
        )
        assert lens.field_polarity_source == (
            part.data["field_polarity_source"]
        )


def test_default_stem_signs_are_loaded_from_selected_tomls():
    state = default_state()
    lenses = _lens_by_key(state)

    assert state.illumination_mode == "STEM"
    assert lenses[MINI_CONDENSER].polarity == -1
    assert lenses[PROBE_TL22_LENS].polarity == -1
    assert lenses[PROBE_TL21_LENS].polarity == 1
    assert lenses[PROBE_TL12_LENS].polarity == 1
    assert lenses[IMAGE_CORRECTOR_TL11_LENS].polarity == -1
    assert lenses[IMAGE_CORRECTOR_TL12_LENS].polarity == 1
    assert lenses[IMAGE_CORRECTOR_TL21_LENS].polarity == 1
    assert lenses[IMAGE_CORRECTOR_TL22_LENS].polarity == -1

    for key in (
        OBJECTIVE_LENS,
        DIFFRACTION_LENS,
        INTERMEDIATE_LENS,
        PROJECTOR_LENS_1,
        PROJECTOR_LENS_2,
    ):
        assert lenses[key].polarity == 1


def test_tem_stem_mode_toml_reverses_only_mini_condenser_effective_field():
    state = default_state()
    positions = {lens.key: lens.z_mm for lens in state.lenses}

    apply_preset(state, "TEM image")
    tem_polarities = {
        lens.key: lens.polarity for lens in state.lenses
    }
    assert tem_polarities[MINI_CONDENSER] == 1

    apply_preset(state, "STEM image")
    stem_polarities = {
        lens.key: lens.polarity for lens in state.lenses
    }
    assert stem_polarities[MINI_CONDENSER] == -1
    assert {
        key
        for key in tem_polarities
        if tem_polarities[key] != stem_polarities[key]
    } == {MINI_CONDENSER}
    assert {lens.key: lens.z_mm for lens in state.lenses} == positions


def test_new_assembly_restores_manifest_signs_without_moving_lenses():
    state = default_state()
    catalog = AssemblyCatalog()
    selection = AssemblySelection(
        "FEG",
        "C3 + Probe Corrector + Image Corrector",
        "Energy Filter",
    )
    catalog.apply(state, selection)
    positions = {lens.key: lens.z_mm for lens in state.lenses}
    expected = _configured_polarities(state)
    for key in expected:
        _lens_by_key(state)[key].polarity *= -1

    catalog.apply(state, selection)
    lenses = _lens_by_key(state)

    assert {lens.key: lens.z_mm for lens in state.lenses} == positions
    assert {
        key: lenses[key].polarity for key in expected
    } == expected


def test_recalculation_preserves_runtime_polarity_override():
    state = default_state()
    lens = _lens_by_key(state)[PROJECTOR_LENS_1]
    configured = int(
        state._module_manifest_parts[PROJECTOR_LENS_1]
        .data["field_polarity"]
    )
    lens.polarity = -configured

    apply_physical_layout_to_state(
        state, preserve_operating_parameters=True
    )

    assert lens.polarity == -configured
    assert lens.field_polarity_status == "provisional_model_assumption"


@pytest.mark.parametrize("invalid", (0, 2, -2, 1.0, True))
def test_manifest_rejects_invalid_magnetic_field_polarity(invalid):
    document = deepcopy(module_manifest.read_document(
        module_manifest.MODULE_ROOT
        / "project_and_recording_system"
        / "EnergyFilter.toml"
    ))
    part = next(
        part for part in document["parts"]
        if part["key"] == PROJECTOR_LENS_1
    )
    part["field_polarity"] = invalid

    with pytest.raises(ValueError, match="field_polarity must be integer"):
        module_manifest.validate_document(document)


def test_manifest_requires_polarity_provenance():
    document = deepcopy(module_manifest.read_document(
        module_manifest.MODULE_ROOT
        / "project_and_recording_system"
        / "EnergyFilter.toml"
    ))
    part = next(
        part for part in document["parts"]
        if part["key"] == PROJECTOR_LENS_2
    )
    del part["field_polarity_source"]

    with pytest.raises(ValueError, match="field_polarity_source"):
        module_manifest.validate_document(document)
