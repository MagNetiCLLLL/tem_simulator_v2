"""Vacuum is explicit; no library is allowed to supply material implicitly."""
from copy import deepcopy
from dataclasses import fields
import shutil

import pytest

from specimen_inputs import SI_CIF, imported_sample
from temsim.optics.column import default_state
from temsim.optics.model import Sample, State
from temsim.specimen.source import (
    active_cif_path, specimen_is_vacuum, specimen_interactions_active,
    validate_sample_source,
)
from temsim.specimen.scene import SpecimenScene
from temsim.calculation_cache import _cif_content_identity, calculation_signatures
from temsim.calculation_manifest import capture_external_input_identities


def test_default_vacuum_has_no_material_or_implicit_file_dependencies():
    state = default_state()
    assert state.sample.specimen_mode == "vacuum"
    assert not state.sample.inserted
    assert "reference_sample_key" not in {field.name for field in fields(Sample)}
    assert active_cif_path(state.sample) == ""
    scene = SpecimenScene.from_state(state, include_eds_materials=True)
    assert scene.is_vacuum and not scene.structure_available
    assert scene.sample_material is scene.support_material is None
    assert not any(row.role.startswith("specimen:") for row in capture_external_input_identities(state))
    assert State.from_dict(state.to_dict()).sample == state.sample


def test_imported_material_and_vacuum_preserve_incident_beam_identity():
    state = default_state()
    before = calculation_signatures(state)
    imported_sample(state)
    after = calculation_signatures(state)
    assert after["incident"] == before["incident"]
    assert after["elastic"] != before["elastic"]
    assert specimen_interactions_active(state.sample)
    state.sample.specimen_mode = "vacuum"
    # Even a retained import path cannot become active material in vacuum.
    assert active_cif_path(state.sample) == ""
    assert specimen_is_vacuum(state.sample)
    assert not specimen_interactions_active(state.sample)


def test_inserted_unconfigured_material_is_rejected_before_page_work():
    from temsim.simulation_workflow import calculate_workflow
    state = default_state()
    state.sample.specimen_mode, state.sample.inserted = "atomic", True
    with pytest.raises(ValueError, match="Import a CIF"):
        validate_sample_source(state.sample)
    with pytest.raises(ValueError, match="Import a CIF"):
        calculate_workflow(state, workflow="sample")


@pytest.mark.parametrize("sample_values", [
    {"specimen_mode": "reference"}, {"reference_sample_key": "si_110"},
])
def test_retired_selection_is_not_silently_converted(sample_values):
    data = default_state().to_dict()
    data["sample"].update(sample_values)
    original = deepcopy(data)
    with pytest.raises(ValueError, match="Reference CIF selection has been removed"):
        State.from_dict(data)
    assert data == original


def test_cif_bytes_invalidate_material_cache_but_sidecars_are_not_consumed(tmp_path):
    path = tmp_path / "Si.cif"
    shutil.copyfile(SI_CIF, path)
    state = default_state()
    imported_sample(state, path=path)
    before = _cif_content_identity(state.to_dict())
    path.with_suffix(".toml").write_text("invalid metadata = [")
    assert _cif_content_identity(state.to_dict()) == before
    path.write_bytes(path.read_bytes() + b"\n# revised input\n")
    assert _cif_content_identity(state.to_dict()) != before
    roles = {row.role for row in capture_external_input_identities(state)}
    assert "specimen:cif" in roles
    assert "specimen:reference_metadata" not in roles
