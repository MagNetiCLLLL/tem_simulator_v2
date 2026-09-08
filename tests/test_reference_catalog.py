from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.optics.model import Sample, State
from temsim.specimen import reference_catalog as catalog
from temsim.specimen.geometry import quaternion_to_matrix
from temsim.specimen.source import active_cif_path, migrate_legacy_structure_source
from temsim.specimen.rutherford import read_cif_composition
from temsim.calculation_cache import _cif_content_identity
from temsim.calculation_manifest import capture_external_input_identities


def test_user_si_crystal_and_default_orientation_are_physical():
    sample = Sample()
    composition = read_cif_composition(active_cif_path(sample))
    assert dict(composition.atoms_per_cell) == {14: 8.0}
    assert composition.volume_nm3 == pytest.approx(0.16131810739, rel=1e-7)
    assert dict(composition.number_densities_atoms_nm3)[14] * sample.thickness_nm == pytest.approx(247.9572858376)
    rotation = quaternion_to_matrix(sample.specimen_orientation_quaternion_wxyz)
    assert rotation @ (np.array([1, 1, 0]) / np.sqrt(2)) == pytest.approx([0, 0, 1], abs=1e-12)


def test_reference_switch_preserves_size_and_manual_overrides():
    sample = Sample(size_x_nm=12, size_y_nm=12, thickness_nm=7, real_tail_atomic_number=8)
    catalog.apply_reference_sample(sample, "au_001")
    assert sample.reference_sample_key == "au_001"
    assert sample.zone_axis_uvw == (0, 0, 1)
    assert (sample.size_x_nm, sample.size_y_nm, sample.thickness_nm) == (12, 12, 7)
    assert sample.real_tail_atomic_number == 8


def test_reference_mode_case_keeps_thermal_and_metadata_dependencies():
    sample = Sample(specimen_mode=" REFERENCE ")
    assert catalog.reference_thermal_sigma(sample) == 0.085
    assert "Loane" in catalog.reference_thermal_source(sample)
    identity = _cif_content_identity({"sample": vars(sample)})
    assert identity["metadata"]["available"]


@pytest.mark.parametrize("vacuum", ["retracted", "zero_thickness"])
def test_missing_reference_does_not_block_explicit_vacuum(vacuum):
    from temsim.specimen.scene import SpecimenScene
    state = default_state()
    state.sample.reference_sample_key = "missing_after_file_rename"
    if vacuum == "retracted":
        state.sample.inserted = False
    else:
        state.sample.thickness_nm = 0
    scene = SpecimenScene.from_state(state)
    assert scene.interacting_thickness_nm == 0
    assert not scene.structure_available
    assert catalog.reference_thermal_sigma(state.sample) == 0
    assert not any(row.role == "specimen:cif" for row in capture_external_input_identities(state))
    state.sample.inserted = True
    state.sample.thickness_nm = 5
    with pytest.raises(ValueError, match="unavailable"):
        SpecimenScene.from_state(state)


def test_directory_discovery_refresh_and_missing_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "REFERENCE_DIRECTORY", tmp_path)
    assert catalog.available_reference_samples() == ()
    (tmp_path / "New.cif").write_text("data_new\n")
    entry, = catalog.refresh_reference_samples()
    assert entry.key == "New"
    assert entry.zone_axis == (0, 0, 1)
    assert entry.thermal_sigma_angstrom == 0
    assert entry.inelastic_preset_key == ""
    with pytest.raises(ValueError, match="unavailable"):
        catalog.get_reference_sample("missing")


def test_default_roundtrip_and_legacy_migration_preserve_dimensions():
    state = default_state()
    data = state.to_dict()
    assert not any(key.startswith("virtual_") for key in data["sample"])
    restored = State.from_dict(data)
    assert restored.sample.thickness_nm == 5
    assert restored.sample.size_x_nm == 10
    data["schema_version"] = 76
    data["sample"].update(specimen_mode="virtual", specimen_preset_key="si_110", thickness_nm=27, size_x_nm=1000,
                           specimen_rotation_x_deg=0, specimen_rotation_y_deg=0,
                           specimen_orientation_quaternion_wxyz=[1, 0, 0, 0])
    restored = State.from_dict(data)
    assert restored.sample.specimen_mode == "reference"
    assert restored.sample.thickness_nm == 27
    assert restored.sample.size_x_nm == 1000
    assert restored.sample.real_tail_material_source == "structure"  # explicit value survives
    old = migrate_legacy_structure_source({"specimen_mode": "virtual", "specimen_preset_key": "vacuum"})
    assert old["inserted"] is False
    with pytest.raises(ValueError, match="unavailable"):
        migrate_legacy_structure_source({"specimen_mode": "virtual", "specimen_preset_key": "amorphous_carbon"})


def test_reference_content_and_metadata_are_cache_and_manifest_inputs(tmp_path, monkeypatch):
    from shutil import copyfile
    original = catalog.get_reference_sample("si_110")
    copyfile(original.cif_path, tmp_path / "Si.cif")
    copyfile(original.metadata_path, tmp_path / "Si.toml")
    monkeypatch.setattr(catalog, "REFERENCE_DIRECTORY", tmp_path)
    state = default_state()
    initial = _cif_content_identity(state.to_dict())
    with (tmp_path / "Si.cif").open("a") as stream:
        stream.write("\n# refreshed source\n")
    changed = _cif_content_identity(state.to_dict())
    assert changed != initial
    with (tmp_path / "Si.toml").open("a") as stream:
        stream.write("\n# refreshed metadata\n")
    assert _cif_content_identity(state.to_dict()) != changed
    roles = {item.role for item in capture_external_input_identities(state)}
    assert {"specimen:cif", "specimen:reference_metadata"} <= roles


def test_changed_cif_cannot_borrow_stale_material_thermal_model(tmp_path, monkeypatch):
    from shutil import copyfile
    si, au = catalog.get_reference_sample("si_110"), catalog.get_reference_sample("au_001")
    copyfile(au.cif_path, tmp_path / "Si.cif")
    copyfile(si.metadata_path, tmp_path / "Si.toml")
    monkeypatch.setattr(catalog, "REFERENCE_DIRECTORY", tmp_path)
    with pytest.raises(ValueError, match="thermal material"):
        catalog.reference_thermal_sigma(Sample())
    with pytest.raises(ValueError, match="thermal material"):
        catalog.reference_thermal_source(Sample())
