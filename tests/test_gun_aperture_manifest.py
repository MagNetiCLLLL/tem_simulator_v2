"""Gun Aperture is an upstream physical restriction, not accelerator hardware."""
from pathlib import Path

import pytest

from temsim.module_manifest import read_document, validate_document


ROOT = Path(__file__).resolve().parents[1] / "configs/instruments/gun"


@pytest.mark.parametrize("filename", ["FEG.toml", "FEG_Mono.toml"])
def test_default_gun_aperture_fits_downstream_of_gun_lens_in_both_guns(filename):
    document = read_document(ROOT / filename)
    validate_document(document)
    rows = {row["key"]: row for row in document["parts"]}
    aperture = rows["feg_dpa_aperture"]
    assert aperture["name"] == "Gun Aperture"
    assert "parent_key" not in aperture
    assert aperture["local_start_z_mm"] == 23.0
    assert aperture["optical_reference_local_z_mm"] == 24.0
    assert aperture["local_end_z_mm"] == 25.0
    assert rows["feg_extractor"]["local_end_z_mm"] < aperture["local_start_z_mm"]
    assert rows["feg_electrostatic_lens"]["local_end_z_mm"] < aperture["local_start_z_mm"]
    assert aperture["local_end_z_mm"] < rows["feg_accelerator"]["local_start_z_mm"]
    assert set(aperture["aperture_functions"]) == {"electron_interception", "differential_pumping_restriction"}
    assert rows["feg_electrostatic_lens"]["order"] < aperture["order"] < rows["feg_accelerator"]["order"]
    if "feg_monochromator_wien" in rows:
        assert aperture["local_end_z_mm"] < rows["feg_monochromator_wien"]["local_start_z_mm"]


@pytest.mark.parametrize("position", [7.0, 12.0, 18.0, 120.0])
def test_manifest_rejects_placements_before_gun_lens_exit_or_inside_accelerator(position):
    document = read_document(ROOT / "FEG.toml")
    aperture = next(row for row in document["parts"] if row["key"] == "feg_dpa_aperture")
    aperture.update(local_start_z_mm=position - 1, local_center_z_mm=position,
                    local_end_z_mm=position + 1, optical_reference_local_z_mm=position)
    with pytest.raises(ValueError, match="Gun Aperture"):
        validate_document(document)


def test_manifest_accepts_a_separate_free_gap_after_gun_lens():
    document = read_document(ROOT / "FEG.toml")
    aperture = next(row for row in document["parts"] if row["key"] == "feg_dpa_aperture")
    aperture.update(local_start_z_mm=25., local_center_z_mm=26.,
                    local_end_z_mm=27., optical_reference_local_z_mm=26.)
    validate_document(document)


def test_manifest_rejects_old_accelerator_parent_even_with_new_position():
    document = read_document(ROOT / "FEG.toml")
    aperture = next(row for row in document["parts"] if row["key"] == "feg_dpa_aperture")
    aperture["parent_key"] = "feg_accelerator"
    with pytest.raises(ValueError, match="Gun Aperture"):
        validate_document(document)
