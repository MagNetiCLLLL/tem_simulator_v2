from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
import pytest

from temsim import module_manifest
from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.optics.column import default_state


COLUMN_NAMES = (
    "C2",
    "C3",
    "C3 + Probe Corrector",
    "C3 + Image Corrector",
    "C3 + Probe Corrector + Image Corrector",
)
CONDENSER_KEYS = ("condenser_lens_1", "condenser_lens_2")


@pytest.mark.parametrize("column", COLUMN_NAMES)
def test_c1_c2_ratio_and_field_calibration_are_toml_authoritative(column):
    state = default_state()
    assembly = AssemblyCatalog().apply(
        state,
        AssemblySelection("FEG", column, "Energy Filter"),
    )
    lenses = {lens.key: lens for lens in state.lenses}
    c1 = assembly.part("condenser_lens_1")
    c2 = assembly.part("condenser_lens_2")

    assert c1.length_mm == pytest.approx(100.0)
    assert c2.length_mm == pytest.approx(200.0)
    assert c2.length_mm / c1.length_mm == pytest.approx(2.0)
    assert c1.length_mm + c2.length_mm == pytest.approx(300.0)
    assert c1.end_z_mm == pytest.approx(c2.start_z_mm)

    for key in CONDENSER_KEYS:
        part = assembly.part(key).data
        lens = lenses[key]
        assert lens.b0_t == pytest.approx(part["maximum_peak_field_t"])
        assert lens.a_mm == pytest.approx(part["field_half_width_mm"])
        assert lens.max_percent == pytest.approx(
            part["maximum_excitation_percent"]
        )
        assert lens.normalise_profile_peak is bool(
            part["normalise_field_profile_peak"]
        )
        assert np.asarray([
            [term.amplitude, term.offset, term.sigma]
            for term in lens.gaussian
        ]) == pytest.approx(np.asarray(part["field_profile_terms"]))
        assert lens.field_calibration_status == (
            part["field_calibration_status"]
        )
        assert lens.field_calibration_source == part["field_calibration_source"]
        assert lens.b0_t**2 * lens.a_mm == pytest.approx(22.5)

    assert state.condenser_system.condenser_lens_1.focal_length_mm() == (
        pytest.approx(0.8947412148057265, rel=2.0e-10)
    )
    assert state.condenser_system.condenser_lens_2.focal_length_mm() == (
        pytest.approx(7.03427403538115, rel=2.0e-10)
    )


def test_custom_column_toml_overrides_python_c1_field_defaults(tmp_path: Path):
    root = tmp_path / "instruments"
    shutil.copytree(module_manifest.MODULE_ROOT, root)
    path = root / "column" / "C3_ProbeCorrector.toml"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "condenser_lens_1 = 1.7748239349298847",
        "condenser_lens_1 = 2.0",
        1,
    )
    text = text.replace(
        "maximum_peak_field_t = 1.7748239349298847",
        "maximum_peak_field_t = 2.0",
        1,
    )
    path.write_text(text, encoding="utf-8")

    state = default_state()
    AssemblyCatalog(root).apply(
        state,
        AssemblySelection("FEG", "C3 + Probe Corrector", "Energy Filter"),
    )

    assert state.condenser_lens_1.lens.b0_t == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("maximum_peak_field_t", 0.0, "maximum_peak_field_t"),
        ("field_half_width_mm", -1.0, "field_half_width_mm"),
        ("maximum_excitation_percent", 101.0, "excitation percentages"),
        ("field_profile_terms", [[1.0, 0.0, 0.0]], "sigma"),
        (
            "normalise_field_profile_peak",
            "yes",
            "normalise_field_profile_peak",
        ),
    ),
)
def test_manifest_rejects_invalid_condenser_field_calibration(
    field, value, message
):
    document = deepcopy(module_manifest.read_document(
        module_manifest.MODULE_ROOT / "column" / "C3_ProbeCorrector.toml"
    ))
    part = next(
        item for item in document["parts"]
        if item["key"] == "condenser_lens_1"
    )
    part[field] = value

    with pytest.raises(ValueError, match=message):
        module_manifest.validate_document(document)


def test_manifest_rejects_c1_c2_ratio_or_peak_field_drift():
    document = deepcopy(module_manifest.read_document(
        module_manifest.MODULE_ROOT / "column" / "C3_ProbeCorrector.toml"
    ))
    document["geometry"]["c1_c2_lens_length_ratio_c2_to_c1"] = 1.5
    with pytest.raises(ValueError, match="declared total and ratio"):
        module_manifest.validate_document(document)

    document = deepcopy(module_manifest.read_document(
        module_manifest.MODULE_ROOT / "column" / "C3_ProbeCorrector.toml"
    ))
    part = next(
        item for item in document["parts"]
        if item["key"] == "condenser_lens_1"
    )
    part["maximum_peak_field_t"] += 0.1
    with pytest.raises(ValueError, match="runtime and mechanical design"):
        module_manifest.validate_document(document)
