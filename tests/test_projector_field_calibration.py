from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
import pytest

from temsim import module_manifest
from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.optics.direct_alignment import (
    PROJECTOR_KEYS,
    projector_field_calibration_rows,
)


def test_selected_recording_toml_owns_projector_field_calibration():
    state = default_state()
    catalog = AssemblyCatalog()
    assembly = catalog.apply(state, catalog.default_selection())
    rows = {row["key"]: row for row in projector_field_calibration_rows(state)}

    for key in PROJECTOR_KEYS:
        part = assembly.part(key).data
        lens = next(item for item in state.lenses if item.key == key)
        assert lens.b0_t == pytest.approx(part["maximum_peak_field_t"])
        assert lens.a_mm == pytest.approx(part["field_half_width_mm"])
        assert lens.max_percent == pytest.approx(
            part["maximum_excitation_percent"]
        )
        assert np.asarray([
            [term.amplitude, term.offset, term.sigma]
            for term in lens.gaussian
        ]) == pytest.approx(np.asarray(part["field_profile_terms"]))
        assert rows[key]["status"] == part["field_calibration_status"]
        assert rows[key]["source"] == part["field_calibration_source"]


def test_saved_state_keeps_excitation_but_omits_toml_field_calibration():
    state = default_state()
    payload = state.to_dict()
    rows = {
        item["key"]: item for item in payload["lenses"]
        if item["key"] in PROJECTOR_KEYS
    }

    for key in PROJECTOR_KEYS:
        assert "percent" in rows[key]
        for field in ("b0_t", "a_mm", "max_percent", "gaussian"):
            assert field not in rows[key]


def test_custom_recording_toml_overrides_python_projector_defaults(
    tmp_path: Path,
):
    root = tmp_path / "instruments"
    shutil.copytree(module_manifest.MODULE_ROOT, root)
    path = root / "project_and_recording_system" / "EnergyFilter.toml"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "maximum_peak_field_t = 0.42",
        "maximum_peak_field_t = 0.5",
        1,
    )
    path.write_text(text, encoding="utf-8")

    state = default_state()
    catalog = AssemblyCatalog(root)
    catalog.apply(state, catalog.default_selection())

    assert state.projector_lens_p1.b0_t == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("maximum_peak_field_t", 0.0, "maximum_peak_field_t"),
        ("field_half_width_mm", -1.0, "field_half_width_mm"),
        ("maximum_excitation_percent", 101.0, "excitation percentages"),
        ("field_profile_terms", [[1.0, 0.0, 0.0]], "sigma"),
        ("field_calibration_status", "OEM-ish", "field_calibration_status"),
        ("field_calibration_source", "", "field_calibration_source"),
    ),
)
def test_manifest_rejects_invalid_projector_field_calibration(
    field, value, message
):
    document = deepcopy(module_manifest.read_document(
        module_manifest.MODULE_ROOT
        / "project_and_recording_system"
        / "EnergyFilter.toml"
    ))
    part = next(
        item for item in document["parts"]
        if item["key"] == "projector_lens_1"
    )
    part[field] = value

    with pytest.raises(ValueError, match=message):
        module_manifest.validate_document(document)
