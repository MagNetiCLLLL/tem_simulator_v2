from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.component_keys import (
    APERTURE_KEYS,
    POST_PROJECTOR_DETECTOR_CHAMBER,
    PROJECTION_CHAMBER_DPA_APERTURE,
)
from temsim.gui.diagnostic_tabs import PhysicalLayoutView
from temsim.mechanical_profiles import (
    FIXED_DIFFERENTIAL_PUMPING_APERTURE,
    POST_PROJECTOR_DETECTOR_CHAMBER as DETECTOR_CHAMBER_PROFILE,
)
from temsim.module_manifest import validate_document
from temsim.optics.column import default_state
from temsim.simulation_pipeline import aperture_stop_records


PROJECT_AND_RECORDING_ROOT = (
    Path(__file__).parents[1]
    / "configs"
    / "instruments"
    / "project_and_recording_system"
)


def _document(filename: str) -> dict:
    return tomllib.loads(
        (PROJECT_AND_RECORDING_ROOT / filename).read_text(encoding="utf-8")
    )


def _parts_by_key(document: dict) -> dict[str, dict]:
    return {str(part["key"]): part for part in document["parts"]}


@pytest.mark.parametrize(
    "filename", ("EnergyFilter.toml", "NoEnergyFilter.toml")
)
def test_post_p2_chamber_preserves_detector_planes_and_marks_non_oem_geometry(
    filename,
):
    document = validate_document(_document(filename))
    by_key = _parts_by_key(document)
    chamber = by_key[POST_PROJECTOR_DETECTOR_CHAMBER]
    dpa = by_key[PROJECTION_CHAMBER_DPA_APERTURE]
    p2_end = float(by_key["projector_lens_2_housing"]["local_end_z_mm"])

    assert chamber["mechanical_profile"] == DETECTOR_CHAMBER_PROFILE
    assert chamber["mechanical_only"] is True
    assert chamber["axial_vacuum_context_only"] is True
    assert chamber["local_start_z_mm"] == pytest.approx(p2_end)
    assert chamber["local_end_z_mm"] == pytest.approx(1100.0)
    assert chamber["mechanical_inner_diameter_mm"] == pytest.approx(180.0)
    assert chamber["mechanical_outer_diameter_mm"] == pytest.approx(200.0)
    assert chamber["contained_recording_plane_keys"] == [
        "haadf", "flu_screen", "df", "bf",
    ]
    assert chamber["upstream_boundary_aperture_key"] == (
        PROJECTION_CHAMBER_DPA_APERTURE
    )
    assert "provisional_non_oem" in chamber["mechanical_geometry_status"]

    assert dpa["mechanical_profile"] == (
        FIXED_DIFFERENTIAL_PUMPING_APERTURE
    )
    assert dpa["mechanical_part_role"] == "fixed_vacuum_restriction"
    assert dpa["mechanical_only"] is False
    assert dpa["branch"] == "common"
    assert dpa["local_start_z_mm"] == pytest.approx(p2_end)
    assert dpa["local_center_z_mm"] == pytest.approx(p2_end)
    assert dpa["local_end_z_mm"] == pytest.approx(p2_end)
    assert dpa["length_mm"] == pytest.approx(0.0)
    assert dpa["mechanical_bore_diameter_mm"] == pytest.approx(12.0)
    assert dpa["reference_bore_diameter_mm"] == pytest.approx(0.2)
    assert "not_oem" in dpa["design_bore_status"]
    assert "not_titan_oem" in dpa["reference_bore_status"]
    assert dpa["aperture_adjustability"] == (
        "fixed_non_retractable_hardware_toml_design_variable"
    )
    assert dpa["conjugate_plane_status"] == (
        "operating_mode_dependent_not_imposed_by_mechanical_layout"
    )
    assert dpa["optical_constraint_policy"] == (
        "always_inserted_hard_edge_no_automatic_preset_recalculation"
    )
    assert dpa["optical_reference_local_z_mm"] == pytest.approx(p2_end)
    assert "aperture_plate_attachment" not in dpa

    expected_gaps_mm = {
        "haadf": 7.25,
        "flu_screen": 127.25,
        "df": 217.25,
        "bf": 287.25,
        "camera": 399.75,
    }
    for key, expected_gap_mm in expected_gaps_mm.items():
        signal_z = float(by_key[key]["optical_reference_local_z_mm"])
        assert signal_z - p2_end == pytest.approx(expected_gap_mm)
    assert float(by_key["camera"]["optical_reference_local_z_mm"]) > float(
        chamber["local_end_z_mm"]
    )


def test_post_p2_chamber_must_start_at_p2_end():
    document = _document("EnergyFilter.toml")
    chamber = _parts_by_key(document)[POST_PROJECTOR_DETECTOR_CHAMBER]
    chamber["local_start_z_mm"] += 1.0
    chamber["local_center_z_mm"] += 0.5
    chamber["length_mm"] -= 1.0

    with pytest.raises(ValueError, match="must start at the P2 housing end"):
        validate_document(document)


def test_projection_chamber_dpa_stop_follows_its_adjustable_mechanics():
    document = _document("EnergyFilter.toml")
    dpa = _parts_by_key(document)[PROJECTION_CHAMBER_DPA_APERTURE]
    dpa["local_start_z_mm"] += 1.0
    dpa["local_center_z_mm"] += 1.0
    dpa["local_end_z_mm"] += 1.0

    with pytest.raises(
        ValueError,
        match="DPA stop and mechanical plane must coincide",
    ):
        validate_document(document)
    dpa["optical_reference_local_z_mm"] += 1.0
    dpa["mechanical_bore_diameter_mm"] = 0.4
    validate_document(document)
    assert dpa["reference_bore_diameter_mm"] == 0.2  # Literature reference unchanged.


def test_physical_layout_draws_post_p2_detector_chamber_without_moving_planes(
    qtbot,
):
    state = default_state()
    catalog = AssemblyCatalog()
    assembly = catalog.apply(state, catalog.default_selection())
    layout = apply_physical_layout_to_state(state)
    view = PhysicalLayoutView()
    qtbot.addWidget(view)

    view.display_result(SimpleNamespace(assembly=assembly, layout=layout))

    assert len(view._detector_chamber_items) == 2
    assert all(
        "absolute dimensions" in item.toolTip()
        and "7.25 mm" in item.toolTip()
        and "mechanical-only" in item.toolTip()
        for item in view._detector_chamber_items
    )
    assert (
        view._record_by_key[POST_PROJECTOR_DETECTOR_CHAMBER].profile
        == DETECTOR_CHAMBER_PROFILE
    )
    dpa_record = view._record_by_key[PROJECTION_CHAMBER_DPA_APERTURE]
    assert dpa_record.profile == FIXED_DIFFERENTIAL_PUMPING_APERTURE
    assert dpa_record.center_z_mm == pytest.approx(
        assembly.part("projector_lens_2").end_z_mm
    )
    assert dpa_record.bore_diameter_mm == pytest.approx(
        assembly.part(PROJECTION_CHAMBER_DPA_APERTURE).data["mechanical_bore_diameter_mm"]
    )
    assert dpa_record.optical_references_mm == (dpa_record.center_z_mm,)
    assert POST_PROJECTOR_DETECTOR_CHAMBER in view._component_label_items
    assert PROJECTION_CHAMBER_DPA_APERTURE in (
        view._component_label_items
    )
    assert PROJECTION_CHAMBER_DPA_APERTURE in set(
        view._selectable_item_keys.values()
    )
    assert PROJECTION_CHAMBER_DPA_APERTURE in APERTURE_KEYS
    assert PROJECTION_CHAMBER_DPA_APERTURE in {
        record["key"] for record in aperture_stop_records(state)
    }
    assert assembly.part("haadf").start_z_mm - assembly.part(
        "projector_lens_2"
    ).end_z_mm == pytest.approx(7.25)
