from pathlib import Path
from types import SimpleNamespace
import math
import tomllib

import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.component_keys import EDS_DETECTOR_SYSTEM
from temsim.detector.eds_geometry import (
    EDS_DETECTOR_DEFINITION_FIELD,
    EDSDetectorArrayGeometry,
    assess_axisymmetric_pole_centerline,
    load_eds_detector_definition,
)
from temsim.gui.diagnostic_tabs import PhysicalLayoutView
from temsim.module_manifest import validate_document
from temsim.optics.column import default_state
from temsim.paths import EDS_DETECTOR_CONFIG_ROOT


COLUMN_ROOT = (
    Path(__file__).parents[1] / "configs" / "instruments" / "column"
)
PROJECT_ROOT = Path(__file__).parents[1]
UNKNOWN_EDS_DIMENSIONS = {
    "active_area_per_segment_mm2",
    "sample_to_sensor_distance_mm",
    "mechanical_outer_diameter_mm",
    "sensor_face_width_mm",
    "sensor_face_height_mm",
    "detector_package_length_mm",
}


def _column_document(name: str = "C2.toml") -> dict:
    return tomllib.loads((COLUMN_ROOT / name).read_text(encoding="utf-8"))


def _part(document: dict, key: str) -> dict:
    return next(part for part in document["parts"] if part["key"] == key)


def test_only_one_generic_eds_detector_definition_is_installed():
    definitions = sorted(EDS_DETECTOR_CONFIG_ROOT.glob("*.toml"))

    assert [path.name for path in definitions] == ["EDS.toml"]
    definition = load_eds_detector_definition("EDS.toml")
    assert definition["eds_system_key"] == "eds"
    assert definition["segment_count"] == 6


def test_eds_definition_is_included_in_installed_project_data():
    pyproject = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    data_files = pyproject["tool"]["setuptools"]["data-files"]
    assert data_files["configs/detectors/eds"] == [
        "configs/detectors/eds/*.toml"
    ]


def test_every_column_installs_the_same_evidence_bounded_eds_array():
    paths = sorted(COLUMN_ROOT.glob("*.toml"))

    assert len(paths) == 5
    for path in paths:
        raw_document = _column_document(path.name)
        raw_detectors = [
            part
            for part in raw_document["parts"]
            if part["key"] == EDS_DETECTOR_SYSTEM
        ]
        assert len(raw_detectors) == 1
        assert raw_detectors[0][EDS_DETECTOR_DEFINITION_FIELD] == (
            "EDS.toml"
        )
        assert "eds_system_key" not in raw_detectors[0]

        document = validate_document(raw_document)
        detector = _part(document, EDS_DETECTOR_SYSTEM)
        sample = _part(document, "sample")
        geometry = EDSDetectorArrayGeometry.from_part_data(detector)

        assert detector["mechanical_only"] is True
        assert detector["axial_vacuum_context_only"] is True
        assert detector["parent_key"] == "objective_lens"
        assert detector["local_start_z_mm"] == pytest.approx(
            sample["local_center_z_mm"]
        )
        assert detector["local_center_z_mm"] == pytest.approx(
            sample["local_center_z_mm"]
        )
        assert detector["local_end_z_mm"] == pytest.approx(
            sample["local_center_z_mm"]
        )
        assert geometry.system_key == "eds"
        assert geometry.segment_count == 6
        assert geometry.azimuth_centers_deg == pytest.approx(
            (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
        )
        assert geometry.takeoff_angle_deg == pytest.approx(32.06)
        assert geometry.minimum_unshadowed_solid_angle_sr == pytest.approx(
            4.45
        )
        assert geometry.analytical_holder_solid_angle_sr == pytest.approx(
            4.04
        )
        assert geometry.windowless is True
        assert detector.keys().isdisjoint(UNKNOWN_EDS_DIMENSIONS)


def test_eds_angular_summaries_do_not_infer_a_detector_size():
    document = validate_document(_column_document())
    detector = _part(document, EDS_DETECTOR_SYSTEM)
    geometry = EDSDetectorArrayGeometry.from_part_data(detector)

    assert (
        geometry.minimum_unshadowed_solid_angle_per_segment_sr
        == pytest.approx(4.45 / 6.0)
    )
    assert (
        geometry.analytical_holder_solid_angle_per_segment_sr
        == pytest.approx(4.04 / 6.0)
    )
    assert geometry.minimum_unshadowed_fraction_of_4pi == pytest.approx(
        4.45 / (4.0 * math.pi)
    )
    assert geometry.equivalent_circular_cone_half_angle_deg() == (
        pytest.approx(28.1203, abs=1.0e-4)
    )
    assert geometry.equivalent_circular_cone_half_angle_deg(
        analytical_holder=True
    ) == pytest.approx(26.7682, abs=1.0e-4)
    assert geometry.minimum_axis_separation_deg == pytest.approx(
        50.1427, abs=1.0e-4
    )
    assert geometry.equivalent_circular_cones_overlap is True

    # A size can only produce a distance after adding explicit face-shape and
    # orientation assumptions.  A 2.25 mm-radius circular disk is a useful
    # inverse-formula check, not an installed production dimension.
    illustrative_area = math.pi * 2.25**2
    assert geometry.equivalent_circular_face_distance_mm(
        illustrative_area
    ) == pytest.approx(4.21029, abs=1.0e-5)


def test_eds_reference_ray_clears_constraint_derived_pole_profile():
    for path in sorted(COLUMN_ROOT.glob("*.toml")):
        document = validate_document(_column_document(path.name))
        detector = _part(document, EDS_DETECTOR_SYSTEM)
        objective = _part(document, "objective_lens")
        upper = _part(document, "objective_upper_pole")
        lower = _part(document, "objective_lower_pole")
        geometry = EDSDetectorArrayGeometry.from_part_data(detector)

        assert objective["s_twin_pole_gap_mm"] == pytest.approx(5.4)
        for pole in (upper, lower):
            assert pole["mechanical_outer_diameter_mm"] == pytest.approx(96.0)
            assert pole["mechanical_tip_diameter_mm"] == pytest.approx(8.0)
            assert pole["pole_nose_axial_length_mm"] == pytest.approx(27.5584)
            assert pole["pole_cone_angle_to_axis_deg"] == pytest.approx(57.94)

        assessment = assess_axisymmetric_pole_centerline(
            takeoff_angle_deg=geometry.takeoff_angle_deg,
            pole_gap_mm=objective["s_twin_pole_gap_mm"],
            pole_bore_diameter_mm=upper["mechanical_bore_diameter_mm"],
            pole_tip_diameter_mm=upper["mechanical_tip_diameter_mm"],
            pole_outer_diameter_mm=upper["mechanical_outer_diameter_mm"],
            pole_nose_axial_length_mm=upper["pole_nose_axial_length_mm"],
        )
        assert assessment.clears_requested_margin is True
        assert assessment.ray_radius_at_face_mm == pytest.approx(4.310851)
        assert assessment.minimum_radial_clearance_mm == pytest.approx(
            0.310851, abs=1.0e-5
        )
        assert assessment.maximum_tip_diameter_mm_for_margin == pytest.approx(
            8.621703, abs=1.0e-5
        )
        assert assessment.minimum_nose_axial_length_mm_for_margin == (
            pytest.approx(27.363667, abs=1.0e-5)
        )


def test_previous_objective_profile_blocked_the_same_reference_ray():
    assessment = assess_axisymmetric_pole_centerline(
        takeoff_angle_deg=32.06,
        pole_gap_mm=5.4,
        pole_bore_diameter_mm=5.76,
        pole_tip_diameter_mm=10.0,
        pole_outer_diameter_mm=96.0,
        pole_nose_axial_length_mm=0.38 * 62.0,
    )

    assert assessment.clears_requested_margin is False
    assert assessment.face_radial_clearance_mm == pytest.approx(-0.689149)
    assert assessment.shoulder_radial_clearance_mm == pytest.approx(-6.072979)


def test_manifest_rejects_an_invented_eds_active_area():
    document = _column_document()
    detector = _part(document, EDS_DETECTOR_SYSTEM)
    detector["active_area_per_segment_mm2"] = 30.0

    with pytest.raises(
        ValueError,
        match="public sources do not support these physical dimensions",
    ):
        validate_document(document)


def test_manifest_rejects_an_unsourced_eds_azimuth_phase_change():
    document = _column_document()
    detector = _part(document, EDS_DETECTOR_SYSTEM)
    detector["azimuth_centers_deg"] = [
        30.0,
        90.0,
        150.0,
        210.0,
        270.0,
        330.0,
    ]

    with pytest.raises(ValueError, match="must come only"):
        validate_document(document)


def test_eds_is_not_an_axial_vacuum_wall_or_optical_component():
    state = default_state()
    catalog = AssemblyCatalog()
    assembly = catalog.apply(state, catalog.default_selection())
    layout = apply_physical_layout_to_state(state)

    detector = assembly.part(EDS_DETECTOR_SYSTEM)
    sample = assembly.part("sample")
    assert detector.data[EDS_DETECTOR_DEFINITION_FIELD] == "EDS.toml"
    assert detector.data["eds_system_key"] == "eds"
    assert detector.center_z_mm == pytest.approx(sample.center_z_mm)
    assert detector.length_mm == pytest.approx(0.0)
    assert all(
        segment.key != EDS_DETECTOR_SYSTEM
        for segment in assembly.vacuum_bore_segments
    )
    assert EDS_DETECTOR_SYSTEM not in {
        component.key for component in layout
    }


def test_physical_layout_draws_two_eds_azimuthal_projections(qtbot):
    state = default_state()
    catalog = AssemblyCatalog()
    assembly = catalog.apply(state, catalog.default_selection())
    layout = apply_physical_layout_to_state(state)
    view = PhysicalLayoutView()
    qtbot.addWidget(view)

    view.display_result(SimpleNamespace(assembly=assembly, layout=layout))

    assert {
        role: len(items) for role, items in view._eds_detector_items.items()
    } == {
        "centerline": 2,
        "acceptance": 4,
        "active_face": 2,
        "housing": 2,
    }
    assert len(view._eds_detector_labels) == 1
    assert view._label_callouts["eds:detector_array"].component_key == (
        EDS_DETECTOR_SYSTEM
    )
    assert "crystal and package dimensions are unavailable" in (
        view.eds_legend.toolTip()
    )
    tooltip = view._eds_detector_items["active_face"][0].toolTip()
    assert "dimensions are not public" in tooltip
    assert "display-only clearance" in tooltip
    assert "not an electron recording plane" in tooltip
    assert "six such circular cones overlap" in tooltip
    assert "minimum radial clearance 0.3109 mm" in tooltip
    assert "does not change these intersections" in tooltip

    pole_radius = max(
        0.5 * view._record_by_key[key].outer_diameter_mm
        for key in ("objective_upper_pole", "objective_lower_pole")
    )
    minimum_solid_radius = (
        pole_radius + view.EDS_POLE_DISPLAY_CLEARANCE_MM
    )
    for face in view._eds_detector_items["active_face"]:
        _z_values, radius_values = face.getData()
        assert min(abs(float(value)) for value in radius_values) >= (
            minimum_solid_radius - 1.0e-9
        )
    for housing in view._eds_detector_items["housing"]:
        assert min(
            abs(float(point.y())) for point in housing.polygon()
        ) >= minimum_solid_radius - 1.0e-9

    view.focus_component(assembly.part(EDS_DETECTOR_SYSTEM))
    assert len(view.summary.text()) < 160
    assert "Active area, sensor distance" in view.summary.toolTip()
    assert "non-dimensional schematics" in view.summary.toolTip()
