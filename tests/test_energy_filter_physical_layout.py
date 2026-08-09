from pathlib import Path
from types import SimpleNamespace
import tomllib

import numpy as np
import pytest
from PySide6.QtCore import Qt

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.gui.diagnostic_tabs import EnergyFilterView
from temsim import module_manifest
from temsim.module_manifest import validate_document
from temsim.optics.column import default_state
from temsim.optics.energy_filter_sector import (
    multipole_housing_bank_polygons_xz_mm,
    sector_from_energy_filter,
    sector_radial_aperture_paths_xz_mm,
)


ENERGY_FILTER_MANIFEST = (
    Path(__file__).parents[1]
    / "configs"
    / "instruments"
    / "project_and_recording_system"
    / "EnergyFilter.toml"
)


class _SceneClick:
    def __init__(self, scene_position):
        self._scene_position = scene_position
        self.accepted = False

    @staticmethod
    def button():
        return Qt.MouseButton.LeftButton

    @staticmethod
    def double():
        return False

    def scenePos(self):
        return self._scene_position

    def accept(self):
        self.accepted = True


def _energy_filter_state():
    state = default_state()
    assembly = AssemblyCatalog().apply(
        state,
        AssemblySelection("FEG", "C3", "Energy Filter"),
    )
    return state, assembly


def test_m12_mechanical_envelopes_are_toml_owned_and_separate_from_field():
    state, assembly = _energy_filter_state()
    manifest = assembly.part("energy_filter").data
    energy_filter = state.energy_filter

    assert energy_filter.m12_housing_geometry_status == (
        "provisional_derived_envelope"
    )
    assert "not_manufacturer_dimension" in (
        energy_filter.m12_housing_geometry_source
    )
    assert len(energy_filter.multipoles) == 10
    for element in energy_filter.multipoles:
        manifest = assembly.part(element.key).data
        assert element.housing_length_m * 1.0e3 == pytest.approx(
            manifest["housing_length_mm"]
        )
        assert element.length_m * 1.0e3 == pytest.approx(
            manifest["magnetic_support_length_mm"]
        )
        assert element.bore_radius_m * 1.0e3 == pytest.approx(
            manifest["mechanical_bore_radius_mm"]
        )
        assert element.outer_radius_m * 1.0e3 == pytest.approx(
            manifest["mechanical_outer_radius_mm"]
        )
        assert element.housing_length_m >= element.length_m
        assert element._manifest_definition_id == assembly.part(
            element.key
        ).definition_id
        assert element._individual_pole_assignment_status == "not_public"


def test_m12_planar_housing_uses_oriented_length_outer_radius_and_bore():
    state, _assembly = _energy_filter_state()
    for element in (state.energy_filter.multipoles[0],
                    state.energy_filter.multipoles[3]):
        banks = multipole_housing_bank_polygons_xz_mm(element)
        points = np.concatenate(banks)
        centre_xz_mm = element.frame.origin_m[[0, 2]] * 1.0e3
        tangent_xz = element.frame.rotation_local_to_global[(0, 2), 2]
        transverse_xz = element.frame.rotation_local_to_global[(0, 2), 0]
        along_mm = (points - centre_xz_mm) @ tangent_xz
        transverse_mm = (points - centre_xz_mm) @ transverse_xz

        assert np.ptp(along_mm) == pytest.approx(
            element.housing_length_m * 1.0e3
        )
        assert np.max(np.abs(transverse_mm)) == pytest.approx(
            element.outer_radius_m * 1.0e3
        )
        assert np.min(np.abs(transverse_mm)) == pytest.approx(
            element.bore_radius_m * 1.0e3
        )


def test_sector_planar_edges_use_radial_aperture_not_out_of_plane_pole_gap():
    state, _assembly = _energy_filter_state()
    energy_filter = state.energy_filter
    sector = sector_from_energy_filter(energy_filter)
    paths = sector_radial_aperture_paths_xz_mm(sector)
    centre_xz_mm = sector.centre_m[[0, 2]] * 1.0e3
    radii_mm = sorted(
        float(np.linalg.norm(path[0] - centre_xz_mm))
        for path in paths
    )

    assert radii_mm == pytest.approx([
        energy_filter.prism_radius_mm
        - energy_filter.sector_radial_aperture_mm,
        energy_filter.prism_radius_mm
        + energy_filter.sector_radial_aperture_mm,
    ])
    assert radii_mm != pytest.approx([
        energy_filter.prism_radius_mm - 0.5 * energy_filter.pole_gap_mm,
        energy_filter.prism_radius_mm + 0.5 * energy_filter.pole_gap_mm,
    ])


def test_manifest_rejects_m12_housing_shorter_than_magnetic_support():
    document = tomllib.loads(ENERGY_FILTER_MANIFEST.read_text(encoding="utf-8"))
    multipole = next(
        part for part in document["parts"]
        if part["key"] == "energy_filter_multipole_01"
    )
    multipole["housing_length_mm"] = 19.0

    with pytest.raises(ValueError, match="cannot be shorter"):
        validate_document(document)


def test_manifest_rejects_geometry_duplicated_on_branch_interface():
    document = tomllib.loads(ENERGY_FILTER_MANIFEST.read_text(encoding="utf-8"))
    interface = next(
        part for part in document["parts"] if part["key"] == "energy_filter"
    )
    interface["prism_radius_mm"] = 135.0

    with pytest.raises(ValueError, match="must not duplicate"):
        validate_document(document)


def test_energy_filter_view_draws_scaled_housing_banks(qtbot):
    state, _assembly = _energy_filter_state()
    result = SimpleNamespace(state_snapshot=state, energy_filter=None)
    view = EnergyFilterView()
    qtbot.addWidget(view)
    view.resize(1200, 800)
    view.show()

    view.display_result(result)
    qtbot.wait(20)

    assert len(view._multipole_housing_items) == 20
    assert len(view._prism_clear_aperture_items) == 2
    assert len(view._label_callouts) == 19
    assert len(view._device_body_items) == 12
    assert all(
        "TOML mechanical envelope" in item.toolTip()
        for item in view._multipole_housing_items
    )
    first_housing = view._multipole_housing_items[0]
    assert first_housing.pen().color().name() == "#c084fc"
    assert first_housing.pen().widthF() == pytest.approx(0.8)
    assert first_housing.brush().color().alpha() == 105
    assert all(
        callout.leader.opts["pen"].style() == Qt.PenStyle.DashLine
        for callout in view._label_callouts.values()
    )

    selected = []
    view.component_selected.connect(selected.append)
    slit_callout = view._label_callouts[
        "device:energy_filter_slit"
    ]
    assert slit_callout.label.isVisible()
    label_click = _SceneClick(
        slit_callout.label.sceneBoundingRect().center()
    )
    view._component_item_clicked(label_click)
    assert label_click.accepted
    assert selected[-1] == "energy_filter_slit"

    dynamic_callout = view._label_callouts[
        "device:energy_filter_dynamic_focus_electrostatic_quadrupole"
    ]
    dynamic_click = _SceneClick(
        dynamic_callout.label.sceneBoundingRect().center()
    )
    view._component_item_clicked(dynamic_click)
    assert dynamic_click.accepted
    assert selected[-1] == (
        "energy_filter_dynamic_focus_electrostatic_quadrupole"
    )

    m01_body = view._multipole_housing_items[0]
    body_click = _SceneClick(m01_body.mapToScene(
        m01_body.boundingRect().center()
    ))
    view._component_item_clicked(body_click)
    assert body_click.accepted
    assert selected[-1] == "energy_filter_multipole_01"

    view_box = view.plot.getViewBox()
    assert view_box.state["aspectLocked"] is False
    assert view_box.state["mouseEnabled"] == [True, True]
    view_box.setRange(
        xRange=(0.0, 200.0),
        yRange=(-300.0, 0.0),
        padding=0.0,
    )
    x_before = tuple(view_box.viewRange()[0])
    view.plot.setYRange(-80.0, 20.0, padding=0.0)
    assert tuple(view_box.viewRange()[0]) == pytest.approx(x_before)
    assert "adjustable non-OEM envelopes" in view.summary.text()
    assert "zoomed independently" in view.summary.text()


def test_public_iliad_topology_and_zebra_active_areas_have_unique_toml_rows():
    document = tomllib.loads(ENERGY_FILTER_MANIFEST.read_text(encoding="utf-8"))
    by_key = {part["key"]: part for part in document["parts"]}
    interface = by_key["energy_filter"]
    multipole_keys = tuple(
        f"energy_filter_multipole_{index:02d}" for index in range(1, 11)
    )

    assert interface["confirmed_large_tapered_prism_count"] == 1
    assert interface["confirmed_multipole_count"] == 10
    assert interface["multipole_numbering_status"] == (
        "simulator_m01_m10_indices_not_public_production_labels_or_exact_order"
    )
    assert "prism_radius_mm" not in interface
    assert "multipole_01_s_mm" not in interface
    assert all(key in by_key for key in multipole_keys)
    assert all(
        by_key[key]["name"].endswith("(model index)")
        for key in multipole_keys
    )
    assert all(
        by_key[key]["individual_pole_assignment_status"] == "not_public"
        for key in multipole_keys
    )
    assert by_key[
        "energy_filter_dynamic_focus_electrostatic_quadrupole"
    ]["optical_model_status"] == (
        "mechanical_layout_only_dynamic_focus_field_not_implemented"
    )

    zebra = by_key["energy_filter_zebra"]
    assert zebra["strip_count"] == 5
    assert zebra["pixels_per_strip"] == 2048
    assert zebra["strip_active_width_mm"] == pytest.approx(28.672)
    assert zebra["strip_active_height_mm"] == pytest.approx(0.800)
    assert zebra["alignment_active_width_mm"] == pytest.approx(28.672)
    assert zebra["alignment_active_height_mm"] == pytest.approx(3.584)
    assert zebra["strip_center_pitch_status"] == (
        "adjustable_unknown_not_public"
    )

    state, _assembly = _energy_filter_state()
    detector = state.energy_filter.zebra_detector
    assert detector.spectral_width_mm == pytest.approx(28.672)
    assert detector.spectral_height_mm == pytest.approx(0.800)
    assert detector.alignment_width_mm == pytest.approx(28.672)
    assert detector.alignment_height_mm == pytest.approx(3.584)


def test_saved_energy_filter_state_omits_manifest_owned_internal_geometry():
    state, _assembly = _energy_filter_state()
    saved = state.to_dict()["energy_filter"]

    assert not (
        set(module_manifest.ENERGY_FILTER_GEOMETRY_FIELDS) & saved.keys()
    )
    assert "distance_from_sector_exit_m" not in saved["energy_slit"]
    assert "clear_height_m" not in saved["energy_slit"]
    assert "housing_length_mm" not in saved["bias_tube"]
    assert "electrode_gap_mm" not in saved["fast_shutter"]
    assert "electrode_length_mm" not in saved["camera_deflector"]
    assert "spectral_clear_height_mm" not in saved["zebra_detector"]
    assert "pixels_per_strip" not in saved["zebra_detector"]
