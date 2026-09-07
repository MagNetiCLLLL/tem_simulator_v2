import json
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QDoubleSpinBox, QLabel

from temsim.assembly_catalog import AssemblyCatalog
from temsim.calculation_cache import calculation_signatures
from temsim.gui.sample_panel import SamplePage
from temsim.gui.eds_panel import EDSPage
from temsim.optics.column import default_state
from temsim.specimen.atomistic import atomistic_capability


def test_sample_page_binds_modes_envelope_and_safe_offscreen_view(qtbot):
    state = default_state()
    page = SamplePage()
    qtbot.addWidget(page)

    page.set_state(state)
    assert page.envelope_shape.currentData() == "disk"
    assert page.scalar_controls["size_x_nm"].value() == pytest.approx(
        3_000_000.0
    )
    assert page.scalar_controls["size_y_nm"].isHidden()
    page.mode.setCurrentIndex(page.mode.findData("virtual"))
    page.scalar_controls["size_x_nm"].setValue(250.0)
    page.inserted.setChecked(False)

    assert state.sample.specimen_mode == "virtual"
    assert state.sample.size_x_nm == 250.0
    assert state.sample.size_y_nm == 250.0
    assert state.sample.inserted is False
    assert page.real_group.isHidden() is True
    assert page.virtual_group.isHidden() is False
    assert page.scene.opengl_available is False
    assert "offscreen" in page.scene.opengl_detail


def test_sample_page_applies_extensible_interaction_table(qtbot):
    state = default_state()
    state.sample.specimen_mode = "virtual"
    page = SamplePage()
    qtbot.addWidget(page)
    page.set_state(state)
    page.interaction_table.setRowCount(0)
    page._append_table_row(
        page.interaction_table,
        (
            True,
            "absorbed",
            "absorption",
            0.2,
            json.dumps({}),
        ),
    )

    page._apply_interactions()

    assert state.sample.virtual_interactions == [
        {
            "enabled": True,
            "name": "absorbed",
            "kind": "absorption",
            "probability": 0.2,
        }
    ]


def test_sample_page_owns_shared_wave_and_virtual_interaction_controls(qtbot):
    state = default_state()
    state.illumination_mode = "TEM"
    page = SamplePage()
    qtbot.addWidget(page)
    page.set_state(state)
    assert "total λ" in page.inelastic_summary.text()
    assert "Silicon" in page.inelastic_summary.text()

    preset_index = page.preset.findData("si_110")
    assert preset_index >= 0
    page.preset.setCurrentIndex(preset_index)
    page.tem_wave_enabled.setChecked(True)
    page.wave_grid.setValue(64)
    page.wave_scalar_controls["wave_slice_thickness_angstrom"].setValue(1.5)
    page.frozen_enabled.setChecked(True)
    page.frozen_configurations.setValue(6)
    page.frozen_sigma.setValue(0.075)
    page.frozen_seed.setValue(42)
    page.inelastic_scalar_controls[
        "real_plasmon_mean_free_path_nm"
    ].setValue(175.0)
    page.inelastic_scalar_controls[
        "real_absorption_mean_free_path_nm"
    ].setValue(900.0)

    assert state.sample.specimen_preset_key == "si_110"
    assert state.sample.wave_enabled is True
    assert state.sample.wave_grid_pixels == 64
    assert state.sample.wave_slice_thickness_angstrom == pytest.approx(1.5)
    assert state.sample.wave_frozen_phonon_enabled is True
    assert state.sample.wave_frozen_phonon_configurations == 6
    assert state.sample.wave_frozen_phonon_sigma_angstrom == pytest.approx(
        0.075
    )
    assert state.sample.wave_frozen_phonon_seed == 42
    assert state.sample.real_inelastic_enabled is True
    assert state.sample.real_plasmon_mean_free_path_nm == pytest.approx(175.0)
    assert state.sample.real_absorption_mean_free_path_nm == pytest.approx(900.0)
    assert page.findChild(QDoubleSpinBox, "sampleGVector") is None

    page.mode.setCurrentIndex(page.mode.findData("virtual"))
    page.diffraction_enabled.setChecked(False)

    assert state.sample.specimen_mode == "virtual"
    assert state.sample.diffraction_enabled is False
    assert page.real_group.isHidden() is True
    assert page.virtual_group.isHidden() is False


def test_mode_is_the_only_structure_source_selector(qtbot):
    state = default_state()
    state.sample.specimen_preset_key = "si_110"
    state.sample.cif_path = "remembered-real-sample.cif"
    page = SamplePage()
    qtbot.addWidget(page)
    page.set_state(state)

    assert not hasattr(page, "structure_source")
    assert page.mode.currentData() == "virtual"
    assert page.virtual_group.isAncestorOf(page.preset)
    assert page.preset.isEnabled()
    assert page.virtual_group.isHidden() is False
    assert page.real_group.isHidden() is True
    assert page.preset.currentData() == "si_110"

    page.mode.setCurrentIndex(
        page.mode.findData("atomic")
    )
    page.cif_path.setText("ideal-sample.cif")
    page._cif_edited()

    assert state.sample.specimen_mode == "atomic"
    assert state.sample.cif_path == "ideal-sample.cif"
    assert state.sample.specimen_preset_key == "si_110"
    assert page.real_group.isHidden() is False
    assert page.virtual_group.isHidden() is True
    assert page.real_group.isAncestorOf(page.cif_path)

    page.mode.setCurrentIndex(
        page.mode.findData("virtual")
    )

    assert state.sample.specimen_mode == "virtual"
    assert state.sample.specimen_preset_key == "si_110"
    assert state.sample.cif_path == "ideal-sample.cif"


def test_sample_page_gates_tem_wave_control_by_illumination_mode(qtbot):
    state = default_state()
    page = SamplePage()
    qtbot.addWidget(page)

    state.illumination_mode = "STEM"
    page.set_state(state)
    assert not page.tem_wave_enabled.isEnabled()

    state.illumination_mode = "TEM"
    page.set_state(state)
    assert page.tem_wave_enabled.isEnabled()

    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = ""
    page.set_state(state)
    assert not page.tem_wave_enabled.isEnabled()

    state.sample.cif_path = "real-sample.cif"
    page.set_state(state)
    assert page.tem_wave_enabled.isEnabled()


def test_dedicated_eds_page_uses_calculated_sample_plane_rays(qtbot):
    state = default_state()
    page = EDSPage()
    qtbot.addWidget(page)
    page.set_state(state)

    material_index = page.eds_support_material.findData("copper")
    page.eds_support_material.setCurrentIndex(material_index)
    page.eds_scalar_controls["eds_support_offset_x_um"].setValue(-60.0)

    assert state.sample.eds_support_material_key == "copper"
    assert state.sample.eds_support_offset_x_um == pytest.approx(-60.0)
    assert page._eds_result is None
    assert "run High accuracy" in page.eds_summary.text()
    assert "Ultra" not in page.eds_group.title()

    catalog = AssemblyCatalog()
    assembly = catalog.apply(state, catalog.default_selection())
    ray_count = 2
    calculation_state = type(state).from_dict(state.to_dict())
    calculation_state.electron_gun.emitter.ray_count = ray_count
    state.electron_gun.emitter.ray_count = ray_count + 3
    result_signatures = calculation_signatures(calculation_state)
    simulation = SimpleNamespace(
        incident=SimpleNamespace(
            alive=np.ones(ray_count, dtype=bool),
            x=np.zeros((1, ray_count)),
            y=np.zeros((1, ray_count)),
            tx=np.asarray(((-1.0e-3, 1.0e-3),)),
            ty=np.asarray(((0.5e-3, -0.5e-3),)),
            energy_offset_ev=np.asarray((-0.1, 0.1)),
            ray_weight=np.asarray((0.4, 0.6)),
        )
    )
    page.display_result(
        SimpleNamespace(
            assembly=assembly,
            simulation=simulation,
            state_snapshot=calculation_state,
            signatures=result_signatures,
        )
    )
    page.eds_acquire.click()

    assert page._eds_result is not None
    assert page._eds_result.metrics["system_name"] == "EDS"
    assert page._eds_result.metrics["elastic_trajectory_generation"] is True
    assert page._eds_result.metrics["trajectory_count"] == ray_count
    assert page._eds_result.metrics["reaching_sample_ray_count"] == ray_count
    assert page._elastic_result is page._eds_result.elastic_transport
    assert (
        page._specimen_interactions.metrics["dependency_signatures"]["eds"]
        == result_signatures["eds"]
    )
    assert result_signatures["eds"] != calculation_signatures(state)["eds"]
    assert page.spectrum_plot.listDataItems()
    assert not hasattr(page, "eds_trajectory_plot")
    assert not hasattr(page, "eds_trajectory_yz_plot")
    # Histories remain available to the shared sample scene, not another plot.
    assert page._elastic_result.trajectories
    assert {
        line.source_key for line in page._eds_result.lines
    } >= {"sample", "support:bar"}
    assert "EDS point |" in page.eds_summary.text()
    assert "Ultra" not in page.eds_summary.text()


def test_eds_page_rejects_enrichment_from_a_different_calculation_identity(
    qtbot,
):
    page = EDSPage()
    qtbot.addWidget(page)
    retained = object()
    page._result = SimpleNamespace(
        signatures={"eds": "high-15k"},
        specimen_interactions=retained,
    )
    errors = []
    page.error.connect(errors.append)
    mismatched = SimpleNamespace(
        metrics={"dependency_signatures": {"eds": "live-1k"}}
    )

    stored = page._store_specimen_interactions(mismatched)

    assert stored is False
    assert page._result.specimen_interactions is retained
    assert page._specimen_interactions is None
    assert errors and "does not match" in errors[-1]


def test_eds_spectrum_hover_reports_nearest_energy_bin_and_counts(qtbot):
    page = EDSPage()
    qtbot.addWidget(page)
    page.resize(1000, 700)
    page.show()
    spectrum = SimpleNamespace(
        energy_bin_centres_ev=np.asarray((1000.0, 2000.0, 3000.0)),
        expected_counts=np.asarray((10.0, 25.5, 8.0)),
        sampled_counts=None,
    )

    page._plot_spectrum(spectrum)
    qtbot.wait(20)
    scene_position = page.spectrum_plot.getViewBox().mapViewToScene(
        QPointF(2.1, 25.5)
    )
    page._spectrum_mouse_moved(scene_position)

    assert page.spectrum_hover_readout.text() == (
        "Energy 2 keV | Expected counts 25.5"
    )
    assert page._spectrum_cursor.isVisible()
    assert page._spectrum_cursor.value() == pytest.approx(2.0)


def test_sample_page_does_not_show_eds_controls_or_trajectory_plot(qtbot):
    page = SamplePage()
    qtbot.addWidget(page)
    page.set_state(default_state())

    assert page.eds_group.isHidden()
    assert page.eds_trajectory_plot.isHidden()


def test_sample_page_contains_only_structure_and_labels_ball_elements(
    qtbot,
    tmp_path,
):
    if not atomistic_capability().available:
        pytest.skip("Atomistic CIF backend unavailable")
    from ase.build import bulk
    from ase.io import write

    path = tmp_path / "nacl.cif"
    write(path, bulk("NaCl", "rocksalt", a=5.64))
    state = default_state()
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = str(path)
    state.sample.size_x_nm = 1.2
    state.sample.size_y_nm = 1.2
    state.sample.thickness_nm = 1.2
    page = SamplePage()
    qtbot.addWidget(page)

    page.set_state(state)

    assert not hasattr(page, "image_panels")
    assert page._snapshot is None  # Hidden pages defer expensive atom rendering.
    page.show()
    qtbot.waitUntil(lambda: page._snapshot is not None)
    assert page._snapshot.atomic_numbers.size > 2
    assert page._snapshot.atom_bond_pairs.shape[0] > 0
    legend_text = "\n".join(
        label.text() for label in page.element_legend.findChildren(QLabel)
    )
    assert "Na — Sodium" in legend_text
    assert "Cl — Chlorine" in legend_text
