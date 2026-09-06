from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QPointF

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.gui.diagnostic_tabs import EnergyFilterView, MagneticFieldView
from temsim.gui.main_window import MainWindow
from temsim.optics.column import default_state


def _energy_filter_state():
    state = default_state()
    AssemblyCatalog().apply(
        state,
        AssemblySelection("FEG", "C3", "Energy Filter"),
    )
    return state


def test_energy_filter_view_reads_cached_spectrum_and_eftem_image(qtbot):
    state = _energy_filter_state()
    forward = SimpleNamespace(
        energy_loss_ev=np.asarray((0.0, 1.0, 2.0)),
        detected_expected_counts=np.asarray((5.0, 11.0, 3.0)),
        detected_sampled_counts=None,
    )
    branch = SimpleNamespace(
        paths_u_mm=[],
        paths_v_mm=[],
        colours=[],
        status="cached forward chain",
        eels_forward=forward,
        eftem_image=None,
    )
    result = SimpleNamespace(state_snapshot=state, energy_filter=branch)
    view = EnergyFilterView()
    qtbot.addWidget(view)
    view.resize(1000, 700)
    view.show()

    view.display_result(result)

    np.testing.assert_allclose(
        view.spectrum_curve.getData()[1], (5.0, 11.0, 3.0)
    )
    scene_point = view.spectrum_plot.getViewBox().mapViewToScene(
        QPointF(1.0, 11.0)
    )
    view._spectrum_mouse_moved(scene_point)
    assert "Energy 1 eV" in view.spectrum_status.text()
    assert "counts 11" in view.spectrum_status.text()

    state.energy_filter.operating_mode = "eftem"
    branch.eftem_image = np.arange(12, dtype=float).reshape(3, 4)
    view.display_result(result)

    assert view.eftem_image_item.isVisible()
    assert "4 × 3 px" in view.eftem_status.text()


def test_magnetic_field_map_controls_require_explicit_source_reference(qtbot):
    view = MagneticFieldView()
    qtbot.addWidget(view)
    view.field_map_lens.addItem("Objective", "objective_lens")

    assert not view.field_map_import.isEnabled()
    view.field_map_provenance.setCurrentIndex(
        view.field_map_provenance.findData("measured")
    )
    view.field_map_reference.setValue(73.0)
    view.field_map_polarity.setCurrentIndex(
        view.field_map_polarity.findData(-1)
    )

    assert view.field_map_import.isEnabled()
    assert "No unit inference" in view.field_map_import.toolTip()
    assert "sourced axial Bz" in view.digital_twin_status.text()


def test_main_window_field_map_handler_binds_and_clears_live_state(
    qtbot, tmp_path, monkeypatch
):
    window = MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    monkeypatch.setattr(window, "_runtime_parameter_changed", lambda *_: None)
    stale_calls = []
    monkeypatch.setattr(
        window, "_mark_operating_preset_stale", lambda: stale_calls.append(True)
    )
    lens_key = window.state.lenses[0].key
    provider = window._native_lens_field_provider(lens_key)
    from temsim.physics.lens_field_provider import lens_geometry_binding

    binding = lens_geometry_binding(window.state, lens_key, provider)
    field_path = tmp_path / "measured_map.npz"
    metadata = {
        "map_type": "axisymmetric_rz",
        "geometry_fingerprint": binding.geometry_fingerprint,
    }
    import json

    np.savez(
        field_path,
        r_m=np.asarray((0.0, 1.0e-3)),
        z_m=np.asarray((-1.0e-3, 1.0e-3)),
        br_t=np.zeros((2, 2)),
        bz_t=np.ones((2, 2)),
        metadata_json=np.asarray(json.dumps(metadata)),
    )

    window._import_lens_field_map(
        lens_key, str(field_path), "measured", 73.0, 1
    )

    descriptor = window.state.lens_field_map_descriptors[lens_key]
    assert descriptor["provenance_kind"] == "measured"
    assert descriptor["reference_excitation_percent"] == 73.0
    assert "map bound" in window.workspace.magnetic_field.field_map_status.text()
    assert len(stale_calls) == 1

    window._clear_lens_field_map(lens_key)

    assert lens_key not in window.state.lens_field_map_descriptors
    assert "provisional TOML Gaussian" in (
        window.workspace.magnetic_field.field_map_status.text()
    )
    assert len(stale_calls) == 2
