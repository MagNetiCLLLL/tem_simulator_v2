from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QPointF

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.gui.diagnostic_tabs import MagneticFieldView
from temsim.gui.energy_filter_outputs import EnergyFilterOutputsView
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
    state.energy_filter._last_slit_metrics = SimpleNamespace(
        dispersion_um_per_ev=1.25, non_isochromaticity_ev_rms=0.125
    )
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
    view = EnergyFilterOutputsView()
    qtbot.addWidget(view)
    view.resize(1000, 700)
    view.show()

    received = []
    view.result_displayed.connect(received.append)
    view.display_result(result)

    assert received == [result]
    assert received[0] is result
    assert not hasattr(view, "plot")
    assert not hasattr(view, "fit_all")
    assert view.output_tabs.objectName() == "energyFilterOutputTabs"
    assert [view.output_tabs.tabText(index) for index in range(
        view.output_tabs.count()
    )] == ["EELS spectrum", "EFTEM image"]
    assert "dispersion 1.25 um/eV" in view.summary.text()
    assert "non-iso RMS 0.125 eV" in view.summary.text()

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
    forward.detected_sampled_counts = np.asarray((6.0, 10.0, 4.0))
    branch.eftem_image = np.arange(12, dtype=float).reshape(3, 4)
    view.display_result(result)

    np.testing.assert_allclose(
        view.spectrum_curve.getData()[1], (6.0, 10.0, 4.0)
    )
    assert "sampled total counts 20" in view.spectrum_status.text()
    assert view.eftem_image_item.isVisible()
    assert "4 × 3 px" in view.eftem_status.text()

    stale_signals = []
    view.result_stale.connect(lambda: stale_signals.append(True))
    view.mark_result_stale()

    assert stale_signals == [True]
    assert view._result is result
    assert "inputs changed" in view.summary.text()
    assert "inputs changed" in view.spectrum_status.text()
    assert "inputs changed" in view.eftem_status.text()
    np.testing.assert_allclose(view.eftem_image_item.image, branch.eftem_image)
    scene_point = view.spectrum_plot.getViewBox().mapViewToScene(
        QPointF(1.0, 10.0)
    )
    view._spectrum_mouse_moved(scene_point)
    assert "inputs changed" in view.spectrum_status.text()
    assert "counts 10" in view.spectrum_status.text()

    # Workspace invalidation can override the tooltip; new results must clear it.
    view.summary.setToolTip("Previous calculation; update the outputs.")
    view.display_result(result)
    assert "Previous calculation" not in view.summary.toolTip()
    assert "Ray Diagram" in view.summary.toolTip()
    assert "inputs changed" not in view.spectrum_status.text()


@pytest.mark.parametrize("missing", ["filter", "disabled", "branch"])
def test_energy_filter_outputs_clear_cached_readouts_when_unavailable(qtbot, missing):
    state = _energy_filter_state()
    state.energy_filter.operating_mode = "eftem"
    result = SimpleNamespace(
        state_snapshot=state,
        energy_filter=SimpleNamespace(
            eels_forward=SimpleNamespace(
                energy_loss_ev=np.asarray((0.0, 1.0)),
                detected_expected_counts=np.asarray((1.0, 2.0)),
            ),
            eftem_image=np.ones((2, 2)),
        ),
    )
    view = EnergyFilterOutputsView()
    qtbot.addWidget(view)
    view.display_result(result)
    view.spectrum_cursor.show()
    view.spectrum_point.show()
    received = []
    view.result_displayed.connect(received.append)

    if missing == "filter":
        state.energy_filter = None
    elif missing == "disabled":
        state.energy_filter.enabled = False
    else:
        result.energy_filter = None
    view.display_result(result)

    assert received[0] is result
    assert view._spectrum_energy_ev.size == 0
    assert view._spectrum_counts.size == 0
    assert not view.spectrum_cursor.isVisible()
    assert not view.spectrum_point.isVisible()
    assert not view.eftem_image_item.isVisible()
    assert view.eftem_image_item.image is None
    assert "No cached" in view.spectrum_status.text()
    if missing == "branch":
        assert "No Energy Filter result" in view.summary.text()
    else:
        assert "not installed or enabled" in view.summary.text()


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
