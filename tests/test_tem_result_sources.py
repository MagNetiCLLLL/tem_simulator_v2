"""Unified TEM presentation uses stored products, never another calculation."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.gui.visualization import WaveImagingView


def wave(value):
    axes = np.array([-1., 0., 1.])
    return SimpleNamespace(
        image_intensity=np.arange(9, dtype=float).reshape(3, 3) + value,
        diffraction_intensity=np.arange(9, dtype=float).reshape(3, 3) + value + 10.,
        camera_x_mm=axes, camera_y_mm=axes, x_angstrom=axes, y_angstrom=axes,
        spatial_frequency_inv_angstrom=axes,
        preset_key="reference", preset_name=f"Specimen {value}",
        metrics={"surviving_rays": 3},
    )


def readout(result):
    return SimpleNamespace(
        wave=result, stem=None, coordinates={"lens:objective:percent": 65.},
        notes=("TEM: completed recording reused.",),
        state_snapshot=SimpleNamespace(sample=SimpleNamespace(thickness=42.)),
    )


@pytest.fixture
def view(qtbot, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Changing display source must not request a calculation")
    monkeypatch.setattr("temsim.simulation_pipeline.calculate", forbidden)
    monkeypatch.setattr("temsim.physics.wave_imaging.reproject_wave_image", forbidden)
    from temsim.gui.calculation_controller import CalculationController
    monkeypatch.setattr(CalculationController, "submit", forbidden)
    monkeypatch.setattr(CalculationController, "submit_background", forbidden)
    widget = WaveImagingView()
    qtbot.addWidget(widget)
    return widget


def select(view, source):
    view.image_source.setCurrentIndex(view.image_source.findData(source))


def test_switching_reuses_main_and_bank_products_without_editing_parameters(view):
    main, bank = wave(1), wave(2)
    current_state = SimpleNamespace(sample=SimpleNamespace(thickness=10.))
    before = deepcopy(vars(current_state.sample))
    view.display_result(main, current_state, "High accuracy")
    view.set_bank_readout(readout(bank))
    assert view.image_source.currentData() == "current"
    np.testing.assert_array_equal(view.image.image, main.image_intensity.T)
    select(view, "bank")
    np.testing.assert_array_equal(view.image.image, bank.image_intensity.T)
    np.testing.assert_array_equal(view.diffraction.image, bank.diffraction_intensity.T)
    assert "captured settings" in view.image_source_status.text()
    assert "65" in view.image_source_status.toolTip()
    select(view, "current")
    np.testing.assert_array_equal(view.image.image, main.image_intensity.T)
    assert vars(current_state.sample) == before
    assert view._current_wave_presentation[0] is main


def test_new_main_result_is_retained_without_overwriting_selected_bank(view):
    main, bank, newest = wave(1), wave(2), wave(3)
    view.display_result(main)
    view.set_bank_readout(readout(bank))
    select(view, "bank")
    view.display_result(newest)
    np.testing.assert_array_equal(view.image.image, bank.image_intensity.T)
    select(view, "current")
    np.testing.assert_array_equal(view.image.image, newest.image_intensity.T)


def test_pending_bank_and_stale_current_are_independent(view):
    main, bank = wave(1), wave(2)
    view.display_result(main)
    view.set_bank_readout(readout(bank))
    select(view, "bank")
    view.mark_result_stale()
    assert "inputs changed" not in view.image_source_status.text()
    view.mark_bank_readout_pending("Readout failed; previous point retained")
    assert "previous readout" in view.image_source_status.text()
    assert "failed" in view.image_source_status.toolTip()
    np.testing.assert_array_equal(view.image.image, bank.image_intensity.T)
    select(view, "current")
    assert "inputs changed" in view.image_source_status.text()
    np.testing.assert_array_equal(view.image.image, main.image_intensity.T)
    view.set_bank_readout(readout(wave(4)))
    assert "inputs changed" in view.image_source_status.text()
    select(view, "bank")
    assert "previous readout" not in view.image_source_status.text()


def test_missing_bank_product_clears_bank_only(view):
    main = wave(1)
    view.display_result(main)
    select(view, "bank")
    assert view.image.image is None and view.diffraction.image is None
    assert "No Advanced bank" in view.image_source_status.text()
    view.set_bank_readout(readout(wave(2)))
    assert view.image.image is not None
    view.set_bank_readout(readout(None))
    assert view.image.image is None and view.diffraction.image is None
    assert "No TEM image" in view.summary.text()
    select(view, "current")
    np.testing.assert_array_equal(view.image.image, main.image_intensity.T)


def test_returning_to_source_preserves_its_user_plot_ranges(view, qtbot):
    view.resize(1000, 700)
    view.show()
    view.display_result(wave(1))
    qtbot.wait(20)
    view.image.getView().setRange(xRange=(-.6, .6), yRange=(-.5, .5), padding=0)
    view.diffraction.getView().setRange(xRange=(-.7, .7), yRange=(-.4, .4), padding=0)
    before = [plot.getView().viewRange() for plot in (view.image, view.diffraction)]
    view.set_bank_readout(readout(wave(2)))
    select(view, "bank")
    select(view, "current")
    for plot, expected in zip((view.image, view.diffraction), before):
        np.testing.assert_allclose(plot.getView().viewRange(), expected, rtol=0, atol=1e-10)

