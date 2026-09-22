"""Current-pixel electron counts are separate from cached-bank signals."""
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QTableWidgetItem

from temsim.detector.particle_readout import ParticleDetectorReadout
from temsim.gui import interactive_calculation as gui


@pytest.fixture
def pixel_page(qtbot, monkeypatch, tmp_path):
    settings = QSettings(str(tmp_path / "pixel.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(gui, "QSettings", lambda: settings)
    page = gui.InteractiveCalculationPage()
    qtbot.addWidget(page)
    yield page
    page.shutdown()


def test_current_pixel_counts_preserve_true_zero_and_do_not_replace_bank_readout(pixel_page):
    page = pixel_page
    page.signal_table.setRowCount(1)
    page.signal_table.setItem(0, 0, QTableWidgetItem("Saved bank detector"))
    page.result_status.setText("Advanced bank | readout ready")
    page.set_particle_signals((
        ParticleDetectorReadout("camera", "Pixelated camera", "AVAILABLE",
                                simulated_electrons=3.75, electrons_per_second=1.25e8),
        ParticleDetectorReadout("bf", "BF Detector", "AVAILABLE",
                                simulated_electrons=0., electrons_per_second=0.),
    ))
    table = page.particle_signal_table
    assert table.rowCount() == 2
    assert [table.item(0, col).text() for col in range(4)] == [
        "Pixelated camera", "3.75", "125000000", "Available"]
    assert table.item(1, 1).text() == table.item(1, 2).text() == "0"
    assert page.signal_table.item(0, 0).text() == "Saved bank detector"
    assert page.result_status.text() == "Advanced bank | readout ready"
    page.set_particle_signals(())
    assert table.rowCount() == 0
    assert "No completed" in page.particle_signal_status.text()
    assert page.signal_table.rowCount() == 1


@pytest.mark.parametrize("status", ["NOT_INSERTED", "READOUT_DISABLED", "NOT_REACHED", "NOT_CALCULATED"])
def test_unavailable_detector_does_not_masquerade_as_zero(pixel_page, status):
    pixel_page.set_particle_signals((ParticleDetectorReadout(
        "camera", "Pixelated camera", status, simulated_electrons=0., electrons_per_second=0.),))
    table = pixel_page.particle_signal_table
    assert table.item(0, 1).text() == table.item(0, 2).text() == "—"
    assert table.item(0, 3).text() == status.replace("_", " ").capitalize()


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_missing_numeric_readout_remains_unavailable(pixel_page, value):
    pixel_page.set_particle_signals((ParticleDetectorReadout(
        "camera", "Pixelated camera", "AVAILABLE", simulated_electrons=value, electrons_per_second=value),))
    table = pixel_page.particle_signal_table
    assert table.item(0, 1).text() == table.item(0, 2).text() == "—"


def test_particle_status_distinguishes_material_transport_from_legacy_optical_preview(pixel_page):
    pixel_page._live_mode = True
    result = SimpleNamespace(simulation=SimpleNamespace(metrics={"particle_tuning": True, "tuning_quality": "Preview"}))
    pixel_page.display_tuning_status(result)
    assert "Particle transport and detector signals updated" in pixel_page.live_status.text()
    assert "not calculated" not in pixel_page.live_status.text()
    result.simulation.metrics.pop("particle_tuning")
    pixel_page.display_tuning_status(result)
    assert "sample signals not calculated" in pixel_page.live_status.text()
