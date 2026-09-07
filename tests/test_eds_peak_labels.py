"""EDS annotations are display-only and separate simulated/reference lines."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPointF

from temsim.detector.eds_line_library import element_lines
from temsim.gui.eds_panel import EDSPage
from temsim.optics.column import default_state


def _line(z=14, counts=50, source="sample"):
    reference = next(line for line in element_lines(z) if line.transition == "K-L3")
    return SimpleNamespace(atomic_number=z, transition=reference.transition,
                           energy_ev=reference.energy_ev, expected_detected_counts=counts,
                           source_key=source)


def _spectrum(lines=None, sampled=False):
    axis = np.arange(5.0, 10000, 10.0)
    values = np.zeros_like(axis)
    lines = (_line(),) if lines is None else tuple(lines)
    for line in lines:
        values += line.expected_detected_counts * np.exp(-0.5 * ((axis - line.energy_ev) / 55)**2)
    return SimpleNamespace(energy_bin_centres_ev=axis, expected_counts=values,
                           sampled_counts=np.rint(values) if sampled else None, lines=lines,
                           metrics={"energy_resolution_fwhm_ev": 130.0})


@pytest.fixture
def page(qtbot):
    result = EDSPage()
    qtbot.addWidget(result)
    result.resize(1100, 600)
    result.show()
    return result


def _draw(page, qtbot, x_range=(0.1, 8.0)):
    page.spectrum_plot.setXRange(*x_range, padding=0)
    page.spectrum_plot.setYRange(0, 80, padding=0)
    qtbot.wait(40)
    page.peak_labels._draw()
    return [item for item in page.peak_labels._items if isinstance(item, pg.TextItem)]


def test_simulated_labels_deduplicate_vacancies_and_name_sources(page, qtbot):
    page._plot_spectrum(_spectrum((_line(counts=20), _line(counts=30, source="support"))))
    texts = _draw(page, qtbot)

    assert len(texts) == 1
    assert texts[0].toPlainText() == "Si Kα1"
    assert "50 counts" in texts[0].toolTip()
    assert "sample, support" in texts[0].toolTip()
    assert "130 eV FWHM" in page.peak_labels.status.text()
    assert "xraylib" in page.peak_labels.status.toolTip()


def test_hover_keeps_energy_counts_and_adds_actual_line_identity(page, qtbot):
    spectrum = _spectrum(sampled=True)
    page._plot_spectrum(spectrum)
    _draw(page, qtbot)
    position = page.spectrum_plot.getViewBox().mapViewToScene(QPointF(1.74, 40))
    page._spectrum_mouse_moved(position)

    text = page.spectrum_hover_readout.text()
    assert "Energy" in text and "Sampled counts" in text
    assert "Simulated: Si Kα1 (K-L3)" in text
    assert "Reference" not in text


def test_reference_lines_are_explicit_and_do_not_mutate_result_or_trigger_calculation(page, qtbot):
    spectrum = _spectrum()
    page._plot_spectrum(spectrum)
    state = default_state()
    page._state = state
    before = deepcopy(state.sample)
    expected = spectrum.expected_counts.copy()
    changed = []
    page.parameters_changed.connect(changed.append)
    _draw(page, qtbot)
    ranges = deepcopy(page.spectrum_plot.getViewBox().viewRange())
    page.peak_labels.element.setCurrentIndex(page.peak_labels.element.findData(26))
    qtbot.wait(40)
    texts = [item for item in page.peak_labels._items if isinstance(item, pg.TextItem)]

    assert texts and all(item.toPlainText().startswith("Ref Fe ") for item in texts)
    assert "Reference only" in page.peak_labels.status.text()
    assert "Reference: Fe Kα1" in page.peak_labels.nearby_text(6.4039)
    assert changed == []
    assert state.sample == before
    np.testing.assert_array_equal(spectrum.expected_counts, expected)
    np.testing.assert_allclose(page.spectrum_plot.getViewBox().viewRange(), ranges)


def test_vacuum_and_zero_detected_lines_do_not_get_automatic_element_peaks(page, qtbot):
    page._plot_spectrum(_spectrum((_line(counts=0),)))
    assert _draw(page, qtbot) == []
    assert page.peak_labels.nearby_text(1.74) == ""
    assert "0 simulated lines" in page.peak_labels.status.text()


def test_toggle_and_new_result_remove_old_labels(page, qtbot):
    page._plot_spectrum(_spectrum())
    assert _draw(page, qtbot)
    page.peak_labels.enabled.setChecked(False)
    qtbot.wait(40)
    assert not page.peak_labels._items
    assert page.peak_labels.nearby_text(1.74) == ""
    page.peak_labels.enabled.setChecked(True)
    page._plot_spectrum(_spectrum((_line(z=26),)))
    texts = _draw(page, qtbot)
    assert all("Si" not in item.toPlainText() for item in texts)
    assert any("Fe" in item.toPlainText() for item in texts)
    page._reset_spectrum_hover()
    assert not page.peak_labels._items
    assert not page.peak_labels._candidates


def test_annotation_count_is_bounded_and_view_range_is_respected(page, qtbot):
    lines = [_line(z) for z in range(10, 40)]
    page._plot_spectrum(_spectrum(lines))
    texts = _draw(page, qtbot, x_range=(0.1, 8.0))
    assert 0 < len(texts) <= 16
    assert all(0.1 <= item.pos().x() <= 8.0 for item in texts)
    texts = _draw(page, qtbot, x_range=(1.65, 1.8))
    assert len(texts) == 1
    assert "Si" in texts[0].toPlainText()


def test_edge_labels_and_empty_reference_library_are_explicit(page, qtbot):
    page._plot_spectrum(_spectrum())
    texts = _draw(page, qtbot, x_range=(1.739, 8.0))
    assert len(texts) == 1
    assert texts[0].anchor.x() == 0.0
    page.peak_labels.element.setCurrentIndex(page.peak_labels.element.findData(1))
    qtbot.wait(40)
    assert not page.peak_labels._items
    assert "Reference only | No tabulated lines" == page.peak_labels.status.text()


def test_changing_annotation_scope_clears_previous_hover_identity(page, qtbot):
    page._plot_spectrum(_spectrum())
    _draw(page, qtbot)
    position = page.spectrum_plot.getViewBox().mapViewToScene(QPointF(1.74, 40))
    page._spectrum_mouse_moved(position)
    assert "Simulated: Si" in page.spectrum_hover_readout.text()
    page.peak_labels.element.setCurrentIndex(page.peak_labels.element.findData(26))
    assert "Simulated: Si" not in page.spectrum_hover_readout.text()
    page.peak_labels.element.setCurrentIndex(0)
    page._spectrum_mouse_moved(position)
    assert "Simulated: Si" in page.spectrum_hover_readout.text()
    page.peak_labels.enabled.setChecked(False)
    assert "Simulated: Si" not in page.spectrum_hover_readout.text()


def test_missing_response_metadata_is_not_claimed_ideal(page):
    spectrum = _spectrum(())
    spectrum.metrics = {}
    page._plot_spectrum(spectrum)
    assert "Resolution unavailable" in page.peak_labels.status.text()
    assert "ideal response" not in page.peak_labels.status.text()
    spectrum.metrics = {"energy_resolution_fwhm_ev": 0.0}
    page._plot_spectrum(spectrum)
    assert "ideal response" in page.peak_labels.status.text()
