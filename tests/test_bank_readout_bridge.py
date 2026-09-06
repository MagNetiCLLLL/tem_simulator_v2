"""Advanced-bank publication and provenance; no high-accuracy calculations."""
from types import SimpleNamespace

import pyqtgraph as pg
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QDoubleSpinBox, QTabWidget

import temsim.interactive_calculation as backend
from temsim.gui import interactive_calculation as gui
from temsim.optics.column import default_state


@pytest.fixture
def page(qtbot, monkeypatch, tmp_path):
    settings = QSettings(str(tmp_path / "bank.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(gui, "QSettings", lambda: settings)
    widget = gui.InteractiveCalculationPage()
    qtbot.addWidget(widget)
    yield widget
    widget.shutdown()


def _readout(value=4.0):
    return backend.InteractiveReadout(
        {"detector:camera:z_mm": value}, {"camera": 0.25}, {"camera": 12.5},
        object(), object(), ("Independent completed bank images.",),
        state_snapshot=SimpleNamespace(marker=value),
    )


def _events(page):
    events = []
    page.readout_status_changed.connect(lambda text: events.append(("status", text)))
    page.readout_updated.connect(lambda result: events.append(("ready", result)))
    return events


def _coordinates_widget(page, value):
    control = QDoubleSpinBox(page)
    control.setRange(1, 10)
    control.setValue(value)
    page.live_widgets = {"detector:camera:z_mm": control}
    return control


def test_bank_page_has_only_physical_readout_and_publishes_without_calculation(page, monkeypatch):
    for name in ("views", "tem_plot", "tem_image", "stem_choice", "stem_plot", "stem_image", "_show_stem"):
        assert not hasattr(page, name)
    assert not page.readout_panel.findChildren(pg.PlotWidget)
    assert not page.readout_panel.findChildren(QTabWidget)
    assert page.readout_panel.isAncestorOf(page.signal_table)
    monkeypatch.setattr(backend, "calculate", lambda *a, **k: pytest.fail("Unexpected propagation"))
    for name in ("build", "read"):
        monkeypatch.setattr(page.controller, name, lambda *a, **k: pytest.fail("Unexpected bank work"))
    events = _events(page)
    result = _readout()
    page._readout_ready(result)
    assert events == [("ready", result)]
    assert page._readout is result
    assert page.signal_table.item(0, 0).text() == "camera"
    assert page.signal_table.item(0, 1).text() == "0.25"
    assert page.signal_table.item(0, 2).text() == "12.5"
    assert "Advanced bank" in page.notes.text()
    assert "Illuminating Image" in page.notes.text()
    assert "Scanning Image" in page.notes.text()
    assert page.notes.toolTip() == result.notes[0]
    assert not page.timer.isActive()


def test_changed_failed_and_cancelled_readout_retains_last_complete_images(page):
    result = _readout()
    page._readout_ready(result)
    events = _events(page)
    page._queue_read()
    page.timer.stop()
    page.show_error("Requested detector is outside retained trajectories")
    page._cancel()
    assert len(events) == 3
    assert all(kind == "status" and "Previous bank readout" in text for kind, text in events)
    assert page._readout is result
    assert page.signal_table.item(0, 2).text() == "12.5"


def test_live_tuning_updates_never_replace_or_invalidate_bank_display(page):
    result = _readout()
    page._readout_ready(result)
    events = _events(page)
    page._live_mode = True
    page._queue_read()
    page.timer.stop()
    page.display_tuning_status(SimpleNamespace(simulation=SimpleNamespace(metrics={"tuning_quality": "Preview"})))
    page.show_error("Live tuning could not update")
    assert events == []
    assert page._readout is result
    assert page.result_status.text() == "Advanced bank | readout ready"


def test_rejected_live_tuning_start_does_not_mark_independent_bank_previous(page):
    result = _readout()
    page._readout_ready(result)
    events = _events(page)
    page.start_live_tuning()
    assert events == []
    assert page._readout is result
    assert not page._live_mode
    assert page.result_status.text() == "Advanced bank | readout ready"


def test_capture_and_new_bank_mark_previous_without_erasing_readout(page):
    result = _readout()
    page._readout_ready(result)
    events = _events(page)
    page.timer.start(10000)
    page.set_source(default_state())
    assert not page.timer.isActive()
    page._bank_ready(SimpleNamespace(plan=SimpleNamespace(ranges=()), points=(), retained_bytes=0))
    page.timer.stop()
    assert len(events) == 2
    assert "captured" in events[0][1]
    assert "preparing readout" in events[1][1]
    assert page._readout is result


def test_immediate_readout_success_is_last_event_not_overwritten_by_finished(page, monkeypatch):
    result = _readout()
    _coordinates_widget(page, 4.0)
    events = _events(page)
    monkeypatch.setattr(page.controller, "read", lambda coordinates: page._readout_ready(result))
    page._read()
    page._busy(False)
    assert events[-1] == ("ready", result)
    assert events[0][0] == "status"
    assert len(events) == 2
    assert page.result_status.text() == "Advanced bank | readout ready"


def test_build_marks_previous_before_any_immediate_result(page, monkeypatch):
    previous, replacement = _readout(3.0), _readout(4.0)
    page._readout_ready(previous)
    page.source_state = object()
    monkeypatch.setattr(page, "plan", lambda: SimpleNamespace(point_count=1))
    monkeypatch.setattr(page.controller, "build", lambda *args: page._readout_ready(replacement))
    events = _events(page)
    page.start_build()
    assert events[0][0] == "status"
    assert events[-1] == ("ready", replacement)
    assert page._readout is replacement


def test_readout_finished_during_debounce_cannot_publish_superseded_controls(page):
    previous = _readout(3.0)
    page._readout_ready(previous)
    _coordinates_widget(page, 5.0)
    events = _events(page)
    page._queue_read()
    page.timer.stop()
    page._readout_ready(_readout(4.0))
    assert page._readout is previous
    assert all(kind == "status" for kind, _ in events)
    newest = _readout(5.0)
    page._readout_ready(newest)
    assert events[-1] == ("ready", newest)


def test_readout_state_matches_selected_detector_and_is_detached(monkeypatch):
    state = default_state()
    control = next(c for c in backend.available_controls(state)
                   if c.group == "detector" and c.field == "outer_width_mm")
    selected = control.current * 1.5
    plan = backend.InteractivePlan((backend.CalculationRange(control, control.current, selected),), 1024)
    result = SimpleNamespace(state_snapshot=state, wave_imaging=None, stem_scan=None)
    bank = backend.InteractiveBank(plan, state, backend.external_model_signature(state),
                                   (backend.BankPoint((), result, ()),), 0)
    monkeypatch.setattr(backend, "calculate", lambda *a, **k: pytest.fail("Unexpected propagation"))
    original = state.to_dict()
    readout = backend.read_bank(bank, {control.identity: selected})
    assert readout.state_snapshot is not state
    detector = next(d for d in readout.state_snapshot.recording_planes if d.key == control.key)
    assert detector.outer_width_mm == selected
    assert readout.coordinates[control.identity] == selected
    detector.outer_width_mm = selected * 2
    assert state.to_dict() == original
    assert next(d for d in state.recording_planes if d.key == control.key).outer_width_mm == control.current


def test_readout_preserves_legacy_positional_constructor():
    result = backend.InteractiveReadout({}, {}, {}, "wave", "stem", ("notes",))
    assert result.wave == "wave" and result.stem == "stem"
    assert result.notes == ("notes",)
    assert result.state_snapshot is None
