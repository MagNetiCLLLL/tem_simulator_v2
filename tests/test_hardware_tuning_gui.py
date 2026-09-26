"""Manual hardware controls over real state; no transport or alignment solves."""
from copy import deepcopy

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QLabel, QGroupBox, QLineEdit, QCheckBox

from temsim.gui.hardware_tuning_panel import HardwareTuningPanel
from temsim.hardware_tuning import TUNING_TASKS
from temsim.optics.column import default_state
from temsim.runtime_parameters import runtime_targets


@pytest.fixture
def panel(qtbot):
    view = HardwareTuningPanel()
    qtbot.addWidget(view)
    view.set_state(default_state())
    view.resize(760, 540)
    view.show()
    return view


def _edit(view, identity, text):
    editor = view.editors[identity]
    assert isinstance(editor, QLineEdit)
    editor.setText(text)
    editor.editingFinished.emit()


def _assert_control_geometry(view):
    """Test actual settled rectangles, including controls below the viewport."""
    groups = [box for box in view.findChildren(QGroupBox) if box.isVisibleTo(view)]
    assert groups
    previous_bottom = -1
    for box in groups:
        assert box.height() >= box.minimumSizeHint().height(), box.title()
        assert box.geometry().top() > previous_bottom, box.title()
        previous_bottom = box.geometry().bottom()
    for identity, editor in view.editors.items():
        box = editor.parentWidget()
        assert isinstance(box, QGroupBox)
        assert editor.isVisibleTo(view), identity
        assert editor.height() >= editor.minimumSizeHint().height(), identity
        assert box.contentsRect().contains(editor.geometry()), identity
        label = box.layout().labelForField(editor)
        assert isinstance(label, QLabel), identity
        # A zero-width label is still "visible" to Qt and its editor can work,
        # so signals and scrollbar tests alone miss this regression.
        longest_word = max(label.text().split(), key=len)
        required_width = label.fontMetrics().horizontalAdvance(longest_word)
        assert label.contentsRect().width() >= required_width, (identity, label.text())
        assert label.height() >= label.heightForWidth(label.width()), (identity, label.text())
        assert box.contentsRect().contains(label.geometry()), identity
        assert not label.geometry().intersects(editor.geometry()), identity


def test_selecting_tasks_is_read_only_and_uses_in_place_hardware_fields(panel):
    before = deepcopy(panel._state.to_dict())
    changed = []
    panel.runtime_changed.connect(changed.append)
    for task in TUNING_TASKS:
        panel.select_task(task.key)
        assert panel.selected_key == task.key
        assert panel.title.text() == task.label
        assert panel.description.text() == task.description
        targets = runtime_targets(panel._state)
        for (key, field), editor in panel.editors.items():
            assert key in targets
            value = getattr(targets[key].obj, field)
            if isinstance(editor, QLineEdit):
                assert float(editor.text()) == pytest.approx(value)
            elif isinstance(editor, QCheckBox):
                assert editor.isChecked() == value
    assert panel._state.to_dict() == before
    assert changed == []


def test_shared_shift_and_tilt_edit_one_physical_pair_once(panel):
    panel.select_task("beam_shift")
    identity = ("beam_deflector", "upper_x_mrad")
    target = runtime_targets(panel._state)[identity[0]].obj
    changed = []
    panel.runtime_changed.connect(changed.append)
    _edit(panel, identity, "0.125")
    assert target.upper_x_mrad == 0.125
    assert changed == ["beam_deflector.upper_x_mrad"]
    panel.select_task("beam_tilt")
    assert panel.editors[identity].text() == "0.125"
    _edit(panel, identity, "0.125")
    assert changed == ["beam_deflector.upper_x_mrad"]
    _edit(panel, identity, "-0.25")
    panel.select_task("beam_shift")
    assert target.upper_x_mrad == -0.25
    assert panel.editors[identity].text() == "-0.25"
    assert len(changed) == 2


@pytest.mark.parametrize("invalid", ["nan", "inf", "not a number", "-1"])
def test_invalid_lens_drive_preserves_state_and_restores_current_value(panel, invalid):
    panel.select_task("condenser_current")
    identity = ("condenser_lens_1", "percent")
    target = runtime_targets(panel._state)[identity[0]].obj
    before = target.percent
    changed = []
    panel.runtime_changed.connect(changed.append)
    _edit(panel, identity, invalid)
    assert target.percent == before
    assert float(panel.editors[identity].text()) == before
    assert changed == []
    assert panel.status.isVisible()
    assert panel.status.text()


def test_enable_toggle_updates_actual_lens_and_availability_message(panel):
    panel.select_task("condenser_current")
    identity = ("condenser_lens_1", "enabled")
    target = runtime_targets(panel._state)[identity[0]].obj
    changed = []
    panel.runtime_changed.connect(changed.append)
    panel.editors[identity].setChecked(False)
    assert not target.enabled
    assert changed == ["condenser_lens_1.enabled"]
    assert any("Disabled:" in label.text() for label in panel.findChildren(QLabel))
    panel.editors[identity].setChecked(True)
    assert target.enabled
    assert changed == ["condenser_lens_1.enabled"] * 2


@pytest.mark.parametrize("period", ["0", "-0.25"])
def test_nonpositive_wobble_period_rejects_without_mutating_component(panel, period):
    panel.select_task("beam_wobble")
    target = panel._state.ac_deflector
    before = deepcopy(vars(target))
    changed = []
    panel.runtime_changed.connect(changed.append)
    _edit(panel, ("ac_deflector", "wobble_period_s"), period)
    assert vars(target) == before
    assert changed == []
    assert "period must be positive" in panel.status.text()
    assert float(panel.editors[("ac_deflector", "wobble_period_s")].text()) == before["wobble_period_s"]


def test_wobble_and_active_raster_are_rejected_without_mutating_component(panel):
    target = panel._state.ac_deflector
    target.scan_enabled = True
    target.wobble_enabled = False
    target.validate()
    panel.select_task("beam_wobble")
    before = deepcopy(vars(target))
    changed = []
    panel.runtime_changed.connect(changed.append)
    panel.editors[("ac_deflector", "wobble_enabled")].setChecked(True)
    assert vars(target) == before
    assert changed == []
    assert "mutually exclusive" in panel.status.text()
    assert not panel.editors[("ac_deflector", "wobble_enabled")].isChecked()


@pytest.mark.parametrize("field", ["kick_x_mrad", "upper_coil_gain"])
def test_scan_drive_limit_rejection_preserves_all_state_including_derived_lower_gain(panel, field):
    target = panel._state.ac_deflector
    # A nonzero static command makes an excessive upper gain fail the physical
    # foil-drive limit after its setter has recalculated the lower gain.
    target.kick_x_mrad = 1.0
    target.validate()
    panel.select_task("scan_static")
    before = deepcopy(vars(target))
    changed = []
    panel.runtime_changed.connect(changed.append)
    _edit(panel, ("ac_deflector", field), str(target.maximum_kick_mrad * 2.0))
    assert vars(target) == before
    assert "lower_coil_gain" in before
    assert changed == []
    assert "limit" in panel.status.text()
    assert float(panel.editors[("ac_deflector", field)].text()) == before[field]


def test_valid_scan_gain_updates_derived_lower_gain_and_emits_once(panel):
    target = panel._state.ac_deflector
    panel.select_task("scan_static")
    changed = []
    panel.runtime_changed.connect(changed.append)
    _edit(panel, ("ac_deflector", "upper_coil_gain"), "0.75")
    assert target.upper_coil_gain == 0.75
    assert target.lower_coil_gain == -0.75
    assert changed == ["ac_deflector.upper_coil_gain"]
    target.validate()


def test_valid_wobble_period_and_enable_are_accepted_when_raster_is_off(panel):
    target = panel._state.ac_deflector
    target.scan_enabled = False
    target.wobble_enabled = False
    panel.select_task("beam_wobble")
    changed = []
    panel.runtime_changed.connect(changed.append)
    _edit(panel, ("ac_deflector", "wobble_period_s"), "0.25")
    assert target.wobble_period_s == 0.25
    assert changed == ["ac_deflector.wobble_period_s"]
    panel.editors[("ac_deflector", "wobble_enabled")].setChecked(True)
    assert target.wobble_enabled
    assert not target.scan_enabled
    assert changed == ["ac_deflector.wobble_period_s", "ac_deflector.wobble_enabled"]
    target.validate()


def test_state_replacement_does_not_leave_editors_bound_to_old_instrument(panel):
    panel.select_task("condenser_current")
    identity = ("condenser_lens_1", "percent")
    old_state = panel._state
    old_target = runtime_targets(old_state)[identity[0]].obj
    old_value = old_target.percent
    old_editor = panel.editors[identity]
    replacement = default_state()
    new_target = runtime_targets(replacement)[identity[0]].obj
    new_target.percent = 42.0
    panel.set_state(replacement)
    assert panel.editors[identity] is not old_editor
    assert float(panel.editors[identity].text()) == 42.0
    # A delayed commit from a replaced form must not affect either instrument.
    old_editor.setText("17")
    old_editor.editingFinished.emit()
    assert new_target.percent == 42.0
    assert old_target.percent == old_value
    _edit(panel, identity, "43.5")
    assert new_target.percent == 43.5
    assert old_target.percent == old_value


def test_external_change_rejects_stale_draft_and_refreshes_live_value(panel):
    panel.select_task("condenser_current")
    identity = ("condenser_lens_1", "percent")
    target = runtime_targets(panel._state)[identity[0]].obj
    editor = panel.editors[identity]
    changed = []
    panel.runtime_changed.connect(changed.append)
    editor.setText("45")
    target.percent = 46.0
    editor.editingFinished.emit()
    assert target.percent == 46.0
    assert float(panel.editors[identity].text()) == 46.0
    assert "changed elsewhere" in panel.status.text()
    assert changed == []


def test_unsupported_task_has_explicit_notice_and_no_editors(panel):
    panel.select_task("condenser_threefold")
    assert not panel.editors
    assert "Unsupported" in panel.description.text()
    assert any("No editable hardware" in label.text() for label in panel.findChildren(QLabel))


def test_missing_hardware_removes_editors_when_configuration_changes(panel):
    panel.select_task("condenser_current")
    assert ("condenser_lens_3", "percent") in panel.editors
    panel._state.layout_c3_hardware = "two_condenser"
    panel.set_state(panel._state)
    assert ("condenser_lens_3", "percent") not in panel.editors
    assert any("condenser lens 3 is not installed" in label.text()
               for label in panel.findChildren(QLabel))


def test_two_column_compact_page_scrolls_and_exposes_no_manufacturer_labels(panel, qtbot):
    panel.select_task("probe_corrector")
    qtbot.wait(10)
    assert panel.splitter.orientation() == Qt.Orientation.Horizontal
    assert panel.splitter.count() == 2
    assert panel.scroll.widgetResizable()
    assert panel.scroll.verticalScrollBar().maximum() > 0
    assert panel.scroll.horizontalScrollBar().maximum() == 0
    assert panel.width() == 760
    assert panel.height() == 540
    labels = [panel.title.text(), panel.description.text()]
    labels += [label.text() for label in panel.findChildren(QLabel)]
    labels += [box.title() for box in panel.findChildren(QGroupBox)]
    labels += [task.label + " " + task.category + " " + task.description for task in TUNING_TASKS]
    text = " ".join(labels).lower()
    for brand in ("iliad", "ultra", "spectra", "ceta", "s-corr", "thermo"):
        assert brand not in text
    panel.search.setText("Gun tilt")
    assert not panel._items["gun_tilt"].isHidden()
    assert panel._items["beam_shift"].isHidden()


@pytest.mark.parametrize("task", ["gun_shift", "probe_corrector"])
def test_compact_parameter_labels_and_group_bodies_retain_readable_geometry(panel, qtbot, task):
    from temsim.app import APPLICATION_STYLE

    panel.setStyleSheet(APPLICATION_STYLE)
    panel.select_task(task)
    qtbot.wait(50)
    _assert_control_geometry(panel)
    assert (panel.width(), panel.height()) == (760, 540)
    if task == "probe_corrector":
        assert panel.scroll.widget().height() > panel.scroll.viewport().height()
        assert panel.scroll.verticalScrollBar().maximum() > 0
        last = list(panel.editors.values())[-1]
        panel.scroll.ensureWidgetVisible(last)
        qtbot.wait(20)
        _assert_control_geometry(panel)
        assert panel.scroll.verticalScrollBar().value() > 0


@pytest.fixture
def window(qtbot, monkeypatch, tmp_path):
    from temsim.gui import main_window, interactive_calculation, calculation_controller

    settings = QSettings(str(tmp_path / "hardware-tuning.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(main_window, "QSettings", lambda: settings)
    monkeypatch.setattr(interactive_calculation, "QSettings", lambda: settings)
    monkeypatch.setattr(calculation_controller, "default_artifact_cache_root", lambda: tmp_path / "artifacts")
    for name in ("INITIAL_PREVIEW_DELAY_MS", "PREVIEW_DEBOUNCE_MS", "DESIGN_EXPLORER_DEBOUNCE_MS"):
        monkeypatch.setattr(main_window.MainWindow, name, 60_000)
    monkeypatch.setattr(main_window.MainWindow, "_apply_state_operating_modes", lambda *_: object())
    view = main_window.MainWindow()
    qtbot.addWidget(view)
    view.preview_timer.stop()
    monkeypatch.setattr(view.calculations.pool, "start", lambda *_: pytest.fail("GUI must not execute transport"))
    monkeypatch.setattr(view.direct_alignments, "submit", lambda *_args, **_kwargs: pytest.fail("Manual controls must not solve Direct Alignment"))
    view.show()
    yield view
    view.preview_timer.stop()
    view.close()


def test_main_window_edit_uses_existing_invalidation_and_keeps_parameter_panel_in_sync(window, monkeypatch):
    page = window.workspace.hardware_tuning
    assert window.workspace.tabs.tabText(window.workspace.tabs.indexOf(page)) == "Hardware tuning"
    window._select_component_from_workspace("condenser_lens_1")
    window.workspace.tabs.setCurrentWidget(page)
    page.select_task("condenser_current")
    old_revision = window._physical_revision
    old_generation = window.calculations.generation
    assert not window.preview_timer.isActive()
    old_navigation = window._selected_component_key
    before = deepcopy(window.state.to_dict())
    page.select_task("gun_shift")
    page.select_task("condenser_current")
    assert window.state.to_dict() == before
    assert window._physical_revision == old_revision
    assert window._selected_component_key == old_navigation
    assert window.workspace.tabs.currentWidget() is page
    assert not window.preview_timer.isActive()

    stale = []
    original_mark_stale = window.workspace.mark_ray_stale
    def mark_stale(state):
        stale.append(state)
        original_mark_stale(state)
    monkeypatch.setattr(window.workspace, "mark_ray_stale", mark_stale)
    _edit(page, ("condenser_lens_1", "percent"), "43.5")
    assert window.parameter_panel.lens_excitation.value() == 43.5
    assert window._physical_revision == old_revision + 1
    assert window.calculations.generation > old_generation
    assert stale == [window.state]
    assert window.preview_timer.isActive()
    assert window.workspace.tabs.currentWidget() is page
    assert window._selected_component_key == old_navigation

    # The original component editor remains the same authority and refreshes
    # the already-open hardware page through the ordinary runtime path.
    window.parameter_panel.lens_excitation.setValue(44.25)
    assert float(page.editors[("condenser_lens_1", "percent")].text()) == 44.25
    assert window._physical_revision == old_revision + 2


def test_hardware_splitter_participates_in_existing_layout_persistence(window, qtbot):
    page = window.workspace.hardware_tuning
    window.workspace.tabs.setCurrentWidget(page)
    qtbot.wait(20)
    manager = window.workspace_layouts
    assert manager.splitters["hardwareTuningSplitter"] is page.splitter
    page.splitter.setSizes([210, 650])
    page.splitter.splitterMoved.emit(page.splitter.sizes()[0], 1)
    manager.save_current()
    state = manager.splitter_states["hardwareTuningSplitter"]["default"]
    assert state == page.splitter.saveState()
    original = page.splitter.sizes()
    page.splitter.setSizes([400, 460])
    assert page.splitter.sizes() != original
    assert page.splitter.restoreState(state)
    assert page.splitter.sizes() == original


@pytest.mark.parametrize("task", ["gun_shift", "probe_corrector"])
def test_published_hardware_tab_retains_readable_controls(window, qtbot, task):
    from temsim.app import APPLICATION_STYLE

    window.setStyleSheet(APPLICATION_STYLE)
    page = window.workspace.hardware_tuning
    window.workspace.tabs.setCurrentWidget(page)
    page.select_task(task)
    qtbot.wait(50)
    _assert_control_geometry(page)
    if task == "probe_corrector":
        assert page.scroll.widget().height() > page.scroll.viewport().height()
        assert page.scroll.verticalScrollBar().maximum() > 0
