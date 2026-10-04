"""Real Qt list controls; no numerical propagation is claimed by these tests."""

from dataclasses import FrozenInstanceError

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QPushButton

from temsim.gui.coherent_state_list import TipStateEntry, TipStateList


@pytest.fixture
def widget(qtbot):
    view = TipStateList()
    qtbot.addWidget(view)
    view.resize(540, 350)
    view.show()
    return view


def test_list_defaults_and_explicit_actions(widget, qtbot):
    assert widget.entries() == ()
    assert widget.selected_id() is None
    assert widget.display_mode() == "current"
    assert not widget.replace_button.isEnabled()
    assert not widget.restore_button.isEnabled()
    assert not widget.remove_button.isEnabled()
    assert widget.findChild(QPushButton, "calculateCoherentStates") is None
    assert "Calculate beam above" in widget.help.text()
    assert widget.table.maximumHeight() == 200
    assert widget.help.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
    add = QSignalSpy(widget.add_requested)
    qtbot.mouseClick(widget.add_button, Qt.MouseButton.LeftButton)
    assert add.count() == 1
    assert widget.entries() == ()  # Controller owns the physical capture.


def test_stable_ids_selection_and_no_duplicate_notifications(widget):
    edits = QSignalSpy(widget.entries_edited)
    display = QSignalSpy(widget.display_changed)
    widget.add_entry("first-id", "Electron state 1", "0.4 eV")
    assert widget.entries() == (TipStateEntry("first-id", "Electron state 1", 1., True),)
    assert widget.selected_id() == "first-id"
    assert (edits.count(), display.count()) == (1, 1)
    widget.add_entry("second-id", "Electron state 2", "0.8 eV")
    assert widget.selected_id() == "second-id"
    assert (edits.count(), display.count()) == (2, 2)
    widget.set_status("first-id", "Ready at Z 100 mm")
    widget.set_summary("first-id", "0.5 eV")
    assert widget.selected_id() == "second-id"
    assert (edits.count(), display.count()) == (2, 2)
    assert "Ready at Z 100 mm" in widget.table.item(0, 3).text()
    assert "0.5 eV" in widget.table.item(0, 3).toolTip()
    widget.remove_entry("first-id")
    assert widget.selected_id() == "second-id"
    assert widget.entries()[0].id == "second-id"
    assert (edits.count(), display.count()) == (3, 3)
    widget.remove_entry("second-id")
    assert widget.selected_id() is None
    assert not widget.restore_button.isEnabled()
    assert (edits.count(), display.count()) == (4, 4)


def test_editing_name_weight_visibility_and_mode_only_updates_display_metadata(widget):
    widget.add_entry("id", "First", "initial state")
    edits = QSignalSpy(widget.entries_edited)
    display = QSignalSpy(widget.display_changed)
    widget.table.item(0, 1).setText("Renamed")
    widget.table.cellWidget(0, 2).setValue(2.5)
    widget.table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    widget.mode.setCurrentIndex(2)
    assert widget.entries() == (TipStateEntry("id", "Renamed", 2.5, False),)
    assert widget.display_mode() == "overlay"
    assert (edits.count(), display.count()) == (3, 4)
    widget.table.cellWidget(0, 2).setValue(0.)
    assert widget.entries()[0].weight == 0.
    assert widget.table.cellWidget(0, 2).minimum() == 0.
    assert widget.table.cellWidget(0, 2).decimals() == 4
    with pytest.raises(FrozenInstanceError):
        widget.entries()[0].weight = 100.


def test_selected_actions_emit_captured_identity_without_mutation(widget, qtbot):
    widget.add_entry("first", "State 1", "0.3 eV")
    widget.add_entry("second", "State 2", "0.4 eV")
    spies = {name: QSignalSpy(getattr(widget, f"{name}_requested"))
             for name in ("replace", "restore", "remove")}
    before = widget.entries()
    for name in ("replace", "restore", "remove"):
        qtbot.mouseClick(getattr(widget, f"{name}_button"), Qt.MouseButton.LeftButton)
        assert spies[name].count() == 1
        assert spies[name].at(0) == ["second"]
    assert widget.entries() == before
    widget.table.selectRow(0)
    qtbot.mouseClick(widget.restore_button, Qt.MouseButton.LeftButton)
    assert spies["restore"].at(1) == ["first"]


def test_selection_changes_display_only_in_selected_mode(widget):
    widget.add_entry("a", "A", "")
    widget.add_entry("b", "B", "")
    display = QSignalSpy(widget.display_changed)
    widget.table.selectRow(0)
    assert display.count() == 0
    widget.mode.setCurrentIndex(1)
    assert widget.display_mode() == "selected"
    assert display.count() == 1
    widget.table.selectRow(1)
    assert display.count() == 2
    widget.set_status("b", "Calculating")
    assert display.count() == 2
    widget.table.clearSelection()
    assert widget.selected_id() is None
    assert not widget.restore_button.isEnabled()
    assert display.count() == 3


def test_invalid_ids_and_unknown_updates_do_not_mutate_rows(widget):
    widget.add_entry("id", "A", "summary")
    before = widget.entries()
    with pytest.raises(ValueError, match="Duplicate"):
        widget.add_entry("id", "Duplicate", "")
    with pytest.raises(ValueError, match="non-empty"):
        widget.add_entry("", "Empty", "")
    with pytest.raises(ValueError, match="needs a name"):
        widget.add_entry("other", "  ", "")
    for action in (lambda: widget.remove_entry("unknown"),
                   lambda: widget.set_status("unknown", "Ready"),
                   lambda: widget.set_summary("unknown", "summary")):
        with pytest.raises(KeyError):
            action()
    assert widget.entries() == before
    widget.table.item(0, 1).setText("")
    assert widget.entries()[0].name == "Unnamed state"
