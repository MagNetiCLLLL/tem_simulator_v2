"""Real CIF sources drive the visible structure and read-only material summary."""
from copy import deepcopy
from pathlib import Path
import shutil
import tomllib
from types import SimpleNamespace

import numpy as np
import pytest
import tomli_w
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QLabel

from temsim.gui.sample_panel import SamplePage
from temsim.optics.column import default_state
from specimen_inputs import SI_CIF, imported_sample
from temsim.specimen.rutherford import resolve_tail_material
from temsim.specimen.source import active_cif_path, validate_sample_source


def test_import_retains_vacuum_until_source_is_selected_and_can_be_reused(qtbot, monkeypatch):
    page, state = _page(qtbot)
    assert [page.mode.itemData(i) for i in range(page.mode.count())] == ["vacuum", "atomic"]
    assert not hasattr(page, "preset")
    assert not hasattr(page, "refresh_references")
    assert page.cif_browse.isEnabled() and not page.inserted.isEnabled()
    assert not state.sample.inserted
    changes, calculations, dialogs = [], [], []
    page.parameters_changed.connect(changes.append)
    page.calculation_requested.connect(lambda: calculations.append(True))

    def choose_cif(*args):
        dialogs.append(True)
        return str(SI_CIF), ""

    monkeypatch.setattr(QFileDialog, "getOpenFileName", choose_cif)
    page.cif_browse.click()
    assert state.sample.specimen_mode == "vacuum" and not state.sample.inserted
    assert state.sample.cif_path == str(SI_CIF)
    assert active_cif_path(state.sample) == ""
    assert page.mode.currentData() == "vacuum"
    assert "CIF retained: Si.cif" in page.source_note.text()
    assert not page.apply_zone.isEnabled()
    assert changes == ["sample.cif_path"]
    page.show()
    qtbot.waitUntil(lambda: page._snapshot is not None)
    assert page._snapshot.mode == "vacuum"
    assert page._snapshot.atomic_numbers.size == 0
    page.mode.setCurrentIndex(page.mode.findData("atomic"))
    assert state.sample.specimen_mode == "atomic" and state.sample.inserted
    assert page.apply_zone.isEnabled()
    qtbot.waitUntil(lambda: page._snapshot is not None and page._snapshot.atomic_numbers.size > 0)
    assert set(page._snapshot.atomic_numbers) == {14}
    page.mode.setCurrentIndex(page.mode.findData("vacuum"))
    assert not state.sample.inserted
    assert page.mode.currentData() == "vacuum"
    qtbot.waitUntil(lambda: page._snapshot.mode == "vacuum")
    assert page._snapshot.atomic_numbers.size == 0
    assert "no specimen interactions" in page.scene_status.text()
    assert "Vacuum sample" in page.full_sample_label.text()
    assert state.sample.cif_path == str(SI_CIF)
    page.mode.setCurrentIndex(page.mode.findData("atomic"))
    qtbot.waitUntil(lambda: page._snapshot.mode == "atomic")
    assert set(page._snapshot.atomic_numbers) == {14}
    assert dialogs == [True]  # Only the explicit Open CIF click opens a dialog.
    assert calculations == []
    assert changes == ["sample.cif_path"] + ["sample.specimen_mode"] * 3


def test_failed_import_preserves_existing_source_and_does_not_invalidate(qtbot, tmp_path):
    state = default_state()
    imported_sample(state)
    page, _ = _page(qtbot, state)
    before = deepcopy(state.sample)
    errors, changes = [], []
    page.error.connect(errors.append)
    page.parameters_changed.connect(changes.append)
    path = tmp_path / "broken.cif"
    path.write_text("not a CIF")
    page.cif_path.setText(str(path))
    page._cif_edited()
    assert state.sample == before
    assert len(errors) == 1 and "CIF import failed" in errors[0]
    assert changes == []
    assert page.cif_path.text() == before.cif_path


def test_selecting_unconfigured_cif_does_not_open_a_dialog_or_silently_use_vacuum(qtbot, monkeypatch):
    page, state = _page(qtbot)
    dialogs = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: dialogs.append(True))
    page.mode.setCurrentIndex(page.mode.findData("atomic"))
    assert dialogs == []
    assert page.mode.currentData() == state.sample.specimen_mode == "atomic"
    assert state.sample.inserted
    assert "Use Open CIF" in page.source_note.text()
    with pytest.raises(ValueError, match="Import a CIF"):
        validate_sample_source(state.sample)
    page.mode.setCurrentIndex(page.mode.findData("vacuum"))
    validate_sample_source(state.sample)
    assert dialogs == []


@pytest.mark.parametrize("mode,inserted", [("vacuum", False), ("atomic", True), ("atomic", False)])
def test_open_cancel_replace_and_clear_preserve_source_and_holder(qtbot, monkeypatch, mode, inserted):
    state = default_state()
    state.sample.specimen_mode = mode
    state.sample.inserted = inserted
    page, _ = _page(qtbot, state)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("", ""))
    before = deepcopy(state.sample)
    changes = []
    page.parameters_changed.connect(changes.append)
    page.cif_browse.click()
    assert state.sample == before
    assert changes == []
    page.cif_path.setText(str(SI_CIF))
    page._cif_edited()
    assert state.sample.specimen_mode == mode and state.sample.inserted == inserted
    assert state.sample.cif_path == str(SI_CIF)
    page.cif_browse.click()
    assert state.sample.cif_path == str(SI_CIF)
    assert changes == ["sample.cif_path"]
    from specimen_inputs import AU_CIF
    page.cif_path.setText(str(AU_CIF))
    page._cif_edited()
    assert state.sample.specimen_mode == mode and state.sample.inserted == inserted
    assert state.sample.cif_path == str(AU_CIF)
    page.cif_path.clear()
    page._cif_edited()
    assert state.sample.specimen_mode == mode and state.sample.inserted == inserted
    assert not state.sample.cif_path


def test_completed_vacuum_sample_is_reported_without_inventing_interactions(qtbot):
    page, state = _page(qtbot)
    result = SimpleNamespace(workflow="sample", simulation=SimpleNamespace(), state_snapshot=state)
    page.display_result(result)
    assert page.calculation_bar.status.text() == "Completed result displayed."
    assert "Vacuum" in page.calculate_button.toolTip()


def _page(qtbot, state=None):
    state = state or default_state()
    page = SamplePage()
    qtbot.addWidget(page)
    page.set_state(state)
    return page, state


def test_auto_tail_summary_uses_structure_and_thickness_without_overwriting_manual(qtbot):
    state = default_state()
    imported_sample(state)
    state.sample.real_tail_atomic_number = 29
    state.sample.real_tail_areal_density_atoms_nm2 = 123.5
    state.sample.real_tail_screening_angle_mrad = 3.75
    page, state = _page(qtbot, state)
    original = resolve_tail_material(state.sample, 5.0, state.beam_voltage_kv)
    element = original.elements[0]
    assert "Si (Z=14)" in page.tail_material_summary.text()
    assert f"{element.areal_density_atoms_nm2:.6g} atoms/nm²" in page.tail_material_summary.text()
    assert not page.tail_atomic_number.isEnabled()
    assert not page.tail_density.isEnabled()
    assert not page.tail_screening.isEnabled()
    page.scalar_controls["thickness_nm"].setValue(10.0)
    assert f"{element.areal_density_atoms_nm2 * 2:.6g} atoms/nm²" in page.tail_material_summary.text()
    assert state.sample.real_tail_atomic_number == 29
    assert state.sample.real_tail_areal_density_atoms_nm2 == 123.5
    assert state.sample.real_tail_screening_angle_mrad == 3.75
    page.tail_material_source.setCurrentIndex(page.tail_material_source.findData("manual"))
    page.tail_screening_source.setCurrentIndex(page.tail_screening_source.findData("manual"))
    assert page.tail_atomic_number.isEnabled() and page.tail_density.isEnabled()
    assert page.tail_screening.isEnabled()
    assert "Cu (Z=29)" in page.tail_material_summary.text()
    assert "123.5 atoms/nm²" in page.tail_material_summary.text()
    assert "screening 3.75 mrad" in page.tail_material_summary.text()
    page.tail_material_source.setCurrentIndex(0)
    page.tail_screening_source.setCurrentIndex(0)
    assert "Si (Z=14)" in page.tail_material_summary.text()
    assert page.tail_density.value() == 123.5
    assert page.tail_screening.value() == 3.75


def test_invalid_imported_structure_reports_auto_tail_error_without_manual_fallback(qtbot, tmp_path):
    state = default_state()
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = str(tmp_path / "missing.cif")
    page, state = _page(qtbot, state)
    assert "Tail material unavailable" in page.tail_material_summary.text()
    assert state.sample.real_tail_material_source == "structure"
    assert page.tail_material_source.currentData() == "structure"
    assert not page.tail_atomic_number.isEnabled()
    page.show()
    qtbot.waitUntil(lambda: page._snapshot is not None)
    assert page._snapshot.atomic_numbers.size == 0
    assert "Geometry retained" in page.atom_display_label.text()


def test_open_mcif_zone_alignment_and_visible_atoms(qtbot, tmp_path):
    source = SI_CIF
    imported = tmp_path / "structure.mcif"
    shutil.copyfile(source, imported)
    state = default_state()
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = str(imported)
    state.sample.size_x_nm = state.sample.size_y_nm = state.sample.thickness_nm = 1
    page, state = _page(qtbot, state)
    errors = []
    page.error.connect(errors.append)
    for control, value in zip(page.zone_controls[1], (0, 0, 1)):
        control.setValue(value)
    for control, value in zip(page.in_plane_controls[1], (1, 0, 0)):
        control.setValue(value)
    qtbot.mouseClick(page.apply_zone, Qt.MouseButton.LeftButton)
    assert errors == []
    assert state.sample.zone_axis_uvw == (0, 0, 1)
    page.show()
    qtbot.waitUntil(lambda: page._snapshot is not None)
    assert page._snapshot.atomic_numbers.size > 0
    assert set(page._snapshot.atomic_numbers) == {14}
    assert "Spheres unavailable" not in page.atom_display_label.text()
