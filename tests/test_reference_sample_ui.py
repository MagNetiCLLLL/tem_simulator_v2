"""Real CIF sources drive the visible structure and read-only material summary."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from temsim.gui.sample_panel import SamplePage
from temsim.optics.column import default_state
from temsim.specimen.reference_catalog import get_reference_sample
from temsim.specimen.rutherford import resolve_tail_material


def _page(qtbot, state=None):
    state = state or default_state()
    page = SamplePage()
    qtbot.addWidget(page)
    page.set_state(state)
    return page, state


def test_default_reference_renders_actual_cif_atoms_and_five_nm_thickness(qtbot):
    page, state = _page(qtbot)
    assert state.sample.specimen_mode == "reference"
    assert page.mode.currentText() == "Reference CIF"
    assert page.scalar_controls["size_x_nm"].value() == pytest.approx(10.0)
    assert page.scalar_controls["thickness_nm"].value() == pytest.approx(5.0)
    page.resize(1280, 760)
    page.show()
    qtbot.waitUntil(lambda: page._snapshot is not None)
    snapshot = page._snapshot
    reference = get_reference_sample(state.sample.reference_sample_key)
    assert Path(snapshot.cif_path).resolve() == Path(reference.cif_path).resolve()
    assert snapshot.mode == "reference"
    assert snapshot.atomic_numbers.size > 0
    assert np.all(snapshot.atomic_numbers == 14)
    assert snapshot.atom_bond_pairs.shape[0] > 0
    assert not snapshot.regions
    assert snapshot.size_nm == (10.0, 10.0, 5.0)
    legend = "\n".join(label.text() for label in page.element_legend.findChildren(QLabel))
    assert "Si — Silicon" in legend
    assert "CIF" in page.scene_status.toolTip()


def test_auto_tail_summary_uses_structure_and_thickness_without_overwriting_manual(qtbot):
    state = default_state()
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


def test_open_cif_without_completed_wave_updates_atoms_even_with_old_result(qtbot, tmp_path):
    from ase.build import bulk
    from ase.io import write

    path = tmp_path / "NaCl.cif"
    write(path, bulk("NaCl", "rocksalt", a=5.64))
    page, state = _page(qtbot)
    page.display_result(SimpleNamespace(
        state_snapshot=SimpleNamespace(sample=deepcopy(state.sample)),
        stem_scan=None, wave_imaging=None,
    ))
    page.show()
    qtbot.waitUntil(lambda: page._snapshot is not None)
    page.mode.setCurrentIndex(page.mode.findData("atomic"))
    page.cif_path.setText(str(path))
    page._cif_edited()
    qtbot.waitUntil(lambda: page._snapshot is not None and page._snapshot.cif_path == str(path))
    assert set(page._snapshot.atomic_numbers) == {11, 17}
    assert "Na (Z=11)" in page.tail_material_summary.text()
    assert "Cl (Z=17)" in page.tail_material_summary.text()
    assert "Structural preview" in page.scene_status.text()
    page.mode.setCurrentIndex(page.mode.findData("reference"))
    assert state.sample.cif_path == str(path)
    assert set(page._snapshot.atomic_numbers) == {14}


def test_catalog_refresh_preserves_selection_and_reports_missing_reference(qtbot, monkeypatch):
    import temsim.gui.sample_panel as panel

    page, state = _page(qtbot)
    key = state.sample.reference_sample_key
    references = list(panel.available_reference_samples())
    added = SimpleNamespace(name="User crystal", key="new_user_crystal")
    refreshed = []
    monkeypatch.setattr(panel, "refresh_reference_samples", lambda: refreshed.append(True))
    monkeypatch.setattr(panel, "available_reference_samples", lambda: tuple(references))
    references.append(added)
    qtbot.mouseClick(page.refresh_references, Qt.MouseButton.LeftButton)
    assert refreshed
    assert page.preset.currentData() == key
    assert page.preset.findData(added.key) >= 0
    references[:] = [added]
    original = panel.active_cif_path

    def path_for(sample):
        if sample.specimen_mode == "reference" and sample.reference_sample_key == key:
            raise ValueError(f"Reference CIF missing: {key}")
        return original(sample)

    monkeypatch.setattr(panel, "active_cif_path", path_for)
    qtbot.mouseClick(page.refresh_references, Qt.MouseButton.LeftButton)
    assert page.preset.currentData() == key
    assert state.sample.reference_sample_key == key
    assert "Missing reference" in page.preset.currentText()
    assert "Reference CIF missing" in page.source_note.text()
    assert not page.tem_wave_enabled.isEnabled()


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


def test_bad_catalog_does_not_prevent_opening_sample_editor(qtbot, monkeypatch):
    import temsim.gui.sample_panel as panel

    def invalid_catalog():
        raise ValueError("Invalid user sidecar metadata")

    monkeypatch.setattr(panel, "available_reference_samples", invalid_catalog)
    page = SamplePage()
    qtbot.addWidget(page)
    assert page.preset.count() == 0
    assert "Invalid user sidecar metadata" in page.source_note.text()
    assert page.mode.findData("atomic") >= 0


def test_reference_selection_applies_declared_axes_and_keeps_finite_envelope(qtbot):
    page, state = _page(qtbot)
    key = "au_001"
    reference = get_reference_sample(key)
    original_size = (state.sample.size_x_nm, state.sample.size_y_nm, state.sample.thickness_nm)
    page.preset.setCurrentIndex(page.preset.findData(key))
    assert state.sample.reference_sample_key == key
    assert tuple(control.value() for control in page.zone_controls[1]) == reference.zone_axis
    assert tuple(control.value() for control in page.in_plane_controls[1]) == reference.in_plane_axis
    assert (state.sample.size_x_nm, state.sample.size_y_nm, state.sample.thickness_nm) == original_size
    assert "Au (Z=79)" in page.tail_material_summary.text()


def test_reference_refresh_and_source_controls_fit_visible_sidebar(qtbot):
    page, _state = _page(qtbot)
    page.resize(1280, 760)
    page.show()
    qtbot.waitUntil(lambda: page._snapshot is not None)
    page.controls_scroll.ensureWidgetVisible(page.refresh_references)
    viewport = page.controls_scroll.viewport()
    qtbot.waitUntil(lambda: page.controls_scroll.widget().width() <= viewport.width())
    for control in (page.reference_source_widget, page.refresh_references, page.cif_source_widget,
                    page.zone_controls[0], page.in_plane_controls[0]):
        left = control.mapTo(viewport, control.rect().topLeft()).x()
        right = control.mapTo(viewport, control.rect().topRight()).x()
        assert 0 <= left <= right < viewport.width()
