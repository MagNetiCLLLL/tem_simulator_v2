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


@pytest.fixture
def copied_reference_library(tmp_path, monkeypatch):
    from temsim.specimen import reference_catalog

    reference = get_reference_sample("si_110")
    shutil.copyfile(reference.cif_path, tmp_path / "Si.cif")
    shutil.copyfile(reference.metadata_path, tmp_path / "Si.toml")
    monkeypatch.setattr(reference_catalog, "REFERENCE_DIRECTORY", tmp_path)
    return tmp_path


@pytest.mark.parametrize("change", ["cif", "metadata", "missing_cif", "invalid_metadata"])
def test_refresh_active_reference_invalidates_results_without_changing_sample(
    qtbot, copied_reference_library, change,
):
    page, state = _page(qtbot)
    state.sample.wave_frozen_phonon_enabled = True
    before = deepcopy(vars(state.sample))
    sentinel = object()
    page._eds_result = page._elastic_result = page._specimen_interactions = sentinel
    changes = []
    page.parameters_changed.connect(changes.append)
    directory = copied_reference_library
    if change == "cif":
        path = directory / "Si.cif"
        path.write_bytes(path.read_bytes() + b"\n# file revision\n")
    elif change == "metadata":
        path = directory / "Si.toml"
        metadata = tomllib.loads(path.read_text(encoding="utf-8"))
        metadata["thermal_sigma_angstrom"] = 0.095
        path.write_text(tomli_w.dumps(metadata), encoding="utf-8")
    elif change == "missing_cif":
        (directory / "Si.cif").unlink()
    else:
        (directory / "Si.toml").write_text("invalid = [", encoding="utf-8")
    qtbot.mouseClick(page.refresh_references, Qt.MouseButton.LeftButton)
    assert changes == ["sample.reference_source"]
    assert page._eds_result is page._elastic_result is page._specimen_interactions is None
    assert vars(state.sample) == before
    qtbot.mouseClick(page.refresh_references, Qt.MouseButton.LeftButton)
    assert changes == ["sample.reference_source"]


@pytest.mark.parametrize("mode", ["reference", "atomic"])
def test_refresh_unmodified_or_unselected_reference_preserves_results(
    qtbot, copied_reference_library, mode,
):
    state = default_state()
    state.sample.specimen_mode = mode
    state.sample.cif_path = str(copied_reference_library / "Si.cif")
    page, _ = _page(qtbot, state)
    sentinel = object()
    page._eds_result = sentinel
    changes = []
    page.parameters_changed.connect(changes.append)
    qtbot.mouseClick(page.refresh_references, Qt.MouseButton.LeftButton)
    shutil.copyfile(copied_reference_library / "Si.cif", copied_reference_library / "other.cif")
    qtbot.mouseClick(page.refresh_references, Qt.MouseButton.LeftButton)
    assert changes == []
    assert page._eds_result is sentinel
    if mode == "reference":
        assert page.preset.findData("other") >= 0
    else:
        # Reference refresh is disabled while an external CIF owns the source.
        assert not page.refresh_references.isEnabled()
        assert page.preset.findData("other") == -1


def test_reference_refresh_reaches_main_window_stale_path(
    qtbot, monkeypatch, copied_reference_library, tmp_path,
):
    from PySide6.QtCore import QSettings
    from temsim.gui import main_window

    settings = str(tmp_path / "window.ini")
    monkeypatch.setattr(main_window, "QSettings", lambda: QSettings(settings, QSettings.Format.IniFormat))
    window = main_window.MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    monkeypatch.setattr(window.calculations, "submit_background", lambda *_args, **_kwargs: None)
    stale = []
    monkeypatch.setattr(window.workspace, "mark_high_accuracy_stale", lambda: stale.append(True))
    path = copied_reference_library / "Si.toml"
    metadata = tomllib.loads(path.read_text(encoding="utf-8"))
    metadata["thermal_sigma_angstrom"] = 0.095
    path.write_text(tomli_w.dumps(metadata), encoding="utf-8")
    qtbot.mouseClick(window.workspace.sample_page.refresh_references, Qt.MouseButton.LeftButton)
    window.preview_timer.stop()
    assert stale == [True]


def test_open_mcif_zone_alignment_and_visible_atoms(qtbot, tmp_path):
    source = get_reference_sample("si_110").cif_path
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
