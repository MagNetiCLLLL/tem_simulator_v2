"""Render the integrated parameter inspector using disposable instrument files."""
import os
from pathlib import Path
import shutil
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def main():
    from PySide6.QtCore import QSettings, Qt
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QApplication
    from temsim.app import APPLICATION_STYLE
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.gui import main_window
    from temsim.manifest_editor import ManifestEditor
    from temsim.paths import INSTRUMENT_CONFIG_ROOT

    app = QApplication([])
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
    app.setFont(QFont("Segoe UI", 9))
    app.setStyle("Fusion")
    app.setStyleSheet(APPLICATION_STYLE)
    project = Path(__file__).resolve().parents[1]
    temporary = Path(tempfile.mkdtemp(prefix="parameter_preview_", dir=project / "tmp"))
    root = temporary / "instruments"
    shutil.copytree(INSTRUMENT_CONFIG_ROOT, root)
    main_window.AssemblyCatalog = lambda: AssemblyCatalog(root)
    main_window.ManifestEditor = lambda: ManifestEditor(root)
    main_window.QSettings = lambda: QSettings(str(temporary / "preview.ini"), QSettings.Format.IniFormat)
    main_window.MainWindow.schedule_preview = lambda *args: None
    window = main_window.MainWindow()
    window.preview_timer.stop()
    window.resize(1910, 1040)
    window.show()
    layout = window.workspace.physical_layout
    page = layout.model_editor
    window.workspace.tabs.setCurrentWidget(layout)
    layout.tabs.setCurrentWidget(page)
    window._select_physical_component("condenser_aperture_2")
    app.processEvents()
    page.view.set_topology_selection([dict(key="condenser_aperture_2", region="body", kind="face", id="plate_positive")], emit=True)
    app.processEvents()
    window.grab().save(str(project / "tmp" / "parameter_definitions_c2.png"))

    key = "intermediate_lens_excitation_coil"
    window._select_physical_component(key)
    window.set_simulation_mode("custom")
    window.state.lens_field_map_descriptors["intermediate_lens"] = {
        "solver": "axisymmetric_linear_fem", "ampere_turns": 1000., "relative_permeability": 500.}
    window._refresh_parameter_simulation_context()
    page.session.set_model_3d(key, {"schema_version": 1, "base": {"kind": "existing"},
                                  "transform": {"scale_xy": [1.1, 1.]}})
    page._draft_changed()
    for row in range(page.dimensions.rowCount()):
        if tuple(page.dimensions.item(row, 1).data(Qt.ItemDataRole.UserRole)) == ("parts", key, "mechanical_outer_diameter_mm"):
            page.dimensions.setCurrentCell(row, 0)
    page.view.fit_all()
    app.processEvents()
    window.grab().save(str(project / "tmp" / "parameter_definitions_coil.png"))
    page.revert()

    dialog = page.show_dimension_audit()
    dialog.search.setText("condenser_aperture_2")
    app.processEvents()
    dialog.grab().save(str(project / "tmp" / "parameter_definitions_audit.png"))
    assert not page.session.dirty
    for source in INSTRUMENT_CONFIG_ROOT.rglob("*.toml"):
        assert source.read_bytes() == (root / source.relative_to(INSTRUMENT_CONFIG_ROOT)).read_bytes()
    print("Rendered C2, coil draft / field-use and audit. All copied source files remain unchanged.")
    window.close()


if __name__ == "__main__":
    main()
