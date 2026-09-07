"""Render the actual editor with a temporary, unsaved mechanical design."""
import os
from pathlib import Path
import shutil
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def main():
    from PySide6.QtCore import QSettings, QPointF
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
    temporary = Path(tempfile.mkdtemp(prefix="cad_preview_", dir=project / "tmp"))
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
    key = "intermediate_lens_excitation_coil"
    window._select_physical_component(key)
    layout = window.workspace.physical_layout
    window.workspace.tabs.setCurrentWidget(layout)
    page = layout.model_editor
    layout.tabs.setCurrentWidget(page)
    app.processEvents()
    page.session.set_model_3d(key, {
        "schema_version": 1,
        "base": {"kind": "box", "width_mm": 80.0, "height_mm": 60.0, "length_mm": 20.0},
        "transform": {"scale_xy": [1.0, 1.0], "offset_mm": [0.0, 0.0, 0.0], "rotation_deg": [0.0, 0.0, 0.0]},
        "features": [
            {"id": "mounting_hole", "kind": "hole", "axis": "z", "center_mm": [18.0, 0.0, 0.0], "diameter_mm": 14.0, "depth_mm": 30.0},
            {"id": "alignment_groove", "kind": "slot", "axis": "z", "center_mm": [-18.0, 0.0, 9.0], "width_mm": 8.0, "length_mm": 28.0, "depth_mm": 8.0, "rotation_deg": 35.0},
        ],
    })
    page._draft_changed()
    page.view.fit_all()
    page.view.set_isometric_view()
    page.selection_mode.setCurrentIndex(page.selection_mode.findData("edge"))
    app.processEvents()
    for mesh in page._mesh_records:
        for edge in mesh["edges"]:
            if "mounting_hole" in str(edge["id"]) and "positive" in str(edge["id"]):
                page.view.set_topology_selection([dict(key=key, region="body", kind="edge", id=edge["id"])], emit=True)
                break
    app.processEvents()
    window.grab().save(str(project / "tmp" / "part_features_parameters.png"))
    page.parameter_tabs.setCurrentIndex(page.parameter_tabs.count() - 1)
    app.processEvents()
    window.grab().save(str(project / "tmp" / "part_features_editor.png"))
    print("Preview images:", project / "tmp" / "part_features_parameters.png", project / "tmp" / "part_features_editor.png")
    window.close()


if __name__ == "__main__":
    main()
