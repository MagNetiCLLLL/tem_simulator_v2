"""Verify the real main-window aperture wiring and render its corrected view."""
import os
from pathlib import Path
import shutil
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def main():
    import numpy as np
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
    temporary = Path(tempfile.mkdtemp(prefix="aperture_preview_", dir=project / "tmp"))
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
    key = "condenser_aperture_2"
    window._select_physical_component(key)
    layout = window.workspace.physical_layout
    window.workspace.tabs.setCurrentWidget(layout)
    page = layout.model_editor
    layout.tabs.setCurrentWidget(page)
    app.processEvents()
    original = page.session.path.read_bytes()

    def opening_radius():
        mesh = next(mesh for mesh in page._mesh_records if mesh["key"] == key)
        faces = mesh["faces"][np.asarray(mesh["face_groups"]) == "working_opening"]
        points = mesh["vertices"][np.unique(faces)]
        assert np.isclose(np.ptp(mesh["vertices"][:, 2]), .2)
        return np.linalg.norm(points[:, :2], axis=1)

    assert np.allclose(opening_radius(), .05)
    target = window._runtime_targets[key].obj
    target.radius_mm = .1
    window._runtime_parameter_changed("radius_mm")
    assert np.allclose(opening_radius(), .1)
    target.radius_mm = .05
    window._runtime_parameter_changed("radius_mm")
    assert np.allclose(opening_radius(), .05)
    assert not page.session.dirty and page.session.path.read_bytes() == original
    page.view.fit_all()
    page.view.set_isometric_view()
    page.view.set_topology_selection([dict(key=key, region="body", kind="face", id="plate_positive")], emit=True)
    app.processEvents()
    labels = [page.dimensions.item(row, 0).text() for row in range(page.dimensions.rowCount())]
    assert "Material outer diameter" not in labels
    assert "Working opening diameter" in labels
    destination = project / "tmp" / "aperture_model_corrected.png"
    window.grab().save(str(destination))
    print("Main-window live opening verified; source unchanged. Screenshot:", destination)
    window.close()


if __name__ == "__main__":
    main()
