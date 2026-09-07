"""Isolated GUI navigation regression run; never uses personal QSettings."""
from pathlib import Path
import os
import tempfile
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication, QSettings, QStandardPaths
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QApplication
    import pytest

    output = Path(tempfile.mkdtemp(prefix="physical_navigation_tests_", dir=root / "tmp"))
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(output / "settings"))
    QCoreApplication.setOrganizationName("TEMNavigationVerification")
    QCoreApplication.setApplicationName("Regression")
    QStandardPaths.setTestModeEnabled(True)
    application = QApplication.instance() or QApplication([])
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
    application.setFont(QFont("Segoe UI", 9))
    paths = ["tests/test_part_model_view.py", "tests/test_part_model_document.py",
             "tests/test_part_model_3d.py", "tests/test_part_model_editor.py"]
    paths += [f"tests/test_gui_shell.py::{name}" for name in (
        "test_component_tree_double_click_reveals_without_changing_auto_zoom",
        "test_runtime_control_double_click_has_no_false_physical_position",
        "test_component_navigation_filters_only_the_active_assembly",
        "test_layout_selection_opens_energy_slit_editor_and_updates_window",
        "test_layout_selection_opens_unmodelled_iliad_component_toml",
        "test_ray_plot_marks_every_component_centre_and_detected_crossover",
    )]
    paths = sys.argv[1:] or paths
    return pytest.main(["-o", "addopts=", "-q", "--tb=short", "-ra", "--durations=5",
        "-o", f"cache_dir={output / 'cache'}", f"--basetemp={output / 'cases'}", *paths])


if __name__ == "__main__":
    raise SystemExit(main())
