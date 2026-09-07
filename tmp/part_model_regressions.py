"""Isolated regression verification for file-backed 3D component editing."""
from pathlib import Path
import os
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication, QSettings, QStandardPaths
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QApplication
    import pytest

    output = Path(tempfile.mkdtemp(prefix="part_model_tests_", dir=root / "tmp"))
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(output / "settings"))
    QCoreApplication.setOrganizationName("TEMPartModelVerification")
    QCoreApplication.setApplicationName("Regression")
    QStandardPaths.setTestModeEnabled(True)
    application = QApplication.instance() or QApplication([])
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"
    if font_path.is_file():
        QFontDatabase.addApplicationFont(str(font_path))
        application.setFont(QFont("Segoe UI", 9))
    names = """
        test_part_model_3d test_part_model_features test_part_model_document test_part_model_view
        test_part_model_editor test_part_materials
        test_part_geometry test_part_geometry_editor test_part_geometry_validation
        test_part_geometry_integration test_manifest_length_editing
        test_manifest_length_panel test_projector_custom_clearance test_manifest_editing
        test_mvp_core test_toml_authority test_nanopulser_assembly
        test_fixed_apertures test_lens_field_provider test_input_policy
        test_magnetic_circuits test_nonlinear_magnetostatics test_magnetic_validation
        test_lens_material_defaults
    """.split()
    paths = [str(root / "tests" / f"{name}.py") for name in names]
    paths += [f"tests/test_gui_shell.py::{name}" for name in (
        "test_manifest_geometry_reload_retains_lens_strengths_until_explicit_preset",
        "test_layout_selection_opens_unmodelled_iliad_component_toml",
        "test_lens_selection_exposes_live_excitation_control",
        "test_workspace_action_buttons_fit_without_a_window_state_change",
    )]
    print("Verification output:", output, flush=True)
    return pytest.main([
        "-o", "addopts=", "-q", "--tb=short", "-ra", "--durations=10",
        "-o", f"cache_dir={output / 'cache'}", f"--basetemp={output / 'cases'}",
        f"--junitxml={root / 'tmp' / 'part_model_regressions.xml'}", *paths,
    ])


if __name__ == "__main__":
    raise SystemExit(main())
