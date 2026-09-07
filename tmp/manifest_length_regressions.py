"""Isolated Qt and geometry verification for editable part lengths."""
from pathlib import Path
import os
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication, QSettings, QStandardPaths
    import pytest

    output = Path(tempfile.mkdtemp(prefix="manifest_length_tests_", dir=root / "tmp"))
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(output / "settings"))
    QCoreApplication.setOrganizationName("TEMManifestLengthVerification")
    QCoreApplication.setApplicationName("Regression")
    QStandardPaths.setTestModeEnabled(True)
    names = """
        test_manifest_length_editing test_manifest_length_panel test_projector_custom_clearance
        test_manifest_editing test_mvp_core test_toml_authority test_nanopulser_assembly
        test_fixed_apertures test_lens_field_provider test_input_policy
    """.split()
    paths = [str(root / "tests" / f"{name}.py") for name in names]
    paths += [f"tests/test_gui_shell.py::{name}" for name in (
        "test_manifest_geometry_reload_retains_lens_strengths_until_explicit_preset",
        "test_layout_selection_opens_unmodelled_iliad_component_toml",
        "test_lens_selection_exposes_live_excitation_control",
    )]
    print("Verification output:", output, flush=True)
    return pytest.main([
        "-o", "addopts=", "-q", "--tb=short", "-ra", "--durations=10",
        "-o", f"cache_dir={output / 'cache'}", f"--basetemp={output / 'cases'}",
        f"--junitxml={root / 'tmp' / 'manifest_length_regressions.xml'}", *paths,
    ])


if __name__ == "__main__":
    raise SystemExit(main())
