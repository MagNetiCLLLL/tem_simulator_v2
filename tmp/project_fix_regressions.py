"""Run isolated Qt/physics regression checks from a real Windows main module."""
from pathlib import Path
import os
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication, QSettings, QStandardPaths
    import pytest

    output = Path(tempfile.mkdtemp(prefix="project_fix_tests_", dir=root / "tmp"))
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(output / "settings"))
    QCoreApplication.setOrganizationName("TEMProjectFixVerification")
    QCoreApplication.setApplicationName("Regression")
    QStandardPaths.setTestModeEnabled(True)
    names = """
    test_mvp_core test_manifest_editing test_toml_authority
    test_profile_optional_values test_sample_profile_v2 test_aberration_model
    test_artifact_quota test_calculation_manifest_artifacts
    test_calculation_controller test_background_calculation_requests
    test_calculation_cache_reuse test_cache_preferences
    test_stem_recording_cache_version test_interactive_calculation
    test_fourdstem test_fourdstem_cache_products test_fourdstem_user_wiring
    test_record_plane test_first_order_transfer test_scan_system test_stem_recording_deflection
    test_stem_sampling test_stem_sampling_gui test_wave_imaging
    test_stem_detector_control test_vector_field_transport
    test_tem_result_sources test_stem_result_sources test_shared_image_sources
    test_bank_readout_bridge test_stem_cuda_pipeline test_wave_fft
    """.split()
    # Include the independently added physical-routing regressions once ready.
    names += [path.stem for path in sorted((root / "tests").glob("test_*descan*.py"))
              if path.stem not in names]
    paths = [str(root / "tests" / f"{name}.py") for name in names]
    print("Verification output:", output, flush=True)
    return pytest.main([
        "-o", "addopts=", "-q", "--tb=short", "-ra", "--durations=12",
        "-o", f"cache_dir={output / 'cache'}",
        f"--basetemp={output / 'cases'}",
        f"--junitxml={root / 'tmp' / 'project_fix_regressions.xml'}",
        *paths,
    ])


if __name__ == "__main__":
    raise SystemExit(main())
