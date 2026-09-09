"""Advertised CIF/MCIF imports retain atoms across GUI and numerical inputs."""

from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from temsim.optics.model import Sample
from temsim.specimen import reference_catalog
from temsim.specimen.atomistic import build_cif_equilibrium_atoms
from temsim.specimen.geometry import build_sample_geometry_snapshot


@pytest.mark.parametrize("extension", [".mcif", ".MCIF"])
def test_mcif_reference_display_and_iam_atoms_match_cif(tmp_path, monkeypatch, extension):
    source = reference_catalog.get_reference_sample("si_110").cif_path
    original = tmp_path / "original.cif"
    imported = tmp_path / ("imported" + extension)
    shutil.copyfile(source, original)
    shutil.copyfile(source, imported)
    monkeypatch.setattr(reference_catalog, "REFERENCE_DIRECTORY", tmp_path)
    baseline = Sample(size_x_nm=1, size_y_nm=1, thickness_nm=1)
    candidate = Sample(size_x_nm=1, size_y_nm=1, thickness_nm=1)
    reference_catalog.apply_reference_sample(baseline, "original")
    reference_catalog.apply_reference_sample(candidate, "imported")
    assert candidate.specimen_orientation_quaternion_wxyz == pytest.approx(
        baseline.specimen_orientation_quaternion_wxyz
    )
    expected = build_sample_geometry_snapshot(baseline)
    observed = build_sample_geometry_snapshot(candidate)
    assert observed.atomic_numbers.size > 0
    np.testing.assert_array_equal(observed.atomic_numbers, expected.atomic_numbers)
    np.testing.assert_allclose(observed.atom_positions_nm, expected.atom_positions_nm)
    kwargs = dict(thickness_angstrom=10, field_of_view_angstrom=10)
    expected_atoms, expected_periods = build_cif_equilibrium_atoms(original, **kwargs)
    observed_atoms, observed_periods = build_cif_equilibrium_atoms(imported, **kwargs)
    np.testing.assert_array_equal(observed_atoms.numbers, expected_atoms.numbers)
    np.testing.assert_allclose(observed_atoms.positions, expected_atoms.positions)
    np.testing.assert_allclose(observed_periods, expected_periods)


def test_wheel_contains_reference_formats_with_original_bytes(tmp_path):
    """Build a temporary source copy, never the user's checkout or environment."""
    root = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    project.mkdir()
    for name in ("pyproject.toml", "LICENSE"):
        shutil.copyfile(root / name, project / name)
    shutil.copytree(root / "src", project / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    shutil.copytree(root / "configs", project / "configs")
    reference_dir = project / "configs" / "reference_samples"
    contents = (reference_dir / "Si.cif").read_bytes()
    filenames = [f"Test{index}{extension}" for index, extension in enumerate((".cif", ".CIF", ".mcif", ".MCIF"))]
    for name in filenames:
        (reference_dir / name).write_bytes(contents)
    completed = subprocess.run(
        [sys.executable, "-c", "from setuptools.build_meta import build_wheel; build_wheel('dist')"],
        cwd=project, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    wheel, = (project / "dist").glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        for name in filenames:
            member, = [p for p in archive.namelist() if p.endswith("/configs/reference_samples/" + name)]
            assert archive.read(member) == contents


def test_mcif_scan_sampling_hint_and_match_grid_action(qtbot, tmp_path, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QMessageBox
    from temsim.calculation_cache import calculation_signatures
    from temsim.detector.stem_signal import StemScanResult
    from temsim.gui.scan_panel import ScanControlView
    from temsim.optics.column import default_state
    from temsim.physics.stem_sampling import detector_sampling_report

    path = tmp_path / "sampling.mcif"
    shutil.copyfile(reference_catalog.get_reference_sample("si_110").cif_path, path)
    state = default_state()
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = str(path)
    state.sample.wave_grid_pixels = 256
    view = ScanControlView()
    qtbot.addWidget(view)
    view.set_state(state)
    warning = view._sample_scale_warning(
        sample=state.sample, pixel_nm=0.2, fov_x_nm=0.4, fov_y_nm=0.4,
    )
    assert "shortest CIF atom spacing" in warning
    assert "atomic columns are undersampled" in warning
    report = detector_sampling_report(
        {"bf": (0.0, 10.0), "df": (16.0, 112.0), "haadf": (60.0, 331.0)},
        maximum_angle_mrad=42.0, wavelength_angstrom=0.019687,
        requested_fov_angstrom=40.0, requested_grid_pixels=256,
        bandwidth_fraction=2 / 3, probe_semiangle_mrad=25.0,
    )
    x, y = np.meshgrid([-0.0001, 0.0001], [-0.0001, 0.0001])
    frame = StemScanResult(
        scan_x_um=x, scan_y_um=y,
        fractions={key: np.full((2, 2), 0.1) for key in ("haadf", "df", "bf")},
        detector_signals={},
        metrics={"model": "multislice_angle_resolved", "detector_sampling": report,
                 "sampling_state_signature": calculation_signatures(state)["stem"]},
    )
    view._set_stem_frame(frame)
    assert view.match_detector_sampling.isEnabled()
    before = state.to_dict()
    errors = []
    view.error.connect(errors.append)
    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Yes)
    qtbot.mouseClick(view.match_detector_sampling, Qt.MouseButton.LeftButton)
    assert errors == []
    assert state.sample.wave_grid_pixels == report["recommended_grid_pixels"]
    after = state.to_dict()
    after["sample"]["wave_grid_pixels"] = before["sample"]["wave_grid_pixels"]
    assert after == before
    assert view._stem_frame is frame
    assert not view.match_detector_sampling.isEnabled()
