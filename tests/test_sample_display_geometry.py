"""Full specimen, local material, and capped atom rendering stay distinct."""

from dataclasses import asdict

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.specimen.atomistic import atomistic_capability
from temsim.specimen.geometry import (
    build_sample_geometry_snapshot,
    quaternion_from_zone_axes,
)


def _sample():
    sample = default_state().sample
    sample.specimen_mode = "atomic"
    sample.envelope_shape = "disk"
    sample.size_x_nm = sample.size_y_nm = 10.0
    sample.thickness_nm = 10.0
    return sample


@pytest.fixture
def silicon_cif(tmp_path):
    if not atomistic_capability().available:
        pytest.skip("Atomistic CIF backend unavailable")
    from ase.build import bulk
    from ase.io import write

    path = tmp_path / "display-only-silicon.cif"
    unit = bulk("Si", "diamond", a=5.43, cubic=True)
    write(path, unit)
    return path, np.asarray(unit.cell.array)


def test_local_material_metadata_does_not_replace_the_full_physical_envelope():
    sample = _sample()
    snapshot = build_sample_geometry_snapshot(
        sample,
        scan_x_um=np.asarray(((-0.00031, 0.00031),)),
        scan_y_um=np.asarray(((-0.00031, 0.00031),)),
        load_atoms=False,
    )

    assert snapshot.size_nm == (10.0, 10.0, 10.0)
    assert snapshot.centre_nm == (0.0, 0.0, 0.0)
    assert snapshot.envelope_shape == "disk"
    assert snapshot.local_material_bounds_nm == pytest.approx(
        (-0.31, 0.31, -0.31, 0.31, -5.0, 5.0)
    )
    assert snapshot.atom_display_size_nm is None
    assert not snapshot.atom_display_capped


def test_explicit_wave_roi_overrides_scan_padding_and_clips_only_material_bounds():
    sample = _sample()
    sample.centre_x_nm, sample.centre_y_nm = 2.0, -3.0
    override = (-5.0, 4.0, -9.0, -1.0)
    snapshot = build_sample_geometry_snapshot(
        sample,
        scan_x_um=np.asarray(((0.0, 0.0002),)),
        scan_y_um=np.asarray(((0.0, 0.0002),)),
        probe_padding_nm=100.0,
        calculation_roi_bounds_nm_override=override,
        load_atoms=False,
    )

    assert snapshot.calculation_roi_bounds_nm == override
    assert snapshot.scan_fov_bounds_nm == pytest.approx((0.0, 0.2, 0.0, 0.2))
    assert snapshot.local_material_bounds_nm == (-3.0, 4.0, -8.0, -1.0, -5.0, 5.0)
    assert snapshot.centre_nm == (2.0, -3.0, 0.0)
    assert snapshot.size_nm == (10.0, 10.0, 10.0)


@pytest.mark.parametrize(
    "bounds",
    ((0.0, 1.0, 0.0), (0.0, np.nan, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 2.0, 1.0)),
)
def test_invalid_explicit_calculation_bounds_are_rejected(bounds):
    with pytest.raises(ValueError, match="Calculation ROI bounds"):
        build_sample_geometry_snapshot(
            _sample(), calculation_roi_bounds_nm_override=bounds, load_atoms=False
        )


def test_no_calculation_retains_full_structural_request_without_claiming_a_roi():
    snapshot = build_sample_geometry_snapshot(_sample(), load_atoms=False)

    assert snapshot.calculation_roi_bounds_nm is None
    assert snapshot.scan_fov_bounds_nm is None
    assert snapshot.local_material_bounds_nm == (-5.0, 5.0, -5.0, 5.0, -5.0, 5.0)


def test_disk_corner_outside_material_never_opens_cif_or_creates_atoms():
    sample = _sample()
    sample.cif_path = "this-file-must-not-be-opened.cif"
    snapshot = build_sample_geometry_snapshot(
        sample,
        calculation_roi_bounds_nm_override=(4.0, 4.5, 4.0, 4.5),
    )

    assert snapshot.calculation_roi_bounds_nm == (4.0, 4.5, 4.0, 4.5)
    assert snapshot.local_material_bounds_nm is None
    assert snapshot.atomic_numbers.size == 0
    assert snapshot.atom_display_size_nm is None
    assert not snapshot.atom_display_capped
    assert any("vacuum only" in warning for warning in snapshot.warnings)


def test_zero_thickness_has_no_local_material_volume():
    sample = _sample()
    sample.thickness_nm = 0.0
    sample.cif_path = "this-file-must-not-be-opened.cif"
    snapshot = build_sample_geometry_snapshot(sample)

    assert snapshot.size_nm == (10.0, 10.0, 0.0)
    assert snapshot.local_material_bounds_nm is None
    assert snapshot.atomic_numbers.size == 0


def test_display_cap_keeps_uncapped_material_bounds_and_sample_settings(silicon_cif):
    path, _cell = silicon_cif
    sample = _sample()
    sample.cif_path = str(path)
    before = asdict(sample)
    full_roi = (-4.0, 4.0, -4.0, 4.0)
    capped = build_sample_geometry_snapshot(
        sample,
        calculation_roi_bounds_nm_override=full_roi,
        maximum_display_atoms=100,
    )

    assert capped.calculation_roi_bounds_nm == full_roi
    assert capped.local_material_bounds_nm == (*full_roi, -5.0, 5.0)
    assert capped.size_nm == (10.0, 10.0, 10.0)
    assert capped.atom_display_capped
    assert 0 < capped.atomic_numbers.size < 500
    assert np.all(np.asarray(capped.atom_display_size_nm) < (8.0, 8.0, 10.0))
    assert asdict(sample) == before
    assert not capped.atom_positions_nm.flags.writeable


def test_rotated_cif_atoms_are_clipped_to_lab_disk_not_a_rotated_or_square_box(silicon_cif):
    path, cell = silicon_cif
    sample = _sample()
    sample.cif_path = str(path)
    sample.size_x_nm = sample.size_y_nm = 2.0
    sample.thickness_nm = 1.0
    sample.centre_x_nm, sample.centre_y_nm = 3.0, -2.0
    sample.specimen_orientation_quaternion_wxyz = quaternion_from_zone_axes(
        cell, (1, 1, 0), (1, -1, 0)
    )
    snapshot = build_sample_geometry_snapshot(sample, maximum_display_atoms=2_500)

    assert snapshot.local_material_bounds_nm == (2.0, 4.0, -3.0, -1.0, -0.5, 0.5)
    assert snapshot.atom_display_size_nm == (2.0, 2.0, 1.0)
    assert not snapshot.atom_display_capped
    assert snapshot.atomic_numbers.size > 0
    # Atom positions already contain the lattice rotation and are expressed
    # relative to the sample centre in laboratory axes. No second rotation is
    # appropriate for these positions or the physical envelope.
    xyz = snapshot.atom_positions_nm
    assert np.all(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 <= 1.0 + 1.0e-10)
    assert np.all(np.abs(xyz[:, 2]) <= 0.5 + 1.0e-10)


def test_atoms_remain_inside_an_off_centre_requested_local_window(silicon_cif):
    path, _cell = silicon_cif
    sample = _sample()
    sample.cif_path = str(path)
    sample.size_x_nm = sample.size_y_nm = 2.0
    sample.thickness_nm = 1.0
    sample.centre_x_nm, sample.centre_y_nm = 3.0, -2.0
    roi = (3.2, 3.9, -2.3, -1.7)
    snapshot = build_sample_geometry_snapshot(
        sample, calculation_roi_bounds_nm_override=roi, maximum_display_atoms=2_500
    )

    assert snapshot.local_material_bounds_nm == (*roi, -0.5, 0.5)
    assert not snapshot.atom_display_capped
    assert snapshot.atomic_numbers.size > 0
    lab = snapshot.atom_positions_nm + np.asarray(snapshot.centre_nm)
    lower = np.asarray(snapshot.local_material_bounds_nm)[::2]
    upper = np.asarray(snapshot.local_material_bounds_nm)[1::2]
    assert np.all(lab >= lower - 1.0e-9)
    assert np.all(lab <= upper + 1.0e-9)
    assert np.all((lab[:, 0] - 3.0)**2 + (lab[:, 1] + 2.0)**2 <= 1.0 + 1.0e-10)
