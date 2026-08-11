from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.physics.stem_wave_imaging import (
    PhysicalAngularDetector,
    integrate_angular_intensity,
)
from temsim.physics.wave_imaging import prepare_specimen_potentials
from temsim.specimen.geometry import (
    build_sample_geometry_snapshot,
    quaternion_from_zone_axes,
    quaternion_to_matrix,
)
from temsim.specimen.presets import load_specimen_preset
from temsim.specimen.atomistic import (
    atomistic_capability,
    build_cif_equilibrium_atoms,
)
from temsim.specimen.virtual import (
    build_virtual_angular_distribution,
    physical_screened_rutherford_probability,
    virtual_density_at_scan,
)


def test_zone_axis_and_in_plane_axis_share_one_right_handed_orientation():
    cell = np.diag((0.4, 0.5, 0.6))
    quaternion = quaternion_from_zone_axes(cell, (1, 1, 0), (1, -1, 0))
    rotation = quaternion_to_matrix(quaternion)
    zone = np.asarray((1, 1, 0)) @ cell
    in_plane = np.asarray((1, -1, 0)) @ cell
    zone /= np.linalg.norm(zone)
    in_plane -= np.dot(in_plane, zone) * zone
    in_plane /= np.linalg.norm(in_plane)

    assert rotation @ zone == pytest.approx((0.0, 0.0, 1.0), abs=1.0e-12)
    assert rotation @ in_plane == pytest.approx((1.0, 0.0, 0.0), abs=1.0e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_geometry_snapshot_reports_fov_roi_and_vacuum_outside_finite_sample():
    sample = default_state().sample
    sample.size_x_nm = 10.0
    sample.size_y_nm = 8.0
    scan_x = np.asarray(((-7.0e-3, 7.0e-3), (-7.0e-3, 7.0e-3)))
    scan_y = np.asarray(((-2.0e-3, -2.0e-3), (2.0e-3, 2.0e-3)))

    snapshot = build_sample_geometry_snapshot(
        sample,
        scan_x_um=scan_x,
        scan_y_um=scan_y,
        probe_padding_nm=1.0,
        load_atoms=False,
    )

    assert snapshot.scan_fov_bounds_nm == pytest.approx((-7.0, 7.0, -2.0, 2.0))
    assert snapshot.calculation_roi_bounds_nm == pytest.approx((-8.0, 8.0, -3.0, 3.0))
    assert any("vacuum" in warning for warning in snapshot.warnings)


def test_virtual_regions_and_grayscale_map_leave_outside_as_vacuum(tmp_path):
    sample = default_state().sample
    sample.specimen_mode = "virtual"
    sample.size_x_nm = 20.0
    sample.size_y_nm = 20.0
    density_path = tmp_path / "density.npy"
    np.save(density_path, np.asarray(((0.0, 1.0), (0.0, 1.0))))
    sample.virtual_regions = [
        {
            "name": "mapped island",
            "kind": "map",
            "enabled": True,
            "density": 0.8,
            "centre_x_nm": 0.0,
            "centre_y_nm": 0.0,
            "size_x_nm": 10.0,
            "size_y_nm": 10.0,
            "map_path": str(density_path),
        }
    ]
    sample.virtual_probe_convolution_enabled = False
    scan_x = np.asarray(((-20.0, -4.0, 4.0, 20.0),)) * 1.0e-3
    scan_y = np.zeros_like(scan_x)

    density = virtual_density_at_scan(sample, scan_x, scan_y)

    assert density[0, 0] == 0.0
    assert density[0, -1] == 0.0
    assert density[0, 1] < density[0, 2]
    assert density.max() <= 0.8


def test_absolute_virtual_probabilities_conserve_and_reject_overbooking():
    sample = default_state().sample
    sample.virtual_interactions = [
        {
            "name": "spots",
            "kind": "diffraction_spots",
            "enabled": True,
            "probability": 0.25,
            "angle_mrad": 8.0,
            "spot_count": 4,
        },
        {
            "name": "lost",
            "kind": "absorption",
            "enabled": True,
            "probability": 0.15,
        },
    ]
    distribution = build_virtual_angular_distribution(sample, beam_energy_kv=300.0)

    assert distribution.scattered_probability == pytest.approx(0.25)
    assert distribution.absorbed_probability == pytest.approx(0.15)
    assert distribution.direct_probability == pytest.approx(0.60)
    assert distribution.total_probability == pytest.approx(1.0)

    sample.virtual_interactions[1]["probability"] = 0.8
    with pytest.raises(ValueError, match="above one"):
        build_virtual_angular_distribution(sample, beam_energy_kv=300.0)


def test_screened_rutherford_uses_solid_angle_integral_and_poisson_probability():
    low_density_probability, cross_section = physical_screened_rutherford_probability(
        atomic_number=14,
        areal_density_atoms_nm2=1.0,
        beam_energy_kv=300.0,
        screening_angle_mrad=5.0,
        minimum_angle_mrad=20.0,
        maximum_angle_mrad=100.0,
    )
    high_density_probability, same_cross_section = physical_screened_rutherford_probability(
        atomic_number=14,
        areal_density_atoms_nm2=10.0,
        beam_energy_kv=300.0,
        screening_angle_mrad=5.0,
        minimum_angle_mrad=20.0,
        maximum_angle_mrad=100.0,
    )

    assert cross_section > 0.0
    assert same_cross_section == pytest.approx(cross_section)
    assert 0.0 < low_density_probability < high_density_probability < 1.0


def test_physical_detector_mask_retains_signed_anisotropic_transfer():
    class RectangularDetector:
        @staticmethod
        def hit_mask(x_mm, y_mm):
            return (np.abs(x_mm) <= 1.0) & (np.abs(y_mm) <= 0.25)

    detector = PhysicalAngularDetector(
        key="rect",
        detector=RectangularDetector(),
        sample_to_detector_m_per_rad=np.asarray(((1.0, 0.5), (-0.25, 2.0))),
        inner_mrad=0.0,
        outer_mrad=2.0,
    ).validate()
    angle_x = np.asarray(((0.0, 1.0), (0.0, -1.0)))
    angle_y = np.asarray(((0.0, 0.0), (0.2, 0.0)))
    radius = np.hypot(angle_x, angle_y)
    intensity = np.ones_like(radius)

    fractions, masks, uncollected = integrate_angular_intensity(
        intensity,
        radius,
        (detector,),
        angle_x_mrad=angle_x,
        angle_y_mrad=angle_y,
    )

    expected = detector.acceptance_mask(angle_x, angle_y)
    assert np.array_equal(masks["rect"], expected)
    assert fractions["rect"] == pytest.approx(np.mean(expected))
    assert fractions["rect"] + uncollected == pytest.approx(1.0)


def test_atomic_roi_fully_outside_finite_sample_is_explicit_vacuum():
    state = default_state()
    state.sample.size_x_nm = 10.0
    state.sample.size_y_nm = 10.0
    state.sample.cif_path = "this-file-must-not-be-opened.cif"
    prepared = prepare_specimen_potentials(
        state,
        load_specimen_preset("vacuum"),
        field_of_view_angstrom_override=20.0,
        calculation_roi_centre_nm=(100.0, 100.0),
        calculation_roi_bounds_nm=(99.0, 101.0, 99.0, 101.0),
    )

    assert prepared.metrics["potential_model"] == "finite_sample_vacuum_outside"
    assert np.count_nonzero(prepared.mean_projected_potential_v_angstrom) == 0


def test_cif_builder_generates_only_roi_neighbourhood_for_macroscopic_sample(
    tmp_path,
):
    if not atomistic_capability().available:
        pytest.skip("Atomistic CIF backend unavailable")
    from ase import Atoms
    from ase.io import write

    path = tmp_path / "large-sample.cif"
    write(
        path,
        Atoms(
            symbols="Si",
            scaled_positions=((0.25, 0.25, 0.25),),
            cell=np.diag((5.0, 5.0, 5.0)),
            pbc=True,
        ),
    )

    atoms, _periods = build_cif_equilibrium_atoms(
        path,
        thickness_angstrom=10.0,
        field_of_view_angstrom=20.0,
        specimen_size_xy_angstrom=(1.0e7, 1.0e7),
        specimen_centre_xy_angstrom=(0.0, 0.0),
        calculation_roi_centre_xy_angstrom=(2.0e6, -3.0e6),
        rotation_deg_xyz=(10.0, 20.0, 30.0),
    )

    assert 0 < len(atoms) < 100_000
    assert atoms.cell.lengths() == pytest.approx((20.0, 20.0, 10.0))


def test_cif_structure_display_repeats_cell_builds_bonds_and_caps_only_rendering(
    tmp_path,
):
    if not atomistic_capability().available:
        pytest.skip("Atomistic CIF backend unavailable")
    from ase.build import bulk
    from ase.io import write

    path = tmp_path / "nacl.cif"
    unit = bulk("NaCl", "rocksalt", a=5.64)
    write(path, unit)
    sample = default_state().sample
    sample.cif_path = str(path)
    sample.size_x_nm = 20.0
    sample.size_y_nm = 20.0
    sample.thickness_nm = 10.0

    snapshot = build_sample_geometry_snapshot(
        sample,
        maximum_display_atoms=1_000,
    )

    assert snapshot.atomic_numbers.size > len(unit)
    assert set(np.unique(snapshot.atomic_numbers)) == {11, 17}
    assert snapshot.atom_bond_pairs.shape[1] == 2
    assert snapshot.atom_bond_pairs.shape[0] > 0
    assert max(snapshot.atom_bond_pairs.ravel()) < snapshot.atomic_numbers.size
    assert all(
        displayed <= requested
        for displayed, requested in zip(
            snapshot.atom_display_size_nm,
            (sample.size_x_nm, sample.size_y_nm, sample.thickness_nm),
        )
    )
    assert any("calculation ROI is unchanged" in text for text in snapshot.warnings)
    assert snapshot.atom_positions_nm.flags.writeable is False
    assert snapshot.atom_bond_pairs.flags.writeable is False
