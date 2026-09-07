"""Finite CIF enumeration against the previous small covering-cube reference."""

from types import SimpleNamespace

import numpy as np
import pytest

from temsim.specimen import atomistic


@pytest.fixture
def finite_cif(monkeypatch, tmp_path):
    ase = pytest.importorskip("ase")
    import ase.io

    unit = ase.Atoms(
        numbers=[14, 32], positions=[[0, 0, 0], [1.0, 1.5, 2.0]],
        cell=np.diag([4.0, 5.0, 6.0]), pbc=True,
    )
    path = tmp_path / "fixture.cif"
    path.touch()
    monkeypatch.setattr(ase.io, "read", lambda _path: unit.copy())
    backend = SimpleNamespace(orthogonalize_cell=lambda value, **_kw: value)
    monkeypatch.setattr(atomistic, "_require_backend", lambda: (backend, ase.Atoms))
    return path, unit


def _old_reference(unit, *, fov, size, thickness, rotation, sample_centre, roi_centre):
    """Independent original ASE.repeat/crop algorithm, only for small cases."""
    periods = unit.cell.lengths()
    sample_centre = np.asarray(sample_centre)
    roi_centre = np.asarray(roi_centre)
    fov = np.broadcast_to(fov, (2,))
    size = np.asarray(size)
    intersection_min = np.maximum(sample_centre - size / 2, roi_centre - fov / 2)
    intersection_max = np.minimum(sample_centre + size / 2, roi_centre + fov / 2)
    cover = np.linalg.norm(np.r_[intersection_max - intersection_min, thickness]) + 2 * max(periods)
    repeats = tuple(int(value) for value in np.ceil(cover / periods) + 1)
    repeated = unit.repeat(repeats)
    repeated.wrap(eps=1e-12)
    source_centre = np.r_[0.5 * (intersection_min + intersection_max) - sample_centre, 0] @ rotation
    shift = np.rint(source_centre / periods) * periods
    positions = repeated.positions - (np.asarray(repeats) // 2) * periods + shift
    rotated = positions @ rotation.T
    local_xy = rotated[:, :2] + sample_centre - roi_centre
    keep = np.all(np.abs(rotated) <= np.r_[size, thickness] / 2 + 1e-9, axis=1)
    keep &= np.all(np.abs(local_xy) <= fov / 2 + 1e-9, axis=1)
    positions = np.column_stack((local_xy[keep] + fov / 2, rotated[keep, 2] + thickness / 2))
    return repeated.numbers[keep], positions


@pytest.mark.parametrize("angles", [(0, 0, 0), (45, 0, 0), (31, -22, 67), (89.9, 13, -101)])
@pytest.mark.parametrize("centres", [((0, 0), (0, 0)), ((8.2, -11.5), (13.8, -8.6))])
def test_chunked_sites_match_original_order_and_lattice_phase(finite_cif, angles, centres):
    path, unit = finite_cif
    rotation = atomistic._rotation_matrix_xyz(angles)
    common = dict(fov=(17.3, 13.1), size=(29.7, 24.4), thickness=8.7,
                  rotation=rotation, sample_centre=centres[0], roi_centre=centres[1])
    numbers, expected = _old_reference(unit, **common)
    atoms, _ = atomistic.build_cif_equilibrium_atoms(
        path, thickness_angstrom=common["thickness"], field_of_view_angstrom=common["fov"],
        specimen_size_xy_angstrom=common["size"], rotation_matrix=rotation,
        specimen_centre_xy_angstrom=centres[0], calculation_roi_centre_xy_angstrom=centres[1],
    )
    np.testing.assert_array_equal(atoms.numbers, numbers)
    np.testing.assert_allclose(atoms.positions, expected, rtol=0, atol=1e-10)


def test_vacuum_window_does_not_increase_material_sites(finite_cif):
    path, _ = finite_cif
    results = []
    for fov in (50.0, 500000.0):
        atoms, _ = atomistic.build_cif_equilibrium_atoms(
            path, thickness_angstrom=8, field_of_view_angstrom=fov,
            specimen_size_xy_angstrom=(20, 24), rotation_deg_xyz=(35, -22, 10),
        )
        results.append((atoms.numbers, atoms.positions - [fov / 2, fov / 2, 4]))
        np.testing.assert_array_equal(atoms.cell.lengths(), [fov, fov, 8])
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_allclose(results[0][1], results[1][1], rtol=0, atol=2e-11)


def test_disk_crops_actual_atoms_not_only_a_potential_pixel_mask(finite_cif):
    path, _ = finite_cif
    kwargs = dict(thickness_angstrom=12, field_of_view_angstrom=50,
                  specimen_size_xy_angstrom=(32, 24), rotation_deg_xyz=(11, 43, -7))
    box, _ = atomistic.build_cif_equilibrium_atoms(path, **kwargs)
    disk, _ = atomistic.build_cif_equilibrium_atoms(path, specimen_envelope_shape="disk", **kwargs)
    positions = box.positions[:, :2] - 25
    keep = np.sum((positions / [16, 12]) ** 2, axis=1) <= 1
    assert 0 < np.count_nonzero(keep) < len(box)
    np.testing.assert_array_equal(disk.numbers, box.numbers[keep])
    np.testing.assert_array_equal(disk.positions, box.positions[keep])


def test_far_off_centre_roi_preserves_crystal_phase(finite_cif):
    path, unit = finite_cif
    rotation = atomistic._rotation_matrix_xyz((13, 46, -33))
    centre = (100013.7, -200005.8)
    numbers, expected = _old_reference(
        unit, fov=(14, 13), size=(1e7, 1e7), thickness=11, rotation=rotation,
        sample_centre=(0, 0), roi_centre=centre,
    )
    atoms, _ = atomistic.build_cif_equilibrium_atoms(
        path, thickness_angstrom=11, field_of_view_angstrom=(14, 13),
        specimen_size_xy_angstrom=(1e7, 1e7), rotation_matrix=rotation,
        calculation_roi_centre_xy_angstrom=centre,
    )
    np.testing.assert_array_equal(atoms.numbers, numbers)
    np.testing.assert_allclose(atoms.positions, expected, rtol=0, atol=1e-9)


def test_exact_faces_and_small_remainder_batches(finite_cif, monkeypatch):
    path, unit = finite_cif
    monkeypatch.setattr(atomistic, "_CIF_ENUMERATION_CHUNK_ATOMS", 7)
    numbers, expected = _old_reference(unit, fov=16, size=(16, 16), thickness=12,
        rotation=np.eye(3), sample_centre=(0, 0), roi_centre=(0, 0))
    atoms, _ = atomistic.build_cif_equilibrium_atoms(path, thickness_angstrom=12,
        field_of_view_angstrom=16, specimen_size_xy_angstrom=(16, 16))
    np.testing.assert_array_equal(atoms.numbers, numbers)
    np.testing.assert_allclose(atoms.positions, expected, rtol=0, atol=1e-12)
    assert np.any(atoms.positions[:, 0] == 0)
    assert np.any(atoms.positions[:, 0] == 16)


def test_retained_atom_limit_not_covering_cube_estimate(finite_cif, monkeypatch):
    path, _ = finite_cif
    kwargs = dict(thickness_angstrom=10, field_of_view_angstrom=20,
                  specimen_size_xy_angstrom=(20, 20), rotation_deg_xyz=(31, 22, 47))
    baseline, _ = atomistic.build_cif_equilibrium_atoms(path, **kwargs)
    monkeypatch.setattr(atomistic, "MAX_CIF_SUPERCELL_ATOMS", len(baseline))
    exact, _ = atomistic.build_cif_equilibrium_atoms(path, **kwargs)
    np.testing.assert_array_equal(exact.positions, baseline.positions)
    monkeypatch.setattr(atomistic, "MAX_CIF_SUPERCELL_ATOMS", len(baseline) - 1)
    with pytest.raises(ValueError, match="retained atoms"):
        atomistic.build_cif_equilibrium_atoms(path, **kwargs)


def test_rotated_thin_100nm_box_does_not_repeat_a_cube(finite_cif, monkeypatch):
    path, _ = finite_cif
    from ase import Atoms
    monkeypatch.setattr(Atoms, "repeat", lambda *_a, **_k: pytest.fail("covering cube was repeated"))
    # This fixture's low-density cell isolates enumeration resource scaling;
    # the user's denser CIF can still exceed the retained-atom cap.
    atoms, _ = atomistic.build_cif_equilibrium_atoms(
        path, thickness_angstrom=100, field_of_view_angstrom=1000,
        specimen_size_xy_angstrom=(1000, 1000), rotation_deg_xyz=(0, 45, 22),
    )
    assert 1_500_000 < len(atoms) < 1_800_000
    assert np.all((atoms.positions >= -1e-9) & (atoms.positions <= [1000, 1000, 100] + np.zeros(3) + 1e-9))


@pytest.mark.parametrize("kwargs,match", [
    ({"calculation_roi_centre_xy_angstrom": (1000, 1000)}, "does not intersect"),
    ({"field_of_view_angstrom": 0.01, "thickness_angstrom": 0.01,
      "calculation_roi_centre_xy_angstrom": (0.1, 0.1)}, "contains no atomic sites"),
    ({"rotation_matrix": np.zeros((3, 3))}, "orthonormal"),
    ({"specimen_envelope_shape": "triangle"}, "envelope shape"),
])
def test_invalid_or_empty_roi(finite_cif, kwargs, match):
    path, _ = finite_cif
    options = dict(thickness_angstrom=10, field_of_view_angstrom=20,
                   specimen_size_xy_angstrom=(20, 20))
    options.update(kwargs)
    with pytest.raises(ValueError, match=match):
        atomistic.build_cif_equilibrium_atoms(path, **options)


def test_enumeration_work_guard_is_distinct_from_atom_storage(finite_cif, monkeypatch):
    path, _ = finite_cif
    monkeypatch.setattr(atomistic, "_MAX_CIF_ENUMERATION_COLUMNS", 1)
    with pytest.raises(ValueError, match="lattice-column work limit"):
        atomistic.build_cif_equilibrium_atoms(path, thickness_angstrom=10, field_of_view_angstrom=20)


def test_candidate_work_guard_bounds_rejected_sites(finite_cif, monkeypatch):
    path, _ = finite_cif
    monkeypatch.setattr(atomistic, "_MAX_CIF_ENUMERATION_CANDIDATES", 1)
    with pytest.raises(ValueError, match="candidate-site work limit"):
        atomistic.build_cif_equilibrium_atoms(path, thickness_angstrom=10, field_of_view_angstrom=20)


def test_coarse_finite_potential_sampling_is_rejected_before_backend_division():
    pytest.importorskip("abtem")
    from ase import Atoms
    atoms = Atoms("Si", positions=[[250000, 250000, 1]], cell=[500000, 500000, 2], pbc=False)
    with pytest.raises(ValueError, match="sampling is too coarse for Si"):
        atomistic._build_one_potential(atoms, gpts_xy=(256, 256), target_slice_thickness_angstrom=2)
