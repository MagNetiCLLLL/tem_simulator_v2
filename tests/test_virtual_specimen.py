import math

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.optics.column import default_state
from temsim.physics.simulation import run
from temsim.specimen.atomistic import (
    atomistic_capability,
    build_atomistic_potential_ensemble,
    build_cif_equilibrium_atoms,
)
from temsim.specimen.presets import load_specimen_preset
from temsim.specimen.virtual import virtual_scattering_branches


def test_virtual_specimen_builds_explicit_spots_and_isotropic_ring():
    sample = default_state().sample
    sample.specimen_mode = "virtual"
    sample.virtual_diffraction_angle_mrad = 6.0
    sample.virtual_diffraction_azimuth_deg = 90.0
    sample.virtual_diffraction_relative_weight = 0.5
    sample.virtual_scattering_angle_mrad = 20.0
    sample.virtual_scattering_relative_weight = 0.8
    sample.virtual_scattering_azimuth_samples = 4

    branches = virtual_scattering_branches(sample)

    assert [branch.kind for branch in branches] == [
        "transmitted",
        "diffraction_spot",
        "diffraction_spot",
        "isotropic_ring",
        "isotropic_ring",
        "isotropic_ring",
        "isotropic_ring",
    ]
    assert branches[1].kick_x_rad == pytest.approx(0.0, abs=1.0e-15)
    assert branches[1].kick_y_rad == pytest.approx(6.0e-3)
    ring = branches[3:]
    assert sum(branch.relative_weight for branch in ring) == pytest.approx(0.8)
    assert {
        round(math.hypot(branch.kick_x_rad, branch.kick_y_rad) * 1.0e3, 12)
        for branch in ring
    } == {20.0}


def test_virtual_specimen_rejects_non_paraxial_user_angle():
    sample = default_state().sample
    sample.specimen_mode = "virtual"
    sample.virtual_scattering_angle_mrad = 201.0

    with pytest.raises(ValueError, match="paraxial"):
        virtual_scattering_branches(sample)


def test_ray_simulation_uses_virtual_channels_in_both_transverse_axes():
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    layout = apply_physical_layout_to_state(state)
    state.acceleration_enabled = False
    state.acceleration_backend = "CPU"
    state.step_mm = 5.0
    state.history_step_mm = 5.0
    state.electron_gun.emitter.ray_count = 9
    state.sample.specimen_mode = "virtual"
    state.sample.diffraction_enabled = True
    state.sample.virtual_diffraction_angle_mrad = 5.0
    state.sample.virtual_diffraction_azimuth_deg = 90.0
    state.sample.virtual_scattering_angle_mrad = 12.0
    state.sample.virtual_scattering_azimuth_samples = 4

    simulation = run(state, resolved_layout=layout)

    assert simulation.metrics["specimen_mode"] == "virtual"
    assert simulation.metrics["sample_scattering_model"] == (
        "user_defined_virtual_angular_channels"
    )
    assert {
        "000",
        "virtual_+g",
        "virtual_-g",
        "virtual_ring_001",
        "virtual_ring_002",
        "virtual_ring_003",
        "virtual_ring_004",
    } == set(simulation.branches)
    plus = simulation.branches["virtual_+g"]
    direct = simulation.branches["000"]
    assert np.mean(plus.ty[0] - direct.ty[0]) == pytest.approx(
        5.0e-3,
        rel=1.0e-6,
        abs=1.0e-9,
    )


def test_specimen_mode_and_cif_path_round_trip():
    state = default_state()
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = "example.cif"
    state.sample.specimen_rotation_x_deg = 12.5
    state.sample.specimen_rotation_y_deg = -3.25
    state.sample.specimen_rotation_z_deg = 91.0

    restored = type(state).from_dict(state.to_dict())

    assert restored.sample.specimen_mode == "atomic"
    assert restored.sample.cif_path == "example.cif"
    assert (
        restored.sample.specimen_rotation_x_deg,
        restored.sample.specimen_rotation_y_deg,
        restored.sample.specimen_rotation_z_deg,
    ) == pytest.approx((12.5, -3.25, 91.0))


def test_cif_import_builds_exact_finite_rotated_specimen_box(tmp_path):
    if not atomistic_capability().available:
        pytest.skip("Atomistic CIF backend unavailable")
    from ase import Atoms
    from ase.io import write

    path = tmp_path / "two_atom.cif"
    atoms = Atoms(
        symbols="Si2",
        scaled_positions=((0.15, 0.2, 0.25), (0.65, 0.7, 0.75)),
        cell=np.diag((5.0, 5.0, 5.0)),
        pbc=True,
    )
    write(path, atoms)

    imported, source_periods = build_cif_equilibrium_atoms(
        path,
        thickness_angstrom=8.0,
        field_of_view_angstrom=10.0,
        rotation_deg_xyz=(10.0, 20.0, 30.0),
    )

    assert len(imported) > 0
    assert imported.cell.lengths() == pytest.approx((10.0, 10.0, 8.0))
    assert not np.any(imported.pbc)
    assert np.all(imported.positions >= -1.0e-9)
    assert np.all(
        imported.positions
        <= np.asarray((10.0, 10.0, 8.0)) + 1.0e-9
    )
    assert all(value > 0.0 for value in source_periods)


def test_custom_cif_builds_finite_slice_iam_potential(tmp_path):
    if not atomistic_capability().available:
        pytest.skip("Atomistic CIF backend unavailable")
    from ase import Atoms
    from ase.io import write

    path = tmp_path / "silicon.cif"
    write(
        path,
        Atoms(
            symbols="Si",
            scaled_positions=((0.25, 0.25, 0.25),),
            cell=np.diag((4.0, 4.0, 4.0)),
            pbc=True,
        ),
    )

    ensemble = build_atomistic_potential_ensemble(
        load_specimen_preset("vacuum"),
        thickness_angstrom=4.0,
        field_of_view_angstrom=8.0,
        pixels=32,
        target_slice_thickness_angstrom=2.0,
        frozen_phonon_enabled=False,
        frozen_phonon_configurations=1,
        thermal_sigma_override_angstrom=0.0,
        thermal_seed=7,
        cif_path=str(path),
        rotation_deg_xyz=(0.0, 0.0, 15.0),
    )

    assert ensemble.source_kind == "cif"
    assert ensemble.source_path == str(path.resolve())
    assert ensemble.rotation_deg_xyz == pytest.approx((0.0, 0.0, 15.0))
    assert ensemble.slice_count >= 1
    assert ensemble.mean_projected_potential_v_angstrom.shape == (
        ensemble.grid_shape_yx
    )
    assert np.all(np.isfinite(ensemble.mean_projected_potential_v_angstrom))
