from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.physics import compute_backend
from temsim.physics.multislice import propagate_multislice
from temsim.physics.stem_wave_imaging import (
    AngularDetector,
    simulate_angle_resolved_stem,
)
from temsim.physics.wave_imaging import (
    interaction_constant_rad_per_v_angstrom,
    prepare_specimen_potentials,
    simulate_wave_image,
)
from temsim.specimen import atomistic
from temsim.specimen.atomistic import (
    AtomisticBackendUnavailable,
    atomistic_capability,
    build_atomistic_potential_ensemble,
    build_equilibrium_atoms,
)
from temsim.specimen.presets import load_specimen_preset


def _require_atomistic_backend():
    capability = atomistic_capability()
    if not capability.available:
        pytest.skip(capability.detail)


def _incident_bundle():
    return SimpleNamespace(
        alive=np.ones(5, dtype=bool),
        ray_weight=np.asarray([0.6, 0.1, 0.1, 0.1, 0.1]),
        x=np.zeros((1, 5), dtype=float),
        y=np.zeros((1, 5), dtype=float),
        tx=np.asarray([[0.0, 2.0e-3, -2.0e-3, 0.0, 0.0]]),
        ty=np.asarray([[0.0, 0.0, 0.0, 2.0e-3, -2.0e-3]]),
    )


def _small_atomistic_state():
    state = default_state()
    state.acceleration_enabled = False
    state.acceleration_backend = "CPU"
    state.sample.specimen_preset_key = "si_110"
    state.sample.thickness_nm = 0.4
    state.sample.wave_grid_pixels = 32
    state.sample.wave_field_of_view_angstrom = 8.0
    state.sample.wave_multislice_enabled = True
    state.sample.wave_atomistic_enabled = True
    state.sample.wave_slice_thickness_angstrom = 2.0
    return state


def test_atomistic_preset_owns_crystal_and_thermal_provenance():
    silicon = load_specimen_preset("si_110")
    gold = load_specimen_preset("au_001")

    assert silicon.atomistic is not None
    assert silicon.atomistic.zone_axis == (1, 1, 0)
    assert silicon.atomistic.thermal_sigma_angstrom == pytest.approx(0.085)
    assert "10.1107/S0108767391000375" in (
        silicon.atomistic.thermal_sigma_reference
    )
    assert gold.atomistic is not None
    assert gold.atomistic.zone_axis == (0, 0, 1)
    assert "10.1107/S0567739470000141" in (
        gold.atomistic.thermal_sigma_reference
    )


def test_oriented_silicon_supercell_preserves_diamond_number_density():
    _require_atomistic_backend()
    preset = load_specimen_preset("si_110")
    atoms, periods = build_equilibrium_atoms(
        preset,
        thickness_angstrom=4.0,
        field_of_view_angstrom=8.0,
    )

    expected_density = 8.0 / preset.atomistic.lattice_constant_angstrom**3
    assert len(atoms) / atoms.get_volume() == pytest.approx(
        expected_density, rel=1.0e-12
    )
    assert np.asarray(atoms.cell.array) == pytest.approx(
        np.diag(atoms.cell.lengths()), abs=1.0e-12
    )
    assert periods == pytest.approx(tuple(atoms.cell.lengths()))


def test_lobato_slice_units_match_abtem_transmission_function():
    _require_atomistic_backend()
    import abtem

    preset = load_specimen_preset("si_110")
    atoms, _ = build_equilibrium_atoms(
        preset,
        thickness_angstrom=4.0,
        field_of_view_angstrom=8.0,
    )
    requested_sampling = 8.0 / 32
    gpts_xy = tuple(
        max(16, int(round(length / requested_sampling)))
        for length in atoms.cell.lengths()[:2]
    )
    slices, thicknesses = atomistic._build_one_potential(
        atoms,
        gpts_xy=gpts_xy,
        target_slice_thickness_angstrom=2.0,
    )
    built = abtem.Potential(
        atoms,
        gpts=gpts_xy,
        slice_thickness=2.0,
        parametrization="lobato",
        projection="finite",
        periodic=True,
        device="cpu",
    ).build(lazy=False)
    reference = np.asarray(
        built.transmission_function(energy=200_000.0).array
    ).transpose(0, 2, 1)
    reference = np.fft.fftshift(reference, axes=(-2, -1))
    sigma = interaction_constant_rad_per_v_angstrom(200.0)

    assert slices.shape[0] == thicknesses.size
    assert np.exp(1j * sigma * slices) == pytest.approx(
        reference, rel=2.0e-6, abs=2.0e-7
    )


def test_frozen_phonon_seed_is_reproducible_and_changes_configurations():
    _require_atomistic_backend()
    preset = load_specimen_preset("si_110")
    parameters = dict(
        thickness_angstrom=4.0,
        field_of_view_angstrom=8.0,
        pixels=32,
        target_slice_thickness_angstrom=2.0,
        frozen_phonon_enabled=True,
        frozen_phonon_configurations=2,
        thermal_sigma_override_angstrom=0.0,
    )
    first = build_atomistic_potential_ensemble(
        preset, thermal_seed=321, **parameters
    )
    repeated = build_atomistic_potential_ensemble(
        preset, thermal_seed=321, **parameters
    )
    changed = build_atomistic_potential_ensemble(
        preset, thermal_seed=322, **parameters
    )

    assert all(
        np.array_equal(left, right)
        for left, right in zip(
            first.configurations_v_angstrom,
            repeated.configurations_v_angstrom,
        )
    )
    assert not np.array_equal(
        first.configurations_v_angstrom[0],
        changed.configurations_v_angstrom[0],
    )
    expected_mean = np.mean(
        [
            np.sum(configuration, axis=0, dtype=np.float64)
            for configuration in first.configurations_v_angstrom
        ],
        axis=0,
    )
    assert first.mean_projected_potential_v_angstrom == pytest.approx(
        expected_mean
    )


def test_atomistic_ensemble_rejects_unsafe_potential_storage_before_build():
    _require_atomistic_backend()
    preset = load_specimen_preset("si_110")

    with pytest.raises(ValueError, match="4 GiB"):
        build_atomistic_potential_ensemble(
            preset,
            thickness_angstrom=4.0,
            field_of_view_angstrom=8.0,
            pixels=20_000,
            target_slice_thickness_angstrom=2.0,
            frozen_phonon_enabled=True,
            frozen_phonon_configurations=64,
            thermal_sigma_override_angstrom=0.0,
            thermal_seed=1,
        )


def test_rectangular_atomic_slices_match_between_cpu_and_cuda():
    _require_atomistic_backend()
    if not compute_backend.cupy_capability().available:
        pytest.skip("CuPy CUDA backend unavailable")
    preset = load_specimen_preset("si_110")
    ensemble = build_atomistic_potential_ensemble(
        preset,
        thickness_angstrom=4.0,
        field_of_view_angstrom=8.0,
        pixels=32,
        target_slice_thickness_angstrom=2.0,
        frozen_phonon_enabled=False,
        frozen_phonon_configurations=1,
        thermal_sigma_override_angstrom=0.0,
        thermal_seed=0,
    )
    potential = ensemble.configurations_v_angstrom[0]
    ny, nx = potential.shape[-2:]
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, ny),
        np.linspace(-1.0, 1.0, nx),
        indexing="ij",
    )
    wave = np.exp(-(xx**2 + 1.3 * yy**2) + 0.2j * xx)
    wave /= np.linalg.norm(wave)
    common = dict(
        pixel_size_angstrom=(
            ensemble.sampling_angstrom_xy[1],
            ensemble.sampling_angstrom_xy[0],
        ),
        wavelength_angstrom=0.025079,
        interaction_constant_rad_per_v_angstrom=(
            interaction_constant_rad_per_v_angstrom(200.0)
        ),
        total_thickness_angstrom=None,
        target_slice_thickness_angstrom=2.0,
        slice_thicknesses_angstrom=ensemble.slice_thicknesses_angstrom,
        bandwidth_fraction=2.0 / 3.0,
    )
    cpu, _ = propagate_multislice(
        wave, potential, compute_backend="NumPy CPU", **common
    )
    gpu, diagnostics = propagate_multislice(
        wave, potential, compute_backend="CuPy CUDA", **common
    )

    relative_error = np.linalg.norm(gpu - cpu) / np.linalg.norm(cpu)
    assert diagnostics.compute_backend == "CuPy CUDA"
    assert relative_error < 5.0e-5


def test_tem_atomistic_frozen_phonons_average_intensities_on_rectangular_grid():
    _require_atomistic_backend()
    state = _small_atomistic_state()
    state.illumination_mode = "TEM"
    state.sample.wave_frozen_phonon_enabled = True
    state.sample.wave_frozen_phonon_configurations = 2
    state.sample.wave_frozen_phonon_seed = 77

    result = simulate_wave_image(
        state, SimpleNamespace(incident=_incident_bundle())
    )

    assert result.metrics["specimen_atomistic_applied"] is True
    assert result.metrics["specimen_frozen_phonon_applied"] is True
    assert result.metrics["specimen_model"] == (
        "atomistic_frozen_phonon_multislice"
    )
    assert result.metrics["displayed_intensity_average"] == (
        "incoherent frozen-phonon intensity mean"
    )
    assert result.image_intensity.shape == result.projected_potential_v_angstrom.shape
    assert result.image_intensity.shape[0] != result.image_intensity.shape[1]
    assert np.isfinite(
        result.metrics["image_configuration_relative_standard_error"]
    )
    assert result.metrics["image_configuration_relative_standard_error"] > 0.0


def test_stem_atomistic_frozen_phonons_average_detector_intensities():
    _require_atomistic_backend()
    state = _small_atomistic_state()
    state.illumination_mode = "STEM"
    state.sample.wave_frozen_phonon_enabled = True
    state.sample.wave_frozen_phonon_configurations = 2
    result = simulate_angle_resolved_stem(
        state,
        SimpleNamespace(incident=_incident_bundle()),
        (AngularDetector("bf", 0.0, 20.0),),
        np.zeros((1, 1)),
        np.zeros((1, 1)),
    )

    assert result.metrics["specimen_atomistic_applied"] is True
    assert result.metrics["specimen_model"] == (
        "atomistic_frozen_phonon_multislice"
    )
    assert result.metrics["displayed_intensity_average"] == (
        "incoherent frozen-phonon intensity mean"
    )
    relative_error = result.metrics[
        "detector_configuration_relative_standard_error"
    ]["bf"]
    assert np.isfinite(relative_error)
    assert relative_error >= 0.0


def test_missing_atomistic_backend_has_an_explicit_legacy_fallback(monkeypatch):
    state = _small_atomistic_state()
    preset = load_specimen_preset("si_110")

    def unavailable(*_args, **_kwargs):
        raise AtomisticBackendUnavailable("synthetic missing atomistic backend")

    monkeypatch.setattr(
        "temsim.physics.wave_imaging.build_atomistic_potential_ensemble",
        unavailable,
    )
    prepared = prepare_specimen_potentials(state, preset)

    assert prepared.metrics["atomistic_requested"] is True
    assert prepared.metrics["atomistic_applied"] is False
    assert prepared.metrics["potential_model"] == "analytic_projected_columns"
    assert "synthetic missing" in prepared.metrics[
        "atomistic_fallback_reason"
    ]
