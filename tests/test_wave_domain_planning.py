"""Bounded checks for padded wave windows; no full-column calculation."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.physics import wave_imaging
from temsim.physics.stem_wave_imaging import (
    AngularDetector,
    simulate_angle_resolved_stem,
)
from temsim.physics.wave_imaging import prepare_specimen_potentials
from temsim.physics.wave_sampling import plan_wave_sampling
from temsim.specimen.atomistic import atomistic_capability
from temsim.specimen.presets import load_specimen_preset


def _plan(**overrides):
    inputs = dict(
        reference_fov_angstrom=40.0,
        reference_pixels=256,
        requested_fov_angstrom=40.0,
        thickness_angstrom=10.0,
        target_slice_thickness_angstrom=2.0,
    )
    inputs.update(overrides)
    return plan_wave_sampling(**inputs)


def _small_wave_state():
    state = default_state()
    state.acceleration_enabled = False
    state.illumination_mode = "STEM"
    # A finite reference CIF now supplies matter. The planning tests remain
    # bounded to a 2 nm / 0.4 nm slab and a 64-pixel base grid.
    state.sample.specimen_mode = "reference"
    state.sample.reference_sample_key = "si_110"
    state.sample.envelope_shape = "rectangle"
    state.sample.size_x_nm = 2.0
    state.sample.size_y_nm = 2.0
    state.sample.thickness_nm = 0.4
    state.sample.wave_grid_pixels = 64
    state.sample.wave_field_of_view_angstrom = 40.0
    state.sample.wave_atomistic_enabled = True
    state.sample.wave_multislice_enabled = True
    state.sample.wave_slice_thickness_angstrom = 2.0
    state.probe_corrector_installed = False
    state.objective_lens.cs_mm = 0.0
    state.objective_lens.cc_mm = 0.0
    return state


def _waist_bundle(waist_nm, *, centre_nm=(0.0, 0.0)):
    """Five weighted 30 mrad rays; only a local coherent-probe fixture."""
    tx = np.asarray((0.0, 0.03, -0.03, 0.0, 0.0))
    ty = np.asarray((0.0, 0.0, 0.0, 0.03, -0.03))
    return SimpleNamespace(
        alive=np.ones(5, dtype=bool),
        ray_weight=np.full(5, 0.2),
        x=(centre_nm[0] * 1.0e-9 - tx * waist_nm * 1.0e-9)[None, :],
        y=(centre_nm[1] * 1.0e-9 - ty * waist_nm * 1.0e-9)[None, :],
        tx=tx[None, :],
        ty=ty[None, :],
    )


@pytest.mark.parametrize("requested_fov", (20.0, 40.0, 40.1, 80.0, 101.0))
def test_dynamic_window_preserves_or_improves_reference_sampling(requested_fov):
    plan = _plan(requested_fov_angstrom=requested_fov)

    assert plan.field_of_view_angstrom == requested_fov
    assert plan.pixels >= 256
    assert plan.sampling_angstrom <= 40.0 / 256
    assert plan.reference_sampling_angstrom == 40.0 / 256
    assert plan.pixels * plan.sampling_angstrom == pytest.approx(requested_fov)
    assert plan.expanded is (requested_fov > 40.0)
    if plan.expanded:
        assert plan.pixels % 2 == 0


def test_unchanged_odd_reference_grid_is_not_silently_rounded():
    plan = _plan(reference_pixels=33)

    assert plan.pixels == 33
    assert not plan.expanded
    with pytest.raises(FrozenInstanceError):
        plan.pixels = 64


def test_potential_estimate_includes_all_slices_configurations_and_mean():
    plan = _plan(
        requested_fov_angstrom=80.0,
        thickness_angstrom=9.0,
        configuration_count=3,
    )

    assert plan.estimated_potential_bytes == 512**2 * (4 * 5 * 3 + 8)
    analytic = _plan(
        requested_fov_angstrom=80.0,
        thickness_angstrom=9.0,
        configuration_count=3,
        atomistic=False,
    )
    assert analytic.estimated_potential_bytes == 512**2 * 8


def test_limit_rejects_instead_of_coarsening_or_clipping_the_window():
    required_bytes = 512**2 * (4 * 5 + 8)
    with pytest.raises(ValueError, match="potential storage alone"):
        _plan(requested_fov_angstrom=80.0, max_potential_bytes=required_bytes - 1)

    accepted = _plan(
        requested_fov_angstrom=80.0, max_potential_bytes=required_bytes
    )
    assert accepted.pixels == 512
    assert accepted.field_of_view_angstrom == 80.0


def test_vacuum_working_memory_is_guarded_even_when_potential_storage_fits():
    # Planning only: no 16,384-square NumPy array is allocated. Its vacuum
    # potential fits the 4 GiB potential cap, but its probe/FFT buffers do not.
    with pytest.raises(ValueError, match="wave working memory"):
        _plan(requested_fov_angstrom=2_560.0, atomistic=False)

    planned = _plan(
        requested_fov_angstrom=2_560.0,
        atomistic=False,
        max_working_bytes=512 * 1024**3,
    )
    assert planned.pixels == 16_384
    assert planned.estimated_potential_bytes == 2 * 1024**3
    assert planned.estimated_working_bytes == 16_384**2 * (1024 + 48 + 2 * 8)
    assert planned.estimated_working_bytes > 8 * 1024**3


@pytest.mark.parametrize(
    "override",
    (
        {"requested_fov_angstrom": np.inf},
        {"reference_fov_angstrom": 0.0},
        {"thickness_angstrom": -1.0},
        {"target_slice_thickness_angstrom": 0.0},
        {"reference_pixels": 31},
        {"reference_pixels": np.nan},
        {"configuration_count": 1.5},
        {"configuration_count": np.inf},
        {"max_working_bytes": np.nan},
    ),
)
def test_invalid_domain_inputs_are_rejected(override):
    with pytest.raises(ValueError):
        _plan(**override)


def test_finite_material_mask_keeps_the_full_padded_wave_window():
    state = _small_wave_state()
    original_settings = deepcopy(asdict(state.sample))
    prepared = prepare_specimen_potentials(
        state,
        load_specimen_preset("si_110"),
        field_of_view_angstrom_override=80.0,
        calculation_roi_bounds_nm=(-0.1, 0.1, -0.1, 0.1),
    )

    assert prepared.metrics["wave_window_bounds_nm"] == (-4.0, 4.0, -4.0, 4.0)
    assert prepared.metrics["atom_generation_bounds_nm"] == (-1.0, 1.0, -1.0, 1.0)
    assert prepared.mean_projected_potential_v_angstrom.shape == (128, 128)
    spacing = float(np.diff(prepared.x_angstrom)[0])
    assert spacing == pytest.approx(40.0 / 64)
    assert spacing * prepared.x_angstrom.size == pytest.approx(80.0)
    outside = (
        (np.abs(prepared.x_angstrom[None, :]) > 10.0)
        | (np.abs(prepared.y_angstrom[:, None]) > 10.0)
    )
    assert np.count_nonzero(prepared.mean_projected_potential_v_angstrom[outside]) == 0
    assert np.any(prepared.mean_projected_potential_v_angstrom[~outside] > 0.0)
    assert asdict(state.sample) == original_settings


def test_square_wave_window_can_intersect_matter_outside_the_scan_rectangle():
    state = _small_wave_state()
    # The short scan rectangle misses the specimen, but its 8 nm square FFT
    # window reaches back to it. It must not be replaced with all vacuum.
    prepared = prepare_specimen_potentials(
        state,
        load_specimen_preset("si_110"),
        field_of_view_angstrom_override=80.0,
        calculation_roi_centre_nm=(3.0, 0.0),
        calculation_roi_bounds_nm=(2.9, 3.1, -0.1, 0.1),
    )

    assert prepared.metrics["potential_model"] != "finite_sample_vacuum_outside"
    assert prepared.metrics["wave_window_bounds_nm"] == (-1.0, 7.0, -4.0, 4.0)
    assert np.any(prepared.mean_projected_potential_v_angstrom > 0.0)


def test_cif_crystal_phase_at_common_lab_positions_survives_roi_translation():
    import abtem

    state = _small_wave_state()
    state.sample.centre_x_nm = 3.125
    state.sample.centre_y_nm = -2.5
    preset = load_specimen_preset("si_110")
    step_nm = state.sample.wave_field_of_view_angstrom / state.sample.wave_grid_pixels * 0.1
    cx, cy = state.sample.centre_x_nm, state.sample.centre_y_nm

    def prepare_at(x_nm, y_nm):
        return prepare_specimen_potentials(
            state,
            preset,
            calculation_roi_centre_nm=(x_nm, y_nm),
            calculation_roi_bounds_nm=(x_nm - 0.1, x_nm + 0.1, y_nm - 0.1, y_nm + 0.1),
        )

    # The former analytic fixture used float64. Retain its strict translation
    # tolerance by building this small IAM fixture at the same precision;
    # ordinary abTEM float32 accumulation need not be bitwise shift invariant.
    with abtem.config.set({"precision": "float64"}):
        original = prepare_at(cx, cy)
        shifted = prepare_at(cx + step_nm, cy + step_nm)
    assert np.any(original.mean_projected_potential_v_angstrom > 0.0)
    # A one-pixel positive shift of the window exposes the same laboratory
    # positions at array indices one lower. Both crystal phase and finite
    # specimen masking must remain anchored to the specimen, not the window.
    np.testing.assert_allclose(
        original.mean_projected_potential_v_angstrom[1:, 1:],
        shifted.mean_projected_potential_v_angstrom[:-1, :-1],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(original.x_angstrom, shifted.x_angstrom)


def test_window_outside_real_specimen_is_full_vacuum_without_opening_cif(monkeypatch):
    state = _small_wave_state()
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = "must-not-be-read.cif"
    state.sample.wave_atomistic_enabled = True

    def unexpected_builder(*args, **kwargs):
        pytest.fail("A disjoint wave window must not construct any CIF atoms")

    monkeypatch.setattr(wave_imaging, "build_atomistic_potential_ensemble", unexpected_builder)
    prepared = prepare_specimen_potentials(
        state,
        load_specimen_preset("si_110"),
        field_of_view_angstrom_override=80.0,
        calculation_roi_centre_nm=(100.0, 100.0),
        calculation_roi_bounds_nm=(99.9, 100.1, 99.9, 100.1),
    )

    assert prepared.metrics["potential_model"] == "finite_sample_vacuum_outside"
    assert prepared.metrics["atom_generation_bounds_nm"] is None
    assert prepared.mean_projected_potential_v_angstrom.shape == (128, 128)
    assert not np.any(prepared.mean_projected_potential_v_angstrom)


def test_oversized_defocus_window_fails_before_atoms_without_mutating_prior_result(monkeypatch):
    state = _small_wave_state()
    preset = load_specimen_preset("si_110")
    previous = prepare_specimen_potentials(state, preset)
    previous_potential = previous.mean_projected_potential_v_angstrom.copy()
    previous_metrics = deepcopy(previous.metrics)
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = "must-not-be-read.cif"
    state.sample.wave_atomistic_enabled = True
    original_settings = deepcopy(asdict(state.sample))

    def unexpected_builder(*args, **kwargs):
        pytest.fail("Oversized wave-domain planning must precede atom generation")

    monkeypatch.setattr(wave_imaging, "build_atomistic_potential_ensemble", unexpected_builder)
    with pytest.raises(ValueError, match="retain.*sampling"):
        prepare_specimen_potentials(
            state,
            preset,
            field_of_view_angstrom_override=500_000.0,  # 50 micrometres
            calculation_roi_bounds_nm=(-25_000.0, 25_000.0, -25_000.0, 25_000.0),
        )

    assert asdict(state.sample) == original_settings
    assert np.array_equal(previous.mean_projected_potential_v_angstrom, previous_potential)
    assert previous.metrics == previous_metrics


def test_vacuum_stem_current_is_preserved_when_defocus_expands_the_domain():
    state = _small_wave_state()
    state.sample.inserted = False
    state.sample.wave_grid_pixels = 256
    scan = np.zeros((1, 1))
    results = [
        simulate_angle_resolved_stem(
            state,
            SimpleNamespace(incident=_waist_bundle(waist_nm)),
            (AngularDetector("all", 0.0, 50.0),),
            scan,
            scan,
        )
        for waist_nm in (0.0, -40.0, 40.0)
    ]

    assert results[0].metrics["grid_pixels"] == 256
    for result in results:
        assert result.metrics["pixel_size_angstrom"] <= 40.0 / 256 + 1.0e-12
        assert result.fractions["all"][0, 0] == pytest.approx(1.0, abs=1.0e-10)
        assert result.uncollected_fraction[0, 0] == pytest.approx(0.0, abs=1.0e-10)
        assert result.metrics["wave_intensity_conservation_within_0_1_percent"]
    for result in results[1:]:
        assert result.metrics["grid_pixels"] > results[0].metrics["grid_pixels"]
        assert result.metrics["field_of_view_angstrom"] > 40.0
        sampling = result.metrics["detector_sampling"]
        assert sampling["requested_grid_pixels"] == 256
        assert sampling["execution_grid_pixels"] == result.metrics["grid_pixels"]
        # Match detector sampling edits the *reference* grid, not the expanded
        # execution grid. Only a 32-pixel rounding block may differ here.
        assert sampling["recommended_grid_pixels"] <= (
            results[0].metrics["detector_sampling"]["recommended_grid_pixels"] + 32
        )
        next_plan = _plan(
            reference_pixels=sampling["recommended_grid_pixels"],
            requested_fov_angstrom=result.metrics["field_of_view_angstrom"],
            atomistic=False,
        )
        assert next_plan.pixels >= result.metrics["grid_pixels"]
        assert next_plan.pixels <= (
            result.metrics["grid_pixels"]
            * sampling["recommended_grid_pixels"] / 256 + 2
        )


def test_stem_preparation_uses_lab_beam_origin_minus_baseline_scan_offset():
    state = _small_wave_state()
    state.sample.inserted = False
    state.sample.wave_grid_pixels = 256
    result = simulate_angle_resolved_stem(
        state,
        SimpleNamespace(incident=_waist_bundle(0.0, centre_nm=(3.0, -4.0))),
        (AngularDetector("all", 0.0, 50.0),),
        np.asarray([[0.0002]]),  # 0.2 nm nominal scan X
        np.asarray([[-0.0001]]),  # -0.1 nm nominal scan Y
        baseline_scan_offset_um=(0.001, -0.002),
    )

    assert result.metrics["specimen_calculation_roi_centre_nm"] == pytest.approx((2.2, -2.1))
    assert result.metrics["specimen_wave_window_bounds_nm"] == pytest.approx((0.2, 4.2, -4.1, -0.1))
    assert result.fractions["all"][0, 0] == pytest.approx(1.0, abs=1.0e-10)


def test_finite_silicon_cif_stem_responds_to_focus_with_a_padded_wave_domain(tmp_path):
    """Real small CPU multislice, not a full-column or experimental reference."""
    if not atomistic_capability().available:
        pytest.skip("Atomistic CIF backend unavailable")
    from ase.build import bulk
    from ase.io import write

    # Synthetic fixture only; no user CIF is read or changed. The conventional
    # diamond cell contains eight atoms before finite-envelope repetition.
    cif_path = tmp_path / "finite-silicon.cif"
    write(cif_path, bulk("Si", "diamond", a=5.43, cubic=True))
    state = _small_wave_state()
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = str(cif_path)
    state.sample.wave_atomistic_enabled = True
    state.sample.wave_grid_pixels = 256
    scan_x, scan_y = np.meshgrid(
        np.asarray((-0.0001, 0.0, 0.0001)),
        np.asarray((-0.0001, 0.0, 0.0001)),
        indexing="xy",
    )
    detectors = (AngularDetector("bf", 0.0, 20.0), AngularDetector("df", 20.0, 40.0))
    focused, defocused = (
        simulate_angle_resolved_stem(
            state,
            SimpleNamespace(incident=_waist_bundle(waist_nm)),
            detectors,
            scan_x,
            scan_y,
        )
        for waist_nm in (0.0, 30.0)
    )

    for result in (focused, defocused):
        assert result.metrics["specimen_atomistic_applied"]
        assert 0 < result.metrics["specimen_atom_count"] < 1_000
        assert result.metrics["specimen_atom_generation_bounds_nm"] == (-1.0, 1.0, -1.0, 1.0)
        # The fixed 2/3 anti-alias bandwidth can discard high-angle material
        # scattering. Bound that reported loss; do not silently renormalise a
        # failing 0.1% conservation flag into a successful one.
        relative_change = result.metrics["specimen_maximum_relative_intensity_change"]
        assert np.isfinite(relative_change)
        assert 0.0 <= relative_change < 0.005
        assert result.metrics["wave_intensity_conservation_within_0_1_percent"] == (
            relative_change <= 0.001
        )
        assert result.metrics["specimen_final_integrated_intensity"] <= (
            result.metrics["specimen_initial_integrated_intensity"] + 1.0e-10
        )
        assert result.metrics["pixel_size_angstrom"] <= 40.0 / 256 + 1.0e-12
        for image in result.fractions.values():
            assert image.shape == (3, 3)
            assert np.all(np.isfinite(image))
            assert np.all(image >= 0.0)
            assert np.any(image > 0.0)
        total = sum(result.fractions.values()) + result.uncollected_fraction
        np.testing.assert_allclose(total, 1.0, atol=1.0e-10)
    assert defocused.metrics["grid_pixels"] > focused.metrics["grid_pixels"]
    assert defocused.metrics["field_of_view_angstrom"] > focused.metrics["field_of_view_angstrom"]
    # Defocus changes the physically propagated signal; no forced BF/DF
    # contrast sign or golden image is imposed on this thin finite crystal.
    assert any(
        not np.allclose(focused.fractions[key], defocused.fractions[key], atol=1.0e-6)
        for key in ("bf", "df")
    )
