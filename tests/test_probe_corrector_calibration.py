"""Accept nanoprobe correction on the actual source and physical sample plane."""

from types import SimpleNamespace

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.optics.probe_calibration import (
    IncidentProbeMeasurement,
    IncidentProbeModel,
    fit_probe_hexapoles,
)
from temsim.physics.core import hexapole_field_components


def _preset_state():
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    apply_operating_mode_pair(state, "nano_probe", "diffraction")
    return state


@pytest.fixture(scope="module")
def calibrated_probe():
    # All comparisons reuse identical emission positions, directions, energies
    # and weights. This avoids attributing source sampling changes to correction.
    numba = pytest.importorskip("numba")
    previous_threads = numba.get_num_threads()
    numba.set_num_threads(min(4, previous_threads))
    try:
        state = _preset_state()
        state.electron_gun.emitter.ray_count = 1000
        state.acceleration_enabled = True
        state.acceleration_backend = "Numba CPU"
        state.history_step_mm = 0.5
        state.step_mm = 0.05
        model = IncidentProbeModel(state)

        def trace(step, **options):
            # trace(step_mm=...) caps the state's step; update both so the
            # nominal 0.1 mm run really uses the coarser integration grid.
            state.step_mm = step
            return model.trace(step_mm=step, **options)

        corrected = trace(0.05)
        linear = trace(0.05, spherical=False, hexapoles=False)
        uncorrected = trace(0.05, spherical=True, hexapoles=False)
        coarse = trace(0.1)
        refined = trace(0.025)
        state.step_mm = 0.05
        yield SimpleNamespace(
            state=state, model=model, corrected=corrected, linear=linear,
            uncorrected=uncorrected, coarse=coarse, refined=refined,
        )
    finally:
        numba.set_num_threads(previous_threads)


def test_corrected_finite_source_probe_is_small_focused_and_nearly_round(calibrated_probe):
    state = calibrated_probe.state
    statistics = calibrated_probe.corrected.statistics
    assert statistics.radius_rms_m * 1.0e9 < 0.2
    assert 2.0 * statistics.radius_95_m * 1.0e9 < 0.7
    assert statistics.threefold_moment < 0.15
    assert abs(statistics.waist_offset_m) * 1.0e9 < 2.0
    assert statistics.convergence_95_mrad == pytest.approx(24.62489, abs=0.15)
    assert statistics.surviving_fraction > 0.5
    assert state.hp2_hexapole.enabled and state.hp1_hexapole.enabled
    assert state.hp2_hexapole.strength_m3 > 0.0
    assert state.hp1_hexapole.strength_m3 > 0.0
    assert state.objective_lens.cs_mm > 0.0
    normal, skew = hexapole_field_components(
        [state.hp2_hexapole.z_mm, state.hp1_hexapole.z_mm], state,
    )
    assert np.all(np.hypot(normal, skew) > 1.0e5)


def test_corrector_removes_nonlinear_error_without_losing_pupil_current(calibrated_probe):
    corrected = calibrated_probe.corrected
    linear = calibrated_probe.linear
    uncorrected = calibrated_probe.uncorrected
    np.testing.assert_array_equal(corrected.alive, linear.alive)
    np.testing.assert_array_equal(uncorrected.alive, linear.alive)
    np.testing.assert_array_equal(corrected.weights, linear.weights)
    mask = linear.alive
    weights = linear.weights[mask]
    weights = weights / weights.sum()

    def error_nm(measurement):
        # Absolute same-ray position differences retain centroid errors too;
        # defocusing, recentering or circularising the displayed spot cannot
        # pass this physical correction criterion.
        dx = measurement.x_m[mask] - linear.x_m[mask]
        dy = measurement.y_m[mask] - linear.y_m[mask]
        return float(np.sqrt(np.sum(weights * (dx * dx + dy * dy))) * 1.0e9)

    corrected_error = error_nm(corrected)
    uncorrected_error = error_nm(uncorrected)
    assert corrected_error < 0.01
    assert corrected_error < 0.1 * uncorrected_error


def test_probe_size_and_focus_survive_integration_step_refinement(calibrated_probe):
    coarse = calibrated_probe.coarse
    medium = calibrated_probe.corrected
    fine = calibrated_probe.refined
    np.testing.assert_array_equal(coarse.alive, fine.alive)
    np.testing.assert_array_equal(medium.alive, fine.alive)
    coarse_stats, medium_stats, fine_stats = (
        probe.statistics for probe in (coarse, medium, fine)
    )
    for statistics in (coarse_stats, medium_stats):
        assert abs(statistics.radius_rms_m - fine_stats.radius_rms_m) * 1.0e9 < 0.005
    assert abs(medium_stats.waist_offset_m - fine_stats.waist_offset_m) * 1.0e9 < 0.1
    assert abs(medium_stats.waist_offset_m - fine_stats.waist_offset_m) < (
        abs(coarse_stats.waist_offset_m - medium_stats.waist_offset_m)
    )
    assert fine_stats.threefold_moment < 0.15


def test_nanoprobe_preset_restores_calibrated_hexapoles_after_channel_edits():
    state = _preset_state()
    state.hp2_hexapole.strength_m3 *= 0.5
    state.hp2_hexapole.orientation_rad = 0.1
    state.hp1_hexapole.strength_m3 *= 1.2
    state.hp1_hexapole.orientation_rad = -0.1
    next(lens for lens in state.lenses if lens.key == "condenser_lens_3").percent = 20.0

    applied = apply_operating_mode_pair(state, "nano_probe", "diffraction")

    assert state.hp2_hexapole.strength_m3 == pytest.approx(626910.5744377297)
    assert state.hp2_hexapole.orientation_rad == pytest.approx(0.0)
    assert state.hp1_hexapole.strength_m3 == pytest.approx(374822.52529434016)
    assert state.hp1_hexapole.orientation_rad == pytest.approx(-0.04231957336232493)
    assert next(lens for lens in state.lenses if lens.key == "condenser_lens_3").percent == pytest.approx(21.273622908333973)
    assert applied.condenser.targets["target_convergence_sem_angle_mrad"] == pytest.approx(25.0)


def test_probe_mode_roundtrip_restores_each_modes_own_hexapole_reference():
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())

    def fields(candidate):
        return tuple(
            (part.enabled, part.strength_m3, part.orientation_rad)
            for part in (candidate.hp2_hexapole, candidate.hp1_hexapole)
        )

    apply_operating_mode_pair(state, "micro_probe", "imaging")
    fresh_micro = fields(state)
    apply_operating_mode_pair(state, "nano_probe", "diffraction")
    nano = fields(state)
    assert nano != fresh_micro

    # A saved calculation/profile must retain the selected working point too.
    restored = type(state).from_dict(state.to_dict())
    assert fields(restored) == nano
    apply_operating_mode_pair(restored, "micro_probe", "imaging")
    assert fields(restored) == fresh_micro
    apply_operating_mode_pair(restored, "nano_probe", "diffraction")
    assert fields(restored) == nano


@pytest.mark.parametrize("invalid", ["empty", "blocked", "zero_weight", "nan_weight", "nan_position"])
def test_hexapole_fit_rejects_invalid_baseline_before_changing_fields(invalid):
    size = 0 if invalid == "empty" else 3
    x = np.zeros(size)
    alive = np.ones(size, dtype=bool)
    weights = np.ones(size)
    if invalid == "blocked":
        alive[:] = False
    elif invalid == "zero_weight":
        weights[:] = 0.0
    elif invalid == "nan_weight":
        weights[0] = np.nan
    elif invalid == "nan_position":
        x[0] = np.nan
    baseline = IncidentProbeMeasurement(
        x, np.zeros(size), np.zeros(size), np.zeros(size), alive, weights,
    )

    def hexapole(strength):
        return SimpleNamespace(
            enabled=True, strength_m3=strength, orientation_rad=0.0,
            maximum_strength_m3=1.0e6,
        )

    state = SimpleNamespace(
        hp2_hexapole=hexapole(6.0e5), hp1_hexapole=hexapole(3.5e5),
    )
    model = SimpleNamespace(state=state, trace=lambda **_options: baseline)
    with pytest.raises(ValueError):
        fit_probe_hexapoles(model)
    assert state.hp2_hexapole.strength_m3 == 6.0e5
    assert state.hp1_hexapole.strength_m3 == 3.5e5
    assert state.hp1_hexapole.orientation_rad == 0.0
