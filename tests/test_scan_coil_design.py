"""Analytical optics and a bounded real-field smoke test for AC design advice."""

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from temsim.optics import scan_coil_design as design
from temsim.physics.first_order import TransverseTransfer


@dataclass
class _Coils:
    upper_z_mm: float = 10.
    lower_z_mm: float = 20.
    scan_reference: str = "sample_centre"
    scan_field_of_view_x_nm: float = 200.
    scan_field_of_view_y_nm: float = 400.
    maximum_kick_mrad: float = 100.
    kick_x_mrad: float = 0.
    kick_y_mrad: float = 0.
    enabled: bool = True

    def coil_kicks_mrad(self, x, y):
        return ((x, y), (-x, -y))


def _state(**kwargs):
    return SimpleNamespace(ac_deflector=_Coils(),
                           sample=SimpleNamespace(z_mm=100., upper_surface_z_mm=99.9),
                           g=0., lens_z_mm=None, focal_mm=(20., 20.), **kwargs)


def _drift(length_m):
    result = np.eye(4)
    result[:2, 2:] = np.eye(2) * length_m
    return result


def _absolute_map(state, z):
    if state.g:
        generator = np.zeros((4, 4))
        generator[:2, 2:] = np.eye(2)
        generator[2:, 2:] = np.array(((0., 2. * state.g), (-2. * state.g, 0.)))
        return expm(generator * float(z) * 1e-3)
    if state.lens_z_mm is not None and z > state.lens_z_mm:
        lens = np.eye(4)
        lens[2:, :2] = -np.diag(1e3 / np.asarray(state.focal_mm))
        return _drift((z-state.lens_z_mm)*1e-3) @ lens @ _drift(state.lens_z_mm*1e-3)
    return _drift(z*1e-3)


@pytest.fixture
def analytical(monkeypatch):
    monkeypatch.setattr(design, "_private_state", deepcopy)

    def trace(state, source, targets, **_kwargs):
        # Deliberate cache write emulates the real observer and tests isolation.
        state.trace_calls = getattr(state, "trace_calls", 0) + 1
        inverse = np.linalg.inv(_absolute_map(state, source))
        result = {}
        for z in targets:
            value = _absolute_map(state, float(z)) @ inverse
            result[float(z)] = TransverseTransfer(
                source, float(z), value[:2, :2], value[:2, 2:],
                value[2:, :2], value[2:, 2:])
        return result

    def canonical(state, z, **_kwargs):
        basis = np.eye(4)
        basis[2:, :2] = np.array(((0., state.g), (-state.g, 0.)))
        return basis

    monkeypatch.setattr(design, "trace_transverse_transfers", trace)
    monkeypatch.setattr(design, "canonical_source_basis", canonical)
    return _state()


def test_zero_field_matches_exact_two_kick_shift_and_preserves_state(analytical):
    before = deepcopy(vars(analytical))
    result = design.evaluate_scan_coil_design(analytical, pivot_step_mm=2.)
    expected = np.diag((.01, .02))  # 100/200 nm half-FOV over a 10 mm baseline.
    assert result.controllable and result.kick_limit_pass
    assert np.asarray(result.upper_kick_matrix_mrad) == pytest.approx(expected, abs=1e-14)
    assert np.asarray(result.lower_kick_matrix_mrad) == pytest.approx(-expected, abs=1e-14)
    assert np.asarray(result.lower_from_upper) == pytest.approx(-np.eye(2), abs=1e-14)
    assert result.position_relative_residual < 1e-12
    assert result.angle_residual_rad < 1e-17
    assert not result.pivot.is_common_pivot
    assert result.pivot.relative_residual == pytest.approx(1.)
    assert vars(analytical) == before

    assert not result.geometry_qualified


def test_strong_uniform_field_distinguishes_mechanical_from_canonical(analytical):
    analytical.g = -100.  # radians/metre; exact uniform-solenoid response.
    mechanical = design.evaluate_scan_coil_design(analytical, pivot_step_mm=2.)
    canonical = design.evaluate_scan_coil_design(analytical, target="canonical", pivot_step_mm=2.)
    expected = np.diag((100e-9, 200e-9))
    for result in (mechanical, canonical):
        assert result.controllable
        assert np.asarray(result.sample_position_matrix_m) == pytest.approx(expected, abs=1e-19)
        assert result.angle_relative_residual < 1e-12
    assert np.linalg.norm(mechanical.sample_mechanical_angle_matrix_rad) < 1e-17
    assert np.linalg.norm(mechanical.sample_canonical_angle_matrix_rad) > 1e-6
    assert np.linalg.norm(canonical.sample_canonical_angle_matrix_rad) < 1e-17
    assert np.linalg.norm(canonical.sample_mechanical_angle_matrix_rad) > 1e-6
    assert not np.allclose(mechanical.lower_from_upper, canonical.lower_from_upper)


def test_round_lens_has_full_two_dimensional_front_focal_scan_pivot(analytical):
    analytical.lens_z_mm = 60.
    result = design.evaluate_scan_coil_design(analytical, pivot_step_mm=1.3)
    assert result.pivot.is_common_pivot
    assert result.pivot.z_mm == pytest.approx(40., abs=1e-5)
    assert max(result.pivot.singular_values_m) < 1e-13
    assert result.pivot.kind == "common_pivot_candidate"


def test_astigmatic_two_axis_crossings_are_not_a_common_pivot(analytical):
    analytical.lens_z_mm = 60.
    analytical.focal_mm = (20., 30.)
    result = design.evaluate_scan_coil_design(analytical, pivot_step_mm=.5)
    assert not result.pivot.is_common_pivot
    assert result.pivot.relative_residual > .1
    # Separate X/Y zeros are at 40 and 30 mm; their matrix minimum is between.
    assert 30 < result.pivot.z_mm < 40
    assert result.pivot.kind == "closest_approach"


def test_wide_fov_axis_does_not_hide_failure_to_pivot_the_other_axis(analytical):
    analytical.lens_z_mm = 60.
    analytical.focal_mm = (20., 30.)
    analytical.ac_deflector.scan_field_of_view_x_nm = 2e8
    analytical.ac_deflector.scan_field_of_view_y_nm = .002
    result = design.evaluate_scan_coil_design(analytical, pivot_step_mm=.5)
    assert not result.pivot.is_common_pivot
    assert result.pivot.relative_residual > .1
    assert 30 < result.pivot.z_mm < 40


def test_physical_limit_includes_full_fov_corners_and_static_bias(analytical):
    ac = analytical.ac_deflector
    ac.maximum_kick_mrad = .021
    assert design.evaluate_scan_coil_design(analytical, pivot_step_mm=5.).kick_limit_pass
    ac.kick_y_mrad = .002
    result = design.evaluate_scan_coil_design(analytical, pivot_step_mm=5.)
    assert result.upper_scan_peak_mrad == pytest.approx((.01, .02))
    assert result.upper_peak_mrad == pytest.approx((.01, .022))
    assert result.lower_peak_mrad == pytest.approx((.01, .022))
    assert not result.kick_limit_pass
    assert result.peak_kick_mrad == pytest.approx(.022)


@pytest.mark.parametrize(("upper", "lower"), ((-1., 20.), (20., 10.), (10., 10.),
                                                     (10., 100.), (10., float("nan"))))
def test_candidate_positions_must_be_ordered_and_above_sample(analytical, upper, lower):
    with pytest.raises(ValueError, match="coil planes"):
        design.evaluate_scan_coil_design(analytical, upper_z_mm=upper, lower_z_mm=lower)


@pytest.mark.parametrize("kwargs", (
    {"target": "field_peak"}, {"pivot_step_mm": 0}, {"pivot_step_mm": .0001},
    {"condition_limit": .9}, {"maximum_step_mm": -1}, {"pivot_relative_tolerance": 1.},
))
def test_invalid_diagnostic_options_rejected(analytical, kwargs):
    with pytest.raises(ValueError):
        design.evaluate_scan_coil_design(analytical, **kwargs)


def test_scan_entrance_reference_is_honoured(analytical):
    analytical.ac_deflector.scan_reference = "sample_entrance"
    result = design.evaluate_scan_coil_design(analytical, pivot_step_mm=4.)
    assert result.sample_z_mm == 99.9
    assert result.scan_reference == "sample_entrance"


def test_nearly_coincident_coils_are_not_silently_accepted(analytical):
    result = design.evaluate_scan_coil_design(
        analytical, upper_z_mm=19.999999999, lower_z_mm=20., pivot_step_mm=4.)
    assert not result.controllable
    assert result.condition_number > result.condition_limit
    assert not result.kick_limit_pass
    assert result.upper_kick_matrix_mrad is None and result.pivot is None


def test_position_search_is_bounded_ranked_cancellable_and_read_only(analytical):
    progress = []
    before = deepcopy(vars(analytical))
    result = design.search_scan_coil_positions(
        analytical, upper_z_range_mm=(5., 15.), coil_gap_mm=10.,
        candidate_count=3, pivot_step_mm=5., progress_callback=lambda *row: progress.append(row))
    assert len(result.candidates) == 3
    assert all(item.lower_z_mm - item.upper_z_mm == 10. for item in result.candidates)
    assert {item.upper_z_mm for item in result.candidates} == {5., 10., 15.}
    assert result.best is not None
    assert progress == [(0, 3), (1, 3), (2, 3), (3, 3)]
    assert vars(analytical) == before
    with pytest.raises(InterruptedError, match="cancelled"):
        design.search_scan_coil_positions(
            analytical, upper_z_range_mm=(5., 15.), coil_gap_mm=10.,
            cancel_check=lambda: True)
    assert vars(analytical) == before

    progress.clear()
    with pytest.raises(InterruptedError, match="cancelled"):
        design.search_scan_coil_positions(
            analytical, upper_z_range_mm=(5., 15.), coil_gap_mm=10.,
            pivot_step_mm=5., progress_callback=lambda *row: progress.append(row),
            cancel_check=lambda: bool(progress and progress[-1][0] == 1))
    assert progress == [(0, 3), (1, 3)]
    assert vars(analytical) == before


def test_search_never_clips_an_illegal_interval_or_count(analytical):
    for kwargs in ({"upper_z_range_mm": (90., 95.), "coil_gap_mm": 10.},
                   {"upper_z_range_mm": (5., 15.), "coil_gap_mm": 10., "candidate_count": True},
                   {"upper_z_range_mm": (15., 5.), "coil_gap_mm": 10.}):
        with pytest.raises(ValueError):
            design.search_scan_coil_positions(analytical, **kwargs)


def test_real_default_full_field_preserves_exact_input_graph():
    from temsim.instrument_snapshot import encode_instrument
    from temsim.optics.column import default_state

    state = default_state()
    state.acceleration_enabled = False
    before = encode_instrument(state)
    mechanical = design.evaluate_scan_coil_design(state, pivot_step_mm=.5)
    canonical = design.evaluate_scan_coil_design(state, target="canonical", pivot_step_mm=.5)
    assert encode_instrument(state) == before
    for result in (mechanical, canonical):
        assert result.controllable and result.kick_limit_pass
        assert result.position_relative_residual < 1e-10
        assert result.angle_relative_residual < 1e-10
        assert result.pivot.interval_z_mm == (state.ac_deflector.lower_z_mm, state.sample.z_mm)
    assert np.linalg.norm(mechanical.sample_canonical_angle_matrix_rad) > 1e-9
    assert np.linalg.norm(canonical.sample_mechanical_angle_matrix_rad) > 1e-9
    assert not mechanical.pivot.is_common_pivot
    assert canonical.pivot.is_common_pivot
