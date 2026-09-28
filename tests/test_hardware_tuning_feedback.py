"""Executed-history feedback: weights, angle conventions and unavailable data."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import math

import numpy as np
import pytest

from temsim.hardware_tuning_feedback import (
    baseline_difference, captured_hardware_values, observe_retained_beam, recorded_waists,
)
from temsim.optics.column import default_state
from temsim.runtime_parameters import runtime_targets
from temsim.instrument_snapshot import decode_instrument, encode_instrument


def retained_result(state=None, *, request="feedback-first", x_shift=0.0):
    state = decode_instrument(encode_instrument(state)) if state is not None else default_state()
    z = float(state.sample.z_mm)
    branch = SimpleNamespace(
        name="incident", interaction_kind="incident", z=np.array([0., z]),
        x=np.tile(np.array([-3e-6, -1e-6, 1e-6, 3e-6]) + x_shift, (2, 1)),
        y=np.zeros((2, 4)), tx=np.full((2, 4), 0.2), ty=np.zeros((2, 4)),
        blocked_z=np.array([np.nan, np.nan, np.nan, z / 2]),
        ray_weight=np.array([0.1, 0.2, 0.3, 0.4]), weight=1.,
        source_ray_id=np.arange(4, dtype=np.int64), source_azimuth_rad=np.zeros(4),
    )
    return SimpleNamespace(
        state_snapshot=state, signatures={"request": request}, model_signature="captured-model",
        simulation=SimpleNamespace(incident=branch, branches={}, real_interactions=None,
                                   metrics={"effective_source_current_pa": 20., "branch_weights_are_absolute": True}),
        lens_crossovers=(), calculation_manifest=None,
    )


def test_weighted_observation_preserves_source_transmission_and_uses_angles_not_slopes():
    result = retained_result()
    before = deepcopy(vars(result.simulation.incident))
    row = observe_retained_beam(result, result.state_snapshot.sample.z_mm)
    assert row.status == "AVAILABLE"
    assert row.ray_count == 3
    assert row.source_fraction == pytest.approx(0.6)
    assert row.current_pa == pytest.approx(12.0)
    assert row.centroid_x_um == pytest.approx(-1 / 3)
    assert row.direction_x_mrad == pytest.approx(math.atan(0.2) * 1000.)
    assert row.direction_x_mrad != 200.
    assert row.angular_rms_mrad == pytest.approx(0., abs=1e-12)
    assert row.rms_radius_um == pytest.approx(math.sqrt(20 / 9))
    assert row.result_id == "feedback-first" and row.model_id == "captured-model"
    assert "interpolation" in row.provenance
    for key, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(result.simulation.incident, key), value)


def test_feedback_never_extrapolates_past_retained_range_or_invents_missing_weights():
    result = retained_result()
    row = observe_retained_beam(result, result.state_snapshot.sample.z_mm + 1.)
    assert row.status == "UNAVAILABLE"
    assert row.current_pa is None and row.centroid_x_um is None
    assert "downstream" in row.reason
    result.simulation.incident.ray_weight = None
    row = observe_retained_beam(result, 0.)
    assert row.status == "UNAVAILABLE"
    assert row.source_fraction is None
    assert "probabilities are unavailable" in row.reason


def test_single_positive_path_retains_direction_but_does_not_report_a_zero_beam_width():
    result = retained_result()
    result.simulation.incident.ray_weight[:] = [1, 0, 0, 0]
    row = observe_retained_beam(result, 0.)
    assert row.status == "INSUFFICIENT_POPULATION"
    assert row.ray_count == 1 and row.current_pa == 20
    assert row.centroid_x_um == -3
    assert row.diameter95_um is None and row.angular_rms_mrad is None


def test_zero_survivors_keep_zero_sampled_current_but_no_centroid():
    result = retained_result()
    result.simulation.incident.blocked_z[:] = 0.5
    row = observe_retained_beam(result, 1.)
    assert row.status == "NO_SAMPLED_SURVIVORS"
    assert row.source_fraction == 0 and row.current_pa == 0
    assert row.centroid_x_um is None


def test_baseline_delta_requires_matching_plane_population_and_available_statistics():
    result = retained_result()
    first = observe_retained_beam(result, 0.)
    second = observe_retained_beam(retained_result(request="second", x_shift=2e-6), 0.)
    difference, reason = baseline_difference(second, first)
    assert difference["centroid_x_um"] == pytest.approx(2.)
    assert "other settings" in reason
    assert baseline_difference(replace(second, z_mm=1.), first)[0] is None
    assert baseline_difference(replace(second, population="Specimen exit"), first)[0] is None
    assert baseline_difference(replace(second, centroid_x_um=None), first)[0] is None


def test_captured_hardware_controls_are_detached_and_do_not_restore_state():
    result = retained_result()
    old = captured_hardware_values(result.state_snapshot, "condenser_current")
    runtime_targets(result.state_snapshot)["condenser_lens_1"].obj.percent = 42.
    current = captured_hardware_values(result.state_snapshot, "condenser_current")
    assert old != current
    assert next(row[-1] for row in current if row[:2] == ("condenser_lens_1", "percent")) == 42.


def test_focus_uses_only_recorded_matching_finite_verified_waists():
    result = retained_result()
    result.lens_crossovers = (
        dict(source_lens_key="objective_lens", z_mm=1600., rms_radius_mm=0.001, verified=True),
        dict(source_lens_key="condenser_lens_1", z_mm=100., rms_radius_mm=0.002, verified=True),
        dict(source_lens_key="objective_lens", z_mm=math.nan, rms_radius_mm=0.001, verified=True),
        dict(source_lens_key="objective_lens", z_mm=1602., rms_radius_mm=0.001, verified=False),
    )
    assert recorded_waists(result, {"objective_lens"}) == (("objective_lens", 1600., 1.),)
    assert recorded_waists(result, {"mini_condenser"}) == ()


def test_missing_execution_and_nonfinite_observation_are_explicit():
    assert observe_retained_beam(None, 0.).status == "UNAVAILABLE"
    with pytest.raises(ValueError, match="finite"):
        observe_retained_beam(retained_result(), math.nan)


@pytest.mark.parametrize("signatures", [{}, {"request": None}, {"request": ""}, {"request": "   "}, None])
def test_missing_request_identity_never_qualifies_feedback(signatures):
    result = retained_result()
    result.signatures = signatures
    row = observe_retained_beam(result, 0.)
    assert row.status == "UNAVAILABLE"
    assert row.result_id == "UNRECORDED"
    assert "request identity" in row.reason
    assert row.current_pa is None and row.centroid_x_um is None
