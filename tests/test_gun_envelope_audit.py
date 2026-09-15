"""Readout tests only; no expensive gun or wave propagation."""
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.check_gun_envelope import plane_summary


def trace_fixture():
    z = np.tile(np.array([0., 1., 2.])[:, None], (1, 3))
    x = np.tile(np.array([0., .001, .003])[:, None], (1, 3))
    return SimpleNamespace(
        blocked_z_mm=np.array([np.nan, np.nan, .5]),
        exit_bundle=SimpleNamespace(weight=np.array([1., 0., 1.])),
        equal_time_history=SimpleNamespace(z_mm=z, x_m=x, y_m=np.zeros_like(x),
            tx_rad=np.full_like(x, .1), ty_rad=np.zeros_like(x)),
        # A coarse presentation line is deliberately not the physical history.
        z_mm=np.array([0., 2.]), x_m=x[[0, 2]],
    )


def test_physical_crossing_not_coarse_display_interpolation():
    result = plane_summary(trace_fixture(), 1.)
    assert result["reaching_rays"] == 1
    assert result["axis_centred_envelope_diameter_mm"] == pytest.approx(2.)
    assert result["axis_centred_rms_radius_mm"] == pytest.approx(1.)
    assert result["max_polar_angle_deg"] == pytest.approx(np.degrees(np.arctan(.1)))


def test_roundoff_at_exit_is_not_a_missing_crossing():
    trace = trace_fixture()
    trace.equal_time_history.z_mm[-1] = np.nextafter(2., 0.)
    assert plane_summary(trace, 2.)["axis_centred_envelope_diameter_mm"] == pytest.approx(6.)


def test_actual_missing_history_is_not_silently_extrapolated():
    with pytest.raises(ValueError, match="no recorded crossing"):
        plane_summary(trace_fixture(), 3.)


def test_no_reaching_particles():
    trace = trace_fixture()
    trace.blocked_z_mm[:] = .5
    assert plane_summary(trace, 1.) == {"z_mm": 1., "reaching_rays": 0}
