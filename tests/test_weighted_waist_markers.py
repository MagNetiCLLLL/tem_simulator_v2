from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics.beam_waist import detect_beam_waist
from temsim.physics.all_lens_crossovers import detect_all_lens_crossovers


def bundle():
    z = np.linspace(0., 10., 101)
    # Two symmetric angular populations with different focal planes. Physical
    # flux, not the number of numerical representatives, fixes their RMS waist.
    slope = np.r_[-np.ones(4), np.ones(4)]*.01
    focus = np.tile([2., 2., 8., 8.], 2)
    x = (z[:, None]-focus)*1e-3*slope
    return SimpleNamespace(z=z, x=x, y=np.zeros_like(x),
        tx=np.broadcast_to(slope, x.shape).copy(), ty=np.zeros_like(x),
        blocked_z=np.full(8, np.nan), ray_weight=np.tile([9., 9., 1., 1.], 2))


def test_marker_uses_current_weights_and_is_invariant_to_ray_splitting():
    b = bundle()
    expected = detect_beam_waist(b, 0., 10.)
    assert expected["z_mm"] == pytest.approx(2.6)
    # Oversample only the high-z focus, without adding physical current.
    repeats = np.tile([1, 1, 31, 31], 2)
    copies = SimpleNamespace(z=b.z, **{key: np.repeat(getattr(b, key), repeats, axis=1)
        for key in ("x", "y", "tx", "ty")}, blocked_z=np.repeat(b.blocked_z, repeats),
        ray_weight=np.repeat(b.ray_weight/repeats, repeats))
    actual = detect_beam_waist(copies, 0., 10.)
    assert actual["z_mm"] == expected["z_mm"]
    assert actual["rms_radius_mm"] == pytest.approx(expected["rms_radius_mm"])
    lens = SimpleNamespace(key="condenser_lens_1", name="C1", z_mm=1., enabled=True)
    assert detect_all_lens_crossovers([copies], [lens])[0]["z_mm"] == expected["z_mm"]


def test_zero_current_rays_do_not_create_a_waist():
    b = bundle()
    b.ray_weight[:] = 0.
    assert detect_beam_waist(b, 0., 10.) is None


def test_no_waist_created_by_population_clipping():
    b = bundle()
    b.x = np.broadcast_to(np.linspace(-.01, .01, 8), b.x.shape).copy()
    b.tx[:] = 0.
    b.blocked_z[:4] = 5.
    assert detect_beam_waist(b, 0., 10.) is None


def test_equal_weight_historical_branches_and_weight_validation():
    b = bundle()
    b.ray_weight = None
    assert detect_beam_waist(b, 0., 10.)["z_mm"] == pytest.approx(5.)
    b.ray_weight = np.ones(8)
    b.ray_weight[0] = -1.
    with pytest.raises(ValueError, match="waist history"):
        detect_beam_waist(b, 0., 10.)
