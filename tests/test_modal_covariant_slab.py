"""Optional complete constant-slab coordinates, not physical-source acceptance."""
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm
from threadpoolctl import threadpool_limits

from temsim.physics.covariant_boundary import _slab, _compose
from temsim.physics.modal_covariant_slab import modal_covariant_slab
from temsim.physics.scattering_load import covariant_carrier_slab


def direct_transfer(q, g, width, k):
    n = len(q)
    eye = np.eye(n)
    chart = np.block([[eye, eye], [1j*eye, -1j*eye]])
    a = np.linalg.solve(chart, np.block([[-1j*g, k*eye], [-q/k, -1j*g]])@chart)
    t = expm(a*width)
    r = -np.linalg.solve(t[n:, n:], t[n:, :n])
    tr = np.linalg.solve(t[n:, n:], eye)
    return r, tr, t[:n, :n]+t[:n, n:]@r, t[:n, n:]@tr


@pytest.mark.parametrize("width", (.001, .1, -.1, 0.))
@pytest.mark.parametrize("q", (np.diag([4., 9.]), np.diag([-4., -9.]),
                              np.array([[-4., 1j], [-1j, 9.]])))
def test_all_channels_match_independent_short_transfer(q, width):
    g = np.array([[.3, 1j], [-1j, -.4]])
    result, _ = modal_covariant_slab(q, g, width, 2.)
    np.testing.assert_allclose(result, direct_transfer(q, g, width, 2.), atol=3e-13, rtol=3e-13)


def test_long_mixed_slab_avoids_growing_transfer_and_keeps_phase():
    q = np.array([[-4., .2j], [-.2j, 9.]])
    g = np.array([[.3, .1], [.1, -.4]])
    result, record = modal_covariant_slab(q, g, 100., 2.)
    reference, _ = _slab(q, g, 100., 2., lambda: False, maximum_seed_log_growth=.25)
    np.testing.assert_allclose(result, reference, atol=2e-11, rtol=2e-11)
    assert record["evanescent_directional_solutions"] == 2
    assert record["retained_directional_solutions"] == 4
    half, _ = modal_covariant_slab(q, g, 50., 2.)
    np.testing.assert_allclose(result, _compose(half, half), atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("factor", (1., .5, -.25, .01))
def test_retained_actual_gun_operator(factor, record_property):
    path = Path(__file__).resolve().parents[1]/"docs/development/evidence/20260913-quartic-failure-capture/failed_slab.npz"
    with np.load(path, allow_pickle=False) as data:
        width, kappa, carrier = data["width_kappa_carrier"]
        q = carrier**2*np.eye(len(data["residual"]))+data["residual"]
        g = data["connection"]
    with threadpool_limits(1):
        result, record = modal_covariant_slab(q, g, width*factor, kappa)
        reference, _ = _slab(q, g, width*factor, kappa, lambda: False, maximum_seed_log_growth=.25)
    difference = float(np.linalg.norm(np.asarray(result)-reference))
    record_property("full_complex_difference", difference)
    record_property("current_residual", record["slab_unitarity_residual"])
    assert difference < 1e-9
    assert record["retained_directional_solutions"] == 128
    assert record["evanescent_directional_solutions"] == 60


def test_degenerate_grazing_chart_rejected_without_removing_channel():
    with pytest.raises(ValueError, match="chart|Grazing"):
        modal_covariant_slab(np.diag([0., 4.]), np.zeros((2, 2)), 1., 1.)


def test_cancel_before_eigensystem():
    with pytest.raises(InterruptedError):
        modal_covariant_slab(np.eye(1), np.zeros((1, 1)), 1., 1., lambda: True)


@pytest.mark.parametrize("kappa", (0., -1., float("nan"), True))
def test_invalid_kappa(kappa):
    with pytest.raises(ValueError, match="kappa"):
        modal_covariant_slab(np.eye(1), np.zeros((1, 1)), 1., kappa)


def test_unresolved_modal_chart_falls_back_to_complete_doubling(monkeypatch):
    import temsim.physics.modal_covariant_slab as module
    q = np.diag([-4., 9.])
    g = np.array([[.3, .1], [.1, -.4]])
    def reject(*args, **kwargs):
        raise ValueError("Injected unresolved chart")
    monkeypatch.setattr(module, "modal_covariant_slab", reject)
    result, record = covariant_carrier_slab(q-np.eye(2), g, 1., 2., 1.)
    reference, _ = _slab(q, g, 1., 2., lambda: False)
    np.testing.assert_array_equal(result, reference)
    assert record["modal_fallback_reason"] == "Injected unresolved chart"


def test_mixed_method_selection_preserves_full_reference():
    q = np.diag([-4., 9.])
    g = np.array([[.3, .1], [.1, -.4]])
    result, record = covariant_carrier_slab(q-np.eye(2), g, 1., 2., 1.)
    reference, baseline = covariant_carrier_slab(q-np.eye(2), g, 1., 2., 1., mixed_coordinates="doubling")
    np.testing.assert_allclose(result, reference, atol=3e-13, rtol=3e-13)
    assert record["modal_covariant_slab"] is True
    assert "scattering_doublings" in baseline
    with pytest.raises(ValueError, match="mixed-channel"):
        covariant_carrier_slab(q-np.eye(2), g, 1., 2., 1., mixed_coordinates="unknown")
