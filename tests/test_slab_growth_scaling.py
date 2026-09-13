"""Constant-slab roundoff regressions, not full-source convergence acceptance."""
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm
from threadpoolctl import threadpool_limits

from temsim.physics.covariant_boundary import _slab, _unitarity


def test_growth_bound_controls_transfer_and_keeps_noncommuting_phase():
    q = np.array([[-40., 2j], [-2j, 5.]])
    g = np.array([[20., 3.], [3., -30.]])
    width, kappa = .15, 2.
    eye = np.eye(2)
    chart = np.block([[eye, eye], [1j*eye, -1j*eye]])
    generator = np.linalg.solve(chart,
        np.block([[-1j*g, kappa*eye], [-q/kappa, -1j*g]])@chart)
    result, row = _slab(q, g, width, kappa, lambda: False)
    seed = generator*width/2**row["scattering_doublings"]
    assert np.linalg.norm(expm(seed), ord=2) <= np.exp(row["seed_log_growth_bound"])*(1+1e-14)
    assert np.linalg.cond(expm(seed)) <= np.e*(1+1e-14)
    assert row["seed_generator_norm"] <= 16.
    # Independent full transfer solve is safe for this short fixture.
    full = expm(generator*width)
    a, b, c, d = full[:2, :2], full[:2, 2:], full[2:, :2], full[2:, 2:]
    reflection = -np.linalg.solve(d, c)
    transmission = np.linalg.solve(d, eye)
    reference = reflection, transmission, a+b@reflection, b@transmission
    np.testing.assert_allclose(result, reference, atol=3e-13, rtol=3e-13)
    assert _unitarity(result) < 1e-12


def test_large_pure_connection_preserves_absolute_complex_phase():
    kappa, width = 2., 1.
    g = np.array([[1000., 40j], [-40j, -1200.]])
    values, vectors = np.linalg.eigh(g)
    result, row = _slab(kappa*kappa*np.eye(2), g, width, kappa, lambda: False)
    right = (vectors*np.exp(1j*(kappa-values)*width))@vectors.conj().T
    left = (vectors*np.exp(1j*(kappa+values)*width))@vectors.conj().T
    np.testing.assert_allclose(result, (g*0, left, right, g*0), atol=2e-11, rtol=2e-11)
    assert row["seed_log_growth_bound"] == 0.
    assert row["scattering_doublings"] < 10


def test_captured_quartic_gun_slab_preserves_all_channels(record_property):
    path = Path(__file__).resolve().parents[1]/(
        "docs/development/evidence/20260913-quartic-failure-capture/failed_slab.npz")
    # Actual failure from the full, unchanged physical gun. Pin the fixture;
    # it contains only the operator/connection, not a replacement source.
    assert sha256(path.read_bytes()).hexdigest() == "9d82a2232648173d3405c203c7400d442465e963a1a187b01656ba53c3e7bef8"
    with np.load(path, allow_pickle=False) as data:
        width, kappa, carrier = data["width_kappa_carrier"]
        q = carrier**2*np.eye(len(data["residual"]))+data["residual"]
        g = data["connection"]
    with threadpool_limits(1):
        result, row = _slab(q, g, width, kappa, lambda: False)
        refined, fine = _slab(q, g, width, kappa, lambda: False, maximum_seed_log_growth=.25)
    full = np.block([[result[0], result[1]], [result[2], result[3]]])
    reference = np.block([[refined[0], refined[1]], [refined[2], refined[3]]])
    error = float(np.linalg.norm(full-reference, ord=np.inf))
    record_property("constant_slab_complex_matrix_error", error)
    record_property("default_maximum_unitarity_residual", row["maximum_unitarity_residual"])
    record_property("refined_maximum_unitarity_residual", fine["maximum_unitarity_residual"])
    assert full.shape == (128, 128)
    assert error < 1e-9
    assert row["maximum_unitarity_residual"] < 1e-9
    assert fine["maximum_unitarity_residual"] < 1e-9
    assert fine["scattering_doublings"] == row["scattering_doublings"]+1


@pytest.mark.parametrize("bound", (0, -.1, 1., np.inf, np.nan, True))
def test_invalid_seed_growth_bound_rejected(bound):
    with pytest.raises(ValueError, match="log-growth"):
        _slab(np.eye(1), np.zeros((1, 1)), 1., 1., lambda: False, maximum_seed_log_growth=bound)


def test_strict_unitarity_rejection_not_replaced_with_renormalization():
    with pytest.raises(ValueError, match="unitarity"):
        _unitarity((np.zeros((2, 2)), np.eye(2)*(1+1e-8), np.eye(2), np.zeros((2, 2))))
