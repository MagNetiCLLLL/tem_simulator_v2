"""Occupied-wave loss, not the worst-case operator defect or norm fitting."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import eval_laguerre

from temsim.physics.radial_gun_wave import aperture_projection
from temsim.physics.radial_mask_ledger import mask_loss_budget, physical_to_chart


def test_disk_loss_matches_independent_spatial_integral_for_both_ports():
    left = np.array([.8, .1j, -.3, .2-.1j])
    right = np.array([.1j, .2, .1, .05j])
    radius, width, k = 1.7, 1.2, 3.1
    p = aperture_projection(radius, width, len(left), 16)
    result = mask_loss_budget(p, left, right, k)
    physical_loss = 0.
    for v in (left, right):
        def density(x):
            return np.exp(-x)*abs(sum(c*eval_laguerre(n, x) for n, c in enumerate(v)))**2
        physical_loss += k*quad(density, (radius/width)**2, np.inf, epsabs=1e-12)[0]
    assert result["mask_absorbed"] == pytest.approx(physical_loss, abs=2e-12)
    removed = k*sum(np.vdot(v, v).real-np.vdot(p@v, p@v).real for v in (left, right))
    assert result["total_removed"] == pytest.approx(removed, abs=2e-14)
    assert result["unresolved_transmitted"] > .01


def test_occupied_loss_can_vanish_despite_nonzero_operator_defect():
    p = aperture_projection(3., 1., 32, 64)
    wave = np.zeros(32); wave[0] = 1.
    result = mask_loss_budget(p, wave, np.zeros(32), 1.)
    assert np.linalg.norm(p-p@p, ord=2) > .2
    assert result["unresolved_transmitted"] < 1e-4


def test_chart_recovery_preserves_physical_current_and_both_directions():
    a = np.array([.2+.3j, 1.]); b = np.array([.1j, .1-.2j])
    alpha, log, k = 700., -.031, 3.
    u, du = a+b, 1j*k*(a-b)
    field, derivative = u/np.sqrt(alpha), (alpha*du+log*u)/np.sqrt(alpha)
    actual = physical_to_chart(field, derivative, alpha, log, k)
    np.testing.assert_allclose(actual, (a, b), atol=1e-15)
    assert np.vdot(field, derivative).imag == pytest.approx(k*(np.vdot(a, a)-np.vdot(b, b)).real)
