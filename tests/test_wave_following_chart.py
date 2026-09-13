"""Numerical-coordinate checks, never full source/image acceptance fixtures."""
import numpy as np
import pytest
from scipy.special import eval_laguerre, eval_genlaguerre

from temsim.physics.wave_following_chart import (radial_wave_moments, ExecutedWaveChart,
    WaveFollowingNumerics, chart_from_executed_wave)


def test_gaussian_and_pure_radial_modes_have_exact_moments_even_with_large_chirp():
    for order in (0, 1, 12, 63):
        a = np.zeros(64, complex); a[order] = (2+3j)*1e-200
        result = radial_wave_moments(a, 3., 1e12, 100.)
        assert result["radius_squared_nm2"] == pytest.approx(9*(2*order+1))
        assert result["optimal_width_nm"] == pytest.approx(3.)
        assert result["optimal_curvature_per_nm"] == 1e12
        assert result["minimum_mean_radial_order"] == pytest.approx(order, abs=1e-12)


def test_moments_match_independent_real_space_complex_field_and_gradient():
    a = np.array((1+.2j, .3-.8j, .4+.6j, -.7j))
    b, c, k = 2.3, .04, 120.
    r = np.linspace(0, 35, 80001); x = (r/b)**2
    polynomials = np.stack([eval_laguerre(n, x) for n in range(len(a))], axis=1)
    derivatives = np.stack([np.zeros_like(x) if n == 0 else -eval_genlaguerre(n-1, 1, x)
        for n in range(len(a))], axis=1)
    factor = np.exp(-x/2+1j*k*c*r*r/2)/(np.sqrt(np.pi)*b)
    f = factor*(polynomials@a)
    df = factor*((derivatives@a)*2*r/b**2 + (polynomials@a)*(-r/b**2+1j*k*c*r))
    integral = lambda v: np.trapezoid(2*np.pi*r*v, r)
    norm = integral(abs(f)**2)
    r2 = integral(r*r*abs(f)**2)/norm
    covariance = integral((np.conj(f)*r*df).imag)/norm
    p2 = integral(abs(df)**2)/norm
    expected = (r2/(p2-covariance*covariance/r2))**.25
    actual = radial_wave_moments(a, b, c, k)
    assert actual["radius_squared_nm2"] == pytest.approx(r2, rel=2e-8)
    assert actual["optimal_width_nm"] == pytest.approx(expected, rel=2e-8)
    assert actual["optimal_curvature_per_nm"] == pytest.approx(covariance/(k*r2), rel=2e-8)
    assert radial_wave_moments(a*np.exp(.72j), b, c, k) == pytest.approx(actual)


@pytest.mark.parametrize("smooth", [0., .05])
def test_executed_chart_derivatives_and_unchanged_tip_matching_coordinates(smooth):
    z = np.geomspace(2., 1000., 80)
    settings = WaveFollowingNumerics(iterations=1, smoothing_log_z_width=smooth)
    chart = ExecutedWaveChart(z, 3*np.sqrt(z), .2/z, settings, {"executed": "fixture"})
    def base(v):
        return (1+v*.01, .03+v*.00001, .01/(1+v*.01), .00001)
    for v in (2., 5., 10.):
        assert chart.evaluate(v, base(v)) == base(v)
    for v in (10., 15., 45., 99., 100., 200., 900.):
        h = 1e-4
        center = chart.evaluate(v, base(v))
        low, high = (chart.evaluate(w, base(w)) for w in (v-h, v+h))
        assert (np.log(high[0])-np.log(low[0]))/(2*h) == pytest.approx(center[2], abs=2e-10)
        assert (high[1]-low[1])/(2*h) == pytest.approx(center[3], abs=2e-10)
    with pytest.raises(ValueError, match="extrapolate"):
        chart.evaluate(1001., base(1001.))


def test_all_executed_state_and_normal_derivative_affect_chart_provenance():
    settings = WaveFollowingNumerics(iterations=1)
    fields = np.array([[1., .1j]]*6, complex)
    derivatives = fields*.7j
    frame = {"rows": [{"z_nm": z} for z in (10., 100., 100., 500., 1000.)],
        "start_nm": 2., "frames": [(3., .01)]*5, "initial_width_nm": 3.,
        "initial_curvature_per_nm": .01, "reference_wave_number_per_nm": 100., "numerics": {}}
    chart, report = chart_from_executed_wave(fields, derivatives, frame, settings)
    assert report["unique_planes"] == 5
    changed = derivatives.copy(); changed[2, 1] += .01
    assert chart_from_executed_wave(fields, changed, frame, settings)[0].digest != chart.digest
    assert chart_from_executed_wave(fields*np.exp(.3j), derivatives*np.exp(.3j), frame, settings)[0].digest != chart.digest


@pytest.mark.parametrize("smooth", [0., .05])
def test_tabulated_chart_preserves_independent_nonparaxial_free_wave_phase(smooth):
    from threadpoolctl import threadpool_limits
    from scripts.inspect_moving_radial_frame import calculate
    z = np.linspace(2, 102, 201)
    propagation = z-2
    chart = ExecutedWaveChart(z, 1.2*np.sqrt(1+(propagation/100)**2),
        propagation/(10000+propagation**2),
        WaveFollowingNumerics(iterations=1, transition_start_nm=12, transition_end_nm=52,
                              smoothing_log_z_width=smooth),
        {"scope": "Independent free-wave coordinate fixture, not a source"})
    with threadpool_limits(1):
        result = calculate(.05, count=16, chart_override=lambda value, base: chart.evaluate(value+2, base))
    assert result["relative_complex_error"] < 1e-6


def test_smoothed_quartic_coordinate_derivative_and_explicit_identity():
    z = np.geomspace(2., 1000., 160)
    widths = np.sqrt(z)*(1+.005*np.sin(90*np.log(z)))
    arguments = (z, widths, .2/z)
    settings = WaveFollowingNumerics(iterations=1, phase_order=4, smoothing_log_z_width=.05)
    chart = ExecutedWaveChart(*arguments, settings, {"fixture": True}, quartic_per_nm4=.03/z**2)
    original = ExecutedWaveChart(*arguments, WaveFollowingNumerics(iterations=1, phase_order=4),
                                {"fixture": True}, quartic_per_nm4=.03/z**2)
    assert chart.digest != original.digest
    for value in (10., 15., 45., 99., 100., 200., 900.):
        h = 1e-4
        low, high = chart.quartic_phase(value-h)[0], chart.quartic_phase(value+h)[0]
        assert (high-low)/(2*h) == pytest.approx(chart.quartic_phase(value)[1], abs=1e-10)
    # A chart is only coordinates: it neither contains a replacement field
    # nor changes the original wave-moment data supplied to it.
    np.testing.assert_array_equal(widths, np.sqrt(z)*(1+.005*np.sin(90*np.log(z))))


@pytest.mark.parametrize("a", ([0, 0], [float("nan"), 1], [[1, 0]]))
def test_missing_or_invalid_field_cannot_supply_a_chart(a):
    with pytest.raises(ValueError):
        radial_wave_moments(a, 3., .1, 100.)


@pytest.mark.parametrize("settings", (WaveFollowingNumerics(iterations=True), WaveFollowingNumerics(iterations=-1),
    WaveFollowingNumerics(width_multiplier=0), WaveFollowingNumerics(transition_end_nm=1),
    WaveFollowingNumerics(smoothing_log_z_width=-.1), WaveFollowingNumerics(smoothing_log_z_width=float('nan'))))
def test_invalid_numerical_controls_rejected(settings):
    with pytest.raises(ValueError):
        settings.validate()
