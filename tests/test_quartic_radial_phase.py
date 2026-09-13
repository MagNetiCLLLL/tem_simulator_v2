"""Independent coordinate calculus and phase export; not imaging acceptance."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.special import eval_laguerre, eval_genlaguerre, roots_laguerre
from threadpoolctl import threadpool_limits

from temsim.physics.quartic_radial_phase import quartic_operators, quartic_wave_moments, apply_quartic_on_grid
from temsim.physics.wave_following_chart import ExecutedWaveChart, WaveFollowingNumerics


def test_streamed_envelope_matches_independent_polynomials_without_dropping_coefficients():
    from temsim.physics.quartic_radial_phase import radial_envelope
    from temsim.physics.radial_gun_wave import basis_values
    rng=np.random.default_rng(17)
    a=rng.normal(size=64)+1j*rng.normal(size=64)
    r=np.linspace(0,60,8193)
    actual=radial_envelope(r,2.3,a)
    expected=basis_values(r,2.3,0.,1.,len(a))@a
    np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-12)


def test_kinetic_connection_and_outside_closure_match_real_space_derivatives():
    count, b, c, k, rate, prime, gamma, gamma_prime = 7, 1.7, .04, 20., .03, -.002, .003, .0007
    x, weight = roots_laguerre(96)
    r = b*np.sqrt(x)
    p = np.stack([eval_laguerre(n, x) for n in range(count)], axis=1)
    dp = np.stack([np.zeros_like(x) if n == 0 else -eval_genlaguerre(n-1, 1, x)
        for n in range(count)], axis=1)
    dr = (2*r/b**2)[:, None]*(dp-.5*p)+1j*(k*c*r+4*gamma*r**3)[:, None]*p
    dz = 1j*(.5*k*prime*r*r+gamma_prime*r**4)[:, None]*p-rate*(p+2*x[:, None]*(dp-.5*p))
    expected_k = dr.conj().T@(weight[:, None]*dr)
    expected_g = -1j*p.T@(weight[:, None]*dz)
    expected_c = dz.conj().T@(weight[:, None]*dz)-expected_g@expected_g
    kinetic, connection, closure = quartic_operators(count, b, c, k, rate, prime, gamma, gamma_prime)
    np.testing.assert_allclose(kinetic, expected_k, rtol=3e-11, atol=5e-11)
    np.testing.assert_allclose(connection, expected_g, rtol=3e-11, atol=5e-11)
    np.testing.assert_allclose(closure, expected_c, rtol=3e-10, atol=5e-10)
    np.testing.assert_allclose(kinetic, kinetic.conj().T, atol=1e-12)
    assert np.linalg.eigvalsh(closure).min() > -1e-12
    assert closure[-2, -2] > 0  # Not the former last-diagonal-only closure.


def test_zero_quartic_reduces_to_existing_quadratic_operators():
    from temsim.physics.radial_gun_wave import laguerre_operators
    n, b, c, k, rate, prime = 16, 3., .02, 100., .003, .00001
    x, d, t = laguerre_operators(n)
    kinetic, connection, closure = quartic_operators(n, b, c, k, rate, prime)
    np.testing.assert_allclose(kinetic, t/b**2+(k*c*b)**2*x-2j*k*c*d, atol=1e-12)
    np.testing.assert_allclose(connection, .5*k*prime*b*b*x+1j*rate*d, atol=1e-12)
    expected = np.zeros((n,n)); expected[-1,-1] = n*n*((.5*k*prime*b*b)**2+rate**2)
    np.testing.assert_allclose(closure, expected, atol=1e-12)


def test_quartic_moments_keep_existing_phase_and_match_gradient_least_squares():
    b, c, k, gamma = 2.3, .04, 120., .0001
    a = np.array((1+.2j, .3-.8j, .4+.6j, -.7j))
    x, w = roots_laguerre(64); rho = np.sqrt(x)
    p = np.stack([eval_laguerre(n,x) for n in range(len(a))],axis=1)
    dp = np.stack([np.zeros_like(x) if n==0 else -eval_genlaguerre(n-1,1,x) for n in range(len(a))],axis=1)
    f = p@a; derivative = 2*rho*((dp-.5*p)@a)
    norm = w@abs(f)**2
    design = np.stack((rho*f,rho**3*f),axis=1)
    gram = (design.conj().T@(w[:,None]*design)).real/norm
    rhs = (design.conj().T@(w*derivative)).imag/norm
    fit = np.linalg.solve(gram,rhs)
    intrinsic = (w@abs(derivative)**2/norm-rhs@fit)/b**2
    actual = quartic_wave_moments(a,b,c,k,gamma)
    assert actual['optimal_curvature_per_nm'] == pytest.approx(c+fit[0]/(k*b*b),abs=1e-13)
    assert actual['optimal_quartic_phase_per_nm4'] == pytest.approx(gamma+fit[1]/(4*b**4),abs=1e-13)
    assert actual['intrinsic_momentum_squared_nm2'] == pytest.approx(intrinsic,rel=1e-12)
    original = quartic_wave_moments([1,0,0],b,c,k,gamma)
    assert original['optimal_width_nm'] == pytest.approx(b)
    assert original['optimal_quartic_phase_per_nm4'] == gamma


def test_quartic_chart_derivative_includes_transition_and_is_bound_to_identity():
    z = np.geomspace(2,1000,80)
    settings = WaveFollowingNumerics(iterations=1,phase_order=4)
    chart = ExecutedWaveChart(z,np.sqrt(z),1/z,settings,{},quartic_per_nm4=.001/z**2)
    assert chart.quartic_phase(2) == (0.,0.)
    for value in (10.,15.,50.,99.,100.,200.):
        h=1e-4
        assert (chart.quartic_phase(value+h)[0]-chart.quartic_phase(value-h)[0])/(2*h) == pytest.approx(chart.quartic_phase(value)[1],abs=1e-14)
    other = ExecutedWaveChart(z,np.sqrt(z),1/z,settings,{},quartic_per_nm4=.002/z**2)
    assert other.digest != chart.digest


def test_nonparaxial_free_wave_keeps_phase_through_changing_quartic_coordinates():
    from scripts.inspect_moving_radial_frame import calculate
    def phase(z):
        t=z/100
        return .005*t**3*(10+t*(-15+6*t)), .005*30*t*t*(1-t)**2/100
    with threadpool_limits(1):
        coarse=calculate(.05,count=32,quartic_chart=phase)
        result=calculate(.05,count=48,quartic_chart=phase)
    assert result['relative_complex_error'] < 2e-6
    assert result['relative_complex_error'] < coarse['relative_complex_error']/5


def test_export_and_downstream_radial_replay_retain_quartic_phase():
    from temsim.physics.radial_column_wave import from_executed_gun
    n=256; step=.04; axis=(np.arange(n)-n//2)*step
    xx,yy=np.meshgrid(axis,axis); radius=np.hypot(xx,yy)
    amplitude=np.exp(-radius**2/2)/np.sqrt(np.pi)*step
    result=apply_quartic_on_grid(amplitude,radius,step,.001)
    np.testing.assert_allclose(result,amplitude*np.exp(.001j*radius**4))
    payload={'mode_id':'fixture','width_nm':1.,'coefficients_real':[1.], 'coefficients_imag':[0.],
             'curvature_m1':0.,'quartic_phase_per_nm4':.001}
    mode=SimpleNamespace(mode_id='fixture',plane=SimpleNamespace(probability=1.))
    replay=from_executed_gun(mode,payload,8192)
    r=replay.radius_m*1e9
    np.testing.assert_allclose(replay.amplitude,np.exp(-r*r/2+.001j*r**4)/np.sqrt(np.pi)*1e9,rtol=1e-13,atol=1e-7)
    with pytest.raises(ValueError,match='finer Cartesian grid'):
        apply_quartic_on_grid(amplitude,radius,step,100.)


def test_comparison_cannot_ignore_quartic_phase_when_density_is_identical():
    from scripts.compare_gun_boundaries import boundary_metrics
    from scripts.inspect_basis_support import sampled_basis
    row={'z_nm':1.,'width_nm':1.,'reference_k_per_nm':100.,'curvature_per_nm':0.,
         'coefficients_real':[1.],'coefficients_imag':[0.]}
    changed={**row,'quartic_phase_per_nm4':.1}
    metrics=boundary_metrics(row,changed)
    assert metrics['relative_complex_l2'] > .1
    assert metrics['relative_amplitude_l2'] < 1e-14
    r=np.linspace(0,3,100)
    np.testing.assert_allclose(sampled_basis(changed,r,1)[:,0],sampled_basis(row,r,1)[:,0]*np.exp(.1j*r**4))


def test_initial_radial_sampling_refines_executed_field_without_dropping_phase():
    from temsim.physics.radial_column_wave import from_executed_gun, RadialExecution
    payload={'mode_id':'fixture','width_nm':1.,'coefficients_real':[1.], 'coefficients_imag':[0.],
             'curvature_m1':0.,'quartic_phase_per_nm4':.01}
    mode=SimpleNamespace(mode_id='fixture',plane=SimpleNamespace(probability=1.))
    load=lambda samples: from_executed_gun(mode,payload,samples)
    execution=RadialExecution(load,128,32768)
    assert execution.samples > 128
    assert execution.refinements[0]['kind'] == 'input_replay'
    r=execution.wave.radius_m*1e9
    np.testing.assert_allclose(execution.wave.amplitude,np.exp(-r*r/2+.01j*r**4)/np.sqrt(np.pi)*1e9,rtol=1e-13,atol=1e-7)
    with pytest.raises(ValueError,match='above its 128 budget'):
        RadialExecution(load,128,128)


def test_quartic_metadata_cannot_be_discarded_or_extrapolated():
    z=np.array([2.,10.,100.,1000.])
    with pytest.raises(ValueError,match='silently omitted'):
        ExecutedWaveChart(z,z,1/z,WaveFollowingNumerics(),{},quartic_per_nm4=1/z**4)
    chart=ExecutedWaveChart(z,z,1/z,WaveFollowingNumerics(phase_order=4),{},quartic_per_nm4=1/z**4)
    with pytest.raises(ValueError,match='extrapolate'):
        chart.quartic_phase(1001.)
    with pytest.raises(ValueError,match='phase order'):
        WaveFollowingNumerics(phase_order=True).validate()
