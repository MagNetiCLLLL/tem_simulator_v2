"""Analytic and backend checks of shared post-gun electric transport."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad

from temsim.physics import core
from temsim.physics.closed_gun_field import ClosedGunField
from temsim.physics.instrument_electric import InstrumentElectricField
from temsim.physics.electrostatic_column_transport import _momentum_speed


def fixture(*, gradient=2e6, radial=0., step=.1, mode='analytical'):
    state=SimpleNamespace(lenses=[],stigmators=[],corrector_elements=[],deflectors=[],
        beam_voltage_kv=300.,step_mm=step,history_step_mm=1.,acceleration_enabled=False,
        acceleration_backend='CPU',simulation_mode=mode,projector_mode='diffraction',
        equivalent_image_lenses_enabled=False,sample=SimpleNamespace(z_mm=20.),simulation_time_s=0.)
    plan=core.build_propagation_plan(state,0.,10.,checkpoint_z_mm=(0.,4.,10.))
    r=np.array([0.,.01,.02]);z=np.array([0.,.005,.02])
    voltage=gradient*z[None,:]+radial*r[:,None]**2
    base=ClosedGunField({},r,z,voltage)
    field=InstrumentElectricField(base,base,SimpleNamespace(exit_plane_z_mm=0.),
        np.array([[-.02,-.02,0.],[.02,.02,.02]]),'fixture','fixture','fixture',())
    plan=replace(plan,electric_field=field,electric_field_identity='fixture',
                 electric_reference_invariant_ev=300000.)
    return state,plan


def trace(state,plan,*,tx=1e-3,energy=300000.,time=0.,**kwargs):
    count=np.size(energy)
    sources=(np.zeros(count),np.broadcast_to(tx,(count,)).copy(),np.zeros(count),np.zeros(count))
    energy_output=[]
    result=core.execute_propagation_plan(state,plan,*sources,
        initial_kinetic_energy_ev=np.broadcast_to(energy,(count,)),
        initial_time_s=np.broadcast_to(time,(count,)),return_flight_times=True,
        energy_output=energy_output,**kwargs)
    return result,energy_output[0]


def test_axial_electric_acceleration_compresses_angle_and_keeps_actual_energy():
    state,plan=fixture()
    result,energy=trace(state,plan)
    points=result[-1]
    p0=_momentum_speed(300000.)[0]
    expected_k=300000.+2e6*points.z_mm*1e-3
    expected_angle=np.array([1e-3*p0/_momentum_speed(k)[0] for k in expected_k])
    np.testing.assert_allclose(points.kinetic_energy_ev[:,0],expected_k,rtol=2e-15)
    np.testing.assert_allclose(points.tx_rad[:,0],expected_angle,rtol=2e-13)
    expected_x=quad(lambda zz:1e-3*p0/_momentum_speed(300000.+2e6*zz)[0],0.,.01,epsabs=1e-20)[0]
    expected_t=quad(lambda zz:np.sqrt(1.+(1e-3*p0/_momentum_speed(300000.+2e6*zz)[0])**2)/_momentum_speed(300000.+2e6*zz)[1],0.,.01,epsabs=1e-24)[0]
    assert points.x_m[-1,0]==pytest.approx(expected_x,rel=2e-13)
    assert points.flight_time_s[-1,0]==pytest.approx(expected_t,rel=2e-13)
    np.testing.assert_array_equal(energy[-1],points.kinetic_energy_ev[-1])


@pytest.mark.parametrize('mode',['analytical','ideal'])
def test_electric_cutoff_resume_uses_actual_energy_and_fixed_ideal_reference(mode):
    state,plan=fixture(mode=mode,radial=2e7)
    full,_=trace(state,plan,energy=np.array([280000.,310000.]))
    cp=full[-1];row=1;energies=[]
    resumed=core.execute_propagation_plan(state,plan,cp.x_m[row],cp.tx_rad[row],cp.y_m[row],cp.ty_rad[row],
        start_index=int(plan.checkpoint_index[row]),include_initial_plane_kicks=False,
        initial_time_s=cp.flight_time_s[row],return_flight_times=True,
        initial_kinetic_energy_ev=cp.kinetic_energy_ev[row],energy_output=energies)[-1]
    for name in ('x_m','tx_rad','y_m','ty_rad','flight_time_s','kinetic_energy_ev'):
        np.testing.assert_allclose(getattr(resumed,name)[-1],getattr(cp,name)[-1],rtol=4e-14,atol=1e-24)
    if mode=='ideal':
        np.testing.assert_array_equal(cp.x_m[:,0],cp.x_m[:,1])
        assert cp.flight_time_s[-1,0]>cp.flight_time_s[-1,1]


@pytest.mark.parametrize('backend',['Numba CPU','CUDA GPU'])
def test_closed_electric_and_magnetic_actions_match_reference_backend(monkeypatch,backend):
    if backend=='CUDA GPU':
        from temsim.physics.compute_backend import cuda_capability
        if not cuda_capability().available:pytest.skip('No CUDA hardware')
    elif not core.NUMBA_AVAILABLE:pytest.skip('No Numba')
    state,plan=fixture(radial=2e7,step=.2)
    magnetic=np.full_like(plan.magnetic_t,.015)
    mid=np.full_like(plan.midpoint_magnetic_t,.015)
    kicks=plan.kick_x_rad.copy();kicks[np.searchsorted(plan.z_mm,4.)]=1e-4
    cs=plan.cs_kick_m3.copy();cs[np.searchsorted(plan.z_mm,4.)]=1e8
    plan=replace(plan,magnetic_t=magnetic,midpoint_magnetic_t=mid,kick_x_rad=kicks,cs_kick_m3=cs)
    reference,_=trace(state,plan,energy=np.array([270000.,300000.,330000.]))
    monkeypatch.setattr(core,'choose_ray_backend',lambda *_a,**_kw:(backend,None))
    state.acceleration_enabled=True;state.acceleration_backend='Require GPU' if backend=='CUDA GPU' else backend
    actual,_=trace(state,plan,energy=np.array([270000.,300000.,330000.]))
    for name in ('x_m','tx_rad','y_m','ty_rad','flight_time_s','kinetic_energy_ev'):
        np.testing.assert_allclose(getattr(actual[-1],name),getattr(reference[-1],name),rtol=3e-11,atol=1e-21)


def test_constant_potential_reduces_to_existing_magnetic_column():
    state,plan=fixture(gradient=0.)
    field=np.full_like(plan.magnetic_t,.02);mid=np.full_like(plan.midpoint_magnetic_t,.02)
    plan=replace(plan,magnetic_t=field,midpoint_magnetic_t=mid)
    actual,_=trace(state,plan)
    magnetic=replace(plan,electric_field=None,electric_field_identity=None,electric_reference_invariant_ev=None)
    expected,_=trace(state,magnetic)
    for name in ('x_m','tx_rad','y_m','ty_rad','flight_time_s'):
        np.testing.assert_allclose(getattr(actual[-1],name),getattr(expected[-1],name),rtol=5e-14,atol=1e-21)


def test_missing_runtime_field_cannot_execute_an_archived_electric_plan():
    state,plan=fixture()
    with pytest.raises(ValueError,match='captured electric field reconstructed'):
        trace(state,replace(plan,electric_field=None))


def test_unknown_electric_identity_cannot_admit_common_prefix():
    _,plan=fixture()
    plan=replace(plan,electric_field_identity=None)
    assert core.propagation_plan_common_prefix_nodes(plan,plan)==0


def test_electric_resume_requires_actual_checkpoint_energy():
    state,plan=fixture()
    with pytest.raises(ValueError,match="checkpoint's actual kinetic energy"):
        core.execute_propagation_plan(state,plan,*(np.zeros(1),)*4,start_index=1)


def test_paused_wave_rejects_active_electric_column_before_allocating_wave():
    from temsim.physics.wave_field_admission import require_supported_wave_dipoles
    state,plan=fixture()
    with pytest.raises(ValueError,match='Paused wave transport'):
        require_supported_wave_dipoles(state,0.,10.,plan)


def test_clipped_unknown_particles_keep_nan_energy_without_invalidating_live_ray():
    state,plan=fixture()
    result=core.execute_propagation_plan(state,plan,
        np.array([0.,np.nan]),np.array([1e-3,np.nan]),np.array([0.,np.nan]),np.array([0.,np.nan]),
        initial_kinetic_energy_ev=np.array([300000.,np.nan]),defer_nonfinite_until_clipping=True)[-1]
    assert np.isfinite(result.kinetic_energy_ev[:,0]).all()
    assert np.isnan(result.kinetic_energy_ev[:,1]).all()


@pytest.mark.parametrize('mode', ['analytical', 'ideal'])
def test_alignment_observers_include_captured_electric_field(monkeypatch, mode):
    from temsim.optics.direct_alignment import _LiveFirstOrderModel, diffraction_transfer
    from temsim.physics.first_order import trace_transverse_transfer
    state, plan = fixture(mode=mode)
    state.sample.z_mm = 0.
    # An isolated analytic E field is explicitly bound to this mathematical gun
    # fixture; no production physical gun is replaced or allowed to omit E.
    state.electron_gun = SimpleNamespace(exit_plane_z_mm=0.)
    monkeypatch.setattr('temsim.physics.instrument_electric.capture_instrument_electric_field',
                        lambda _state: plan.electric_field)
    model = _LiveFirstOrderModel(state, 0., 10., (), step_mm=.1)
    assert model.full_field_transfer
    expected = trace_transverse_transfer(state, 0., 10., maximum_step_mm=.025)
    np.testing.assert_allclose(model.matrix([]), expected.matrix, rtol=2e-12, atol=1e-18)
    actual = diffraction_transfer(state, 10.)
    np.testing.assert_allclose(actual.matrix, expected.matrix, rtol=2e-12, atol=1e-18)
    np.testing.assert_allclose(actual.position_offset_m, expected.position_offset_m, atol=1e-20)
    # Longitudinal acceleration changes the angular map even with no magnetic
    # elements: a B-only observer would incorrectly return a unit diagonal.
    assert actual.k_diff[0, 0] < .99


def test_alignment_reduced_model_requires_explicit_constant_field_proof(monkeypatch):
    from temsim.optics.direct_alignment import _active_column_electric_field
    state, _ = fixture()
    def forbidden(_state):
        raise AssertionError('A no-gun mathematical fixture must not require a physical E provider')
    monkeypatch.setattr('temsim.physics.instrument_electric.capture_instrument_electric_field', forbidden)
    assert _active_column_electric_field(state, 0., 10.) is None
    state.electron_gun = SimpleNamespace()
    intervals = []
    explicit_constant = SimpleNamespace(is_constant_on_interval=lambda lo, hi: intervals.append((lo, hi)) or True)
    monkeypatch.setattr('temsim.physics.instrument_electric.capture_instrument_electric_field',
                        lambda _state: explicit_constant)
    assert _active_column_electric_field(state, 2., 10.) is None
    assert intervals == [(2., 10.)]


def test_equivalent_image_search_preserves_lens_actions_with_electric_field(monkeypatch):
    from temsim.optics import direct_alignment as alignment
    state, plan = fixture()
    state.electron_gun = SimpleNamespace()
    state.equivalent_image_lenses_enabled = True
    state.lenses = [SimpleNamespace(key=key, percent=10.) for key in alignment.IMAGE_KEYS]
    monkeypatch.setattr('temsim.physics.instrument_electric.capture_instrument_electric_field',
                        lambda _state: plan.electric_field)
    monkeypatch.setattr(alignment, 'equivalent_image_calibrations', lambda *_args:
                        tuple(SimpleNamespace(key=key, maximum_percent=100.) for key in alignment.IMAGE_KEYS))
    def complete_trace(captured, source, target, **_kwargs):
        assert captured.equivalent_image_lenses_enabled
        assert [lens.percent for lens in captured.lenses] == [20.] * len(alignment.IMAGE_KEYS)
        assert source == 0. and target == 10.
        return SimpleNamespace(matrix=np.eye(4) * 2.)
    monkeypatch.setattr(alignment, 'trace_transverse_transfer', complete_trace)
    def forbidden(*_args):
        raise AssertionError('The thin-lens-only formula must not silently omit an active E field')
    monkeypatch.setattr(alignment, 'equivalent_image_transfer_matrix', forbidden)
    model = alignment._EquivalentImageFirstOrderModel(state, 0., 10.)
    np.testing.assert_array_equal(model.matrix([20.] * len(alignment.IMAGE_KEYS)), np.eye(4) * 2.)
    assert [lens.percent for lens in state.lenses] == [10.] * len(alignment.IMAGE_KEYS)
    assert state.equivalent_image_lenses_enabled


def test_alignment_real_rays_keep_executed_gun_energy(monkeypatch):
    from temsim.optics import direct_alignment as alignment
    state, plan = fixture()
    state.electron_gun = SimpleNamespace()
    monkeypatch.setattr('temsim.physics.instrument_electric.capture_instrument_electric_field',
                        lambda _state: plan.electric_field)
    model = alignment._LiveFirstOrderModel(state, 0., 10., (), step_mm=.1)
    energy = np.array([275000., 320000.])
    def complete_propagate(_state, _start, _end, *rays, **kwargs):
        np.testing.assert_array_equal(kwargs['initial_kinetic_energy_ev'], energy)
        return (np.array([10.]), *(np.asarray(ray)[None, :] for ray in rays))
    monkeypatch.setattr(alignment, 'propagate', complete_propagate)
    result = model.rays_at([], np.zeros((4, 2)), (10.,), initial_kinetic_energy_ev=energy)
    assert result.shape == (1, 4, 2)


def test_medium_grid_and_transport_start_with_explicit_actual_energy(monkeypatch):
    state, plan = fixture()
    state.vacuum_map = SimpleNamespace(enabled=True, max_transport_nodes=100000)
    energy = np.array([210000., 285000.])
    captures = []
    def build(*_args, **kwargs):
        np.testing.assert_array_equal(kwargs['medium_energy_ev'], energy)
        return plan
    def medium(_state, _plan, _count, actual_energy, **_kwargs):
        np.testing.assert_array_equal(actual_energy, energy)
        captures.append(actual_energy.copy())
        return object()
    monkeypatch.setattr(core, 'build_propagation_plan', build)
    monkeypatch.setattr('temsim.physics.residual_medium.ColumnMediumTransport', medium)
    monkeypatch.setattr(core, 'execute_propagation_plan', lambda *_args, **_kwargs: (None,) * 6)
    core.propagate(state, 0., 10., *(np.zeros(2),) * 4,
                   particle_medium=True, energy_offset_ev=np.zeros(2), initial_kinetic_energy_ev=energy)
    assert len(captures) == 1


def test_proven_constant_electric_fast_path_keeps_fixed_ideal_reference():
    state, plan = fixture(gradient=0., mode='ideal')
    plan = replace(plan, magnetic_t=np.full_like(plan.magnetic_t, .02),
                   midpoint_magnetic_t=np.full_like(plan.midpoint_magnetic_t, .02),
                   electric_reference_invariant_ev=270000.)
    expected, _ = trace(state, plan)
    field = plan.electric_field
    proven_constant = SimpleNamespace(is_constant_on_interval=lambda *_args: True,
        potential_rise_v_at_global_positions=field.potential_rise_v_at_global_positions)
    actual, _ = trace(state, replace(plan, electric_field=proven_constant))
    for name in ('x_m', 'tx_rad', 'y_m', 'ty_rad', 'flight_time_s', 'kinetic_energy_ev'):
        np.testing.assert_allclose(getattr(actual[-1], name), getattr(expected[-1], name),
                                   rtol=1e-12, atol=1e-21)


@pytest.mark.parametrize('enabled,policy,expected', [
    (True, 'Auto', core.BACKEND_NUMBA),
    (True, 'CPU', core.BACKEND_CPU),
    (False, 'Auto', core.BACKEND_CPU),
])
def test_long_small_electric_batch_uses_compiled_auto_without_overriding_cpu(monkeypatch, enabled, policy, expected):
    from temsim.physics import electrostatic_column_transport as electric
    state, plan = fixture(step=.02)
    state.acceleration_enabled = enabled
    state.acceleration_backend = policy
    monkeypatch.setattr(core, 'NUMBA_AVAILABLE', True)
    monkeypatch.setattr(core, 'choose_ray_backend', lambda *_args, **_kwargs: (core.BACKEND_CPU, None))
    calls = []
    def kernel(inputs, *, backend, serial, **_kwargs):
        calls.append((backend, serial))
        count, saved, checkpoints = len(inputs[10]), len(inputs[16]), len(inputs[17])
        phase = [np.zeros((saved, count)) for _ in range(4)]
        captured = [np.zeros((checkpoints, count)) for _ in range(4)]
        return (*phase, *captured, np.zeros((saved, count)), np.zeros((checkpoints, count)),
                np.full((saved, count), 300000.), np.full((checkpoints, count), 300000.)), backend, None
    monkeypatch.setattr(electric, 'electrostatic_column_rk4', kernel)
    core.execute_propagation_plan(state, plan, *(np.zeros(9),)*4)
    assert calls == [(expected, True)]
