"""Analytical focus-plane/gate tests; actual gun verification is separate."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.surface_probe_focus import (
    SurfaceFocusMeasurement, local_waist_from_radii, measure_surface_focus, surface_z_mm,
)
from temsim.physics.beam_statistics import transverse_beam_statistics


def test_surface_uses_half_thickness_without_moving_sample_or_objective():
    state=SimpleNamespace(sample=SimpleNamespace(z_mm=1599.2,thickness_nm=10.))
    assert surface_z_mm(state)==pytest.approx(1599.199995,abs=1e-10)
    assert state.sample.z_mm==1599.2
    state.sample.thickness_nm=0
    assert surface_z_mm(state)==1599.2
    state.sample.thickness_nm=-1
    with pytest.raises(ValueError):
        surface_z_mm(state)


@pytest.mark.parametrize('focus_nm',[-20.,0.,.2,20.])
def test_one_sided_variance_derivative_finds_analytical_drift_waist(focus_nm):
    h=1e-7
    z=np.array([-2*h,-h,0.])
    r=np.sqrt((2e-6)**2+.03**2*(z-focus_nm*1e-9)**2)
    offset,second=local_waist_from_radii(r,h)
    assert offset==pytest.approx(focus_nm,abs=.0001)
    assert second==pytest.approx(2*.03**2,rel=1e-6)


def test_envelope_maximum_is_not_a_probe_waist():
    z=np.array([-2e-7,-1e-7,0.])
    offset,second=local_waist_from_radii(np.sqrt((2e-6)**2-.03**2*z*z),1e-7)
    assert second<0
    assert np.isnan(offset)


@pytest.mark.parametrize('r,h',[([1,2],1.),([1,-1,1],1.),([1,1,np.nan],1.),([1,1,1],0.)])
def test_invalid_focus_derivatives_rejected(r,h):
    with pytest.raises(ValueError):
        local_waist_from_radii(r,h)


def fixture_measurement():
    phi=np.arange(32)*2*np.pi/32
    # Position and angle in quadrature: stationary radial variance, alpha=30 mrad.
    stats=transverse_beam_statistics(1e-9*np.cos(phi),1e-9*np.sin(phi),
        -.03*np.sin(phi),.03*np.cos(phi),weights=np.ones(32)/32)
    return SurfaceFocusMeasurement(100.,stats,0.,.001,(1e-9,1e-9,1e-9),.05)


def test_acceptance_requires_current_correct_angle_and_true_minimum():
    m=fixture_measurement()
    assert m.accepts(30.)
    assert not m.accepts(0.)
    assert not m.accepts(25.)
    assert not replace(m,variance_second_derivative=-1).accepts(30.)
    assert not replace(m,local_waist_offset_nm=2.).accepts(30.)
    assert not replace(m,statistics=replace(m.statistics,surviving_rays=3)).accepts(30.)
    assert not replace(m,statistics=replace(m.statistics,surviving_fraction=0.)).accepts(30.)
    assert not replace(m,statistics=replace(m.statistics,waist_offset_m=1e-6)).accepts(30.)


def test_no_gun_execution_when_medium_not_qualified():
    state=SimpleNamespace(electron_gun=SimpleNamespace(source_representation='classical_particles'),
                          vacuum_map=SimpleNamespace(enabled=True))
    with pytest.raises(ValueError,match='residual-medium'):
        measure_surface_focus(state)


def test_small_physical_current_is_not_rejected_as_missing_crossovers():
    m=fixture_measurement()
    small=replace(m,statistics=replace(m.statistics,surviving_fraction=1e-5),effective_rays=24.)
    assert small.accepts(30.)
    assert not small.accepts(30.,minimum_current_fraction=.01)
    assert not replace(small,effective_rays=2.).accepts(30.)
    assert not replace(small,effective_rays=float('inf')).accepts(30.)
    assert not replace(small,effective_rays=float('nan')).accepts(30.)
    assert replace(small,effective_rays=16.-np.finfo(float).eps*16).accepts(30.)
    assert not replace(small,statistics=replace(small.statistics,surviving_fraction=0.)).accepts(30.)


def test_measure_uses_exact_full_precision_checkpoints_not_plot_history(monkeypatch):
    from temsim.optics.column import default_state
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.optics.electron_gun import source
    from temsim.optics import surface_probe_focus as module
    state=default_state()
    before=capture_instrument_snapshot(state).digest
    angle=np.arange(33)*2*np.pi/32
    x=1e-9*np.cos(angle)
    y=1e-9*np.sin(angle)
    tx=-.03*np.sin(angle)
    ty=.03*np.cos(angle)
    # A zero-current support ray is deliberately far from the physical probe.
    x[-1]=1.
    weights=np.r_[np.full(32,1/32),0.]
    alive=np.ones(33,bool)
    alive[0]=False
    emitted=SimpleNamespace(x_m=x,tx_rad=tx,y_m=y,ty_rad=ty,
        weight=weights,alive=alive,energy_offset_ev=np.arange(33)*.01)
    gun=SimpleNamespace(exit_bundle=emitted,blocked_z_mm=np.r_[10.,np.full(32,np.nan)],
                        blocked_key=['earlier stop']+['']*32)
    calls=[]
    monkeypatch.setattr(source,'trace_source_to_exit',lambda s:calls.append(s) or gun)

    def execute(s,start,end,*args,**kwargs):
        assert end==state.sample.upper_surface_z_mm
        planes=kwargs['checkpoint_z_mm']
        assert all(z in kwargs['save_z_mm'] for z in planes)
        assert max(planes)==end and min(planes)<end
        assert kwargs['return_checkpoints']
        assert kwargs['energy_offset_ev'] is emitted.energy_offset_ev
        dz=(np.array(planes)-end)[:,None]*1e-3
        cp=SimpleNamespace(z_mm=np.array(planes),x_m=x+dz*tx,y_m=y+dz*ty,
                           tx_rad=np.broadcast_to(tx,(3,33)),ty_rad=np.broadcast_to(ty,(3,33)))
        # These discarded drawing histories would incorrectly imply zero angle.
        history=np.zeros((2,33),np.float32)
        return np.array([start,end]),history,history,history,history,cp

    monkeypatch.setattr(module,'propagate',execute)
    result=measure_surface_focus(state)
    expected=transverse_beam_statistics(x,y,tx,ty,alive=alive&(weights>0),weights=weights)
    assert calls==[state]
    assert result.statistics==expected
    assert result.statistics.surviving_rays==31
    assert result.statistics.convergence_95_mrad>29.
    assert capture_instrument_snapshot(state).digest==before


@pytest.mark.parametrize('reject_refinement',[False,True])
def test_refinement_is_detached_and_failed_step_check_cannot_supply_a_profile(monkeypatch,reject_refinement):
    from temsim.optics.column import default_state
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.optics import surface_probe_focus as module
    state=default_state()
    before=capture_instrument_snapshot(state).digest

    def fixture_forward(s,*,step_mm):
        lenses={lens.key:lens for lens in s.lenses}
        m=fixture_measurement()
        stats=replace(m.statistics,
            convergence_95_rad=(30.+.1*(lenses['condenser_lens_3'].percent-35.))*1e-3,
            waist_offset_m=(lenses['objective_lens'].percent-70.)*1e-6)
        return replace(m,statistics=stats,step_mm=step_mm,
                       local_waist_offset_nm=2. if reject_refinement and step_mm==.05 else stats.waist_offset_m*1e9)

    monkeypatch.setattr(module,'measure_surface_focus',fixture_forward)
    candidate,evidence=module.refine_surface_focus(state,30.,initial=[40.,69.])
    assert capture_instrument_snapshot(state).digest==before
    assert evidence['sampling_convergence']=='NOT_TESTED'
    if reject_refinement:
        assert candidate is None
        assert evidence['status']=='FAIL'
    else:
        assert candidate is not state
        assert evidence['status']=='PASS_AT_FIXED_PARTICLE_BUDGET'
        assert evidence['strengths']==pytest.approx({'condenser_lens_3':35.,'objective_lens':70.})
        # Put the two permitted controls back; the entire original graph must match.
        previous={lens.key:lens for lens in state.lenses}
        for lens in candidate.lenses:
            if lens.key in evidence['strengths']:
                lens.percent=previous[lens.key].percent
        assert capture_instrument_snapshot(candidate).digest==before
