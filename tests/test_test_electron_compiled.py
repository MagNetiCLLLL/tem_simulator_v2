"""Compiled/reference parity and genuine streamed-state regression checks."""
from dataclasses import replace
import numpy as np
import pytest
from temsim.magnetic_test_particle import TestElectronSettings,trace_test_electron
from temsim.test_electron_compiled import prepare_compiled_fields,compiled_fields

@pytest.fixture(scope='module')
def captured_scene():
    from temsim.cpu_resources import numerical_job
    from temsim.optics.column import default_state
    from temsim.magnetic_field_scene import prepare_magnetic_scene
    from temsim.test_electron_scene import prepare_test_electron_scene
    with numerical_job(1):
        state=default_state()
        magnetic=prepare_magnetic_scene(state,z_limits_mm=(0.,3026.4))
        return prepare_test_electron_scene(state,magnetic,z_limits_mm=(0.,3026.4))


def settings(scene,**kwargs):
    return TestElectronSettings(kinetic_energy_ev=scene.initial_energy_ev,position_m=scene.initial_position_m,
        max_path_length_m=kwargs.pop('max_path_length_m',.55),step_m=.001,max_steps=20000,**kwargs)


def test_compiled_scalar_fields_match_native_every_source_and_support(captured_scene):
    scene=captured_scene
    data=prepare_compiled_fields(scene)
    assert data is not None
    for source in scene.magnetic_scene._sources:
        for z in np.linspace(source.bounds_m[0,2],source.bounds_m[1,2],9):
            for fraction in (0.,.013,.39,1.01):
                point=np.array([fraction*source.radius_m, -.23*fraction*source.radius_m,z])
                reference=scene.diagnostic_fields_at_global_position(point)
                valid,magnetic,electric,potential=compiled_fields(point,data)
                assert valid == (reference is not None)
                if valid:
                    np.testing.assert_allclose(magnetic,reference[0],rtol=2e-7,atol=1e-12)
                    np.testing.assert_allclose(electric,reference[1],rtol=1e-12,atol=1e-9)
                    assert potential == pytest.approx(reference[2],rel=2e-15,abs=1e-10)


@pytest.mark.parametrize('angle,azimuth,length',[(0.,0.,3.0264),(2.61,0.,3.0264),(5.,45.,.55),(175.,45.,.001)])
def test_compiled_complete_paths_match_reference_and_hardware_stops(captured_scene,angle,azimuth,length):
    scene=captured_scene
    cfg=settings(scene,polar_angle_deg=angle,azimuth_angle_deg=azimuth,max_path_length_m=length)
    reference=trace_test_electron(scene,cfg,use_compiled=False)
    actual=trace_test_electron(scene,cfg,use_compiled=True)
    assert actual.reason==reference.reason
    assert actual.completed==reference.completed
    assert actual.steps==reference.steps
    np.testing.assert_allclose(actual.positions_m,reference.positions_m,rtol=2e-7,atol=2e-12)
    np.testing.assert_allclose(actual.momentum_kg_m_per_s,reference.momentum_kg_m_per_s,rtol=2e-7,atol=1e-29)
    np.testing.assert_allclose(actual.time_s,reference.time_s,rtol=2e-10,atol=1e-17)
    np.testing.assert_allclose(actual.path_length_m,reference.path_length_m,rtol=2e-10,atol=2e-12)
    assert actual.energy_invariant_error_ev<1e-5


def test_compiled_streaming_prefix_is_executed_immutable_and_does_not_change_result(captured_scene):
    cfg=settings(captured_scene,polar_angle_deg=5.,azimuth_angle_deg=45.)
    prefixes=[]
    result=trace_test_electron(captured_scene,cfg,progress=prefixes.append,progress_interval_s=0.)
    plain=trace_test_electron(captured_scene,cfg)
    assert len(prefixes)>1
    assert all(a.steps<b.steps for a,b in zip(prefixes,prefixes[1:]))
    for prefix in prefixes:
        assert prefix.reason=='in_progress' and not prefix.completed
        for name,value in vars(prefix).items():
            if isinstance(value,np.ndarray):
                assert not value.flags.writeable
                np.testing.assert_array_equal(value,getattr(result,name)[:len(value)])
    for name,value in vars(result).items():
        if isinstance(value,np.ndarray):np.testing.assert_array_equal(value,getattr(plain,name))


def test_compiled_progress_cancellation_returns_last_executed_state(captured_scene):
    prefixes=[]
    result=trace_test_electron(captured_scene,settings(captured_scene),progress=prefixes.append,
                              progress_interval_s=0.,cancelled=lambda:bool(prefixes))
    assert result.reason=='cancelled' and not result.completed
    assert len(prefixes)==1
    np.testing.assert_array_equal(result.positions_m,prefixes[0].positions_m)


def test_progress_interval_validation_and_unsupported_provider_fallback(captured_scene):
    from test_magnetic_test_particle import UniformElectromagneticScene,UniformScene
    for scene in (UniformElectromagneticScene(),UniformScene()):
        assert prepare_compiled_fields(scene) is None
        cfg=TestElectronSettings(kinetic_energy_ev=200000.,max_path_length_m=.002,step_m=.0001)
        prefixes=[]
        result=trace_test_electron(scene,cfg,progress=prefixes.append,progress_interval_s=0.)
        assert prefixes and all(p.reason=='in_progress' and not p.completed for p in prefixes)
        np.testing.assert_array_equal(result.positions_m,trace_test_electron(scene,cfg,use_compiled=False).positions_m)
    for interval in (-1.,np.nan,np.inf):
        with pytest.raises(ValueError,match='Progress interval'):
            trace_test_electron(captured_scene,settings(captured_scene),progress_interval_s=interval)

@pytest.mark.parametrize('category',['stigmator','corrector','hexapole','deflector','gun_deflector','gun_stigmator'])
def test_nonzero_configured_magnetic_components_are_not_omitted(captured_scene,category):
    from copy import deepcopy
    from temsim.magnetic_field_scene import _EquivalentDeflectorField,_MultipoleField
    from temsim.optics.electron_gun.alignment import GunDeflector,GunStigmator
    scene=captured_scene
    sources=list(scene.magnetic_scene._sources)
    target=None
    for index,source in enumerate(sources):
        provider=source.provider
        match=((category=='stigmator' and type(provider) is _MultipoleField and provider.state.stigmators)
               or(category=='corrector' and type(provider) is _MultipoleField and provider.state.corrector_elements and hasattr(provider.state.corrector_elements[0],'quadrupole_strength_m2'))
               or(category=='hexapole' and type(provider) is _MultipoleField and provider.state.corrector_elements and hasattr(provider.state.corrector_elements[0],'hexapole_strength_components_m3'))
               or(category=='deflector' and type(provider) is _EquivalentDeflectorField)
               or(category=='gun_deflector' and type(provider) is GunDeflector)
               or(category=='gun_stigmator' and type(provider) is GunStigmator))
        if not match:continue
        provider=deepcopy(provider)
        if category=='stigmator':
            component=provider.state.stigmators[0]
            component.strength_x_percent=13.;component.strength_y_percent=-7.
        elif category=='corrector':
            component=provider.state.corrector_elements[0]
            component.strength_m2=21.
        elif category=='hexapole':
            provider.state.simulation_mode='custom'
            component=provider.state.corrector_elements[0]
            component.strength_m3=4e7;component.orientation_rad=.23
        elif category=='deflector':
            provider=replace(provider,bx_t=.0013,by_t=-.0021)
        elif category=='gun_deflector':
            provider.upper_field_x_mt=1.1;provider.lower_field_y_mt=-2.3
        else:
            provider.gradient_t_per_m=3.4;provider.rotation_deg=13.
        sources[index]=replace(source,provider=provider,known_zero=False)
        target=sources[index]
        break
    assert target is not None
    changed=replace(scene,magnetic_scene=replace(scene.magnetic_scene,_sources=tuple(sources)))
    data=prepare_compiled_fields(changed)
    assert data is not None
    active=False
    for z in np.linspace(target.bounds_m[0,2],target.bounds_m[1,2],31):
        p=np.array([target.radius_m*.001,-target.radius_m*.0007,z])
        reference=changed.diagnostic_fields_at_global_position(p)
        valid,magnetic,electric,potential=compiled_fields(p,data)
        assert valid==(reference is not None)
        if valid:
            np.testing.assert_allclose(magnetic,reference[0],rtol=2e-7,atol=1e-12)
            active=active or bool(np.linalg.norm(target.provider.field_at_global_positions_t(p[None,:]))>0.)
    assert active


def test_unavailable_or_failed_compilation_retains_complete_reference_solver(captured_scene,monkeypatch,caplog):
    import temsim.test_electron_compiled as fast
    from numba.core.errors import TypingError
    cfg=settings(captured_scene,max_path_length_m=.0001)
    expected=trace_test_electron(captured_scene,cfg,use_compiled=False)
    def unavailable(*args,**kwargs):
        raise TypingError('controlled unsupported compiler fixture')
    monkeypatch.setattr(fast,'compiled_block',unavailable)
    fallback=trace_test_electron(captured_scene,cfg)
    assert 'complete reference solver' in caplog.text
    np.testing.assert_array_equal(fallback.positions_m,expected.positions_m)
    np.testing.assert_array_equal(fallback.kinetic_energy_ev,expected.kinetic_energy_ev)
    monkeypatch.setattr(fast,'njit',None)
    assert fast.prepare_compiled_fields(captured_scene) is None
    missing=trace_test_electron(captured_scene,cfg)
    np.testing.assert_array_equal(missing.positions_m,expected.positions_m)


def test_compiled_magnetic_inputs_are_frozen_and_rebuilt_after_excitation_change(captured_scene):
    from copy import deepcopy
    scene=captured_scene
    before=prepare_compiled_fields(scene)
    sources=list(scene.magnetic_scene._sources)
    original=sources[0]
    owner=original.provider.native_provider.state
    provider=deepcopy(original.provider,{id(owner):owner})
    provider.native_provider.lens.percent*=1.3
    sources[0]=replace(original,provider=provider)
    changed=replace(scene,magnetic_scene=replace(scene.magnetic_scene,_sources=tuple(sources)))
    after=prepare_compiled_fields(changed)
    assert all(not a.flags.writeable for a in before)
    assert not np.array_equal(before[-1],after[-1])
    point=np.array([0.,0.,np.mean(original.bounds_m[:,2])])
    first=compiled_fields(point,before)[1]
    second=compiled_fields(point,after)[1]
    np.testing.assert_allclose(second,changed.diagnostic_fields_at_global_position(point)[0],rtol=1e-12,atol=1e-15)
    assert not np.array_equal(first,second)

@pytest.mark.parametrize('kind',['lens','multipole'])
@pytest.mark.parametrize('replacement',['instance','subclass','class'])
def test_replaced_native_field_laws_use_full_reference(captured_scene,monkeypatch,kind,replacement):
    from copy import deepcopy
    from types import MethodType
    from temsim.magnetic_field_scene import _MultipoleField
    sources=list(captured_scene.magnetic_scene._sources)
    if kind=='lens':
        index=0
        original=sources[index].provider
        owner=original.native_provider.state
        provider=deepcopy(original,{id(owner):owner})
        target=provider.native_provider
        method='magnetic_field_t'
    else:
        index=next(i for i,source in enumerate(sources)
                   if type(source.provider) is _MultipoleField and source.provider.state.corrector_elements
                   and hasattr(source.provider.state.corrector_elements[0],'quadrupole_strength_m2'))
        provider=deepcopy(sources[index].provider)
        target=provider.state.corrector_elements[0]
        method='quadrupole_strength_m2'
    original_method=getattr(type(target),method)
    def changed_law(self,z):
        return 1.2*original_method(self,z)+np.ones_like(np.asarray(z))*1e-4
    if replacement=='instance':
        setattr(target,method,MethodType(changed_law,target))
    elif replacement=='subclass':
        # A tempting module name is not evidence of a supported physical law.
        derived=type('DifferentFieldLaw',(type(target),),{'__module__':type(target).__module__,method:changed_law})
        target.__class__=derived
    else:
        monkeypatch.setattr(type(target),method,changed_law)
    sources[index]=replace(sources[index],provider=provider,known_zero=False)
    changed=replace(captured_scene,magnetic_scene=replace(captured_scene.magnetic_scene,_sources=tuple(sources)))
    assert prepare_compiled_fields(changed) is None
    center=float(np.mean(sources[index].bounds_m[:,2]))
    cfg=TestElectronSettings(kinetic_energy_ev=300000.,position_m=(1e-8,0.,center),
                             max_path_length_m=1e-5,step_m=2e-6)
    expected=trace_test_electron(changed,cfg,use_compiled=False)
    actual=trace_test_electron(changed,cfg)
    assert actual.reason==expected.reason
    np.testing.assert_array_equal(actual.positions_m,expected.positions_m)
    np.testing.assert_array_equal(actual.kinetic_energy_ev,expected.kinetic_energy_ev)


def test_replaced_whole_magnetic_provider_is_not_packed(captured_scene,monkeypatch):
    from copy import deepcopy
    sources=list(captured_scene.magnetic_scene._sources)
    original=sources[0].provider
    owner=original.native_provider.state
    provider=deepcopy(original,{id(owner):owner})
    original_method=type(provider).field_at_global_positions_t
    def changed_law(self,points):return 1.5*original_method(self,points)
    monkeypatch.setattr(type(provider),'field_at_global_positions_t',changed_law)
    sources[0]=replace(sources[0],provider=provider)
    changed=replace(captured_scene,magnetic_scene=replace(captured_scene.magnetic_scene,_sources=tuple(sources)))
    assert prepare_compiled_fields(changed) is None
