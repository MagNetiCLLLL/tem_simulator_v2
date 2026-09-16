from pathlib import Path

import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.optics.assembly_illumination import seed_illumination
from temsim.optics.column import default_state


@pytest.fixture
def operator_identity(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from check_illumination_equivalence import operator_identity
    return operator_identity


def assembled(column='C3 + Probe Corrector', recording='Energy Filter'):
    state=default_state()
    AssemblyCatalog().apply(state,AssemblySelection('FEG',column,recording))
    seed_illumination(state,'micro_probe')
    state.electron_gun.emitter.ray_count=32
    return state


def test_custom_analytic_plan_equivalence_for_downstream_variants(operator_identity):
    state=assembled()
    assert state.simulation_mode=='custom'
    identity=operator_identity(state,step_mm=.1)
    other=assembled('C3 + Probe Corrector + Image Corrector','No Energy Filter')
    assert identity['signature']==operator_identity(other,step_mm=.1)['signature']


@pytest.mark.parametrize('change',['lens','pupil','flux','source','upper_surface'])
def test_consumed_upstream_inputs_change_equivalence_identity(operator_identity,change):
    state=assembled()
    before=operator_identity(state,step_mm=.1)['signature']
    if change=='lens':
        next(l for l in state.lenses if l.key=='condenser_lens_1').percent+=.01
    elif change=='pupil': state.condenser_aperture_2.diameter_mm*=.9
    elif change=='flux': state.column_current_limit_percent=.001
    elif change=='source': state.electron_gun.emitter.ray_count+=1
    else: state.sample.thickness_nm+=2
    assert before!=operator_identity(state,step_mm=.1)['signature']


def test_active_vacuum_is_not_aliased(operator_identity):
    state=assembled()
    state.vacuum_map.enabled=True
    with pytest.raises(ValueError,match='inactive vacuum'):
        operator_identity(state,step_mm=.1)


def test_historical_microprobe_baseline_requires_explicit_permission(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from validate_assembly_illumination import reference_microprobe_needs_baseline_approval
    selection=dict(gun='FEG',column='C3 + Probe Corrector',beam_blanker='None')
    assert reference_microprobe_needs_baseline_approval(selection,'micro_probe')
    assert not reference_microprobe_needs_baseline_approval(selection,'micro_probe',approved=True)
    assert not reference_microprobe_needs_baseline_approval(selection,'nano_probe')
    assert not reference_microprobe_needs_baseline_approval(dict(selection,column='C3'),'micro_probe')


def test_reference_microprobe_approval_records_six_ordered_intervals(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from validate_assembly_illumination import reference_microprobe_baseline
    baseline=reference_microprobe_baseline()
    assert baseline['authorised'] is True
    assert baseline['count']==len(baseline['intervals'])==6
    assert baseline['intervals'][0]==['between',['condenser_lens_1'],['condenser_lens_2']]
    assert baseline['intervals'][-2:]==[['between',['mini_condenser'],['objective_lens','sample']]]*2
