from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.optics.illumination_current import (
    aperture_gate, current_gate, set_calibration_flux, load_current_aperture_limits,
)


def measurement(fraction):
    return SimpleNamespace(statistics=SimpleNamespace(surviving_fraction=fraction))


def test_authoritative_current_and_pupil_targets():
    limits = load_current_aperture_limits()
    assert (limits.minimum_current_pa,limits.maximum_current_pa)==(2.,200.)
    assert (limits.minimum_c2_diameter_um,limits.maximum_c2_diameter_um)==(20.,250.)


def test_flux_setting_preserves_physical_source_losses_and_snapshot():
    state = default_state()
    state.condenser_aperture_2.diameter_mm = .12
    before = state.to_dict()
    old_digest = capture_instrument_snapshot(state).physical_digest
    m = measurement(.01)
    audit = set_calibration_flux(state,m)
    after = state.to_dict()
    assert audit['specimen_current_pa'] == pytest.approx(100.)
    assert audit['source_to_specimen_fraction'] == .01
    assert audit['effective_source_current_pa'] == pytest.approx(10000.)
    assert audit['available_at_full_flux_pa'] == pytest.approx(state.electron_gun.emitted_current_a*1e12*.01)
    assert audit['policy']=='EXPLICIT_UPSTREAM_TOTAL_FLUX'
    assert audit['passed']
    # No emitter, lens, pupil, geometry or distribution modification.
    after['column_current_limit_percent']=before['column_current_limit_percent']
    assert before==after
    snapshot = capture_instrument_snapshot(state)
    assert snapshot.physical_digest != old_digest
    assert snapshot.restore().column_current_limit_percent==state.column_current_limit_percent
    assert m.statistics.surviving_fraction==.01


def test_validation_uses_fixed_flux_instead_of_renormalising_transmission():
    state=default_state()
    set_calibration_flux(state,measurement(.5))
    limit=state.column_current_limit_percent
    assert current_gate(state,measurement(.25))['specimen_current_pa']==pytest.approx(50.)
    assert not current_gate(state,measurement(.0001))['passed']
    assert state.column_current_limit_percent==limit


def test_available_current_below_preferred_setpoint_is_not_amplified():
    state=default_state()
    fraction=20./(state.electron_gun.emitted_current_a*1e12)
    audit=set_calibration_flux(state,measurement(fraction))
    assert state.column_current_limit_percent==100.
    assert audit['specimen_current_pa']==pytest.approx(20.)
    assert audit['passed']


@pytest.mark.parametrize('fraction',[0.,1e-20,-.1,1.1,np.nan,np.inf])
def test_impossible_current_rejected_without_mutation(fraction):
    state=default_state()
    before=state.to_dict()
    with pytest.raises(ValueError):
        set_calibration_flux(state,measurement(fraction))
    assert state.to_dict()==before


@pytest.mark.parametrize('diameter,passed',[(.0199,False),(.02,True),(.25,True),(.25001,False),(np.nan,False)])
def test_c2_diameter_limits(diameter,passed):
    state=default_state()
    state.condenser_aperture_2.diameter_mm=diameter
    assert aperture_gate(state)['passed'] is passed
    if not passed:
        with pytest.raises(ValueError,match='C2 aperture'):
            set_calibration_flux(state,measurement(.5))


def test_retracted_c2_is_not_a_qualified_pupil():
    state=default_state()
    state.condenser_aperture_2.enabled=False
    assert not aperture_gate(state)['passed']


def test_limits_reject_invalid_order(tmp_path):
    limits=asdict(load_current_aperture_limits())
    limits['minimum_current_pa']=201.
    path=tmp_path/'targets.toml'
    path.write_text('schema="assembly-illumination-targets-v1"\n[current_and_aperture]\n'+
                    '\n'.join(f'{key}={value}' for key,value in limits.items()),encoding='utf-8')
    with pytest.raises(ValueError,match='ordered'):
        load_current_aperture_limits(path)


def test_fixed_setting_validation_requires_numerical_stability(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from validate_assembly_illumination import convergence_gate
    row=dict(measurement=dict(statistics=dict(convergence_95_rad=.03,radius_95_m=.4e-9,surviving_fraction=.5)))
    assert convergence_gate([row,deepcopy(row)],'nano_probe')['passed']
    other=deepcopy(row)
    other['measurement']['statistics']['radius_95_m']=.42e-9
    assert not convergence_gate([row,other],'nano_probe')['passed']
    assert not convergence_gate([row],'nano_probe')['passed']
    other['measurement']['statistics']['radius_95_m']=np.nan
    assert not convergence_gate([row,other],'nano_probe')['passed']
