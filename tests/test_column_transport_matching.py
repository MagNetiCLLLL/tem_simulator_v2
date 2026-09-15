"""Recovery transaction/readout tests; physical acceptance is a separate CLI run."""
from types import SimpleNamespace
from dataclasses import asdict

import numpy as np
import pytest

from temsim.alignment_transaction import AlignmentRequest, _allowed_state, solve_alignment_candidate
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.optics.transport_matching import DEFINITION, LENSES, target_plane, transport_measurement


def fake_result(stops, weights):
    z=np.array([0.,1.,2.,3.])
    shape=(4,len(stops))
    branch=SimpleNamespace(z=z,x=np.full(shape,1e-5),y=np.zeros(shape),
        tx=np.zeros(shape),ty=np.zeros(shape),ray_weight=np.asarray(weights),
        blocked_z=np.asarray(stops),alive=np.zeros(len(stops),bool),weight=1.)
    return SimpleNamespace(incident=branch,branches={'000':branch})


def test_detector_absorption_is_success_before_detector_and_earlier_stops_never_pass():
    # Every final alive flag is false; only the later detector hit is success.
    result=transport_measurement(fake_result([2.5,1.,np.nan,2.],[.2,.3,0.,.5]),2.)
    assert result['rays']==1
    assert result['source_fraction']==pytest.approx(.2)
    assert result['radius_mm']==pytest.approx(.01)
    assert result['finite']


def test_nonfinite_survivor_rejected_but_post_stop_placeholders_do_not_count():
    result=fake_result([3.,1.],[.5,.5])
    result.incident.x[1,0]=np.nan
    assert not transport_measurement(result,2.)['finite']
    result.incident.x[1,0]=0.
    result.incident.x[2:,1]=np.nan
    assert transport_measurement(result,2.)['finite']


def test_target_is_actual_permanent_projection_chamber_plane():
    state=default_state()
    actual=next(a for a in state.apertures if a.key=='projection_chamber_dpa_aperture')
    assert target_plane(state)==actual.z_mm
    actual.enabled=False
    assert actual.enabled  # The permanent aperture cannot be retracted.
    with pytest.raises(ValueError,match='installed'):
        target_plane(SimpleNamespace(apertures=[],sample=state.sample))


def test_recovery_changes_exactly_three_controls_and_preserves_source_apertures_and_mode():
    state=default_state()
    before=capture_instrument_snapshot(state)
    request=AlignmentRequest.capture(state,'column_transport',.01,revision=4)
    strengths=dict(zip(LENSES,(7.9,25.7,40.3)))
    updated=_allowed_state(request,strengths)
    assert capture_instrument_snapshot(state).digest==before.digest
    for lens in updated.lenses:
        if lens.key in LENSES:
            lens.percent=next(old.percent for old in state.lenses if old.key==lens.key)
    assert capture_instrument_snapshot(updated).digest==before.digest
    with pytest.raises(ValueError,match='exact registered'):
        _allowed_state(request,{**strengths,'objective_lens':50.})


def test_cancelled_recovery_executes_no_gun_and_retains_working_point(monkeypatch):
    state=default_state()
    before=capture_instrument_snapshot(state).digest
    request=AlignmentRequest.capture(state,'column_transport',.01,revision=0)
    from temsim.alignment_transaction import AlignmentCancelled
    monkeypatch.setattr(type(state.electron_gun),'trace_to_exit',lambda *_:pytest.fail('Gun must not run'))
    with pytest.raises(AlignmentCancelled):
        solve_alignment_candidate(request,cancelled=lambda:True)
    assert capture_instrument_snapshot(state).digest==before


def test_zero_current_does_not_report_a_successful_beam():
    state=default_state()
    state.column_current_limit_percent=0.
    request=AlignmentRequest.capture(state,'column_transport',.01,revision=0)
    with pytest.raises(ValueError,match='No source current'):
        solve_alignment_candidate(request)


def test_transport_is_not_an_imaging_focus_definition():
    assert set(DEFINITION.devices)==set(LENSES)
    assert 'focus' in DEFINITION.calibration_status
    assert DEFINITION.key not in {'nanoprobe_convergence','microprobe_illumination'}
    assert 'achieved_convergence' not in asdict(DEFINITION)['targets']


def test_match_button_starts_transaction_without_a_new_image_option(qtbot,monkeypatch,tmp_path):
    from PySide6.QtCore import QSettings
    from temsim.gui import main_window, interactive_calculation, calculation_controller
    settings=QSettings(str(tmp_path/'workspace.ini'),QSettings.Format.IniFormat)
    monkeypatch.setattr(main_window,'QSettings',lambda:settings)
    monkeypatch.setattr(interactive_calculation,'QSettings',lambda:settings)
    monkeypatch.setattr(calculation_controller,'default_artifact_cache_root',lambda:tmp_path/'artifacts')
    monkeypatch.setattr(main_window.MainWindow,'INITIAL_PREVIEW_DELAY_MS',60000)
    window=main_window.MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    window.show()
    window.workspace.show_ray_diagram()
    button=window.workspace.match_transport
    assert button.parentWidget() is window.workspace.view_controls_panel
    assert window.workspace.view_controls_panel.layout().indexOf(button)>=0
    assert button.isVisibleTo(window)
    calls=[]
    monkeypatch.setattr(window,'apply_direct_alignment',lambda *args:calls.append(args))
    button.click()
    assert calls==[('column_transport',.01)]
    window.close()
