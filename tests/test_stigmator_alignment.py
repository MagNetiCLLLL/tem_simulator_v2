"""Synthetic transaction tests are separate from actual tip-ray response checks."""
from dataclasses import replace
import numpy as np
import pytest
from temsim.optics.column import default_state
from temsim.assembly_catalog import AssemblyCatalog
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.alignment_transaction import AlignmentRequest, AlignmentCommitGate, AlignmentCancelled, solve_alignment_candidate
from temsim.stigmator_alignment import KEY, StigmatorAlignmentOptions, capability, component


def state():
    result = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(result, catalog.default_selection())
    component(result).field_model = "normal_skew"
    return result


def fixture_response(candidate, *, deficient=False, **kwargs):
    stig = component(candidate)
    a, b = .15+.003*stig.strength_x_percent, -.1+(0. if deficient else .003*stig.strength_y_percent)
    covariance = np.array([[1+a, b], [b, 1-a]])
    angles = np.arange(32)*2*np.pi/32
    xy = np.linalg.cholesky(covariance) @ np.array([np.cos(angles), np.sin(angles)])*1e-9
    arrays = dict(x_m=xy[0], y_m=xy[1], tx_rad=np.zeros(32), ty_rad=np.zeros(32),
                  alive=np.ones(32, bool), weight=np.full(32, 1/32))
    return dict(effective_samples=32., current_pa=10., diameter95_um=.004), arrays, dict(
        plane_z_mm=candidate.sample.upper_surface_z_mm, source_current_a=1e-11), None


def test_geometric_shape_transaction_is_bounded_and_atomic(monkeypatch):
    source = state()
    original = capture_instrument_snapshot(source)
    request = AlignmentRequest.capture(source, KEY, 0., revision=0, options=StigmatorAlignmentOptions(maximum_evaluations=32))
    monkeypatch.setattr("temsim.alignment_constraints.evaluate_incident", fixture_response)
    candidate = solve_alignment_candidate(request)
    assert candidate.result.success
    assert candidate.validation["control_authority"]["rank"] == 2
    assert candidate.validation["observables"] == ("shape_normal", "shape_skew")
    assert capture_instrument_snapshot(source).digest == original.digest
    gate = AlignmentCommitGate()
    changed = gate.apply(source, candidate, revision=0)
    assert component(changed).strength_x_percent == pytest.approx(-50., abs=.01)
    assert component(changed).strength_y_percent == pytest.approx(100/3, abs=.01)
    assert [(l.key, l.percent) for l in changed.lenses] == [(l.key, l.percent) for l in source.lenses]
    assert capture_instrument_snapshot(gate.undo()).digest == original.digest
    with pytest.raises(ValueError, match="[Ss][Tt][Aa][Ll][Ee]|revision"):
        AlignmentCommitGate().apply(source, candidate, revision=1)


@pytest.mark.parametrize("problem", ["rank", "size", "validation"])
def test_failed_checks_preserve_instrument(monkeypatch, problem):
    source = state()
    original = capture_instrument_snapshot(source).digest
    original_step = source.step_mm
    def response(candidate, **kwargs):
        values = fixture_response(candidate, deficient=problem == "rank")
        if problem == "validation" and candidate.step_mm < original_step:
            raise ValueError("Independent step fixture failed")
        return values
    monkeypatch.setattr("temsim.alignment_constraints.evaluate_incident", response)
    options = StigmatorAlignmentOptions(maximum_diameter_nm=1. if problem == "size" else 10.)
    request = AlignmentRequest.capture(source, KEY, 0., revision=0, options=options)
    candidate = solve_alignment_candidate(request)
    assert not candidate.result.success
    assert candidate.status == "FAILED"
    assert capture_instrument_snapshot(source).digest == original


def test_legacy_and_active_vacuum_keep_explicit_boundaries():
    source = state()
    component(source).field_model = "legacy_difference"
    assert not capability(source)[0]
    with pytest.raises(ValueError, match="Independent X/Y"):
        AlignmentRequest.capture(source, KEY, 0., revision=0, options=StigmatorAlignmentOptions())
    component(source).field_model = "normal_skew"
    request = AlignmentRequest.capture(source, KEY, 0., revision=0, options=StigmatorAlignmentOptions())
    with pytest.raises(AlignmentCancelled):
        solve_alignment_candidate(request, cancelled=lambda: True)
    source.vacuum_map.enabled = True
    with pytest.raises(ValueError, match="vacuum"):
        AlignmentRequest.capture(source, KEY, 0., revision=0, options=StigmatorAlignmentOptions())
    assert source.vacuum_map.enabled is True


def test_actual_tip_origin_population_responds_to_both_channels():
    from temsim.alignment_constraints import evaluate_incident
    source = state()
    source.electron_gun.emitter.ray_count = 49
    source.step_mm = 1.
    before = evaluate_incident(source)[1]
    stig = component(source)
    results = []
    for x, y in ((.1, 0.), (0., .1)):
        stig.strength_x_percent, stig.strength_y_percent = x, y
        after = evaluate_incident(source)[1]
        live = before["alive"] & after["alive"]
        assert np.count_nonzero(live) >= 3
        results.append(np.concatenate([(after[k]-before[k])[live] for k in ("x_m", "y_m", "tx_rad", "ty_rad")]))
    assert np.linalg.matrix_rank(np.column_stack(results)) == 2


def test_ui_does_not_enable_legacy_rank_one_task(qtbot):
    from temsim.gui.direct_alignment_panel import DirectAlignmentPanel
    source = state()
    panel = DirectAlignmentPanel()
    qtbot.addWidget(panel)
    panel.set_state(source)
    assert panel.stigmator_alignment_editor.apply_button.isEnabled()
    assert isinstance(panel.constraint_options(KEY), StigmatorAlignmentOptions)
    component(source).field_model = "legacy_difference"
    panel.set_state(source)
    assert not panel.stigmator_alignment_editor.apply_button.isEnabled()
