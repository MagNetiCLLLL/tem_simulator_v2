"""Physical response checks and separately labelled transaction fixtures."""
from dataclasses import replace
import numpy as np
import pytest

from temsim.beam_alignment import KEY, BeamAlignmentOptions, capability, allowed_state
from temsim.alignment_transaction import AlignmentRequest, AlignmentCommitGate, solve_alignment_candidate
from temsim.assembly_catalog import AssemblyCatalog
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state


def state():
    result = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(result, catalog.default_selection())
    return result


def test_current_condenser_stigmator_has_one_independent_field_direction():
    from temsim.physics.core import multipole_focusing_fields
    source = state()
    stig = source.condenser_stigmator
    z = np.array([stig.z_mm])
    stig.strength_x_percent, stig.strength_y_percent = 1., 0.
    x = multipole_focusing_fields(z, source)
    stig.strength_x_percent, stig.strength_y_percent = 0., -1.
    y = multipole_focusing_fields(z, source)
    for a, b in zip(x, y, strict=True):
        np.testing.assert_array_equal(a, b)
    assert abs(x[0][0]) > 0


def test_actual_tip_origin_beam_responds_to_physical_deflector():
    from temsim.alignment_constraints import evaluate_incident
    source = state()
    source.electron_gun.emitter.ray_count = 9
    source.step_mm = 1.
    before = evaluate_incident(source)[0]
    source.beam_deflector.lower_x_mrad += .001
    after = evaluate_incident(source)[0]
    assert "centroid_x_um" in before and "centroid_x_um" in after
    assert not np.allclose([before[k] for k in ("centroid_x_um", "mean_tx_mrad")],
                           [after[k] for k in ("centroid_x_um", "mean_tx_mrad")], rtol=0, atol=1.e-10)


def test_beam_transaction_changes_only_registered_kicks(monkeypatch):
    source = state()
    initial = capture_instrument_snapshot(source)
    options = BeamAlignmentOptions(targets=(.1, -.1, .01, -.01), maximum_evaluations=16)
    request = AlignmentRequest.capture(source, KEY, 0., revision=0, options=options)
    def response(candidate, **kwargs):
        d = candidate.beam_deflector
        metrics = dict(centroid_x_um=d.upper_x_mrad+d.lower_x_mrad, centroid_y_um=d.upper_y_mrad+d.lower_y_mrad,
                       mean_tx_mrad=d.upper_x_mrad-d.lower_x_mrad, mean_ty_mrad=d.upper_y_mrad-d.lower_y_mrad,
                       effective_samples=32., current_pa=10.)
        arrays = dict(x_m=np.zeros(32), y_m=np.zeros(32), tx_rad=np.zeros(32), ty_rad=np.zeros(32),
                      alive=np.ones(32, bool), weight=np.full(32, 1./32))
        return metrics, arrays, dict(plane_z_mm=candidate.sample.upper_surface_z_mm, source_current_a=1.e-11), None
    monkeypatch.setattr("temsim.alignment_constraints.evaluate_incident", response)
    result = solve_alignment_candidate(request)
    assert result.result.success
    assert result.validation["control_authority"]["rank"] == 4
    assert capture_instrument_snapshot(source).digest == initial.digest
    gate = AlignmentCommitGate()
    changed = gate.apply(source, result, revision=0)
    assert changed.beam_deflector.lower_x_mrad != source.beam_deflector.lower_x_mrad
    assert [(l.key, l.percent) for l in changed.lenses] == [(l.key, l.percent) for l in source.lenses]
    assert capture_instrument_snapshot(gate.undo()).digest == initial.digest
    with pytest.raises(ValueError, match="registered"):
        allowed_state(request, {**result.result.strengths, "beam_deflector.z_mm": 500.})


def test_beam_ui_availability_and_targets_are_read_only(qtbot):
    from temsim.gui.direct_alignment_panel import DirectAlignmentPanel
    source = state()
    panel = DirectAlignmentPanel()
    qtbot.addWidget(panel)
    panel.set_state(source)
    before = capture_instrument_snapshot(source).digest
    editor = panel.beam_alignment_editor
    assert editor.apply_button.isEnabled()
    editor.targets[0].setValue(.2)
    assert panel.constraint_options(KEY).targets[0] == .2
    assert capture_instrument_snapshot(source).digest == before
    source.beam_deflector.enabled = False
    panel.set_state(source)
    assert not editor.apply_button.isEnabled()
    with pytest.raises(ValueError, match="disabled"):
        AlignmentRequest.capture(source, KEY, 0., revision=0, options=BeamAlignmentOptions())
