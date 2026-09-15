"""A refined source must remain usable at the small live-preview budget."""
from dataclasses import replace
from threading import Event

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.gui.calculation_request import CapturedCalculationRequest, apply_request_numerics
from temsim.physics.optical_tuning import prepare_tuning_snapshot, resolve_tuning_ray_count


def source():
    s = default_state()
    model = s.electron_gun.emitter.surface_model
    s.electron_gun.emitter.surface_model = replace(model,emission=replace(model.emission,
        spatial_sampling='apex_stratified_v1',directions_per_position=72,
        angular_sampling='tangent_stratified_v2',angular_refinement_gain=80.,
        angular_stratum_allocation=(1,1,1,1,32,1,1,1,1)))
    return s


@pytest.mark.parametrize('quality,requested,expected,support',[
    ('Preview',49,82,1),('Medium',193,193,33),('High accuracy',10369,10369,0)])
def test_requested_count_manifest_and_emitted_population_agree(quality,requested,expected,support):
    s = source()
    emission = s.electron_gun.emitter.surface_model.emission
    assert resolve_tuning_ray_count(s,quality,requested) == expected
    apply_request_numerics(s,quality,requested,1.)
    if quality != 'High accuracy':
        prepare_tuning_snapshot(s,quality)
    b = s.electron_gun.emit()
    assert len(b.weight) == s.electron_gun.emitter.ray_count == expected
    assert np.count_nonzero(b.weight == 0) == support
    assert b.weight.sum() == pytest.approx(1.)
    assert s.electron_gun.emitter.surface_model.emission == emission


def test_background_capture_records_the_actual_preview_budget_without_mutating_live_state():
    s = source()
    before = s.electron_gun.emitter.ray_count
    request = CapturedCalculationRequest.capture(s,'Preview',49,1.)
    prepared = request.prepare(Event())
    assert request.ray_count == prepared.ray_count == 82
    assert prepared.snapshot.electron_gun.emitter.ray_count == 82
    assert s.electron_gun.emitter.ray_count == before


def test_ordinary_source_and_explicit_high_accuracy_budgets_are_not_changed():
    assert resolve_tuning_ray_count(default_state(),'Preview',49) == 49
    assert resolve_tuning_ray_count(source(),'High accuracy',49) == 49


def test_unresolved_preview_does_not_report_an_artificial_nanoprobe(monkeypatch):
    from types import SimpleNamespace
    from temsim.physics.optical_tuning import tuning_metrics
    from temsim.physics import beam_statistics
    branch = SimpleNamespace(z=np.array([0.,1.]),x=np.zeros((2,3)),y=np.zeros((2,3)),
        tx=np.zeros((2,3)),ty=np.zeros((2,3)),blocked_z=np.full(3,np.nan),
        alive=np.ones(3,bool),ray_weight=np.array([.00007,.000006,.0000004]))
    measured = SimpleNamespace(surviving_fraction=float(branch.ray_weight.sum()))
    monkeypatch.setattr(beam_statistics,'branch_sample_statistics',lambda _b: measured)
    metrics = tuning_metrics(default_state(),branch)
    assert metrics['sample_statistics_status'] == 'INSUFFICIENT_EFFECTIVE_RAYS'
    assert np.isnan(metrics['sample_convergence_95_mrad'])
    assert np.isnan(metrics['sample_illumination_diameter_95_um'])
    assert metrics['sample_beam_surviving_fraction'] == pytest.approx(branch.ray_weight.sum())
    assert metrics['sample_beam_surviving_rays'] == 3
