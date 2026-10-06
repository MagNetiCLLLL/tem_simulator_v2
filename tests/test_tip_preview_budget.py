"""A refined source must remain usable at the small live-preview budget."""
from dataclasses import replace
from threading import Event

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim import module_manifest
from temsim.optics.electron_gun.tip_assembly import model_from_part
from temsim.gui.calculation_request import CapturedCalculationRequest, apply_request_numerics
from temsim.physics.optical_tuning import prepare_tuning_snapshot, resolve_tuning_ray_count


def source():
    s = default_state()
    model = model_from_part(module_manifest.part_data("gun/FEG.toml", "feg_tip"))
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


def product_source(kind):
    from temsim.optics.electron_gun.emitter import EmissionQuadrature
    from temsim.optics.electron_gun.tip_surface import SharedSurfaceCoherence

    state = source() if kind == "stratified_surface" else default_state()
    emitter = state.electron_gun.emitter
    if kind == "curved":
        emitter.curvature_nm_inv = .01
    elif kind == "shared_surface":
        model = model_from_part(module_manifest.part_data("gun/FEG.toml", "feg_tip"))
        emitter.surface_model = replace(model, emission=replace(model.emission,
            flux_profile="cosine_cap", cap_half_angle_deg=10.,
            energy_distribution="gamma", kinetic_mean_ev=.7, kinetic_sigma_ev=.12,
            maximum_angle_deg=0.), coherence=SharedSurfaceCoherence()).validate()
    emitter.quadrature = EmissionQuadrature(spatial=31, directions=3, energies=7)
    return state


@pytest.mark.parametrize("kind", ["flat", "curved", "stratified_surface", "shared_surface"])
@pytest.mark.parametrize("quality,requested", [("Preview", 49), ("Medium", 193)])
def test_product_source_preview_preserves_complete_weighted_population(kind, quality, requested):
    state = product_source(kind)
    emitter = state.electron_gun.emitter
    plan, surface = emitter.quadrature, emitter.surface_model
    reference = emitter.emit(plan.total)
    # Exercise replacement of stale probe flags as well as ordinary preparation.
    emitter._tuning_boundary_probes = emitter._tuning_surface_probes = 33
    apply_request_numerics(state, quality, requested, 1.)
    prepare_tuning_snapshot(state, quality, particle_signals=True)
    actual = emitter.emit()
    assert emitter.ray_count == len(actual.weight) == plan.total
    assert emitter.quadrature == plan and emitter.surface_model == surface
    assert not np.any(actual.weight == 0)
    assert actual.weight.sum() == pytest.approx(1.)
    assert vars(actual).keys() == vars(reference).keys()
    for key, expected in vars(reference).items():
        np.testing.assert_array_equal(getattr(actual, key), expected)
    if kind == "stratified_surface":
        assert np.unique(actual.weight).size > 1  # Preserve importance weights too.
    assert resolve_tuning_ray_count(state, "High accuracy", requested) == requested


@pytest.mark.parametrize("quality,requested", [("Preview", 49), ("Medium", 193)])
def test_product_preview_capture_records_full_budget_and_keeps_live_source(quality, requested):
    state = product_source("shared_surface")
    before = state.to_dict()
    total = state.electron_gun.emitter.quadrature.total
    request = CapturedCalculationRequest.capture(state, quality, requested, 1.)
    prepared = request.prepare(Event())
    assert request.ray_count == prepared.ray_count == total
    prepare_tuning_snapshot(prepared.snapshot, quality, particle_signals=True)
    assert len(prepared.snapshot.electron_gun.emit().weight) == total
    assert state.to_dict() == before


@pytest.mark.parametrize("boundary", ["forward_gaussian_schell", "driven_gaussian_schell"])
@pytest.mark.parametrize("quality,requested,support", [("Preview", 49, 0), ("Medium", 193, 33)])
def test_gaussian_schell_preview_keeps_source_and_support_weights(boundary, quality, requested, support):
    from temsim.optics.electron_gun.tip_coherence import TipCoherence

    state = default_state()
    emitter = state.electron_gun.emitter
    emitter.coherence = TipCoherence(boundary_model=boundary,
        offset_x_nm=.3, tilt_y_mrad=.1, incoherent_angle_rms_mrad=.2)
    if boundary == "forward_gaussian_schell":
        # An explicitly broad fixture admits the historical paraxial Wigner
        # law; the production narrow tip requires the driven boundary instead.
        emitter.virtual_source_fwhm_nm = 1000.
    coherence = emitter.coherence
    # This legacy sampler moment-matches energies for the full requested count
    # before zero-current support probes replace the tail. Preserve that law.
    reference = emitter.emit(requested)
    apply_request_numerics(state, quality, requested, 1.)
    prepare_tuning_snapshot(state, quality, particle_signals=True)
    actual = emitter.emit()
    assert emitter.coherence == coherence and emitter.quadrature is None
    assert np.count_nonzero(actual.weight == 0) == support
    assert actual.weight.sum() == pytest.approx(1.)
    for key, expected in vars(reference).items():
        if key == "weight":
            expected = np.full(requested, 1/(requested-support))
        np.testing.assert_array_equal(getattr(actual, key)[:requested-support], expected[:requested-support])
    # Keeping a product budget must not bypass unsupported source admission.
    from temsim.optics.electron_gun.emitter import EmissionQuadrature
    emitter.quadrature = EmissionQuadrature(3, 3, 3)
    apply_request_numerics(state, quality, requested, 1.)
    prepare_tuning_snapshot(state, quality, particle_signals=True)
    with pytest.raises(ValueError, match="Product quadrature is classical only"):
        emitter.emit()


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
