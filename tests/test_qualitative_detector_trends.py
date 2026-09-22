"""Detector limits and explicit optical-reference diagnostics, not image acceptance."""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.detector.plane_image import detector_response_image
from temsim.detector.stem_detector import create_stem_detectors
from temsim.physics.interaction_budget import plane_interaction_budget


def diagnostic_fixture(weights, *, offset=(0., 0.), positions=None):
    """Isolated detector input fixture; never an active application source."""
    weights = np.asarray(weights, dtype=float)
    detector = create_stem_detectors()[2]
    detector.geometry = 'disk'
    detector.inner_diameter_mm = 0.
    detector.outer_width_mm = 2.
    detector.centre_offset_x_mm, detector.centre_offset_y_mm = offset
    detector.z_mm = 2.
    detector.point_spread_model = 'none'
    detector.point_spread_sigma_x_mm = detector.point_spread_sigma_y_mm = 0.
    if positions is None:
        positions = np.tile(offset, (len(weights), 1))
    positions = np.asarray(positions) * 1e-3
    branch = SimpleNamespace(name='reference', colour=(1, 1, 1),
        interaction_kind='vacuum', z=np.array([1., 3.]),
        x=np.tile(positions[:, 0], (2, 1)), y=np.tile(positions[:, 1], (2, 1)),
        tx=np.zeros((2, len(weights))), ty=np.zeros((2, len(weights))),
        alive=np.ones(len(weights), dtype=bool), blocked_z=np.full(len(weights), np.nan),
        blocked_key=[''] * len(weights), ray_weight=weights, weight=1.)
    incident = SimpleNamespace(**{**vars(branch), 'z': np.array([0., 1.])})
    simulation = SimpleNamespace(incident=incident, branches={'reference': branch},
        metrics={'branch_weights_are_absolute': True}, real_interactions=None)
    state = SimpleNamespace(sample=SimpleNamespace(z_mm=1.), recording_planes=[detector])
    return simulation, state, detector


def test_zero_weight_particles_never_create_detector_signal():
    simulation, state, detector = diagnostic_fixture([0., 0.])
    response = detector_response_image(simulation, state, detector.key)
    assert response.accepted_weight == 0.
    assert response.response_weight == 0.
    assert not np.any(response.intensity)


@pytest.mark.parametrize('weight', [-1., np.nan, np.inf])
def test_invalid_detector_weights_are_not_silently_replaced(weight):
    simulation, state, detector = diagnostic_fixture([weight])
    with pytest.raises(ValueError, match='weights.*finite.*non-negative'):
        detector_response_image(simulation, state, detector.key)


def test_translating_detector_and_hits_preserves_collection():
    results = []
    for offset in [(0., 0.), (3., -4.)]:
        simulation, state, detector = diagnostic_fixture([.25, .75], offset=offset)
        results.append(detector_response_image(simulation, state, detector.key, pixels=128))
    for response in results:
        assert response.accepted_weight == pytest.approx(1., abs=1e-14)
        assert response.response_weight == pytest.approx(1., abs=1e-14)
    np.testing.assert_allclose(np.asarray(results[1].extent) - results[0].extent,
                               [3., 3., -4., -4.], rtol=0, atol=1e-14)


def test_budget_cannot_restore_a_ray_lost_before_the_sample():
    simulation, state, _ = diagnostic_fixture([.25, .75])
    simulation.incident.blocked_z = np.array([.5, np.nan])
    budget = plane_interaction_budget(SimpleNamespace(simulation=simulation,
        state_snapshot=state), 2.)
    assert budget.source_fraction_at_plane == pytest.approx(.75)
    assert budget.pre_sample_stopped_source_fraction == pytest.approx(.25)
    assert budget.conservation_error < 1e-14


def test_detector_spread_rotates_covariance_and_keeps_absolute_weight():
    from temsim.detector.point_spread import DetectorPointSpread, apply_point_spread
    ideal = np.zeros((257, 257))
    ideal[128, 128] = 7.
    pitch = .005  # mm, fixed for every comparison
    yy, xx = (np.indices(ideal.shape) - 128) * pitch
    covariances = []
    peaks = []
    for scale, angle in [(1., 0.), (2., 0.), (1., 45.)]:
        psf = DetectorPointSpread('gaussian', .04*scale, .02*scale, angle,
            'provisional_model_parameter', 'isolated reference')
        response = apply_point_spread(ideal, psf, pixel_size_x_mm=pitch, pixel_size_y_mm=pitch)
        total = response.sum()
        assert total == pytest.approx(7., abs=2e-13)
        covariance = np.array([[np.sum(response*xx*xx), np.sum(response*xx*yy)],
                               [np.sum(response*xx*yy), np.sum(response*yy*yy)]]) / total
        theta = np.deg2rad(angle)
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        reference = rotation @ np.diag([(.04*scale)**2, (.02*scale)**2]) @ rotation.T
        np.testing.assert_allclose(covariance, reference, rtol=.002, atol=1e-15)
        covariances.append(covariance)
        peaks.append(response.max())
    assert peaks[1] < peaks[0]
    np.testing.assert_allclose(covariances[1], 4*covariances[0], rtol=.002, atol=1e-15)


def test_pixel_response_dose_gain_saturation_and_shot_noise_are_distinct():
    from temsim.physics.fourdstem import PixelatedDetectorResponse
    dose = np.full((320, 320), 40.)
    response = PixelatedDetectorResponse(quantum_efficiency=.25,
        gain_counts_per_electron=2., offset_counts=3.)
    np.testing.assert_array_equal(response.apply(dose), np.full_like(dose, 23.))
    np.testing.assert_array_equal(response.apply(2*dose)-3, 2*(response.apply(dose)-3))
    saturated = PixelatedDetectorResponse(quantum_efficiency=.25,
        gain_counts_per_electron=2., offset_counts=3., saturation_electrons=12.)
    np.testing.assert_array_equal(saturated.apply(2*dose), np.full_like(dose, 27.))
    noisy = PixelatedDetectorResponse(quantum_efficiency=.25, poisson_enabled=True, seed=791)
    counts = noisy.apply(dose)
    np.testing.assert_array_equal(counts, noisy.apply(dose))
    # Independent Poisson reference: mean=lambda and variance=lambda. Six
    # standard errors are fixed before sampling, not fitted to the realization.
    n, lam = counts.size, 10.
    assert abs(np.mean(counts, dtype=float)-lam) < 6*np.sqrt(lam/n)
    assert abs(np.var(counts, dtype=float)-lam) < 6*np.sqrt((lam+2*lam**2)/n)
    np.testing.assert_array_equal(dose, np.full_like(dose, 40.))


def test_fixed_total_frame_time_redistributes_dose_without_creating_electrons():
    from temsim.detector.stem_signal import StemScanResult, reweight_stem_scan
    from temsim.optics.column import default_state
    state = default_state()
    state.sample.stem_poisson_enabled = False
    state.ac_deflector.scan_frame_period_s = .12
    state.column_current_limit_percent = 1e-6
    fraction = .3  # held recorded collection; isolated readout check
    totals = []
    for shape in [(3, 4), (6, 8)]:
        frame = StemScanResult(np.zeros(shape), np.zeros(shape),
            {'bf': np.full(shape, fraction)}, {})
        result = reweight_stem_scan(state, frame)
        totals.append(result.expected_electrons['bf'].sum())
        expected_total = state.electron_gun.emitted_current_a * 1e-8 * .12 * fraction / 1.602176634e-19
        assert totals[-1] == pytest.approx(expected_total, rel=1e-13)
        assert result.dwell_time_s == pytest.approx(.12/np.prod(shape))
        state.ac_deflector.scan_frame_period_s *= 2
        longer = reweight_stem_scan(state, frame)
        np.testing.assert_allclose(longer.expected_electrons['bf'], 2*result.expected_electrons['bf'], rtol=1e-14)
        np.testing.assert_array_equal(longer.current_pa['bf'], result.current_pa['bf'])
        state.ac_deflector.scan_frame_period_s /= 2
    assert totals[0] == pytest.approx(totals[1], rel=1e-14)


def test_tip_origin_detector_losses_survive_readout_toggle_and_step_refinement(record_property):
    """Executed vacuum specimen path; no replacement source or image claim."""
    import json
    from temsim.optics.column import default_state
    from temsim.physics.simulation import run
    from temsim.simulation_modes import switch_mode
    state = default_state()
    state.electron_gun.emitter.ray_count = 49
    state.sample.inserted = False
    state.acceleration_enabled = False
    state.acceleration_backend = 'CPU'
    switch_mode(state, 'analytical')
    assert state.electron_gun.emitter.coherence is None
    assert not state.vacuum_map.enabled
    records = []
    traces = []
    for step in [.125, .0625]:
        state.step_mm = step
        simulation = run(state, optical_only=True)
        assert len(simulation.branches) == 1
        branch = next(iter(simulation.branches.values()))
        assert np.count_nonzero(simulation.incident.alive) >= 3
        np.testing.assert_array_equal(branch.ray_weight, simulation.incident.ray_weight)
        weights = np.asarray(branch.ray_weight)
        stopped = np.isfinite(branch.blocked_z)
        assert weights.sum() == pytest.approx(1., abs=1e-14)
        assert weights[stopped].sum() + weights[~stopped].sum() == pytest.approx(1., abs=1e-14)
        fractions = {str(key): float(weights[np.asarray(branch.blocked_key) == key].sum())
                     for key in set(branch.blocked_key) if key}
        assert fractions.get('haadf', 0.) > 0.
        assert fractions.get('flu_screen', 0.) > 0.
        result = SimpleNamespace(simulation=simulation, state_snapshot=state)
        for detector in state.recording_planes:
            selected = detector.z_mm + .001  # after interception, in mm
            budget = plane_interaction_budget(result, selected)
            expected = weights[np.isnan(branch.blocked_z) | (branch.blocked_z >= selected)].sum()
            assert budget.source_fraction_at_plane == pytest.approx(expected, abs=1e-14)
            assert budget.conservation_error < 2e-14
        records.append(fractions)
        traces.append(simulation)
    assert records[0] == pytest.approx(records[1], abs=1e-14)
    # Electronic readout is changed on the same installed geometry. The
    # executed physical losses and downstream trajectories must remain fixed.
    state.haadf_detector.readout_enabled = False
    hidden = run(state, existing_simulation=traces[-1], optical_only=True)
    before = next(iter(traces[-1].branches.values()))
    after = next(iter(hidden.branches.values()))
    for field in ['blocked_z', 'blocked_key', 'ray_weight', 'x', 'y', 'tx', 'ty']:
        np.testing.assert_array_equal(getattr(after, field), getattr(before, field))
    record_property('actual_tip_chain_stop_fractions_by_column_step', json.dumps(records))
    record_property('column_steps_mm', json.dumps([.125, .0625]))
