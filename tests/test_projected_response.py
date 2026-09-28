"""Small conditional-response fixtures; no gun solve, wave solve or full column."""
from types import SimpleNamespace
import math

import numpy as np
import pytest

from temsim.detector import projected_response as response
from temsim.specimen.downstream_transport import GeometricSpecimenExit
from temsim.specimen.elastic_transport import (
    incident_rays_from_simulation, screened_rutherford_total_cross_section_cm2,
)
from temsim.specimen.projected_scattering import ProjectedElement


def _state():
    return SimpleNamespace(
        lenses=[], stigmators=[], deflectors=[], corrector_elements=[], apertures=[],
        beam_voltage_kv=300., step_mm=.05, history_step_mm=.1,
        acceleration_enabled=False, acceleration_backend="CPU", simulation_mode="custom",
        projector_mode="diffraction", equivalent_image_lenses_enabled=False,
        sample=SimpleNamespace(z_mm=0., thickness_nm=5.),
        vacuum_map=SimpleNamespace(enabled=False), chromatic_aberration_enabled=False,
        objective_lens=SimpleNamespace(cc_mm=0.), condenser_system={},
        recording_planes=[], stem_detectors=[], column_inner_diameter_mm=20.,
    )


def _simulation(count=4):
    x = np.linspace(-2., 2., count)*1e-9
    y = np.linspace(3., -1., count)*1e-9
    energy = np.linspace(150_000., 300_000., count)
    return SimpleNamespace(incident=SimpleNamespace(
        z=np.array([0.]), x=x[None], y=y[None],
        tx=np.full((1, count), .01), ty=np.full((1, count), -.02),
        alive=np.ones(count, dtype=bool), blocked_z=np.full(count, np.nan),
        energy_offset_ev=energy-300_000., kinetic_energy_ev=energy[None],
        ray_weight=np.arange(1, count+1, dtype=float)/sum(range(1, count+1)),
        source_ray_id=np.arange(count)+100, source_azimuth_rad=np.zeros(count),
    ))


def _projection(state, simulation):
    rays = incident_rays_from_simulation(state, simulation).rays
    energies = np.array([ray.kinetic_energy_ev for ray in rays])
    weights = np.array([ray.weight for ray in rays])
    sigma = np.array([screened_rutherford_total_cross_section_cm2(14, energy)*1e14
                      for energy in energies])
    mean = weights @ sigma
    element = ProjectedElement(14, np.zeros(1), np.zeros(1), mean,
                               energies, weights*sigma/mean)
    return SimpleNamespace(elements=(element,))


def _detector(z, key="detector"):
    return SimpleNamespace(z_mm=z, key=key, inserted=True, readout_enabled=False,
                           hit_mask=lambda x, y: np.ones_like(x, dtype=bool))


def test_uses_executed_source_identity_state_and_conditional_weights_without_mutation(monkeypatch):
    state, simulation = _state(), _simulation(120)
    simulation.incident.alive[2] = False
    simulation.incident.blocked_z[2] = -1.
    before = simulation.incident.ray_weight.copy()
    projection = _projection(state, simulation)
    calls = []
    marker = object()

    def captured(_state, local_simulation, elastic, distribution, **kwargs):
        calls.append((local_simulation, elastic.terminal_electrons, distribution, kwargs))
        assert math.fsum(local_simulation.incident.ray_weight) == pytest.approx(1.)
        assert local_simulation.incident.ray_weight[2] == 0.
        return GeometricSpecimenExit((), {})

    monkeypatch.setattr(response, "build_geometric_specimen_exit", captured)
    result = response.build_projected_responses(state, simulation, projection, marker, stop_z_mm=.2)
    assert tuple(result) == ("direct", "vacuum", "Z14")
    assert len(calls[0][1].weight) <= 64
    assert len(calls[2][1].weight) == (len(response._polar_edges())-1)*16
    assert calls[0][2] is marker and calls[1][2] is None and calls[2][2] is marker
    for local_simulation, terminal, _, options in calls:
        assert local_simulation is not simulation
        assert options["capture_sections"] is False
        assert options["recording_interceptions"] is False
        assert sum(terminal.weight) == pytest.approx(1., abs=2e-15)
        indices = terminal.source_ray_index
        assert 2 not in indices
        np.testing.assert_allclose(terminal.position_nm[:, :2],
            np.column_stack((simulation.incident.x[0, indices], simulation.incident.y[0, indices]))*1e9,
            rtol=4e-16, atol=5e-16)
        np.testing.assert_array_equal(terminal.kinetic_energy_ev, simulation.incident.kinetic_energy_ev[0, indices])
        assert terminal.reference_time_offset_s is None
    np.testing.assert_array_equal(calls[0][1].material_path_nm, 5.)
    np.testing.assert_array_equal(calls[1][1].material_path_nm, 0.)
    np.testing.assert_array_equal(simulation.incident.ray_weight, before)
    assert not simulation.incident.alive[2]
    assert all(not item.checkpoints and not item.segments and not item.dependency_signature for item in result.values())
    assert all(item.metrics["source_survival_applied"] is False for item in result.values())


def test_scattering_rotation_is_relative_to_each_actual_incident_direction():
    state, simulation = _state(), _simulation()
    simulation.incident.tx[0] = [-.3, .1, .02, .4]
    simulation.incident.ty[0] = [.2, -.1, -.4, .02]
    rays = incident_rays_from_simulation(state, simulation).rays
    element = _projection(state, simulation).elements[0]
    rows, theta, phi, weights = response._angular_rows(rays, element)
    terminal = response._terminal(rays, rows, weights, 5., theta=theta, phi=phi)
    reference = np.array([rays[index].direction for index in rows])
    np.testing.assert_allclose(np.einsum("ij,ij->i", terminal.direction, reference), np.cos(theta), atol=4e-15)
    assert "backscattered" in terminal.outcome
    forward = np.array(terminal.outcome) == "transmitted"
    assert 0. < sum(terminal.weight[forward]) < 1.
    # Backward probabilities remain in the same unit ledger.
    assert sum(terminal.weight[~forward]) > 0.


def test_auxiliary_real_drift_keeps_detector_planes_and_defers_recording_only():
    state, simulation = _state(), _simulation()
    detector = _detector(.137)
    state.recording_planes = [detector]
    state.stem_detectors = [detector]
    # If the auxiliary path attempted to create material cache identities,
    # this fixture would fail rather than silently create a restart object.
    state.to_dict = lambda: (_ for _ in ()).throw(AssertionError("No auxiliary checkpoint"))
    result = response.build_projected_responses(state, simulation, SimpleNamespace(elements=()))
    direct = result["direct"]
    assert not direct.checkpoints and not direct.segments
    assert len(direct.branches) == 1
    branch = direct.branches[0]
    np.testing.assert_array_equal(branch.z, (0., .137))
    np.testing.assert_allclose(branch.x[-1], simulation.incident.x[0] + .137e-3*.01, atol=2e-19)
    np.testing.assert_allclose(branch.y[-1], simulation.incident.y[0] - .137e-3*.02, atol=2e-19)
    assert np.all(branch.alive)  # root's per-pixel collector owns this detector
    assert all(key == "" for key in branch.blocked_key)
    assert branch.weight == pytest.approx(1.)
    np.testing.assert_array_equal(branch.kinetic_energy_ev[0], simulation.incident.kinetic_energy_ev[0])
    assert np.all(np.isnan(branch.flight_time_s))


def test_auxiliary_still_clips_physical_aperture_before_deferred_detector():
    state, simulation = _state(), _simulation()
    state.apertures = [SimpleNamespace(key="closed_aperture", z_mm=.073, installed=True,
        enabled=True, radius_mm=0., offset_x_mm=0., offset_y_mm=0.)]
    detector = _detector(.137)
    state.recording_planes = [detector]
    state.stem_detectors = [detector]
    result = response.build_projected_responses(state, simulation, SimpleNamespace(elements=()))
    for item in result.values():
        branch = item.branches[0]
        assert .073 in branch.z
        assert not np.any(branch.alive)
        np.testing.assert_array_equal(branch.blocked_z, .073)
        assert set(branch.blocked_key) == {"closed_aperture"}


def test_auxiliary_still_clips_column_wall():
    state, simulation = _state(), _simulation()
    state.column_inner_diameter_mm = .001
    result = response.build_projected_responses(state, simulation, SimpleNamespace(elements=()), stop_z_mm=.2)
    branch = result["direct"].branches[0]
    assert not np.any(branch.alive)
    assert set(branch.blocked_key) == {"column_wall"}
    assert np.all((branch.blocked_z > 0.) & (branch.blocked_z < .2))


def test_material_absorption_is_once_and_vacuum_response_never_gets_it():
    from test_downstream_transport import _test_inelastic_distribution
    state, simulation = _state(), _simulation()
    results = response.build_projected_responses(state, simulation, SimpleNamespace(elements=()),
        _test_inelastic_distribution(), stop_z_mm=.1)
    assert sum(branch.weight for branch in results["direct"].branches) == pytest.approx(.9)
    assert results["direct"].metrics["inelastic_absorbed_source_probability"] == pytest.approx(.1)
    assert sum(branch.weight for branch in results["vacuum"].branches) == pytest.approx(1.)
    assert results["vacuum"].metrics["inelastic_absorbed_source_probability"] == 0.


@pytest.mark.parametrize("stop", [.1, .2])
def test_filter_is_not_silently_bypassed(stop):
    state, simulation = _state(), _simulation()
    state.energy_filter_installed = True
    state.energy_filter = SimpleNamespace(enabled=True, entrance_z_mm=.1)
    with pytest.raises(ValueError, match="cannot yet traverse"):
        response.build_projected_responses(state, simulation, SimpleNamespace(elements=()), stop_z_mm=stop)


def test_detector_before_filter_does_not_traverse_filter():
    state, simulation = _state(), _simulation()
    state.energy_filter_installed = True
    state.energy_filter = SimpleNamespace(enabled=True, entrance_z_mm=.2)
    state.stem_detectors = [_detector(.1)]
    result = response.build_projected_responses(state, simulation, SimpleNamespace(elements=()))
    assert result["direct"].branches[0].z[-1] == .1


def test_real_scattered_response_keeps_nontransmitted_probability():
    state, simulation = _state(), _simulation()
    result = response.build_projected_responses(state, simulation, _projection(state, simulation), stop_z_mm=.1)
    scattered = result["Z14"]
    assert 0. < scattered.metrics["elastic_nontransmitted_conditional_probability"] < 1.
    forward = sum(branch.weight for branch in scattered.branches)
    assert forward < 1.
    assert forward + scattered.metrics["elastic_nontransmitted_conditional_probability"] == pytest.approx(1.)
    assert not scattered.checkpoints and not scattered.segments


def test_angular_refinement_preserves_full_sphere_probability(monkeypatch):
    state, simulation = _state(), _simulation()
    rays = incident_rays_from_simulation(state, simulation).rays
    element = _projection(state, simulation).elements[0]
    for polar, azimuth in ((16, 8), (32, 16), (64, 32)):
        monkeypatch.setattr(response, "POLAR_BIN_COUNT", polar)
        monkeypatch.setattr(response, "AZIMUTH_SAMPLE_COUNT", azimuth)
        rows, theta, phi, weights = response._angular_rows(rays, element)
        assert len(rows) == (len(response._polar_edges())-1)*azimuth
        assert sum(weights) == pytest.approx(1., abs=1e-14)
        assert np.all((theta >= 0.) & (theta <= math.pi))
        assert np.all((phi >= 0.) & (phi < 2*math.pi))


@pytest.mark.parametrize("energy", [200_000., 300_000.])
@pytest.mark.parametrize("collapsed_sine", [False, True])
def test_zero_mass_neighbouring_angle_edges_are_omitted_without_reweighting(energy, collapsed_sine):
    state, simulation = _state(), _simulation()
    simulation.incident.kinetic_energy_ev[:] = energy
    rays = incident_rays_from_simulation(state, simulation).rays
    element = _projection(state, simulation).elements[0]
    edges = (np.array([0., np.nextafter(math.pi, 0.), math.pi]) if collapsed_sine
             else np.array([0., .1, np.nextafter(.1, np.inf), math.pi]))
    _, expected_phi, expected_weights = response.screened_angular_quadrature(
        element, edges, azimuth_samples=response.AZIMUTH_SAMPLE_COUNT)
    positive = expected_weights > 0.
    if collapsed_sine:
        assert np.any(~positive)  # Both last edges map to exactly sin²(theta/2)=1.
    else:
        assert np.all(positive)  # Stable interval integration retains tiny positive mass.
    rows, theta, phi, weights = response._angular_rows(rays, element, edges)
    np.testing.assert_array_equal(weights, expected_weights[positive])
    np.testing.assert_array_equal(phi, expected_phi[positive])
    assert len(rows) == len(theta) == len(phi) == len(weights)
    assert np.all(np.isfinite(theta))
    assert math.fsum(weights) == math.fsum(expected_weights)


def test_response_cache_binds_hardware_incident_and_angular_inputs_and_detaches_outputs(monkeypatch):
    from collections import OrderedDict
    from dataclasses import replace
    from temsim.optics.column import default_state
    from temsim.physics.simulation import Branch
    state = default_state()
    simulation = _simulation()
    simulation.incident.z = np.array([state.sample.z_mm])
    projection = _projection(state, simulation)
    stop = float(state.sample.z_mm)+.1
    monkeypatch.setattr(response, "_RESPONSE_CACHE", OrderedDict())
    monkeypatch.setattr(response, "_RESPONSE_CACHE_SIZE", 0)
    calls = []

    def fake(_state, _simulation, _elastic, _inelastic, **kwargs):
        calls.append(1)
        z = np.asarray(kwargs["save_z_mm"])
        zeros = np.zeros((len(z), 1))
        branch = Branch("fixture", (1, 1, 1), z, zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(),
                        np.array([True]), np.array([np.nan]), [""], 1., np.zeros(1), ray_weight=np.ones(1))
        return GeometricSpecimenExit((branch,), {})

    monkeypatch.setattr(response, "build_geometric_specimen_exit", fake)
    first = response.build_projected_responses(state, simulation, projection, stop_z_mm=stop)
    assert len(calls) == 3
    first["direct"].branches[0].x[:] = 42.
    first["direct"].metrics["mutated"] = True
    second = response.build_projected_responses(state, simulation, projection, stop_z_mm=stop)
    assert len(calls) == 3
    assert second["direct"].metrics["response_cache_hit"] is True
    assert "mutated" not in second["direct"].metrics
    np.testing.assert_array_equal(second["direct"].branches[0].x, 0.)
    second["direct"].branches[0].x[:] = 7.
    third = response.build_projected_responses(state, simulation, projection, stop_z_mm=stop)
    np.testing.assert_array_equal(third["direct"].branches[0].x, 0.)

    # Pixel-domain optical-depth maps are not transport inputs.
    changed_map = SimpleNamespace(elements=(replace(projection.elements[0],
        optical_depth=np.array([1., 2., 3.]), scattered_fraction=np.array([.2, .4, .5])),))
    response.build_projected_responses(state, simulation, changed_map, stop_z_mm=stop)
    assert len(calls) == 3
    simulation.incident.tx[0, 0] += .001
    response.build_projected_responses(state, simulation, projection, stop_z_mm=stop)
    assert len(calls) == 6
    state.step_mm *= .5
    response.build_projected_responses(state, simulation, projection, stop_z_mm=stop)
    assert len(calls) == 9
    changed_cdf = SimpleNamespace(elements=(replace(projection.elements[0],
        cross_section_weights=np.roll(projection.elements[0].cross_section_weights, 1)),))
    response.build_projected_responses(state, simulation, changed_cdf, stop_z_mm=stop)
    assert len(calls) == 12
    response.build_projected_responses(state, simulation, projection, stop_z_mm=stop+.001)
    assert len(response._RESPONSE_CACHE) <= 4
    assert response._RESPONSE_CACHE_SIZE <= response.RESPONSE_CACHE_BYTES


def test_failure_or_cancelled_completion_does_not_enter_response_cache(monkeypatch):
    from collections import OrderedDict
    from temsim.optics.column import default_state
    state = default_state()
    simulation = _simulation()
    simulation.incident.z = np.array([state.sample.z_mm])
    monkeypatch.setattr(response, "_RESPONSE_CACHE", OrderedDict())
    monkeypatch.setattr(response, "_RESPONSE_CACHE_SIZE", 0)
    monkeypatch.setattr(response, "build_geometric_specimen_exit", lambda *args, **kwargs: GeometricSpecimenExit((), {}))

    def cancel(done, total, _message):
        if done == total:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        response.build_projected_responses(state, simulation, SimpleNamespace(elements=()),
            stop_z_mm=state.sample.z_mm+.1, progress_callback=cancel)
    assert not response._RESPONSE_CACHE


@pytest.mark.parametrize("physical_stop", [False, True])
def test_auxiliary_nonfinite_history_distinguishes_prior_intercept_from_field_domain(monkeypatch, physical_stop):
    from temsim.specimen import downstream_transport
    state, simulation = _state(), _simulation()
    if physical_stop:
        state.apertures = [SimpleNamespace(key="closed_aperture", z_mm=.05, installed=True,
            enabled=True, radius_mm=0., offset_x_mm=0., offset_y_mm=0.)]

    def truncated(_state, start, stop, x, tx, y, ty, *args, **kwargs):
        assert kwargs["defer_nonfinite_until_clipping"] is True
        z = np.array([0., .05, .1, .15, .2])
        arrays = [np.broadcast_to(values, (len(z), len(x))).copy() for values in (x, tx, y, ty)]
        for array in arrays:
            array[3:, 0] = np.nan
        energy = np.broadcast_to(kwargs["initial_kinetic_energy_ev"], arrays[0].shape).copy()
        energy[3:, 0] = np.nan
        kwargs["energy_output"].append(energy)
        return (z, *arrays, np.full_like(arrays[0], np.nan))

    monkeypatch.setattr(downstream_transport, "propagate", truncated)
    result = response.build_projected_responses(state, simulation, SimpleNamespace(elements=()), stop_z_mm=.2)
    for item in result.values():
        branch = item.branches[0]
        if physical_stop:
            assert set(branch.blocked_key) == {"closed_aperture"}
            assert item.metrics["numerical_domain_truncated_conditional_probability"] == 0.
        else:
            assert branch.blocked_key[0] == "projected_field_domain"
            assert not branch.alive[0] and np.all(branch.alive[1:])
            assert branch.blocked_z[0] == .15
            assert item.metrics["numerical_domain_truncated_conditional_probability"] == pytest.approx(.1)
        assert not item.segments and not item.checkpoints


def test_detector_angle_boundary_hints_match_analytic_cdf_at_two_quadrature_resolutions(monkeypatch):
    from temsim.specimen.elastic_transport import screened_rutherford_angle_cdf
    state, simulation = _state(), _simulation()
    simulation.incident.x[:] = simulation.incident.y[:] = 0.
    simulation.incident.tx[:] = simulation.incident.ty[:] = 0.
    simulation.incident.kinetic_energy_ev[:] = 200_000.
    simulation.incident.energy_offset_ev[:] = -100_000.
    state.column_inner_diameter_mm = 1e6
    definitions = {"BF": (0., .005), "DF": (.005, .03), "HAADF": (.03, .2)}
    for name, (inner, outer) in definitions.items():
        z = .1
        r0, r1 = z*math.tan(inner), z*math.tan(outer)
        state.recording_planes.append(SimpleNamespace(key=name, z_mm=z, inserted=True,
            inner_diameter_mm=2*r0, outer_width_mm=2*r1,
            hit_mask=lambda x, y, lo=r0, hi=r1: (np.hypot(x, y) >= lo) & (np.hypot(x, y) <= hi)))
    projection = _projection(state, simulation)
    measured = []
    for polar, azimuth in ((32, 16), (64, 32)):
        monkeypatch.setattr(response, "POLAR_BIN_COUNT", polar)
        monkeypatch.setattr(response, "AZIMUTH_SAMPLE_COUNT", azimuth)
        result = response.build_projected_responses(state, simulation, projection, stop_z_mm=.1)["Z14"]
        values = {}
        for detector in state.recording_planes:
            values[detector.key] = sum(branch.weight * np.sum(branch.ray_weight[
                detector.hit_mask(branch.x[-1]*1e3, branch.y[-1]*1e3)]) for branch in result.branches)
            lo, hi = definitions[detector.key]
            expected = (screened_rutherford_angle_cdf(hi, 14, 200_000.)
                        - screened_rutherford_angle_cdf(lo, 14, 200_000.))
            assert values[detector.key] == pytest.approx(expected, abs=2e-13)
        measured.append(values)
    for name in definitions:
        assert measured[0][name] == pytest.approx(measured[1][name], abs=2e-13)


def test_vacuum_response_defers_recording_inside_medium_without_mutating_instrument(monkeypatch):
    from temsim.physics import residual_medium
    from temsim.physics.recording_clipping import clip_recording_planes
    from temsim.vacuum import Medium, ResolvedMedium, VacuumMap
    state, simulation = _state(), _simulation()
    state.sample.inserted = False
    state.sample.thickness_nm = 0.
    state.vacuum_map = VacuumMap(enabled=True)
    state.step_mm = .025
    first, last = _detector(.05, "first"), _detector(.2, "last")
    state.recording_planes = [first, last]
    state.stem_detectors = [first, last]
    original_planes = state.recording_planes
    region = ResolvedMedium("fixture_gas", "Short dilute gas", 0., .2,
                            Medium(pressure_mbar=1e-20))
    # Isolate the medium/recording coupling with a declared tiny gas interval;
    # use the real stepwise ColumnMediumTransport, not a fake propagator.
    monkeypatch.setattr(residual_medium, "resolve_regions", lambda _state: (region,))
    result = response.build_projected_responses(state, simulation, SimpleNamespace(elements=()))
    assert state.recording_planes is original_planes
    assert state.recording_planes == [first, last]
    for item in result.values():
        branch = item.branches[0]
        assert np.all(branch.alive)
        assert set(branch.blocked_key) == {""}
        report = branch.vacuum_report
        gas = next(row for row in report["regions"] if row["key"] == "fixture_gas")
        assert gas["mean_path_m"] == pytest.approx(.2e-3*math.sqrt(1.+.01**2+.02**2), rel=1e-12)
        # Recording remains a sequential physical collector after the raster
        # position is known: the first absorbing detector owns every ray.
        alive, blocked, keys = clip_recording_planes(state, branch.z, branch.x, branch.y,
            branch.alive, branch.blocked_z, branch.blocked_key)
        assert not np.any(alive)
        assert set(keys) == {"first"}
        np.testing.assert_array_equal(blocked, .05)


def test_real_instrument_state_vacuum_response_keeps_property_backed_recorders():
    from temsim.optics.column import default_state
    from temsim.physics.recording_clipping import clip_recording_planes
    state = default_state()
    state.vacuum_map.enabled = True
    state.sample.inserted = False
    state.sample.thickness_nm = 0.
    state.step_mm = .5
    state.acceleration_backend = "Numba CPU"
    simulation = _simulation()
    simulation.incident.tx[:] = simulation.incident.ty[:] = 0.
    simulation.incident.kinetic_energy_ev[:] = 300_000.
    simulation.incident.energy_offset_ev[:] = 0.
    start = float(state.sample.z_mm)
    simulation.incident.z = np.array([start])
    first, last = state.fluorescent_screen, state.camera
    for plane in state.recording_planes:
        plane.inserted = False
    first.inserted = last.inserted = True
    # Real components resolve Z from their mechanical anchors on serialising.
    # Keep that geometry and test physical object/state preservation, rather
    # than list identity (canonical serialisation reorders the list container).
    state.to_dict()
    original_planes = tuple(state.recording_planes)
    original_settings = [(plane.inserted, plane.z_mm) for plane in original_planes]
    result = response.build_projected_responses(state, simulation, SimpleNamespace(elements=()),
                                                stop_z_mm=last.z_mm)
    assert all(actual is expected for actual, expected in zip(state.recording_planes, original_planes))
    assert [(plane.inserted, plane.z_mm) for plane in state.recording_planes] == original_settings
    assert state.fluorescent_screen is first and state.camera is last
    assert first.inserted and last.inserted
    for item in result.values():
        branch = item.branches[0]
        assert np.all(branch.alive)
        assert set(branch.blocked_key) == {""}
        gas_path = sum(region["mean_path_m"] for region in branch.vacuum_report["regions"])
        # Gas transport continues to the camera, not only to the nearer screen.
        assert gas_path == pytest.approx((last.z_mm-start)*1e-3, rel=1e-6)
        alive, blocked, keys = clip_recording_planes(state, branch.z, branch.x, branch.y,
            branch.alive, branch.blocked_z, branch.blocked_key)
        assert not np.any(alive)
        assert set(keys) == {first.key}
        np.testing.assert_array_equal(blocked, first.z_mm)
