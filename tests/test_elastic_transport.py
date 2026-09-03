import math
from types import SimpleNamespace

import numpy as np
import pytest
import xraylib
from scipy.integrate import quad

from temsim.detector.eds_signal import EDSMaterial, elemental_material
from temsim.optics.column import default_state
from temsim.specimen.elastic_transport import (
    ElasticTransportGeometry,
    IncidentElectronRay,
    incident_rays_from_simulation,
    elastic_mean_free_path_nm,
    rotate_direction_after_scatter,
    sample_screened_rutherford_angle,
    screened_rutherford_angle_cdf,
    screened_rutherford_differential_cross_section_cm2_sr,
    screened_rutherford_parameter,
    screened_rutherford_total_cross_section_cm2,
    simulate_elastic_point_transport,
)


def _incident_rays(count, *, x_nm=0.0, y_nm=0.0, energy_ev=300_000.0):
    return tuple(
        IncidentElectronRay(
            source_ray_index=index,
            position_xy_nm=(x_nm, y_nm),
            direction=(0.0, 0.0, 1.0),
            kinetic_energy_ev=energy_ev,
            weight=1.0 / count,
        )
        for index in range(count)
    )


def test_screened_rutherford_reference_values_and_declared_units():
    assert screened_rutherford_parameter(14, 200_000.0) == pytest.approx(
        9.874935747058295e-05, rel=1.0e-14
    )
    assert screened_rutherford_total_cross_section_cm2(
        14, 200_000.0
    ) == pytest.approx(1.890013470524361e-18, rel=1.0e-14)
    assert screened_rutherford_total_cross_section_cm2(
        79, 200_000.0
    ) == pytest.approx(1.8983252085642375e-17, rel=1.0e-14)
    with pytest.raises(ValueError, match="Z=1..99"):
        screened_rutherford_total_cross_section_cm2(100, 200_000.0)
    with pytest.raises(ValueError, match="positive"):
        screened_rutherford_total_cross_section_cm2(14, 0.0)


def test_differential_cross_section_integrates_to_total_cross_section():
    atomic_number = 14
    energy_ev = 200_000.0
    integrated, _error = quad(
        lambda theta: 2.0
        * math.pi
        * math.sin(theta)
        * screened_rutherford_differential_cross_section_cm2_sr(
            theta, atomic_number, energy_ev
        ),
        0.0,
        math.pi,
        epsabs=1.0e-30,
        epsrel=1.0e-10,
        limit=300,
    )
    assert integrated == pytest.approx(
        screened_rutherford_total_cross_section_cm2(
            atomic_number, energy_ev
        ),
        rel=1.0e-10,
    )


def test_inverse_angle_sampler_matches_analytical_cdf_and_is_reproducible():
    reference_rng = np.random.default_rng(73)
    expected_cdf = float(reference_rng.random())
    expected_phi = 2.0 * math.pi * float(reference_rng.random())
    theta, phi = sample_screened_rutherford_angle(
        np.random.default_rng(73), 29, 300_000.0
    )

    assert screened_rutherford_angle_cdf(
        theta, 29, 300_000.0
    ) == pytest.approx(expected_cdf, abs=2.0e-15)
    assert phi == pytest.approx(expected_phi, abs=2.0e-15)
    assert screened_rutherford_angle_cdf(
        math.pi, 29, 300_000.0
    ) == pytest.approx(1.0)


def test_mixture_mean_free_path_uses_macroscopic_number_density_sum():
    material = EDSMaterial(
        key="mixture",
        name="mixture",
        density_g_cm3=3.1,
        mass_fractions=((6, 0.2), (14, 0.8)),
        provenance="unit test",
    )
    energy_ev = 200_000.0
    expected_rate_cm_inverse = sum(
        material.density_g_cm3
        * fraction
        * 6.02214076e23
        / xraylib.AtomicWeight(atomic_number)
        * screened_rutherford_total_cross_section_cm2(
            atomic_number, energy_ev
        )
        for atomic_number, fraction in material.mass_fractions
    )

    assert elastic_mean_free_path_nm(material, energy_ev) == pytest.approx(
        1.0e7 / expected_rate_cm_inverse
    )


def test_scatter_rotation_preserves_norm_and_requested_polar_angle():
    initial = np.asarray((0.2, -0.3, 0.9), dtype=float)
    initial /= np.linalg.norm(initial)
    theta = 0.37
    rotated = rotate_direction_after_scatter(initial, theta, 1.2)

    assert np.linalg.norm(rotated) == pytest.approx(1.0, abs=2.0e-15)
    assert float(initial @ rotated) == pytest.approx(
        math.cos(theta), abs=2.0e-15
    )


def test_incident_bundle_uses_only_survivors_and_preserves_phase_space():
    state = default_state()
    simulation = SimpleNamespace(
        incident=SimpleNamespace(
            alive=np.asarray((True, False, True)),
            x=np.asarray(((1.0e-9, 99.0e-9, 3.0e-9),)),
            y=np.asarray(((2.0e-9, 99.0e-9, 6.0e-9),)),
            tx=np.asarray(((1.0e-3, 99.0, 3.0e-3),)),
            ty=np.asarray(((2.0e-3, 99.0, -2.0e-3),)),
            energy_offset_ev=np.asarray((-1.0, 88.0, 2.0)),
            ray_weight=np.asarray((0.2, 0.5, 0.3)),
        )
    )

    bundle = incident_rays_from_simulation(
        state, simulation, target_x_nm=10.0, target_y_nm=-5.0
    )

    assert bundle.emitted_ray_count == 3
    assert bundle.reaching_ray_count == 2
    assert bundle.surviving_fraction == pytest.approx(0.5)
    assert [ray.source_ray_index for ray in bundle.rays] == [0, 2]
    assert [ray.weight for ray in bundle.rays] == pytest.approx((0.4, 0.6))
    assert sum(
        ray.weight * ray.position_xy_nm[0] for ray in bundle.rays
    ) == pytest.approx(10.0)
    assert sum(
        ray.weight * ray.position_xy_nm[1] for ray in bundle.rays
    ) == pytest.approx(-5.0)
    assert bundle.chief_angle_mrad == pytest.approx((2.2, -0.4))
    assert bundle.energy_range_ev == pytest.approx((299_999.0, 300_002.0))
    assert bundle.rays[0].direction[0] > 0.0
    assert bundle.rays[0].direction[1] > 0.0


def test_incident_bundle_interpolates_an_upstream_boundary_before_later_stop():
    state = default_state()
    sample_z = float(state.sample.z_mm)
    simulation = SimpleNamespace(
        incident=SimpleNamespace(
            z=np.asarray((sample_z - 10.0, sample_z)),
            alive=np.asarray((True, False)),
            blocked_z=np.asarray((np.nan, sample_z - 1.0)),
            x=np.asarray(((0.0, 2.0e-9), (10.0e-9, 12.0e-9))),
            y=np.zeros((2, 2)),
            tx=np.asarray(((0.0, 2.0e-3), (1.0e-3, 3.0e-3))),
            ty=np.zeros((2, 2)),
            energy_offset_ev=np.zeros(2),
            ray_weight=np.asarray((0.25, 0.75)),
        )
    )

    bundle = incident_rays_from_simulation(
        state, simulation, boundary_z_mm=sample_z - 5.0
    )

    assert bundle.boundary_z_mm == pytest.approx(sample_z - 5.0)
    assert bundle.reaching_ray_count == 2
    assert bundle.surviving_fraction == pytest.approx(1.0)
    assert [ray.position_xy_nm[0] for ray in bundle.rays] == pytest.approx(
        (5.0, 7.0)
    )


def test_finite_geometry_finds_sample_face_and_mesh_sidewall():
    state = default_state()
    geometry = ElasticTransportGeometry.from_state(state)
    start = np.asarray(
        (0.0, 0.0, geometry.sample_top_nm - 8.0 * geometry.epsilon_nm)
    )
    distance = geometry.next_region_boundary_distance_nm(
        start,
        (0.0, 0.0, 1.0),
        None,
        maximum_distance_nm=1.0,
    )
    assert distance == pytest.approx(8.0 * geometry.epsilon_nm)

    state.sample.eds_support_material_key = "copper"
    grid_geometry = ElasticTransportGeometry.from_state(state)
    inside_opening = np.asarray(
        (0.0, 0.0, grid_geometry.support_top_nm + 1.0)
    )
    sidewall = grid_geometry.next_region_boundary_distance_nm(
        inside_opening,
        (1.0, 0.0, 0.0),
        None,
        maximum_distance_nm=1.0e6,
    )
    expected = 0.5 * grid_geometry.support_grid.mesh.hole_width_um * 1000.0
    assert sidewall == pytest.approx(expected)
    after = grid_geometry.region_at(
        inside_opening + np.asarray((sidewall + 1.0e-3, 0.0, 0.0))
    )
    assert after is not None
    assert after.source_key == "support:bar"


def test_finite_geometry_uses_the_circular_disk_sidewall():
    state = default_state()
    geometry = ElasticTransportGeometry.from_state(state)
    inside = np.asarray((0.0, 0.0, 0.0))
    region = geometry.region_at(inside)

    assert region is not None
    assert geometry.region_at((1_400_000.0, 1_400_000.0, 5.0)) is None
    distance = geometry.next_region_boundary_distance_nm(
        inside,
        (1.0, 0.0, 0.0),
        region,
        maximum_distance_nm=2_000_000.0,
    )
    assert distance == pytest.approx(1_500_000.0)


def test_tilted_incident_ray_is_back_projected_from_the_reference_plane():
    state = default_state()
    direction = np.asarray((0.1, -0.05, 1.0), dtype=float)
    direction /= np.linalg.norm(direction)
    ray = IncidentElectronRay(
        source_ray_index=0,
        position_xy_nm=(12.0, -7.0),
        direction=tuple(float(value) for value in direction),
        kinetic_energy_ev=300_000.0,
        weight=1.0,
    )

    result = simulate_elastic_point_transport(
        state,
        incident_rays=(ray,),
        seed=3,
        stored_trajectory_count=1,
    )

    start = result.trajectories[0].points_nm[0]
    projected_to_reference = start[:2] - (
        start[2] / direction[2]
    ) * direction[:2]
    assert projected_to_reference == pytest.approx((12.0, -7.0))
    assert start[2] < -0.5 * state.sample.thickness_nm


def test_point_transport_is_seeded_and_aggregates_real_material_paths():
    state = default_state()
    state.sample.eds_elastic_seed = 101
    rays = _incident_rays(12)
    first = simulate_elastic_point_transport(state, incident_rays=rays)
    second = simulate_elastic_point_transport(state, incident_rays=rays)

    assert first.metrics["total_elastic_events"] > 0
    assert first.metrics["terminal_mean_energy_ev"] == pytest.approx(
        first.metrics["incident_mean_energy_ev"]
    )
    assert sum(first.metrics["outcome_counts"].values()) == 12
    assert first.metrics["event_limit_fraction"] == 0.0
    assert sum(
        track.electron_weight * track.path_length_nm
        for track in first.eds_tracks
    ) == pytest.approx(
        first.metrics["mean_material_path_nm"]
    )
    assert {track.history for track in first.eds_tracks} >= {
        "straight_primary",
        "elastic_scattered",
    }
    assert all(
        track.source_ray_index is not None for track in first.eds_tracks
    )
    assert first.metrics == second.metrics
    assert first.terminal_electrons is not None
    assert first.terminal_electrons.position_nm.shape == (12, 3)
    assert first.terminal_electrons.direction.shape == (12, 3)
    assert len(first.terminal_electrons.outcome) == 12
    assert first.material_flights
    assert all(
        np.linalg.norm(
            np.asarray(flight.end_nm) - np.asarray(flight.start_nm)
        )
        > 0.0
        for flight in first.material_flights
    )
    for left, right in zip(
        first.trajectories, second.trajectories, strict=True
    ):
        assert np.array_equal(left.points_nm, right.points_nm)


def test_point_transport_progress_reports_completed_histories():
    state = default_state()
    progress = []

    simulate_elastic_point_transport(
        state,
        incident_rays=_incident_rays(3),
        progress_callback=lambda completed, total, label: progress.append(
            (completed, total, label)
        ),
    )

    assert progress[0] == (0, 3, "Preparing elastic specimen histories")
    assert [completed for completed, _total, _label in progress] == [0, 1, 2, 3]
    assert all(total == 3 for _completed, total, _label in progress)
    assert progress[-1][2] == "Elastic specimen history 3/3"


def test_retracted_holder_removes_sample_and_support_from_transport():
    state = default_state()
    state.sample.inserted = False
    state.sample.eds_support_material_key = "gold"
    result = simulate_elastic_point_transport(
        state, incident_rays=_incident_rays(3, x_nm=60_000.0)
    )

    assert result.eds_tracks == ()
    assert result.metrics["total_elastic_events"] == 0
    assert result.metrics["transmitted_fraction"] == 1.0


def test_heavy_support_sets_accuracy_warning_and_event_guard():
    state = default_state()
    state.sample.thickness_nm = 0.0
    state.sample.eds_support_material_key = "gold"
    state.sample.eds_support_mesh_key = "square_200"
    result = simulate_elastic_point_transport(
        state,
        incident_rays=_incident_rays(1, x_nm=60_000.0),
        maximum_events_per_trajectory=1,
    )

    assert result.metrics["rutherford_heavy_element_warning"] is True
    assert result.metrics["total_elastic_events"] == 1
    assert result.metrics["event_limit_fraction"] == 1.0
    assert result.trajectories[0].outcome == "event_limit"


def test_elemental_mean_free_path_decreases_from_si_to_au_at_200_kev():
    silicon = elemental_material(14, density_g_cm3=2.33)
    gold = elemental_material(79)

    assert elastic_mean_free_path_nm(gold, 200_000.0) < (
        elastic_mean_free_path_nm(silicon, 200_000.0)
    )
