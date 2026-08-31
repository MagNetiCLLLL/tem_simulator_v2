import math

import numpy as np
import pytest
import xraylib
from scipy.integrate import quad

from temsim.detector.eds_signal import EDSMaterial, elemental_material
from temsim.optics.column import default_state
from temsim.specimen.elastic_transport import (
    ElasticTransportGeometry,
    elastic_mean_free_path_nm,
    rotate_direction_after_scatter,
    sample_screened_rutherford_angle,
    screened_rutherford_angle_cdf,
    screened_rutherford_differential_cross_section_cm2_sr,
    screened_rutherford_parameter,
    screened_rutherford_total_cross_section_cm2,
    simulate_elastic_point_transport,
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


def test_finite_geometry_finds_sample_face_and_mesh_sidewall():
    state = default_state()
    geometry = ElasticTransportGeometry.from_state(state)
    start = np.asarray((0.0, 0.0, -8.0 * geometry.epsilon_nm))
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


def test_point_transport_is_seeded_and_aggregates_real_material_paths():
    state = default_state()
    state.sample.eds_elastic_trajectory_count = 12
    state.sample.eds_elastic_seed = 101
    first = simulate_elastic_point_transport(state, x_nm=0.0, y_nm=0.0)
    second = simulate_elastic_point_transport(state, x_nm=0.0, y_nm=0.0)

    assert first.metrics["total_elastic_events"] > 0
    assert sum(first.metrics["outcome_counts"].values()) == 12
    assert first.metrics["event_limit_fraction"] == 0.0
    assert sum(track.path_length_nm for track in first.eds_tracks) == pytest.approx(
        first.metrics["mean_material_path_nm"]
    )
    assert {track.history for track in first.eds_tracks} >= {
        "straight_primary",
        "elastic_scattered",
    }
    assert first.metrics == second.metrics
    for left, right in zip(
        first.trajectories, second.trajectories, strict=True
    ):
        assert np.array_equal(left.points_nm, right.points_nm)


def test_retracted_holder_removes_sample_and_support_from_transport():
    state = default_state()
    state.sample.inserted = False
    state.sample.eds_support_material_key = "gold"
    result = simulate_elastic_point_transport(
        state, x_nm=60_000.0, y_nm=0.0, trajectory_count=3
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
        x_nm=60_000.0,
        y_nm=0.0,
        trajectory_count=1,
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

