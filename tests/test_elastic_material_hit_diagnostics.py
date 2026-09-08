"""Finite-specimen hit accounting must not confuse plane current with matter."""

import pytest

from temsim.optics.column import default_state
from temsim.specimen.elastic_transport import simulate_elastic_point_transport
from temsim.specimen.interaction_types import IncidentElectronRay


def _state():
    state = default_state()
    state.sample.size_x_nm = 10.0
    state.sample.size_y_nm = 10.0
    state.sample.thickness_nm = 5.0
    state.sample.eds_support_material_key = "vacuum"
    return state


def _ray(index, x_nm, weight):
    return IncidentElectronRay(index, (x_nm, 0.0), (0.0, 0.0, 1.0), 300_000.0, weight)


def test_material_hits_retain_weights_and_exclude_zero_weight_histories():
    result = simulate_elastic_point_transport(
        _state(), incident_rays=(_ray(0, 0.0, 0.2), _ray(1, 50.0, 0.8), _ray(2, 0.0, 0.0)),
        stored_trajectory_count=0,
    )
    metrics = result.metrics
    assert metrics["trajectory_count"] == 3
    assert metrics["positive_weight_trajectory_count"] == 2
    assert metrics["sample_hit_trajectory_count"] == 1
    assert metrics["material_hit_trajectory_count"] == 1
    assert metrics["sample_hit_weight_fraction"] == pytest.approx(0.2)
    assert metrics["material_hit_weight_fraction"] == pytest.approx(0.2)
    assert metrics["incident_position_centroid_nm"] == pytest.approx((40.0, 0.0))
    assert metrics["incident_position_rms_radius_nm"] == pytest.approx(20.0)
    assert metrics["material_sampling_status"] == "sampled_material_hits"
    assert {track.source_ray_index for track in result.eds_tracks} == {0}
    # Accounting must cover all histories, even when none are retained for 3D.
    assert result.trajectories == ()


def test_no_sampled_hits_are_distinct_from_no_material():
    state = _state()
    rays = (_ray(0, -50.0, 0.5), _ray(1, 50.0, 0.5))
    miss = simulate_elastic_point_transport(state, incident_rays=rays)
    assert miss.metrics["material_sampling_status"] == "no_sampled_material_hits"
    assert miss.metrics["sample_hit_trajectory_count"] == 0
    assert miss.metrics["material_hit_weight_fraction"] == 0.0
    assert miss.eds_tracks == ()

    state.sample.inserted = False
    vacuum = simulate_elastic_point_transport(state, incident_rays=rays)
    assert vacuum.metrics["material_sampling_status"] == "no_material"
    assert vacuum.metrics["sample_hit_trajectory_count"] == 0
    assert vacuum.eds_tracks == ()


def test_hits_do_not_require_elastic_collisions():
    state = _state()
    result = simulate_elastic_point_transport(state, incident_rays=(_ray(0, 0.0, 1.0),), seed=0)
    assert result.metrics["total_elastic_events"] == 0
    assert result.metrics["sample_hit_trajectory_count"] == 1
    assert result.metrics["sample_hit_weight_fraction"] == pytest.approx(1.0)
    assert sum(track.path_length_nm for track in result.eds_tracks) > 0.0
