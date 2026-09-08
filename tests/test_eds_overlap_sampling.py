"""Density/normalization tests; no column propagation or multislice needed."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import ncx2

from temsim.optics.column import default_state
from temsim.specimen.interaction_types import IncidentElectronRay, IncidentRayBundle
from temsim.specimen.overlap_sampling import build_overlap_sampling_plan


def _gaussian_positions(sigma=50.0, radial_count=32, angular_count=32):
    quantiles = (np.arange(radial_count) + 0.5) / radial_count
    radius = sigma * np.sqrt(-2.0 * np.log1p(-quantiles))
    angle = 2.0 * np.pi * np.arange(angular_count) / angular_count
    return np.column_stack((
        (radius[:, None] * np.cos(angle)[None, :]).ravel(),
        (radius[:, None] * np.sin(angle)[None, :]).ravel(),
    ))


def _bundle(positions, weights=None, slopes=None, energies=None):
    count = len(positions)
    weights = np.full(count, 1.0 / count) if weights is None else np.asarray(weights)
    slopes = np.zeros((count, 2)) if slopes is None else np.asarray(slopes)
    energies = np.full(count, 300_000.0) if energies is None else np.asarray(energies)
    directions = np.column_stack((slopes, np.ones(count)))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    rays = tuple(IncidentElectronRay(
        source_ray_index=index, position_xy_nm=tuple(position),
        direction=tuple(direction), kinetic_energy_ev=float(energy), weight=float(weight),
    ) for index, position, direction, energy, weight in zip(
        range(count), positions, directions, energies, weights, strict=True,
    ))
    centroid = np.average(positions, axis=0, weights=weights)
    return IncidentRayBundle(
        rays=rays, emitted_ray_count=2*count, reaching_ray_count=count,
        surviving_fraction=0.4, original_centroid_nm=tuple(centroid),
        target_centroid_nm=tuple(centroid), chief_angle_mrad=(0.0, 0.0),
        energy_range_ev=(float(energies.min()), float(energies.max())),
    )


def _transport(hits=0, **extra):
    return SimpleNamespace(metrics={
        "sample_hit_trajectory_count": hits, "sample_axial_field_t": 0.0,
        "sample_face_fields_t": (0.0, 0.0), **extra,
    })


@pytest.fixture
def state():
    result = default_state()
    result.sample.eds_overlap_sampling_enabled = True
    result.sample.eds_overlap_sampling_points = 1024
    result.sample.eds_support_material_key = "vacuum"
    result.sample.size_x_nm = result.sample.size_y_nm = 10.0
    result.sample.thickness_nm = 5.0
    return result


@pytest.mark.parametrize("centre", ((0.0, 0.0), (45.0, -20.0)))
def test_rare_gaussian_overlap_matches_analytic_kernel_mixture(state, centre):
    positions = _gaussian_positions()
    bundle = _bundle(positions)
    state.sample.centre_x_nm, state.sample.centre_y_nm = centre
    plan = build_overlap_sampling_plan(state, bundle, _transport())
    assert plan.metrics["eds_overlap_sampling_status"] == "active"
    kernel_cov = np.asarray(plan.metrics["eds_overlap_kernel_covariance_nm2"])
    np.testing.assert_allclose(kernel_cov, np.eye(2) * kernel_cov[0, 0], atol=1.0e-12)
    kernel_variance = kernel_cov[0, 0]
    exact_kde = float(np.mean(ncx2.cdf(
        25.0 / kernel_variance, 2,
        np.sum((positions - centre)**2, axis=1) / kernel_variance,
    )))
    mass = math.fsum(ray.weight for ray in plan.rays)
    assert mass == pytest.approx(exact_kde, rel=0.02)
    assert 0.0 < mass < 0.01
    assert mass == plan.metrics["eds_overlap_probability_mass"]
    # The reference Gaussian and the bandwidth-smoothed mixture are distinct
    # models; finite sampling/bandwidth bias must remain visible in this test.
    gaussian_probability = float(ncx2.cdf(25.0 / 2500.0, 2, np.dot(centre, centre)/2500.0))
    assert mass == pytest.approx(gaussian_probability, rel=0.15)
    assert bundle.surviving_fraction == 0.4
    assert math.fsum(ray.weight for ray in bundle.rays) == 1.0
    assert plan.metrics["eds_overlap_quadrature_effective_point_count"] > 32


def test_absolute_tiny_mass_is_not_renormalized_to_one(state):
    plan = build_overlap_sampling_plan(state, _bundle(_gaussian_positions(5000.0)), _transport())
    assert plan.rays
    assert 0.0 < math.fsum(ray.weight for ray in plan.rays) < 1.0e-6
    assert all(0.0 < ray.weight < 1.0e-8 for ray in plan.rays)


def test_tiny_gaussian_tails_do_not_underflow_the_weight_diagnostic(state):
    state.sample.centre_x_nm = 600.0
    plan = build_overlap_sampling_plan(state, _bundle(_gaussian_positions()), _transport())
    assert plan.rays
    assert 0.0 < plan.metrics["eds_overlap_probability_mass"] < 1.0e-160
    assert math.isfinite(plan.metrics["eds_overlap_quadrature_effective_point_count"])


def test_unequal_weights_preserve_parent_direction_energy_and_source_identity(state):
    cloud = _gaussian_positions(30.0, 8, 16)
    positions = np.vstack((cloud + (-160.0, 0.0), cloud + (160.0, 0.0)))
    count = len(cloud)
    weights = np.r_[np.full(count, 0.2/count), np.full(count, 0.8/count)]
    slopes = np.vstack((np.tile((-0.02, 0.0), (count, 1)), np.tile((0.02, 0.0), (count, 1))))
    energies = np.r_[np.full(count, 200_000.0), np.full(count, 300_000.0)]
    bundle = _bundle(positions, weights, slopes, energies)
    state.sample.centre_x_nm = 160.0
    first = build_overlap_sampling_plan(state, bundle, _transport())
    second = build_overlap_sampling_plan(state, bundle, _transport())
    assert first == second
    assert first.rays
    assert len(first.parent_source_ray_indices) == len(first.rays)
    for ray, parent_id in zip(first.rays, first.parent_source_ray_indices, strict=True):
        parent = bundle.rays[parent_id]
        assert ray.source_ray_index >= bundle.emitted_ray_count
        assert ray.direction == parent.direction
        assert ray.kinetic_energy_ev == parent.kinetic_energy_ev
    assert np.mean(np.asarray(first.parent_source_ray_indices) >= count) > 0.99
    assert first.metrics["eds_overlap_effective_original_ray_count"] == pytest.approx(1.0/np.sum(weights**2))
    np.testing.assert_array_equal([ray.position_xy_nm for ray in bundle.rays], positions)
    np.testing.assert_array_equal([ray.weight for ray in bundle.rays], weights)


def test_annular_non_gaussian_beam_keeps_a_depressed_centre(state):
    angle = np.arange(1024) * (2.0 * np.pi / 1024)
    bundle = _bundle(200.0 * np.column_stack((np.cos(angle), np.sin(angle))))
    centre = build_overlap_sampling_plan(state, bundle, _transport())
    state.sample.centre_x_nm = 200.0
    ring = build_overlap_sampling_plan(state, bundle, _transport())
    assert centre.rays and ring.rays
    # Gaussian kernels have nonzero tails: do not assert or claim a perfectly
    # reconstructed hard hole. A single covariance Gaussian would peak at 0.
    assert centre.metrics["eds_overlap_probability_mass"] < 0.01 * ring.metrics["eds_overlap_probability_mass"]
    assert "Gaussian tails" in centre.metrics["eds_overlap_sampling_limitations"]


def test_tilted_side_entry_uses_padded_domain_and_original_node_count(state):
    positions = _gaussian_positions()
    slopes = np.tile((0.08, 0.0), (len(positions), 1))
    state.sample.envelope_shape = "rectangle"
    state.sample.centre_x_nm, state.sample.centre_y_nm = 20.0, -10.0
    plan = build_overlap_sampling_plan(state, _bundle(positions, slopes=slopes), _transport())
    assert plan.rays
    nodes = np.asarray([ray.position_xy_nm for ray in plan.rays])
    assert np.any(np.abs(nodes[:, 0] - 20.0) > 5.0)
    assert np.max(np.abs(nodes[:, 0] - 20.0)) <= 5.2 + 1.0e-7
    assert np.max(np.abs(nodes[:, 1] + 10.0)) <= 5.0 + 1.0e-7
    assert plan.metrics["eds_overlap_side_pad_nm"][0] == pytest.approx(0.2, abs=1.0e-7)
    # The rectangle's projected straight-ray silhouette has this area.
    assert plan.metrics["eds_overlap_domain_area_nm2"] == pytest.approx(104.0, abs=1.0e-5)


@pytest.mark.parametrize(("setting", "value", "status"), (
    ("eds_overlap_sampling_enabled", False, "disabled"),
    ("inserted", False, "no_material"),
    ("thickness_nm", 0.0, "no_material"),
    ("eds_support_material_key", "copper", "unsupported_support"),
    ("eds_overlap_sampling_points", 8192, "invalid_point_count"),
    ("thickness_nm", 100.0, "unsupported_thickness_or_angle"),
))
def test_unsupported_or_inactive_settings_preserve_original_estimate(state, setting, value, status):
    setattr(state.sample, setting, value)
    plan = build_overlap_sampling_plan(state, _bundle(_gaussian_positions()), _transport())
    assert not plan.rays
    assert plan.metrics["eds_overlap_sampling_status"] == status
    assert plan.metrics["eds_overlap_sampling_detail"]


def test_focused_and_sufficient_hit_cases_do_not_add_a_second_estimate(state):
    focused = build_overlap_sampling_plan(state, _bundle(_gaussian_positions(1.0)), _transport())
    sufficient = build_overlap_sampling_plan(state, _bundle(_gaussian_positions()), _transport(32))
    assert focused.metrics["eds_overlap_sampling_status"] == "sample_not_small_relative_to_beam"
    assert sufficient.metrics["eds_overlap_sampling_status"] == "adequate_original_hits"
    assert not focused.rays and not sufficient.rays


def test_covariance_and_effective_sample_size_are_not_fabricated(state):
    line = _bundle(np.column_stack((np.arange(128), np.zeros(128))))
    singular = build_overlap_sampling_plan(state, line, _transport())
    assert singular.metrics["eds_overlap_sampling_status"] == "degenerate_covariance"
    positions = _gaussian_positions(50.0, 8, 8)
    weights = np.full(len(positions), 0.5 / (len(positions)-1))
    weights[0] = 0.5
    inadequate = build_overlap_sampling_plan(state, _bundle(positions, weights), _transport())
    assert inadequate.metrics["eds_overlap_sampling_status"] == "insufficient_effective_rays"
    assert not singular.rays and not inadequate.rays


@pytest.mark.parametrize(("extra", "status"), (
    ({"sample_field_geometry_material_coupled": True}, "unsupported_mapped_field"),
    ({"sample_field_source": "same runtime measured/FEM and fallback providers"}, "unsupported_mapped_field"),
    ({"sample_axial_field_t": 1.0, "sample_face_fields_t": (0.5, 1.5)}, "unsupported_local_field"),
    ({"sample_face_fields_t": None}, "missing_field_diagnostic"),
))
def test_unbounded_or_nonuniform_field_domains_fall_back(state, extra, status):
    plan = build_overlap_sampling_plan(state, _bundle(_gaussian_positions()), _transport(**extra))
    assert not plan.rays
    assert plan.metrics["eds_overlap_sampling_status"] == status


def test_mass_above_one_is_rejected_without_clipping(state, monkeypatch):
    import temsim.specimen.overlap_sampling as sampling
    original = sampling.logsumexp
    monkeypatch.setattr(sampling, "logsumexp", lambda *a, **kw: original(*a, **kw) + math.log(1.0e9))
    plan = build_overlap_sampling_plan(state, _bundle(_gaussian_positions()), _transport())
    assert not plan.rays
    assert plan.metrics["eds_overlap_sampling_status"] == "invalid_probability_mass"
    assert plan.metrics["eds_overlap_probability_mass"] > 1.0
