"""Geometry and electrostatic boundaries for the executed gun-field provider."""
from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from temsim.physics.continuous_curvature_conductor import (
    ContinuousCurvatureConductor,
    continuous_curvature_conductor,
)


@pytest.fixture
def curved_gun():
    from temsim.optics.column import default_state

    gun = default_state().electron_gun
    gun.emitter.curvature_nm_inv = 0.01
    return gun


@pytest.mark.parametrize("curvature", [1e-14, 1e-8, 0.001, 0.01, 0.1])
def test_conductor_preserves_existing_spherical_emission_and_source_arrays(curved_gun, curvature):
    emitter = curved_gun.emitter
    emitter.curvature_nm_inv = curvature
    before = emitter.emit(193)
    source_inputs = copy.deepcopy(vars(emitter))
    geometry = continuous_curvature_conductor(emitter)
    after = emitter.emit(193)
    assert emitter.surface_model is None
    assert repr(vars(emitter)) == repr(source_inputs)
    for key in ("x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev", "weight", "ray_id",
                "surface_position_m", "surface_normal", "surface_direction"):
        np.testing.assert_array_equal(getattr(after, key), getattr(before, key))
    points = before.surface_position_m
    radius = np.hypot(points[:, 0], points[:, 1])
    np.testing.assert_allclose(geometry.surface_z_m(radius), points[:, 2],
                               rtol=3e-15, atol=1e-24)
    np.testing.assert_allclose(geometry.normal_at_positions(points), before.surface_normal,
                               rtol=0, atol=8e-16)


def test_weak_curvature_allows_sphere_clipped_before_cone(curved_gun):
    curved_gun.emitter.curvature_nm_inv = 1e-8
    geometry = continuous_curvature_conductor(curved_gun.emitter)
    assert geometry.apex_radius_nm > geometry.shank_length_um * 1000.0
    assert np.isfinite(geometry.radius_m(geometry.back_z_m))
    assert geometry.radius_m(geometry.back_z_m) > 0
    assert geometry.radius_m(geometry.back_z_m - 1e-12) == 0
    assert geometry.radius_m(1e-12) == 0
    assert geometry.surface_z_m(0.0) == 0


def test_cap_cone_inverse_and_normals_agree_at_tangent_join(curved_gun):
    geometry = continuous_curvature_conductor(curved_gun.emitter)
    radius = geometry.apex_radius_nm * 1e-9
    angle = np.deg2rad(geometry.cone_half_angle_deg)
    join = radius * np.cos(angle)
    r = np.array([0.0, 0.5 * join, join, 1.2 * join])
    z = geometry.surface_z_m(r)
    np.testing.assert_allclose(geometry.radius_m(z), r, rtol=1e-13, atol=1e-22)
    points = np.column_stack((r, np.zeros_like(r), z))
    normals = geometry.normal_at_positions(points)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-14)
    np.testing.assert_allclose(normals[2], [np.cos(angle), 0.0, np.sin(angle)], atol=1e-14)
    np.testing.assert_allclose(normals[3], normals[2], atol=1e-14)
    inside = points.copy()
    inside[:, 2] -= 1e-12
    outside = points.copy()
    outside[:, 2] += 1e-12
    assert np.all(geometry.material_mask(inside))
    assert not np.any(geometry.material_mask(outside))


def test_adapter_is_frozen_and_detached_from_emitter(curved_gun):
    geometry = continuous_curvature_conductor(curved_gun.emitter)
    with pytest.raises(FrozenInstanceError):
        geometry.apex_radius_nm = 42.0
    curved_gun.emitter.curvature_nm_inv = 0.02
    assert geometry.apex_radius_nm == 100.0


def test_emission_beyond_cap_cone_join_is_rejected_without_conversion(curved_gun):
    curved_gun.emitter.curvature_nm_inv = 0.1
    curved_gun.emitter.tip_cone_half_angle_deg = 60.0
    # The original spherical emitter is valid; this cone would intersect its
    # outer emitting support and cannot silently replace those launch normals.
    original = curved_gun.emitter.emit(193)
    with pytest.raises(ValueError, match="spherical cap"):
        continuous_curvature_conductor(curved_gun.emitter)
    assert curved_gun.emitter.surface_model is None
    np.testing.assert_array_equal(curved_gun.emitter.emit(193).surface_position_m,
                                   original.surface_position_m)


@pytest.mark.parametrize("arguments", [
    (0.0, 5.0, 1000.0, 1.0), (np.inf, 5.0, 1000.0, 1.0),
    (100.0, 0.0, 1000.0, 1.0), (100.0, 90.0, 1000.0, 1.0),
    (100.0, 5.0, -1.0, 1.0), (100.0, 5.0, 1000.0, -1.0),
    (100.0, 5.0, 0.0001, 10.0),
])
def test_invalid_conductor_or_support_beyond_back_is_rejected(arguments):
    with pytest.raises(ValueError):
        ContinuousCurvatureConductor(*arguments).validate()


@pytest.fixture(scope="module")
def continuous_fields(tmp_path_factory):
    from temsim.optics.column import default_state
    from temsim.physics.continuous_gun_field import build_continuous_gun_field

    results = []
    for curvature, apex, nodes in [(1e-14, 16, 64), (0.01, 16, 64), (0.01, 32, 128)]:
        gun = default_state().electron_gun
        gun.emitter.curvature_nm_inv = curvature
        cache = tmp_path_factory.mktemp(f"continuous-{curvature}-{apex}")
        field = build_continuous_gun_field(gun, cells_per_bore=4,
            apex_cells_per_radius=apex, tip_nodes=nodes, cache_dir=cache)
        field._regular.compiled = False  # Independent NumPy interpolation here.
        results.append((gun, field, cache, apex, nodes))
    return results


def test_curved_requests_bind_geometry_and_do_not_admit_a_planar_substitution(curved_gun):
    from temsim.physics.closed_gun_field import build_closed_gun_field
    from temsim.physics.continuous_gun_field import continuous_field_request
    from temsim.physics.planar_gun_field import request_digest

    original = continuous_field_request(curved_gun)
    assert original["source_admission"] == "continuous_curvature_classical_tip"
    assert original["domain"]["entrance_m"] < 0.0
    assert original["cathode_geometry"]["apex_radius_nm"] == 100.0
    with pytest.raises(ValueError, match="curved"):
        build_closed_gun_field(curved_gun)
    for name, value in (("curvature_nm_inv", 0.02), ("tip_cone_half_angle_deg", 6.0),
                        ("mechanical_length_mm", 1.2)):
        changed = copy.deepcopy(curved_gun)
        setattr(changed.emitter, name, value)
        assert request_digest(continuous_field_request(changed)) != request_digest(original)


def test_true_weak_curvature_limit_keeps_exact_launch_points_and_finite_force(continuous_fields):
    gun, field, *_ = continuous_fields[0]
    emitted = gun.emit(193)
    points = emitted.surface_position_m.copy()
    assert field.geometry.radius_m(field.geometry.back_z_m) > field.r[-1]
    potential, electric = field.interpolate(points)
    assert np.isfinite(potential).all() and np.isfinite(electric).all()
    assert np.all(electric[:, 2] < 0.0)  # The emitted electrons see forward extraction force.
    assert np.max(np.abs(potential)) < 1e-3
    assert not np.any(field.tip_material_mask(points))
    assert np.all(field.tip_material_mask(points - [0.0, 0.0, 1e-12]))
    np.testing.assert_array_equal(points, emitted.surface_position_m)


def test_curved_launch_boundary_error_decreases_on_mesh_refinement(continuous_fields):
    reports = []
    for gun, field, *_ in continuous_fields[1:]:
        points = gun.emit(193).surface_position_m
        report = field.launch_boundary_report(points)
        assert report["launch_positions_changed"] is False
        assert report["maximum_launch_potential_error_v"] < 1e-3
        assert report["maximum_surface_representation_error_m"] < 0.01e-9
        assert field.report["linear_residual"] < 1e-9
        reports.append(report)
    assert reports[1]["maximum_launch_potential_error_v"] < reports[0]["maximum_launch_potential_error_v"] / 2
    assert reports[1]["maximum_surface_representation_error_m"] < reports[0]["maximum_surface_representation_error_m"] / 2


def test_curved_field_force_matches_same_potential_and_does_not_extrapolate(continuous_fields):
    field = continuous_fields[1][1]
    points = np.array([[1.23e-5, 0.71e-5, 0.02717], [0.77e-5, -0.23e-5, 0.09371]])
    potential, electric = field.interpolate(points)
    delta = 1e-9
    numerical = np.empty_like(electric)
    for axis in range(3):
        offset = np.eye(3)[axis] * delta
        numerical[:, axis] = -(field.interpolate(points + offset)[0]
                                - field.interpolate(points - offset)[0]) / (2 * delta)
    np.testing.assert_allclose(electric, numerical, rtol=1e-5, atol=0.1)
    with pytest.raises(ValueError, match="outside"):
        field.interpolate([[0.0, 0.0, field.z[-1] + 1e-3]])


def test_curved_cache_restores_full_cut_operator_and_rejects_corruption(continuous_fields):
    from temsim.physics.continuous_gun_field import build_continuous_gun_field

    gun, field, cache, apex, nodes = continuous_fields[1]
    loaded = build_continuous_gun_field(gun, cells_per_bore=4,
        apex_cells_per_radius=apex, tip_nodes=nodes, cache_dir=cache)
    loaded._regular.compiled = False
    assert loaded.report["cache_hit"] is True
    points = np.r_[gun.emit(193).surface_position_m,
                   [[1.23e-5, 0.71e-5, 0.02717], [0.77e-5, -0.23e-5, 0.09371]]]
    for actual, expected in zip(loaded.interpolate(points), field.interpolate(points)):
        np.testing.assert_array_equal(actual, expected)
    archive = next(cache.glob("*.npz"))
    data = bytearray(archive.read_bytes())
    data[len(data) // 2] ^= 1
    archive.write_bytes(data)
    with pytest.raises(ValueError, match="checksum"):
        build_continuous_gun_field(gun, cells_per_bore=4,
            apex_cells_per_radius=apex, tip_nodes=nodes, cache_dir=cache)


def test_closed_field_reuses_across_gun_instances_and_source_counts(monkeypatch, tmp_path):
    from collections import OrderedDict
    from temsim.optics.column import default_state
    from temsim.physics import closed_gun_field as closed

    monkeypatch.setattr(closed, "_MEMORY_FIELDS", OrderedDict())
    monkeypatch.setattr(closed, "field_cache_directory", lambda: tmp_path)
    gun = default_state().electron_gun
    closed.closed_field_request(gun)  # Bind the actual resolved environment first.
    gun._gun_field_cells_per_bore = 4
    other = copy.deepcopy(gun)
    other.emitter.ray_count *= 2
    other.emitter.emission_current_na *= 2
    first = closed.closed_field(gun)
    second = closed.closed_field(other)
    assert second is first
    assert gun.emitter.surface_model is None and other.emitter.surface_model is None
    del other._closed_gun_field
    closed._MEMORY_FIELDS.clear()
    restored = closed.closed_field(other)
    assert restored.report["cache_hit"] is True
    np.testing.assert_array_equal(restored.voltage, first.voltage)


def test_closed_liner_geometry_is_fixed_when_numerical_endpoint_changes():
    from temsim.optics.column import default_state
    from temsim.physics.closed_gun_field import closed_field_request

    gun = default_state().electron_gun
    short = closed_field_request(gun, cells_per_bore=4, exit_extension_mm=100.0)
    long = closed_field_request(gun, cells_per_bore=4, exit_extension_mm=200.0)
    assert short["grounded_liner"] == long["grounded_liner"]
    assert short["domain"]["gun_exit_m"] == long["domain"]["gun_exit_m"]
    assert short["domain"]["exit_m"] < long["domain"]["exit_m"]
    assert short["ground_assignment"] == long["ground_assignment"]


def test_closed_liner_bore_changes_cache_and_broken_continuity_is_rejected():
    from temsim.optics.column import default_state
    from temsim.physics.closed_gun_field import closed_field_request
    from temsim.physics.planar_gun_field import request_digest

    gun = default_state().electron_gun
    base = closed_field_request(gun, cells_per_bore=4)
    rows = [dict(start_z_mm=p["start_m"] * 1000, end_z_mm=p["stop_m"] * 1000,
                 inner_diameter_mm=p["inner_m"] * 2000,
                 outer_diameter_mm=p["outer_m"] * 2000) for p in base["grounded_liner"]]
    before = closed_field_request(gun, liner_segments=rows, cells_per_bore=4)
    changed = copy.deepcopy(rows)
    changed[0]["inner_diameter_mm"] *= 0.9
    after = closed_field_request(gun, liner_segments=changed, cells_per_bore=4)
    assert request_digest(before) != request_digest(after)
    changed[0]["start_z_mm"] += 0.1
    with pytest.raises(ValueError, match="start"):
        closed_field_request(gun, liner_segments=changed, cells_per_bore=4)
