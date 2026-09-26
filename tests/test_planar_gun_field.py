"""Bounded checks of the opt-in planar-electrode electrostatic prototype.

The harmonic fixture checks interpolation and Laplace convergence independently
of a gun. The small annular fixture checks configuration and cache boundaries;
neither fixture qualifies the complete electron-gun transport chain.
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from temsim.physics.axisymmetric_cut_field import AxisymmetricCutField
from temsim.physics import planar_gun_field as planar


CATHODE = "planar_equipotential"


class FixtureLens(SimpleNamespace):
    def potential_rise_from_tip_v(self, extraction_kv, high_tension_kv):
        references = {"tip": 0.0, "extractor": extraction_kv,
                      "ground": high_tension_kv}
        return 1000.0 * (references[self.voltage_reference] + self.voltage_kv)


@pytest.fixture
def small_gun():
    """All gun electrodes in a millimetre-scale domain; no particle execution."""
    geometry = dict(mechanical_length_mm=0.1,
                    mechanical_clear_bore_diameter_mm=0.4,
                    mechanical_outer_diameter_mm=2.0)
    return SimpleNamespace(
        emitter=SimpleNamespace(surface_model=None, curvature_nm_inv=0.0,
                                coherence=None, ray_count=11, emission_current_na=2.0),
        extractor=SimpleNamespace(mechanical_center_from_tip_mm=0.4,
                                  voltage_kv=4.0, **geometry),
        electrostatic_lens=FixtureLens(mechanical_center_from_tip_mm=0.9,
                                      voltage_kv=1.2,
                                      voltage_reference="extractor",
                                      potential_scale=4.22125, **geometry),
        accelerator=SimpleNamespace(
            mechanical_center_from_tip_mm=3.75,
            mechanical_length_mm=4.58,
            mechanical_clear_bore_diameter_mm=0.4,
            mechanical_outer_diameter_mm=2.0,
            _electrode_thickness_mm=0.08, high_tension_kv=300.0,
            stages=[SimpleNamespace(center_from_tip_mm=1.5 + 0.5 * i,
                                    voltage_fraction=(i + 1) / 10)
                    for i in range(10)]),
        monochromator_installed=False, exit_plane_z_mm=7.0,
    )


def request(gun, **kwargs):
    return planar.planar_field_request(gun, cathode_boundary=CATHODE,
                                       cells_per_bore=4, **kwargs)


def harmonic_field(count):
    # Exact axisymmetric vacuum solution: Laplacian(z²-r²/2) = 0.
    radial = np.linspace(0.0, 0.02, count)
    axial = np.linspace(-0.02, 0.02, count)
    rr, zz = np.meshgrid(radial, axial, indexing="ij")
    values = 1e5 + 3e6 * zz + 2e8 * (zz**2 - 0.5 * rr**2)
    fixed = np.zeros_like(values, dtype=bool)
    fixed[-1, :] = fixed[:, 0] = fixed[:, -1] = True
    with threadpool_limits(limits=1):
        solved = AxisymmetricCutField(radial, axial, fixed, values)
    field = planar.PlanarGunField(
        {"fixture": "harmonic-vacuum", "mesh_count": count},
        radial, axial, solved.nodal_voltage,
        {"linear_residual": solved.residual},
    )
    return field, solved.residual


@pytest.fixture(scope="module")
def harmonic():
    return harmonic_field(33)[0]


def exact_potential(points):
    p = np.asarray(points)
    return 1e5 + 3e6 * p[:, 2] + 2e8 * (
        p[:, 2]**2 - 0.5 * (p[:, 0]**2 + p[:, 1]**2))


def test_harmonic_potential_and_field_converge_on_refinement():
    errors = []
    for count in (17, 33, 65):
        field, residual = harmonic_field(count)
        mids = 0.5 * (field.z[:-1] + field.z[1:])
        points = np.column_stack((np.full_like(mids, 0.005317),
                                  np.full_like(mids, 0.002173), mids))
        potential, electric = field.interpolate(points)
        spacing = np.max(np.diff(field.z))
        error = np.max(np.abs(potential - exact_potential(points)))
        errors.append(error)
        # Bilinear(r²,z) is exact in r² and first order in z. The quadratic
        # interpolation remainder is |Phi_zz| dz²/8; gradients are O(dz).
        assert error <= 2e8 * spacing**2 / 4 + 1e-5
        np.testing.assert_allclose(electric[:, :2], 2e8 * points[:, :2],
                                   rtol=0, atol=0.1)
        np.testing.assert_allclose(electric[:, 2], -3e6 - 4e8 * mids,
                                   rtol=0, atol=2e8 * spacing + 0.1)
        assert residual <= 1e-9
    assert errors[0] / errors[1] >= 3.9
    assert errors[1] / errors[2] >= 3.9


def test_electric_force_is_gradient_of_the_same_potential(harmonic):
    points = np.array([[0.004319, 0.002717, 0.003716],
                       [0.006317, -0.004121, -0.007321],
                       [-0.011197, 0.003129, 0.012137]])
    potential, electric = harmonic.interpolate(points)
    delta = 1e-8
    numerical = np.empty_like(electric)
    for axis in range(3):
        offset = np.eye(3)[axis] * delta
        numerical[:, axis] = -(
            harmonic.interpolate(points + offset)[0]
            - harmonic.interpolate(points - offset)[0]) / (2 * delta)
    np.testing.assert_allclose(electric, numerical, rtol=0, atol=0.1)
    np.testing.assert_array_equal(
        harmonic.potential_v_at_global_positions(points), potential)
    np.testing.assert_array_equal(
        harmonic.field_at_global_positions_v_per_m(points), electric)


def test_axis_regularity_and_rotation_covariance(harmonic):
    points = np.array([[0.004319, 0.002717, 0.003716],
                       [0.006317, -0.004121, -0.007321],
                       [0.0, 0.0, 0.012137]])
    voltage, electric = harmonic.interpolate(points)
    rotated = points[:, [1, 0, 2]] * [-1.0, 1.0, 1.0]
    r_voltage, r_electric = harmonic.interpolate(rotated)
    np.testing.assert_allclose(r_voltage, voltage, rtol=0, atol=1e-7)
    np.testing.assert_allclose(r_electric,
                               electric[:, [1, 0, 2]] * [-1.0, 1.0, 1.0],
                               rtol=0, atol=0.1)
    np.testing.assert_array_equal(electric[-1, :2], [0.0, 0.0])


@pytest.mark.parametrize("points", [
    [[0.02001, 0.0, 0.0]], [[0.0, 0.0, -0.02001]],
    [[0.0, 0.0, 0.02001]], [[0.0, 0.0, np.nan]],
    [[np.inf, 0.0, 0.0]], [[0.0, 0.0]], 0.0,
])
def test_invalid_or_outside_field_queries_are_rejected(harmonic, points):
    with pytest.raises(ValueError):
        harmonic.interpolate(points)


def test_single_point_and_shaped_queries_preserve_input_shape(harmonic):
    point = np.array([0.004319, 0.002717, 0.003716])
    single = harmonic.interpolate(point)
    batch = harmonic.interpolate(np.broadcast_to(point, (2, 4, 3)))
    assert single[0].shape == ()
    assert single[1].shape == (3,)
    assert batch[0].shape == (2, 4)
    assert batch[1].shape == (2, 4, 3)
    np.testing.assert_array_equal(batch[0], np.full((2, 4), single[0]))
    np.testing.assert_array_equal(batch[1], np.broadcast_to(single[1], (2, 4, 3)))


def test_default_electrodes_use_actual_voltages_and_keep_analytic_default():
    from temsim.optics.column import default_state
    from temsim.optics.electron_gun.electrostatic import FegElectrostaticField

    gun = default_state().electron_gun
    emitter_before = copy.deepcopy(vars(gun.emitter))
    result = request(gun)
    by_key = {ring["key"]: ring for ring in result["rings"]}
    assert set(by_key) == {"extractor", "electrostatic_lens"} | {
        f"accelerator:{index}" for index in range(10)}
    assert by_key["extractor"]["potential_rise_v"] == 4000.0
    assert by_key["electrostatic_lens"]["potential_rise_v"] == 5200.0
    expected = 4000.0 + np.arange(1, 11) / 10 * (300000.0 - 4000.0)
    np.testing.assert_allclose(
        [by_key[f"accelerator:{i}"]["potential_rise_v"] for i in range(10)],
        expected, rtol=0, atol=1e-9)
    assert result["domain"]["entrance_m"] == 0.0
    assert isinstance(gun.electric_field, FegElectrostaticField)
    # Constructing a field request must not replace or edit source parameters.
    assert repr(vars(gun.emitter)) == repr(emitter_before)


def test_small_electrode_solve_preserves_all_equipotentials(small_gun):
    original = copy.deepcopy(small_gun)
    with threadpool_limits(limits=1):
        field = planar.build_planar_gun_field(
            small_gun, cathode_boundary=CATHODE, cells_per_bore=4)
    assert field.r.size * field.z.size < 30000
    for ring in field.request["rings"]:
        query = [[0.5 * (ring["inner_m"] + ring["outer_m"]), 0.0,
                  0.5 * (ring["start_m"] + ring["stop_m"])]]
        value, electric = field.interpolate(query)
        np.testing.assert_allclose(value, ring["potential_rise_v"],
                                   rtol=0, atol=1e-7)
        np.testing.assert_allclose(electric, 0.0, rtol=0, atol=0.1)
    cathode = np.column_stack((field.r, np.zeros_like(field.r),
                               np.full_like(field.r, field.z[0])))
    exit_plane = cathode.copy()
    exit_plane[:, 2] = field.z[-1]
    np.testing.assert_allclose(field.interpolate(cathode)[0], 0.0, atol=1e-7)
    np.testing.assert_allclose(field.interpolate(exit_plane)[0], 300000.0,
                               rtol=0, atol=1e-7)
    assert repr(small_gun) == repr(original)


@pytest.mark.parametrize("path,name,value", [
    ("extractor", "voltage_kv", 4.5),
    ("extractor", "mechanical_center_from_tip_mm", 0.45),
    ("extractor", "mechanical_length_mm", 0.12),
    ("extractor", "mechanical_clear_bore_diameter_mm", 0.5),
    ("extractor", "mechanical_outer_diameter_mm", 2.2),
    ("electrostatic_lens", "voltage_kv", 1.4),
    ("electrostatic_lens", "voltage_reference", "tip"),
    ("electrostatic_lens", "mechanical_center_from_tip_mm", 0.95),
    ("electrostatic_lens", "mechanical_length_mm", 0.12),
    ("electrostatic_lens", "mechanical_clear_bore_diameter_mm", 0.5),
    ("electrostatic_lens", "mechanical_outer_diameter_mm", 2.2),
    ("accelerator", "high_tension_kv", 200.0),
    ("accelerator", "mechanical_clear_bore_diameter_mm", 0.5),
    ("accelerator", "mechanical_outer_diameter_mm", 2.2),
    ("accelerator", "_electrode_thickness_mm", 0.1),
    ("", "exit_plane_z_mm", 8.0),
])
def test_cache_identity_binds_consumed_electrode_inputs(small_gun, path, name, value):
    before = planar.request_digest(request(small_gun))
    changed = copy.deepcopy(small_gun)
    setattr(getattr(changed, path) if path else changed, name, value)
    assert planar.request_digest(request(changed)) != before


@pytest.mark.parametrize("name,value", [("center_from_tip_mm", 1.6),
                                         ("voltage_fraction", 0.11)])
def test_cache_identity_binds_individual_stage(small_gun, name, value):
    before = planar.request_digest(request(small_gun))
    setattr(small_gun.accelerator.stages[0], name, value)
    assert planar.request_digest(request(small_gun)) != before


@pytest.mark.parametrize("options", [{"cells_per_bore": 8},
                                        {"outer_factor": 3.0},
                                        {"exit_extension_mm": 1.0}])
def test_cache_identity_binds_grid_and_boundaries(small_gun, options):
    before = planar.request_digest(request(small_gun))
    kwargs = {"cells_per_bore": 4, **options}
    changed = planar.planar_field_request(small_gun, cathode_boundary=CATHODE,
                                         **kwargs)
    assert planar.request_digest(changed) != before


def test_uncharged_field_key_excludes_particle_count_current_and_analytic_scale(small_gun):
    before = planar.request_digest(request(small_gun))
    small_gun.emitter.ray_count *= 2
    small_gun.emitter.emission_current_na *= 3
    small_gun.electrostatic_lens.potential_scale *= 4
    assert planar.request_digest(request(small_gun)) == before


@pytest.mark.parametrize("boundary", [None, "", "flat", "analytic", {}, 0])
def test_planar_cathode_is_an_explicit_boundary_choice(small_gun, boundary):
    with pytest.raises((TypeError, ValueError)):
        planar.planar_field_request(small_gun, cathode_boundary=boundary)


def test_missing_cathode_choice_is_rejected(small_gun):
    with pytest.raises(TypeError):
        planar.planar_field_request(small_gun)


@pytest.mark.parametrize("attribute,value", [
    ("surface_model", SimpleNamespace()), ("curvature_nm_inv", 0.1),
    ("coherence", SimpleNamespace()),
])
def test_unsupported_source_models_are_not_silently_converted(small_gun, attribute, value):
    setattr(small_gun.emitter, attribute, value)
    with pytest.raises(ValueError):
        request(small_gun)
    assert getattr(small_gun.emitter, attribute) is value


def test_requested_monochromator_is_not_silently_bypassed(small_gun):
    small_gun.monochromator_installed = True
    with pytest.raises(ValueError):
        request(small_gun)


@pytest.mark.parametrize("options", [
    {"cells_per_bore": 0}, {"cells_per_bore": 3}, {"cells_per_bore": 65},
    {"cells_per_bore": 4.5}, {"cells_per_bore": True},
    {"outer_factor": 1.0}, {"outer_factor": np.nan},
    {"exit_extension_mm": -1.0}, {"exit_extension_mm": np.inf},
])
def test_invalid_numerical_or_outer_boundary_settings_are_rejected(small_gun, options):
    with pytest.raises((TypeError, ValueError)):
        planar.planar_field_request(small_gun, cathode_boundary=CATHODE, **options)


@pytest.mark.parametrize("path,name,value", [
    ("extractor", "mechanical_clear_bore_diameter_mm", 0.0),
    ("extractor", "mechanical_outer_diameter_mm", 0.3),
    ("extractor", "mechanical_length_mm", -0.1),
    ("extractor", "mechanical_center_from_tip_mm", 0.01),
    ("extractor", "voltage_kv", np.nan),
    ("electrostatic_lens", "mechanical_center_from_tip_mm", 0.4),
    ("accelerator", "_electrode_thickness_mm", 0.6),
    ("", "exit_plane_z_mm", 5.0),
])
def test_invalid_or_overlapping_electrodes_are_rejected(small_gun, path, name, value):
    setattr(getattr(small_gun, path) if path else small_gun, name, value)
    with pytest.raises(ValueError):
        request(small_gun)


def test_cache_round_trip_is_lossless_and_field_arrays_are_read_only(harmonic, tmp_path):
    planar.save_cached_field(harmonic, tmp_path)
    loaded = planar.load_cached_field(harmonic.request, tmp_path)
    assert loaded is not None
    points = np.array([[0.004319, 0.002717, 0.003716], [0.0, 0.0, 0.012137]])
    for actual, expected in zip(loaded.interpolate(points), harmonic.interpolate(points)):
        np.testing.assert_array_equal(actual, expected)
    for name in ("r", "z", "voltage"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(harmonic, name))
        assert not getattr(loaded, name).flags.writeable


def test_changed_request_does_not_reuse_a_field(harmonic, tmp_path):
    planar.save_cached_field(harmonic, tmp_path)
    changed = copy.deepcopy(harmonic.request)
    changed["mesh_count"] += 1
    assert planar.load_cached_field(changed, tmp_path) is None


@pytest.mark.parametrize("kind", ["archive", "manifest"])
def test_corrupted_cache_is_rejected(harmonic, tmp_path, kind):
    planar.save_cached_field(harmonic, tmp_path)
    if kind == "archive":
        archive = next(tmp_path.glob("*.npz"))
        data = bytearray(archive.read_bytes())
        data[len(data) // 2] ^= 1
        archive.write_bytes(data)
    else:
        manifest = next(tmp_path.glob("*.json"))
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["request"] = {"different": "boundary"}
        manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        planar.load_cached_field(harmonic.request, tmp_path)
