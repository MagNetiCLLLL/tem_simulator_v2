"""Independent mathematical/transport regressions, not full-source acceptance."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.linalg import expm

from temsim.optics.model import Stigmator
from temsim.optics.column import default_state
from temsim.physics import ray_integrator as rk
from temsim.physics.core import build_propagation_plan, propagation_plan_common_prefix_nodes
from temsim.physics.ray_device_cache import plan_identity


def test_independent_channels_and_double_angle_orientation():
    stig = Stigmator("test", "test", 0., field_model="normal_skew")
    columns = []
    for x, y in ((10., 0.), (0., 10.)):
        stig.strength_x_percent, stig.strength_y_percent = x, y
        xx, yy, xy = stig.quadrupole_tensor_m2(np.array([0.]))
        assert xx + yy == pytest.approx([0.])
        columns.append((xx[0], xy[0]))
    assert np.linalg.matrix_rank(np.array(columns)) == 2
    stig.strength_x_percent = stig.strength_y_percent = 10.
    xx, _, xy = stig.quadrupole_tensor_m2(0.)
    assert np.rad2deg(.5*np.arctan2(xy, xx)) == pytest.approx(22.5)
    stig.strength_x_percent = -10.
    stig.strength_y_percent = 0.
    xx, _, xy = stig.quadrupole_tensor_m2(0.)
    assert .5*np.arctan2(xy, xx) == pytest.approx(np.pi/2)


def test_legacy_default_and_equal_inputs_still_cancel():
    stig = Stigmator("old", "old", 0., strength_x_percent=10., strength_y_percent=10.)
    assert stig.field_model == "legacy_difference"
    assert np.asarray(stig.quadrupole_tensor_m2(0.)) == pytest.approx(np.zeros(3))
    stig.strength_y_percent = 0.
    assert stig.quadrupole_tensor_m2(0.) == pytest.approx((15., -15., 0.))


def tensor_inputs():
    n = 100
    stages = 2*n+1
    x = np.array([1.e-6, -2.e-6, 3.e-6])
    return (np.full(stages, 200.), np.full(stages, -200.),
            np.zeros(stages), np.zeros(stages), np.zeros(stages), np.ones(3),
            np.zeros(n+1), np.zeros(n+1), np.zeros(n+1), np.full(n, 1.e-4),
            x, x*100., -x*.5, x*30., np.zeros(n+1), np.zeros(n+1),
            np.array([0, n], dtype=np.int64), np.array([n], dtype=np.int64),
            np.full(stages, 80.))


@pytest.mark.parametrize("engine", [rk.vectorised_rk4, rk.serial_rk4, rk.parallel_rk4])
def test_full_tensor_matches_independent_matrix_exponential(engine):
    inputs = tensor_inputs()
    result = engine(*inputs)
    generator = np.array([[0., 1., 0., 0.], [-200., 0., -80., 0.],
                          [0., 0., 0., 1.], [-80., 0., 200., 0.]])
    expected = expm(generator*.01) @ np.array(inputs[10:14])
    np.testing.assert_allclose(np.array(result[4:])[:, 0, :], expected, rtol=3e-10, atol=1e-15)
    from temsim.physics.vector_field_transport import vector_map_rk4
    mapped = vector_map_rk4(*inputs, z_mm=np.linspace(0., 10., 101), mapped_fields=())
    for a, b in zip(result[4:], mapped[4:]):
        np.testing.assert_allclose(a, b, rtol=2e-13, atol=1e-15)


def test_cuda_tensor_matches_cpu_when_available():
    from temsim.physics.compute_backend import cuda_capability
    if not cuda_capability().available:
        pytest.skip("No actual CUDA device; CPU fixtures do not qualify GPU")
    inputs = tensor_inputs()
    actual = rk.cuda_rk4(*inputs)
    expected = rk.vectorised_rk4(*inputs)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=3e-7, atol=1e-14)


def test_skew_has_cache_identity_and_legacy_zero_skew_compatibility():
    inputs = tensor_inputs()
    changed = (*inputs[:18], np.zeros_like(inputs[18]))
    assert plan_identity(inputs) != plan_identity(changed)
    assert plan_identity(changed) == plan_identity(inputs[:18])


def test_plan_and_snapshot_preserve_skew_without_default_retuning():
    state = default_state()
    original_percent = [lens.percent for lens in state.lenses]
    stig = state.stigmators[0]
    assert stig.field_model == "legacy_difference"
    stig.field_model = "normal_skew"
    stig.strength_y_percent = 10.
    z0, z1 = stig.z_mm - 15., stig.z_mm + 15.
    before = build_propagation_plan(state, z0, z1, maximum_step_mm=.5)
    assert np.max(np.abs(before.sxy_m2)) > 1.
    payload = state.to_dict()
    restored = type(state).from_dict(payload)
    restored_stig = next(s for s in restored.stigmators if s.key == stig.key)
    assert restored_stig.field_model == "normal_skew"
    assert restored_stig.channel_y_angle_deg == 45.
    stig.strength_y_percent = -10.
    after = build_propagation_plan(state, z0, z1, maximum_step_mm=.5)
    assert before.signature != after.signature
    assert propagation_plan_common_prefix_nodes(before, after) < len(before.z_mm)
    assert [lens.percent for lens in state.lenses] == original_percent


def test_manifest_rejects_dependent_channels_and_wrong_scan_clock():
    from temsim.module_manifest import _validate_alignment_structure
    with pytest.raises(ValueError, match="independent"):
        _validate_alignment_structure([dict(stigmator_structure="two_quadrupole_bases",
            channel_x_angle_deg=0., channel_y_angle_deg=90.)])
    with pytest.raises(ValueError, match="drive basis"):
        _validate_alignment_structure([dict(scan_structure="two_axial_xy_dipole_pairs",
            field_basis="column_xy", drive_clock="independent")])


@pytest.mark.parametrize("module", ["EnergyFilter", "NoEnergyFilter"])
def test_both_projector_assemblies_use_explicit_stigmator_basis(module):
    from temsim.module_manifest import part_data
    part = part_data(f"project_and_recording_system/{module}.toml", "diffraction_stigmator")
    assert (part["channel_x_angle_deg"], part["channel_y_angle_deg"]) == (0., 45.)
    assert part["coil_count"] == 8
    assert part["coil_topology_evidence"] == "generic_principle_not_oem_geometry"
    assert part["field_geometry_status"] == "coincident_effective_fields_not_measured_coil_map"


def test_old_profile_does_not_inherit_a_new_field_law():
    from temsim.profile_io import apply_profile_values
    state = default_state()
    stig = state.stigmators[0]
    stig.field_model = "normal_skew"
    assert apply_profile_values(state, {stig.key: {"strength_x_percent": 10., "strength_y_percent": 10.}}) == []
    assert stig.field_model == "legacy_difference"
    np.testing.assert_array_equal(stig.quadrupole_tensor_m2(stig.z_mm), np.zeros(3))
    assert apply_profile_values(state, {stig.key: {"field_model": "normal_skew"}}) == []
    assert stig.field_model == "normal_skew"
