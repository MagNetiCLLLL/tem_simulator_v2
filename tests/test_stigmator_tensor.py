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


def test_current_default_equal_inputs_produce_two_independent_components():
    stig = Stigmator("test", "test", 0., strength_x_percent=10., strength_y_percent=10.)
    assert stig.field_model == "normal_skew"
    assert stig.quadrupole_tensor_m2(0.) == pytest.approx((30., -30., 30.))
    stig.strength_y_percent = 0.
    assert stig.quadrupole_tensor_m2(0.) == pytest.approx((30., -30., 0.))


@pytest.mark.parametrize("model", ["legacy_difference", "", None, "unknown"])
def test_unsupported_field_law_is_rejected_without_remapping(model):
    stig = Stigmator("test", "test", 0., field_model=model)
    with pytest.raises(ValueError, match="normal_skew"):
        stig.quadrupole_tensor_m2(0.)
    assert stig.field_model == model


def test_column_requires_full_tensor_instead_of_inventing_a_rank_one_law():
    from temsim.physics.core import multipole_focusing_fields, skew_quadrupole_field
    incomplete = SimpleNamespace(enabled=True, z_mm=0., length_mm=8.,
                                 max_strength_m2=300., strength_x_percent=10., strength_y_percent=10.)
    state = SimpleNamespace(stigmators=[incomplete], corrector_elements=[])
    for function in (multipole_focusing_fields, skew_quadrupole_field):
        with pytest.raises(AttributeError, match="quadrupole_tensor_m2"):
            function(np.array([0.]), state)


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


def test_skew_has_cache_identity_and_zero_skew_is_explicit():
    inputs = tensor_inputs()
    changed = (*inputs[:18], np.zeros_like(inputs[18]))
    assert plan_identity(inputs) != plan_identity(changed)
    assert plan_identity(changed) == plan_identity((*inputs[:18], np.zeros_like(inputs[18])))


@pytest.mark.parametrize("timed", [False, True])
def test_device_identity_rejects_old_input_lengths_without_skew(timed):
    inputs = tensor_inputs()[:18]
    if timed:
        inputs = (*inputs, np.zeros(3), np.ones(3))
    with pytest.raises(ValueError, match="nineteen arrays"):
        plan_identity(inputs)


def test_plan_and_snapshot_preserve_skew_without_default_retuning():
    state = default_state()
    original_percent = [lens.percent for lens in state.lenses]
    stig = state.stigmators[0]
    assert stig.field_model == "normal_skew"
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


def test_partial_profile_preserves_current_stigmator_model():
    from temsim.profile_io import apply_profile_values
    state = default_state()
    stig = state.stigmators[0]
    stig.field_model = "normal_skew"
    apply_profile_values(state, {stig.key: {"strength_x_percent": 10., "strength_y_percent": 10.}})
    assert stig.field_model == "normal_skew"
    assert np.linalg.norm(stig.quadrupole_tensor_m2(stig.z_mm)) > 0


@pytest.mark.parametrize("kind", ["condenser", "objective", "diffraction"])
def test_current_stigmator_loaders_require_explicit_physical_model(kind):
    from dataclasses import asdict
    from importlib import import_module
    module = import_module(f"temsim.optics.{kind}_stigmator")
    create = getattr(module, f"create_{kind}_stigmator")
    restore = getattr(module, f"{kind}_stigmator_from_dict")
    record = asdict(create())
    restored = restore(record)
    assert restored.field_model == "normal_skew"
    for value in ("legacy_difference", None):
        invalid = dict(record)
        if value is None:
            invalid.pop("field_model")
        else:
            invalid["field_model"] = value
        with pytest.raises(ValueError, match="normal_skew"):
            restore(invalid)


def test_current_field_model_is_serialized_but_not_an_editable_mode():
    from temsim.runtime_parameters import RuntimeTarget, editable_parameters
    state = default_state()
    stig = state.stigmators[0]
    target = RuntimeTarget(stig.key, stig.name, stig)
    names = {item.name for item in editable_parameters(target)}
    assert "strength_x_percent" in names and "strength_y_percent" in names
    assert "field_model" not in names
    record = next(row for row in state.to_dict()["stigmators"] if row["key"] == stig.key)
    assert record["field_model"] == "normal_skew"
