"""One physical orientation across controls, persistence and detached recipes."""
from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("field", (
    "g_inv_nm", "excitation_error_inv_nm", "rocking_width_inv_nm",
    "diffuse_broadening_mrad", "diffraction_enabled",
))
def test_retired_qualitative_scattering_controls_are_rejected(field):
    from temsim.optics.column import default_state

    state = default_state()
    assert not hasattr(state.sample, field)
    payload = state.to_dict()
    payload["sample"][field] = 1.
    with pytest.raises(ValueError, match="Legacy specimen inputs"):
        type(state).from_dict(payload)
    setattr(state.sample, field, 1.)
    with pytest.raises(ValueError, match="Retired qualitative specimen"):
        state.to_dict()

from temsim.optics.model import Sample
from temsim.specimen.geometry import (
    IDENTITY_QUATERNION_WXYZ, quaternion_from_euler_xyz_deg, quaternion_to_matrix,
    quaternion_to_euler_xyz_deg,
    sample_orientation_quaternion, sample_orientation_euler_xyz_deg,
    set_sample_orientation, set_sample_orientation_euler_xyz_deg,
)
from temsim.design_experiments import parameter_value, replace_parameter, validate_runtime_sweep_path


def test_sample_persists_only_the_physical_quaternion():
    names = {field.name for field in fields(Sample)}
    assert "specimen_orientation_quaternion_wxyz" in names
    assert not any(f"specimen_rotation_{axis}_deg" in names for axis in "xyz")
    sample = Sample()
    set_sample_orientation_euler_xyz_deg(sample, (15., -12., 36.))
    assert sample_orientation_euler_xyz_deg(sample) == pytest.approx((15., -12., 36.))
    set_sample_orientation(sample, IDENTITY_QUATERNION_WXYZ)
    assert sample_orientation_quaternion(sample) == IDENTITY_QUATERNION_WXYZ
    assert not any(f"specimen_rotation_{axis}_deg" in vars(sample) for axis in "xyz")


def test_retired_euler_cannot_override_even_an_identity_quaternion():
    sample = SimpleNamespace(specimen_orientation_quaternion_wxyz=IDENTITY_QUATERNION_WXYZ,
                             specimen_rotation_x_deg=45.)
    with pytest.raises(ValueError, match="Retired sample Euler"):
        sample_orientation_quaternion(sample)


@pytest.mark.parametrize("axis", "xyz")
def test_derived_angle_control_changes_quaternion_without_adding_fields(axis):
    payload = {"sample": {"specimen_orientation_quaternion_wxyz": quaternion_from_euler_xyz_deg((10., 20., 30.))}}
    path = f"sample.orientation_euler_{axis}_deg"
    changed = replace_parameter(payload, path, 12.)
    assert validate_runtime_sweep_path(changed, path) == pytest.approx(12.)
    assert set(changed["sample"]) == {"specimen_orientation_quaternion_wxyz"}
    assert parameter_value(payload, path) == pytest.approx((10., 20., 30.)["xyz".index(axis)])
    assert changed != payload
    with pytest.raises(ValueError, match="Retired"):
        replace_parameter(payload, f"sample.specimen_rotation_{axis}_deg", 12.)


@pytest.mark.parametrize("angles", [
    (0., 0., 0.), (12., -25., 38.), (17., 90., 3.), (17., -90., 3.),
    (17., 89.999999, 3.), (17., -89.999999, 3.),
    (175., 90.000001, -173.), (-178., -90.000001, 176.),
    (120., 165., -140.), (-160., -140., 179.),
])
def test_angle_conversions_preserve_the_rotation_matrix(angles):
    sample = Sample()
    set_sample_orientation_euler_xyz_deg(sample, angles)
    quaternion = sample_orientation_quaternion(sample)
    roundtrip = quaternion_from_euler_xyz_deg(sample_orientation_euler_xyz_deg(sample))
    np.testing.assert_allclose(quaternion_to_matrix(roundtrip), quaternion_to_matrix(quaternion), rtol=0., atol=2e-15)


def test_random_quaternions_and_sign_representations_preserve_rotation():
    random = np.random.default_rng(20260920)
    quaternions = random.normal(size=(1000, 4))
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    for quaternion in quaternions:
        original = quaternion_to_matrix(quaternion)
        for representation in (quaternion, -quaternion):
            angles = quaternion_to_euler_xyz_deg(representation)
            restored = quaternion_to_matrix(quaternion_from_euler_xyz_deg(angles))
            np.testing.assert_allclose(restored, original, rtol=0., atol=2e-15)


@pytest.mark.parametrize("pitch", [-180.000001, -180., -179.999999,
    -90.000001, -90., -89.999999, 89.999999, 90., 90.000001,
    179.999999, 180., 180.000001])
def test_euler_chart_boundary_preserves_rotation_for_both_quaternion_signs(pitch):
    quaternion = np.asarray(quaternion_from_euler_xyz_deg((137., pitch, -155.)))
    original = quaternion_to_matrix(quaternion)
    for representation in (quaternion, -quaternion):
        restored = quaternion_to_matrix(quaternion_from_euler_xyz_deg(
            quaternion_to_euler_xyz_deg(representation)))
        np.testing.assert_allclose(restored, original, rtol=0., atol=2e-15)


def test_current_state_round_trip_rejects_retired_euler_payload():
    from temsim.optics.column import default_state
    state = default_state()
    set_sample_orientation_euler_xyz_deg(state.sample, (12., -25., 38.))
    payload = state.to_dict()
    restored = type(state).from_dict(payload)
    assert sample_orientation_quaternion(restored.sample) == sample_orientation_quaternion(state.sample)
    for axis in "xyz":
        with pytest.raises(ValueError, match="Legacy specimen"):
            type(state).from_dict({**payload, "sample": {**payload["sample"], f"specimen_rotation_{axis}_deg": 0.}})


def test_detached_snapshot_recipe_applies_the_derived_angle_to_real_orientation():
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.column import default_state
    from temsim.design_explorer import capture_design_snapshot, HighAccuracyRequest
    from temsim.design_experiments import recipe_from_snapshot, SweepAxis, plan_parameter_sweep
    from temsim.design_sweep_execution import rebuild_recipe_state, _assert_sweep_coordinates_applied
    state = default_state()
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    catalog.apply(state, selection)
    before = sample_orientation_quaternion(state.sample)
    snapshot = capture_design_snapshot(state, selection, slot="A", request=HighAccuracyRequest(9, 2.5))
    recipe = recipe_from_snapshot(snapshot, name="Physical orientation control")
    sweep = plan_parameter_sweep(recipe, (SweepAxis("sample.orientation_euler_x_deg", (12., 18.), "deg"),))
    for point in sweep.points:
        rebuilt, _ = rebuild_recipe_state(recipe, state_payload=point.state_payload, catalog=catalog)
        _assert_sweep_coordinates_applied(rebuilt, point, sweep.axes)
        assert sample_orientation_euler_xyz_deg(rebuilt.sample)[0] == pytest.approx(point.coordinates[sweep.axes[0].path])
    assert sample_orientation_quaternion(state.sample) == before
