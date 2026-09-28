"""Provenance and cache admission, using tiny arrays without a field solve.

The arrays below are identity fixtures, not an electrostatic accuracy study.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.diagnostic_field_identity import (
    can_reuse_electric_field, electric_field_identity,
    electric_request_physical_id, field_array_digest, identity_digest,
)
from temsim.physics.closed_gun_field import ClosedGunField, closed_field_request
from temsim.physics.continuous_gun_field import ContinuousGunField, continuous_field_request
from temsim.physics.planar_gun_field import PlanarGunField, request_digest


@pytest.fixture
def gun():
    from temsim.optics.column import default_state
    return default_state().electron_gun


def tiny_field(request, *, voltage_offset=0.0):
    domain = request["domain"]
    r = np.linspace(0., domain["outer_radius_m"], 3)
    z = np.linspace(domain["entrance_m"], domain["exit_m"], 4)
    voltage = np.broadcast_to(z * 1e5 + voltage_offset, (3, 4))
    return ClosedGunField(request, r, z, voltage,
                          {"request_sha256": request_digest(request)})


def test_namespaces_are_explicit_and_input_objects_are_not_hashed():
    assert identity_digest("field", {"a": 1, "b": (2, 3)}) == identity_digest("field", {"b": [2, 3], "a": 1})
    assert identity_digest("field", [1]) != identity_digest("execution", [1])
    for value in (float("nan"), object(), SimpleNamespace(x=1)):
        with pytest.raises((TypeError, ValueError)):
            identity_digest("field", value)
    with pytest.raises(ValueError):
        identity_digest("", {})


def test_array_digest_binds_values_shape_and_precision():
    values = np.array([1., 2.])
    assert field_array_digest(values) == field_array_digest(values.copy())
    assert field_array_digest(values) != field_array_digest(values.reshape(1, 2))
    assert field_array_digest(values) != field_array_digest(values.astype(np.float32))
    assert field_array_digest(values) != field_array_digest(values + 1)
    for value in (np.array([np.nan]), np.array([object()]), np.array(["hidden object"])):
        with pytest.raises(ValueError):
            field_array_digest(value)


def test_full_request_and_actual_arrays_both_bind_numerical_identity(gun):
    request = closed_field_request(gun)
    first, second = tiny_field(request), tiny_field(request, voltage_offset=.01)
    a, b = electric_field_identity(first), electric_field_identity(second)
    assert a.status == b.status == "known"
    assert a.physical_id == b.physical_id
    assert a.request_id == b.request_id == request_digest(request)
    assert a.numerical_id != b.numerical_id
    assert a.support_r_z_m == (first.r[-1], first.z[0], first.z[-1])
    assert a.to_dict()["numerical_id"] == a.numerical_id


@pytest.mark.parametrize("name,value", [
    ("ray_count", 17), ("emission_current_na", 2.), ("emission_energy_ev", 1.),
    ("angular_rms_mrad", .5), ("angular_cutoff_mrad", 2.), ("name", "Emission tip A"),
])
def test_charge_free_field_excludes_unused_source_and_display_inputs(gun, name, value):
    before = closed_field_request(gun)
    setattr(gun.emitter, name, value)
    after = closed_field_request(gun)
    assert request_digest(before) == request_digest(after)
    assert electric_request_physical_id(before) == electric_request_physical_id(after)


@pytest.mark.parametrize("component,name,value", [
    ("extractor", "voltage_kv", 4.1),
    ("extractor", "mechanical_center_from_tip_mm", .0001),
    ("accelerator", "high_tension_kv", 200.),
])
def test_actual_voltage_and_electrode_geometry_change_physical_identity(gun, component, name, value):
    before = closed_field_request(gun)
    if name == "mechanical_center_from_tip_mm":
        value += getattr(getattr(gun, component), name)
    setattr(getattr(gun, component), name, value)
    after = closed_field_request(gun)
    assert electric_request_physical_id(before) != electric_request_physical_id(after)
    assert electric_field_identity(tiny_field(before)).numerical_id != electric_field_identity(tiny_field(after)).numerical_id


def test_complete_liner_and_potential_reference_are_physical_inputs(gun):
    request = closed_field_request(gun)
    changed = deepcopy(request)
    changed["grounded_liner"][-1]["stop_m"] += .01  # Beyond either numerical cut.
    assert electric_request_physical_id(changed) != electric_request_physical_id(request)
    changed = deepcopy(request)
    changed["potential_reference"] = "ground_referenced_volts"
    assert electric_request_physical_id(changed) != electric_request_physical_id(request)


def test_cut_and_mesh_changes_preserve_physics_but_not_numerical_identity(gun):
    original = closed_field_request(gun, exit_extension_mm=100., cells_per_bore=4)
    for changed in (closed_field_request(gun, exit_extension_mm=200., cells_per_bore=4),
                    closed_field_request(gun, exit_extension_mm=100., cells_per_bore=8)):
        a, b = electric_field_identity(tiny_field(original)), electric_field_identity(tiny_field(changed))
        assert a.status == b.status == "known"
        assert a.physical_id == b.physical_id
        assert a.numerical_id != b.numerical_id
        assert a.request_id != b.request_id


def test_curved_geometry_and_refinement_have_distinct_dependencies(gun):
    flat = closed_field_request(gun)
    gun.emitter.curvature_nm_inv = .01
    first = continuous_field_request(gun, apex_cells_per_radius=16)
    refined = continuous_field_request(gun, apex_cells_per_radius=32)
    assert electric_request_physical_id(first) != electric_request_physical_id(flat)
    assert electric_request_physical_id(first) == electric_request_physical_id(refined)
    assert request_digest(first) != request_digest(refined)
    gun.emitter.curvature_nm_inv = .02
    second = continuous_field_request(gun)
    assert electric_request_physical_id(first) != electric_request_physical_id(second)


def test_larger_immutable_covering_field_can_be_reused_without_changing_identity(gun):
    smaller = closed_field_request(gun, exit_extension_mm=100.)
    larger = closed_field_request(gun, exit_extension_mm=200.)
    field = tiny_field(larger)
    before = electric_field_identity(field)
    assert can_reuse_electric_field(field, smaller, (0., smaller["domain"]["exit_m"]))
    assert electric_field_identity(field) == before
    assert field.request == larger  # Never relabel the larger solve as the requested smaller one.
    assert not can_reuse_electric_field(tiny_field(smaller), larger, (0., larger["domain"]["exit_m"]))
    assert not can_reuse_electric_field(field, smaller, (0., larger["domain"]["exit_m"] + .001))
    assert not can_reuse_electric_field(field, smaller, (0., .01), radial_limit_m=field.r[-1] * 2)


@pytest.mark.parametrize("kind", ["mesh", "solver", "reference", "boundary", "voltage"])
def test_matching_geometry_or_label_is_insufficient_for_reuse(gun, kind):
    request = closed_field_request(gun)
    changed = deepcopy(request)
    if kind == "mesh":
        changed["numerics"]["cells_per_bore"] *= 2
    elif kind == "solver":
        changed["implementation_sha256"]["closed_gun_field.py"] = "different implementation"
    elif kind == "reference":
        changed["potential_reference"] = "another reference"
    elif kind == "boundary":
        changed["boundary_conditions"]["radial_outer"] = "different boundary"
    else:
        changed["rings"][0]["potential_rise_v"] += 1.
    assert not can_reuse_electric_field(tiny_field(request), changed, (0., .01))


def test_mutable_or_stale_field_products_remain_unknown_and_not_reusable(gun):
    request = closed_field_request(gun)
    for change in (lambda field: field.voltage.setflags(write=True),
                   lambda field: field.report.pop("request_sha256"),
                   lambda field: field.report.update(request_sha256="stale"),
                   lambda field: field.request["domain"].update(exit_m=999.)):
        field = tiny_field(request)
        change(field)
        assert electric_field_identity(field).status == "unknown"
        assert not can_reuse_electric_field(field, request, (0., .01))


def test_truncated_or_altered_liner_cannot_masquerade_as_same_physics(gun):
    request = closed_field_request(gun)
    damaged = deepcopy(request)
    row = next(row for row in damaged["rings"] if row["key"].startswith("grounded_outlet"))
    row["potential_rise_v"] += 1.
    assert electric_request_physical_id(damaged) is None
    assert electric_field_identity(tiny_field(damaged)).status == "unknown"


def test_unsupported_or_historical_provider_does_not_gain_an_identity():
    historical = PlanarGunField({}, [0., 1.], [0., 1.], [[0., 1.], [0., 1.]])
    for provider in (historical, SimpleNamespace(request={}), object()):
        identity = electric_field_identity(provider)
        assert identity.status == "unknown"
        assert identity.physical_id is identity.numerical_id is identity.request_id is None
        assert identity.reason


def tiny_continuous_field(gun):
    # No solve: construct the explicit tiny array contract consumed by this
    # provider. This checks provenance only, not curved-field accuracy.
    gun.emitter.curvature_nm_inv = .01
    request = continuous_field_request(gun)
    basic = tiny_field(request)
    from temsim.physics.axisymmetric_cut_field import AxisymmetricCutField
    fem = object.__new__(AxisymmetricCutField)
    fem.r, fem.z, fem.nodal_voltage = basic.r, basic.z, basic.voltage
    for name in ("lookup", "cut_cells", "origin", "inverse", "gradient", "phi0", "boundary_s", "boundary_z"):
        array = np.zeros((2, 2))
        array.setflags(write=False)
        setattr(fem, name, array)
    return ContinuousGunField(request, fem, {})


def test_continuous_numerical_identity_binds_all_interpolation_arrays(gun):
    field = tiny_continuous_field(gun)
    before = electric_field_identity(field)
    assert before.status == "known"
    changed = field._regular.slope.copy() + 1.
    changed.setflags(write=False)
    field._regular.slope = changed
    after = electric_field_identity(field)
    assert after.physical_id == before.physical_id
    assert after.numerical_id != before.numerical_id


@pytest.mark.parametrize("name", ["interpolate", "tip_material_mask", "field_at_global_positions_v_per_m"])
def test_instance_field_method_override_does_not_inherit_known_identity(gun, name):
    field = tiny_field(closed_field_request(gun))
    setattr(field, name, lambda points: None)
    assert electric_field_identity(field).status == "unknown"
    assert not can_reuse_electric_field(field, field.request, (0., .01))


def test_class_field_method_override_does_not_inherit_known_identity(gun, monkeypatch):
    field = tiny_field(closed_field_request(gun))
    monkeypatch.setattr(ClosedGunField, "interpolate", lambda self, points: None)
    assert electric_field_identity(field).status == "unknown"


@pytest.mark.parametrize("owner", ["_regular", "_fem"])
def test_custom_curved_interpolation_remains_unknown(gun, owner):
    field = tiny_continuous_field(gun)
    getattr(field, owner).interpolate = lambda points: None
    assert electric_field_identity(field).status == "unknown"


def test_old_solved_request_is_also_bound_to_current_interpolation_implementation(gun, monkeypatch):
    field = tiny_field(closed_field_request(gun))
    before = electric_field_identity(field)
    monkeypatch.setattr("temsim.diagnostic_field_identity._current_interpolation_identity", lambda cls: "updated evaluator")
    after = electric_field_identity(field)
    assert before.physical_id == after.physical_id and before.request_id == after.request_id
    assert before.numerical_id != after.numerical_id


def test_curved_interpolation_backend_changes_numerical_not_physical_identity(gun):
    field = tiny_continuous_field(gun)
    before = electric_field_identity(field)
    field._regular.compiled = False
    after = electric_field_identity(field)
    assert before.physical_id == after.physical_id and before.request_id == after.request_id
    assert before.numerical_id != after.numerical_id
