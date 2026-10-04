"""Identity checks for the actual curved-tip field; not field convergence."""
from copy import copy

import numpy as np
import pytest

from temsim.diagnostic_field_identity import electric_field_identity


@pytest.fixture(scope="module")
def grounded():
    from temsim.optics.column import default_state
    from temsim.optics.electron_gun.tip_surface import load_tip_surface_reference
    from temsim.physics.grounded_tip_field import grounded_field
    gun = default_state().electron_gun
    gun.emitter.surface_model = load_tip_surface_reference()
    return gun, grounded_field(gun)


def detached_interpolator(field):
    changed = copy(field)
    changed._fem = copy(field._fem)
    changed._regular = copy(field._regular)
    changed._regular.field = changed._fem
    return changed


def test_cached_curved_field_has_stable_content_identity(grounded):
    from temsim.physics.grounded_tip_field import grounded_field
    gun, field = grounded
    first = electric_field_identity(field)
    second = electric_field_identity(grounded_field(gun))
    assert first.status == "known"
    assert first == second
    assert first.physical_id and first.numerical_id and first.request_id


def test_curved_field_identity_binds_executed_gradients_and_axis_interpolation(grounded):
    _, field = grounded
    before = electric_field_identity(field)
    changed = detached_interpolator(field)
    gradient = field._fem.gradient.copy()
    gradient.flat[0] = np.nextafter(gradient.flat[0], np.inf)
    gradient.setflags(write=False)
    changed._fem.gradient = gradient
    after = electric_field_identity(changed)
    assert after.status == "known"
    assert after.physical_id == before.physical_id
    assert after.numerical_id != before.numerical_id
    changed = detached_interpolator(field)
    changed._regular.fraction *= .5
    assert electric_field_identity(changed).numerical_id != before.numerical_id
    changed = copy(field)
    changed.report = {**field.report, "axis_core_fraction": field.report["axis_core_fraction"] * .5}
    assert electric_field_identity(changed).numerical_id != before.numerical_id
    changed = detached_interpolator(field)
    changed._regular.interpolate = lambda positions: None
    assert electric_field_identity(changed).status == "unknown"


def test_curved_field_identity_rejects_mutable_or_unmatched_geometry(grounded):
    from dataclasses import replace
    _, field = grounded
    changed = detached_interpolator(field)
    changed._fem.gradient = field._fem.gradient.copy()
    assert electric_field_identity(changed).status == "unknown"
    changed = copy(field)
    changed.geometry = replace(field.geometry, apex_radius_nm=field.geometry.apex_radius_nm * 1.1)
    assert electric_field_identity(changed).status == "unknown"
