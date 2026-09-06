"""Exact bounded analytical reference-plane caching across State snapshots."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

import temsim.optics.objective_lens as objective_module
from temsim.optics.column import default_state
from temsim.optics.objective_lens import ObjectiveLensComponent


@pytest.fixture
def state():
    value = default_state()
    objective_module._PLANE_ZERO_CACHE.clear()
    return value


def arguments(state):
    return (state.beam_voltage_kv, state.sample.z_mm + state.sample.thickness_nm * .5e-6,
            state.sample.z_mm + state.objective_lens.assembly_length_mm / 2,
            (0, 0), .02)


@pytest.mark.parametrize("index,step", (((0, 0), .02), ((0, 1), .1)))
def test_cached_root_matches_original_solver_exactly(state, index, step):
    lens = state.objective_lens
    values = list(arguments(state))
    values[3:] = (index, step)
    expected = lens._solve_first_matrix_zero(*values)
    actual = lens._first_matrix_zero(*values)
    assert actual == expected
    assert lens._first_matrix_zero(*values) == expected


def test_equivalent_independent_lenses_share_scalar_roots(state, monkeypatch):
    calls = []
    monkeypatch.setattr(ObjectiveLensComponent, "_solve_first_matrix_zero",
                        lambda self, *args: calls.append(args) or 42.)
    lens = state.objective_lens
    other = replace(lens, upper_gaussian=deepcopy(lens.upper_gaussian),
                    lower_gaussian=deepcopy(lens.lower_gaussian))
    assert lens._first_matrix_zero(*arguments(state)) == 42.
    assert other._first_matrix_zero(*arguments(state)) == 42.
    assert len(calls) == 1


@pytest.mark.parametrize("name", (
    "enabled", "percent", "polarity", "upper_b0_t", "lower_b0_t",
    "upper_a_mm", "lower_a_mm", "upper_field_center_z_mm", "lower_field_center_z_mm",
    "upper_gaussian.amplitude", "upper_gaussian.offset", "upper_gaussian.sigma",
    "lower_gaussian.amplitude", "lower_gaussian.offset", "lower_gaussian.sigma",
))
def test_every_consumed_field_parameter_invalidates_root(state, monkeypatch, name):
    calls = []
    monkeypatch.setattr(ObjectiveLensComponent, "_solve_first_matrix_zero",
                        lambda self, *args: calls.append(args) or None)
    lens = state.objective_lens
    args = arguments(state)
    lens._first_matrix_zero(*args)
    lens._first_matrix_zero(*args)
    target = lens
    if "." in name:
        group, name = name.split(".")
        target = getattr(lens, group)[0]
    previous = getattr(target, name)
    updated = not previous if name == "enabled" else -previous if name == "polarity" else np.nextafter(float(previous), np.inf)
    setattr(target, name, updated)
    lens._first_matrix_zero(*args)
    assert len(calls) == 2


@pytest.mark.parametrize("argument", range(5))
def test_voltage_boundary_index_and_resolution_are_exact_keys(state, monkeypatch, argument):
    calls = []
    monkeypatch.setattr(ObjectiveLensComponent, "_solve_first_matrix_zero",
                        lambda self, *args: calls.append(args) or None)
    args = list(arguments(state))
    state.objective_lens._first_matrix_zero(*args)
    args[argument] = (0, 1) if argument == 3 else np.nextafter(args[argument], np.inf)
    state.objective_lens._first_matrix_zero(*args)
    assert len(calls) == 2


def test_scalar_root_cache_is_bounded_and_retains_no_lens_objects(state, monkeypatch):
    monkeypatch.setattr(ObjectiveLensComponent, "_solve_first_matrix_zero", lambda *_args: 42.)
    monkeypatch.setattr(objective_module, "_PLANE_ZERO_CACHE_LIMIT", 3)
    lens = state.objective_lens
    for value in range(5):
        lens.percent = value
        lens._first_matrix_zero(*arguments(state))
    assert len(objective_module._PLANE_ZERO_CACHE) == 3
    assert set(objective_module._PLANE_ZERO_CACHE.values()) == {42.}
    assert all(lens is not item for key in objective_module._PLANE_ZERO_CACHE for item in key)


def test_custom_field_override_bypasses_analytical_cache(state, monkeypatch):
    calls = []
    lens = state.objective_lens
    monkeypatch.setattr(ObjectiveLensComponent, "_solve_first_matrix_zero",
                        lambda self, *args: calls.append(args) or None)
    lens.magnetic_field_t = lambda z: np.zeros_like(z)
    lens._first_matrix_zero(*arguments(state))
    lens._first_matrix_zero(*arguments(state))
    assert len(calls) == 2
    assert not objective_module._PLANE_ZERO_CACHE
