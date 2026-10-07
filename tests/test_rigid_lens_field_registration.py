"""Small CPU-only checks of rigid field placement and map reuse."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.lens_pose import lens_pose_registration
from temsim.component_keys import CONDENSER_LENS_KEYS, IMAGE_CORRECTOR_LENS_KEYS
from temsim.optics.column import default_state
from temsim.physics.lens_field_provider import (
    CoordinateRegistration, FieldMapError, FieldMapProvenance, FrozenAnalyticField,
    MagneticFieldMap, MappedLensFieldProvider, active_mapped_providers,
    active_vector_providers, bind_imported_lens_field_map,
    freeze_vector_provider, lens_geometry_binding, resolve_runtime_lens_field_provider,
)


KEY = "condenser_lens_1"
ROUND_LENS_KEYS = (*CONDENSER_LENS_KEYS, "adapter_lens", "probe_tl22_lens", "probe_tl21_lens",
                   "probe_tl12_lens", "mini_condenser", "objective_lens", *IMAGE_CORRECTOR_LENS_KEYS,
                   "diffraction_lens", "intermediate_lens", "projector_lens_1", "projector_lens_2")


@pytest.fixture(scope="module")
def all_analytic_lens_types():
    # This test only queries existing field laws; no trajectory/FEM solve.
    state = default_state()
    state.simulation_mode = "analytical"
    assert {lens.key for lens in state.lenses} == set(ROUND_LENS_KEYS)
    return state


@pytest.mark.parametrize("lens_key", ROUND_LENS_KEYS)
def test_frozen_gaussian_reproduces_each_native_lens_family(all_analytic_lens_types, lens_key):
    from temsim.lens_pose import rotation_matrix_mrad
    state = all_analytic_lens_types
    lens = next(item for item in state.lenses if item.key == lens_key)
    native = state.condenser_system[lens_key] if lens_key in CONDENSER_LENS_KEYS else lens
    lens.enabled = True
    lens.percent = lens.max_percent * .43
    provider = resolve_runtime_lens_field_provider(state, lens_key, native)
    support = provider.native_field_support_mm()
    points = np.column_stack((np.linspace(-2e-5, 2e-5, 11), np.full(11, 1e-5),
                              np.linspace(support[0], support[1], 11)*1e-3))
    rotation = rotation_matrix_mrad((2., -3., 1.))
    pivot = np.array((0., 0., lens.z_mm*1e-3))
    translation = pivot-rotation@pivot+np.array((2e-5, -3e-5, 4e-5))
    registration = CoordinateRegistration(tuple(translation), tuple(map(tuple, rotation)))
    moved_points = points@rotation.T+translation
    posed = replace(provider, registration=registration)
    for polarity in (1, -1):
        lens.polarity = polarity
        frozen = freeze_vector_provider(posed)
        assert np.max(np.abs(native.magnetic_field_t(points[:, 2]*1e3))) > 0.
        assert frozen.field_at_global_positions_t(moved_points) == pytest.approx(
            posed.field_at_global_positions_t(moved_points), rel=2e-7, abs=2e-10)
        # Compare the local longitudinal component directly with the native
        # implementation, including objective upper/lower and normalized lenses.
        local_field = frozen.field_at_global_positions_t(moved_points)@rotation
        assert local_field[:, 2] == pytest.approx(native.magnetic_field_t(points[:, 2]*1e3), rel=2e-10, abs=2e-12)
    lens.enabled = False
    assert not np.any(freeze_vector_provider(posed).field_at_global_positions_t(moved_points))


def test_frozen_analytic_slope_law_retains_paraxial_limit_under_tiny_pose():
    state, native = _single_lens_state()
    provider = resolve_runtime_lens_field_provider(state, KEY, native)
    frozen = freeze_vector_provider(provider)
    positions = np.array(((1e-4, -2e-4, native.lens.z_mm*1e-3),
                          (-2e-4, 1e-4, (native.lens.z_mm+1.)*1e-3)))
    slopes = np.array(((.08, -.05), (-.1, .04)))
    charge_over_p = np.array((-1000., -900.))
    field = frozen.field_at_global_positions_t(positions)
    expected = charge_over_p[:, None]*np.column_stack((slopes[:, 1]*field[:, 2]-field[:, 1],
                                                       field[:, 0]-slopes[:, 0]*field[:, 2]))
    assert frozen.slope_derivative(positions, slopes, charge_over_p) == pytest.approx(expected)
    infinitesimal = replace(frozen, registration=CoordinateRegistration((1e-16, 0., 0.)))
    assert infinitesimal.slope_derivative(positions, slopes, charge_over_p) == pytest.approx(expected, abs=2e-9, rel=2e-10)


def test_analytic_slope_law_rejects_backwards_local_axis():
    state, native = _single_lens_state()
    provider = resolve_runtime_lens_field_provider(state, KEY, native)
    reversed_axis = CoordinateRegistration(rotation_local_to_global=((-1., 0., 0.), (0., 1., 0.), (0., 0., -1.)))
    frozen = freeze_vector_provider(replace(provider, registration=reversed_axis))
    with pytest.raises(FieldMapError, match="forward local-axis"):
        frozen.slope_derivative(np.array(((0., 0., native.lens.z_mm*1e-3),)), np.zeros((1, 2)), -1000.)


def _edit_pose(state, **values):
    state._resolved_assembly = replace(
        state._resolved_assembly,
        parts=tuple(replace(part, data={**part.data, **values}) if part.key == KEY else part
                    for part in state._resolved_assembly.parts),
    )


def _single_lens_state():
    state = default_state()
    state.simulation_mode = "analytical"
    for lens in state.lenses:
        lens.enabled = lens.key == KEY
    return state, state.condenser_system[KEY]


def test_analytic_lens_pose_transforms_query_and_field_together():
    state, native = _single_lens_state()
    baseline = resolve_runtime_lens_field_provider(state, KEY, native)
    points = np.array(((1e-5, -2e-5, native.lens.z_mm*1e-3),
                       (-2e-5, 3e-5, (native.lens.z_mm+2)*1e-3)))
    reference = baseline.field_at_global_positions_t(points)
    _edit_pose(state, offset_x_mm=.1, offset_y_mm=-.05,
               rotation_x_mrad=2., rotation_y_mrad=-3., rotation_z_mrad=4.)
    posed = resolve_runtime_lens_field_provider(state, KEY, native)
    registration = lens_pose_registration(state, KEY)
    moved_points = points @ registration.rotation_array.T + registration.origin_array_m
    expected = reference @ registration.rotation_array.T
    assert posed is not baseline
    assert posed.binding == baseline.binding
    assert posed.field_at_global_positions_t(moved_points) == pytest.approx(expected, rel=1e-8, abs=1e-11)
    assert resolve_runtime_lens_field_provider(state, KEY, native) is posed
    assert active_mapped_providers(state) == ()
    assert active_vector_providers(state) == (posed,)


def test_frozen_analytic_pose_and_excitation_do_not_follow_later_edits():
    state, native = _single_lens_state()
    _edit_pose(state, offset_x_mm=.02, rotation_y_mrad=1.)
    provider = resolve_runtime_lens_field_provider(state, KEY, native)
    frozen = freeze_vector_provider(provider)
    assert isinstance(frozen, FrozenAnalyticField)
    local = np.array(((1e-5, 0., (native.lens.z_mm+1.)*1e-3),))
    points = local @ provider.registration.rotation_array.T + provider.registration.origin_array_m
    before = frozen.field_at_global_positions_t(points)
    assert before == pytest.approx(provider.field_at_global_positions_t(points), rel=2e-8, abs=1e-11)
    fingerprint = frozen.fingerprint
    native.lens.percent *= .4
    _edit_pose(state, offset_x_mm=.04)
    assert frozen.field_at_global_positions_t(points) == pytest.approx(before, rel=0, abs=0)
    assert frozen.fingerprint == fingerprint
    assert freeze_vector_provider(resolve_runtime_lens_field_provider(state, KEY, native)).fingerprint != fingerprint


def test_zero_pose_keeps_axial_fast_path_and_coil_dimensions_do_not_scale_analytic_field():
    state, native = _single_lens_state()
    baseline = resolve_runtime_lens_field_provider(state, KEY, native)
    values = baseline.magnetic_field_t(np.array((native.lens.z_mm, native.lens.z_mm+1)))
    assert active_vector_providers(state) == ()
    _edit_pose(state, mechanical_outer_diameter_mm=170.)
    changed = resolve_runtime_lens_field_provider(state, KEY, native)
    assert changed.magnetic_field_t(np.array((native.lens.z_mm, native.lens.z_mm+1))) == pytest.approx(values)
    assert active_vector_providers(state) == ()


def test_mapped_lens_pose_composes_existing_registration_without_rebinding():
    state, native = _single_lens_state()
    state.simulation_mode = "custom"
    binding = lens_geometry_binding(state, KEY, native)
    r = np.array((0., .001, .002))
    z = np.array((-.002, 0., .002))
    field_map = MagneticFieldMap(
        "axisymmetric_rz", (r, z), (np.zeros((3, 3)), np.full((3, 3), .3)),
        CoordinateRegistration((1e-4, -2e-4, native.lens.z_mm*1e-3)),
        binding.geometry_fingerprint, 100., 1,
        FieldMapProvenance("measured", "captured.npz", "a"*64, "synthetic rigid-pose regression"),
    )
    bind_imported_lens_field_map(state, KEY, field_map, native_provider=native)
    original = resolve_runtime_lens_field_provider(state, KEY, native)
    points = np.array((field_map.registration.origin_global_m,))
    reference = original.field_at_global_positions_t(points)
    _edit_pose(state, offset_x_mm=.05, rotation_x_mrad=3., rotation_y_mrad=2.)
    posed = resolve_runtime_lens_field_provider(state, KEY, native)
    registration = lens_pose_registration(state, KEY)
    moved_points = points @ registration.rotation_array.T + registration.origin_array_m
    assert isinstance(posed, MappedLensFieldProvider)
    assert posed.binding == binding
    assert state._lens_field_map_bindings[KEY] is field_map
    assert posed.field_map is not field_map
    assert np.shares_memory(posed.field_map.components_t[1], field_map.components_t[1])
    assert posed.field_at_global_positions_t(moved_points) == pytest.approx(reference @ registration.rotation_array.T)
    assert resolve_runtime_lens_field_provider(state, KEY, native) is posed
    assert freeze_vector_provider(posed).field_support_mm == posed.field_support_mm()


def test_posed_scene_bounds_identity_and_python_field_follow_lens():
    from temsim.magnetic_field_scene import prepare_magnetic_scene
    from temsim.physics.compiled_magnetic_field import prepare_compiled_magnetic_sources
    state, native = _single_lens_state()
    original = prepare_magnetic_scene(state, z_limits_mm=(native.lens.z_mm-2, native.lens.z_mm+2))
    _edit_pose(state, offset_x_mm=.05, rotation_y_mrad=1.)
    scene = prepare_magnetic_scene(state, z_limits_mm=(native.lens.z_mm-2, native.lens.z_mm+2))
    source = next(source for source in scene._sources if source.key == KEY)
    local = np.array(((0., 0., native.lens.z_mm*1e-3),))
    registration = lens_pose_registration(state, KEY)
    moved = local @ registration.rotation_array.T + registration.origin_array_m
    assert source.contains(moved)[0]
    assert source.bounds_m[0, 0] != next(s for s in original._sources if s.key == KEY).bounds_m[0, 0]
    assert source.identity.physical_identity is not None
    assert source.identity.physical_identity != next(s for s in original._sources if s.key == KEY).identity.physical_identity
    assert prepare_compiled_magnetic_sources((source,)) is None
    provider = resolve_runtime_lens_field_provider(state, KEY, native)
    assert source.provider.field_at_global_positions_t(moved) == pytest.approx(provider.field_at_global_positions_t(moved))


def test_first_order_uses_small_central_differences_for_posed_analytic_lens(monkeypatch):
    from temsim.physics import first_order, instrument_magnetic
    state, native = _single_lens_state()
    # Remove the independent electric-field trigger: lens pose alone must
    # select local finite differences instead of metre-sized basis rays.
    state.electron_gun = None
    _edit_pose(state, offset_x_mm=.01, rotation_y_mrad=.5)
    captured = []
    monkeypatch.setattr(instrument_magnetic, "active_column_events", lambda _state: ())
    monkeypatch.setattr(instrument_magnetic, "events_overlapping_interval", lambda *_: ())
    monkeypatch.setattr(first_order, "propagate", lambda *args, **kwargs: captured.append(args) or "executed")
    result, steps, central = first_order._trace_basis(state, native.lens.z_mm-1, native.lens.z_mm+1)
    assert result == "executed" and central
    assert steps == pytest.approx((1e-8, 1e-8, 1e-6, 1e-6))
    x, tx, y, ty = captured[0][3:7]
    assert np.asarray((x, y, tx, ty)).shape == (4, 9)
    assert np.asarray((x, y, tx, ty))[:, 1:5] == pytest.approx(-np.asarray((x, y, tx, ty))[:, 5:])


def test_nonlinear_readiness_reports_different_channel_poses_before_solving(monkeypatch):
    from temsim.physics import nonlinear_circuits
    from temsim.simulation_modes import nonlinear_mode_issues
    state, native = _single_lens_state()
    other = next(lens for lens in state.lenses if lens.key != KEY)
    descriptors = {key: {"solver": "axisymmetric_nonlinear_fem", "ampere_turns": 100.}
                   for key in (KEY, other.key)}
    monkeypatch.setattr(nonlinear_circuits, "operator_settings", lambda _row: {"same": "operator"})
    _edit_pose(state, rotation_y_mrad=.5)
    assert any("same rigid placement" in issue for issue in nonlinear_mode_issues(state, descriptors))
