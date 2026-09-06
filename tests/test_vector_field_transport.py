"""Analytic Lorentz fixtures, not measured/FEM microscope validation."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from temsim.optics.column import default_state
from temsim.optics.direct_alignment import _LiveFirstOrderModel, diffraction_transfer
from temsim.optics.equivalent_image_lenses import (
    equivalent_image_calibrations, equivalent_image_lenses_enabled,
)
from temsim.physics.core import (
    E, build_propagation_plan, electron, execute_propagation_plan, propagate,
    propagation_plan_common_prefix_nodes,
)
from temsim.physics.first_order import trace_transverse_transfer
from temsim.physics.lens_field_provider import (
    CoordinateRegistration, FieldMapProvenance, MagneticFieldMap,
    bind_imported_lens_field_map, lens_geometry_binding,
)


def _state():
    lens = SimpleNamespace(
        key="mapped_test_lens", name="Synthetic lens", z_mm=0.5,
        enabled=True, percent=100.0, max_percent=200.0, polarity=1,
        field_support_mm=lambda *args: (-1.0, 2.0),
        magnetic_field_t=lambda z: np.full_like(np.asarray(z, dtype=float), 0.9),
    )
    return SimpleNamespace(
        lenses=[lens], stigmators=[], corrector_elements=[],
        beam_voltage_kv=300.0, step_mm=0.1, history_step_mm=0.1,
        acceleration_enabled=False, acceleration_backend="CPU",
        projector_mode="diffraction", equivalent_image_lenses_enabled=False,
        sample=SimpleNamespace(z_mm=0.0),
    )


def _map(state, values=(0.003, 0.0, 0.0), *, lens=None,
         registration=None, axes=None, components=None, rz=False):
    lens = state.lenses[0] if lens is None else lens
    if axes is None:
        axes = ((np.linspace(0.0, 0.01, 3), np.linspace(0.0, 0.001, 3))
                if rz else (np.linspace(-0.01, 0.01, 3),)*2
                + (np.linspace(0.0, 0.001, 3),))
    if components is None:
        components = tuple(np.full(tuple(a.size for a in axes), b) for b in values)
    result = MagneticFieldMap(
        map_type="axisymmetric_rz" if rz else "cartesian_xyz",
        axes_m=axes, components_t=components,
        registration=registration or CoordinateRegistration(),
        geometry_fingerprint=lens_geometry_binding(state, lens.key, lens).geometry_fingerprint,
        reference_excitation_percent=100.0, reference_polarity=1,
        provenance=FieldMapProvenance("fem", "synthetic-test-only.npz", "0"*64,
                                    "Analytic regression fixture, not FEM data"),
    )
    bind_imported_lens_field_map(state, lens.key, result, native_provider=lens)
    return result


def _trace(state, inputs=None, start=0.0, stop=1.0, **kwargs):
    inputs = (np.zeros(1),)*4 if inputs is None else inputs
    return propagate(state, start, stop, *inputs,
                     include_spherical_aberration=False, include_hexapole=False, **kwargs)


@pytest.mark.parametrize("field,coordinate,sign", [
    ((0.03, 0.0, 0.0), 2, -1), ((0.0, 0.03, 0.0), 0, 1),
])
def test_uniform_transverse_field_matches_relativistic_circle(field, coordinate, sign):
    state = _state()
    _map(state, field)
    result = _trace(state)
    k = sign*E*0.03/electron(state)[1]
    length = 1e-3
    position = k*length**2/(1 + np.sqrt(1-(k*length)**2))
    slope = k*length/np.sqrt(1-(k*length)**2)
    assert result[coordinate+1][-1,0] == pytest.approx(position, rel=2e-8)
    assert result[coordinate+2][-1,0] == pytest.approx(slope, rel=2e-8)
    # The native 0.9 T model must not also rotate/focus this mapped lens.
    other = 2-coordinate
    assert result[other+1][-1,0] == pytest.approx(0.0, abs=1e-16)
    assert result[other+2][-1,0] == pytest.approx(0.0, abs=1e-16)


def test_uniform_axial_field_rotates_slopes_without_double_focusing():
    state = _state()
    _map(state, (0.0, 0.0, 0.05))
    tx = 0.03
    result = _trace(state, (np.zeros(1), np.array([tx]), np.zeros(1), np.zeros(1)))
    omega = -E*0.05/electron(state)[1]*np.sqrt(1+tx*tx)
    angle = omega*1e-3
    expected = [tx*np.sin(angle)/omega, tx*np.cos(angle),
                tx*(np.cos(angle)-1)/omega, -tx*np.sin(angle)]
    np.testing.assert_allclose([item[-1,0] for item in result[1:]], expected, rtol=2e-8)
    np.testing.assert_allclose(result[2]**2+result[4]**2, tx*tx, rtol=1e-9)


def test_vector_field_uses_each_ray_position_and_energy():
    state = _state()
    axes = (np.linspace(-.01,.01,3),)*2 + (np.linspace(0,.001,3),)
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    _map(state, axes=axes, components=(yy*2.0, xx*2.0, zz*0))
    # Divergence-free quadrupole, no on-axis B at all.
    inputs = (np.array([1e-4,-1e-4]), np.zeros(2), np.zeros(2), np.zeros(2))
    result = _trace(state, inputs)
    assert result[2][-1,0] > 0
    assert result[2][-1,1] == pytest.approx(-result[2][-1,0], rel=1e-12)
    _map(state, (.003,0,0))
    result = _trace(state, (np.zeros(2),)*4, energy_offset_ev=np.array([0,300000]))
    assert abs(result[4][-1,0]) > abs(result[4][-1,1]) > 0


def test_rotated_box_support_covers_volume_not_only_axis():
    state = _state()
    rotation = ((0.,0.,1.), (0.,1.,0.), (-1.,0.,0.))
    axes = (np.linspace(-.002,.002,3), np.linspace(-.001,.001,3),
            np.linspace(-.001,.001,3))
    field_map = _map(state, (.003,0,0), axes=axes, registration=CoordinateRegistration(
        origin_global_m=(0,0,.010), rotation_local_to_global=rotation))
    assert field_map.field_support_mm == pytest.approx((8,12))
    assert field_map.field_at_global_positions_t([[0,0,.011]])[0,2] == pytest.approx(-.003)
    plan = build_propagation_plan(state, 9,11, include_spherical_aberration=False)
    assert len(plan.mapped_fields) == 1
    assert np.all(plan.magnetic_t == 0)
    result = _trace(state, (np.zeros(1), np.array([.01]), np.zeros(1), np.zeros(1)),
                    start=9, stop=11)
    assert result[4][-1,0] < 0


def test_registration_does_not_retain_mutable_caller_arrays():
    origin = np.zeros(3)
    rotation = np.eye(3)
    registration = CoordinateRegistration(origin, rotation)
    origin[2] = 1
    rotation[0,0] = 0
    assert registration.origin_global_m == (0.,0.,0.)
    np.testing.assert_array_equal(registration.rotation_array, np.eye(3))


def test_tilted_rz_support_contains_radial_extent():
    state = _state()
    c = np.sqrt(.5)
    rotation = ((c,0,c),(0,1,0),(-c,0,c))
    field_map = _map(state, (0,.003), rz=True,
        axes=(np.linspace(0,.002,3), np.linspace(-.001,.001,3)),
        registration=CoordinateRegistration((0,0,.010), rotation))
    assert field_map.field_support_mm == pytest.approx((10-3*c,10+3*c))


def test_plan_freezes_map_and_resume_does_not_repeat_kicks():
    state = _state()
    _map(state)
    plan = build_propagation_plan(state, -.5,1.5,
        events=((.5,.001,-.002),), checkpoint_z_mm=(.5,), save_z_mm=(.5,),
        include_spherical_aberration=False)
    initial = (np.array([0,1e-6]), np.zeros(2), np.zeros(2), np.zeros(2))
    full = execute_propagation_plan(state, plan, *initial)
    checkpoint = full[-1]
    index = int(np.argmin(abs(plan.z_mm-.5)))
    state.lenses[0].percent = 10
    _map(state, (.01,.002,0))
    replay = execute_propagation_plan(state, plan, *initial)
    for old, current in zip(full[:5], replay[:5]):
        np.testing.assert_array_equal(old, current)
    resumed = execute_propagation_plan(state, plan, checkpoint.x_m[0],
        checkpoint.tx_rad[0], checkpoint.y_m[0], checkpoint.ty_rad[0],
        start_index=index, include_initial_plane_kicks=False)
    for old, current in zip(full[1:5], resumed[1:5]):
        np.testing.assert_array_equal(old[-1], current[-1])


def test_changed_transverse_map_invalidates_only_affected_prefix():
    state = _state()
    first_map = _map(state)
    old = build_propagation_plan(state, -2,2, include_spherical_aberration=False)
    second_map = _map(state, (-.003,0,0))
    new = build_propagation_plan(state, -2,2, include_spherical_aberration=False)
    assert first_map.content_fingerprint != second_map.content_fingerprint
    np.testing.assert_array_equal(old.magnetic_t, new.magnetic_t)
    assert old.signature != new.signature
    count = propagation_plan_common_prefix_nodes(old, new)
    assert 0 < count < len(old.z_mm)
    assert old.z_mm[count-1] < 0.0


def test_mapped_first_order_and_alignment_use_local_affine_transfer():
    state = _state()
    _map(state, (.003,.002,0))
    raw = trace_transverse_transfer(state, 0,1)
    assert raw.position_offset_m[0] > 0
    assert raw.position_offset_m[1] < 0
    small = np.array([1e-9,-2e-9,1e-7,-2e-7])
    result = _trace(state, tuple(np.array([small[i]]) for i in (0,2,1,3)))
    predicted = raw.matrix @ small + np.array((*raw.position_offset_m,*raw.angle_offset_rad))
    actual = np.array([result[i][-1,0] for i in (1,3,2,4)])
    np.testing.assert_allclose(actual, predicted, atol=2e-13)
    model = _LiveFirstOrderModel(state, 0,1, (state.lenses[0].key,), step_mm=.1)
    np.testing.assert_allclose(model.matrix([100]), raw.matrix, rtol=1e-9, atol=1e-12)
    assert state.lenses[0].percent == 100
    assert state.equivalent_image_lenses_enabled is False
    np.testing.assert_allclose(diffraction_transfer(state,1).matrix, raw.matrix,
                               atol=2e-10, rtol=1e-7)


def test_vector_transport_converges_against_independent_arclength_lorentz_ode():
    state = _state()
    axes = (np.linspace(-.01,.01,3),)*2 + (np.linspace(0,.01,3),)
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    _map(state, axes=axes, components=(-.5*xx+.0002, -.5*yy-.0003, .03+zz))
    direction = np.array([.2,.1,1.0])
    direction /= np.linalg.norm(direction)
    initial = np.r_[[0.,0.,0.],direction]
    charge_over_p = -E/electron(state)[1]
    def rhs(_s, values):
        x,y,z = values[:3]
        b = np.array([-.5*x+.0002, -.5*y-.0003, .03+z])
        return np.r_[values[3:], charge_over_p*np.cross(values[3:],b)]
    def at_target(_s, values):
        return values[2]-.01
    at_target.terminal = True
    reference = solve_ivp(rhs, (0,.02), initial, events=at_target,
                          method="DOP853", rtol=2.3e-14, atol=1e-17)
    assert reference.t_events[0].size == 1
    end = reference.y[:,-1]
    expected = np.array([end[0],end[3]/end[5],end[1],end[4]/end[5]])
    errors = []
    for step in (.5,.25,.125):
        state.step_mm = step
        result = _trace(state, tuple(np.array([v]) for v in (0,.2,0,.1)), stop=10)
        actual = np.array([item[-1,0] for item in result[1:]])
        errors.append(np.linalg.norm(actual-expected))
    assert errors[0] > 8*errors[1] > 64*errors[2]
    assert errors[-1] < 1e-10


def test_overlapping_map_fields_add_without_native_field_or_kick():
    state = _state()
    _map(state, (.003,.002,.001))
    single = _trace(state)
    second_lens = SimpleNamespace(**{**vars(state.lenses[0]), "key": "second_map"})
    state.lenses.append(second_lens)
    _map(state, (.0015,.001,.0005))
    _map(state, (.0015,.001,.0005), lens=second_lens)
    combined = _trace(state)
    for a,b in zip(single[1:], combined[1:]):
        np.testing.assert_allclose(a,b, rtol=1e-12, atol=1e-16)


def test_equivalent_image_calibration_tracks_map_content_including_zero():
    state = default_state()
    lens = state.projector_lens_p1
    z_m = lens.z_mm*1e-3 + np.linspace(-.002,.002,3)
    axes = (np.linspace(0,.002,3), z_m)
    def power():
        return next(item.power_at_100_percent_m1 for item in equivalent_image_calibrations(
            state, state.sample.z_mm, 3000) if item.key == lens.key)
    native_power = power()
    assert native_power > 0
    _map(state, (0,0), lens=lens, rz=True, axes=axes)
    assert power() == 0
    _map(state, (0,.05), lens=lens, rz=True, axes=axes)
    mapped_power = power()
    expected = (E/(2*electron(state)[1]))**2 * .05**2 * .004
    assert mapped_power == pytest.approx(expected, rel=1e-10)
    _map(state, (0,.1), lens=lens, rz=True, axes=axes)
    assert power() == pytest.approx(4*mapped_power, rel=1e-10)
    # Aligned RZ can use the explicit thin-lens option, a 3D box cannot.
    state.projector_mode = "image"
    state.equivalent_image_lenses_enabled = True
    assert equivalent_image_lenses_enabled(state)
    spanning = build_propagation_plan(state, state.sample.z_mm-1, lens.z_mm+3,
                                      include_spherical_aberration=False)
    assert len(spanning.mapped_fields) == 1
    assert not np.any(spanning.thin_power_m1)
    assert not np.any(spanning.thin_rotation_rad)
    _map(state, lens=lens, registration=CoordinateRegistration((0,0,lens.z_mm*.001)))
    assert not equivalent_image_lenses_enabled(state)
    plan = build_propagation_plan(state, lens.z_mm, lens.z_mm+1,
                                  include_spherical_aberration=True)
    assert len(plan.mapped_fields) == 1
    assert not np.any(plan.thin_power_m1)
    assert not np.any(plan.cs_kick_m3)


def test_mapped_grid_and_double_precision_are_included_in_memory_guard():
    from temsim.gui.calculation_controller import estimate_calculation_memory_bytes
    state = default_state()
    state.sample.inserted = False
    state.sample.wave_enabled = False
    native = estimate_calculation_memory_bytes(state, "High accuracy", 100, .1)
    lens = state.projector_lens_p1
    z = lens.z_mm*.001 + np.linspace(-.001,.001,3)
    _map(state, lens=lens, axes=(np.linspace(-.01,.01,3),)*2+(z,))
    coarse = estimate_calculation_memory_bytes(state, "High accuracy", 100, .1)
    _map(state, lens=lens, axes=(np.linspace(-.01,.01,3),)*2+
         (lens.z_mm*.001+np.linspace(-.001,.001,101),))
    refined = estimate_calculation_memory_bytes(state, "High accuracy", 100, .1)
    assert native < coarse < refined
