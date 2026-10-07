"""Short CPU traces for physical lens installation errors, never a full column."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.lens_pose import lens_pose_registration
from temsim.optics.column import default_state
from temsim.physics.core import build_propagation_plan, execute_propagation_plan, propagation_plan_common_prefix_nodes
from temsim.physics.posed_aberrations import FrozenLensAberrationKick
from temsim.physics.lens_field_provider import CoordinateRegistration

KEY = "condenser_lens_1"


def _state(**pose):
    source = default_state()
    lens = next(lens for lens in source.lenses if lens.key == KEY)
    lens.cs_mm = .1
    state = SimpleNamespace(lenses=[lens], condenser_system=source.condenser_system,
        _resolved_assembly=source._resolved_assembly, stigmators=[], corrector_elements=[],
        simulation_mode="analytical", beam_voltage_kv=300., step_mm=.5, history_step_mm=.5,
        acceleration_enabled=False, acceleration_backend="CPU", projector_mode="diffraction",
        equivalent_image_lenses_enabled=False, sample=SimpleNamespace(z_mm=lens.z_mm+100.))
    _pose(state, **pose)
    return state


def _pose(state, **values):
    state._resolved_assembly = replace(state._resolved_assembly, parts=tuple(
        replace(part, data={**part.data, **values}) if part.key == KEY else part
        for part in state._resolved_assembly.parts))


def _plan(state, **options):
    z = state.lenses[0].z_mm
    return build_propagation_plan(state, z-4., z+4., checkpoint_z_mm=(z,),
                                  include_hexapole=False, **options)


def test_displaced_lens_transports_about_its_own_axis_and_freezes_plan():
    state = _state(offset_x_mm=.01)
    plan = _plan(state)
    assert len(plan.mapped_fields) == len(plan.posed_spherical_kicks) == 1
    assert not np.any(plan.magnetic_t)  # no duplicate coaxial lens field
    assert not np.any(plan.cs_kick_m3)  # no duplicate global-axis Cs kick
    initial = (np.array([1e-5+2e-6]), np.array([1e-5]), np.array([3e-6]), np.array([0.]))
    before = execute_propagation_plan(state, plan, *initial)
    _pose(state, offset_x_mm=.03)
    current = _plan(state)
    shifted = execute_propagation_plan(state, current, initial[0]+2e-5, *initial[1:])
    np.testing.assert_allclose(shifted[1]-2e-5, before[1], rtol=1e-8, atol=1e-12)
    for a, b in zip(shifted[2:5], before[2:5]):
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-12)
    repeated = execute_propagation_plan(state, plan, *initial)
    for a, b in zip(repeated[:5], before[:5]):
        np.testing.assert_array_equal(a, b)
    assert current.signature != plan.signature
    assert propagation_plan_common_prefix_nodes(plan, current) < len(plan.z_mm)


def test_tilted_lens_axis_ray_remains_straight_without_cuda():
    state = _state(rotation_x_mrad=1., rotation_y_mrad=-2., offset_x_mm=.02)
    plan = _plan(state)
    registration = lens_pose_registration(state, KEY)
    direction = registration.rotation_array[:, 2]
    slope = direction[:2]/direction[2]
    origin = registration.origin_array_m
    initial_xy = origin[:2]+(plan.z_mm[0]*1e-3-origin[2])*slope
    result = execute_propagation_plan(state, plan, np.array([initial_xy[0]]), np.array([slope[0]]),
                                      np.array([initial_xy[1]]), np.array([slope[1]]))
    dz = (result[0]-result[0][0])*1e-3
    np.testing.assert_allclose(result[1][:, 0], initial_xy[0]+slope[0]*dz, atol=2e-12)
    np.testing.assert_allclose(result[3][:, 0], initial_xy[1]+slope[1]*dz, atol=2e-12)
    np.testing.assert_allclose(result[2][:, 0], slope[0], atol=2e-11)
    np.testing.assert_allclose(result[4][:, 0], slope[1], atol=2e-11)


def test_infinitesimal_pose_does_not_change_paraxial_model_or_add_aberration():
    state = _state()
    # Converge the original coarse-grid RK4 before comparing equations. The
    # spatial-field branch independently limits its step to resolve the lens.
    state.step_mm = .025
    # A visible finite ray angle exposes an accidental full-Lorentz cubic term.
    inputs = (np.array([1e-5]), np.array([.02]), np.array([-2e-5]), np.array([-.01]))
    baseline = execute_propagation_plan(state, _plan(state), *inputs)
    _pose(state, offset_x_mm=1e-12)
    posed = execute_propagation_plan(state, _plan(state), *inputs)
    for reference, actual in zip(baseline[1:5], posed[1:5]):
        np.testing.assert_allclose(actual[-1], reference[-1], rtol=3e-6, atol=1e-10)


def test_pose_cs_frame_and_checkpoint_resume_do_not_repeat_impulse():
    registration = CoordinateRegistration((.001, -.002, 0.))
    kick = FrozenLensAberrationKick("lens", .01, 1e12, registration)
    x, tx, y, ty = kick.apply(np.array([.001+1e-5]), np.zeros(1), np.array([-.002]), np.zeros(1))
    assert tx[0] == pytest.approx(-.001)
    assert ty[0] == pytest.approx(0.)
    assert x[0] == pytest.approx(.001+1e-5)
    state = _state(offset_x_mm=.01, offset_z_mm=.2)
    plan = _plan(state)
    event = plan.posed_spherical_kicks[0]
    assert event.z_mm == pytest.approx(state.lenses[0].z_mm+.2)
    index = int(np.argmin(abs(plan.z_mm-event.z_mm)))
    plan = replace(plan, checkpoint_index=np.array([index]))
    inputs = (np.array([1.2e-5]), np.zeros(1), np.zeros(1), np.zeros(1))
    whole = execute_propagation_plan(state, plan, *inputs)
    cp = whole[-1]
    resumed = execute_propagation_plan(state, plan, cp.x_m[0], cp.tx_rad[0], cp.y_m[0], cp.ty_rad[0],
                                      start_index=index, include_initial_plane_kicks=False)
    for a, b in zip(resumed[1:5], whole[1:5]):
        np.testing.assert_allclose(a[-1], b[-1], atol=1e-14)


def test_pose_plan_roundtrips_section_safe_data_codec():
    from temsim.particle_section_io import _pack, _unpack, _data_classes
    state = _state(offset_x_mm=.01, rotation_y_mrad=1.)
    plan = _plan(state)
    arrays = {}
    encoded = _pack(plan, arrays, {}, _data_classes())
    decoded = _unpack(encoded, arrays, _data_classes())
    assert decoded.signature == plan.signature
    assert decoded.mapped_fields[0].fingerprint == plan.mapped_fields[0].fingerprint
    assert decoded.posed_spherical_kicks == plan.posed_spherical_kicks


def test_existing_section_plan_without_pose_field_remains_readable():
    from temsim.particle_section_io import _pack, _unpack, _data_classes
    state = _state()
    plan = _plan(state)
    arrays = {}
    encoded = _pack(plan, arrays, {}, _data_classes())
    encoded["fields"].pop("posed_spherical_kicks")
    decoded = _unpack(encoded, arrays, _data_classes())
    assert decoded.posed_spherical_kicks == ()
    np.testing.assert_array_equal(decoded.z_mm, plan.z_mm)
