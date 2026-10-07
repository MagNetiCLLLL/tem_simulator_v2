"""Shared particle/diagnostic field contract for placed main-column controls."""
from copy import copy, deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from temsim.optics.column import default_state
from temsim.physics.core import build_propagation_plan, electron, execute_propagation_plan
from temsim.physics.instrument_magnetic import active_column_events, column_dipole_fields
from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.posed_column_fields import FrozenPosedMultipole, capture_posed_column_fields


@pytest.fixture(scope="module")
def native_state():
    return default_state()


def component_state(native, key, *, angle_mrad=0., offset_mm=0.):
    collection = next(name for name in ("stigmators", "corrector_elements", "deflectors")
                      if any(item.key == key for item in getattr(native, name)))
    component = deepcopy(next(item for item in getattr(native, collection) if item.key == key))
    component.enabled = True
    if hasattr(component, "quadrupole_tensor_m2"):
        component.strength_x_percent, component.strength_y_percent = 7., -4.
    elif hasattr(component, "quadrupole_strength_m2"):
        component.strength_m2 = 30.
    elif hasattr(component, "hexapole_strength_components_m3"):
        component.strength_m3, component.orientation_rad = 10000., .2
    else:
        component.kick_x_mrad, component.kick_y_mrad = .02, -.01
    assembly = replace(native._resolved_assembly, parts=tuple(
        replace(part, data={**part.data, "rotation_y_mrad": angle_mrad, "offset_x_mm": offset_mm})
        if part.key == key else part for part in native._resolved_assembly.parts))
    state = SimpleNamespace(stigmators=[], corrector_elements=[], deflectors=[], lenses=[],
        _resolved_assembly=assembly, condenser_system=native.condenser_system, sample=native.sample,
        simulation_mode="analytical", beam_voltage_kv=300., step_mm=.05, history_step_mm=.05,
        acceleration_enabled=False, acceleration_backend="CPU", projector_mode="diffraction",
        equivalent_image_lenses_enabled=False, simulation_time_s=0., apertures=[], recording_planes=[])
    setattr(state, collection, [component])
    return state, component


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_rotated_multipole_curl_and_jets_match_independent_differences(degree):
    rotation = Rotation.from_rotvec((.017, -.023, .12)).as_matrix()
    registration = CoordinateRegistration((2e-5, -3e-5, .4), tuple(map(tuple, rotation)))
    field = FrozenPosedMultipole("x", "x", registration, ((degree, 0, .7), (0, degree, -.3)),
        "gaussian", 0., .006, (-30., 30.), .003, 3e-22)
    point, step = np.array((1e-4, -2e-4, .405)), 1e-7
    a, gradient, hessian = field.vector_potential_jet(point)
    np.testing.assert_allclose(a, field.vector_potential_at_global_positions_t_m(point), rtol=1e-14)
    curl = np.array((gradient[2, 1]-gradient[1, 2], gradient[0, 2]-gradient[2, 0],
                     gradient[1, 0]-gradient[0, 1]))
    np.testing.assert_allclose(curl, field.field_at_global_positions_t(point), rtol=3e-14, atol=1e-18)
    for delta, axis in zip(np.eye(3)*step, range(3)):
        plus, gp, _ = field.vector_potential_jet(point+delta)
        minus, gm, _ = field.vector_potential_jet(point-delta)
        np.testing.assert_allclose((plus-minus)/(2*step), gradient[:, axis], rtol=2e-6, atol=1e-12)
        np.testing.assert_allclose((gp-gm)/(2*step), hessian[:, :, axis], rtol=2e-6, atol=1e-12)


@pytest.mark.parametrize("key", ["condenser_stigmator", "probe_qph2_quadrupole", "probe_hp2_hexapole", "probe_dph2_deflector"])
def test_real_component_plan_excludes_old_channels_and_matches_zero_pose_limit(native_state, key):
    unposed, component = component_state(native_state, key)
    almost, _ = component_state(native_state, key, angle_mrad=1e-8)
    assert capture_posed_column_fields(unposed) == ()
    frozen, = capture_posed_column_fields(almost)
    centre = component.z_mm
    span = float(getattr(component, "effective_thickness_mm",
                        getattr(component, "effective_length_mm", getattr(component, "length_mm", 1.))))*.3
    plans = [build_propagation_plan(state, centre-span, centre+span, active_column_events(state))
             for state in (unposed, almost)]
    assert len(plans[1].mapped_fields) == 1
    for name in ("sx_m2", "sy_m2", "sxy_m2", "hex_normal_m3", "hex_skew_m3", "dipole_bx_t", "dipole_by_t", "kick_x_rad", "kick_y_rad"):
        np.testing.assert_array_equal(getattr(plans[1], name), 0.)
    origins = np.array((-2e-6, 0., 3e-6))
    results = [execute_propagation_plan(state, plan, origins, np.zeros(3), origins*.4, np.zeros(3))
               for state, plan in zip((unposed, almost), plans)]
    for index in (1, 2, 3, 4):
        np.testing.assert_allclose(results[0][index][-1], results[1][index][-1], rtol=2e-7, atol=2e-12)
    assert frozen.fingerprint


def test_posed_dipole_field_and_scene_share_registration_without_double_count(native_state):
    from temsim.magnetic_field_scene import prepare_magnetic_scene
    state, component = component_state(native_state, "probe_dph2_deflector", angle_mrad=8., offset_mm=.04)
    field, = capture_posed_column_fields(state)
    coil, = column_dipole_fields(state)
    local = np.array(((0., 0., field.center_z_m), (0., 0., field.native_support_mm[1]*.001+.002)))
    points = local@field.registration.rotation_array.T+field.registration.origin_array_m
    expected = field.field_at_global_positions_t(points)
    np.testing.assert_allclose(coil.field_at_global_positions_t(points), expected, atol=1e-18)
    scene = prepare_magnetic_scene(state)
    np.testing.assert_allclose(scene.field_at_global_positions_t(points[:1]), expected[:1], atol=1e-18)
    assert len(scene._sources) == 1
    assert scene.physical_identity is not None and scene.numerical_identity is not None
    np.testing.assert_array_equal(expected[1], 0.)


def test_pose_capture_is_empty_for_minimal_zero_field_state():
    assert capture_posed_column_fields(SimpleNamespace()) == ()


def test_unbound_posed_deflector_override_is_rejected(native_state):
    state, component = component_state(native_state, "probe_dph2_deflector", angle_mrad=1.)
    event, = active_column_events(state)
    with pytest.raises(ValueError, match="override does not match"):
        build_propagation_plan(state, component.z_mm-1., component.z_mm+1.,
            ((event[0], event[1]+1e-3, event[2]),))


def test_dynamic_posed_coil_recaptures_arrival_drive_once_in_executed_column(native_state):
    from temsim.physics.column_wave import _prepare_column, _propagate_column, _mode_dipoles
    from temsim.physics.wave_grid import WaveGridNumerics
    from temsim.physics.wave_reference import AxialWaveReference
    from test_shared_deflector_fields import paired_state
    from test_wave_detector_readout import checkpoint

    state, host, channel = paired_state()
    state._resolved_assembly = replace(native_state._resolved_assembly, parts=tuple(
        replace(part, data={**part.data, "rotation_y_mrad": 1., "offset_x_mm": .01})
        if part.key == host.key else part for part in native_state._resolved_assembly.parts))
    state.condenser_system = native_state.condenser_system
    state.recording_planes = []
    state.projector_mode = "diffraction"
    channel.scan_enabled = True
    channel.scan_pixels_x = channel.scan_lines = 2
    channel.scan_frame_period_s = 1.
    channel.set_scan_command_matrix_mrad(((2., 0.), (0., 0.)))
    captured = capture_posed_column_fields(state,
        arrival_time=lambda z: .125 if z < (host.upper_z_mm+host.lower_z_mm)/2 else .625)
    assert [field.captured_time_s for field in captured] == [.125, .625]
    for field in captured:
        expected = next(coil for coil in column_dipole_fields(state, time_s=field.captured_time_s)
                        if coil.key == field.lens_key)
        assert field.event_dx_rad == expected.event_dx_rad
        assert field.polynomial_terms == ((0, 1, expected.bx_t), (1, 0, -expected.by_t))

    # A short interior section isolates per-arrival drive ownership from the
    # separately tested native hard-edge axial integration error.
    low, high = host.upper_z_mm-.1, host.upper_z_mm+.1
    prepared = _prepare_column(state, low, high, .01)
    plan, _, _, owners = prepared
    assert len(plan.mapped_fields) == 1
    assert not np.any(plan.dipole_bx_t) and not np.any(plan.dipole_by_t)
    initial = checkpoint()
    mode = replace(initial.beam.modes[0], axial_reference=AxialWaveReference(1e-8, 1e-22))
    initial = replace(initial, plane_z_mm=low, beam=replace(initial.beam, modes=(mode,)))
    output = _propagate_column(state, initial, high, _prepared=prepared, tip_time_s=.125,
        grid_numerics=WaveGridNumerics(compute_backend="CPU"))
    record = output.record["modes"][0]
    saved, = record["posed_column_fields"]
    assert saved["lens_key"] == host.key+":0"
    assert saved["captured_time_s"] > .125
    expected = next(field for field in column_dipole_fields(state, time_s=saved["captured_time_s"])
                    if field.key == saved["lens_key"])
    assert saved["event_dx_rad"] == expected.event_dx_rad
    assert saved["event_dx_rad"] != plan.mapped_fields[0].event_dx_rad
    assert saved["drive_keys"] == (host.key, channel.key)
    for component in _mode_dipoles(plan, owners, record["deflector_actions"], state):
        np.testing.assert_array_equal(component, 0.)
    assert output.beam.total_weight == pytest.approx(initial.beam.total_weight, rel=1e-10)


def test_shifted_dynamic_coil_support_and_record_offsets_follow_its_placed_field(native_state):
    from temsim.physics.instrument_magnetic import events_overlapping_interval
    from temsim.physics.record_plane import _dynamic_column_coils, _scan_deflection_offsets
    from test_shared_deflector_fields import paired_state

    state, host, channel = paired_state()
    displacement_mm = 10.
    state._resolved_assembly = replace(native_state._resolved_assembly, parts=tuple(
        replace(part, data={**part.data, "offset_z_mm": displacement_mm, "rotation_y_mrad": 1.})
        if part.key == host.key else part for part in native_state._resolved_assembly.parts))
    state.condenser_system = native_state.condenser_system
    state.projector_mode = "diffraction"
    channel.scan_enabled = True
    channel.scan_pixels_x = channel.scan_lines = 2
    channel.scan_frame_period_s = 1.
    channel.set_scan_command_matrix_mrad(((2., 0.), (0., 0.)))
    upper = column_dipole_fields(state)[0]
    assert upper.arrival_z_mm == pytest.approx(host.upper_z_mm+displacement_mm, abs=1e-4)
    # The callback must receive the placed centre, while the nominal centre
    # remains the stable ownership/event-matching key.
    queried = []
    def arrival(z):
        queried.append(z)
        return .125 if z > host.upper_z_mm+displacement_mm-.01 else 0.
    frozen = capture_posed_column_fields(state, arrival_time=arrival)
    assert queried[0] == upper.arrival_z_mm
    assert frozen[0].captured_time_s == .125
    assert frozen[0].event_z_mm == host.upper_z_mm

    low, high = upper.arrival_z_mm-.1, upper.arrival_z_mm+.1
    assert upper.upper_m*1e3 < low  # The old axial window would miss it entirely.
    active = _dynamic_column_coils(state, low, high)
    assert [coil.key for coil in active] == [upper.key]
    events = events_overlapping_interval(state, active_column_events(state), low, high)
    assert [row[0] for row in events] == [host.upper_z_mm]
    positions = []
    for time in (0., .125):
        timed = copy(state)
        timed.simulation_time_s = time
        plan = build_propagation_plan(timed, low, high, active_column_events(timed),
            include_spherical_aberration=False, include_hexapole=False, maximum_step_mm=.02)
        assert len(plan.mapped_fields) == 1
        ray = execute_propagation_plan(timed, plan, *(np.zeros(1) for _ in range(4)))
        positions.append(np.array((ray[1][-1, 0], ray[3][-1, 0])))
    expected = np.asarray(positions)-positions[0]
    assert np.linalg.norm(expected[1]) > 1e-10
    offsets, = _scan_deflection_offsets(state, low, (SimpleNamespace(z_mm=high),),
        np.array((0., .125)), .02, (SimpleNamespace(position_offset_m=positions[0]),))
    np.testing.assert_allclose(offsets, expected, atol=2e-15, rtol=2e-10)


def test_placed_hexapole_first_order_observer_keeps_actual_feeddown(native_state):
    from temsim.physics.first_order import trace_transverse_transfer
    state, component = component_state(native_state, "probe_hp2_hexapole", angle_mrad=1., offset_mm=.03)
    low, high = component.z_mm-.2, component.z_mm+.2
    observed = trace_transverse_transfer(state, low, high, maximum_step_mm=.025)
    # Independent half-sized symmetric perturbations execute the nonlinear
    # shared field, so a translated hexapole's linear feeddown cannot vanish.
    steps = np.array((5e-9, 5e-9, 5e-7, 5e-7))
    basis = np.column_stack((np.diag(steps), -np.diag(steps)))
    plan = build_propagation_plan(state, low, high, (), include_spherical_aberration=False,
        include_hexapole=True, maximum_step_mm=.025)
    ray = execute_propagation_plan(state, plan, basis[0], basis[2], basis[1], basis[3])
    endpoint = np.array([ray[i][-1] for i in (1, 3, 2, 4)])
    independent = (endpoint[:, :4]-endpoint[:, 4:])/(2*steps)
    assert np.linalg.norm(independent[2:, :2]) > 1e-5
    np.testing.assert_allclose(observed.matrix, independent, rtol=2e-7, atol=2e-10)


def test_actual_posed_conjugate_atlas_captures_full_chief_gauge(native_state):
    from temsim.optics.direct_alignment import canonical_transfers, canonical_source_basis
    from temsim.physics.conjugate_planes import build_conjugate_atlas
    state, component = component_state(native_state, "condenser_stigmator", angle_mrad=8., offset_mm=.04)
    low, high = component.z_mm-.25, component.z_mm+.25
    metrics = {"optical_tuning": True, "sample_scattering_applied": False,
        "optical_execution_extent": {"coordinate_system": "column_axial_z_mm",
            "physics_scope": "optical_reference_without_specimen_interactions",
            "start_z_mm": low, "completed_z_mm": high}}
    result = SimpleNamespace(state_snapshot=state,
        simulation=SimpleNamespace(section_checkpoint=None, metrics=metrics))
    atlas = build_conjugate_atlas(result)
    maps = canonical_transfers(state, low, atlas.z_mm)
    expected = np.stack([canonical_source_basis(state, z,
        position_xy_m=maps[float(z)].position_offset_m)[2:, :2] for z in atlas.z_mm])
    np.testing.assert_allclose(atlas.source_c_m1, expected, rtol=2e-12, atol=1e-14)
    assert np.max(abs(atlas.source_c_m1[:, 0, 0])) > 1e-8
    np.testing.assert_allclose(atlas.transfer_matrix(low, high), maps[high].matrix,
                               rtol=2e-9, atol=2e-10)
