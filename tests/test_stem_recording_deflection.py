"""Recording deflection checks against independently propagated ray positions."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.physics.core import propagate
from temsim.physics.diffraction_memory import MemoryDiffractionSink
from temsim.physics.fourdstem import (
    FourDSTEMCalibration, FourDSTEMWriter, integrate_runtime_recording_planes,
)
from temsim.physics import record_plane
from temsim.physics.record_plane import PlaneStop, build_record_plane_plan, route_record_planes
from temsim.physics.stem_sampling import detector_angular_bounds
from temsim.physics.stem_wave_imaging import AngularDetector, simulate_angle_resolved_stem


@pytest.fixture
def recording_column(monkeypatch):
    """Actual column fields/descan; three small synthetic recording surfaces."""
    state = default_state()
    state.acceleration_enabled = False
    state.simulation_mode = "ideal"
    state.step_mm = 5.0
    for collection in (state.deflectors, state.corrector_elements):
        for component in collection:
            component.enabled = False
    descan = state.descan_deflector
    descan.enabled = True
    descan.scan_enabled = True
    descan.kick_x_mrad = 0.4
    descan.kick_y_mrad = -0.15
    descan.scan_pixels_x = 3
    descan.scan_lines = 1
    descan.scan_frame_period_s = 1.0
    descan.scan_amplitude_x_mrad = 1.0
    descan.scan_amplitude_y_mrad = 0.0
    # A nonzero reference time catches accidental addition of the baseline
    # raster kick a second time when the per-frame correction is applied.
    state.simulation_time_s = 0.37
    end = float(descan.lower_z_mm)
    planes = (
        PlaneStop("ap", "Aperture", end + 1, "aperture", "disk", radius_mm=0.025),
        PlaneStop("bf", "BF", end + 5, "detector", "disk",
                  outer_width_mm=0.025, readout_enabled=True),
        PlaneStop("camera", "Camera", end + 10, "detector", "disk",
                  outer_width_mm=10.0, readout_enabled=True),
    )
    monkeypatch.setattr(record_plane, "runtime_recording_stops", lambda *_: planes)
    return state, descan, planes


def _reference_positions(state, descan, planes, times, positions, angles):
    """Direct column propagation, without the plan or its offset helpers."""
    output = [[] for _ in planes]
    for time, position, angle in zip(times.ravel(), positions, angles):
        z, x, tx, y, ty = propagate(
            state, state.sample.z_mm, planes[-1].z_mm,
            position[..., 0].ravel(), angle[..., 0].ravel(),
            position[..., 1].ravel(), angle[..., 1].ravel(),
            events=descan.kick_events(time_s=float(time)),
            include_spherical_aberration=False, include_hexapole=False,
            save_z_mm=tuple(plane.z_mm for plane in planes),
        )
        for rows, plane in zip(output, planes):
            index = np.argmin(abs(z - plane.z_mm))
            rows.append(np.stack((x[index], y[index]), axis=-1).reshape(position.shape))
    return tuple(np.asarray(rows) for rows in output)


@pytest.mark.parametrize("scanning", [False, True])
def test_static_and_raster_descan_match_ray_propagation_and_conserve_weight(recording_column, scanning):
    state, descan, planes = recording_column
    descan.scan_enabled = scanning
    times = np.array([[0.1, 0.5, 0.9]])
    plan = build_record_plane_plan(state, scan_times_s=times)
    position = np.array([[1e-6, -2e-6], [-3e-6, 1e-6], [4e-6, 0.0]])
    angle = np.array([[0.1e-3, -0.2e-3], [0.0, 0.1e-3], [-0.2e-3, 0.0]])
    weights = np.array([0.2, 0.3, 0.5])
    result = route_record_planes(plan, position, angle, weights=weights)
    reference = _reference_positions(state, descan, planes, times, position, angle)
    alive = np.ones(3, dtype=bool)
    for event, expected in zip(result.interactions, reference):
        np.testing.assert_allclose(event.projected_position_m, expected, rtol=3e-6, atol=1e-10)
        np.testing.assert_array_equal(event.incident_mask, alive)
        if event.plane.kind == "aperture":
            blocked = alive & ~event.plane.transmission_mask(expected[..., 0], expected[..., 1])
        else:
            blocked = alive & event.plane.hit_mask(expected[..., 0], expected[..., 1])
        np.testing.assert_array_equal(event.intercepted_mask, blocked)
        alive &= ~blocked
    assert result.balance_error == pytest.approx(0.0, abs=1e-15)
    assert result.physically_intercepted_weight + result.surviving_weight == pytest.approx(1.0)
    # The static transfer includes the DC command, while scan offsets contain
    # only the change relative to simulation_time_s.
    assert any(np.any(transfer.position_offset_m) for transfer in plan.transfers)
    assert any(np.any(offset) for offset in plan.scan_position_offsets_m) == scanning
    if not scanning:
        assert plan.scan_position_offsets_m == ()


def _tiny_wave_inputs(state):
    state.illumination_mode = "STEM"
    state.sample.specimen_preset_key = "vacuum"
    state.sample.thickness_nm = 0.0
    state.sample.wave_grid_pixels = 32
    state.sample.wave_field_of_view_angstrom = 16.0
    state.sample.wave_multislice_enabled = False
    state.sample.wave_atomistic_enabled = False
    incident = SimpleNamespace(
        alive=np.ones(5, dtype=bool), ray_weight=np.array([.6, .1, .1, .1, .1]),
        x=np.full((1, 5), 1e-9), y=np.full((1, 5), -2e-9),
        tx=np.array([[0., .002, -.002, 0., 0.]]),
        ty=np.array([[0., 0., 0., .002, -.002]]),
    )
    scan = np.array([[-1e-4, 0., 1e-4]])
    return SimpleNamespace(incident=incident), scan, np.zeros_like(scan)


def test_wave_plan_without_legacy_shifts_matches_stored_cube_and_ray_reference(recording_column, tmp_path):
    state, descan, planes = recording_column
    simulation, scan_x, scan_y = _tiny_wave_inputs(state)
    times = np.array([[0.1, 0.5, 0.9]])
    plan = build_record_plane_plan(state, scan_times_s=times)
    sink = MemoryDiffractionSink(1_000_000, "test")
    wave = simulate_angle_resolved_stem(
        state, simulation, (AngularDetector("bf", 0., 5.), AngularDetector("camera", 5., 50.)),
        scan_x, scan_y, record_plane_plan=plan, diffraction_sink=sink,
    )
    artifact = wave.fourdstem_artifact
    calibration = artifact.calibration
    np.testing.assert_array_equal(calibration.scan_times_s, times)
    for chunk in (1, 2, 3):
        replay = integrate_runtime_recording_planes(artifact, None, plan, chunk_scan_points=chunk)
        for key in ("bf", "camera"):
            np.testing.assert_allclose(replay.images[key], wave.fractions[key], atol=1e-7)
        assert replay.balance_error == pytest.approx(0., abs=1e-7)

    # Independent detector interception of every FFT angular pixel, including
    # sample position and all static/scan kicks exactly once.
    shape = artifact.data.shape[-2:]
    positions = np.broadcast_to(np.stack((calibration.scan_x_um.ravel(), calibration.scan_y_um.ravel()), -1)[:, None, None, :] * 1e-6, (3, *shape, 2))
    angles = np.broadcast_to(np.stack((calibration.angle_x_mrad, calibration.angle_y_mrad), -1)[None] * 1e-3, positions.shape)
    reference = _reference_positions(state, descan, planes, times, positions, angles)
    alive = np.ones(artifact.data.reshape(3, *shape).shape, dtype=bool)
    for plane, expected in zip(planes, reference):
        if plane.kind == "aperture":
            alive &= plane.transmission_mask(expected[..., 0], expected[..., 1])
        else:
            hits = alive & plane.hit_mask(expected[..., 0], expected[..., 1])
            values = np.sum(artifact.data.reshape(3, *shape) * hits, axis=(-2, -1))
            np.testing.assert_allclose(wave.fractions[plane.key].ravel(), values, atol=1e-7)
            alive &= ~hits

    writer = FourDSTEMWriter(tmp_path / "times.npy", calibration, record_plane_plan=plan)
    for index in range(3):
        writer.write_frame(0, index, artifact.data[0, index])
    stored = writer.finalize()
    np.testing.assert_array_equal(stored.calibration.scan_times_s, times)
    assert stored.calibration.digest == calibration.digest
    descan.kick_x_mrad += 1.0
    changed_plan = build_record_plane_plan(state, scan_times_s=stored.calibration.scan_times_s)
    changed = integrate_runtime_recording_planes(stored, None, changed_plan)
    assert changed.plan_changed_since_capture
    assert changed_plan.fingerprint != plan.fingerprint
    assert any(not np.allclose(changed.images[key], wave.fractions[key]) for key in changed.images)
    wrong_times = build_record_plane_plan(state, scan_times_s=times + 0.01)
    assert wrong_times.fingerprint != changed_plan.fingerprint
    with pytest.raises(ValueError, match="scan times differ"):
        integrate_runtime_recording_planes(stored, None, wrong_times)


def test_legacy_cube_identity_is_preserved_but_dynamic_replay_requires_times(recording_column, tmp_path):
    state, _, _ = recording_column
    calibration = FourDSTEMCalibration.from_rectilinear_axes([0., 1., 2.], [0.], [0.], [0.])
    path = tmp_path / "legacy.npz"
    np.savez(path, scan_x_um=calibration.scan_x_um, scan_y_um=calibration.scan_y_um,
             angle_x_mrad=calibration.angle_x_mrad, angle_y_mrad=calibration.angle_y_mrad)
    loaded = FourDSTEMCalibration.load(path)
    assert loaded.scan_times_s is None
    assert loaded.digest == calibration.digest
    for invalid_times in (np.array([[0., np.nan, 1.]]), np.zeros((2, 2))):
        with pytest.raises(ValueError, match="scan times"):
            replace(calibration, scan_times_s=invalid_times)
    with pytest.raises(ValueError, match="scan times"):
        build_record_plane_plan(state, scan_times_s=np.array([[np.nan]]))
    with pytest.raises(ValueError, match="scan times"):
        build_record_plane_plan(state, scan_times_s=np.array([0., 1.]))
    plan = build_record_plane_plan(state)
    with pytest.raises(ValueError, match="recorded scan times"):
        integrate_runtime_recording_planes(np.ones(calibration.shape), loaded, plan)


def test_sampling_bounds_include_scan_deflection_exactly_once(recording_column):
    state, _, _ = recording_column
    times = np.array([[0.1, 0.5, 0.9]])
    plan = build_record_plane_plan(state, scan_times_s=times)
    detector = AngularDetector("bf", 0., 5.)
    position = np.array([[1e-6, -2e-6], [-3e-6, 1e-6], [4e-6, 0.]])
    bound = detector_angular_bounds([detector], positions_m=position, record_plane_plan=plan)["bf"]
    individual = []
    for index in range(3):
        transfers = tuple(replace(transfer, position_offset_m=np.asarray(transfer.position_offset_m) + offset.reshape(-1, 2)[index])
                          for transfer, offset in zip(plan.transfers, plan.scan_position_offsets_m))
        single = replace(plan, transfers=transfers, scan_times_s=None, scan_position_offsets_m=())
        individual.append(detector_angular_bounds([detector], positions_m=position[index:index+1], record_plane_plan=single)["bf"])
    assert bound[1] == pytest.approx(max(value[1] for value in individual))


def test_legacy_pair_and_wobble_events_follow_the_same_recording_path(recording_column):
    state, descan, planes = recording_column
    descan.enabled = False
    z = float(descan.upper_z_mm)
    legacy = SimpleNamespace(
        enabled=True, upper_z_mm=z, lower_z_mm=z + 1,
        upper_x_mrad=0.2, upper_y_mrad=-0.1, lower_x_mrad=0.3, lower_y_mrad=0.1,
    )
    wobble = SimpleNamespace(
        enabled=True, z_mm=z + 2, wobble_enabled=True,
        kick_events=lambda time_s=0.: ((z + 2, 0.001 * np.sin(2 * np.pi * time_s), 0.),),
    )
    state.deflectors = [legacy]
    state.corrector_elements = [wobble]
    times = np.array([[0., .25, .5]])
    plan = build_record_plane_plan(state, scan_times_s=times)
    assert plan.time_dependent_deflection
    routed = route_record_planes(plan, np.zeros((3, 2)), np.zeros((3, 2)))
    actual = []
    for time in times.ravel():
        events = ((z, .0002, -.0001), (z + 1, .0003, .0001), *wobble.kick_events(time))
        zero = np.zeros(1)
        zs, x, tx, y, ty = propagate(
            state, state.sample.z_mm, planes[-1].z_mm, zero, zero, zero, zero,
            events=events, include_spherical_aberration=False, include_hexapole=False,
        )
        actual.append((x[-1, 0], y[-1, 0]))
    np.testing.assert_allclose(routed.interactions[-1].projected_position_m, actual, rtol=3e-6, atol=1e-10)
