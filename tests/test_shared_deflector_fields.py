"""Shared electrical drives use one physical field per host coil.

Small real component objects isolate ownership and finite-field admission;
these tests do not trace the gun or claim a measured coil calibration.
"""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from temsim.optics.ac_deflector import create_ac_deflector
from temsim.optics.beam_deflector import create_beam_deflector
from temsim.optics.descan_deflector import create_descan_deflector
from temsim.optics.image_diffraction_deflector import create_image_diffraction_deflector
from temsim.physics.instrument_magnetic import active_column_events, column_dipole_fields


def paired_state(kind="descan"):
    host, channel = ((create_beam_deflector(), create_ac_deflector()) if kind == "scan"
                     else (create_image_diffraction_deflector(), create_descan_deflector()))
    host.upper_x_mrad, host.upper_y_mrad = .4, -.2
    host.lower_x_mrad, host.lower_y_mrad = -.1, .3
    channel.kick_x_mrad, channel.kick_y_mrad = .6, -.4
    channel.scan_enabled = False
    if hasattr(channel, "wobble_enabled"):
        channel.wobble_enabled = False
    state = SimpleNamespace(deflectors=[host], corrector_elements=[channel], stigmators=[],
        lenses=[], apertures=[], beam_voltage_kv=300., simulation_time_s=0.,
        sample=SimpleNamespace(z_mm=1599.2, thickness_nm=0.), simulation_mode="analytical",
        step_mm=.2, history_step_mm=.2, equivalent_image_lenses_enabled=False,
        acceleration_enabled=False, acceleration_backend="CPU")
    return state, host, channel


def test_static_and_channel_drives_sum_at_two_host_foils():
    state, host, channel = paired_state()
    # Legacy channel coordinates and thickness cannot create another coil.
    object.__setattr__(channel, "z_mm", 9000.)
    object.__setattr__(channel, "effective_thickness_mm", .01)
    fields = column_dipole_fields(state)
    assert len(fields) == 2
    assert [field.key for field in fields] == [f"{host.key}:0", f"{host.key}:1"]
    np.testing.assert_allclose([(field.event_dx_rad, field.event_dy_rad) for field in fields],
                               np.asarray(((.7, -.4), (.2, .1)))*1e-3, atol=1e-18)
    assert [field.event_z_mm for field in fields] == [host.upper_z_mm, host.lower_z_mm]
    for field in fields:
        assert (field.upper_m-field.lower_m)*1e3 == pytest.approx(host.thickness_mm)
        assert field.drive_keys == (host.key, channel.key)
        assert not field.dynamic


def test_host_disable_suppresses_all_drives_but_channel_disable_keeps_alignment():
    state, host, channel = paired_state()
    channel.enabled = False
    fields = column_dipole_fields(state)
    np.testing.assert_allclose([(f.event_dx_rad, f.event_dy_rad) for f in fields],
                               np.asarray(((.4, -.2), (-.1, .3)))*1e-3)
    assert all(field.drive_keys == (host.key,) for field in fields)
    channel.enabled = True
    host.enabled = False
    assert column_dipole_fields(state) == ()


def test_missing_host_cannot_silently_discard_an_active_channel():
    state, _, _ = paired_state()
    state.deflectors = []
    with pytest.raises(ValueError, match="missing physical deflector host"):
        column_dipole_fields(state)


def test_legal_individual_drives_cannot_exceed_the_shared_coil_limit_together():
    state, host, channel = paired_state()
    host.maximum_kick_mrad = .6
    # Host alignment is at most 0.4 mrad and the channel's upper drive is
    # 0.3 mrad, but their physical sum is 0.7 mrad.
    with pytest.raises(ValueError, match="summed alignment and scan drive"):
        column_dipole_fields(state)


def test_dynamic_scan_stays_attached_to_host_for_recording_and_wave_arrival():
    from temsim.physics.column_wave import _component_events
    from temsim.physics.record_plane import _dynamic_column_coils
    state, host, channel = paired_state()
    channel.scan_enabled = True
    channel.scan_pixels_x = channel.scan_lines = 2
    channel.scan_frame_period_s = 1.
    channel.set_scan_command_matrix_mrad(((2., 0.), (0., 0.)))
    baseline = column_dipole_fields(state)
    peak = column_dipole_fields(state, time_s=.125)
    np.testing.assert_allclose([p.event_dx_rad-b.event_dx_rad for p, b in zip(peak, baseline)],
                               (.5e-3, .5e-3), atol=1e-18)
    assert all(field.dynamic for field in peak)
    low, high = host.upper_z_mm-20., host.lower_z_mm+20.
    assert [field.key for field in _dynamic_column_coils(state, low, high)] == [field.key for field in peak]
    events, owners = _component_events(state, low, high, arrival_time=lambda _z: .125)
    np.testing.assert_allclose(events, [(f.event_z_mm, f.event_dx_rad, f.event_dy_rad) for f in peak])
    assert len(owners) == 2
    assert all(row["component"] == host.key and row["finite_field"] for row in owners)
    assert all(row["drive_keys"] == (host.key, channel.key) and row["arrival_time_s"] == .125 for row in owners)


def test_summed_commands_enter_particle_plan_once_as_finite_fields():
    from temsim.physics.core import build_propagation_plan
    state, host, _ = paired_state()
    fields = column_dipole_fields(state)
    plan = build_propagation_plan(state, host.upper_z_mm-20., host.lower_z_mm+20.,
        active_column_events(state), include_spherical_aberration=False, include_hexapole=False)
    assert np.all(plan.kick_x_rad == 0.)
    assert np.all(plan.kick_y_rad == 0.)
    midpoint = .5*(plan.z_mm[:-1]+plan.z_mm[1:])*1e-3
    positions = np.column_stack((np.zeros_like(midpoint), np.zeros_like(midpoint), midpoint))
    expected = sum(field.field_at_global_positions_t(positions) for field in fields)
    np.testing.assert_allclose(plan.dipole_bx_t[1::3], expected[:, 0])
    np.testing.assert_allclose(plan.dipole_by_t[1::3], expected[:, 1])


def test_upstream_wave_events_use_summed_host_commands():
    from temsim.physics.gun_wave_transport import upstream_component_events
    state, host, _ = paired_state()
    state.electron_gun = SimpleNamespace(exit_plane_z_mm=450.)
    state.sample.z_mm = host.lower_z_mm+20.
    events, owners = upstream_component_events(state)
    assert events == list(active_column_events(state))
    assert len(owners) == 2
    assert all(row["component_id"] == host.key for row in owners)


def test_diagnostic_field_uses_host_bore_and_no_channel_sources():
    from temsim.magnetic_field_scene import _extra_sources
    state, host, channel = paired_state()
    host.mechanical_clear_bore_diameter_mm = 1.
    object.__setattr__(channel, "mechanical_clear_bore_diameter_mm", .01)
    sources = _extra_sources(state, (-np.inf, np.inf), [])
    assert len(sources) == 2
    assert all(source.key.startswith(host.key+":") for source in sources)
    assert all(source.radius_m == pytest.approx(.5e-3) for source in sources)


def test_channel_edit_restarts_at_host_field_entrance(monkeypatch):
    from temsim.physics import particle_sections
    state, host, channel = paired_state()
    state.electron_gun = SimpleNamespace(exit_plane_z_mm=450.)
    monkeypatch.setattr(particle_sections, "section_limits", lambda _s: (450., 2000.))
    boundary = particle_sections._selection_boundary(state, [channel.key])
    assert boundary == pytest.approx(host.upper_z_mm-.5*host.thickness_mm)


def test_ac_scan_and_upstream_beam_pair_keep_separate_physical_sources():
    state, beam, ac = paired_state("scan")
    ac.wobble_enabled = True
    ac.wobble_amplitude_x_mrad = 2.
    ac.wobble_period_s = 1.
    fields = column_dipole_fields(state, time_s=.25)
    assert len(fields) == 4
    assert {field.key for field in fields} == {
        "beam_deflector:0", "beam_deflector:1", "ac_deflector:0", "ac_deflector:1"}
    assert ac.upper_z_mm > beam.lower_z_mm
    beam_fields = [field for field in fields if field.key.startswith("beam_deflector:")]
    ac_fields = [field for field in fields if field.key.startswith("ac_deflector:")]
    np.testing.assert_allclose([field.event_dx_rad for field in beam_fields], (.4e-3, -.1e-3))
    np.testing.assert_allclose([field.event_dx_rad for field in ac_fields], (1.3e-3, -1.3e-3))
    assert all(field.dynamic for field in ac_fields)
    assert all(not field.dynamic for field in beam_fields)
    beam.enabled = False
    assert {field.key for field in column_dipole_fields(state)} == {"ac_deflector:0", "ac_deflector:1"}


def test_host_disable_stops_scan_preview_without_losing_channel_preference():
    from temsim.optics.shared_deflectors import bind_shared_deflector_channels, shared_channel_enabled
    from temsim.physics.scan_geometry import _scan_kicks_mrad
    state, host, channel = paired_state()
    bind_shared_deflector_channels(state)
    channel.scan_enabled = True
    times = np.asarray((.1, .5, .9))
    assert np.any(_scan_kicks_mrad(channel, times))
    host.enabled = False
    assert not shared_channel_enabled(channel)
    assert np.all(_scan_kicks_mrad(channel, times) == 0.)
    assert channel.enabled and channel.scan_enabled
    host.enabled = True
    assert np.any(_scan_kicks_mrad(channel, times))


def held_state():
    from temsim.optics.shared_deflectors import bind_shared_deflector_channels
    state, host, descan = paired_state()
    ac = create_ac_deflector()
    ac.wobble_enabled = False
    ac.scan_enabled = descan.scan_enabled = True
    ac.calibration_mode = "held"
    state.corrector_elements.append(ac)
    state.ac_deflector, state.descan_deflector = ac, descan
    bind_shared_deflector_channels(state)
    record = dict(version=1, ac_ratio=[[-1., 0.], [0., -1.]],
        descan_ratio=[[1., 0.], [0., 1.]], command_mrad=[[.01, 0.], [0., .01]],
        fov_nm=[ac.scan_field_of_view_x_nm, ac.scan_field_of_view_y_nm],
        target_key="selected_area_aperture", target_z_mm=host.lower_z_mm+10.,
        descan_calibrated=True, reference="sample_centre")
    return state, host, record


@pytest.mark.parametrize("saved_host", [None, "descan_deflector", "another_host"])
def test_old_or_different_descan_host_requires_explicit_recalibration(saved_host):
    from temsim.physics.scan_calibration import restore_held, validate_record
    state, _, record = held_state()
    if saved_host is not None:
        record["descan_physical_host_key"] = saved_host
    text = json.dumps(record)
    assert validate_record(text)["version"] == 1  # Still readable as historical data.
    state.ac_deflector.calibration_record_json = text
    with pytest.raises(ValueError, match="Use Calibrate and hold"):
        restore_held(state)


def test_held_host_identity_survives_lens_and_host_position_changes():
    from temsim.physics.scan_calibration import restore_held
    state, host, record = held_state()
    record["descan_physical_host_key"] = host.key
    state.ac_deflector.calibration_record_json = json.dumps(record)
    state.lenses = [SimpleNamespace(percent=42.)]
    host.mechanical_center_below_sample_mm += 5.
    np.testing.assert_array_equal(restore_held(state), record["command_mrad"])


def test_disabled_host_does_not_require_new_descan_calibration():
    from temsim.physics.scan_calibration import restore_held
    state, host, record = held_state()
    record["descan_calibrated"] = False
    host.enabled = False
    state.ac_deflector.calibration_record_json = json.dumps(record)
    np.testing.assert_array_equal(restore_held(state), record["command_mrad"])


def test_new_held_record_captures_physical_host(monkeypatch):
    from temsim.physics.scan_calibration import capture_record
    state, host, _ = held_state()
    state.descan_deflector.set_image_plane_coupling(((1., 0.), (0., 1.)),
        target_key="selected_area_aperture", target_z_mm=host.lower_z_mm+10.)
    monkeypatch.setattr("temsim.instrument_snapshot.capture_instrument_snapshot",
                        lambda _s: SimpleNamespace(digest="isolated-calibration-fixture"))
    assert json.loads(capture_record(state, descan_calibrated=True))["descan_physical_host_key"] == host.key


def test_disabled_host_is_not_solved_by_scan_calibration(monkeypatch):
    from temsim.physics import scan_geometry
    state, host, _ = held_state()
    host.enabled = False
    monkeypatch.setattr(scan_geometry, "calibrate_ac_scan_scale", lambda _s: (np.eye(2)*.01, 0.))
    monkeypatch.setattr(scan_geometry, "calibrate_descan_image_plane",
                        lambda _s: pytest.fail("A disabled physical host cannot be calibrated"))
    _, _, result = scan_geometry.calibrate_scan_system(state)
    assert result is None
