"""Finite scan calibration checked against analytic and executed coil fields."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from temsim.immutable_json import json_digest
from temsim.instrument_snapshot import encode_instrument, decode_instrument
from temsim.optics.column import default_state
from temsim.optics.direct_alignment import canonical_source_basis
from temsim.optics.scan_coil_design import evaluate_scan_coil_design
from temsim.physics import core
from temsim.physics.finite_scan_response import finite_scan_responses
from temsim.physics.instrument_magnetic import active_column_events, column_dipole_fields
from temsim.physics.scan_geometry import calibrate_scan_system


@pytest.fixture(scope="module")
def native():
    s = default_state()
    s.acceleration_enabled = True
    s.acceleration_backend = "Numba CPU"
    return encode_instrument(s)


def zero_field(native):
    s = decode_instrument(native)
    s._propagation_energy_kev = s.beam_voltage_kv
    s.electron_gun = None
    for c in (*s.lenses, *s.stigmators, *s.corrector_elements, *s.deflectors):
        c.enabled = False
    s.ac_deflector.enabled = True
    s.ac_deflector.scan_enabled = False
    s.ac_deflector.wobble_enabled = False
    return s


def direct_phase(s, source, targets):
    zero = np.zeros(1)
    points = core.propagate(s, source, max(targets), zero, zero, zero, zero,
        events=active_column_events(s), include_spherical_aberration=False,
        include_hexapole=True, maximum_step_mm=.1,
        checkpoint_z_mm=targets, return_checkpoints=True)[-1]
    return {float(z): np.array([points.x_m[i,0], points.y_m[i,0],
                               points.tx_rad[i,0], points.ty_rad[i,0]])
            for i,z in enumerate(points.z_mm)}


def test_zero_field_exact_length_partial_response_and_no_double_kick(native):
    s = zero_field(native)
    coil = next(c for c in column_dipole_fields(s) if c.key == "ac_deflector:0")
    lo, hi = coil.field_support_mm
    z = [lo-1., lo, coil.event_z_mm, hi, hi+10.]
    before = json_digest(encode_instrument(s))
    response = finite_scan_responses(s, s.ac_deflector, z)
    length = (hi-lo)*1e-3
    for i, station in enumerate(z):
        elapsed = max(0., min(station, hi)-lo)*1e-3
        after = max(0., station-hi)*1e-3
        expected = np.vstack((np.eye(2)*(elapsed**2/(2*length)+after*elapsed/length),
                              np.eye(2)*elapsed/length))
        np.testing.assert_allclose(response.upper[i], expected, rtol=2e-13, atol=2e-14)
        np.testing.assert_allclose(response.lower[i], 0., atol=1e-18)
    assert json_digest(encode_instrument(s)) == before


def test_uniform_axial_field_matches_forced_matrix_exponential(native, monkeypatch):
    s = zero_field(native)
    b = .7
    monkeypatch.setattr(core, "fields", lambda z, *_args, **_kw:
                        (np.full_like(np.asarray(z), b, dtype=float),
                         np.zeros_like(np.asarray(z)), np.zeros_like(np.asarray(z))))
    coil = next(c for c in column_dipole_fields(s) if c.key == "ac_deflector:0")
    lo, hi = coil.field_support_mm
    stop = hi+10.
    charge, momentum, _ = core.electron(s)
    g = charge*b/(2*momentum)
    generator = np.zeros((4,4))
    generator[:2,2:] = np.eye(2)
    generator[2:,2:] = [[0.,2*g],[-2*g,0.]]
    augmented = np.zeros((6,6))
    augmented[:4,:4] = generator
    augmented[:4,4:] = np.eye(4)[:,2:] / ((hi-lo)*1e-3)
    expected = (expm(generator*(stop-hi)*1e-3)
                @ expm(augmented*(hi-lo)*1e-3)[:4,4:])
    result = finite_scan_responses(s,s.ac_deflector,[stop],maximum_step_mm=.02)
    np.testing.assert_allclose(result.upper[0],expected,rtol=2e-9,atol=2e-11)
    thin = expm(generator*(stop-coil.event_z_mm)*1e-3)[:,2:]
    assert np.linalg.norm(result.upper[0]-thin) > 1e-5


def test_tilted_finite_coil_matches_actual_provider_transport(native):
    s = zero_field(native)
    key = "ac_deflector"
    s._resolved_assembly = replace(s._resolved_assembly, parts=tuple(
        replace(p,data={**p.data,"rotation_y_mrad":8.,"rotation_z_mrad":150.})
        if p.key == key else p for p in s._resolved_assembly.parts))
    s.ac_deflector.set_pure_shift_coupling(np.zeros((2,2)))
    coil = next(c for c in column_dipole_fields(s) if c.key == key+":0")
    stop = max(c.field_support_mm[1] for c in column_dipole_fields(s))+2.
    response = finite_scan_responses(s,s.ac_deflector,[stop],maximum_step_mm=.1)
    baseline = direct_phase(s,response.source_z_mm,[stop])[stop]
    s.ac_deflector.kick_x_mrad = .002
    moved = direct_phase(s,response.source_z_mm,[stop])[stop]-baseline
    command = np.array([.002*s.ac_deflector.upper_coil_gain*1e-3,0.])
    np.testing.assert_allclose(moved,response.upper[0]@command,rtol=2e-6,atol=2e-12)
    assert abs(response.upper[0,3,0]) > .1  # actual registered cross-axis field
    assert coil.registration is not None


def test_production_solve_matches_executed_finite_ac_and_descan(native):
    s = decode_instrument(native)
    s.descan_deflector.descan_target_key = "selected_area_aperture"
    s.ac_deflector.scan_pixels_x = s.ac_deflector.scan_lines = 4
    s.ac_deflector.scan_pixel_size_nm = .1
    calibrate_scan_system(s)
    targets = [s.sample.z_mm,s.selected_area_aperture.z_mm]
    source = min(c.field_support_mm[0] for c in column_dipole_fields(s)
                 if c.key.startswith("ac_deflector:"))
    s.ac_deflector.scan_enabled = s.descan_deflector.scan_enabled = False
    baseline = direct_phase(s,source,targets)
    s.ac_deflector.scan_enabled = s.descan_deflector.scan_enabled = True
    desired = np.diag([s.ac_deflector.scan_field_of_view_x_nm,
                       s.ac_deflector.scan_field_of_view_y_nm])*.5e-9
    for t in (.01,.17,.43):
        s.simulation_time_s = t
        actual = direct_phase(s,source,targets)
        sample = actual[targets[0]]-baseline[targets[0]]
        expected = desired @ s.ac_deflector.scan_factors(t)
        np.testing.assert_allclose(sample[:2],expected,rtol=2e-6,atol=2e-16)
        assert np.linalg.norm(sample[2:]) < 2e-12
        assert np.linalg.norm(actual[targets[1]][:2]-baseline[targets[1]][:2]) < 3e-13


def test_finite_canonical_design_preserves_state_and_both_position_axes(native):
    s = decode_instrument(native)
    before = json_digest(encode_instrument(s))
    result = evaluate_scan_coil_design(s,target="canonical",response_model="finite_coil",
                                       pivot_step_mm=2.,maximum_step_mm=.1)
    assert result.response_model == "finite_coil"
    assert "finite-coil" in result.scope
    response = finite_scan_responses(s,s.ac_deflector,[s.sample.z_mm],maximum_step_mm=.05)
    phase = (response.upper[0]@np.asarray(result.upper_kick_matrix_mrad)
             +response.lower[0]@np.asarray(result.lower_kick_matrix_mrad))*1e-3
    g = canonical_source_basis(s,s.sample.z_mm)[2:,:2]
    desired = np.diag(result.field_of_view_nm)*.5e-9
    np.testing.assert_allclose(phase[:2],desired,atol=2e-15)
    assert np.linalg.norm(phase[2:]-g@phase[:2]) < 2e-12
    assert np.linalg.norm(phase[2:]) > 1e-8
    assert result.pivot.is_common_pivot
    assert json_digest(encode_instrument(s)) == before


def test_raster_time_is_neutralized_without_mutating_caller(native):
    s = decode_instrument(native)
    first = finite_scan_responses(s,s.ac_deflector,[s.sample.z_mm])
    s.simulation_time_s = .371
    s.ac_deflector.set_scan_command_matrix_mrad(np.eye(2)*.01)
    second = finite_scan_responses(s,s.ac_deflector,[s.sample.z_mm])
    np.testing.assert_allclose(first.upper,second.upper,rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(first.lower,second.lower,rtol=1e-12,atol=1e-14)
    assert s.simulation_time_s == .371


def test_static_ac_alignment_is_retained_in_the_common_affine_reference(native):
    s = decode_instrument(native)
    s.ac_deflector.scan_enabled = s.descan_deflector.scan_enabled = False
    s.ac_deflector.kick_x_mrad = .004
    s.ac_deflector.kick_y_mrad = -.003
    stop = s.selected_area_aperture.z_mm
    response = finite_scan_responses(s,s.descan_deflector,[stop],maximum_step_mm=.1)
    actual = direct_phase(s,response.source_z_mm,[stop])[stop]
    np.testing.assert_allclose(response.reference[0],actual,rtol=2e-10,atol=1e-14)
    assert np.linalg.norm(actual[:2]) > 1e-9
    assert response.source_z_mm < s.ac_deflector.upper_z_mm


def test_scale_failure_rolls_back_coupling_and_flags(monkeypatch):
    from temsim.physics import scan_geometry as scan
    from temsim.optics.ac_deflector import create_ac_deflector
    ac = create_ac_deflector()
    state = SimpleNamespace(ac_deflector=ac,sample=SimpleNamespace(z_mm=2000.))
    before = dict(ac.__dict__)
    monkeypatch.setattr(scan,"_coil_phase_responses",lambda *_args:
                        (np.vstack((np.zeros((2,2)),2*np.eye(2))),
                         np.vstack((np.zeros((2,2)),np.eye(2)))))
    with pytest.raises(ValueError,match="singular"):
        scan.calibrate_ac_scan_scale(state)
    assert ac.__dict__ == before


def test_consumption_starts_at_finite_field_entrance(native,monkeypatch):
    from temsim.physics import scan_geometry as scan
    s = decode_instrument(native)
    coil = next(c for c in column_dipole_fields(s) if c.key == "ac_deflector:0")
    stop = .5*(coil.field_support_mm[0]+coil.event_z_mm)
    calls = []
    monkeypatch.setattr(scan,"calibrate_ac_scan_scale",lambda _s:
                        calls.append("AC") or (np.eye(2),0.))
    result = scan.calibrate_scan_system(s,observation_stop_z_mm=stop)
    assert calls == ["AC"]
    assert result is not None and result[2] is None


def test_truncated_preview_does_not_refit_the_descan_target(native):
    from temsim.physics import scan_geometry as scan
    s = decode_instrument(native)
    s.descan_deflector.descan_target_key = "selected_area_aperture"
    scan.calibrate_scan_system(s)
    accepted = np.asarray(s.descan_deflector.image_plane_lower_ratio_matrix)
    stop = float(s.descan_deflector.lower_z_mm)+1.
    assert stop < s.selected_area_aperture.z_mm
    geometry = scan.calculate_scan_geometry(s,observation_stop_z_mm=stop)
    np.testing.assert_allclose(s.descan_deflector.image_plane_lower_ratio_matrix,
                               accepted,rtol=1e-12,atol=1e-12)
    assert geometry.descan_target_z_mm == s.selected_area_aperture.z_mm


@pytest.mark.parametrize("kwargs",[{"response_model":"thin_guess"},{"response_model":"finite_coil","lower_z_mm":1599.1}])
def test_finite_design_rejects_unknown_model_or_field_overlapping_sample(native,kwargs):
    with pytest.raises(ValueError):
        evaluate_scan_coil_design(decode_instrument(native),**kwargs)
