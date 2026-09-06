import json
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics.axisymmetric_magnetostatics import MU0, solve_axisymmetric, solve_geometry_field_map
from temsim.physics.multiplane_wave import PlaneWave, propagate_plane_wave, intermediate_apertures
from temsim.specimen.vector_field_transport import SpecimenFieldTransport
from temsim.optics.field_aberrations import fit_wave_gradient


def _constant_transport(field):
    class Provider:
        def field_support_mm(self):
            return -10, 10

        def field_at_global_positions_t(self, positions):
            return np.broadcast_to(field, np.asarray(positions).shape)
    transport = SpecimenFieldTransport.__new__(SpecimenFieldTransport)
    transport.origin_m = np.zeros(3)
    transport.providers = (Provider(),)
    transport.spatial_step_nm = 100
    return transport


def test_vector_specimen_flight_transverse_field_and_plane_reversal():
    transport = _constant_transport((1.0, .5, .2))
    start, direction = np.array((2., -3., 0.)), np.array((0., 0., 1.))
    end, outgoing = transport.to_plane(start, direction, 10000, energy_ev=300000)
    assert np.linalg.norm(outgoing) == pytest.approx(1, abs=1e-14)
    assert end[2] == pytest.approx(10000, abs=1e-7)
    assert abs(outgoing[1]) > 1e-4
    restored, restored_direction = transport.to_plane(end, outgoing, 0, energy_ev=300000)
    np.testing.assert_allclose(restored, start, atol=1e-4)
    np.testing.assert_allclose(restored_direction, direction, atol=1e-10)


def test_magnetostatics_manufactured_solution_converges():
    errors = []
    for nodes in (18, 36):
        r, z = np.linspace(0, 1, nodes), np.linspace(-1, 1, 2*nodes-1)
        solution = solve_axisymmetric(r, z, lambda r, z: 1,
                                     lambda r, z: (8*r*(1-z*z) + 2*r*(1-r*r))/MU0)
        exact = r[:, None]*(1-r[:, None]**2)*(1-z[None, :]**2)
        errors.append(np.linalg.norm(solution.a_phi_tm-exact) / np.linalg.norm(exact))
        assert solution.relative_residual < 1e-10
        assert np.max(np.abs(solution.br_t[0])) == 0
    assert errors[1] < .4*errors[0]
    assert errors[1] < .01


def _coil_binding():
    import hashlib
    geometry = {"lens_assembly": {"parts": [{"key": "coil", "start_z_mm": -10, "end_z_mm": 10,
        "data": {"mechanical_profile": "magnetic_excitation_coil", "mechanical_inner_diameter_mm": 18,
                 "mechanical_outer_diameter_mm": 22}}]}}
    encoded = json.dumps(geometry, sort_keys=True, separators=(",", ":"))
    return SimpleNamespace(canonical_geometry_json=encoded, geometry_fingerprint=hashlib.sha256(encoded.encode()).hexdigest())


def test_generated_coil_field_is_cached_and_scales_with_current():
    binding = _coil_binding()
    settings = {"relative_permeability": 1, "ampere_turns": 100, "radial_nodes": 30, "axial_nodes": 60, "padding_factor": 4}
    field = solve_geometry_field_map(binding, settings)
    assert solve_geometry_field_map(binding, settings) is field
    assert field.geometry_fingerprint == binding.geometry_fingerprint
    twice = solve_geometry_field_map(binding, dict(settings, ampere_turns=200))
    np.testing.assert_allclose(twice.components_t[1], 2*field.components_t[1], atol=1e-12)
    assert field.field_at_global_positions_t(np.zeros(3))[2] > 0
    # Analytical centre field of a finite solenoid, averaged over the uniform
    # winding radius from 9 to 11 mm. This checks magnitude, not only linearity.
    reference = MU0*100/2 * (np.arcsinh(.011/.010)-np.arcsinh(.009/.010))/.002
    assert field.field_at_global_positions_t(np.zeros(3))[2] == pytest.approx(reference, rel=.03)


def _wave():
    axis = np.arange(32)-16
    y, x = np.meshgrid(axis, axis, indexing="ij")
    amplitude = np.exp(-(x*x+y*y)/20).astype(complex)
    amplitude /= np.linalg.norm(amplitude)
    return PlaneWave(amplitude, np.diag((1e-9, 1e-9)), np.zeros(2))


def test_coherent_intermediate_fourier_and_mask_preserve_absolute_flux():
    wave = _wave()
    identity = propagate_plane_wave(wave, np.eye(4), np.zeros(4), 2e-12)
    np.testing.assert_allclose(identity.amplitude, wave.amplitude)
    matrix = np.block([[np.zeros((2, 2)), .01*np.eye(2)], [-100*np.eye(2), np.zeros((2, 2))]])
    diffraction = propagate_plane_wave(wave, matrix, np.zeros(4), 2e-12)
    assert diffraction.probability == pytest.approx(1)
    mask = diffraction.coordinates_m()[0] > 0
    clipped = PlaneWave(np.where(mask, diffraction.amplitude, 0j), diffraction.basis_m, diffraction.origin_m,
                        diffraction.curvature_m1, diffraction.tilt_rad)
    downstream = propagate_plane_wave(clipped, matrix, np.zeros(4), 2e-12)
    assert 0 < downstream.probability < 1
    assert downstream.probability == pytest.approx(clipped.probability)


def test_wave_gradient_fit_recovers_cs_and_threefold_term():
    alpha = .025
    phase = np.arange(24) * 2*np.pi/24
    r = np.linspace(.1, 1, 9)
    theta = alpha * np.column_stack(((r[:, None]*np.cos(phase)).ravel(), (r[:, None]*np.sin(phase)).ravel()))
    x, y = theta.T
    cs_m, a2_m = .0012, .0003
    residual = -np.column_stack((cs_m*x*(x*x+y*y) + a2_m*(x*x-y*y),
                                 cs_m*y*(x*x+y*y) - 2*a2_m*x*y))
    fitted, rms, condition = fit_wave_gradient(theta, residual, alpha)
    assert fitted[7] == pytest.approx(cs_m*1e3, rel=1e-6)
    assert fitted[5] == pytest.approx(a2_m*1e3, rel=1e-6)
    assert rms < 1e-15
    assert condition < 100


def test_intermediate_apertures_use_existing_stop_units():
    from temsim.optics.column import default_state
    state = default_state()
    aperture = next(item for item in state.apertures if item.z_mm > state.sample.z_mm)
    aperture.enabled = True
    aperture.radius_mm = .01
    stops = intermediate_apertures(state, aperture.z_mm + 1)
    selected = next(item for item in stops if item.key == aperture.key)
    assert selected.transmission_mask(1e-6, 0)
    assert not selected.transmission_mask(1e-3, 0)


def test_camera_pipeline_applies_intermediate_mask_without_renormalising():
    from temsim.optics.column import default_state
    from temsim.physics.camera_wave import project_wave_to_recording_plane
    state = default_state()
    for detector in state.stem_detectors:
        detector.inserted = False
        detector.readout_enabled = False
    state.camera.inserted = True
    state.fluorescent_screen.inserted = False
    state.projector_mode = "image"
    state.objective_aperture.enabled = True
    state.objective_aperture.radius_mm = .02
    axis = np.linspace(-8, 8, 32)
    wave = np.exp(-(axis[:, None]**2 + axis[None, :]**2)/8).astype(complex)
    opened = project_wave_to_recording_plane(state, wave, axis, axis, .0197)
    assert opened.method == "multiplane_coherent_lct_with_apertures"
    assert 0 < opened.metrics["camera_collected_zero_loss_relative_intensity"] <= 1 + 1e-10
    state.objective_aperture.radius_mm = 0
    closed = project_wave_to_recording_plane(state, wave, axis, axis, .0197)
    assert not np.any(closed.intensity)
    assert closed.metrics["intermediate_aperture_transmissions"][0]["transmitted_probability"] == 0


def test_field_derived_mode_is_exclusive_and_invalidates_on_mode_change(monkeypatch):
    from temsim.optics.column import default_state
    from temsim.optics import aberrations, field_aberrations
    state = default_state()
    state.image_aberrations = {"mode": "field_derived", "c3_mm": 900}
    fitted = aberrations.EffectiveAberrationSet("objective image", "field-derived", c3_mm=.42)
    calls = []
    def derive(state, system):
        calls.append(system)
        return fitted, fitted, {}
    monkeypatch.setattr(field_aberrations, "derive_field_aberrations", derive)
    assert aberrations.active_effective_aberrations(state, "image").c3_mm == .42
    assert aberrations.active_effective_aberrations(state, "image").c3_mm == .42
    assert calls == ["image"]
    state.image_aberrations["mode"] = "manual"
    assert aberrations.active_effective_aberrations(state, "image").c3_mm == 900


def test_field_aberrations_use_production_vector_rays():
    from test_vector_field_transport import _state, _map
    from temsim.optics.field_aberrations import derive_field_aberrations
    state = _state()
    _map(state, (0., 0., .4))
    state.objective_image_plane_z_mm = .8
    state.image_aberrations = {"mode": "field_derived", "fit_semiangle_mrad": 10, "c3_mm": 900}
    before, after, evidence = derive_field_aberrations(state, "image")
    assert before is after
    assert abs(after.c3_mm) < 10
    assert np.isfinite(after.cc_mm)
    assert evidence["fit_rms_m"] < 1e-10
    assert evidence["unmapped_round_lenses"] == ()
    assert evidence["correction_comparison_available"] is False
    assert "ray_error_rms_after" not in evidence
