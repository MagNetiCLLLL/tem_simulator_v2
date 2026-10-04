"""Shared injection inputs, not a claim of exact classical quantum phase space."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from temsim.optics.electron_gun.field_emission import FieldEmissionGun
from temsim.optics.electron_gun.tip_surface import (
    SharedSurfaceCoherence, SurfaceCoherence, TipSurfaceModel,
    cap_flux_area_quantiles, emit_surface, load_tip_surface_reference,
)


def shared_model(*, mean_ev=.7, rms_ev=.12, cap_deg=10.):
    original = load_tip_surface_reference()
    return replace(original, emission=replace(original.emission,
        flux_profile="cosine_cap", cap_half_angle_deg=cap_deg,
        energy_distribution="gamma" if rms_ev else "monoenergetic",
        kinetic_mean_ev=mean_ev, kinetic_sigma_ev=rms_ev,
        maximum_angle_deg=0.), coherence=SharedSurfaceCoherence()).validate()


def test_shared_record_has_one_energy_owner_and_roundtrips_without_conversion():
    model = shared_model()
    record = model.to_dict()
    assert set(record["coherence"]) == {"model", "edge_phase_rad"}
    assert TipSurfaceModel.from_dict(record) == model
    record["coherence"]["mean_energy_ev"] = 30.
    with pytest.raises(ValueError, match="Invalid tip surface model"):
        TipSurfaceModel.from_dict(record)
    old = replace(load_tip_surface_reference(), coherence=SurfaceCoherence())
    assert TipSurfaceModel.from_dict(old.to_dict()) == old
    with pytest.raises(ValueError, match="propagated as a wave"):
        emit_surface(old, 49)


@pytest.mark.parametrize("cap_deg", [.1, 10., 45.])
def test_flux_cdf_matches_independent_spherical_area_integral(cap_deg):
    model = shared_model(cap_deg=cap_deg)
    quantiles = np.array([0., .001, .01, .1, .5, .9, .99, .999, 1.])
    area = cap_flux_area_quantiles(model, quantiles)
    angle = np.deg2rad(cap_deg)
    theta = 2*np.arcsin(np.sqrt(area)*np.sin(angle/2))
    def density(t):
        return np.sin(t)*np.cos(np.pi*t/(2*angle))**4
    normalisation = quad(density, 0., angle, epsabs=1e-14)[0]
    actual = np.array([quad(density, 0., t, epsabs=1e-14)[0]/normalisation for t in theta])
    np.testing.assert_allclose(actual, quantiles, atol=3e-8, rtol=0.)
    amplitude = model.source_amplitude(model.geometry.apex_radius_nm*np.sin(theta))
    np.testing.assert_allclose(abs(amplitude)**2, np.cos(np.pi*theta/(2*angle))**4, atol=1e-14)


def test_shared_particles_start_on_real_conductor_with_normal_direction_and_current():
    model = shared_model()
    gun = FieldEmissionGun()
    gun.emitter.surface_model = model
    bundle = gun.emit(3000)
    radius_m = model.geometry.apex_radius_nm*1e-9
    normals = (bundle.surface_position_m+[0., 0., radius_m])/radius_m
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1., atol=2e-15)
    np.testing.assert_allclose(bundle.surface_direction, normals, atol=2e-15)
    assert np.min(bundle.surface_position_m[:, 2]) < 0
    assert np.max(bundle.surface_position_m[:, 2]) <= 0
    assert np.sum(bundle.weight) == pytest.approx(1.)
    assert gun.emitter.emitted_current_a == pytest.approx(model.current_na*1e-9)
    np.testing.assert_allclose(bundle.energy_offset_ev+model.mean_energy_ev,
                               bundle.surface_energy_ev, atol=1e-16)


@pytest.mark.parametrize("mean_ev,rms_ev", [(.3, .1), (.7, .12), (2., .4), (.8, 0.)])
def test_wave_and_particle_energy_quadratures_read_same_editable_spectrum(mean_ev, rms_ev):
    model = shared_model(mean_ev=mean_ev, rms_ev=rms_ev)
    energies, weights = model.energy_quadrature(7)
    assert weights@energies == pytest.approx(mean_ev, abs=1e-14)
    assert np.sqrt(weights@(energies-mean_ev)**2) == pytest.approx(rms_ev, abs=1e-14)
    _, _, ray_energy, ray_weight = emit_surface(model, 16384)
    # Independent CDF quadrature converges to the same physical spectrum;
    # finite ray samples are not forcibly recentered or assigned wave weights.
    assert ray_weight@ray_energy == pytest.approx(mean_ev, rel=8e-4)
    assert np.sqrt(ray_weight@(ray_energy-mean_ev)**2) == pytest.approx(rms_ev, rel=3e-3, abs=1e-15)


@pytest.mark.parametrize("change", [
    {"flux_profile": "uniform_area"},
    {"energy_distribution": "normal_tangential_exponential"},
    {"maximum_angle_deg": 1.},
    {"spatial_sampling": "apex_stratified_v1"},
])
def test_incompatible_shared_phase_space_is_not_silently_replaced(change):
    model = shared_model()
    with pytest.raises(ValueError):
        replace(model, emission=replace(model.emission, **change)).validate()
    assert model.shared_boundary
    assert model.emission.maximum_angle_deg == 0.


def test_unimplemented_surface_phase_gradient_is_explicit():
    model = shared_model()
    with pytest.raises(ValueError, match="oblique shared source-port"):
        replace(model, coherence=SharedSurfaceCoherence(edge_phase_rad=.1)).validate()


def test_classical_and_shared_representation_use_identical_injection_samples():
    coherent = shared_model()
    classical = replace(coherent, coherence=None)
    for a, b in zip(emit_surface(coherent, 193), emit_surface(classical, 193)):
        np.testing.assert_array_equal(a, b)
    # Current defined as the area-mean flux retains the same physical current.
    mean_flux = 7e7
    density = replace(coherent, emission=replace(coherent.emission,
        current_na=None, flux_electrons_per_nm2_s=mean_flux)).validate()
    assert density.current_na == pytest.approx(mean_flux*density.emission.area_nm2(density.geometry)*1.602176634e-10)


def test_explicit_product_ray_budget_remains_valid_for_shared_surface():
    from temsim.optics.electron_gun.emitter import EmissionQuadrature
    gun = FieldEmissionGun()
    gun.emitter.surface_model = shared_model()
    gun.emitter.quadrature = EmissionQuadrature(spatial=31, directions=3, energies=7)
    bundle = gun.emit(gun.emitter.quadrature.total)
    assert len(bundle.weight) == 651
    assert np.sum(bundle.weight) == pytest.approx(1.)
    radius = gun.emitter.surface_model.geometry.apex_radius_nm*1e-9
    normal = (bundle.surface_position_m+[0., 0., radius])/radius
    np.testing.assert_allclose(bundle.surface_direction, normal, atol=2e-15)
    assert len(np.unique(bundle.surface_position_m, axis=0)) == 31
    assert len(np.unique(bundle.surface_energy_ev)) == 7


def test_shared_source_metadata_labels_injection_and_geometric_comparison():
    from temsim.optics.electron_gun.tip_coherence import describe_physical_tip_source
    gun = FieldEmissionGun()
    gun.emitter.surface_model = shared_model()
    source = describe_physical_tip_source(gun.emitter)
    assert "Incident injection" in source["current_definition"]
    assert "not an exact quantum Wigner" in source["particle_representation"]
    assert source["mean_kinetic_energy_ev"] == pytest.approx(.7)
    assert source["energy_rms_ev"] == pytest.approx(.12)


@pytest.mark.parametrize("prior_signature", [True, False])
def test_assembly_geometry_edits_refresh_shared_source_without_resetting_emission(prior_signature):
    from temsim import module_manifest
    from temsim.optics.electron_gun.tip_assembly import apply_tip_part
    gun = FieldEmissionGun()
    part = dict(module_manifest.part_data("gun/FEG.toml", "feg_tip"))
    apply_tip_part(gun.emitter, part)
    model = shared_model()
    gun.emitter.surface_model = model
    if not prior_signature:
        del gun.emitter._tip_assembly_signature
    before_key = gun._cache_key(49)
    before_positions = gun.emit(49).surface_position_m
    part["tip_radius_nm"] = 125.
    apply_tip_part(gun.emitter, part)
    changed = gun.emitter.surface_model
    assert changed.shared_boundary
    assert changed.coherence == model.coherence
    assert changed.emission == model.emission
    assert changed.geometry.apex_radius_nm == gun.emitter.tip_radius_nm == 125.
    np.testing.assert_allclose(gun.emit(49).surface_position_m, before_positions*1.25)
    assert gun._cache_key(49) != before_key


def test_historical_reservoir_is_not_reinterpreted_by_assembly_refresh():
    from temsim import module_manifest
    from temsim.optics.electron_gun.tip_assembly import apply_tip_part
    gun = FieldEmissionGun()
    part = dict(module_manifest.part_data("gun/FEG.toml", "feg_tip"))
    apply_tip_part(gun.emitter, part)
    historical = replace(load_tip_surface_reference(), coherence=SurfaceCoherence())
    gun.emitter.surface_model = historical
    part["tip_radius_nm"] = 125.
    apply_tip_part(gun.emitter, part)
    assert gun.emitter.surface_model is historical


def test_numerical_sampling_comparison_admits_shared_geometric_rays_only():
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.optics.column import default_state
    from temsim.optics.electron_gun.tip_coherence import TipCoherence
    from temsim.sampling_convergence import ConvergenceRequest
    from temsim.working_point import WorkingPointCheckpoint
    state = default_state()
    state.electron_gun.emitter.surface_model = shared_model()
    point = WorkingPointCheckpoint(capture_instrument_snapshot(state), {}, state.sample.z_mm, "inputs", {})
    prepared = ConvergenceRequest(point, "spatial", 54, 3, 3, 3).prepare()
    assert prepared.electron_gun.emitter.surface_model == shared_model()
    assert prepared.electron_gun.emitter.quadrature.total == 27
    assert prepared.electron_gun.emitter.emit().weight.sum() == pytest.approx(1.)
    assert state.electron_gun.emitter.quadrature is None
    state.electron_gun.emitter.surface_model = None
    state.electron_gun.emitter.coherence = TipCoherence()
    gaussian = WorkingPointCheckpoint(capture_instrument_snapshot(state), {}, state.sample.z_mm, "inputs", {})
    with pytest.raises(ValueError, match="sampling comparison"):
        ConvergenceRequest(gaussian, "spatial", 54, 3, 3, 3).prepare()
