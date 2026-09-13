"""Review b637e35: isolated counterexamples, not full TEM/STEM acceptance."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.artifact_store import ArtifactStore
from temsim.calculation_cache import calculation_signatures
from temsim.calculation_manifest import capture_calculation_manifest
from temsim.optics.column import default_state
from temsim.optics.electron_gun.field_emission import FieldEmissionGun
from temsim.optics.electron_gun.tip_coherence import TipCoherence, TipWaveNumerics, generate_tip_emission
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.specimen_wave_transport import _slice_phase, _bandlimit_mode
from temsim.physics.wave_flux import WaveMode
from temsim.physics.wave_grid import WaveSamplingError


def phase_fixture(n=100, incoming=.3, added=.4, carrier=False):
    # Fixed physical period; n=200/400 are independent finer-grid references.
    spacing = 1e-8/n
    axis = (np.arange(n)-n//2)*spacing
    phase = 2*np.pi*incoming*axis/1e-10
    amplitude = np.ones((n, n), complex)/n
    tilt = None
    if carrier:
        from temsim.optics.electron_gun.tip_coherence import wavelength_m
        tilt = np.array((incoming*float(wavelength_m(300000.))/1e-10, 0.))
    else:
        amplitude = amplitude*np.exp(1j*phase)[None, :]
    mode = WaveMode(PlaneWave(amplitude, np.eye(2)*spacing, np.zeros(2), tilt_rad=tilt),
                    1., "isolated-operator", "review-alias", 300.)
    potential = np.broadcast_to(2*np.pi*added*axis/1e-10, (n, n))
    return mode, potential, axis


@pytest.mark.parametrize("carrier", [False, True])
def test_combined_specimen_bandwidth_rejects_before_aliasing(carrier):
    mode, potential, axis = phase_fixture(carrier=carrier)
    original = mode.plane.amplitude.copy()
    with pytest.raises(WaveSamplingError, match="bandwidth|sampling"):
        _slice_phase(mode, potential, axis, axis, 1., 1.)
    np.testing.assert_array_equal(mode.plane.amplitude, original)


def test_independent_fine_grids_retain_correct_complex_phase_and_frequency():
    outputs = []
    for n in (200, 400):
        mode, potential, axis = phase_fixture(n)
        result = _slice_phase(mode, potential, axis, axis, 1., 1.)
        expected = np.broadcast_to(np.exp(2j*np.pi*.7*axis/1e-10)/n, (n, n))
        np.testing.assert_allclose(result.plane.amplitude, expected, atol=2e-15, rtol=0)
        power = abs(np.fft.fft2(result.plane.amplitude))**2
        _, col = np.unravel_index(power.argmax(), power.shape)
        assert np.fft.fftfreq(n, d=1e-8/n)[col] == pytest.approx(.7/1e-10, rel=1e-14)
        outputs.append(result.plane.amplitude*n)
    np.testing.assert_allclose(outputs[0], outputs[1][::2, ::2], atol=2e-13, rtol=0)


def test_undersampled_analytic_carrier_cannot_be_folded_by_band_filter():
    mode, _, _ = phase_fixture(incoming=.7, added=0., carrier=True)
    with pytest.raises(WaveSamplingError):
        _bandlimit_mode(mode, 2/3)


@pytest.mark.parametrize("field", ["upper_b0_t", "lower_b0_t", "upper_a_mm", "lower_a_mm",
    "upper_field_center_z_mm", "lower_field_center_z_mm", "cc_mm"])
@pytest.mark.parametrize("assembled", [False, True])
def test_every_live_objective_field_parameter_invalidates_incident(field, assembled):
    state = default_state()
    if assembled:
        catalog = AssemblyCatalog()
        catalog.apply(state, catalog.default_selection())
    baseline = calculation_signatures(state)
    setattr(state.objective_lens, field, float(getattr(state.objective_lens, field))+.001)
    changed = calculation_signatures(state)
    assert changed["incident"] != baseline["incident"]


@pytest.mark.parametrize("profile", ["upper_gaussian", "lower_gaussian"])
@pytest.mark.parametrize("field", ["amplitude", "offset", "sigma"])
def test_both_objective_profiles_are_fully_bound(profile, field):
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    baseline = calculation_signatures(state)
    terms = list(getattr(state.objective_lens, profile))
    terms[0] = replace(terms[0], **{field: getattr(terms[0], field)+.01})
    setattr(state.objective_lens, profile, terms)
    assert calculation_signatures(state)["incident"] != baseline["incident"]


def test_artifact_store_rejects_old_lower_pole_but_keeps_history(tmp_path):
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    old = capture_calculation_manifest(state)
    state.objective_lens.lower_b0_t *= 1.01
    new = capture_calculation_manifest(state)
    store = ArtifactStore(tmp_path/"cache", quota_bytes=10_000_000)
    store.put_array_bundle(old, product_key="incident", dependency_signature=old.calculation_signatures["incident"],
                           arrays={"old": np.array([1.])}, metadata={"test": "old-pole"})
    assert store.get_array_bundle(new, product_key="incident", dependency_signature=new.calculation_signatures["incident"]) is None
    assert store.get_array_bundle(old, product_key="incident", dependency_signature=old.calculation_signatures["incident"]) is not None


def test_display_only_lens_changes_do_not_invalidate_physics():
    state = default_state()
    before = calculation_signatures(state)
    state.objective_lens.colour = "#123456"
    state.objective_lens.name = "Display label only"
    assert calculation_signatures(state) == before


def test_wave_and_particle_reject_nonforward_tip_without_changing_parameters():
    gun = FieldEmissionGun()
    gun.emitter.energy_spread_fwhm_ev = 0.
    gun.emitter.coherence = TipCoherence(tilt_x_mrad=1500.)
    before = gun.to_dict()
    with pytest.raises(ValueError, match="forward|domain"):
        generate_tip_emission(gun, TipWaveNumerics(energy_samples=1))
    with pytest.raises(ValueError, match="forward|domain"):
        gun.emit(33)
    assert gun.to_dict() == before


def test_small_phase_modulation_still_checks_its_diffraction_sidebands():
    mode, _, axis = phase_fixture(incoming=.3, added=0.)
    # A small local derivative alone would pass; the +.4 sideband added to
    # the incident +.3 carrier cannot be represented on this grid.
    potential = np.broadcast_to(.2*np.sin(2*np.pi*.4*axis/1e-10), (100, 100))
    with pytest.raises(WaveSamplingError):
        _slice_phase(mode, potential, axis, axis, 1., 1.)


def test_legal_tip_domain_preserves_statistics_and_records_error_budget():
    from temsim.optics.electron_gun.tip_coherence import tip_covariance
    gun = FieldEmissionGun()
    # Explicit isolated valid-domain fixture, NOT a replacement of user
    # defaults or an assertion of acceptance for the actual microscope.
    gun.emitter.virtual_source_fwhm_nm = 100.
    gun.emitter.coherence = TipCoherence(incoherent_angle_rms_mrad=2.)
    before = gun.to_dict()
    emission = generate_tip_emission(gun, TipWaveNumerics(energy_samples=9))
    report = emission.record["source_domain"]
    assert report["paraxial_generator_relative_error_bound"] <= .01
    assert report["tail_probability_budget"] == 1e-8
    assert report["nonforward_probability_upper_bound"] <= 1e-8
    assert report["minimum_evaluated_energy_ev"] == gun.emitter.minimum_kinetic_energy_ev
    gun.emitter.energy_spread_fwhm_ev = 0.
    bundle = gun.emit(65536)
    momentum = np.array((bundle.tx_rad, bundle.ty_rad))
    momentum /= np.sqrt(1+np.sum(momentum**2, axis=0))
    values = np.vstack((bundle.x_m, bundle.y_m, momentum))
    expected = tip_covariance(gun.emitter, gun.emitter.emission_energy_ev)
    scale = np.sqrt(np.diag(expected))
    np.testing.assert_allclose(np.cov(values, bias=True)/np.outer(scale, scale),
                               expected/np.outer(scale, scale), atol=.002, rtol=0)
    gun.emitter.energy_spread_fwhm_ev = before["components"][gun.emitter.key]["energy_spread_fwhm_ev"]
    assert gun.to_dict() == before


def test_unqualified_default_coherent_tip_is_not_silently_recalibrated():
    from temsim.optics.electron_gun.tip_source_domain import TipSourceDomainError
    from temsim.instrument_snapshot import encode_instrument, decode_instrument
    gun = FieldEmissionGun()
    gun.emitter.energy_spread_fwhm_ev = 0.
    gun.emitter.coherence = TipCoherence()
    state = default_state()
    state.electron_gun = gun
    before = encode_instrument(state)
    for action in (lambda: generate_tip_emission(gun, TipWaveNumerics(energy_samples=1)), lambda: gun.emit(33)):
        with pytest.raises(TipSourceDomainError, match="paraxial") as error:
            action()
        assert error.value.report["paraxial_generator_relative_error_bound"] > .01
    assert encode_instrument(decode_instrument(before)) == before
    assert encode_instrument(state) == before


def test_forward_domain_check_does_not_depend_on_lucky_finite_sampling():
    from temsim.optics.electron_gun.tip_source_domain import TipSourceDomainError
    gun = FieldEmissionGun()
    gun.emitter.virtual_source_fwhm_nm = 100.
    gun.emitter.coherence = TipCoherence(incoherent_angle_rms_mrad=200.)
    reports = []
    for action in (lambda: generate_tip_emission(gun), lambda: gun.emit(9), lambda: gun.emit(4096)):
        with pytest.raises(TipSourceDomainError) as error:
            action()
        reports.append(error.value.report)
    assert reports[0] == reports[1] == reports[2]


def test_changed_lower_pole_warm_trace_matches_cold_and_gui_has_no_whole_product_hit():
    from temsim.instrument_snapshot import encode_instrument, decode_instrument
    from temsim.calculation_cache import matching_products
    from temsim.gui.calculation_controller import CalculationController
    from temsim.physics.simulation import run
    from temsim.simulation_pipeline import CalculationResult
    from test_segmented_column_cache import _small_vacuum_state, _assert_incident_equal
    state = _small_vacuum_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    state.sample.inserted = False
    # Explicitly retain this resolved assembly: the low-level field edit is
    # the counterexample; do not overwrite it by re-applying TOML defaults.
    previous = run(state, resolved_layout=state._resolved_assembly, optical_only=True)
    old_signatures = calculation_signatures(state)
    result = CalculationResult(simulation=previous, energy_filter=None, signatures=old_signatures,
                               calculated_products=frozenset({"column", "incident"}))
    controller = CalculationController(persistent_cache_enabled=False)
    controller._cache_result(result)
    old_x = previous.incident.x.copy()
    state.objective_lens.lower_b0_t *= 1.01
    new_signatures = calculation_signatures(state)
    assert not matching_products(old_signatures, new_signatures)
    # A valid earlier prefix may be passed to the solver, but never the
    # complete old incident/column product after changing the lower pole.
    assert controller._seed_reuse_score(result, new_signatures) <= 1
    cold_state = decode_instrument(encode_instrument(state))
    warm = run(state, resolved_layout=state._resolved_assembly, existing_simulation=previous, optical_only=True)
    cold = run(cold_state, resolved_layout=cold_state._resolved_assembly, optical_only=True)
    _assert_incident_equal(warm.incident, cold.incident)
    assert warm.metrics["column_segment_cache"]["mode"] != "full_incident"
    np.testing.assert_array_equal(previous.incident.x, old_x)
