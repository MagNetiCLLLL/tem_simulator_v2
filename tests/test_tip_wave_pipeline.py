from dataclasses import replace

import numpy as np
import pytest

from temsim.detector.wave_readout import WaveReadoutOptions
from temsim.optics.column import default_state
from temsim.optics.electron_gun.tip_coherence import TipCoherence, TipWaveNumerics
from temsim.physics.tip_gun_wave import GunWaveNumerics
from temsim.physics.tip_wave_pipeline import TipWaveRequest, simulate_tip_wave
from temsim.physics.wave_execution import WaveExecutionOptions


@pytest.mark.parametrize("segmented", [False, True])
def test_default_coherent_tip_domain_rejection_does_not_execute_or_publish(monkeypatch, tmp_path, segmented):
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.optics.electron_gun.tip_source_domain import TipSourceDomainError
    from temsim.physics import tip_gun_wave, tip_wave_pipeline
    state = default_state()
    state.electron_gun.emitter.coherence = TipCoherence()
    state.electron_gun.emitter.energy_spread_fwhm_ev = 0.
    before = capture_instrument_snapshot(state).digest
    stages = tuple(tip_wave_pipeline._STAGES)
    monkeypatch.setattr(tip_gun_wave, "_axial_grid", lambda *a, **kw: pytest.fail("Rejected tip must not transport"))
    request = TipWaveRequest(stop="gun_exit", source=TipWaveNumerics(energy_samples=1),
        execution=WaveExecutionOptions(segmented=segmented, cache_directory=str(tmp_path/"executed")),
        gun=GunWaveNumerics(field_step_mm=.1, bore_step_mm=2.))
    with pytest.raises(TipSourceDomainError, match="paraxial domain"):
        simulate_tip_wave(state, request)
    assert capture_instrument_snapshot(state).digest == before
    assert tuple(tip_wave_pipeline._STAGES) == stages
    assert not any(p.is_file() for p in tmp_path.rglob("*"))



def test_cancel_and_invalid_numerics_do_not_start_transport():
    state = default_state()
    with pytest.raises(InterruptedError):
        simulate_tip_wave(state, cancelled=lambda: True)
    for request in (TipWaveRequest(column_step_mm=0), TipWaveRequest(detector_pixels=True),
                    TipWaveRequest(maximum_readout_bytes=-1)):
        with pytest.raises(ValueError):
            request.validate()


def test_unexecuted_equivalent_state_is_not_a_pipeline_option():
    state = default_state()
    state.equivalent_image_lenses_enabled = True
    with pytest.raises(ValueError, match="executed upstream caches"):
        simulate_tip_wave(state, TipWaveRequest(stop="gun_exit"))


def test_installed_energy_filter_is_not_silently_bypassed():
    state = default_state()
    _install_filter(state)
    # A selected physical path crossing the real entrance remains unsupported.
    state.energy_filter.entrance_z_mm = min(d.z_mm for d in state.stem_detectors)-1.
    with pytest.raises(ValueError, match="energy filter requires"):
        simulate_tip_wave(state, TipWaveRequest(stop="detector"))


def test_filter_after_selected_detector_does_not_prevent_upstream_calculation(monkeypatch):
    """Control fixture checks filter ordering, not unsupported column physics."""
    import temsim.physics.tip_wave_pipeline as pipeline
    state = default_state()
    _install_filter(state)
    state.camera.inserted = state.camera.readout_enabled = True
    before = state.energy_filter.entrance_z_mm
    assert state.energy_filter_installed
    assert all(d.z_mm < before for d in (*state.stem_detectors, state.camera, state.fluorescent_screen))
    def gun_started(*args, **kwargs):
        raise RuntimeError("physical upstream gun calculation reached")
    monkeypatch.setattr(pipeline, "build_tip_gun_checkpoint", gun_started)
    monkeypatch.setattr(pipeline, "_prepare_column", _quiet_prepared_column)
    for key in ("haadf", "bf", "camera"):
        with pytest.raises(RuntimeError, match="upstream gun calculation reached"):
            simulate_tip_wave(state, TipWaveRequest(stop="detector", detector_key=key))
    assert state.energy_filter_installed and state.energy_filter.entrance_z_mm == before


def test_partial_stage_cannot_cross_a_filter_moved_upstream():
    state = default_state()
    _install_filter(state)
    state.energy_filter.entrance_z_mm = state.electron_gun.exit_plane_z_mm-1.
    with pytest.raises(ValueError, match="energy filter requires"):
        simulate_tip_wave(state, TipWaveRequest(stop="gun_exit"))


def test_material_multislice_with_actual_atomistic_potential_preserves_branches(monkeypatch):
    """An independent specimen operator fixture, not source-chain acceptance."""
    from temsim.physics.specimen_wave_transport import _propagate_specimen
    from temsim.physics.multiplane_wave import PlaneWave
    from temsim.physics.tip_gun_wave import TipGunCheckpoint
    from temsim.physics.wave_flux import BeamState, WaveMode
    from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE
    from temsim.physics.wave_reference import AxialWaveReference
    state = default_state()
    for collection in (state.lenses, state.stigmators, state.corrector_elements, state.deflectors):
        for component in collection:
            component.enabled = False
    state.apertures = []
    from pathlib import Path
    # This independent material fixture has an explicitly zero-field analytic
    # column plan. It retains actual CIF potential and frozen-phonon propagation.
    monkeypatch.setattr("temsim.physics.column_wave._prepare_column", _quiet_prepared_column)
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = str(Path(__file__).parent/"fixtures"/"cif"/"Si.cif")
    state.sample.inserted = True
    # Explicit test-channel inputs, not CIF-derived material calibration.
    state.sample.real_plasmon_mean_free_path_nm = 100.
    state.sample.real_plasmon_energy_ev = 16.
    state.sample.thickness_nm = .4
    state.sample.wave_slice_thickness_angstrom = 2.
    state.sample.wave_grid_pixels = 128
    state.sample.wave_field_of_view_angstrom = 30.
    state.sample.wave_frozen_phonon_enabled = True
    state.sample.wave_frozen_phonon_configurations = 2
    state.sample.wave_frozen_phonon_sigma_angstrom = .075  # explicit test displacement
    axis = (np.arange(128)-64)*2e-11
    xx, yy = np.meshgrid(axis, axis)
    amplitude = np.exp(-(xx*xx+yy*yy)/(2*(.2e-9)**2)).astype(complex)
    amplitude /= np.linalg.norm(amplitude)
    reference = AxialWaveReference(2e-8, 1e-22)
    mode = WaveMode(PlaneWave(amplitude, np.eye(2)*2e-11, np.zeros(2)), .4, TIP_REFERENCE, "specimen-fixture", 300., reference)
    checkpoint = TipGunCheckpoint(BeamState((mode,), TIP_REFERENCE), state.sample.z_mm-state.sample.thickness_nm*.5e-6,
        1e-9, {"fixture": "independent physical specimen operator; no source-chain qualification"})
    result = _propagate_specimen(state, checkpoint)
    assert len(result.beam.modes) == 2
    assert 0 < result.beam.total_weight < .4
    assert result.record["potential"]["atomistic_applied"]
    assert result.record["potential"]["frozen_phonon_applied"]
    assert all("phonon:" in m.mode_id for m in result.beam.modes)
    from temsim.physics.tip_gun_wave import _momentum_velocity
    momentum, velocity = _momentum_velocity(300000.)
    for m, row in zip(result.beam.modes, result.record["modes"]):
        assert m.axial_reference.flight_time_s-reference.flight_time_s == pytest.approx(.4e-9/velocity, rel=2e-6, abs=1e-23)
        assert m.axial_reference.longitudinal_action_j_s-reference.longitudinal_action_j_s == pytest.approx(.4e-9*momentum, rel=2e-6, abs=1e-38)
        balance = row["probability_balance"]
        assert balance["accounted_weight"] == pytest.approx(row["input_weight"], abs=1e-13)
        assert balance["first_events"]["real_plasmon"] > 0
    assert all(m.plane.amplitude.imag.max() > 0 for m in result.beam.modes)
    assert result.plane_z_mm == pytest.approx(state.sample.z_mm+state.sample.thickness_nm*.5e-6)


def _install_filter(state):
    from temsim.assembly_catalog import AssemblyCatalog
    catalog = AssemblyCatalog()
    catalog.apply(state, replace(catalog.default_selection(), recording="Energy Filter"))
    assert state.energy_filter_installed


@pytest.mark.parametrize("z", (None, True, float("nan"), float("inf"), "500", [500.]))
def test_virtual_plane_requires_an_explicit_finite_numeric_z(z):
    with pytest.raises(ValueError, match="finite numeric"):
        TipWaveRequest(stop="plane", observation_z_mm=z).validate()


def test_virtual_plane_request_does_not_mix_with_named_stages_or_detectors():
    with pytest.raises(ValueError, match="only used"):
        TipWaveRequest(stop="gun_exit", observation_z_mm=500.).validate()
    for options in ({"detector_key": "camera"}, {"detector_keys": ("bf",)}):
        with pytest.raises(ValueError, match="cannot also select"):
            TipWaveRequest(stop="plane", observation_z_mm=500., **options).validate()


def _quiet_observation_state():
    """Independent column/control fixture; does not qualify a tip producer."""
    state = default_state()
    for collection in (state.lenses, state.stigmators, state.corrector_elements, state.deflectors):
        for component in collection:
            component.enabled = False
    for plane in state.recording_planes:
        plane.inserted = False
    state.apertures = []
    return state


def _fixture_tip_checkpoint(z_mm, *, narrow=False):
    """Executed-input test double, never an admitted configurable source."""
    from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE
    from temsim.physics.multiplane_wave import PlaneWave
    from temsim.physics.tip_gun_wave import TipGunCheckpoint
    from temsim.physics.wave_flux import BeamState, WaveMode
    from temsim.physics.wave_reference import AxialWaveReference
    if narrow:
        n, step, sigma = 256, 5e-10, 5e-9
    else:
        # Preserve physical width and spacing, but resolve the periodic-domain
        # boundary. At 64 cells this Gaussian has ~4.6e-10 probability above
        # 0.8 Nyquist from its nonzero edge, exceeding the 1e-12 spectral guard.
        # A 128-cell window puts that artificial tail below 5e-28; reducing
        # sigma or weakening phase expansion would change the intended case.
        n, step, sigma = 128, 1e-6, 6e-6
    x = (np.arange(n)-n//2)*step
    a = np.exp(-(x[:, None]**2+x[None, :]**2)/(4*sigma**2)).astype(complex)*np.exp(.37j)
    a /= np.linalg.norm(a)
    plane = PlaneWave(a, np.eye(2)*step, np.array((2e-9, -3e-9)),
                      tilt_rad=np.array((2e-4, -1e-4)) if narrow else np.zeros(2))
    mode = WaveMode(plane, .4, TIP_REFERENCE, "isolated-fixture:0", 300., AxialWaveReference(2e-8, 1e-22))
    return TipGunCheckpoint(BeamState((mode,), TIP_REFERENCE), z_mm, 1e-9,
        {"fixture": "independent executed-input mathematics/control; not full tip-chain qualification"})


def test_continuation_gaussian_fixture_resolves_its_periodic_boundary():
    """A larger domain fixes Gaussian truncation without relaxing sampling."""
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
    from temsim.physics.wave_grid import WaveSamplingError
    wave = _fixture_tip_checkpoint(1.).beam.modes[0].plane
    assert wave.amplitude.shape == (128, 128)
    np.testing.assert_array_equal(wave.basis_m, np.eye(2)*1e-6)
    # The central subdomain is the original 64-cell Gaussian, up to a scalar
    # normalisation which cancels from the fractional spectral probability.
    truncated = replace(wave, amplitude=wave.amplitude[32:96, 32:96])
    def high_frequency_fraction(amplitude):
        spectrum = abs(np.fft.fft2(amplitude))**2
        frequency = abs(2*np.pi*np.fft.fftfreq(amplitude.shape[0]))
        return spectrum[frequency >= .8*np.pi, :].sum()/spectrum.sum()
    assert high_frequency_fraction(truncated.amplitude) > 1e-12
    assert high_frequency_fraction(wave.amplitude) < 1e-12
    wavelength = float(wavelength_m(300000.))
    with pytest.raises(WaveSamplingError, match="undersampled"):
        truncated.full_amplitude(wavelength)
    np.testing.assert_array_equal(wave.full_amplitude(wavelength), wave.amplitude)


def _quiet_prepared_column(_state, start, stop, maximum_step_mm):
    """Analytic vacuum plan; actual complex column integrator runs on it."""
    from types import SimpleNamespace
    count = max(1, int(np.ceil((stop-start)/maximum_step_mm)))
    z = np.linspace(start, stop, count+1)
    zeros = np.zeros(count)
    plan = SimpleNamespace(z_mm=z, step_m=np.diff(z)*1e-3,
        midpoint_magnetic_t=zeros, midpoint_sx_m2=zeros, midpoint_sy_m2=zeros,
        midpoint_sxy_m2=zeros, midpoint_hex_normal_m3=zeros, midpoint_hex_skew_m3=zeros,
        kick_x_rad=np.zeros(count+1), kick_y_rad=np.zeros(count+1),
        cs_kick_m3=np.zeros(count+1), electric_field=None,
        signature=f"analytic-vacuum-fixture:{start:.17g}:{stop:.17g}:{maximum_step_mm:.17g}")
    return plan, np.full(count+1, np.inf), {}, []


def _observation_request(tmp_path, z, *, segmented=False, **options):
    return TipWaveRequest(stop="plane", observation_z_mm=z, column_step_mm=.1,
        execution=WaveExecutionOptions(segmented=segmented, segment_steps=2,
            cache_directory=str(tmp_path/"waves"), maximum_disk_cache_bytes=1<<28), **options)


def test_virtual_plane_inside_gun_is_rejected_before_execution(monkeypatch, tmp_path):
    import temsim.physics.tip_wave_pipeline as pipeline
    state = _quiet_observation_state()
    monkeypatch.setattr(pipeline, "build_tip_gun_checkpoint", lambda *_a, **_kw: pytest.fail("Unsupported gun plane must not execute"))
    with pytest.raises(ValueError, match="inside the gun"):
        simulate_tip_wave(state, _observation_request(tmp_path, state.electron_gun.exit_plane_z_mm-1.))
    assert not (tmp_path/"waves").exists()


def test_virtual_material_internal_plane_requires_truncated_operator(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from temsim.specimen.scene import SpecimenScene
    import temsim.physics.tip_wave_pipeline as pipeline
    state = _quiet_observation_state()
    monkeypatch.setattr(SpecimenScene, "from_state", lambda *_a: SimpleNamespace(is_vacuum=False))
    monkeypatch.setattr(pipeline, "build_tip_gun_checkpoint", lambda *_a, **_kw: pytest.fail("Unsupported specimen plane must not execute"))
    with pytest.raises(ValueError, match="inside the specimen"):
        simulate_tip_wave(state, _observation_request(tmp_path, state.sample.z_mm))


@pytest.mark.parametrize("offset", (0., 1.))
def test_virtual_plane_cannot_cross_installed_filter(monkeypatch, tmp_path, offset):
    import temsim.physics.tip_wave_pipeline as pipeline
    # Install against a complete physical assembly before any fixture removes
    # apertures. The filter rejects before column or gun execution.
    state = default_state()
    _install_filter(state)
    monkeypatch.setattr(pipeline, "build_tip_gun_checkpoint", lambda *_a, **_kw: pytest.fail("Filter cannot be bypassed"))
    with pytest.raises(ValueError, match="energy filter requires"):
        simulate_tip_wave(state, _observation_request(tmp_path, state.energy_filter.entrance_z_mm+offset))


def test_virtual_plane_rejects_unknown_post_gun_electric_provider(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import temsim.physics.tip_wave_pipeline as pipeline
    state = _quiet_observation_state()
    def prepared(*args):
        plan, radii, stops, owners = _quiet_prepared_column(*args)
        plan.electric_field = SimpleNamespace(is_constant_on_interval=lambda *_a: False)
        return plan, radii, stops, owners
    monkeypatch.setattr(pipeline, "_prepare_column", prepared)
    monkeypatch.setattr(pipeline, "build_tip_gun_checkpoint", lambda *_a, **_kw: pytest.fail("Unsupported E must not execute gun"))
    with pytest.raises(ValueError, match="Non-polynomial electric maps"):
        simulate_tip_wave(state, _observation_request(tmp_path, state.electron_gun.exit_plane_z_mm+.1))


def test_virtual_plane_does_not_omit_unknown_electric_field_inside_material(monkeypatch, tmp_path):
    """Control fixture checks the same electric gate through material slices."""
    from types import SimpleNamespace
    from temsim.specimen.scene import SpecimenScene
    import temsim.physics.tip_wave_pipeline as pipeline
    state = _quiet_observation_state()
    entrance = state.sample.z_mm-state.sample.thickness_nm*.5e-6
    exit_z = state.sample.z_mm+state.sample.thickness_nm*.5e-6
    monkeypatch.setattr(SpecimenScene, "from_state", lambda *_a: SimpleNamespace(is_vacuum=False))
    def prepared(_state, start, stop, step):
        plan, radii, stops, owners = _quiet_prepared_column(_state, start, stop, step)
        if start == entrance and stop == exit_z:
            plan.electric_field = SimpleNamespace(is_constant_on_interval=lambda *_a: False)
        return plan, radii, stops, owners
    monkeypatch.setattr(pipeline, "_prepare_column", prepared)
    monkeypatch.setattr(pipeline, "build_tip_gun_checkpoint", lambda *_a, **_kw: pytest.fail("Unimplemented specimen E must not execute gun"))
    with pytest.raises(ValueError, match="Non-polynomial electric maps"):
        simulate_tip_wave(state, _observation_request(tmp_path, exit_z+.1))


def test_virtual_gun_exit_uses_only_executed_tip_checkpoint(monkeypatch, tmp_path):
    import temsim.physics.tip_wave_pipeline as pipeline
    state = _quiet_observation_state()
    checkpoint = _fixture_tip_checkpoint(state.electron_gun.exit_plane_z_mm)
    monkeypatch.setattr(pipeline, "build_tip_gun_checkpoint", lambda *_a, **_kw: checkpoint)
    monkeypatch.setattr(pipeline, "_prepare_column", lambda *_a: pytest.fail("Gun-exit observation must not transport a column"))
    result = simulate_tip_wave(state, _observation_request(tmp_path, checkpoint.plane_z_mm), use_cache=False)
    assert result.checkpoint is checkpoint
    assert result.detector is None and result.detector_readouts == ()
    assert result.checkpoint.beam.reference_plane == checkpoint.beam.reference_plane


def test_virtual_plane_actual_column_matches_complex_analytic_gaussian_drift(monkeypatch, tmp_path):
    """Isolated physical vacuum reference, not microscope/source qualification."""
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
    import temsim.physics.tip_wave_pipeline as pipeline
    state = _quiet_observation_state()
    original = _fixture_tip_checkpoint(state.electron_gun.exit_plane_z_mm, narrow=True)
    monkeypatch.setattr(pipeline, "build_tip_gun_checkpoint", lambda *_a, **_kw: original)
    monkeypatch.setattr(pipeline, "_prepare_column", _quiet_prepared_column)
    from temsim.physics import column_wave
    monkeypatch.setattr(column_wave, "_prepare_column", _quiet_prepared_column)
    target = original.plane_z_mm+.001234567
    result = simulate_tip_wave(state, _observation_request(tmp_path, target), use_cache=False)
    old, mode = original.beam.modes[0], result.checkpoint.beam.modes[0]
    wave, lam, sigma = mode.plane, float(wavelength_m(300000.)), 5e-9
    distance = (target-original.plane_z_mm)*1e-3
    local = wave.coordinates_m()-wave.origin_m[:, None, None]
    precision = 1j*lam/(4*np.pi*sigma**2)
    factor = 1+distance*precision
    centre = old.plane.amplitude.shape[0]//2
    expected = (old.plane.amplitude[centre, centre]/factor
        *np.exp(1j*np.pi/lam*(precision/factor)*np.sum(local**2, axis=0))
        *np.exp(2j*np.pi/lam*np.einsum("i,iyx->yx", old.plane.tilt_rad, local))
        *np.exp(1j*np.pi/lam*distance*np.dot(old.plane.tilt_rad, old.plane.tilt_rad)))
    expected *= np.sqrt(abs(np.linalg.det(wave.basis_m)/np.linalg.det(old.plane.basis_m)))
    assert result.checkpoint.plane_z_mm == target
    assert mode.mode_id == old.mode_id
    assert mode.weight_per_reference_electron == pytest.approx(old.weight_per_reference_electron, abs=1e-12)
    assert np.linalg.norm(wave.full_amplitude(lam)-expected) < 2e-8
    assert mode.axial_reference.flight_time_s > old.axial_reference.flight_time_s


@pytest.mark.parametrize("segmented", (False, True))
def test_virtual_plane_absorbs_all_prior_detectors_even_with_readout_disabled(monkeypatch, tmp_path, segmented):
    """Calling-chain fixture with the real physical stop kernel and wave cache."""
    import temsim.physics.tip_wave_pipeline as pipeline
    from temsim.physics import column_wave
    state = _quiet_observation_state()
    start = state.electron_gun.exit_plane_z_mm
    plane = state.camera
    plane.inserted = True
    plane.readout_enabled = False
    plane.z_mm, plane.outer_width_mm = start+.1, .01
    original = _fixture_tip_checkpoint(start)
    calls = []
    def producer(*_a, **_kw):
        calls.append("gun")
        return original
    monkeypatch.setattr(pipeline, "build_tip_gun_checkpoint", producer)
    monkeypatch.setattr(pipeline, "_prepare_column", _quiet_prepared_column)
    monkeypatch.setattr(column_wave, "_prepare_column", _quiet_prepared_column)
    request = _observation_request(tmp_path, start+.2, segmented=segmented)
    first = simulate_tip_wave(state, request)
    second = simulate_tip_wave(state, request)
    assert calls == ["gun"]
    assert second.propagation_cache_hit
    assert 0 < first.checkpoint.beam.total_weight < original.beam.total_weight
    assert first.checkpoint.beam.total_weight == pytest.approx(second.checkpoint.beam.total_weight, rel=1e-12)
    assert first.checkpoint.plane_z_mm == start+.2
    assert first.detector is None and first.detector_readouts == ()
    assert first.checkpoint.beam.modes[0].mode_id == original.beam.modes[0].mode_id


def test_virtual_plane_coincident_with_detector_returns_incident_state(monkeypatch, tmp_path):
    import temsim.physics.tip_wave_pipeline as pipeline
    from temsim.physics import column_wave
    state = _quiet_observation_state()
    original = _fixture_tip_checkpoint(state.electron_gun.exit_plane_z_mm)
    state.camera.inserted = True
    state.camera.z_mm = original.plane_z_mm+.1
    monkeypatch.setattr(pipeline, "build_tip_gun_checkpoint", lambda *_a, **_kw: original)
    monkeypatch.setattr(pipeline, "_prepare_column", _quiet_prepared_column)
    monkeypatch.setattr(column_wave, "_prepare_column", _quiet_prepared_column)
    monkeypatch.setattr(pipeline, "_apply_recording_stop", lambda *_a: pytest.fail("The coincident virtual plane is incident, not absorbing"))
    result = simulate_tip_wave(state, _observation_request(tmp_path, state.camera.z_mm), use_cache=False)
    assert result.checkpoint.beam.total_weight == pytest.approx(original.beam.total_weight, abs=1e-12)


@pytest.mark.parametrize("face", ("entrance", "exit", "after"))
def test_virtual_material_boundaries_execute_the_existing_specimen_stage(monkeypatch, tmp_path, face):
    """Control-only doubles verify ownership/order, not multislice accuracy."""
    from types import SimpleNamespace
    from temsim.physics.wave_execution import InelasticWaveNumerics
    from temsim.specimen.scene import SpecimenScene
    import temsim.physics.tip_wave_pipeline as pipeline
    state = _quiet_observation_state()
    entrance = state.sample.z_mm-state.sample.thickness_nm*.5e-6
    exit_z = state.sample.z_mm+state.sample.thickness_nm*.5e-6
    original = _fixture_tip_checkpoint(state.electron_gun.exit_plane_z_mm)
    calls = []
    monkeypatch.setattr(SpecimenScene, "from_state", lambda *_a: SimpleNamespace(is_vacuum=False))
    monkeypatch.setattr(pipeline, "build_tip_gun_checkpoint", lambda *_a, **_kw: original)
    monkeypatch.setattr(pipeline, "_prepare_column", _quiet_prepared_column)
    def column(_state, upstream, stop, **_options):
        prepared = _options["_prepared"]
        assert prepared is not None
        assert prepared[0].z_mm[0] == upstream.plane_z_mm
        assert prepared[0].z_mm[-1] == stop
        assert stop <= entrance or upstream.plane_z_mm >= exit_z
        calls.append(("column", upstream.plane_z_mm, stop))
        return replace(upstream, plane_z_mm=stop,
                       record={"fixture_column": (upstream.plane_z_mm, stop), "upstream_digest": upstream.digest})
    def specimen(_state, upstream, **_options):
        calls.append(("specimen", upstream.plane_z_mm, exit_z))
        mode = upstream.beam.modes[0]
        shifted = replace(mode, plane=replace(mode.plane, amplitude=mode.plane.amplitude*1j))
        return replace(upstream, beam=replace(upstream.beam, modes=(shifted,)), plane_z_mm=exit_z,
                       record={"fixture_specimen": True, "upstream_digest": upstream.digest})
    monkeypatch.setattr(pipeline, "_propagate_column", column)
    monkeypatch.setattr(pipeline, "_propagate_specimen", specimen)
    target = {"entrance": entrance, "exit": exit_z, "after": exit_z+.2}[face]
    request = _observation_request(tmp_path, target, inelastic=InelasticWaveNumerics(method="zero_loss"))
    result = simulate_tip_wave(state, request, use_cache=False)
    assert calls[0] == ("column", original.plane_z_mm, entrance)
    assert sum(c[0] == "specimen" for c in calls) == (0 if face == "entrance" else 1)
    assert result.checkpoint.plane_z_mm == target
    expected = original.beam.modes[0].plane.amplitude*(1 if face == "entrance" else 1j)
    np.testing.assert_array_equal(result.checkpoint.beam.modes[0].plane.amplitude, expected)


def test_cancelled_virtual_result_cannot_be_published(monkeypatch, tmp_path):
    """A late cancellation never turns a completed double into a current result."""
    import temsim.physics.tip_wave_pipeline as pipeline
    state = _quiet_observation_state()
    original = _fixture_tip_checkpoint(state.electron_gun.exit_plane_z_mm)
    cancelled = [False]
    monkeypatch.setattr(pipeline, "build_tip_gun_checkpoint", lambda *_a, **_kw: original)
    monkeypatch.setattr(pipeline, "_prepare_column", _quiet_prepared_column)
    def transport(_state, upstream, stop, **_kw):
        cancelled[0] = True
        return replace(upstream, plane_z_mm=stop)
    monkeypatch.setattr(pipeline, "_propagate_column", transport)
    with pytest.raises(InterruptedError):
        simulate_tip_wave(state, _observation_request(tmp_path, original.plane_z_mm+.1),
                          use_cache=False, cancelled=lambda: cancelled[0])


def _prefix_fixture(monkeypatch):
    """Bounded actual column/stop execution; gun input remains a test double."""
    from collections import OrderedDict
    import temsim.physics.tip_wave_pipeline as pipeline
    from temsim.physics import column_wave
    state = _quiet_observation_state()
    start = state.electron_gun.exit_plane_z_mm
    state.sample.z_mm = start+.025
    original = _fixture_tip_checkpoint(start)
    gun_calls, column_calls = [], []
    monkeypatch.setattr(pipeline, "_STAGES", OrderedDict())
    monkeypatch.setattr(pipeline, "_OBSERVATION_PREFIXES", OrderedDict())
    def producer(*_a, **_kw):
        gun_calls.append("gun")
        return original
    actual_column = column_wave._propagate_column
    def transport(_state, upstream, stop, **options):
        column_calls.append((upstream.plane_z_mm, stop))
        return actual_column(_state, upstream, stop, **options)
    monkeypatch.setattr(pipeline, "build_tip_gun_checkpoint", producer)
    monkeypatch.setattr(pipeline, "_prepare_column", _quiet_prepared_column)
    monkeypatch.setattr(column_wave, "_prepare_column", _quiet_prepared_column)
    monkeypatch.setattr(pipeline, "_propagate_column", transport)
    monkeypatch.setattr(column_wave, "_propagate_column", transport)
    return state, original, gun_calls, column_calls


@pytest.mark.parametrize("segmented", (False, True))
def test_virtual_plane_resumes_nearest_completed_post_sample_observation(monkeypatch, tmp_path, segmented):
    """Actual complex drift/control fixture, not an admitted full tip chain."""
    state, original, gun_calls, calls = _prefix_fixture(monkeypatch)
    start = original.plane_z_mm
    first = simulate_tip_wave(state, _observation_request(tmp_path, start+.1, segmented=segmented))
    calls.clear()
    second_request = _observation_request(tmp_path, start+.2, segmented=segmented)
    second = simulate_tip_wave(state, second_request)
    assert gun_calls == ["gun"]
    assert second.propagation_cache_hit
    assert calls[0][0] == first.checkpoint.plane_z_mm
    assert calls[-1][1] == second_request.observation_z_mm
    assert all(a >= first.checkpoint.plane_z_mm for a, _ in calls)
    calls.clear()
    same = simulate_tip_wave(state, second_request)
    assert same.propagation_cache_hit and calls == []
    assert same.checkpoint.digest == second.checkpoint.digest
    np.testing.assert_array_equal(same.checkpoint.beam.modes[0].plane.amplitude,
                                  second.checkpoint.beam.modes[0].plane.amplitude)
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
    fresh = simulate_tip_wave(state, second_request, use_cache=False)
    cached_mode, fresh_mode = second.checkpoint.beam.modes[0], fresh.checkpoint.beam.modes[0]
    wavelength = float(wavelength_m(cached_mode.energy_kev*1000.))
    # Compare the complex fields directly; no fitted phase or normalisation.
    assert np.linalg.norm(cached_mode.plane.full_amplitude(wavelength)
                          - fresh_mode.plane.full_amplitude(wavelength)) < 2e-11
    assert cached_mode.weight_per_reference_electron == pytest.approx(
        fresh_mode.weight_per_reference_electron, abs=1e-12)
    assert cached_mode.axial_reference.flight_time_s == pytest.approx(
        fresh_mode.axial_reference.flight_time_s, rel=1e-12)


@pytest.mark.parametrize("segmented", (False, True))
def test_virtual_plane_never_resumes_from_a_downstream_observation(monkeypatch, tmp_path, segmented):
    state, original, _gun_calls, calls = _prefix_fixture(monkeypatch)
    start = original.plane_z_mm
    simulate_tip_wave(state, _observation_request(tmp_path, start+.2, segmented=segmented))
    calls.clear()
    result = simulate_tip_wave(state, _observation_request(tmp_path, start+.1, segmented=segmented))
    assert calls[0][0] == start
    assert result.checkpoint.plane_z_mm == start+.1


@pytest.mark.parametrize("segmented", (False, True))
def test_virtual_plane_prefix_at_detector_absorbs_once_when_continuing(monkeypatch, tmp_path, segmented):
    """Real stop execution checks incident and already-absorbed boundaries."""
    from temsim.detector import wave_readout
    import temsim.physics.tip_wave_pipeline as pipeline
    state, original, _gun_calls, calls = _prefix_fixture(monkeypatch)
    start = original.plane_z_mm
    state.camera.z_mm, state.camera.outer_width_mm = start+.1, .01
    state.camera.inserted = True
    state.camera.readout_enabled = False
    actual_stop, stops = wave_readout._apply_recording_stop, []
    def absorb(checkpoint, plane):
        stops.append(plane.z_mm)
        return actual_stop(checkpoint, plane)
    monkeypatch.setattr(pipeline, "_apply_recording_stop", absorb)
    monkeypatch.setattr(wave_readout, "_apply_recording_stop", absorb)
    incident = simulate_tip_wave(state, _observation_request(tmp_path, state.camera.z_mm, segmented=segmented))
    assert stops == []
    assert incident.checkpoint.beam.total_weight == pytest.approx(original.beam.total_weight, abs=1e-12)
    calls.clear()
    transmitted = simulate_tip_wave(state, _observation_request(tmp_path, start+.2, segmented=segmented))
    assert stops == [state.camera.z_mm]
    assert calls[0][0] == state.camera.z_mm
    assert 0 < transmitted.checkpoint.beam.total_weight < incident.checkpoint.beam.total_weight
    after = simulate_tip_wave(state, _observation_request(tmp_path, start+.3, segmented=segmented))
    assert stops == [state.camera.z_mm]
    assert after.checkpoint.beam.total_weight == pytest.approx(transmitted.checkpoint.beam.total_weight, rel=1e-12)


@pytest.mark.parametrize("change", ("state", "column_step", "source", "grid", "inelastic", "epoch"))
def test_virtual_plane_prefix_invalidates_consumed_input_changes(monkeypatch, tmp_path, change):
    from temsim.physics.wave_execution import InelasticWaveNumerics
    state, original, _gun_calls, calls = _prefix_fixture(monkeypatch)
    start = original.plane_z_mm
    request = _observation_request(tmp_path, start+.1)
    simulate_tip_wave(state, request)
    calls.clear()
    next_request = replace(request, observation_z_mm=start+.2)
    if change == "state":
        state.lenses[0].percent += .1
    elif change == "column_step":
        next_request = replace(next_request, column_step_mm=.05)
    elif change == "source":
        next_request = replace(next_request, source=replace(request.source, energy_samples=1))
    elif change == "grid":
        next_request = replace(next_request, wave_grid=replace(request.wave_grid, maximum_pixels=16384))
    elif change == "inelastic":
        next_request = replace(next_request, inelastic=InelasticWaveNumerics(seed=42))
    else:
        next_request = replace(next_request, tip_time_s=1e-8)
    simulate_tip_wave(state, next_request)
    assert calls[0][0] == start


def test_virtual_plane_disk_prefix_checks_array_corruption_before_reuse(monkeypatch, tmp_path):
    """Even an exact-Z observation validates the existing committed arrays."""
    from pathlib import Path
    import temsim.physics.tip_wave_pipeline as pipeline
    state, original, _gun_calls, calls = _prefix_fixture(monkeypatch)
    request = _observation_request(tmp_path, original.plane_z_mm+.1, segmented=True)
    completed = simulate_tip_wave(state, request)
    prefixes = tuple(pipeline._OBSERVATION_PREFIXES)
    modes = completed.checkpoint.beam.modes
    path = Path(modes.root)/modes.rows[0]["arrays"]["amplitude"]["file"]
    path.write_bytes(path.read_bytes()+b"checksum regression")
    calls.clear()
    with pytest.raises(ValueError, match="checksum"):
        simulate_tip_wave(state, request)
    assert calls == []
    assert tuple(pipeline._OBSERVATION_PREFIXES) == prefixes


@pytest.mark.parametrize("segmented", (False, True))
def test_failed_virtual_continuation_does_not_register_an_observation(monkeypatch, tmp_path, segmented):
    import temsim.physics.tip_wave_pipeline as pipeline
    from temsim.physics import column_wave
    state, original, _gun_calls, _calls = _prefix_fixture(monkeypatch)
    simulate_tip_wave(state, _observation_request(tmp_path, original.plane_z_mm+.1, segmented=segmented))
    before = tuple(pipeline._OBSERVATION_PREFIXES.items())
    def fail(*_a, **_kw):
        raise InterruptedError("late continuation cancellation fixture")
    monkeypatch.setattr(pipeline, "_propagate_column", fail)
    monkeypatch.setattr(column_wave, "_propagate_column", fail)
    with pytest.raises(InterruptedError):
        simulate_tip_wave(state, _observation_request(tmp_path, original.plane_z_mm+.2, segmented=segmented))
    assert tuple(pipeline._OBSERVATION_PREFIXES.items()) == before


def test_virtual_observation_prefix_index_is_bounded_and_ram_budgeted(monkeypatch, tmp_path):
    import temsim.physics.tip_wave_pipeline as pipeline
    state, original, _gun_calls, _calls = _prefix_fixture(monkeypatch)
    monkeypatch.setattr(pipeline, "_MAXIMUM_OBSERVATION_PREFIXES", 2)
    for offset in (.1, .2, .3):
        simulate_tip_wave(state, _observation_request(tmp_path, original.plane_z_mm+offset))
    assert len(pipeline._OBSERVATION_PREFIXES) == 2
    assert {key[1] for key in pipeline._OBSERVATION_PREFIXES} == {
        original.plane_z_mm+.2, original.plane_z_mm+.3}
    pipeline._OBSERVATION_PREFIXES.clear()
    request = _observation_request(tmp_path, original.plane_z_mm+.4)
    request = replace(request, execution=replace(request.execution, maximum_ram_cache_bytes=1))
    simulate_tip_wave(state, request)
    assert pipeline._OBSERVATION_PREFIXES == {}


@pytest.mark.parametrize("segmented", (False, True))
def test_pre_sample_virtual_plane_reuses_executed_upstream_state(monkeypatch, tmp_path, segmented):
    """An upstream observation is a real continuation, not a new source."""
    state, original, gun_calls, calls = _prefix_fixture(monkeypatch)
    start = original.plane_z_mm
    state.sample.z_mm = start+1.
    first = simulate_tip_wave(state, _observation_request(tmp_path, start+.1, segmented=segmented))
    calls.clear()
    result = simulate_tip_wave(state, _observation_request(tmp_path, start+.2, segmented=segmented))
    assert gun_calls == ["gun"]
    assert calls[0][0] == first.checkpoint.plane_z_mm
    assert result.propagation_cache_hit
    assert result.checkpoint.plane_z_mm < state.sample.z_mm


def test_backward_unvisited_plane_reuses_nearest_committed_column_segment(monkeypatch, tmp_path):
    """Actual vacuum complex propagation starts before Z, never runs backwards."""
    import temsim.physics.tip_wave_pipeline as pipeline
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
    state, original, gun_calls, calls = _prefix_fixture(monkeypatch)
    start = original.plane_z_mm
    simulate_tip_wave(state, _observation_request(tmp_path, start+.8, segmented=True))
    target = start+.5
    candidates = [key[1] for key in pipeline._OBSERVATION_PREFIXES if key[1] <= target]
    nearest = max(candidates)
    assert start < nearest < target
    calls.clear()
    request = _observation_request(tmp_path, target, segmented=True)
    result = simulate_tip_wave(state, request)
    assert gun_calls == ["gun"] and result.propagation_cache_hit
    assert calls[0][0] == nearest
    assert all(nearest <= lower < upper <= target for lower, upper in calls)
    fresh = simulate_tip_wave(state, request, use_cache=False)
    cached_mode, fresh_mode = result.checkpoint.beam.modes[0], fresh.checkpoint.beam.modes[0]
    wavelength = float(wavelength_m(cached_mode.energy_kev*1000.))
    assert np.linalg.norm(cached_mode.plane.full_amplitude(wavelength)
                          - fresh_mode.plane.full_amplitude(wavelength)) < 2e-11
    assert cached_mode.weight_per_reference_electron == pytest.approx(
        fresh_mode.weight_per_reference_electron, abs=1e-12)
    assert cached_mode.axial_reference.flight_time_s == pytest.approx(
        fresh_mode.axial_reference.flight_time_s, rel=1e-12)


def test_verified_prefix_only_admits_remaining_column_span(monkeypatch, tmp_path):
    import temsim.physics.tip_wave_pipeline as pipeline
    state, original, _gun_calls, _calls = _prefix_fixture(monkeypatch)
    start = original.plane_z_mm
    first = simulate_tip_wave(state, _observation_request(tmp_path, start+.1, segmented=True))
    admitted = []
    def prepare(_state, lower, upper, step):
        admitted.append((lower, upper))
        return _quiet_prepared_column(_state, lower, upper, step)
    monkeypatch.setattr(pipeline, "_prepare_column", prepare)
    request = _observation_request(tmp_path, start+.2, segmented=True)
    simulate_tip_wave(state, request)
    assert admitted and all(lower >= first.checkpoint.plane_z_mm for lower, _ in admitted)
    admitted.clear()
    simulate_tip_wave(state, request)
    assert admitted == []


def test_backward_detector_plane_keeps_incident_checkpoint(monkeypatch, tmp_path):
    """Intermediate checkpoint discovery must not substitute transmitted flux."""
    state, original, _gun_calls, calls = _prefix_fixture(monkeypatch)
    start = original.plane_z_mm
    state.camera.z_mm, state.camera.outer_width_mm = start+.1, .01
    state.camera.inserted = True
    state.camera.readout_enabled = False
    downstream = simulate_tip_wave(state, _observation_request(tmp_path, start+.8, segmented=True))
    assert downstream.checkpoint.beam.total_weight < original.beam.total_weight
    calls.clear()
    incident = simulate_tip_wave(state, _observation_request(tmp_path, state.camera.z_mm, segmented=True))
    assert calls == []
    assert incident.checkpoint.beam.total_weight == pytest.approx(original.beam.total_weight, abs=1e-12)


def test_cancelled_observation_keeps_only_completed_intermediate_segments(monkeypatch, tmp_path):
    import temsim.physics.tip_wave_pipeline as pipeline
    state, original, gun_calls, calls = _prefix_fixture(monkeypatch)
    start = original.plane_z_mm
    cancelled = [False]
    def progress(_done, _total, label):
        if label.startswith("Column segment saved"):
            cancelled[0] = True
    with pytest.raises(InterruptedError):
        simulate_tip_wave(state, _observation_request(tmp_path, start+.8, segmented=True),
                          cancelled=lambda: cancelled[0], progress_callback=progress)
    endpoints = [key[1] for key in pipeline._OBSERVATION_PREFIXES]
    assert start < max(endpoints) < start+.8
    committed_z = max(endpoints)
    calls.clear()
    result = simulate_tip_wave(state, _observation_request(tmp_path, committed_z+.05, segmented=True))
    assert gun_calls == ["gun"] and result.propagation_cache_hit
    assert calls[0][0] == committed_z


def test_live_observation_session_captures_once_and_keeps_input_state_independent(monkeypatch, tmp_path):
    """Actual complex drift uses a fixed captured tip, never the later live edits."""
    import temsim.physics.tip_wave_pipeline as pipeline
    from temsim.instrument_snapshot import InstrumentSnapshot
    state, original, gun_calls, calls = _prefix_fixture(monkeypatch)
    start = original.plane_z_mm
    captures, restores, identities = [], [], []
    actual_capture = pipeline.capture_instrument_snapshot
    actual_restore = InstrumentSnapshot.restore
    actual_identity = InstrumentSnapshot.digest.fget
    def capture(value):
        captures.append(value)
        return actual_capture(value)
    def restore(snapshot):
        restores.append(snapshot)
        return actual_restore(snapshot)
    def digest(snapshot):
        identities.append(snapshot)
        return actual_identity(snapshot)
    monkeypatch.setattr(pipeline, "capture_instrument_snapshot", capture)
    monkeypatch.setattr(InstrumentSnapshot, "restore", restore)
    monkeypatch.setattr(InstrumentSnapshot, "digest", property(digest))
    request = _observation_request(tmp_path, start+.8, segmented=True)
    session = pipeline.TipWaveObservationSession(state, request)
    assert captures == restores == identities == []
    first = session.observe(start+.8)
    from temsim.physics.scan_preparation import scan_drive_identity
    assert first.resolved_scan_identity == scan_drive_identity(state)
    state.lenses[0].percent += .5
    state.sample.z_mm += 1.
    calls.clear()
    second = session.observe(start+.5)
    assert len(captures) == len(restores) == len(identities) == 1
    assert first.instrument_digest == second.instrument_digest
    assert first.resolved_scan_identity == second.resolved_scan_identity
    assert gun_calls == ["gun"] and second.propagation_cache_hit
    assert start < calls[0][0] < start+.5
    assert second.request.observation_z_mm == start+.5
    assert session.observe(start+.5).checkpoint.digest == second.checkpoint.digest
    assert len(captures) == len(restores) == len(identities) == 1


def test_live_session_rechecks_source_before_reusing_an_exact_z(monkeypatch, tmp_path):
    import temsim.physics.tip_wave_pipeline as pipeline
    from temsim import calculation_manifest
    state, original, _gun_calls, calls = _prefix_fixture(monkeypatch)
    request = _observation_request(tmp_path, original.plane_z_mm+.1, segmented=True)
    session = pipeline.TipWaveObservationSession(state, request)
    session.observe(request.observation_z_mm)
    calls.clear()
    monkeypatch.setattr(calculation_manifest, "solver_source_identity", lambda: "modified-source")
    with pytest.raises(ValueError, match="Solver implementation changed"):
        session.observe(request.observation_z_mm)
    assert calls == []


def test_live_session_rejects_external_change_and_keeps_checkpoint_unpublished(monkeypatch, tmp_path):
    import temsim.physics.tip_wave_pipeline as pipeline
    state, original, _gun_calls, calls = _prefix_fixture(monkeypatch)
    path = tmp_path / "external-field.dat"
    path.write_bytes(b"original input")
    state.lens_field_map_descriptors = {"objective_lens": {"source_path": str(path)}}
    request = _observation_request(tmp_path, original.plane_z_mm+.1, segmented=True)
    session = pipeline.TipWaveObservationSession(state, request)
    session.observe(request.observation_z_mm)
    before = tuple(pipeline._OBSERVATION_PREFIXES.items())
    calls.clear()
    path.write_bytes(b"modified input")
    with pytest.raises(ValueError, match="Changed"):
        session.observe(request.observation_z_mm+.1)
    assert calls == []
    assert tuple(pipeline._OBSERVATION_PREFIXES.items()) == before


def test_live_observation_session_rejects_parallel_use_and_releases_after_failure(monkeypatch, tmp_path):
    import temsim.physics.tip_wave_pipeline as pipeline
    state, original, _gun_calls, _calls = _prefix_fixture(monkeypatch)
    request = _observation_request(tmp_path, original.plane_z_mm+.1)
    session = pipeline.TipWaveObservationSession(state, request)
    actual_execute = pipeline._execute_tip_wave
    def interrupted(*_args, **_options):
        with pytest.raises(RuntimeError, match="Only one observation"):
            session.observe(request.observation_z_mm)
        raise InterruptedError("controlled live cancellation")
    monkeypatch.setattr(pipeline, "_execute_tip_wave", interrupted)
    with pytest.raises(InterruptedError, match="controlled live cancellation"):
        session.observe(request.observation_z_mm)
    monkeypatch.setattr(pipeline, "_execute_tip_wave", actual_execute)
    assert session.observe(request.observation_z_mm).checkpoint.plane_z_mm == request.observation_z_mm


def test_live_observation_session_rejects_invalid_targets_before_snapshot_capture(monkeypatch, tmp_path):
    import temsim.physics.tip_wave_pipeline as pipeline
    state = _quiet_observation_state()
    request = _observation_request(tmp_path, state.electron_gun.exit_plane_z_mm)
    session = pipeline.TipWaveObservationSession(state, request)
    monkeypatch.setattr(pipeline, "capture_instrument_snapshot", lambda *_a:
                        pytest.fail("Cancelled/invalid live observation must not capture inputs"))
    for target in (True, None, float("nan")):
        with pytest.raises(ValueError, match="finite numeric"):
            session.observe(target)
    with pytest.raises(InterruptedError):
        session.observe(request.observation_z_mm, cancelled=lambda: True)
    with pytest.raises(ValueError, match="stop='plane'"):
        pipeline.TipWaveObservationSession(state, replace(request, stop="gun_exit", observation_z_mm=None))


def test_live_session_real_field_plan_matches_fresh_column_execution(monkeypatch, tmp_path):
    """Real local lens/E-field providers; gun production alone is a test double."""
    from collections import OrderedDict
    import temsim.physics.tip_wave_pipeline as pipeline
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
    from temsim.physics.column_wave import _prepare_column
    from temsim.optics.model import Gaussian
    state = _quiet_observation_state()
    start = state.electron_gun.exit_plane_z_mm
    lens = state.lenses[0]
    lens.enabled = True
    lens.z_mm, lens.a_mm, lens.b0_t, lens.percent = start+.125, .125, .002, 100.
    lens.gaussian = [Gaussian(1., 0., 1.)]
    state.step_mm = .03125
    original = _fixture_tip_checkpoint(start)
    monkeypatch.setattr(pipeline, "_STAGES", OrderedDict())
    monkeypatch.setattr(pipeline, "_OBSERVATION_PREFIXES", OrderedDict())
    monkeypatch.setattr(pipeline, "build_tip_gun_checkpoint", lambda *_a, **_k: original)
    request = replace(_observation_request(tmp_path, start+.25, segmented=True), column_step_mm=.03125)
    prepared = _prepare_column(state, start, start+.375, request.column_step_mm)
    assert np.max(abs(prepared[0].midpoint_magnetic_t)) > 0.
    assert prepared[0].electric_field is not None
    session = pipeline.TipWaveObservationSession(state, request)
    session.observe(start+.25)
    continued = session.observe(start+.375)
    fresh = pipeline.simulate_tip_wave(state, replace(request, observation_z_mm=start+.375), use_cache=False)
    actual, expected = continued.checkpoint.beam.modes[0], fresh.checkpoint.beam.modes[0]
    wavelength = float(wavelength_m(actual.energy_kev*1000.))
    np.testing.assert_allclose(actual.plane.basis_m, expected.plane.basis_m, rtol=1e-12, atol=1e-18)
    assert np.linalg.norm(actual.plane.full_amplitude(wavelength)
                          - expected.plane.full_amplitude(wavelength)) < 2e-10
    assert actual.weight_per_reference_electron == pytest.approx(expected.weight_per_reference_electron, rel=1e-12)
    assert actual.energy_kev == pytest.approx(expected.energy_kev, rel=1e-14)
    assert actual.axial_reference.flight_time_s == pytest.approx(expected.axial_reference.flight_time_s, rel=1e-12)


@pytest.mark.parametrize("segmented", (False, True))
def test_virtual_column_reuses_admitted_exact_plan_with_identical_complex_field(monkeypatch, tmp_path, segmented):
    """Real weak-lens/E-field column; the gun producer is an isolated fixture."""
    import temsim.physics.tip_wave_pipeline as pipeline
    from temsim.physics import column_wave
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
    from temsim.optics.model import Gaussian
    state = _quiet_observation_state()
    start = state.electron_gun.exit_plane_z_mm
    target = start+.25
    lens = state.lenses[0]
    lens.enabled = True
    lens.z_mm, lens.a_mm, lens.b0_t, lens.percent = start+.125, .125, .002, 100.
    lens.gaussian = [Gaussian(1., 0., 1.)]
    state.step_mm = .03125
    original = _fixture_tip_checkpoint(start)
    monkeypatch.setattr(pipeline, "build_tip_gun_checkpoint", lambda *_a, **_k: original)
    prepares = []
    prepare = column_wave._prepare_column
    def counted_prepare(working, lower, upper, step):
        prepares.append((lower, upper))
        return prepare(working, lower, upper, step)
    monkeypatch.setattr(pipeline, "_prepare_column", counted_prepare)
    monkeypatch.setattr(column_wave, "_prepare_column", counted_prepare)
    request = replace(_observation_request(tmp_path, target, segmented=segmented), column_step_mm=.03125)
    reused = simulate_tip_wave(state, request, use_cache=False)
    assert prepares == [(start, target)]

    # Execute the same input/steps with the old separate preparation path.
    name = "_propagate_column_segmented" if segmented else "_propagate_column"
    transport = getattr(pipeline, name)
    def separately_prepared(working, upstream, stop, **options):
        assert options.pop("_prepared") is not None
        return transport(working, upstream, stop, **options)
    monkeypatch.setattr(pipeline, name, separately_prepared)
    prepares.clear()
    fresh = simulate_tip_wave(state, request, use_cache=False)
    assert prepares == [(start, target), (start, target)]
    actual, expected = reused.checkpoint.beam.modes[0], fresh.checkpoint.beam.modes[0]
    wavelength = float(wavelength_m(actual.energy_kev*1000.))
    np.testing.assert_allclose(actual.plane.basis_m, expected.plane.basis_m, rtol=1e-12, atol=1e-18)
    assert np.linalg.norm(actual.plane.full_amplitude(wavelength)-expected.plane.full_amplitude(wavelength)) < 2e-10
    assert actual.weight_per_reference_electron == pytest.approx(expected.weight_per_reference_electron, rel=1e-12)
    assert actual.energy_kev == pytest.approx(expected.energy_kev, rel=1e-14)
    assert actual.axial_reference == expected.axial_reference


@pytest.mark.parametrize("segmented", (False, True))
def test_admitted_plan_cannot_cross_a_detector_absorption_boundary(monkeypatch, tmp_path, segmented):
    """Real complex propagation and stop kernel on an analytic vacuum plan."""
    import temsim.physics.tip_wave_pipeline as pipeline
    from temsim.physics import column_wave
    from temsim.detector.wave_readout import _apply_recording_stop
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
    state = _quiet_observation_state()
    start = state.electron_gun.exit_plane_z_mm
    target = start+.2
    detector = state.camera
    detector.inserted, detector.readout_enabled = True, False
    detector.z_mm, detector.outer_width_mm = start+.1, .01
    original = _fixture_tip_checkpoint(start)
    monkeypatch.setattr(pipeline, "build_tip_gun_checkpoint", lambda *_a, **_k: original)
    prepares = []
    def counted_prepare(working, lower, upper, step):
        prepares.append((lower, upper))
        return _quiet_prepared_column(working, lower, upper, step)
    monkeypatch.setattr(pipeline, "_prepare_column", counted_prepare)
    monkeypatch.setattr(column_wave, "_prepare_column", counted_prepare)
    name = "_propagate_column_segmented" if segmented else "_propagate_column"
    transport = getattr(pipeline, name)
    def split_transport(working, upstream, stop, **options):
        # The whole preflight span contains this detector. Neither physical
        # interval may borrow that plan, even though it has the same fields.
        assert options["_prepared"] is None
        return transport(working, upstream, stop, **options)
    monkeypatch.setattr(pipeline, name, split_transport)
    result = simulate_tip_wave(state, _observation_request(tmp_path, target, segmented=segmented), use_cache=False)
    assert prepares == [(start, target), (start, detector.z_mm), (detector.z_mm, target)]
    incident = column_wave._propagate_column(state, original, detector.z_mm, maximum_step_mm=.1)
    transmitted = _apply_recording_stop(incident, detector)
    reference = column_wave._propagate_column(state, transmitted, target, maximum_step_mm=.1)
    actual, expected = result.checkpoint.beam.modes[0], reference.beam.modes[0]
    assert 0 < result.checkpoint.beam.total_weight < original.beam.total_weight
    assert actual.weight_per_reference_electron == pytest.approx(expected.weight_per_reference_electron, rel=1e-12)
    wavelength = float(wavelength_m(actual.energy_kev*1000.))
    np.testing.assert_allclose(actual.plane.basis_m, expected.plane.basis_m, rtol=1e-12, atol=1e-18)
    assert np.linalg.norm(actual.plane.full_amplitude(wavelength)-expected.plane.full_amplitude(wavelength)) < 2e-10


def test_segmented_column_rejects_prepared_plan_with_different_endpoints(monkeypatch, tmp_path):
    from temsim.physics.column_wave import _propagate_column_segmented
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    state = _quiet_observation_state()
    start = state.electron_gun.exit_plane_z_mm
    original = _fixture_tip_checkpoint(start)
    prepared = _quiet_prepared_column(state, start, start+.2, .1)
    store = ExecutedWaveStore(tmp_path/"waves", "isolated-plan-validation", 1<<24)
    monkeypatch.setattr(store, "writer", lambda *_a: pytest.fail("A mismatched plan cannot publish"))
    with pytest.raises(ValueError, match="exact executed interval"):
        _propagate_column_segmented(state, original, start+.1, store=store,
            segment_steps=2, _prepared=prepared)
