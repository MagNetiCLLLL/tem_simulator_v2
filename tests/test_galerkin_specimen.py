"""Actual atomistic potential refinement, not physical-tip source acceptance."""
from dataclasses import replace, asdict

import numpy as np
import pytest

from temsim.immutable_json import json_digest
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.specimen_wave_transport import _propagate_specimen, _material_wave_axes, _material_phase
from temsim.physics.wave_grid import WaveGridNumerics, WaveSamplingError
from test_material_grid_refinement import state_and_wave


def material_fixture():
    state, checkpoint = state_and_wave()
    state.sample.wave_frozen_phonon_enabled = False
    state.sample.wave_grid_pixels = 128
    state.sample.wave_field_of_view_angstrom = 30.
    x = (np.arange(128)-64)*2e-11
    xx, yy = np.meshgrid(x, x)
    a = np.exp(-(xx**2+yy**2)/(2*(.2e-9)**2)).astype(complex)
    a /= np.linalg.norm(a)
    mode = replace(checkpoint.beam.modes[0], plane=PlaneWave(a, np.eye(2)*2e-11, np.zeros(2)))
    return state, replace(checkpoint, beam=replace(checkpoint.beam, modes=(mode,)))


def test_actual_atomistic_complex_field_and_fixed_observation_band_converge(record_property):
    state, checkpoint = material_fixture()
    source_id = checkpoint.digest
    outputs, readings, losses = {}, {}, {}
    for n, q in ((128, 2), (128, 4), (128, 8), (256, 2), (512, 2)):
        state.sample.wave_grid_pixels = n
        before = vars(state.sample).copy()
        result = _propagate_specimen(state, checkpoint,
            grid_numerics=WaveGridNumerics(specimen_quadrature_factor=q))
        assert vars(state.sample) == before
        assert checkpoint.digest == source_id
        assert result.record["potential"]["atomistic_applied"]
        assert result.record["material_grid_refinement"]["factor"] == 1
        m = result.beam.modes[0]
        a = m.plane.amplitude*np.sqrt(m.weight_per_reference_electron)
        coefficients = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(a), norm="ortho"))
        size = coefficients.shape[0]
        frequency = np.fft.fftshift(np.fft.fftfreq(size, d=m.plane.basis_m[0, 0]))
        fy, fx = np.meshgrid(frequency, frequency)
        # Constant 8 nm^-1 observation band. This is not a physical detector
        # propagated through a projector, and not a Nyquist-relative window.
        readings[(n, q)] = float(np.sum(abs(coefficients[np.hypot(fx, fy) <= 8e9])**2))
        outputs[(n, q)] = coefficients[size//2-16:size//2+16, size//2-16:size//2+16]
        losses[(n, q)] = result.record["modes"][0]["numerical_band_loss"]
        assert abs(result.record["modes"][0]["probability_balance"]["residual"]) < 1e-13
        record_property(f"grid_{n}_quadrature_{q}", str({"observation_weight": readings[(n, q)],
            "numerical_band_loss": losses[(n, q)], "output_weight": m.weight_per_reference_electron}))
    difference = lambda a, b: np.linalg.norm(outputs[a]-outputs[b])
    # Regression accuracy for THIS specimen/beam; no universal image claim.
    assert difference((256, 2), (512, 2)) < .5*difference((128, 2), (256, 2))
    assert difference((128, 4), (128, 8)) < .5*difference((128, 2), (128, 4))
    assert difference((256, 2), (512, 2))/np.linalg.norm(outputs[(512, 2)]) < 1e-3
    assert abs(readings[(256, 2)]/readings[(512, 2)]-1) < 1e-4
    assert losses[(512, 2)] < losses[(256, 2)] < losses[(128, 2)]
    record_property("complex_difference_256_512", float(difference((256, 2), (512, 2))))
    record_property("quadrature_difference_4_8", float(difference((128, 4), (128, 8))))


@pytest.mark.parametrize("n", [63, 64])
@pytest.mark.parametrize("factor", [2, 3, 4])
def test_quadrature_does_not_shift_the_wave_origin(n, factor):
    origin, dx = 2e-9, 1e-11
    fine = origin+(np.arange(n*factor)-(n*factor)//2)*dx/factor
    x, y = _material_wave_axes(fine, fine, WaveGridNumerics(specimen_quadrature_factor=factor))
    np.testing.assert_allclose(x, origin+(np.arange(n)-n//2)*dx, atol=1e-22, rtol=0)
    np.testing.assert_array_equal(x, y)


def carrier_fixture(shape=(4, 6), *, carrier=True):
    """Independent small affine lattice; carrier is evaluated analytically."""
    _, checkpoint = state_and_wave()
    rng = np.random.default_rng(37024)
    amplitude = rng.normal(size=shape)+1j*rng.normal(size=shape)
    amplitude /= np.linalg.norm(amplitude)
    basis, origin = np.diag((2e-11, 3e-11)), np.array((.43e-9, -.19e-9))
    curvature = np.array(((2e9, -.6e9), (-.6e9, -1e9))) if carrier else None
    tilt = np.array((.04, -.03)) if carrier else None
    wave = PlaneWave(amplitude, basis, origin, curvature, tilt)
    return replace(checkpoint.beam.modes[0], plane=wave)


def independent_carrier(mode, shape):
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
    ny, nx = shape
    old_ny, old_nx = mode.plane.amplitude.shape
    yy, xx = np.meshgrid(np.arange(ny)-ny//2, np.arange(nx)-nx//2, indexing="ij")
    physical = mode.plane.basis_m @ np.vstack((xx.ravel()*old_nx/nx, yy.ravel()*old_ny/ny))
    q = np.zeros((2, 2)) if mode.plane.curvature_m1 is None else mode.plane.curvature_m1
    t = np.zeros(2) if mode.plane.tilt_rad is None else mode.plane.tilt_rad
    phase = (.5*np.sum(physical*(q @ physical), axis=0)+t @ physical)
    return np.exp(2j*np.pi*phase/float(wavelength_m(mode.energy_kev*1000)))


@pytest.mark.parametrize("carrier", (False, True))
def test_material_phase_matches_independent_carrier_conjugated_hamiltonian(carrier):
    from scipy.linalg import expm
    from test_galerkin_potential import fourier_basis
    mode = carrier_fixture(carrier=carrier)
    wave, shape = mode.plane, mode.plane.amplitude.shape
    quadrature = tuple(2*n for n in shape)
    rng = np.random.default_rng(1114)
    potential = rng.uniform(-2., 3., quadrature)
    sigma, fraction = 1.3, .4
    x = wave.origin_m[0]+(np.arange(quadrature[1])-quadrature[1]//2)*wave.basis_m[0, 0]/2
    y = wave.origin_m[1]+(np.arange(quadrature[0])-quadrature[0]//2)*wave.basis_m[1, 1]/2
    # Build U_fine R directly from physical exponentials, independently of
    # FFT or production carrier expansion. U* V U=V for this scalar potential.
    fine_basis = independent_carrier(mode, quadrature)[:, None]*fourier_basis(shape, quadrature)
    coarse_basis = independent_carrier(mode, shape)[:, None]*fourier_basis(shape, shape)
    h = fine_basis.conj().T @ ((sigma*fraction*potential).ravel()[:, None]*fine_basis)
    np.testing.assert_allclose(h, h.conj().T, atol=3e-15)
    incident = independent_carrier(mode, shape)*wave.amplitude.ravel()
    expected = coarse_basis @ expm(1j*h) @ coarse_basis.conj().T @ incident
    result, record = _material_phase(mode, potential, x, y, sigma, fraction,
        numerics=WaveGridNumerics(), cancelled=lambda: False)
    actual = independent_carrier(result, shape)*result.plane.amplitude.ravel()
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=3e-14)
    assert abs(np.vdot(actual, actual).real-np.vdot(incident, incident).real) < 1e-13
    assert result.weight_per_reference_electron == mode.weight_per_reference_electron
    assert result.energy_kev == mode.energy_kev and result.axial_reference == mode.axial_reference
    for name in ("basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(result.plane, name), getattr(wave, name))
    assert record["basis"] == "carrier-covariant Fourier"
    assert record["carrier_retained"]


def test_constant_material_phase_retains_unresolved_carrier_without_expanding_it():
    mode = carrier_fixture()
    wave = mode.plane
    ny, nx = wave.amplitude.shape
    x = wave.origin_m[0]+(np.arange(2*nx)-nx)*wave.basis_m[0, 0]/2
    y = wave.origin_m[1]+(np.arange(2*ny)-ny)*wave.basis_m[1, 1]/2
    potential = np.full((2*ny, 2*nx), 2.3)
    result, _ = _material_phase(mode, potential, x, y, 1.2, .4,
        numerics=WaveGridNumerics(), cancelled=lambda: False)
    np.testing.assert_allclose(result.plane.amplitude, wave.amplitude*np.exp(1.104j), atol=2e-15)
    np.testing.assert_array_equal(result.plane.curvature_m1, wave.curvature_m1)
    np.testing.assert_array_equal(result.plane.tilt_rad, wave.tilt_rad)
    # The historical sampled method still requires the full laboratory
    # carrier to fit its lattice and must reject this deliberately coarse case.
    with pytest.raises(WaveSamplingError):
        _material_phase(mode, potential, x, y, 1.2, .4,
            numerics=WaveGridNumerics(specimen_phase_method="sampled"), cancelled=lambda: False)


@pytest.mark.parametrize("carrier", (False, True))
def test_carrier_frame_bandlimit_matches_dense_projector_and_records_actual_loss(carrier):
    from test_galerkin_potential import fourier_basis
    from temsim.physics.specimen_wave_transport import _bandlimit_mode
    mode = carrier_fixture((6, 8), carrier=carrier)
    shape, wave, fraction = mode.plane.amplitude.shape, mode.plane, .65
    unitary = independent_carrier(mode, shape)[:, None]*fourier_basis(shape, shape)
    ky, kx = np.meshgrid(np.arange(shape[0])-shape[0]//2,
                         np.arange(shape[1])-shape[1]//2, indexing="ij")
    reciprocal = np.linalg.inv(wave.basis_m).T @ np.vstack((kx.ravel()/shape[1], ky.ravel()/shape[0]))
    cutoff = .5*fraction/np.linalg.svd(wave.basis_m, compute_uv=False).max()
    retained = np.linalg.norm(reciprocal, axis=0) <= cutoff
    projector = (unitary*retained) @ unitary.conj().T
    np.testing.assert_allclose(projector @ projector, projector, atol=2e-15)
    incident = independent_carrier(mode, shape)*wave.amplitude.ravel()
    expected = projector @ incident*np.sqrt(mode.weight_per_reference_electron)
    result, loss = _bandlimit_mode(mode, fraction, frame="carrier")
    actual = independent_carrier(result, shape)*result.plane.amplitude.ravel()*np.sqrt(result.weight_per_reference_electron)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-14)
    expected_weight = float(np.vdot(expected, expected).real)
    assert result.weight_per_reference_electron == pytest.approx(expected_weight, rel=0, abs=2e-15)
    assert loss == pytest.approx(mode.weight_per_reference_electron-expected_weight, rel=0, abs=2e-15)
    assert 0 < loss < mode.weight_per_reference_electron
    assert result.energy_kev == mode.energy_kev and result.axial_reference == mode.axial_reference
    np.testing.assert_array_equal(result.plane.curvature_m1, wave.curvature_m1)
    np.testing.assert_array_equal(result.plane.tilt_rad, wave.tilt_rad)
    if not carrier:
        legacy, legacy_loss = _bandlimit_mode(mode, fraction)
        np.testing.assert_array_equal(result.plane.amplitude, legacy.plane.amplitude)
        assert result.weight_per_reference_electron == legacy.weight_per_reference_electron
        assert loss == legacy_loss
    else:
        # This finite moving basis intentionally differs from the old fixed
        # laboratory projection, whose direct carrier expansion is unresolved.
        with pytest.raises(WaveSamplingError):
            _bandlimit_mode(mode, fraction)


@pytest.mark.parametrize("resident", (False, True))
def test_gpu_material_phase_interpolation_and_bandlimit_preserve_carrier_and_weights(monkeypatch, resident):
    from test_galerkin_potential import cupy_device_or_skip
    from temsim.physics.specimen_wave_transport import _bandlimit_mode
    cp = cupy_device_or_skip()
    import cupyx.scipy.ndimage as device_ndimage
    mode = carrier_fixture((6, 8))
    wave = mode.plane
    ny, nx = wave.amplitude.shape
    x = wave.origin_m[0]+(np.arange(2*nx)-nx)*wave.basis_m[0, 0]/2
    y = wave.origin_m[1]+(np.arange(2*ny)-ny)*wave.basis_m[1, 1]/2
    potential = np.random.default_rng(7842).uniform(-1., 2., (2*ny, 2*nx))
    reference, _ = _material_phase(mode, potential, x, y, 1.1, .4,
        numerics=WaveGridNumerics(), cancelled=lambda: False)
    reference, expected_loss = _bandlimit_mode(reference, .65, frame="carrier")
    calls, actual_interpolation = [], device_ndimage.map_coordinates
    def track_interpolation(values, coordinates, **kwargs):
        assert isinstance(values, cp.ndarray) and values.dtype == cp.float64
        assert isinstance(coordinates, cp.ndarray) and coordinates.dtype == cp.float64
        calls.append(values.shape)
        return actual_interpolation(values, coordinates, **kwargs)
    monkeypatch.setattr(device_ndimage, "map_coordinates", track_interpolation)
    numerical = WaveGridNumerics(compute_backend="Require GPU", maximum_device_working_bytes=32*1024**2)
    incoming = replace(mode, plane=replace(wave, amplitude=cp.asarray(wave.amplitude))) if resident else mode
    result, record = _material_phase(incoming, potential, x, y, 1.1, .4,
        numerics=numerical, cancelled=lambda: False)
    result, loss = _bandlimit_mode(result, .65, frame="carrier", numerics=numerical)
    assert isinstance(result.plane.amplitude, cp.ndarray if resident else np.ndarray)
    assert calls == [potential.shape]
    actual = cp.asnumpy(result.plane.amplitude) if resident else result.plane.amplitude
    np.testing.assert_allclose(actual, reference.plane.amplitude, rtol=2e-13, atol=3e-14)
    for name in ("basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(result.plane, name), getattr(wave, name))
    assert result.weight_per_reference_electron == pytest.approx(reference.weight_per_reference_electron, rel=0, abs=2e-15)
    assert loss == pytest.approx(expected_loss, rel=0, abs=2e-15)
    assert result.energy_kev == mode.energy_kev and result.axial_reference == mode.axial_reference
    assert record["compute_backend"] == "cupy" and record["carrier_retained"]
    assert record["numeric_precision"] == "complex128 / float64"


def test_gpu_material_restarts_full_quadrature_and_phase_after_resource_failure(monkeypatch):
    from test_galerkin_potential import cupy_device_or_skip
    from temsim.physics import galerkin_potential as operator
    from temsim.physics.compute_backend import GPUExecutionError
    cp = cupy_device_or_skip()
    mode = carrier_fixture()
    wave, (ny, nx) = mode.plane, mode.plane.amplitude.shape
    x = wave.origin_m[0]+(np.arange(2*nx)-nx)*wave.basis_m[0, 0]/2
    y = wave.origin_m[1]+(np.arange(2*ny)-ny)*wave.basis_m[1, 1]/2
    potential = np.random.default_rng(9842).uniform(-1., 2., (2*ny, 2*nx))
    original = wave.amplitude.copy()
    reference, _ = _material_phase(mode, potential, x, y, 1., .5,
        numerics=WaveGridNumerics(), cancelled=lambda: False)
    attempts, actual_operator = [], operator._resident_potential_action
    def fail_gpu(amplitude, phase, *, xp, **kwargs):
        attempts.append("numpy" if xp is np else "cupy")
        if xp is cp:
            cp.fft.fft2(amplitude)
            raise GPUExecutionError("out_of_memory", "controlled material FFT allocation")
        return actual_operator(amplitude, phase, xp=xp, **kwargs)
    monkeypatch.setattr(operator, "_resident_potential_action", fail_gpu)
    result, record = _material_phase(mode, potential, x, y, 1., .5,
        numerics=WaveGridNumerics(compute_backend="Prefer GPU"), cancelled=lambda: False)
    assert attempts == ["cupy", "numpy"]
    np.testing.assert_array_equal(result.plane.amplitude, reference.plane.amplitude)
    np.testing.assert_array_equal(mode.plane.amplitude, original)
    np.testing.assert_array_equal(result.plane.curvature_m1, wave.curvature_m1)
    assert record["compute_backend"] == "numpy" and "out_of_memory" in record["fallback_reason"]


def test_gpu_repeated_material_operators_release_private_split_blocks():
    from test_galerkin_potential import cupy_device_or_skip
    from temsim.physics.wave_device import release_device_memory
    cp = cupy_device_or_skip()
    mode = carrier_fixture((256, 256))
    wave = mode.plane
    x = wave.origin_m[0]+(np.arange(512)-256)*wave.basis_m[0, 0]/2
    y = wave.origin_m[1]+(np.arange(512)-256)*wave.basis_m[1, 1]/2
    potential = np.random.default_rng(9843).uniform(-1., 1., (512, 512))
    numerics = WaveGridNumerics(compute_backend="Require GPU")
    def execute():
        result, record = _material_phase(mode, potential, x, y, 1., .5,
            numerics=numerics, cancelled=lambda: False)
        assert isinstance(result.plane.amplitude, np.ndarray)
        assert record["compute_backend"] == "cupy"
        return result.plane.amplitude
    def used_bytes():
        release_device_memory()
        cp.cuda.get_current_stream().synchronize()
        free, total = cp.cuda.runtime.memGetInfo()
        return int(total-free)
    # Real FFT/interpolation warmup precedes measurement. Before the fix,
    # caller temporaries outlived their private pool and leaked ~4 MiB/call.
    reference = execute()
    execute()
    before = used_bytes()
    for _ in range(12):
        result = execute()
    after = used_bytes()
    np.testing.assert_array_equal(result, reference)
    assert after-before < 16*1024**2


def test_gpu_retired_material_pool_preserves_live_device_output():
    from test_galerkin_potential import cupy_device_or_skip
    from temsim.physics.wave_device import device_scope, release_device_memory
    cp = cupy_device_or_skip()
    mode = carrier_fixture((64, 64))
    wave = mode.plane
    x = wave.origin_m[0]+(np.arange(128)-64)*wave.basis_m[0, 0]/2
    y = wave.origin_m[1]+(np.arange(128)-64)*wave.basis_m[1, 1]/2
    potential = np.random.default_rng(9844).uniform(-1., 1., (128, 128))
    incoming = replace(mode, plane=replace(wave, amplitude=cp.asarray(wave.amplitude)))
    numerics = WaveGridNumerics(compute_backend="Require GPU")
    result, _ = _material_phase(incoming, potential, x, y, 1., .5,
        numerics=numerics, cancelled=lambda: False)
    expected = cp.asnumpy(result.plane.amplitude)
    # Retired pools may still own a live result. Explicit cleanup and a new
    # allocating scope must reclaim only unused blocks, never that result.
    release_device_memory()
    with device_scope("Require GPU", required_bytes=4*1024**2):
        scratch = cp.full((256, 256), 7+3j, dtype=cp.complex128)
        assert bool(cp.all(scratch == 7+3j))
        del scratch
    release_device_memory()
    assert isinstance(result.plane.amplitude, cp.ndarray)
    np.testing.assert_array_equal(cp.asnumpy(result.plane.amplitude), expected)
    del result, incoming
    release_device_memory()


def test_method_and_quadrature_are_distinct_cache_inputs():
    cases = [WaveGridNumerics(), WaveGridNumerics(specimen_phase_method="sampled"),
             WaveGridNumerics(specimen_quadrature_factor=4)]
    assert len({json_digest(asdict(n)) for n in cases}) == 3
    for kw in ({"specimen_phase_method": "auto"}, {"specimen_quadrature_factor": 1},
               {"specimen_quadrature_factor": True}, {"specimen_quadrature_factor": 2.5}):
        with pytest.raises(ValueError):
            WaveGridNumerics(**kw).validate()


def test_material_numerics_reuse_every_executed_upstream_column_segment(tmp_path, monkeypatch):
    from temsim.physics import column_wave
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    from test_wave_detector_readout import checkpoint as column_fixture
    from test_segmented_tip_wave import quiet_state
    state, source = quiet_state(), column_fixture(.4)
    store = ExecutedWaveStore(tmp_path, "independent-column-cache-fixture", 1<<28)
    def run(numerics):
        return column_wave._propagate_column_segmented(state, source, 2000.2,
            store=store, segment_steps=1, maximum_step_mm=.1, grid_numerics=numerics)
    first, hit = run(WaveGridNumerics())
    assert not hit
    def forbidden(*a, **kw):
        pytest.fail("Changing only specimen numerics must not rerun an upstream column segment")
    monkeypatch.setattr(column_wave, "_propagate_column", forbidden)
    for numerics in (WaveGridNumerics(specimen_phase_method="sampled"),
                     WaveGridNumerics(specimen_quadrature_factor=4)):
        reused, hit = run(numerics)
        assert hit and reused.digest == first.digest
    assert WaveGridNumerics().column_identity() != WaveGridNumerics(maximum_pixels=512).column_identity()


def test_actual_inelastic_cache_distinguishes_potential_quadrature(tmp_path, monkeypatch):
    from temsim.physics import inelastic_wave
    from temsim.physics.wave_execution import InelasticWaveNumerics
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    state, source = material_fixture()
    store = ExecutedWaveStore(tmp_path, "independent-inelastic-quadrature-fixture", 1<<28)
    def run(factor):
        return inelastic_wave._propagate_inelastic_specimen(state, source,
            numerics=InelasticWaveNumerics(trajectories_per_mode=1), store=store,
            maximum_step_mm=.5, grid_numerics=WaveGridNumerics(specimen_quadrature_factor=factor),
            tip_time_s=0., cancelled=lambda: False, progress_callback=None, verify=lambda: None, use_cache=True)
    coarse, fine = run(2), run(4)
    assert coarse.digest != fine.digest
    for result, factor in ((coarse, 2), (fine, 4)):
        row = result.record["modes"][0]["steps"][0]["phase_before"]
        assert row["potential_quadrature_shape"] == (128*factor, 128*factor)
    def forbidden(*a, **kw):
        pytest.fail("An unchanged, completed material stage must be reused")
    monkeypatch.setattr(inelastic_wave, "_propagate_inelastic_attempt", forbidden)
    assert run(2).digest == coarse.digest
    assert run(4).digest == fine.digest
