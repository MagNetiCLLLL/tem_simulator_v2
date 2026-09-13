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


def test_galerkin_cannot_expand_an_undersampled_analytical_carrier():
    state, checkpoint = material_fixture()
    mode = checkpoint.beam.modes[0]
    mode = replace(mode, plane=replace(mode.plane, tilt_rad=np.array((.1, 0.))))
    x = (np.arange(256)-128)*1e-11
    with pytest.raises(WaveSamplingError):
        _material_phase(mode, np.zeros((256, 256)), x, x, 1., 1.,
            numerics=WaveGridNumerics(), cancelled=lambda: False)


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
