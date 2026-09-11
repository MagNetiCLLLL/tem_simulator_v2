"""Retained illumination mathematics and the HANDOFF v2 production gate.

Withdrawn production-pupil acceptance is preserved in historical/, not reported
as current source-chain validation. Flux/aperture acceptance remains unchanged.
"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics import wave_imaging as imaging
from temsim.physics.illumination import (
    default_illumination_config, gaussian_quadrature,
    current_angle_quantiles, validate_illumination_config,
)
from fixtures.illumination import PupilState, source_nodes
from test_tem_flux_contract import tem_benchmark


def test_at11_uniform_disk_has_distinct_current_quantiles():
    axis = np.linspace(-.05, .05, 801)
    fx, fy = np.meshgrid(axis, axis)
    pupil = PupilState(semi_axes_mrad=(20., 20.))
    spectrum = pupil.spectrum(fx, fy, 1.)
    q = current_angle_quantiles(fx, fy, spectrum, 1.)
    assert q["alpha_95_current_rad"] == pytest.approx(np.sqrt(.95)*.02, abs=2e-5)
    assert q["alpha_99_current_rad"] == pytest.approx(np.sqrt(.99)*.02, abs=2e-5)
    assert pupil.metadata()["alpha_edge_rad"] == .02


def test_sampled_intensity_and_amplitude_semantics():
    fy, fx = np.meshgrid(np.arange(-5., 6.), np.arange(-5., 6.), indexing="ij")
    data = [[0., 0., 0.], [1., 4., 0.], [0., 0., 0.]]
    pupil = PupilState(shape="sampled", values=data, semantics="intensity")
    # Finer than the supplied pupil lattice: enough occupied cells.
    fy, fx = np.meshgrid(np.arange(-5., 5., .25), np.arange(-5., 5., .25), indexing="ij")
    amp = pupil.spectrum(fx/1000, fy/1000, 1.)
    assert amp[20, 20] == 2.
    assert amp[20, 16] == 1.
    assert replace(pupil, semantics="amplitude").spectrum(fx/1000, fy/1000, 1.)[20, 20] == 4.


def test_source_priors_and_independent_dimensions():
    config = default_illumination_config()
    config["positions_nm"] = gaussian_quadrature([.1, 0.], 3)
    config["angles_mrad"] = [[0., 1., .25], [0., -1., .75]]
    config["energies_ev"] = gaussian_quadrature([.5], 3)
    nodes = source_nodes(config)
    assert len(nodes) == 18 and sum(n.weight for n in nodes) == pytest.approx(1.)
    assert sum(n.weight*n.position_nm[0]**2 for n in nodes) == pytest.approx(.01)
    assert sum(n.weight*n.energy_offset_ev**2 for n in nodes) == pytest.approx(.25)
    config["energies_ev"] = [[0., .5]]
    with pytest.raises(ValueError, match="sum to 1"):
        validate_illumination_config(config)


def test_clipped_declared_pupil_is_rejected_before_normalisation():
    fx, fy = np.meshgrid(np.linspace(-.01, .01, 32), np.linspace(-.01, .01, 32))
    with pytest.raises(ValueError, match="outside the wave grid"):
        PupilState().spectrum(fx, fy, 1.)


def test_legacy_reports_unknown_edge(tem_benchmark):
    result = tem_benchmark.run()
    assert result.metrics["alpha_edge_rad"] is None
    assert result.metrics["illumination_model"] == "ray_conditioned_reduced_order"


def test_at13_beam_state_intensity_sum_excludes_interference():
    from temsim.physics.wave_flux import WaveMode, BeamState
    axis = np.arange(64.) - 32
    plane = np.ones((64, 64), complex)
    tilted = np.broadcast_to(np.exp(2j*np.pi*axis/16), plane.shape)
    a = WaveMode.from_legacy(plane, axis, axis, reference_discrete_norm=4096., prior_weight=.3, mode_id="a", energy_kev=300.)
    b = WaveMode.from_legacy(tilted, axis, axis, reference_discrete_norm=4096., prior_weight=.7, mode_id="b", energy_kev=300.)
    intensity = BeamState((a, b)).cell_probabilities()
    np.testing.assert_allclose(intensity, 1/4096, atol=1e-15)
    coherent = np.abs(np.sqrt(.3)*a.plane.amplitude + np.sqrt(.7)*b.plane.amplitude)**2
    assert np.max(np.abs(coherent-intensity)) > .8/4096


def test_at15_analytic_gaussian_lct_centre_width_and_curvature():
    from temsim.physics.wave_flux import WaveMode, BeamState, TEM_REFERENCE_PLANE
    from temsim.physics.multiplane_wave import PlaneWave
    from temsim.physics.core import electron
    n, step, sigma, z = 256, .2e-9, 2e-9, 10e-6
    axis = (np.arange(n)-n//2)*step
    x, y = np.meshgrid(axis, axis)
    amplitude = np.exp(-(x*x+y*y)/(4*sigma*sigma)).astype(complex)
    amplitude /= np.sqrt(np.sum(np.abs(amplitude)**2))
    centre, tilt, curvature = np.array([1e-9, -2e-9]), np.array([.001, -.002]), 2e4
    plane = PlaneWave(amplitude, np.eye(2)*step, centre, np.eye(2)*curvature, tilt)
    beam = BeamState((WaveMode(plane, .7, TEM_REFERENCE_PLANE, "gaussian", 300.),))
    matrix = np.block([[np.eye(2), z*np.eye(2)], [np.zeros((2, 2)), np.eye(2)]])
    output = beam.propagate(lambda _: (matrix, np.zeros(4)))
    p = output.modes[0].plane
    xy = p.coordinates_m()
    weights = np.abs(p.amplitude)**2
    mean = np.sum(xy*weights, axis=(1, 2))
    np.testing.assert_allclose(mean, centre+z*tilt, atol=1e-16)
    variance = np.sum((xy-mean[:, None, None])**2*weights, axis=(1, 2))
    wavelength = electron(SimpleNamespace(beam_voltage_kv=300.))[2]*1e-9
    expected = (1+z*curvature)**2*sigma**2 + (wavelength*z/(4*np.pi*sigma))**2
    np.testing.assert_allclose(variance, expected, rtol=1e-8, atol=1e-28)
    # Add sampled residual phase to the analytic carrier, rather than judging
    # curvature from the carrier alone after an angular-spectrum step.
    row = n//2
    phase = np.unwrap(np.angle(p.amplitude[row]))
    local_x = xy[0, row]-p.origin_m[0]
    mask = weights[row] > weights[row].max()*1e-5
    fit = np.polyfit(local_x[mask]/1e-9, phase[mask], 2)
    measured_curvature = p.curvature_m1[0, 0] + fit[0]*1e18*wavelength/np.pi
    beta = wavelength/(4*np.pi*sigma**2)
    analytic = (curvature*(1+z*curvature) + z*beta**2)/((1+z*curvature)**2+(z*beta)**2)
    assert measured_curvature == pytest.approx(analytic, rel=1e-7)
    assert output.total_weight == .7


def test_energy_override_changes_actual_lens_map_without_retuning():
    from temsim.optics.column import default_state
    from temsim.physics.illumination import state_at_energy
    from temsim.physics.first_order import trace_transverse_transfer
    state = default_state()
    z = state.objective_lens.z_mm
    excitation = tuple((l.key, l.percent) for l in state.lenses)
    nominal = trace_transverse_transfer(state, z-5., z+5., maximum_step_mm=.2)
    shifted = trace_transverse_transfer(state_at_energy(state, state.beam_voltage_kv*.9), z-5., z+5., maximum_step_mm=.2)
    assert np.max(np.abs(shifted.matrix-nominal.matrix)) > 1e-6
    assert tuple((l.key, l.percent) for l in state.lenses) == excitation

@pytest.mark.parametrize("shape", ["ellipse", "rectangle"])
def test_independent_pupils_rejected_by_production_tem(tem_benchmark, shape):
    config = default_illumination_config()
    config["pupil"]["shape"] = shape
    tem_benchmark.state.sample.wave_illumination = config
    with pytest.raises(ValueError, match="historical only"):
        tem_benchmark.run()


def test_independent_modes_rejected_before_stem_capture():
    from test_stem_finite_absorption import _state, _simulation
    from temsim.physics import stem_wave_imaging as stem
    state = _state()
    state.sample.wave_illumination = default_illumination_config()
    sink = SimpleNamespace(begin=lambda *a: pytest.fail("No partial cube should be started"))
    with pytest.raises(ValueError, match="historical only"):
        stem.simulate_angle_resolved_stem(state, _simulation(), [],
            np.array([[0.]]), np.array([[0.]]), diffraction_sink=sink)


def test_historical_profile_parser_does_not_enable_old_source(tmp_path):
    from temsim.optics.column import default_state
    from temsim.profile_io import save_profile, read_profile, apply_profile_values
    from temsim.assembly_catalog import AssemblyCatalog
    state = default_state()
    state.sample.wave_illumination = default_illumination_config()
    path = tmp_path / "historical.json"
    save_profile(path, state, AssemblyCatalog().default_selection())
    _, payload = read_profile(path)
    target = default_state()
    before = deepcopy(target.sample.wave_illumination)
    with pytest.raises(ValueError, match="historical only"):
        apply_profile_values(target, payload)
    assert target.sample.wave_illumination == before
