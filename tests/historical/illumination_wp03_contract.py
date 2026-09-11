# Historical WP-03 acceptance, withdrawn by HANDOFF v2. Not collected by pytest.
# Independent specimen-entrance sources are no longer production inputs.
"""WP-03 manufactured illumination benchmarks; not a material calibration."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics import wave_imaging as imaging
from temsim.physics.illumination import (
    PupilState, default_illumination_config, gaussian_quadrature,
    current_angle_quantiles, source_nodes, validate_illumination_config,
)
from test_tem_flux_contract import tem_benchmark  # shared analytic column fixture

_real_incident_wave = imaging._incident_wave


def test_at11_uniform_disk_has_distinct_current_quantiles():
    axis = np.linspace(-.05, .05, 801)
    fx, fy = np.meshgrid(axis, axis)
    pupil = PupilState(semi_axes_mrad=(20., 20.))
    spectrum = pupil.spectrum(fx, fy, 1.)
    q = current_angle_quantiles(fx, fy, spectrum, 1.)
    assert q["alpha_95_current_rad"] == pytest.approx(np.sqrt(.95)*.02, abs=2e-5)
    assert q["alpha_99_current_rad"] == pytest.approx(np.sqrt(.99)*.02, abs=2e-5)
    assert pupil.metadata()["alpha_edge_rad"] == .02


def test_at12_rotated_offset_pupil_reaches_production_tem(tem_benchmark, monkeypatch):
    f = tem_benchmark
    monkeypatch.setattr(imaging, "_incident_wave", _real_incident_wave)
    f.state.objective_aperture.enabled = False
    config = default_illumination_config()
    config["pupil"].update(shape="rectangle", semi_axes_mrad=[4., 1.5], offset_mrad=[3., -2.], basis=[[0., -1.], [1., 0.]])
    f.state.sample.wave_illumination = config
    result = f.run()
    weight = result.absolute_diffraction_probability
    from temsim.physics.core import electron
    a = result.spatial_frequency_inv_angstrom * electron(f.state)[2] * 10 * 1000
    ax, ay = np.meshgrid(a, a)
    cx, cy = np.sum(ax*weight), np.sum(ay*weight)
    assert cx == pytest.approx(3., abs=.4) and cy == pytest.approx(-2., abs=.4)
    assert np.sum((ay-cy)**2*weight) > 3*np.sum((ax-cx)**2*weight)
    assert result.metrics["illumination_model"] == "specimen_entrance_pupil_modes"


def test_at13_tem_two_modes_are_incoherent_and_replayable(tem_benchmark, monkeypatch):
    f = tem_benchmark
    monkeypatch.setattr(imaging, "_incident_wave", _real_incident_wave)
    f.state.objective_aperture.enabled = False
    config = default_illumination_config()
    config["pupil"]["semi_axes_mrad"] = [3., 2.]
    config["angles_mrad"] = [[-4., 0., .25], [4., 0., .75]]
    f.state.sample.wave_illumination = config
    mixed = f.run()
    singles = []
    for x, y, _ in config["angles_mrad"]:
        f.state.sample.wave_illumination = {**config, "angles_mrad": [[x, y, 1.]]}
        singles.append(f.run())
    expected = .25*singles[0].camera_intensity + .75*singles[1].camera_intensity
    np.testing.assert_allclose(mixed.camera_intensity, expected, rtol=1e-10, atol=1e-12)
    expected_diffraction = .25*singles[0].absolute_diffraction_probability + .75*singles[1].absolute_diffraction_probability
    np.testing.assert_allclose(mixed.absolute_diffraction_probability, expected_diffraction, atol=1e-12)
    f.state.sample.wave_illumination = config
    replay = imaging.reproject_wave_image(f.state, mixed)
    np.testing.assert_allclose(replay.camera_intensity, mixed.camera_intensity, rtol=1e-10, atol=1e-12*np.max(mixed.camera_intensity))
    assert mixed.metrics["illumination_mode_count"] == 2


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


def test_energy_modes_update_wavelength_scattering_and_keep_hardware(tem_benchmark, monkeypatch):
    f = tem_benchmark
    monkeypatch.setattr(imaging, "_incident_wave", _real_incident_wave)
    f.state.objective_aperture.enabled = False
    config = default_illumination_config()
    config["pupil"]["semi_axes_mrad"] = [3., 2.]
    config["energies_ev"] = [[-1000., .4], [1000., .6]]
    f.state.sample.wave_illumination = config
    controls = [(l.key, l.percent, l.z_mm) for l in f.state.lenses]
    voltage = f.state.beam_voltage_kv
    result = f.run()
    records = result.metrics["illumination_executed_modes"]
    assert records[0]["wavelength_angstrom"] > records[1]["wavelength_angstrom"]
    assert records[0]["interaction_constant_rad_per_v_angstrom"] != records[1]["interaction_constant_rad_per_v_angstrom"]
    assert result.projector_checkpoint.configuration_energies_kev == (voltage-1., voltage+1.)
    assert [(l.key, l.percent, l.z_mm) for l in f.state.lenses] == controls
    assert f.state.beam_voltage_kv == voltage and not hasattr(f.state, "_propagation_energy_kev")
    replay = imaging.reproject_wave_image(f.state, result)
    np.testing.assert_allclose(replay.camera_intensity, result.camera_intensity, rtol=1e-10, atol=1e-12*result.camera_intensity.max())
    for record in replay.metrics["wave_execution_manifest"]["illumination"]["illumination_executed_modes"]:
        assert "objective_aperture_policy" not in record
        assert "camera_collected_zero_loss_relative_intensity" not in record


def test_stem_source_energy_modes_match_separate_detector_integrations(monkeypatch):
    from test_stem_finite_absorption import _state, _simulation, _empty_grid
    from temsim.physics import stem_wave_imaging as stem
    state = _state()
    state.sample.wave_multislice_enabled = False
    _empty_grid(monkeypatch, pixels=128, fov_nm=8.)
    config = default_illumination_config()
    config["pupil"].update(shape="rectangle", semi_axes_mrad=[3., 1.5], offset_mrad=[2., 0.])
    config["positions_nm"] = [[-.1, 0., .25], [.1, 0., .75]]
    config["energies_ev"] = [[-1000., .4], [1000., .6]]
    state.sample.wave_illumination = config
    scan = np.array([[-.0001, 0., .0001]])
    detectors = (stem.AngularDetector("BF", 0., 2.), stem.AngularDetector("DF", 2., 8.))
    def run():
        return stem.simulate_angle_resolved_stem(state, _simulation(), detectors, scan, np.zeros_like(scan), compute_sample_overlap=True)
    mixed = run()
    separate = {k: np.zeros_like(scan) for k in mixed.fractions}
    for node in source_nodes(config):
        state.sample.wave_illumination = {**config, "positions_nm": [[*node.position_nm, 1.]],
                                         "angles_mrad": [[*node.tilt_mrad, 1.]], "energies_ev": [[node.energy_offset_ev, 1.]]}
        result = run()
        for k in separate:
            separate[k] += node.weight*result.fractions[k]
    for k in separate:
        np.testing.assert_allclose(mixed.fractions[k], separate[k], atol=1e-12)
    assert mixed.metrics["illumination_mode_count"] == 4
    assert np.mean(mixed.fractions["DF"]) > 0


def test_illumination_serialization_cache_and_dialog(qtbot, tmp_path):
    from temsim.optics.column import default_state
    from temsim.optics.model import State
    from temsim.calculation_cache import calculation_signatures
    from temsim.gui.illumination_dialog import IlluminationDialog
    state = default_state()
    before = calculation_signatures(state)
    config = default_illumination_config()
    config["pupil"]["semi_axes_mrad"] = [3., 2.]
    state.sample.wave_illumination = config
    after = calculation_signatures(state)
    assert before["incident"] == after["incident"]
    assert before["wave_source"] != after["wave_source"]
    assert before["stem"] != after["stem"]
    restored = State.from_dict(state.to_dict())
    assert restored.sample.wave_illumination == config
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.profile_io import save_profile, read_profile, apply_profile_values
    selection = AssemblyCatalog().default_selection()
    path = tmp_path / "illumination.toml"
    save_profile(path, state, selection)
    _, values = read_profile(path)
    other = default_state()
    apply_profile_values(other, values)
    assert validate_illumination_config(other.sample.wave_illumination) == validate_illumination_config(config)
    dialog = IlluminationDialog(config, state.beam_voltage_kv)
    qtbot.addWidget(dialog)
    dialog.accept()
    assert dialog.config == validate_illumination_config(config)
    dialog.editor.setPlainText('{"model": "specimen_entrance_pupil_modes", "energies_ev": [[0, -1]]}')
    assert not dialog.buttons.button(dialog.buttons.StandardButton.Ok).isEnabled()


@pytest.mark.parametrize("dimension", ["position", "energy"])
def test_at14_source_and_energy_quadrature_convergence(monkeypatch, record_property, dimension):
    import json
    from test_stem_finite_absorption import _state, _simulation
    from temsim.physics import stem_wave_imaging as stem
    from temsim.physics.illumination import convergence_report
    state = _state()
    state.sample.wave_multislice_enabled = False
    state.sample.centre_x_nm = 3.
    state.sample.size_x_nm = 6.  # A physical edge at x=0; source overlap is nontrivial.
    n, fov = 128, 80.
    axis = (np.arange(n)-n//2) * fov/n
    xx, yy = np.meshgrid(axis, axis)
    potential = 1200*np.cos(2*np.pi*xx/4.) * (xx >= 0)
    prepared = SimpleNamespace(x_angstrom=axis, y_angstrom=axis,
        potential_configurations_v_angstrom=(potential,), mean_projected_potential_v_angstrom=potential,
        slice_thicknesses_angstrom=None, metrics={"calculation_roi_centre_nm": (0., 0.), "atomistic_applied": False, "frozen_phonon_applied": False})
    preset = SimpleNamespace(key="manufactured_edge_grating", pixels=n, field_of_view_angstrom=fov)
    monkeypatch.setattr(stem, "_wave_grid", lambda *_: (preset, prepared))
    config = default_illumination_config()
    config["pupil"]["semi_axes_mrad"] = [2., 2.]
    scan = np.linspace(-.0001, .0001, 5)[None, :]
    values, orders = [], [3, 5, 7]
    for order in orders:
        if dimension == "position":
            config["positions_nm"] = gaussian_quadrature([.05, 0.], order)
        else:
            config["energies_ev"] = gaussian_quadrature([1000.], order)
        state.sample.wave_illumination = config
        result = stem.simulate_angle_resolved_stem(state, _simulation(), (stem.AngularDetector("BF", 0., 1.),),
                              scan, np.zeros_like(scan), compute_sample_overlap=True)
        values.append(np.stack((result.fractions["BF"], result.sample_overlap_fraction)))
    report = convergence_report(values, orders, observable="BF probability and incident finite-material overlap", unit="probability per conditional incident electron",
                                absolute_floor=1e-7, relative_target=.01)
    record_property("wp03_convergence_" + dimension, json.dumps(report))
    assert report["status"] == "PASS", report
    assert np.ptp(values[-1][1]) > .01  # not a constant all-vacuum or all-material fixture


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


def test_mode_energy_validation_uses_nominal_reference_once():
    from temsim.optics.column import default_state
    from temsim.physics.illumination import state_for_source_node, illumination_config
    state = default_state()
    config = default_illumination_config()
    config["energies_ev"] = [[-299000., .5], [0., .5]]
    state.sample.wave_illumination = config
    mode = state_for_source_node(state, source_nodes(config)[0])
    assert mode.beam_voltage_kv == 1.
    assert illumination_config(mode)["energies_ev"] == config["energies_ev"]


def test_explicit_stem_grid_covers_all_source_positions(monkeypatch):
    from test_stem_finite_absorption import _state, _simulation
    from temsim.physics import stem_wave_imaging as stem
    from temsim.physics.illumination import state_for_source_node
    state = _state()
    state.sample.wave_multislice_enabled = False
    config = default_illumination_config()
    config["pupil"]["semi_axes_mrad"] = [2., 1.]
    config["positions_nm"] = [[-1., 0., .5], [1., 0., .5]]
    state.sample.wave_illumination = config
    seen = []
    def prepare(*args, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(metrics={})
    monkeypatch.setattr(stem, "prepare_specimen_potentials", prepare)
    for node in source_nodes(config):
        mode = state_for_source_node(state, node)
        stem._wave_grid(mode, _simulation(), np.array([[node.position_nm[0]*1e-3]]), np.array([[0.]]))
    assert seen[0] == seen[1]
    x0, x1, _, _ = seen[0]["calculation_roi_bounds_nm"]
    assert x0 < -1. and x1 > 1.


def test_multimode_capture_rejects_before_writing(monkeypatch):
    from test_stem_finite_absorption import _state, _simulation
    from temsim.physics import stem_wave_imaging as stem
    state = _state()
    config = default_illumination_config()
    config["positions_nm"] = [[-1., 0., .5], [1., 0., .5]]
    state.sample.wave_illumination = config
    sink = SimpleNamespace(begin=lambda *args: pytest.fail("No partial cube should be started"))
    with pytest.raises(ValueError, match="mode-resolved angular calibration"):
        stem.simulate_angle_resolved_stem(state, _simulation(), [], np.zeros((1, 1)), np.zeros((1, 1)), diffraction_sink=sink)
