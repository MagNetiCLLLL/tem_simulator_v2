"""Non-wave CIF readout integration with isolated executed-incident fixtures.

The real CIF projection, detector geometry, sequential absorption and current
normalisation run here. Only the downstream response is an analytic straight
drift fixture; this does not qualify a tip-to-column calculation or Bragg/BF
contrast. The fixture is deliberately independent of the user's CIF file.
"""
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.calculation_cache import calculation_signatures
from temsim.detector import projected_response, projected_stem, stem_signal
from temsim.optics.column import default_state
from temsim.physics.simulation import Branch
from temsim.specimen.downstream_transport import GeometricSpecimenExit


FCC_CELL_ANGSTROM = 3.82166108


def _fcc_cif(tmp_path):
    """The supplied file's FCC cell/site, not a substituted diamond lattice."""
    path = tmp_path / "fcc_si.cif"
    path.write_text(f"""data_Si
_cell_length_a {FCC_CELL_ANGSTROM}
_cell_length_b {FCC_CELL_ANGSTROM}
_cell_length_c {FCC_CELL_ANGSTROM}
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'Fm-3m'
_symmetry_Int_Tables_number 225
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Si0 Si 0 0 0 1
""", encoding="utf-8")
    return path


def _state(path):
    state = default_state()
    sample = state.sample
    sample.inserted = True
    sample.specimen_mode = "atomic"
    sample.cif_path = str(path)
    sample.stem_particle_model = "projected_atoms"
    sample.stem_wave_enabled = False
    sample.stem_poisson_enabled = False
    sample.stem_fourdstem_enabled = False
    sample.eds_support_material_key = "vacuum"
    sample.size_x_nm = sample.size_y_nm = 6.
    sample.thickness_nm = 1.
    sample.centre_x_nm = sample.centre_y_nm = 0.
    sample.scan_origin_x_nm = sample.scan_origin_y_nm = 0.
    sample.specimen_orientation_quaternion_wxyz = (1., 0., 0., 0.)
    state.ac_deflector.enabled = state.ac_deflector.scan_enabled = True
    state.ac_deflector.scan_pixels_x = state.ac_deflector.scan_lines = 17
    state.ac_deflector.scan_pixel_size_nm = .02
    state.descan_deflector.enabled = False
    # Independent analytic detector geometry, with real detector hit masks.
    # Three disjoint angular intervals: [40,100], [10,40], [0,10] mrad.
    by_key = {detector.key: detector for detector in state.stem_detectors}
    for plane in state.recording_planes:
        plane.inserted = plane.key in by_key
    for key, distance, inner, outer in (
        ("haadf", 100., .04, .10), ("df", 110., .01, .04),
        ("bf", 120., 0., .01),
    ):
        detector = by_key[key]
        detector.set_optical_reference_z_mm(state.selected_area_aperture.z_mm,
                                            sample.z_mm + distance)
        detector.geometry = "disk" if inner == 0. else "annulus"
        detector.inner_diameter_mm = 2*distance*inner
        detector.outer_width_mm = 2*distance*outer
        detector.centre_offset_x_mm = detector.centre_offset_y_mm = 0.
        detector.readout_enabled = True
    return state


def _incident(sigma_nm=.025, survival=1.):
    positions = np.sqrt(2)*sigma_nm*np.array(((1., 0.), (-1., 0.), (0., 1.), (0., -1.)))
    positions = np.tile(positions, (2, 1))
    branch = SimpleNamespace(
        x=positions[None, :, 0]*1e-9, y=positions[None, :, 1]*1e-9,
        tx=np.zeros((1, 8)), ty=np.zeros((1, 8)),
        alive=np.array([True]*4+[False]*4),
        ray_weight=np.r_[np.full(4, survival/4), np.full(4, (1-survival)/4)],
        kinetic_energy_ev=np.full((1, 8), 300_000.), energy_offset_ev=np.zeros(8),
    )
    return SimpleNamespace(incident=branch, metrics={})


def _install_drift_responses(monkeypatch, *, absorbed=.0, stop_first_scattered=False,
                             domain_scattered_indices=()):
    """Conditional samples have known drift intersections, not preset images."""
    captured = []

    def build(state, simulation, projection, real_interactions=None, *, stop_z_mm, **kwargs):
        captured.append(projection)
        z = np.array([state.sample.z_mm] + sorted({plane.z_mm for plane in state.recording_planes
                                                  if plane.inserted and plane.z_mm <= stop_z_mm}))
        result = {}
        for key in ("vacuum", "direct", *(f"Z{element.atomic_number}" for element in projection.elements)):
            scattered = key.startswith("Z")
            slopes = np.array((.08, .03, .002, .25)) if scattered else np.zeros(4)
            ray_weights = np.array((.3, .3, .3, .1)) if scattered else np.full(4, .25)
            x = (z[:, None]-state.sample.z_mm)*1e-3*slopes
            blocked = np.full(4, np.nan)
            blocked_keys = [""]*4
            if stop_first_scattered and scattered:
                blocked[0] = state.sample.z_mm + 1.
                blocked_keys[0] = "aperture"
            if scattered:
                for index in domain_scattered_indices:
                    # After the HAADF plane, before DF/BF: a prior real hit
                    # owns that particle; an as-yet unmeasured one is unknown.
                    blocked[index] = state.sample.z_mm + 105.
                    blocked_keys[index] = "projected_field_domain"
            loss = 0. if key == "vacuum" else absorbed
            branch = Branch(key, (1., 1., 1.), z, x, np.zeros_like(x),
                            np.broadcast_to(slopes, x.shape).copy(), np.zeros_like(x),
                            np.isnan(blocked), blocked,
                            blocked_keys,
                            1-loss, np.zeros(4), ray_weights,
                            kinetic_energy_ev=np.full_like(x, 300_000.))
            result[key] = GeometricSpecimenExit((branch,), {
                "inelastic_absorbed_source_probability": loss,
                "model": "analytic_drift_integration_fixture",
            })
        return result

    monkeypatch.setattr(projected_response, "build_projected_responses", build)
    # A descan-compensated drift readout: the actual atomic scan positions still
    # change, but no additive centre shift at the recording planes is needed.
    monkeypatch.setattr(projected_stem, "paired_kick_response", lambda *_a: np.zeros((2, 2)))
    # Collection labels are not part of this physical-mask integration test.
    monkeypatch.setattr(stem_signal, "collection_angle", lambda *_a: None)
    return captured


def _scan(state, simulation, *, count=25, sample_response=None):
    axis_um = np.linspace(-FCC_CELL_ANGSTROM*.05, FCC_CELL_ANGSTROM*.05, count)*1e-3
    x, y = np.meshgrid(axis_um, axis_um)
    x += state.sample.scan_origin_x_nm*1e-3
    y += state.sample.scan_origin_y_nm*1e-3
    return projected_stem.acquire_projected_stem_scan(
        simulation, state, state.stem_detectors,
        scan_x_um=x, scan_y_um=y, kick_grid_mrad=np.zeros((*x.shape, 2)),
        baseline_scan_mrad=np.zeros(2), baseline_descan_scan_mrad=np.zeros(2),
        sample_response=np.eye(2) if sample_response is None else sample_response,
        scan_times_s=np.zeros_like(x),
    )


def _assert_conserved(frame):
    total = (sum(frame.fractions.values()) + frame.uncollected_fraction
             + frame.absorbed_fraction + frame.truncated_fraction)
    np.testing.assert_allclose(total, 1., rtol=0., atol=2e-14)
    assert frame.metrics["source_probability_conservation_error"] < 2e-14


def test_fcc_cif_has_particle_contrast_and_actual_wide_probe_suppresses_it(tmp_path, monkeypatch):
    path = _fcc_cif(tmp_path)
    before = sha256(path.read_bytes()).hexdigest()
    state = _state(path)
    _install_drift_responses(monkeypatch)
    narrow = _scan(state, _incident(.025))
    broad = _scan(state, _incident(.25))
    for key in ("haadf", "df", "bf"):
        assert np.ptp(narrow.fractions[key]) > 1e-3
        assert np.ptp(broad.fractions[key]) < .01*np.ptp(narrow.fractions[key])
    _assert_conserved(narrow)
    _assert_conserved(broad)
    assert narrow.metrics["model"] == "projected_atomic_scattering"
    assert narrow.metrics["coherent_imaging"] is False
    assert narrow.metrics["quantitative_model"] is False
    np.testing.assert_allclose(narrow.metrics["probe_sigma_principal_nm"], (.025, .025))
    np.testing.assert_allclose(broad.metrics["probe_sigma_principal_nm"], (.25, .25))
    assert sha256(path.read_bytes()).hexdigest() == before


def test_source_survival_and_conditional_absorption_are_each_applied_once(tmp_path, monkeypatch):
    state = _state(_fcc_cif(tmp_path))
    captured = _install_drift_responses(monkeypatch, absorbed=.2)
    full = _scan(state, _incident(survival=1.), count=9)
    partial = _scan(state, _incident(survival=.37), count=9)
    for key in full.fractions:
        np.testing.assert_allclose(partial.fractions[key], .37*full.fractions[key], rtol=2e-14)
        np.testing.assert_allclose(partial.expected_electrons[key], .37*full.expected_electrons[key], rtol=2e-14)
    np.testing.assert_allclose(partial.absorbed_fraction, .37*full.absorbed_fraction, rtol=2e-14)
    np.testing.assert_allclose(partial.absorbed_fraction, .37*.2*captured[-1].material_overlap, atol=2e-15)
    np.testing.assert_allclose(partial.uncollected_fraction, .63+.37*full.uncollected_fraction, atol=2e-15)
    _assert_conserved(partial)


def test_aperture_stop_is_not_revived_for_any_later_detector(tmp_path, monkeypatch):
    state = _state(_fcc_cif(tmp_path))
    _install_drift_responses(monkeypatch)
    baseline = _scan(state, _incident(), count=9)
    _install_drift_responses(monkeypatch, stop_first_scattered=True)
    stopped = _scan(state, _incident(), count=9)
    np.testing.assert_array_equal(stopped.fractions["haadf"], 0.)
    for key in ("bf", "df"):
        np.testing.assert_array_equal(stopped.fractions[key], baseline.fractions[key])
    np.testing.assert_allclose(stopped.uncollected_fraction,
                               baseline.uncollected_fraction+baseline.fractions["haadf"], atol=2e-15)
    _assert_conserved(stopped)


def test_field_domain_unknown_excludes_previously_detected_particles(tmp_path, monkeypatch):
    state = _state(_fcc_cif(tmp_path))
    _install_drift_responses(monkeypatch, absorbed=.2)
    baseline = _scan(state, _incident(survival=.8), count=9)
    captured = _install_drift_responses(monkeypatch, absorbed=.2,
                                         domain_scattered_indices=(0, 1, 3))
    limited = _scan(state, _incident(survival=.8), count=9)
    # 30% of the scattered response hit HAADF before field failure. The
    # unmeasured 30% DF and 10% outward sample remain numerically unresolved.
    np.testing.assert_array_equal(limited.fractions["haadf"], baseline.fractions["haadf"])
    np.testing.assert_array_equal(limited.fractions["bf"], baseline.fractions["bf"])
    np.testing.assert_array_equal(limited.fractions["df"], 0.)
    expected_unknown = .8*.8*.4*captured[-1].scattered_fraction
    np.testing.assert_allclose(limited.truncated_fraction, expected_unknown, rtol=2e-14)
    np.testing.assert_allclose(limited.uncollected_fraction, .2, atol=2e-15)
    np.testing.assert_array_equal(limited.absorbed_fraction, baseline.absorbed_fraction)
    assert limited.metrics["maximum_numerical_truncation_fraction"] == pytest.approx(expected_unknown.max())
    _assert_conserved(limited)


def test_public_readout_disabled_still_intercepts_and_preserves_other_channels(tmp_path, monkeypatch):
    state = _state(_fcc_cif(tmp_path))
    _install_drift_responses(monkeypatch)
    monkeypatch.setattr(stem_signal, "physical_angular_detectors", lambda *_a: ([], {}))
    monkeypatch.setattr(stem_signal, "paired_kick_response", lambda *_a: np.eye(2)*1e-3)
    simulation = _incident()
    acquire = lambda: stem_signal.acquire_stem_scan(simulation, state, pixels_x=5, pixels_y=5, scan_calibrated=True)
    visible = acquire()
    haadf = next(detector for detector in state.stem_detectors if detector.key == "haadf")
    haadf.readout_enabled = False
    hidden = acquire()
    assert "haadf" not in hidden.fractions
    assert hidden.metrics["unreadout_detector_keys"] == ("haadf",)
    assert hidden.metrics["unreadout_intercepted_mean_fraction"]["haadf"] == pytest.approx(visible.fractions["haadf"].mean())
    for key in ("bf", "df"):
        np.testing.assert_array_equal(hidden.fractions[key], visible.fractions[key])
    np.testing.assert_array_equal(hidden.uncollected_fraction, visible.uncollected_fraction)
    np.testing.assert_array_equal(hidden.absorbed_fraction, visible.absorbed_fraction)
    total = sum(hidden.fractions.values()) + visible.fractions["haadf"] + hidden.uncollected_fraction + hidden.absorbed_fraction
    np.testing.assert_allclose(total, 1., atol=2e-14)


def test_physical_detector_position_changes_interception_not_atomic_probability(tmp_path, monkeypatch):
    state = _state(_fcc_cif(tmp_path))
    captured = _install_drift_responses(monkeypatch)
    centred = _scan(state, _incident(), count=9)
    next(detector for detector in state.stem_detectors if detector.key == "haadf").centre_offset_x_mm = -50.
    moved = _scan(state, _incident(), count=9)
    assert captured[0].identity == captured[1].identity
    np.testing.assert_array_equal(moved.fractions["haadf"], 0.)
    for key in ("bf", "df"):
        np.testing.assert_array_equal(moved.fractions[key], centred.fractions[key])
    np.testing.assert_allclose(moved.uncollected_fraction, centred.uncollected_fraction+centred.fractions["haadf"], atol=2e-15)
    _assert_conserved(moved)


def test_scan_origin_moves_atomic_sampling_and_detector_positions_together(tmp_path, monkeypatch):
    state = _state(_fcc_cif(tmp_path))
    captured = _install_drift_responses(monkeypatch)
    baseline = _scan(state, _incident(), count=9)
    # An explicitly declared affine magnification makes this tiny origin
    # displacement observable at the macroscopic mask. This regression checks
    # the coordinate plumbing, not calibration of a physical condenser lens.
    monkeypatch.setattr(projected_stem, "paired_kick_response", lambda *_a: np.eye(2)*1e5)
    state.sample.scan_origin_x_nm = .05
    shifted = _scan(state, _incident(), count=9)
    np.testing.assert_allclose(shifted.scan_x_um, baseline.scan_x_um+.05e-3, atol=1e-18)
    np.testing.assert_array_equal(shifted.scan_y_um, baseline.scan_y_um)
    assert captured[0].identity != captured[1].identity
    assert not np.allclose(captured[0].optical_depth, captured[1].optical_depth)
    # sample_response=1 mm/rad -> origin_command=5e-8 rad. The downstream
    # response is 1e8 mm/rad, so the direct beam shifts 5 mm into HAADF.
    # Without origin_command it would remain at BF even as the image moved.
    np.testing.assert_allclose(shifted.fractions["haadf"],
                               captured[-1].direct_fraction+.6*captured[-1].scattered_fraction,
                               rtol=2e-14)
    np.testing.assert_array_equal(shifted.fractions["bf"], 0.)
    np.testing.assert_array_equal(shifted.fractions["df"], 0.)
    _assert_conserved(shifted)


def test_nonzero_origin_rejects_singular_calibration_before_response_transport(tmp_path, monkeypatch):
    state = _state(_fcc_cif(tmp_path))
    captured = _install_drift_responses(monkeypatch)
    singular = np.array(((1., 0.), (0., 0.)))
    # No inversion is necessary at the zero origin.
    _scan(state, _incident(), count=3, sample_response=singular)
    assert len(captured) == 1
    state.sample.scan_origin_y_nm = .1
    with pytest.raises(ValueError, match="scan calibration.*scan origin"):
        _scan(state, _incident(), count=3, sample_response=singular)
    assert len(captured) == 1


def test_calculation_does_not_replace_or_mutate_executed_incident_state(tmp_path, monkeypatch):
    state = _state(_fcc_cif(tmp_path))
    _install_drift_responses(monkeypatch)
    simulation = _incident(.04, .7)
    checkpoint = object()
    simulation.incident_checkpoints = checkpoint
    original = simulation.incident
    before = {key: value.copy() for key, value in vars(original).items()}
    frame = _scan(state, simulation, count=7)
    assert simulation.incident is original and simulation.incident_checkpoints is checkpoint
    for key, value in before.items():
        np.testing.assert_array_equal(getattr(original, key), value)
    assert frame.probe_state.surviving_fraction == pytest.approx(.7)


def test_missing_cif_is_explicit_and_does_not_fall_back_to_uniform_material(tmp_path, monkeypatch):
    state = _state(tmp_path / "absent.cif")
    called = _install_drift_responses(monkeypatch)
    with pytest.raises((FileNotFoundError, ValueError), match="CIF|cif|structure|file"):
        _scan(state, _incident(), count=3)
    assert called == []


def test_particle_model_changes_stem_only_not_executed_upstream_dependencies():
    state = default_state()
    before = calculation_signatures(state)
    state.sample.stem_particle_model = "material_paths"
    after = calculation_signatures(state)
    assert {key for key in before if before[key] != after[key]} == {"request", "stem", "stem_transport"}
