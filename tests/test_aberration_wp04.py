"""WP-04 analytic, independent-kernel and production-field acceptance."""

from copy import copy, deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt

from temsim.optics.aberrations import EffectiveAberrationSet, aberration_phase_rad
from temsim.optics.aberration_basis import (
    CARTESIAN_NAMES, WAVE_TERMS, cartesian_coefficients, predict_displacement_m,
)
from temsim.optics.field_aberrations import derive_field_aberrations, fit_wave_gradient, pupil_rays
from temsim.optics.aberration_validation import (
    CorrectorControl, _control_snapshot, compare_refinements, convergence_study, local_corrector_response,
)


def test_a5_rotation_scaling_and_zero():
    theta = pupil_rays(.032, holdout=True)
    wl = .0197
    coeff = EffectiveAberrationSet("sample", "manual", a5_mm=17, a5_azimuth_deg=13)
    chi = aberration_phase_rad(theta[:, 0]/wl, theta[:, 1]/wl, wl, coeff)
    angle = np.arctan2(theta[:, 1], theta[:, 0])
    expected = 2*np.pi/(wl*1e-10) * 17e-3/6 * np.linalg.norm(theta, axis=1)**6 * np.cos(6*(angle-np.deg2rad(13)))
    np.testing.assert_allclose(chi, expected, rtol=1e-12, atol=1e-12)
    for orientation, scale in ((73, 1), (43, -1)):
        rotated = aberration_phase_rad(theta[:, 0]/wl, theta[:, 1]/wl, wl, replace(coeff, a5_azimuth_deg=orientation))
        np.testing.assert_allclose(rotated, scale*chi, atol=1e-12)
    np.testing.assert_allclose(aberration_phase_rad(theta[:, 0]/wl/2, theta[:, 1]/wl/2, wl, coeff), chi/64)
    assert not np.any(aberration_phase_rad(theta[:, 0]/wl, theta[:, 1]/wl, wl, replace(coeff, a5_mm=0)))


def _mixed_coefficients():
    return EffectiveAberrationSet("sample", "manual", c1_mm=.00031, a1_mm=.0007, a1_azimuth_deg=17,
        b2_mm=.02, b2_azimuth_deg=-19, a2_mm=.03, a2_azimuth_deg=31,
        c3_mm=1.2, s3_mm=.8, s3_azimuth_deg=-22, a3_mm=.7, a3_azimuth_deg=37,
        c5_mm=29, a5_mm=17, a5_azimuth_deg=13)


def test_phase_matches_abtem_c56_and_mixed_polar_kernel(record_property):
    import abtem
    from abtem.transfer import Aberrations
    coefficients = _mixed_coefficients()
    parameters = {term.polar_code: getattr(coefficients, term.field)*1e7 for term in WAVE_TERMS}
    parameters.update({"phi"+term.polar_code[1:]: np.deg2rad(getattr(coefficients, term.azimuth_field))
                       for term in WAVE_TERMS if term.azimuth_field})
    theta = pupil_rays(.032, holdout=True)
    with abtem.config.set({"precision": "float64"}):
        external = Aberrations(energy=300e3, **parameters)
        expected = external._evaluate_from_angular_grid(np.linalg.norm(theta, axis=1), np.arctan2(theta[:, 1], theta[:, 0]))
        chi = aberration_phase_rad(theta[:, 0]/external.wavelength, theta[:, 1]/external.wavelength, external.wavelength, coefficients)
    np.testing.assert_allclose(np.exp(-1j*chi), expected, atol=1e-11, rtol=1e-11)
    record_property("wp04_external_kernel", json.dumps({"software": "abTEM", "version": abtem.__version__,
        "mapping": "A5_mm * 1e7 = C56_A; orientation_deg * pi/180 = phi56_rad", "status": "PASS",
        "max_complex_error": float(np.max(np.abs(np.exp(-1j*chi)-expected))),
        "scope": "Independent phase kernel, not material or field-solver validation"}))


def _independent_gradient(theta, c):
    # Explicit analytic derivatives, independent of the production basis builder.
    x, y = theta.T
    z, r2 = x+1j*y, x*x+y*y
    gx = c.c1_mm*x + c.c3_mm*r2*x + c.c5_mm*r2*r2*x
    gy = c.c1_mm*y + c.c3_mm*r2*y + c.c5_mm*r2*r2*y
    for amplitude, orientation, m in ((c.a1_mm, c.a1_azimuth_deg, 2), (c.a2_mm, c.a2_azimuth_deg, 3),
                                     (c.a3_mm, c.a3_azimuth_deg, 4), (c.a5_mm, c.a5_azimuth_deg, 6)):
        v = amplitude*np.exp(-1j*m*np.deg2rad(orientation))*z**(m-1)
        gx += v.real
        gy -= v.imag
    for amplitude, orientation, m, power in ((c.b2_mm, c.b2_azimuth_deg, 1, 3), (c.s3_mm, c.s3_azimuth_deg, 2, 4)):
        v = amplitude*np.exp(-1j*m*np.deg2rad(orientation))
        gx += (2*x*(v*z**m).real + r2*(m*v*z**(m-1)).real)/power
        gy += (2*y*(v*z**m).real + r2*(1j*m*v*z**(m-1)).real)/power
    return -1e-3*np.column_stack((gx, gy))


@pytest.mark.parametrize("alpha", [.01, .02, .032, .04])
def test_mixed_coefficients_recover_with_independent_holdout(alpha):
    training, holdout, truth = pupil_rays(alpha), pupil_rays(alpha, holdout=True), _mixed_coefficients()
    fitted, rms, condition = fit_wave_gradient(training, _independent_gradient(training, truth), alpha)
    np.testing.assert_allclose(fitted, cartesian_coefficients(truth), atol=1e-6, rtol=1e-7)
    np.testing.assert_allclose(predict_displacement_m(holdout, fitted), _independent_gradient(holdout, truth), atol=2e-20)
    assert rms < 1e-20 and condition < 100
    assert not np.any(np.all(training[:, None] == holdout[None, :], axis=2))


def test_missing_term_is_visible_in_holdout_error():
    training, holdout = pupil_rays(.032), pupil_rays(.032, holdout=True)
    # Curl-free sixth-degree C54 wave term absent from the declared model.
    def missing(theta):
        x, y = theta.T
        q, r2 = x+1j*y, x*x+y*y
        return -.2/6*np.column_stack((2*x*(q**4).real + r2*(4*q**3).real,
                                      2*y*(q**4).real + r2*(4j*q**3).real))
    coefficients, rms, _ = fit_wave_gradient(training, missing(training), .032)
    error = predict_displacement_m(holdout, coefficients) - missing(holdout)
    assert rms > 1e-10 and np.sqrt(np.mean(error**2)) > 1e-10


@pytest.mark.parametrize("angles, displacement, alpha", [
    ([[0, 0]], [[0, 0]], .01), ([[np.nan, 0]], [[0, 0]], .01),
    ([[.02, 0]], [[0, 0]], .01), ([[0, 0]], [[0, np.inf]], .01),
    ([[0, 0]], [[0]], .01), ([[0, 0]], [[0, 0]], 0),
])
def test_invalid_or_underconstrained_fit_fails(angles, displacement, alpha):
    with pytest.raises(ValueError):
        fit_wave_gradient(angles, displacement, alpha)


class _EnergyState(SimpleNamespace):
    @property
    def beam_voltage_kv(self):
        return getattr(self, "_propagation_energy_kev", 300.0)


def _field_state(nodes=9, step=.02, alpha=10):
    from test_vector_field_transport import _state, _map
    attributes = vars(_state()).copy()
    attributes.pop("beam_voltage_kv")
    state = _EnergyState(**attributes)
    state.step_mm = step
    state.objective_image_plane_z_mm = .8
    state.image_aberrations = {"mode": "field_derived", "fit_semiangle_mrad": alpha, "fit_step_mm": step}
    axes = (np.linspace(0, .0004, nodes), np.linspace(0, .001, 2*nodes-1))
    r, z = np.meshgrid(*axes, indexing="ij")
    # Divergence-free quadratic magnetic potential; independently sampled maps.
    curvature, b0 = .3/.001**2, .4
    br = -b0*curvature*r*z
    bz = b0*(1+curvature*(z*z-r*r/2))
    _map(state, axes=axes, components=(br, bz), rz=True)
    return state


def test_field_fit_reports_independent_rays_grid_partial_support_and_fixed_energy(monkeypatch, record_property):
    from temsim.physics import core
    state = _field_state()
    original = deepcopy([(lens.percent, lens.z_mm) for lens in state.lenses])
    calls = []
    real_execute = core.execute_propagation_plan
    def traced(snapshot, plan, *args, **kwargs):
        calls.append((snapshot.beam_voltage_kv, [(l.percent, l.z_mm) for l in snapshot.lenses], (plan.z_mm[0], plan.z_mm[-1])))
        return real_execute(snapshot, plan, *args, **kwargs)
    monkeypatch.setattr(core, "execute_propagation_plan", traced)
    _, result, evidence = derive_field_aberrations(state, "image")
    assert result.c1_mm == 0 and abs(result.cc_mm) > 1e-6
    assert evidence["holdout_ray_count"] == 161 and evidence["training_ray_count"] == 192
    assert evidence["holdout_independent"] and evidence["field_grids"][0]["shape"] == [9, 17]
    assert "C54" in evidence["unimplemented_terms"]
    assert evidence["chromatic_convergence"]["status"] == "PASS"
    assert len(set(energy for energy, _, _ in calls)) == 7
    assert all(hardware == original and interval == (0, .8) for _, hardware, interval in calls)
    assert [(l.percent, l.z_mm) for l in state.lenses] == original and state.beam_voltage_kv == 300
    record_property("wp04_cc_energy", json.dumps(evidence["chromatic_convergence"]))
    extra = copy(state.lenses[0])
    extra.key, extra.z_mm, extra.percent = "unmapped", .6, 0
    state.lenses.append(extra)
    _, result, evidence = derive_field_aberrations(state, "image")
    assert result.status == "partial_field_support" and "unmapped" in evidence["unmapped_round_lenses"]


def _probe_observable(state, coeff):
    # Vacuum probe encircled probability, not a specimen detector image.
    axis = np.linspace(-.008, .008, 64)
    tx, ty = np.meshgrid(axis, axis)
    pupil = tx*tx+ty*ty <= .008**2
    spectrum = pupil*np.exp(-1j*aberration_phase_rad(tx/.0197, ty/.0197, .0197, coeff))
    intensity = abs(np.fft.fftshift(np.fft.ifft2(spectrum)))**2
    return float(intensity[31:34, 31:34].sum()/intensity.sum())


def test_aperture_step_and_independent_grid_studies(record_property):
    floors = np.array([1e-7, 1e-7, 1e-7, 1e-5, 1e-5, 1e-5, 1e-5, .01, .01, .01, .01, .01, 10, 10, 10])
    studies = {
        "semiangle_mrad": [(str(alpha), _field_state(alpha=alpha, step=.003)) for alpha in (10, 20, 32, 40)],
        "ray_step_mm": [(str(step), _field_state(step=step)) for step in (.012, .006, .003)],
        "field_grid_radial_nodes": [(str(nodes), _field_state(nodes=nodes, step=.003)) for nodes in (5, 9, 17)],
    }
    for axis, cases in studies.items():
        report = convergence_study(cases, system="image", validation_semiangle_mrad=8,
            coefficient_floors_mm=floors, displacement_floor_m=1e-11,
            observable=_probe_observable, observable_unit="probability", observable_floor=1e-7)
        assert len(report["cases"]) == len(cases)
        assert report["status"] in {"PASS", "INCONCLUSIVE"}
        if axis == "field_grid_radial_nodes":
            fingerprints = [row["fit"]["field_grids"][0]["content_fingerprint"] for row in report["cases"]]
            assert len(set(fingerprints)) == 3
        counts = [row["fit"]["integration_grids"][0]["interval_count"] for row in report["cases"]]
        assert len(set(counts)) == (3 if axis == "ray_step_mm" else 1)
        report["fixture"] = "Analytic divergence-free field; vacuum probe encircled probability; not material / OEM validation"
        record_property("wp04_convergence_" + axis, json.dumps(report))


def test_large_coefficient_cannot_hide_small_coefficient_nonconvergence():
    result = compare_refinements([[0, 1e6], [1e-3, 1e6], [2e-3, 1e6]], [1, 2, 3], unit="mm", absolute_floor=[1e-6, 1.])
    assert result["status"] == "INCONCLUSIVE"


def test_settings_dialog_validates_a5_and_preserves_disabled_defaults(qtbot):
    from temsim.gui.aberration_dialog import AberrationSettingsDialog
    from temsim.optics.aberrations import SYSTEM_COEFFICIENT_ROWS
    dialog = AberrationSettingsDialog({"mode": "manual"}, system="probe", state_step_mm=.1)
    qtbot.addWidget(dialog)
    row = next(i for i, term in enumerate(SYSTEM_COEFFICIENT_ROWS) if term[0] == "A5")
    dialog.table.item(row, 0).setCheckState(Qt.CheckState.Checked)
    dialog.table.item(row, 2).setText("nan")
    dialog.accept()
    assert dialog.result_options is None and "finite" in dialog.error.text()
    dialog.table.item(row, 2).setText("15.2")
    dialog.table.item(row, 3).setText("27")
    dialog.accept()
    assert dialog.result_options["a5_mm"] == 15.2 and dialog.result_options["a5_azimuth_deg"] == 27
    assert "c3_mm" not in dialog.result_options


def test_a5_state_profile_cache_and_production_stem_phase(tmp_path):
    from temsim.optics.column import default_state
    from temsim.optics.model import State
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.profile_io import save_profile, read_profile, apply_profile_values
    from temsim.calculation_cache import calculation_signatures
    from temsim.physics.stem_wave_imaging import _probe_spectrum
    state = default_state()
    state.probe_aberrations = {"c3_mm": 0, "c1_mm": 0}
    axis = np.linspace(-1.5, 1.5, 32)
    stats = {"mean_tx_rad": 0, "mean_ty_rad": 0, "convergence_95_rad": .025, "waist_offset_m": 1e-8}
    first = _probe_spectrum(state, stats, axis, axis, .0197)
    # Resolve optical component ownership before taking signatures.
    baseline = calculation_signatures(state)
    state.probe_aberrations.update(a5_mm=15.2, a5_azimuth_deg=27)
    state.image_aberrations.update(a5_mm=3, a5_azimuth_deg=-4)
    current = calculation_signatures(state)
    assert current["incident"] == baseline["incident"] and current["wave"] != baseline["wave"]
    second = _probe_spectrum(state, stats, axis, axis, .0197)
    assert np.max(abs(first-second)) > .01
    np.testing.assert_allclose(abs(first), abs(second), atol=1e-15)
    restored = State.from_dict(state.to_dict())
    assert restored.probe_aberrations == state.probe_aberrations and restored.image_aberrations == state.image_aberrations
    path = tmp_path / "a5.toml"
    save_profile(path, state, AssemblyCatalog().default_selection())
    _, values = read_profile(path)
    restored.probe_aberrations = {}
    assert apply_profile_values(restored, values) == []
    assert restored.probe_aberrations == state.probe_aberrations


def test_control_snapshot_preserves_geometry_aliases_and_isolation():
    from temsim.optics.column import default_state
    state = default_state()
    state._field_provider_diagnostics = {"sentinel": {"status": "original"}}
    work = _control_snapshot(state)
    assert not hasattr(work, "_field_provider_diagnostics")
    assert work._resolved_assembly is state._resolved_assembly
    assert work.objective_lens is not state.objective_lens and work.objective_lens in work.lenses
    before = [l.percent for l in state.lenses]
    work.objective_lens.percent *= .99
    assert [l.percent for l in state.lenses] == before
    assert work._module_optical_offsets_mm == state._module_optical_offsets_mm


def test_local_response_uses_joint_cartesian_validation_without_mutating_controls(monkeypatch):
    from temsim.optics import aberration_validation
    state = _field_state()
    state.corrector_elements = [SimpleNamespace(key="hp1", strength_m3=0., maximum_strength_m3=20),
                                SimpleNamespace(key="hp2", strength_m3=0., maximum_strength_m3=20)]
    def evaluate(work, system):
        x, y = [v.strength_m3 for v in work.corrector_elements]
        ax, ay = 2*x+y, -x+3*y
        coeff = EffectiveAberrationSet("sample", "field-derived", a5_mm=abs(complex(ax, ay)),
            a5_azimuth_deg=float(np.degrees(np.angle(complex(ax, ay)))/6), c3_mm=.2*x*y)
        return coeff, coeff, {"fixture": "Known polynomial control map"}
    monkeypatch.setattr(aberration_validation, "derive_field_aberrations", evaluate)
    report = local_corrector_response(state, "image", [CorrectorControl("hp1", "strength_m3", .1), CorrectorControl("hp2", "strength_m3", .1)])
    names = report["coefficient_names"]
    jacobian = np.asarray(report["jacobian"])
    np.testing.assert_allclose(jacobian[names.index("A5_x")], [2, 1])
    np.testing.assert_allclose(jacobian[names.index("A5_y")], [-1, 3])
    assert abs(report["joint_holdouts"][0]["error_mm"][names.index("C3")]) > 0
    assert all(c.strength_m3 == 0 for c in state.corrector_elements)
    assert report["autotuning_applied"] is False


def test_local_response_reaches_production_hexapole_fields(record_property):
    from temsim.optics.hexapole import HexapoleComponent
    state = _field_state(nodes=5, step=.04)
    state.corrector_elements = [HexapoleComponent(
        name=key, key=key, z_mm=z, strength_m3=1e8, maximum_strength_m3=1e10,
        effective_length_mm=.1, enabled=True, colour="#fff",
        mechanical_center_from_tip_mm=z, mechanical_length_mm=.2,
        mechanical_outer_diameter_mm=2, mechanical_clear_bore_diameter_mm=1,
        optical_reference_from_tip_mm=z, orientation_rad=angle)
        for key, z, angle in (("hp1", .2, 0), ("hp2", .6, .3))]
    report = local_corrector_response(state, "image", [CorrectorControl("hp1", "strength_m3", 1e6),
                                                       CorrectorControl("hp2", "strength_m3", 1e6)])
    jacobian = np.asarray(report["jacobian"])
    assert np.isfinite(jacobian).all()
    assert np.all(np.linalg.norm(jacobian, axis=0) > 1e-15)
    assert all(h.strength_m3 == 1e8 for h in state.corrector_elements)
    assert report["baseline_fit"]["holdout_independent"]
    report["fixture"] = "Analytic field map with two distributed paraxial hexapoles; control units are m^-3, not coil amperes"
    record_property("wp04_corrector_response", json.dumps(report))
