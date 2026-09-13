"""Synthetic audit-format fixtures, never a physical gun acceptance record."""
import json

import numpy as np
import pytest

from scripts.summarize_gun_refinement import summarize


def fixture_report(directory, *, quartic=0.):
    directory.mkdir()
    np.savez(directory/"complex_states.npz", near_z_nm=np.array([[0., 2.]]))
    boundary = {"z_nm": 2., "width_nm": 1., "reference_k_per_nm": 1.,
        "curvature_per_nm": 0., "quartic_phase_per_nm4": quartic,
        "coefficients_real": [1.], "coefficients_imag": [0.], "net_current_fraction": .8}
    mode = {"energy_ev": .3, "mixture_weight": 1., "exit_fraction": .8,
        "far_field": {"exit_energy_ev": 300000.3, "reference_carrier_phase_mod_rad": 0.},
        "boundary_states": [boundary], "mask_absorbed_fraction": .2,
        "unresolved_mask_fraction": 0., "near_flux": {"side": 0.},
        "mask_losses": [{"kind": "aperture", "component": "fixture", "z_nm": 1., "radius_nm": 2.}]}
    payload = {"mode_id": "fixture", "width_nm": 1., "curvature_m1": 0.,
        "quartic_phase_per_nm4": quartic, "coefficients_real": [1.], "coefficients_imag": [0.]}
    report = {"instrument_settings_digest": "SYNTHETIC-ISOLATED-FIXTURE",
        "implementation": "SYNTHETIC-ISOLATED-FIXTURE", "implementation_unchanged": True,
        "status": "DIAGNOSTIC_GUN_ONLY_NOT_IMAGES",
        "gun": {"current_a": 8e-8, "mode_records": [mode], "radial_output_modes": [payload]}}
    path = directory/"report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_equal_pair_pass_is_not_full_source_admission(tmp_path):
    first, second = [fixture_report(tmp_path/name) for name in ("a", "b")]
    result = summarize(first, second)
    assert result["coordinate_comparison_status"] == "PASSES_THIS_PAIR_ONLY"
    assert result["source_to_image_admission"] == "NOT_GRANTED"
    assert result["relative_complex_l2"] == 0.
    assert len(result["inputs"][0]["complex_states_sha256"]) == 64
    assert result["mask_ledger"][0]["physical_mask_events"] == 1


def test_equal_current_and_density_do_not_hide_phase_error(tmp_path):
    a, b = fixture_report(tmp_path/"a"), fixture_report(tmp_path/"b", quartic=.1)
    result = summarize(a, b)
    assert result["relative_current_change"] == 0.
    assert result["relative_complex_l2"] > .1
    assert result["exit_comparison_quadrature_change"] < 2e-5
    assert result["coordinate_comparison_status"] == "FAILED"


@pytest.mark.parametrize("change,match", (
    (lambda r: r.update(status="FAILED"), "completed"),
    (lambda r: r.update(implementation_unchanged=False), "unchanged"),
    (lambda r: r.update(implementation="changed"), "same solver"),
    (lambda r: r["gun"]["mode_records"][0]["mask_losses"].clear(), "Physical masks"),
    (lambda r: r.update(instrument_settings_digest="changed"), "physical source"),
    (lambda r: r["gun"]["mode_records"].clear(), "every energy mode"),
))
def test_incomplete_changed_or_omitted_physics_is_not_refinement(tmp_path, change, match):
    a, b = [fixture_report(tmp_path/name) for name in ("a", "b")]
    report = json.loads(b.read_text(encoding="utf-8"))
    change(report)
    b.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        summarize(a, b)
