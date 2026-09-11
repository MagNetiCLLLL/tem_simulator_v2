"""Independent competing-rate probability checks; no source qualification."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics.specimen_wave_channels import _attenuate_zero_loss
from temsim.physics.wave_reference import AxialWaveReference
from test_wave_detector_readout import checkpoint


def material():
    return SimpleNamespace(plasmon_mean_free_path_nm=100., ionisation_mean_free_path_nm=200.,
        other_mean_free_path_nm=np.inf, absorption_mean_free_path_nm=400.,
        total_inelastic_mean_free_path_nm=1/(1/100+1/200),
        channels=(SimpleNamespace(key="real_plasmon", energy_loss_ev=16., characteristic_angle_mrad=.03),
                  SimpleNamespace(key="real_ionisation", energy_loss_ev=100., characteristic_angle_mrad=.2)))


def test_material_phase_and_competing_hazards_close_against_analytic_survival():
    mode = replace(checkpoint(.37).beam.modes[0], axial_reference=AxialWaveReference(1e-8, 2e-22))
    inside = np.ones(mode.plane.amplitude.shape, bool)
    result, record = _attenuate_zero_loss(mode, inside, 30., material())
    survival = np.exp(-30*(1/100+1/200+1/400))
    assert result.weight_per_reference_electron == pytest.approx(.6*survival, abs=1e-15)
    np.testing.assert_allclose(result.plane.amplitude, mode.plane.amplitude, atol=1e-16)
    assert result.axial_reference is mode.axial_reference
    assert result.energy_kev == mode.energy_kev
    events = {r["kind"]: r for r in record["events"]}
    removed = .6*(1-survival)
    assert events["real_plasmon"]["weight"] == pytest.approx(removed*4/7)
    assert events["real_ionisation"]["weight"] == pytest.approx(removed*2/7)
    assert events["effective_absorption"]["weight"] == pytest.approx(removed/7)
    assert events["real_other_inelastic"]["weight"] == 0
    assert events["real_plasmon"]["status"] == "OUTGOING_WAVE_NOT_COMPUTED"
    assert all(r["phase"] == "UNDEFINED" for r in record["events"])
    assert result.weight_per_reference_electron+sum(r["weight"] for r in record["events"]) == pytest.approx(.6)


def test_finite_material_and_slice_subdivision_preserve_spatial_coherence_and_first_event_weights():
    mode = checkpoint(.72).beam.modes[0]
    inside = mode.plane.coordinates_m()[0] > 0
    whole, ledger = _attenuate_zero_loss(mode, inside, 50., material())
    current, weights = mode, np.zeros(4)
    for _ in range(20):
        current, part = _attenuate_zero_loss(current, inside, 2.5, material())
        weights += [event["weight"] for event in part["events"]]
    np.testing.assert_allclose(weights, [event["weight"] for event in ledger["events"]], atol=5e-16)
    np.testing.assert_allclose(current.plane.amplitude, whole.plane.amplitude, atol=1e-15)
    assert current.weight_per_reference_electron == pytest.approx(whole.weight_per_reference_electron, abs=1e-15)
    # Outside the actual finite material, weighted complex amplitude is unchanged.
    np.testing.assert_allclose(np.sqrt(whole.weight_per_reference_electron)*whole.plane.amplitude[~inside],
        np.sqrt(mode.weight_per_reference_electron)*mode.plane.amplitude[~inside], atol=1e-16)


def test_vacuum_tiny_depth_and_opaque_limit_do_not_create_negative_or_duplicate_populations():
    mode = checkpoint().beam.modes[0]
    for depth, mask in ((0., True), (1e-12, True), (1e9, True), (1e9, False)):
        result, record = _attenuate_zero_loss(mode, np.full(mode.plane.amplitude.shape, mask), depth, material())
        assert all(r["weight"] >= 0 for r in record["events"])
        assert result.weight_per_reference_electron+sum(r["weight"] for r in record["events"]) == pytest.approx(.6, abs=1e-15)
        if not mask or depth == 0:
            assert record["removed_weight"] == 0
        if mask and depth == 1e9:
            assert result.weight_per_reference_electron == 0
            assert result.plane.probability == 0


def test_inconsistent_material_cannot_publish_a_balanced_looking_ledger():
    m = material(); m.total_inelastic_mean_free_path_nm = 1000
    mode = checkpoint().beam.modes[0]
    with pytest.raises(ValueError, match="do not sum"):
        _attenuate_zero_loss(mode, np.ones(mode.plane.amplitude.shape, bool), 5., m)
