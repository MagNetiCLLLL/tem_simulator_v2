"""Pure control records: no emission, field solve or particle propagation."""
from copy import copy, deepcopy
from dataclasses import asdict

import pytest

from temsim.optics.electron_gun.emitter import EmissionQuadrature
from temsim.optics.electron_gun.field_emission import FieldEmissionGun
from temsim.optics.electron_gun.thermionic import ThermionicGun
from temsim.optics.electron_gun.profile_controls import (
    capture_gun_profile_controls, prepare_gun_profile_controls,
    apply_prepared_gun_profile_controls,
)


@pytest.fixture
def gun():
    return FieldEmissionGun()


def test_current_non_scalar_controls_preserve_geometry_and_component_identities(gun):
    source = FieldEmissionGun()
    source.emitter.quadrature = EmissionQuadrature(3, 5, 7)
    source.emitter.ray_count = 105
    source.accelerator.stages[0].voltage_fraction *= .9
    source.accelerator.stages[0].soft_edge_mm += .25
    payload = capture_gun_profile_controls(source)
    assert all(set(row) == {"voltage_fraction", "soft_edge_mm"}
               for row in payload["accelerator_stages"])
    original = (gun.emitter, gun.accelerator, *gun.accelerator.stages)
    positions = [stage.center_from_tip_mm for stage in gun.accelerator.stages]
    gun.emitter.ray_count = 105
    prepared = prepare_gun_profile_controls(gun, payload)
    assert gun.emitter.quadrature is None
    apply_prepared_gun_profile_controls(gun, prepared)
    assert capture_gun_profile_controls(gun) == payload
    assert all(a is b for a, b in zip(original, (gun.emitter, gun.accelerator, *gun.accelerator.stages)))
    assert [stage.center_from_tip_mm for stage in gun.accelerator.stages] == positions


def test_explicit_empty_quadrature_clears_an_old_selection(gun):
    payload = capture_gun_profile_controls(gun)
    assert payload["tip_quadrature"] == {}
    gun.emitter.quadrature = EmissionQuadrature(3, 3, 3)
    gun.emitter.ray_count = 27
    apply_prepared_gun_profile_controls(gun, prepare_gun_profile_controls(gun, payload))
    assert gun.emitter.quadrature is None


def test_validation_uses_incoming_scalar_candidates_without_mutating_either(gun):
    emitter = copy(gun.emitter)
    emitter.ray_count = 105
    accelerator = copy(gun.accelerator)
    accelerator.high_tension_kv = 120.
    payload = capture_gun_profile_controls(gun)
    payload["tip_quadrature"] = asdict(EmissionQuadrature(3, 5, 7))
    prepared = prepare_gun_profile_controls(gun, payload, emitter=emitter, accelerator=accelerator)
    assert prepared.tip_quadrature.total == 105
    assert gun.emitter.quadrature is emitter.quadrature is None
    assert gun.accelerator.high_tension_kv == 300.
    assert accelerator.high_tension_kv == 120.
    with pytest.raises(ValueError, match="ray count"):
        prepare_gun_profile_controls(gun, payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0.1", 0., -1., 1.1])
def test_invalid_stage_fraction_is_rejected_without_any_change(gun, value):
    original = capture_gun_profile_controls(gun)
    payload = deepcopy(original)
    payload["accelerator_stages"][0]["voltage_fraction"] = value
    with pytest.raises(ValueError):
        prepare_gun_profile_controls(gun, payload)
    assert capture_gun_profile_controls(gun) == original


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "4", 0., -1.])
def test_invalid_stage_width_is_rejected_without_any_change(gun, value):
    original = capture_gun_profile_controls(gun)
    payload = deepcopy(original)
    payload["accelerator_stages"][0]["soft_edge_mm"] = value
    with pytest.raises(ValueError):
        prepare_gun_profile_controls(gun, payload)
    assert capture_gun_profile_controls(gun) == original


@pytest.mark.parametrize("change", ["extra", "missing", "position", "count", "order", "last"])
def test_invalid_complete_stage_contract_is_rejected(gun, change):
    original = capture_gun_profile_controls(gun)
    payload = deepcopy(original)
    if change == "extra":
        payload["old_gun"] = {}
    elif change == "missing":
        payload.pop("tip_quadrature")
    elif change == "position":
        payload["accelerator_stages"][0]["center_from_tip_mm"] = 1.
    elif change == "count":
        payload["accelerator_stages"].pop()
    elif change == "order":
        payload["accelerator_stages"][1]["voltage_fraction"] = .01
    else:
        payload["accelerator_stages"][-1]["voltage_fraction"] = .95
    with pytest.raises(ValueError):
        prepare_gun_profile_controls(gun, payload)
    assert capture_gun_profile_controls(gun) == original


@pytest.mark.parametrize("row", [[], {"spatial": 3},
    {"spatial": True, "directions": 3, "energies": 3, "schema": "tip-product-cdf-v1"},
    {"spatial": 3, "directions": 3, "energies": 3, "schema": "unknown"},
    {"spatial": 3, "directions": 3, "energies": 3, "schema": "tip-product-cdf-v1", "old": 1},
])
def test_bad_quadrature_records_are_rejected(gun, row):
    payload = capture_gun_profile_controls(gun)
    payload["tip_quadrature"] = row
    with pytest.raises(ValueError):
        prepare_gun_profile_controls(gun, payload)
    assert gun.emitter.quadrature is None


def test_quadrature_cannot_silently_replace_a_coherent_source(gun):
    from temsim.optics.electron_gun.tip_coherence import TipCoherence
    payload = capture_gun_profile_controls(gun)
    payload["tip_quadrature"] = asdict(EmissionQuadrature(3, 3, 3))
    emitter = copy(gun.emitter)
    emitter.ray_count = 27
    emitter.coherence = TipCoherence()
    with pytest.raises(ValueError, match="classical only"):
        prepare_gun_profile_controls(gun, payload, emitter=emitter)
    assert gun.emitter.quadrature is None


def test_thermionic_stages_round_trip_without_a_feg_quadrature_record():
    gun = ThermionicGun()
    payload = capture_gun_profile_controls(gun)
    assert set(payload) == {"accelerator_stages"}
    payload["accelerator_stages"][0]["voltage_fraction"] *= .9
    apply_prepared_gun_profile_controls(gun, prepare_gun_profile_controls(gun, payload))
    assert capture_gun_profile_controls(gun) == payload
    with pytest.raises(ValueError, match="current control tables"):
        prepare_gun_profile_controls(gun, {**payload, "tip_quadrature": {}})


def test_writer_rejects_a_selected_quadrature_with_the_wrong_ray_budget(gun):
    gun.emitter.quadrature = EmissionQuadrature(3, 3, 3)
    with pytest.raises(ValueError, match="ray count"):
        capture_gun_profile_controls(gun)


def test_grounded_surface_uses_its_existing_boundary_and_edge_contract(gun):
    from temsim.optics.electron_gun.tip_surface import load_tip_surface_reference

    gun.emitter.surface_model = load_tip_surface_reference()
    # Analytic ramp edge widths do not enter the grounded Laplace field.
    gun.accelerator.stages[0].soft_edge_mm = 0.
    payload = capture_gun_profile_controls(gun)
    assert prepare_gun_profile_controls(gun, payload).stage_values[0][1] == 0.
    payload["accelerator_stages"][-1]["voltage_fraction"] = .999999
    with pytest.raises(ValueError, match="final accelerator fraction"):
        prepare_gun_profile_controls(gun, payload)
