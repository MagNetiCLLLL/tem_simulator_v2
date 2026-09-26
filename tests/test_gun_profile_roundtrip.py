"""Current profile round trips include tip quadrature and every accelerator stage."""
from copy import deepcopy
from dataclasses import replace
import tomllib

import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.optics.electron_gun.emitter import EmissionQuadrature
from temsim.optics.electron_gun.profile_controls import capture_gun_profile_controls
from temsim.profile_io import save_profile, read_profile, apply_profile_values


def configured(gun_name="FEG"):
    state = default_state()
    catalog = AssemblyCatalog()
    selection = replace(catalog.default_selection(), gun=gun_name)
    catalog.apply(state, selection)
    stages = state.electron_gun.accelerator.stages
    for index, stage in enumerate(stages):
        stage.voltage_fraction *= .85 + .15 * index / (len(stages) - 1)
        stage.soft_edge_mm += .125 * (index + 1)
    if state.electron_gun.type_key == "cold_feg":
        state.electron_gun.emitter.quadrature = EmissionQuadrature(3, 5, 7)
        state.electron_gun.emitter.ray_count = 105
    return state, selection


@pytest.mark.parametrize("gun_name", ["FEG", "FEG + Mono", "Thermionic"])
def test_profile_restores_quadrature_and_all_accelerator_controls(tmp_path, gun_name):
    state, selection = configured(gun_name)
    before = capture_gun_profile_controls(state.electron_gun)
    path = tmp_path / "gun.toml"
    save_profile(path, state, selection)
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    assert document["gun_controls"] == before
    selected, values = read_profile(path)
    restored = default_state()
    AssemblyCatalog().apply(restored, selected)
    gun = restored.electron_gun
    original = (gun.emitter, gun.accelerator, *gun.accelerator.stages)
    centers = [stage.center_from_tip_mm for stage in gun.accelerator.stages]
    if gun.type_key == "cold_feg":
        # The incoming ray budget and incoming product must be checked together.
        gun.emitter.quadrature = EmissionQuadrature(3, 3, 3)
        gun.emitter.ray_count = 27
    apply_profile_values(restored, values)
    assert capture_gun_profile_controls(gun) == before
    assert all(a is b for a, b in zip(original, (gun.emitter, gun.accelerator, *gun.accelerator.stages), strict=True))
    assert [stage.center_from_tip_mm for stage in gun.accelerator.stages] == centers
    if gun.type_key == "cold_feg":
        assert gun.emitter.ray_count == gun.emitter.quadrature.total == 105


def test_explicit_current_profile_can_clear_quadrature(tmp_path):
    state, selection = configured()
    state.electron_gun.emitter.quadrature = None
    state.electron_gun.emitter.ray_count = 1000
    path = tmp_path / "clear.toml"
    save_profile(path, state, selection)
    _, values = read_profile(path)
    current, _ = configured()
    apply_profile_values(current, values)
    assert current.electron_gun.emitter.quadrature is None
    assert current.electron_gun.emitter.ray_count == 1000


@pytest.mark.parametrize("damage", [
    lambda controls: controls["accelerator_stages"][0].update(voltage_fraction=-1.),
    lambda controls: controls["accelerator_stages"][0].update(center_from_tip_mm=999.),
    lambda controls: controls["tip_quadrature"].update(spatial=5),
    lambda controls: controls.pop("accelerator_stages"),
])
def test_invalid_gun_controls_leave_entire_profile_transaction_unchanged(damage):
    state, _ = configured()
    before = deepcopy(state.to_dict())
    controls = capture_gun_profile_controls(state.electron_gun)
    damage(controls)
    with pytest.raises(ValueError):
        apply_profile_values(state, {"__gun_controls__": controls, "objective_lens": {"percent": 13.5}})
    assert state.to_dict() == before


def test_partial_profile_without_gun_controls_retains_current_quadrature_and_stages():
    state, _ = configured()
    before = capture_gun_profile_controls(state.electron_gun)
    apply_profile_values(state, {"objective_lens": {"percent": 13.5}})
    assert capture_gun_profile_controls(state.electron_gun) == before
