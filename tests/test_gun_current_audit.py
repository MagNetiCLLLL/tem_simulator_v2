"""Current attribution is accounting, never a normalization or admission."""
from copy import deepcopy
import json

import pytest

from scripts.audit_gun_current import compare_budgets, current_budget


def report(exit_fraction=.79, physical_loss=.1, unresolved=.01):
    return {"status": "DIAGNOSTIC_GUN_ONLY_NOT_IMAGES", "implementation_unchanged": True,
        "implementation": "test-implementation", "instrument_settings_digest": "same-physics",
        "gun": {"current_a": exit_fraction*100e-9,
            "record": {"source": {"emission": {"current_na": 100.}}},
            "mode_records": [{"energy_ev": .3, "mixture_weight": 1.,
                "near_flux": {"incoming": 1., "reflected": .08, "side": .02, "top": .9},
                "exit_fraction": exit_fraction,
                "mask_losses": [{"component": "DPA", "kind": "aperture", "z_nm": 100., "radius_nm": 2.,
                    "mask_absorbed": physical_loss, "unresolved_transmitted": unresolved}],
                "boundary_states": [{"z_nm": 1., "net_current_fraction": .9},
                                    {"z_nm": 50., "net_current_fraction": .9}]}]}}


def test_ledger_separates_physical_mask_from_unresolved_modes():
    result = current_budget(report())
    assert result["current_na"] == pytest.approx({"reflected_to_tip": 8., "numerical_side_escape": 2.,
        "entering_gun": 90., "physical_mask_absorption": 10., "unresolved_mask_modes": 1., "exit": 79.})
    assert abs(result["total_accounting_residual_na"]) < 1e-12
    assert abs(result["modes"][0]["gun_balance_residual"]) < 1e-12
    assert result["modes"][0]["pre_mask_sampled_current_range"] == 0


def test_signed_balance_failure_is_preserved_not_rescaled():
    original = report(.78)
    copy = deepcopy(original)
    budget = current_budget(original)
    assert budget["total_accounting_residual_na"] == pytest.approx(1.)
    assert budget["modes"][0]["gun_balance_residual"] == pytest.approx(.01)
    assert original == copy


def test_complete_energy_mixture_is_weighted_not_equal_averaged():
    original = report()
    mode = deepcopy(original["gun"]["mode_records"][0])
    mode.update(energy_ev=.5, mixture_weight=.75, exit_fraction=.7)
    mode["mask_losses"][0]["mask_absorbed"] = .19
    original["gun"]["mode_records"][0]["mixture_weight"] = .25
    original["gun"]["mode_records"].append(mode)
    original["gun"]["current_a"] = (.25*.79+.75*.7)*100e-9
    budget = current_budget(original)
    assert budget["current_na"]["exit"] == pytest.approx(72.25)
    assert budget["current_na"]["physical_mask_absorption"] == pytest.approx(16.75)
    assert budget["components"][0]["events_per_mode"] == [1, 1]


@pytest.mark.parametrize("weight", [float("nan"), -.1, .9, 1.+1e-10])
def test_missing_or_invalid_mixture_is_not_normalized(weight):
    original = report()
    original["gun"]["mode_records"][0]["mixture_weight"] = weight
    with pytest.raises(ValueError, match="full energy mixture"):
        current_budget(original)


def test_missing_implementation_proof_and_inconsistent_exit_are_rejected():
    original = report()
    original["implementation_unchanged"] = False
    with pytest.raises(ValueError, match="unchanged-implementation"):
        current_budget(original)
    original = report()
    original["gun"]["current_a"] *= 2
    with pytest.raises(ValueError, match="exit current"):
        current_budget(original)


def test_mode_cannot_skip_an_installed_aperture():
    original = report()
    mode = deepcopy(original["gun"]["mode_records"][0])
    original["gun"]["mode_records"][0]["mixture_weight"] = .5
    mode.update(mixture_weight=.5, mask_losses=[])
    original["gun"]["mode_records"].append(mode)
    with pytest.raises(ValueError, match="same physical masks"):
        current_budget(original)


def test_comparison_attributes_delta_and_rejects_physical_changes(tmp_path):
    paths = [tmp_path/"a.json", tmp_path/"b.json"]
    first, second = report(), report(.84, .055, .005)
    for path, data in zip(paths, (first, second)):
        path.write_text(json.dumps(data), encoding="utf-8")
    budget = compare_budgets(*paths)
    assert budget["second_minus_first_na"]["exit"] == pytest.approx(5.)
    assert budget["component_changes"][0]["physical_mask_absorption_change_na"] == pytest.approx(-4.5)
    assert budget["component_changes"][0]["unresolved_mask_modes_change_na"] == pytest.approx(-.5)
    for field, value, match in (("instrument_settings_digest", "changed", "physical inputs"),
                               ("implementation", "changed", "same numerical implementation")):
        changed = {**second, field: value}
        paths[1].write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            compare_budgets(*paths)


def test_comparison_checks_masks_even_if_report_claims_same_identity(tmp_path):
    paths = [tmp_path/"a.json", tmp_path/"b.json"]
    first, second = report(), report()
    second["gun"]["mode_records"][0]["mask_losses"][0]["radius_nm"] = 3.
    for path, data in zip(paths, (first, second)):
        path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Physical masks differ"):
        compare_budgets(*paths)
