"""Reference CIFs use the real physics routes, with no legacy virtual inputs."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from temsim.specimen import reference_catalog
from temsim.specimen.inelastic import real_inelastic_distribution
from temsim.specimen.rutherford import read_cif_composition


def _state(key="si_110", **sample_changes):
    values = dict(specimen_mode="reference", reference_sample_key=key,
                  specimen_preset_key="si_110", thickness_nm=5.0, inserted=True,
                  real_inelastic_enabled=True, cif_path="dormant-missing.cif")
    values.update(sample_changes)
    return SimpleNamespace(sample=SimpleNamespace(**values), beam_voltage_kv=200.0)


@pytest.mark.parametrize("key,expected_mfp", [("si_110", 145.0), ("au_001", 84.0)])
def test_reference_uses_its_explicit_material_anchor_not_numerical_template(key, expected_mfp):
    state = _state(key)
    result = real_inelastic_distribution(state)
    assert result.material_key == key
    assert result.total_inelastic_mean_free_path_nm == pytest.approx(expected_mfp)
    assert result.mean_inelastic_events == pytest.approx(5.0 / expected_mfp)
    assert result.total_probability == pytest.approx(1)
    assert any("elements were checked" in warning for warning in result.warnings)


def test_external_atomic_cif_never_inherits_reference_inelastic_anchor():
    state = _state("si_110", specimen_mode="atomic",
                   cif_path=str(reference_catalog.get_reference_sample("au_001").cif_path))
    result = real_inelastic_distribution(state)
    assert result.model == "material_data_unavailable"
    assert result.mean_inelastic_events == 0
    assert any("never borrows" in warning for warning in result.warnings)
    state.sample.real_plasmon_mean_free_path_nm = 50
    state.sample.real_plasmon_energy_ev = 20
    overridden = real_inelastic_distribution(state)
    assert overridden.plasmon_mean_free_path_nm == 50
    assert overridden.mean_inelastic_events == pytest.approx(.1)


def test_mismatched_reference_inelastic_anchor_is_rejected(monkeypatch):
    wrong = replace(reference_catalog.get_reference_sample("si_110"), inelastic_preset_key="au_001")
    monkeypatch.setattr(reference_catalog, "get_reference_sample", lambda _key: wrong)
    with pytest.raises(ValueError, match="CIF elements.*do not match"):
        real_inelastic_distribution(_state())


def test_reference_without_explicit_anchor_does_not_borrow_template(monkeypatch):
    unassigned = replace(reference_catalog.get_reference_sample("au_001"), inelastic_preset_key="")
    monkeypatch.setattr(reference_catalog, "get_reference_sample", lambda _key: unassigned)
    result = real_inelastic_distribution(_state("au_001"))
    assert result.model == "material_data_unavailable"
    assert any("no explicitly assigned" in warning for warning in result.warnings)


def _occupancy_cif(tmp_path, rows):
    path = tmp_path / "disorder.cif"
    path.write_text("""data_disorder
_cell_length_a 5
_cell_length_b 5
_cell_length_c 5
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_Int_Tables_number 1
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
""" + rows + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("rows", ["Si1 Si 0 0 0 .5", "Si1 Si 0 0 0 .5\nGe1 Ge 0 0 0 .5"])
def test_atomistic_potential_rejects_partial_or_mixed_sites_before_orthogonalizing(monkeypatch, tmp_path, rows):
    from temsim.specimen import atomistic

    path = _occupancy_cif(tmp_path, rows)
    composition = read_cif_composition(path)
    assert composition.partial_occupancy or composition.mixed_occupancy
    def no_potential(*_args, **_kwargs):
        raise AssertionError("Disorder must fail before constructing an atomistic potential")
    monkeypatch.setattr(atomistic, "_require_backend", lambda: (SimpleNamespace(orthogonalize_cell=no_potential), Atoms))
    with pytest.raises(ValueError, match="explicitly occupied/disordered supercell"):
        atomistic.build_cif_equilibrium_atoms(path, thickness_angstrom=50, field_of_view_angstrom=100)


def test_partial_reference_composition_does_not_use_full_material_imfp(monkeypatch, tmp_path):
    partial = replace(reference_catalog.get_reference_sample("si_110"),
                      cif_path=_occupancy_cif(tmp_path, "Si1 Si 0 0 0 .5"))
    monkeypatch.setattr(reference_catalog, "get_reference_sample", lambda _key: partial)
    result = real_inelastic_distribution(_state())
    assert result.model == "material_data_unavailable"
    assert any("full-material inelastic anchors are not applied" in warning for warning in result.warnings)


def test_live_reference_column_ignores_legacy_virtual_rows_and_budget_remains_real(monkeypatch):
    from temsim.optics.column import default_state
    from temsim.physics.simulation import run
    from temsim.physics.interaction_budget import plane_interaction_budget
    from temsim.specimen import virtual

    state = default_state()
    state.sample.specimen_mode = "reference"
    state.sample.reference_sample_key = "si_110"
    state.sample.virtual_interactions = [{"kind": "invalid obsolete channel", "probability": 1000}]
    state.sample.virtual_regions = [{"kind": "invalid obsolete region"}]
    state.sample.diffraction_enabled = True
    state.step_mm = state.history_step_mm = 5.0
    emitter = getattr(state.electron_gun, "emitter", state.electron_gun)
    emitter.ray_count = 9
    def no_virtual(*_args, **_kwargs):
        raise AssertionError("Reference material must not consume virtual channels")
    monkeypatch.setattr(virtual, "virtual_scattering_branches", no_virtual)
    monkeypatch.setattr(virtual, "build_virtual_angular_distribution", no_virtual)
    simulation = run(state)
    assert simulation.real_interactions is not None
    assert simulation.metrics["sample_scattering_model"] == "real_material_inelastic_poisson_plus_elastic_wave"
    assert simulation.metrics["branch_weights_are_absolute"]
    assert all(branch.interaction_kind.startswith("real_") for branch in simulation.branches.values())
    assert sum(branch.weight for branch in simulation.branches.values()) == pytest.approx(simulation.real_interactions.tracked_probability)
    budget = plane_interaction_budget(SimpleNamespace(simulation=simulation, state_snapshot=state), state.sample.z_mm)
    assert budget.material_name == simulation.real_interactions.material_name
    assert budget.conservation_error < 1e-10
    assert all(channel.key.startswith("real_") for channel in budget.channels)
