import math
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.optics.energy_filter_raytrace import (
    _branch_probabilities as energy_filter_branch_probabilities,
)
from temsim.physics.interaction_budget import plane_interaction_budget
from temsim.physics.simulation import Branch, Simulation
from temsim.specimen.inelastic import (
    beb_ionisation_cross_section_m2,
    real_inelastic_distribution,
    real_inelastic_ray_branches,
)
from temsim.specimen.presets import load_specimen_preset


def test_si_inelastic_budget_uses_measured_anchor_and_poisson_statistics():
    state = default_state()
    state.electron_gun.high_tension_kv = 200.0
    state.sample.specimen_preset_key = "si_110"
    state.sample.thickness_nm = 10.0

    distribution = real_inelastic_distribution(state)
    channels = {channel.key: channel for channel in distribution.channels}
    mean = 10.0 / 145.0
    plasmon_mean = 10.0 / 168.0
    ionisation_mean = mean - plasmon_mean

    assert distribution.total_inelastic_mean_free_path_nm == pytest.approx(145.0)
    assert distribution.total_probability == pytest.approx(1.0)
    assert distribution.mean_inelastic_events == pytest.approx(mean)
    assert channels["real_zero_loss"].probability == pytest.approx(
        math.exp(-mean)
    )
    assert channels["real_plasmon"].probability == pytest.approx(
        math.exp(-mean) * plasmon_mean
    )
    assert channels["real_ionisation"].probability == pytest.approx(
        math.exp(-mean) * ionisation_mean
    )
    assert channels["real_plural_inelastic"].probability == pytest.approx(
        1.0 - math.exp(-mean) * (1.0 + mean)
    )


def test_voltage_scaling_and_beb_cross_section_are_physical():
    state = default_state()
    state.sample.specimen_preset_key = "si_110"
    state.sample.thickness_nm = 10.0
    state.electron_gun.high_tension_kv = 200.0
    at_200 = real_inelastic_distribution(state)
    state.electron_gun.high_tension_kv = 300.0
    at_300 = real_inelastic_distribution(state)

    assert at_300.plasmon_mean_free_path_nm > at_200.plasmon_mean_free_path_nm
    assert at_300.ionisation_mean_free_path_nm > at_200.ionisation_mean_free_path_nm
    assert beb_ionisation_cross_section_m2(300_000.0, 99.2) > 0.0
    assert beb_ionisation_cross_section_m2(50.0, 99.2) == 0.0


def test_effective_absorption_is_independent_and_probability_conserving():
    state = default_state()
    state.electron_gun.high_tension_kv = 200.0
    state.sample.specimen_preset_key = "si_110"
    state.sample.thickness_nm = 100.0
    state.sample.real_absorption_mean_free_path_nm = 100.0

    distribution = real_inelastic_distribution(state)
    branches = real_inelastic_ray_branches(distribution)

    assert distribution.absorbed_probability == pytest.approx(1.0 - math.exp(-1.0))
    assert distribution.tracked_probability == pytest.approx(math.exp(-1.0))
    assert sum(branch.probability for branch in branches) == pytest.approx(
        math.exp(-1.0)
    )
    assert distribution.total_probability == pytest.approx(1.0)


def test_material_presets_keep_inelastic_provenance_out_of_python_constants():
    silicon = load_specimen_preset("si_110").inelastic
    gold = load_specimen_preset("au_001").inelastic
    vacuum = load_specimen_preset("vacuum").inelastic

    assert silicon is not None and silicon.total_mean_free_path_nm == 145.0
    assert gold is not None and gold.total_mean_free_path_nm == 84.0
    assert "10.1103/PhysRevB.77.104102" in silicon.reference
    assert vacuum is None


def test_custom_cif_never_silently_borrows_selected_preset_inelastic_data():
    state = default_state()
    state.sample.specimen_preset_key = "si_110"
    state.sample.cif_path = "custom-silicon.cif"

    distribution = real_inelastic_distribution(state)

    assert distribution.material_key == "cif:custom-silicon.cif"
    assert distribution.model == "material_data_unavailable"
    assert distribution.mean_inelastic_events == 0.0
    assert any("never borrows" in warning for warning in distribution.warnings)


def test_custom_cif_accepts_one_complete_explicit_channel_pair():
    state = default_state()
    state.sample.specimen_preset_key = "si_110"
    state.sample.cif_path = "custom-silicon.cif"
    state.sample.thickness_nm = 25.0
    state.sample.real_plasmon_mean_free_path_nm = 100.0
    state.sample.real_plasmon_energy_ev = 15.0

    distribution = real_inelastic_distribution(state)
    channels = {channel.key: channel for channel in distribution.channels}

    assert distribution.model != "material_data_unavailable"
    assert distribution.plasmon_mean_free_path_nm == pytest.approx(100.0)
    assert math.isinf(distribution.ionisation_mean_free_path_nm)
    assert channels["real_plasmon"].probability > 0.0
    assert distribution.total_probability == pytest.approx(1.0)


def _branch(name, kind, weight, blocked_z):
    ray_count = 2
    z = np.array((1.0, 2.0))
    zeros = np.zeros((2, ray_count))
    return Branch(
        name=name,
        colour=(0.5, 0.5, 0.5),
        z=z,
        x=zeros.copy(),
        y=zeros.copy(),
        tx=zeros.copy(),
        ty=zeros.copy(),
        alive=np.isnan(blocked_z),
        blocked_z=np.asarray(blocked_z, dtype=float),
        blocked_key=["stop" if np.isfinite(value) else "" for value in blocked_z],
        weight=weight,
        energy_offset_ev=np.zeros(ray_count),
        ray_weight=np.array((0.4, 0.6)),
        interaction_kind=kind,
    )


def test_selected_plane_budget_uses_all_weighted_rays_and_conserves_source():
    incident = _branch("incident", "incident", 1.0, (math.nan, math.nan))
    zero = _branch("000", "real_zero_loss", 0.7, (1.5, math.nan))
    plasmon = _branch(
        "real_plasmon", "real_plasmon", 0.3, (math.nan, math.nan)
    )
    simulation = Simulation(
        incident=incident,
        branches={"000": zero, "real_plasmon": plasmon},
        metrics={
            "branch_weights_are_absolute": True,
            "sample_scattering_model": "test",
        },
    )
    state = SimpleNamespace(
        sample=SimpleNamespace(
            z_mm=1.0,
            specimen_mode="atomic",
            inserted=True,
            diffraction_enabled=True,
        )
    )
    result = SimpleNamespace(simulation=simulation, state_snapshot=state)

    budget = plane_interaction_budget(result, 2.0)
    channels = {channel.key: channel for channel in budget.channels}

    assert budget.source_fraction_at_plane == pytest.approx(0.72)
    assert budget.downstream_stopped_source_fraction == pytest.approx(0.28)
    assert channels["real_zero_loss"].source_fraction_at_plane == pytest.approx(0.42)
    assert channels["real_plasmon"].source_fraction_at_plane == pytest.approx(0.30)
    assert budget.conservation_error < 1.0e-12


def test_energy_filter_preserves_absolute_weight_and_absorbed_remainder():
    zero = _branch("000", "real_zero_loss", 0.6, (math.nan, math.nan))
    plasmon = _branch(
        "real_plasmon", "real_plasmon", 0.25, (math.nan, math.nan)
    )
    simulation = SimpleNamespace(
        branches={"000": zero, "real_plasmon": plasmon},
        metrics={"branch_weights_are_absolute": True},
    )

    probabilities = energy_filter_branch_probabilities(simulation)

    assert probabilities[id(zero)] == pytest.approx(0.6)
    assert probabilities[id(plasmon)] == pytest.approx(0.25)
    assert sum(probabilities.values()) == pytest.approx(0.85)
