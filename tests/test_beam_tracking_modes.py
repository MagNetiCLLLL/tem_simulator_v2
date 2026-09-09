"""Recorded interaction categories remain displayable after lens focusing."""

from types import SimpleNamespace

import numpy as np
import pytest

from temsim.gui.beam_tracking_modes import branch_interaction_style


@pytest.mark.parametrize("channel", (
    "real_plasmon", "real_ionisation", "real_other_inelastic", "real_plural_inelastic",
))
@pytest.mark.parametrize("tag", ("sample_region_elastic", "unknown", ""))
def test_generated_elastic_inelastic_names_keep_both_recorded_processes(channel, tag):
    branch = SimpleNamespace(name=f"specimen_elastic:{channel}", interaction_kind=tag,
                             colour=(1., .32, .48), weight=.2)
    before = vars(branch).copy()
    key, label, rgb, symbol = branch_interaction_style(branch)
    assert key == "elastic+" + channel
    assert label.startswith("Elastic + ")
    assert rgb != (255, 82, 122)
    assert all(isinstance(value, int) and 0 <= value <= 255 for value in rgb)
    assert symbol == "star"
    assert vars(branch) == before


@pytest.mark.parametrize("prefix,key,symbol", (
    ("specimen_primary", "sample_region_primary", "o"),
    ("specimen_elastic", "sample_region_elastic", "t"),
))
@pytest.mark.parametrize("zero_loss", ("000", "zero_loss"))
def test_exact_generated_zero_loss_names_distinguish_primary_from_elastic(prefix, key, symbol, zero_loss):
    actual = branch_interaction_style(SimpleNamespace(name=f"{prefix}:{zero_loss}"))
    assert actual[0] == key
    assert actual[3] == symbol


@pytest.mark.parametrize("channel", (
    "real_plasmon", "real_ionisation", "real_other_inelastic", "real_plural_inelastic",
))
def test_ordinary_and_finite_primary_inelastic_populations_share_class(channel):
    ordinary = SimpleNamespace(name=channel, interaction_kind=channel)
    primary = SimpleNamespace(name=f"specimen_primary:{channel}", interaction_kind=channel)
    assert branch_interaction_style(ordinary) == branch_interaction_style(primary)
    assert branch_interaction_style(ordinary)[3] == "d"
    assert branch_interaction_style(SimpleNamespace(name=channel))[0] == channel


def test_recognized_explicit_metadata_wins_and_unknown_names_are_not_inferred():
    conflicting = SimpleNamespace(name="specimen_elastic:real_plasmon", interaction_kind="vacuum")
    assert branch_interaction_style(conflicting)[0] == "vacuum"
    for name, kind in (
        ("specimen_elastic:channelling", "unknown"),
        ("my_plasmon_ray", "unknown"),
        ("000", "channelling"),
    ):
        branch = SimpleNamespace(name=name, interaction_kind=kind, energy_offset_ev=-1000, tx=0.0)
        assert branch_interaction_style(branch) == ("unknown", "Unknown interaction", (148, 163, 184), "x")
    explicit = SimpleNamespace(name="legacy_unspecified_channel", interaction_kind="sample_region_elastic")
    assert branch_interaction_style(explicit)[0] == "sample_region_elastic"


def test_ordinary_colours_are_retained_without_mutating_channel_metadata():
    branch = SimpleNamespace(name="whatever", interaction_kind="real_plasmon", colour=(.2, .4, .6))
    assert branch_interaction_style(branch)[2] == (51, 102, 153)
    assert branch.interaction_kind == "real_plasmon"
    branch.colour = (float("nan"), 0., 1.)
    assert branch_interaction_style(branch)[2] == (41, 209, 245)


def _branch(name, kind):
    x = np.asarray(([-1e-4, 0, 1e-4], [-2e-4, 0, 2e-4]))
    return SimpleNamespace(
        name=name, interaction_kind=kind, z=np.asarray((10., 20.)),
        x=x, y=x*.3, tx=np.zeros_like(x), ty=np.zeros_like(x),
        blocked_z=np.full(3, np.nan), ray_weight=np.full(3, 1/3),
    )


def test_ray_diagram_keeps_elastic_and_mixed_legends_after_focusing(qtbot):
    from temsim.gui.visualization import VisualizationWorkspace

    view = VisualizationWorkspace()
    qtbot.addWidget(view)
    view.ray_colour_mode.setCurrentIndex(view.ray_colour_mode.findData("interaction"))
    incident = _branch("incident", "incident")
    branches = (
        incident,
        _branch("specimen_primary:000", "sample_region_primary"),
        _branch("specimen_elastic:000", "sample_region_elastic"),
        _branch("specimen_primary:real_plasmon", "real_plasmon"),
        _branch("specimen_elastic:real_plasmon", "sample_region_elastic"),
    )
    simulation = SimpleNamespace(incident=incident, metrics={"sample_convergence_99_mrad": 1.})
    expected = {branch_interaction_style(branch)[0] for branch in branches}
    view._sync_ray_curves(simulation, branches)
    assert set(view._ray_legend_items) == expected
    assert {key[0] for key in view._ray_items_by_group} == expected
    assert view._ray_legend_items["elastic+real_plasmon"].opts["name"] == "Elastic + plasmon / low loss"
    before = {id(branch): branch_interaction_style(branch) for branch in branches}
    for branch in branches:
        branch.x[-1] = 0.0  # Synthetic projector focus; no additional collision.
        branch.y[-1] = 0.0
    view._sync_ray_curves(simulation, branches)
    assert set(view._ray_legend_items) == expected
    assert {id(branch): branch_interaction_style(branch) for branch in branches} == before
