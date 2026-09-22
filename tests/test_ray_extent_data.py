"""Presentation metadata fixtures; no ray propagation is claimed here."""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.gui.ray_extent_data import completed_ray_extent


def result(target=750., resumable=700., *, checkpoint=True):
    gun = SimpleNamespace(z_mm=np.array([0., 450.]))
    section = SimpleNamespace(gun_trace=gun, segments=(SimpleNamespace(
        checkpoints=SimpleNamespace(z_mm=np.array([450., 500.])),),))
    return SimpleNamespace(simulation=SimpleNamespace(
        metrics={"section_target_z_mm": target, "section_resumable_through_z_mm": resumable},
        section_checkpoint=section if checkpoint else None, gun_trace=gun,
        incident=SimpleNamespace(z=np.array([0., 500.])),
        branches={"display_tail": SimpleNamespace(z=np.array([500., 1400.]))}))


def test_completion_uses_executed_metadata_not_display_tail_or_incident_end():
    assert completed_ray_extent(result()) == dict(
        start_z_mm=0., completed_z_mm=750., resumable_z_mm=700.)


def test_legacy_display_without_checkpoint_cannot_claim_completed_extent():
    for value in (None, result(checkpoint=False)):
        assert all(v is None for v in completed_ray_extent(value).values())


@pytest.mark.parametrize("target", [None, float("nan"), float("inf"), -1., "750", True])
def test_invalid_completion_metadata_remains_unavailable(target):
    assert completed_ray_extent(result(target=target))["completed_z_mm"] is None


@pytest.mark.parametrize("resumable", [None, float("nan"), float("inf"), -1., 751.])
def test_invalid_or_missing_restart_position_does_not_hide_completed_result(resumable):
    actual = completed_ray_extent(result(resumable=resumable))
    assert actual["completed_z_mm"] == 750.
    assert actual["resumable_z_mm"] is None
