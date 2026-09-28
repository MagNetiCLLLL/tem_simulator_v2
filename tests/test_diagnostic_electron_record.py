"""Ownership rules only; no field preparation or numerical integration."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.diagnostic_electron_record import (
    ElectronRecord, progress_belongs_to_record, progress_is_current,
    record_request_key, request_matches_record, trajectory_is_current,
    trajectory_is_visible,
)
from temsim.magnetic_test_particle import TestElectronSettings, TestElectronTrajectory


def trajectory(reason="path_length", completed=True):
    vectors, scalars = np.zeros((1, 3)), np.zeros(1)
    return TestElectronTrajectory(
        positions_m=vectors, directions=vectors, time_s=scalars, path_length_m=scalars,
        energy_invariant_error_ev=0., energy_invariant_relative_error=0.,
        reason=reason, completed=completed, steps=0, kinetic_energy_ev=scalars,
        electrostatic_potential_v=scalars, speed_m_per_s=scalars,
        momentum_kg_m_per_s=vectors, notes=(),
    )


@pytest.fixture
def record():
    settings = TestElectronSettings()
    return ElectronRecord(
        key="electron-1", label="Electron 1", colour="#ffd166", settings=settings,
        trajectory=trajectory(), trajectory_generation=3, trajectory_settings=settings,
        revision=4,
    )


def test_current_trajectory_requires_executed_settings_and_field_generation(record):
    assert trajectory_is_current(record, 3)
    assert not trajectory_is_current(record, 4)
    assert not trajectory_is_current(record, 3, typing=True)
    record.settings = replace(record.settings, kinetic_energy_ev=.4)
    assert not trajectory_is_current(record, 3)
    assert trajectory_is_visible(record, 3)
    assert not trajectory_is_visible(record, 4)


def test_restored_history_is_visible_but_never_current_even_with_same_generation(record):
    record.historical = True
    assert trajectory_is_visible(record, 3)
    assert trajectory_is_visible(record, 12)
    assert not trajectory_is_current(record, 3)
    assert not trajectory_is_current(record, 12)


@pytest.mark.parametrize("reason", ["cancelled", "in_progress"])
def test_unaccepted_prefix_cannot_enter_terminal_current_or_display_slot(record, reason):
    record.trajectory = trajectory(reason, False)
    assert not trajectory_is_visible(record, 3)
    assert not trajectory_is_current(record, 3)
    record.historical = True
    assert trajectory_is_visible(record, 3)
    assert not trajectory_is_current(record, 3)


def test_executed_terminal_budget_outcome_retains_its_unsuccessful_status(record):
    record.trajectory = trajectory("step_limit", False)
    assert trajectory_is_current(record, 3)
    assert not record.trajectory.completed
    assert record.trajectory.reason == "step_limit"


def test_no_trajectory_is_never_current_or_visible(record):
    record.trajectory = None
    assert not trajectory_is_current(record, 3)
    assert not trajectory_is_visible(record, 3)
    record.historical = True
    assert not trajectory_is_visible(record, 3)


@pytest.mark.parametrize("changed", ["generation", "settings", "key", "revision"])
def test_request_identity_rejects_each_changed_ownership_component(record, changed):
    key = record_request_key(record, 3)
    assert request_matches_record(record, key, 3)
    if changed == "generation":
        assert not request_matches_record(record, key, 4)
    else:
        setattr(record, changed, {
            "settings": replace(record.settings, kinetic_energy_ev=.4),
            "key": "another-electron", "revision": 5,
        }[changed])
        assert not request_matches_record(record, key, 3)


@pytest.mark.parametrize("key", [None, (), (3,), [3, TestElectronSettings(), "electron-1", 4]])
def test_invalid_or_missing_request_is_rejected(record, key):
    assert not request_matches_record(record, key, 3)
    assert not request_matches_record(None, key, 3)


def test_progress_current_requires_record_id_generation_settings_and_edit_revision(record):
    record.progress_trajectory = trajectory("in_progress", False)
    record.progress_key = record_request_key(record, 3)
    assert progress_belongs_to_record(record, 3)
    assert progress_is_current(record, 3)
    assert not progress_is_current(record, 3, typing=True)
    record.revision += 1
    assert progress_belongs_to_record(record, 3)
    assert not progress_is_current(record, 3)
    record.settings = replace(record.settings, kinetic_energy_ev=.5)
    assert progress_belongs_to_record(record, 3)
    assert not progress_is_current(record, 3)
    assert not progress_belongs_to_record(record, 4)
    record.key = "another-electron"
    assert not progress_belongs_to_record(record, 3)


def test_historical_prefix_is_not_mislabelled_as_current(record):
    record.progress_trajectory = trajectory("in_progress", False)
    record.progress_key = record_request_key(record, 3)
    record.historical = True
    assert not progress_is_current(record, 3)


@pytest.mark.parametrize("reason,completed", [("path_length", True), ("cancelled", False), ("in_progress", True)])
def test_nonprefix_payload_is_not_accepted_as_live_progress(record, reason, completed):
    record.progress_key = record_request_key(record, 3)
    record.progress_trajectory = trajectory(reason, completed)
    assert not progress_belongs_to_record(record, 3)
    assert not progress_is_current(record, 3)


def test_record_mutable_display_caches_are_independent(record):
    another = ElectronRecord("electron-2", "Electron 2", "#abcdef", record.settings)
    record._paths[True] = ("only-first",)
    assert another._paths == {}
    assert not another.historical


def test_existing_gui_import_preserves_the_public_record_type():
    from temsim.gui.magnetic_test_electron import ElectronRecord as GuiRecord
    assert GuiRecord is ElectronRecord
