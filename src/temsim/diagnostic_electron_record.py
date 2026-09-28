"""Independent virtual-electron records and executed-result ownership.

The controller advances ``generation`` whenever it replaces its captured field
scene. A generation therefore identifies one controller-owned field snapshot,
not a physical source or a persistent field identity. Session restoration keeps
saved results historical until a new execution establishes current ownership.

These predicates decide presentation and exact diagnostic reuse only. They do
not validate particle physics, admit a main simulation source, or execute work.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from temsim.magnetic_test_particle import TestElectronSettings, TestElectronTrajectory


@dataclass
class ElectronRecord:
    key: str
    label: str
    colour: str
    settings: TestElectronSettings
    checked: bool = True
    trajectory: TestElectronTrajectory | None = None
    revision: int = 0
    attempted: bool = False
    error: str | None = None
    ready_at: float = 0.
    trajectory_generation: int = -1
    trajectory_settings: TestElectronSettings | None = None
    progress_trajectory: TestElectronTrajectory | None = None
    progress_key: tuple | None = None
    _path_signature: object | None = None
    _paths: dict = field(default_factory=dict)
    historical: bool = False
    history_dependency: object | None = None


def record_request_key(record: ElectronRecord, generation: int) -> tuple:
    """Identify the requested record version in one captured field generation."""
    return generation, record.settings, record.key, record.revision


def request_matches_record(record: ElectronRecord | None, key, generation: int) -> bool:
    """Reject a removed record, another electron, old fields or an old edit."""
    return (record is not None and isinstance(key, tuple) and len(key) == 4
            and key == record_request_key(record, generation))


def trajectory_is_visible(record: ElectronRecord, generation: int) -> bool:
    """Allow own-field earlier settings and explicitly labelled saved history.

    A live cancelled/in-progress trajectory belongs in the progress slot. A
    saved historical prefix remains inspectable with its incomplete status;
    visibility must never make it current or eligible for exact reuse.
    """
    result = record.trajectory
    return (result is not None and (record.historical or (
        result.reason not in {"cancelled", "in_progress"}
        and record.trajectory_generation == generation)))


def trajectory_is_current(record: ElectronRecord, generation: int, *, typing: bool = False) -> bool:
    """Test exact executed settings and captured fields, excluding saved history.

    The terminal solver outcome need not be successful: an executed step-limit
    outcome is still the outcome of those settings. It must retain its reason
    and completion flag and cannot become a production continuation checkpoint.
    """
    return (not typing and not record.historical
            and trajectory_is_visible(record, generation)
            and record.trajectory.reason not in {"cancelled", "in_progress"}
            and record.trajectory_generation == generation
            and record.trajectory_settings == record.settings)


def progress_belongs_to_record(record: ElectronRecord, generation: int) -> bool:
    """Allow an actual unfinished prefix for this electron and captured field.

    During continuous slider changes the active sampled request may predate
    the latest settings. Its prefix may stay visible as earlier-settings work.
    """
    key, result = record.progress_key, record.progress_trajectory
    return (isinstance(key, tuple) and len(key) == 4
            and key[0] == generation and key[2] == record.key
            and result is not None and result.reason == "in_progress"
            and not result.completed)


def progress_is_current(record: ElectronRecord, generation: int, *, typing: bool = False) -> bool:
    """Test whether a prefix also belongs to the currently requested edit."""
    return (not typing and not record.historical
            and progress_belongs_to_record(record, generation)
            and request_matches_record(record, record.progress_key, generation))


__all__ = [
    "ElectronRecord", "record_request_key", "request_matches_record",
    "trajectory_is_current", "trajectory_is_visible", "progress_belongs_to_record",
    "progress_is_current",
]
