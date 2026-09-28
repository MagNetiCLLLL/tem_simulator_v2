"""Validate detached diagnostic payloads without changing numerical results.

These checks establish the private IPC data contract only. They do not qualify
energy conservation, solver accuracy, or a microscope calculation.
"""
from __future__ import annotations

import math

import numpy as np

from temsim.magnetic_test_particle import TestElectronSettings, TestElectronTrajectory


class ElectronProtocolValueError(ValueError):
    """A response payload cannot be accepted as the requested diagnostic data."""


def _require(condition, message):
    if not condition:
        raise ElectronProtocolValueError(message)


def _real_scalar(value, name, *, positive=False, allow_nan=False):
    _require(isinstance(value, (int, float, np.integer, np.floating))
             and not isinstance(value, (bool, np.bool_)), f"{name} must be a real scalar")
    _require((allow_nan and math.isnan(value)) or math.isfinite(value), f"{name} must be finite")
    if not math.isnan(value):
        _require(value > 0. if positive else value >= 0., f"{name} must be {'positive' if positive else 'nonnegative'}")


def _identity(value, name):
    _require(value is None or (isinstance(value, str) and bool(value.strip())),
             f"{name} must be a nonempty identity or None")


def _notes(value):
    _require(isinstance(value, tuple) and all(isinstance(note, str) for note in value),
             "notes must be a tuple of text entries")


def _box(value, name):
    try:
        bounds = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ElectronProtocolValueError(f"{name} must be numeric XYZ data") from exc
    _require(bounds.dtype.kind in "fiu" and bounds.shape == (2, 3)
             and np.isfinite(bounds).all() and np.all(bounds[1] > bounds[0]),
             f"{name} must be a finite increasing XYZ box")


def validate_scene_metadata(value, *, process_identity):
    """Return the original owned metadata, or reject malformed display inputs."""
    # Import lazily: the execution backend imports these validators itself.
    from temsim.test_electron_execution import RemoteElectronScene

    _require(isinstance(value, RemoteElectronScene), "Prepared scene has an invalid metadata type")
    _require(isinstance(process_identity, str) and bool(process_identity.strip())
             and value.process_identity == process_identity, "Prepared scene process identity does not match")
    _require(isinstance(value.token, str) and bool(value.token.strip()), "Prepared scene token is empty or invalid")
    try:
        point = np.asarray(value.initial_position_m)
        for name in ("bounds_m", "diagnostic_bounds_m"):
            _box(getattr(value, name), "Prepared scene " + name)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ElectronProtocolValueError):
            raise
        raise ElectronProtocolValueError("Prepared scene coordinates must be numeric XYZ data") from exc
    _require(point.dtype.kind in "fiu" and point.shape == (3,) and np.isfinite(point).all(),
             "Prepared scene initial position must be finite XYZ metres")
    _real_scalar(value.initial_energy_ev, "Prepared scene initial energy", positive=True)
    _real_scalar(value.default_path_length_m, "Prepared scene default path length", positive=True)
    for name in ("physical_identity", "numerical_identity", "transport_identity"):
        _identity(getattr(value, name), name)
    _notes(value.notes)
    _require(isinstance(value.support_metadata, tuple), "Prepared scene support metadata must be a tuple")
    from temsim.magnetic_field_scene import MagneticFieldSupport
    for support in value.support_metadata:
        _require(isinstance(support, MagneticFieldSupport), "Prepared scene support entry has an invalid type")
        for name in ("key", "model", "limitation"):
            _require(isinstance(getattr(support, name), str), f"Support {name} must be text")
        _box(support.bounds_m, "Support bounds")
        _real_scalar(support.radial_limit_m, "Support radial limit", positive=True)
        for name in ("physical_identity", "numerical_identity"):
            _identity(getattr(support, name), "Support " + name)
        for name in ("reference_momentum_kg_m_s", "reference_charge_c", "captured_time_s"):
            scalar = getattr(support, name)
            _require(scalar is None or (isinstance(scalar, (int, float, np.integer, np.floating))
                     and not isinstance(scalar, (bool, np.bool_)) and math.isfinite(scalar)),
                     f"Support {name} must be a finite scalar or None")
    return value


def _array(value, name, shape, *, finite=True):
    _require(isinstance(value, np.ndarray) and value.dtype.kind in "fiu" and value.shape == shape,
             f"Trajectory {name} must be a real array with shape {shape}")
    if finite:
        _require(np.isfinite(value).all(), f"Trajectory {name} must contain finite values")
    return value


def validate_trajectory_payload(value, *, settings, scene, execution_identity=None, progress=False):
    """Check shape, chronology, status and captured identities; return unchanged.

    Completed domain exits and physical stops retain the solver's meaning.
    Unknown initial electric potential is accepted only on an unpropagated
    outside-domain/cancelled result, with unknown energy-invariant readouts.
    A zero speed/direction/momentum is permitted; no physical accuracy test is
    hidden inside this IPC validation.
    """
    _require(isinstance(value, TestElectronTrajectory), "Diagnostic response has an invalid trajectory type")
    _require(isinstance(settings, TestElectronSettings), "Trajectory validation requires its executed settings")
    _require(isinstance(value.steps, (int, np.integer)) and not isinstance(value.steps, (bool, np.bool_))
             and 0 <= value.steps <= settings.max_steps, "Trajectory steps exceed the executed request budget")
    count = int(value.steps) + 1
    for name in ("positions_m", "directions", "momentum_kg_m_per_s"):
        _array(getattr(value, name), name, (count, 3))
    for name in ("time_s", "path_length_m", "kinetic_energy_ev", "speed_m_per_s"):
        array = _array(getattr(value, name), name, (count,))
        _require(np.all(array >= 0.), f"Trajectory {name} cannot be negative")
    for name in ("time_s", "path_length_m"):
        array = getattr(value, name)
        _require(array[0] == 0. and np.all(array[1:] >= array[:-1]),
                 f"Trajectory {name} must start at zero and be nondecreasing")
    _require(isinstance(value.reason, str) and bool(value.reason.strip()), "Trajectory stop reason must be nonempty text")
    _require(isinstance(value.completed, (bool, np.bool_)), "Trajectory completed must be Boolean")
    failures = {"initial_outside_domain", "step_limit", "cancelled", "numerical_limit", "in_progress"}
    completed = value.reason not in failures and not value.reason.startswith("unsupported_field:")
    _require(bool(value.completed) == completed, "Trajectory reason and completed flag disagree")
    _require((value.reason == "in_progress") == bool(progress),
             "Progress must be an unfinished prefix and a terminal result must not be a prefix")
    potential = _array(value.electrostatic_potential_v, "electrostatic_potential_v", (count,), finite=False)
    unknown_initial = (count == 1 and value.reason in {"initial_outside_domain", "cancelled"}
                       and bool(np.isnan(potential[0])))
    _require(unknown_initial or np.isfinite(potential).all(), "Trajectory electric potential has unexpected nonfinite values")
    for name in ("energy_invariant_error_ev", "energy_invariant_relative_error"):
        scalar = getattr(value, name)
        _real_scalar(scalar, name, allow_nan=unknown_initial)
        _require(not unknown_initial or math.isnan(scalar), "An unknown initial potential cannot have a known energy invariant")
    for name, expected in (
        ("physical_field_identity", getattr(scene, "physical_identity", None)),
        ("numerical_field_identity", getattr(scene, "numerical_identity", None)),
        ("execution_identity", execution_identity),
    ):
        _identity(getattr(value, name), name)
        _identity(expected, "Expected " + name)
        _require(getattr(value, name) == expected, f"Trajectory {name} does not match the executed request")
    _notes(value.notes)
    return value


__all__ = ["ElectronProtocolValueError", "validate_scene_metadata", "validate_trajectory_payload"]
