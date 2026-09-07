"""Relativistic Lorentz propagation for static electromagnetic fields.

All public quantities use SI units.  Momentum, rather than velocity or ray
slope, is the dynamical variable:

    dx/dt = p / (gamma*m)
    dp/dt = q * (E + v cross B)

The module is independent of TEM GUI state and of any particular optical
element.  A field provider only needs to implement the small global-position
interface declared below.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np
from scipy.integrate import solve_ivp


ELEMENTARY_CHARGE_C = 1.602176634e-19
ELECTRON_MASS_KG = 9.1093837015e-31
SPEED_OF_LIGHT_M_PER_S = 299_792_458.0


class MagneticFieldProvider(Protocol):
    def field_at_global_positions_t(self, positions_m):
        ...


class ElectricFieldProvider(Protocol):
    def field_at_global_positions_v_per_m(self, positions_m):
        ...


@dataclass(frozen=True)
class ParticleSpecies:
    """Charge and rest mass shared by one traced particle population."""

    charge_c: float
    rest_mass_kg: float
    name: str = "particle"

    def __post_init__(self):
        if not math.isfinite(float(self.charge_c)):
            raise ValueError("Particle charge must be finite.")
        if (
            not math.isfinite(float(self.rest_mass_kg))
            or self.rest_mass_kg <= 0.0
        ):
            raise ValueError("Particle rest mass must be finite and positive.")


ELECTRON = ParticleSpecies(
    charge_c=-ELEMENTARY_CHARGE_C,
    rest_mass_kg=ELECTRON_MASS_KG,
    name="electron",
)


@dataclass
class RelativisticPhaseSpace:
    """Synchronized position and mechanical momentum at one time."""

    position_m: np.ndarray
    momentum_kg_m_per_s: np.ndarray
    time_s: float = 0.0

    def __post_init__(self):
        position = np.asarray(self.position_m, dtype=float)
        momentum = np.asarray(self.momentum_kg_m_per_s, dtype=float)
        if (
            position.ndim == 0
            or position.shape[-1] != 3
            or momentum.shape != position.shape
        ):
            raise ValueError(
                "Position and momentum must share a final three-vector axis."
            )
        if (
            not np.all(np.isfinite(position))
            or not np.all(np.isfinite(momentum))
            or not math.isfinite(float(self.time_s))
        ):
            raise ValueError("Relativistic phase-space values must be finite.")
        self.position_m = position.copy()
        self.momentum_kg_m_per_s = momentum.copy()
        self.time_s = float(self.time_s)

    def copy(self):
        return RelativisticPhaseSpace(
            self.position_m,
            self.momentum_kg_m_per_s,
            self.time_s,
        )


@dataclass(frozen=True)
class TrajectoryHistory:
    """Synchronized fixed-step or adaptive trajectory samples."""

    time_s: np.ndarray
    position_m: np.ndarray
    momentum_kg_m_per_s: np.ndarray
    method: str
    function_evaluations: int = 0
    terminated_by_event: bool = False

    @property
    def final_state(self):
        return RelativisticPhaseSpace(
            self.position_m[-1],
            self.momentum_kg_m_per_s[-1],
            self.time_s[-1],
        )


def relativistic_gamma(momentum_kg_m_per_s, species=ELECTRON):
    """Return gamma from mechanical momentum."""

    momentum = np.asarray(momentum_kg_m_per_s, dtype=float)
    if momentum.ndim == 0 or momentum.shape[-1] != 3:
        raise ValueError("Momentum must have a final three-vector axis.")
    scaled_squared = np.sum(momentum * momentum, axis=-1) / (
        species.rest_mass_kg * SPEED_OF_LIGHT_M_PER_S
    ) ** 2
    return np.sqrt(1.0 + scaled_squared)


def velocity_from_momentum_m_per_s(
    momentum_kg_m_per_s,
    species=ELECTRON,
):
    momentum = np.asarray(momentum_kg_m_per_s, dtype=float)
    gamma = relativistic_gamma(momentum, species)
    return momentum / (
        gamma[..., np.newaxis] * species.rest_mass_kg
    )


def kinetic_energy_ev_from_momentum(
    momentum_kg_m_per_s,
    species=ELECTRON,
):
    gamma = relativistic_gamma(momentum_kg_m_per_s, species)
    energy_j = (
        (gamma - 1.0)
        * species.rest_mass_kg
        * SPEED_OF_LIGHT_M_PER_S**2
    )
    return energy_j / ELEMENTARY_CHARGE_C


def momentum_from_kinetic_energy_ev(
    kinetic_energy_ev,
    direction,
    species=ELECTRON,
):
    """Return momentum vectors with unit-vector ``direction``."""

    energy_ev = np.asarray(kinetic_energy_ev, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if direction.ndim == 0 or direction.shape[-1] != 3:
        raise ValueError("Direction must have a final three-vector axis.")
    if np.any(~np.isfinite(energy_ev)) or np.any(energy_ev < 0.0):
        raise ValueError("Kinetic energy must be finite and non-negative.")
    if np.any(~np.isfinite(direction)):
        raise ValueError("Momentum direction must be finite.")
    magnitude = np.linalg.norm(direction, axis=-1)
    if np.any(magnitude <= 0.0):
        raise ValueError("Momentum direction must be non-zero.")
    energy_j = energy_ev * ELEMENTARY_CHARGE_C
    rest_energy_j = (
        species.rest_mass_kg * SPEED_OF_LIGHT_M_PER_S**2
    )
    momentum_magnitude = np.sqrt(
        energy_j * (energy_j + 2.0 * rest_energy_j)
    ) / SPEED_OF_LIGHT_M_PER_S
    unit_direction = direction / magnitude[..., np.newaxis]
    return unit_direction * np.asarray(
        momentum_magnitude
    )[..., np.newaxis]


def _validate_time_step(time_step_s):
    time_step = float(time_step_s)
    if not math.isfinite(time_step) or time_step == 0.0:
        raise ValueError("Time step must be finite and non-zero.")
    return time_step


def _magnetic_field_t(provider, positions_m):
    evaluator = getattr(
        provider,
        "field_at_global_positions_t",
        None,
    )
    if not callable(evaluator):
        raise TypeError(
            "Magnetic field provider must implement "
            "field_at_global_positions_t."
        )
    field = np.asarray(evaluator(positions_m), dtype=float)
    if field.shape != np.asarray(positions_m).shape:
        raise ValueError(
            "Magnetic field values must match the position-array shape."
        )
    if not np.all(np.isfinite(field)):
        raise ValueError("Magnetic field values must be finite.")
    return field


def _electric_field_v_per_m(provider, positions_m):
    if provider is None:
        return np.zeros_like(positions_m, dtype=float)
    evaluator = getattr(
        provider,
        "field_at_global_positions_v_per_m",
        None,
    )
    if not callable(evaluator):
        raise TypeError(
            "Electric field provider must implement "
            "field_at_global_positions_v_per_m."
        )
    field = np.asarray(evaluator(positions_m), dtype=float)
    if field.shape != np.asarray(positions_m).shape:
        raise ValueError(
            "Electric field values must match the position-array shape."
        )
    if not np.all(np.isfinite(field)):
        raise ValueError("Electric field values must be finite.")
    return field


def lorentz_derivative(
    position_m,
    momentum_kg_m_per_s,
    magnetic_field,
    *,
    electric_field=None,
    species=ELECTRON,
):
    """Return ``(dx/dt, dp/dt)`` for the relativistic Lorentz equation."""

    position = np.asarray(position_m, dtype=float)
    momentum = np.asarray(momentum_kg_m_per_s, dtype=float)
    if position.shape != momentum.shape:
        raise ValueError("Position and momentum shapes must match.")
    velocity = velocity_from_momentum_m_per_s(momentum, species)
    magnetic = _magnetic_field_t(magnetic_field, position)
    electric = _electric_field_v_per_m(electric_field, position)
    momentum_derivative = species.charge_c * (
        electric + np.cross(velocity, magnetic)
    )
    return velocity, momentum_derivative


def boris_step(
    state,
    time_step_s,
    magnetic_field,
    *,
    electric_field=None,
    species=ELECTRON,
):
    """Advance one synchronized relativistic Boris drift-kick-drift step."""

    if not isinstance(state, RelativisticPhaseSpace):
        raise TypeError("Boris step requires RelativisticPhaseSpace.")
    time_step = _validate_time_step(time_step_s)
    position = state.position_m
    momentum = state.momentum_kg_m_per_s

    velocity_before = velocity_from_momentum_m_per_s(momentum, species)
    midpoint_position = position + 0.5 * time_step * velocity_before
    electric = _electric_field_v_per_m(
        electric_field,
        midpoint_position,
    )
    magnetic = _magnetic_field_t(
        magnetic_field,
        midpoint_position,
    )

    half_electric_impulse = (
        0.5 * species.charge_c * time_step * electric
    )
    momentum_minus = momentum + half_electric_impulse
    gamma_minus = relativistic_gamma(momentum_minus, species)
    rotation_t = (
        species.charge_c
        * time_step
        * magnetic
        / (
            2.0
            * species.rest_mass_kg
            * gamma_minus[..., np.newaxis]
        )
    )
    rotation_s = 2.0 * rotation_t / (
        1.0
        + np.sum(rotation_t * rotation_t, axis=-1)[..., np.newaxis]
    )
    momentum_prime = momentum_minus + np.cross(
        momentum_minus,
        rotation_t,
    )
    momentum_plus = momentum_minus + np.cross(
        momentum_prime,
        rotation_s,
    )
    momentum_after = momentum_plus + half_electric_impulse
    velocity_after = velocity_from_momentum_m_per_s(
        momentum_after,
        species,
    )
    position_after = (
        midpoint_position + 0.5 * time_step * velocity_after
    )
    return RelativisticPhaseSpace(
        position_after,
        momentum_after,
        state.time_s + time_step,
    )


def rk4_step(
    state,
    time_step_s,
    magnetic_field,
    *,
    electric_field=None,
    species=ELECTRON,
):
    """Advance one classical fourth-order Runge-Kutta Lorentz step."""

    if not isinstance(state, RelativisticPhaseSpace):
        raise TypeError("RK4 step requires RelativisticPhaseSpace.")
    time_step = _validate_time_step(time_step_s)
    position = state.position_m
    momentum = state.momentum_kg_m_per_s

    k1_x, k1_p = lorentz_derivative(
        position,
        momentum,
        magnetic_field,
        electric_field=electric_field,
        species=species,
    )
    k2_x, k2_p = lorentz_derivative(
        position + 0.5 * time_step * k1_x,
        momentum + 0.5 * time_step * k1_p,
        magnetic_field,
        electric_field=electric_field,
        species=species,
    )
    k3_x, k3_p = lorentz_derivative(
        position + 0.5 * time_step * k2_x,
        momentum + 0.5 * time_step * k2_p,
        magnetic_field,
        electric_field=electric_field,
        species=species,
    )
    k4_x, k4_p = lorentz_derivative(
        position + time_step * k3_x,
        momentum + time_step * k3_p,
        magnetic_field,
        electric_field=electric_field,
        species=species,
    )
    position_after = position + time_step * (
        k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x
    ) / 6.0
    momentum_after = momentum + time_step * (
        k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p
    ) / 6.0
    return RelativisticPhaseSpace(
        position_after,
        momentum_after,
        state.time_s + time_step,
    )


def integrate_fixed_steps(
    initial_state,
    time_step_s,
    step_count,
    magnetic_field,
    *,
    method="boris",
    electric_field=None,
    species=ELECTRON,
):
    """Return synchronized history for a fixed number of time steps."""

    if not isinstance(initial_state, RelativisticPhaseSpace):
        raise TypeError(
            "Trajectory integration requires RelativisticPhaseSpace."
        )
    time_step = _validate_time_step(time_step_s)
    if isinstance(step_count, bool) or not isinstance(
        step_count, (int, np.integer)
    ):
        raise TypeError("Step count must be an integer.")
    step_count = int(step_count)
    if step_count < 0:
        raise ValueError("Step count must not be negative.")
    method_key = str(method).strip().lower()
    steppers = {
        "boris": boris_step,
        "rk4": rk4_step,
    }
    if method_key not in steppers:
        raise ValueError("Method must be 'boris' or 'rk4'.")

    state = initial_state.copy()
    times = np.empty(step_count + 1, dtype=float)
    positions = np.empty(
        (step_count + 1,) + state.position_m.shape,
        dtype=float,
    )
    momenta = np.empty_like(positions)
    times[0] = state.time_s
    positions[0] = state.position_m
    momenta[0] = state.momentum_kg_m_per_s
    stepper = steppers[method_key]
    for index in range(1, step_count + 1):
        state = stepper(
            state,
            time_step,
            magnetic_field,
            electric_field=electric_field,
            species=species,
        )
        times[index] = state.time_s
        positions[index] = state.position_m
        momenta[index] = state.momentum_kg_m_per_s
    return TrajectoryHistory(
        time_s=times,
        position_m=positions,
        momentum_kg_m_per_s=momenta,
        method=method_key,
    )


def integrate_adaptive_reference(
    initial_state,
    duration_s,
    magnetic_field,
    *,
    electric_field=None,
    species=ELECTRON,
    relative_tolerance=1.0e-10,
    position_tolerance_m=1.0e-12,
    momentum_tolerance_kg_m_per_s=1.0e-32,
    maximum_step_s=math.inf,
    stop_event=None,
):
    """Integrate a high-accuracy DOP853 reference trajectory.

    This solver is intended for validation, not production ray populations.
    ``stop_event`` may be a callable accepting
    ``(position_m, momentum_kg_m_per_s, time_s)`` and returning one scalar.
    Its optional SciPy-compatible ``terminal`` and ``direction`` attributes
    are preserved.
    """

    if not isinstance(initial_state, RelativisticPhaseSpace):
        raise TypeError(
            "Adaptive integration requires RelativisticPhaseSpace."
        )
    duration = _validate_time_step(duration_s)
    relative_tolerance = float(relative_tolerance)
    position_tolerance = float(position_tolerance_m)
    momentum_tolerance = float(momentum_tolerance_kg_m_per_s)
    maximum_step = float(maximum_step_s)
    if (
        not math.isfinite(relative_tolerance)
        or relative_tolerance <= 0.0
    ):
        raise ValueError("Relative tolerance must be positive and finite.")
    if (
        not math.isfinite(position_tolerance)
        or position_tolerance <= 0.0
        or not math.isfinite(momentum_tolerance)
        or momentum_tolerance <= 0.0
    ):
        raise ValueError(
            "Adaptive absolute tolerances must be positive and finite."
        )
    if (
        math.isnan(maximum_step)
        or maximum_step <= 0.0
    ):
        raise ValueError("Maximum adaptive step must be positive.")

    state_shape = initial_state.position_m.shape
    value_count = int(np.prod(state_shape))
    initial_values = np.concatenate((
        initial_state.position_m.reshape(-1),
        initial_state.momentum_kg_m_per_s.reshape(-1),
    ))
    absolute_tolerance = np.concatenate((
        np.full(value_count, position_tolerance),
        np.full(value_count, momentum_tolerance),
    ))

    def unpack(values):
        position = values[:value_count].reshape(state_shape)
        momentum = values[value_count:].reshape(state_shape)
        return position, momentum

    def derivative(_time_s, values):
        position, momentum = unpack(values)
        velocity, momentum_derivative = lorentz_derivative(
            position,
            momentum,
            magnetic_field,
            electric_field=electric_field,
            species=species,
        )
        return np.concatenate((
            velocity.reshape(-1),
            momentum_derivative.reshape(-1),
        ))

    events = None
    if stop_event is not None:
        if not callable(stop_event):
            raise TypeError("Adaptive stop event must be callable.")

        def event_wrapper(time_s, values):
            position, momentum = unpack(values)
            event_value = float(
                stop_event(position, momentum, time_s)
            )
            if not math.isfinite(event_value):
                raise ValueError(
                    "Adaptive stop event must return a finite scalar."
                )
            return event_value

        event_wrapper.terminal = bool(
            getattr(stop_event, "terminal", True)
        )
        event_wrapper.direction = float(
            getattr(stop_event, "direction", 0.0)
        )
        events = event_wrapper

    start_time = float(initial_state.time_s)
    solution = solve_ivp(
        derivative,
        (start_time, start_time + duration),
        initial_values,
        method="DOP853",
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        max_step=maximum_step,
        events=events,
    )
    if not solution.success:
        raise RuntimeError(
            "Adaptive DOP853 integration failed: "
            + str(solution.message)
        )
    values = solution.y.T
    positions = values[:, :value_count].reshape(
        (len(solution.t),) + state_shape
    )
    momenta = values[:, value_count:].reshape(
        (len(solution.t),) + state_shape
    )
    return TrajectoryHistory(
        time_s=np.asarray(solution.t, dtype=float),
        position_m=positions,
        momentum_kg_m_per_s=momenta,
        method="adaptive_dop853",
        function_evaluations=int(solution.nfev),
        terminated_by_event=bool(
            events is not None
            and solution.t_events
            and len(solution.t_events[0]) > 0
        ),
    )
