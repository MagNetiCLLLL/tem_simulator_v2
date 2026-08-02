"""Fast coupled IL/P1/P2 magnification and camera-length controller.

The optimiser uses the same distributed three-Gaussian field model as the ray
tracer, but propagates only a 2x2 paraxial transfer matrix on a coarser temporary
z grid. A full-resolution transfer calculation is performed once at the end for
reporting. The full ray simulation remains in MainWindow.calc().
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np

from temsim.physics.core import transfer
from temsim.component_keys import (
    INTERMEDIATE_LENS,
    PROJECTOR_LENS_1,
    PROJECTOR_LENS_2,
)

E = 1.602176634e-19
M = 9.1093837015e-31
C = 299792458.0
KEYS = (INTERMEDIATE_LENS, PROJECTOR_LENS_1, PROJECTOR_LENS_2)

@dataclass
class Result:
    success: bool
    requested: float
    achieved: float
    relay_error: float
    strengths: dict
    message: str
    fast_iterations: int
    full_transfer_calls: int

def _lens_map(state):
    return {lens.key: lens for lens in state.lenses}

def _get_vector(state):
    lenses = _lens_map(state)
    return np.array([lenses[key].percent for key in KEYS], dtype=float)

def _set_vector(state, vector):
    lenses = _lens_map(state)
    for key, value in zip(KEYS, vector):
        lenses[key].percent = float(value)

def _electron_momentum(voltage_kv):
    kinetic = E * float(voltage_kv) * 1000.0
    rest = M * C * C
    return math.sqrt(kinetic * kinetic + 2.0 * kinetic * rest) / C

def _lens_profile(z, lens):
    """Bz profile at 100 percent excitation."""
    profile = np.zeros_like(z)
    if not getattr(lens, "enabled", True):
        return profile
    for gaussian in lens.gaussian:
        sigma = max(1.0e-12, gaussian.sigma * lens.a_mm)
        centre = lens.z_mm + gaussian.offset * lens.a_mm
        profile += (
            lens.b0_t
            * gaussian.amplitude
            * np.exp(-0.5 * ((z - centre) / sigma) ** 2)
        )
    return profile

class FastProjectorModel:
    """Cached fast transfer model for repeated IL/P1/P2 evaluations."""

    def __init__(self, state, step_mm=2.0):
        self.state = state
        self.step_mm = max(0.5, float(step_mm))
        self.z = np.arange(
            state.sample.z_mm,
            state.camera.z_mm + self.step_mm / 2.0,
            self.step_mm,
            dtype=float,
        )
        self.h_m = self.step_mm * 1.0e-3
        self.lenses = _lens_map(state)

        self.variable_profiles = np.vstack(
            [_lens_profile(self.z, self.lenses[key]) for key in KEYS]
        )
        self.fixed_field = np.zeros_like(self.z)
        for lens in state.lenses:
            if lens.key not in KEYS:
                self.fixed_field += _lens_profile(self.z, lens) * lens.percent / 100.0

        momentum = _electron_momentum(state.beam_voltage_kv)
        self.field_to_k = (E / (2.0 * momentum)) ** 2

        # Stigmators are intentionally not included in the scalar x transfer
        # used for magnification control. Their normal operating values should
        # be small and they must not drive the projector magnification table.

    def matrix(self, vector):
        field = self.fixed_field + np.tensordot(
            np.asarray(vector, dtype=float) / 100.0,
            self.variable_profiles,
            axes=(0, 0),
        )
        k = self.field_to_k * field * field

        # Propagate the 2x2 matrix with a symplectic kick-drift step. This is
        # much cheaper than RK4 ray propagation and stable for optimisation.
        a, b, c, d = 1.0, 0.0, 0.0, 1.0
        h = self.h_m
        for value in k[:-1]:
            c -= h * value * a
            d -= h * value * b
            a += h * c
            b += h * d
        return np.array([[a, b], [c, d]], dtype=float)

def actual_value(state):
    matrix = transfer(state, state.sample.z_mm, state.camera.z_mm)
    if state.projector_mode == "image":
        return abs(float(matrix[0, 0]))
    return abs(float(matrix[0, 1]))

def slider_to_target(state, value):
    u = np.clip(float(value), 0.0, 100.0) / 100.0
    if state.projector_mode == "image":
        return 10.0 ** (2.0 + 4.0 * u)
    return 10.0 ** (
        math.log10(0.05)
        + u * (math.log10(30.0) - math.log10(0.05))
    )

def target_to_slider(state, target):
    target = max(float(target), 1.0e-15)
    if state.projector_mode == "image":
        return float(
            np.clip((math.log10(target) - 2.0) / 4.0 * 100.0, 0.0, 100.0)
        )
    return float(
        np.clip(
            (math.log10(target) - math.log10(0.05))
            / (math.log10(30.0) - math.log10(0.05))
            * 100.0,
            0.0,
            100.0,
        )
    )

def _residual(state, model, vector, target, initial):
    matrix = model.matrix(vector)
    if state.projector_mode == "image":
        achieved = max(abs(float(matrix[0, 0])), 1.0e-15)
        primary = np.array(
            [math.log(achieved / target), float(matrix[0, 1]) / 0.10]
        )
    else:
        achieved = max(abs(float(matrix[0, 1])), 1.0e-15)
        primary = np.array(
            [math.log(achieved / target), float(matrix[0, 0])]
        )

    # Keeps the solution on the nearby projector branch.
    regularisation = 0.001 * (vector - initial) / np.maximum(abs(initial), 25.0)
    return np.r_[primary, regularisation]

def optimise(state, target, iterations=16, optimiser_step_mm=0.5):
    """Optimise IL/P1/P2 quickly and validate once at full resolution."""
    target = float(target)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("Target must be a positive finite value")

    model = FastProjectorModel(state, step_mm=optimiser_step_mm)
    initial = _get_vector(state)
    vector = initial.copy()
    lenses = _lens_map(state)
    lower = np.zeros(3)
    upper = np.array([lenses[key].max_percent for key in KEYS], dtype=float)
    best = vector.copy()
    best_cost = float("inf")
    damping = 2.0e-2
    completed = 0

    for completed in range(1, int(iterations) + 1):
        residual = _residual(state, model, vector, target, initial)
        cost = float(residual @ residual)
        if cost < best_cost:
            best_cost = cost
            best = vector.copy()
        if cost < 1.0e-7:
            break

        # Forward differences require only three extra fast matrix evaluations,
        # rather than six central-difference evaluations.
        jacobian = np.empty((len(residual), 3), dtype=float)
        for column in range(3):
            step = max(0.08, abs(vector[column]) * 0.003)
            trial = vector.copy()
            trial[column] = min(upper[column], trial[column] + step)
            denominator = trial[column] - vector[column]
            if denominator <= 0.0:
                jacobian[:, column] = 0.0
            else:
                jacobian[:, column] = (
                    _residual(state, model, trial, target, initial) - residual
                ) / denominator

        lhs = jacobian.T @ jacobian + damping * np.eye(3)
        rhs = -(jacobian.T @ residual)
        delta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        delta = np.clip(delta, -15.0, 15.0)
        trial = np.clip(vector + delta, lower, upper)
        trial_residual = _residual(state, model, trial, target, initial)
        if float(trial_residual @ trial_residual) < cost:
            vector = trial
            damping = max(1.0e-7, damping * 0.35)
        else:
            damping = min(1.0e5, damping * 6.0)

    # Capture the last accepted vector as well.
    final_fast_cost = float(
        _residual(state, model, vector, target, initial)
        @ _residual(state, model, vector, target, initial)
    )
    if final_fast_cost < best_cost:
        best = vector.copy()

    _set_vector(state, best)

    # Only one expensive full RK4 transfer call per slider release.
    full_matrix = transfer(state, state.sample.z_mm, state.camera.z_mm)
    if state.projector_mode == "image":
        achieved = abs(float(full_matrix[0, 0]))
        relay_error = abs(float(full_matrix[0, 1]))
        unit = "x"
    else:
        achieved = abs(float(full_matrix[0, 1]))
        relay_error = abs(float(full_matrix[0, 0]))
        unit = "m"

    relative_error = abs(math.log(max(achieved, 1.0e-15) / target))
    success = relative_error < 0.05 and relay_error < 0.02
    strengths = {key: float(value) for key, value in zip(KEYS, best)}
    message = (
        f"Target {target:.5g} {unit}; actual {achieved:.5g} {unit}; "
        f"relay residual {relay_error:.3g}; fast iterations {completed}."
    )
    if not success:
        message += " Best local solution shown."

    return Result(
        success,
        target,
        achieved,
        relay_error,
        strengths,
        message,
        completed,
        1,
    )
