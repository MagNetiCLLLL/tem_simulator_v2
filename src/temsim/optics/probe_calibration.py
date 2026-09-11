"""Physical two-hexapole calibration against the incident source bundle.

This is an engineering calibration of this simulator, not an OEM current
table.  No ray positions are rescaled or circularised: candidate fields are
always traced through the column and its physical apertures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from temsim.physics.aperture_clipping import clip_segment
from temsim.physics.beam_statistics import transverse_beam_statistics
from temsim.physics.column_wall import clip_column_wall
from temsim.physics.core import propagate


@dataclass(frozen=True)
class IncidentProbeMeasurement:
    x_m: np.ndarray
    y_m: np.ndarray
    tx_rad: np.ndarray
    ty_rad: np.ndarray
    alive: np.ndarray
    weights: np.ndarray

    @property
    def statistics(self):
        return transverse_beam_statistics(
            self.x_m, self.y_m, self.tx_rad, self.ty_rad,
            alive=self.alive, weights=self.weights,
        )


class IncidentProbeModel:
    """Reuse one physical gun emission while changing downstream optics."""

    def __init__(self, state):
        self.state = state
        from temsim.optics.electron_gun.source import trace_source_to_exit
        self.gun_trace = trace_source_to_exit(state)
        self.emitted = self.gun_trace.exit_bundle

    def trace(self, *, spherical=True, hexapoles=True, step_mm=None):
        from temsim.optics.direct_alignment import _pre_sample_kick_events

        state = self.state
        source = float(state.electron_gun.exit_plane_z_mm)
        sample = float(state.sample.z_mm)
        apertures = tuple(
            float(aperture.z_mm) for aperture in state.apertures
            if aperture.enabled and getattr(aperture, "installed", True)
            and source <= float(aperture.z_mm) <= sample
        )
        bundle = self.emitted
        z, x, tx, y, ty = propagate(
            state, source, sample,
            bundle.x_m, bundle.tx_rad, bundle.y_m, bundle.ty_rad,
            events=_pre_sample_kick_events(state),
            energy_offset_ev=bundle.energy_offset_ev,
            save_z_mm=apertures, maximum_step_mm=step_mm,
            include_spherical_aberration=spherical,
            include_hexapole=hexapoles,
        )
        alive = np.asarray(bundle.alive, dtype=bool).copy()
        blocked_z = np.asarray(self.gun_trace.blocked_z_mm, dtype=float).copy()
        blocked_key = list(self.gun_trace.blocked_key)
        alive, blocked_z, blocked_key = clip_segment(
            state, z, x, y, alive, blocked_z, blocked_key,
        )
        alive, _, _ = clip_column_wall(
            state, z, x, y, alive, blocked_z, blocked_key,
        )
        return IncidentProbeMeasurement(
            np.asarray(x[-1], dtype=float), np.asarray(y[-1], dtype=float),
            np.asarray(tx[-1], dtype=float), np.asarray(ty[-1], dtype=float),
            alive, np.asarray(bundle.weight, dtype=float),
        )


def fit_probe_hexapoles(model: IncidentProbeModel, *, maximum_evaluations=40):
    """Fit HP2/HP1 amplitude and relative azimuth to nonlinear ray errors.

    The target is the same source propagated through the linear column, so
    simply broadening a spot until it looks round cannot reduce this error.
    Fields stay enabled.  This mutates only the supplied calibration state;
    callers must work on a snapshot and validate before installing a result.
    """

    state = model.state
    hp2 = state.hp2_hexapole
    hp1 = state.hp1_hexapole
    if not (hp2.enabled and hp1.enabled):
        raise ValueError("Both principal probe hexapoles must be enabled")
    baseline = model.trace(spherical=False, hexapoles=False)
    if (not np.all(np.isfinite(baseline.weights))
            or np.any(baseline.weights < 0.0)):
        raise ValueError("Calibration weights must be finite and nonnegative")
    mask = baseline.alive & (baseline.weights > 0.0)
    weight = baseline.weights[mask]
    total_weight = float(weight.sum())
    if not np.any(mask) or not np.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("Calibration requires surviving rays with positive current")
    weight = np.sqrt(weight / total_weight)
    scale = 1.0e5
    initial = np.array((
        hp2.strength_m3 / scale,
        hp1.strength_m3 / scale,
        hp1.orientation_rad,
    ))

    def install(vector):
        hp2.strength_m3 = float(vector[0] * scale)
        hp1.strength_m3 = float(vector[1] * scale)
        hp1.orientation_rad = float(vector[2])

    def residual(vector):
        install(vector)
        measured = model.trace()
        if not np.all(measured.alive[mask]):
            raise ValueError("Corrector candidate clips the calibration pupil")
        dx = (measured.x_m[mask] - baseline.x_m[mask]) * 1.0e9
        dy = (measured.y_m[mask] - baseline.y_m[mask]) * 1.0e9
        return np.r_[weight * dx, weight * dy]

    before = residual(initial)
    try:
        result = least_squares(
            residual, initial,
            bounds=(
                (0.01, 0.01, -np.pi / 3),
                (hp2.maximum_strength_m3 / scale,
                 hp1.maximum_strength_m3 / scale, np.pi / 3),
            ),
            diff_step=1.0e-3, x_scale="jac", max_nfev=maximum_evaluations,
            ftol=1.0e-9, xtol=1.0e-9, gtol=1.0e-9,
        )
        after = residual(result.x)
        if not result.success or np.linalg.norm(after) > np.linalg.norm(before):
            raise ValueError(f"Probe-corrector fit did not converge: {result.message}")
    except Exception:
        install(initial)
        raise
    install(result.x)
    return {
        "hp2_strength_m3": float(hp2.strength_m3),
        "hp1_strength_m3": float(hp1.strength_m3),
        "hp1_orientation_rad": float(hp1.orientation_rad),
        "nonlinear_error_before_nm": float(np.linalg.norm(before)),
        "nonlinear_error_after_nm": float(np.linalg.norm(after)),
        "evaluations": int(result.nfev),
    }
