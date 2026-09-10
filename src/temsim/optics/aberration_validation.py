"""Explicit numerical studies; reports do not tune or certify the microscope."""

from copy import copy, deepcopy
from dataclasses import asdict, dataclass
import numpy as np

from temsim.optics.aberration_basis import (
    ABERRATION_SCHEMA, CARTESIAN_NAMES, cartesian_coefficients, predict_displacement_m,
)
from temsim.optics.field_aberrations import derive_field_aberrations, pupil_rays


def compare_refinements(values, levels, *, unit, absolute_floor, relative_target=.01):
    """Per-component bounds prevent a large C5 from concealing an A1 failure."""
    arrays = np.asarray(values, float)
    floors = np.asarray(absolute_floor, float)
    if (arrays.ndim < 1 or len(arrays) < 3 or len(levels) != len(arrays)
            or not np.isfinite(arrays).all() or not np.isfinite(floors).all()
            or np.any(floors < 0) or not np.isfinite(relative_target) or relative_target < 0):
        raise ValueError("Convergence requires at least three finite cases and nonnegative tolerances")
    rows = []
    for level, previous, current in zip(levels[1:], arrays, arrays[1:]):
        delta = np.abs(current-previous)
        bound = floors + relative_target*np.abs(current)
        rows.append({"level": level, "absolute_change": delta.tolist(), "acceptance_bound": bound.tolist(),
                     "status": "PASS" if np.all(delta <= bound) else "FAIL"})
    return {"unit": unit, "levels": list(levels), "values": arrays.tolist(),
            "absolute_floor": floors.tolist(), "relative_target": relative_target, "comparisons": rows,
            "status": "PASS" if all(row["status"] == "PASS" for row in rows[-2:]) else "INCONCLUSIVE"}


def convergence_study(cases, *, system, validation_semiangle_mrad,
                      coefficient_floors_mm, displacement_floor_m, relative_target=.01,
                      observable=None, observable_unit=None, observable_floor=None):
    """Fit independently prepared states in declared refinement order.

    cases = [(label, state), ...]. For a field-grid study callers must supply
    newly solved fields / independent maps; this function never resamples an
    existing map and presents that as convergence. Use one varying numerical
    axis per study. An optional physical observable runs on each state/result.
    """
    cases = list(cases)
    if len(cases) < 3 or len({label for label, _ in cases}) != len(cases):
        raise ValueError("Provide at least three uniquely labelled refinement cases")
    alpha = float(validation_semiangle_mrad)*1e-3
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Declare a positive common validation semi-angle")
    if np.asarray(coefficient_floors_mm).shape != (len(CARTESIAN_NAMES),):
        raise ValueError("Declare an absolute mm floor for every Cartesian coefficient")
    rows, coefficients, displacements, signals = [], [], [], []
    for label, state in cases:
        _, result, evidence = derive_field_aberrations(state, system)
        if alpha*1e3 > evidence["fit_semiangle_mrad"]*(1+1e-12):
            raise ValueError("Common validation pupil extends outside a fitted aperture")
        vector = cartesian_coefficients(result)  # Transport focus remains excluded.
        coefficients.append(vector)
        displacements.append(predict_displacement_m(pupil_rays(alpha, holdout=True), vector))
        rows.append({"label": label, "coefficients": asdict(result), "fit": evidence})
        if observable is not None:
            if not observable_unit or observable_floor is None:
                raise ValueError("Declare physical observable units and absolute tolerance")
            signals.append(observable(state, result))
    labels = [label for label, _ in cases]
    comparisons = {
        "coefficients": compare_refinements(coefficients, labels, unit="mm", absolute_floor=coefficient_floors_mm, relative_target=relative_target),
        "common_pupil_displacement": compare_refinements(displacements, labels, unit="m", absolute_floor=displacement_floor_m, relative_target=relative_target),
    }
    if observable is not None:
        comparisons["target_signal"] = compare_refinements(signals, labels, unit=observable_unit,
                                                           absolute_floor=observable_floor, relative_target=relative_target)
    return {"schema": ABERRATION_SCHEMA, "system": system, "cases": rows,
            "validation_semiangle_mrad": validation_semiangle_mrad,
            "cartesian_names": CARTESIAN_NAMES, "comparisons": comparisons,
            "status": "PASS" if all(row["status"] == "PASS" for row in comparisons.values()) else "INCONCLUSIVE",
            "scope": "Numerical study at declared settings only; field provenance and missing basis terms remain limitations"}


@dataclass(frozen=True)
class CorrectorControl:
    key: str
    field: str
    step: float

    @property
    def unit(self):
        if self.field in {"percent", "strength_percent", "strength_x_percent", "strength_y_percent"}:
            return "percentage_point"
        if self.field in {"rotation_deg", "azimuth_deg"}:
            return "degree"
        if self.field in {"strength_m3", "strength_m2", "orientation_rad"}:
            return {"strength_m3": "m^-3", "strength_m2": "m^-2", "orientation_rad": "rad"}[self.field]
        raise ValueError("Use an explicit existing excitation or orientation control; no current calibration is inferred")


def _control_snapshot(state):
    """Copy physical objects together to retain aliases, without State.from_dict.

    Immutable field maps / assembly bindings can be shared; private solver and
    transfer caches are discarded. No preset application or initialization.
    """
    result = copy(state)
    names = ("lenses", "corrector_elements", "stigmators", "deflectors", "sample", "electron_gun")
    memo = {id(state): result}
    assembly = getattr(state, "_resolved_assembly", None)
    if assembly is not None:
        memo[id(assembly)] = assembly
    objects = deepcopy({name: getattr(state, name) for name in names if hasattr(state, name)}, memo)
    for name in list(vars(result)):
        value = getattr(result, name)
        if name.startswith("_") and ("cache" in name or "propagation_plan" in name or name == "_field_provider_diagnostics"):
            delattr(result, name)
        elif name.startswith("_") and (id(value) in memo or type(value).__module__.startswith("temsim.optics.")):
            setattr(result, name, deepcopy(value, memo))
    for name, value in objects.items():
        setattr(result, name, value)
    result.lens_field_map_descriptors = deepcopy(getattr(state, "lens_field_map_descriptors", {}))
    result._lens_field_map_bindings = dict(getattr(state, "_lens_field_map_bindings", {}))
    return result


def local_corrector_response(state, system, controls, *, validation_fraction=.5):
    """Central local Jacobian in Cartesian coefficients, with joint holdouts.

    Units follow each existing control, not inferred mm/A. Finite perturbation
    bounds and joint holdout errors are evidence at this working point, not a
    certified trust region. No control values are written back to the state.
    """
    controls = tuple(controls)
    if not 2 <= len(controls) <= 8 or len({(c.key, c.field) for c in controls}) != len(controls):
        raise ValueError("Choose 2–8 distinct corrector controls")
    if not np.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("Joint validation fraction must lie inside (0, 1)")
    frozen = _control_snapshot(state)

    def find(snapshot, control):
        candidates = {id(item): item for name in ("corrector_elements", "lenses", "stigmators")
                      for item in getattr(snapshot, name, ()) if item.key == control.key}
        if len(candidates) != 1:
            raise ValueError(f"Corrector control must resolve uniquely: {control.key}")
        item = next(iter(candidates.values()))
        if not hasattr(item, control.field):
            raise ValueError(f"Missing control {control.key}.{control.field}")
        return item

    base = np.array([float(getattr(find(frozen, c), c.field)) for c in controls])
    steps = np.array([c.step for c in controls], float)
    units = [c.unit for c in controls]
    if not np.isfinite(base).all() or not np.isfinite(steps).all() or np.any(steps <= 0):
        raise ValueError("Control working points and positive steps must be finite")

    def evaluate(offset):
        work = _control_snapshot(frozen)
        for control, value in zip(controls, base + offset):
            item = find(work, control)
            if control.unit == "percentage_point":
                maximum = float(getattr(item, "max_percent", 100))
                minimum = -maximum if control.field in {"strength_x_percent", "strength_y_percent"} else 0
                if not minimum <= value <= maximum:
                    raise ValueError(f"Perturbation exceeds excitation limits for {control.key}")
            if control.field in {"strength_m3", "strength_m2"} and abs(value) > float(getattr(item, "maximum_" + control.field, np.inf)):
                raise ValueError(f"Perturbation exceeds field-strength limits for {control.key}")
            setattr(item, control.field, float(value))
        _, result, evidence = derive_field_aberrations(work, system)
        return np.r_[cartesian_coefficients(result), result.cc_mm], evidence

    origin, evidence = evaluate(np.zeros_like(base))
    matrices = []
    for fraction in (1., .5):
        columns = []
        for i, step in enumerate(steps * fraction):
            delta = np.zeros_like(base)
            delta[i] = step
            columns.append((evaluate(delta)[0] - evaluate(-delta)[0]) / (2*step))
        matrices.append(np.column_stack(columns))
    jacobian = matrices[-1]
    holdouts = []
    for signs in (np.ones(len(base)), (-1.)**np.arange(len(base))):
        offset = validation_fraction * steps * signs
        actual, _ = evaluate(offset)
        predicted = origin + jacobian @ offset
        holdouts.append({"control_offsets": offset.tolist(), "actual_mm": actual.tolist(),
                         "predicted_mm": predicted.tolist(), "error_mm": (actual-predicted).tolist()})
    return {"schema": ABERRATION_SCHEMA, "system": system,
            "working_point": [{**asdict(c), "value": float(v), "unit": u,
                               "tested_interval": [float(v-c.step), float(v+c.step)]} for c, v, u in zip(controls, base, units)],
            "coefficient_names": (*CARTESIAN_NAMES, "Cc"), "coefficient_unit": "mm",
            "base_coefficients_mm": origin.tolist(), "jacobian": jacobian.tolist(),
            "jacobian_step_halving_change": (matrices[1]-matrices[0]).tolist(),
            "joint_holdouts": holdouts, "baseline_fit": evidence,
            "status": "LOCAL_RESPONSE_ONLY; validity not certified outside tested perturbations",
            "autotuning_applied": False}
