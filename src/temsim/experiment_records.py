"""Declared perturbations, resumable scalar receipts and experiment exports."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import csv
from hashlib import sha256
import json
import math
import os
from tempfile import NamedTemporaryFile

import numpy as np

from temsim.design_experiments import (ParameterSweep, SweepPoint, SweepAxis, ToleranceRule,
    parameter_value, replace_parameter, validate_runtime_sweep_path, recipe_from_dict)
from temsim.immutable_json import freeze_json, thaw_json, json_digest


@dataclass(frozen=True)
class Perturbation:
    path: str
    standard_deviation: float
    unit: str = ""

    def __post_init__(self):
        if not math.isfinite(self.standard_deviation) or self.standard_deviation <= 0:
            raise ValueError("Perturbation standard deviation must be finite and positive")


def plan_robustness(recipe, perturbations, *, samples=16, seed=0, maximum_points=64):
    perturbations = tuple(perturbations)
    if not perturbations or len({p.path for p in perturbations}) != len(perturbations):
        raise ValueError("Specify distinct registered perturbation controls")
    if type(samples) is not int or not 2 <= samples <= maximum_points:
        raise ValueError("Robustness sample count exceeds the numerical budget")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Specify a nonnegative 32-bit perturbation seed")
    for p in perturbations:
        validate_runtime_sweep_path(recipe.state_payload, p.path)
    centres = np.array([parameter_value(recipe.state_payload, p.path) for p in perturbations], float)
    draws = centres + np.random.default_rng(seed).normal(size=(samples, len(perturbations))) * np.array([p.standard_deviation for p in perturbations])
    points = []
    for index, vector in enumerate(draws):
        payload = recipe.state_payload
        coordinates = {p.path: float(value) for p, value in zip(perturbations, vector, strict=True)}
        for path, value in coordinates.items():
            payload = replace_parameter(payload, path, value)
        points.append(SweepPoint(index, coordinates, payload, json_digest(payload)))
    axes = tuple(SweepAxis(p.path, tuple(float(v) for v in draws[:, i]), p.unit) for i, p in enumerate(perturbations))
    return ParameterSweep(recipe.digest, axes, tuple(points), kind="normal_perturbations",
        assumptions=dict(distribution="Independent untruncated normal perturbations of explicitly selected controls",
            scope="User-declared simulation assumption; not an OEM specification or measured noise model",
            seed=seed, samples=samples, perturbations=[asdict(p) for p in perturbations]))


def experiment_document(recipe, sweep, result, tolerance_rules=()):
    if result.recipe_digest != recipe.digest or sweep.recipe_digest != recipe.digest:
        raise ValueError("Experiment results and plan must belong to the same recipe")
    document = dict(schema="temsim-experiment-record-v1", recipe=recipe.to_dict(),
        plan=dict(axes=sweep.axes, kind=sweep.kind, assumptions=sweep.assumptions,
                  points=[dict(index=p.index, coordinates=p.coordinates, state_digest=p.state_digest) for p in sweep.points]),
        result=result, tolerance_rules=tuple(tolerance_rules))
    document = thaw_json(freeze_json(document))
    document["digest"] = json_digest(document)
    return document


def _atomic_json(path, document):
    destination = Path(path)
    temporary = None
    try:
        with NamedTemporaryFile(dir=destination.parent, prefix=".experiment-", suffix=".tmp", delete=False, mode="w", encoding="utf-8") as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def save_experiment(path, recipe, sweep, result, tolerance_rules=()):
    _atomic_json(path, experiment_document(recipe, sweep, result, tolerance_rules))


def load_experiment(path):
    path = Path(path)
    if path.stat().st_size > 256 * 1024**2:
        raise ValueError("Experiment record exceeds the 256 MiB input-record budget")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "temsim-experiment-record-v1" or document.get("digest") != json_digest({k: v for k, v in document.items() if k != "digest"}):
        raise ValueError("Experiment record schema or checksum is invalid")
    recipe = recipe_from_dict(document["recipe"])
    plan = document["plan"]
    axes = tuple(SweepAxis(row["path"], tuple(row["values"]), row["unit"]) for row in plan["axes"])
    if len(plan["points"]) > 256:
        raise ValueError("Experiment record has too many points")
    points = []
    for row in plan["points"]:
        payload = recipe.state_payload
        if set(row["coordinates"]) != {axis.path for axis in axes}:
            raise ValueError("Experiment point coordinates do not match its axes")
        if plan["kind"] != "geometry":
            for key, value in row["coordinates"].items():
                validate_runtime_sweep_path(payload, key)
                payload = replace_parameter(payload, key, value)
        if json_digest(payload) != row["state_digest"]:
            raise ValueError("Experiment point input checksum is invalid")
        points.append(SweepPoint(row["index"], row["coordinates"], payload, row["state_digest"]))
    sweep = ParameterSweep(recipe.digest, axes, tuple(points), plan["kind"], plan["assumptions"])
    if sweep.kind == "geometry":
        from temsim.geometry_experiments import plan_geometry_sweep
        settings = sweep.assumptions
        expected = plan_geometry_sweep(recipe, settings["module"], settings["part"], settings["dimension"],
            axes[0].values, optimization=settings.get("optimization"), maximum_points=256)
        if json_digest(expected) != json_digest(sweep):
            raise ValueError("Geometry record does not match its declared plan")
    from temsim.design_sweep_execution import (SweepExecutionResult, SweepPointExecution, SweepMetricDefinition,
                                              PointToleranceAssessment)
    from temsim.design_experiments import SensitivityEstimate, ToleranceResult
    raw = document["result"]
    result = SweepExecutionResult(recipe_digest=raw["recipe_digest"],
        point_results=tuple(SweepPointExecution(**row) for row in raw["point_results"]),
        sensitivities=tuple(SensitivityEstimate(**row) for row in raw["sensitivities"]),
        tolerances=tuple(PointToleranceAssessment(row["point_index"], tuple(ToleranceResult(**v) for v in row["results"])) for row in raw["tolerances"]),
        metric_definitions=tuple(SweepMetricDefinition(**row) for row in raw["metric_definitions"]),
        cancelled=bool(raw["cancelled"]), sweep_axes=axes, analysis_notes=tuple(raw["analysis_notes"]),
        execution_inputs=raw["execution_inputs"], execution_identity=raw["execution_identity"])
    if result.recipe_digest != recipe.digest:
        raise ValueError("Result belongs to a different recipe")
    seen = set()
    for row in result.point_results:
        if row.point_index in seen or not 0 <= row.point_index < len(points):
            raise ValueError("Experiment result has invalid point indices")
        point = points[row.point_index]
        if row.state_digest != point.state_digest or dict(row.coordinates) != dict(point.coordinates):
            raise ValueError("Experiment result coordinates do not match captured inputs")
        seen.add(row.point_index)
    return recipe, sweep, result, tuple(ToleranceRule(**row) for row in document["tolerance_rules"])


def pareto_points(result, objectives):
    """Return all nondominated complete candidates for explicit objectives."""
    if not objectives or any(direction not in {"min", "max"} for direction in objectives.values()):
        raise ValueError("Declare each comparison objective as min or max")
    eligible = [row for row in result.point_results if row.status == "COMPLETE" and all(k in row.metrics for k in objectives)]
    vectors = {row.point_index: np.array([row.metrics[k] * (1 if d == "min" else -1) for k, d in objectives.items()]) for row in eligible}
    return tuple(index for index, values in vectors.items() if not any(
        other != index and np.all(candidate <= values) and np.any(candidate < values)
        for other, candidate in vectors.items()))


def with_local_sensitivities(sweep, rows, definitions):
    """Neighbor differences as explicit objectives, without extra solver calls.

    Other coordinates must match exactly. Failed/missing neighbors cannot be
    skipped to create a smoother sensitivity. Three-point unequal-grid central
    differences or two-point one-sided boundary differences are labelled.
    """
    from temsim.design_sweep_execution import SweepMetricDefinition
    by_index = {row.point_index: row for row in rows}
    added = {row.point_index: {} for row in rows}
    evidence = {row.point_index: {} for row in rows}
    extra = {}
    for axis in sweep.axes:
        other = tuple(a.path for a in sweep.axes if a.path != axis.path)
        groups = {}
        for point in sweep.points:
            groups.setdefault(tuple(point.coordinates[p] for p in other), []).append(point)
        for group in groups.values():
            group.sort(key=lambda point: point.coordinates[axis.path])
            if len(group) < 2:
                continue
            for i, point in enumerate(group):
                neighbors = group[max(0,i-1):min(len(group),i+2)]
                if any(p.index not in by_index or by_index[p.index].status != "COMPLETE" for p in neighbors):
                    continue
                x = np.array([p.coordinates[axis.path] for p in neighbors])
                if np.any(np.diff(x) <= 0):
                    continue
                for definition in definitions:
                    if any(definition.key not in by_index[p.index].metrics for p in neighbors):
                        continue
                    y = np.array([by_index[p.index].metrics[definition.key] for p in neighbors])
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        derivative = float(np.gradient(y, x)[next(j for j,p in enumerate(neighbors) if p.index == point.index)])
                    if not math.isfinite(derivative):
                        continue
                    key = f"sensitivity_abs/{definition.key}/{axis.path}"
                    added[point.index][key] = abs(derivative)
                    evidence[point.index][key] = dict(signed_derivative=derivative,
                        point_indices=[p.index for p in neighbors],
                        method="three-point central difference" if len(neighbors) == 3 else "two-point boundary difference",
                        qualification="Observed finite difference; numerical convergence not established")
                    extra[key] = SweepMetricDefinition(key, f"|Sensitivity: {definition.label} / {axis.path}|",
                        f"{definition.unit}/{axis.unit or 'control unit'}", "Executed neighboring point observations")
    measured = tuple(replace(row, metrics={**row.metrics, **added[row.point_index]},
        evidence={**row.evidence, "local_sensitivities": evidence[row.point_index]})
        if added[row.point_index] else row for row in rows)
    return measured, (*definitions, *extra.values())


def export_experiment_table(path, recipe, sweep, result):
    """Scalar CSV plus provenance manifest; no calculation arrays exported."""
    path = Path(path)
    metric_keys = [definition.key for definition in result.metric_definitions]
    parameters = [axis.path for axis in sweep.axes]
    def safe_cell(value):
        return "'" + value if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r", "\n")) else value
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=".experiment-", suffix=".tmp", delete=False,
                mode="w", newline="", encoding="utf-8-sig") as stream:
            temporary = Path(stream.name)
            writer = csv.writer(stream)
            writer.writerow([safe_cell(v) for v in ["point", "status", "failure_reason", "numerical_status", *parameters, *metric_keys]])
            for row in result.point_results:
                writer.writerow([safe_cell(v) for v in [row.point_index + 1, row.status, row.failure_reason, row.numerical_status,
                    *(row.coordinates.get(k, "") for k in parameters), *(row.metrics.get(k, "") for k in metric_keys)]])
            stream.flush()
            os.fsync(stream.fileno())
        checksum = sha256(temporary.read_bytes()).hexdigest()
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    manifest = dict(schema="temsim-experiment-export-v1", execution_identity=result.execution_identity,
        execution_inputs=thaw_json(result.execution_inputs), recipe_identity=recipe.digest,
        units={row.key: row.unit for row in result.metric_definitions},
        parameter_units={axis.path: axis.unit for axis in sweep.axes},
        assumptions=thaw_json(sweep.assumptions), cancelled=result.cancelled,
        candidate_evidence=[dict(point=row.point_index, status=row.status, evidence=thaw_json(row.evidence)) for row in result.point_results],
        status="Executed scalar observations; numerical and physical qualification are separate",
        array_policy="No calculation arrays in this export",
        output_file=path.name, output_sha256=checksum)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    _atomic_json(manifest_path, manifest)
    return manifest_path
