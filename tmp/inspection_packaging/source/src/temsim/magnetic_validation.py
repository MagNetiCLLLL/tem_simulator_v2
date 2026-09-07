"""Detached, dependency-keyed magnetic mesh/domain studies.

SI internally. The optical checks are local paraxial vacuum tests in a Larmor
frame, not specimen rays, image results or Cs/Cc validation. Every comparison
uses fixed physical planes and the same collimated test radius. No presets.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import json
import math
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from temsim.physics.lens_field_provider import _canonical, _fingerprint, _part_geometry_data, _runtime_excitation

VALIDATION_VERSION = "magnetic-convergence-v1"


class ValidationCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class ValidationOptions:
    relative_tolerance: float = .01
    field_absolute_t: float = 1e-6
    length_absolute_mm: float = 1e-5
    rotation_absolute_deg: float = 1e-4
    beam_absolute_um: float = 1e-4
    mesh_factor: float = 1.5
    boundary_factor: float = 1.5
    test_radius_um: float = 1.0

    def validate(self):
        if not all(math.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError("Validation settings must be finite and positive")
        if not 1e-6 <= self.relative_tolerance <= .2:
            raise ValueError("Relative tolerance must lie between 0.0001% and 20%")
        if not (1.1 <= self.mesh_factor <= 2 and 1.1 <= self.boundary_factor <= 2):
            raise ValueError("Refinement factors must lie between 1.1 and 2")
        return self


@dataclass(frozen=True)
class FieldProblem:
    key: str
    geometry_json: str
    settings_json: str
    scale: float


@dataclass(frozen=True)
class Comparison:
    stage: str
    metric: str
    unit: str
    value: float | None
    absolute_change: float | None
    relative_change: float | None
    allowed_change: float
    passed: bool | None


@dataclass(frozen=True)
class CaseResult:
    label: str
    metrics: tuple[tuple[str, float | None], ...]
    axis_field_t: np.ndarray
    meshes: tuple[tuple[int, int], ...]
    residuals: tuple[float, ...]
    problems: tuple[FieldProblem, ...]


@dataclass(frozen=True)
class MagneticScene:
    r_m: np.ndarray
    z_m: np.ndarray
    magnitude_t: np.ndarray
    flux_per_radian_wb: np.ndarray
    regions: np.ndarray  # 0 vacuum, 1 magnetic body, 2 coil
    material_bounds_mm: tuple[float, float, float]


@dataclass(frozen=True)
class ValidationReport:
    signature: str
    selected_key: str
    beam_voltage_kv: float
    options: ValidationOptions
    z_m: np.ndarray
    cases: tuple[CaseResult, ...]
    comparisons: tuple[Comparison, ...]
    scene: MagneticScene

    @property
    def passed(self):
        return bool(self.comparisons) and all(c.passed is True for c in self.comparisons)


class ValidationCache:
    """Separate from image/ray results; retain at most four completed studies."""

    def __init__(self, capacity=4):
        self.capacity = max(1, int(capacity))
        self._reports = OrderedDict()

    def get(self, signature):
        result = self._reports.get(signature)
        if result is not None:
            self._reports.move_to_end(signature)
        return result

    def put(self, report):
        self._reports[report.signature] = report
        self._reports.move_to_end(report.signature)
        while len(self._reports) > self.capacity:
            self._reports.popitem(last=False)


def input_signature(state, key, options):
    from temsim.physics.axisymmetric_magnetostatics import SOLVER_VERSION
    from temsim.physics.lens_field_provider import _GEOMETRY_ATTRIBUTES, _FIELD_STRUCTURE_PROFILES
    from temsim.magnetic_circuits import circuit_channels, belongs_to_circuit, MAGNETIC_BODIES
    from temsim.simulation_modes import mode_key
    options.validate()
    by_key = {p.key: p for p in state._resolved_assembly.parts}
    descriptors = state.lens_field_map_descriptors
    nonlinear = descriptors.get(key, {}).get("solver") == "axisymmetric_nonlinear_fem"
    keys = ({k for k, row in descriptors.items() if row.get("solver") == "axisymmetric_nonlinear_fem"}
            if nonlinear else set(circuit_channels(by_key, key)))
    owned = [p for p in by_key.values() if p.data.get("mechanical_profile") in MAGNETIC_BODIES | {"magnetic_excitation_coil"}
             and any(belongs_to_circuit(p, channel, by_key) for channel in keys)]
    if owned:
        lo, hi = min(p.start_z_mm for p in owned), max(p.end_z_mm for p in owned)
        centre = .5*(lo+hi)
        padding = max(float(descriptors.get(k, {}).get("padding_factor", 2)) for k in keys)
        half = .5*(hi-lo)*padding*options.boundary_factor**2
        affected = {p.key for p in by_key.values() if p.start_z_mm <= centre+half and p.end_z_mm >= centre-half}
    else:
        affected = set(by_key)
    # Include parents: their TOML intervals define split Objective bodies.
    for part_key in tuple(affected):
        part = by_key[part_key]
        seen = set()
        while part.parent_key in by_key and part.parent_key not in seen:
            seen.add(part.parent_key)
            affected.add(part.parent_key)
            part = by_key[part.parent_key]
    excitation_keys = set(keys)
    if nonlinear:
        excitation_keys.update(p.data.get("field_source_key", p.parent_key) for p in by_key.values()
                               if p.key in affected and p.data.get("mechanical_profile") == "magnetic_excitation_coil")
    parts = tuple((p.key, p.parent_key, p.start_z_mm, p.end_z_mm, _part_geometry_data(p))
                  for p in state._resolved_assembly.parts
                  if p.key in affected and p.data.get("mechanical_profile") in _FIELD_STRUCTURE_PROFILES)
    lenses = tuple((l.key, _runtime_excitation(l),
                    tuple((name, getattr(l, name)) for name in _GEOMETRY_ATTRIBUTES if hasattr(l, name)))
                   for l in state.lenses if l.key in excitation_keys)
    return _fingerprint((VALIDATION_VERSION, SOLVER_VERSION, key, asdict(options.validate()), mode_key(state),
                         float(state.beam_voltage_kv), parts, lenses, {k: descriptors.get(k) for k in sorted(keys)}))


def snapshot_state(state):
    """Capture on the GUI thread; resolved production assemblies are immutable."""
    if hasattr(state, "to_dict"):
        result = type(state).from_dict(state.to_dict())
        result._resolved_assembly = state._resolved_assembly
        return result
    return deepcopy(state)  # Small synthetic reference fixtures only.


def prepare_problems(state, selected_key, *, mesh_multiplier=1., padding_multiplier=1.):
    """Build exact solver inputs without solving or touching the live state."""
    from temsim.component_keys import CONDENSER_LENS_KEYS
    from temsim.magnetic_circuits import circuit_channels
    from temsim.physics.lens_field_provider import lens_geometry_binding, _validate_shared_linear_recipes
    from temsim.physics.nonlinear_circuits import SOLVER, resolve_nonlinear_provider
    from temsim.simulation_modes import uses_field_maps
    if not uses_field_maps(state):
        raise ValueError("Select Custom, Linear Geometry or Nonlinear Material mode first; saved recipes are inactive")
    work = snapshot_state(state)
    lenses = {l.key: l for l in work.lenses}
    if selected_key not in lenses or not lenses[selected_key].enabled:
        raise ValueError("Select an enabled round lens")
    descriptors = work.lens_field_map_descriptors
    solver = descriptors.get(selected_key, {}).get("solver")
    nonlinear = solver == SOLVER
    if solver not in {SOLVER, "axisymmetric_linear_fem"}:
        raise ValueError("Configure a generated geometry field before running convergence checks")
    keys = (tuple(sorted(k for k, row in descriptors.items() if row.get("solver") == SOLVER and k in lenses))
            if nonlinear else circuit_channels({p.key: p for p in work._resolved_assembly.parts}, selected_key))
    for key in keys:
        row = descriptors.get(key, {})
        if row.get("solver") != solver:
            raise ValueError(f"Configure every circuit channel with the same field solver: {key}")
        for name, default, limit in (("radial_nodes", 40, 256), ("axial_nodes", 80, 512)):
            nodes = int(math.ceil((int(row.get(name, default))-1)*mesh_multiplier))+1
            if nodes > limit:
                raise ValueError(f"Refinement exceeds {name} limit {limit}; lower the base mesh or refinement factor")
            row[name] = nodes
        row["padding_factor"] = float(row.get("padding_factor", 2))*padding_multiplier
        if row["padding_factor"] > 10:
            raise ValueError("Boundary study exceeds padding factor 10; lower the base padding or expansion factor")
    native = work.condenser_system[selected_key] if selected_key in CONDENSER_LENS_KEYS else lenses[selected_key]
    if nonlinear:
        geometry, settings, channels = resolve_nonlinear_provider(
            work, selected_key, native, lens_geometry_binding(work, selected_key, native), prepare_only=True)
        return (FieldProblem("Joint B-H: " + ", ".join(channels), _json(geometry), _json(settings), 1.),)
    result = []
    for key in keys:
        source = work.condenser_system[key] if key in CONDENSER_LENS_KEYS else lenses[key]
        enabled, percent, polarity = _runtime_excitation(source)
        binding = lens_geometry_binding(work, key, source)
        _validate_shared_linear_recipes(work, key, binding)
        result.append(FieldProblem(key, binding.canonical_geometry_json, _json(descriptors[key]),
                                   percent*.01*polarity if enabled else 0.))
    return tuple(result)


def _json(value):
    return json.dumps(_canonical(value), sort_keys=True, allow_nan=False)


@lru_cache(maxsize=12)
def _validation_solve(geometry_json, settings_json):
    # Refined studies must not evict production maps from their own LRU.
    from temsim.physics.axisymmetric_magnetostatics import _solve_bound_geometry
    return _solve_bound_geometry.__wrapped__(geometry_json, settings_json)


def _solve(problem, baseline=False):
    if baseline:
        from temsim.physics.axisymmetric_magnetostatics import _solve_bound_geometry
        return _solve_bound_geometry(problem.geometry_json, problem.settings_json)
    return _validation_solve(problem.geometry_json, problem.settings_json)


def _extend_axis(axis, low, high):
    spacing = float(np.max(np.diff(axis)))
    left_n = max(0, math.ceil((axis[0]-low)/spacing))
    right_n = max(0, math.ceil((high-axis[-1])/spacing))
    return np.r_[np.linspace(low, axis[0], left_n+1)[:-1], axis,
                 np.linspace(axis[-1], high, right_n+1)[1:]]


def extend_problem_grid(problem, previous_axes):
    settings = json.loads(problem.settings_json)
    from temsim.magnetic_circuits import MAGNETIC_BODIES
    parts = json.loads(problem.geometry_json)["lens_assembly"]["parts"]
    used = [p for p in parts if p["data"].get("mechanical_profile") in MAGNETIC_BODIES | {"magnetic_excitation_coil"}]
    radius = max(p["data"]["mechanical_outer_diameter_mm"] for p in used)*.5e-3
    low, high = min(p["start_z_mm"] for p in used)*1e-3, max(p["end_z_mm"] for p in used)*1e-3
    centre, half, padding = .5*(low+high), .5*(high-low), settings["padding_factor"]
    r = _extend_axis(previous_axes[0], 0, radius*padding)
    z = _extend_axis(previous_axes[1], centre-half*padding, centre+half*padding)
    settings["validation_grid_axes_m"] = [r.tolist(), z.tolist()]
    return replace(problem, settings_json=_json(settings))


def _immutable(value):
    array = np.asarray(value).copy()
    array.setflags(write=False)
    return array


def paraxial_metrics(z_m, bz_t, voltage_kv, radius_um, *, material_parts=()):
    """Local collimated bundle in the rotating frame; finite-length, not thin-lens."""
    from temsim.physics.core import electron
    voltage = float(voltage_kv)
    if not math.isfinite(voltage) or voltage <= 0:
        raise ValueError("Beam voltage must be positive and finite")
    charge, momentum, _ = electron(SimpleNamespace(beam_voltage_kv=voltage))
    g = -charge*bz_t/(2*momentum)
    if not (np.all(np.isfinite(z_m)) and np.all(np.isfinite(bz_t)) and np.all(np.diff(z_m) > 0)):
        raise ValueError("Paraxial field samples must be finite on an increasing Z axis")
    def trace(subdivisions):
        # RK4 integrates each piecewise-linear field interval explicitly, so a
        # narrow sampled peak cannot be skipped by a global adaptive step.
        y = np.array((1., 0., 0., 1.))
        largest_slope = 0.
        radii, positions = [], []
        def rhs(values, strength):
            k = strength*strength
            return np.array((values[1], -k*values[0], values[3], -k*values[2]))
        for i, length in enumerate(np.diff(z_m)):
            h = float(length)/subdivisions
            for j in range(subdivisions):
                ga = g[i]+(g[i+1]-g[i])*j/subdivisions
                gc = g[i]+(g[i+1]-g[i])*(j+1)/subdivisions
                gm = .5*(ga+gc)
                k1 = rhs(y, ga)
                k2 = rhs(y+.5*h*k1, gm)
                k3 = rhs(y+.5*h*k2, gm)
                k4 = rhs(y+h*k3, gc)
                y = y+h*(k1+2*k2+2*k3+k4)/6.
                if not np.all(np.isfinite(y)):
                    raise ValueError("Paraxial validation trace became nonfinite")
                largest_slope = max(largest_slope, abs(y[1])*radius_um*1e-6)
                radii.append(abs(y[0])*radius_um*1e-6)
                positions.append(z_m[i]+(j+1)*h)
        if material_parts:
            from temsim.physics.axisymmetric_magnetostatics import _part_mask
            if any(np.any(_part_mask(p, np.asarray(radii), np.asarray(positions))) for p in material_parts):
                raise ValueError("Paraxial test bundle intersects magnetic material or a coil; reduce its radius")
        return y, largest_slope
    result, _ = trace(1)
    for subdivisions in (2, 4, 8, 16):
        refined, slope = trace(subdivisions)
        if np.allclose(result, refined, rtol=1e-8, atol=1e-10):
            break
        result = refined
    else:
        raise ValueError("Paraxial integration step-refinement check failed")
    if slope > .1:
        raise ValueError("Test beam exceeds the 100 mrad paraxial diagnostic limit; reduce its radius")
    a, c, b, d = refined
    if abs(a*d-b*c-1) > 1e-5:
        raise ValueError("Paraxial transfer failed the unit-determinant check")
    power = -c
    focal = 1e3/power if abs(power) > 1e-9 else None
    return {"Peak axial field": float(np.max(np.abs(bz_t))),
            "Effective focal length": focal,
            "Larmor rotation": float(np.degrees(np.trapezoid(g, z_m))),
            "Exit test-beam radius": abs(float(a))*radius_um}


def compare_cases(before, after, options, stage):
    values, previous = dict(after.metrics), dict(before.metrics)
    definitions = (("Axial field curve", "T", options.field_absolute_t),
                   ("Effective focal length", "mm", options.length_absolute_mm),
                   ("Larmor rotation", "deg", options.rotation_absolute_deg),
                   ("Exit test-beam radius", "um", options.beam_absolute_um))
    result = []
    for name, unit, absolute in definitions:
        if name == "Axial field curve":
            old, new = before.axis_field_t, after.axis_field_t
        else:
            old, new = previous[name], values[name]
        if old is None or new is None:
            result.append(Comparison(stage, name, unit, new, None, None, absolute, None))
            continue
        delta = float(np.max(np.abs(np.asarray(new)-old)))
        scale = float(np.max(np.abs(new)))
        allowed = absolute + options.relative_tolerance*scale
        result.append(Comparison(stage, name, unit, scale if name == "Axial field curve" else float(new),
                                 delta, delta/scale if scale > 0 else None, allowed, delta <= allowed))
    return tuple(result)


def make_scene(problems, details):
    from temsim.physics.axisymmetric_magnetostatics import _part_mask
    from temsim.magnetic_circuits import MAGNETIC_BODIES
    rmax = min(item[0].axes_m[0][-1] for item in details)
    zlo = max(item[0].axes_m[1][0] for item in details)
    zhi = min(item[0].axes_m[1][-1] for item in details)
    r, z = np.linspace(0, rmax, 180), np.linspace(zlo, zhi, 320)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    positions = np.stack((rr, np.zeros_like(rr), zz), axis=-1)
    field, flux, regions = np.zeros(rr.shape+(3,)), np.zeros(rr.shape), np.zeros(rr.shape, np.uint8)
    parts = {}
    for problem, (mapped, solution) in zip(problems, details):
        field += problem.scale*mapped.field_at_global_positions_t(positions)
        potential = RegularGridInterpolator(mapped.axes_m, solution.a_phi_tm)(np.stack((rr, zz), axis=-1))
        flux += problem.scale*rr*potential
        geometry = json.loads(problem.geometry_json)["lens_assembly"]
        parts.update({p["key"]: p for p in geometry["parts"]+geometry.get("magnetostatic_neighbours", [])})
    material = [p for p in parts.values() if p["data"].get("mechanical_profile") in MAGNETIC_BODIES | {"magnetic_excitation_coil"}]
    for part in material:
        mask = _part_mask(part, rr, zz)
        regions[mask] = 2 if part["data"]["mechanical_profile"] == "magnetic_excitation_coil" else 1
    bounds = (min(p["start_z_mm"] for p in material), max(p["end_z_mm"] for p in material),
              max(p["data"]["mechanical_outer_diameter_mm"] for p in material)/2)
    return MagneticScene(*map(_immutable, (r, z, np.linalg.norm(field, axis=-1), flux, regions)), bounds)


def run_validation(snapshot, key, options=ValidationOptions(), *, progress=None, cancelled=None):
    options.validate()
    signature = input_signature(snapshot, key, options)
    def check():
        if cancelled and cancelled():
            raise ValidationCancelled("Validation cancelled; existing results retained")
    cases, comparisons, finest_axes = [], [], None
    z = None
    plan = []
    for index in range(5):
        check()
        plan.append(prepare_problems(snapshot, key, mesh_multiplier=options.mesh_factor**min(index, 2),
                                     padding_multiplier=options.boundary_factor**max(0, index-2)))
    for index in range(5):
        check()
        label = f"Mesh {index+1}/3" if index < 3 else f"Boundary {index-2}/2"
        if progress:
            progress(index, 5, label + " — solving")
        problems = plan[index]
        if index >= 3:
            problems = tuple(extend_problem_grid(p, axes) for p, axes in zip(problems, finest_axes))
        details = []
        for problem in problems:
            check()
            details.append(_solve(problem, baseline=index == 0))
        check()
        if index >= 2:
            finest_axes = tuple(detail[0].axes_m for detail in details)
        if z is None:
            lo = max(detail[0].axes_m[1][0] for detail in details)
            hi = min(detail[0].axes_m[1][-1] for detail in details)
            knots = np.concatenate([d[0].axes_m[1] for d in details])
            knots = np.unique(knots[(knots >= lo) & (knots <= hi)])
            from temsim.physics.axisymmetric_magnetostatics import _merged_axis
            z = _merged_axis(np.r_[np.linspace(lo, hi, 1025), knots, .5*(knots[1:]+knots[:-1])])
        positions = np.column_stack((np.zeros_like(z), np.zeros_like(z), z))
        bz = sum((p.scale*d[0].field_at_global_positions_t(positions)[:, 2] for p, d in zip(problems, details)), start=np.zeros_like(z))
        from temsim.magnetic_circuits import MAGNETIC_BODIES
        material_parts = {}
        for problem in problems:
            assembly = json.loads(problem.geometry_json)["lens_assembly"]
            for part in assembly["parts"]+assembly.get("magnetostatic_neighbours", []):
                if part["data"].get("mechanical_profile") in MAGNETIC_BODIES | {"magnetic_excitation_coil"}:
                    material_parts[part["key"]] = part
        metrics = paraxial_metrics(z, bz, snapshot.beam_voltage_kv, options.test_radius_um,
                                  material_parts=tuple(material_parts.values()))
        case = CaseResult(label, tuple(metrics.items()), _immutable(bz),
                          tuple(tuple(len(a) for a in d[0].axes_m) for d in details),
                          tuple(d[1].relative_residual for d in details), problems)
        if cases:
            comparisons.extend(compare_cases(cases[-1], case, options, label))
        cases.append(case)
        if progress:
            progress(index+1, 5, label + " — complete")
    check()
    scene = make_scene(problems, details)
    check()
    return ValidationReport(signature, key, float(snapshot.beam_voltage_kv), options, _immutable(z), tuple(cases), tuple(comparisons), scene)


def report_payload(report):
    """Reproducible export/reference interface; external validation stays absent."""
    from temsim.physics.axisymmetric_magnetostatics import SOLVER_VERSION
    return dict(schema=VALIDATION_VERSION, field_solver_version=SOLVER_VERSION,
                signature=report.signature, selected_key=report.selected_key, beam_voltage_kv=report.beam_voltage_kv,
                options=asdict(report.options), sampled_checks_passed=report.passed,
                external_validation="Not checked", cs_cc_validation="Not checked",
                scope="Static axisymmetric fields; local paraxial vacuum test, not specimen/detector imaging",
                z_m=report.z_m.tolist(), comparisons=[asdict(c) for c in report.comparisons],
                cases=[dict(label=c.label, metrics=dict(c.metrics), meshes=c.meshes, residuals=c.residuals,
                            bz_t=c.axis_field_t.tolist(),
                            problems=[dict(key=p.key, geometry=json.loads(p.geometry_json),
                                           settings=json.loads(p.settings_json), scale=p.scale) for p in c.problems])
                       for c in report.cases])
