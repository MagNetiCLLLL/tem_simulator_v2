"""Bounded, detached optical banks and independent physical signal readout.

Lens and pre-specimen aperture axes are solved at explicit grid nodes. Post-
specimen stops are replayed on retained, unmasked trajectory coordinates; wave
readout propagates complex configurations again, never subtracts intensities.
No bank result is published into the main-window high-accuracy result cache.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
import math
from types import MappingProxyType

import numpy as np

from temsim.calculation_cache import calculation_signatures, external_model_signature
from temsim.design_sweep_execution import SweepCalculationCache
from temsim.physics.recording_stop import determine_tem_stop_z
from temsim.simulation_pipeline import calculate


MAX_OPTICAL_POINTS = 256


class InteractiveCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class RangeControl:
    group: str
    key: str
    field: str
    label: str
    unit: str
    current: float
    stage: str  # optical nodes or continuous readout

    @property
    def identity(self):
        return f"{self.group}:{self.key}:{self.field}"


@dataclass(frozen=True)
class CalculationRange:
    control: RangeControl
    minimum: float
    maximum: float
    points: int = 5

    def __post_init__(self):
        if not all(math.isfinite(float(v)) for v in (self.minimum, self.maximum)):
            raise ValueError("Range endpoints must be finite")
        if self.minimum >= self.maximum:
            raise ValueError("Enter an explicit minimum smaller than the maximum")
        if isinstance(self.points, bool) or int(self.points) != self.points or not 2 <= self.points <= 129:
            raise ValueError("Use 2 to 129 optical samples per axis")

    @property
    def values(self):
        return tuple(float(v) for v in np.linspace(self.minimum, self.maximum, self.points))

    def validate_value(self, value):
        value = float(value)
        if not math.isfinite(value) or not self.minimum <= value <= self.maximum:
            raise ValueError(f"{self.control.label}: outside the declared range; rebuild the bank")
        return value


@dataclass(frozen=True)
class InteractivePlan:
    ranges: tuple[CalculationRange, ...]
    cache_budget_bytes: int
    precompute: bool = True

    def __post_init__(self):
        if not self.ranges:
            raise ValueError("Add at least one calculation range; no implicit range is used")
        ids = [r.control.identity for r in self.ranges]
        if len(set(ids)) != len(ids):
            raise ValueError("A control can have only one range")
        if self.cache_budget_bytes <= 0:
            raise ValueError("Enter a positive cache budget")
        if self.point_count > MAX_OPTICAL_POINTS:
            raise ValueError(f"The range has {self.point_count} optical combinations; limit is {MAX_OPTICAL_POINTS}")

    @property
    def optical_ranges(self):
        return tuple(r for r in self.ranges if r.control.stage == "optical")

    @property
    def point_count(self):
        return math.prod(r.points for r in self.optical_ranges) if self.precompute else 1


def available_controls(state):
    controls = []
    def add(group, obj, field, label, unit, stage):
        controls.append(RangeControl(group, obj.key, field, f"{obj.name} / {label}",
                                     unit, float(getattr(obj, field)), stage))
    for lens in state.lenses:
        if bool(getattr(lens, "installed", True)) and bool(lens.enabled):
            add("lens", lens, "percent", "Excitation", "%", "optical")
    seen = set()
    apertures = list(state.apertures)
    for name in ("dpa_aperture", "c1_aperture"):
        aperture = getattr(state.electron_gun, name, None)
        if aperture is not None:
            apertures.append(aperture)
    for aperture in apertures:
        if (aperture.key in seen or not bool(getattr(aperture, "installed", True))
                or not bool(getattr(aperture, "enabled", True))):
            continue
        seen.add(aperture.key)
        stage = "readout" if aperture.z_mm > state.sample.z_mm else "optical"
        if hasattr(aperture, "diameter_mm"):
            add("aperture", aperture, "diameter_mm", "Diameter", "mm", stage)
        for field, label in (("offset_x_mm", "Centre X"), ("offset_y_mm", "Centre Y")):
            if hasattr(aperture, field):
                add("aperture", aperture, field, label, "mm", stage)
    for detector in state.recording_planes:
        if not bool(getattr(detector, "inserted", False)):
            continue
        for field, label in (("z_mm", "Z"), ("outer_width_mm", "Active width"),
                             ("centre_offset_x_mm", "Centre X"), ("centre_offset_y_mm", "Centre Y")):
            if hasattr(detector, field):
                add("detector", detector, field, label, "mm", "readout")
        if str(detector.geometry) == "annulus":
            add("detector", detector, "inner_diameter_mm", "Inner diameter", "mm", "readout")
    return tuple(controls)


def _target(state, control):
    if control.group == "lens":
        candidates = state.lenses
    elif control.group == "detector":
        candidates = state.recording_planes
    elif control.group == "aperture":
        candidates = list(state.apertures) + [getattr(state.electron_gun, name, None)
                                            for name in ("dpa_aperture", "c1_aperture")]
    else:
        raise ValueError("Unknown interactive control group")
    for item in candidates:
        if item is not None and item.key == control.key:
            return item
    raise ValueError(f"Control no longer installed: {control.label}")


def _validated_assignment(state, control, value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Control value must be finite")
    if control.field in {"diameter_mm", "inner_diameter_mm"} and value < 0:
        raise ValueError("Aperture and detector diameters cannot be negative")
    if control.field == "outer_width_mm" and value <= 0:
        raise ValueError("Detector width must be positive")
    obj = _target(state, control)
    if (control.group == "aperture" and control.field == "diameter_mm"
            and hasattr(obj, "maximum_radius_mm")
            and value > 2.0 * float(obj.maximum_radius_mm)):
        raise ValueError(f"{obj.name} opening exceeds its clear bore")
    if control.group == "lens" and not 0 <= value <= float(getattr(obj, "max_percent", 100.)):
        raise ValueError("Lens excitation is outside this component's supported range")
    if control.group == "detector" and control.field == "z_mm":
        if value <= state.sample.z_mm:
            raise ValueError("Recording planes must remain downstream of the sample")
    return obj, value


def _assign(state, control, value):
    obj, value = _validated_assignment(state, control, value)
    if control.group == "detector" and control.field == "z_mm":
        obj.set_optical_reference_z_mm(state.selected_area_aperture.z_mm, value)
    else:
        setattr(obj, control.field, value)


def apply_live_tuning_values(state, axes_and_values):
    """Validate scalar edits without constructing a full instrument snapshot.

    Only installed live lens/aperture controls are allowed. Stage, bounds and
    component setter validation finish before writing any live scalar. Worker
    submission still creates its own complete, detached physical snapshot.
    """
    from copy import copy
    allowed = {control.identity: control for control in available_controls(state)}
    pending = []
    seen = set()
    for axis, value in axes_and_values:
        control = axis.control
        if control.group not in {"lens", "aperture"}:
            raise ValueError("Use Advanced bank for detached detector geometry readout")
        active = allowed.get(control.identity)
        if active is None or active.stage != control.stage:
            raise ValueError("A tuning component changed; capture settings again")
        if control.identity in seen:
            raise ValueError("A live control can be changed only once per update")
        seen.add(control.identity)
        target, value = _validated_assignment(state, control, axis.validate_value(value))
        # These supported setters change scalar excitation/radius/offset only.
        # Validate against a shallow component copy, not the whole microscope.
        draft = copy(target)
        setattr(draft, control.field, value)
        pending.append((target, control.field, value, getattr(target, control.field)))
    applied = []
    try:
        for target, field, value, old in pending:
            applied.append((target, field, old))
            setattr(target, field, value)
    except Exception:
        for target, field, old in reversed(applied):
            setattr(target, field, old)
        raise


def detached_state(state):
    from temsim.gui.calculation_controller import CalculationController
    return CalculationController._calculation_snapshot(
        state, "High accuracy", state.electron_gun.ray_count, state.step_mm)


def validate_plan(state, plan):
    allowed = {c.identity: c for c in available_controls(state)}
    for axis in plan.ranges:
        c = axis.control
        if c.identity not in allowed or c.stage != allowed[c.identity].stage:
            raise ValueError("The range contains an unavailable or misclassified control")
        for endpoint in (axis.minimum, axis.maximum):
            _assign(detached_state(state), c, endpoint)
        if c.group == "detector" and c.field == "z_mm" and axis.maximum > determine_tem_stop_z(state):
            raise ValueError(f"Detector Z must not exceed the retained column limit ({determine_tem_stop_z(state):.6g} mm)")
    # Every combination must have a physical annulus, not just its endpoints in isolation.
    for detector in state.recording_planes:
        if str(detector.geometry) != "annulus":
            continue
        outer_min, inner_max = detector.outer_width_mm, detector.inner_diameter_mm
        for axis in plan.ranges:
            if axis.control.group == "detector" and axis.control.key == detector.key:
                if axis.control.field == "outer_width_mm":
                    outer_min = axis.minimum
                elif axis.control.field == "inner_diameter_mm":
                    inner_max = axis.maximum
        if inner_max >= outer_min:
            raise ValueError(f"{detector.name}: every inner diameter must be below every outer diameter")


@dataclass(frozen=True)
class ReplayBranch:
    branch: object
    weights: np.ndarray  # absolute emitted-source fractions; never renormalised after clipping
    alive: np.ndarray
    blocked_z: np.ndarray
    blocked_key: tuple[str, ...]


def prepare_replay(result):
    from temsim.physics.column_wall import clip_column_wall
    from temsim.detector.stem_signal import _branch_probabilities, _normalised_ray_weights
    s, sim = result.state_snapshot, result.simulation
    specimen_exit = result.specimen_exit
    if specimen_exit is not None:
        branches = specimen_exit.branches
        probabilities = tuple(b.weight for b in branches)
    else:
        branches, probabilities = _branch_probabilities(sim)
    replay = []
    for b, probability in zip(branches, probabilities):
        n = b.x.shape[1]
        if specimen_exit is not None:
            alive, blocked, keys = np.ones(n, bool), np.full(n, np.nan), [""] * n
        else:
            # Only pre-specimen losses are inherited. All downstream detector
            # and aperture masks will be evaluated independently at readout.
            alive = np.asarray(sim.incident.alive).copy()
            blocked = np.asarray(sim.incident.blocked_z).copy()
            keys = list(sim.incident.blocked_key)
        alive, blocked, keys = clip_column_wall(s, b.z, b.x, b.y, alive, blocked, keys)
        weights = float(probability) * _normalised_ray_weights(b)
        if weights.shape != (n,) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("Replay requires finite non-negative source weights for every ray")
        for a in (weights, alive, blocked):
            a.setflags(write=False)
        replay.append(ReplayBranch(b, weights, alive, blocked, tuple(keys)))
    return tuple(replay)


@dataclass(frozen=True)
class BankPoint:
    coordinates: tuple[float, ...]
    result: object
    replay: tuple[ReplayBranch, ...]


@dataclass(frozen=True)
class InteractiveBank:
    plan: InteractivePlan
    source_state: object
    external_signature: str
    points: tuple[BankPoint, ...]
    retained_bytes: int
    external_inputs: tuple = ()


def build_bank(state, plan, *, seeds=(), retained_roots=(), progress=None, cancelled=lambda: False, calculator=calculate):
    """Transactionally build a pinned bank; cancellation never returns a partial bank."""
    from temsim.gui.calculation_controller import (
        estimate_result_cache_bytes, estimate_calculation_memory_bytes,
        HIGH_ACCURACY_MEMORY_BUDGET_BYTES,
    )
    if not plan.precompute:
        raise ValueError("A live tuning plan does not build a high-accuracy bank")
    validate_plan(state, plan)
    from temsim.calculation_manifest import capture_calculation_manifest, assert_external_input_identities_unchanged
    identities = (capture_calculation_manifest(state, ray_count=state.electron_gun.ray_count,
                                               step_mm=state.step_mm).external_inputs
                  if hasattr(state, "to_dict") else ())
    if bool(getattr(state.sample, "stem_fourdstem_enabled", False)):
        raise ValueError("Disable file-backed 4D-STEM capture before building a bank; range points must not overwrite one output file")
    external = external_model_signature(state)
    cache = SweepCalculationCache(maximum_results=1)
    for seed in seeds:
        cache.put(seed)
    completed = []
    for index, coordinates in enumerate(product(*(r.values for r in plan.optical_ranges))):
        if cancelled():
            raise InteractiveCancelled("Bank build cancelled; previous bank retained")
        assert_external_input_identities_unchanged(identities)
        if external_model_signature(state) != external:
            raise ValueError("External model changed during bank construction; capture again")
        point_state = detached_state(state)
        for axis, value in zip(plan.optical_ranges, coordinates):
            _assign(point_state, axis.control, value)
        existing, exact = cache.select(calculation_signatures(point_state))
        objective_pupil_range = any(r.control.key == getattr(getattr(point_state, "objective_aperture", None), "key", None)
                                    and r.control.group == "aperture" for r in plan.ranges)
        cached_wave = getattr(existing, "wave_imaging", None)
        if (objective_pupil_range and cached_wave is not None and not getattr(
                cached_wave.projector_checkpoint, "unapertured_wave_configurations", ())):
            existing = replace(existing, wave_imaging=None)
            exact = False
        from temsim.specimen.source import specimen_structure_available
        memory_capture = bool(point_state.ac_deflector.enabled and point_state.ac_deflector.scan_enabled
                              and point_state.sample.stem_wave_enabled
                              and str(point_state.illumination_mode).upper() == "STEM"
                              and specimen_structure_available(point_state.sample))
        if memory_capture and getattr(getattr(existing, "stem_scan", None), "fourdstem_artifact", None) is None:
            exact = False  # A previous image alone cannot serve changed physical stops.
        retained = estimate_result_cache_bytes(completed, existing, retained_roots, seeds)
        estimate = estimate_calculation_memory_bytes(point_state, "High accuracy", point_state.electron_gun.ray_count, point_state.step_mm)
        if estimate + retained > HIGH_ACCURACY_MEMORY_BUDGET_BYTES:
            raise ValueError("Optical working memory plus retained results exceed the 24 GiB application budget")
        def report(done, total, stage):
            if cancelled():
                raise InteractiveCancelled("Bank build cancelled; previous bank retained")
            if progress:
                progress(index, plan.point_count, done, total, stage)
        if exact:
            result = existing
        else:
            extra = {}
            if memory_capture:
                from temsim.physics.diffraction_memory import MemoryDiffractionSink
                remaining = min(plan.cache_budget_bytes - estimate_result_cache_bytes(completed),
                                HIGH_ACCURACY_MEMORY_BUDGET_BYTES - estimate - retained)
                extra["diffraction_sink"] = MemoryDiffractionSink(
                    remaining, calculation_signatures(point_state)["fourdstem_cube"], cancelled)
            result = calculator(point_state, existing_result=existing, progress_callback=report, **extra)
        for axis, requested in zip(plan.optical_ranges, coordinates):
            actual = float(getattr(_target(result.state_snapshot, axis.control), axis.control.field))
            if not math.isclose(actual, requested, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Assembly resolution changed {axis.control.label}; requested point was not calculated")
        if cancelled():
            raise InteractiveCancelled("Bank build cancelled; previous bank retained")
        if external_model_signature(state) != external:
            raise ValueError("External model changed during calculation; result discarded")
        assert_external_input_identities_unchanged(identities)
        completed.append(BankPoint(coordinates, result, prepare_replay(result)))
        retained = estimate_result_cache_bytes(completed)
        if retained > plan.cache_budget_bytes:
            raise ValueError("Bank exceeds the declared cache budget; reduce optical samples or increase the budget")
        if len(completed) == 1 and plan.point_count > 1 and retained * plan.point_count > plan.cache_budget_bytes:
            # Early conservative guard. Shared products may reduce the final size,
            # but a full optical bank is optional; live tuning needs no bank.
            raise ValueError(f"First point uses {retained / 1024**3:.2f} GiB; projected bank exceeds the declared cache budget. Use live tuning, fewer points, or increase the budget.")
        cache.put(result)
        report(1, 1, "Optical point stored")
    return InteractiveBank(plan, state, external, tuple(completed), retained, identities)


@dataclass(frozen=True)
class InteractiveReadout:
    coordinates: object
    detector_fractions: object
    detector_current_pa: object
    wave: object = None
    stem: object = None
    notes: tuple[str, ...] = ()
    # Detached state after applying the selected bank readout controls. Viewers
    # must not borrow current main-window geometry for a bank image.
    state_snapshot: object = None


def read_bank(bank, coordinates, *, cancelled=lambda: False):
    """Evaluate physical stops without running the source/specimen ray solver."""
    from temsim.physics.recording_clipping import clip_recording_planes
    from temsim.physics.beam_current import effective_source_current_pa
    from temsim.physics.wave_imaging import reproject_wave_image
    from temsim.calculation_manifest import assert_external_input_identities_unchanged
    assert_external_input_identities_unchanged(bank.external_inputs)
    declared = {axis.control.identity for axis in bank.plan.ranges}
    if set(coordinates) != declared:
        raise ValueError("Supply exactly the declared controls; unspecified edits cannot reuse this bank")
    values = {r.control.identity: r.validate_value(coordinates[r.control.identity]) for r in bank.plan.ranges}
    if external_model_signature(bank.source_state) != bank.external_signature:
        raise ValueError("The captured external model has changed; rebuild the bank")
    optical = tuple(values[r.control.identity] for r in bank.plan.optical_ranges)
    point = next((p for p in bank.points if p.coordinates == optical), None)
    if point is None:
        raise ValueError("Select a precomputed optical node; lens/image interpolation is not enabled")
    state = detached_state(point.result.state_snapshot)
    for axis in bank.plan.ranges:
        if axis.control.stage == "readout":
            _assign(state, axis.control, values[axis.control.identity])
    changed = any(r.control.stage == "readout" and values[r.control.identity] != float(
        getattr(_target(point.result.state_snapshot, r.control), r.control.field)) for r in bank.plan.ranges)
    if point.result.wave_imaging is not None and changed:
        from temsim.physics.wave_imaging import estimate_tem_wave_memory_bytes
        from temsim.gui.calculation_controller import HIGH_ACCURACY_MEMORY_BUDGET_BYTES
        if bank.retained_bytes + estimate_tem_wave_memory_bytes(state) > HIGH_ACCURACY_MEMORY_BUDGET_BYTES:
            raise ValueError("Pinned bank plus TEM working memory exceed the application budget")
    fractions = {p.key: 0.0 for p in state.recording_planes if p.inserted}
    for entry in point.replay:
        if cancelled():
            raise InteractiveCancelled("Readout superseded")
        b = entry.branch
        for p in state.recording_planes:
            if p.inserted and not b.z[0] <= p.z_mm <= b.z[-1]:
                raise ValueError("Requested detector plane is outside retained trajectories")
        _, _, keys = clip_recording_planes(state, b.z, b.x, b.y,
                                           entry.alive, entry.blocked_z, entry.blocked_key)
        keys = np.asarray(keys)
        for p in state.recording_planes:
            if p.key in fractions and bool(getattr(p, "readout_enabled", True)):
                fractions[p.key] += float(entry.weights[keys == p.key].sum())
    source = effective_source_current_pa(state)
    notes = ["Ray readout: retained trajectory sampling; emitted-source weights preserved."]
    wave = None
    if point.result.wave_imaging is not None:
        try:
            wave = (reproject_wave_image(state, point.result.wave_imaging) if changed
                    else point.result.wave_imaging)
            notes.append("TEM: downstream coherent propagation updated." if changed else "TEM: completed recording reused.")
        except ValueError as exc:
            notes.append(f"TEM unavailable: {exc}")
    else:
        notes.append("TEM was not calculated in the captured settings.")
    stem = point.result.stem_scan
    if stem is not None and changed:
        from temsim.physics.diffraction_memory import can_recollect_stem, recollect_stem
        if can_recollect_stem(stem):
            stem = recollect_stem(state, stem)
            notes.append("STEM: angle-resolved intensity routing approximation, not coherent arbitrary-plane imaging.")
        else:
            stem = None
            notes.append("STEM unavailable for changed stops: raw angular frames without an uncached high-angle tail are required.")
    if cancelled():
        raise InteractiveCancelled("Readout superseded")
    assert_external_input_identities_unchanged(bank.external_inputs)
    return InteractiveReadout(MappingProxyType(values), MappingProxyType(fractions),
                              MappingProxyType({k: v * source for k, v in fractions.items()}),
                              wave, stem, tuple(notes), state_snapshot=state)
