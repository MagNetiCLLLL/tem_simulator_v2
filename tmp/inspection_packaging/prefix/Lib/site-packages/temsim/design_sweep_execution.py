"""Executable, detached parameter sweeps for the Design Explorer.

The objects in :mod:`temsim.design_experiments` deliberately describe only a
plan.  This module is the execution boundary: every point is reconstructed
from the immutable recipe, resolved against the selected TOML assembly, and
sent through the same high-accuracy pipeline as the main window.  It never
borrows or mutates the live GUI ``State``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
import math
from time import perf_counter
from types import MappingProxyType

from temsim.assembly_catalog import (
    AssemblyCatalog,
    AssemblySelection,
)
from temsim.calculation_cache import (
    calculation_signatures,
    external_model_signature,
    matching_products,
)
from temsim.calculation_manifest import (
    assert_external_input_identities_unchanged,
    assert_external_inputs_unchanged,
    capture_calculation_manifest,
    resolved_assembly_geometry_fingerprint,
)
from temsim.design_experiments import (
    DesignRecipe,
    MetricObservation,
    ParameterSweep,
    SensitivityEstimate,
    SweepAxis,
    ToleranceResult,
    ToleranceRule,
    estimate_sensitivities,
    evaluate_tolerances,
    parameter_tokens,
    parameter_value,
    validate_runtime_sweep_path,
)
from temsim.immutable_json import json_digest, thaw_json
from temsim.simulation_pipeline import CalculationResult, calculate


MAX_EXECUTABLE_SWEEP_POINTS = 256


class SweepExecutionCancelled(RuntimeError):
    """Raised internally when a cooperative sweep cancellation is observed."""


@dataclass(frozen=True, slots=True)
class SweepMetricDefinition:
    """Public identity, units and source for one extracted solver metric."""

    key: str
    label: str
    unit: str
    source: str


SWEEP_METRICS = (
    SweepMetricDefinition(
        "sample_illumination_diameter_95_um",
        "Sample illumination D95",
        "um",
        "CalculationResult.simulation.metrics",
    ),
    SweepMetricDefinition(
        "sample_convergence_95_mrad",
        "Sample convergence 95%",
        "mrad",
        "CalculationResult.simulation.metrics",
    ),
    SweepMetricDefinition(
        "sample_waist_offset_mm",
        "Sample waist offset",
        "mm",
        "CalculationResult.simulation.metrics",
    ),
    SweepMetricDefinition(
        "sample_surviving_current_pa",
        "Current reaching sample",
        "pA",
        "CalculationResult.simulation.metrics",
    ),
    SweepMetricDefinition(
        "image_conjugacy_residual_m_per_rad",
        "Image conjugacy residual",
        "m/rad",
        "CalculationResult.simulation.metrics",
    ),
    SweepMetricDefinition(
        "diffraction_conjugacy_residual",
        "Diffraction conjugacy residual",
        "dimensionless",
        "CalculationResult.simulation.metrics",
    ),
    SweepMetricDefinition(
        "magnification",
        "Image magnification",
        "x",
        "CalculationResult.simulation.metrics",
    ),
    SweepMetricDefinition(
        "effective_camera_length_m",
        "Effective camera length",
        "m",
        "CalculationResult.simulation.metrics",
    ),
    SweepMetricDefinition(
        "eds_total_expected_counts",
        "EDS expected counts",
        "counts",
        "CalculationResult.specimen_interactions.eds_spectrum.metrics"
        ".total_expected_counts",
    ),
)


@dataclass(frozen=True, slots=True)
class SweepProgress:
    point_index: int
    completed_points: int
    total_points: int
    stage_completed: int
    stage_total: int
    stage: str


@dataclass(frozen=True, slots=True)
class SweepPointExecution:
    point_index: int
    coordinates: Mapping[str, float]
    state_digest: str
    geometry_fingerprint: str
    request_signature: str
    metrics: Mapping[str, float]
    calculated_products: tuple[str, ...]
    reused_products: tuple[str, ...]
    duration_s: float
    complete_cache_hit: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coordinates",
            MappingProxyType({
                str(key): float(value)
                for key, value in self.coordinates.items()
            }),
        )
        object.__setattr__(
            self,
            "metrics",
            MappingProxyType({
                str(key): float(value)
                for key, value in self.metrics.items()
            }),
        )


@dataclass(frozen=True, slots=True)
class PointToleranceAssessment:
    point_index: int
    results: tuple[ToleranceResult, ...]


@dataclass(frozen=True, slots=True)
class SweepExecutionResult:
    recipe_digest: str
    point_results: tuple[SweepPointExecution, ...]
    sensitivities: tuple[SensitivityEstimate, ...]
    tolerances: tuple[PointToleranceAssessment, ...]
    metric_definitions: tuple[SweepMetricDefinition, ...]
    cancelled: bool
    sweep_axes: tuple[SweepAxis, ...] = field(default_factory=tuple)
    analysis_notes: tuple[str, ...] = ()

    @property
    def completed_points(self) -> int:
        return len(self.point_results)


class SweepCalculationCache:
    """Small rolling cache used only during one detached sweep.

    ``CalculationResult`` can contain large arrays, so this cache is bounded
    by a deliberately small entry count.  Unchanged products inside adjacent
    points are shared by the normal dependency-scoped calculation pipeline.
    Persistent incident seeds are handled separately by ``ArtifactStore``.
    """

    _WEIGHTS = {
        "incident": 128,
        "column": 64,
        "wave_source": 64,
        "wave": 32,
        "elastic": 16,
        "stem_transport": 16,
        "stem": 16,
        "fourdstem_cube": 16,
        "eds": 8,
        "energy_filter": 4,
        "scan_geometry": 2,
        "scan_ray_paths": 2,
        "sample_region": 2,
        "sample_downstream": 1,
    }

    def __init__(self, *, maximum_results: int = 1) -> None:
        maximum = int(maximum_results)
        if not 1 <= maximum <= 8:
            raise ValueError("Sweep result-cache size must be between 1 and 8")
        self.maximum_results = maximum
        self._results: OrderedDict[str, CalculationResult] = OrderedDict()

    def select(
        self, signatures: Mapping[str, str]
    ) -> tuple[CalculationResult | None, bool]:
        request = str(signatures.get("request", ""))
        exact = self._results.get(request)
        if exact is not None:
            self._results.move_to_end(request)
            return exact, True
        if not self._results:
            return None, False

        def score(result: CalculationResult) -> int:
            reusable = matching_products(result.signatures, dict(signatures))
            return sum(self._WEIGHTS.get(key, 1) for key in reusable)

        key, result = max(
            reversed(tuple(self._results.items())),
            key=lambda item: score(item[1]),
        )
        if score(result) <= 0:
            return None, False
        self._results.move_to_end(key)
        return result, False

    def put(self, result: CalculationResult) -> None:
        request = str((result.signatures or {}).get("request", ""))
        if not request:
            return
        self._results[request] = result
        self._results.move_to_end(request)
        while len(self._results) > self.maximum_results:
            self._results.popitem(last=False)

    def clear(self) -> None:
        self._results.clear()


def _selection_from_recipe(recipe: DesignRecipe) -> AssemblySelection:
    values = recipe.selection
    return AssemblySelection(
        gun=str(values["gun"]),
        column=str(values["column"]),
        recording=str(values["recording"]),
        beam_blanker=str(values.get("beam_blanker", "None")),
    )


def _runtime_operating_values(state) -> dict[str, dict[str, object]]:
    """Capture scalar user controls that an assembly resolution may touch."""

    from temsim.runtime_parameters import editable_parameters, runtime_targets

    return {
        key: {
            parameter.name: deepcopy(parameter.value)
            for parameter in editable_parameters(target)
            if parameter.value is not None
        }
        for key, target in runtime_targets(state).items()
    }


def rebuild_recipe_state(
    recipe: DesignRecipe,
    *,
    state_payload: Mapping[str, object] | None = None,
    catalog: AssemblyCatalog | None = None,
    validate_recipe_geometry: bool = False,
):
    """Rebuild a detached state and selected assembly without applying a preset.

    The serialized state owns user controls and lens excitations; the selected
    TOMLs own component geometry and hardware topology.  Resolving the latter
    with ``preserve_operating_parameters=True`` is therefore essential.  A
    second layout validation reasserts geometry after restoring scalar runtime
    controls and still does not invoke an operating-mode preset solver.
    """

    from temsim.column.state_layout import apply_physical_layout_to_state
    from temsim.optics.model import State
    from temsim.profile_io import apply_profile_values

    payload = thaw_json(state_payload or recipe.state_payload)
    simulation_time_s = float(payload.pop("simulation_time_s", 0.0))
    state = State.from_dict(payload)
    state.simulation_time_s = simulation_time_s
    operating_values = _runtime_operating_values(state)
    selected_catalog = catalog or AssemblyCatalog()
    selection = selected_catalog.normalise_selection(
        _selection_from_recipe(recipe)
    )
    resolved_assembly = selected_catalog.apply(
        state,
        selection,
        preserve_operating_parameters=True,
    )
    skipped = apply_profile_values(state, operating_values)
    if skipped:
        raise ValueError(
            "Recipe contains controls unavailable in its selected assembly: "
            + ", ".join(skipped)
        )
    apply_physical_layout_to_state(
        state,
        preserve_operating_parameters=True,
        assembly_root=selected_catalog.root,
        assembly=resolved_assembly,
    )
    if (
        tuple(state._resolved_assembly.selected_module_paths)
        != tuple(resolved_assembly.selected_module_paths)
    ):
        raise RuntimeError(
            "Detached recipe rebuild replaced its selected TOML assembly"
        )
    # State serialization normalises a few coupled read-backs (notably the
    # Energy Filter entrance plane).  Capture uses that same serialization
    # boundary before recording its geometry fingerprint, so cross it here as
    # well to avoid treating sub-picometre floating round-off as changed TOML.
    external_model_signature(state)
    actual_geometry = resolved_assembly_geometry_fingerprint(state)
    if (
        validate_recipe_geometry
        and recipe.geometry_fingerprint
        and actual_geometry != recipe.geometry_fingerprint
    ):
        raise ValueError(
            "The selected TOML assembly geometry has changed since this "
            "design was captured; capture A/B again before running a sweep"
        )
    return state, selection


def _prepare_high_accuracy_request(state, recipe: DesignRecipe) -> None:
    request = recipe.request
    emitter = getattr(state.electron_gun, "emitter", None)
    if emitter is not None:
        emitter.ray_count = int(request.ray_count)
    else:
        state.electron_gun.ray_count = int(request.ray_count)
    state.step_mm = float(request.step_mm)
    state.history_step_mm = max(float(request.step_mm), 0.5)


def _assert_recipe_external_inputs(
    state: object, recipe: DesignRecipe
) -> None:
    """Verify a rebuilt state still uses the exact captured external model."""

    expected = str(recipe.external_model_signature)
    if int(recipe.schema_version) < 2 or not expected:
        raise ValueError(
            "This legacy design recipe lacks captured external-model "
            "provenance; capture A/B again before running a sweep"
        )
    actual = external_model_signature(state)
    if actual != expected:
        raise ValueError(
            "The CIF, selected TOML assembly, field map, or other external "
            "model differs from the captured design; capture A/B again"
        )
    assert_external_input_identities_unchanged(recipe.external_inputs)


def _runtime_target_for_path(state: object, path: str):
    """Resolve the object owning one allowlisted serialized runtime leaf."""

    from temsim.runtime_parameters import RuntimeTarget

    current = state
    parsed = parameter_tokens(path)
    for name, selector in parsed[:-1]:
        if isinstance(current, Mapping):
            current = current[name]
        elif isinstance(current, (list, tuple)):
            matches = [
                item for item in current
                if str(getattr(item, "key", "")) == name
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Runtime sweep target {path!r} is unavailable"
                )
            current = matches[0]
        else:
            current = getattr(current, name)
        if selector is not None:
            if selector.isdigit():
                current = current[int(selector)]
            else:
                matches = [
                    item for item in current
                    if str(getattr(item, "key", "")) == selector
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Runtime sweep target {path!r} is unavailable"
                    )
                current = matches[0]
    return RuntimeTarget(str(path), str(path), current), parsed[-1][0]


def _assert_sweep_coordinates_applied(
    state: object,
    point,
    axes: Sequence,
) -> None:
    """Reject a point if assembly reconstruction erased its requested value."""

    from temsim.runtime_parameters import validate_runtime_assignment

    serialized = state.to_dict()
    for axis in axes:
        path = axis.path
        expected = float(point.coordinates[path])
        planned = validate_runtime_sweep_path(point.state_payload, path)
        if not math.isclose(
            planned, expected, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"Sweep point {point.index} coordinate {path!r} does not "
                "match its state payload"
            )
        target, field_name = _runtime_target_for_path(state, path)
        old_value = getattr(target.obj, field_name)
        candidate: object = expected
        if (
            isinstance(old_value, int)
            and not isinstance(old_value, bool)
            and expected.is_integer()
        ):
            candidate = int(expected)
        validate_runtime_assignment(target, field_name, candidate)
        actual = validate_runtime_sweep_path(serialized, path)
        if not math.isclose(
            actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"Sweep parameter {path!r} was not retained after selected "
                "assembly reconstruction"
            )


def extract_sweep_metrics(result: CalculationResult) -> dict[str, float]:
    """Extract only named, finite, traceable metrics from a real result."""

    values: dict[str, object] = dict(result.simulation.metrics or {})
    spectrum = getattr(
        getattr(result, "specimen_interactions", None),
        "eds_spectrum",
        None,
    )
    spectrum_metrics = dict(getattr(spectrum, "metrics", {}) or {})
    if "total_expected_counts" in spectrum_metrics:
        values["eds_total_expected_counts"] = spectrum_metrics[
            "total_expected_counts"
        ]
    extracted = {}
    for definition in SWEEP_METRICS:
        value = values.get(definition.key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        converted = float(value)
        if math.isfinite(converted):
            extracted[definition.key] = converted
    return extracted


def _persistent_incident_seed(
    artifact_store,
    manifest,
    state,
) -> CalculationResult | None:
    if artifact_store is None:
        return None
    try:
        simulation = artifact_store.get_incident_simulation_seed(manifest)
    except Exception:
        return None
    if simulation is None:
        return None
    return CalculationResult(
        simulation=simulation,
        energy_filter=None,
        state_snapshot=state,
        signatures={
            "incident": str(
                manifest.calculation_signatures.get("incident", "")
            )
        },
    )


def execute_parameter_sweep(
    recipe: DesignRecipe,
    sweep: ParameterSweep,
    *,
    catalog: AssemblyCatalog | None = None,
    artifact_store=None,
    calculation_cache: SweepCalculationCache | None = None,
    tolerance_rules: Sequence[ToleranceRule] = (),
    progress_callback: Callable[[SweepProgress], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
    calculator: Callable[..., CalculationResult] = calculate,
    maximum_points: int = MAX_EXECUTABLE_SWEEP_POINTS,
) -> SweepExecutionResult:
    """Execute a bounded sweep through the production calculation pipeline.

    Cancellation is cooperative and is checked before every point and on every
    inner solver progress callback.  A cancelled in-flight point is discarded;
    only complete point observations are returned.
    """

    if sweep.recipe_digest != recipe.digest:
        raise ValueError("Sweep was not planned from this design recipe")
    total = len(sweep.points)
    if total <= 0:
        raise ValueError("Sweep has no calculation points")
    if total > int(maximum_points):
        raise ValueError(
            f"Sweep has {total} points; execution limit is {maximum_points}"
        )
    for expected, point in enumerate(sweep.points):
        if point.index != expected:
            raise ValueError("Sweep point indices must be contiguous")
        if json_digest(point.state_payload) != point.state_digest:
            raise ValueError(f"Sweep point {point.index} state checksum failed")
    for axis in sweep.axes:
        validate_runtime_sweep_path(recipe.state_payload, axis.path)

    selected_catalog = catalog or AssemblyCatalog()
    # Check that immutable recipe inputs can still be reconstructed before any
    # expensive point begins.  Point payloads may intentionally alter geometry,
    # so the base recipe is the only geometry identity checked against capture.
    base_state, base_selection = rebuild_recipe_state(
        recipe,
        catalog=selected_catalog,
        validate_recipe_geometry=True,
    )
    _assert_recipe_external_inputs(base_state, recipe)
    _prepare_high_accuracy_request(base_state, recipe)
    baseline_manifest = capture_calculation_manifest(
        base_state,
        selection=base_selection,
        ray_count=recipe.request.ray_count,
        step_mm=recipe.request.step_mm,
    )
    cache = calculation_cache or SweepCalculationCache()
    point_results: list[SweepPointExecution] = []
    observations: list[MetricObservation] = []
    assessments: list[PointToleranceAssessment] = []
    cancelled = False

    def cancellation_requested() -> bool:
        return bool(cancel_requested is not None and cancel_requested())

    for point in sweep.points:
        if cancellation_requested():
            cancelled = True
            break
        assert_external_inputs_unchanged(baseline_manifest)
        state, selection = rebuild_recipe_state(
            recipe,
            state_payload=point.state_payload,
            catalog=selected_catalog,
        )
        _assert_recipe_external_inputs(state, recipe)
        _assert_sweep_coordinates_applied(state, point, sweep.axes)
        _prepare_high_accuracy_request(state, recipe)
        signatures = calculation_signatures(state)
        existing, exact_hit = cache.select(signatures)
        # Keep the detached route under the same workstation budget enforced
        # by the main High-accuracy controller.  This import is intentionally
        # local so the reusable sweep model remains independent of Qt at load.
        from temsim.gui.calculation_controller import (
            HIGH_ACCURACY_MEMORY_BUDGET_BYTES,
            estimate_calculation_memory_bytes,
            estimate_result_cache_bytes,
            format_memory_size,
        )
        estimate = estimate_calculation_memory_bytes(
            state,
            "High accuracy",
            recipe.request.ray_count,
            recipe.request.step_mm,
        )
        if estimate > HIGH_ACCURACY_MEMORY_BUDGET_BYTES:
            raise ValueError(
                "Sweep point needs approximately "
                f"{format_memory_size(estimate)}, above the "
                f"{format_memory_size(HIGH_ACCURACY_MEMORY_BUDGET_BYTES)} "
                "High-accuracy memory budget"
            )
        if (
            existing is not None
            and not exact_hit
            and estimate_result_cache_bytes(existing)
            > HIGH_ACCURACY_MEMORY_BUDGET_BYTES - estimate
        ):
            existing = None
        manifest = None
        if artifact_store is not None:
            try:
                manifest = capture_calculation_manifest(
                    state,
                    selection=selection,
                    ray_count=recipe.request.ray_count,
                    step_mm=recipe.request.step_mm,
                )
            except Exception:
                manifest = None
        if existing is None and manifest is not None:
            existing = _persistent_incident_seed(
                artifact_store, manifest, state
            )

        started = perf_counter()
        try:
            if exact_hit and existing is not None:
                result = replace(
                    existing,
                    cache_hit=True,
                    calculated_products=frozenset(),
                    reused_products=frozenset(
                        set(existing.calculated_products)
                        | set(existing.reused_products)
                    ),
                )
                if progress_callback is not None:
                    progress_callback(SweepProgress(
                        point.index,
                        len(point_results),
                        total,
                        1,
                        1,
                        "Reusing completed point",
                    ))
            else:
                def inner_progress(
                    completed: int, stage_total: int, stage: str
                ) -> None:
                    if cancellation_requested():
                        raise SweepExecutionCancelled()
                    if progress_callback is not None:
                        progress_callback(SweepProgress(
                            point.index,
                            len(point_results),
                            total,
                            int(completed),
                            int(stage_total),
                            str(stage),
                        ))

                kwargs = {"progress_callback": inner_progress}
                if existing is not None:
                    kwargs["existing_result"] = existing
                result = calculator(state, **kwargs)
        except SweepExecutionCancelled:
            cancelled = True
            break
        if not isinstance(result, CalculationResult):
            raise TypeError("Sweep calculator did not return CalculationResult")
        duration = perf_counter() - started
        cache.put(result)
        if manifest is not None:
            expected_incident = str(
                manifest.calculation_signatures.get("incident", "")
            )
            if str((result.signatures or {}).get("incident", "")) == expected_incident:
                try:
                    artifact_store.put_incident_simulation_seed(
                        manifest, result.simulation
                    )
                except Exception:
                    pass

        metrics = extract_sweep_metrics(result)
        observations.append(MetricObservation(point.index, metrics))
        assessments.append(PointToleranceAssessment(
            point.index,
            evaluate_tolerances(metrics, tolerance_rules),
        ))
        point_results.append(SweepPointExecution(
            point_index=point.index,
            coordinates=point.coordinates,
            state_digest=point.state_digest,
            geometry_fingerprint=(
                resolved_assembly_geometry_fingerprint(state)
            ),
            request_signature=str(
                (result.signatures or {}).get("request", "")
            ),
            metrics=metrics,
            calculated_products=tuple(sorted(result.calculated_products)),
            reused_products=tuple(sorted(result.reused_products)),
            duration_s=duration,
            complete_cache_hit=bool(result.cache_hit),
        ))
        if progress_callback is not None:
            progress_callback(SweepProgress(
                point.index,
                len(point_results),
                total,
                1,
                1,
                "Point complete",
            ))

    notes = []
    sensitivities: tuple[SensitivityEstimate, ...] = ()
    if not cancelled and len(observations) == total:
        try:
            sensitivities = estimate_sensitivities(sweep, observations)
        except ValueError as exc:
            notes.append(f"Sensitivity unavailable: {exc}")
    elif cancelled:
        notes.append("Sensitivity omitted because the sweep was cancelled")
    if point_results and not any(item.metrics for item in point_results):
        notes.append("No finite configured sweep metrics were produced")
    return SweepExecutionResult(
        recipe_digest=recipe.digest,
        point_results=tuple(point_results),
        sensitivities=sensitivities,
        tolerances=tuple(assessments),
        metric_definitions=SWEEP_METRICS,
        cancelled=cancelled,
        sweep_axes=tuple(sweep.axes),
        analysis_notes=tuple(notes),
    )


__all__ = (
    "MAX_EXECUTABLE_SWEEP_POINTS",
    "PointToleranceAssessment",
    "SWEEP_METRICS",
    "SweepCalculationCache",
    "SweepExecutionCancelled",
    "SweepExecutionResult",
    "SweepMetricDefinition",
    "SweepPointExecution",
    "SweepProgress",
    "execute_parameter_sweep",
    "extract_sweep_metrics",
    "rebuild_recipe_state",
)
