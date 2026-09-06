"""State recipes, undo history and bounded Design Explorer experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import itertools
import math
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType

import numpy as np

from temsim.design_explorer import DesignSnapshot, HighAccuracyRequest
from temsim.calculation_manifest import ExternalInputIdentity
from temsim.immutable_json import (
    canonical_json_bytes,
    freeze_json,
    json_digest,
    thaw_json,
)


DESIGN_RECIPE_SCHEMA_VERSION = 2
_SUPPORTED_RECIPE_SCHEMAS = frozenset({1, DESIGN_RECIPE_SCHEMA_VERSION})
_PATH_TOKEN = re.compile(r"(?P<name>[^.\[\]]+)(?:\[(?P<selector>[^\]]+)\])?")

# A design sweep changes operating controls only.  In particular it must not
# make the serialized State a competing owner for TOML/component geometry.
# These explicit families cover the controls exposed by Design Explorer plus
# useful source/current and specimen simulation controls.  File/source keys,
# component placement, lens/pole geometry and solver output paths are absent
# by construction.
_ROOT_RUNTIME_SWEEP_FIELDS = frozenset({"column_current_limit_percent"})
_GUN_RUNTIME_SWEEP_FIELDS = frozenset({
    "angular_cutoff_mrad",
    "angular_rms_mrad",
    "beam_blanked",
    "boersch_sigma_ev",
    "centre_offset_um",
    "electric_field_v_per_m",
    "electric_quadrupole_gradient_v_per_m2",
    "emission_current_na",
    "emission_energy_ev",
    "energy_half_range_ev",
    "energy_spread_fwhm_ev",
    "gap_um",
    "gradient_t_per_m",
    "high_tension_kv",
    "lower_field_x_mt",
    "lower_field_y_mt",
    "magnetic_field_mt",
    "minimum_kinetic_energy_ev",
    "potential_scale",
    "requested_pass_window_ev",
    "rotation_deg",
    "upper_field_x_mt",
    "upper_field_y_mt",
    "vacuum_pa",
    "virtual_source_fwhm_nm",
    "voltage_kv",
    "young_decay_width_ev",
})
_SAMPLE_RUNTIME_SWEEP_FIELDS = frozenset({
    "centre_x_nm",
    "centre_y_nm",
    "diffuse_broadening_mrad",
    "eds_detector_efficiency",
    "eds_elastic_max_events",
    "eds_elastic_seed",
    "eds_energy_resolution_fwhm_ev",
    "eds_poisson_seed",
    "eds_spectrum_bin_width_ev",
    "eds_spectrum_max_energy_ev",
    "eds_support_offset_x_um",
    "eds_support_offset_y_um",
    "eds_support_rotation_deg",
    "excitation_error_inv_nm",
    "g_inv_nm",
    "real_absorption_mean_free_path_nm",
    "real_ionisation_energy_ev",
    "real_ionisation_mean_free_path_nm",
    "real_other_inelastic_energy_ev",
    "real_other_inelastic_mean_free_path_nm",
    "real_plasmon_energy_ev",
    "real_plasmon_mean_free_path_nm",
    "real_tail_areal_density_atoms_nm2",
    "real_tail_atomic_number",
    "real_tail_max_angle_mrad",
    "real_tail_screening_angle_mrad",
    "rocking_width_inv_nm",
    "sample_region_downstream_distance_um",
    "sample_region_photon_path_count",
    "sample_region_secondary_path_count",
    "sample_region_seed",
    "sample_region_upstream_distance_um",
    "scan_origin_x_nm",
    "scan_origin_y_nm",
    "size_x_nm",
    "size_y_nm",
    "specimen_rotation_x_deg",
    "specimen_rotation_y_deg",
    "specimen_rotation_z_deg",
    "stem_fourdstem_charge_spread_sigma_px",
    "stem_fourdstem_dark_electrons_per_pixel",
    "stem_fourdstem_gain_counts_per_electron",
    "stem_fourdstem_offset_counts",
    "stem_fourdstem_quantum_efficiency",
    "stem_fourdstem_read_noise_electrons_rms",
    "stem_fourdstem_saturation_electrons",
    "stem_fourdstem_seed",
    "stem_fourdstem_virtual_inner_mrad",
    "stem_fourdstem_virtual_outer_mrad",
    "stem_poisson_seed",
    "thickness_nm",
    "virtual_diffraction_angle_mrad",
    "virtual_diffraction_azimuth_deg",
    "virtual_diffraction_relative_weight",
    "virtual_scattering_angle_mrad",
    "virtual_scattering_azimuth_samples",
    "virtual_scattering_relative_weight",
    "wave_bandwidth_fraction",
    "wave_defocus_nm",
    "wave_field_of_view_angstrom",
    "wave_frozen_phonon_configurations",
    "wave_frozen_phonon_seed",
    "wave_frozen_phonon_sigma_angstrom",
    "wave_grid_pixels",
    "wave_probe_padding_factor",
    "wave_slice_thickness_angstrom",
})


def _normalise_parameter_path(path: str) -> str:
    value = str(path).strip()
    if value.startswith("state."):
        value = value[len("state."):]
    if not value:
        raise ValueError("Parameter path cannot be empty")
    return value


def _tokens(path: str) -> tuple[tuple[str, str | None], ...]:
    normalized = _normalise_parameter_path(path)
    result = []
    position = 0
    for match in _PATH_TOKEN.finditer(normalized):
        if match.start() != position or (
            match.start() > 0 and normalized[match.start() - 1] != "."
        ):
            raise ValueError(f"Invalid parameter path: {path!r}")
        result.append((match.group("name"), match.group("selector")))
        position = match.end() + 1
    if not result or position - 1 != len(normalized):
        raise ValueError(f"Invalid parameter path: {path!r}")
    return tuple(result)


def parameter_tokens(path: str) -> tuple[tuple[str, str | None], ...]:
    """Return the normalized stable path tokens used by recipe operations."""

    return _tokens(path)


def _select_sequence_item(values: list[object], selector: str) -> object:
    if selector.isdigit():
        index = int(selector)
        try:
            return values[index]
        except IndexError as exc:
            raise KeyError(f"Sequence index {index} is out of range") from exc
    matches = [
        item
        for item in values
        if isinstance(item, Mapping) and str(item.get("key", "")) == selector
    ]
    if len(matches) != 1:
        raise KeyError(
            f"Expected one component with key {selector!r}, found {len(matches)}"
        )
    return matches[0]


def replace_parameter(
    payload: Mapping[str, object], path: str, value: object
) -> dict[str, object]:
    """Return a detached payload with one existing stable path replaced."""

    result = thaw_json(freeze_json(payload))
    current: object = result
    parsed = _tokens(path)
    for index, (name, selector) in enumerate(parsed):
        if not isinstance(current, dict) or name not in current:
            raise KeyError(f"Unknown parameter path: {path!r}")
        final = index == len(parsed) - 1
        target = current[name]
        if selector is not None:
            if not isinstance(target, list):
                raise KeyError(f"Path segment {name!r} is not a sequence")
            selected = _select_sequence_item(target, selector)
            if final:
                selected_index = target.index(selected)
                target[selected_index] = thaw_json(freeze_json(value))
                return result
            current = selected
        elif final:
            current[name] = thaw_json(freeze_json(value))
            return result
        else:
            current = target
    raise RuntimeError("Parameter replacement did not reach a leaf")


def parameter_value(payload: Mapping[str, object], path: str) -> object:
    """Read one existing stable parameter path from a serialized State."""

    current: object = payload
    for name, selector in _tokens(path):
        if not isinstance(current, Mapping) or name not in current:
            raise KeyError(f"Unknown parameter path: {path!r}")
        current = current[name]
        if selector is not None:
            if not isinstance(current, (list, tuple)):
                raise KeyError(f"Path segment {name!r} is not a sequence")
            current = _select_sequence_item(list(current), selector)
    return current


def validate_runtime_sweep_path(
    payload: Mapping[str, object], path: str
) -> float:
    """Validate and read an explicitly supported numeric sweep control.

    The validator is intentionally narrower than generic recipe editing.
    Runtime strength/source/specimen controls may be swept, while any field
    capable of changing component geometry, an external-input selection, or
    a disk output destination is rejected before detached state rebuilds.
    """

    normalized = _normalise_parameter_path(path)
    parsed = _tokens(normalized)
    allowed = False
    if (
        len(parsed) == 1
        and parsed[0][1] is None
        and parsed[0][0] in _ROOT_RUNTIME_SWEEP_FIELDS
    ):
        allowed = True
    elif (
        len(parsed) == 2
        and parsed[0][0] == "lenses"
        and parsed[0][1] is not None
        and parsed[1] == ("percent", None)
    ):
        allowed = True
    elif (
        len(parsed) == 2
        and parsed[0] == ("sample", None)
        and parsed[1][1] is None
        and parsed[1][0] in _SAMPLE_RUNTIME_SWEEP_FIELDS
    ):
        allowed = True
    elif (
        len(parsed) >= 3
        and parsed[0] == ("electron_gun", None)
        and all(selector is None for _name, selector in parsed)
        and parsed[-1][0] in _GUN_RUNTIME_SWEEP_FIELDS
    ):
        allowed = True
    if not allowed:
        raise ValueError(
            f"Parameter {normalized!r} is not an allowed runtime sweep "
            "control; TOML-owned geometry, positions, profiles, external "
            "input selections, and output paths cannot be swept"
        )
    value = parameter_value(payload, normalized)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Runtime sweep parameter {normalized!r} must be numeric"
        )
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(
            f"Runtime sweep parameter {normalized!r} must be finite"
        )
    return numeric


@dataclass(frozen=True, slots=True)
class ParameterEdit:
    path: str
    value: object

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalise_parameter_path(self.path))
        object.__setattr__(self, "value", freeze_json(self.value))


@dataclass(frozen=True, slots=True)
class DesignRecipe:
    name: str
    created_at_utc: str
    selection: Mapping[str, str]
    base_state_payload: Mapping[str, object]
    request: HighAccuracyRequest
    geometry_fingerprint: str
    external_model_signature: str = ""
    external_inputs: tuple[ExternalInputIdentity, ...] = ()
    edits: tuple[ParameterEdit, ...] = ()
    parent_digest: str = ""
    schema_version: int = DESIGN_RECIPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.schema_version) not in _SUPPORTED_RECIPE_SCHEMAS:
            raise ValueError("Unsupported design-recipe schema")
        name = str(self.name).strip()
        if not name:
            raise ValueError("Design recipe needs a name")
        if not isinstance(self.request, HighAccuracyRequest):
            raise TypeError("Recipe request must be HighAccuracyRequest")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "selection", freeze_json(self.selection))
        object.__setattr__(
            self, "base_state_payload", freeze_json(self.base_state_payload)
        )
        object.__setattr__(
            self,
            "external_model_signature",
            str(self.external_model_signature),
        )
        external_inputs = tuple(self.external_inputs)
        if any(
            not isinstance(row, ExternalInputIdentity)
            for row in external_inputs
        ):
            raise TypeError(
                "Recipe external inputs must be identity records"
            )
        object.__setattr__(self, "external_inputs", external_inputs)
        object.__setattr__(self, "edits", tuple(self.edits))

    @property
    def state_payload(self) -> Mapping[str, object]:
        payload = thaw_json(self.base_state_payload)
        for edit in self.edits:
            payload = replace_parameter(payload, edit.path, edit.value)
        return freeze_json(payload)

    @property
    def digest(self) -> str:
        identity = {
            "schema_version": self.schema_version,
            "selection": self.selection,
            "base_state_payload": self.base_state_payload,
            "request": self.request,
            "geometry_fingerprint": self.geometry_fingerprint,
            "edits": self.edits,
            "parent_digest": self.parent_digest,
        }
        if int(self.schema_version) >= 2:
            identity.update({
                "external_model_signature": self.external_model_signature,
                "external_inputs": self.external_inputs,
            })
        return json_digest(identity)

    def with_edits(
        self, name: str, edits: Sequence[ParameterEdit]
    ) -> "DesignRecipe":
        return DesignRecipe(
            name=name,
            created_at_utc=datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            selection=self.selection,
            base_state_payload=self.state_payload,
            request=self.request,
            geometry_fingerprint=self.geometry_fingerprint,
            external_model_signature=self.external_model_signature,
            external_inputs=self.external_inputs,
            edits=tuple(edits),
            parent_digest=self.digest,
            schema_version=self.schema_version,
        )

    def to_dict(self) -> dict[str, object]:
        document = {
            "schema_version": self.schema_version,
            "name": self.name,
            "created_at_utc": self.created_at_utc,
            "selection": thaw_json(self.selection),
            "base_state_payload": thaw_json(self.base_state_payload),
            "request": {
                "ray_count": self.request.ray_count,
                "step_mm": self.request.step_mm,
            },
            "geometry_fingerprint": self.geometry_fingerprint,
            "edits": [
                {"path": edit.path, "value": thaw_json(edit.value)}
                for edit in self.edits
            ],
            "parent_digest": self.parent_digest,
            "digest": self.digest,
        }
        if int(self.schema_version) >= 2:
            document.update({
                "external_model_signature": self.external_model_signature,
                "external_inputs": [
                    thaw_json(freeze_json(row))
                    for row in self.external_inputs
                ],
            })
        return document


def recipe_from_snapshot(
    snapshot: DesignSnapshot, *, name: str
) -> DesignRecipe:
    return DesignRecipe(
        name=name,
        created_at_utc=snapshot.captured_at_utc,
        selection=snapshot.selection,
        base_state_payload=snapshot.state_payload,
        request=snapshot.request,
        geometry_fingerprint=str(
            getattr(snapshot, "geometry_fingerprint", "")
        ),
        external_model_signature=str(
            getattr(snapshot, "external_model_signature", "")
        ),
        external_inputs=tuple(getattr(snapshot, "external_inputs", ())),
    )


def save_recipe(path: str | Path, recipe: DesignRecipe) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(recipe.to_dict()))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_recipe(path: str | Path) -> DesignRecipe:
    import json

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Design recipe must be a JSON object")
    request = document.get("request", {})
    edits = tuple(
        ParameterEdit(str(row["path"]), row.get("value"))
        for row in document.get("edits", ())
    )
    raw_external_inputs = document.get("external_inputs", ())
    if not isinstance(raw_external_inputs, (list, tuple)) or any(
        not isinstance(row, Mapping) for row in raw_external_inputs
    ):
        raise ValueError("Recipe external_inputs must be a list of records")
    external_inputs = tuple(
        ExternalInputIdentity(
            role=str(row.get("role", "")),
            path=str(row.get("path", "")),
            available=bool(row.get("available", False)),
            size_bytes=int(row.get("size_bytes", 0)),
            sha256=str(row.get("sha256", "")),
        )
        for row in raw_external_inputs
    )
    recipe = DesignRecipe(
        name=str(document.get("name", "")),
        created_at_utc=str(document.get("created_at_utc", "")),
        selection=document.get("selection", {}),
        base_state_payload=document.get("base_state_payload", {}),
        request=HighAccuracyRequest(
            int(request["ray_count"]), float(request["step_mm"])
        ),
        geometry_fingerprint=str(document.get("geometry_fingerprint", "")),
        external_model_signature=str(
            document.get("external_model_signature", "")
        ),
        external_inputs=external_inputs,
        edits=edits,
        parent_digest=str(document.get("parent_digest", "")),
        schema_version=int(document.get("schema_version", 0)),
    )
    expected = str(document.get("digest", ""))
    if expected and expected != recipe.digest:
        raise ValueError("Design recipe checksum does not match its content")
    return recipe


class StateHistory:
    """Bounded in-memory history of immutable design recipes."""

    def __init__(self, *, limit: int = 100) -> None:
        self.limit = int(limit)
        if self.limit <= 0:
            raise ValueError("State-history limit must be positive")
        self._entries: list[DesignRecipe] = []
        self._cursor = -1

    @property
    def current(self) -> DesignRecipe | None:
        return self._entries[self._cursor] if self._cursor >= 0 else None

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        return 0 <= self._cursor < len(self._entries) - 1

    def record(self, recipe: DesignRecipe) -> bool:
        if self.current is not None and self.current.digest == recipe.digest:
            return False
        del self._entries[self._cursor + 1:]
        self._entries.append(recipe)
        if len(self._entries) > self.limit:
            del self._entries[: len(self._entries) - self.limit]
        self._cursor = len(self._entries) - 1
        return True

    def undo(self) -> DesignRecipe:
        if not self.can_undo:
            raise IndexError("No earlier design state")
        self._cursor -= 1
        return self._entries[self._cursor]

    def redo(self) -> DesignRecipe:
        if not self.can_redo:
            raise IndexError("No later design state")
        self._cursor += 1
        return self._entries[self._cursor]


@dataclass(frozen=True, slots=True)
class SweepAxis:
    path: str
    values: tuple[float, ...]
    unit: str = ""

    def __post_init__(self) -> None:
        path = _normalise_parameter_path(self.path)
        values = tuple(float(value) for value in self.values)
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError("Sweep-axis values must be finite and nonempty")
        if len(set(values)) != len(values):
            raise ValueError("Sweep-axis values must be unique")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class SweepPoint:
    index: int
    coordinates: Mapping[str, float]
    state_payload: Mapping[str, object]
    state_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinates", freeze_json(self.coordinates))
        object.__setattr__(self, "state_payload", freeze_json(self.state_payload))


@dataclass(frozen=True, slots=True)
class ParameterSweep:
    recipe_digest: str
    axes: tuple[SweepAxis, ...]
    points: tuple[SweepPoint, ...]


def plan_parameter_sweep(
    recipe: DesignRecipe,
    axes: Sequence[SweepAxis],
    *,
    maximum_points: int = 256,
) -> ParameterSweep:
    """Create a deterministic Cartesian sweep without running calculations."""

    axes = tuple(axes)
    if not axes:
        raise ValueError("A parameter sweep needs at least one axis")
    paths = [axis.path for axis in axes]
    if len(set(paths)) != len(paths):
        raise ValueError("A parameter sweep cannot repeat an axis path")
    base_payload = recipe.state_payload
    for path in paths:
        validate_runtime_sweep_path(base_payload, path)
    count = math.prod(len(axis.values) for axis in axes)
    if count > int(maximum_points):
        raise ValueError(
            f"Parameter sweep has {count} points; limit is {maximum_points}"
        )
    points = []
    for index, coordinates in enumerate(
        itertools.product(*(axis.values for axis in axes))
    ):
        payload = thaw_json(recipe.state_payload)
        coordinate_map = dict(zip(paths, coordinates, strict=True))
        for path, value in coordinate_map.items():
            payload = replace_parameter(payload, path, value)
        points.append(SweepPoint(
            index=index,
            coordinates=coordinate_map,
            state_payload=payload,
            state_digest=json_digest(payload),
        ))
    return ParameterSweep(recipe.digest, axes, tuple(points))


@dataclass(frozen=True, slots=True)
class MetricObservation:
    point_index: int
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        converted = {str(key): float(value) for key, value in self.metrics.items()}
        if any(not math.isfinite(value) for value in converted.values()):
            raise ValueError("Observed metrics must be finite")
        object.__setattr__(self, "metrics", MappingProxyType(converted))


@dataclass(frozen=True, slots=True)
class SensitivityEstimate:
    metric: str
    parameter_path: str
    derivative: float
    r_squared: float
    sample_count: int


def estimate_sensitivities(
    sweep: ParameterSweep,
    observations: Sequence[MetricObservation],
) -> tuple[SensitivityEstimate, ...]:
    """Fit local first-order derivatives for every common output metric."""

    by_index = {observation.point_index: observation for observation in observations}
    if len(by_index) != len(observations):
        raise ValueError("Each sweep point can have only one observation")
    selected = [
        (point, by_index[point.index])
        for point in sweep.points
        if point.index in by_index
    ]
    if len(selected) < len(sweep.axes) + 1:
        raise ValueError("Too few observations for the requested sensitivity fit")
    common_metrics = set.intersection(*(
        set(observation.metrics) for _point, observation in selected
    ))
    if not common_metrics:
        raise ValueError("Observations have no common metric")
    x = np.asarray([
        [point.coordinates[axis.path] for axis in sweep.axes]
        for point, _observation in selected
    ], dtype=float)
    centre = np.mean(x, axis=0)
    scale = np.ptp(x, axis=0)
    if np.any(scale <= 0.0):
        raise ValueError("Every sensitivity axis needs at least two values")
    design = np.column_stack((np.ones(x.shape[0]), (x - centre) / scale))
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Sensitivity sweep is rank deficient")
    estimates = []
    for metric in sorted(common_metrics):
        y = np.asarray([
            observation.metrics[metric]
            for _point, observation in selected
        ], dtype=float)
        coefficients, _residuals, _rank, _singular = np.linalg.lstsq(
            design, y, rcond=None
        )
        fitted = design @ coefficients
        total = float(np.sum((y - np.mean(y)) ** 2))
        residual = float(np.sum((y - fitted) ** 2))
        r_squared = 1.0 if total == 0.0 and residual == 0.0 else (
            1.0 - residual / total if total > 0.0 else 0.0
        )
        for axis_index, axis in enumerate(sweep.axes):
            estimates.append(SensitivityEstimate(
                metric=metric,
                parameter_path=axis.path,
                derivative=float(coefficients[axis_index + 1] / scale[axis_index]),
                r_squared=r_squared,
                sample_count=len(selected),
            ))
    return tuple(estimates)


@dataclass(frozen=True, slots=True)
class ToleranceRule:
    metric: str
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if self.minimum is None and self.maximum is None:
            raise ValueError("A tolerance needs a minimum or maximum")
        if self.minimum is not None and not math.isfinite(float(self.minimum)):
            raise ValueError("Tolerance minimum must be finite")
        if self.maximum is not None and not math.isfinite(float(self.maximum)):
            raise ValueError("Tolerance maximum must be finite")
        if (
            self.minimum is not None
            and self.maximum is not None
            and float(self.minimum) > float(self.maximum)
        ):
            raise ValueError("Tolerance minimum cannot exceed maximum")


@dataclass(frozen=True, slots=True)
class ToleranceResult:
    metric: str
    value: float | None
    passed: bool
    reason: str


def evaluate_tolerances(
    metrics: Mapping[str, float], rules: Sequence[ToleranceRule]
) -> tuple[ToleranceResult, ...]:
    rows = []
    for rule in rules:
        if rule.metric not in metrics:
            rows.append(ToleranceResult(
                rule.metric, None, False, "Metric was not calculated"
            ))
            continue
        value = float(metrics[rule.metric])
        if not math.isfinite(value):
            rows.append(ToleranceResult(
                rule.metric, value, False, "Metric is not finite"
            ))
            continue
        passed = (
            (rule.minimum is None or value >= float(rule.minimum))
            and (rule.maximum is None or value <= float(rule.maximum))
        )
        rows.append(ToleranceResult(
            rule.metric,
            value,
            passed,
            "Within tolerance" if passed else "Outside tolerance",
        ))
    return tuple(rows)


__all__ = (
    "DESIGN_RECIPE_SCHEMA_VERSION",
    "DesignRecipe",
    "MetricObservation",
    "ParameterEdit",
    "ParameterSweep",
    "SensitivityEstimate",
    "StateHistory",
    "SweepAxis",
    "SweepPoint",
    "ToleranceResult",
    "ToleranceRule",
    "estimate_sensitivities",
    "evaluate_tolerances",
    "load_recipe",
    "parameter_tokens",
    "parameter_value",
    "plan_parameter_sweep",
    "recipe_from_snapshot",
    "replace_parameter",
    "save_recipe",
    "validate_runtime_sweep_path",
)
