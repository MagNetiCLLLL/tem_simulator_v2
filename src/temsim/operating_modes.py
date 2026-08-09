"""Load and apply assembly-aware condenser/projector operating modes."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import tomllib

from temsim.paths import OPERATING_MODE_CONFIG_ROOT
from temsim.optics.excitation_policy import is_saturated_excitation


@dataclass(frozen=True)
class OperatingModeDefinition:
    key: str
    name: str
    family: str
    calibration_status: str
    compatible_columns: tuple[str, ...]
    compatible_recording_systems: tuple[str, ...]
    devices: dict[str, dict[str, object]]
    apertures: dict[str, dict[str, object]]
    targets: dict[str, object]
    calibration_reference: str


@dataclass(frozen=True)
class CrossoverConstraint:
    key: str
    upstream_lens: str
    downstream_lens: str
    target_z_source: str
    applies_to_modes: tuple[str, ...]
    status: str
    note: str


@dataclass(frozen=True)
class DirectAlignmentDefinition:
    """One user-level coupled adjustment backed by a live optical solve."""

    key: str
    name: str
    family: str
    mode_key: str
    unit: str
    minimum: float
    maximum: float
    default_value: float
    devices: tuple[str, ...]
    observable: str
    constraint: str
    calibration_status: str
    calibration_reference: str
    targets: dict[str, object]


@dataclass(frozen=True)
class OperatingModeCatalog:
    modes: tuple[OperatingModeDefinition, ...]
    crossover_constraints: tuple[CrossoverConstraint, ...]
    direct_alignments: tuple[DirectAlignmentDefinition, ...]
    source_path: Path


@dataclass(frozen=True)
class AppliedOperatingModes:
    condenser: OperatingModeDefinition
    projector: OperatingModeDefinition
    changed_devices: tuple[str, ...]

    @property
    def summary(self) -> str:
        convergence = self.condenser.targets.get(
            "achieved_convergence_sem_angle_mrad"
        )
        relay_um = self.projector.targets.get("achieved_relay_error_um")
        details = []
        if convergence is not None:
            details.append(f"sample semi-angle {float(convergence):.3f} mrad")
        if relay_um is not None:
            details.append(f"conjugate residual {float(relay_um):.3f} µm")
        suffix = "; ".join(details)
        return (
            f"{self.condenser.name} + {self.projector.name}"
            + (f": {suffix}" if suffix else "")
        )


@lru_cache(maxsize=1)
def load_operating_mode_catalog() -> OperatingModeCatalog:
    """Load mode storage; this does not change the microscope state."""
    path = OPERATING_MODE_CONFIG_ROOT / "catalog.toml"
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    if int(document.get("format_version", 0)) != 1:
        raise ValueError(f"{path}: unsupported operating-mode format")

    modes = tuple(
        OperatingModeDefinition(
            key=str(item["key"]),
            name=str(item["name"]),
            family=str(item["family"]),
            calibration_status=str(item["calibration_status"]),
            compatible_columns=tuple(
                str(value) for value in item["compatible_columns"]
            ),
            compatible_recording_systems=tuple(
                str(value)
                for value in item["compatible_recording_systems"]
            ),
            devices={
                str(key): dict(value)
                for key, value in item.get("devices", {}).items()
            },
            apertures={
                str(key): dict(value)
                for key, value in item.get("apertures", {}).items()
            },
            targets={
                str(key): value
                for key, value in item.get("targets", {}).items()
            },
            calibration_reference=str(
                item.get("calibration_reference", "")
            ),
        )
        for item in document.get("modes", ())
    )
    mode_keys = [mode.key for mode in modes]
    if len(set(mode_keys)) != len(mode_keys):
        raise ValueError(f"{path}: duplicate operating-mode key")
    if {mode.family for mode in modes} - {"condenser", "projector"}:
        raise ValueError(f"{path}: unsupported operating-mode family")
    for mode in modes:
        for key, values in mode.devices.items():
            if "percent" in values:
                percent = float(values["percent"])
                if not 0.0 <= percent <= 100.0:
                    raise ValueError(
                        f"{path}: {mode.key}.{key} exceeds 100%"
                    )
                if is_saturated_excitation(percent):
                    raise ValueError(
                        f"{path}: {mode.key}.{key} reaches 100%; enlarge "
                        "the lens maximum field and rebase the default into "
                        "the 30-70% operating window"
                    )
        for key, values in mode.apertures.items():
            for field in ("diameter_mm", "radius_mm"):
                if field in values and float(values[field]) <= 0.0:
                    raise ValueError(
                        f"{path}: {mode.key}.{key}.{field} must be positive"
                    )

    constraints = tuple(
        CrossoverConstraint(
            key=str(item["key"]),
            upstream_lens=str(item["upstream_lens"]),
            downstream_lens=str(item["downstream_lens"]),
            target_z_source=str(item["target_z_source"]),
            applies_to_modes=tuple(
                str(value) for value in item.get("applies_to_modes", ())
            ),
            status=str(item.get("status", "pending")),
            note=str(item.get("note", "")),
        )
        for item in document.get("crossover_constraints", ())
    )
    known_modes = set(mode_keys)
    for constraint in constraints:
        unknown = set(constraint.applies_to_modes) - known_modes
        if unknown:
            raise ValueError(
                f"{path}: crossover {constraint.key} references unknown modes"
            )

    direct_alignments = tuple(
        DirectAlignmentDefinition(
            key=str(item["key"]),
            name=str(item["name"]),
            family=str(item["family"]),
            mode_key=str(item["mode_key"]),
            unit=str(item["unit"]),
            minimum=float(item["minimum"]),
            maximum=float(item["maximum"]),
            default_value=float(item["default_value"]),
            devices=tuple(str(value) for value in item["devices"]),
            observable=str(item["observable"]),
            constraint=str(item["constraint"]),
            calibration_status=str(item["calibration_status"]),
            calibration_reference=str(item["calibration_reference"]),
            targets={
                str(key): value
                for key, value in item.get("targets", {}).items()
            },
        )
        for item in document.get("direct_alignments", ())
    )
    direct_keys = [definition.key for definition in direct_alignments]
    if len(set(direct_keys)) != len(direct_keys):
        raise ValueError(f"{path}: duplicate direct-alignment key")
    expected_direct_keys = {
        "nanoprobe_convergence",
        "microprobe_illumination",
        "image_magnification",
        "diffraction_camera_length",
    }
    if set(direct_keys) != expected_direct_keys:
        raise ValueError(
            f"{path}: direct alignments must define exactly "
            + ", ".join(sorted(expected_direct_keys))
        )
    expected_devices = {
        "nanoprobe_convergence": (
            "condenser_lens_2", "condenser_lens_3",
        ),
        "microprobe_illumination": (
            "condenser_lens_2", "condenser_lens_3",
        ),
        "image_magnification": (
            "diffraction_lens", "intermediate_lens",
            "projector_lens_1", "projector_lens_2",
        ),
        "diffraction_camera_length": (
            "diffraction_lens", "intermediate_lens",
            "projector_lens_1", "projector_lens_2",
        ),
    }
    for definition in direct_alignments:
        if definition.family not in {"condenser", "projector"}:
            raise ValueError(
                f"{path}: {definition.key} has unsupported family"
            )
        if definition.mode_key not in known_modes:
            raise ValueError(
                f"{path}: {definition.key} references unknown mode "
                f"{definition.mode_key!r}"
            )
        mode = next(item for item in modes if item.key == definition.mode_key)
        if mode.family != definition.family:
            raise ValueError(
                f"{path}: {definition.key} family does not match its mode"
            )
        if not (
            0.0 < definition.minimum
            <= definition.default_value
            <= definition.maximum
        ):
            raise ValueError(
                f"{path}: {definition.key} has an invalid target range"
            )
        if definition.devices != expected_devices[definition.key]:
            raise ValueError(
                f"{path}: {definition.key} must couple "
                + ", ".join(expected_devices[definition.key])
            )
        if not definition.unit.strip():
            raise ValueError(f"{path}: {definition.key} unit is empty")
        if not definition.calibration_status.strip():
            raise ValueError(
                f"{path}: {definition.key} calibration status is empty"
            )
        if not definition.calibration_reference.strip():
            raise ValueError(
                f"{path}: {definition.key} calibration reference is empty"
            )
    return OperatingModeCatalog(
        modes, constraints, direct_alignments, path
    )


def mode_by_key(
    key: str, catalog: OperatingModeCatalog | None = None
) -> OperatingModeDefinition:
    catalog = catalog or load_operating_mode_catalog()
    try:
        return next(mode for mode in catalog.modes if mode.key == str(key))
    except StopIteration as exc:
        choices = ", ".join(mode.key for mode in catalog.modes)
        raise KeyError(f"Unknown operating mode {key!r}; choices: {choices}") from exc


def direct_alignment_by_key(
    key: str, catalog: OperatingModeCatalog | None = None
) -> DirectAlignmentDefinition:
    catalog = catalog or load_operating_mode_catalog()
    try:
        return next(
            definition
            for definition in catalog.direct_alignments
            if definition.key == str(key)
        )
    except StopIteration as exc:
        choices = ", ".join(
            definition.key for definition in catalog.direct_alignments
        )
        raise KeyError(
            f"Unknown direct alignment {key!r}; choices: {choices}"
        ) from exc


def compatible_modes(
    family: str,
    column_name: str,
    recording_name: str,
    catalog: OperatingModeCatalog | None = None,
) -> tuple[OperatingModeDefinition, ...]:
    """Return catalog modes that can be shown for one loaded assembly."""

    catalog = catalog or load_operating_mode_catalog()

    def matches(values: tuple[str, ...], selected: str) -> bool:
        return "*" in values or selected in values

    return tuple(
        mode
        for mode in catalog.modes
        if mode.family == family
        and matches(mode.compatible_columns, column_name)
        and matches(mode.compatible_recording_systems, recording_name)
    )


def _apply_values(state, mode: OperatingModeDefinition) -> tuple[str, ...]:
    # Imported lazily to keep catalog parsing independent of the runtime GUI.
    from temsim.runtime_parameters import (
        runtime_targets,
        validate_runtime_assignment,
    )

    targets = runtime_targets(state)
    changed = []
    for group in (mode.devices, mode.apertures):
        for key, values in group.items():
            try:
                target = targets[key]
            except KeyError as exc:
                raise ValueError(
                    f"Operating mode {mode.key!r} references missing device {key!r}"
                ) from exc
            for field, raw_value in values.items():
                runtime_field = {
                    "field_polarity": "polarity",
                }.get(field, field)
                if not hasattr(target.obj, runtime_field):
                    raise ValueError(
                        f"Operating mode {mode.key!r}: {key}.{field} does not exist"
                    )
                value = validate_runtime_assignment(
                    target, runtime_field, raw_value
                )
                setattr(target.obj, runtime_field, value)
            changed.append(key)
    return tuple(changed)


def apply_operating_mode_pair(
    state,
    condenser_key: str,
    projector_key: str,
    *,
    column_name: str | None = None,
    recording_name: str | None = None,
    catalog: OperatingModeCatalog | None = None,
) -> AppliedOperatingModes:
    """Apply one independently selectable illumination/projector preset pair."""

    catalog = catalog or load_operating_mode_catalog()
    condenser = mode_by_key(condenser_key, catalog)
    projector = mode_by_key(projector_key, catalog)
    if condenser.family != "condenser" or projector.family != "projector":
        raise ValueError("A mode pair requires one condenser and one projector mode")

    def require_compatible(mode: OperatingModeDefinition) -> None:
        if (
            column_name is not None
            and "*" not in mode.compatible_columns
            and column_name not in mode.compatible_columns
        ):
            raise ValueError(f"{mode.name} is not calibrated for {column_name}")
        if (
            recording_name is not None
            and "*" not in mode.compatible_recording_systems
            and recording_name not in mode.compatible_recording_systems
        ):
            raise ValueError(f"{mode.name} is not calibrated for {recording_name}")

    require_compatible(condenser)
    require_compatible(projector)

    state.illumination_mode = {
        "micro_probe": "TEM",
        "nano_probe": "STEM",
    }[condenser.key]
    state.projector_mode = {
        "imaging": "image",
        "diffraction": "diffraction",
    }[projector.key]
    changed = _apply_values(state, condenser) + _apply_values(state, projector)
    if not bool(getattr(state, "monochromator_installed", False)):
        state.condenser_aperture_3.radius_mm = (
            state.condenser_aperture_3.maximum_radius_mm
        )

    state.electron_gun.electrostatic_lens.voltage_kv = 1.2
    state.sync_objective()
    return AppliedOperatingModes(condenser, projector, changed)
