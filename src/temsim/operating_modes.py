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
    applies_to_modes: tuple[str, ...]
    state_parameters: tuple[str, ...]

    @property
    def definition_id(self) -> str:
        if self.observable == "sample_current_weighted_95_percent_semi_angle":
            return "chief-ray-current-contained-semiangle-95-v1"
        if self.observable == "sample_current_weighted_95_percent_diameter":
            return "chief-ray-current-contained-diameter-95-v1"
        return self.observable + "-v1"

    @property
    def active_mode_keys(self) -> tuple[str, ...]:
        return self.applies_to_modes or (self.mode_key,)


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
        condenser_is_retained = self.condenser.calibration_status.startswith(
            "retained_not_recomputed_"
        )
        convergence = self.condenser.targets.get(
            "achieved_convergence_sem_angle_mrad"
        )
        relay_um = self.projector.targets.get("achieved_relay_error_um")
        details = []
        if convergence is not None:
            label = (
                "stored reference semi-angle"
                if condenser_is_retained
                else "sample semi-angle"
            )
            details.append(f"{label} {float(convergence):.3f} mrad")
        if relay_um is not None:
            details.append(f"conjugate residual {float(relay_um):.3f} µm")
        if condenser_is_retained:
            details.append(
                "condenser preset not recalculated for current geometry"
            )
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
            if "radius_mm" in values:
                raise ValueError(
                    f"{path}: {mode.key}.{key} must use diameter_mm; "
                    "radius_mm is an internal legacy representation"
                )
            if "diameter_mm" not in values:
                raise ValueError(
                    f"{path}: {mode.key}.{key} must define diameter_mm"
                )
            if float(values["diameter_mm"]) <= 0.0:
                raise ValueError(
                    f"{path}: {mode.key}.{key}.diameter_mm must be positive"
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
            applies_to_modes=tuple(
                str(value)
                for value in item.get("applies_to_modes", ())
            ),
            state_parameters=tuple(
                str(value)
                for value in item.get("state_parameters", ())
            ),
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
            "objective_lens", "diffraction_lens", "intermediate_lens",
            "projector_lens_1", "projector_lens_2",
        ),
        "diffraction_camera_length": (
            "diffraction_lens", "intermediate_lens",
            "projector_lens_1", "projector_lens_2",
        ),
    }
    expected_state_parameters = {
        "nanoprobe_convergence": (),
        "microprobe_illumination": (),
        "image_magnification": (),
        "diffraction_camera_length": (),
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
        unknown_modes = set(definition.active_mode_keys) - known_modes
        if unknown_modes:
            raise ValueError(
                f"{path}: {definition.key} references unknown active modes"
            )
        if any(
            next(item for item in modes if item.key == mode_key).family
            != definition.family
            for mode_key in definition.active_mode_keys
        ):
            raise ValueError(
                f"{path}: {definition.key} active modes do not match its family"
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
        if (
            definition.state_parameters
            != expected_state_parameters[definition.key]
        ):
            raise ValueError(
                f"{path}: {definition.key} has an invalid state parameter set"
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


def _apply_operating_mode_pair(
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
    # A physical TEM screen/camera path and the inserted STEM detector stack
    # are mutually exclusive presets.  In TEM, retract annular/disk STEM
    # detectors before wave propagation; in STEM, retract the fluorescent
    # screen and Camera so they cannot stop rays between detector channels.
    if state.illumination_mode == "TEM":
        for detector in getattr(state, "stem_detectors", ()):
            detector.inserted = False
            detector.readout_enabled = False
        if not (
            bool(getattr(state.fluorescent_screen, "inserted", False))
            or bool(getattr(state.camera, "inserted", False))
        ):
            # The Camera branch owns the complete configured camera-length
            # range.  Keep a real recording target after a STEM -> TEM switch;
            # users can insert the FluScreen instead and re-run alignment for
            # that upstream plane.
            state.camera.inserted = True
    else:
        for detector in getattr(state, "stem_detectors", ()):
            detector.inserted = True
            detector.readout_enabled = True
        state.fluorescent_screen.inserted = False
        state.camera.inserted = False
    changed = _apply_values(state, condenser) + _apply_values(state, projector)
    if not bool(getattr(state, "monochromator_installed", False)):
        state.condenser_aperture_3.radius_mm = (
            state.condenser_aperture_3.maximum_radius_mm
        )

    state.electron_gun.electrostatic_lens.voltage_kv = 1.2
    state.sync_objective()
    if bool(getattr(getattr(state, "nanopulser", None), "installed", False)):
        from temsim.optics.condenser_recalibration import (
            recalibrate_nanopulser_condenser,
        )
        condenser = recalibrate_nanopulser_condenser(state, condenser, catalog)
    return AppliedOperatingModes(condenser, projector, changed)


def apply_operating_mode_pair(
    state,
    condenser_key: str,
    projector_key: str,
    *,
    column_name: str | None = None,
    recording_name: str | None = None,
    catalog: OperatingModeCatalog | None = None,
) -> AppliedOperatingModes:
    """Apply presets; an installed NanoPulser requires a validated live solve.

    The additional gun-to-C1 distance changes the incident beam.  Do not leave
    a partially applied pair when its recalculation cannot meet the requested
    sample illumination and focus constraints.
    """
    arguments = dict(
        column_name=column_name, recording_name=recording_name, catalog=catalog,
    )
    if not bool(getattr(getattr(state, "nanopulser", None), "installed", False)):
        return _apply_operating_mode_pair(
            state, condenser_key, projector_key, **arguments
        )

    from temsim.runtime_parameters import runtime_targets

    targets = runtime_targets(state)
    saved = []
    for key in (condenser_key, projector_key):
        definition = mode_by_key(key, catalog)
        for group in (definition.devices, definition.apertures):
            for device_key, values in group.items():
                obj = targets[device_key].obj
                for field in values:
                    field = {"field_polarity": "polarity"}.get(field, field)
                    saved.append((obj, field, getattr(obj, field)))
    saved.extend((
        (state, "illumination_mode", state.illumination_mode),
        (state, "projector_mode", state.projector_mode),
        (state.electron_gun.electrostatic_lens, "voltage_kv",
         state.electron_gun.electrostatic_lens.voltage_kv),
        (state.condenser_aperture_3, "radius_mm", state.condenser_aperture_3.radius_mm),
    ))
    for device in (*getattr(state, "stem_detectors", ()),
                   state.fluorescent_screen, state.camera):
        for field in ("inserted", "readout_enabled"):
            if hasattr(device, field):
                saved.append((device, field, getattr(device, field)))
    try:
        return _apply_operating_mode_pair(
            state, condenser_key, projector_key, **arguments
        )
    except Exception:
        for obj, field, value in reversed(saved):
            setattr(obj, field, value)
        state.sync_objective()
        raise
