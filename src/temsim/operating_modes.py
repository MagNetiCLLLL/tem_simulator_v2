"""Read assembly-aware operating-mode definitions without applying them."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import tomllib

from temsim.paths import OPERATING_MODE_CONFIG_ROOT


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
class OperatingModeCatalog:
    modes: tuple[OperatingModeDefinition, ...]
    crossover_constraints: tuple[CrossoverConstraint, ...]
    source_path: Path


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
            if "percent" in values and not (
                0.0 <= float(values["percent"]) <= 100.0
            ):
                raise ValueError(f"{path}: {mode.key}.{key} exceeds 100%")
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
    return OperatingModeCatalog(modes, constraints, path)
