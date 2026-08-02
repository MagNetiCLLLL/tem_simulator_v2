"""Load analytic projected-column specimen definitions from TOML.

The Python code owns only the generic schema and validation.  All concrete
materials, lattice dimensions, atom-column positions and projected strengths
live in ``specimen_presets/*.toml``.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import tomllib

from temsim.paths import SPECIMEN_CONFIG_ROOT

PRESET_DIRECTORY = SPECIMEN_CONFIG_ROOT


@dataclass(frozen=True)
class SpecimenColumn:
    x_fraction: float
    y_fraction: float
    atomic_number: int
    occupancy: float
    sigma_angstrom: float
    projected_potential_v_angstrom: float


@dataclass(frozen=True)
class SpecimenPreset:
    key: str
    name: str
    description: str
    zone_axis: str
    reference_thickness_nm: float
    unit_cell_x_angstrom: float
    unit_cell_y_angstrom: float
    field_of_view_angstrom: float
    pixels: int
    columns: tuple[SpecimenColumn, ...]
    source_path: Path


def _preset_paths() -> tuple[Path, ...]:
    return tuple(sorted(PRESET_DIRECTORY.glob("*.toml")))


@lru_cache(maxsize=1)
def _preset_index() -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in _preset_paths():
        with path.open("rb") as stream:
            key = str(tomllib.load(stream).get("key", "")).strip()
        if not key:
            raise ValueError(f"Specimen preset {path} has no key.")
        if key in result:
            raise ValueError(f"Duplicate specimen preset key: {key}")
        result[key] = path
    return result


def available_specimen_presets() -> tuple[tuple[str, str], ...]:
    """Return ``(key, display name)`` pairs in file-name order."""
    items = []
    for key, path in _preset_index().items():
        with path.open("rb") as stream:
            data = tomllib.load(stream)
        items.append((key, str(data.get("name", key))))
    return tuple(items)


def default_specimen_preset_key() -> str:
    """Read the single TOML preset marked as the application default."""
    defaults = []
    for key, path in _preset_index().items():
        with path.open("rb") as stream:
            if bool(tomllib.load(stream).get("default", False)):
                defaults.append(key)
    if len(defaults) != 1:
        raise ValueError(
            "Exactly one specimen preset TOML must set default = true."
        )
    return defaults[0]


@lru_cache(maxsize=None)
def load_specimen_preset(key: str) -> SpecimenPreset:
    try:
        path = _preset_index()[str(key)]
    except KeyError as exc:
        choices = ", ".join(_preset_index())
        raise ValueError(
            f"Unknown specimen preset {key!r}; available presets: {choices}"
        ) from exc
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    cell = data.get("unit_cell", {})
    grid = data.get("grid", {})
    columns = tuple(
        SpecimenColumn(
            x_fraction=float(item["x_fraction"]),
            y_fraction=float(item["y_fraction"]),
            atomic_number=int(item["atomic_number"]),
            occupancy=float(item.get("occupancy", 1.0)),
            sigma_angstrom=float(item["sigma_angstrom"]),
            projected_potential_v_angstrom=float(
                item["projected_potential_v_angstrom"]
            ),
        )
        for item in data.get("columns", [])
    )
    preset = SpecimenPreset(
        key=str(data["key"]),
        name=str(data["name"]),
        description=str(data.get("description", "")),
        zone_axis=str(data.get("zone_axis", "")),
        reference_thickness_nm=float(data["reference_thickness_nm"]),
        unit_cell_x_angstrom=float(cell["x_angstrom"]),
        unit_cell_y_angstrom=float(cell["y_angstrom"]),
        field_of_view_angstrom=float(grid["field_of_view_angstrom"]),
        pixels=int(grid["pixels"]),
        columns=columns,
        source_path=path,
    )
    if preset.reference_thickness_nm <= 0.0:
        raise ValueError(f"{path}: reference_thickness_nm must be positive.")
    if min(preset.unit_cell_x_angstrom, preset.unit_cell_y_angstrom) <= 0.0:
        raise ValueError(f"{path}: unit-cell dimensions must be positive.")
    if preset.field_of_view_angstrom <= 0.0 or preset.pixels < 32:
        raise ValueError(f"{path}: wave grid is invalid.")
    for column in preset.columns:
        if not (0.0 <= column.x_fraction < 1.0):
            raise ValueError(f"{path}: column x_fraction must be in [0, 1).")
        if not (0.0 <= column.y_fraction < 1.0):
            raise ValueError(f"{path}: column y_fraction must be in [0, 1).")
        if column.atomic_number < 1 or column.occupancy < 0.0:
            raise ValueError(f"{path}: column atom/occupancy is invalid.")
        if column.sigma_angstrom <= 0.0:
            raise ValueError(f"{path}: column sigma must be positive.")
    return preset
