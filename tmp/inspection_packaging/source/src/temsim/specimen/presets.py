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
class AtomisticCrystal:
    """TOML-owned equilibrium crystal and thermal-displacement metadata."""

    generator: str
    chemical_symbol: str
    atomic_number: int
    crystal_structure: str
    lattice_constant_angstrom: float
    zone_axis: tuple[int, int, int]
    thermal_sigma_angstrom: float
    thermal_sigma_reference: str


@dataclass(frozen=True)
class InelasticMaterial:
    """Material-owned inputs for the real-specimen energy-loss model.

    The two mean free paths are experimental anchors at
    ``reference_energy_kev``.  Their inverse difference is the non-plasmon
    (predominantly core-loss/ionisation) rate.  A representative loss energy
    is used for chromatic transport and for the compact ray-diagram
    quadrature; it is not an EELS line-shape model.
    """

    reference_energy_kev: float
    total_mean_free_path_nm: float
    plasmon_mean_free_path_nm: float
    plasmon_energy_ev: float
    ionisation_energy_ev: float
    collection_semiangle_mrad: float
    reference: str
    applicability: str


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
    atomistic: AtomisticCrystal | None
    inelastic: InelasticMaterial | None
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
    atomistic_data = data.get("atomistic")
    atomistic = None
    if atomistic_data is not None:
        zone_axis = tuple(int(value) for value in atomistic_data["zone_axis"])
        atomistic = AtomisticCrystal(
            generator=str(atomistic_data["generator"]),
            chemical_symbol=str(atomistic_data["chemical_symbol"]),
            atomic_number=int(atomistic_data["atomic_number"]),
            crystal_structure=str(atomistic_data["crystal_structure"]),
            lattice_constant_angstrom=float(
                atomistic_data["lattice_constant_angstrom"]
            ),
            zone_axis=zone_axis,
            thermal_sigma_angstrom=float(
                atomistic_data.get("thermal_sigma_angstrom", 0.0)
            ),
            thermal_sigma_reference=str(
                atomistic_data.get("thermal_sigma_reference", "")
            ),
        )
    inelastic_data = data.get("inelastic")
    inelastic = None
    if inelastic_data is not None:
        inelastic = InelasticMaterial(
            reference_energy_kev=float(
                inelastic_data.get("reference_energy_kev", 200.0)
            ),
            total_mean_free_path_nm=float(
                inelastic_data["total_mean_free_path_nm"]
            ),
            plasmon_mean_free_path_nm=float(
                inelastic_data["plasmon_mean_free_path_nm"]
            ),
            plasmon_energy_ev=float(inelastic_data["plasmon_energy_ev"]),
            ionisation_energy_ev=float(
                inelastic_data["ionisation_energy_ev"]
            ),
            collection_semiangle_mrad=float(
                inelastic_data.get("collection_semiangle_mrad", 20.0)
            ),
            reference=str(inelastic_data.get("reference", "")),
            applicability=str(inelastic_data.get("applicability", "")),
        )
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
        atomistic=atomistic,
        inelastic=inelastic,
        source_path=path,
    )
    if preset.reference_thickness_nm <= 0.0:
        raise ValueError(f"{path}: reference_thickness_nm must be positive.")
    if min(preset.unit_cell_x_angstrom, preset.unit_cell_y_angstrom) <= 0.0:
        raise ValueError(f"{path}: unit-cell dimensions must be positive.")
    if preset.field_of_view_angstrom <= 0.0 or preset.pixels < 32:
        raise ValueError(f"{path}: wave grid is invalid.")
    if preset.atomistic is not None:
        crystal = preset.atomistic
        if crystal.generator != "ase_bulk_surface":
            raise ValueError(f"{path}: unsupported atomistic generator.")
        if not crystal.chemical_symbol.strip():
            raise ValueError(f"{path}: atomistic chemical symbol is empty.")
        if not 1 <= crystal.atomic_number <= 118:
            raise ValueError(f"{path}: atomistic atomic number is invalid.")
        if crystal.lattice_constant_angstrom <= 0.0:
            raise ValueError(f"{path}: atomistic lattice constant must be positive.")
        if len(crystal.zone_axis) != 3 or not any(crystal.zone_axis):
            raise ValueError(f"{path}: atomistic zone axis must be a nonzero triplet.")
        if crystal.thermal_sigma_angstrom < 0.0:
            raise ValueError(f"{path}: thermal sigma cannot be negative.")
        if (
            crystal.thermal_sigma_angstrom > 0.0
            and not crystal.thermal_sigma_reference.strip()
        ):
            raise ValueError(
                f"{path}: a nonzero thermal sigma requires a source reference."
            )
    if preset.inelastic is not None:
        material = preset.inelastic
        positive_values = (
            material.reference_energy_kev,
            material.total_mean_free_path_nm,
            material.plasmon_mean_free_path_nm,
            material.plasmon_energy_ev,
            material.ionisation_energy_ev,
            material.collection_semiangle_mrad,
        )
        if any(value <= 0.0 for value in positive_values):
            raise ValueError(f"{path}: inelastic material values must be positive.")
        if (
            material.total_mean_free_path_nm
            > material.plasmon_mean_free_path_nm + 1.0e-12
        ):
            raise ValueError(
                f"{path}: total inelastic MFP cannot exceed the plasmon-component MFP."
            )
        if not material.reference.strip():
            raise ValueError(f"{path}: inelastic material data need a reference.")
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
