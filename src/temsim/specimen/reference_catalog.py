"""Discover real reference structures and their explicit modelling assumptions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib

from temsim.paths import CONFIG_ROOT

REFERENCE_DIRECTORY = CONFIG_ROOT / "reference_samples"
DEFAULT_REFERENCE_KEY = "si_110"


@dataclass(frozen=True)
class ReferenceSample:
    key: str
    name: str
    cif_path: Path
    zone_axis: tuple[int, int, int]
    in_plane_axis: tuple[int, int, int]
    thermal_sigma_angstrom: float
    description: str
    template_preset_key: str
    inelastic_preset_key: str = ""
    chemical_symbol: str = ""
    thermal_source: str = ""
    metadata_path: Path | None = None


def _axis(data, key, default):
    value = data.get(key, default)
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(isinstance(v, bool) or not isinstance(v, int) for v in value)
            or not any(value)):
        raise ValueError(f"Reference {key} must contain three integers and be nonzero.")
    return tuple(value)


def available_reference_samples() -> tuple[ReferenceSample, ...]:
    entries = []
    keys = set()
    for path in sorted(REFERENCE_DIRECTORY.glob("*"), key=lambda p: p.name.lower()):
        if path.suffix.lower() not in {".cif", ".mcif"} or not path.is_file():
            continue
        metadata_path = path.with_suffix(".toml")
        data = tomllib.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        key = str(data.get("key", path.stem)).strip()
        if not key or key in keys:
            raise ValueError(f"Duplicate or empty reference sample key: {key!r}")
        keys.add(key)
        sigma = float(data.get("thermal_sigma_angstrom", 0.0))
        thermal_source = str(data.get("thermal_source", "")).strip()
        if not math.isfinite(sigma) or sigma < 0 or (sigma > 0 and not thermal_source):
            raise ValueError(f"Reference {key}: positive thermal RMS requires its source.")
        entries.append(ReferenceSample(
            key=key, name=str(data.get("name", path.stem)), cif_path=path.resolve(),
            zone_axis=_axis(data, "zone_axis", (0, 0, 1)),
            in_plane_axis=_axis(data, "in_plane_axis", (1, 0, 0)),
            thermal_sigma_angstrom=sigma, description=str(data.get("description", "")),
            template_preset_key=str(data.get("template_preset_key", "si_110")),
            inelastic_preset_key=str(data.get("inelastic_preset_key", "")),
            chemical_symbol=str(data.get("chemical_symbol", "")), thermal_source=thermal_source,
            metadata_path=metadata_path.resolve() if metadata_path.exists() else None,
        ))
    return tuple(entries)


def refresh_reference_samples() -> tuple[ReferenceSample, ...]:
    return available_reference_samples()


def get_reference_sample(key: str) -> ReferenceSample:
    for entry in available_reference_samples():
        if entry.key == key:
            return entry
    raise ValueError(f"Reference sample {key!r} is unavailable. Add its CIF to {REFERENCE_DIRECTORY} and Refresh.")


def apply_reference_sample(sample, key: str) -> ReferenceSample:
    from temsim.specimen.cif_io import read_cif_atoms
    from temsim.specimen.geometry import quaternion_from_zone_axes, set_sample_orientation
    from temsim.specimen.rutherford import composition_from_atoms

    entry = get_reference_sample(key)
    unit = read_cif_atoms(str(entry.cif_path), index=0)
    composition = composition_from_atoms(unit, source_path=str(entry.cif_path))
    from ase.data import atomic_numbers
    if entry.chemical_symbol and {z for z, _ in composition.number_densities_atoms_nm3} != {atomic_numbers[entry.chemical_symbol]}:
        raise ValueError(f"Reference {key}: CIF composition does not match its declared {entry.chemical_symbol} material.")
    quaternion = quaternion_from_zone_axes(unit.cell.array, entry.zone_axis, entry.in_plane_axis)
    sample.specimen_mode = "reference"
    sample.reference_sample_key = entry.key
    sample.specimen_preset_key = entry.template_preset_key
    sample.zone_axis_uvw = entry.zone_axis
    sample.in_plane_axis_uvw = entry.in_plane_axis
    set_sample_orientation(sample, quaternion)
    return entry


def reference_thermal_sigma(sample) -> float:
    from temsim.specimen.source import specimen_is_vacuum
    explicit = float(getattr(sample, "wave_frozen_phonon_sigma_angstrom", 0.0))
    if specimen_is_vacuum(sample):
        return explicit
    if explicit > 0 or str(getattr(sample, "specimen_mode", "reference")).strip().lower() != "reference":
        return explicit
    return _thermal_reference(sample).thermal_sigma_angstrom


def reference_thermal_source(sample) -> str:
    from temsim.specimen.source import specimen_is_vacuum
    if specimen_is_vacuum(sample):
        return ""
    if (float(getattr(sample, "wave_frozen_phonon_sigma_angstrom", 0.0)) > 0
            or str(getattr(sample, "specimen_mode", "reference")).strip().lower() != "reference"):
        return ""
    return _thermal_reference(sample).thermal_source


def _thermal_reference(sample) -> ReferenceSample:
    from ase.data import atomic_numbers
    from temsim.specimen.rutherford import read_cif_composition
    entry = get_reference_sample(getattr(sample, "reference_sample_key", DEFAULT_REFERENCE_KEY))
    if entry.chemical_symbol:
        composition = read_cif_composition(entry.cif_path)
        if {z for z, _ in composition.atoms_per_cell} != {atomic_numbers[entry.chemical_symbol]}:
            raise ValueError(f"Reference {entry.key}: CIF composition no longer matches its declared {entry.chemical_symbol} thermal material.")
    return entry
