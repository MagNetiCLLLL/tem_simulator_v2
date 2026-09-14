"""Editable beam-path environments, separate from mechanical vacuum bores.

Pressure is in mbar, geometry in mm. Resolved boundaries follow the installed
hardware; a cell replaces the ambient medium only in its finite volume.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import tomllib

from temsim.paths import CONFIG_ROOT

VACUUM_SCHEMA = "classical-vacuum-map-v1"
DEFAULT_PATH = CONFIG_ROOT / "environments" / "vacuum_map.toml"


@dataclass
class Medium:
    phase: str = "gas"
    formula: str = "N2"
    pressure_mbar: float = 1e-7
    temperature_k: float = 293.0
    density_kg_m3: float = 1000.0
    # Optional measured removal cross section PER MOLECULE. Elastic scattering
    # is not removal and must never be counted again in this coefficient.
    removal_cross_section_m2: float = 0.0
    removal_reference: str = ""

    def validate(self):
        from ase.formula import Formula
        from ase.data import atomic_numbers
        if self.phase not in {"gas", "liquid", "vacuum"}:
            raise ValueError("Medium phase must be gas, liquid or vacuum")
        try:
            atoms = Formula(self.formula).count()
            valid = atoms and all(k in atomic_numbers and n > 0 for k, n in atoms.items())
        except (ValueError, KeyError):
            valid = False
        if not valid:
            raise ValueError(f"Invalid medium chemical formula: {self.formula}")
        for name in ("pressure_mbar", "temperature_k", "density_kg_m3", "removal_cross_section_m2"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Medium {name} must be finite and non-negative")
        if self.temperature_k <= 0 or self.density_kg_m3 <= 0:
            raise ValueError("Temperature and liquid mass density must be positive")
        if self.removal_cross_section_m2 and not self.removal_reference.strip():
            raise ValueError("A nonzero removal cross section needs a measurement/model reference")
        return self

    def number_density_m3(self):
        if self.phase == "vacuum":
            return 0.0
        from scipy.constants import Boltzmann, atomic_mass
        if self.phase == "gas":
            return self.pressure_mbar * 100.0 / (Boltzmann * self.temperature_k)
        from ase.formula import Formula
        from ase.data import atomic_masses, atomic_numbers
        mass = sum(atomic_masses[atomic_numbers[k]] * v for k, v in Formula(self.formula).count().items())
        return self.density_kg_m3 / (mass * atomic_mass)


@dataclass
class VacuumRegion:
    key: str
    name: str
    start_anchor: str
    end_anchor: str
    start_offset_mm: float = 0.0
    end_offset_mm: float = 0.0
    medium: Medium = field(default_factory=Medium)
    pressure_reference: str = "user-defined"


@dataclass
class SpecimenCell:
    inserted: bool = False
    diameter_mm: float = 0.01
    length_mm: float = 0.001
    offset_x_mm: float = 0.0
    offset_y_mm: float = 0.0
    offset_z_mm: float = 0.0
    medium: Medium = field(default_factory=lambda: Medium(pressure_mbar=1.0))


@dataclass
class VacuumMap:
    enabled: bool = True
    seed: int = 914
    schema: str = VACUUM_SCHEMA
    provenance: str = ""
    regions: list[VacuumRegion] = field(default_factory=list)
    cell: SpecimenCell = field(default_factory=SpecimenCell)
    max_optical_depth_per_step: float = 0.02
    max_transport_nodes: int = 2000000

    def validate(self):
        if self.schema != VACUUM_SCHEMA:
            raise ValueError(f"Unknown vacuum model: {self.schema}")
        if not isinstance(self.enabled, bool) or not isinstance(self.cell.inserted, bool):
            raise ValueError("Vacuum enabled and cell inserted must be booleans")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Vacuum random seed must be an integer from 0 to 4294967295")
        if not math.isfinite(self.max_optical_depth_per_step) or not 0 < self.max_optical_depth_per_step <= .05:
            raise ValueError("Vacuum maximum step optical depth must be in (0, 0.05]")
        if type(self.max_transport_nodes) is not int or not 2 <= self.max_transport_nodes <= 10000000:
            raise ValueError("Vacuum integration budget must be 2 to 10000000 nodes")
        keys = [r.key for r in self.regions]
        if any(not k or k == "specimen_cell" or k.startswith("transition:") for k in keys) or len(set(keys)) != len(keys):
            raise ValueError("Vacuum region keys must be nonempty and unique; specimen_cell and transition: are reserved")
        for r in self.regions:
            r.medium.validate()
            for value in (r.start_offset_mm, r.end_offset_mm):
                if not math.isfinite(value):
                    raise ValueError("Vacuum boundary offsets must be finite")
        self.cell.medium.validate()
        for key in ("diameter_mm", "length_mm", "offset_x_mm", "offset_y_mm", "offset_z_mm"):
            if not math.isfinite(getattr(self.cell, key)):
                raise ValueError(f"Cell {key} must be finite")
        if self.cell.diameter_mm <= 0 or self.cell.length_mm <= 0:
            raise ValueError("Cell dimensions must be positive")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("Vacuum map must be a table")
        data = dict(data)
        try:
            data["regions"] = [VacuumRegion(**{**r, "medium": Medium(**r.get("medium", {}))})
                               for r in data.get("regions", [])]
            c = data.get("cell", {})
            data["cell"] = SpecimenCell(**{**c, "medium": Medium(**c.get("medium", {}))})
            return cls(**data).validate()
        except (TypeError, AttributeError) as exc:
            raise ValueError(f"Invalid vacuum map fields: {exc}") from exc

    @classmethod
    def historical(cls):
        return cls(enabled=False, provenance="Historical profile: no residual-medium transport was specified")

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        with Path(path).open("rb") as stream:
            return cls.from_dict(tomllib.load(stream))

    def save(self, path):
        import os
        import tempfile
        import tomli_w
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=target.parent, prefix=target.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(tomli_w.dumps(self.to_dict()))
            os.replace(name, target)
        finally:
            Path(name).unlink(missing_ok=True)

    def signature(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class ResolvedMedium:
    key: str
    name: str
    start_z_mm: float
    end_z_mm: float
    medium: Medium
    radius_mm: float | None = None
    center_x_mm: float = 0.0
    center_y_mm: float = 0.0
    # Automatic gap: endpoint partial pressures are blended linearly in Z.
    # Different gases retain their endpoint temperatures and cross sections.
    end_medium: Medium | None = None


def fill_region_gaps(resolved):
    result = []
    for region in sorted(resolved, key=lambda r: r.start_z_mm):
        if result:
            left = result[-1]
            gap = region.start_z_mm-left.end_z_mm
            if gap < -1e-8:
                raise ValueError(f"Vacuum regions {left.name} / {region.name} overlap")
            if gap > 1e-8:
                if left.medium.phase == "liquid" or region.medium.phase == "liquid":
                    raise ValueError("A liquid interface needs an explicit boundary; automatic vacuum transitions support gas/vacuum only")
                result.append(ResolvedMedium(f"transition:{left.key}:{region.key}",
                    f"{left.name} → {region.name}", left.end_z_mm, region.start_z_mm,
                    left.medium, end_medium=region.medium))
        result.append(region)
    return result


@dataclass(frozen=True)
class ModuleAxialRange:
    key: str
    source_file: str
    origin_z_mm: float
    start_z_mm: float
    end_z_mm: float


def module_axial_ranges(state):
    """Use installed module transforms, not file-local coordinates or widths."""
    assembly = getattr(state, "_resolved_assembly", None)
    installed = {p.key: p for p in getattr(assembly, "parts", ())}
    result = []
    previous_exit = 0.0
    for module in getattr(assembly, "modules", ()):
        origin = next((installed[p.key].center_z_mm-p.center_z_mm
                       for p in module.parts if p.key in installed),
                      previous_exit-module.entrance_z_mm)
        start, end = origin+module.entrance_z_mm, origin+module.exit_z_mm
        result.append(ModuleAxialRange(module.key, module.source_file, origin, start, end))
        previous_exit = end
    return tuple(result)


def boundary_anchors(state):
    from temsim.component_keys import PROJECTION_CHAMBER_DPA_APERTURE
    gun = state.electron_gun
    # Accelerator entrance comes from the installed physical electrode object.
    anchors = {"axis_origin": 0.0, "source": 0.0, "gun_exit": float(gun.exit_plane_z_mm),
               "sample": float(state.sample.z_mm)}
    for module in module_axial_ranges(state):
        for suffix, z in (("origin", module.origin_z_mm), ("start", module.start_z_mm),
                          ("end", module.end_z_mm)):
            anchors[f"module:{module.key}.{suffix}"] = z
    assembly = getattr(state, "_resolved_assembly", None)
    for p in getattr(assembly, "parts", ()):
        for short, attr in (("start", "start_z_mm"), ("center", "center_z_mm"), ("end", "end_z_mm")):
            if hasattr(p, attr):
                anchors[f"{p.key}.{short}"] = float(getattr(p, attr))
    anchors["source"] = min(0., anchors.get("feg_tip.start", 0.))
    anchors["gun_acceleration_start"] = float(gun.accelerator.mechanical_center_from_tip_mm-gun.accelerator.mechanical_length_mm/2)
    dpa = next((p for p in getattr(assembly, "parts", ()) if p.key == PROJECTION_CHAMBER_DPA_APERTURE), None)
    if dpa is not None:
        anchors["projection_dpa"] = float(dpa.center_z_mm)
    else:
        dpa = next((p for p in state.apertures if p.key == PROJECTION_CHAMBER_DPA_APERTURE), None)
        if dpa is not None:
            anchors["projection_dpa"] = float(dpa.z_mm)
    anchors["column_end"] = max(float(getattr(assembly, "exit_z_mm", 0)),
                                float(state.camera.z_mm), float(state.fluorescent_screen.z_mm))
    return anchors


def resolve_regions(state, *, include_cell=True, include_disabled=False):
    config = state.vacuum_map.validate()
    if not config.enabled and not include_disabled:
        return ()
    anchors = boundary_anchors(state)
    resolved = []
    for r in config.regions:
        try:
            a = anchors[r.start_anchor] + r.start_offset_mm
            b = anchors[r.end_anchor] + r.end_offset_mm
        except KeyError as exc:
            raise ValueError(f"Vacuum region {r.name}: missing installed boundary {exc.args[0]}") from exc
        if b <= a:
            raise ValueError(f"Vacuum region {r.name}: end must follow start")
        resolved.append(ResolvedMedium(r.key, r.name, a, b, r.medium))
    resolved.sort(key=lambda r: r.start_z_mm)
    if not resolved:
        if not config.enabled:
            return ()
        raise ValueError("Enabled vacuum map requires beam-path regions")
    keys = {r.key for r in resolved}
    if not {"column", "specimen", "post_column", "projection"}.issubset(keys) or not any(k.startswith("gun") for k in keys):
        raise ValueError("Vacuum map must retain gun, column, specimen, post-column and projection regions")
    resolved = fill_region_gaps(resolved)
    if abs(resolved[0].start_z_mm-anchors["source"]) > 1e-8 or abs(resolved[-1].end_z_mm-anchors["column_end"]) > 1e-8:
        raise ValueError("Vacuum map must cover the full source-to-column-end beam path")
    projection = next(r for r in resolved if r.key == "projection")
    if abs(projection.start_z_mm-anchors["projection_dpa"]) > 1e-8:
        raise ValueError("Projection chamber must begin at the installed projection DPA")
    cell = config.cell
    if include_cell and cell.inserted:
        centre = anchors["sample"] + cell.offset_z_mm
        a, b = centre-cell.length_mm/2, centre+cell.length_mm/2
        ambient = next((r for r in resolved if r.key == "specimen"), None)
        if ambient is None or a < ambient.start_z_mm or b > ambient.end_z_mm:
            raise ValueError("The inserted cell must fit inside the specimen environment region")
        # A displaced finite specimen remains a separately modeled solid. The
        # transport excludes its volume rather than counting liquid/gas there.
        resolved.append(ResolvedMedium("specimen_cell", "Inserted specimen cell", a, b,
                                       cell.medium, cell.diameter_mm/2, cell.offset_x_mm, cell.offset_y_mm))
    return tuple(resolved)


def bind_gun_environment(state):
    gun = state.electron_gun
    regions = tuple(r for r in resolve_regions(state, include_cell=False)
                    if r.start_z_mm < gun.exit_plane_z_mm and r.end_z_mm > 0)
    gun._vacuum_regions = regions
    gun._vacuum_seed = state.vacuum_map.seed
    gun._vacuum_max_step_tau = state.vacuum_map.max_optical_depth_per_step
    return regions


def ensure_standalone_gun_environment(gun):
    """Direct gun callers get the normal map; a bound historical map stays off."""
    if hasattr(gun, "_vacuum_regions"):
        return
    config = VacuumMap.load()
    surface = getattr(gun.emitter, "surface_model", None)
    anchors = {"source": -surface.geometry.shank_length_um*.001 if surface is not None else 0., "gun_exit": float(gun.exit_plane_z_mm),
               "gun_acceleration_start": float(gun.accelerator.mechanical_center_from_tip_mm-gun.accelerator.mechanical_length_mm/2)}
    gun._vacuum_regions = tuple(fill_region_gaps([ResolvedMedium(r.key, r.name,
        anchors[r.start_anchor]+r.start_offset_mm, anchors[r.end_anchor]+r.end_offset_mm, r.medium)
        for r in config.regions if config.enabled and r.start_anchor in anchors and r.end_anchor in anchors]))
    gun._vacuum_seed = config.seed
    gun._vacuum_max_step_tau = config.max_optical_depth_per_step
