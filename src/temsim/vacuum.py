"""Editable beam-path environments, separate from mechanical vacuum bores.

Pressure is in mbar, geometry in mm. Resolved boundaries follow the installed
hardware; a cell replaces the ambient medium only in its finite volume.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import tomllib

from temsim.paths import CONFIG_ROOT

CELL_KEYS = {"specimen_cell", "cell_window_upstream", "cell_window_downstream"}

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
    # Optional mole fractions override formula; never silently normalised.
    mixture_mole_fractions: dict[str, float] = field(default_factory=dict)

    def atomic_stoichiometry(self):
        from ase.formula import Formula
        atoms = {}
        for formula, fraction in (self.mixture_mole_fractions or {self.formula: 1.0}).items():
            if not isinstance(formula, str) or not formula.strip():
                raise ValueError("Every medium species needs a chemical formula")
            species = Formula(formula).count()
            if not species or any(count <= 0 for count in species.values()):
                raise ValueError(f"Invalid medium species: {formula}")
            for symbol, count in species.items():
                atoms[symbol] = atoms.get(symbol, 0.0)+fraction*count
        return atoms

    def validate(self):
        from ase.data import atomic_numbers
        if self.phase not in {"gas", "liquid", "solid", "vacuum"}:
            raise ValueError("Medium phase must be gas, liquid, solid or vacuum")
        if not isinstance(self.mixture_mole_fractions, dict):
            raise ValueError("Mixture must map chemical formulas to mole fractions")
        if self.mixture_mole_fractions:
            fractions = list(self.mixture_mole_fractions.values())
            if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in fractions):
                raise ValueError("Mixture mole fractions must be finite and positive")
            if not math.isclose(sum(fractions), 1.0, rel_tol=0, abs_tol=1e-8):
                raise ValueError("Mixture mole fractions must sum to one")
        try:
            atoms = self.atomic_stoichiometry()
            valid = atoms and all(k in atomic_numbers and n > 0 for k, n in atoms.items())
        except (ValueError, KeyError, TypeError):
            valid = False
        if not valid:
            raise ValueError(f"Invalid medium composition: {self.mixture_mole_fractions or self.formula}")
        for name in ("pressure_mbar", "temperature_k", "density_kg_m3", "removal_cross_section_m2"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Medium {name} must be finite and non-negative")
        if self.temperature_k <= 0 or self.density_kg_m3 <= 0:
            raise ValueError("Temperature and condensed-medium mass density must be positive")
        if self.removal_cross_section_m2 and not self.removal_reference.strip():
            raise ValueError("A nonzero removal cross section needs a measurement/model reference")
        return self

    def number_density_m3(self):
        if self.phase == "vacuum":
            return 0.0
        from scipy.constants import Boltzmann, atomic_mass
        if self.phase == "gas":
            return self.pressure_mbar * 100.0 / (Boltzmann * self.temperature_k)
        from ase.data import atomic_masses, atomic_numbers
        mass = sum(atomic_masses[atomic_numbers[k]] * v for k, v in self.atomic_stoichiometry().items())
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
class CellWindow:
    # Zero thickness preserves old windowless cell profiles without migration.
    thickness_nm: float = 0.0
    material: str = "SiN (Si3N4 approximation)"
    medium: Medium = field(default_factory=lambda: Medium(phase="solid", formula="Si3N4", density_kg_m3=3100.0))
    reference: str = "Editable bulk-density approximation; see CELL_ENVIRONMENT.md"
    diameter_mm: float = 0.0  # Zero inherits the cell aperture (historical maps).

    def radius_mm(self, cell):
        return (self.diameter_mm or cell.diameter_mm) / 2

    def validate(self):
        self.medium.validate()
        if self.medium.phase != "solid":
            raise ValueError("Cell window material must be solid")
        if type(self.thickness_nm) not in (int, float) or not math.isfinite(self.thickness_nm) or self.thickness_nm < 0:
            raise ValueError("Window thickness must be finite and non-negative")
        if not isinstance(self.material, str) or not self.material.strip():
            raise ValueError("Window material needs a name")
        if type(self.diameter_mm) not in (int, float) or not math.isfinite(self.diameter_mm) or self.diameter_mm < 0:
            raise ValueError("Window diameter must be finite and non-negative (zero follows cell aperture)")
        return self


@dataclass
class SpecimenCell:
    inserted: bool = False
    diameter_mm: float = 0.01
    length_mm: float = 0.001  # Inner-face separation (cell gap), excluding windows.
    offset_x_mm: float = 0.0
    offset_y_mm: float = 0.0
    offset_z_mm: float = 0.0
    medium: Medium = field(default_factory=lambda: Medium(pressure_mbar=1.0))
    upstream_window: CellWindow = field(default_factory=CellWindow)
    downstream_window: CellWindow = field(default_factory=CellWindow)
    pressure_gradient_enabled: bool = False
    end_pressure_mbar: float = 1.0  # Downstream inner face; upstream uses medium.pressure_mbar.


@dataclass
class VacuumMap:
    enabled: bool = False
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
        if any(not k or k in CELL_KEYS or k.startswith("transition:") for k in keys) or len(set(keys)) != len(keys):
            raise ValueError("Vacuum region keys must be nonempty and unique; cell layer keys and transition: are reserved")
        for r in self.regions:
            r.medium.validate()
            for value in (r.start_offset_mm, r.end_offset_mm):
                if not math.isfinite(value):
                    raise ValueError("Vacuum boundary offsets must be finite")
        self.cell.medium.validate()
        if self.cell.medium.phase == "solid":
            raise ValueError("Cell interior must be gas, liquid or vacuum")
        self.cell.upstream_window.validate()
        self.cell.downstream_window.validate()
        if not isinstance(self.cell.pressure_gradient_enabled, bool):
            raise ValueError("Cell pressure-gradient setting must be boolean")
        pressure = self.cell.end_pressure_mbar
        if type(pressure) not in (int, float) or not math.isfinite(pressure) or pressure < 0:
            raise ValueError("Cell downstream pressure must be finite and non-negative")
        if self.cell.pressure_gradient_enabled and self.cell.medium.phase != "gas":
            raise ValueError("A cell pressure gradient requires a gas; liquid density is a separate input")
        for key in ("diameter_mm", "length_mm", "offset_x_mm", "offset_y_mm", "offset_z_mm"):
            if not math.isfinite(getattr(self.cell, key)):
                raise ValueError(f"Cell {key} must be finite")
        if self.cell.diameter_mm <= 0 or self.cell.length_mm <= 0:
            raise ValueError("Cell dimensions must be positive")
        for window in (self.cell.upstream_window, self.cell.downstream_window):
            if window.thickness_nm and window.radius_mm(self.cell) < self.cell.diameter_mm / 2:
                raise ValueError("Each nonzero window must cover the cell aperture")
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
            windows = {}
            for key in ("upstream_window", "downstream_window"):
                window = dict(c.get(key, {}))
                if "medium" in window:
                    window["medium"] = Medium(**window["medium"])
                windows[key] = CellWindow(**window)
            data["cell"] = SpecimenCell(**{**c, "medium": Medium(**c.get("medium", {})), **windows})
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
                if left.medium.phase in {"liquid", "solid"} or region.medium.phase in {"liquid", "solid"}:
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
    from temsim.assembly_navigation import assembly_sections, component_anchor
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
                anchors[component_anchor(p, short)] = float(getattr(p, attr))
    for section in assembly_sections(assembly):
        for point in ("origin", "start", "end"):
            anchors[f"{section.key}.{point}"] = float(getattr(section, point+"_z_mm"))
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
    if include_cell and config.cell.inserted:
        layers = resolve_cell_layers(state)
        ambient = next((r for r in resolved if r.key == "specimen"), None)
        outer_a = min(r.start_z_mm for r in layers)
        outer_b = max(r.end_z_mm for r in layers)
        if ambient is None or outer_a < ambient.start_z_mm or outer_b > ambient.end_z_mm:
            raise ValueError("The inserted cell must fit inside the specimen environment region")
        resolved.extend(layers)
    return tuple(resolved)


def resolve_cell_layers(state):
    """Single geometry source for Physical Layout and executed transport.

    Uses applied inputs even when transport is disabled; no display thickness,
    inferred frame, optical aperture or second specimen is introduced.
    """
    cell = state.vacuum_map.validate().cell
    if not cell.inserted:
        return ()
    centre = float(state.sample.z_mm) + cell.offset_z_mm
    a, b = centre-cell.length_mm/2, centre+cell.length_mm/2
    up, down = cell.upstream_window, cell.downstream_window
    outer_a, outer_b = a-up.thickness_nm*1e-6, b+down.thickness_nm*1e-6
    if b <= a or (up.thickness_nm > 0 and outer_a >= a) or (down.thickness_nm > 0 and outer_b <= b):
        raise ValueError("Cell gap or window thickness is below the numerical resolution at this Z position")
    endpoint = replace(cell.medium, pressure_mbar=cell.end_pressure_mbar) if cell.pressure_gradient_enabled else None
    layers = [ResolvedMedium("specimen_cell", "Cell interior", a, b, cell.medium,
                             cell.diameter_mm/2, cell.offset_x_mm, cell.offset_y_mm, endpoint)]
    from temsim.specimen.source import specimen_is_vacuum
    sample = state.sample
    for key, label, window, lo, hi in (
            ("cell_window_upstream", "Upstream window", up, outer_a, a),
            ("cell_window_downstream", "Downstream window", down, b, outer_b)):
        if window.thickness_nm == 0:
            continue
        footprint = replace(cell, diameter_mm=2*window.radius_mm(cell))
        if sample.inserted and not specimen_is_vacuum(sample) and sample_overlaps_window(sample, footprint, lo, hi):
            raise ValueError("Sample intersects a cell window. Increase the cell gap or change the cell offset; edit the specimen only in Sample.")
        layers.append(ResolvedMedium(key, f"{label} · {window.material}", lo, hi,
            window.medium, window.radius_mm(cell), cell.offset_x_mm, cell.offset_y_mm))
    return tuple(layers)


def sample_overlaps_window(sample, cell, lo, hi):
    """Finite Sample envelope versus window cylinder; all lengths here in mm."""
    half_z = sample.thickness_nm*.5e-6
    tolerance = 1e-10  # 0.0001 nm, only for face-touch roundoff.
    if sample.z_mm+half_z <= lo+tolerance or sample.z_mm-half_z >= hi-tolerance:
        return False
    dx = sample.centre_x_nm*1e-6-cell.offset_x_mm
    dy = sample.centre_y_nm*1e-6-cell.offset_y_mm
    if sample.envelope_shape == "disk":
        a, b = sample.size_x_nm*.5e-6, sample.size_y_nm*.5e-6
        radius = cell.diameter_mm/2
        if a == b:
            return math.hypot(dx, dy) < radius+a
        # Match specimen_interval's elliptical envelope for unequal X/Y sizes.
        # Closest point on a convex ellipse via a bracketed Lagrange multiplier.
        # Normalisation avoids squaring nanometre-valued mm lengths in the root.
        from scipy.optimize import brentq
        scale = max(a, b, abs(dx), abs(dy), radius)
        a, b, x, y = a/scale, b/scale, abs(dx)/scale, abs(dy)/scale
        if (x/a)**2+(y/b)**2 <= 1:
            return True
        def equation(lam):
            return (a*x/(lam+a*a))**2+(b*y/(lam+b*b))**2-1
        lam = brentq(equation, 0, max(a*x+b*y, 1.0), xtol=1e-14, rtol=1e-14)
        distance = math.hypot(x-a*a*x/(lam+a*a), y-b*b*y/(lam+b*b))
        return distance < radius/scale
    dx = max(abs(dx)-sample.size_x_nm*.5e-6, 0)
    dy = max(abs(dy)-sample.size_y_nm*.5e-6, 0)
    return math.hypot(dx, dy) < cell.diameter_mm/2


def bind_gun_environment(state):
    gun = state.electron_gun
    regions = tuple(r for r in resolve_regions(state, include_cell=False)
                    if r.start_z_mm < gun.exit_plane_z_mm and r.end_z_mm > 0)
    gun._vacuum_regions = regions
    gun._vacuum_seed = state.vacuum_map.seed
    gun._vacuum_max_step_tau = state.vacuum_map.max_optical_depth_per_step
    return regions


def ensure_standalone_gun_environment(gun):
    """Direct gun callers respect the configured opt-in; bound maps stay intact."""
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
