"""TOML-backed TEM support-grid geometry and material lookup."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import tomllib

from temsim.paths import SPECIMEN_SUPPORT_CONFIG_ROOT


@dataclass(frozen=True, slots=True)
class SupportMaterial:
    key: str
    name: str
    atomic_number: int
    density_g_cm3: float
    status: str

    @property
    def is_vacuum(self) -> bool:
        return self.atomic_number == 0


@dataclass(frozen=True, slots=True)
class SupportMesh:
    key: str
    name: str
    mesh_count_per_inch: int
    pitch_um: float
    hole_width_um: float
    bar_width_um: float
    open_area_percent: float


@dataclass(frozen=True, slots=True)
class SupportGrid:
    material: SupportMaterial
    mesh: SupportMesh
    outer_diameter_mm: float
    foil_thickness_um: float
    rim_width_um: float
    geometry_status: str
    geometry_source: str
    geometry_source_url: str

    def region_at_nm(
        self,
        x_nm: float,
        y_nm: float,
        *,
        offset_x_um: float = 0.0,
        offset_y_um: float = 0.0,
        rotation_deg: float = 0.0,
    ) -> str:
        """Classify a laboratory point as vacuum, opening, bar, rim or outside."""

        values = (x_nm, y_nm, offset_x_um, offset_y_um, rotation_deg)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Support-grid coordinates must be finite")
        if self.material.is_vacuum:
            return "vacuum"
        x_um = float(x_nm) * 1.0e-3 - float(offset_x_um)
        y_um = float(y_nm) * 1.0e-3 - float(offset_y_um)
        angle = math.radians(float(rotation_deg))
        cosine = math.cos(angle)
        sine = math.sin(angle)
        local_x = cosine * x_um + sine * y_um
        local_y = -sine * x_um + cosine * y_um
        radius_um = math.hypot(local_x, local_y)
        outer_radius_um = 0.5 * self.outer_diameter_mm * 1000.0
        if radius_um > outer_radius_um:
            return "outside"
        if radius_um >= outer_radius_um - self.rim_width_um:
            return "rim"
        pitch = self.mesh.pitch_um
        half_hole = 0.5 * self.mesh.hole_width_um
        # The offset denotes the centre of a mesh opening. Zero therefore
        # starts unobstructed rather than at a bar intersection.
        cell_x = (local_x + 0.5 * pitch) % pitch - 0.5 * pitch
        cell_y = (local_y + 0.5 * pitch) % pitch - 0.5 * pitch
        if abs(cell_x) <= half_hole and abs(cell_y) <= half_hole:
            return "opening"
        return "bar"

    def material_path_length_nm(
        self, x_nm: float, y_nm: float, **kwargs
    ) -> float:
        region = self.region_at_nm(x_nm, y_nm, **kwargs)
        if region in {"bar", "rim"}:
            return self.foil_thickness_um * 1000.0
        return 0.0


@dataclass(frozen=True, slots=True)
class SupportCatalog:
    materials: dict[str, SupportMaterial]
    meshes: dict[str, SupportMesh]
    outer_diameter_mm: float
    foil_thickness_um: float
    rim_width_um: float
    geometry_status: str
    geometry_source: str
    geometry_source_url: str


@lru_cache(maxsize=1)
def load_support_catalog(path: Path | None = None) -> SupportCatalog:
    source = Path(path or (SPECIMEN_SUPPORT_CONFIG_ROOT / "catalog.toml"))
    with source.open("rb") as stream:
        data = tomllib.load(stream)
    if int(data.get("format_version", 0)) != 1:
        raise ValueError("Unsupported support-grid catalog format")
    if data.get("catalog_type") != "tem_support_grid":
        raise ValueError("Invalid support-grid catalog type")
    materials = {
        str(key): SupportMaterial(
            key=str(key),
            name=str(row["name"]),
            atomic_number=int(row["atomic_number"]),
            density_g_cm3=float(row["density_g_cm3"]),
            status=str(row["status"]),
        )
        for key, row in data.get("materials", {}).items()
    }
    meshes = {
        str(key): SupportMesh(
            key=str(key),
            name=str(row["name"]),
            mesh_count_per_inch=int(row["mesh_count_per_inch"]),
            pitch_um=float(row["pitch_um"]),
            hole_width_um=float(row["hole_width_um"]),
            bar_width_um=float(row["bar_width_um"]),
            open_area_percent=float(row["open_area_percent"]),
        )
        for key, row in data.get("meshes", {}).items()
    }
    if not materials or not meshes:
        raise ValueError("Support-grid catalog needs materials and meshes")
    for material in materials.values():
        if material.atomic_number < 0 or material.atomic_number > 99:
            raise ValueError(
                f"Invalid support atomic number: {material.key}"
            )
        if material.density_g_cm3 < 0.0:
            raise ValueError(f"Invalid support density: {material.key}")
        if material.is_vacuum != (material.density_g_cm3 == 0.0):
            raise ValueError(
                "Only the virtual vacuum support may have zero density"
            )
    for mesh in meshes.values():
        if min(
            mesh.pitch_um, mesh.hole_width_um, mesh.bar_width_um
        ) <= 0.0:
            raise ValueError(
                f"Invalid support mesh dimensions: {mesh.key}"
            )
        if not math.isclose(
            mesh.hole_width_um + mesh.bar_width_um,
            mesh.pitch_um,
            abs_tol=1.1,
        ):
            raise ValueError(f"Support mesh pitch mismatch: {mesh.key}")
        if not 0.0 < mesh.open_area_percent < 100.0:
            raise ValueError(f"Invalid support open area: {mesh.key}")
    catalog = SupportCatalog(
        materials=materials,
        meshes=meshes,
        outer_diameter_mm=float(data["outer_diameter_mm"]),
        foil_thickness_um=float(data["foil_thickness_um"]),
        rim_width_um=float(data["rim_width_um"]),
        geometry_status=str(data["geometry_status"]),
        geometry_source=str(data["geometry_source"]),
        geometry_source_url=str(data["geometry_source_url"]),
    )
    if min(
        catalog.outer_diameter_mm,
        catalog.foil_thickness_um,
        catalog.rim_width_um,
    ) <= 0.0:
        raise ValueError(
            "Support-grid envelope dimensions must be positive"
        )
    return catalog


def available_support_materials() -> tuple[tuple[str, str], ...]:
    return tuple(
        (key, material.name)
        for key, material in load_support_catalog().materials.items()
    )


def available_support_meshes() -> tuple[tuple[str, str], ...]:
    return tuple(
        (key, mesh.name)
        for key, mesh in load_support_catalog().meshes.items()
    )


def resolve_support_grid(material_key: str, mesh_key: str) -> SupportGrid:
    catalog = load_support_catalog()
    try:
        material = catalog.materials[str(material_key)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown support material: {material_key}"
        ) from exc
    try:
        mesh = catalog.meshes[str(mesh_key)]
    except KeyError as exc:
        raise ValueError(f"Unknown support mesh: {mesh_key}") from exc
    return SupportGrid(
        material=material,
        mesh=mesh,
        outer_diameter_mm=catalog.outer_diameter_mm,
        foil_thickness_um=catalog.foil_thickness_um,
        rim_width_um=catalog.rim_width_um,
        geometry_status=catalog.geometry_status,
        geometry_source=catalog.geometry_source,
        geometry_source_url=catalog.geometry_source_url,
    )
