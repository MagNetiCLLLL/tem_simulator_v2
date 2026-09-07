"""TOML-backed mechanical shapes, independent of any GUI or rendering engine.

The initial primitive is an annular cylinder. Its dimensions describe the
material body; the separate vacuum passage is a column constraint. Renderers
may use the same shape as an axial section or revolve it into a 3-D surface.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
from numbers import Real

from temsim.mechanical_profiles import MAGNETIC_LENS_MECHANICAL_PROFILES


def _dimension(name, value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number in mm")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number in mm") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number in mm")
    return result


@dataclass(frozen=True)
class AnnularPartGeometry:
    key: str
    name: str
    profile: str
    center_z_mm: float
    length_mm: float
    inner_diameter_mm: float
    outer_diameter_mm: float
    vacuum_inner_diameter_mm: float
    material_class: str
    center_fraction: float = 0.5

    def __post_init__(self):
        if not self.key or self.profile not in MAGNETIC_LENS_MECHANICAL_PROFILES:
            raise ValueError("Dimension editing supports annular coils, housings and yokes")
        for field in ("center_z_mm", "length_mm", "inner_diameter_mm", "outer_diameter_mm",
                      "vacuum_inner_diameter_mm", "center_fraction"):
            object.__setattr__(self, field, _dimension(field, getattr(self, field)))
        if self.length_mm <= 0:
            raise ValueError("Length must be greater than zero")
        if not 0 <= self.center_fraction <= 1:
            raise ValueError("The fixed centre must lie within the axial envelope")
        if not all(math.isfinite(value) for value in (self.start_z_mm, self.end_z_mm)):
            raise ValueError("Axial endpoints must be finite")
        if self.inner_diameter_mm < 0 or self.outer_diameter_mm <= self.inner_diameter_mm:
            raise ValueError("Diameters must satisfy 0 <= inner diameter < outer diameter")
        if self.vacuum_inner_diameter_mm <= 0:
            raise ValueError("The vacuum passage must have a positive diameter")
        if self.inner_diameter_mm < self.vacuum_inner_diameter_mm - 1e-9:
            raise ValueError("The material's inner diameter must clear the vacuum passage")

    @property
    def start_z_mm(self):
        return self.center_z_mm - self.center_fraction * self.length_mm

    @property
    def end_z_mm(self):
        return self.center_z_mm + (1.0 - self.center_fraction) * self.length_mm

    @property
    def thickness_mm(self):
        """Radial wall/winding thickness, not a diameter or vacuum bore."""
        return (self.outer_diameter_mm - self.inner_diameter_mm) * 0.5

    def with_dimension(self, name, value, *, thickness_anchor="inner"):
        value = _dimension(name, value)
        if name in {"length_mm", "inner_diameter_mm", "outer_diameter_mm"}:
            return replace(self, **{name: value})
        if name != "thickness_mm":
            raise ValueError(f"Unsupported dimension: {name}")
        if value <= 0:
            raise ValueError("Radial thickness must be greater than zero")
        if thickness_anchor == "inner":
            return replace(self, outer_diameter_mm=self.inner_diameter_mm + 2 * value)
        if thickness_anchor == "outer":
            return replace(self, inner_diameter_mm=self.outer_diameter_mm - 2 * value)
        raise ValueError("Thickness anchor must be inner or outer")

    def updates_from(self, original):
        """Only changed material dimensions become authoritative TOML edits."""
        protected = ("key", "profile", "center_z_mm", "center_fraction",
                     "vacuum_inner_diameter_mm", "material_class")
        if any(getattr(self, field) != getattr(original, field) for field in protected):
            raise ValueError("Dimension edits cannot move the centre or change the vacuum passage/material")
        fields = {}
        if self.length_mm != original.length_mm:
            fields.update(length_mm=self.length_mm, local_start_z_mm=self.start_z_mm,
                          local_end_z_mm=self.end_z_mm)
        if self.inner_diameter_mm != original.inner_diameter_mm:
            fields["mechanical_inner_diameter_mm"] = self.inner_diameter_mm
        if self.outer_diameter_mm != original.outer_diameter_mm:
            fields["mechanical_outer_diameter_mm"] = self.outer_diameter_mm
        return {("parts", self.key, field): value for field, value in fields.items()}

    def to_shape_spec(self):
        """A renderer-neutral derived shape; TOML remains the source of truth."""
        return {
            "primitive": "annular_cylinder", "axis": "z", "units": "mm",
            "part_key": self.key, "start_z_mm": self.start_z_mm, "end_z_mm": self.end_z_mm,
            "center_z_mm": self.center_z_mm, "inner_radius_mm": self.inner_diameter_mm / 2,
            "outer_radius_mm": self.outer_diameter_mm / 2, "material_class": self.material_class,
        }


def geometry_from_part(part: Mapping, *, parent: Mapping | None = None):
    """Read an annular primitive; pass its parent to check assembly-owned shapes."""
    if part.get("mechanical_profile") not in MAGNETIC_LENS_MECHANICAL_PROFILES:
        raise ValueError("Graphical dimensions currently support coils, housings and magnetic yokes")
    if part.get("magnetic_radial_profile_mm") is not None:
        raise ValueError("This part has a shaped radial profile; edit its profile in TOML")
    if part.get("material_intervals_mm") is not None:
        raise ValueError("This part has explicit material sections; edit those sections in TOML")
    parent = parent or {}
    if any(
        data.get("magnetic_lens_keys") or data.get("shared_housing_key")
        or data.get("magnetic_circuit_topology") == "shared_pole_multi_gap"
        for data in (part, parent)
    ):
        raise ValueError("This part belongs to a shared magnetic structure; edit its assembly in TOML")
    if part.get("mechanical_profile") in {"magnetic_excitation_coil", "magnetic_lens_yoke"}:
        # Objective material is defined by two parent-owned intervals, not the
        # child's display envelope. Never offer handles for that envelope.
        split_fields = ("upper_yoke_start_local_z_mm", "upper_yoke_end_local_z_mm",
                        "lower_yoke_start_local_z_mm", "lower_yoke_end_local_z_mm")
        if part.get("parent_key") == "objective_lens" or any(field in parent for field in split_fields):
            raise ValueError("This part has split material sections defined by its parent; edit those sections in TOML")
    required = ("key", "local_start_z_mm", "local_center_z_mm", "local_end_z_mm", "length_mm",
                "mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm", "vacuum_inner_diameter_mm")
    missing = [field for field in required if field not in part]
    if missing:
        raise ValueError("Missing mechanical geometry: " + ", ".join(missing))
    start, center, end, length = (
        _dimension(field, part[field])
        for field in ("local_start_z_mm", "local_center_z_mm", "local_end_z_mm", "length_mm")
    )
    if not start <= center <= end or not math.isclose(end - start, length, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Part length and axial endpoints must be consistent before editing")
    return AnnularPartGeometry(
        key=str(part["key"]), name=str(part.get("name", part["key"])),
        profile=str(part["mechanical_profile"]), center_z_mm=center, length_mm=length,
        inner_diameter_mm=part["mechanical_inner_diameter_mm"],
        outer_diameter_mm=part["mechanical_outer_diameter_mm"],
        vacuum_inner_diameter_mm=part["vacuum_inner_diameter_mm"],
        material_class=str(part.get("material_class", "Unspecified")),
        center_fraction=(center - start) / length if length > 0 else 0.5,
    )
