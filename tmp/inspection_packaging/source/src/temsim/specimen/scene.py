"""Immutable specimen scene shared by particle, EDS and wave calculations.

``Sample.z_mm`` is the optical reference plane at the specimen centre.  Local
scene coordinates therefore place a finite specimen symmetrically about
``z=0``; the support grid starts at the downstream specimen face.  Keeping
this convention in one snapshot prevents individual solvers from silently
interpreting the same state as either a centre plane or an entrance surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from temsim.specimen.envelope import (
    canonical_sample_envelope_shape,
    envelope_contains_xy,
    envelope_intersects_bounds,
)
from temsim.specimen.geometry import sample_orientation_quaternion
from temsim.specimen.source import (
    active_cif_path,
    active_specimen_source,
    selected_reference_preset_key,
    specimen_mode,
    specimen_structure_available,
    wave_template_preset_key,
)
from temsim.specimen.support import SupportGrid, resolve_support_grid


@dataclass(frozen=True, slots=True)
class SceneMaterialRegion:
    """One axial material interval through the scene at a fixed X/Y point."""

    source_key: str
    material: Any
    z_start_nm: float
    z_end_nm: float

    @property
    def path_length_nm(self) -> float:
        return max(float(self.z_end_nm) - float(self.z_start_nm), 0.0)

    @property
    def emitting_layer_thickness_nm(self) -> float:
        return self.path_length_nm


@dataclass(frozen=True, slots=True)
class SpecimenScene:
    """Validated, solver-neutral snapshot of the active specimen and holder."""

    inserted: bool
    reference_z_mm: float
    mode: str
    source_kind: str
    source_key: str
    structure_available: bool
    cif_path: str
    preset_key: str
    wave_template_key: str
    envelope_shape: str
    centre_xy_nm: tuple[float, float]
    size_xy_nm: tuple[float, float]
    thickness_nm: float
    orientation_quaternion_wxyz: tuple[float, float, float, float]
    support_grid: SupportGrid
    support_offset_xy_um: tuple[float, float]
    support_rotation_deg: float
    sample_material: Any | None = None
    support_material: Any | None = None

    @classmethod
    def from_state(
        cls,
        state,
        *,
        include_eds_materials: bool = False,
    ) -> "SpecimenScene":
        sample = state.sample
        reference_z_mm = float(getattr(sample, "z_mm", 0.0))
        centre_xy_nm = (
            float(getattr(sample, "centre_x_nm", 0.0)),
            float(getattr(sample, "centre_y_nm", 0.0)),
        )
        size_xy_nm = (
            float(getattr(sample, "size_x_nm", 0.0)),
            float(getattr(sample, "size_y_nm", 0.0)),
        )
        thickness_nm = float(getattr(sample, "thickness_nm", 0.0))
        support_offset_xy_um = (
            float(getattr(sample, "eds_support_offset_x_um", 0.0)),
            float(getattr(sample, "eds_support_offset_y_um", 0.0)),
        )
        support_rotation_deg = float(
            getattr(sample, "eds_support_rotation_deg", 0.0)
        )
        if not all(
            math.isfinite(value)
            for value in (
                reference_z_mm,
                *centre_xy_nm,
                *size_xy_nm,
                thickness_nm,
                *support_offset_xy_um,
                support_rotation_deg,
            )
        ):
            raise ValueError("Specimen scene geometry must be finite.")
        if min(size_xy_nm) <= 0.0:
            raise ValueError("Specimen X/Y envelope size must be positive.")
        if thickness_nm < 0.0:
            raise ValueError("Specimen thickness cannot be negative.")

        mode = specimen_mode(sample)
        source_kind = active_specimen_source(sample)
        cif_path = active_cif_path(sample)
        preset_key = selected_reference_preset_key(sample)
        if source_kind == "cif":
            source_key = f"cif:{Path(cif_path).name}" if cif_path else "unconfigured"
        else:
            source_key = f"preset:{preset_key}" if preset_key else "unconfigured"
        inserted = bool(getattr(sample, "inserted", True))
        support_grid = resolve_support_grid(
            str(getattr(sample, "eds_support_material_key", "vacuum")),
            str(getattr(sample, "eds_support_mesh_key", "square_200")),
        )

        sample_material = None
        support_material = None
        if include_eds_materials:
            # Local import avoids a module cycle: eds_signal also consumes the
            # scene for its straight-path reference calculation.
            from temsim.detector.eds_signal import (
                material_from_sample,
                material_from_support_grid,
            )

            sample_material = material_from_sample(state)
            support_material = (
                material_from_support_grid(support_grid) if inserted else None
            )

        return cls(
            inserted=inserted,
            reference_z_mm=reference_z_mm,
            mode=mode,
            source_kind=source_kind,
            source_key=source_key,
            structure_available=specimen_structure_available(sample),
            cif_path=cif_path,
            preset_key=preset_key,
            wave_template_key=wave_template_preset_key(
                sample, inserted=inserted
            ),
            envelope_shape=canonical_sample_envelope_shape(
                getattr(sample, "envelope_shape", "rectangle")
            ),
            centre_xy_nm=centre_xy_nm,
            size_xy_nm=size_xy_nm,
            thickness_nm=thickness_nm,
            orientation_quaternion_wxyz=sample_orientation_quaternion(sample),
            support_grid=support_grid,
            support_offset_xy_um=support_offset_xy_um,
            support_rotation_deg=support_rotation_deg,
            sample_material=sample_material if inserted else None,
            support_material=support_material if inserted else None,
        )

    @property
    def interacting_thickness_nm(self) -> float:
        """Return the finite specimen-region propagation span.

        An explicit Vacuum preset retains the user-defined free-space span for
        wave propagation, while ``matter_thickness_nm`` remains zero. A
        retracted or unconfigured specimen has no local propagation region.
        """

        if not self.inserted or not self.structure_available:
            return 0.0
        return self.thickness_nm

    @property
    def matter_thickness_nm(self) -> float:
        """Return thickness containing matter rather than free-space vacuum."""

        return 0.0 if self.is_vacuum else self.interacting_thickness_nm

    @property
    def is_vacuum(self) -> bool:
        """Whether this immutable active source represents vacuum."""

        return bool(
            not self.inserted
            or self.thickness_nm <= 0.0
            or (self.mode == "virtual" and self.preset_key == "vacuum")
        )

    @property
    def sample_top_nm(self) -> float:
        return -0.5 * self.thickness_nm

    @property
    def sample_bottom_nm(self) -> float:
        return 0.5 * self.thickness_nm

    @property
    def support_top_nm(self) -> float:
        return self.sample_bottom_nm

    @property
    def support_bottom_nm(self) -> float:
        return self.support_top_nm + self.support_grid.foil_thickness_um * 1000.0

    @property
    def sample_top_z_mm(self) -> float:
        return self.reference_z_mm + self.sample_top_nm * 1.0e-6

    @property
    def sample_bottom_z_mm(self) -> float:
        return self.reference_z_mm + self.sample_bottom_nm * 1.0e-6

    def sample_contains_xy(self, x_nm, y_nm):
        return envelope_contains_xy(
            self.envelope_shape,
            x_nm,
            y_nm,
            centre_xy_nm=self.centre_xy_nm,
            size_xy_nm=self.size_xy_nm,
        )

    def sample_intersects_bounds(
        self, bounds_nm: tuple[float, float, float, float]
    ) -> bool:
        return envelope_intersects_bounds(
            self.envelope_shape,
            bounds_nm,
            centre_xy_nm=self.centre_xy_nm,
            size_xy_nm=self.size_xy_nm,
        )

    def support_region_at_xy(self, x_nm: float, y_nm: float) -> str:
        return self.support_grid.region_at_nm(
            x_nm,
            y_nm,
            offset_x_um=self.support_offset_xy_um[0],
            offset_y_um=self.support_offset_xy_um[1],
            rotation_deg=self.support_rotation_deg,
        )

    def axial_material_regions(
        self, x_nm: float, y_nm: float
    ) -> tuple[SceneMaterialRegion, ...]:
        """Return actual sample/support intervals along a +Z axial track."""

        if not self.inserted:
            return ()
        regions: list[SceneMaterialRegion] = []
        if (
            self.sample_material is not None
            and self.thickness_nm > 0.0
            and self.sample_contains_xy(x_nm, y_nm)
        ):
            regions.append(
                SceneMaterialRegion(
                    source_key="sample",
                    material=self.sample_material,
                    z_start_nm=self.sample_top_nm,
                    z_end_nm=self.sample_bottom_nm,
                )
            )
        support_region = self.support_region_at_xy(x_nm, y_nm)
        if self.support_material is not None and support_region in {"bar", "rim"}:
            regions.append(
                SceneMaterialRegion(
                    source_key=f"support:{support_region}",
                    material=self.support_material,
                    z_start_nm=self.support_top_nm,
                    z_end_nm=self.support_bottom_nm,
                )
            )
        return tuple(regions)
