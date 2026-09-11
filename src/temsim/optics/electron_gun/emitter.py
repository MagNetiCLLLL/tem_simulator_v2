"""Cold field-emitter geometry and deterministic emission phase space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from temsim.component_keys import FEG_TIP
from temsim.optics.electron_gun.base import EmissionBundle
from temsim.physics.chromatic import cold_feg_energy_offsets
from temsim.optics.electron_gun.tip_coherence import TipCoherence


def _radical_inverse(indices: np.ndarray, base: int) -> np.ndarray:
    values = np.zeros(indices.size, dtype=float)
    fraction = 1.0 / float(base)
    work = indices.astype(np.int64, copy=True)
    while np.any(work):
        values += (work % base) * fraction
        work //= base
        fraction /= float(base)
    return values


def _halton_dimensions(count: int, bases: Iterable[int]) -> list[np.ndarray]:
    indices = np.arange(18, 18 + count, dtype=np.int64)
    return [_radical_inverse(indices, base) for base in bases]


def _truncated_gaussian_disk(radial_u, azimuth_u, sigma, cutoff):
    if sigma <= 0.0 or cutoff <= 0.0:
        return np.zeros_like(radial_u), np.zeros_like(radial_u)
    retained = 1.0 - np.exp(-(cutoff * cutoff) / (2.0 * sigma * sigma))
    radius = sigma * np.sqrt(
        -2.0 * np.log(np.maximum(1.0 - radial_u * retained, 1e-15))
    )
    azimuth = 2.0 * np.pi * azimuth_u
    return radius * np.cos(azimuth), radius * np.sin(azimuth)


@dataclass
class ColdFieldEmitter:
    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    name: str = "Cold Field Emission Tip"
    key: str = FEG_TIP
    tip_radius_nm: float = 100.0
    tip_cone_half_angle_deg: float = 5.0
    emitter_material: str = "W(310)"
    work_function_ev: float = 4.4
    vacuum_pa: float = 1.0e-8
    emission_current_na: float = 10000.0
    emission_energy_ev: float = 0.3
    minimum_kinetic_energy_ev: float = 0.01
    virtual_source_fwhm_nm: float = 5.0
    angular_rms_mrad: float = 1.0
    angular_cutoff_mrad: float = 3.0
    energy_spread_fwhm_ev: float = 0.3
    young_decay_width_ev: float = 0.20
    boersch_sigma_ev: float = 0.10
    energy_half_range_ev: float = 1.0
    ray_count: int = 1000

    @property
    def coherence(self) -> TipCoherence | None:
        # Keep the pre-existing dataclass field schema and absent-parameter
        # graph byte-for-byte compatible with archived classical gun data.
        # A selected tip model is an explicit captured instance attribute.
        return self.__dict__.get("_tip_coherence")

    @coherence.setter
    def coherence(self, value):
        if value is None:
            self.__dict__.pop("_tip_coherence", None)
        else:
            self.__dict__["_tip_coherence"] = value

    @property
    def label(self):
        return self.name

    @property
    def effective_aperture_radius_mm(self):
        return 0.5 * self.mechanical_outer_diameter_mm

    @property
    def emitted_current_a(self):
        return float(self.emission_current_na) * 1.0e-9

    @property
    def optical_reference_from_tip_mm(self):
        return 0.0

    @property
    def kind(self):
        return "cold_field_emitter"

    @property
    def shape_profile(self):
        return "feg_tip"

    def validate(self):
        nonnegative = (
            "tip_radius_nm",
            "work_function_ev",
            "vacuum_pa",
            "emission_current_na",
            "emission_energy_ev",
            "minimum_kinetic_energy_ev",
            "virtual_source_fwhm_nm",
            "angular_rms_mrad",
            "angular_cutoff_mrad",
            "energy_spread_fwhm_ev",
            "young_decay_width_ev",
            "boersch_sigma_ev",
            "energy_half_range_ev",
        )
        for attribute in nonnegative:
            value = float(getattr(self, attribute))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{self.name} {attribute} must be finite and non-negative.")
        if self.coherence is not None:
            if not isinstance(self.coherence, TipCoherence):
                raise ValueError("FEG tip coherence must be a TipCoherence parameter record")
            self.coherence.validate()
            if self.virtual_source_fwhm_nm <= 0.0:
                raise ValueError("A coherent tip needs positive spatial FWHM")
        if int(self.ray_count) < 9:
            raise ValueError("Cold FEG ray count must be at least 9.")
        if self.emission_energy_ev <= 0.0:
            raise ValueError("Cold FEG emission energy must be positive.")
        if self.minimum_kinetic_energy_ev >= self.emission_energy_ev:
            raise ValueError(
                "Cold FEG minimum kinetic energy must be below its mean "
                "emission energy."
            )
        if self.mechanical_length_mm <= 0.0:
            raise ValueError("Cold FEG visible tip length must be positive.")
        return self

    def emit(self, count: int | None = None) -> EmissionBundle:
        self.validate()
        n = int(self.ray_count if count is None else count)
        if n < 9:
            raise ValueError("Cold FEG emission requires at least 9 rays.")
        if self.coherence is not None:
            from temsim.optics.electron_gun.tip_coherence import tip_particle_samples
            bundle = tip_particle_samples(self, n)
            if getattr(self, "_tuning_boundary_probes", 0):
                from temsim.physics.optical_tuning import add_source_support_probes
                bundle = add_source_support_probes(bundle, self)
            return bundle
        u_r, u_phi, u_a, u_theta, u_young, u_boersch = _halton_dimensions(
            n, (2, 3, 5, 7, 11, 13)
        )
        spatial_sigma_m = self.virtual_source_fwhm_nm / 2.354820045 * 1e-9
        x, y = _truncated_gaussian_disk(
            u_r, u_phi, spatial_sigma_m, 3.0 * spatial_sigma_m
        )
        angular_sigma = self.angular_rms_mrad * 1e-3
        angular_cutoff = self.angular_cutoff_mrad * 1e-3
        tx, ty = _truncated_gaussian_disk(
            u_a, u_theta, angular_sigma, angular_cutoff
        )
        energy = cold_feg_energy_offsets(
            n,
            self.energy_spread_fwhm_ev,
            self.energy_half_range_ev,
            self.young_decay_width_ev,
            self.boersch_sigma_ev,
            quantiles=(u_young, u_boersch),
            mean_kinetic_energy_ev=self.emission_energy_ev,
            minimum_kinetic_energy_ev=self.minimum_kinetic_energy_ev,
        )
        bundle = EmissionBundle(
            x_m=x,
            y_m=y,
            tx_rad=tx,
            ty_rad=ty,
            energy_offset_ev=energy,
            weight=np.full(n, 1.0 / n, dtype=float),
            ray_id=np.arange(n, dtype=np.int64),
        )
        if getattr(self, "_tuning_boundary_probes", 0):
            from temsim.physics.optical_tuning import add_source_support_probes
            bundle = add_source_support_probes(bundle, self)
        return bundle

    def draw_layout(self):
        return {
            "key": self.key,
            "mechanical_center_from_tip_mm": self.mechanical_center_from_tip_mm,
            "mechanical_length_mm": self.mechanical_length_mm,
            "mechanical_outer_diameter_mm": self.mechanical_outer_diameter_mm,
            "shape_profile": self.shape_profile,
        }
