"""Canonical shared-column Condenser Stigmator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from temsim import module_manifest
from temsim.component_keys import (
    CONDENSER_STIGMATOR,
    canonical_stigmator_key,
)
from temsim.optics.model import Stigmator

_DEFAULT_OBJECTIVE_MODULE_PATH = "column/C3_ProbeCorrector.toml"
_DEFAULT_COLUMN_ORIGIN_Z_MM = module_manifest.port_z_mm(
    "gun/FEG.toml", "exit"
)
_DEFAULT_PART = module_manifest.part_data(
    _DEFAULT_OBJECTIVE_MODULE_PATH, CONDENSER_STIGMATOR
)
_DEFAULT_SAMPLE = module_manifest.part_data(
    _DEFAULT_OBJECTIVE_MODULE_PATH, "sample"
)
_DEFAULT_CENTER_ABOVE_SAMPLE_MM = (
    float(_DEFAULT_SAMPLE["local_center_z_mm"])
    - float(_DEFAULT_PART["local_center_z_mm"])
)
_DEFAULT_OPTICAL_REFERENCE_Z_MM = (
    _DEFAULT_COLUMN_ORIGIN_Z_MM
    + float(_DEFAULT_PART["optical_reference_local_z_mm"])
)


@dataclass(frozen=True)
class CondenserStigmatorDefinition:
    key: str
    label: str
    mechanical_center_above_sample_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    mechanical_clear_bore_diameter_mm: float
    optical_reference_z_mm: float
    effective_length_mm: float
    maximum_strength_m2: float
    colour: str
    owner: str = "condenser"
    kind: str = "stigmator"
    shape_profile: str = "quadrupole_body"
    interaction_kind: str = "distributed_quadrupole_field"

    @property
    def effective_aperture_radius_mm(self):
        return self.mechanical_clear_bore_diameter_mm / 2.0

    def create_component(self):
        return CondenserStigmatorComponent(
            name=self.label,
            key=self.key,
            z_mm=self.optical_reference_z_mm,
            length_mm=self.effective_length_mm,
            max_strength_m2=self.maximum_strength_m2,
            strength_x_percent=0.0,
            strength_y_percent=0.0,
            enabled=True,
            colour=self.colour,
            mechanical_center_above_sample_mm=(
                self.mechanical_center_above_sample_mm
            ),
            mechanical_length_mm=self.mechanical_length_mm,
            mechanical_outer_diameter_mm=(
                self.mechanical_outer_diameter_mm
            ),
            mechanical_clear_bore_diameter_mm=(
                self.mechanical_clear_bore_diameter_mm
            ),
            optical_reference_z_mm=self.optical_reference_z_mm,
            corrector=self.owner,
        )


@dataclass
class CondenserStigmatorComponent(Stigmator):
    mechanical_center_above_sample_mm: float = _DEFAULT_CENTER_ABOVE_SAMPLE_MM
    mechanical_length_mm: float = float(_DEFAULT_PART["length_mm"])
    mechanical_outer_diameter_mm: float = float(
        _DEFAULT_PART["mechanical_outer_diameter_mm"]
    )
    mechanical_clear_bore_diameter_mm: float = float(
        _DEFAULT_PART["mechanical_clear_bore_diameter_mm"]
    )
    optical_reference_z_mm: float = _DEFAULT_OPTICAL_REFERENCE_Z_MM
    corrector: str = "condenser"

    EXPECTED_KEY: ClassVar[str] = CONDENSER_STIGMATOR
    KIND: ClassVar[str] = "stigmator"
    SHAPE_PROFILE: ClassVar[str] = "quadrupole_body"
    INTERACTION_KIND: ClassVar[str] = "distributed_quadrupole_field"

    def __post_init__(self):
        object.__setattr__(self, "_position_coupling_ready", True)

    def __setattr__(self, name, value):
        if name in {
            "z_mm",
            "mechanical_center_above_sample_mm",
            "optical_reference_z_mm",
        }:
            value = float(value)
        ready = self.__dict__.get("_position_coupling_ready", False)
        if ready and name == "mechanical_center_above_sample_mm":
            delta_mm = value - float(
                self.mechanical_center_above_sample_mm
            )
            object.__setattr__(self, name, value)
            optical = float(self.optical_reference_z_mm) - delta_mm
            object.__setattr__(self, "optical_reference_z_mm", optical)
            object.__setattr__(self, "z_mm", float(self.z_mm) - delta_mm)
            return
        if ready and name == "optical_reference_z_mm":
            delta_mm = value - float(self.optical_reference_z_mm)
            object.__setattr__(self, name, value)
            object.__setattr__(self, "z_mm", float(self.z_mm) + delta_mm)
            return
        object.__setattr__(self, name, value)

    @property
    def owner(self):
        return self.corrector

    @property
    def kind(self):
        return self.KIND

    @property
    def shape_profile(self):
        return self.SHAPE_PROFILE

    @property
    def interaction_kind(self):
        return self.INTERACTION_KIND

    @property
    def optical_active(self):
        return bool(self.enabled)

    @property
    def effective_aperture_radius_mm(self):
        return self.mechanical_clear_bore_diameter_mm / 2.0

    @property
    def strength_x_m2(self):
        return (
            float(self.max_strength_m2)
            * float(self.strength_x_percent)
            / 100.0
        )

    @property
    def strength_y_m2(self):
        return (
            float(self.max_strength_m2)
            * float(self.strength_y_percent)
            / 100.0
        )

    def validate(self):
        if self.key != self.EXPECTED_KEY:
            raise ValueError(
                "Condenser Stigmator key is not canonical."
            )
        if self.mechanical_center_above_sample_mm < 0.0:
            raise ValueError(
                "Condenser Stigmator must sit above the sample."
            )
        if self.mechanical_length_mm <= 0.0:
            raise ValueError(
                "Condenser Stigmator mechanical length must be positive."
            )
        if not (
            0.0
            < self.mechanical_clear_bore_diameter_mm
            < self.mechanical_outer_diameter_mm
        ):
            raise ValueError(
                "Condenser Stigmator bore must fit inside its body."
            )
        if not 0.0 < self.length_mm <= self.mechanical_length_mm:
            raise ValueError(
                "Condenser Stigmator field length must fit its body."
            )
        if self.max_strength_m2 <= 0.0:
            raise ValueError(
                "Condenser Stigmator maximum strength must be positive."
            )
        return self

    def quadrupole_strengths_m2(self, z_mm):
        """Return the legacy-compatible continuous +x/-y field pair."""

        z = np.asarray(z_mm, dtype=float)
        if not self.enabled:
            zero = np.zeros_like(z)
            return zero, zero
        sigma_mm = max(float(self.length_mm) / 2.355, 1e-12)
        envelope = np.exp(
            -0.5 * ((z - float(self.z_mm)) / sigma_mm) ** 2
        )
        signed_strength = 0.5 * (
            self.strength_x_m2 - self.strength_y_m2
        )
        x_strength = signed_strength * envelope
        return x_strength, -x_strength

    def draw_layout(self):
        return {
            "key": self.key,
            "mechanical_center_above_sample_mm": (
                self.mechanical_center_above_sample_mm
            ),
            "mechanical_length_mm": self.mechanical_length_mm,
            "mechanical_outer_diameter_mm": (
                self.mechanical_outer_diameter_mm
            ),
            "mechanical_clear_bore_diameter_mm": (
                self.mechanical_clear_bore_diameter_mm
            ),
            "shape_profile": self.shape_profile,
        }

    def draw_ray_overlay(self):
        return {
            "key": self.key,
            "optical_reference_z_mm": self.z_mm,
            "effective_length_mm": self.length_mm,
            "strength_x_m2": self.strength_x_m2,
            "strength_y_m2": self.strength_y_m2,
            "enabled": self.enabled,
        }


CONDENSER_STIGMATOR_DEFINITION = CondenserStigmatorDefinition(
    key=CONDENSER_STIGMATOR,
    label="Condenser Stigmator",
    mechanical_center_above_sample_mm=_DEFAULT_CENTER_ABOVE_SAMPLE_MM,
    mechanical_length_mm=float(_DEFAULT_PART["length_mm"]),
    mechanical_outer_diameter_mm=float(
        _DEFAULT_PART["mechanical_outer_diameter_mm"]
    ),
    mechanical_clear_bore_diameter_mm=float(
        _DEFAULT_PART["mechanical_clear_bore_diameter_mm"]
    ),
    optical_reference_z_mm=_DEFAULT_OPTICAL_REFERENCE_Z_MM,
    effective_length_mm=float(_DEFAULT_PART["effective_length_mm"]),
    maximum_strength_m2=300.0,
    colour="#8e24aa",
)


def create_condenser_stigmator():
    return CONDENSER_STIGMATOR_DEFINITION.create_component()


def condenser_stigmator_from_dict(data):
    values = dict(data)
    values["key"] = canonical_stigmator_key(
        values.get("key", "")
    )
    component = create_condenser_stigmator()
    for attribute in (
        "strength_x_percent",
        "strength_y_percent",
        "enabled",
        "colour",
        "max_strength_m2",
    ):
        if attribute in values:
            object.__setattr__(component, attribute, values[attribute])
    component.key = CONDENSER_STIGMATOR
    component.corrector = CONDENSER_STIGMATOR_DEFINITION.owner
    return component.validate()
