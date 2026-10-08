"""Canonical Mini Condenser with integrated and standalone installations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar

from temsim import module_manifest
from temsim.component_keys import MINI_CONDENSER, require_current_lens_key
from temsim.operating_modes import (
    MINI_CONDENSER_MODE_CONTROL_SCHEMA,
    load_mini_condenser_default_excitation_percent,
)
from temsim.optics.condenser_lens import AxialFieldTerm
from temsim.optics.round_lens import (
    RoundLensComponent,
    restore_round_lens,
)

_DEFAULT_COLUMN_ORIGIN_Z_MM = module_manifest.port_z_mm(
    "gun/FEG.toml", "exit"
)
_INTEGRATED_PART = module_manifest.part_data(
    "column/C3_ProbeCorrector.toml", MINI_CONDENSER
)
_STANDALONE_PART = module_manifest.part_data(
    "column/C3.toml", MINI_CONDENSER
)


@dataclass(frozen=True)
class MiniCondenserGeometry:
    installation: str
    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    bore_diameter_mm: float
    pole_gap_mm: float
    optical_reference_from_tip_mm: float
    owner: str


@dataclass(frozen=True)
class MiniCondenserDefinition:
    key: str
    label: str
    integrated_geometry: MiniCondenserGeometry
    standalone_geometry: MiniCondenserGeometry
    reference_peak_field_t: float
    field_scale_half_width_mm: float
    default_excitation_percent: float
    maximum_excitation_percent: float
    colour: str
    kind: str = "round_lens"
    shape_profile: str = "magnetic_lens_yoke"
    interaction_kind: str = "axial_magnetic_field"

    def geometry_for(self, installation):
        if installation == "integrated":
            return self.integrated_geometry
        if installation == "standalone":
            return self.standalone_geometry
        raise ValueError(f"Unsupported Mini Condenser installation: {installation}")

    def create_component(self):
        integrated = self.integrated_geometry
        standalone = self.standalone_geometry
        return MiniCondenserComponent(
            name=self.label,
            key=self.key,
            z_mm=integrated.optical_reference_from_tip_mm,
            b0_t=self.reference_peak_field_t,
            a_mm=self.field_scale_half_width_mm,
            percent=self.default_excitation_percent,
            max_percent=self.maximum_excitation_percent,
            colour=self.colour,
            gaussian=[
                AxialFieldTerm(0.09, -1.0, 0.90),
                AxialFieldTerm(0.82, 0.0, 0.55),
                AxialFieldTerm(0.09, 1.0, 0.90),
            ],
            enabled=True,
            cs_mm=None,
            cc_mm=None,
            polarity=int(_INTEGRATED_PART["field_polarity"]),
            normalise_profile_peak=False,
            mechanical_center_from_tip_mm=(
                integrated.mechanical_center_from_tip_mm
            ),
            mechanical_length_mm=integrated.mechanical_length_mm,
            mechanical_outer_diameter_mm=(
                integrated.mechanical_outer_diameter_mm
            ),
            bore_diameter_mm=integrated.bore_diameter_mm,
            pole_gap_mm=integrated.pole_gap_mm,
            optical_reference_from_tip_mm=(
                integrated.optical_reference_from_tip_mm
            ),
            corrector="shared_column",
            standalone_mechanical_length_mm=(
                standalone.mechanical_length_mm
            ),
            standalone_mechanical_outer_diameter_mm=(
                standalone.mechanical_outer_diameter_mm
            ),
            standalone_bore_diameter_mm=standalone.bore_diameter_mm,
            standalone_pole_gap_mm=standalone.pole_gap_mm,
            active_installation="integrated",
        )


@dataclass
class MiniCondenserComponent(RoundLensComponent):
    standalone_mechanical_length_mm: float = float(
        _STANDALONE_PART["length_mm"]
    )
    standalone_mechanical_outer_diameter_mm: float = float(
        _STANDALONE_PART["mechanical_outer_diameter_mm"]
    )
    standalone_bore_diameter_mm: float = float(
        _STANDALONE_PART["bore_diameter_mm"]
    )
    standalone_pole_gap_mm: float = float(
        _STANDALONE_PART["pole_gap_mm"]
    )
    active_installation: str = "integrated"

    EXPECTED_KEY: ClassVar[str] = MINI_CONDENSER

    @property
    def signed_excitation_percent(self) -> float:
        """Project control sign: Microprobe positive, Nanoprobe negative.

        This is not an OEM current calibration or a standalone focal power.
        Nanoprobe keeps finite excitation; optical off refers to the coupled
        CM/objective action, not removal of the CM magnetic field.
        """
        return float(self.percent) * int(self.polarity)

    @signed_excitation_percent.setter
    def signed_excitation_percent(self, value: float) -> None:
        if isinstance(value, bool):
            raise ValueError("Mini Condenser signed excitation must be numeric")
        signed = float(value)
        if not math.isfinite(signed) or not 0.0 < abs(signed) <= self.max_percent:
            raise ValueError(
                "Mini Condenser signed excitation must have a finite nonzero "
                "magnitude within its configured range; optical off is not zero current"
            )
        self.percent = abs(signed)
        self.polarity = 1 if signed > 0.0 else -1

    def validate_probe_mode(self, mode_key: str) -> float:
        """Return the requested signed drive without changing the component."""
        if mode_key not in {"micro_probe", "nano_probe"}:
            raise ValueError(f"Unsupported Mini Condenser probe mode: {mode_key}")
        magnitude = float(self.percent)
        if not self.enabled or not math.isfinite(magnitude) or not 0.0 < magnitude <= self.max_percent:
            raise ValueError(
                "Mini Condenser mode switching requires enabled, finite nonzero excitation"
            )
        return magnitude if mode_key == "micro_probe" else -magnitude

    def set_probe_mode(self, mode_key: str):
        """Reverse the control sign; this alone does not qualify probe optics.

        Zero excitation and disabled hardware are rejected explicitly, never
        interpreted as the optical off state of Nanoprobe.
        """
        self.signed_excitation_percent = self.validate_probe_mode(mode_key)
        return self

    def __setattr__(self, name, value):
        ready = self.__dict__.get("_position_coupling_ready", False)
        if name in {
            "mechanical_center_from_tip_mm",
            "optical_reference_from_tip_mm",
        }:
            value = float(value)
        if ready and name == "mechanical_center_from_tip_mm":
            delta = value - float(self.mechanical_center_from_tip_mm)
            object.__setattr__(self, name, value)
            object.__setattr__(
                self,
                "optical_reference_from_tip_mm",
                float(self.optical_reference_from_tip_mm) + delta,
            )
            object.__setattr__(self, "z_mm", float(self.z_mm) + delta)
            return
        if ready and name == "optical_reference_from_tip_mm":
            delta = value - float(self.optical_reference_from_tip_mm)
            object.__setattr__(self, name, value)
            object.__setattr__(self, "z_mm", float(self.z_mm) + delta)
            return
        super().__setattr__(name, value)

    @property
    def owner(self):
        return self.geometry_for(self.active_installation).owner

    @property
    def length_mm(self):
        return self.geometry_for(
            self.active_installation
        ).mechanical_length_mm

    def select_installation(self, installation):
        if installation not in {"integrated", "standalone"}:
            raise ValueError(
                f"Unsupported Mini Condenser installation: {installation}"
            )
        object.__setattr__(self, "active_installation", installation)
        return self

    def geometry_for(self, installation):
        if installation == "integrated":
            return MiniCondenserGeometry(
                installation="integrated",
                mechanical_center_from_tip_mm=(
                    self.mechanical_center_from_tip_mm
                ),
                mechanical_length_mm=self.mechanical_length_mm,
                mechanical_outer_diameter_mm=(
                    self.mechanical_outer_diameter_mm
                ),
                bore_diameter_mm=self.bore_diameter_mm,
                pole_gap_mm=self.pole_gap_mm,
                optical_reference_from_tip_mm=(
                    self.optical_reference_from_tip_mm
                ),
                owner="probe_corrector",
            )
        if installation == "standalone":
            return MiniCondenserGeometry(
                installation="standalone",
                mechanical_center_from_tip_mm=(
                    self.mechanical_center_from_tip_mm
                ),
                mechanical_length_mm=(
                    self.standalone_mechanical_length_mm
                ),
                mechanical_outer_diameter_mm=(
                    self.standalone_mechanical_outer_diameter_mm
                ),
                bore_diameter_mm=self.standalone_bore_diameter_mm,
                pole_gap_mm=self.standalone_pole_gap_mm,
                optical_reference_from_tip_mm=(
                    self.optical_reference_from_tip_mm
                ),
                owner="condenser",
            )
        raise ValueError(f"Unsupported Mini Condenser installation: {installation}")

    def validate(self):
        super().validate()
        standalone = self.geometry_for("standalone")
        if standalone.mechanical_center_from_tip_mm < 0.0:
            raise ValueError(
                "Standalone Mini Condenser centre must follow the tip."
            )
        if standalone.mechanical_length_mm <= 0.0:
            raise ValueError(
                "Standalone Mini Condenser length must be positive."
            )
        if (
            standalone.mechanical_outer_diameter_mm
            <= standalone.bore_diameter_mm
        ):
            raise ValueError(
                "Standalone Mini Condenser outer diameter must exceed its bore."
            )
        if not (
            0.0
            < standalone.pole_gap_mm
            <= standalone.mechanical_length_mm
        ):
            raise ValueError(
                "Standalone Mini Condenser pole gap must fit its body."
            )
        return self

    def draw_layout(self):
        geometry = self.geometry_for(self.active_installation)
        return {
            "key": self.key,
            "installation": geometry.installation,
            "mechanical_center_from_tip_mm": (
                geometry.mechanical_center_from_tip_mm
            ),
            "mechanical_length_mm": geometry.mechanical_length_mm,
            "mechanical_outer_diameter_mm": (
                geometry.mechanical_outer_diameter_mm
            ),
            "bore_diameter_mm": geometry.bore_diameter_mm,
            "pole_gap_mm": geometry.pole_gap_mm,
            "shape_profile": self.shape_profile,
        }


MINI_CONDENSER_DEFINITION = MiniCondenserDefinition(
    key=MINI_CONDENSER,
    label="Mini Condenser Lens",
    integrated_geometry=MiniCondenserGeometry(
        installation="integrated",
        mechanical_center_from_tip_mm=(
            _DEFAULT_COLUMN_ORIGIN_Z_MM
            + float(_INTEGRATED_PART["local_center_z_mm"])
        ),
        mechanical_length_mm=float(_INTEGRATED_PART["length_mm"]),
        mechanical_outer_diameter_mm=float(
            _INTEGRATED_PART["mechanical_outer_diameter_mm"]
        ),
        bore_diameter_mm=float(_INTEGRATED_PART["bore_diameter_mm"]),
        pole_gap_mm=float(_INTEGRATED_PART["pole_gap_mm"]),
        optical_reference_from_tip_mm=(
            _DEFAULT_COLUMN_ORIGIN_Z_MM
            + float(_INTEGRATED_PART["optical_reference_local_z_mm"])
        ),
        owner="probe_corrector",
    ),
    standalone_geometry=MiniCondenserGeometry(
        installation="standalone",
        mechanical_center_from_tip_mm=(
            _DEFAULT_COLUMN_ORIGIN_Z_MM
            + float(_STANDALONE_PART["local_center_z_mm"])
        ),
        mechanical_length_mm=float(_STANDALONE_PART["length_mm"]),
        mechanical_outer_diameter_mm=float(
            _STANDALONE_PART["mechanical_outer_diameter_mm"]
        ),
        bore_diameter_mm=float(_STANDALONE_PART["bore_diameter_mm"]),
        pole_gap_mm=float(_STANDALONE_PART["pole_gap_mm"]),
        optical_reference_from_tip_mm=(
            _DEFAULT_COLUMN_ORIGIN_Z_MM
            + float(_STANDALONE_PART["optical_reference_local_z_mm"])
        ),
        owner="condenser",
    ),
    reference_peak_field_t=0.45,
    field_scale_half_width_mm=8.0,
    default_excitation_percent=load_mini_condenser_default_excitation_percent(),
    maximum_excitation_percent=100.0,
    colour="#0097a7",
)


def create_mini_condenser():
    return MINI_CONDENSER_DEFINITION.create_component()


def mini_condenser_from_dict(data):
    values = dict(data)
    values["key"] = require_current_lens_key(values.get("key"), expected=MINI_CONDENSER)
    component = create_mini_condenser()
    manifest_owned = {
        "z_mm",
        "mechanical_center_from_tip_mm",
        "mechanical_length_mm",
        "mechanical_outer_diameter_mm",
        "bore_diameter_mm",
        "pole_gap_mm",
        "optical_reference_from_tip_mm",
        "standalone_mechanical_length_mm",
        "standalone_mechanical_outer_diameter_mm",
        "standalone_bore_diameter_mm",
        "standalone_pole_gap_mm",
    }
    restored = restore_round_lens(
        component,
        {
            key: value
            for key, value in values.items()
            if key not in manifest_owned
        },
    )
    restored.key = MINI_CONDENSER
    restored.corrector = "shared_column"
    return restored.validate()
