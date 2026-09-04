"""Optional electrostatic beam blanker with a physical downstream stop.

This is an illustrative, static paraxial model, not the proprietary Iliad
NanoPulser design. The plate field is represented by its integrated transverse
kick at the plate centre. Open/blanked states do not simulate switching edges,
RF fields, bunch formation or a pulse repetition rate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


ELECTRON_REST_ENERGY_EV = (
    9.1093837015e-31 * 299792458.0**2 / 1.602176634e-19
)


def electrostatic_deflection_rad(
    voltage_v: float, plate_length_mm: float,
    plate_gap_mm: float, beam_voltage_kv: float,
) -> float:
    """Return the signed small-angle plate deflection at nominal beam energy.

    ``voltage_v`` is the potential of the +axis plate minus the -axis plate.
    Electrons therefore deflect toward +axis for a positive voltage. From
    ``delta(theta) = q E L / (p v)``, the voltage-equivalent relativistic
    ``p v / e`` is ``K (K + 2 mc²) / (K + mc²)``, with K and mc² in eV.
    """

    values = tuple(map(float, (
        voltage_v, plate_length_mm, plate_gap_mm, beam_voltage_kv,
    )))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Blanker voltage and geometry must be finite")
    voltage, length, gap, beam_kv = values
    if length <= 0.0 or gap <= 0.0 or beam_kv <= 0.0:
        raise ValueError("Blanker plate length, gap and beam energy must be positive")
    kinetic_ev = beam_kv * 1000.0
    momentum_velocity_ev = (
        kinetic_ev * (kinetic_ev + 2.0 * ELECTRON_REST_ENERGY_EV)
        / (kinetic_ev + ELECTRON_REST_ENERGY_EV)
    )
    return voltage * length / gap / momentum_velocity_ev


@dataclass
class NanoPulser:
    """Operating state plus TOML-resolved geometry for an optional blanker."""

    installed: bool = False
    blanked: bool = False
    voltage_v: float = 500.0
    azimuth_deg: float = 0.0
    z_mm: float = 20.0
    stop_z_mm: float = 60.0
    plate_length_mm: float = 10.0
    plate_gap_mm: float = 1.0
    aperture_radius_mm: float = 0.1
    mechanical_center_from_tip_mm: float = 20.0
    mechanical_length_mm: float = 10.0
    mechanical_outer_diameter_mm: float = 40.0
    mechanical_clear_bore_diameter_mm: float = 1.0

    @property
    def name(self):
        return "NanoPulser electrostatic blanker"

    @property
    def key(self):
        return "nanopulser_deflector"

    @property
    def kind(self):
        return "electrostatic_deflector"

    @property
    def enabled(self):
        return self.installed

    @property
    def optical_reference_from_tip_mm(self):
        return self.z_mm

    @property
    def optical_active(self):
        return self.installed and self.blanked

    @property
    def effective_aperture_radius_mm(self):
        return 0.5 * self.mechanical_clear_bore_diameter_mm

    @property
    def aperture(self):
        # Construct from the owner each time so a geometry rebuild can never
        # leave behind an aperture at a previous module position.
        from temsim.optics.model import Aperture

        return Aperture(
            name="NanoPulser blanking aperture",
            key="nanopulser_aperture",
            z_mm=float(self.stop_z_mm),
            radius_mm=float(self.aperture_radius_mm),
            enabled=bool(self.installed),
        )

    def validate(self):
        if not isinstance(self.installed, bool) or not isinstance(self.blanked, bool):
            raise ValueError("NanoPulser installed and blanked states must be Boolean")
        numeric_fields = (
            "voltage_v", "azimuth_deg", "z_mm", "stop_z_mm",
            "plate_length_mm", "plate_gap_mm", "aperture_radius_mm",
            "mechanical_center_from_tip_mm", "mechanical_length_mm",
            "mechanical_outer_diameter_mm", "mechanical_clear_bore_diameter_mm",
        )
        for name in numeric_fields:
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"NanoPulser {name} must be finite")
        for name in numeric_fields[4:]:
            if name == "mechanical_center_from_tip_mm":
                continue
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"NanoPulser {name} must be positive")
        if self.z_mm < 0.0 or self.mechanical_center_from_tip_mm < 0.0:
            raise ValueError("NanoPulser must follow the source tip")
        if self.stop_z_mm <= self.z_mm + 0.5 * self.plate_length_mm:
            raise ValueError("NanoPulser stop must follow the deflection plates")
        if self.plate_length_mm > self.mechanical_length_mm:
            raise ValueError("NanoPulser plates must fit inside their body")
        if self.mechanical_clear_bore_diameter_mm >= self.mechanical_outer_diameter_mm:
            raise ValueError("NanoPulser bore must fit inside its body")
        return self

    def kick_events(self, beam_voltage_kv):
        """Yield a physical impulse only when the installed beam is blanked.

        The column's nominal kinetic energy sets the deflection calibration;
        the sub-eV source energy distribution does not modify this impulse.
        """

        self.validate()
        if not self.installed or not self.blanked:
            return ()
        angle = electrostatic_deflection_rad(
            self.voltage_v, self.plate_length_mm,
            self.plate_gap_mm, beam_voltage_kv,
        )
        if abs(angle) > 0.1:
            raise ValueError("NanoPulser deflection exceeds the 100 mrad paraxial range")
        azimuth = math.radians(float(self.azimuth_deg))
        return ((
            float(self.z_mm), angle * math.cos(azimuth),
            angle * math.sin(azimuth),
        ),)

    def to_dict(self):
        """Persist operating choices; instrument TOML owns every dimension."""

        return {
            "installed": bool(self.installed),
            "blanked": bool(self.blanked),
            "voltage_v": float(self.voltage_v),
            "azimuth_deg": float(self.azimuth_deg),
        }

    @classmethod
    def from_dict(cls, values=None):
        values = {} if values is None else dict(values)
        allowed = {"installed", "blanked", "voltage_v", "azimuth_deg"}
        return cls(**{key: value for key, value in values.items() if key in allowed}).validate()


def nanopulser_from_dict(values=None):
    return NanoPulser.from_dict(values)


def ensure_nanopulser(state):
    component = getattr(state, "nanopulser", None)
    if component is None:
        component = NanoPulser()
        state.nanopulser = component
    elif isinstance(component, dict):
        component = NanoPulser.from_dict(component)
        state.nanopulser = component
    return component.validate()
