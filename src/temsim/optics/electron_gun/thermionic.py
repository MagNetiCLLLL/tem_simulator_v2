"""Physics-based thermionic source using the common gun-exit contract.

The emission boundary combines Richardson-Laue-Dushman supply, Schottky
barrier lowering and the Child-Langmuir space-charge limit.  Launch positions
and velocities sample the flux-weighted planar Maxwell-Boltzmann distribution;
the resulting rays continue through finite gun fields with relativistic Boris
integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from temsim import module_manifest
from temsim.component_keys import (
    THERMIONIC_ACCELERATOR,
    THERMIONIC_ANODE_APERTURE,
    THERMIONIC_C1_APERTURE,
    THERMIONIC_CATHODE,
    THERMIONIC_DEFLECTOR,
    THERMIONIC_GUN_LENS,
    THERMIONIC_STIGMATOR,
    THERMIONIC_WEHNELT,
)
from temsim.optics.electron_gun.alignment import GunDeflector, GunStigmator
from temsim.optics.electron_gun.aperture import GunAperture
from temsim.optics.electron_gun.base import EmissionBundle
from temsim.optics.electron_gun.electrostatic import (
    AcceleratorColumn,
    AcceleratorStage,
    ElectrostaticGunLens,
    ExtractorElectrode,
)
from temsim.optics.electron_gun.emitter import (
    _halton_dimensions,
)
from temsim.optics.electron_gun.field_emission import (
    FieldEmissionGun,
    _apply_part_geometry,
    _restore_component_settings,
)
from temsim.physics.relativistic_lorentz import (
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
    SPEED_OF_LIGHT_M_PER_S,
)


BOLTZMANN_CONSTANT_J_PER_K = 1.380649e-23
VACUUM_PERMITTIVITY_F_PER_M = 8.8541878128e-12
GAMMA_SHAPE_2_FWHM_FACTOR = 2.446386037
_THERMIONIC_MODULE_PATH = "gun/Thermionic.toml"


@dataclass
class ThermionicEmitter:
    """Physics-based planar thermionic-cathode emission model."""

    mechanical_center_from_tip_mm: float
    mechanical_length_mm: float
    mechanical_outer_diameter_mm: float
    name: str = "LaB6 Thermionic Cathode"
    key: str = THERMIONIC_CATHODE
    cathode_material: str = "LaB6"
    cathode_temperature_k: float = 1800.0
    work_function_ev: float = 2.7
    richardson_constant_a_cm2_k2: float = 29.0
    vacuum_pa: float = 1.0e-5
    emitting_radius_um: float = 10.0
    cathode_anode_gap_mm: float = 1.0
    extraction_field_scale: float = 1.0
    ray_count: int = 1000

    @property
    def label(self):
        return self.name

    @property
    def effective_aperture_radius_mm(self):
        return 0.5 * self.mechanical_outer_diameter_mm

    @property
    def emitted_current_a(self):
        return self.emission_diagnostics(0.0, 0.0)[
            "temperature_limited_current_a"
        ]

    @property
    def emission_energy_ev(self):
        """Mean kinetic energy of the emitted flux (Gamma shape 2)."""

        return 2.0 * self.thermal_energy_ev

    @property
    def energy_spread_fwhm_ev(self):
        return GAMMA_SHAPE_2_FWHM_FACTOR * self.thermal_energy_ev

    @property
    def thermal_energy_ev(self):
        return (
            BOLTZMANN_CONSTANT_J_PER_K
            * float(self.cathode_temperature_k)
            / ELEMENTARY_CHARGE_C
        )

    @property
    def emitting_area_m2(self):
        return math.pi * (float(self.emitting_radius_um) * 1.0e-6) ** 2

    @staticmethod
    def schottky_lowering_ev(field_v_per_m):
        field = max(0.0, float(field_v_per_m))
        return math.sqrt(
            ELEMENTARY_CHARGE_C
            * field
            / (4.0 * math.pi * VACUUM_PERMITTIVITY_F_PER_M)
        )

    def temperature_limited_current_density_a_m2(self, field_v_per_m):
        lowering_ev = self.schottky_lowering_ev(field_v_per_m)
        effective_work_function_ev = max(
            0.0, float(self.work_function_ev) - lowering_ev
        )
        exponent = -effective_work_function_ev / max(
            self.thermal_energy_ev, 1.0e-30
        )
        richardson_a_m2_k2 = (
            float(self.richardson_constant_a_cm2_k2) * 1.0e4
        )
        return (
            richardson_a_m2_k2
            * float(self.cathode_temperature_k) ** 2
            * math.exp(max(exponent, -745.0))
        )

    @staticmethod
    def child_langmuir_current_density_a_m2(
        extraction_voltage_v, gap_m
    ):
        voltage = max(0.0, float(extraction_voltage_v))
        gap = max(float(gap_m), 1.0e-12)
        return (
            (4.0 / 9.0)
            * VACUUM_PERMITTIVITY_F_PER_M
            * math.sqrt(
                2.0 * ELEMENTARY_CHARGE_C / ELECTRON_MASS_KG
            )
            * voltage**1.5
            / gap**2
        )

    def emission_diagnostics(
        self, cathode_field_v_per_m, extraction_voltage_v
    ):
        gap_m = float(self.cathode_anode_gap_mm) * 1.0e-3
        temperature_limited = (
            self.temperature_limited_current_density_a_m2(
                cathode_field_v_per_m
            )
        )
        space_charge_limited = (
            self.child_langmuir_current_density_a_m2(
                extraction_voltage_v, gap_m
            )
        )
        active_density = min(
            temperature_limited, space_charge_limited
        )
        area = self.emitting_area_m2
        return {
            "cathode_field_v_per_m": max(
                0.0, float(cathode_field_v_per_m)
            ),
            "extraction_voltage_v": max(
                0.0, float(extraction_voltage_v)
            ),
            "schottky_lowering_ev": self.schottky_lowering_ev(
                cathode_field_v_per_m
            ),
            "effective_work_function_ev": max(
                0.0,
                float(self.work_function_ev)
                - self.schottky_lowering_ev(cathode_field_v_per_m),
            ),
            "temperature_limited_current_density_a_m2": (
                temperature_limited
            ),
            "space_charge_limited_current_density_a_m2": (
                space_charge_limited
            ),
            "active_current_density_a_m2": active_density,
            "temperature_limited_current_a": temperature_limited * area,
            "space_charge_limited_current_a": space_charge_limited * area,
            "emitted_current_a": active_density * area,
            "limiting_regime": (
                "temperature"
                if temperature_limited <= space_charge_limited
                else "space_charge"
            ),
            "mean_emission_energy_ev": self.emission_energy_ev,
            "energy_spread_fwhm_ev": self.energy_spread_fwhm_ev,
            "normalized_rms_emittance_m_rad": (
                float(self.emitting_radius_um)
                * 0.5e-6
                * math.sqrt(
                    self.thermal_energy_ev
                    / (
                        ELECTRON_MASS_KG
                        * SPEED_OF_LIGHT_M_PER_S**2
                        / ELEMENTARY_CHARGE_C
                    )
                )
            ),
        }

    @property
    def optical_reference_from_tip_mm(self):
        return 0.0

    @property
    def kind(self):
        return "thermionic_cathode"

    @property
    def shape_profile(self):
        return "thermionic_cathode"

    def validate(self):
        for attribute in ("work_function_ev", "vacuum_pa"):
            if float(getattr(self, attribute)) < 0.0:
                raise ValueError(
                    f"{self.name} {attribute} must not be negative."
                )
        for attribute in (
            "cathode_temperature_k",
            "richardson_constant_a_cm2_k2",
            "emitting_radius_um",
            "cathode_anode_gap_mm",
        ):
            if float(getattr(self, attribute)) <= 0.0:
                raise ValueError(
                    f"{self.name} {attribute} must be positive."
                )
        if float(self.extraction_field_scale) < 0.0:
            raise ValueError(
                f"{self.name} extraction_field_scale must not be negative."
            )
        if int(self.ray_count) < 9:
            raise ValueError("Thermionic ray count must be at least 9.")
        if self.mechanical_length_mm <= 0.0:
            raise ValueError("Thermionic cathode length must be positive.")
        return self

    def emit(self, count: int | None = None) -> EmissionBundle:
        self.validate()
        n = int(self.ray_count if count is None else count)
        if n < 9:
            raise ValueError("Thermionic emission requires at least 9 rays.")
        u_r, u_phi, u_v, u_theta, u_normal = _halton_dimensions(
            n, (2, 3, 5, 7, 11)
        )
        radius_m = (
            float(self.emitting_radius_um) * 1.0e-6 * np.sqrt(u_r)
        )
        azimuth = 2.0 * np.pi * u_phi
        x = radius_m * np.cos(azimuth)
        y = radius_m * np.sin(azimuth)

        thermal_velocity_sigma = math.sqrt(
            BOLTZMANN_CONSTANT_J_PER_K
            * float(self.cathode_temperature_k)
            / ELECTRON_MASS_KG
        )
        transverse_radius = thermal_velocity_sigma * np.sqrt(
            -2.0 * np.log(np.maximum(u_v, 1.0e-15))
        )
        transverse_azimuth = 2.0 * np.pi * u_theta
        velocity_x = transverse_radius * np.cos(transverse_azimuth)
        velocity_y = transverse_radius * np.sin(transverse_azimuth)
        # Flux weighting multiplies the half-Maxwellian by v_z, yielding a
        # Rayleigh distribution for the positive normal velocity.
        velocity_z = thermal_velocity_sigma * np.sqrt(
            -2.0 * np.log(np.maximum(1.0 - u_normal, 1.0e-15))
        )
        tx = velocity_x / velocity_z
        ty = velocity_y / velocity_z
        kinetic_energy_ev = (
            0.5
            * ELECTRON_MASS_KG
            * (
                velocity_x**2
                + velocity_y**2
                + velocity_z**2
            )
            / ELEMENTARY_CHARGE_C
        )
        energy = kinetic_energy_ev - self.emission_energy_ev
        return EmissionBundle(
            x_m=x,
            y_m=y,
            tx_rad=tx,
            ty_rad=ty,
            energy_offset_ev=energy,
            weight=np.full(n, 1.0 / n, dtype=float),
            ray_id=np.arange(n, dtype=np.int64),
        )

    def draw_layout(self):
        return {
            "key": self.key,
            "mechanical_center_from_tip_mm": self.mechanical_center_from_tip_mm,
            "mechanical_length_mm": self.mechanical_length_mm,
            "mechanical_outer_diameter_mm": self.mechanical_outer_diameter_mm,
            "shape_profile": self.shape_profile,
        }


def _part(key):
    return (
        module_manifest.part_geometry(_THERMIONIC_MODULE_PATH, key),
        module_manifest.part_data(_THERMIONIC_MODULE_PATH, key),
    )


def _emitter():
    geometry, part = _part(THERMIONIC_CATHODE)
    return ThermionicEmitter(
        mechanical_center_from_tip_mm=geometry.center_z_mm,
        mechanical_length_mm=geometry.length_mm,
        mechanical_outer_diameter_mm=float(part["outer_diameter_mm"]),
    )


def _wehnelt():
    geometry, part = _part(THERMIONIC_WEHNELT)
    return ExtractorElectrode(
        name="Wehnelt / Extraction Electrode",
        key=THERMIONIC_WEHNELT,
        mechanical_center_from_tip_mm=geometry.center_z_mm,
        mechanical_length_mm=geometry.length_mm,
        mechanical_outer_diameter_mm=float(part["outer_diameter_mm"]),
        mechanical_clear_bore_diameter_mm=float(
            part["bore_diameter_mm"]
        ),
        voltage_kv=0.8,
        transition_start_mm=0.1,
        transition_end_mm=8.0,
    )


def _gun_lens():
    geometry, part = _part(THERMIONIC_GUN_LENS)
    return ElectrostaticGunLens(
        name="Thermionic Gun Lens",
        key=THERMIONIC_GUN_LENS,
        mechanical_center_from_tip_mm=geometry.center_z_mm,
        mechanical_length_mm=geometry.length_mm,
        mechanical_outer_diameter_mm=float(part["outer_diameter_mm"]),
        mechanical_clear_bore_diameter_mm=float(
            part["bore_diameter_mm"]
        ),
        voltage_kv=1.0,
        potential_scale=3.0,
        soft_edge_mm=2.0,
    )


def _anode_aperture():
    geometry, part = _part(THERMIONIC_ANODE_APERTURE)
    return GunAperture(
        name="Thermionic Anode Aperture",
        key=THERMIONIC_ANODE_APERTURE,
        mechanical_center_from_tip_mm=geometry.center_z_mm,
        mechanical_length_mm=geometry.length_mm,
        mechanical_outer_diameter_mm=float(part["outer_diameter_mm"]),
        mechanical_bore_diameter_mm=float(part["bore_diameter_mm"]),
        plate_thickness_mm=float(part["active_length_mm"]),
        radius_mm=2.0,
        maximum_radius_mm=3.0,
        colour="#ef6c00",
    )


def _accelerator():
    geometry, part = _part(THERMIONIC_ACCELERATOR)
    centers = [float(value) for value in part["stage_centers_z_mm"]]
    return AcceleratorColumn(
        name="Thermionic Accelerator Tube",
        key=THERMIONIC_ACCELERATOR,
        mechanical_center_from_tip_mm=geometry.center_z_mm,
        mechanical_length_mm=geometry.length_mm,
        mechanical_outer_diameter_mm=float(part["outer_diameter_mm"]),
        mechanical_clear_bore_diameter_mm=float(
            part["bore_diameter_mm"]
        ),
        stages=[
            AcceleratorStage(
                center,
                float(index + 1) / len(centers),
                4.0,
            )
            for index, center in enumerate(centers)
        ],
    )


def _deflector():
    geometry, part = _part(THERMIONIC_DEFLECTOR)
    centers = [
        float(value)
        for value in part["interaction_centers_local_z_mm"]
    ]
    return GunDeflector(
        name="Thermionic Gun Deflector Pair",
        key=THERMIONIC_DEFLECTOR,
        mechanical_center_from_tip_mm=geometry.center_z_mm,
        mechanical_length_mm=geometry.length_mm,
        mechanical_outer_diameter_mm=float(part["outer_diameter_mm"]),
        mechanical_clear_bore_diameter_mm=float(
            part["bore_diameter_mm"]
        ),
        upper_center_from_tip_mm=centers[0],
        lower_center_from_tip_mm=centers[1],
        coil_length_mm=float(part["active_length_mm"]),
    )


def _stigmator():
    geometry, part = _part(THERMIONIC_STIGMATOR)
    return GunStigmator(
        name="Thermionic Gun Stigmator",
        key=THERMIONIC_STIGMATOR,
        mechanical_center_from_tip_mm=geometry.center_z_mm,
        mechanical_length_mm=geometry.length_mm,
        mechanical_outer_diameter_mm=float(part["outer_diameter_mm"]),
        mechanical_clear_bore_diameter_mm=float(
            part["bore_diameter_mm"]
        ),
        effective_length_mm=float(part["active_length_mm"]),
    )


def _c1_aperture():
    geometry, part = _part(THERMIONIC_C1_APERTURE)
    return GunAperture(
        name="Thermionic Gun Exit Aperture",
        key=THERMIONIC_C1_APERTURE,
        mechanical_center_from_tip_mm=geometry.center_z_mm,
        mechanical_length_mm=geometry.length_mm,
        mechanical_outer_diameter_mm=float(part["outer_diameter_mm"]),
        mechanical_bore_diameter_mm=float(part["bore_diameter_mm"]),
        plate_thickness_mm=float(part["active_length_mm"]),
        radius_mm=2.0,
        maximum_radius_mm=3.0,
        colour="#fb8c00",
    )


@dataclass
class ThermionicGun(FieldEmissionGun):
    """Thermionic gun family with its own persistent component instances."""

    emitter: ThermionicEmitter = field(default_factory=_emitter)
    extractor: ExtractorElectrode = field(default_factory=_wehnelt)
    electrostatic_lens: ElectrostaticGunLens = field(default_factory=_gun_lens)
    dpa_aperture: GunAperture = field(default_factory=_anode_aperture)
    accelerator: AcceleratorColumn = field(default_factory=_accelerator)
    deflector: GunDeflector = field(default_factory=_deflector)
    stigmator: GunStigmator = field(default_factory=_stigmator)
    c1_aperture: GunAperture = field(default_factory=_c1_aperture)
    monochromator: object = field(default=None, init=False, repr=False)

    type_key = "thermionic"
    display_name = "Thermionic source (LaB6)"

    @property
    def cathode_extraction_voltage_v(self):
        return max(0.0, float(self.extractor.voltage_kv) * 1000.0)

    @property
    def cathode_extraction_field_v_per_m(self):
        gap_m = max(
            float(self.emitter.cathode_anode_gap_mm) * 1.0e-3,
            1.0e-12,
        )
        return (
            self.cathode_extraction_voltage_v
            / gap_m
            * float(self.emitter.extraction_field_scale)
        )

    @property
    def emission_diagnostics(self):
        return self.emitter.emission_diagnostics(
            self.cathode_extraction_field_v_per_m,
            self.cathode_extraction_voltage_v,
        )

    @property
    def emitted_current_a(self):
        return self.emission_diagnostics["emitted_current_a"]

    def apply_manifest_geometry(self, monochromator_installed=None):
        for component in self.base_components:
            _apply_part_geometry(component, _THERMIONIC_MODULE_PATH)
        self._resolved_exit_plane_z_mm = module_manifest.port_z_mm(
            _THERMIONIC_MODULE_PATH, "exit"
        )
        return self

    @property
    def exit_plane_z_mm(self):
        resolved = getattr(self, "_resolved_exit_plane_z_mm", None)
        if resolved is not None:
            return float(resolved)
        return module_manifest.port_z_mm(
            _THERMIONIC_MODULE_PATH, "exit"
        )

    def to_dict(self):
        payload = super().to_dict()
        payload["integrator"]["model"] = "thermionic_rld_cl_boris"
        return payload


def thermionic_gun_from_dict(data=None):
    if data is None:
        return ThermionicGun()
    values = dict(data)
    if values.get("type", "thermionic") != "thermionic":
        raise ValueError("ThermionicGun data must have type 'thermionic'.")
    gun = ThermionicGun()
    component_data = dict(values.get("components", {}))
    for component in gun.components:
        row = component_data.get(component.key)
        if row is None:
            raise ValueError(
                f"Missing thermionic-gun component: {component.key}"
            )
        _restore_component_settings(component, row)
    integrator = dict(values.get("integrator", {}))
    if integrator.get("method", "boris") != "boris":
        raise ValueError("Thermionic gun integrator must be Boris.")
    gun.trace_step_mm = float(
        integrator.get("trace_step_mm", gun.trace_step_mm)
    )
    gun.drift_step_mm = float(
        integrator.get("drift_step_mm", gun.drift_step_mm)
    )
    gun.history_step_mm = float(
        integrator.get("history_step_mm", gun.history_step_mm)
    )
    gun.apply_manifest_geometry()
    return gun.validate()
