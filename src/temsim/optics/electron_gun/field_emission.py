"""Canonical cold field-emission gun assembly."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections import OrderedDict
import json

from temsim import module_manifest
from temsim.component_keys import (
    C1_APERTURE,
    FEG_ACCELERATOR,
    FEG_DEFLECTOR,
    FEG_ELECTROSTATIC_LENS,
    FEG_EXTRACTOR,
    FEG_MONOCHROMATOR_WIEN,
    FEG_STIGMATOR,
    FEG_TIP,
    GUN_EXTRACTOR_APERTURE,
    THERMIONIC_ACCELERATOR,
    THERMIONIC_ANODE_APERTURE,
    THERMIONIC_C1_APERTURE,
    THERMIONIC_CATHODE,
    THERMIONIC_DEFLECTOR,
    THERMIONIC_GUN_LENS,
    THERMIONIC_STIGMATOR,
    THERMIONIC_WEHNELT,
)
from temsim.mechanical_axis import (
    MechanicalNestingPermission,
    resolve_mechanical_axis,
)
from temsim.optics.electron_gun.alignment import (
    FegMagneticField,
    GunDeflector,
    GunStigmator,
)
from temsim.optics.electron_gun.aperture import (
    GunAperture,
    create_c1_aperture,
    create_dpa_aperture,
)
from temsim.optics.electron_gun.electrostatic import (
    AcceleratorColumn,
    AcceleratorStage,
    ElectrostaticGunLens,
    ExtractorElectrode,
    FegElectrostaticField,
)
from temsim.optics.electron_gun.emitter import ColdFieldEmitter
from temsim.optics.electron_gun.monochromator import (
    CombinedElectricField,
    CombinedMagneticField,
    WienMonochromatorAssembly,
    monochromator_from_dict,
)
from temsim.optics.electron_gun.tracing import trace_feg_to_exit
import numpy as np


_SHARED_TRACE_CACHE = OrderedDict()
_SHARED_TRACE_CACHE_LIMIT = 16
_FEG_MODULE_PATH = "gun/FEG.toml"
_TOML_GEOMETRY_COMPONENT_KEYS = frozenset((
    FEG_TIP,
    FEG_EXTRACTOR,
    FEG_ELECTROSTATIC_LENS,
    FEG_ACCELERATOR,
    GUN_EXTRACTOR_APERTURE,
    FEG_DEFLECTOR,
    FEG_STIGMATOR,
    C1_APERTURE,
    FEG_MONOCHROMATOR_WIEN,
    THERMIONIC_CATHODE,
    THERMIONIC_WEHNELT,
    THERMIONIC_GUN_LENS,
    THERMIONIC_ACCELERATOR,
    THERMIONIC_ANODE_APERTURE,
    THERMIONIC_DEFLECTOR,
    THERMIONIC_STIGMATOR,
    THERMIONIC_C1_APERTURE,
))
_TOML_GEOMETRY_ATTRIBUTES = frozenset((
    "mechanical_center_from_tip_mm",
    "mechanical_length_mm",
    "mechanical_outer_diameter_mm",
    "mechanical_clear_bore_diameter_mm",
    "mechanical_bore_diameter_mm",
    "plate_thickness_mm",
    "upper_center_from_tip_mm",
    "lower_center_from_tip_mm",
    "coil_length_mm",
    "effective_length_mm",
    "active_length_mm",
))


def _feg_part_geometry(key):
    return module_manifest.part_geometry(_FEG_MODULE_PATH, key)


def _feg_part_data(key):
    return module_manifest.part_data(_FEG_MODULE_PATH, key)


def _create_emitter():
    geometry = _feg_part_geometry(FEG_TIP)
    part = _feg_part_data(FEG_TIP)
    return ColdFieldEmitter(
        mechanical_center_from_tip_mm=geometry.center_z_mm,
        mechanical_length_mm=geometry.length_mm,
        mechanical_outer_diameter_mm=float(part["outer_diameter_mm"]),
    )


def _create_extractor():
    geometry = _feg_part_geometry(FEG_EXTRACTOR)
    part = _feg_part_data(FEG_EXTRACTOR)
    return ExtractorElectrode(
        mechanical_center_from_tip_mm=geometry.center_z_mm,
        mechanical_length_mm=geometry.length_mm,
        mechanical_outer_diameter_mm=float(part["outer_diameter_mm"]),
        mechanical_clear_bore_diameter_mm=float(
            part["bore_diameter_mm"]
        ),
    )


def _create_electrostatic_lens():
    geometry = _feg_part_geometry(FEG_ELECTROSTATIC_LENS)
    part = _feg_part_data(FEG_ELECTROSTATIC_LENS)
    return ElectrostaticGunLens(
        mechanical_center_from_tip_mm=geometry.center_z_mm,
        mechanical_length_mm=geometry.length_mm,
        mechanical_outer_diameter_mm=float(part["outer_diameter_mm"]),
        mechanical_clear_bore_diameter_mm=float(
            part["bore_diameter_mm"]
        ),
    )


def _create_accelerator():
    geometry = _feg_part_geometry(FEG_ACCELERATOR)
    part = _feg_part_data(FEG_ACCELERATOR)
    centers = [float(value) for value in part["stage_centers_z_mm"]]
    return AcceleratorColumn(
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


def _create_deflector():
    geometry = _feg_part_geometry(FEG_DEFLECTOR)
    part = _feg_part_data(FEG_DEFLECTOR)
    centers = [
        float(value)
        for value in part["interaction_centers_local_z_mm"]
    ]
    if len(centers) != 2:
        raise ValueError("FEG deflector requires two interaction centres.")
    return GunDeflector(
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


def _create_stigmator():
    geometry = _feg_part_geometry(FEG_STIGMATOR)
    part = _feg_part_data(FEG_STIGMATOR)
    return GunStigmator(
        mechanical_center_from_tip_mm=geometry.center_z_mm,
        mechanical_length_mm=geometry.length_mm,
        mechanical_outer_diameter_mm=float(part["outer_diameter_mm"]),
        mechanical_clear_bore_diameter_mm=float(
            part["bore_diameter_mm"]
        ),
        effective_length_mm=float(part["active_length_mm"]),
    )


def _apply_part_geometry(component, module_path):
    geometry = module_manifest.part_geometry(module_path, component.key)
    part = module_manifest.part_data(module_path, component.key)
    component.mechanical_center_from_tip_mm = geometry.center_z_mm
    component.mechanical_length_mm = geometry.length_mm
    if (
        "optical_reference_local_z_mm" in part
        and hasattr(component, "optical_reference_from_tip_mm")
    ):
        descriptor = getattr(
            type(component), "optical_reference_from_tip_mm", None
        )
        if not (
            isinstance(descriptor, property)
            and descriptor.fset is None
        ):
            component.optical_reference_from_tip_mm = float(
                part["optical_reference_local_z_mm"]
            )
    component.mechanical_outer_diameter_mm = float(
        part["outer_diameter_mm"]
    )
    if hasattr(component, "mechanical_clear_bore_diameter_mm"):
        component.mechanical_clear_bore_diameter_mm = float(
            part["bore_diameter_mm"]
        )
    if hasattr(component, "mechanical_bore_diameter_mm"):
        component.mechanical_bore_diameter_mm = float(
            part["bore_diameter_mm"]
        )
    if isinstance(component, AcceleratorColumn):
        centers = [
            float(value) for value in part["stage_centers_z_mm"]
        ]
        if len(centers) != len(component.stages):
            raise ValueError(
                "FEG accelerator TOML stage count changed; "
                "restart the state to rebuild its physics records."
            )
        for stage, center in zip(component.stages, centers):
            stage.center_from_tip_mm = center
    elif isinstance(component, GunAperture):
        component.plate_thickness_mm = float(part["active_length_mm"])
    elif isinstance(component, GunDeflector):
        centers = [
            float(value)
            for value in part["interaction_centers_local_z_mm"]
        ]
        if len(centers) != 2:
            raise ValueError(
                "FEG deflector requires two interaction centres."
            )
        component.upper_center_from_tip_mm = centers[0]
        component.lower_center_from_tip_mm = centers[1]
        component.coil_length_mm = float(part["active_length_mm"])
    elif isinstance(component, GunStigmator):
        component.effective_length_mm = float(part["active_length_mm"])
    elif component.key == FEG_MONOCHROMATOR_WIEN:
        component.active_length_mm = float(part["active_length_mm"])
    return component


def _component_payload(component):
    payload = asdict(component)
    if component.key in _TOML_GEOMETRY_COMPONENT_KEYS:
        for attribute in _TOML_GEOMETRY_ATTRIBUTES:
            payload.pop(attribute, None)
    if component.key in {
        FEG_ACCELERATOR,
        THERMIONIC_ACCELERATOR,
    }:
        for stage in payload["stages"]:
            stage.pop("center_from_tip_mm", None)
    return payload


def _restore_component_settings(component, row):
    allowed = component.__dataclass_fields__
    for attribute, value in row.items():
        if attribute not in allowed:
            continue
        if (
            component.key in _TOML_GEOMETRY_COMPONENT_KEYS
            and attribute in _TOML_GEOMETRY_ATTRIBUTES
        ):
            continue
        if (
            isinstance(component, AcceleratorColumn)
            and attribute == "stages"
        ):
            if len(value) != len(component.stages):
                raise ValueError(
                    "Saved accelerator stage count does not match TOML."
                )
            for stage, saved_stage in zip(component.stages, value):
                for stage_attribute, stage_value in saved_stage.items():
                    if stage_attribute == "center_from_tip_mm":
                        continue
                    if (
                        stage_attribute
                        in AcceleratorStage.__dataclass_fields__
                    ):
                        setattr(stage, stage_attribute, stage_value)
        else:
            setattr(component, attribute, value)
    return component


@dataclass
class FieldEmissionGun:
    emitter: ColdFieldEmitter = field(default_factory=_create_emitter)
    extractor: ExtractorElectrode = field(default_factory=_create_extractor)
    electrostatic_lens: ElectrostaticGunLens = field(
        default_factory=_create_electrostatic_lens
    )
    dpa_aperture: GunAperture = field(default_factory=create_dpa_aperture)
    accelerator: AcceleratorColumn = field(default_factory=_create_accelerator)
    deflector: GunDeflector = field(default_factory=_create_deflector)
    stigmator: GunStigmator = field(default_factory=_create_stigmator)
    c1_aperture: GunAperture = field(default_factory=create_c1_aperture)
    monochromator: WienMonochromatorAssembly = field(
        default_factory=WienMonochromatorAssembly
    )
    trace_step_mm: float = 0.2
    drift_step_mm: float = 2.0
    history_step_mm: float = 2.0

    type_key = "cold_feg"
    display_name = "Cold field emission gun (FEG)"

    def __post_init__(self):
        self._trace_cache_key = None
        self._trace_cache = None
        self._bind_c1_mechanism()

    def _bind_c1_mechanism(self):
        if self.type_key == "cold_feg" and self.monochromator is not None:
            self.c1_aperture.bind_slit_profile(self.monochromator.slit)
            self.c1_aperture.select_slit_mode(
                self.monochromator_installed
            )
        else:
            self.c1_aperture.select_slit_mode(False)
        return self.c1_aperture

    def apply_manifest_geometry(self, monochromator_installed=None):
        installed = (
            self.monochromator_installed
            if monochromator_installed is None
            else bool(monochromator_installed)
        )
        module_path = (
            "gun/FEG_Mono.toml" if installed else _FEG_MODULE_PATH
        )
        for component in self.base_components:
            _apply_part_geometry(component, module_path)
        if installed:
            _apply_part_geometry(self.monochromator.wien, module_path)
        return self

    @property
    def base_components(self):
        return (
            self.emitter,
            self.extractor,
            self.electrostatic_lens,
            self.dpa_aperture,
            self.accelerator,
            self.deflector,
            self.stigmator,
            self.c1_aperture,
        )

    @property
    def monochromator_installed(self):
        return (
            self.type_key == "cold_feg"
            and self.monochromator is not None
            and bool(self.monochromator.installed)
        )

    @property
    def components(self):
        base = self.base_components
        if not self.monochromator_installed:
            return base
        return (
            *base[:3],
            self.monochromator.wien,
            *base[3:],
        )

    @property
    def bore_components(self):
        components = (
            self.extractor,
            self.electrostatic_lens,
            self.accelerator,
            self.deflector,
            self.stigmator,
        )
        if self.monochromator_installed:
            return (
                *components[:2],
                self.monochromator.wien,
                *components[2:],
            )
        return components

    @property
    def mechanical_axis_order(self):
        order = [
            self.emitter.key,
            self.extractor.key,
            self.electrostatic_lens.key,
        ]
        if self.monochromator_installed:
            order.append(self.monochromator.wien.key)
        order.extend((
            self.accelerator.key,
            self.deflector.key,
            self.stigmator.key,
            self.c1_aperture.key,
        ))
        return tuple(order)

    @property
    def mechanical_nesting_permissions(self):
        return (
            MechanicalNestingPermission(
                self.dpa_aperture.key,
                self.accelerator.key,
                (
                    "The DPA/anode aperture is mounted inside the "
                    "accelerator envelope."
                ),
            ),
        )

    def resolve_mechanical_axis(self):
        return resolve_mechanical_axis(
            self.components,
            self.mechanical_axis_order,
            self.mechanical_nesting_permissions,
        )

    @property
    def exit_plane_z_mm(self):
        module_path = (
            "gun/FEG_Mono.toml"
            if self.monochromator_installed
            else _FEG_MODULE_PATH
        )
        return module_manifest.port_z_mm(module_path, "exit")

    @property
    def nominal_exit_energy_ev(self):
        return self.accelerator.high_tension_kv * 1000.0

    @property
    def high_tension_kv(self):
        return self.accelerator.high_tension_kv

    @high_tension_kv.setter
    def high_tension_kv(self, value):
        self.accelerator.high_tension_kv = float(value)

    @property
    def emitted_current_a(self):
        return self.emitter.emitted_current_a

    @property
    def ray_count(self):
        return int(self.emitter.ray_count)

    @property
    def diagnostic_waist_region_mm(self):
        lens = self.electrostatic_lens
        start = (
            lens.optical_reference_from_tip_mm
            + 0.5 * lens.mechanical_length_mm
            + lens.soft_edge_mm
        )
        return start, self.exit_plane_z_mm

    @property
    def electric_field(self):
        base = FegElectrostaticField(
            self.emitter,
            self.extractor,
            self.electrostatic_lens,
            self.accelerator,
        )
        if not self.monochromator_installed:
            return base
        return CombinedElectricField(
            base, self.monochromator.field_provider
        )

    @property
    def magnetic_field(self):
        base = FegMagneticField(self.deflector, self.stigmator)
        if not self.monochromator_installed:
            return base
        return CombinedMagneticField(
            base, self.monochromator.field_provider
        )

    def component(self, key):
        return next(item for item in self.components if item.key == key)

    def validate(self):
        for component in self.base_components:
            component.validate()
        if self.monochromator is not None:
            self.monochromator.validate()
        self._bind_c1_mechanism()
        if (
            self.trace_step_mm <= 0.0
            or self.drift_step_mm <= 0.0
            or self.history_step_mm <= 0.0
        ):
            raise ValueError("Electron-gun tracing steps must be positive.")
        if self.dpa_aperture.z_mm >= self.c1_aperture.z_mm:
            raise ValueError("DPA aperture must precede C1 aperture.")
        return self

    def emit(self, count=None):
        return self.emitter.emit(count)

    def _cache_key(self, count):
        payload = self.to_dict()
        payload["requested_count"] = count
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def trace_to_exit(self, count=None):
        self.validate()
        key = self._cache_key(count)
        if key != self._trace_cache_key:
            cached = _SHARED_TRACE_CACHE.get(key)
            if cached is None:
                cached = trace_feg_to_exit(self, count)
                _SHARED_TRACE_CACHE[key] = cached
                while len(_SHARED_TRACE_CACHE) > _SHARED_TRACE_CACHE_LIMIT:
                    _SHARED_TRACE_CACHE.popitem(last=False)
            else:
                _SHARED_TRACE_CACHE.move_to_end(key)
            self._trace_cache = cached
            self._trace_cache_key = key
        return self._trace_cache

    @property
    def local_wien_reference_energy_ev(self):
        if self.type_key != "cold_feg" or self.monochromator is None:
            raise ValueError("Only a cold FEG can own a monochromator.")
        base = FegElectrostaticField(
            self.emitter,
            self.extractor,
            self.electrostatic_lens,
            self.accelerator,
        )
        position = np.array([[
            0.0,
            0.0,
            self.monochromator.wien.optical_reference_from_tip_mm * 1.0e-3,
        ]])
        potential_v = float(
            base.potential_v_at_global_positions(position)[0]
        )
        return max(
            0.0,
            float(self.emitter.emission_energy_ev) + potential_v,
        )

    def match_monochromator_to_local_energy(self):
        if self.type_key != "cold_feg" or self.monochromator is None:
            raise ValueError("Only a cold FEG can own a monochromator.")
        return self.monochromator.match_to_energy(
            self.local_wien_reference_energy_ev
        )

    def match_monochromator_to_current_ht(self):
        """Compatibility alias; the upstream Wien matches its local energy."""

        return self.match_monochromator_to_local_energy()

    def install_monochromator(self):
        """Load the installed FEG geometry from its TOML module."""

        if self.type_key != "cold_feg" or self.monochromator is None:
            raise ValueError("Only a cold FEG can install a monochromator.")
        if self.monochromator_installed:
            return self.monochromator
        self.monochromator.installation_model_version = 3
        self.monochromator.installed = True
        self.monochromator.accelerator_restore_profile = None
        self.apply_manifest_geometry(True)
        self._bind_c1_mechanism()
        self.validate()
        return self.monochromator

    def remove_monochromator(self):
        """Load the uninstalled FEG geometry from its TOML module."""

        if self.type_key != "cold_feg" or self.monochromator is None:
            return None
        self.monochromator.installed = False
        self.monochromator.accelerator_restore_profile = None
        self.apply_manifest_geometry(False)
        self._bind_c1_mechanism()
        self.validate()
        return self.monochromator

    def migrate_legacy_monochromator_bay(self):
        """Discard legacy geometry and reload the selected TOML module."""

        if self.monochromator is None:
            return None
        self.monochromator.installation_model_version = 3
        self.monochromator.accelerator_restore_profile = None
        self.monochromator.installed = True
        self.apply_manifest_geometry(True)
        self._bind_c1_mechanism()
        self.validate()
        return self.monochromator

    @property
    def field_supports_mm(self):
        lens = self.electrostatic_lens
        supports = [
            (
                self.extractor.transition_start_mm,
                self.extractor.transition_end_mm,
            ),
            (
                lens.optical_reference_from_tip_mm
                - 0.5 * lens.mechanical_length_mm
                - lens.soft_edge_mm,
                lens.optical_reference_from_tip_mm
                + 0.5 * lens.mechanical_length_mm
                + lens.soft_edge_mm,
            ),
        ]
        supports.extend(
            (
                stage.center_from_tip_mm
                + self.accelerator.field_center_offset_mm
                - stage.soft_edge_mm,
                stage.center_from_tip_mm
                + self.accelerator.field_center_offset_mm
                + stage.soft_edge_mm,
            )
            for stage in self.accelerator.stages
        )
        deflector = self.deflector
        half_deflector = (
            0.5 * deflector.coil_length_mm + deflector.soft_edge_mm
        )
        supports.extend((
            (
                deflector.upper_center_from_tip_mm
                + deflector.field_center_offset_mm
                - half_deflector,
                deflector.upper_center_from_tip_mm
                + deflector.field_center_offset_mm
                + half_deflector,
            ),
            (
                deflector.lower_center_from_tip_mm
                + deflector.field_center_offset_mm
                - half_deflector,
                deflector.lower_center_from_tip_mm
                + deflector.field_center_offset_mm
                + half_deflector,
            ),
        ))
        stigmator = self.stigmator
        half_stigmator = (
            0.5 * stigmator.effective_length_mm + stigmator.soft_edge_mm
        )
        supports.append((
            stigmator.optical_reference_from_tip_mm - half_stigmator,
            stigmator.optical_reference_from_tip_mm + half_stigmator,
        ))
        if self.monochromator_installed:
            supports.append(self.monochromator.wien.field_support_mm)
        return tuple(sorted(supports))

    def integration_step_mm_at(self, z_mm):
        z = float(z_mm)
        if (
            self.monochromator_installed
            and self.monochromator.wien.field_support_mm[0]
            <= z
            <= self.monochromator.wien.field_support_mm[1]
        ):
            return min(
                self.trace_step_mm,
                self.monochromator.trace_step_mm,
            )
        for start, end in self.field_supports_mm:
            if start <= z <= end:
                return self.trace_step_mm
            if z < start:
                return min(self.drift_step_mm, max(start - z, self.trace_step_mm))
        return self.drift_step_mm

    def draw_layout(self):
        return tuple(component.draw_layout() for component in self.components)

    def draw_ray_overlay(self):
        return self.trace_to_exit()

    def to_dict(self):
        payload = {
            "type": self.type_key,
            "integrator": {
                "method": "boris",
                "trace_step_mm": self.trace_step_mm,
                "drift_step_mm": self.drift_step_mm,
                "history_step_mm": self.history_step_mm,
            },
            "components": {
                component.key: _component_payload(component)
                for component in self.base_components
            },
        }
        if self.type_key == "cold_feg" and self.monochromator is not None:
            payload["monochromator"] = self.monochromator.to_dict()
        return payload


def field_emission_gun_from_dict(data=None):
    if data is None:
        return FieldEmissionGun()
    values = dict(data)
    if values.get("type", "cold_feg") != "cold_feg":
        raise ValueError("FieldEmissionGun data must have type 'cold_feg'.")
    gun = FieldEmissionGun()
    gun.monochromator = monochromator_from_dict(
        values.get("monochromator")
    )
    gun._bind_c1_mechanism()
    component_data = dict(values.get("components", {}))
    for component in gun.base_components:
        row = component_data.get(component.key)
        if row is None:
            raise ValueError(f"Missing electron-gun component: {component.key}")
        _restore_component_settings(component, row)
    integrator = dict(values.get("integrator", {}))
    if integrator.get("method", "boris") != "boris":
        raise ValueError("Production FEG integrator must be Boris.")
    gun.trace_step_mm = float(
        integrator.get("trace_step_mm", gun.trace_step_mm)
    )
    gun.drift_step_mm = float(
        integrator.get("drift_step_mm", gun.drift_step_mm)
    )
    gun.history_step_mm = float(
        integrator.get("history_step_mm", gun.history_step_mm)
    )
    gun._bind_c1_mechanism()
    if (
        gun.monochromator_installed
        and (
            int(gun.monochromator.installation_model_version) < 2
            or gun.monochromator.wien.mechanical_center_from_tip_mm
            > gun.accelerator.mechanical_center_from_tip_mm
        )
    ):
        gun.migrate_legacy_monochromator_bay()
    gun.apply_manifest_geometry(gun.monochromator_installed)
    return gun.validate()
