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
    "blanking_field_y_mt",
))


def _feg_part_geometry(key):
    return module_manifest.part_geometry(_FEG_MODULE_PATH, key)


def _feg_part_data(key):
    return module_manifest.part_data(_FEG_MODULE_PATH, key)


def _create_emitter():
    geometry = _feg_part_geometry(FEG_TIP)
    part = _feg_part_data(FEG_TIP)
    emitter = ColdFieldEmitter(
        mechanical_center_from_tip_mm=geometry.center_z_mm,
        mechanical_length_mm=geometry.length_mm,
        mechanical_outer_diameter_mm=float(part["outer_diameter_mm"]),
    )
    emitter._tip_reference_file = str(part.get("tip_surface_reference_file", "sources/cold_feg_tip.toml"))
    from temsim.optics.electron_gun.tip_assembly import apply_tip_part
    apply_tip_part(emitter, part)
    return emitter


def _create_extractor():
    geometry = _feg_part_geometry(FEG_EXTRACTOR)
    part = _feg_part_data(FEG_EXTRACTOR)
    return _apply_electrical_defaults(ExtractorElectrode(
        mechanical_center_from_tip_mm=geometry.center_z_mm,
        mechanical_length_mm=geometry.length_mm,
        mechanical_outer_diameter_mm=float(part["outer_diameter_mm"]),
        mechanical_clear_bore_diameter_mm=float(
            part["bore_diameter_mm"]
        ),
    ), part)


def _create_electrostatic_lens():
    geometry = _feg_part_geometry(FEG_ELECTROSTATIC_LENS)
    part = _feg_part_data(FEG_ELECTROSTATIC_LENS)
    return _apply_electrical_defaults(ElectrostaticGunLens(
        mechanical_center_from_tip_mm=geometry.center_z_mm,
        mechanical_length_mm=geometry.length_mm,
        mechanical_outer_diameter_mm=float(part["outer_diameter_mm"]),
        mechanical_clear_bore_diameter_mm=float(
            part["bore_diameter_mm"]
        ),
    ), part)


def _create_accelerator():
    geometry = _feg_part_geometry(FEG_ACCELERATOR)
    part = _feg_part_data(FEG_ACCELERATOR)
    centers = [float(value) for value in part["stage_centers_z_mm"]]
    return _apply_electrical_defaults(AcceleratorColumn(
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
    ), part)


def _apply_electrical_defaults(component, part):
    """Apply a changed TOML default, retaining runtime edits on geometry reload."""
    if isinstance(component, AcceleratorColumn) and "electrode_thickness_mm" in part:
        component._electrode_thickness_mm = float(part["electrode_thickness_mm"])
    for field, attribute in (("default_voltage_kv", "voltage_kv"),
                             ("default_high_tension_kv", "high_tension_kv"),
                             ("default_voltage_reference", "voltage_reference")):
        if field in part and getattr(component, "_assembly_"+field, None) != part[field]:
            setattr(component, attribute, str(part[field]) if field == "default_voltage_reference" else float(part[field]))
            setattr(component, "_assembly_"+field, part[field])
    return component


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
        blanking_field_y_mt=float(part.get("blanking_field_y_mt", 50.0)),
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
    _apply_electrical_defaults(component, part)
    if isinstance(component, ColdFieldEmitter):
        from temsim.optics.electron_gun.tip_assembly import apply_tip_part
        apply_tip_part(component, part)
        component._tip_reference_file = str(part.get("tip_surface_reference_file", "sources/cold_feg_tip.toml"))
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
        component.blanking_field_y_mt = float(part.get("blanking_field_y_mt", 50.0))
    elif isinstance(component, GunStigmator):
        component.effective_length_mm = float(part["active_length_mm"])
    elif component.key == FEG_MONOCHROMATOR_WIEN:
        component.active_length_mm = float(part["active_length_mm"])
    return component


def _apply_resolved_part_geometry(component, part):
    """Apply one already-resolved assembly part without reopening a TOML."""

    data = part.data
    _apply_electrical_defaults(component, data)
    if isinstance(component, ColdFieldEmitter):
        from temsim.optics.electron_gun.tip_assembly import apply_tip_part
        apply_tip_part(component, data)
        component._tip_reference_file = str(data.get("tip_surface_reference_file", "sources/cold_feg_tip.toml"))
    local_shift_mm = (
        float(part.center_z_mm) - float(data["local_center_z_mm"])
    )

    def absolute(local_value):
        return float(local_value) + local_shift_mm

    if hasattr(component, "name"):
        try:
            object.__setattr__(component, "name", str(part.name))
        except (AttributeError, TypeError):
            pass
    component.mechanical_center_from_tip_mm = float(part.center_z_mm)
    component.mechanical_length_mm = float(part.length_mm)
    if (
        "optical_reference_local_z_mm" in data
        and hasattr(component, "optical_reference_from_tip_mm")
    ):
        descriptor = getattr(
            type(component), "optical_reference_from_tip_mm", None
        )
        if not (
            isinstance(descriptor, property)
            and descriptor.fset is None
        ):
            component.optical_reference_from_tip_mm = absolute(
                data["optical_reference_local_z_mm"]
            )
    component.mechanical_outer_diameter_mm = float(
        data["outer_diameter_mm"]
    )
    if hasattr(component, "mechanical_clear_bore_diameter_mm"):
        component.mechanical_clear_bore_diameter_mm = float(
            data["bore_diameter_mm"]
        )
    if hasattr(component, "mechanical_bore_diameter_mm"):
        component.mechanical_bore_diameter_mm = float(
            data["bore_diameter_mm"]
        )
    if isinstance(component, AcceleratorColumn):
        centers = [
            absolute(value) for value in data["stage_centers_z_mm"]
        ]
        if len(centers) != len(component.stages):
            raise ValueError(
                "Resolved accelerator TOML stage count changed; "
                "restart the state to rebuild its physics records."
            )
        for stage, center in zip(component.stages, centers):
            stage.center_from_tip_mm = center
    elif isinstance(component, GunAperture):
        component.plate_thickness_mm = float(data["active_length_mm"])
    elif isinstance(component, GunDeflector):
        centers = [
            absolute(value)
            for value in data["interaction_centers_local_z_mm"]
        ]
        if len(centers) != 2:
            raise ValueError(
                "Resolved gun deflector requires two interaction centres."
            )
        component.upper_center_from_tip_mm = centers[0]
        component.lower_center_from_tip_mm = centers[1]
        component.coil_length_mm = float(data["active_length_mm"])
        component.blanking_field_y_mt = float(data.get("blanking_field_y_mt", 50.0))
    elif isinstance(component, GunStigmator):
        component.effective_length_mm = float(data["active_length_mm"])
    elif component.key == FEG_MONOCHROMATOR_WIEN:
        component.active_length_mm = float(data["active_length_mm"])

    component._manifest_source_file = str(part.source_file)
    component._manifest_part_key = str(part.key)
    component._manifest_definition_id = str(part.definition_id)
    return component


def _component_payload(component, *, include_geometry=False):
    payload = asdict(component)
    if isinstance(component, ColdFieldEmitter) and component.quadrature is not None:
        payload["quadrature"] = asdict(component.quadrature)
    if isinstance(component, ColdFieldEmitter) and component.curvature_nm_inv:
        payload["curvature_nm_inv"] = component.curvature_nm_inv
        payload["emission_geometry_model"] = component.curvature_model
    if isinstance(component, ColdFieldEmitter) and component.coherence is not None:
        payload["coherence"] = asdict(component.coherence)
    if isinstance(component, ColdFieldEmitter) and component.surface_model is not None:
        payload["surface_model"] = component.surface_model.to_dict()
    if not include_geometry and component.key in _TOML_GEOMETRY_COMPONENT_KEYS:
        for attribute in _TOML_GEOMETRY_ATTRIBUTES:
            payload.pop(attribute, None)
    if not include_geometry and component.key in {
        FEG_ACCELERATOR,
        THERMIONIC_ACCELERATOR,
    }:
        for stage in payload["stages"]:
            stage.pop("center_from_tip_mm", None)
    return payload


def _restore_component_settings(component, row):
    allowed = component.__dataclass_fields__
    extra = {"quadrature", "curvature_nm_inv", "emission_geometry_model", "coherence", "surface_model"} if isinstance(component, ColdFieldEmitter) else set()
    unknown = set(row) - set(allowed) - extra
    if unknown:
        raise ValueError(f"Unknown {component.key} fields: {', '.join(sorted(unknown))}")
    if row.get("key") != component.key:
        raise ValueError(f"Electron-gun component key must be {component.key}")
    if isinstance(component, ColdFieldEmitter):
        from temsim.optics.electron_gun.emitter import EmissionQuadrature
        component.quadrature = None if row.get("quadrature") is None else EmissionQuadrature(**row["quadrature"])
        from temsim.optics.electron_gun.tip_curvature import MODEL
        component.curvature_nm_inv = row.get("curvature_nm_inv", 0.0)
        if component.curvature_nm_inv and "emission_geometry_model" not in row:
            raise ValueError("Curved tip records require an explicit emission_geometry_model")
        component.curvature_model = row.get("emission_geometry_model", MODEL)
    if isinstance(component, ElectrostaticGunLens) and "voltage_reference" not in row:
        raise ValueError("Electrostatic gun lens requires explicit voltage_reference")
    if isinstance(component, ColdFieldEmitter) and "surface_model" not in row:
        component.surface_model = None  # Explicit planar record; never infer a curved source.
    for attribute, value in row.items():
        if isinstance(component, ColdFieldEmitter) and attribute == "surface_model":
            from temsim.optics.electron_gun.tip_surface import TipSurfaceModel
            component.surface_model = None if value is None else TipSurfaceModel.from_dict(value)
            continue
        if isinstance(component, ColdFieldEmitter) and attribute == "coherence":
            from temsim.optics.electron_gun.tip_coherence import TipCoherence
            component.coherence = None if value is None else TipCoherence(**value).validate()
            continue
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
    # Opt-in v2 equivalent-source shelf. Legacy parameters are not migrated.
    source_representation: str = "classical_particles"
    effective_source: object | None = None

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
        self._resolved_exit_plane_z_mm = module_manifest.port_z_mm(
            module_path, "exit"
        )
        self.__dict__.pop("_grounded_outlet_liner_segments", None)
        from temsim.physics.gun_field_environment import ensure_gun_field_environment
        ensure_gun_field_environment(self)
        return self

    def apply_resolved_manifest_geometry(
        self, parts, *, exit_plane_z_mm
    ):
        """Accept the selected assembly as the sole runtime geometry source."""

        for component in self.components:
            try:
                part = parts[component.key]
            except KeyError as exc:
                raise ValueError(
                    "Selected gun assembly is missing runtime component "
                    f"{component.key!r}"
                ) from exc
            _apply_resolved_part_geometry(component, part)
        self._resolved_exit_plane_z_mm = float(exit_plane_z_mm)
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
        resolved = getattr(self, "_resolved_exit_plane_z_mm", None)
        if resolved is not None:
            return float(resolved)
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
        if self.uses_geometry_electric_field:
            # A search region, not an assertion that the electric field ends.
            return lens.mechanical_center_from_tip_mm + .5*lens.mechanical_length_mm, self.exit_plane_z_mm
        start = (
            lens.optical_reference_from_tip_mm
            + 0.5 * lens.mechanical_length_mm
            + lens.soft_edge_mm
        )
        return start, self.exit_plane_z_mm

    @property
    def uses_geometry_electric_field(self):
        """Select the field family without building a field or changing emission."""
        return self.type_key == "cold_feg"

    @property
    def base_electric_field(self):
        """The gun electrodes, before any installed Wien field is added."""
        if self.type_key == "cold_feg" and self.emitter.surface_model is not None:
            from temsim.physics.grounded_tip_field import grounded_field
            return grounded_field(self)
        if self.uses_geometry_electric_field:
            from temsim.physics.gun_field_environment import ensure_gun_field_environment
            ensure_gun_field_environment(self)
            if float(self.emitter.curvature_nm_inv) != 0.:
                from temsim.physics.continuous_gun_field import continuous_field
                return continuous_field(self)
            from temsim.physics.closed_gun_field import closed_field
            return closed_field(self)
        return FegElectrostaticField(
            self.emitter, self.extractor, self.electrostatic_lens, self.accelerator,
        )

    @property
    def electric_field(self):
        base = self.base_electric_field
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
        from temsim.optics.electron_gun.source_policy import require_physical_gun_source
        require_physical_gun_source(self)
        from temsim.physics.gun_field_environment import ensure_gun_field_environment
        ensure_gun_field_environment(self)
        surface = getattr(self.emitter, "surface_model", None) is not None
        for component in self.base_components:
            if self.uses_geometry_electric_field and any(component is item for item in (self.extractor, self.electrostatic_lens, self.accelerator)):
                component.validate(grounded=True)
            else:
                component.validate()
        if surface:
            from temsim.physics.grounded_tip_field import field_request
            field_request(self)  # validate physical boundary inputs, not old ramps
        elif self.uses_geometry_electric_field:
            if float(self.emitter.curvature_nm_inv) != 0.:
                from temsim.physics.continuous_gun_field import continuous_field_request
                continuous_field_request(self)
            else:
                from temsim.physics.closed_gun_field import closed_field_request
                closed_field_request(self)
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
        if not (0.0 <= self.dpa_aperture.z_mm < self.c1_aperture.z_mm
                <= self.exit_plane_z_mm < float("inf")):
            raise ValueError("Gun apertures must lie between the tip and the gun exit plane.")
        return self

    def emit(self, count=None):
        from temsim.optics.electron_gun.source_policy import require_physical_gun_source
        require_physical_gun_source(self)
        return self.emitter.emit(count)

    def _cache_key(self, count):
        from temsim.vacuum import ensure_standalone_gun_environment
        ensure_standalone_gun_environment(self)
        from temsim.physics.gun_field_environment import ensure_gun_field_environment
        ensure_gun_field_environment(self)
        payload = self.to_dict()
        # Profiles defer geometry to TOML; executed caches must retain it.
        # Hard stops and alignment coils also matter outside the field solve.
        payload["trace_geometry_schema"] = "physical-aperture-separate-exit-v2"
        payload["executed_components"] = {
            component.key: _component_payload(component, include_geometry=True)
            for component in self.components
        }
        payload["exit_plane_z_mm"] = float(self.exit_plane_z_mm)
        payload["requested_count"] = self.ray_count if count is None else count
        payload["particle_integrator_schema"] = "discrete-gradient-compiled-v2"
        from temsim.optics.electron_gun.tracing import (
            ANALYTIC_ENERGY_SCHEMA, ANALYTIC_STEP_SCHEMA,
            ANALYTIC_MAXIMUM_RELATIVE_IMPULSE,
        )
        payload["analytic_energy_schema"] = ANALYTIC_ENERGY_SCHEMA
        payload["analytic_step_schema"] = ANALYTIC_STEP_SCHEMA
        payload["analytic_maximum_relative_impulse"] = ANALYTIC_MAXIMUM_RELATIVE_IMPULSE
        payload["tuning_sampling_schema"] = "physical-tip-grouped-support-v2"
        payload["tuning_surface_probes"] = int(getattr(self.emitter, "_tuning_surface_probes", 0))
        payload["tuning_boundary_probes"] = int(getattr(self.emitter, "_tuning_boundary_probes", 0))
        from dataclasses import asdict
        payload["vacuum_regions"] = [asdict(r) for r in getattr(self, "_vacuum_regions", ())]
        payload["vacuum_seed"] = getattr(self, "_vacuum_seed", 914)
        payload["vacuum_max_step_tau"] = getattr(self, "_vacuum_max_step_tau", .02)
        from temsim.physics.residual_medium import MODEL
        payload["vacuum_model"] = MODEL
        # This field is TOML-owned and deliberately omitted from profiles,
        # but changing its calibration must invalidate cached gun trajectories.
        payload["blanking_field_y_mt"] = float(self.deflector.blanking_field_y_mt)
        if self.type_key == "cold_feg" and self.emitter.surface_model is not None:
            from temsim.physics.grounded_tip_field import field_request
            payload["grounded_field"] = field_request(self)
        elif self.uses_geometry_electric_field:
            if float(self.emitter.curvature_nm_inv) != 0.:
                from temsim.physics.continuous_gun_field import continuous_field_request
                payload["closed_electrode_field"] = continuous_field_request(self)
            else:
                from temsim.physics.closed_gun_field import closed_field_request
                payload["closed_electrode_field"] = closed_field_request(self)
        payload["geometry_transport_schema"] = "electrode-discrete-gradient-actual-endpoint-v1"
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def trace_to_exit(self, count=None, *, cancelled=None):
        if cancelled is not None and cancelled():
            raise RuntimeError("Superseded optical tuning request")
        self.validate()
        key = self._cache_key(count)
        if key != self._trace_cache_key:
            cached = _SHARED_TRACE_CACHE.get(key)
            if cached is None:
                cached = (trace_feg_to_exit(self, count) if cancelled is None else
                          trace_feg_to_exit(self, count, cancelled=cancelled))
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
        base = self.base_electric_field
        position = np.array([[
            0.0,
            0.0,
            self.monochromator.wien.optical_reference_from_tip_mm * 1.0e-3,
        ]])
        potential = getattr(base, "potential_rise_v_at_global_positions",
                            base.potential_v_at_global_positions)
        potential_v = float(potential(position)[0] - potential(np.zeros((1, 3)))[0])
        launch_energy = (self.emitter.surface_model.emission.mean_energy_ev
                         if self.emitter.surface_model is not None
                         else self.emitter.emission_energy_ev)
        return max(
            0.0,
            float(launch_energy) + potential_v,
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
        if not self.monochromator_installed:
            return self.monochromator
        self.monochromator.installed = False
        self.monochromator.accelerator_restore_profile = None
        self.apply_manifest_geometry(False)
        self._bind_c1_mechanism()
        self.validate()
        return self.monochromator

    @property
    def field_supports_mm(self):
        if getattr(self.emitter, "surface_model", None) is not None:
            # Laplace fields extend through the vacuum domain; legacy compact
            # ramps cannot declare a field-free section of this solved gun.
            return ((-self.emitter.surface_model.geometry.shank_length_um*.001, self.exit_plane_z_mm),)
        if self.uses_geometry_electric_field:
            # Every vacuum interval participates in the coupled Laplace solve.
            # No compact analytic window can declare an internal drift gap.
            entrance = (-float(self.emitter.mechanical_length_mm)
                        if self.emitter.curvature_nm_inv != 0. else 0.)
            return ((entrance, self.exit_plane_z_mm),)
        lens = self.electrostatic_lens
        supports = [
            (
                self.extractor.transition_start_mm + self.extractor.field_center_offset_mm,
                self.extractor.transition_end_mm + self.extractor.field_center_offset_mm,
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
        # The equal-time bundle is not a common-Z plane. Slow electrons can
        # still be accelerating after the leading electron leaves a field.
        z = np.asarray(z_mm, dtype=float)
        lo, hi = float(np.min(z)), float(np.max(z))
        if getattr(self, "uses_geometry_electric_field", False):
            # The solved vacuum is active everywhere along the executed gun.
            # Do not rebuild field requests or invent internal drift gaps here.
            step = self.trace_step_mm
        else:
            step = self.drift_step_mm
            for start, end in self.field_supports_mm:
                if lo <= end and hi >= start:
                    step = min(step, self.trace_step_mm)
                elif hi < start:
                    step = min(step, max(start - hi, 1e-10))
        if self.monochromator_installed:
            start, end = self.monochromator.wien.field_support_mm
            if lo <= end and hi >= start:
                step = min(step, self.monochromator.trace_step_mm)
        return step

    def draw_layout(self):
        return tuple(component.draw_layout() for component in self.components)

    def draw_ray_overlay(self):
        return self.trace_to_exit()

    def to_dict(self):
        payload = {
            "type": self.type_key,
            "integrator": {
                "method": ("static_discrete_gradient" if self.uses_geometry_electric_field else "boris"),
                "trace_step_mm": self.trace_step_mm,
                "drift_step_mm": self.drift_step_mm,
                "history_step_mm": self.history_step_mm,
            },
            "components": {
                component.key: _component_payload(component)
                for component in self.base_components
            },
        }
        if self.effective_source is not None:
            payload["effective_source"] = asdict(self.effective_source)
        if self.source_representation != "classical_particles":
            payload["source_representation"] = self.source_representation
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
    parameters = None
    if "effective_source" in values:
        from temsim.optics.electron_gun.effective_source import EffectiveGunSource
        parameters = EffectiveGunSource(**values["effective_source"])
    representation = str(values.get("source_representation", "classical_particles"))
    if representation not in {"classical_particles", "effective_gaussian_schell"}:
        raise ValueError("Unknown electron-gun source representation")
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
    expected_method = "static_discrete_gradient" if gun.uses_geometry_electric_field else "boris"
    if integrator.get("method", "boris") != expected_method:
        raise ValueError("Saved FEG integrator does not match its tip model.")
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
    gun.apply_manifest_geometry(gun.monochromator_installed)
    gun.validate()
    # Preserve historical data without activation. Production source guards
    # reject an exit source; its binding cannot replace actual gun transport.
    gun.source_representation, gun.effective_source = representation, parameters
    return gun
