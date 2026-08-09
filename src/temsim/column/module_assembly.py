from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
from pathlib import Path
from types import MappingProxyType

from temsim import module_manifest
from temsim.component_keys import (
    AC_DEFLECTOR,
    BEAM_DEFLECTOR,
    C1_APERTURE,
    CONDENSER_APERTURE_2,
    CONDENSER_APERTURE_3,
    CONDENSER_DEFLECTOR,
    CONDENSER_LENS_1,
    CONDENSER_STIGMATOR,
    DESCAN_DEFLECTOR,
    ENERGY_FILTER_BIAS_TUBE,
    ENERGY_FILTER_CAMERA_DEFLECTOR,
    ENERGY_FILTER_DYNAMIC_FOCUS_QUADRUPOLE,
    ENERGY_FILTER_EFTEM_OUTPUT_PLANE,
    ENERGY_FILTER_MULTIPOLE_KEYS,
    ENERGY_FILTER_SHUTTER,
    ENERGY_FILTER_SLIT,
    ENERGY_FILTER_TAPERED_PRISM,
    ENERGY_FILTER_ZEBRA,
    FEG_ACCELERATOR,
    FEG_DEFLECTOR,
    FEG_ELECTROSTATIC_LENS,
    FEG_EXTRACTOR,
    FEG_MONOCHROMATOR_WIEN,
    FEG_STIGMATOR,
    FEG_TIP,
    GUN_EXTRACTOR_APERTURE,
    IMAGE_DIFFRACTION_DEFLECTOR,
    MINI_CONDENSER,
    OBJECTIVE_APERTURE,
    OBJECTIVE_LENS,
    OBJECTIVE_STIGMATOR,
    PROBE_CORRECTOR_KEYS,
    PROBE_DP12_SCAN_DEFLECTOR,
    THERMIONIC_ACCELERATOR,
    THERMIONIC_ANODE_APERTURE,
    THERMIONIC_C1_APERTURE,
    THERMIONIC_CATHODE,
    THERMIONIC_DEFLECTOR,
    THERMIONIC_GUN_LENS,
    THERMIONIC_STIGMATOR,
    THERMIONIC_WEHNELT,
)


_TOML_OWNED_GEOMETRY_KEYS = module_manifest.all_part_keys()
TOML_OWNED_GEOMETRY_KEYS = _TOML_OWNED_GEOMETRY_KEYS
_RUNTIME_POSITION_OWNED_KEYS = frozenset()

STRUCTURAL_FIELD_SOURCES = MappingProxyType({
    "mechanical_outer_diameter_mm": (
        "mechanical_outer_diameter_mm",
        "outer_diameter_mm",
    ),
    "mechanical_clear_bore_diameter_mm": (
        "mechanical_clear_bore_diameter_mm",
        "bore_diameter_mm",
    ),
    "mechanical_bore_diameter_mm": (
        "mechanical_bore_diameter_mm",
        "bore_diameter_mm",
    ),
    "bore_diameter_mm": (
        "bore_diameter_mm",
        "mechanical_bore_diameter_mm",
        "mechanical_clear_bore_diameter_mm",
    ),
    "pole_gap_mm": ("pole_gap_mm",),
    "effective_length_mm": ("effective_length_mm", "active_length_mm"),
    "effective_thickness_mm": (
        "effective_thickness_mm",
        "active_length_mm",
    ),
    "mechanical_coil_length_mm": ("mechanical_coil_length_mm",),
    "mechanical_inter_coil_gap_mm": (
        "mechanical_inter_coil_gap_mm",
    ),
    "inter_coil_gap_mm": ("mechanical_inter_coil_gap_mm",),
    "thickness_mm": ("effective_thickness_mm", "active_length_mm"),
    "plate_thickness_mm": ("plate_thickness_mm", "active_length_mm"),
    "maximum_radius_mm": ("maximum_radius_mm",),
    "outer_width_mm": ("outer_width_mm",),
    "inner_diameter_mm": ("inner_diameter_mm",),
})


@dataclass(frozen=True)
class ModulePart:
    key: str
    name: str
    branch: str
    start_z_mm: float
    center_z_mm: float
    end_z_mm: float
    length_mm: float
    parent_key: str | None
    data: Mapping


@dataclass(frozen=True)
class ModuleDefinition:
    type: str
    key: str
    entrance_interface: str
    entrance_z_mm: float
    exit_interface: str
    exit_z_mm: float
    length_mm: float
    parts: tuple[ModulePart, ...]
    source_file: str
    geometry: Mapping


@dataclass(frozen=True)
class AssemblyPart:
    module_key: str
    source_file: str
    key: str
    name: str
    branch: str
    start_z_mm: float
    center_z_mm: float
    end_z_mm: float
    length_mm: float
    parent_key: str | None
    data: Mapping

    @property
    def definition_id(self):
        """Stable, variant-scoped authority for this assembled part."""

        return f"{self.source_file}::parts[{self.key}]"


@dataclass(frozen=True)
class VacuumBoreSegment:
    """One axial section of the electron-accessible vacuum envelope."""

    key: str
    name: str
    start_z_mm: float
    end_z_mm: float
    inner_diameter_mm: float


@dataclass(frozen=True)
class VacuumLinerSegment:
    """Thin non-magnetic tube surrounding one vacuum-bore segment."""

    key: str
    name: str
    start_z_mm: float
    end_z_mm: float
    inner_diameter_mm: float
    outer_diameter_mm: float
    wall_thickness_mm: float


@dataclass(frozen=True)
class ModuleAssembly:
    modules: tuple[ModuleDefinition, ...]
    parts: tuple[AssemblyPart, ...]
    exit_z_mm: float


@dataclass(frozen=True)
class ResolvedAssembly(ModuleAssembly):
    """One immutable, self-contained reading of the selected module TOMLs."""

    root: Path
    selected_module_paths: tuple[tuple[str, str], ...]
    vacuum_bore_segments: tuple[VacuumBoreSegment, ...]
    vacuum_liner_segments: tuple[VacuumLinerSegment, ...]

    def selected_path(self, module_type):
        for selected_type, path in self.selected_module_paths:
            if selected_type == module_type:
                return path
        raise KeyError(module_type)

    def part(self, key):
        for part in self.parts:
            if part.key == key:
                return part
        raise KeyError(key)

    @property
    def part_authorities(self):
        """Map every active key to its one selected TOML definition."""

        return MappingProxyType({
            part.key: part.definition_id for part in self.parts
        })

    @property
    def maximum_vacuum_inner_diameter_mm(self):
        return max(
            segment.inner_diameter_mm for segment in self.vacuum_bore_segments
        )


def _module_vacuum_segments(module, origin, resolved_parts):
    """Resolve overlaps by keeping the narrowest electron-accessible bore."""

    module_start = origin + module.entrance_z_mm
    module_end = origin + module.exit_z_mm
    parts = [part for part in resolved_parts if part.module_key == module.key]
    breakpoints = {float(module_start), float(module_end)}
    for part in parts:
        if part.end_z_mm > part.start_z_mm:
            breakpoints.add(max(float(part.start_z_mm), float(module_start)))
            breakpoints.add(min(float(part.end_z_mm), float(module_end)))
    points = sorted(
        value for value in breakpoints
        if module_start <= value <= module_end
    )
    drift_diameter = float(module.geometry["vacuum_drift_inner_diameter_mm"])
    segments = []
    for start, end in zip(points, points[1:]):
        if end <= start:
            continue
        midpoint = 0.5 * (start + end)
        active = [
            part for part in parts
            if (
                part.start_z_mm <= midpoint <= part.end_z_mm
                and "vacuum_inner_diameter_mm" in part.data
            )
        ]
        if active:
            owner = min(
                active,
                key=lambda part: (
                    float(part.data["vacuum_inner_diameter_mm"]),
                    float(part.length_mm),
                ),
            )
            diameter = float(owner.data["vacuum_inner_diameter_mm"])
            key = owner.key
            name = owner.name
        else:
            diameter = drift_diameter
            key = f"@vacuum_drift:{module.key}"
            name = f"{module.key} vacuum drift"
        segment = VacuumBoreSegment(key, name, start, end, diameter)
        previous = segments[-1] if segments else None
        if (
            previous is not None
            and previous.key == segment.key
            and math.isclose(
                previous.inner_diameter_mm,
                segment.inner_diameter_mm,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and math.isclose(
                previous.end_z_mm, segment.start_z_mm,
                rel_tol=0.0, abs_tol=1.0e-12,
            )
        ):
            segments[-1] = replace(previous, end_z_mm=segment.end_z_mm)
        else:
            segments.append(segment)
    return tuple(segments)


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({
            key: _freeze(item)
            for key, item in value.items()
        })
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _read(path):
    return module_manifest.read_document(path)


def _load_module(path, source_file=None):
    if source_file is None:
        source_file = Path(path).name
    data = module_manifest.validate_document(_read(path))
    module = data["module"]
    entrance = data["ports"]["entrance"]
    exit_port = data["ports"]["exit"]
    module_length_mm = float(data["geometry"]["length_mm"])
    parts = tuple(
        ModulePart(
            str(part["key"]),
            str(part["name"]),
            str(part["branch"]),
            float(part["local_start_z_mm"]),
            float(part["local_center_z_mm"]),
            float(part["local_end_z_mm"]),
            float(part["length_mm"]),
            part.get("parent_key"),
            _freeze(dict(part)),
        )
        for part in sorted(data["parts"], key=lambda part: int(part["order"]))
    )
    if data["coordinate_system"] != "module_local_z_mm":
        raise ValueError(f"Invalid coordinate system in {path}")
    if len({part.key for part in parts}) != len(parts):
        raise ValueError(f"Duplicate part key in {path}")
    orders = [int(part.data["order"]) for part in parts]
    if len(set(orders)) != len(orders):
        raise ValueError(f"Duplicate part order in {path}")
    for part in parts:
        if not part.start_z_mm <= part.center_z_mm <= part.end_z_mm:
            raise ValueError(f"Invalid part range for {part.key} in {path}")
        envelope_length_mm = part.end_z_mm - part.start_z_mm
        if abs(part.length_mm - envelope_length_mm) > 1.0e-9:
            raise ValueError(
                f"Part length mismatch for {part.key} in {path}: "
                f"length_mm={part.length_mm}, "
                f"envelope={envelope_length_mm}"
            )
    port_span_mm = (
        float(exit_port["local_z_mm"])
        - float(entrance["local_z_mm"])
    )
    if abs(module_length_mm - port_span_mm) > 1.0e-9:
        raise ValueError(
            f"Module length mismatch in {path}: "
            f"length_mm={module_length_mm}, port_span={port_span_mm}"
        )
    return ModuleDefinition(
        str(module["type"]),
        str(module["key"]),
        str(entrance["interface"]),
        float(entrance["local_z_mm"]),
        str(exit_port["interface"]),
        float(exit_port["local_z_mm"]),
        module_length_mm,
        parts,
        str(source_file),
        _freeze(dict(data["geometry"])),
    )


def _one(entries, **selection):
    matches = [
        entry
        for entry in entries
        if all(entry.get(key) == value for key, value in selection.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one module for {selection}, found {len(matches)}")
    return matches[0]


def _selected_entries(configuration, catalog):
    gun_name = {
        "cold_feg": "FEG",
        "thermionic": "Thermionic",
    }.get(configuration.electron_gun_type)
    if gun_name is None:
        gun_component_keys = {
            str(component.key)
            for component in configuration.gun_components
        }
        if FEG_TIP in gun_component_keys:
            gun_name = "FEG"
        elif THERMIONIC_CATHODE in gun_component_keys:
            gun_name = "Thermionic"
    if gun_name is None:
        raise ValueError(f"Unsupported electron gun {configuration.electron_gun_type}")
    corrector = configuration.corrector.value
    probe = corrector in {"probe_corrector", "double_corrector"}
    image = corrector in {"image_corrector", "double_corrector"}
    c3 = configuration.c3_hardware.value == "three_condenser"
    selected = {
        "gun": _one(
            catalog["gun_variants"],
            electron_gun=gun_name,
            monochromator=bool(configuration.monochromator_installed),
        ),
        "column": _one(
            catalog["column_variants"],
            c3_lens=c3,
            probe_corrector=probe,
            image_corrector=image,
        ),
        "project_and_recording_system": _one(
            catalog["project_and_recording_system_variants"],
            energy_filter=bool(configuration.energy_filter_selected),
        ),
    }
    return tuple(
        selected[module_type]
        for module_type in catalog["assembly"]["order"]
    )


def selected_module_paths(configuration, root=None):
    root = Path(root) if root is not None else module_manifest.MODULE_ROOT
    catalog = _read(root / "catalog.toml")
    entries = _selected_entries(configuration, catalog)
    return {
        module_type: str(entry["file"])
        for module_type, entry in zip(
            catalog["assembly"]["order"],
            entries,
        )
    }


def resolve_module_assembly(configuration, root=None):
    root = (
        Path(root)
        if root is not None
        else module_manifest.MODULE_ROOT
    )
    catalog = _read(root / "catalog.toml")
    entries = _selected_entries(configuration, catalog)
    selected_paths = tuple(
        (
            str(module_type),
            str(entry["file"]),
        )
        for module_type, entry in zip(
            catalog["assembly"]["order"],
            entries,
        )
    )
    modules = tuple(
        _load_module(root / entry["file"], entry["file"])
        for entry in entries
    )
    parts = []
    module_origins = []
    absolute_exit = 0.0
    previous_interface = None
    for module in modules:
        if previous_interface is None:
            origin = -module.entrance_z_mm
        else:
            if module.entrance_interface != previous_interface:
                raise ValueError(
                    f"Interface mismatch: {previous_interface} -> "
                    f"{module.entrance_interface}"
                )
            origin = absolute_exit - module.entrance_z_mm
        module_origins.append(origin)
        for part in module.parts:
            parts.append(AssemblyPart(
                module.key,
                module.source_file,
                part.key,
                part.name,
                part.branch,
                origin + part.start_z_mm,
                origin + part.center_z_mm,
                origin + part.end_z_mm,
                part.length_mm,
                part.parent_key,
                part.data,
            ))
        absolute_exit = origin + module.exit_z_mm
        previous_interface = module.exit_interface
    if len({part.key for part in parts}) != len(parts):
        raise ValueError("Duplicate part key in assembled column")
    vacuum_bore_segments = []
    vacuum_liner_segments = []
    for module, origin in zip(modules, module_origins):
        module_segments = _module_vacuum_segments(module, origin, parts)
        wall = float(module.geometry["vacuum_liner_wall_thickness_mm"])
        vacuum_bore_segments.extend(module_segments)
        vacuum_liner_segments.extend(
            VacuumLinerSegment(
                key=f"@vacuum_liner:{segment.key}",
                name=f"{segment.name} vacuum liner",
                start_z_mm=segment.start_z_mm,
                end_z_mm=segment.end_z_mm,
                inner_diameter_mm=segment.inner_diameter_mm,
                outer_diameter_mm=(
                    segment.inner_diameter_mm + 2.0 * wall
                ),
                wall_thickness_mm=wall,
            )
            for segment in module_segments
        )
    return ResolvedAssembly(
        modules=modules,
        parts=tuple(parts),
        exit_z_mm=absolute_exit,
        root=root.resolve(),
        selected_module_paths=selected_paths,
        vacuum_bore_segments=tuple(vacuum_bore_segments),
        vacuum_liner_segments=tuple(vacuum_liner_segments),
    )


def apply_module_assembly(
    configuration,
    layout,
    root=None,
    assembly=None,
):
    if assembly is None:
        assembly = getattr(configuration, "resolved_assembly", None)
    if assembly is None:
        assembly = resolve_module_assembly(configuration, root)
    original_source_to_sample = layout.source_to_sample_mm
    original_positions = {
        component.key: (
            original_source_to_sample - component.local_s_range_mm[1],
            original_source_to_sample - component.local_s_center_mm,
            original_source_to_sample - component.local_s_range_mm[0],
        )
        for component in layout
    }
    parts = {part.key: part for part in assembly.parts}
    component_key_list = [component.key for component in layout]
    component_keys = set(component_key_list)
    if len(component_keys) != len(component_key_list):
        duplicates = sorted({
            key for key in component_key_list
            if component_key_list.count(key) > 1
        })
        raise ValueError(
            f"Duplicate component key in optics layout: {duplicates}"
        )
    physical_keys = {
        component.key
        for component in layout
        if (
            component.mechanical_shape is None
            or component.mechanical_shape.profile != "reference_plane"
        )
    }
    missing = physical_keys - parts.keys()
    extra = {
        key for key in parts.keys() - component_keys
        if not (
            bool(parts[key].data.get("mechanical_only", False))
            or bool(parts[key].data.get("branch_path_only", False))
        )
    }
    if missing or extra:
        raise ValueError(
            f"Module/layout part mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    sample_z_mm = parts["sample"].center_z_mm
    physical_original = {
        key: original_positions[key][1] for key in physical_keys
    }
    replacements = []
    for component in layout:
        part = parts.get(component.key)
        if part is None:
            old_start, old_center, old_end = original_positions[component.key]
            anchor = min(
                physical_original,
                key=lambda key: abs(physical_original[key] - old_center),
            )
            shift = parts[anchor].center_z_mm - physical_original[anchor]
            start_z_mm = old_start + shift
            center_z_mm = old_center + shift
            end_z_mm = old_end + shift
            name = component.name
            branch = component.branch
            parent_key = component.nested_parent_key
        else:
            start_z_mm = part.start_z_mm
            center_z_mm = part.center_z_mm
            end_z_mm = part.end_z_mm
            name = part.name
            branch = type(component.branch)(part.branch)
            parent_key = part.parent_key
        optical_reference = component.optical_reference_plane_z_mm
        interaction_planes = component.optical_interaction_planes_z_mm
        if part is not None:
            if component.key == OBJECTIVE_LENS:
                optical_reference = _absolute_part_value(
                    part, "virtual_reference_local_z_mm"
                )
            elif "optical_reference_local_z_mm" in part.data:
                optical_reference = _absolute_part_value(
                    part, "optical_reference_local_z_mm"
                )
            if "interaction_centers_local_z_mm" in part.data:
                interaction_planes = tuple(
                    float(part.center_z_mm)
                    + float(value)
                    - float(part.data["local_center_z_mm"])
                    for value in part.data[
                        "interaction_centers_local_z_mm"
                    ]
                )
        start_s_mm = sample_z_mm - end_z_mm
        center_s_mm = sample_z_mm - center_z_mm
        end_s_mm = sample_z_mm - start_z_mm
        shape = component.mechanical_shape
        external_envelope = component.external_envelope
        if shape is not None:
            shape_updates = {"axial_length_mm": end_z_mm - start_z_mm}
            if (
                part is not None
                and "transverse_envelope_x_mm" in part.data
                and "transverse_envelope_y_mm" in part.data
            ):
                external_envelope = (
                    f"{float(part.data['transverse_envelope_x_mm']):g} x "
                    f"{float(part.data['transverse_envelope_y_mm']):g} mm "
                    "transverse envelope"
                )
                shape_updates["external_envelope"] = external_envelope
            for field in (
                "mechanical_outer_diameter_mm", "outer_diameter_mm",
            ):
                if part is not None and field in part.data:
                    shape_updates["outer_diameter_mm"] = float(part.data[field])
                    break
            for field in (
                "mechanical_clear_bore_diameter_mm",
                "mechanical_bore_diameter_mm",
                "bore_diameter_mm",
            ):
                if part is not None and field in part.data:
                    shape_updates["active_diameter_mm"] = float(part.data[field])
                    break
            for field in (
                "mechanical_tip_diameter_mm",
                "pole_piece_tip_diameter_mm",
                "plate_thickness_mm",
                "effective_length_mm",
                "effective_thickness_mm",
            ):
                if part is not None and field in part.data:
                    shape_updates["active_length_mm"] = float(part.data[field])
                    break
            shape = replace(shape, **shape_updates)
        replacements.append(replace(
            component,
            name=name,
            branch=branch,
            mechanical=replace(
                component.mechanical,
                start_s_mm=start_s_mm,
                end_s_mm=end_s_mm,
            ),
            field_support=replace(
                component.field_support,
                start_s_mm=start_s_mm,
                end_s_mm=end_s_mm,
            ),
            local_s_center_mm=center_s_mm,
            local_s_range_mm=(start_s_mm, end_s_mm),
            rendered_z_center_mm=center_s_mm,
            rendered_z_range_mm=(start_s_mm, end_s_mm),
            external_envelope=external_envelope,
            mechanical_shape=shape,
            optical_reference_plane_z_mm=optical_reference,
            optical_interaction_planes_z_mm=interaction_planes,
            nested_parent_key=parent_key,
        ))
    by_key = {component.key: component for component in replacements}
    resolved = []
    for component in replacements:
        upstream = by_key.get(component.upstream_key)
        downstream = by_key.get(component.downstream_key)
        resolved.append(replace(
            component,
            upstream_clearance_mm=(
                upstream.local_s_range_mm[0]
                - component.local_s_range_mm[1]
                if upstream is not None
                else None
            ),
            downstream_clearance_mm=(
                component.local_s_range_mm[0]
                - downstream.local_s_range_mm[1]
                if downstream is not None
                else None
            ),
        ))
    return type(layout)(resolved)


def module_coordinate_offsets(configuration, layout):
    from temsim.column.layout import _build_optics_layout_metadata

    reference = _build_optics_layout_metadata(configuration)
    reference_positions = {
        component.key: (
            reference.source_to_sample_mm - component.local_s_center_mm
        )
        for component in reference
    }
    return {
        component.key: (
            layout.source_to_sample_mm
            - component.local_s_center_mm
            - reference_positions[component.key]
        )
        for component in layout
        if component.key in reference_positions
    }


def _absolute_part_value(part, local_key):
    return (
        float(part.center_z_mm)
        + float(part.data[local_key])
        - float(part.data["local_center_z_mm"])
    )


def apply_column_manifest_geometry(
    state,
    configuration,
    assembly=None,
    *,
    preserve_operating_parameters=True,
    root=None,
):
    """Inject selected TOML geometry into every available runtime component."""

    from temsim.optics.energy_filter import ensure_energy_filter
    ensure_energy_filter(state)
    operating_values = {}
    if preserve_operating_parameters:
        for lens in getattr(state, "lenses", ()):
            operating_values[str(lens.key)] = {
                name: getattr(lens, name)
                for name in (
                    "enabled", "percent", "cs_mm", "cc_mm", "polarity", "colour"
                )
                if hasattr(lens, name)
            }
    if assembly is None:
        assembly = getattr(configuration, "resolved_assembly", None)
    if assembly is None:
        assembly = resolve_module_assembly(configuration, root=root)
    parts = {
        part.key: part
        for part in assembly.parts
    }
    for key in (
        CONDENSER_LENS_1,
        "condenser_lens_2",
        "condenser_lens_3",
    ):
        part = parts.get(key)
        if part is None:
            continue
        component = getattr(state, key)
        geometry = {
            "mechanical_center_from_tip_mm": float(part.center_z_mm),
            "mechanical_length_mm": float(part.length_mm),
            "optical_reference_from_tip_mm": _absolute_part_value(
                part, "optical_reference_local_z_mm"
            ),
        }
        optional_fields = {
            "mechanical_outer_diameter_mm": "mechanical_outer_diameter_mm",
            "bore_diameter_mm": "bore_diameter_mm",
            "pole_gap_mm": "pole_gap_mm",
            "effective_aperture_radius_mm": "effective_aperture_radius_mm",
        }
        geometry.update({
            attribute: float(part.data[field])
            for attribute, field in optional_fields.items()
            if field in part.data
        })
        component.apply_manifest_geometry(**geometry)
    for key, component in (
        (CONDENSER_APERTURE_2, state.condenser_aperture_2),
        (CONDENSER_APERTURE_3, state.condenser_aperture_3),
    ):
        part = parts.get(key)
        if part is None:
            continue
        geometry = dict(
            z_mm=_absolute_part_value(
                part,
                "optical_reference_local_z_mm",
            ),
            mechanical_length_mm=float(part.length_mm),
        )
        for field in (
            "mechanical_outer_diameter_mm",
            "mechanical_bore_diameter_mm",
            "plate_thickness_mm",
            "maximum_radius_mm",
        ):
            if field in part.data:
                geometry[field] = float(part.data[field])
        component.apply_manifest_geometry(**geometry)
    _apply_manifest_runtime_geometry(state, parts, assembly)
    for lens in getattr(state, "lenses", ()):
        for name, value in operating_values.get(str(lens.key), {}).items():
            object.__setattr__(lens, name, value)
    for key in _TOML_OWNED_GEOMETRY_KEYS:
        state.component_placements.pop(key, None)
        for prefix in (
            "lens",
            "aperture",
            "deflector",
            "stigmator",
            "corrector",
            "recording_plane",
        ):
            state.layout_reference_positions.pop(
                f"{prefix}:{key}",
                None,
            )
    state._module_manifest_parts = parts
    state._resolved_assembly = assembly
    state._module_optical_offsets_mm = {}
    state._module_gun_offsets_mm = {}
    return parts


def _state_targets(state):
    targets = {}

    def add_target(key, kind, item):
        key = str(key)
        existing = targets.get(key)
        if existing is not None and existing[1] is not item:
            raise ValueError(
                f"Duplicate runtime component key {key!r}: "
                f"{type(existing[1]).__name__} and "
                f"{type(item).__name__}"
            )
        targets[key] = (kind, item)

    for items in (
        state.lenses,
        state.apertures,
        state.stigmators,
        getattr(state, "corrector_elements", ()),
        getattr(state, "recording_planes", ()),
    ):
        for item in items:
            if hasattr(item, "z_mm"):
                add_target(item.key, "single", item)
    for pair in state.deflectors:
        add_target(pair.key, "pair", pair)
    if hasattr(state.sample, "z_mm"):
        add_target("sample", "single", state.sample)
    return targets


def _set_pair_positions(item, positions):
    upper, lower = (float(value) for value in positions)
    upper_descriptor = getattr(type(item), "upper_z_mm", None)
    lower_descriptor = getattr(type(item), "lower_z_mm", None)
    read_only_pair = (
        isinstance(upper_descriptor, property)
        and upper_descriptor.fset is None
        and isinstance(lower_descriptor, property)
        and lower_descriptor.fset is None
    )
    if read_only_pair:
        center = 0.5 * (upper + lower)
        if hasattr(item, "z_mm"):
            item.z_mm = center
        elif hasattr(item, "optical_center_from_tip_mm"):
            item.optical_center_from_tip_mm = center
        if (
            upper != lower
            and hasattr(item, "optical_plane_separation_mm")
        ):
            descriptor = getattr(
                type(item), "optical_plane_separation_mm", None
            )
            if not (
                isinstance(descriptor, property)
                and descriptor.fset is None
            ):
                item.optical_plane_separation_mm = lower - upper
        return
    if hasattr(item, "optical_upper_reference_from_tip_mm"):
        item.optical_upper_reference_from_tip_mm = upper
        item.optical_lower_reference_from_tip_mm = lower
    elif hasattr(item, "optical_upper_reference_z_mm"):
        item.optical_upper_reference_z_mm = upper
        item.optical_lower_reference_z_mm = lower
    item.upper_z_mm = upper
    item.lower_z_mm = lower


def _set_single_position(item, reference):
    reference = float(reference)
    for attribute in (
        "optical_reference_from_tip_mm",
        "optical_reference_z_mm",
        "optical_center_from_tip_mm",
    ):
        descriptor = getattr(type(item), attribute, None)
        if (
            hasattr(item, attribute)
            and not (
                isinstance(descriptor, property)
                and descriptor.fset is None
            )
        ):
            try:
                setattr(item, attribute, reference)
            except AttributeError:
                pass
    if hasattr(item, "z_mm"):
        try:
            item.z_mm = reference
        except AttributeError:
            object.__setattr__(item, "z_mm", reference)


def _set_manifest_authority(item, part):
    """Attach non-serialised provenance for the active TOML definition."""

    if hasattr(item, "name"):
        try:
            object.__setattr__(item, "name", str(part.name))
        except (AttributeError, TypeError):
            pass
    object.__setattr__(item, "_manifest_source_file", str(part.source_file))
    object.__setattr__(item, "_manifest_part_key", str(part.key))
    object.__setattr__(
        item, "_manifest_definition_id", str(part.definition_id)
    )


def _set_mechanical_geometry(item, part, parts):
    center = float(part.center_z_mm)
    length = float(part.length_mm)
    if hasattr(item, "mechanical_center_from_tip_mm"):
        try:
            item.mechanical_center_from_tip_mm = center
        except (AttributeError, TypeError):
            pass
    if hasattr(item, "mechanical_center_below_sample_mm") and "sample" in parts:
        try:
            item.mechanical_center_below_sample_mm = (
                center - float(parts["sample"].center_z_mm)
            )
        except (AttributeError, TypeError):
            pass
    anchor_key = getattr(item, "anchor_key", None)
    if anchor_key in parts:
        center_offset = center - float(parts[anchor_key].center_z_mm)
        for attribute in (
            "mechanical_center_downstream_of_anchor_mm",
            "layout_center_downstream_of_anchor_mm",
        ):
            if not hasattr(item, attribute):
                continue
            try:
                setattr(item, attribute, center_offset)
            except (AttributeError, TypeError):
                pass
    if hasattr(item, "mechanical_length_mm"):
        try:
            item.mechanical_length_mm = length
        except (AttributeError, TypeError):
            pass
    for attribute, source_fields in STRUCTURAL_FIELD_SOURCES.items():
        if not hasattr(item, attribute):
            continue
        try:
            current_value = getattr(item, attribute)
        except (AttributeError, TypeError):
            continue
        if current_value is None:
            continue
        source_field = next(
            (field for field in source_fields if field in part.data),
            None,
        )
        if source_field is None:
            raise ValueError(
                f"TOML authority {part.definition_id} is missing the "
                f"structural value for {attribute}"
            )
        try:
            setattr(item, attribute, float(part.data[source_field]))
        except (AttributeError, TypeError):
            pass
    if hasattr(item, "layout_length_mm"):
        try:
            item.layout_length_mm = length
        except (AttributeError, TypeError):
            pass
    if hasattr(item, "mechanical_center_above_sample_mm") and "sample" in parts:
        try:
            item.mechanical_center_above_sample_mm = (
                float(parts["sample"].center_z_mm) - center
            )
        except (AttributeError, TypeError):
            pass
    if (
        "effective_thickness_mm" in part.data
        and hasattr(item, "thickness_mm")
    ):
        try:
            item.thickness_mm = float(part.data["effective_thickness_mm"])
        except (AttributeError, TypeError):
            pass
    if (
        "mechanical_inter_coil_gap_mm" in part.data
        and hasattr(item, "inter_coil_gap_mm")
    ):
        try:
            item.inter_coil_gap_mm = float(
                part.data["mechanical_inter_coil_gap_mm"]
            )
        except (AttributeError, TypeError):
            pass


def _apply_manifest_field_polarity(item, part):
    """Apply the selected TOML's signed Bz convention and its provenance."""

    if not module_manifest.part_requires_field_polarity(part.data):
        return
    if not hasattr(item, "polarity"):
        raise ValueError(
            f"Magnetic lens {part.key} has no runtime polarity attribute"
        )
    object.__setattr__(item, "polarity", int(part.data["field_polarity"]))
    object.__setattr__(
        item,
        "field_polarity_status",
        str(part.data["field_polarity_status"]),
    )
    object.__setattr__(
        item,
        "field_polarity_source",
        str(part.data["field_polarity_source"]),
    )


def _apply_detector_orientation_calibration(item, part):
    """Apply TOML-owned detector/display axes without changing geometry."""

    fields = (
        "detector_axis_rotation_deg",
        "detector_flip_x",
        "detector_flip_y",
        "detector_orientation_uncertainty_deg",
        "detector_orientation_status",
        "detector_orientation_source",
    )
    if not any(field in part.data for field in fields):
        return
    for field in fields:
        if field not in part.data or not hasattr(item, field):
            raise ValueError(
                f"Detector {part.key} cannot apply TOML calibration field {field}"
            )
        object.__setattr__(item, field, part.data[field])
    validate = getattr(item, "validate", None)
    if callable(validate):
        validate()


def _apply_objective_manifest_geometry(state, parts):
    component = getattr(state, "objective_lens", None)
    part = parts.get(OBJECTIVE_LENS)
    if component is None or part is None:
        return
    _set_manifest_authority(component, part)
    upper = _absolute_part_value(
        part, "upper_field_reference_local_z_mm"
    )
    lower = _absolute_part_value(
        part, "lower_field_reference_local_z_mm"
    )
    virtual = _absolute_part_value(part, "virtual_reference_local_z_mm")
    upper_pole = parts.get("objective_upper_pole")
    lower_pole = parts.get("objective_lower_pole")
    from temsim.optics.condenser_lens import AxialFieldTerm

    def profile_terms(prefix):
        amplitudes = tuple(
            part.data[f"{prefix}_field_profile_amplitudes"]
        )
        offsets = tuple(part.data[f"{prefix}_field_profile_offsets"])
        sigmas = tuple(part.data[f"{prefix}_field_profile_sigmas"])
        return [
            AxialFieldTerm(
                float(amplitude),
                float(offset),
                float(sigma),
            )
            for amplitude, offset, sigma in zip(
                amplitudes, offsets, sigmas
            )
        ]

    upper_yoke_start = _absolute_part_value(
        part, "upper_yoke_start_local_z_mm"
    )
    upper_yoke_end = _absolute_part_value(
        part, "upper_yoke_end_local_z_mm"
    )
    lower_yoke_start = _absolute_part_value(
        part, "lower_yoke_start_local_z_mm"
    )
    lower_yoke_end = _absolute_part_value(
        part, "lower_yoke_end_local_z_mm"
    )
    object.__setattr__(component, "_position_coupling_ready", False)
    object.__setattr__(component, "z_mm", virtual)
    object.__setattr__(component, "virtual_lens_reference_z_mm", virtual)
    object.__setattr__(component, "upper_field_center_z_mm", upper)
    object.__setattr__(component, "lower_field_center_z_mm", lower)
    object.__setattr__(
        component,
        "upper_objective_lens_center_z_mm",
        0.5 * (upper_yoke_start + upper_yoke_end),
    )
    object.__setattr__(
        component,
        "lower_objective_lens_center_z_mm",
        0.5 * (lower_yoke_start + lower_yoke_end),
    )
    object.__setattr__(
        component,
        "upper_objective_lens_axial_length_mm",
        upper_yoke_end - upper_yoke_start,
    )
    object.__setattr__(
        component,
        "lower_objective_lens_axial_length_mm",
        lower_yoke_end - lower_yoke_start,
    )
    sample = getattr(state, "sample", None)
    sample_thickness_mm = (
        float(getattr(sample, "thickness_nm", 0.0)) * 0.5e-6
    )
    object.__setattr__(
        component,
        "virtual_lens_offset_below_lower_surface_mm",
        virtual
        - float(parts["sample"].center_z_mm)
        - sample_thickness_mm,
    )
    object.__setattr__(
        component,
        "upper_field_center_above_sample_mm",
        float(parts["sample"].center_z_mm) - upper,
    )
    object.__setattr__(
        component,
        "sample_axial_offset_mm",
        float(parts["sample"].center_z_mm)
        - 0.5
        * (
            float(upper_pole.end_z_mm)
            + float(lower_pole.start_z_mm)
        ),
    )
    object.__setattr__(component, "assembly_length_mm", float(part.length_mm))
    object.__setattr__(
        component,
        "assembly_outer_diameter_mm",
        float(part.data["mechanical_outer_diameter_mm"]),
    )
    object.__setattr__(
        component,
        "inner_face_gap_mm",
        float(part.data["s_twin_pole_gap_mm"]),
    )
    object.__setattr__(
        component,
        "upper_b0_t",
        float(part.data["upper_peak_field_t"]),
    )
    object.__setattr__(
        component,
        "lower_b0_t",
        float(part.data["lower_peak_field_t"]),
    )
    object.__setattr__(
        component,
        "upper_a_mm",
        float(part.data["upper_field_half_width_mm"]),
    )
    object.__setattr__(
        component,
        "lower_a_mm",
        float(part.data["lower_field_half_width_mm"]),
    )
    object.__setattr__(component, "upper_gaussian", profile_terms("upper"))
    object.__setattr__(component, "lower_gaussian", profile_terms("lower"))
    object.__setattr__(
        component,
        "max_percent",
        float(part.data["maximum_excitation_percent"]),
    )
    object.__setattr__(
        component,
        "nominal_voltage_kv",
        float(part.data["nominal_voltage_kv"]),
    )
    object.__setattr__(
        component,
        "nominal_focal_length_mm",
        float(part.data["nominal_focal_length_mm"]),
    )
    object.__setattr__(
        component,
        "nominal_back_focal_plane_z_mm",
        _absolute_part_value(
            part, "nominal_back_focal_plane_local_z_mm"
        ),
    )
    object.__setattr__(
        component,
        "nominal_image_plane_z_mm",
        _absolute_part_value(part, "nominal_image_plane_local_z_mm"),
    )
    object.__setattr__(
        component,
        "cc_mm",
        float(part.data["chromatic_aberration_mm"]),
    )
    object.__setattr__(
        component,
        "cs_mm",
        float(part.data["spherical_aberration_mm"]),
    )
    _apply_manifest_field_polarity(component, part)
    if upper_pole is not None:
        object.__setattr__(
            component,
            "upper_pole_piece_center_z_mm",
            float(upper_pole.center_z_mm),
        )
        object.__setattr__(
            component,
            "upper_pole_piece_axial_length_mm",
            float(upper_pole.length_mm),
        )
        object.__setattr__(
            component,
            "upper_pole_piece_outer_diameter_mm",
            float(upper_pole.data["mechanical_outer_diameter_mm"]),
        )
        object.__setattr__(
            component,
            "upper_pole_piece_tip_diameter_mm",
            float(upper_pole.data["mechanical_tip_diameter_mm"]),
        )
    if lower_pole is not None:
        object.__setattr__(
            component,
            "lower_pole_piece_center_z_mm",
            float(lower_pole.center_z_mm),
        )
        object.__setattr__(
            component,
            "pole_piece_axial_length_mm",
            float(lower_pole.length_mm),
        )
        object.__setattr__(
            component,
            "pole_piece_outer_diameter_mm",
            float(lower_pole.data["mechanical_outer_diameter_mm"]),
        )
        object.__setattr__(
            component,
            "pole_piece_tip_diameter_mm",
            float(lower_pole.data["mechanical_tip_diameter_mm"]),
        )
        object.__setattr__(
            component,
            "pole_piece_bore_diameter_mm",
            float(lower_pole.data["mechanical_bore_diameter_mm"]),
        )
    if upper_pole is not None and lower_pole is not None:
        object.__setattr__(
            component,
            "pole_piece_center_separation_mm",
            float(lower_pole.center_z_mm)
            - float(upper_pole.center_z_mm),
        )
    object.__setattr__(component, "_position_coupling_ready", True)
    component.validate()
    anchor_key = (
        PROBE_DP12_SCAN_DEFLECTOR
        if PROBE_DP12_SCAN_DEFLECTOR in parts
        else BEAM_DEFLECTOR
    )
    anchor = parts[anchor_key]
    resolved = {
        "anchor": float(anchor.end_z_mm),
        "condenser_stigmator": float(
            parts["condenser_stigmator"].center_z_mm
        ),
        "ac_scan_coil": float(parts["ac_deflector"].center_z_mm),
        "mini_condenser": float(parts["mini_condenser"].center_z_mm),
        "objective_upper_lens": float(
            component.upper_objective_lens_center_z_mm
        ),
        "objective_upper_pole": float(upper_pole.center_z_mm),
        "sample": float(parts["sample"].center_z_mm),
        "objective_aperture": float(
            parts["objective_aperture"].center_z_mm
        ),
        "objective_lower_pole": float(lower_pole.center_z_mm),
        "objective_lower_lens": float(
            component.lower_objective_lens_center_z_mm
        ),
        "objective_lower_lens_start": lower_yoke_start,
        "objective_lower_lens_end": lower_yoke_end,
        "descan_deflector": float(parts["descan_deflector"].center_z_mm),
        "objective_stigmator": float(
            parts["objective_stigmator"].center_z_mm
        ),
        "image_diffraction_deflector": float(
            parts["image_diffraction_deflector"].center_z_mm
        ),
    }
    state._upper_objective_package_anchor_key = anchor_key
    state._upper_objective_package_resolved_positions_mm = resolved
    state._ac_downstream_resolved_positions_mm = resolved
    state._ac_downstream_resolved_mode = (
        "probe"
        if anchor_key == PROBE_DP12_SCAN_DEFLECTOR
        else "standalone"
    )


def _apply_manifest_runtime_geometry(state, parts, assembly):
    gun = state.electron_gun
    if not hasattr(gun, "apply_resolved_manifest_geometry"):
        raise ValueError(
            f"Electron gun {type(gun).__name__} cannot accept a resolved "
            "TOML assembly"
        )
    gun_module = next(
        (module for module in assembly.modules if module.type == "gun"),
        None,
    )
    if gun_module is None:
        raise ValueError("Resolved assembly has no gun module")
    gun.apply_resolved_manifest_geometry(
        parts,
        exit_plane_z_mm=(
            float(gun_module.exit_z_mm)
            - float(gun_module.entrance_z_mm)
        ),
    )
    targets = _state_targets(state)
    for key, target in targets.items():
        part = parts.get(key)
        if part is None:
            continue
        _, item = target
        _set_manifest_authority(item, part)
        if key == OBJECTIVE_LENS:
            continue
        _set_mechanical_geometry(item, part, parts)
        _apply_manifest_field_polarity(item, part)
        _apply_detector_orientation_calibration(item, part)
        if "interaction_centers_local_z_mm" in part.data:
            positions = tuple(
                float(part.center_z_mm)
                + float(value)
                - float(part.data["local_center_z_mm"])
                for value in part.data["interaction_centers_local_z_mm"]
            )
            _set_pair_positions(item, positions)
        if "optical_reference_local_z_mm" in part.data:
            reference = _absolute_part_value(
                part, "optical_reference_local_z_mm"
            )
            _set_single_position(
                item,
                reference,
            )
            anchor_key = getattr(item, "anchor_key", None)
            if (
                anchor_key in parts
                and hasattr(
                    item,
                    "optical_reference_downstream_of_anchor_mm",
                )
            ):
                item.optical_reference_downstream_of_anchor_mm = (
                    reference - float(parts[anchor_key].center_z_mm)
                )
    if "sample" in parts:
        state.sample.z_mm = float(parts["sample"].center_z_mm)
    _apply_objective_manifest_geometry(state, parts)
    objective_aperture = getattr(state, "objective_aperture", None)
    objective_lens = getattr(state, "objective_lens", None)
    if objective_aperture is not None and objective_lens is not None:
        objective_aperture.validate_co_located_with_mechanics(
            state.sample.z_mm
        ).validate_between_poles(objective_lens)
    _apply_energy_filter_manifest_geometry(state, parts)


def _apply_energy_filter_manifest_geometry(state, parts):
    interface = parts.get("energy_filter")
    energy_filter = getattr(state, "energy_filter", None)
    if interface is None or energy_filter is None:
        return
    _set_manifest_authority(energy_filter, interface)
    for field in module_manifest.ENERGY_FILTER_MECHANICAL_METADATA_FIELDS:
        setattr(energy_filter, field, str(interface.data[field]))

    prism = parts[ENERGY_FILTER_TAPERED_PRISM]
    energy_filter.prism_radius_mm = float(prism.data["prism_radius_mm"])
    energy_filter.bend_angle_deg = float(prism.data["bend_angle_deg"])
    energy_filter.prism_radial_field_index = float(
        prism.data["prism_radial_field_index"]
    )
    energy_filter.prism_entrance_s_mm = float(
        prism.data["path_entrance_mm"]
    )
    energy_filter.prism_fringe_mm = float(prism.data["fringe_length_mm"])
    energy_filter.pole_gap_mm = float(prism.data["pole_gap_mm"])
    energy_filter.sector_radial_aperture_mm = float(
        prism.data["radial_clear_half_width_mm"]
    )
    energy_filter._prism_manifest_part_key = str(prism.key)
    energy_filter._prism_geometry_status = str(
        prism.data["mechanical_geometry_status"]
    )
    energy_filter._prism_geometry_source = str(
        prism.data["mechanical_geometry_source"]
    )

    from temsim.physics.finite_multipole_field import SoftEdgeEnvelope
    multipoles = tuple(getattr(energy_filter, "multipoles", ()) or ())
    for index, element in enumerate(multipoles, start=1):
        if element is None:
            continue
        part = parts[ENERGY_FILTER_MULTIPOLE_KEYS[index - 1]]
        _set_manifest_authority(element, part)
        path_value = float(part.data["path_center_mm"])
        position_field = (
            f"multipole_{index:02d}_s_mm"
            if index <= 3
            else f"multipole_{index:02d}_d_mm"
        )
        setattr(energy_filter, position_field, path_value)
        element.bore_radius_m = (
            float(part.data["mechanical_bore_radius_mm"]) * 1.0e-3
        )
        element.outer_radius_m = (
            float(part.data["mechanical_outer_radius_mm"]) * 1.0e-3
        )
        element.housing_length_m = (
            float(part.data["housing_length_mm"]) * 1.0e-3
        )
        element.pole_zero_angle_rad = math.radians(float(
            part.data["pole_zero_angle_deg"]
        ))
        element.field_backend.envelope = SoftEdgeEnvelope(
            length_m=(
                float(part.data["magnetic_support_length_mm"]) * 1.0e-3
            ),
            entrance_soft_edge_m=(
                float(part.data["entrance_soft_edge_mm"]) * 1.0e-3
            ),
            exit_soft_edge_m=(
                float(part.data["exit_soft_edge_mm"]) * 1.0e-3
            ),
        )
        element._mechanical_geometry_status = str(
            part.data["mechanical_geometry_status"]
        )
        element._mechanical_geometry_source = str(
            part.data["mechanical_geometry_source"]
        )
        element._individual_pole_assignment_status = str(
            part.data["individual_pole_assignment_status"]
        )
        element.__post_init__()

    energy_filter.entrance_multipole_s_mm = float(
        energy_filter.multipole_03_s_mm
    )
    energy_filter.exit_multipole_d_mm = float(
        energy_filter.multipole_04_d_mm
    )

    energy_slit = getattr(energy_filter, "energy_slit", None)
    if energy_slit is not None:
        slit_part = parts[ENERGY_FILTER_SLIT]
        _set_manifest_authority(energy_slit, slit_part)
        energy_filter.slit_d_mm = float(slit_part.data["path_center_mm"])
        energy_slit.distance_from_sector_exit_m = (
            float(energy_filter.slit_d_mm) * 1.0e-3
        )
        energy_slit.clear_height_m = (
            float(slit_part.data["clear_height_mm"]) * 1.0e-3
        )
        energy_slit.maximum_gap_m = (
            float(slit_part.data["maximum_gap_mm"]) * 1.0e-3
        )
        energy_slit.blade_thickness_m = (
            float(slit_part.data["blade_thickness_mm"]) * 1.0e-3
        )
        energy_slit._mechanical_geometry_status = str(
            slit_part.data["mechanical_geometry_status"]
        )
        energy_slit._mechanical_geometry_source = str(
            slit_part.data["mechanical_geometry_source"]
        )
        energy_slit.__post_init__()

    dynamic_quad = parts[ENERGY_FILTER_DYNAMIC_FOCUS_QUADRUPOLE]
    energy_filter.dynamic_focus_quadrupole_d_mm = float(
        dynamic_quad.data["path_center_mm"]
    )
    energy_filter.dynamic_focus_quadrupole_length_mm = float(
        dynamic_quad.data["housing_length_mm"]
    )
    energy_filter.dynamic_focus_quadrupole_bore_mm = float(
        dynamic_quad.data["clear_bore_diameter_mm"]
    )
    energy_filter.dynamic_focus_quadrupole_outer_mm = float(
        dynamic_quad.data["mechanical_outer_diameter_mm"]
    )
    energy_filter.dynamic_focus_quadrupole_model_status = str(
        dynamic_quad.data["optical_model_status"]
    )
    energy_filter.dynamic_focus_quadrupole_geometry_status = str(
        dynamic_quad.data["mechanical_geometry_status"]
    )
    energy_filter.dynamic_focus_quadrupole_geometry_source = str(
        dynamic_quad.data["mechanical_geometry_source"]
    )

    detector_end_parts = (
        (
            ENERGY_FILTER_BIAS_TUBE,
            "bias_tube",
            "bias_tube_d_mm",
            ("housing_length_mm", "clear_bore_diameter_mm",
             "mechanical_outer_diameter_mm", "maximum_abs_offset_ev"),
        ),
        (
            ENERGY_FILTER_SHUTTER,
            "fast_shutter",
            "fast_shutter_d_mm",
            ("electrode_length_mm", "electrode_gap_mm",
             "mechanical_outer_diameter_mm"),
        ),
        (
            ENERGY_FILTER_CAMERA_DEFLECTOR,
            "camera_deflector",
            "camera_deflector_d_mm",
            ("electrode_length_mm", "electrode_gap_mm",
             "mechanical_outer_diameter_mm"),
        ),
    )
    for key, attribute, position_attribute, geometry_fields in detector_end_parts:
        device_part = parts[key]
        device = getattr(energy_filter, attribute, None)
        setattr(
            energy_filter,
            position_attribute,
            float(device_part.data["path_center_mm"]),
        )
        if device is None:
            continue
        _set_manifest_authority(device, device_part)
        for field in geometry_fields:
            setattr(device, field, float(device_part.data[field]))
        device._mechanical_geometry_status = str(
            device_part.data["mechanical_geometry_status"]
        )
        device._mechanical_geometry_source = str(
            device_part.data["mechanical_geometry_source"]
        )
        if key == ENERGY_FILTER_BIAS_TUBE:
            device.offset_range_status = str(
                device_part.data["offset_range_status"]
            )
        device.validate()

    output_plane = parts[ENERGY_FILTER_EFTEM_OUTPUT_PLANE]
    energy_filter.output_detector_d_mm = float(
        output_plane.data["path_center_mm"]
    )
    energy_filter.output_detector_width_mm = float(
        output_plane.data["active_width_mm"]
    )
    energy_filter.output_plane_geometry_status = str(
        output_plane.data["mechanical_geometry_status"]
    )
    energy_filter.output_plane_geometry_source = str(
        output_plane.data["mechanical_geometry_source"]
    )

    zebra_part = parts[ENERGY_FILTER_ZEBRA]
    energy_filter.zebra_detector_d_mm = float(
        zebra_part.data["path_center_mm"]
    )
    energy_filter.eels_plane_offset_mm = (
        energy_filter.zebra_detector_d_mm
        - energy_filter.output_detector_d_mm
    )
    zebra = getattr(energy_filter, "zebra_detector", None)
    if zebra is not None:
        _set_manifest_authority(zebra, zebra_part)
        zebra.strip_count = int(zebra_part.data["strip_count"])
        zebra.pixels_per_strip = int(zebra_part.data["pixels_per_strip"])
        zebra.spectral_clear_height_mm = float(
            zebra_part.data["strip_active_height_mm"]
        )
        zebra.alignment_pixels_x = int(
            zebra_part.data["alignment_pixels_non_dispersive"]
        )
        zebra.alignment_pixels_y = int(
            zebra_part.data["alignment_pixels_dispersive"]
        )
        zebra.pixel_size_um = float(
            zebra_part.data["strip_pixel_pitch_um"]
        )
        zebra.maximum_spectra_per_s = float(
            zebra_part.data["maximum_spectra_per_s"]
        )
        zebra.provisional_strip_center_pitch_mm = float(
            zebra_part.data["provisional_strip_center_pitch_mm"]
        )
        zebra.strip_center_pitch_status = str(
            zebra_part.data["strip_center_pitch_status"]
        )
        zebra.external_envelope_status = str(
            zebra_part.data["external_envelope_status"]
        )
        zebra._detector_geometry_source = str(
            zebra_part.data["detector_geometry_source"]
        )
        zebra.validate()

    energy_filter.m12_frames_placed = False
    if len(multipoles) == 10:
        from temsim.optics.energy_filter_sector import (
            place_m12_in_sector_frames,
        )
        place_m12_in_sector_frames(energy_filter)
    from temsim.optics.energy_filter_m12 import rigidity_scale
    from temsim.optics.energy_filter_sector import sector_plateau_field_t
    reference_voltage = float(energy_filter.voltage_reference_kv)
    matched_voltage = float(energy_filter.matched_voltage_kv)
    energy_filter.sector_reference_field_t = sector_plateau_field_t(
        reference_voltage,
        float(energy_filter.prism_radius_mm) * 1.0e-3,
        math.radians(float(energy_filter.bend_angle_deg)),
        float(energy_filter.prism_fringe_mm) * 1.0e-3,
        float(energy_filter.prism_radial_field_index),
    )
    energy_filter.sector_field_t = (
        float(energy_filter.sector_reference_field_t)
        * rigidity_scale(matched_voltage, reference_voltage)
    )


def _shift_target(target, shift):
    kind, item = target
    if kind == "pair":
        upper_descriptor = getattr(type(item), "upper_z_mm", None)
        lower_descriptor = getattr(type(item), "lower_z_mm", None)
        read_only_pair = (
            isinstance(upper_descriptor, property)
            and upper_descriptor.fset is None
            and isinstance(lower_descriptor, property)
            and lower_descriptor.fset is None
        )
        if read_only_pair and hasattr(item, "z_mm"):
            item.z_mm = float(item.z_mm) + shift
        else:
            item.upper_z_mm = float(item.upper_z_mm) + shift
            item.lower_z_mm = float(item.lower_z_mm) + shift
    else:
        item.z_mm = float(item.z_mm) + shift


def clear_module_state_offsets(state):
    """Clear obsolete effective-axis bookkeeping without moving components."""

    state._module_optical_offsets_mm = {}
    state._module_gun_offsets_mm = {}


def apply_module_state_offsets(state, configuration, layout, scale):
    """Compatibility entry point; absolute TOML positions replace offsets."""

    del layout, scale
    return apply_column_manifest_geometry(
        state,
        configuration,
        getattr(configuration, "resolved_assembly", None),
    )
