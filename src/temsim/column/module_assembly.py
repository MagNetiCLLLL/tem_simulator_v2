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
    component_keys = {component.key for component in layout}
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
        if not bool(parts[key].data.get("mechanical_only", False))
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
        if shape is not None:
            shape_updates = {"axial_length_mm": end_z_mm - start_z_mm}
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
        assembly = resolve_module_assembly(configuration)
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
    _apply_manifest_runtime_geometry(state, parts)
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
    for items in (
        state.lenses,
        state.apertures,
        state.stigmators,
        getattr(state, "corrector_elements", ()),
        getattr(state, "recording_planes", ()),
    ):
        for item in items:
            if hasattr(item, "z_mm"):
                targets[item.key] = ("single", item)
    for pair in state.deflectors:
        targets[pair.key] = ("pair", pair)
    if hasattr(state.sample, "z_mm"):
        targets["sample"] = ("single", state.sample)
    if "obj_stig" in targets:
        targets["objective_stigmator"] = targets["obj_stig"]
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
    for attribute in (
        "mechanical_outer_diameter_mm",
        "mechanical_clear_bore_diameter_mm",
        "mechanical_bore_diameter_mm",
        "bore_diameter_mm",
        "pole_gap_mm",
        "effective_length_mm",
        "effective_thickness_mm",
        "mechanical_coil_length_mm",
        "mechanical_inter_coil_gap_mm",
        "plate_thickness_mm",
        "maximum_radius_mm",
        "outer_width_mm",
        "inner_diameter_mm",
    ):
        if attribute not in part.data or not hasattr(item, attribute):
            continue
        try:
            setattr(item, attribute, float(part.data[attribute]))
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


def _apply_objective_manifest_geometry(state, parts):
    component = getattr(state, "objective_lens", None)
    part = parts.get(OBJECTIVE_LENS)
    if component is None or part is None:
        return
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
    object.__setattr__(component, "polarity", int(part.data["polarity"]))
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


def _apply_manifest_runtime_geometry(state, parts):
    if hasattr(state.electron_gun, "apply_manifest_geometry"):
        state.electron_gun.apply_manifest_geometry(
            bool(getattr(state, "monochromator_installed", False))
        )
    targets = _state_targets(state)
    for key, target in targets.items():
        part = parts.get(key)
        if part is None or key == OBJECTIVE_LENS:
            continue
        _, item = target
        _set_mechanical_geometry(item, part, parts)
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
    _apply_energy_filter_manifest_geometry(state, parts)


def _apply_energy_filter_manifest_geometry(state, parts):
    part = parts.get("energy_filter")
    energy_filter = getattr(state, "energy_filter", None)
    if part is None or energy_filter is None:
        return
    for field in module_manifest.ENERGY_FILTER_GEOMETRY_FIELDS:
        setattr(energy_filter, field, float(part.data[field]))
    from temsim.physics.finite_multipole_field import SoftEdgeEnvelope
    for role, element in (
        ("entrance", getattr(energy_filter, "entrance_m12", None)),
        ("exit", getattr(energy_filter, "exit_m12", None)),
    ):
        if element is None:
            continue
        element.bore_radius_m = (
            float(part.data[f"{role}_m12_bore_radius_mm"]) * 1.0e-3
        )
        element.outer_radius_m = (
            float(part.data[f"{role}_m12_outer_radius_mm"]) * 1.0e-3
        )
        element.pole_zero_angle_rad = math.radians(float(
            part.data[f"{role}_m12_pole_zero_angle_deg"]
        ))
        element.field_backend.envelope = SoftEdgeEnvelope(
            length_m=(
                float(part.data[f"{role}_m12_length_mm"]) * 1.0e-3
            ),
            entrance_soft_edge_m=(
                float(
                    part.data[
                        f"{role}_m12_entrance_soft_edge_mm"
                    ]
                )
                * 1.0e-3
            ),
            exit_soft_edge_m=(
                float(
                    part.data[f"{role}_m12_exit_soft_edge_mm"]
                )
                * 1.0e-3
            ),
        )
        element.__post_init__()
    energy_slit = getattr(energy_filter, "energy_slit", None)
    if energy_slit is not None:
        energy_slit.distance_from_sector_exit_m = (
            float(energy_filter.slit_d_mm) * 1.0e-3
        )
        energy_slit.clear_height_m = (
            float(part.data["slit_clear_height_mm"]) * 1.0e-3
        )
        energy_slit.maximum_gap_m = (
            float(part.data["slit_maximum_gap_mm"]) * 1.0e-3
        )
        energy_slit.blade_thickness_m = (
            float(part.data["slit_blade_thickness_mm"]) * 1.0e-3
        )
        energy_slit.__post_init__()
    energy_filter.m12_frames_placed = False
    if (
        getattr(energy_filter, "entrance_m12", None) is not None
        and getattr(energy_filter, "exit_m12", None) is not None
    ):
        from temsim.optics.energy_filter_sector import (
            place_m12_in_sector_frames,
        )
        place_m12_in_sector_frames(energy_filter)


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
