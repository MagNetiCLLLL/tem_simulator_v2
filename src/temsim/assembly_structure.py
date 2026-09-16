"""Read-only assembly navigation and path-independent component identities.

This compatibility layer consumes an already resolved assembly. It never
resolves positions, changes ownership, opens files, or participates in transport.
The v1 identity namespace and seeds are a persistence contract. A later module
split/key rename must explicitly carry the exported IDs through identity_map.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from types import MappingProxyType
from uuid import UUID, uuid5

from temsim.component_keys import ENERGY_FILTER_INTERNAL_KEYS, IMAGE_CORRECTOR_KEYS, PROBE_CORRECTOR_KEYS


SCHEMA = "assembly-structure-v1"
NAMESPACE = UUID("8affc27a-0d10-4baa-89ef-54ca435fb428")
GROUPS = (
    ("gun", "Electron gun and acceleration"),
    ("blanker", "Beam blanking"),
    ("illumination", "Illumination"),
    ("probe_correction", "Probe correction"),
    ("objective", "Objective and specimen"),
    ("imaging", "Imaging and projection"),
    ("detection", "Detection and recording"),
    ("energy_filter", "Energy filter"),
    ("unclassified", "Other components"),
)


def stable_id(kind, *keys):
    """UUIDs do not consume file paths, display labels, dimensions or order."""
    return str(uuid5(NAMESPACE, json.dumps([kind, *keys], ensure_ascii=False, separators=(",", ":"))))


def _root_group(part, module_type):
    # Mechanical children inherit their parent's group, even when their optical
    # function is different (e.g. EDS or scan coils in the objective assembly).
    key = part.key
    if module_type == "gun":
        return "gun"
    if module_type == "beam_blanker":
        return "blanker"
    if key == "energy_filter" or key in ENERGY_FILTER_INTERNAL_KEYS or part.branch == "energy_filter":
        return "energy_filter"
    if key in IMAGE_CORRECTOR_KEYS:
        return "imaging"
    if key in PROBE_CORRECTOR_KEYS:
        return "probe_correction"
    if key in {"objective_lens", "objective_aperture", "sample", "sample_stage"}:
        return "objective"
    if part.branch == "detection" or key == "projection_chamber_dpa_aperture":
        return "detection"
    if key in {"condenser_lens_1", "condenser_lens_2", "condenser_lens_3",
               "condenser_aperture_2", "condenser_aperture_3", "condenser_deflector",
               "beam_deflector", "condenser_stigmator"}:
        return "illumination"
    if key in {"selected_area_aperture", "diffraction_lens", "intermediate_lens",
               "projector_lens_1", "projector_lens_2", "diffraction_stigmator",
               "image_diffraction_deflector"}:
        return "imaging"
    return "unclassified"


@dataclass(frozen=True, slots=True)
class ComponentIdentity:
    instance_id: str
    key: str
    module_key: str
    source_file: str
    legacy_authority: str
    name: str
    parent_id: str | None
    group_key: str
    start_z_mm: float
    center_z_mm: float
    end_z_mm: float
    definition_reference: str | None
    path_coordinate: str | None
    path_reference: str | None
    path_center_mm: float | None

    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AssemblyStructure:
    components: tuple[ComponentIdentity, ...]
    module_ids: tuple[tuple[str, str], ...]

    @property
    def by_key(self):
        return MappingProxyType({row.key: row for row in self.components})

    @property
    def by_id(self):
        return MappingProxyType({row.instance_id: row for row in self.components})

    def resolve(self, reference):
        """Read current IDs, canonical keys and historical path authorities."""
        matches = [row for row in self.components if reference in
                   {row.instance_id, row.key, row.legacy_authority}]
        if len(matches) != 1:
            raise KeyError(f"Assembly reference is absent or ambiguous: {reference}")
        return matches[0]

    def identity_map(self):
        return {(row.module_key, row.key): row.instance_id for row in self.components}

    def to_dict(self):
        return {"schema": SCHEMA, "identity_namespace": str(NAMESPACE),
                "coordinate_authority": "unchanged resolved assembly; groups are navigation only",
                "groups": [{"key": key, "id": stable_id("group", key), "name": label} for key, label in GROUPS],
                "modules": [{"key": key, "id": value} for key, value in self.module_ids],
                "components": [row.to_dict() for row in self.components]}


def identity_map_from_document(document):
    """Read an exported binding map; geometry in the export is never applied."""
    if document.get("schema") != SCHEMA or document.get("identity_namespace") != str(NAMESPACE):
        raise ValueError("Unsupported assembly identity map")
    result, used = {}, set()
    for row in document["components"]:
        locator = row["module_key"], row["key"]
        identity = str(UUID(row["instance_id"]))
        if locator in result or identity in used:
            raise ValueError("Duplicate component binding in identity map")
        result[locator] = identity
        used.add(identity)
    return result


def build_assembly_structure(assembly, *, identity_map=None):
    """Project a legacy captured assembly without altering it or its snapshots.

    Explicit IDs are reserved for reviewed migrations: a moved component keeps
    its ID, while an independent copy receives a new one. In v1 the fallback
    seed is the existing module key + component key, never the storage path.
    Shared definition references remain references; they do not merge instances.
    """
    parts = {part.key: part for part in assembly.parts}
    modules = {module.key: module.type for module in assembly.modules}
    if len(parts) != len(assembly.parts) or len(modules) != len(assembly.modules):
        raise ValueError("Assembly structure needs unique component and module keys")
    overrides = dict(identity_map or {})
    locators = {(part.module_key, part.key) for part in assembly.parts}
    if overrides.keys() - locators:
        raise ValueError("Identity map contains an uninstalled component")
    ids = {part.key: str(UUID(overrides[(part.module_key, part.key)]))
           if (part.module_key, part.key) in overrides else stable_id("instance", part.module_key, part.key)
           for part in assembly.parts}
    if len(set(ids.values())) != len(ids):
        raise ValueError("Independent component instances cannot share an identity")
    reserved = {stable_id("group", key) for key, _ in GROUPS} | {stable_id("module", key) for key in modules}
    if reserved.intersection(ids.values()):
        raise ValueError("Component identity collides with an assembly group or module")
    groups, visiting = {}, set()
    def group(key):
        if key in groups:
            return groups[key]
        if key in visiting:
            raise ValueError(f"Cyclic assembly parent relationship: {key}")
        visiting.add(key)
        part = parts[key]
        if part.module_key not in modules:
            raise ValueError(f"Unknown component module: {part.module_key}")
        if part.parent_key:
            if part.parent_key not in parts:
                raise ValueError(f"Missing assembly parent for {key}: {part.parent_key}")
            value = group(part.parent_key)
        else:
            value = _root_group(part, modules[part.module_key])
        visiting.remove(key)
        groups[key] = value
        return value
    components = []
    for part in assembly.parts:
        components.append(ComponentIdentity(
            ids[part.key], part.key, part.module_key, part.source_file,
            part.definition_id, part.name, ids.get(part.parent_key), group(part.key),
            part.start_z_mm, part.center_z_mm, part.end_z_mm,
            part.data.get("tip_definition_file"),
            part.data.get("path_coordinate", "curvilinear_s_mm" if part.data.get("branch_path_only") else None),
            part.data.get("path_reference"),
            part.data.get("path_center_mm")))
    return AssemblyStructure(tuple(components), tuple((key, stable_id("module", key)) for key in modules))
