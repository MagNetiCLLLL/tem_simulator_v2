"""Named physical sections and anchor identities from the captured assembly.

These intervals describe component extents, not separate vacuum chambers.
They can overlap. No files are read and no coordinates are resolved here.
"""
from dataclasses import dataclass
import posixpath

from temsim.assembly_structure import GROUPS, build_assembly_structure, stable_id


@dataclass(frozen=True)
class AssemblySection:
    key: str
    name: str
    module_key: str
    source_file: str
    origin_z_mm: float
    start_z_mm: float
    end_z_mm: float
    part_keys: tuple[str, ...]
    kind: str


def component_anchor(part, point):
    if point not in {"start", "center", "end"}:
        raise ValueError("Choose a component start, center or end")
    return f"component:{stable_id('instance', part.module_key, part.key)}.{point}"


def assembly_sections(assembly):
    """Prefer actual subassemblies; classify remaining parts by function.

    Historical snapshots without composition metadata keep functional sections.
    A missing saved subassembly anchor stays missing, never guessed from files.
    Branch-path-only filter parts retain their separate s coordinates.
    """
    if assembly is None:
        return ()
    identities = build_assembly_structure(assembly).by_key
    result = []
    previous_exit = 0.0
    for module in assembly.modules:
        parts = {p.key: p for p in assembly.parts if p.module_key == module.key}
        origin = next((parts[p.key].center_z_mm-p.center_z_mm for p in module.parts if p.key in parts),
                      previous_exit-module.entrance_z_mm)
        previous_exit = origin+module.exit_z_mm
        owned = set()

        def add(key, name, keys, source, local_origin, kind):
            # Z envelopes are for the straight column only. Filter path s must
            # not masquerade as Z when annotating the vacuum beam path.
            axial = [parts[k] for k in keys if not parts[k].data.get("branch_path_only")]
            if not axial:
                return
            result.append(AssemblySection(key, name, module.key, source, local_origin,
                min(p.start_z_mm for p in axial), max(p.end_z_mm for p in axial), tuple(keys), kind))

        for row in module.geometry.get("navigation_subassemblies", ()):
            keys = tuple(row["part_keys"])
            if not keys or len(set(keys)) != len(keys) or set(keys)-parts.keys() or owned.intersection(keys):
                raise ValueError(f"Invalid captured subassembly membership: {row['key']}")
            owned.update(keys)
            source = posixpath.normpath(posixpath.join(posixpath.dirname(module.source_file.replace('\\', '/')), row["file"]))
            key = "subassembly:"+stable_id("subassembly", module.key, row["key"])
            add(key, row["name"], keys, source, origin+row["origin_z_mm"], "subassembly")
        for group, label in GROUPS:
            keys = tuple(p.key for p in parts.values() if p.key not in owned and identities[p.key].group_key == group)
            if keys:
                z = min(parts[k].start_z_mm for k in keys)
                add("section:"+stable_id("section", module.key, group), label, keys,
                    module.source_file, z, "functional")
    return tuple(sorted(result, key=lambda row: (row.start_z_mm, row.end_z_mm, row.key)))


def section_by_component(assembly):
    return {key: section for section in assembly_sections(assembly) for key in section.part_keys}
