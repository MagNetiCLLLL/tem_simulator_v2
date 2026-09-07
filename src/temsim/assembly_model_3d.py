"""Renderer-neutral physical assembly surfaces in global column millimetres.

This is a view of the captured resolved assembly, not a CAD import or a field
solve. Existing part builders own all shapes and their approximation labels.
The only added transform is the resolved module-local to global Z translation.
"""
from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
from types import MappingProxyType

import numpy as np

from temsim.part_model_3d import TriangleMesh, _extent, _segments, part_model_from_document
from temsim.part_model_apertures import _opening, is_strip_aperture
from temsim.part_materials import configured_region_colour


@dataclass(frozen=True)
class AssemblyModel3D:
    meshes: tuple[TriangleMesh, ...]
    notes: tuple[str, ...] = ()
    omitted_keys: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


def _document_value(value):
    """Detach frozen TOML containers for existing list-based schema readers."""
    if isinstance(value, Mapping):
        return {str(key): _document_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_document_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _contexts(assembly):
    parts = tuple(assembly.parts)
    if len({part.key for part in parts}) != len(parts):
        raise ValueError("Assembly part keys must be unique")
    documents = {}
    for module in assembly.modules:
        documents[module.key] = {
            "module": {"key": module.key},
            "parts": [_document_value(part.data) for part in module.parts],
        }
    for part in parts:
        document = documents.setdefault(part.module_key, {
            "module": {"key": part.module_key}, "parts": [],
        })
        row = _document_value(part.data)
        # Active resolved data wins while inactive source parents remain
        # available solely as profile context, never as extra rendered parts.
        for index, existing in enumerate(document["parts"]):
            if existing["key"] == part.key:
                document["parts"][index] = row
                break
        else:
            document["parts"].append(row)
    return parts, documents


def _runtime_dependencies(parts, runtime_values):
    values = runtime_values or {}
    dependencies = []
    for part in parts:
        row = part.data
        if (not is_strip_aperture(row)
                or row.get("model_3d", {}).get("base", {}).get("kind", "existing") != "existing"):
            continue
        try:
            opening = _opening(row, values.get(part.key), 0)
        except (TypeError, ValueError, OverflowError) as exc:
            opening = ("invalid opening", str(exc))
        dependencies.append((part.key, opening))
    return dependencies


def _liner_rows(assembly):
    return [dict(
        key=segment.key, name=segment.name, mechanical_profile="vacuum_liner",
        local_start_z_mm=segment.start_z_mm, local_center_z_mm=(segment.start_z_mm + segment.end_z_mm) / 2,
        local_end_z_mm=segment.end_z_mm, length_mm=segment.end_z_mm - segment.start_z_mm,
        mechanical_inner_diameter_mm=segment.inner_diameter_mm,
        mechanical_outer_diameter_mm=segment.outer_diameter_mm,
        material_class="nonmagnetic_vacuum_liner",
    ) for segment in getattr(assembly, "vacuum_liner_segments", ())]


def assembly_model_fingerprint(assembly, runtime_values=None) -> str:
    """Exact captured geometry identity, excluding unrelated live excitation.

    Aperture opening/XY changes matter only where the reused part builder
    consumes them. No file is reread and no optical result identity is changed.
    Tessellation is a renderer setting and must be keyed by the caller too.
    """
    parts, documents = _contexts(assembly)
    payload = {
        "schema": "resolved-assembly-surfaces-v1",
        "documents": documents,
        "placements": [(part.module_key, part.key, part.start_z_mm, part.center_z_mm,
                        part.end_z_mm, part.length_mm, part.parent_key) for part in parts],
        "liners": _liner_rows(assembly),
        "runtime": _runtime_dependencies(parts, runtime_values),
    }
    encoded = json.dumps(_document_value(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _immutable_array(value):
    source = np.asarray(value)
    if source.dtype.hasobject:
        # Semantic group labels are strings; object-pointer buffers must never
        # be stored via frombuffer or retained from the source mesh.
        source = source.astype(str)
    return np.frombuffer(source.tobytes(), dtype=source.dtype).reshape(source.shape)


def _immutable(value):
    if isinstance(value, np.ndarray):
        return _immutable_array(value)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _immutable(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_immutable(item) for item in value)
    return value


def _global_mesh(mesh, shift):
    vertices = np.asarray(mesh.vertices, dtype=float).copy()
    vertices[:, 2] += shift
    edges = []
    for edge in mesh.edges:
        record = dict(edge)
        points = np.asarray(record["vertices"], dtype=float).copy()
        points[:, 2] += shift
        record["vertices"] = points
        edges.append(_immutable(record))
    if not np.all(np.isfinite(vertices)):
        raise ValueError(f"{mesh.key}: resolved vertices must remain finite")
    return replace(
        mesh, vertices=_immutable_array(vertices), faces=_immutable_array(mesh.faces),
        face_groups=(_immutable_array(mesh.face_groups) if mesh.face_groups is not None else None),
        surfaces=_immutable(mesh.surfaces), edges=tuple(edges), color=tuple(mesh.color),
    )


def _omission_reason(part, physical_parent_keys):
    row = part.data
    if row.get("branch_path_only"):
        return "branch-path component; its curvilinear geometry belongs to Energy Filter"
    if "model_3d" in row:
        return None
    if ((row.get("mechanical_profile") == "magnetic_lens_assembly"
         or row.get("mechanical_part_role") == "optical_parent")
            and part.key in physical_parent_keys):
        return "optical parent envelope replaced by its physical children"
    if (row.get("virtual") or row.get("virtual_only")
            or "reference_plane" in str(row.get("mechanical_profile", ""))):
        return "virtual/reference plane, not material"
    start, end = _extent(row)
    if start == end:
        return "zero-thickness reference/envelope plane, not a solid"
    return None


def assembly_model_from_assembly(assembly, *, runtime_values=None, angular_segments=16) -> AssemblyModel3D:
    """Build each active physical part once, preserving partial-model errors.

    Supported runtime strip apertures use their current opening and XY offset.
    Insertion/retraction mechanisms and runtime detector positions are not
    inferred; all axial placement comes from the supplied ResolvedAssembly.
    """
    count = _segments(angular_segments)
    parts, documents = _contexts(assembly)
    meshes, notes, omitted, errors = [], [], [], []
    physical_parent_keys = set()
    for part in parts:
        if (part.data.get("mechanical_profile") and not part.data.get("branch_path_only")
                and "reference_plane" not in str(part.data.get("mechanical_profile"))):
            if part.parent_key:
                physical_parent_keys.add(part.parent_key)
            physical_parent_keys.update(part.data.get("magnetic_lens_keys", ()))
    for part in parts:
        try:
            reason = _omission_reason(part, physical_parent_keys)
            if reason:
                omitted.append(part.key)
                notes.append(f"{part.key}: omitted {reason}.")
                continue
            local_centre = float(part.data["local_center_z_mm"])
            shift = float(part.center_z_mm) - local_centre
            if not math.isfinite(shift):
                raise ValueError("Resolved axial translation must be finite")
            for endpoint, field in ((part.start_z_mm, "local_start_z_mm"),
                                    (part.end_z_mm, "local_end_z_mm")):
                if not math.isclose(float(endpoint), float(part.data[field]) + shift,
                                    rel_tol=0, abs_tol=1e-8):
                    raise ValueError("Resolved axial span differs from source geometry; translation alone is insufficient")
            model = part_model_from_document(
                documents[part.module_key], part.key, angular_segments=count,
                include_children=False, runtime_values=runtime_values,
            )
            source = next(row for row in documents[part.module_key]["parts"] if row["key"] == part.key)
            global_meshes = tuple(_global_mesh(replace(
                mesh, color=configured_region_colour(source, mesh.region, mesh.color)), shift)
                for mesh in model.meshes)
            meshes.extend(global_meshes)
            notes.extend(model.notes)
            if not global_meshes:
                omitted.append(part.key)
            for mesh in global_meshes:
                if not mesh.is_exact:
                    notes.append(f"{mesh.key}: {mesh.description}.")
        except (ValueError, TypeError, KeyError, OverflowError, RuntimeError) as exc:
            omitted.append(part.key)
            errors.append(f"{part.key}: {exc}")
    # These material tubes are resolved assembly members but are synthesized
    # from module geometry, not repeated in each module's parts array.
    active_keys = {part.key for part in parts}
    for row in _liner_rows(assembly):
        if row["key"] in active_keys:
            continue
        try:
            model = part_model_from_document({"parts": [row]}, row["key"],
                                             angular_segments=count, include_children=False)
            meshes.extend(_global_mesh(mesh, 0.0) for mesh in model.meshes)
            notes.extend(model.notes)
        except (ValueError, TypeError, KeyError, OverflowError, RuntimeError) as exc:
            omitted.append(row["key"])
            errors.append(f"{row['key']}: {exc}")
    notes.insert(0, "Configured mechanical positions in global column mm; runtime strip openings and XY offsets only. "
                    "No insertion mechanism, detector motion or new lens field is inferred.")
    return AssemblyModel3D(tuple(meshes), tuple(dict.fromkeys(notes)),
                           tuple(dict.fromkeys(omitted)), tuple(errors))
