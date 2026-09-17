"""Reusable physical part files and explicit module-local axial placements.

The legacy module remains the instance/optical authority. Storage composition
does not create a source, change a component key or reinterpret filter path s.
This version supports one level of independent subassemblies, with acyclic
constraints between their parts. Missing/cyclic references fail explicitly.
"""
from copy import deepcopy
from pathlib import Path
from temsim import input_io
import math
import tomllib

from temsim.component_operations import translated_component


POINTS = {"start": "local_start_z_mm", "center": "local_center_z_mm", "end": "local_end_z_mm"}


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def sources(document, path):
    """Validated placement definitions and their independent storage files."""
    result, keys, paths = [], set(), set()
    entries = document.get("subassemblies", ())
    if not isinstance(entries, (list, tuple)):
        raise ValueError("Subassemblies must be an array of tables")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each subassembly must be a table")
        key, reference = entry.get("key"), entry.get("file")
        if not isinstance(key, str) or not key or key in keys:
            raise ValueError("Subassemblies need unique non-empty keys")
        if not isinstance(reference, str) or not reference or Path(reference).is_absolute() or Path(reference).drive:
            raise ValueError("Subassembly file must be a portable relative path")
        source = (Path(path).resolve().parent / reference).resolve()
        if source == Path(path).resolve() or source in paths:
            raise ValueError("A subassembly file may occur only once in a module; make an independent copy first")
        keys.add(key)
        paths.add(source)
        result.append((entry, source))
    return result


def definitions(document, path):
    result = []
    for entry, source in sources(document, path):
        try:
            child = tomllib.loads(input_io.read_text(source, encoding="utf-8-sig"))
        except OSError as exc:
            raise ValueError(f"Subassembly file is unavailable: {source}") from exc
        if child.get("subassemblies"):
            raise ValueError("Nested subassemblies are not supported; use independent part files")
        if child.get("module", {}).get("type") != "subassembly":
            raise ValueError("A referenced part file must declare module.type = subassembly")
        if child.get("coordinate_system") != "module_local_z_mm" or not child.get("parts"):
            raise ValueError(f"Subassembly must contain module-local parts: {source}")
        if any("tip_definition_file" in part for part in child["parts"]):
            raise ValueError("Keep the shared FEG tip in its gun authority; subassemblies must be independent")
        result.append((entry, source, child))
    return result


def dependencies(document, path):
    return {source: input_io.read_bytes(source) for _, source, _ in definitions(document, path)}


def ownership(document, path):
    owners = {}
    for entry, _, child in definitions(document, path):
        for part in child["parts"]:
            key = part["key"]
            if key in owners:
                raise ValueError(f"Duplicate subassembly component key: {key}")
            owners[key] = entry["key"]
    return owners


def _placement(entry):
    value = entry.get("placement", {})
    if not isinstance(value, dict):
        raise ValueError("Subassembly placement must be a table")
    mode = value.get("mode")
    if mode == "fixed":
        if set(value) != {"mode", "origin_z_mm"}:
            raise ValueError("Fixed placement requires only mode and origin_z_mm")
        _number(value["origin_z_mm"], "Origin Z")
    elif mode == "anchor":
        if set(value) != {"mode", "reference_part", "reference_point", "offset_mm", "local_datum_mm"}:
            raise ValueError("Anchor placement requires a component, start/center/end, offset and local datum")
        if value["reference_point"] not in POINTS:
            raise ValueError("Choose start, center or end for an anchor")
        _number(value["offset_mm"], "Anchor offset")
        _number(value["local_datum_mm"], "Subassembly local datum")
    else:
        raise ValueError("Choose fixed or anchor subassembly placement")
    return value


def _place(document, local_parts, owners):
    entries = {entry["key"]: entry for entry in document.get("subassemblies", ())}
    parts = {part["key"]: part for part in local_parts}
    if len(parts) != len(local_parts):
        raise ValueError("Duplicate component key across module and subassemblies")
    origins, active = {}, set()

    def origin(key):
        if key in origins:
            return origins[key]
        if key in active:
            raise ValueError(f"Cyclic subassembly placement at {key}")
        active.add(key)
        placement = _placement(entries[key])
        if placement["mode"] == "fixed":
            value = float(placement["origin_z_mm"])
        else:
            ref = placement["reference_part"]
            if ref not in parts:
                raise ValueError(f"Missing placement anchor component: {ref}")
            parent_origin = origin(owners[ref]) if ref in owners else 0.0
            value = (_number(parts[ref][POINTS[placement["reference_point"]]], "Anchor Z")
                     + parent_origin + placement["offset_mm"] - placement["local_datum_mm"])
        origins[key] = _number(value, "Resolved subassembly origin")
        active.remove(key)
        return origins[key]

    for key in entries:
        origin(key)
    result = deepcopy(document)
    result["parts"] = [translated_component(part, origins.get(owners.get(part["key"]), 0.0))
                       for part in local_parts]
    result["parts"].sort(key=lambda part: part["order"])
    return result, origins


def resolve_document(document, path, *, capture_navigation=False):
    if not document.get("subassemblies"):
        return deepcopy(document)
    local_parts, owners = list(deepcopy(document.get("parts", []))), {}
    navigation = []
    for entry, _, child in definitions(document, path):
        navigation.append({"key": entry["key"], "name": entry.get("name", entry["key"]),
                           "file": entry["file"],
                           "part_keys": [part["key"] for part in child["parts"]]})
        for part in child["parts"]:
            if part["key"] in owners:
                raise ValueError(f"Duplicate subassembly component key: {part['key']}")
            owners[part["key"]] = entry["key"]
            local_parts.append(part)
    result, origins = _place(document, local_parts, owners)
    if capture_navigation:
        # Capture from the same definition read that supplied the physical parts.
        # Runtime navigation / vacuum never reopens potentially changed files.
        for row in navigation:
            row["origin_z_mm"] = origins[row["key"]]
        result["_navigation_subassemblies"] = navigation
    return result


def resolved_origins(document):
    """Read origins from a materialized, already-placed draft."""
    parts = {part["key"]: part for part in document["parts"]}
    origins = {}
    for entry in document.get("subassemblies", ()):
        placement = _placement(entry)
        if placement["mode"] == "fixed":
            value = float(placement["origin_z_mm"])
        else:
            ref = placement["reference_part"]
            if ref not in parts:
                raise ValueError(f"Missing placement anchor component: {ref}")
            value = (parts[ref][POINTS[placement["reference_point"]]]
                     + placement["offset_mm"] - placement["local_datum_mm"])
        origins[entry["key"]] = value
    return origins


def reflow(before, candidate, path, *, owners=None):
    """Keep local geometry edits, then follow only explicitly declared anchors."""
    if not before.get("subassemblies"):
        return candidate
    owners = ownership(before, path) if owners is None else owners
    old = resolved_origins(before)
    local = [translated_component(part, -old.get(owners.get(part["key"]), 0.0))
             for part in candidate["parts"]]
    return _place(candidate, local, owners)[0]


def storage_documents(raw, resolved, path):
    """Separate a reviewed flat draft into its original independent files."""
    owners, origins = ownership(raw, path), resolved_origins(resolved)
    # Re-evaluate the graph even for drafts prepared by another caller.
    local = [translated_component(part, -origins.get(owners.get(part["key"]), 0.0))
             for part in resolved["parts"]]
    checked, _ = _place(resolved, local, owners)
    if checked != resolved:
        raise ValueError("The staged geometry does not satisfy its subassembly placements")
    by_owner = {entry["key"]: [] for entry in raw["subassemblies"]}
    root_parts = []
    for part in local:
        owner = owners.get(part["key"])
        (by_owner[owner] if owner else root_parts).append(part)
    root = deepcopy(resolved)
    root["parts"] = root_parts
    if not root_parts:
        root.pop("parts")
    outputs = {Path(path).resolve(): root}
    for entry, source, child in definitions(raw, path):
        if not by_owner[entry["key"]]:
            raise ValueError("A subassembly cannot be empty")
        child["parts"] = by_owner[entry["key"]]
        outputs[source] = child
    return outputs
