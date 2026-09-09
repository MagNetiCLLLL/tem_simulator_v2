"""Mechanical component drafts, independent of historical module categories.

Coordinates are module-local millimetres. Copying geometry never creates an
optical control or a magnetic circuit. Source documents are never mutated.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import math
from numbers import Real
from pathlib import Path
import re
import tomllib


CUSTOM_COMPONENT_ROLES = frozenset({"custom_mechanical", "custom_mechanical_copy"})


@dataclass(eq=False)
class PartChangeSet(Mapping):
    """Field-compatible updates plus structural changes to the parts array."""

    fields: dict = field(default_factory=dict)
    added_parts: tuple = ()
    removed_keys: tuple = ()
    expected_source_bytes: bytes | None = None

    def __getitem__(self, key):
        return self.fields[key]

    def __iter__(self):
        return iter(self.fields)

    def __len__(self):
        return len(self.fields)

    def __bool__(self):
        return bool(self.fields or self.added_parts or self.removed_keys)

    def __eq__(self, other):
        if isinstance(other, PartChangeSet):
            return (self.fields, self.added_parts, self.removed_keys) == (other.fields, other.added_parts, other.removed_keys)
        if isinstance(other, Mapping):
            return not (self.added_parts or self.removed_keys) and self.fields == dict(other)
        return NotImplemented


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    try:
        value = float(value)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{label} must be {'positive and ' if positive else ''}finite")
    return value


def _key(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", value):
        raise ValueError("A component key must be a non-empty identifier (letters, digits, _, . or -)")
    return value


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return deepcopy(value)


def source_document(source):
    """Snapshot a draft, parsed module, or TOML path before copying anything."""
    if hasattr(source, "document"):
        source = source.document
    if isinstance(source, Mapping):
        return _plain(source)
    with Path(source).open("rb") as stream:
        return tomllib.load(stream)


def validate_component_graph(document):
    rows = document.get("parts", ())
    keys = [row.get("key") for row in rows]
    if any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("Every part must have a unique non-empty key")
    by_key = {row["key"]: row for row in rows}
    for key, row in by_key.items():
        parent = row.get("parent_key")
        if parent is not None and not isinstance(parent, str):
            raise ValueError(f"Parent of {key} must be a component key")
        if parent and parent not in by_key:
            raise ValueError(f"Unknown parent {parent!r} for {key}")
    for key in by_key:
        seen = set()
        current = key
        while current:
            if current in seen:
                raise ValueError("Component ownership must not contain a cycle")
            seen.add(current)
            current = by_key[current].get("parent_key")
    return by_key


def component_subtree(document, key, *, include_children=True):
    if not isinstance(include_children, bool):
        raise ValueError("include_children must be a boolean")
    by_key = validate_component_graph(document)
    if key not in by_key:
        raise ValueError(f"Unknown component: {key}")
    selected = {key}
    if include_children:
        while True:
            expanded = selected | {k for k, row in by_key.items() if row.get("parent_key") in selected}
            if expanded == selected:
                break
            selected = expanded
    return tuple(row["key"] for row in document["parts"] if row["key"] in selected)


def translated_component(part, delta_z_mm):
    """Translate absolute module coordinates, preserving relative CAD/profile data."""
    result = _plain(part)
    delta = _number(delta_z_mm, "Axial displacement")

    def shift(value, label):
        if isinstance(value, (list, tuple)):
            return [shift(item, label) for item in value]
        return _number(_number(value, label) + delta, label)

    for name, value in tuple(result.items()):
        if name in {"local_start_z_mm", "local_center_z_mm", "local_end_z_mm", "material_intervals_mm"} or name.endswith("_local_z_mm"):
            result[name] = shift(value, name)
    return result


def make_component(*, key, name=None, shape="tube", center_z_mm=0.0,
                   length_mm=10.0, inner_diameter_mm=10.0, outer_diameter_mm=20.0,
                   width_mm=20.0, height_mm=20.0, parent_key=None,
                   material_class="non_magnetic_metal"):
    """Create an explicit CAD body excluded from vacuum and field solvers."""
    key = _key(key)
    center = _number(center_z_mm, "Centre")
    length = _number(length_mm, "Length", positive=True)
    if shape not in {"tube", "box", "elliptic_cylinder"}:
        raise ValueError("Component shape must be tube, box or elliptic_cylinder")
    if shape == "tube":
        inner = _number(inner_diameter_mm, "Inner diameter")
        outer = _number(outer_diameter_mm, "Outer diameter", positive=True)
        if not 0 <= inner < outer:
            raise ValueError("Tube dimensions require 0 <= inner diameter < outer diameter")
        width = height = outer
    else:
        width = _number(width_mm, "Width", positive=True)
        height = _number(height_mm, "Height", positive=True)
    if not isinstance(name, (str, type(None))) or (name is not None and not name.strip()):
        raise ValueError("A component name must not be blank")
    if not isinstance(material_class, str) or not material_class.strip():
        raise ValueError("A component material class must not be blank")
    if parent_key is not None and not isinstance(parent_key, str):
        raise ValueError("Parent must be a component key")
    if parent_key:
        _key(parent_key)
    result = {"key": key, "name": name or key.replace("_", " ").title(), "branch": "main", "order": 1,
            "parent_key": parent_key or "", "mechanical_only": True,
            "mechanical_part_role": "custom_mechanical", "mechanical_profile": "custom_mechanical",
            "axial_vacuum_context_only": True, "material_class": material_class,
            "local_start_z_mm": center - length / 2, "local_center_z_mm": center,
            "local_end_z_mm": center + length / 2, "length_mm": length,
            "vacuum_inner_diameter_mm": 1.0,
            "model_3d": {"schema_version": 1,
                         "base": {"kind": "box" if shape == "box" else "elliptic_cylinder",
                                  "width_mm": width, "height_mm": height, "length_mm": length},
                         "transform": {"scale_xy": [1.0, 1.0], "offset_mm": [0.0, 0.0, 0.0],
                                       "rotation_deg": [0.0, 0.0, 0.0]}, "features": []}}
    if shape == "tube":
        # A material annulus has a through bore at every axial length. It must
        # not become a blind hole when its envelope grows beyond a cut tool.
        result.update(mechanical_profile="vacuum_liner", mechanical_inner_diameter_mm=inner,
                      mechanical_outer_diameter_mm=outer)
        result["model_3d"]["base"] = {"kind": "existing"}
    return result


def added_component_document(document, part):
    candidate = _plain(document)
    by_key = validate_component_graph(candidate)
    if not isinstance(part, Mapping):
        raise ValueError("A component must be a table")
    row = _plain(part)
    key = _key(row.get("key"))
    if key in by_key:
        raise ValueError(f"Duplicate component key: {key}")
    if row.get("mechanical_part_role") not in CUSTOM_COMPONENT_ROLES or row.get("mechanical_only") is not True:
        raise ValueError("New components must declare an independent custom mechanical role")
    row["order"] = max((p["order"] for p in candidate["parts"]), default=0) + 1
    candidate["parts"].append(row)
    validate_component_graph(candidate)
    return candidate, key


def placed_component_document(document, key, center_z_mm, *, include_children=True):
    candidate = _plain(document)
    selected = component_subtree(candidate, key, include_children=include_children)
    center = _number(center_z_mm, "Centre")
    original = next(row for row in candidate["parts"] if row["key"] == key)
    delta = center - _number(original["local_center_z_mm"], "Original centre")
    candidate["parts"] = [translated_component(row, delta) if row["key"] in selected else row
                          for row in candidate["parts"]]
    return candidate, selected


_SINGLE_REFERENCES = frozenset({"nested_lens_parent_key", "shared_housing_key", "field_source_key",
                              "upstream_boundary_aperture_key"})
_MULTI_REFERENCES = frozenset({"magnetic_lens_keys", "contained_recording_plane_keys"})


def copied_component_document(document, source, key, new_key, center_z_mm, *, parent_key=None,
                              include_children=True, name=None):
    """Copy a mechanical subtree, never binding to a same-named external owner."""
    source = source_document(source)
    source_by_key = validate_component_graph(source)
    selected = component_subtree(source, key, include_children=include_children)
    from temsim.part_model_3d import part_model_from_document

    def has_solid(part_key):
        from collections import Counter
        import numpy as np

        for mesh in part_model_from_document(source, part_key, include_children=False).meshes:
            edges = Counter(tuple(sorted((int(a), int(b)))) for face in mesh.faces
                            for a, b in zip(face, np.roll(face, -1)))
            if edges and set(edges.values()) == {2}:
                return True
        return False

    if any(source_by_key[part_key].get("branch_path_only", False) for part_key in selected):
        raise ValueError("A branch-path component has no supported axial placement; its curved-path geometry cannot be copied into an axial solid")
    if not any(has_solid(part_key) for part_key in selected):
        raise ValueError("The selected component has no defined 3D solid to copy")
    target = _plain(document)
    target_by_key = validate_component_graph(target)
    new_key = _key(new_key)
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise ValueError("A component name must not be blank")
    if parent_key is not None and not isinstance(parent_key, str):
        raise ValueError("Parent must be a component key")
    if parent_key and parent_key not in target_by_key:
        raise ValueError(f"Unknown parent: {parent_key}")
    names = {key: new_key}
    for old in selected:
        if old != key:
            suffix = old[len(key):] if old.startswith(key + "_") else "__" + old
            names[old] = _key(new_key + suffix)
    collisions = set(names.values()) & set(target_by_key)
    if collisions or len(set(names.values())) != len(names):
        raise ValueError(f"Duplicate copied component keys: {sorted(collisions or names.values())}")
    delta = _number(center_z_mm, "Centre") - _number(source_by_key[key]["local_center_z_mm"], "Original centre")
    next_order = max((row["order"] for row in target["parts"]), default=0) + 1
    from temsim.magnetic_geometry import objective_layer_intervals_mm

    for index, old in enumerate(selected):
        original = source_by_key[old]
        row = _plain(original)
        source_parent = source_by_key.get(original.get("parent_key"), {})
        if (row.get("mechanical_part_role") not in CUSTOM_COMPONENT_ROLES
                and row.get("mechanical_profile") in {"magnetic_excitation_coil", "magnetic_lens_yoke"}):
            intervals = objective_layer_intervals_mm(source_parent, source_parent.get("local_start_z_mm", 0),
                                                   row["mechanical_profile"])
            if intervals:
                row["material_intervals_mm"] = [list(pair) for pair in intervals]
        for field_name in _SINGLE_REFERENCES | _MULTI_REFERENCES:
            if field_name not in row:
                continue
            refs = row[field_name] if field_name in _MULTI_REFERENCES else [row[field_name]]
            missing = [ref for ref in refs if ref and ref not in names]
            # A root's former optical current owner is provenance, not a new
            # coupling to an existing target channel.
            if field_name == "field_source_key" and old == key and missing:
                row.pop(field_name)
                continue
            if missing:
                raise ValueError(f"Cannot copy {old}: include or detach external {field_name} dependencies {missing}")
            mapped = [names[ref] for ref in refs if ref]
            row[field_name] = mapped if field_name in _MULTI_REFERENCES else (mapped[0] if mapped else "")
        row = translated_component(row, delta)
        row.update(key=names[old], order=next_order + index, mechanical_only=True,
                   mechanical_part_role="custom_mechanical_copy", axial_vacuum_context_only=True,
                   geometry_template_key=original.get("geometry_template_key", old),
                   parent_key=(parent_key or "") if old == key else names[original["parent_key"]])
        if old == key:
            row["name"] = name or str(original.get("name", key)) + " (copy)"
        for field_name in tuple(row):
            if field_name.startswith("magnetic_circuit_") or field_name == "field_source_key":
                row.pop(field_name)
        row["copy_provenance"] = {"source_module_key": str(source.get("module", {}).get("key", "")),
                                  "source_part_key": old,
                                  "physics_note": "Independent mechanical copy; no new optical control or magnetic circuit is created."}
        target["parts"].append(row)
    validate_component_graph(target)
    return target, new_key
