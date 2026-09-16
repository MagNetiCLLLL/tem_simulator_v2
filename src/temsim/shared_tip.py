"""One editable tip definition, consumed by every explicitly linked assembly.

Embedded assembly values are portable snapshots. A declared link must resolve;
missing definitions never silently fall back to those snapshots.
"""
from copy import deepcopy
from pathlib import Path
import tomllib

from temsim.optics.electron_gun.tip_assembly import PART_FIELDS, validate_tip_part

LINK = "tip_definition_file"
SHARED_FIELDS = PART_FIELDS | {
    "local_start_z_mm", "local_center_z_mm", "local_end_z_mm",
    "optical_reference_local_z_mm", "outer_diameter_mm", "material_regions",
}


def raw_document(path):
    return tomllib.loads(Path(path).read_text(encoding="utf-8-sig"))


def definition_path(path, part):
    reference = part.get(LINK)
    if reference is None:
        return None
    if part.get("mechanical_only"):
        raise ValueError("A mechanical copy must be detached from the shared emitting-tip definition")
    if part.get("key") != "feg_tip" or not isinstance(reference, str) or not reference.strip():
        raise ValueError("A shared tip link must identify the feg_tip component")
    if Path(reference).is_absolute() or Path(reference).drive:
        raise ValueError("Shared tip references must be relative to the assembly TOML for portable validation and copying")
    result = (Path(path).resolve().parent / reference).resolve()
    if result == Path(path).resolve():
        raise ValueError("A shared tip definition cannot link to itself")
    return result


def dependencies(path, document=None):
    document = raw_document(path) if document is None else document
    from temsim.subassemblies import dependencies as subassembly_dependencies
    result = {source: source.read_bytes() for part in document.get("parts", ())
              if (source := definition_path(path, part)) is not None}
    result.update(subassembly_dependencies(document, path))
    return result


def resolve_document(document, path, *, capture_navigation=False):
    result = deepcopy(document)
    for part in result.get("parts", ()):
        source = definition_path(path, part)
        if source is None:
            continue
        try:
            shared = raw_document(source)
        except OSError as exc:
            raise ValueError(f"Shared FEG tip definition is unavailable: {source}") from exc
        rows = [row for row in shared.get("parts", ()) if row.get("key") == "feg_tip"]
        if len(rows) != 1 or LINK in rows[0]:
            raise ValueError("Shared FEG tip must contain one independent feg_tip definition")
        validate_tip_part(rows[0])
        if "tip_particle_model" not in rows[0]:
            raise ValueError("Shared FEG tip must declare its particle model")
        for field in SHARED_FIELDS:
            if field in rows[0]:
                part[field] = deepcopy(rows[0][field])
            else:
                part.pop(field, None)
    from temsim.subassemblies import resolve_document as resolve_subassemblies
    return resolve_subassemblies(result, path, capture_navigation=capture_navigation)


def materialized_text(text, path):
    """Refresh stored snapshot fields before staging a normal module edit."""
    from temsim import module_manifest
    raw = tomllib.loads(text)
    resolved = resolve_document(raw, path)
    if raw.get("subassemblies"):
        import tomli_w
        return tomli_w.dumps(resolved)
    changes = {("parts", part["key"], field): value
               for original, part in zip(raw.get("parts", ()), resolved.get("parts", ()))
               if LINK in original for field, value in part.items()
               if field in SHARED_FIELDS and original.get(field) != value}
    if any(field in original and field not in part for original, part in
           zip(raw.get("parts", ()), resolved.get("parts", ())) if LINK in original
           for field in SHARED_FIELDS):
        # A removed optional shared field must not survive in a detached copy.
        import tomli_w
        return tomli_w.dumps(resolved)
    return module_manifest.stage_manifest_text(text, changes) if changes else text


def catalog_root_for(path):
    path = Path(path).resolve()
    for parent in path.parents:
        if (parent / "catalog.toml").is_file():
            return parent
        if (parent / "instruments/catalog.toml").is_file():
            return parent / "instruments"
    return None


def catalog_definitions(root):
    root = Path(root).resolve()
    sources = set()
    for path in root.rglob("*.toml"):
        for source in dependencies(path):
            if not source.is_relative_to(root.parent):
                raise ValueError("Shared definitions must stay within this configuration directory")
            sources.add(source)
    return sources


def copy_catalog_tree(source, destination, *, dirs_exist_ok=False):
    """Copy a self-contained catalog plus explicitly referenced definitions."""
    import shutil
    source, destination = Path(source).resolve(), Path(destination).resolve()
    shutil.copytree(source, destination, dirs_exist_ok=dirs_exist_ok)
    copy_catalog_inputs(source, destination, ())


def copy_catalog_inputs(source, destination, relatives):
    """Archive input modules and their declared shared physical definitions."""
    import os
    import shutil
    source, destination = Path(source).resolve(), Path(destination).resolve()
    paths = {(source / relative).resolve() for relative in relatives}
    linked = {dependency for path in paths for dependency in dependencies(path)}
    # A whole copied tree can contain historical modules without any links.
    if not paths:
        linked = catalog_definitions(source)
    for path in paths | linked:
        if not path.is_relative_to(source.parent):
            raise ValueError("Archived input must stay within its configuration directory")
        target = (destination / os.path.relpath(path, source)).resolve()
        if not target.is_relative_to(destination.parent):
            raise ValueError("Copied shared definition must remain in the destination configuration directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
