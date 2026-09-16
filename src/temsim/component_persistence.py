"""Validate component drafts in an isolated catalog before replacing a file."""

from pathlib import Path
from tempfile import TemporaryDirectory
import os
import tomllib

from temsim import module_manifest


def _storage_text(original, document):
    """Preserve part-file comments while the expanded catalog is validated later."""
    from temsim.component_operations import PartChangeSet
    before = tomllib.loads(original)
    old = {part["key"]: part for part in before.get("parts", ())}
    new = {part["key"]: part for part in document.get("parts", ())}
    fields = {("parts", key, field): value for key, part in new.items() if key in old
              for field, value in part.items() if old[key].get(field) != value}
    for key in old.keys() & new.keys():
        for field in old[key].keys() - new[key].keys():
            if field in {"model_3d", "material_regions"}:
                fields[("parts", key, field)] = None if field == "model_3d" else {}
    old_placements = {entry["key"]: entry["placement"] for entry in before.get("subassemblies", ())}
    placements = {entry["key"]: entry["placement"] for entry in document.get("subassemblies", ())
                  if old_placements.get(entry["key"]) != entry["placement"]}
    changes = PartChangeSet(fields, tuple(part for key, part in new.items() if key not in old),
                            tuple(key for key in old if key not in new), placement_updates=placements)
    # Header/port edits still use the same targeted section writer.
    for section in ("module", "geometry"):
        for field, value in document.get(section, {}).items():
            if before.get(section, {}).get(field) != value:
                changes.fields[(section, field)] = value
    for port, values in document.get("ports", {}).items():
        for field, value in values.items():
            if before.get("ports", {}).get(port, {}).get(field) != value:
                changes.fields[("ports", port, field)] = value
    text = module_manifest.stage_manifest_text(original, changes, validate=False)
    parsed = tomllib.loads(text)
    # TOML may retain an explicit empty parts array after removing root additions.
    if not parsed.get("parts"):
        parsed.pop("parts", None)
    if parsed != document:
        raise ValueError("Storage changes cannot be represented without losing unrelated fields")
    return text


def _catalog_snapshot(root):
    from temsim.shared_tip import catalog_definitions
    paths = set(root.rglob("*.toml")) | catalog_definitions(root)
    return {Path(os.path.relpath(path, root)).as_posix(): path.read_bytes() for path in paths}


def save_component_changes(root, module_path, changes, configuration):
    """Commit one destination, checking every compatible assembly first.

    The source revision is carried by the file-backed draft. Checking the whole
    catalog again after validation also detects changed interfaces/dependencies.
    The returned snapshot uses the existing caller's runtime-reload rollback API.
    """
    from temsim.column.module_assembly import resolve_module_assembly
    from temsim.manifest_editor import ManifestEditor
    from temsim.shared_tip import catalog_definitions, materialized_text, definition_path, SHARED_FIELDS
    from temsim.optics.electron_gun.tip_assembly import complete_tip_updates

    root = Path(root).resolve()
    destination = (root / module_path).resolve()
    if not destination.is_relative_to(root) and destination not in catalog_definitions(root):
        raise ValueError("The destination must be a file in this instrument catalog")
    relative = Path(os.path.relpath(destination, root)).as_posix()
    originals = _catalog_snapshot(root)
    if relative not in originals:
        raise ValueError("Choose an existing assembly TOML destination")
    expected = getattr(changes, "expected_source_bytes", None)
    if expected is not None and originals[relative] != expected:
        raise ValueError("The destination file changed outside this editor; reopen it before saving")
    for path, expected in (getattr(changes, "expected_dependency_bytes", None) or {}).items():
        key = Path(os.path.relpath(path, root)).as_posix()
        if originals.get(key) != expected:
            raise ValueError("An assembly dependency changed outside this editor; reopen it before saving")
    original_text = originals[relative].decode("utf-8-sig")
    original_document = tomllib.loads(original_text)
    staged = {relative: module_manifest.stage_manifest_text(materialized_text(original_text, destination), changes)}
    if original_document.get("subassemblies"):
        from temsim.subassemblies import reflow, storage_documents
        candidate = tomllib.loads(staged[relative])
        if not getattr(changes, "subassemblies_resolved", False):
            before = tomllib.loads(materialized_text(original_text, destination))
            candidate = reflow(before, candidate, destination)
        module_manifest.validate_document(candidate)
        documents = storage_documents(original_document, candidate, destination)
        staged = {Path(os.path.relpath(path, root)).as_posix(): _storage_text(
                      originals[Path(os.path.relpath(path, root)).as_posix()].decode("utf-8-sig"), document)
                  for path, document in documents.items()
                  if document != tomllib.loads(originals[Path(os.path.relpath(path, root)).as_posix()].decode("utf-8-sig"))}
    for part in original_document.get("parts", ()):
        shared = definition_path(destination, part)
        shared_updates = {path: value for path, value in changes.items()
                          if len(path) == 3 and path[:2] == ("parts", part["key"]) and path[2] in SHARED_FIELDS}
        if shared is None or not shared_updates:
            continue
        shared_key = Path(os.path.relpath(shared, root)).as_posix()
        text = originals[shared_key].decode("utf-8-sig")
        staged[shared_key] = module_manifest.stage_manifest_text(
            text, complete_tip_updates(tomllib.loads(text), shared_updates))
    with TemporaryDirectory(prefix="temsim-component-validation-") as directory:
        candidate_root = Path(directory) / "instruments"
        for name, content in originals.items():
            candidate = candidate_root / name
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_bytes(content)
        for name, text in staged.items():
            (candidate_root / name).write_bytes(text.encode("utf-8"))
        ManifestEditor(candidate_root).validate_catalog()
        if configuration is not None:
            resolve_module_assembly(configuration, root=candidate_root)
    if _catalog_snapshot(root) != originals:
        raise ValueError("An assembly file changed during validation; reload before saving")
    replaced = []
    try:
        for name, text in staged.items():
            module_manifest._atomic_write_text(root / name, text)
            replaced.append(name)
    except Exception:
        for name in replaced:
            module_manifest._atomic_write_text(root / name, originals[name].decode("utf-8-sig"))
        raise
    return {name: originals[name].decode("utf-8-sig") for name in staged}
