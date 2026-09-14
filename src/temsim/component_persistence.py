"""Validate component drafts in an isolated catalog before replacing a file."""

from pathlib import Path
from tempfile import TemporaryDirectory
import os
import tomllib

from temsim import module_manifest


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
            raise ValueError("A shared tip definition changed outside this editor; reopen it before saving")
    original_text = originals[relative].decode("utf-8-sig")
    original_document = tomllib.loads(original_text)
    staged = {relative: module_manifest.stage_manifest_text(materialized_text(original_text, destination), changes)}
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
