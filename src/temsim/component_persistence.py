"""Validate component drafts in an isolated catalog before replacing a file."""

from pathlib import Path
from tempfile import TemporaryDirectory

from temsim import module_manifest


def _catalog_snapshot(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*.toml")}


def save_component_changes(root, module_path, changes, configuration):
    """Commit one destination, checking every compatible assembly first.

    The source revision is carried by the file-backed draft. Checking the whole
    catalog again after validation also detects changed interfaces/dependencies.
    The returned snapshot uses the existing caller's runtime-reload rollback API.
    """
    from temsim.column.module_assembly import resolve_module_assembly
    from temsim.manifest_editor import ManifestEditor

    root = Path(root).resolve()
    destination = (root / module_path).resolve()
    if not destination.is_relative_to(root):
        raise ValueError("The destination must be a file in this instrument catalog")
    relative = destination.relative_to(root).as_posix()
    originals = _catalog_snapshot(root)
    if relative not in originals:
        raise ValueError("Choose an existing assembly TOML destination")
    expected = getattr(changes, "expected_source_bytes", None)
    if expected is not None and originals[relative] != expected:
        raise ValueError("The destination file changed outside this editor; reopen it before saving")
    original_text = originals[relative].decode("utf-8-sig")
    staged = module_manifest.stage_manifest_text(original_text, changes)
    with TemporaryDirectory(prefix="temsim-component-validation-") as directory:
        candidate_root = Path(directory)
        for name, content in originals.items():
            candidate = candidate_root / name
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_bytes(content)
        (candidate_root / relative).write_bytes(staged.encode("utf-8"))
        ManifestEditor(candidate_root).validate_catalog()
        resolve_module_assembly(configuration, root=candidate_root)
    if _catalog_snapshot(root) != originals:
        raise ValueError("An assembly file changed during validation; reload before saving")
    module_manifest._atomic_write_text(destination, staged)
    return {relative: original_text}
