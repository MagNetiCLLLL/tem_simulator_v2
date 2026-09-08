"""A reference library CIF or an external CIF is the sole physical structure."""

from __future__ import annotations

SPECIMEN_MODES = frozenset({"atomic", "reference"})
LEGACY_ATOMIC_STRUCTURE_SOURCES = frozenset({"preset", "cif"})


def specimen_mode(sample) -> str:
    mode = str(getattr(sample, "specimen_mode", "reference")).strip().lower()
    if mode not in SPECIMEN_MODES:
        raise ValueError("Sample mode must be 'atomic' or 'reference'; Virtual mode has been retired.")
    return mode


def active_specimen_source(sample) -> str:
    specimen_mode(sample)
    return "cif"


def active_cif_path(sample) -> str:
    if specimen_mode(sample) == "atomic":
        return str(getattr(sample, "cif_path", "")).strip()
    from temsim.specimen.reference_catalog import DEFAULT_REFERENCE_KEY, get_reference_sample
    try:
        return str(get_reference_sample(getattr(sample, "reference_sample_key", DEFAULT_REFERENCE_KEY)).cif_path)
    except ValueError:
        if specimen_is_vacuum(sample):
            return ""
        raise


def selected_reference_preset_key(sample) -> str:
    """Numerical/explicit material template, never a substitute for CIF atoms."""
    if specimen_mode(sample) != "reference":
        return ""
    if specimen_is_vacuum(sample):
        return ""
    from temsim.specimen.reference_catalog import DEFAULT_REFERENCE_KEY, get_reference_sample
    return get_reference_sample(getattr(sample, "reference_sample_key", DEFAULT_REFERENCE_KEY)).template_preset_key


def specimen_structure_available(sample) -> bool:
    return bool(active_cif_path(sample))


def specimen_is_vacuum(sample) -> bool:
    if not bool(getattr(sample, "inserted", True)):
        return True
    return float(getattr(sample, "thickness_nm", 0.0)) <= 0.0


def specimen_interactions_active(sample) -> bool:
    return bool(getattr(sample, "inserted", False) and not specimen_is_vacuum(sample)
                and specimen_structure_available(sample))


def wave_template_preset_key(sample, *, inserted: bool = True) -> str:
    if not inserted or not specimen_structure_available(sample):
        return "vacuum"
    from temsim.specimen.presets import default_specimen_preset_key
    return selected_reference_preset_key(sample) or default_specimen_preset_key()


def migrate_legacy_structure_source(
    sample_data: dict,
    *,
    legacy_source: str = "",
    infer_implicit_atomic_preset: bool = False,
) -> dict:
    """Convert old preset sources without filling an unconfigured CIF source.

    Only states predating schema 71 used an empty atomic CIF path to select
    an implicit preset. Later atomic selections, including old profiles,
    own the imported path even when it is empty.
    """
    migrated = dict(sample_data)
    source = str(legacy_source).strip().lower()
    if source and source not in LEGACY_ATOMIC_STRUCTURE_SOURCES:
        raise ValueError("Legacy sample.atomic_structure_source must be preset or cif")
    path = str(migrated.get("cif_path", "")).strip()
    default_mode = "atomic" if infer_implicit_atomic_preset else "reference"
    old_mode = str(migrated.get("specimen_mode", default_mode)).strip().lower()
    implicit_preset = bool(
        infer_implicit_atomic_preset and not source and old_mode == "atomic" and not path
    )
    if source == "cif" or (not source and old_mode == "atomic" and not implicit_preset):
        migrated["specimen_mode"] = "atomic"
        from temsim.specimen.geometry import quaternion_from_euler_xyz_deg
        rotation = tuple(float(migrated.get(f"specimen_rotation_{axis}_deg", 0.0)) for axis in "xyz")
        migrated.setdefault("specimen_orientation_quaternion_wxyz", quaternion_from_euler_xyz_deg(rotation))
        for axis, value in zip("xyz", rotation):
            migrated.setdefault(f"specimen_rotation_{axis}_deg", value)
    elif source == "preset" or old_mode == "virtual" or implicit_preset:
        from types import SimpleNamespace
        from temsim.specimen.reference_catalog import apply_reference_sample
        from temsim.specimen.geometry import sample_orientation_quaternion, quaternion_multiply, set_sample_orientation
        key = str(migrated.get("specimen_preset_key", "si_110")) or "si_110"
        if key == "vacuum":
            key = "si_110"
            migrated["inserted"] = False
        old = SimpleNamespace(**migrated)
        rotation = sample_orientation_quaternion(old)
        apply_reference_sample(old, key)
        set_sample_orientation(old, quaternion_multiply(rotation, old.specimen_orientation_quaternion_wxyz))
        migrated.update(vars(old))
    migrated["virtual_interactions"] = []
    migrated["virtual_regions"] = []
    return migrated
