"""Resolve the single structure source owned by each specimen mode.

The public UI has one source decision only:

* ``atomic`` (Real sample) uses an imported CIF/MCIF.
* ``virtual`` uses one simulator-owned TOML reference specimen.

Both path/key values may remain in a state so switching modes does not discard
the user's work, but only the value owned by the active mode is ever consumed
by a physical calculation.
"""

from __future__ import annotations


SPECIMEN_MODES = frozenset({"atomic", "virtual"})
LEGACY_ATOMIC_STRUCTURE_SOURCES = frozenset({"preset", "cif"})


def specimen_mode(sample) -> str:
    """Return the validated canonical specimen mode without mutating state."""

    mode = str(getattr(sample, "specimen_mode", "virtual")).strip().lower()
    if mode not in SPECIMEN_MODES:
        raise ValueError("Sample mode must be 'atomic' or 'virtual'.")
    return mode


def active_specimen_source(sample) -> str:
    """Return ``cif`` for Real mode or ``preset`` for Virtual mode."""

    return "cif" if specimen_mode(sample) == "atomic" else "preset"


def active_cif_path(sample) -> str:
    """Return the imported structure only when Real mode owns it."""

    if specimen_mode(sample) != "atomic":
        return ""
    return str(getattr(sample, "cif_path", "")).strip()


def selected_reference_preset_key(sample) -> str:
    """Return the active Virtual reference key, resolving its TOML default."""

    if specimen_mode(sample) != "virtual":
        return ""
    from temsim.specimen.presets import default_specimen_preset_key

    return (
        str(getattr(sample, "specimen_preset_key", "")).strip()
        or default_specimen_preset_key()
    )


def specimen_structure_available(sample) -> bool:
    """Return whether the active mode supplies a calculation structure."""

    if specimen_mode(sample) == "atomic":
        return bool(active_cif_path(sample))
    return bool(selected_reference_preset_key(sample))


def wave_template_preset_key(sample, *, inserted: bool = True) -> str:
    """Return the TOML grid/potential template used by a wave calculation.

    Virtual mode uses the selected reference specimen. Real CIF mode still
    needs numerical grid defaults expected by the current wave API; the
    imported CIF replaces the template atoms and material identity.
    """

    if not inserted or not specimen_structure_available(sample):
        return "vacuum"
    reference_key = selected_reference_preset_key(sample)
    if reference_key:
        return reference_key
    from temsim.specimen.presets import default_specimen_preset_key

    return default_specimen_preset_key()


def migrate_legacy_structure_source(
    sample_data: dict,
    *,
    legacy_source: str = "",
) -> dict:
    """Map the retired Real preset/CIF selector onto the owning mode.

    CIF historically had calculation precedence, including before the
    explicit selector was introduced. Otherwise an old Real/TOML selection
    becomes the new Virtual reference mode. Existing Virtual states remain
    Virtual when no retired selector was stored.
    """

    migrated = dict(sample_data)
    source = str(legacy_source).strip().lower()
    if source and source not in LEGACY_ATOMIC_STRUCTURE_SOURCES:
        raise ValueError(
            "Legacy sample.atomic_structure_source must be preset or cif"
        )
    cif_path = str(migrated.get("cif_path", "")).strip()
    old_mode = str(migrated.get("specimen_mode", "atomic")).strip().lower()
    if source == "cif" or (not source and old_mode == "atomic" and cif_path):
        migrated["specimen_mode"] = "atomic"
    elif source == "preset" or (not source and old_mode == "atomic"):
        migrated["specimen_mode"] = "virtual"
    return migrated
