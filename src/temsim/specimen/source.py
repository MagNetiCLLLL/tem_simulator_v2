"""A reference library CIF or an external CIF is the sole physical structure."""

from __future__ import annotations
from temsim import input_io

SPECIMEN_MODES = frozenset({"atomic", "reference"})


def specimen_mode(sample) -> str:
    mode = str(getattr(sample, "specimen_mode", "reference")).strip().lower()
    if mode not in SPECIMEN_MODES:
        raise ValueError("Sample mode must be 'atomic' or 'reference'; Virtual mode has been retired.")
    return mode


def active_specimen_source(sample) -> str:
    specimen_mode(sample)
    return "cif"


@input_io.using_state_inputs
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


@input_io.using_state_inputs
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
