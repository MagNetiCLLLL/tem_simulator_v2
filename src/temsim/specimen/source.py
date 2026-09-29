"""Vacuum or a user-imported CIF is the only active specimen source."""
from __future__ import annotations
from temsim import input_io

SPECIMEN_MODES = frozenset({"vacuum", "atomic"})


def specimen_mode(sample) -> str:
    mode = str(getattr(sample, "specimen_mode", "vacuum")).strip().lower()
    if mode not in SPECIMEN_MODES:
        raise ValueError("Sample mode must be 'vacuum' or 'atomic'. Import a CIF to use a material sample.")
    return mode


def active_specimen_source(sample) -> str:
    return "vacuum" if specimen_mode(sample) == "vacuum" else "cif"


@input_io.using_state_inputs
def active_cif_path(sample) -> str:
    if specimen_mode(sample) == "vacuum":
        return ""
    return str(getattr(sample, "cif_path", "")).strip()


def specimen_structure_available(sample) -> bool:
    return bool(active_cif_path(sample))


def specimen_is_vacuum(sample) -> bool:
    return (specimen_mode(sample) == "vacuum"
            or not bool(getattr(sample, "inserted", False))
            or float(getattr(sample, "thickness_nm", 0.0)) <= 0.0)


def specimen_interactions_active(sample) -> bool:
    return not specimen_is_vacuum(sample) and specimen_structure_available(sample)


def validate_sample_source(sample) -> None:
    """An inserted material request cannot silently calculate an empty sample."""
    if not specimen_is_vacuum(sample) and not specimen_structure_available(sample):
        raise ValueError("Import a CIF/MCIF file before calculating the sample, or select Vacuum sample.")


def wave_template_preset_key(sample, *, inserted: bool = True) -> str:
    if not inserted or specimen_is_vacuum(sample) or not specimen_structure_available(sample):
        return "vacuum"
    from temsim.specimen.presets import default_specimen_preset_key
    # Numerical grid defaults only; atoms and composition always come from CIF.
    return default_specimen_preset_key()
