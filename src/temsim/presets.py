"""Backward-compatible names for the calculated operating-mode catalog."""

from temsim.operating_modes import (
    apply_operating_mode_pair,
    load_operating_mode_catalog,
    mode_by_key,
)


def _lens_values(mode_key: str) -> dict[str, float]:
    mode = mode_by_key(mode_key, load_operating_mode_catalog())
    return {
        key: float(values["percent"])
        for key, values in mode.devices.items()
        if "percent" in values
    }


TEM_ILLUMINATION = _lens_values("micro_probe")
STEM_ILLUMINATION = _lens_values("nano_probe")
IMAGE_PROJECTOR = _lens_values("imaging")
DIFFRACTION_PROJECTOR = _lens_values("diffraction")

P = {
    "TEM image": {
        "illumination": "TEM",
        "projector": "image",
        "mode_keys": ("micro_probe", "imaging"),
        "lens": {**TEM_ILLUMINATION, **IMAGE_PROJECTOR},
    },
    "TEM diffraction": {
        "illumination": "TEM",
        "projector": "diffraction",
        "mode_keys": ("micro_probe", "diffraction"),
        "lens": {**TEM_ILLUMINATION, **DIFFRACTION_PROJECTOR},
    },
    "STEM image": {
        "illumination": "STEM",
        "projector": "image",
        "mode_keys": ("nano_probe", "imaging"),
        "lens": {**STEM_ILLUMINATION, **IMAGE_PROJECTOR},
    },
    "STEM diffraction": {
        "illumination": "STEM",
        "projector": "diffraction",
        "mode_keys": ("nano_probe", "diffraction"),
        "lens": {**STEM_ILLUMINATION, **DIFFRACTION_PROJECTOR},
    },
}


def apply(state, name):
    condenser_key, projector_key = P[name]["mode_keys"]
    return apply_operating_mode_pair(state, condenser_key, projector_key)
