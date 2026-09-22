"""Functional presentation of legacy equipment labels, without editing evidence.

These aliases are a display boundary only. Hardware routing, file paths and
original acquisition metadata retain their exact identifiers.
"""
from __future__ import annotations

import re


_ALIASES = {
    "BM-Ceta": "Column pixelated camera",
    "EF-Ceta": "Energy-filter pixelated camera",
    "Ceta-S": "Pixelated camera",
    "Ceta": "Pixelated camera",
    "Flucam": "Screen camera",
    "SmartCam": "Screen camera",
    "S-CORR": "Hexapole probe corrector",
    "S_CORR": "Hexapole probe corrector",
    "DCORPRIME": "Hexapole probe corrector",
    "DCOR": "Hexapole probe corrector",
    "ASCOR": "Hexapole probe corrector",
    "NanoPulser": "Electrostatic beam blanker",
    "Zebra": "EELS camera",
    "Ultra-X": "EDS detector array",
    "UltraX": "EDS detector array",
    "Super-X": "EDS detector array",
    "SuperX": "EDS detector array",
    "Iliad Ultra": "Transmission electron microscope",
    "Iliad": "Post-column energy filter",
    "Spectra Ultra": "Transmission electron microscope",
    "Themis": "Transmission electron microscope",
    "Titan": "Transmission electron microscope",
    "Talos": "Transmission electron microscope",
    "Tecnai": "Transmission electron microscope",
    "S-TWIN": "Symmetric objective",
    "MerlinEM": "Pixelated electron detector",
    "Merlin": "Pixelated electron detector",
}
_LOOKUP = {key.casefold(): value for key, value in _ALIASES.items()}
_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    + "|".join(re.escape(key) for key in sorted(_ALIASES, key=len, reverse=True))
    + r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def scientific_equipment_text(text: object) -> str:
    """Return a display string; never use it as a device or archive identifier."""
    return _PATTERN.sub(lambda match: _LOOKUP[match.group().casefold()], str(text))
