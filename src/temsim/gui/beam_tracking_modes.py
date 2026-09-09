"""Qt-free display classification of recorded electron interaction histories.

Focusing and specimen exits do not change a recorded interaction category.
The current finite transport stores elastic state in ``interaction_kind`` and
its independent energy-loss channel in an exact generated branch name. This
adapter combines those facts for display only; no trajectory, weight, detector
signal, or physics-cache identity is modified.
"""

from __future__ import annotations

import math


# RGB values are display colours, not material or scattering parameters.
_STYLES = {
    "incident": ("Incident (pre-sample)", (56, 189, 247), "o"),
    "vacuum": ("Vacuum continuation", (148, 163, 184), "o"),
    "optical_reference": ("Optical reference", (148, 189, 214), "o"),
    "real_sample_reference": ("Real sample: reference ray only", (148, 163, 184), "o"),
    "real_zero_loss": ("Zero loss", (184, 194, 209), "o"),
    "real_plasmon": ("Plasmon / low loss", (41, 209, 245), "d"),
    "real_ionisation": ("Core ionisation", (250, 89, 61), "d"),
    "real_other_inelastic": ("Other inelastic", (245, 186, 31), "d"),
    "real_plural_inelastic": ("Plural inelastic", (199, 92, 245), "d"),
    "virtual_interactions_disabled": ("Virtual interactions disabled", (148, 163, 184), "o"),
    "transmitted": ("Transmitted / direct", (74, 222, 128), "o"),
    "diffraction_spots": ("Diffraction spots", (168, 140, 250), "t"),
    "diffuse_ring": ("Diffuse ring", (245, 158, 10), "t"),
    "gaussian_diffuse": ("Gaussian diffuse", (250, 204, 20), "t"),
    "arbitrary_angular": ("Arbitrary angular", (46, 212, 191), "t"),
    "user_screened_power_law": ("User screened power law", (245, 115, 181), "t"),
    "physical_rutherford": ("Rutherford approximation", (250, 79, 97), "t"),
    "sample_region_primary": ("Primary / zero loss", (46, 224, 184), "o"),
    "sample_region_elastic": ("Elastic scattering", (255, 82, 122), "t"),
    "elastic+real_plasmon": ("Elastic + plasmon / low loss", (251, 146, 60), "star"),
    "elastic+real_ionisation": ("Elastic + core ionisation", (232, 121, 249), "star"),
    "elastic+real_other_inelastic": ("Elastic + other inelastic", (245, 158, 11), "star"),
    "elastic+real_plural_inelastic": ("Elastic + plural inelastic", (192, 132, 252), "star"),
    "unknown": ("Unknown interaction", (148, 163, 184), "x"),
}
_ALIASES = {"diffraction_spot": "diffraction_spots", "isotropic_ring": "diffuse_ring"}
_INELASTIC_CHANNELS = frozenset({
    "real_plasmon", "real_ionisation", "real_other_inelastic", "real_plural_inelastic",
})
_ZERO_LOSS_CHANNELS = frozenset({"000", "zero_loss"})


def _generated_kind(name: str) -> str | None:
    """Recognise only the exact names emitted by finite downstream transport."""
    prefix, separator, channel = name.partition(":")
    if not separator or prefix not in {"specimen_primary", "specimen_elastic"}:
        return None
    elastic = prefix == "specimen_elastic"
    if channel in _ZERO_LOSS_CHANNELS:
        return "sample_region_elastic" if elastic else "sample_region_primary"
    if channel in _INELASTIC_CHANNELS:
        return "elastic+" + channel if elastic else channel
    return None


def _branch_rgb(branch, fallback):
    """Retain configured ordinary branch colours when they are well formed."""
    raw = getattr(branch, "colour", None)
    if raw is None:
        return fallback
    try:
        values = tuple(float(value) for value in raw)[:3]
        if len(values) != 3 or any(not math.isfinite(value) or value < 0 for value in values):
            return fallback
        scale = 255 if max(values) <= 1 else 1
        if any(value * scale > 255 for value in values):
            return fallback
        return tuple(round(value * scale) for value in values)
    except (TypeError, ValueError, OverflowError):
        return fallback


def branch_interaction_style(branch) -> tuple[str, str, tuple[int, int, int], str]:
    """Return ``(key, label, RGB_255, pyqtgraph_symbol)`` for recorded history.

    Explicit recognised metadata wins over contradictory names. The generic
    finite elastic tag is enriched only by its supported generated channel
    suffix. Missing/unknown legacy tags permit exact known-name recovery, but
    arbitrary labels, final ray angles and energy offsets are never interpreted
    as proof of an interaction. In particular, no channeling classification is
    inferred from focusing or a crystal orientation.
    """
    raw_kind = str(getattr(branch, "interaction_kind", "") or "").strip().lower()
    kind = _ALIASES.get(raw_kind, raw_kind)
    name = str(getattr(branch, "name", "") or "")
    generated = _generated_kind(name)
    if kind == "sample_region_elastic" and name.startswith("specimen_elastic:"):
        kind = generated or kind
    elif kind in {"", "unknown"}:
        kind = generated or (name if name in _INELASTIC_CHANNELS else None) or {
            "incident": "incident", "000": "transmitted",
            "+g": "diffraction_spots", "-g": "diffraction_spots",
            "virtual_+g": "diffraction_spots", "virtual_-g": "diffraction_spots",
        }.get(name, "unknown")
    elif kind not in _STYLES:
        kind = "unknown"
    label, rgb, symbol = _STYLES[kind]
    # Combined classes have a dedicated palette: the solver's generic elastic
    # colour cannot distinguish their recorded inelastic contribution.
    if kind != "unknown" and not kind.startswith("elastic+"):
        rgb = _branch_rgb(branch, rgb)
    return kind, label, rgb, symbol
