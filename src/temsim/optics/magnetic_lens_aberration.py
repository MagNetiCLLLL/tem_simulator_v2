"""Conventional positive spherical aberration for round magnetic lenses."""

from __future__ import annotations

import math

from temsim.optics.lens_focal_length import focal_length_mm


# The objective's reference values are f=2.8 mm and Cs=1.2 mm.  Use the same
# positive ratio for magnetic lenses that do not carry a measured/explicit Cs.
DEFAULT_CS_TO_FOCAL_LENGTH_RATIO = 3.0 / 7.0


def spherical_aberration_mm(lens, voltage_kv):
    """Return a lens' signed Cs; ``None`` selects a positive physical estimate.

    An explicit zero remains a useful opt-out for paraxial experiments and an
    explicit signed value remains authoritative.  This keeps saved instruments
    backwards compatible while making newly constructed real magnetic lenses
    aberrating by default.
    """

    explicit = getattr(lens, "cs_mm", None)
    if explicit is not None:
        value = float(explicit)
        if not math.isfinite(value):
            raise ValueError(
                f"{getattr(lens, 'name', 'Round lens')} Cs must be finite."
            )
        return value
    try:
        focal = float(focal_length_mm(lens, voltage_kv))
    except Exception:
        return None
    if not math.isfinite(focal) or focal <= 0.0:
        return None
    return DEFAULT_CS_TO_FOCAL_LENGTH_RATIO * focal

