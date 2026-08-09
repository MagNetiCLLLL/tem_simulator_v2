"""Compatibility facade for the TOML-backed Direct Alignment solver.

The original prototype controlled only I/P1/P2, used one laboratory-frame
axis and wrote failed local solutions into the live state.  Projector user
adjustments now have one authoritative implementation in
``temsim.optics.direct_alignment`` and couple D/I/P1/P2 transactionally.
"""

from __future__ import annotations

import math

import numpy as np

from temsim.optics.direct_alignment import (
    DIFFRACTION_CAMERA_LENGTH,
    IMAGE_MAGNIFICATION,
    PROJECTOR_KEYS as KEYS,
    DirectAlignmentResult as Result,
    apply_direct_alignment,
)
from temsim.physics.first_order import (
    linear_map_properties,
    trace_transverse_transfer,
)
from temsim.physics.recording_stop import determine_tem_stop_z


def _alignment_key(state) -> str:
    return (
        IMAGE_MAGNIFICATION
        if str(state.projector_mode).lower() == "image"
        else DIFFRACTION_CAMERA_LENGTH
    )


def actual_value(state) -> float:
    """Return the full transverse sample-to-recording-plane observable."""

    transfer = trace_transverse_transfer(
        state,
        float(state.sample.z_mm),
        float(determine_tem_stop_z(state)),
    )
    block = (
        transfer.j_img
        if _alignment_key(state) == IMAGE_MAGNIFICATION
        else transfer.j_diff_m_per_rad
    )
    return linear_map_properties(block).isotropic_scale


def slider_to_target(state, value) -> float:
    """Map a legacy 0-100 logarithmic slider onto the catalog request range."""

    fraction = float(np.clip(float(value), 0.0, 100.0)) / 100.0
    if _alignment_key(state) == IMAGE_MAGNIFICATION:
        return 10.0 ** (1.0 + 5.0 * fraction)
    return 10.0 ** (
        math.log10(0.05)
        + fraction * (math.log10(30.0) - math.log10(0.05))
    )


def target_to_slider(state, target) -> float:
    """Invert :func:`slider_to_target` for compatibility callers."""

    target = max(float(target), 1.0e-15)
    if _alignment_key(state) == IMAGE_MAGNIFICATION:
        fraction = (math.log10(target) - 1.0) / 5.0
    else:
        fraction = (
            (math.log10(target) - math.log10(0.05))
            / (math.log10(30.0) - math.log10(0.05))
        )
    return float(np.clip(fraction * 100.0, 0.0, 100.0))


def optimise(
    state,
    target,
    iterations=16,
    optimiser_step_mm=0.5,
) -> Result:
    """Delegate to the authoritative transactional D/I/P1/P2 solve.

    The two legacy tuning arguments are retained only for source compatibility;
    numerical steps and iteration limits are TOML-owned calibration metadata.
    """

    del iterations, optimiser_step_mm
    return apply_direct_alignment(state, _alignment_key(state), float(target))
