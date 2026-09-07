"""Select specimen display provenance without calculating or loading atoms.

A completed wave window belongs to its captured sample, never to a later draft.
The geometry renderer may reconstruct a bounded equilibrium-atom view inside
that window; calculation products do not store frozen-phonon atom snapshots.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np


BoundsXY = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class SampleDisplaySource:
    sample: object
    scan_x_um: np.ndarray | None
    scan_y_um: np.ndarray | None
    current_probe_nm: tuple[float, float] | None
    probe_padding_nm: float
    calculation_roi_bounds_nm_override: BoundsXY | None
    provenance_label: str
    provenance_detail: str
    completed_region: bool
    product_kind: str | None = None
    atom_generation_bounds_nm: BoundsXY | None = None
    draft_sample_differs: bool = False


def _bounds(values) -> BoundsXY | None:
    try:
        row = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError):
        return None
    if (len(row) != 4 or not all(math.isfinite(value) for value in row)
            or row[1] <= row[0] or row[3] <= row[2]):
        return None
    return row


def _metrics(product) -> Mapping:
    values = getattr(product, "metrics", None)
    return values if isinstance(values, Mapping) else {}


def _scan_context(stem, sample):
    """Borrow read-only scan views; missing/invalid context remains unknown."""
    scan_x = scan_y = None
    try:
        x = np.asarray(getattr(stem, "scan_x_um", None), dtype=float)
        y = np.asarray(getattr(stem, "scan_y_um", None), dtype=float)
        if (x.ndim >= 1 and x.size and x.shape == y.shape
                and np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            scan_x, scan_y = x.view(), y.view()
            scan_x.setflags(write=False)
            scan_y.setflags(write=False)
    except (TypeError, ValueError, OverflowError):
        pass
    probe = getattr(stem, "probe_state", None)
    centre = None
    padding = 0.0
    if probe is not None:
        try:
            values = tuple(float(value) for value in probe.centroid_nm)
            if len(values) == 2 and all(math.isfinite(value) for value in values):
                centre = values
            candidate = float(probe.radius_99_nm) * float(
                getattr(sample, "wave_probe_padding_factor", 3.0)
            )
            if math.isfinite(candidate) and candidate >= 0.0:
                padding = candidate
        except (AttributeError, TypeError, ValueError, OverflowError):
            pass
    return scan_x, scan_y, centre, padding


def resolve_sample_display_source(
    state,
    result=None,
    *,
    result_is_current: bool | None = None,
) -> SampleDisplaySource:
    """Resolve a captured TEM/STEM window, otherwise an explicitly named preview.

    STEM wins when both products have valid completed-domain metadata. The
    caller supplies current/stale status from its calculation lifecycle; None
    deliberately makes no claim that current optics match the result. Sample
    edits always override a caller's current flag. No file is opened, no cache
    is changed and no ray, potential or atom generation is invoked here.
    """
    draft_sample = state.sample
    captured_state = getattr(result, "state_snapshot", None)
    captured_sample = getattr(captured_state, "sample", None)
    sample = captured_sample if captured_sample is not None else draft_sample
    try:
        draft_differs = not bool(sample == draft_sample)
    except (TypeError, ValueError):
        draft_differs = True

    stem = getattr(result, "stem_scan", None)
    scan_x, scan_y, probe, padding = _scan_context(stem, sample)
    candidates = (
        ("STEM", _metrics(stem)),
        ("TEM", _metrics(getattr(result, "wave_imaging", None))),
    )
    completed = tuple(
        (kind, metrics, bounds)
        for kind, metrics in candidates
        if (bounds := _bounds(metrics.get("specimen_wave_window_bounds_nm"))) is not None
    ) if captured_sample is not None else ()
    if completed:
        kind, metrics, bounds = completed[0]
        if draft_differs or result_is_current is False:
            label = f"Previous {kind} calculation region"
        elif result_is_current is True:
            label = f"Current {kind} calculation region"
        else:
            label = f"Last completed {kind} calculation region"
        detail = (
            "Full outline and calculation window use the same captured sample. "
            "The pink box bounds the specimen's intersection with the completed wave domain; "
            "the wave domain can extend farther into vacuum. Atoms are clipped to the specimen. "
            "Displayed atoms are a bounded equilibrium-structure reconstruction, "
            "not saved frozen-phonon coordinates."
        )
        if draft_differs:
            detail += " The draft sample differs; this view retains the calculated sample."
        elif result_is_current is False:
            detail += " Current calculation settings have changed; the stored region is retained."
        if len(completed) > 1:
            detail += " TEM also has a completed region; this view shows STEM."
        if kind == "TEM":
            # An unrelated raster must not reposition a TEM specimen window.
            scan_x = scan_y = probe = None
            padding = 0.0
        return SampleDisplaySource(
            sample, scan_x, scan_y, probe, padding, bounds, label, detail,
            True, kind, _bounds(metrics.get("specimen_atom_generation_bounds_nm")),
            draft_differs,
        )

    estimated = scan_x is not None
    label = "Estimated scan region" if estimated else "Structural preview"
    if captured_sample is not None:
        label += " | captured sample"
    if draft_differs:
        label += " | draft changed"
    detail = (
        "No completed wave-domain metadata is available. "
        + ("The region is estimated from scan positions and geometric probe padding; "
           if estimated else "The atom window is a structural display preview; ")
        + "it is not evidence of the volume used by a completed TEM/STEM calculation."
    )
    if draft_differs:
        detail += " The view retains the captured sample, not the current draft."
    return SampleDisplaySource(
        sample, scan_x, scan_y, probe, padding, None, label, detail,
        False, draft_sample_differs=draft_differs,
    )


__all__ = ("SampleDisplaySource", "resolve_sample_display_source")
