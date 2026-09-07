"""Small, read-only performance summaries; never inspect numerical array data."""
from __future__ import annotations

from collections.abc import Mapping
import math


_STAGE_LABELS = {
    "Preparing state and physical layout": "State/layout",
    "Tracing the electron column": "Column rays",
    "Calculating the TEM wave image": "TEM wave",
    "Projecting the cached Objective wave to the recording plane": "TEM projection",
    "Transporting electrons through the specimen": "Specimen transport",
    "Calculating the EDS point spectrum": "EDS (including required transport)",
    "Tracing the energy filter": "Energy filter",
    "Solving scan geometry": "Scan geometry",
    "Building scan-ray playback": "Scan playback",
    "Propagating specimen-exit electrons downstream": "Downstream transport",
    "Updating STEM dose readout": "STEM readout",
    "Recollecting cached STEM diffraction": "STEM cached diffraction readout",
    "Calculating the STEM detector frame": "STEM frame",
    "Finalising optical diagnostics": "Diagnostics",
}


def _seconds(value) -> str | None:
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return f"{value:.3f} s" if math.isfinite(value) and value >= 0.0 else None


def calculation_performance_lines(result) -> tuple[str, ...]:
    """Only report this request's work, not timings inherited from old products.

    Stage times are disjoint wall intervals. Potential/GPU diagnostics are
    nested within their wave stage, not extra intervals to add to the total.
    """
    if bool(getattr(result, "cache_hit", False)):
        return ("Performance | Complete result reused; no new physics stages.",)
    performance = getattr(result, "performance", {}) or {}
    stages = performance.get("stages", ())
    lines = []
    stage_names = set()
    for record in stages:
        if not isinstance(record, Mapping):
            continue
        name = str(record.get("stage", ""))
        elapsed = _seconds(record.get("seconds"))
        if name and elapsed is not None:
            stage_names.add(name)
            lines.append(f"Timing | {_STAGE_LABELS.get(name, name)}: {elapsed}")
    calculated = set(getattr(result, "calculated_products", ()))
    reused = set(getattr(result, "reused_products", ()))
    for label, product, field, stage in (
        ("TEM", "wave", "wave_imaging", "Calculating the TEM wave image"),
        ("STEM", "stem", "stem_scan", "Calculating the STEM detector frame"),
    ):
        if product not in calculated or product in reused or stage not in stage_names:
            continue
        output = getattr(result, field, None)
        metrics = getattr(output, "metrics", None) or {}
        if label == "STEM" and ("fourdstem_cube" in reused or metrics.get("interactive_cube_reused")):
            continue
        backend = metrics.get("wave_compute_backend")
        if backend is None:
            continue
        detail = f"{label} compute | {backend}"
        cache_hit = metrics.get("specimen_prepared_specimen_cache_hit")
        elapsed = _seconds(metrics.get("specimen_preparation_seconds"))
        if isinstance(cache_hit, bool):
            detail += " | potential: " + ("cache hit" if cache_hit else "built")
            if elapsed is not None:
                detail += f" ({elapsed})"
        lines.append(detail)
        if "cuda_transmission_cache_hits" in metrics:
            lines.append(
                f"{label} GPU transmission | {metrics['cuda_transmission_cache_hits']} hits"
                f" | {metrics.get('cuda_transmission_cache_builds', 0)} builds"
                f" | {metrics.get('cuda_transmission_cache_bytes', 0) / 1024**2:.1f} MiB"
            )
        fallback = metrics.get("cuda_pipeline_fallback_reason") or metrics.get("fft_fallback_reason")
        if fallback:
            lines.append(f"{label} fallback | {fallback}")
    return tuple(lines)
