"""Read-only timings and reuse evidence for the displayed calculation."""
from __future__ import annotations

import math
from PySide6.QtWidgets import QGroupBox, QPlainTextEdit, QVBoxLayout

from temsim.calculation_performance import calculation_performance_lines, _seconds


def timing_details(result, duration=None):
    """Read small metadata only; inherited cache timings are never current work."""
    lines = []
    elapsed = _seconds(duration)
    if elapsed is not None:
        lines.append(f"Calculation worker: {elapsed} (includes internal cache IO; excludes queue, GUI drawing and section-archive save)")
    if bool(getattr(result, "cache_hit", False)):
        lines.extend(calculation_performance_lines(result))
        return "\n".join(lines)
    performance = getattr(result, "performance", None) or {}
    elapsed = _seconds(performance.get("pipeline_seconds"))
    if elapsed is not None:
        lines.append(f"Pipeline: {elapsed}")
    scope = performance.get("timing_scope")
    if scope:
        lines.append(str(scope))
    lines.extend(calculation_performance_lines(result))
    reused = set(getattr(result, "reused_products", ()) or ())
    if reused:
        names = {"column": "electron column", "incident": "incident beam",
                 "elastic": "material trajectories", "inelastic": "inelastic populations",
                 "eds": "EDS spectrum", "eds_response": "EDS material/photon response",
                 "sample_downstream": "specimen-exit transport",
                 "energy_filter": "energy filter", "stem": "STEM image", "wave": "TEM wave"}
        lines.append("Reused products: " + ", ".join(names.get(key, str(key).replace("_", " "))
                                                      for key in sorted(reused)))
    column_reused = "column" in reused
    metrics = getattr(getattr(result, "simulation", None), "metrics", None) or {}
    gun_reused = metrics.get("section_gun_reused")
    elapsed = _seconds(performance.get("gun_seconds", metrics.get("section_gun_seconds")))
    if not column_reused and isinstance(gun_reused, bool):
        lines.append("Electron gun: " + ("executed state reused" if gun_reused else "calculated from tip emission")
                     + (f" | {elapsed} (included in transport stage)" if elapsed is not None else ""))
    column_cache = metrics.get("column_segment_cache", {}) or {}
    for key, label in (("section_target_z_mm", "Calculated through"),
                       ("section_resume_z_mm", "Reused upstream through")):
        value = metrics.get(key)
        if key == "section_resume_z_mm":
            if column_reused or not metrics.get("section_reused_prefix", column_cache.get("hit", False)):
                continue
            value = metrics.get(key, column_cache.get("resume_z_mm"))
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            lines.append(f"{label}: Z = {value:.9g} mm")
    reasons = {
        "no_prior_checkpoint": "No previous executed section was available.",
        "upstream_inputs_or_model_changed": "Upstream inputs, numerical settings or solver version changed.",
        "invalid_saved_state": "The saved state failed validation.",
        "no_compatible_column_prefix": "No compatible upstream column section was available.",
        "vacuum_restart_required": "Vacuum transport required recalculation.",
        "compatible_executed_prefix": "Compatible executed upstream state reused.",
    }
    reason = metrics.get("section_reuse_reason")
    if reason and not column_reused:
        lines.append("Reuse: " + reasons.get(reason, str(reason).replace("_", " ")))
    if not lines:
        return "Timing metadata unavailable for this retained result."
    lines.append("Stage intervals and included substeps must not be added to the pipeline or worker total.")
    return "\n".join(lines)


class CalculationTimingView(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Calculation details", parent)
        self.setObjectName("calculationTimingGroup")
        self.setCheckable(True)
        self.setChecked(False)
        self.text = QPlainTextEdit()
        self.text.setObjectName("calculationTimingDetails")
        self.text.setReadOnly(True)
        self.text.setMaximumHeight(190)
        self.text.setPlainText("No completed calculation.")
        layout = QVBoxLayout(self)
        layout.addWidget(self.text)
        self.text.hide()
        self.toggled.connect(self.text.setVisible)

    def set_result(self, result, duration=None):
        self.setTitle("Calculation details — displayed result")
        self.text.setPlainText(timing_details(result, duration))

    def mark_stale(self):
        self.setTitle("Calculation details — previous result")
