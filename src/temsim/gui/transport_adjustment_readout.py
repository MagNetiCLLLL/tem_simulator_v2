"""Small, selectable explanation of the captured condenser adjustment."""
from collections.abc import Mapping
import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSizePolicy


class TransportAdjustmentReadout(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("transportAdjustmentReadout")
        self.setWordWrap(True)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse
                                    | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setStyleSheet("color: #bae6fd; background: #172033; "
                          "border: 1px solid #334155; border-radius: 3px; padding: 3px;")
        self.hide()

    def display_result(self, result):
        metrics = getattr(getattr(result, "simulation", None), "metrics", None) or {}
        record = metrics.get("transport_adjustment")
        self.clear()
        self.setToolTip("")
        self.setVisible(record is not None)
        if record is None:
            return
        try:
            if not isinstance(record, Mapping) or record["physics_scope"] != "optical_transport_only":
                raise ValueError("Unsupported adjustment details")
            controls = record["controls"]
            expected = tuple(f"condenser_lens_{i}" for i in (1, 2, 3))
            if tuple(row["key"] for row in controls) != expected:
                raise ValueError("Condenser controls are incomplete")
            changes = []
            for i, row in enumerate(controls, 1):
                before, after = float(row["before_percent"]), float(row["after_percent"])
                if not all(math.isfinite(v) and v >= 0 for v in (before, after)):
                    raise ValueError("Invalid condenser excitation")
                changes.append(f"C{i} {before:.6g}% → {after:.6g}%")
            count = record["source_ray_count"]
            plane, step = float(record["target_plane_z_mm"]), float(record["validation_step_mm"])
            fraction = float(record["source_fraction"])
            if (type(count) is not int or count <= 0 or not math.isfinite(plane)
                    or not math.isfinite(step) or step <= 0 or not 0 <= fraction <= 1):
                raise ValueError("Invalid transport measurement")
            self.setText(
                "Condenser adjustment in this result: " + " | ".join(changes)
                + f"\nOptical validation: {count:,} source samples · "
                  f"{100*fraction:.3g}% of source through Z {plane:.6g} mm · "
                  "Specimen signals deferred · Probe/image focus not calibrated."
            )
            self.setToolTip(
                f"Captured changes in condenser excitation; validation column step {step:g} mm. "
                "Only a subset of the source samples is drawn. The validation result uses "
                "an optical reference through the specimen plane. Use Live tuning → "
                "Working points → Advanced → Undo last apply to undo the latest applied adjustment. "
                "This record belongs to the displayed result, independently of later live edits."
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            self.setText("Condenser adjustment details unavailable in this result.")
