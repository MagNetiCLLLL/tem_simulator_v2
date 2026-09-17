"""Compact observations of an explicitly selected retained result only."""
import json

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel
from temsim.gui.input_policy import WheelSafeComboBox as QComboBox

from temsim.immutable_json import freeze_json, thaw_json
from temsim.sampling_diagnostics import sampling_summary


def result_readout(result, quality):
    """Never read the current instrument or interpolate plot histories here."""
    state = getattr(result, "state_snapshot", None)
    simulation = getattr(result, "simulation", None)
    manifest = getattr(result, "calculation_manifest", None)
    identity = getattr(result, "signatures", {}).get("request", "UNRECORDED")
    summary = dict(result_id=identity, manifest_id=getattr(manifest, "digest", "UNRECORDED"),
        numerical_preset=quality, numerical_validation="NOT_ESTABLISHED", model_qualification="NOT_ESTABLISHED",
        model=str(getattr(state, "simulation_mode", "Unrecorded")),
        illumination_mode=str(getattr(state, "illumination_mode", "Unrecorded")),
        ray_backend=str(getattr(state, "active_backend", "Unrecorded")),
        wave_backend="NOT_COMPUTED", observations=None,
        precision="UNAVAILABLE: no exact incident checkpoint retained at the captured sample centre")
    wave = getattr(result, "wave_imaging", None)
    if wave is not None:
        summary["wave_backend"] = str(wave.metrics.get("wave_compute_backend", "Unrecorded"))
    plane = getattr(getattr(state, "sample", None), "z_mm", None)
    checkpoints = getattr(simulation, "incident_checkpoints", None)
    incident = getattr(simulation, "incident", None)
    if plane is not None and checkpoints is not None and incident is not None:
        matches = np.flatnonzero(np.asarray(checkpoints.z_mm) == float(plane))
        if len(matches) == 1:
            from temsim.physics.beam_current import effective_source_current_a
            arrays = {key: getattr(checkpoints, key)[matches[0]] for key in ("x_m", "y_m", "tx_rad", "ty_rad")}
            arrays.update(weight=incident.ray_weight, alive=incident.alive)
            summary["observations"] = sampling_summary(arrays, plane_z_mm=plane,
                source_current_a=effective_source_current_a(state))
            summary["precision"] = "Exact incident integration checkpoint at captured sample centre"
    return freeze_json(summary)


class ResultReadout(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sharedResultReadout")
        self._records = {}
        self._stale = {}
        self._revision = 0
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.selection = QComboBox()
        self.selection.setObjectName("readoutResultSelection")
        self.selection.addItem("Ray result", "ray")
        self.selection.addItem("Retained High accuracy", "high")
        self.selection.setToolTip("Choose which retained result supplies every readout; selecting does not calculate or change the instrument")
        self.selection.currentIndexChanged.connect(self._refresh)
        layout.addWidget(self.selection)
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.label, 1)
        self._refresh()

    def set_revision(self, revision):
        self._revision = int(revision)
        self._refresh()

    def publish(self, result, quality):
        try:
            row = result_readout(result, quality)
        except (ValueError, TypeError, AttributeError) as exc:
            # A failed readout cannot replace or reinterpret a valid plot.
            row = freeze_json(dict(error=str(exc), result_id=getattr(result, "signatures", {}).get("request", "UNRECORDED")))
        self._records["ray"] = row
        self._stale["ray"] = False
        if str(quality).strip().lower() == "high accuracy":
            self._records["high"] = row
            self._stale["high"] = False
        self._refresh()

    def mark_stale(self, key):
        self._stale[key] = True
        self._refresh()

    def _refresh(self):
        key = self.selection.currentData()
        row = self._records.get(key)
        prefix = f"Live settings revision {self._revision} | "
        if row is None:
            self.label.setText(prefix+"No retained result for this readout")
            self.label.setToolTip("")
            return
        prefix += f"Result {row['result_id'][:12]} | "
        prefix += "STALE — captured inputs differ or recalculation is pending\n" if self._stale.get(key) else "Captured result\n"
        if "error" in row:
            self.label.setText(prefix+"Readout unavailable: "+row["error"])
        else:
            observations = row["observations"]
            def shown(name, factor=1.):
                value = observations.get(name) if observations else None
                return "Unavailable" if value is None else f"{value*factor:.5g}"
            metrics = (f"Z {shown('plane_z_mm')} mm | I {shown('plane_current_a', 1e12)} pA | "
                f"D95 {shown('diameter95_m', 1e6)} um | alpha95 {shown('alpha95_rad', 1e3)} mrad | "
                f"Centre ({shown('centroid_x_m', 1e6)}, {shown('centroid_y_m', 1e6)}) um | "
                f"Transmission {shown('transmission', 100)}%")
            status = (f"\nModel {row['model']} / {row['illumination_mode']} | Preset {row['numerical_preset']} | "
                f"Ray backend {row['ray_backend']} | Numerical validation NOT_ESTABLISHED | Model qualification NOT_ESTABLISHED")
            self.label.setText(prefix+metrics+status)
        self.label.setToolTip(json.dumps(thaw_json(row), indent=2))
