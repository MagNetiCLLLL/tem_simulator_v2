"""Camera/screen reception from a captured calculation, with bounded scan previews."""
from __future__ import annotations

import json
from time import perf_counter

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, QSignalBlocker, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

from temsim.detector.receiver_image import available_receivers, receiver_image
from temsim.detector.receiver_scan import prepare_receiver_scan
from temsim.gui.compact_status_label import CompactStatusLabel
from temsim.gui.input_policy import WheelSafeComboBox, WheelSafeDoubleSpinBox
from temsim.gui.page_calculation import PageCalculationBar


class ReceiverImagingView(QWidget):
    """Display only; instrument changes and transport remain explicit requests.

    Histograms/FFT previews are bounded to 192² pixels, built only while visible.
    A bounded display timer plays retained scan geometry, never new transport.
    """

    calculation_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result = None
        self._state = None
        self._displayed_image = None
        self._scan_preview = None
        self._stale = False
        self._dirty = True
        self._mode_chosen = False
        self._fit_next = True
        self._cache = {}
        self._prepared_scans = {}
        self._play_origin_s = 0.
        self._play_phase_s = 0.
        self._play_frame = 0
        self._loop_after_follow = False
        self._resume_on_show = False
        self._follow_source = None
        self._follow_active = False
        self._follow_suspended = False
        self._follow_time_s = 0.
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(100)
        self._playback_timer.timeout.connect(self._playback_tick)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._render)

        self.calculation_bar = PageCalculationBar("imaging", "calculateReceiverImaging")
        self.calculate_button = self.calculation_bar.button
        self.calculation_bar.requested.connect(self.calculation_requested.emit)
        self.receiver = WheelSafeComboBox()
        self.receiver.setObjectName("imagingReceiver")
        self.receiver.setSizeAdjustPolicy(WheelSafeComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.receiver.setMinimumContentsLength(18)
        self.receiver.setMinimumWidth(160)
        self.receiver.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.receiver.setToolTip("Installed receivers in the captured calculation. Selection does not insert or retract hardware.")
        self.mode = WheelSafeComboBox()
        self.mode.setObjectName("receiverDisplayMode")
        self.mode.addItem("Current position", "current")
        self.mode.addItem("Single scan preview", "frame")
        self.mode.addItem("Scan position preview", "position")
        self.mode.addItem("Frame build-up preview", "accumulate")
        self.mode.activated.connect(self._choose_mode)
        self.show_trajectory = QCheckBox("Scan trajectory")
        self.show_trajectory.setChecked(True)
        self.show_trajectory.setToolTip("First-order centroid trajectory relative to the captured particle image; cyan overlay is not measured intensity.")
        self.fit_receiver_button = QPushButton("Fit receiver")
        self.fit_beam_button = QPushButton("Fit beam")
        row = QHBoxLayout()
        row.addWidget(QLabel("Receiver"))
        row.addWidget(self.receiver, 1)
        row.addWidget(self.mode)
        row.addWidget(self.show_trajectory)
        row.addWidget(self.fit_receiver_button)
        row.addWidget(self.fit_beam_button)

        self.exposure = WheelSafeDoubleSpinBox()
        self.exposure.setDecimals(6)
        self.exposure.setRange(0.000001, 3600.)
        self.exposure.setValue(1.)
        self.exposure.setSuffix(" s")
        self.exposure.setToolTip("Static exposure for the retained beam position. Scan previews have no physical dose estimate.")
        self.quantity = WheelSafeComboBox()
        self.quantity.addItem("Intensity per emitted electron", "probability")
        self.quantity.addItem("Expected incident electrons", "electrons")
        self.log_scale = QCheckBox("Log display")
        self.export_button = QPushButton("Export receiver…")
        self.export_button.setEnabled(False)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Static exposure"))
        row2.addWidget(self.exposure)
        row2.addWidget(self.quantity)
        row2.addWidget(self.log_scale)
        row2.addStretch(1)
        row2.addWidget(self.export_button)

        self.play_button = QPushButton("Play scan")
        self.play_button.setObjectName("receiverPlayScan")
        self.step_button = QPushButton("Next position")
        self.reset_button = QPushButton("Reset scan")
        self.loop_scan = QCheckBox("Loop")
        self.loop_scan.setChecked(True)
        self.follow_scan = QCheckBox("Follow STEM scan")
        self.follow_scan.setChecked(True)
        self.follow_scan.setToolTip("Follow the matching Scanning Image frame. When it finishes, Loop can continue camera preview without acquiring another STEM frame.")
        self.speed = WheelSafeComboBox()
        for label, value in (("0.01×", .01), ("0.1×", .1), ("0.25×", .25), ("0.5×", .5), ("1×", 1.), ("2×", 2.)):
            self.speed.addItem(label, value)
        self.speed.setCurrentIndex(4)
        self.speed.setToolTip("Local preview speed. Refresh is capped at 10 Hz; frames shorter than 0.4 s are slowed for visibility. Next position visits individual retained pixels. Following STEM uses its own clock.")
        playback_row = QHBoxLayout()
        for widget in (self.play_button, self.step_button, self.reset_button,
                       self.loop_scan, self.follow_scan, QLabel("Speed"), self.speed):
            playback_row.addWidget(widget)
        playback_row.addStretch(1)
        self.playback_status = CompactStatusLabel("Scan preview stopped")

        self.position_row = QWidget()
        position_layout = QHBoxLayout(self.position_row)
        position_layout.setContentsMargins(0, 0, 0, 0)
        self.position = QSlider(Qt.Orientation.Horizontal)
        self.position.setObjectName("receiverScanPosition")
        self.position.setRange(0, 0)
        self.position_text = QLabel("No scan")
        position_layout.addWidget(QLabel("Scan position"))
        position_layout.addWidget(self.position, 1)
        position_layout.addWidget(self.position_text)
        self.position_row.hide()
        self.summary = CompactStatusLabel("No receiver result. Run high-accuracy once, then Calculate imaging.")
        self.notice = QLabel("Select an installed Camera or Fluorescent Screen. No image has been calculated.")
        self.notice.setWordWrap(True)
        self.notice.setStyleSheet("color: #fbbf24;")
        self.plot = pg.PlotItem()
        # Coordinates already use mm: leave pg's SI auto-prefixing disabled.
        self.plot.setLabel("bottom", "Receiver X (mm)")
        self.plot.setLabel("left", "Receiver Y (mm)")
        self.plot.setAspectLocked(True)
        self.image = pg.ImageView(view=self.plot)
        self.plot.invertY(False)
        self.image.setObjectName("receiverImageView")
        self.image.getImageItem().setOpts(axisOrder="row-major")
        self.image.ui.roiBtn.hide()
        self.image.ui.menuBtn.hide()
        self.image.setMinimumSize(250, 160)
        self.sensor_outline = pg.PlotCurveItem(pen=pg.mkPen("#64748b", width=1))
        self.plot.addItem(self.sensor_outline)
        self.sensor_outline.setZValue(19)
        self.trajectory = pg.PlotCurveItem(pen=pg.mkPen("#22d3ee", width=1.2))
        self.plot.addItem(self.trajectory)
        self.trajectory.setZValue(20)
        self.marker = pg.ScatterPlotItem(size=8, pen=pg.mkPen("#fbbf24"), brush=pg.mkBrush("#fbbf24"))
        self.plot.addItem(self.marker)
        self.marker.setZValue(21)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.addWidget(self.calculation_bar)
        layout.addLayout(row)
        layout.addLayout(row2)
        layout.addLayout(playback_row)
        layout.addWidget(self.playback_status)
        layout.addWidget(self.position_row)
        layout.addWidget(self.summary)
        layout.addWidget(self.notice)
        layout.addWidget(self.image, 1)

        self.receiver.currentIndexChanged.connect(self._receiver_changed)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.position.valueChanged.connect(self._manual_position)
        self.position.sliderPressed.connect(self.pause_scan)
        self.exposure.valueChanged.connect(self._schedule)
        self.quantity.currentIndexChanged.connect(self._schedule)
        self.log_scale.toggled.connect(self._schedule)
        self.show_trajectory.toggled.connect(self._schedule)
        self.fit_receiver_button.clicked.connect(self.fit_receiver)
        self.fit_beam_button.clicked.connect(self.fit_beam)
        self.export_button.clicked.connect(self._export)
        self.play_button.clicked.connect(self._toggle_play)
        self.step_button.clicked.connect(self._next_position)
        self.reset_button.clicked.connect(self._reset_scan)
        self.follow_scan.toggled.connect(self._follow_toggled)
        self.speed.currentIndexChanged.connect(self._speed_changed)
        self._update_playback_controls()

    def _choose_mode(self, *_args):
        self._mode_chosen = True

    def set_state(self, state):
        """Live state is used only before any result exists."""
        self._state = state
        if self._result is None:
            self._populate_receivers(state)

    def _populate_receivers(self, state):
        old = self.receiver.currentData()
        planes = available_receivers(state)
        with QSignalBlocker(self.receiver):
            self.receiver.clear()
            for plane in planes:
                insertion = "inserted" if bool(getattr(plane, "inserted", False)) else "retracted"
                self.receiver.addItem(f"{plane.name} · Z {plane.z_mm:g} mm · {insertion}", str(plane.key))
            index = self.receiver.findData(old)
            if index < 0:
                index = next((i for i, p in enumerate(planes) if bool(getattr(p, "inserted", False))), 0)
            self.receiver.setCurrentIndex(index if planes else -1)
        self._fit_next = True

    def display_result(self, result):
        def optical_only(candidate):
            metrics = getattr(getattr(candidate, "simulation", None), "metrics", {}) or {}
            return (getattr(candidate, "workflow", "") == "rays"
                    or bool(metrics.get("optical_tuning", False)))

        if (result is not None and optical_only(result) and self._result is not None
                and not optical_only(self._result)):
            # Updating rays does not acquire another physical receiver image.
            # Keep the last reception and its hardware snapshot reviewable.
            self.mark_result_stale()
            return
        self.pause_scan()
        self._follow_source = None
        self._follow_active = self._follow_suspended = False
        self._timer.stop()
        self._result = result
        self._cache.clear()
        self._prepared_scans.clear()
        self._stale = False
        self._displayed_image = self._scan_preview = None
        self.export_button.setEnabled(False)
        state = getattr(result, "state_snapshot", self._state) if result is not None else self._state
        self._populate_receivers(state)
        geometry = getattr(result, "scan_geometry", None)
        times = getattr(geometry, "times_s", None)
        size = np.asarray(times).size if times is not None else 0
        with QSignalBlocker(self.position):
            self.position.setRange(0, max(0, size - 1))
            self.position.setValue(0)
        if not self._mode_chosen:
            with QSignalBlocker(self.mode):
                self.mode.setCurrentIndex(1 if size > 0 else 0)
        # Never leave old pixels visible while a newer result awaits rendering.
        self._clear_pixels()
        self.summary.setText("Captured result ready." if result is not None else "No receiver result. Run high-accuracy once, then Calculate imaging.")
        self.calculation_bar.set_result_available(False)
        self.playback_status.setText("Scan preview stopped")
        self._update_playback_controls()
        self._schedule()

    def mark_result_stale(self, *_args):
        if self._result is not None:
            self.pause_scan()
            self._stale = True
            self.calculation_bar.mark_stale()
            self.playback_status.setText("Inputs changed — Calculate imaging before playing the updated scan")
            self._update_playback_controls()
            self._schedule()

    def _receiver_changed(self, *_args):
        self.pause_scan()
        self._fit_next = True
        self._schedule()

    def _mode_changed(self, *_args):
        if self.mode.currentData() not in {"position", "accumulate"}:
            self.pause_scan()
        self._schedule()

    def _manual_position(self, *_args):
        self.pause_scan()
        self._schedule()

    def _prepared_scan(self):
        key = self.receiver.currentData()
        if self._result is None or key is None:
            return None
        if key not in self._prepared_scans:
            self._prepared_scans[key] = prepare_receiver_scan(self._result, key, pixels=192)
        return self._prepared_scans[key]

    def _has_scan(self):
        return (not self._stale and self._result is not None
                and self.receiver.currentData() is not None and self.position.maximum() > 0)

    def _is_following(self):
        return (self.follow_scan.isChecked() and self._follow_active
                and not self._follow_suspended and self._same_scan_source(self._follow_source))

    def _update_playback_controls(self):
        available = self._has_scan()
        prepared = self._prepared_scans.get(self.receiver.currentData())
        if prepared is not None and getattr(prepared, "status", "GEOMETRIC_PREVIEW") != "GEOMETRIC_PREVIEW":
            available = False
        for button in (self.play_button, self.step_button, self.reset_button):
            button.setEnabled(available)
        running = self._playback_timer.isActive() or self._resume_on_show or self._is_following()
        self.play_button.setText("Pause scan" if running else "Play scan")
        self.speed.setEnabled(not self._is_following())

    def pause_scan(self, *_args):
        self._playback_timer.stop()
        self._loop_after_follow = False
        self._resume_on_show = False
        self._follow_suspended = True
        self.playback_status.setText("Scan preview paused")
        self._update_playback_controls()

    def _toggle_play(self):
        if self._playback_timer.isActive() or self._is_following():
            self.pause_scan()
        else:
            self.play_scan()

    def _select_playback_mode(self):
        if self.mode.currentData() not in {"position", "accumulate"}:
            with QSignalBlocker(self.mode):
                self.mode.setCurrentIndex(self.mode.findData("position"))
        self._mode_chosen = True

    def play_scan(self):
        if not self._has_scan():
            return
        self._select_playback_mode()
        self._loop_after_follow = False
        self._follow_suspended = False
        if self._is_following():
            self.follow_scan_time(self._follow_source, self._follow_time_s)
            self._update_playback_controls()
            return
        if not self.isVisible():
            self._resume_on_show = True
            self._dirty = True
            self._update_playback_controls()
            return
        try:
            prepared = self._prepared_scan()
            available = prepared is not None and prepared.status == "GEOMETRIC_PREVIEW"
        except (ValueError, TypeError, AttributeError, KeyError):
            available = False
        if not available:
            self._render()
            self.playback_status.setText("No usable scan reception — check the receiver status")
            return
        times = np.asarray(prepared.times_s).ravel()
        index = self.position.value()
        if index >= len(times)-1:
            index = 0
        self._play_phase_s = float(times[index]) if index else 0.
        self._play_origin_s = perf_counter()
        self._play_frame = 0
        self._resume_on_show = not self.isVisible()
        self._set_play_position(index)
        if self._displayed_image is None:
            return
        if self.isVisible():
            self._playback_timer.start()
        self.playback_status.setText("Playing scan preview · frame 1")
        self._update_playback_controls()

    def _preview_rate(self, period):
        return min(float(self.speed.currentData()), period / (4.*self._playback_timer.interval()*1e-3))

    def _set_play_position(self, index, *, immediate=True):
        with QSignalBlocker(self.position):
            self.position.setValue(int(index))
        self._dirty = True
        if self.isVisible():
            if immediate:
                self._render()
            elif not self._timer.isActive():
                self._timer.start(self._playback_timer.interval())

    def _playback_tick(self):
        if not self._playback_timer.isActive():
            return
        if not self.isVisible() or not self._has_scan():
            self.pause_scan()
            return
        if self._loop_after_follow:
            self._loop_after_follow = False
            self.play_scan()
            return
        prepared = self._prepared_scan()
        times = np.asarray(prepared.times_s).ravel()
        period = float(prepared.metrics["frame_period_s"])
        rate = self._preview_rate(period)
        elapsed = self._play_phase_s + max(0., perf_counter()-self._play_origin_s)*rate
        frame = int(elapsed // period)
        if frame and not self.loop_scan.isChecked():
            self._set_play_position(len(times)-1)
            self.pause_scan()
            self.playback_status.setText("Single preview frame complete; result retained")
            return
        # Present the completed exposure at each frame boundary. Slow displays
        # skip intermediate refreshes, not detector acquisitions (there are none).
        boundary = frame > self._play_frame
        self._play_frame = frame
        index = len(times)-1 if boundary else max(0, int(np.searchsorted(times, elapsed % period, side="right"))-1)
        self._set_play_position(index)
        if not self._playback_timer.isActive():
            return
        self.playback_status.setText(f"Playing scan preview · frame {max(1, frame) if boundary else frame+1}" + (" complete" if boundary else ""))
        if rate < float(self.speed.currentData()):
            self.playback_status.setText(self.playback_status.text() + " · slowed for display")

    def _next_position(self):
        if not self._has_scan():
            return
        self.pause_scan()
        self._select_playback_mode()
        index = self.position.value()+1
        if index > self.position.maximum():
            index = 0 if self.loop_scan.isChecked() else self.position.maximum()
        self._set_play_position(index)
        self.playback_status.setText("Manual scan position · preview only")

    def _reset_scan(self):
        self.pause_scan()
        self._select_playback_mode()
        self._play_frame = 0
        self._set_play_position(0)
        self.playback_status.setText("Scan preview reset")

    def _speed_changed(self, *_args):
        if self._playback_timer.isActive():
            self.play_scan()

    def _same_scan_source(self, source):
        if source is None or self._result is None or self._stale:
            return False
        if source is self._result:
            return True
        signatures = getattr(source, "signatures", {}) or {}
        ours = getattr(self._result, "signatures", {}) or {}
        return (getattr(source, "scan_geometry", None) is not None
                and source.scan_geometry is getattr(self._result, "scan_geometry", None)
                and all(signatures.get(key) and signatures[key] == ours.get(key)
                        for key in ("column", "sample_downstream")))

    def follow_scan_started(self, source, active):
        if not self._same_scan_source(source):
            return
        was_following = self._is_following()
        new_session = active and (source is not self._follow_source or not self._follow_active)
        self._follow_source, self._follow_active = source, bool(active)
        if active and self.follow_scan.isChecked():
            if new_session:
                self._follow_suspended = False
                self._follow_time_s = 0.
            if not self._follow_suspended:
                self._playback_timer.stop()
                self._resume_on_show = False
                self._select_playback_mode()
                self.playback_status.setText("Following STEM scan · camera preview")
        elif was_following:
            if self.loop_scan.isChecked():
                # Let the completed followed exposure paint before resetting.
                self._loop_after_follow = True
                if self.isVisible():
                    self._render()
                    if self._displayed_image is not None:
                        self._playback_timer.start()
                    else:
                        self._loop_after_follow = False
                        self._update_playback_controls()
                        return
                else:
                    self._resume_on_show = True
                self.playback_status.setText("STEM frame complete · camera preview will loop")
            else:
                self.pause_scan()
                self.playback_status.setText("STEM frame complete; camera preview retained")
        self._update_playback_controls()

    def follow_scan_time(self, source, time_s):
        if not self._same_scan_source(source) or not np.isfinite(time_s):
            return
        self._follow_time_s = float(time_s)
        if not self._is_following():
            return
        times = getattr(getattr(self._result, "scan_geometry", None), "times_s", None)
        if times is None:
            return
        # Receivers get the STEM clock directly; the local speed is irrelevant.
        index = max(0, min(np.asarray(times).size-1,
                          int(np.searchsorted(np.asarray(times).ravel(), time_s, side="right"))-1))
        self._set_play_position(index, immediate=False)
        self.playback_status.setText("Following STEM scan · camera preview")

    def _follow_toggled(self, checked):
        if not checked:
            self.pause_scan()
        elif self._follow_active and self._same_scan_source(self._follow_source):
            self._follow_suspended = False
            self._playback_timer.stop()
            self._resume_on_show = False
            self._select_playback_mode()
            self.follow_scan_time(self._follow_source, self._follow_time_s)
        self._update_playback_controls()

    def _schedule(self, *_args):
        self._dirty = True
        self.export_button.setEnabled(False)
        if self.isVisible():
            self._timer.start()

    def showEvent(self, event):
        super().showEvent(event)
        if self._resume_on_show and self._has_scan():
            self._resume_on_show = False
            self.play_scan()
        elif self._is_following():
            self.follow_scan_time(self._follow_source, self._follow_time_s)
        if self._dirty:
            self._timer.start()

    def hideEvent(self, event):
        self._timer.stop()
        self._resume_on_show = self._playback_timer.isActive() and not self._stale
        self._playback_timer.stop()
        super().hideEvent(event)

    def _clear_pixels(self):
        self.image.clear()
        self.plot.setTitle("No received image")
        self.trajectory.setData([], [])
        self.marker.setData([], [])
        self.sensor_outline.setData([], [])

    def _render(self):
        self._dirty = False
        self._timer.stop()
        self._displayed_image = self._scan_preview = None
        self.export_button.setEnabled(False)
        mode = self.mode.currentData()
        preview_mode = mode != "current"
        self.position_row.setVisible(mode in {"position", "accumulate"})
        self.show_trajectory.setEnabled(preview_mode)
        self.exposure.setEnabled(not preview_mode)
        self.quantity.setEnabled(not preview_mode)
        if preview_mode and self.quantity.currentData() != "probability":
            with QSignalBlocker(self.quantity):
                self.quantity.setCurrentIndex(0)
        key = self.receiver.currentData()
        if self._result is None or key is None:
            self._clear_pixels()
            self.summary.setText("No receiver result. Run high-accuracy once, then Calculate imaging.")
            self.notice.setText("No installed Camera or Fluorescent Screen." if key is None else "No reception calculated; empty is not a zero-signal result.")
            return
        try:
            cache_key = (key, mode, self.position.value() if mode in {"position", "accumulate"} else None,
                         self.exposure.value() if mode == "current" else None)
            if cache_key not in self._cache:
                if preview_mode:
                    value = self._prepared_scan().preview(
                        self.position.value() if mode in {"position", "accumulate"} else None,
                        accumulate=mode == "accumulate")
                else:
                    value = receiver_image(self._result, key, pixels=192, exposure_s=self.exposure.value())
                if len(self._cache) >= 6:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[cache_key] = value
            value = self._cache[cache_key]
            if preview_mode:
                self._scan_preview = value
                image = value.image
            else:
                image = value
            status = value.status
            if image is None or status not in {"AVAILABLE", "GEOMETRIC_PREVIEW"}:
                self._clear_pixels()
                prefix = "Previous calculation · " if self._stale else ""
                self.summary.setText(prefix + status.replace("_", " ") + " · " + value.detail)
                self.notice.setText(value.detail)
                self.calculation_bar.set_result_available(False)
                if self._stale:
                    self.calculation_bar.mark_stale()
                self._playback_timer.stop()
                self._resume_on_show = False
                self._follow_suspended = True
                self._update_playback_controls()
                return
            self._displayed_image = image
            counts = not preview_mode and self.quantity.currentData() == "electrons"
            values = image.expected_electrons if counts else image.response_probability
            if values is None:
                raise ValueError("No physical incident-electron expectation is available for this result")
            values = np.asarray(values)
            contrast_peak = float(np.max(values))
            if mode in {"position", "accumulate"}:
                prepared = self._prepared_scan()
                reference = prepared.base_image
                if mode == "accumulate":
                    full_key = (key, "frame", None, None)
                    if full_key not in self._cache:
                        self._cache[full_key] = prepared.preview()
                    reference = self._cache[full_key].image
                if reference is not None:
                    contrast_peak = float(np.max(reference.response_probability))
            display = np.log10(1. + values / max(contrast_peak, 1e-300) * 1e4) if self.log_scale.isChecked() else values
            max_value = float(np.log10(1.+1e4)) if self.log_scale.isChecked() else contrast_peak
            self.image.setImage(display, autoRange=False, autoLevels=False, levels=(0., max_value if max_value > 0 else 1.))
            dx, dy = float(image.x_mm[1]-image.x_mm[0]), float(image.y_mm[1]-image.y_mm[0])
            self.image.getImageItem().setRect(QRectF(float(image.x_mm[0]-.5*dx), float(image.y_mm[0]-.5*dy), dx*len(image.x_mm), dy*len(image.y_mm)))
            self._draw_sensor_outline(image)
            title = "Geometric scan preview" if preview_mode else "Physical particle reception"
            self.plot.setTitle(title + (" · log display" if self.log_scale.isChecked() else " · linear display"))
            self.trajectory.setData([], [])
            self.marker.setData([], [])
            if preview_mode:
                self._draw_trajectory(value)
            if self._fit_next:
                self.fit_receiver()
                self._fit_next = False
            state = self._result.state_snapshot
            optics_mode = str(getattr(state, "projector_mode", "unknown"))
            prefix = "Previous calculation · inputs changed · " if self._stale else "Captured calculation · "
            detail = f"{image.name} | {optics_mode} mode | "
            if preview_mode:
                detail += f"{self.mode.currentText()} | no physical dose estimate"
                geometry = getattr(self._result, "scan_geometry", None)
                detail += (f" | AC {'ON' if getattr(geometry, 'ac_enabled', False) else 'OFF'}"
                           f" · Descan {'ON' if getattr(geometry, 'descan_enabled', False) else 'OFF'}")
                if mode == "accumulate":
                    detail += f" | frame exposure {100.*float(image.metrics.get('exposure_fraction', 0.)):.1f}%"
                self.notice.setText("Geometry preview: scan-dependent specimen scattering and upstream clipping are not recalculated. Cyan: predicted centroid path.")
            else:
                detail += f"Received {100.*float(image.ideal_probability.sum()):.5g}% | {image.current_pa:.6g} pA | exposure {image.exposure_s:g} s"
                self.notice.setText("Classical particle reception, including calculated scattering. Coherent diffraction/interference is not calculated here.")
                if not np.any(image.ideal_probability):
                    detail += " | No electrons received"
            self.summary.setText(prefix + detail)
            self.notice.setToolTip(value.detail + "\n" + str(image.metrics.get("response_scope", "")))
            self.calculation_bar.set_result_available(True)
            if self._stale:
                self.calculation_bar.mark_stale()
            self.export_button.setEnabled(True)
            self._update_playback_controls()
        except (ValueError, TypeError, AttributeError, KeyError) as error:
            self._clear_pixels()
            self._displayed_image = self._scan_preview = None
            self.summary.setText("Receiver image unavailable · " + str(error))
            self.notice.setText("The retained data cannot provide this receiver image. Calculate imaging to refresh.")
            self.calculation_bar.set_result_available(False)
            self.pause_scan()

    def _draw_sensor_outline(self, image):
        plane = next(p for p in available_receivers(self._result.state_snapshot)
                     if str(p.key) == image.key)
        dx = float(image.x_mm[1] - image.x_mm[0])
        width = float(getattr(plane, "outer_width_mm", dx * len(image.x_mm)))
        cx = float((image.x_mm[0] + image.x_mm[-1]) / 2.)
        cy = float((image.y_mm[0] + image.y_mm[-1]) / 2.)
        geometry = str(getattr(plane, "geometry", "disk" if image.key == "flu_screen" else "square")).lower()
        if geometry in {"square", "rectangle", "camera"}:
            x, y = np.asarray([-1, 1, 1, -1, -1]), np.asarray([-1, -1, 1, 1, -1])
        else:
            angles = np.linspace(0., 2.*np.pi, 129)
            x, y = np.cos(angles), np.sin(angles)
        self.sensor_outline.setData(cx + .5*width*x, cy + .5*width*y)

    def _draw_trajectory(self, preview):
        times = np.asarray(preview.times_s).ravel()
        index = min(self.position.value(), max(0, times.size - 1))
        self.position_text.setText(f"{index+1}/{times.size} · {times[index]:.6g} s")
        if not self.show_trajectory.isChecked():
            return
        # Static centroid + displacement, never add an absolute scan twice.
        cache_key = (preview.key, "trajectory_base")
        if cache_key not in self._cache:
            self._cache[cache_key] = self._prepared_scan().base_image
        base = self._cache[cache_key]
        if base is None:
            return
        probability = np.asarray(base.ideal_probability)
        total = float(probability.sum())
        if total <= 0:
            return
        x0 = float(np.sum(probability * base.x_mm[None, :]) / total)
        y0 = float(np.sum(probability * base.y_mm[:, None]) / total)
        x = x0 + np.asarray(preview.displacement_x_mm).ravel()
        y = y0 + np.asarray(preview.displacement_y_mm).ravel()
        self.trajectory.setData(x, y)
        if self.mode.currentData() in {"position", "accumulate"}:
            self.marker.setData([x[index]], [y[index]])

    def fit_receiver(self):
        image = self._displayed_image
        if image is None:
            return
        dx, dy = image.x_mm[1]-image.x_mm[0], image.y_mm[1]-image.y_mm[0]
        self.plot.setRange(xRange=(image.x_mm[0]-.5*dx, image.x_mm[-1]+.5*dx),
                           yRange=(image.y_mm[0]-.5*dy, image.y_mm[-1]+.5*dy), padding=.02)

    def fit_beam(self):
        image = self._displayed_image
        if image is None:
            return
        values = np.asarray(image.response_probability)
        rows, columns = np.nonzero(values > max(float(values.max())*1e-8, 0.))
        if not len(rows):
            self.fit_receiver()
            return
        dx, dy = image.x_mm[1]-image.x_mm[0], image.y_mm[1]-image.y_mm[0]
        self.plot.setRange(xRange=(image.x_mm[columns.min()]-dx, image.x_mm[columns.max()]+dx),
                           yRange=(image.y_mm[rows.min()]-dy, image.y_mm[rows.max()]+dy), padding=.1)

    def export_result(self, path):
        """Keep physical probabilities, axes and preview provenance in one file."""
        image = self._displayed_image
        if image is None:
            raise ValueError("No receiver image is displayed")
        metadata = dict(image.metrics, status=image.status, detail=image.detail,
                        receiver_name=image.name, stale=self._stale,
                        exposure_s=image.exposure_s, current_pa=image.current_pa)
        arrays = dict(ideal_probability=image.ideal_probability,
                      response_probability=image.response_probability,
                      x_mm=image.x_mm, y_mm=image.y_mm,
                      metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, default=str)))
        if image.expected_electrons is not None:
            arrays["expected_electrons"] = image.expected_electrons
        if self._scan_preview is not None:
            for name in ("times_s", "displacement_x_mm", "displacement_y_mm", "trajectory_x_mm", "trajectory_y_mm"):
                arrays[name] = getattr(self._scan_preview, name)
        np.savez_compressed(path, **arrays)

    def _export(self):
        if self._displayed_image is None:
            return
        self.pause_scan()
        path, _ = QFileDialog.getSaveFileName(self, "Export receiver image", "receiver-image.npz", "NumPy archive (*.npz)")
        if not path:
            return
        try:
            self.export_result(path)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Receiver export", str(error))
