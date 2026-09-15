"""Read-only wave counterpart of the existing transverse ray panel.

Accepts executed checkpoints only through an internal presentation method.
There is no source import, propagation, admission, or aggregate phase here.
"""
from html import escape
import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF
from PySide6.QtGui import QTransform
from PySide6.QtWidgets import QTableWidgetItem

from temsim.gui.transverse_projection import transverse_view_coordinates
from temsim.physics.wave_plane_observables import (
    canonical_angular_spectrum, mode_phase_samples, interaction_weights,
)


class WaveBeamAnalysis:
    MODES = (("Beam current", "wave_current"), ("Canonical angular distribution", "wave_angles"),
             ("Phase — one mode", "wave_phase"), ("Interaction breakdown", "wave_interactions"))
    BINS = 128

    def __init__(self, analysis, checkpoint, axial_bz_t, maximum_working_bytes):
        if not math.isfinite(axial_bz_t):
            raise ValueError("Wave view needs the actual axial magnetic field")
        self.analysis, self.owner = analysis, analysis.owner
        self.checkpoint, self.bz = checkpoint, float(axial_bz_t)
        self.maximum_working_bytes = maximum_working_bytes
        self.saved_mode, self.saved_colour = analysis.mode, analysis.colour_combo.currentData()
        self.ranges = {}
        self.mode = "wave_current"
        self._phase = None
        self._angular = None  # One derived mode only; never retain an unbounded mixture.
        self._hover = None
        self.values = None

    def activate(self):
        a = self.analysis
        for combo in (a.mode_combo, a.colour_combo):
            combo.blockSignals(True)
            combo.clear()
        for label, key in self.MODES:
            a.mode_combo.addItem(label, key)
        a.colour_combo.addItem("All modes (intensity sum)", None)
        for i, mode in enumerate(self.checkpoint.beam.modes):
            a.colour_combo.addItem(f"{mode.mode_id} | {mode.energy_kev:.7g} keV", i)
        for combo in (a.mode_combo, a.colour_combo):
            combo.blockSignals(False)
        a.mode_combo.setToolTip("Read this executed wave checkpoint. No propagation or source changes.")
        a.colour_label.setText("Mode")
        a.colour_combo.setToolTip("Independent modes add intensities, never phases.")
        self.owner.initial_beam_panel.hide()
        a._hover_payload = None
        a._resize_timer.stop()
        self.owner.plot.setAspectLocked(True)
        self.redraw()

    def deactivate(self):
        a = self.analysis
        for combo in (a.mode_combo, a.colour_combo):
            combo.blockSignals(True)
            combo.clear()
        for label, key in a.MODES:
            a.mode_combo.addItem(label, key)
        for label, key in a.COLOUR_MODES:
            a.colour_combo.addItem(label, key)
        a.mode_combo.setCurrentIndex(a.mode_combo.findData(self.saved_mode))
        a.colour_combo.setCurrentIndex(a.colour_combo.findData(self.saved_colour))
        for combo in (a.mode_combo, a.colour_combo):
            combo.blockSignals(False)
        a.mode = self.saved_mode
        a.colour_label.setText("Colour by")
        a.mode_combo.setToolTip("Analyse cached rays at the selected Z. No retracing.")
        a.colour_combo.setToolTip("Source colour follows ancestry; interaction colour follows the recorded channel.")
        a._update_controls()
        self.owner.plot.setAspectLocked(a.mode in a.EQUAL_UNIT_MODES)

    def selection_changed(self):
        self.mode = self.analysis.mode_combo.currentData()
        if self.mode != "wave_phase":
            self._phase = None
        if self.mode != "wave_angles":
            self._angular = None
        if self.mode == "wave_phase" and self.analysis.colour_combo.currentData() is None:
            self.analysis.colour_combo.blockSignals(True)
            self.analysis.colour_combo.setCurrentIndex(1)
            self.analysis.colour_combo.blockSignals(False)
        self.redraw()

    def capture_range(self):
        if self.analysis._drawing:
            return
        self.ranges[self.mode] = tuple((-abs(b-a)/2, abs(b-a)/2)
                                      for a, b in self.owner.plot.viewRange())
        self.redraw()

    def fit(self):
        self.ranges.pop(self.mode, None)
        self.redraw()

    def _modes(self):
        index = self.analysis.colour_combo.currentData()
        if index is None:
            return self.checkpoint.beam.modes
        return (self.checkpoint.beam.modes[index],)

    def _coordinates_and_weight(self, mode):
        from temsim.physics.wave_execution import check_available_memory
        needed = 96*mode.plane.amplitude.size
        if needed > self.maximum_working_bytes:
            raise MemoryError("Wave display exceeds its working-memory budget")
        check_available_memory(needed)
        if self.mode == "wave_angles":
            if self._angular is None or self._angular[0] != mode.mode_id:
                self._angular = (mode.mode_id, canonical_angular_spectrum(mode,
                    maximum_working_bytes=self.maximum_working_bytes))
            result = self._angular[1]
            return result["canonical_angles_rad"]*1e3, result["probability_per_tip_electron"]
        return (mode.plane.coordinates_m()*1e6,
                mode.weight_per_reference_electron*abs(mode.plane.amplitude)**2)

    def _coordinate_corners(self, mode):
        wave = mode.plane
        ny, nx = wave.amplitude.shape
        indices = np.array(((-(nx//2), (nx-1)//2, (nx-1)//2, -(nx//2)),
                            (-(ny//2), -(ny//2), (ny-1)//2, (ny-1)//2)))
        if self.mode == "wave_angles":
            from temsim.optics.electron_gun.tip_coherence import wavelength_m
            return wavelength_m(mode.energy_kev*1000)*1e3*np.linalg.inv(wave.basis_m).T@(indices/np.array((nx, ny))[:, None])
        return (wave.origin_m[:, None]+wave.basis_m@indices)*1e6

    def _draw_current(self):
        from temsim.physics.affine_cell_histogram import affine_cell_histogram
        bounds = self.ranges.get(self.mode)
        modes = self._modes()
        if bounds is None:
            half = np.full(2, 1e-12)
            for mode in modes:
                xy = self._coordinate_corners(mode)
                u, v = transverse_view_coordinates(*xy, self.owner._projection_angle_deg)
                half = np.maximum(half, (np.max(abs(u)), np.max(abs(v))))
            bounds = tuple((-1.08*h, 1.08*h) for h in half)
        phi = math.radians(self.owner._projection_angle_deg)
        rotation = np.array(((math.cos(phi), math.sin(phi)), (-math.sin(phi), math.cos(phi))))
        values = np.zeros((self.BINS, self.BINS))
        display_records = []
        for mode in modes:
            _, weight = self._coordinates_and_weight(mode)
            basis, origin = mode.plane.basis_m*1e6, mode.plane.origin_m*1e6
            if self.mode == "wave_angles":
                from temsim.optics.electron_gun.tip_coherence import wavelength_m
                ny, nx = mode.plane.amplitude.shape
                basis = wavelength_m(mode.energy_kev*1000)*1e3*np.linalg.inv(mode.plane.basis_m).T@np.diag((1/nx, 1/ny))
                origin = np.zeros(2)
            contribution, record = affine_cell_histogram(weight, rotation@basis, rotation@origin, bounds, self.BINS)
            values += contribution
            display_records.append(record)
        values *= self.checkpoint.reference_current_a*1e12
        self.values = values
        self.display_records = tuple(display_records)
        image = pg.ImageItem(values.T, axisOrder="row-major")
        image.setLevels((0., max(float(values.max()), np.finfo(float).tiny)))
        image.setRect(QRectF(bounds[0][0], bounds[1][0], bounds[0][1]-bounds[0][0], bounds[1][1]-bounds[1][0]))
        self.owner.plot.addItem(image)
        self._hover = (values, np.linspace(*bounds[0], self.BINS+1), np.linspace(*bounds[1], self.BINS+1))
        return bounds

    def _draw_phase(self):
        mode, = self._modes()
        if self._phase is None or self._phase[0] != mode.mode_id:
            self._phase = (mode.mode_id, mode_phase_samples(mode,
                maximum_working_bytes=self.maximum_working_bytes))
        result = self._phase[1]
        image = pg.ImageItem(result, axisOrder="row-major")
        image.setAutoDownsample(False)  # Averaging wrapped phase is not a phase readout.
        image.setLookupTable(pg.colormap.get("CET-C2").getLookupTable())
        image.setLevels((-np.pi, np.pi))
        phi = math.radians(self.owner._projection_angle_deg)
        rotation = np.array(((math.cos(phi), math.sin(phi)), (-math.sin(phi), math.cos(phi))))
        ny, nx = mode.plane.amplitude.shape
        basis = rotation@mode.plane.basis_m*1e6
        origin = rotation@mode.plane.origin_m*1e6-basis@np.array((nx//2+.5, ny//2+.5))
        image.setTransform(QTransform(basis[0, 0], basis[1, 0], basis[0, 1], basis[1, 1], *origin))
        self.owner.plot.addItem(image)
        self.values = result
        u, v = transverse_view_coordinates(*self._coordinate_corners(mode), self.owner._projection_angle_deg)
        return self.ranges.get(self.mode, tuple((-max(1e-12, float(abs(x).max()))*1.08,
                                               max(1e-12, float(abs(x).max()))*1.08) for x in (u, v)))

    def _draw_interactions(self):
        rows = interaction_weights(self.checkpoint)
        self.analysis.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            history = row["history"]
            label = "No recorded inelastic branch" if not history else " + ".join(str(event.get("kind", "Recorded event")) for event in history)
            for j, text in enumerate((label, f"{100*row['weight_per_tip_electron']:.6g}", f"{1e12*row['current_a']:.6g}")):
                item = QTableWidgetItem(text)
                item.setToolTip(str(history))
                self.analysis.table.setItem(i, j, item)
        self.analysis.table.resizeRowsToContents()

    def redraw(self):
        a, owner = self.analysis, self.owner
        a._drawing = True
        self._hover = None
        self.values = None
        try:
            owner.plot.clear()
            owner._scatter = None
            a.readout.clear()
            a.legend.show()
            a.readout.show()
            interactions = self.mode == "wave_interactions"
            owner.plot.setVisible(not interactions)
            a.table.setVisible(interactions)
            a.colour_combo.setVisible(not interactions)
            a.colour_label.setVisible(not interactions)
            owner.fit_beam.setEnabled(not interactions)
            owner.fit_beam.setToolTip("Fit the cached wave view. No recalculation.")
            owner.initial_beam_panel.hide()
            unit = "mrad" if self.mode == "wave_angles" else "µm"
            for axis, label in (("bottom", "U"), ("left", "V")):
                owner.plot.getAxis(axis).setLabel(label, units=unit, siPrefixEnableRanges=())
            owner.heading.setText(f"Wave section | Z {self.checkpoint.plane_z_mm:.9g} mm")
            a.legend.setText("Executed wave checkpoint · not detector counts")
            a.legend.setToolTip(escape(str(self.checkpoint.record)))
            if not math.isclose(owner._plane_z_mm, self.checkpoint.plane_z_mm, rel_tol=0., abs_tol=1e-12):
                a.table.setRowCount(0)
                owner.summary.setText(f"Z {owner._plane_z_mm:.9g} mm is not cached. No wave interpolation or propagation was performed.")
                return
            if interactions:
                self._draw_interactions()
            else:
                bounds = self._draw_phase() if self.mode == "wave_phase" else self._draw_current()
                owner.plot.setRange(xRange=bounds[0], yRange=bounds[1], padding=0., disableAutoRange=True)
                self.ranges[self.mode] = owner.plot.viewRange()
            current = 1e12*self.checkpoint.transmitted_current_a
            owner.summary.setText(f"Total forward current {current:.7g} pA | {len(self.checkpoint.beam.modes)} modes")
            if self.mode == "wave_phase":
                a.legend.setText("Single-mode phase (rad) · no mixture phase")
                a.legend.setToolTip(str(self._modes()[0].axial_reference))
            elif self.mode == "wave_angles":
                a.legend.setText(f"Canonical angles | Bz {self.bz:.6g} T")
                a.legend.setToolTip("Full complex phase, in the inherited laboratory gauge. Inside a magnetic field these are not kinetic angles.")
            elif self.mode == "wave_current":
                a.legend.setText(f"Visible current {float(self.values.sum()):.7g} pA · no flux renormalisation")
        except (ValueError, MemoryError) as error:
            owner.plot.clear()
            a.table.setRowCount(0)
            owner.summary.setText(f"Wave view unavailable: {error}")
        finally:
            a._drawing = False

    def mouse_moved(self, position):
        if self._hover is None or not self.owner.plot.sceneBoundingRect().contains(position):
            return
        point = self.owner.plot.getViewBox().mapSceneToView(position)
        values, xe, ye = self._hover
        ix, iy = np.searchsorted(xe, point.x(), side="right")-1, np.searchsorted(ye, point.y(), side="right")-1
        if 0 <= ix < values.shape[0] and 0 <= iy < values.shape[1]:
            unit = "mrad" if self.mode == "wave_angles" else "µm"
            self.analysis.readout.setText(f"U {point.x():.6g}, V {point.y():.6g} {unit} | {values[ix, iy]:.6g} pA / bin")
        else:
            self.analysis.readout.clear()
