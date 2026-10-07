"""Read-only diagnostics of one executed forward-column wave checkpoint.

Heavy derivation runs through the existing numerical job coordinator. Qt only
presents the bounded output for the current selection. No source, propagation,
aggregate mixture phase or individually measured particle path is invented.
"""
from dataclasses import dataclass
import math
from threading import Event

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QRectF, QRunnable, Signal, Slot
from PySide6.QtGui import QTransform
from PySide6.QtWidgets import QTableWidgetItem

from temsim.gui.job_coordinator import CoordinatedPool, ResourceClaim
from temsim.physics.wave_plane_observables import (
    canonical_angular_spectrum, column_mode_observables, mode_phase_samples,
    interaction_weights,
)


def _rotation(angle_deg):
    phi = math.radians(angle_deg)
    return np.array(((math.cos(phi), math.sin(phi)), (-math.sin(phi), math.cos(phi))))


def _lattice(mode, angular):
    wave = mode.plane
    if not angular:
        return wave.basis_m*1e6, wave.origin_m*1e6
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
    ny, nx = wave.amplitude.shape
    basis = wavelength_m(mode.energy_kev*1000)*1e3*np.linalg.inv(wave.basis_m).T@np.diag((1/nx, 1/ny))
    return basis, np.zeros(2)


def _view_bounds(modes, rotation, angular, check_cancelled):
    half = np.full(2, 1e-12)
    if hasattr(modes, "geometries"):
        def lattices():
            # Stored metadata and the small checked affine arrays suffice;
            # do not load every amplitude once just to find display bounds.
            from temsim.optics.electron_gun.tip_coherence import wavelength_m
            for (shape, basis, origin), row in zip(modes.geometries(), modes.rows):
                if angular:
                    ny, nx = shape
                    basis = wavelength_m(row["energy_kev"]*1000)*1e3*np.linalg.inv(basis).T@np.diag((1/nx, 1/ny))
                    origin = np.zeros(2)
                else:
                    basis, origin = basis*1e6, origin*1e6
                yield shape, basis, origin
        geometry = lattices()
    else:
        geometry = ((mode.plane.amplitude.shape, *_lattice(mode, angular)) for mode in modes)
    for (ny, nx), basis, origin in geometry:
        check_cancelled()
        indices = np.array(((-(nx//2)-.5, (nx-1)//2+.5, (nx-1)//2+.5, -(nx//2)-.5),
                            (-(ny//2)-.5, -(ny//2)-.5, (ny-1)//2+.5, (ny-1)//2+.5)))
        xy = rotation@(origin[:, None]+basis@indices)
        half = np.maximum(half, np.max(abs(xy), axis=1))
    return tuple((-1.08*h, 1.08*h) for h in half)


class _ReadoutCancelled(Exception):
    pass


@dataclass(frozen=True)
class _ViewRequest:
    checkpoint: object
    mode: str
    mode_index: int | None
    axial_bz_t: float | None
    maximum_working_bytes: int
    angle_deg: float
    bounds: tuple | None
    bins: int
    flow_bins: int
    magnetic_gauge: object = None


def _derive_view(request, cancel_event):
    """Return arrays and metadata only; this function must never touch Qt."""
    from temsim.physics.affine_cell_histogram import affine_cell_histogram
    from temsim.physics.wave_execution import check_available_memory

    def check_cancelled():
        if cancel_event.is_set():
            raise _ReadoutCancelled()

    check_cancelled()
    checkpoint = request.checkpoint
    if request.mode == "wave_flow" and request.axial_bz_t is None and request.magnetic_gauge is None:
        raise ValueError("Probability flow needs the recorded axial magnetic field at this plane")
    modes = checkpoint.beam.modes
    rows = getattr(modes, "rows", None)
    load_reserve = 0
    if rows is not None:
        selected_rows = rows if request.mode_index is None or request.mode == "wave_interactions" else (rows[request.mode_index],)
        load_reserve = max(2*sum(item["nbytes"] for item in row["arrays"].values()) for row in selected_rows)
        if load_reserve > request.maximum_working_bytes:
            raise MemoryError("Wave display exceeds its working-memory budget before loading the stored mode")
        check_available_memory(load_reserve)
    if request.mode == "wave_interactions":
        return {"rows": interaction_weights(checkpoint)}
    if request.mode_index is not None:
        modes = (modes[request.mode_index],)
    rotation = _rotation(request.angle_deg)
    angular = request.mode == "wave_angles"
    bounds = request.bounds or _view_bounds(modes, rotation, angular, check_cancelled)
    if request.mode == "wave_phase":
        mode, = modes  # There is deliberately no aggregate mixture phase.
        phase = mode_phase_samples(mode, maximum_working_bytes=request.maximum_working_bytes-load_reserve)
        check_cancelled()
        ny, nx = mode.plane.amplitude.shape
        basis = rotation@mode.plane.basis_m*1e6
        origin = rotation@mode.plane.origin_m*1e6-basis@np.array((nx//2+.5, ny//2+.5))
        return {"values": phase, "bounds": bounds, "phase_basis": basis,
                "phase_origin": origin, "phase_reference": str(mode.axial_reference)}

    overhead = 8*(request.bins**2+6*request.flow_bins**2)
    working = request.maximum_working_bytes-overhead-load_reserve
    values = np.zeros((request.bins, request.bins))
    flow = np.zeros((2, request.flow_bins, request.flow_bins)) if request.mode == "wave_flow" else None
    records = []
    for mode in modes:
        check_cancelled()
        bytes_per_cell = 256 if flow is not None else 192 if angular else 96
        needed = bytes_per_cell*mode.plane.amplitude.size
        if type(request.maximum_working_bytes) is not int or working < needed:
            raise MemoryError("Wave display exceeds its working-memory budget")
        check_available_memory(needed+overhead)
        basis, origin = _lattice(mode, angular)
        observable = None
        if flow is not None:
            observable = column_mode_observables(mode, reference_current_a=checkpoint.reference_current_a,
                axial_bz_t=request.axial_bz_t, magnetic_gauge=request.magnetic_gauge,
                plane_z_mm=checkpoint.plane_z_mm, maximum_working_bytes=working)
            weight = observable.cell_probability_per_tip_electron
        elif angular:
            spectrum = canonical_angular_spectrum(mode, maximum_working_bytes=working)
            weight = spectrum["probability_per_tip_electron"]
        else:
            weight = mode.weight_per_reference_electron*abs(mode.plane.amplitude)**2
        check_cancelled()
        contribution, record = affine_cell_histogram(weight, rotation@basis, rotation@origin, bounds, request.bins)
        values += contribution*checkpoint.reference_current_a*1e12
        records.append(record)
        if observable is not None:
            area = abs(float(np.linalg.det(mode.plane.basis_m)))
            for axis in range(2):
                check_cancelled()
                # Signed currents, including the magnetic vector potential,
                # are rotated and integrated before computing a mean slope.
                weights = area*1e12*(rotation[axis, 0]*observable.transverse_current_density_a_per_m2[0]
                                    +rotation[axis, 1]*observable.transverse_current_density_a_per_m2[1])
                component, record = affine_cell_histogram(weights, rotation@basis, rotation@origin,
                                                          bounds, request.flow_bins)
                flow[axis] += component
                records.append(record)
            del observable
        # Do not retain one full diagnostic array per incoherent mode.
        del weight
        if angular:
            del spectrum
    check_cancelled()
    result = {"values": values, "bounds": bounds, "display_records": tuple(records)}
    if flow is not None:
        ratio = request.bins//request.flow_bins
        axial = values.reshape(request.flow_bins, ratio, request.flow_bins, ratio).sum(axis=(1, 3))
        slopes = np.full_like(flow, np.nan)
        np.divide(flow*1e3, axial[None, :, :], out=slopes, where=axial[None, :, :] > 0)
        result.update(flow_current_pA=flow, flow_axial_pA=axial, flow_slopes_mrad=slopes)
    return result


class _ViewSignals(QObject):
    ready = Signal(int, object)
    failed = Signal(int, str)
    finished = Signal(int)


class _ViewWorker(QRunnable):
    def __init__(self, generation, request):
        super().__init__()
        self.generation, self.request = generation, request
        self.cancel_event = Event()
        self.resource_claim = ResourceClaim(max(0, request.maximum_working_bytes)+4*1024**2)
        self.job_input_identity = f"wave-readout:{id(request.checkpoint)}:{request.mode}:{generation}"
        self.signals = _ViewSignals()

    def run(self):
        try:
            result = _derive_view(self.request, self.cancel_event)
            if not self.cancel_event.is_set():
                self.signals.ready.emit(self.generation, result)
        except _ReadoutCancelled:
            pass
        except Exception as error:
            self.signals.failed.emit(self.generation, str(error))
        finally:
            self.signals.finished.emit(self.generation)


class WaveBeamAnalysis(QObject):
    MODES = (("Beam intensity / current", "wave_current"),
             ("Probability flow direction", "wave_flow"),
             ("Canonical angular distribution", "wave_angles"),
             ("Phase — one mode", "wave_phase"), ("Interaction breakdown", "wave_interactions"))
    BINS = 128
    FLOW_BINS = 16

    def __init__(self, analysis, checkpoint, axial_bz_t, maximum_working_bytes, *, magnetic_gauge=None):
        super().__init__(analysis.owner)
        if axial_bz_t is not None and not math.isfinite(axial_bz_t):
            raise ValueError("Wave view needs the actual axial magnetic field")
        self.analysis, self.owner = analysis, analysis.owner
        self.checkpoint, self.bz = checkpoint, None if axial_bz_t is None else float(axial_bz_t)
        self.magnetic_gauge = magnetic_gauge
        self.maximum_working_bytes = maximum_working_bytes
        self.saved_mode, self.saved_colour = analysis.mode, analysis.colour_combo.currentData()
        self.ranges = {}
        self.mode = "wave_current"
        self._hover = None
        self.values = None
        self.flow_slopes_mrad = None
        self.flow_current_pA = None
        self.display_records = ()
        self._active = True
        self._generation = 0
        self._pending_key = self._cached_key = None
        self._data = None
        self.busy = False
        self.pool = CoordinatedPool(self)
        self.pool.coordinator.register_retained(self, "_retained_roots")
        self.pool.coordinator.changed.connect(self._dispose_when_idle)
        pool = self.pool
        self.destroyed.connect(lambda: pool.clear())

    def _retained_roots(self):
        return self.checkpoint, self._data

    def activate(self):
        a = self.analysis
        for combo in (a.mode_combo, a.colour_combo):
            combo.blockSignals(True)
            combo.clear()
        for label, key in self.MODES:
            a.mode_combo.addItem(label, key)
        a.colour_combo.addItem("All modes (intensity/current sum)", None)
        modes = self.checkpoint.beam.modes
        rows = getattr(modes, "rows", None)
        metadata = ((row["mode_id"], row["energy_kev"]) for row in rows) if rows is not None else (
            (mode.mode_id, mode.energy_kev) for mode in modes)
        for i, (mode_id, energy_kev) in enumerate(metadata):
            a.colour_combo.addItem(f"{mode_id} | {energy_kev:.7g} keV", i)
        for combo in (a.mode_combo, a.colour_combo):
            combo.blockSignals(False)
        a.mode_combo.setToolTip("Read this executed wave checkpoint. No propagation or source changes.")
        a.colour_label.setText("Mode")
        a.colour_combo.setToolTip("Independent modes add intensities and currents, never phases. Phase selects one mode.")
        self.owner.initial_beam_panel.hide()
        a.mode_combo.show()
        a.view_label.setText("View")
        a._hover_payload = None
        a._resize_timer.stop()
        self.owner.plot.setAspectLocked(True)
        self.redraw()

    def deactivate(self):
        self._active = False
        # A replacement checkpoint can arrive while FFT or disk loading is
        # unwinding. The coordinator, not a closing view, owns this retired
        # receiver and its pool until Qt confirms worker return.
        self.setParent(self.pool.coordinator)
        self._cancel_pending()
        self._data = None
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
        self._dispose_when_idle()

    @Slot()
    def _dispose_when_idle(self):
        if not self._active and not self.pool.coordinator.has_owner(self.pool):
            self.deleteLater()

    def _cancel_pending(self):
        self._generation += 1
        self._pending_key = None
        self.busy = False
        self.pool.clear()

    def selection_changed(self):
        self.mode = self.analysis.mode_combo.currentData()
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

    def _request_key(self):
        bounds = self.ranges.get(self.mode)
        if bounds is not None:
            bounds = tuple(tuple(map(float, axis)) for axis in bounds)
        return self.mode, self.analysis.colour_combo.currentData(), self.owner._projection_angle_deg, bounds

    def _submit(self, key):
        self._cancel_pending()
        self._pending_key, self.busy = key, True
        self._cached_key, self._data = None, None
        request = _ViewRequest(self.checkpoint, key[0], key[1], self.bz,
            self.maximum_working_bytes, key[2], key[3], self.BINS, self.FLOW_BINS,
            self.magnetic_gauge)
        worker = _ViewWorker(self._generation, request)
        worker.signals.ready.connect(self._ready)
        worker.signals.failed.connect(self._failed)
        worker.signals.finished.connect(self._finished)
        self.pool.start(worker)

    @Slot(int, object)
    def _ready(self, generation, result):
        if not self._active or generation != self._generation:
            return
        self._cached_key, self._data = self._pending_key, result
        self._pending_key, self.busy = None, False
        self.redraw()

    @Slot(int, str)
    def _failed(self, generation, message):
        if not self._active or generation != self._generation:
            return
        self._ready(generation, {"error": message})

    @Slot(int)
    def _finished(self, generation):
        if generation == self._generation:
            self.busy = False
            if self._cached_key is None:
                self._pending_key = None

    def _draw_image(self, data):
        values, bounds = data["values"], data["bounds"]
        image = pg.ImageItem(values.T, axisOrder="row-major")
        image.setLevels((0., max(float(values.max()), np.finfo(float).tiny)))
        image.setRect(QRectF(bounds[0][0], bounds[1][0], bounds[0][1]-bounds[0][0], bounds[1][1]-bounds[1][0]))
        self.owner.plot.addItem(image)
        self._hover = (values, np.linspace(*bounds[0], values.shape[0]+1), np.linspace(*bounds[1], values.shape[1]+1))

    def _draw_phase(self, data):
        image = pg.ImageItem(data["values"], axisOrder="row-major")
        image.setAutoDownsample(False)  # Averaging wrapped phase is not a phase readout.
        image.setLookupTable(pg.colormap.get("CET-C2").getLookupTable())
        image.setLevels((-np.pi, np.pi))
        basis, origin = data["phase_basis"], data["phase_origin"]
        image.setTransform(QTransform(basis[0, 0], basis[1, 0], basis[0, 1], basis[1, 1], *origin))
        self.owner.plot.addItem(image)

    def _draw_flow(self, data):
        slopes, bounds = data["flow_slopes_mrad"], data["bounds"]
        self.flow_slopes_mrad, self.flow_current_pA = slopes, data["flow_current_pA"]
        magnitude = np.hypot(*slopes)
        valid = np.isfinite(magnitude) & (magnitude > 0)
        maximum = float(magnitude[valid].max()) if valid.any() else 0.
        if maximum > 0:
            spacing = (np.asarray(bounds)[:, 1]-np.asarray(bounds)[:, 0])/self.FLOW_BINS
            positions = np.meshgrid(*(np.linspace(*axis, self.FLOW_BINS, endpoint=False)+step/2
                                      for axis, step in zip(bounds, spacing)), indexing="ij")
            centres = np.stack(positions)[:, valid]
            vectors = slopes[:, valid]*(.7*float(spacing.min())/maximum)
            start, end = centres-vectors/2, centres+vectors/2
            normal = np.stack((-vectors[1], vectors[0]))
            head1, head2 = end-.25*vectors+.15*normal, end-.25*vectors-.15*normal
            vertices = np.stack((start, end, head1, end, head2, end), axis=2)
            self.owner.plot.addItem(pg.PlotCurveItem(*vertices.reshape(2, -1), connect="pairs",
                pen=pg.mkPen("#49dcff", width=1.4)))
        self.analysis.legend.setText(
            f"Probability flow J⊥/Jz · longest arrow {maximum:.6g} mrad · not measured particle paths")
        self.analysis.legend.setToolTip(
            "Kinetic probability-current direction, including the recorded magnetic vector potential. "
            "Currents are summed over the selected modes and each display cell before division by axial current. "
            "Arrow length uses a common display scale; it is not a propagated distance or trajectory. "
            "Background: forward current per display bin.")

    def _draw_interactions(self, rows):
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
        if not self._active:
            return
        a, owner = self.analysis, self.owner
        a._drawing = True
        self._hover = None
        self.values = self.flow_slopes_mrad = self.flow_current_pA = None
        try:
            owner.plot.clear()
            owner._scatter = None
            a.readout.clear()
            a.legend.show()
            a.readout.show()
            interactions = self.mode == "wave_interactions"
            owner.plot.setVisible(not interactions)
            a.table.setVisible(interactions)
            a.table.setRowCount(0)
            a.colour_combo.setVisible(not interactions)
            a.colour_label.setVisible(not interactions)
            owner.fit_beam.setEnabled(not interactions)
            owner.fit_beam.setToolTip("Fit the executed wave readout. No propagation.")
            owner.initial_beam_panel.hide()
            unit = "mrad" if self.mode == "wave_angles" else "µm"
            for axis, label in (("bottom", "U"), ("left", "V")):
                owner.plot.getAxis(axis).setLabel(label, units=unit, siPrefixEnableRanges=())
            owner.heading.setText(f"Wave section | Z {self.checkpoint.plane_z_mm:.9g} mm")
            a.legend.setText("Executed wave checkpoint · not detector counts")
            a.legend.setToolTip("Readout of the executed checkpoint; absolute tip weights are retained. No propagation.")
            if not math.isclose(owner._plane_z_mm, self.checkpoint.plane_z_mm, rel_tol=0., abs_tol=1e-12):
                self._cancel_pending()
                owner.summary.setText(f"Z {owner._plane_z_mm:.9g} mm is not cached. No wave interpolation or propagation was performed.")
                return
            key = self._request_key()
            if self._cached_key != key:
                if self._pending_key != key:
                    self._submit(key)
                owner.summary.setText("Preparing wave readout from the executed checkpoint…")
                return
            data = self._data
            if "error" in data:
                owner.summary.setText(f"Wave view unavailable: {data['error']}")
                return
            if interactions:
                self._draw_interactions(data["rows"])
            else:
                self.values = data["values"]
                self.display_records = data.get("display_records", ())
                if self.mode == "wave_phase":
                    self._draw_phase(data)
                else:
                    self._draw_image(data)
                if self.mode == "wave_flow":
                    self._draw_flow(data)
                bounds = data["bounds"]
                owner.plot.setRange(xRange=bounds[0], yRange=bounds[1], padding=0., disableAutoRange=True)
            current = 1e12*self.checkpoint.transmitted_current_a
            owner.summary.setText(f"Total forward current {current:.7g} pA | {len(self.checkpoint.beam.modes)} modes")
            if self.mode == "wave_phase":
                a.legend.setText("Single-mode phase (rad) · no mixture phase")
                a.legend.setToolTip(data["phase_reference"])
            elif self.mode == "wave_angles":
                a.legend.setText("Canonical angles | recorded posed-lens gauge"
                    if self.magnetic_gauge is not None and self.magnetic_gauge.fields
                    else "Canonical angles | Bz not recorded" if self.bz is None
                    else f"Canonical angles | Bz {self.bz:.6g} T")
                a.legend.setToolTip("Full complex phase, in the inherited laboratory gauge. Inside a magnetic field these are not kinetic angles.")
            elif self.mode == "wave_current":
                a.legend.setText(f"Visible current {float(self.values.sum()):.7g} pA · no flux renormalisation")
        finally:
            a._drawing = False
            owner.hardware.redraw()

    def mouse_moved(self, position):
        if self._hover is None or not self.owner.plot.sceneBoundingRect().contains(position):
            return
        point = self.owner.plot.getViewBox().mapSceneToView(position)
        values, xe, ye = self._hover
        ix, iy = np.searchsorted(xe, point.x(), side="right")-1, np.searchsorted(ye, point.y(), side="right")-1
        if 0 <= ix < values.shape[0] and 0 <= iy < values.shape[1]:
            unit = "mrad" if self.mode == "wave_angles" else "µm"
            text = f"U {point.x():.6g}, V {point.y():.6g} {unit} | {values[ix, iy]:.6g} pA / bin"
            if self.flow_slopes_mrad is not None:
                ratio = self.BINS//self.FLOW_BINS
                u, v = self.flow_slopes_mrad[:, ix//ratio, iy//ratio]
                text += f" | mean flow ({u:.6g}, {v:.6g}) mrad" if np.isfinite((u, v)).all() else " | flow undefined at zero current"
            self.analysis.readout.setText(text)
        else:
            self.analysis.readout.clear()
