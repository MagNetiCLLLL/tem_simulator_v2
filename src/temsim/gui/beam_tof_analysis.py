"""Per-path classical arrival-time view of completed transport only."""
import numpy as np
import pyqtgraph as pg

from temsim.gui.transverse_projection import transverse_view_coordinates


class BeamTimeOfFlight:
    def __init__(self, analysis):
        self.analysis = analysis
        self.owner = analysis.owner
        self.reference_s = None
        self.delay_s = np.empty(0)

    def draw(self, data):
        a, owner = self.analysis, self.owner
        a._source_plot_key = None  # Returning to a source preset restores its full population.
        u, v = transverse_view_coordinates(data.x_m, data.y_m, owner._projection_angle_deg)
        tu, tv = transverse_view_coordinates(data.tx, data.ty, owner._projection_angle_deg)
        au, av = np.arctan(tu)*1e3, np.arctan(tv)*1e3
        x, y = ((au, av) if a.mode == "angular" else (u*1e6, au) if a.mode == "phase_u"
                else (v*1e6, av) if a.mode == "phase_v" else (u*1e6, v*1e6))
        pool = np.linspace(0, data.total_column_count-1,
                           min(data.total_column_count, owner.MAX_DISPLAY_RAYS), dtype=int)
        selected = np.isin(data.column_index, pool) & np.isfinite(x) & np.isfinite(y)
        times = (np.asarray(data.flight_time_s, dtype=np.float64) if data.flight_time_s is not None
                 else np.full(len(data.x_m), np.nan))
        known = np.isfinite(times) & (times >= 0)
        self.reference_s = float(np.min(times[known])) if np.any(known) else None
        self.delay_s = times - self.reference_s if self.reference_s is not None else np.full(times.shape, np.nan)
        maximum = float(np.max(self.delay_s[known])) if np.any(known) else 0.
        cmap = pg.colormap.get("viridis")
        brushes = [pg.mkBrush(cmap.mapToQColor(float(value/maximum if maximum > 0 else 0)))
                   if np.isfinite(value) else pg.mkBrush("#94a3b8") for value in self.delay_s[selected]]
        indices = np.flatnonzero(selected)
        records = [{"source_ray_id": int(data.source_ray_id[i]), "column_index": int(data.column_index[i]),
                    "flight_time_s": float(times[i]), "delay_s": float(self.delay_s[i])} for i in indices]
        def hover(_x, _y, record):
            prefix = f"Source {record['source_ray_id']} | path {record['column_index']}"
            if not np.isfinite(record["flight_time_s"]):
                return prefix + "\nFlight time unavailable for this path"
            return (prefix + f"\nFlight time {record['flight_time_s']*1e9:.12g} ns"
                    + f" | Delay {record['delay_s']*1e15:.8g} fs")
        owner._scatter = pg.ScatterPlotItem(x=x[selected], y=y[selected], size=5,
            brush=brushes, pen=pg.mkPen(None), data=records, hoverable=True, tip=hover)
        owner._display_source_ids = data.source_ray_id[selected]
        owner.plot.addItem(owner._scatter)
        owner.plot.addLine(x=0, pen=pg.mkPen("#94a3b8", width=.8))
        owner.plot.addLine(y=0, pen=pg.mkPen("#94a3b8", width=.8))
        a._set_ranges(a._ranges.get(a.mode, a._point_bounds(x, y)))
        unit, scale = ("fs", 1e15) if maximum < 1e-12 else ("ps", 1e12) if maximum < 1e-9 else ("ns", 1e9)
        a.legend.setVisible(True)
        a.legend.setText(
            f'<span style="color:#440154">■</span> 0 — '
            f'<span style="color:#fde725">■</span> {maximum*scale:.6g} {unit} delay'
            if self.reference_s is not None else "Arrival times unavailable in this result")
        a.legend.setToolTip(
            "Delay relative to the earliest timed arrival in the saved plane population, before display subsampling. "
            "Grey: incomplete clock. Interpolation reads saved integration clocks, not drawing-path lengths. "
            "This is classical laboratory time, not quantum phase.")
        a._summary(data, f" | {np.count_nonzero(known):,}/{len(times):,} paths timed")
        if data.status == "Unavailable":
            a.readout.setText(" ".join(data.diagnostics))
        if self.reference_s is not None:
            owner.summary.setToolTip(owner.summary.toolTip() +
                f" Earliest timed arrival: {self.reference_s*1e9:.12g} ns since simultaneous tip emission.")
        source = a.source_data()
        if source is None or not len(indices):
            owner.source_plot.clear("No arriving source paths at this plane")
            return
        rows = source.indices_for(data.source_ray_id[selected])
        positions = np.full((len(rows), 3), np.nan)
        found = rows >= 0
        positions[found] = source.position_m[rows[found]]
        if not np.any(np.all(np.isfinite(positions), axis=1)):
            owner.source_plot.clear("Original emission positions are unavailable in this result.")
            return
        owner.source_plot.set_projection_angle(owner._projection_angle_deg)
        owner.source_plot.set_source(ids=data.source_ray_id[selected], position_m=positions, brushes=brushes,
            azimuth_rad=source.values(data.source_ray_id[selected], "emission_direction"),
            angle_to_normal_rad=source.values(data.source_ray_id[selected], "emission_angle"),
            flight_time_s=times[selected], reference_time_s=self.reference_s,
            status="Arriving paths at this plane. Grey: time unknown; non-arrivals have no arrival time.")
        owner.source_plot.summary.setToolTip(
            "Each displayed downstream path retains its own arrival time, including paths with the same source ID. "
            "Only arriving display paths are shown here; this is not the full emitted population. "
            "Non-arrivals are omitted rather than assigned zero delay.")
