"""Private executed-prefix reuse during detached lens-preset fitting.

Only excitation percentages of declared lenses may vary. The prefix is made
by executing this owned state's physical gun and upstream column; callers
cannot supply a gun-exit or specimen-plane source.
"""
import math
import numpy as np

from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.surface_probe_focus import (
    SurfaceFocusMeasurement, local_waist_from_radii, surface_z_mm, measure_surface_focus,
)


class IlluminationCheckpoint:
    def __init__(self, state, keys, *, step_mm=.05):
        from temsim.physics.core import FIELD_SIGMA_CUTOFF, propagate, active_mapped_providers
        from temsim.optics.direct_alignment import _pre_sample_kick_events
        from temsim.optics.electron_gun.source import trace_source_to_exit
        from temsim.physics.aperture_clipping import clip_segment
        from temsim.physics.column_wall import clip_column_wall
        from temsim.physics.beam_current import effective_source_current_a
        if (state.electron_gun.source_representation != "classical_particles"
                or state.vacuum_map.enabled or effective_source_current_a(state) <= 0):
            raise ValueError("Illumination calibration requires a nonzero classical source and vacuum participation off")
        self._state = capture_instrument_snapshot(state).restore()
        self.keys, self.step_mm = tuple(keys), float(step_mm)
        if not self.keys or not math.isfinite(self.step_mm) or self.step_mm <= 0:
            raise ValueError("Declare varied lens keys and a positive finite step")
        self._lenses = {l.key: l for l in self._state.lenses}
        if any(k not in self._lenses or not self._lenses[k].enabled for k in self.keys):
            raise ValueError("Only installed and enabled lenses can use a fitting checkpoint")
        start = float(self._state.electron_gun.exit_plane_z_mm)
        self._prefix = None
        # Mapped supports and excitation-dependent models need their own
        # dependency proof; use ordinary full transport until qualified.
        if active_mapped_providers(self._state) or any(not hasattr(self._lenses[k], "field_support_mm") for k in self.keys):
            return
        boundary = min(self._lenses[k].field_support_mm(FIELD_SIGMA_CUTOFF)[0] for k in self.keys)-1e-7
        if not start < boundary < surface_z_mm(self._state)-.0002:
            return
        self._gun = trace_source_to_exit(self._state)
        e = self._gun.exit_bundle
        apertures = tuple(a.z_mm for a in self._state.apertures if start < a.z_mm <= boundary)
        z, x, tx, y, ty, cp = propagate(self._state, start, boundary,
            e.x_m, e.tx_rad, e.y_m, e.ty_rad, energy_offset_ev=e.energy_offset_ev,
            events=_pre_sample_kick_events(self._state), save_z_mm=apertures+(boundary,),
            checkpoint_z_mm=(boundary,), return_checkpoints=True, maximum_step_mm=self.step_mm)
        alive, stops, labels = clip_segment(self._state, z, x, y, e.alive.copy(),
            self._gun.blocked_z_mm.copy(), list(self._gun.blocked_key))
        alive, stops, labels = clip_column_wall(self._state, z, x, y, alive, stops, labels)
        if len(cp.z_mm) != 1 or abs(cp.z_mm[0]-boundary) > 1e-10:
            raise ValueError("Missing exact upstream fitting checkpoint")
        self._prefix = (boundary, cp.x_m[0].copy(), cp.tx_rad[0].copy(), cp.y_m[0].copy(),
                        cp.ty_rad[0].copy(), alive.copy(), stops.copy(), tuple(labels))

    @property
    def prefix_z_mm(self):
        return None if self._prefix is None else self._prefix[0]

    def measure(self, values):
        from temsim.physics.core import propagate
        from temsim.optics.direct_alignment import _pre_sample_kick_events
        from temsim.physics.aperture_clipping import clip_segment
        from temsim.physics.column_wall import clip_column_wall
        from temsim.physics.beam_statistics import transverse_beam_statistics
        values = np.asarray(values, float)
        if values.shape != (len(self.keys),) or not np.all(np.isfinite(values)):
            raise ValueError("Finite excitation values must match the declared lenses")
        for key, value in zip(self.keys, values):
            if not 0 <= value <= self._lenses[key].max_percent:
                raise ValueError("Excitation lies outside the configured lens range")
            self._lenses[key].percent = float(value)
        if self._prefix is None:
            return measure_surface_focus(self._state, step_mm=self.step_mm)
        state = self._state
        start, x0, tx0, y0, ty0, alive0, stops0, labels0 = self._prefix
        end = surface_z_mm(state)
        h_mm = .0001
        planes = (end-2*h_mm, end-h_mm, end)
        e = self._gun.exit_bundle
        apertures = tuple(a.z_mm for a in state.apertures if start < a.z_mm < end)
        z, x, tx, y, ty, cp = propagate(state, start, end, x0, tx0, y0, ty0,
            energy_offset_ev=e.energy_offset_ev, events=_pre_sample_kick_events(state),
            save_z_mm=apertures+planes, checkpoint_z_mm=planes, return_checkpoints=True,
            maximum_step_mm=self.step_mm)
        alive, stops, labels = clip_segment(state, z, x, y, alive0.copy(), stops0.copy(), list(labels0))
        alive, _, _ = clip_column_wall(state, z, x, y, alive, stops, labels)
        alive &= e.weight > 0
        if len(cp.z_mm) != 3 or not np.allclose(cp.z_mm, planes, rtol=0, atol=1e-10):
            raise ValueError("Missing exact surface fitting checkpoints")
        rows = []
        for i in range(3):
            vectors = (cp.x_m[i], cp.y_m[i], cp.tx_rad[i], cp.ty_rad[i])
            if any(np.any(~np.isfinite(v[alive])) for v in vectors):
                raise ValueError("Nonfinite current-carrying fitting trajectory")
            rows.append(transverse_beam_statistics(*vectors, alive=alive, weights=e.weight))
        radii = tuple(row.radius_rms_m for row in rows)
        offset, second = local_waist_from_radii(radii, h_mm*1e-3)
        w = e.weight[alive]
        effective = float(w.sum()**2/(w@w)) if w.size and w@w > 0 else 0.
        return SurfaceFocusMeasurement(end, rows[-1], offset, second, radii, self.step_mm, effective)
