"""Stream physical raster/dwell samples through the tip-origin wave pipeline."""
from dataclasses import dataclass, replace
import math

import numpy as np

from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.physics.wave_execution import ScanWaveNumerics
from temsim.physics.tip_wave_pipeline import TipWaveRequest, simulate_tip_wave


@dataclass(frozen=True)
class ScanWaveSample:
    row: int
    column: int
    dwell_index: int
    tip_time_s: float
    dwell_weight: float
    calculation: object


class ScanResponseAccumulator:
    """Small scalar raster only; never retain waves or mix conditional phases."""

    def __init__(self):
        self._values = {}

    def add(self, sample):
        key = (sample.row, sample.column)
        entry = self._values.setdefault(key, [0., 0., set()])
        if sample.dwell_index in entry[2]:
            raise ValueError("A physical dwell sample cannot be counted twice")
        entry[2].add(sample.dwell_index)
        detector = sample.calculation.detector
        response = None if detector is None else detector.record.get("response_weight")
        if response is not None:
            entry[0] += float(response)*sample.dwell_weight
            entry[1] += sample.dwell_weight

    def arrays(self):
        rows = sorted({r for r, _ in self._values})
        columns = sorted({c for _, c in self._values})
        ri, ci = {r: i for i, r in enumerate(rows)}, {c: i for i, c in enumerate(columns)}
        response = np.full((len(rows), len(columns)), np.nan)
        coverage = np.zeros_like(response)
        for (r, c), (value, weight, _) in self._values.items():
            coverage[ri[r], ci[c]] = weight
            if math.isclose(weight, 1., rel_tol=0., abs_tol=1e-10):
                response[ri[r], ci[c]] = value
        return {"physical_rows": np.asarray(rows, dtype=np.int64),
                "physical_columns": np.asarray(columns, dtype=np.int64),
                "response_per_tip_electron": response, "dwell_coverage": coverage}


def physical_scan_samples(state, numerics=ScanWaveNumerics()):
    numerics.validate()
    active = [d for d in (*state.deflectors, *getattr(state, "corrector_elements", ())) if getattr(d, "enabled", False) and getattr(d, "scan_enabled", False)
              and hasattr(d, "scan_pixels_x") and "descan" not in d.key.lower()]
    if len(active) != 1:
        raise ValueError("Select exactly one installed active raster scan controller")
    scan = active[0]
    scan.validate()
    nx, ny, period = int(scan.scan_pixels_x), int(scan.scan_lines), float(scan.scan_frame_period_s)
    count = len(range(0, nx, numerics.stride))*len(range(0, ny, numerics.stride))
    if count > numerics.maximum_positions:
        raise ValueError(f"Physical scan requests {count} positions, above maximum_positions={numerics.maximum_positions}; adjust the numerical stride/budget")
    frame_start = (math.ceil(float(getattr(state, "simulation_time_s", 0.))/period)+numerics.frame_index)*period
    for row in range(0, ny, numerics.stride):
        for column in range(0, nx, numerics.stride):
            for dwell in range(numerics.dwell_samples):
                time = frame_start+(row+(column+(dwell+.5)/numerics.dwell_samples)/nx)/ny*period
                yield row, column, dwell, time, 1./numerics.dwell_samples


def simulate_tip_scan(state, request=TipWaveRequest(), numerics=ScanWaveNumerics(), *,
                      cancelled=lambda: False, progress_callback=None, use_cache=True):
    """Yield one result at a time; consumers save/reduce it before advancing.

Times come from the installed raster, not a shifted specimen-plane pupil.
The column evaluates each coil at its energy-dependent arrival time. Dwell
samples are incoherent exposures; retain phases per sample, never average them.
The inelastic RNG uses common random numbers across dwell/scan positions;
per-position errors cannot be combined as independent scan noise.
"""
    snapshot = capture_instrument_snapshot(state)
    working = snapshot.restore()
    for row, column, dwell, time, weight in physical_scan_samples(working, numerics):
        if cancelled():
            raise InterruptedError("Physical wave scan cancelled")
        result = simulate_tip_wave(working, replace(request, tip_time_s=time), cancelled=cancelled,
                                  progress_callback=progress_callback, use_cache=use_cache)
        snapshot.restore()
        yield ScanWaveSample(row, column, dwell, time, weight, result)
        del result
