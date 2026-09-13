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
    """Dwell-weighted detector rasters (channel, row, column), without waves.

    Missing intensity/positions stay NaN. Channels share a single executed
    dwell; neither channel probabilities nor incoherent phases are summed.
    """

    def __init__(self):
        self._values = {}
        self._detector_keys = None
        self._instrument_digest = None

    def add(self, sample):
        for name in ("row", "column", "dwell_index"):
            value = getattr(sample, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"Scan {name} must be a non-negative integer")
        if (isinstance(sample.dwell_weight, bool) or not math.isfinite(sample.dwell_weight)
                or not 0 < sample.dwell_weight <= 1 or not math.isfinite(sample.tip_time_s)):
            raise ValueError("Scan dwell weight/time must be finite with weight in (0, 1]")
        key = (sample.row, sample.column)
        previous = self._values.get(key)
        if previous is not None and sample.dwell_index in previous[2]:
            raise ValueError("A physical dwell sample cannot be counted twice")
        if previous is not None and previous[0]+sample.dwell_weight > 1+1e-10:
            raise ValueError("Physical dwell weights exceed one exposure")
        calculation = sample.calculation
        readouts = tuple(getattr(calculation, "detector_readouts", ()))
        if not readouts and calculation.detector is not None:
            readouts = (calculation.detector,)
        keys = tuple(str(readout.record.get("detector", "legacy_single_detector")) for readout in readouts)
        if len(set(keys)) != len(keys):
            raise ValueError("A detector readout cannot be counted twice in a dwell")
        if self._detector_keys is not None and keys != self._detector_keys:
            raise ValueError("Detector channels changed within the scan")
        identity = getattr(calculation, "instrument_digest", None)
        if self._values and identity != self._instrument_digest:
            raise ValueError("Instrument inputs changed within the scan")
        responses = [readout.record.get("response_weight") for readout in readouts]
        if any(value is not None and (isinstance(value, bool) or not math.isfinite(value) or value < 0)
               for value in responses):
            raise ValueError("Detector response must be finite and non-negative, or absent")
        # Validate the entire dwell before publishing any channel. A failed
        # channel must not partially increment its neighbours or consume its ID.
        entry = previous if previous is not None else [0., {}, set()]
        for detector_key, response in zip(keys, responses):
            channel = entry[1].setdefault(detector_key, [0., 0.])
            if response is not None:
                channel[0] += float(response)*sample.dwell_weight
                channel[1] += sample.dwell_weight
        entry[0] += sample.dwell_weight
        entry[2].add(sample.dwell_index)
        self._values[key] = entry
        self._detector_keys = keys
        self._instrument_digest = identity

    def arrays(self):
        rows = sorted({r for r, _ in self._values})
        columns = sorted({c for _, c in self._values})
        ri, ci = {r: i for i, r in enumerate(rows)}, {c: i for i, c in enumerate(columns)}
        keys = self._detector_keys or ()
        response = np.full((len(keys), len(rows), len(columns)), np.nan)
        coverage = np.zeros_like(response)
        for (r, c), (_, channels, _) in self._values.items():
            for index, detector_key in enumerate(keys):
                value, weight = channels[detector_key]
                coverage[index, ri[r], ci[c]] = weight
                if math.isclose(weight, 1., rel_tol=0., abs_tol=1e-10):
                    response[index, ri[r], ci[c]] = value
        return {"physical_rows": np.asarray(rows, dtype=np.int64),
                "physical_columns": np.asarray(columns, dtype=np.int64),
                "detector_keys": np.asarray(keys, dtype=np.str_),
                "detector_response_per_tip_electron": response,
                "detector_dwell_coverage": coverage,
                # Compatibility: old consumers see the final detector only,
                # matching TipWaveResult.detector, never a sum of channels.
                "response_per_tip_electron": response[-1] if keys else np.full((len(rows), len(columns)), np.nan),
                "dwell_coverage": coverage[-1] if keys else np.zeros((len(rows), len(columns)))}


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
