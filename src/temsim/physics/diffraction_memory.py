"""Bounded in-memory, pre-detector STEM intensity capture and recollection.

This is the existing angle-resolved first-order recording approximation, not
a complex-wave checkpoint for arbitrary coherent STEM image planes.
"""
from dataclasses import replace
from types import MappingProxyType
import numpy as np

from temsim.physics.fourdstem import FourDSTEMArtifact, PixelatedDetectorResponse


class MemoryDiffractionSink:
    def __init__(self, budget_bytes, signature, cancelled=lambda: False):
        self.budget_bytes = int(budget_bytes)
        self.signature = str(signature)
        self.cancelled = cancelled
        self.data = None

    def begin(self, calibration, valid_reciprocal_mask, *, maximum_isotropic_angle_mrad):
        if self.data is not None:
            raise ValueError("Memory diffraction capture was already started")
        size = int(np.prod(calibration.shape, dtype=object)) * 4
        if size > self.budget_bytes:
            raise ValueError(f"STEM angular cache requires {size / 1024**3:.3f} GiB; reduce scan pixels or the wave grid")
        self.calibration = calibration
        self.valid = np.asarray(valid_reciprocal_mask, bool).copy()
        if self.valid.shape != calibration.shape[-2:]:
            raise ValueError("Diffraction capture mask shape mismatch")
        self.maximum_angle = float(maximum_isotropic_angle_mrad)
        self.data = np.zeros(calibration.shape, dtype=np.float32)
        self.complete = np.zeros(calibration.shape[:2], bool)

    def write_frame(self, scan_y, scan_x, diffraction_probability):
        if self.cancelled():
            raise RuntimeError("Interactive diffraction capture cancelled")
        p = np.asarray(diffraction_probability)
        if p.shape != self.valid.shape or not np.all(np.isfinite(p)) or np.any(p < 0):
            raise ValueError("Invalid diffraction probability frame")
        if p.sum() > 1 + 2e-5:
            raise ValueError("Diffraction probability exceeds its pre-specimen reference")
        self.data[scan_y, scan_x] = np.where(self.valid, p, 0)
        self.complete[scan_y, scan_x] = True

    def finish(self):
        if self.data is None or not np.all(self.complete) or self.cancelled():
            raise ValueError("Incomplete memory diffraction capture cannot be published")
        self.data.setflags(write=False)
        return FourDSTEMArtifact(None, None, None, self.calibration, self.data, MappingProxyType({
            "status": "complete", "storage": "memory", "shape": list(self.data.shape),
            "detector_response": PixelatedDetectorResponse().provenance(),
            "provenance": {"fourdstem_cube_state_signature": self.signature,
                           "stored_frame_quantity": "configuration-averaged diffraction probability",
                           "maximum_isotropic_angle_mrad": self.maximum_angle},
        }))


def can_recollect_stem(frame):
    return bool(frame is not None and frame.fourdstem_artifact is not None
                and not (frame.metrics or {}).get("rutherford_tail_enabled", False))


def recollect_stem(state, frame):
    from temsim.detector.stem_signal import DetectorSignal, _current_values, reweight_stem_scan
    from temsim.physics.fourdstem import integrate_runtime_recording_planes
    from temsim.physics.record_plane import build_record_plane_plan
    if not can_recollect_stem(frame):
        raise ValueError("STEM recollection requires raw angular frames without an uncached high-angle tail")
    artifact = frame.fourdstem_artifact
    # The artifact contains probabilities per incident electron; do not turn
    # them into source fractions by renormalising the newly surviving signal.
    incident_fraction = float(frame.metrics["incident_sample_fraction"])
    if not np.isfinite(incident_fraction) or not -1e-12 <= incident_fraction <= 1.0 + 1e-12:
        raise ValueError("Cached incident sample fraction must be finite and in [0, 1]")
    incident_fraction = min(max(incident_fraction, 0.0), 1.0)
    if frame.absorbed_fraction is None:
        # Older frames without an absorption map used a uniform slab.
        bulk_survival = float(
            frame.metrics["tracked_probability_after_inelastic_absorption"])
        if not np.isfinite(bulk_survival) or not -1e-12 <= bulk_survival <= 1.0 + 1e-12:
            raise ValueError("Cached bulk specimen survival must be finite and in [0, 1]")
        bulk_survival = min(max(bulk_survival, 0.0), 1.0)
        absorbed = np.full(artifact.data.shape[:2], incident_fraction * (1.0 - bulk_survival))
    else:
        absorbed = np.asarray(frame.absorbed_fraction, dtype=float)
        if (absorbed.shape != artifact.data.shape[:2] or not np.all(np.isfinite(absorbed))
                or np.any(absorbed < -1e-12) or np.any(absorbed > incident_fraction + 1e-12)):
            raise ValueError("Cached STEM absorption must be a finite source-fraction map matching the raster")
        absorbed = np.clip(absorbed, 0.0, incident_fraction)
    scale = incident_fraction - absorbed
    routed = integrate_runtime_recording_planes(
        artifact, None, build_record_plane_plan(
            state, scan_times_s=artifact.calibration.scan_times_s, recalibrate_scan=True,
        ),
    )
    detectors = {p.key: p for p in state.stem_detectors if p.inserted and p.readout_enabled}
    fractions = {key: value * scale for key, value in routed.images.items() if key in detectors}
    signals = {}
    for key, values in fractions.items():
        mean = float(np.mean(values))
        simulated, current, rate = _current_values(state, mean)
        signals[key] = DetectorSignal(key, detectors[key].name, mean, simulated, current, rate, None)
    metrics = dict(frame.metrics)
    valid_probability = np.sum(artifact.data, axis=(-2, -1), dtype=np.float64)
    recorded = sum(fractions.values(), np.zeros_like(valid_probability))
    surviving = routed.surviving_weight * scale
    other_stops = np.maximum(valid_probability * scale - recorded - surviving, 0)
    # Match live source-fraction accounting: upstream losses, out-of-band
    # intensity, other stops and the surviving unrecorded beam all belong to
    # this remainder. Truncation is a diagnostic subset, not an extra loss.
    uncollected = np.maximum(1.0 - absorbed - recorded, 0.0)
    conservation_error = float(np.max(np.abs(absorbed + recorded + uncollected - 1.0)))
    metrics.update({"interactive_cube_reused": True, "record_plane_plan_fingerprint": routed.plan_fingerprint,
                    "readout_model": "angle_resolved_first_order_sequential_stops",
                    "other_stop_loss_mean_source_fraction": float(np.mean(other_stops)),
                    "post_recording_surviving_mean_source_fraction": float(np.mean(surviving)),
                    "pre_sample_lost_fraction": 1.0 - incident_fraction,
                    "mean_uncollected_fraction": float(np.mean(uncollected)),
                    "maximum_probability_conservation_error": conservation_error,
                    "real_probability_conserved": conservation_error <= 5.0e-10})
    result = replace(frame, fractions=fractions, detector_signals=signals, metrics=metrics,
                     uncollected_fraction=uncollected,
                     absorbed_fraction=absorbed,
                     truncated_fraction=np.maximum(1 - valid_probability, 0) * scale,
                     high_angle_tail_fraction=None)
    return reweight_stem_scan(state, result)
