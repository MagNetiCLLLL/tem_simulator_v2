"""Narrow high-accuracy adapter for optional 4D-STEM products.

This file intentionally does not own GUI or global pipeline policy.  It turns
one immutable high-accuracy state/simulation pair into a capture sink, then
derives virtual and physical detector images with separate dependency
signatures.  The raw diffraction cube is independent of downstream D/I/P
settings; only the physical-recording-plane product depends on the current
signed record-plane plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping, Sequence

import numpy as np

from temsim.calculation_cache import calculation_signatures
from temsim.physics.beam_current import effective_source_current_a
from temsim.physics.fourdstem import (
    FourDSTEMArtifact,
    FourDSTEMCaptureSink,
    PixelatedDetectorResponse,
    RuntimeDetectorImages,
    VirtualDetector,
    integrate_runtime_recording_planes,
    integrate_virtual_detectors,
)
from temsim.physics.record_plane import (
    RecordPlanePlan,
    build_record_plane_plan,
    record_plane_plan_provenance,
)


ELEMENTARY_CHARGE_C = 1.602176634e-19


def _digest(payload: Mapping) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sample_incident_fraction(simulation) -> float:
    branch = simulation.incident
    alive = np.asarray(branch.alive, dtype=bool)
    if alive.ndim != 1:
        raise ValueError("Incident sample mask must be one-dimensional")
    raw_weights = getattr(branch, "ray_weight", None)
    weights = (
        np.full(alive.shape, 1.0 / max(alive.size, 1), dtype=float)
        if raw_weights is None
        else np.asarray(raw_weights, dtype=float)
    )
    if weights.shape != alive.shape:
        raise ValueError("Incident sample weights must match the alive mask")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Incident sample weights must be finite and non-negative")
    fraction = float(np.sum(weights[alive], dtype=np.float64))
    if fraction > 1.0 + 1.0e-10:
        raise ValueError("Incident sample probability exceeds one")
    return min(max(fraction, 0.0), 1.0)


@dataclass(frozen=True, slots=True)
class FourDSTEMRequest:
    path: Path
    response: PixelatedDetectorResponse = field(
        default_factory=PixelatedDetectorResponse
    )
    frame_count: int | None = None
    overwrite: bool = False
    resume: bool = False
    checkpoint_every: int = 32
    dtype: str = "float32"
    capture_record_plane_provenance: bool = True

    def validate(self) -> "FourDSTEMRequest":
        if not str(self.path):
            raise ValueError("4D-STEM output path must not be empty")
        if self.frame_count is not None and int(self.frame_count) < 1:
            raise ValueError("4D-STEM frame count must be positive")
        if int(self.checkpoint_every) < 1:
            raise ValueError("4D-STEM checkpoint interval must be positive")
        if np.dtype(self.dtype).kind != "f":
            raise ValueError("Raw 4D-STEM probability storage must be floating point")
        self.response.validate()
        return self


@dataclass(frozen=True, slots=True)
class PreparedFourDSTEMCapture:
    sink: FourDSTEMCaptureSink
    record_plane_plan: RecordPlanePlan
    source_dependency_signature: str
    cube_state_signature: str
    virtual_detector_state_signature: str
    physical_recording_state_signature: str
    cube_dependency_signature: str
    incident_fraction: float
    dwell_time_s: float
    expected_electrons_per_frame: float


@dataclass(frozen=True, slots=True)
class FourDSTEMProducts:
    artifact: FourDSTEMArtifact
    physical_recording: RuntimeDetectorImages
    virtual_detector_images: Mapping[str, np.ndarray]
    signatures: Mapping[str, str]


def prepare_fourdstem_capture(
    state,
    simulation,
    request: FourDSTEMRequest,
    *,
    maximum_step_mm: float | None = None,
) -> PreparedFourDSTEMCapture:
    """Freeze dose, source signature, and current runtime recording geometry."""

    request.validate()
    configured_frames = (
        int(state.ac_deflector.scan_pixels_x)
        * int(state.ac_deflector.scan_lines)
    )
    frame_count = configured_frames if request.frame_count is None else int(request.frame_count)
    frame_period_s = float(state.ac_deflector.scan_frame_period_s)
    if not math.isfinite(frame_period_s) or frame_period_s <= 0.0:
        raise ValueError("4D-STEM scan frame period must be finite and positive")
    dwell_time_s = frame_period_s / frame_count
    incident_fraction = _sample_incident_fraction(simulation)
    expected_electrons = (
        effective_source_current_a(state)
        * incident_fraction
        * dwell_time_s
        / ELEMENTARY_CHARGE_C
    )
    plan = build_record_plane_plan(
        state,
        maximum_step_mm=maximum_step_mm,
    )
    # The diffraction cube ends at the specimen exit.  ``wave_source`` keeps
    # illumination, specimen, scan-coil, and numerical wave inputs while
    # intentionally excluding independently repeatable downstream D/I/P
    # projection and recording-plane settings.
    state_signatures = calculation_signatures(state)
    source_signature = state_signatures["wave_source"]
    cube_signature = _digest({
        "model": "configuration_averaged_4dstem_v1",
        "stem_source_signature": source_signature,
        "stored_frame_quantity": "diffraction_probability",
    })
    sink = FourDSTEMCaptureSink(
        request.path,
        electrons_per_frame=expected_electrons,
        response=request.response,
        store_raw_probability=True,
        record_plane_plan=(
            plan if request.capture_record_plane_provenance else None
        ),
        dtype=np.dtype(request.dtype),
        overwrite=request.overwrite,
        resume=request.resume,
        checkpoint_every=request.checkpoint_every,
        provenance={
            "stem_source_signature": source_signature,
            "fourdstem_cube_state_signature": state_signatures[
                "fourdstem_cube"
            ],
            "fourdstem_virtual_detector_state_signature_at_capture": (
                state_signatures["fourdstem_virtual_detectors"]
            ),
            "fourdstem_physical_recording_state_signature_at_capture": (
                state_signatures["fourdstem_physical_recording"]
            ),
            "record_plane_plan": dict(record_plane_plan_provenance(plan)),
            "cube_dependency_signature": cube_signature,
            "dose_and_detector_response": "deferred_to_derived_products",
        },
    )
    return PreparedFourDSTEMCapture(
        sink=sink,
        record_plane_plan=plan,
        source_dependency_signature=source_signature,
        cube_state_signature=state_signatures["fourdstem_cube"],
        virtual_detector_state_signature=state_signatures[
            "fourdstem_virtual_detectors"
        ],
        physical_recording_state_signature=state_signatures[
            "fourdstem_physical_recording"
        ],
        cube_dependency_signature=cube_signature,
        incident_fraction=incident_fraction,
        dwell_time_s=dwell_time_s,
        expected_electrons_per_frame=expected_electrons,
    )


def derive_fourdstem_products(
    prepared: PreparedFourDSTEMCapture,
    artifact: FourDSTEMArtifact,
    *,
    virtual_detectors: Sequence[VirtualDetector] = (),
    chunk_scan_points: int = 1,
) -> FourDSTEMProducts:
    """Derive independently cacheable physical and arbitrary virtual images."""

    provenance = artifact.metadata.get("provenance", {})
    if provenance.get("cube_dependency_signature") != prepared.cube_dependency_signature:
        raise ValueError("4D-STEM artifact belongs to a different source calculation")
    physical = integrate_runtime_recording_planes(
        artifact,
        None,
        prepared.record_plane_plan,
        chunk_scan_points=chunk_scan_points,
    )
    virtual = integrate_virtual_detectors(
        artifact,
        virtual_detectors,
        chunk_scan_points=max(int(chunk_scan_points), 1),
    )
    virtual_signature = _digest({
        "cube": prepared.cube_dependency_signature,
        "detectors": [
            {
                "key": detector.key,
                "weights_sha256": hashlib.sha256(
                    np.ascontiguousarray(detector.weights).view(np.uint8)
                ).hexdigest(),
            }
            for detector in virtual_detectors
        ],
    })
    physical_signature = _digest({
        "cube": prepared.cube_dependency_signature,
        "record_plane_plan": prepared.record_plane_plan.fingerprint,
    })
    return FourDSTEMProducts(
        artifact=artifact,
        physical_recording=physical,
        virtual_detector_images=virtual,
        signatures=MappingProxyType({
            "fourdstem_cube": prepared.cube_dependency_signature,
            "fourdstem_physical_recording": physical_signature,
            "fourdstem_virtual_detectors": virtual_signature,
        }),
    )
