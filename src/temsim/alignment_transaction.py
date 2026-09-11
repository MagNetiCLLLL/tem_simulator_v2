"""One Direct Alignment transaction contract for desktop and synchronous APIs.

Only catalog-owned raw controls are copied out of the private optimiser. The
final forward pass starts from the complete original graph plus those changes.
No preset, blanker override, model switch or sample replacement is permitted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from contextvars import ContextVar
import math
from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from temsim.immutable_json import freeze_json, json_digest
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.operating_modes import direct_alignment_by_key
from temsim.working_point import WorkingPointCheckpoint


class AlignmentCancelled(RuntimeError):
    pass


_CANCELLATION = ContextVar("alignment_cancellation", default=lambda: False)


def check_alignment_cancelled():
    if _CANCELLATION.get()():
        raise AlignmentCancelled("Direct Alignment cancelled; previous working point retained")


@dataclass(frozen=True)
class AlignmentRequest:
    request_id: str
    start_snapshot: object
    revision: int
    key: str
    target: float
    definition_id: str
    registry_digest: str

    @classmethod
    def capture(cls, state, key, target, *, revision):
        from temsim.optics.direct_alignment import _mode_matches
        try:
            definition = direct_alignment_by_key(key)
        except KeyError as exc:
            raise ValueError(f"Unregistered inverse target: {key}") from exc
        if definition.family not in {"condenser", "projector"} or definition.state_parameters:
            raise ValueError("This catalog entry is not an enabled inverse optics target")
        if not _mode_matches(state, definition):
            raise ValueError("Direct Alignment target is not enabled in the current operating mode")
        if not math.isfinite(target) or not definition.minimum <= target <= definition.maximum:
            raise ValueError("Direct Alignment target is outside the registered range")
        lenses = {lens.key: lens for lens in state.lenses}
        if any(key not in lenses or not lenses[key].enabled for key in definition.devices):
            raise ValueError("A registered coupled lens is disabled or absent")
        return cls(str(uuid4()), capture_instrument_snapshot(state), int(revision), str(key), float(target),
                   definition.definition_id, json_digest(asdict(definition)))


@dataclass(frozen=True)
class AlignmentCandidate:
    request: AlignmentRequest
    result: object
    checkpoint: WorkingPointCheckpoint | None
    validation: object
    status: str

    def __post_init__(self):
        from temsim.optics.direct_alignment import DirectAlignmentResult
        if not isinstance(self.result, DirectAlignmentResult):
            raise TypeError("Alignment candidates require an immutable DirectAlignmentResult")
        object.__setattr__(self, "validation", freeze_json(self.validation))


def _allowed_state(request, strengths):
    definition = direct_alignment_by_key(request.key)
    if json_digest(asdict(definition)) != request.registry_digest:
        raise ValueError("Direct Alignment registry changed during the request")
    if set(strengths) != set(definition.devices):
        raise ValueError("Candidate changes do not match the exact registered device set")
    state = request.start_snapshot.restore()
    lenses = {lens.key: lens for lens in state.lenses}
    for key, value in strengths.items():
        if not math.isfinite(value) or not 0 <= value <= float(lenses[key].max_percent):
            raise ValueError("Candidate raw control is outside the physical lens limit")
        lenses[key].percent = float(value)
    return state


def _constraints_pass(definition, measured):
    t = definition.targets
    if definition.key == "nanoprobe_convergence":
        return abs(measured.constraint_value) <= float(t["maximum_waist_offset_mm"])
    if definition.key == "microprobe_illumination":
        return (abs(measured.constraint_value) <= float(t["maximum_curvature_per_m"])
                and measured.convergence_95_mrad <= float(t["maximum_convergence_mrad"])
                and measured.convergence_99_mrad <= float(t["maximum_convergence_99_mrad"]))
    if definition.key == "image_magnification":
        return measured.relay_error_um <= float(t["maximum_relay_error_um"])
    return measured.diffraction_conjugacy_residual <= float(t["maximum_diffraction_conjugacy_residual"])


def solve_alignment_candidate(request, *, cancelled=lambda: False):
    from temsim.optics.direct_alignment import _solve_direct_alignment, _condenser_measurement, _validate_projector_production
    from temsim.physics.beam_statistics import branch_sample_statistics
    from temsim.physics.simulation import run
    from temsim.calculation_manifest import capture_calculation_manifest
    def check_cancel():
        if cancelled():
            raise AlignmentCancelled("Direct Alignment cancelled; previous working point retained")
    check_cancel()
    definition = direct_alignment_by_key(request.key)
    if json_digest(asdict(definition)) != request.registry_digest:
        raise ValueError("Direct Alignment registry changed")
    scratch = request.start_snapshot.restore()
    token = _CANCELLATION.set(cancelled)
    try:
        result = _solve_direct_alignment(scratch, request.key, request.target, definition=definition)
    finally:
        _CANCELLATION.reset(token)
    check_cancel()
    if not result.success:
        return AlignmentCandidate(request, result, None, {"status": "UNREACHABLE", "definition_id": request.definition_id}, "FAILED")
    candidate_state = _allowed_state(request, result.strengths)
    candidate_snapshot = capture_instrument_snapshot(candidate_state)
    # Final numerical refinement is independent of the optimiser, even when
    # the historical catalog used identical search and validation step sizes.
    validation_state = request.start_snapshot.restore()
    for lens in validation_state.lenses:
        if lens.key in result.strengths:
            lens.percent = result.strengths[lens.key]
    validation_state.step_mm = min(float(candidate_state.step_mm), float(result.validation_step_mm)) / 2
    manifest = capture_calculation_manifest(validation_state)
    simulation = run(validation_state, resolved_layout=getattr(validation_state, "_resolved_optics_layout", None), optical_only=True)
    forward_backend = getattr(validation_state, "active_backend", "not_recorded")
    check_cancel()
    if definition.family == "condenser":
        measured = _condenser_measurement(definition, branch_sample_statistics(simulation.incident))
        observable_backend = "CPU (retained particle statistics)"
    else:
        measured = _validate_projector_production(validation_state, definition,
            np.array([result.strengths[key] for key in definition.devices]), validation_state.step_mm)
        observable_backend = measured.execution_backend
    check_cancel()
    error = abs(math.log(max(measured.value, 1e-30) / request.target))
    spread = abs(measured.value - result.achieved) / max(abs(measured.value), 1e-30)
    valid = (math.isfinite(measured.value) and error <= float(definition.targets["maximum_relative_error"])
             and spread <= float(definition.targets["maximum_numerical_spread"]) and _constraints_pass(definition, measured))
    final = replace(result, success=valid, achieved=measured.value,
        constraint_value=measured.constraint_value, constraint_unit=measured.constraint_unit,
        convergence_95_mrad=measured.convergence_95_mrad, convergence_99_mrad=measured.convergence_99_mrad,
        illumination_diameter_95_um=measured.illumination_diameter_95_um,
        relay_error_um=measured.relay_error_um, diffraction_conjugacy_residual=measured.diffraction_conjugacy_residual,
        validation_step_mm=validation_state.step_mm, numerical_spread=spread,
        message=f"Target {request.target:g}; achieved {measured.value:g} {definition.unit}. "
                + ("Forward validation passed." if valid else "Final forward validation failed; no changes applied."))
    checkpoint = WorkingPointCheckpoint.from_result(SimpleNamespace(simulation=simulation,
        calculation_manifest=manifest, signatures=manifest.calculation_signatures), parent_id=request.start_snapshot.digest)
    validation = {"status": "PASS" if valid else "FAIL", "definition_id": request.definition_id,
                  "target": request.target, "achieved": measured.value, "relative_log_error": error,
                  "numerical_spread": spread, "step_mm": validation_state.step_mm,
                  "candidate_snapshot": candidate_snapshot.to_dict(),
                  "forward_snapshot_id": manifest.instrument_snapshot.digest,
                  "requested_backend": candidate_state.acceleration_backend,
                  "actual_backend": forward_backend,
                  "forward_particle_backend": forward_backend,
                  "observable_validation_backend": observable_backend,
                  "independent_reference": "NOT_RUN"}
    return AlignmentCandidate(request, final, checkpoint, validation, "READY_TO_APPLY" if valid else "FAILED")


class AlignmentCommitGate:
    """No parallel live State: commit returns a detached replacement in one step."""
    def __init__(self):
        self._applied = {}
        self._undo = []

    def apply(self, state, candidate, *, revision, previous_checkpoint=None):
        request = candidate.request
        current = capture_instrument_snapshot(state)
        if request.request_id in self._applied:
            expected, applied_revision = self._applied[request.request_id]
            if current.physical_digest != expected or revision != applied_revision:
                raise ValueError("STALE: already-applied request no longer matches the current working point")
            return state
        if revision != request.revision or current.physical_digest != request.start_snapshot.physical_digest:
            raise ValueError("STALE: physical revision or starting working point changed")
        if candidate.status != "READY_TO_APPLY" or not candidate.result.success or candidate.checkpoint is None:
            raise ValueError("Only a complete forward-validated candidate can be applied")
        if candidate.validation.get("status") != "PASS" or candidate.result.key != request.key:
            raise ValueError("Candidate validation does not match the request")
        restored = _allowed_state(request, candidate.result.strengths)
        restored_snapshot = capture_instrument_snapshot(restored)
        if restored_snapshot.digest != candidate.validation["candidate_snapshot"]["digest"]:
            raise ValueError("Candidate raw controls changed after validation")
        if candidate.checkpoint.snapshot.digest != candidate.validation["forward_snapshot_id"]:
            raise ValueError("Candidate forward checkpoint does not match validation")
        self._undo.append((request.request_id, current, previous_checkpoint))
        # The user's current observation plane is display state, not a control
        # solved by Direct Alignment. Preserve it without editing the immutable
        # candidate/checkpoint captured before that display-only movement.
        if hasattr(state, "virtual_observation_z_mm"):
            restored.virtual_observation_z_mm = state.virtual_observation_z_mm
        self._applied[request.request_id] = (restored_snapshot.physical_digest, revision + 1)
        return restored

    def undo(self):
        restored, _ = self.peek_undo()
        self.finish_undo()
        return restored

    def peek_undo(self):
        if not self._undo:
            raise ValueError("No applied Direct Alignment to undo")
        _, snapshot, checkpoint = self._undo[-1]
        restored = snapshot.restore()
        return restored, checkpoint

    def finish_undo(self):
        request_id, _, _ = self._undo.pop()
        self._applied.pop(request_id, None)

    def reject_application(self, request_id):
        if self._undo and self._undo[-1][0] == request_id:
            self.finish_undo()
