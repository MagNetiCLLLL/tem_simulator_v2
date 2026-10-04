"""Own one captured particle/wave pair, without starting another physics engine.

The existing particle controller executes the optical reference and, when
needed, specimen continuation. Its finished result is paired with a wave
session from the very same snapshot. This owner never edits source or optics.
"""
from dataclasses import dataclass
from uuid import uuid4

from PySide6.QtCore import QObject, Signal

from temsim.gui.calculation_controller import CalculationController
from temsim.calculation_cache import state_model_signature
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.specimen.source import specimen_interactions_active


@dataclass(frozen=True)
class BeamPairContext:
    token: str
    physical_identity: str
    particle_result: object
    specimen_active: bool
    specimen_z_mm: float
    instrument_identity: str
    particle_model_identity: str
    particle_ray_count: int
    particle_step_mm: float

    def verify_particle(self):
        state = getattr(self.particle_result, "state_snapshot", None)
        if state is None or state_model_signature(state) != self.particle_model_identity:
            raise ValueError("Executed particle source or optics do not match this pair")
        emitter = state.electron_gun.emitter
        if (int(emitter.ray_count) != self.particle_ray_count
                or float(state.step_mm) != self.particle_step_mm):
            raise ValueError("Executed particle numerical inputs do not match this pair")


class PairedBeamController(QObject):
    """Sequential, cancellable work with isolated publication signals."""

    progress = Signal(str)
    ready = Signal(object, object, object, object, float)
    failed = Signal(str)

    def __init__(self, state_provider, parent=None, *, calculations=None, artifact_store=None):
        super().__init__(parent)
        self.state_provider = state_provider
        self.calculations = calculations or CalculationController(
            self, high_cache_limit=1, high_cache_budget_bytes=0,
            tuning_cache_limit=1, tuning_cache_budget_bytes=0,
            artifact_store=artifact_store, persistent_cache_enabled=artifact_store is not None)
        self.calculations.result_ready.connect(self._particle_ready)
        self.calculations.failed.connect(self._failed)
        self.calculations.progress_changed.connect(self._progress)
        self.token = None
        self.context = None
        self._snapshot = self._state = self._request = self._summary = None
        self._stage = None
        self._duration = 0.
        self.calculations.pool.coordinator.register_retained(self, "retained_roots")

    def retained_roots(self):
        return self._snapshot, self._state, self.context

    def start(self, state, request, summary, ray_count, step_mm):
        # Capture first: malformed requests must not discard a valid old pair.
        snapshot = capture_instrument_snapshot(state)
        frozen = snapshot.restore()
        self.cancel()
        self.token = uuid4().hex
        self._snapshot, self._state = snapshot, frozen
        self._request, self._summary = request, summary
        self._ray_count, self._step_mm = int(ray_count), float(step_mm)
        self._model_identity = state_model_signature(frozen)
        self._duration = 0.
        self._stage = "rays"
        self.progress.emit("Paired beams: calculating particle transport from the captured tip.")
        try:
            self._submit("rays")
        except Exception:
            self.cancel()
            raise
        return self.token

    def _submit(self, workflow, existing_result=None):
        self.calculations.submit_background(
            self._snapshot.restore(), "High accuracy", self._ray_count,
            self._step_mm, workflow=workflow, existing_result=existing_result)

    def is_current(self):
        if self.token is None or self._snapshot is None:
            return False
        try:
            return (capture_instrument_snapshot(self.state_provider()).physical_digest
                    == self._snapshot.physical_digest)
        except (ValueError, OSError, RuntimeError):
            return False

    def _particle_ready(self, quality, result, duration):
        if self.token is None or self._stage not in {"rays", "sample"}:
            return
        if not self.is_current():
            self._failed(quality, "Instrument inputs changed; the captured pair is no longer current.")
            return
        if getattr(result, "workflow", None) != self._stage:
            self._failed(quality, "Particle workflow does not match the captured pair request.")
            return
        self._duration += float(duration)
        sample_active = specimen_interactions_active(self._state.sample)
        if self._stage == "rays" and sample_active:
            self._stage = "sample"
            self.progress.emit("Paired beams: continuing particle specimen interactions from the saved incident beam.")
            try:
                self._submit("sample", result)
            except Exception as error:
                self._failed(quality, str(error))
            return
        self._stage = "wave"
        self.context = BeamPairContext(self.token, self._snapshot.physical_digest,
            result, sample_active, float(self._state.sample.z_mm), self._snapshot.digest,
            self._model_identity, self._ray_count, self._step_mm)
        try:
            self.context.verify_particle()
        except (ValueError, AttributeError) as error:
            self._failed(quality, str(error))
            return
        self.progress.emit("Paired beams: particle result complete; calculating the coherent field from the same captured inputs.")
        self.ready.emit(self._state, self._request, self._summary,
                        self.context, self._duration)

    def _progress(self, _quality, done, total, message):
        if self.token is not None and self._stage in {"rays", "sample"}:
            self.progress.emit(f"Paired particles: {message} ({done}/{total})")

    def _failed(self, _quality, message):
        if self.token is None:
            return
        self.cancel()
        self.failed.emit(str(message))

    def cancel(self):
        self.calculations.invalidate_pending(include_explicit=True)
        self.token = self.context = None
        self._snapshot = self._state = self._request = self._summary = None
        self._stage = None

    def shutdown(self, msecs=3000):
        self.cancel()
        return (self.calculations.pool.waitForDone(msecs)
                and self.calculations.section_file_pool.waitForDone(msecs))
