"""Small GUI-owned request capture, with independent background preparation.

Capture copies editable values but does not reload configuration, resolve the
column, hash external files or solve reference planes. The worker never reads
the live State. Only the recursively frozen installed TOML assembly is shared.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from threading import Event
from types import SimpleNamespace

from temsim.calculation_cache import calculation_signatures, state_model_signature
from temsim.column.state_layout import apply_physical_layout_to_state


class PreparationCancelled(Exception):
    """An obsolete request stopped at a safe preparation boundary."""


class _CapturedModelState(SimpleNamespace):
    def to_dict(self):
        # Signature helpers may normalize their input dictionaries. Keep the
        # original capture independent from every consumer.
        return deepcopy(self._payload)


def reconstruct_calculation_state(
    state_type, payload, resolved_assembly, simulation_time_s,
    quality, ray_count, step_mm,
):
    """Keep the same persistence/assembly authority as synchronous requests."""
    snapshot = state_type.from_dict(payload)
    if resolved_assembly is not None:
        snapshot._resolved_assembly = resolved_assembly
        apply_physical_layout_to_state(
            snapshot, assembly=resolved_assembly,
            preserve_operating_parameters=True,
        )
    if simulation_time_s is not None:
        snapshot.simulation_time_s = float(simulation_time_s)
    emitter = getattr(snapshot.electron_gun, "emitter", None)
    if emitter is not None:
        emitter.ray_count = int(ray_count)
    else:
        snapshot.electron_gun.ray_count = int(ray_count)
    snapshot.step_mm = float(step_mm)
    snapshot.history_step_mm = max(
        float(step_mm), 2.0 if quality == "Preview" else 0.5,
    )
    return snapshot


@dataclass(frozen=True, slots=True)
class PreparedCalculationRequest:
    snapshot: object
    model_signature: str
    request_signatures: dict[str, str]
    ray_count: int
    step_mm: float


@dataclass(frozen=True, slots=True)
class CapturedCalculationRequest:
    """Owned editable-value capture; not a reference to a live instrument.

    The private signature view also copies the small component collections and
    resolved anchors. State.to_dict intentionally omits TOML-owned live geometry;
    retaining that geometry is essential to reproduce the original model tag.
    No trajectory, image, field-map array or previous result on State is copied.
    """

    state_type: type
    quality: str
    ray_count: int
    step_mm: float
    _model_state: _CapturedModelState
    _simulation_time_s: float | None

    @classmethod
    def capture(
        cls, state: object, quality: str, ray_count: int, step_mm: float,
    ) -> CapturedCalculationRequest:
        payload = deepcopy(state.to_dict())
        assembly = getattr(state, "_resolved_assembly", None)
        # Components are small parameter objects, not the State's solver caches.
        # Use one memo so aliases (e.g. an Objective lens in several lists) stay
        # coherent, and the immutable assembly is never passed to deepcopy.
        memo = {} if assembly is None else {id(assembly): assembly}
        names = (
            "lenses", "apertures", "stigmators", "deflectors",
            "corrector_elements", "recording_planes", "stem_detectors",
            "sample", "_upper_objective_package_resolved_positions_mm",
            "_ac_downstream_resolved_positions_mm",
        )
        values = {
            name: deepcopy(getattr(state, name), memo)
            for name in names if hasattr(state, name)
        }
        time_s = (
            float(state.simulation_time_s)
            if hasattr(state, "simulation_time_s") else None
        )
        view = _CapturedModelState(
            _payload=payload, _resolved_assembly=assembly,
            simulation_time_s=0.0 if time_s is None else time_s, **values,
        )
        return cls(
            type(state), str(quality), int(ray_count), float(step_mm), view, time_s,
        )

    def prepare(self, cancel_event: Event) -> PreparedCalculationRequest:
        def check_cancelled():
            if cancel_event.is_set():
                raise PreparationCancelled()

        check_cancelled()
        model_signature = state_model_signature(self._model_state)
        check_cancelled()
        snapshot = reconstruct_calculation_state(
            self.state_type, self._model_state.to_dict(),
            self._model_state._resolved_assembly, self._simulation_time_s,
            self.quality, self.ray_count, self.step_mm,
        )
        check_cancelled()
        signatures = calculation_signatures(snapshot)
        check_cancelled()
        return PreparedCalculationRequest(
            snapshot, model_signature, signatures, self.ray_count, self.step_mm,
        )
