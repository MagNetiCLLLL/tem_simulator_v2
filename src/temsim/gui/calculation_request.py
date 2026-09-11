"""Small GUI-owned request capture, with independent background preparation.

Capture copies editable values but does not reload configuration, resolve the
column, hash external files or solve reference planes. The worker never reads
the live State. Full-graph requests also detach the installed TOML assembly;
lightweight legacy previews may share its recursively frozen definition.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from threading import Event
from types import SimpleNamespace

from temsim.calculation_cache import calculation_signatures, state_model_signature
from temsim.calculation_manifest import (
    ExternalInputIdentity,
    assert_external_input_inventory_unchanged,
    capture_external_input_identities,
)
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.instrument_snapshot import encode_instrument, decode_instrument


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
    return apply_request_numerics(snapshot, quality, ray_count, step_mm)


def apply_request_numerics(snapshot, quality, ray_count, step_mm):
    """Change only the explicitly requested sampling, never optical controls."""
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
    external_inputs: tuple[ExternalInputIdentity, ...] = ()


@dataclass(frozen=True, slots=True)
class CapturedCalculationRequest:
    """Owned editable-value capture; not a reference to a live instrument.

    The private signature view also copies the small component collections and
    resolved anchors. State.to_dict intentionally omits TOML-owned live geometry;
    retaining that geometry is essential to reproduce the original model tag.
    Preview/Medium omit heavy numeric products. High accuracy additionally
    captures the complete parameter graph, including field-map arrays.
    """

    state_type: type
    quality: str
    ray_count: int
    step_mm: float
    _model_state: _CapturedModelState
    _simulation_time_s: float | None
    _instrument_graph: object = None

    @classmethod
    def capture(
        cls, state: object, quality: str, ray_count: int, step_mm: float,
    ) -> CapturedCalculationRequest:
        graph = None
        if quality == "High accuracy" or getattr(getattr(state, "electron_gun", None), "source_representation", "") == "effective_gaussian_schell":
            from temsim.optics.model import State
            if isinstance(state, State):
                graph = encode_instrument(state)
                state = decode_instrument(graph)
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
            "lens_field_map_descriptors",
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
            type(state), str(quality), int(ray_count), float(step_mm), view, time_s, graph,
        )

    def prepare(self, cancel_event: Event) -> PreparedCalculationRequest:
        def check_cancelled():
            if cancel_event.is_set():
                raise PreparationCancelled()

        check_cancelled()
        external_inputs = capture_external_input_identities(self._model_state)
        model_signature = state_model_signature(self._model_state)
        check_cancelled()
        if self._instrument_graph is not None:
            snapshot = apply_request_numerics(decode_instrument(self._instrument_graph),
                self.quality, self.ray_count, self.step_mm)
        else:
            snapshot = reconstruct_calculation_state(
                self.state_type, self._model_state.to_dict(),
                self._model_state._resolved_assembly, self._simulation_time_s,
                self.quality, self.ray_count, self.step_mm,
            )
        check_cancelled()
        signatures = calculation_signatures(snapshot)
        check_cancelled()
        assert_external_input_inventory_unchanged(snapshot, external_inputs)
        return PreparedCalculationRequest(
            snapshot, model_signature, signatures, self.ray_count, self.step_mm,
            external_inputs,
        )
