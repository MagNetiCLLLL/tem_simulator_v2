"""Single-worker asynchronous simulation controller."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import (
    dataclass,
    replace,
)
import math
import os
from pathlib import Path
from threading import Event
from time import perf_counter
from uuid import uuid4
from temsim.job_events import job_event, job_stage
from temsim.physics.backend_execution import capture_worker_backends, classical_backend_preflight
from temsim import input_io
from types import MappingProxyType

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal

from temsim.calculation_cache import (
    calculation_signatures,
    calculation_signatures_for_request,
    matching_products,
    state_model_signature,
)
from temsim.cache_memory import (
    RetainedMemoryLedger,
    estimate_result_cache_bytes,
    retained_memory_inventory,
)
from temsim.artifact_store import ArtifactStore
from temsim.calculation_manifest import (
    CalculationManifest,
    assert_external_input_inventory_unchanged,
    capture_calculation_manifest,
    capture_external_input_identities,
)
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.design_explorer import (
    ProductStageStatus,
    ResultProductSummary,
    evaluate_product_statuses,
    summarise_calculation_result,
)
from temsim.physics.all_lens_crossovers import detect_all_lens_crossovers
from temsim.physics.simulation import (
    MAX_VECTORIZED_POST_RAYS,
    _column_checkpoint_planes,
    run as run_ray_simulation,
)
from temsim.physics.recording_stop import determine_tem_stop_z
from temsim.physics.wave_imaging import (
    estimate_tem_wave_memory_bytes,
    tem_wave_imaging_enabled,
)
from temsim.simulation_pipeline import (
    CalculationResult,
    aperture_stop_records,
    calculate,
)
from temsim.specimen.source import (
    specimen_interactions_active,
)
from temsim.physics.optical_tuning import is_tuning_quality, prepare_tuning_snapshot
from temsim.gui.calculation_request import (
    CapturedCalculationRequest,
    PreparationCancelled,
    reconstruct_calculation_state,
)


HIGH_ACCURACY_MEMORY_BUDGET_BYTES = 24 * 1024**3
HIGH_ACCURACY_RESULT_CACHE_BUDGET_BYTES = 8 * 1024**3
HIGH_ACCURACY_RESULT_CACHE_LIMIT = 32
TUNING_RESULT_CACHE_BUDGET_BYTES = 1024**3
TUNING_RESULT_CACHE_LIMIT = 128
PERSISTENT_ARTIFACT_CACHE_BUDGET_BYTES = 16 * 1024**3


def default_artifact_cache_root() -> Path:
    """Return the bounded, user-writable cache directory for this app."""

    local_app_data = str(os.environ.get("LOCALAPPDATA", "")).strip()
    base = (
        Path(local_app_data)
        if local_app_data
        else Path.home() / ".cache"
    )
    return base / "TEM Simulator v2" / "high_accuracy_artifacts"


def project_artifact_fallback_root() -> Path | None:
    """Find an existing recovery cache in this source checkout only.

    Do not search the working directory, another project, or installed data
    directories. Resolving the candidate also rejects an escaping junction.
    """

    try:
        source = Path(__file__).resolve()
        checkout = source.parents[3]
        if (
            source.relative_to(checkout).as_posix()
            != "src/temsim/gui/calculation_controller.py"
            or not (checkout / "pyproject.toml").is_file()
            or not (checkout / "configs").is_dir()
        ):
            return None
        candidate = checkout / "outputs" / "high_accuracy_artifacts"
        if not candidate.is_dir():
            return None
        resolved = candidate.resolve()
        return resolved if resolved.is_relative_to(checkout) else None
    except (OSError, RuntimeError, ValueError, IndexError):
        return None


@dataclass(frozen=True, slots=True)
class HighAccuracyReusePlan:
    """Small, read-only description of reusable High accuracy products."""

    request_signatures: Mapping[str, str]
    product_statuses: tuple[ProductStageStatus, ...]
    source_request_signature: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_signatures",
            MappingProxyType(dict(self.request_signatures)),
        )


def estimate_calculation_memory_bytes(
    state, quality: str, ray_count: int, step_mm: float
) -> int:
    """Conservatively estimate peak solver memory for one calculation.

    A 24 GiB application budget leaves approximately 8 GiB for Qt, Python,
    the operating system and allocator overhead on the supported 32 GiB
    workstation configuration. High-accuracy estimates include the optional
    TEM wave grid, atomistic slices and frozen-phonon configurations.
    """

    rays = int(ray_count)
    step = float(step_mm)
    if rays <= 0:
        raise ValueError("Ray count must be positive")
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("Integration step must be finite and positive")

    gun_start = float(state.electron_gun.exit_plane_z_mm)
    sample_z = float(state.sample.z_mm)
    stop_z = float(determine_tem_stop_z(state))
    pre_span = max(sample_z - gun_start, 0.0)
    post_span = max(stop_z - sample_z, 0.0)
    pre_nodes = int(math.ceil(pre_span / step)) + 2
    post_nodes = int(math.ceil(post_span / step)) + 2

    history_step = max(step, 2.0 if quality == "Preview" else 0.5)
    pre_history = int(math.ceil(pre_span / history_step)) + 2
    post_history = int(math.ceil(post_span / history_step)) + 2
    from temsim.physics.core import electron
    from temsim.physics.lens_field_provider import FrozenMappedField, active_mapped_providers

    mapped_fields = tuple(FrozenMappedField.from_provider(provider)
                          for provider in active_mapped_providers(state))
    map_storage = 0
    history_itemsize = 8 if mapped_fields else 4
    for item in mapped_fields:
        # Bound extra exact-grid intervals using the solver's own step rule.
        # Counting base nodes as well is intentionally conservative here.
        field_step = item.maximum_step_mm(step, electron(state)[1])
        lower, upper = item.field_map.field_support_mm
        stride = max(1, int(round(history_step/step)))
        for start, stop, before_sample in ((gun_start, sample_z, True),
                                           (sample_z, stop_z, False)):
            length = max(0.0, min(upper,stop)-max(lower,start))
            extra = 2*int(math.ceil(length/field_step)) + 2 if length else 0
            if before_sample:
                pre_nodes += extra
                pre_history += int(math.ceil(extra/stride))
            else:
                post_nodes += extra
                post_history += int(math.ceil(extra/stride))
        map_storage += 4*sum(array.nbytes for array in (
            *item.field_map.axes_m, *item.field_map.components_t))
    specimen_mode = str(
        getattr(state.sample, "specimen_mode", "atomic")
    ).strip().lower()
    if specimen_mode in {"atomic", "reference"} and bool(
        getattr(state.sample, "inserted", True)
    ):
        from temsim.specimen.inelastic import (
            real_inelastic_distribution,
            real_inelastic_ray_branches,
        )

        branch_count = len(
            real_inelastic_ray_branches(
                real_inelastic_distribution(state), ray_count=rays
            )
        )
    else:
        branch_count = 1
    vectorised_branches = min(
        branch_count,
        max(1, MAX_VECTORIZED_POST_RAYS // rays),
    )
    peak_post_rays = rays * vectorised_branches
    # Canonical RK4 uses axial node/midpoint coefficients and separate per-ray
    # momentum. No integration coefficient is a (Z, ray) matrix. Account for
    # retained plans, packed stages, construction temporaries and a possible
    # device copy, plus the NumPy fallback's temporary per-ray stage vectors.
    # Conservative 64-vector allowances cover these two independent scales.
    working = (
        (pre_nodes + post_nodes) * 64
        + max(rays, peak_post_rays) * 64
    ) * 8
    # Mapped transport retains float64 histories for local Jacobians; include
    # that precision and the extra map-grid nodes in the memory guard.
    history = (
        pre_history + branch_count * post_history
    ) * rays * (4 * history_itemsize + 8)  # Float64 executed flight clock.
    checkpoint_count = len(
        _column_checkpoint_planes(gun_start, sample_z, rays)
    )
    checkpoint_storage = checkpoint_count * rays * 5 * 8
    gpu_checkpoint_copy = bool(
        getattr(state, "acceleration_enabled", True)
        and str(getattr(state, "acceleration_backend", "Auto"))
        .strip()
        .lower()
        != "cpu"
    )
    checkpoint_peak = checkpoint_storage * (
        2 if gpu_checkpoint_copy else 1
    )
    # CUDA keeps the active batch's device histories while copying its output
    # to host. Already retained branches are counted once in `history` above.
    history_device_copy = (
        max(pre_history * rays, post_history * peak_post_rays) * (4 * 4 + 8)
        if gpu_checkpoint_copy else 0
    )
    wave_imaging = (
        estimate_tem_wave_memory_bytes(state)
        if quality == "High accuracy"
        else 0
    )
    return int(
        working
        + history
        + history_device_copy
        + checkpoint_peak
        + map_storage
        + wave_imaging
        + 512 * 1024**2
    )


def format_memory_size(byte_count: int) -> str:
    return f"{float(byte_count) / 1024**3:.1f} GiB"


class WorkerSignals(QObject):
    result = Signal(int, str, object, float)
    error = Signal(int, str, str)
    progress = Signal(int, str, int, int, str)
    finished = Signal(int, str)


class PreparationSignals(WorkerSignals):
    prepared = Signal(int, str, object)


def preparation_input_roots(request):
    # Do not follow the lease's store pointer into unrelated cached inputs.
    lease = getattr(request, "_input_assets", None)
    return (getattr(request, "_model_state", None), getattr(request, "_instrument_graph", None),
            getattr(lease, "_items", None))


class PreparationWorker(QRunnable):
    """Prepare one captured request without blocking the Qt event thread."""

    def __init__(self, generation, request, cancel_event):
        super().__init__()
        self.generation = generation
        self.quality = request.quality
        self.request = request
        self.cancel_event = cancel_event
        self.signals = PreparationSignals()
        self._inputs_released = False
        from temsim.gui.job_coordinator import ResourceClaim
        # Preparation is not a 24 GiB solver. Account for captured buffers plus
        # decode/manifest temporaries. This is an ownership estimate, not RSS.
        size = sum(retained_memory_inventory(*preparation_input_roots(request)).values())
        self.resource_claim = ResourceClaim(128 * 1024**2 + 10 * size)

    def release_inputs(self):
        if not self._inputs_released:
            self._inputs_released = True
            assets = getattr(self.request, "_input_assets", None)
            if assets is not None:
                assets.close()

    def job_adapter(self):
        from temsim.gui.job_coordinator import WorkerAdapter
        return WorkerAdapter(self, self.cancel_event, self.resource_claim, self.generation,
            getattr(self, "job_input_identity", ""), self.quality, preparation_input_roots(self.request),
            lambda message: self.signals.error.emit(self.generation, self.quality, message),
            lambda: self.signals.finished.emit(self.generation, self.quality), self.release_inputs,
            request_id=getattr(self, "request_id", ""), backend="CPU preparation")

    def run(self):
        try:
            with job_stage("preparation", backend="CPU"):
                prepared = self.request.prepare(self.cancel_event)
            if not self.cancel_event.is_set():
                self.signals.prepared.emit(self.generation, self.quality, prepared)
        except PreparationCancelled:
            return
        except Exception as exc:
            if not self.cancel_event.is_set():
                self.signals.error.emit(self.generation, self.quality, str(exc))
                self.signals.finished.emit(self.generation, self.quality)
        finally:
            self.release_inputs()
        # Success transfers ownership to a solver or exact-cache delivery.
        # Finishing here would incorrectly release the live-frame scheduler.


class CalculationWorker(QRunnable):
    def __init__(
        self,
        generation: int,
        quality: str,
        state,
        *,
        model_signature: str = "",
        request_signatures: dict[str, str] | None = None,
        existing_result: CalculationResult | None = None,
        artifact_store: ArtifactStore | None = None,
        calculation_manifest: CalculationManifest | None = None,
        external_inputs=None,
        allow_project_artifact_fallback: bool = False,
        artifact_cache_budget_bytes: int = PERSISTENT_ARTIFACT_CACHE_BUDGET_BYTES,
        section_request: dict | None = None,
        particle_tuning: bool = False,
    ) -> None:
        super().__init__()
        self.generation = generation
        self.quality = quality
        self.state = state
        self.model_signature = str(model_signature)
        from temsim.gui.job_coordinator import ResourceClaim
        self.cancel_event = Event()
        self.resource_claim = ResourceClaim()
        self.job_input_identity = self.model_signature
        self.request_signatures = dict(request_signatures or {})
        self.existing_result = existing_result
        self.section_request = section_request
        self.particle_tuning = bool(particle_tuning)
        self.artifact_store = artifact_store
        self.calculation_manifest = calculation_manifest
        if external_inputs is None:
            external_inputs = (calculation_manifest.external_inputs
                               if calculation_manifest is not None else
                               capture_external_input_identities(state))
        self.external_inputs = tuple(external_inputs)
        self.allow_project_artifact_fallback = bool(
            allow_project_artifact_fallback and quality == "High accuracy"
        )
        self.artifact_cache_budget_bytes = int(artifact_cache_budget_bytes)
        self.signals = WorkerSignals()

    def job_adapter(self):
        from temsim.gui.job_coordinator import WorkerAdapter
        return WorkerAdapter(self, self.cancel_event, self.resource_claim, self.generation,
            self.job_input_identity, self.quality, (self.state, self.existing_result),
            lambda message: self.signals.error.emit(self.generation, self.quality, message),
            lambda: self.signals.finished.emit(self.generation, self.quality), lambda: None,
            request_id=getattr(self, "request_id", ""),
            backend=str(getattr(self.state, "acceleration_backend", "Auto")))

    def _load_persistent_incident_seed(
        self,
        existing_result: CalculationResult | None,
    ) -> CalculationResult | None:
        if self.calculation_manifest is None:
            return existing_result
        expected = str(
            self.calculation_manifest.calculation_signatures.get(
                "incident", ""
            )
        )
        existing_signatures = (
            getattr(existing_result, "signatures", None) or {}
        )
        existing_simulation = getattr(existing_result, "simulation", None)
        if (
            expected
            and str(existing_signatures.get("incident", "")) == expected
            and existing_simulation is not None
            and getattr(existing_simulation, "incident_plan", None) is not None
            and getattr(
                existing_simulation, "incident_checkpoints", None
            ) is not None
            and getattr(existing_simulation, "gun_trace", None) is not None
        ):
            return existing_result
        seed = None
        if self.artifact_store is not None:
            try:
                seed = self.artifact_store.get_incident_simulation_seed(
                    self.calculation_manifest
                )
            except Exception:
                # Cache corruption, permissions, or stale external files are
                # a miss, never a reason to discard a completed result.
                pass
        if seed is None and self.allow_project_artifact_fallback:
            try:
                fallback_root = project_artifact_fallback_root()
                if fallback_root is not None:
                    # Use the ordinary verified codec and managed-path checks.
                    # Reads may update cache lock/access metadata; no result is
                    # written here or redirected away from the primary store.
                    fallback = ArtifactStore(
                        fallback_root, quota_bytes=self.artifact_cache_budget_bytes
                    )
                    seed = fallback.get_incident_simulation_seed(
                        self.calculation_manifest
                    )
            except Exception:
                pass
        if seed is None:
            return existing_result
        signatures = dict(existing_signatures)
        signatures["incident"] = expected
        if existing_result is None:
            return CalculationResult(
                simulation=seed,
                energy_filter=None,
                state_snapshot=self.state,
                model_signature=self.model_signature,
                signatures=signatures,
            )
        return replace(
            existing_result,
            simulation=seed,
            signatures=signatures,
        )

    def _persist_incident_seed(self, result: object) -> None:
        if (
            self.artifact_store is None
            or self.calculation_manifest is None
            or not isinstance(result, CalculationResult)
        ):
            return
        signatures = getattr(result, "signatures", None) or {}
        expected = str(
            self.calculation_manifest.calculation_signatures.get(
                "incident", ""
            )
        )
        if str(signatures.get("incident", "")) != expected:
            return
        try:
            self.artifact_store.put_incident_simulation_seed(
                self.calculation_manifest,
                result.simulation,
            )
        except Exception:
            # A cache write is an optional optimisation, never a calculation
            # result dependency.
            return

    @capture_worker_backends
    @input_io.using_state_inputs
    def run(self) -> None:
        started = perf_counter()
        try:
            cancelled = getattr(self, "cancel_event", None)
            if cancelled is not None and cancelled.is_set():
                return
            assert_external_input_inventory_unchanged(self.state, self.external_inputs)
            for requirement in classical_backend_preflight(self.state):
                job_event("backend_preflight", **requirement)
            self.state.active_backend = "CPU"
            self.state._active_backends_used = set()
            # Gun integration and column transport already consume this
            # worker-local callback. Explicit high-accuracy jobs need the
            # same cancellation boundary as live optical tuning.
            if cancelled is not None:
                self.state._tuning_cancelled = cancelled.is_set
            if is_tuning_quality(self.quality):
                prepare_tuning_snapshot(self.state, self.quality,
                    particle_signals=self.particle_tuning or self.section_request is not None)
            if self.section_request is not None or (is_tuning_quality(self.quality) and self.particle_tuning):
                from temsim.simulation_pipeline import calculate_particle_section
                from temsim.detector.particle_readout import measure_particle_detectors
                if self.quality == "High accuracy":
                    self.state._tuning_quality = "High accuracy"
                with job_stage("particle_tuning"):
                    result = calculate_particle_section(self.state,
                        target_z_mm=(self.section_request["target_z_mm"] if self.section_request else None),
                        component_keys=(self.section_request["component_keys"] if self.section_request else ()),
                        existing_result=self.existing_result, progress_callback=self._report_progress)
                result.model_signature = self.model_signature
                result.signatures = dict(result.signatures or {})
                for key in ("request", "section", "particle_tuning"):
                    if key in self.request_signatures:
                        result.signatures[key] = self.request_signatures[key]
                result.particle_signals = measure_particle_detectors(result)
            elif is_tuning_quality(self.quality):
                layout = apply_physical_layout_to_state(self.state)
                with job_stage("ray_transport"):
                    section_options = {}
                    if self.section_request is not None:
                        section_options = {
                            "observation_stop_z_mm": self.section_request["target_z_mm"],
                            "tuning_component_keys": self.section_request["component_keys"],
                        }
                    simulation = run_ray_simulation(
                        self.state, resolved_layout=layout,
                        existing_simulation=getattr(self.existing_result, "simulation", None),
                        optical_only=True,
                        **section_options,
                    )
                lens_crossovers = detect_all_lens_crossovers(
                    [simulation.incident, *simulation.branches.values()],
                    self.state.lenses,
                )
                result = CalculationResult(
                    simulation=simulation,
                    energy_filter=None,
                    state_snapshot=self.state,
                    layout=layout,
                    assembly=self.state._resolved_assembly,
                    lens_crossovers=tuple(lens_crossovers),
                    aperture_stops=aperture_stop_records(self.state),
                    model_signature=self.model_signature,
                    signatures=self.request_signatures,
                )
            else:
                existing_result = self._load_persistent_incident_seed(
                    self.existing_result
                )
                calculation_kwargs = {
                    "progress_callback": self._report_progress,
                }
                if existing_result is not None:
                    calculation_kwargs["existing_result"] = existing_result
                with job_stage("calculation_pipeline"):
                    result = calculate(self.state, **calculation_kwargs)
                if isinstance(result, CalculationResult):
                    result.model_signature = self.model_signature
                    from temsim.detector.particle_readout import measure_particle_detectors
                    if result.simulation is not None:
                        result.particle_signals = measure_particle_detectors(result)
                    assert_external_input_inventory_unchanged(self.state, self.external_inputs)
                    self._persist_incident_seed(result)
            if cancelled is not None and cancelled.is_set():
                return
            assert_external_input_inventory_unchanged(self.state, self.external_inputs)
            if isinstance(result, CalculationResult):
                result.external_inputs = self.external_inputs
                result.performance = dict(result.performance or {},
                    backend_stages=tuple(self.backend_evidence),
                    backend_scope="Last 256 stage calls in this worker, including exact internal reuse; not hardware qualification")
                result.calculation_manifest = self.calculation_manifest
                if result.wave_imaging is not None and self.calculation_manifest is not None:
                    from temsim.physics.wave_imaging import bind_wave_request_manifest
                    result.wave_imaging = bind_wave_request_manifest(result.wave_imaging, self.calculation_manifest)
            job_event("publication_requested", backend=str(getattr(self.state, "active_backend", "unknown")))
            self.signals.result.emit(
                self.generation,
                self.quality,
                result,
                perf_counter() - started,
            )
        except Exception as exc:
            if not getattr(self, "cancel_event", Event()).is_set():
                self.signals.error.emit(self.generation, self.quality, str(exc))
        finally:
            if getattr(self, "_progress_stage", None) is not None:
                job_event("stage_exit", stage=self._progress_stage, outcome="worker_return")
            # A delivered snapshot remains inspectable after the next request
            # cancels this worker's token; cancellation is not result state.
            if hasattr(self.state, "_tuning_cancelled"):
                delattr(self.state, "_tuning_cancelled")
            self.signals.finished.emit(self.generation, self.quality)

    def _report_progress(
        self, completed: int, total: int, stage: str
    ) -> None:
        if self.cancel_event.is_set():
            raise RuntimeError("Calculation cancelled")
        if getattr(self, "_progress_stage", None) != stage:
            if getattr(self, "_progress_stage", None) is not None:
                job_event("stage_exit", stage=self._progress_stage, outcome="next_progress_stage")
            self._progress_stage = str(stage)
            job_event("stage_entry", stage=str(stage), evidence="solver_progress_boundary")
        self.signals.progress.emit(
            self.generation,
            self.quality,
            int(completed),
            int(total),
            str(stage),
        )
        if self.cancel_event.is_set():
            raise RuntimeError("Calculation cancelled")


class SectionFileSignals(QObject):
    result = Signal(str, object)
    error = Signal(str)
    finished = Signal()


class SectionFileWorker(QRunnable):
    """Archive/capture work stays off the GUI thread and shares resource admission."""
    def __init__(self, token, operation, path, result=None, *, automatic=False,
                 maximum_unpacked_bytes, unpacked_size_bytes=None):
        super().__init__()
        self.token, self.operation, self.path = token, operation, Path(path)
        self.payload = result
        self.state = getattr(result, "state_snapshot", None)
        self.automatic = automatic
        self.maximum_unpacked_bytes = int(maximum_unpacked_bytes)
        # A queued archive must not outgrow its admission reservation if the
        # file is replaced before execution. The codec checks the bound again.
        self.read_limit_bytes = (self.maximum_unpacked_bytes if unpacked_size_bytes is None
                                 else min(self.maximum_unpacked_bytes, int(unpacked_size_bytes)))
        self.cancel_event = Event()
        self.signals = SectionFileSignals()
        self.job_input_identity = token
        from temsim.gui.job_coordinator import ResourceClaim
        working_bytes = (2 * 1024**3 if unpacked_size_bytes is None else
                         max(512 * 1024**2, 2 * int(unpacked_size_bytes)))
        self.resource_claim = ResourceClaim(working_bytes=working_bytes)

    @input_io.using_state_inputs
    def run(self):
        from temsim.particle_section_io import (
            archive_section_result, checked_section_archive_info, load_section_result, save_section_result,
        )
        started = perf_counter()
        try:
            if self.cancel_event.is_set():
                return
            if self.operation == "load":
                result = load_section_result(self.path, maximum_unpacked_bytes=self.read_limit_bytes)
                output = {"result": result, "info": result.section_archive_info}
            elif self.automatic:
                output = archive_section_result(self.payload, self.path,
                                                maximum_unpacked_bytes=self.maximum_unpacked_bytes)
            else:
                package = save_section_result(self.payload, self.path, overwrite=True,
                                             maximum_unpacked_bytes=self.maximum_unpacked_bytes)
                output = checked_section_archive_info(self.payload, self.path,
                    maximum_unpacked_bytes=self.maximum_unpacked_bytes,
                    expected_package_digest=package.digest)
            info = output["info"] if self.operation == "load" else output
            info["file_io_seconds"] = max(0.0, perf_counter() - started)
            info["file_io_operation"] = ("load" if self.operation == "load" else
                "verify_existing" if self.automatic and info.get("reused") else "save")
            info["file_io_reused_measurement"] = False
            self.signals.result.emit(self.token, output)
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


class CalculationController(QObject):
    started = Signal(str)
    result_ready = Signal(str, object, float)
    failed = Signal(str, str)
    progress_changed = Signal(str, int, int, str)
    finished = Signal(str)
    section_archive_changed = Signal(object)
    section_loaded = Signal(object, object)

    def __init__(
        self,
        parent=None,
        *,
        high_cache_limit: int = HIGH_ACCURACY_RESULT_CACHE_LIMIT,
        high_cache_budget_bytes: int = HIGH_ACCURACY_RESULT_CACHE_BUDGET_BYTES,
        tuning_cache_limit: int = TUNING_RESULT_CACHE_LIMIT,
        tuning_cache_budget_bytes: int = TUNING_RESULT_CACHE_BUDGET_BYTES,
        artifact_store: ArtifactStore | None = None,
        artifact_cache_root: str | Path | None = None,
        artifact_cache_budget_bytes: int = (
            PERSISTENT_ARTIFACT_CACHE_BUDGET_BYTES
        ),
        persistent_cache_enabled: bool = True,
    ) -> None:
        super().__init__(parent)
        from temsim.gui.job_coordinator import CoordinatedPool
        self.pool = CoordinatedPool(self)
        self.pool.setMaxThreadCount(1)
        self.section_file_pool = CoordinatedPool(self)
        self._section_file_jobs = {}
        self._section_file_pending = {}
        self._section_archive_records = OrderedDict()
        self._loaded_section_seeds = OrderedDict()
        self._loaded_section_memory = RetainedMemoryLedger()
        self._loaded_tuning_sections = OrderedDict()
        self._loaded_tuning_memory = RetainedMemoryLedger()
        self.section_archive_root = Path(artifact_cache_root or default_artifact_cache_root()) / "particle_sections"
        self._generation = 0
        self._finished_generation = -1
        self._requests = {}
        self._high_requests = {}
        self._high_cache: OrderedDict[str, CalculationResult] = OrderedDict()
        self._high_cache_memory = RetainedMemoryLedger()
        self._high_cache_entry_bytes: dict[str, int] = {}
        self._high_cache_limit = max(1, int(high_cache_limit))
        self._high_cache_budget_bytes = max(
            0, int(high_cache_budget_bytes)
        )
        self._tuning_cache: OrderedDict[tuple[str, str], CalculationResult] = OrderedDict()
        self._tuning_cache_memory = RetainedMemoryLedger()
        self._tuning_cache_limit = max(1, int(tuning_cache_limit))
        self._tuning_cache_budget_bytes = max(0, int(tuning_cache_budget_bytes))
        self._cache_hits = {"high": 0, "tuning": 0}
        self._cache_misses = {"high": 0, "tuning": 0}
        self._running_high_key: str | None = None
        self._running_high_generation: int | None = None
        self._cancel_event = Event()
        self._tuning_seeds = {}
        self.pool.coordinator.register_retained(self, "retained_roots")
        self._allow_project_artifact_fallback = bool(
            persistent_cache_enabled
            and artifact_store is None
            and artifact_cache_root is None
        )
        self._artifact_cache_budget_bytes = int(artifact_cache_budget_bytes)
        self._artifact_store = artifact_store
        if self._artifact_store is None and persistent_cache_enabled:
            try:
                self._artifact_store = ArtifactStore(
                    artifact_cache_root or default_artifact_cache_root(),
                    quota_bytes=int(artifact_cache_budget_bytes),
                )
            except Exception:
                # A read-only home, corrupt cache, or invalid environment must
                # never prevent the simulator from starting.
                self._artifact_store = None

    @property
    def generation(self) -> int:
        """Read-only job token; changes on submission or invalidation."""
        return self._generation

    @property
    def has_pending_requests(self):
        return any(not item["cancel"].is_set() for item in self._requests.values())

    def retained_roots(self):
        return (self._high_cache, self._tuning_cache, self._tuning_seeds, self._loaded_section_seeds,
                self._loaded_tuning_sections,
                tuple(worker.payload for worker in self._section_file_jobs.values()))

    def archive_completed_section(self, result, *, path=None):
        """Queue only an already accepted completed result; never execute transport."""
        from temsim.particle_section_io import section_archive_summary
        info = section_archive_summary(result)
        automatic = path is None
        destination = self.section_archive_root if automatic else Path(path)
        identity = info["identity"]
        previous = self._section_archive_records.get(identity)
        if (automatic and previous is not None and self._section_archive_record_current(previous)
                and not self._section_archive_replacement_pending(previous["path"])):
            self.section_archive_changed.emit(dict(previous, status="saved", reused=True,
                file_io_reused_measurement=True))
            return identity
        if previous is not None and not self._section_archive_record_current(previous):
            self._section_archive_records.pop(identity, None)
        pending_key = (identity, os.path.normcase(str(destination.resolve())), automatic)
        if pending_key in self._section_file_pending:
            return identity
        if not automatic:
            # A queued replacement already makes the old association unsafe:
            # another GUI cache hit must not claim its old contents will persist.
            self._forget_section_archive_path(destination)
        token = uuid4().hex
        self._section_file_pending[pending_key] = token
        worker = SectionFileWorker(token, "save", destination, result, automatic=automatic,
                                   maximum_unpacked_bytes=self.section_file_pool.coordinator.ram_budget_bytes)
        self._section_file_jobs[token] = worker
        self.section_archive_changed.emit(dict(info, status="saving", automatic=automatic))
        worker.signals.result.connect(self._section_file_completed)
        worker.signals.error.connect(lambda message: self._section_file_failed(token, info, message))
        worker.signals.finished.connect(lambda: self._section_file_finished(token))
        self.section_file_pool.start(worker)
        return identity

    @staticmethod
    def _section_archive_record_current(info):
        from temsim.particle_section_io import section_file_fingerprint
        try:
            return (bool(info.get("_package_digest"))
                    and tuple(info.get("_file_fingerprint", ())) == section_file_fingerprint(info["path"]))
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def _forget_section_archive_path(self, path):
        from temsim.particle_section_io import section_file_fingerprint
        resolved = os.path.normcase(str(Path(path).resolve()))
        try:
            file_identity = section_file_fingerprint(path)[:2]
        except (OSError, ValueError):
            file_identity = None
        for identity, record in tuple(self._section_archive_records.items()):
            fingerprint = record.get("_file_fingerprint", ())
            if (os.path.normcase(str(Path(record["path"]).resolve())) == resolved
                    or (file_identity is not None and tuple(fingerprint[:2]) == file_identity)):
                del self._section_archive_records[identity]

    def _remember_section_archive(self, info):
        if not info.get("path"):
            return False
        self._forget_section_archive_path(info["path"])
        if not self._section_archive_record_current(info):
            return False
        self._section_archive_records[info["identity"]] = dict(info, status="saved")
        self._section_archive_records.move_to_end(info["identity"])
        while len(self._section_archive_records) > 128:
            self._section_archive_records.popitem(last=False)
        return True

    def _section_archive_replacement_pending(self, path):
        target = Path(path)
        resolved = os.path.normcase(str(target.resolve()))
        for worker in self._section_file_jobs.values():
            if worker.operation != "save" or worker.automatic:
                continue
            if os.path.normcase(str(worker.path.resolve())) == resolved:
                return True
            try:
                if target.samefile(worker.path):
                    return True
            except OSError:
                pass
        return False

    def load_section_archive(self, path):
        from temsim.working_point import WorkingPointArchiveIndex
        maximum = self.section_file_pool.coordinator.ram_budget_bytes
        index = WorkingPointArchiveIndex.read(path, maximum_unpacked_bytes=maximum)
        token = uuid4().hex
        worker = SectionFileWorker(token, "load", path, maximum_unpacked_bytes=maximum,
                                   unpacked_size_bytes=index.unpacked_size_bytes)
        self._section_file_jobs[token] = worker
        info = {"path": str(Path(path).resolve()), "identity": token}
        self.section_archive_changed.emit(dict(info, status="loading"))
        worker.signals.result.connect(self._section_file_completed)
        worker.signals.error.connect(lambda message: self._section_file_failed(token, info, message))
        worker.signals.finished.connect(lambda: self._section_file_finished(token))
        self.section_file_pool.start(worker)

    def _section_file_completed(self, token, output):
        worker = self._section_file_jobs.get(token)
        if worker is None:
            return
        if worker.operation == "load":
            if not self._section_archive_record_current(output["info"]):
                self._section_file_failed(token, output["info"], "Section archive changed after loading; load it again")
                return
            try:
                self.retain_section_seed(output["result"])
            except (ValueError, TypeError, RuntimeError) as exc:
                self._section_file_failed(token, output["info"], str(exc))
                return
            self.section_loaded.emit(output["result"], output["info"])
            return
        info = dict(output, status="saved")
        if not self._remember_section_archive(info):
            self._section_file_failed(token, info, "Section archive changed after saving; save the result again")
            return
        self.section_archive_changed.emit(info)

    def _section_file_failed(self, token, info, message):
        worker = self._section_file_jobs.get(token)
        if worker is not None:
            # Loading displays the operation token until the archive has been
            # admitted. Budget/admission failures occur after decoding exposes
            # a different archive identity, which must not hide this error.
            identity = token if worker.operation == "load" else info.get("identity")
            self.section_archive_changed.emit(dict(info, identity=identity, status="failed", error=str(message),
                                                   operation=worker.operation))

    def _section_file_finished(self, token):
        self._section_file_jobs.pop(token, None)
        for key, pending in tuple(self._section_file_pending.items()):
            if pending == token:
                del self._section_file_pending[key]

    def _request_active(self, generation):
        item = self._requests.get(generation)
        if item is not None:
            return not item["cancel"].is_set()
        return generation == self._generation and generation > self._finished_generation and not self._cancel_event.is_set()

    @property
    def artifact_store(self) -> ArtifactStore | None:
        """Return the shared persistent store for detached calculations."""

        return self._artifact_store

    def configure_cache(
        self,
        *,
        high_cache_budget_bytes: int | None = None,
        tuning_cache_budget_bytes: int | None = None,
        high_cache_limit: int | None = None,
        tuning_cache_limit: int | None = None,
        disk_cache_budget_bytes: int | None = None,
    ) -> None:
        """Change bounded retention without changing results or running work.

        A zero RAM budget disables that history. Reductions evict least-recent
        entries only; displayed results and worker-owned seeds remain valid.
        Disk quota reductions take effect on the next cache write, so applying
        preferences never walks or prunes a large cache on the GUI thread.
        """

        requested = {
            "_high_cache_budget_bytes": (high_cache_budget_bytes, 0),
            "_tuning_cache_budget_bytes": (tuning_cache_budget_bytes, 0),
            "_high_cache_limit": (high_cache_limit, 1),
            "_tuning_cache_limit": (tuning_cache_limit, 1),
        }
        validated = {}
        for name, (value, minimum) in requested.items():
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                    raise ValueError(f"{name.removeprefix('_')} must be an integer >= {minimum}")
                validated[name] = int(value)
        if disk_cache_budget_bytes is not None:
            if (isinstance(disk_cache_budget_bytes, bool)
                    or not isinstance(disk_cache_budget_bytes, (int, np.integer))
                    or disk_cache_budget_bytes <= 0):
                raise ValueError("disk_cache_budget_bytes must be a positive integer")
        for name, value in validated.items():
            setattr(self, name, value)
        if disk_cache_budget_bytes is not None:
            self._artifact_cache_budget_bytes = int(disk_cache_budget_bytes)
            if self._artifact_store is not None:
                self._artifact_store.set_quota_bytes(int(disk_cache_budget_bytes))
        while self._loaded_section_seeds and (self._loaded_section_memory.total_bytes > self._high_cache_budget_bytes
                or len(self._loaded_section_seeds) > self._high_cache_limit):
            key, _ = self._loaded_section_seeds.popitem(last=False)
            self._loaded_section_memory.remove(key)
        while self._loaded_tuning_sections and (self._loaded_tuning_memory.total_bytes > self._tuning_cache_budget_bytes
                or len(self._loaded_tuning_sections) > self._tuning_cache_limit):
            key, _ = self._loaded_tuning_sections.popitem(last=False)
            self._loaded_tuning_memory.remove(key)
        self._trim_high_cache()
        self._trim_tuning_cache()

    def cache_statistics(self) -> dict[str, int | bool]:
        """Return cheap retention estimates, without reading arrays or disk."""

        store = self._artifact_store
        return {
            "high_budget_bytes": self._high_cache_budget_bytes,
            "high_used_bytes": self._high_cache_memory.total_bytes + self._loaded_section_memory.total_bytes,
            "loaded_section_entries": len(self._loaded_section_seeds) + len(self._loaded_tuning_sections),
            "high_entries": len(self._high_cache),
            "high_limit": self._high_cache_limit,
            "high_hits": self._cache_hits["high"],
            "high_misses": self._cache_misses["high"],
            "tuning_budget_bytes": self._tuning_cache_budget_bytes,
            "tuning_used_bytes": self._tuning_cache_memory.total_bytes + self._loaded_tuning_memory.total_bytes,
            "tuning_entries": len(self._tuning_cache),
            "tuning_limit": self._tuning_cache_limit,
            "tuning_hits": self._cache_hits["tuning"],
            "tuning_misses": self._cache_misses["tuning"],
            "disk_enabled": store is not None,
            "disk_budget_bytes": int(store.quota_bytes) if store is not None else 0,
        }

    def clear_memory_cache(self) -> None:
        """Release history references, not displayed products or active jobs."""

        self._high_cache.clear()
        self._high_cache_entry_bytes.clear()
        self._high_cache_memory.clear()
        self._tuning_cache.clear()
        self._loaded_section_seeds.clear()
        self._loaded_section_memory.clear()
        self._loaded_tuning_sections.clear()
        self._loaded_tuning_memory.clear()
        self._tuning_cache_memory.clear()
        self._tuning_seeds.clear()

    def _cache_tuning_result(self, quality: str, result: CalculationResult) -> None:
        signatures = getattr(result, "signatures", None) or {}
        key = (quality, str(signatures.get("request", "")))
        inventory = retained_memory_inventory(result)
        retained_bytes = sum(inventory.values())
        if retained_bytes > self._tuning_cache_budget_bytes:
            return
        self._tuning_cache_memory.replace_inventory(key, inventory)
        self._tuning_cache[key] = result
        self._tuning_cache.move_to_end(key)
        self._trim_tuning_cache()

    def _trim_tuning_cache(self, *, budget_bytes: int | None = None) -> None:
        total_budget = self._tuning_cache_budget_bytes if budget_bytes is None else min(self._tuning_cache_budget_bytes, int(budget_bytes))
        budget = max(0, total_budget - self._loaded_tuning_memory.total_bytes)
        while self._tuning_cache and (
            len(self._tuning_cache) + len(self._loaded_tuning_sections) > self._tuning_cache_limit
            or self._tuning_cache_memory.total_bytes > budget
        ):
            key, _result = self._tuning_cache.popitem(last=False)
            self._tuning_cache_memory.remove(key)
        # Seed ownership follows the same byte budget, never an unbounded
        # second set of references to otherwise evicted histories.
        self._tuning_seeds.clear()
        for (quality, _request), result in self._tuning_cache.items():
            self._tuning_seeds[quality] = result

    def completed_high_accuracy_results(self) -> tuple[CalculationResult, ...]:
        """Return immutable result references that may seed detached work."""

        return tuple(self._high_cache.values())

    def retain_section_seed(self, result: CalculationResult) -> None:
        """Retain an executed restart seed under its quality's shared cache budget."""
        simulation = getattr(result, "simulation", None)
        if getattr(simulation, "section_checkpoint", None) is None:
            raise ValueError("The file contains no executed section checkpoint")
        quality = simulation.metrics.get("tuning_quality")
        if quality not in {"Preview", "Medium", "High accuracy"}:
            raise ValueError("Unknown particle-section tuning quality")
        inventory = retained_memory_inventory(result)
        size = sum(inventory.values())
        budget = self._high_cache_budget_bytes if quality == "High accuracy" else self._tuning_cache_budget_bytes
        if size > budget:
            raise ValueError(f"This section exceeds the {quality} cache budget; increase the cache limit before loading")
        # A loaded restart package lacks optional completed products. It must
        # never enter either complete-result cache and create a false hit.
        from temsim.particle_section_io import section_archive_identity
        key = (quality, section_archive_identity(result))
        seeds = self._loaded_section_seeds if quality == "High accuracy" else self._loaded_tuning_sections
        ledger = self._loaded_section_memory if quality == "High accuracy" else self._loaded_tuning_memory
        limit = self._high_cache_limit if quality == "High accuracy" else self._tuning_cache_limit
        seeds[key] = result
        seeds.move_to_end(key)
        ledger.replace_inventory(key, inventory)
        while len(seeds) > limit or ledger.total_bytes > budget:
            removed, _ = seeds.popitem(last=False)
            ledger.remove(removed)
        if quality == "High accuracy":
            self._trim_high_cache()
        else:
            self._trim_tuning_cache()
        info = getattr(result, "section_archive_info", None)
        if info is not None and info.get("path"):
            self._remember_section_archive(info)

    @staticmethod
    def build_calculation_manifest(
        state,
        ray_count: int,
        step_mm: float,
        *,
        selection=None,
    ) -> CalculationManifest:
        """Freeze the same High-accuracy request used by ``submit``."""

        return capture_calculation_manifest(
            state,
            selection=selection,
            ray_count=ray_count,
            step_mm=step_mm,
        )

    @staticmethod
    def persist_incident_checkpoints(
        store: ArtifactStore,
        manifest: CalculationManifest,
        result: CalculationResult,
    ) -> str | None:
        """Persist the restartable incident phase-space checkpoint subset."""

        signatures = getattr(result, "signatures", None) or {}
        expected = str(manifest.calculation_signatures.get("incident", ""))
        if not expected or str(signatures.get("incident", "")) != expected:
            raise ValueError("Result does not belong to this calculation manifest")
        simulation = getattr(result, "simulation", None)
        checkpoints = getattr(simulation, "incident_checkpoints", None)
        if checkpoints is None:
            return None
        return store.put_propagation_checkpoints(manifest, checkpoints)

    @staticmethod
    def load_persisted_incident_checkpoints(
        store: ArtifactStore,
        manifest: CalculationManifest,
    ):
        """Load safe numeric checkpoints for a later calculation worker."""

        return store.get_propagation_checkpoints(manifest)

    @staticmethod
    def persist_incident_seed(
        store: ArtifactStore,
        manifest: CalculationManifest,
        result: CalculationResult,
    ) -> str:
        """Persist the complete restart seed consumed by ``simulation.run``."""

        signatures = getattr(result, "signatures", None) or {}
        expected = str(manifest.calculation_signatures.get("incident", ""))
        if str(signatures.get("incident", "")) != expected:
            raise ValueError("Result does not belong to this calculation manifest")
        return store.put_incident_simulation_seed(
            manifest, result.simulation
        )

    @staticmethod
    def load_persisted_incident_seed(
        store: ArtifactStore,
        manifest: CalculationManifest,
    ):
        """Restore the complete safe numeric seed used by ``simulation.run``."""

        return store.get_incident_simulation_seed(manifest)

    @staticmethod
    def _calculation_snapshot(state, quality, ray_count, step_mm):
        from temsim.optics.electron_gun.source_policy import require_physical_gun_source
        require_physical_gun_source(getattr(state, "electron_gun", None))
        from temsim.parameter_registry import unmapped_public_inputs
        if quality == "High accuracy" or unmapped_public_inputs(state):
            from temsim.optics.model import State
            if isinstance(state, State):
                from temsim.instrument_snapshot import encode_instrument, decode_instrument
                from temsim.gui.calculation_request import apply_request_numerics
                return apply_request_numerics(decode_instrument(encode_instrument(state)),
                                              quality, ray_count, step_mm)
        return reconstruct_calculation_state(
            type(state), state.to_dict(),
            getattr(state, "_resolved_assembly", None),
            getattr(state, "simulation_time_s", None),
            quality, ray_count, step_mm,
        )

    def _cache_result(self, result: CalculationResult) -> None:
        signatures = getattr(result, "signatures", None) or {}
        key = str(signatures.get("request", ""))
        if not key:
            return
        inventory = retained_memory_inventory(result)
        retained_bytes = sum(inventory.values())
        if retained_bytes > self._high_cache_budget_bytes:
            # The workspace still owns and displays this result. Do not let
            # one exceptionally large bundle erase several useful history
            # entries or remain pinned by the reuse cache as well.
            return
        self._high_cache_memory.replace_inventory(key, inventory)
        self._high_cache[key] = result
        self._high_cache_entry_bytes[key] = retained_bytes
        self._high_cache.move_to_end(key)
        self._trim_high_cache()

    def refresh_cached_result_metadata(
        self, result: CalculationResult | None
    ) -> None:
        """Refresh size metadata after a displayed result is enriched."""

        if result is None:
            return
        signatures = getattr(result, "signatures", None) or {}
        key = str(signatures.get("request", ""))
        if not key or self._high_cache.get(key) is not result:
            return
        inventory = retained_memory_inventory(result)
        retained_bytes = sum(inventory.values())
        if retained_bytes > self._high_cache_budget_bytes:
            self._high_cache.pop(key, None)
            self._high_cache_entry_bytes.pop(key, None)
            self._high_cache_memory.remove(key)
            return
        self._high_cache_memory.replace_inventory(key, inventory)
        self._high_cache_entry_bytes[key] = retained_bytes
        self._trim_high_cache()

    def _high_cache_bytes(self) -> int:
        return self._high_cache_memory.total_bytes

    def _trim_high_cache(
        self,
        *,
        budget_bytes: int | None = None,
        preserve_key: str | None = None,
    ) -> None:
        total_budget = self._high_cache_budget_bytes if budget_bytes is None else min(self._high_cache_budget_bytes, int(budget_bytes))
        budget = max(0, total_budget - self._loaded_section_memory.total_bytes)
        while self._high_cache and (
            len(self._high_cache) + len(self._loaded_section_seeds) > self._high_cache_limit
            or self._high_cache_bytes() > budget
        ):
            removable = next(
                (
                    key
                    for key in self._high_cache
                    if key != preserve_key
                ),
                None,
            )
            if removable is None:
                break
            del self._high_cache[removable]
            self._high_cache_entry_bytes.pop(removable, None)
            self._high_cache_memory.remove(removable)

    @staticmethod
    def _seed_reuse_score(
        result: CalculationResult,
        signatures: dict[str, str],
    ) -> int:
        weights = {
            "incident": 128,
            "column": 64,
            "wave_source": 64,
            "wave": 32,
            "elastic": 16,
            "stem": 16,
            "eds": 8,
            "energy_filter": 4,
            "scan_geometry": 2,
            "scan_ray_paths": 2,
            "sample_region": 2,
            "sample_downstream": 1,
        }
        summary = summarise_calculation_result(result)
        valid_products = (
            summary.available_stage_keys | summary.terminal_stage_keys
        )
        exact_score = sum(
            weights.get(product, 1)
            for product in matching_products(
                getattr(result, "signatures", None), signatures
            )
            if product in valid_products
        )
        simulation = getattr(result, "simulation", None)
        checkpoints = getattr(simulation, "incident_checkpoints", None)
        can_try_prefix = bool(
            getattr(simulation, "incident_plan", None) is not None
            and getattr(simulation, "gun_trace", None) is not None
            and np.size(getattr(checkpoints, "z_mm", ())) > 0
        )
        # A lens/raster edit can invalidate every complete-product signature
        # while leaving the gun-to-edit prefix unchanged. Pass a candidate to
        # run(), which validates the source and exact common integration nodes
        # before resuming. This is NOT permission to reuse any whole product.
        # Any exact-product match outranks a checkpoint-only candidate.
        return 2 * exact_score + int(can_try_prefix)

    def _best_seed_entry(
        self, signatures: dict[str, str]
    ) -> tuple[str, CalculationResult] | None:
        entry = self._peek_best_seed_entry(signatures)
        if entry is not None:
            self._high_cache.move_to_end(entry[0])
        return entry

    def _section_seed_rank(self, result, signatures, model_signature):
        """Rank restart candidates; the transport solver still validates reuse."""
        simulation = getattr(result, "simulation", None)
        metrics = getattr(simulation, "metrics", {}) or {}
        depth = metrics.get("section_resumable_through_z_mm", metrics.get("section_target_z_mm"))
        if (getattr(simulation, "section_checkpoint", None) is None
                or not isinstance(depth, (int, float)) or not math.isfinite(depth)):
            depth = -math.inf
        return (self._seed_reuse_score(result, signatures),
                bool(model_signature and getattr(result, "model_signature", None) == model_signature),
                float(depth))

    def _peek_best_seed_entry(
        self, signatures: dict[str, str]
    ) -> tuple[str, CalculationResult] | None:
        """Inspect the best reusable seed without changing LRU recency."""

        if not self._high_cache:
            return None
        key, result = max(
            reversed(tuple(self._high_cache.items())),
            key=lambda item: self._seed_reuse_score(item[1], signatures),
        )
        if self._seed_reuse_score(result, signatures) <= 0:
            return None
        return key, result

    @staticmethod
    def _requested_design_stage_keys(
        state,
        *,
        has_sample_region: bool = False,
    ) -> frozenset[str]:
        """Return the products requested by the current microscope mode."""

        requested = {"incident", "column", "diagnostics"}
        sample = state.sample
        specimen_active = specimen_interactions_active(sample)
        eds_requested = bool(
            specimen_active and getattr(sample, "eds_enabled", False)
        )
        scan = state.ac_deflector
        descan = state.descan_deflector
        stem_requested = bool(scan.enabled and scan.scan_enabled)
        scan_geometry_requested = bool(
            stem_requested
            or (descan.enabled and descan.scan_enabled)
        )
        if scan_geometry_requested:
            requested.add("scan_geometry")
        if stem_requested:
            requested.add("scan_ray_paths")
        geometric_transport_requested = bool(
            stem_requested
            and specimen_active
            and str(getattr(sample, "specimen_mode", "atomic"))
            .strip()
            .lower()
            in {"atomic", "reference"}
            and not bool(getattr(sample, "stem_wave_enabled", False))
            and any(
                bool(getattr(detector, "inserted", False))
                for detector in state.stem_detectors
            )
        )
        if eds_requested or geometric_transport_requested:
            requested.add("elastic")
        if geometric_transport_requested:
            requested.add("sample_downstream")
        if eds_requested:
            requested.add("eds")
        if tem_wave_imaging_enabled(state):
            requested.update(("wave_source", "wave"))
        if stem_requested and getattr(sample, "stem_image_enabled", True):
            requested.add("stem")
            if bool(
                getattr(sample, "stem_fourdstem_enabled", False)
                and getattr(sample, "stem_wave_enabled", False)
                and str(getattr(state, "illumination_mode", "")).upper()
                == "STEM"
            ):
                requested.update((
                    "fourdstem_cube",
                    "fourdstem_virtual_detectors",
                    "fourdstem_physical_recording",
                ))
        if bool(
            getattr(getattr(state, "energy_filter", None), "enabled", False)
        ):
            requested.add("energy_filter")
        if (
            eds_requested
            and str(getattr(sample, "eds_transport_mode", ""))
            == "elastic_monte_carlo"
            and has_sample_region
        ):
            requested.update(("sample_region", "sample_downstream"))
        return frozenset(requested)

    def describe_high_accuracy_reuse(
        self,
        state,
        ray_count: int,
        step_mm: float,
        *,
        completed_summary: ResultProductSummary | None = None,
    ) -> HighAccuracyReusePlan:
        """Describe one request without mutating cache order or results."""

        request_signatures = calculation_signatures_for_request(
            state,
            ray_count=ray_count,
            step_mm=step_mm,
            minimum_history_step_mm=0.5,
        )
        request_key = request_signatures["request"]

        display_summary = (
            completed_summary
            if completed_summary is not None
            and completed_summary.request_signature == request_key
            else None
        )
        candidates = list(self._high_cache.items())
        source_key = ""
        source_result = None
        if display_summary is None:
            source_entry = next(
                (
                    (key, candidate)
                    for key, candidate in reversed(candidates)
                    if key == request_key
                ),
                None,
            )
            if source_entry is None and candidates:
                source_entry = max(
                    reversed(candidates),
                    key=lambda item: self._seed_reuse_score(
                        item[1], request_signatures
                    ),
                )
                source_result = source_entry[1]
                if self._seed_reuse_score(
                    source_result, request_signatures
                ) <= 0:
                    source_entry = None
                    source_result = None
            if source_entry is not None:
                source_key, source_result = source_entry

        if source_result is not None:
            source_signatures = getattr(source_result, "signatures", None) or {}
            source_request = str(source_signatures.get("request", ""))
            if source_request != request_key:
                estimate = estimate_calculation_memory_bytes(
                    state,
                    "High accuracy",
                    ray_count,
                    step_mm,
                )
                available = max(
                    0,
                    HIGH_ACCURACY_MEMORY_BUDGET_BYTES - int(estimate),
                )
                retained_bytes = self._high_cache_entry_bytes.get(source_key)
                if retained_bytes is None:
                    retained_bytes = estimate_result_cache_bytes(source_result)
                if retained_bytes > available:
                    # submit() would evict this seed before starting work.
                    # Do not advertise a reuse path that cannot be retained.
                    source_result = None

        summary = display_summary or (
            summarise_calculation_result(source_result)
            if source_result is not None
            else None
        )
        requested = self._requested_design_stage_keys(
            state,
            has_sample_region=bool(
                summary is not None
                and "sample_region" in summary.available_stage_keys
            ),
        )
        statuses = evaluate_product_statuses(
            request_signatures,
            summary,
            requested_stage_keys=requested,
        )
        return HighAccuracyReusePlan(
            request_signatures=request_signatures,
            product_statuses=statuses,
            source_request_signature=(
                summary.request_signature if summary is not None else ""
            ),
        )

    def _make_room_for_calculation(
        self,
        estimate_bytes: int,
        seed_entry: tuple[str, CalculationResult] | None,
    ) -> CalculationResult | None:
        available = max(
            0,
            HIGH_ACCURACY_MEMORY_BUDGET_BYTES - int(estimate_bytes),
        )
        # Tuning history competes for the same host memory headroom. Evict
        # cheap preview history before discarding expensive reusable seeds.
        for seeds, ledger in ((self._loaded_tuning_sections, self._loaded_tuning_memory),
                              (self._loaded_section_seeds, self._loaded_section_memory)):
            while seeds and self._loaded_tuning_memory.total_bytes + self._loaded_section_memory.total_bytes > available:
                key, _ = seeds.popitem(last=False)
                ledger.remove(key)
        self._trim_tuning_cache(budget_bytes=max(0, available - self._high_cache_bytes()
                                                  - self._loaded_section_memory.total_bytes))
        available = max(0, available - self._tuning_cache_memory.total_bytes - self._loaded_tuning_memory.total_bytes)
        seed_key = seed_entry[0] if seed_entry is not None else None
        self._trim_high_cache(
            budget_bytes=min(available, self._high_cache_budget_bytes),
            preserve_key=seed_key,
        )
        if self._high_cache_bytes() + self._loaded_section_memory.total_bytes > available and seed_key is not None:
            # A cold calculation is safer than retaining a seed that would
            # push the estimated working set beyond the application budget.
            self._high_cache.pop(seed_key, None)
            self._high_cache_entry_bytes.pop(seed_key, None)
            self._high_cache_memory.remove(seed_key)
            seed_entry = None
            self._trim_high_cache(budget_bytes=available)
        return seed_entry[1] if seed_entry is not None else None

    def _best_seed(
        self, signatures: dict[str, str]
    ) -> CalculationResult | None:
        entry = self._best_seed_entry(signatures)
        return entry[1] if entry is not None else None

    def _begin_request(self, quality="Preview") -> int:
        # Live edits coalesce only live work. Explicit High requests retain
        # their captured state, cancellation token and result association.
        self._cancel_live_requests()
        self._generation += 1
        self._request_input_guard = None
        self._cancel_event = Event()
        self._requests[self._generation] = dict(quality=quality, cancel=self._cancel_event, guard=None, request_id=uuid4().hex)
        return self._generation

    def _trace_request(self, generation, event, **details):
        item = self._requests.get(generation, {})
        self.pool.coordinator.events.record(event, metadata=dict(
            owner=type(self).__name__, generation=generation,
            request_id=item.get("request_id", ""), job_id=item.get("job_id", ""),
            input_identity=item.get("identity", ""), backend="controller",
            working_bytes=0, vram_bytes=0), **details)

    def _cancel_live_requests(self):
        for generation, item in tuple(self._requests.items()):
            if item["quality"] != "High accuracy":
                self._trace_request(generation, "cancellation_request")
                item["cancel"].set()
                del self._requests[generation]
        self.pool.clear(lambda worker: getattr(worker, "quality", "") != "High accuracy")

    def submit_background(
        self, state, quality: str, ray_count: int, step_mm: float,
        *, parent_id=None, section_request=None, particle_tuning=False,
    ) -> None:
        """Capture inputs now and prepare the complete request off-thread.

        High accuracy includes the full graph and pinned input assets. The
        synchronous submit API remains available. Edits immediately after
        capture returns affect a later request, never this request's inputs.
        """
        if quality == "High accuracy" or input_io.archive_payload(state) is not None:
            from copy import copy
            from temsim.physics.source_admission import admit_requested_wave_products
            # This gate only reads controls. A shallow shell contains any lazy
            # State getter aliases; no input buffers or optical fields are copied.
            admit_requested_wave_products(copy(state))
        if not is_tuning_quality(quality) and quality != "High accuracy":
            raise ValueError("Unknown calculation quality")
        from temsim.particle_section_io import normalise_section_request
        section_request = normalise_section_request(section_request, quality=quality)
        if particle_tuning and not is_tuning_quality(quality):
            raise ValueError("Particle live tuning requires Preview or Medium quality")
        if (
            int(ray_count) <= 0
            or not math.isfinite(float(step_mm))
            or float(step_mm) <= 0
        ):
            raise ValueError("Calculation resolution must be positive and finite")
        # Capture may fail before starting a lifecycle. Do not cancel a valid
        # current calculation or leave GUI progress active on such a failure.
        capture_started = perf_counter()
        capture_id = uuid4().hex
        self.pool.coordinator.events.record("capture_entry", request_id=capture_id,
            owner=type(self).__name__, generation=self._generation + 1, quality=quality,
            job_id="", backend="CPU capture", working_bytes=0, vram_bytes=0)
        request = None
        try:
            request = CapturedCalculationRequest.capture(
                state, quality, ray_count, step_mm,
            )
            from temsim.immutable_json import json_digest
            identity = json_digest(dict(graph=request._instrument_graph,
                controls=request._model_state.to_dict(), quality=quality, rays=ray_count, step=step_mm,
                section_request=section_request, particle_tuning=bool(particle_tuning)))
        except Exception as exc:
            if request is not None and request._input_assets is not None:
                request._input_assets.close()
            self.pool.coordinator.events.record("capture_exit", request_id=capture_id,
                outcome="failed", error=str(exc), elapsed_s=perf_counter()-capture_started)
            raise ValueError(f"Could not capture calculation settings: {exc}") from exc
        generation = self._begin_request(quality)
        self._requests[generation]["section_request"] = section_request
        self._requests[generation]["particle_tuning"] = bool(particle_tuning)
        self._requests[generation]["request_id"] = capture_id
        self._trace_request(generation, "capture_exit", outcome="captured", elapsed_s=perf_counter()-capture_started)
        self._requests[generation]["parent_id"] = parent_id
        worker = PreparationWorker(generation, request, self._cancel_event)
        worker.job_input_identity = identity
        worker.request_id = capture_id
        self._requests[generation]["identity"] = worker.job_input_identity
        worker.signals.prepared.connect(self._accept_prepared)
        worker.signals.error.connect(self._accept_error)
        worker.signals.finished.connect(self._accept_finished)
        self.started.emit(quality)
        if self._request_active(generation):
            job_id = self.pool.start(worker)
            if generation in self._requests:
                self._requests[generation]["job_id"] = job_id
        else:
            worker.release_inputs()

    def _accept_prepared(self, generation, quality, prepared) -> None:
        if not self._request_active(generation):
            return
        self._trace_request(generation, "preparation_received")
        if quality == "High accuracy":
            existing = self._high_requests.get(prepared.request_signatures["request"])
            if existing is not None and existing != generation and self._request_active(existing):
                # Same captured inputs already have one execution and one
                # complete result; cancelling this duplicate does not touch it.
                self._accept_finished(generation, quality)
                return
        try:
            requirements = classical_backend_preflight(prepared.snapshot)
            self.progress_changed.emit(quality, 0, 1, "Backend requirements | " + "; ".join(
                f"{row['stage']}: {row['capability']}" for row in requirements))
            self._dispatch_prepared(
                prepared.snapshot, quality, prepared.ray_count, prepared.step_mm,
                model_signature=prepared.model_signature,
                request_signatures=prepared.request_signatures,
                external_inputs=prepared.external_inputs,
                generation=generation, estimate=prepared.memory_estimate, already_started=True,
                calculation_manifest=prepared.calculation_manifest,
            )
        except Exception as exc:
            self._accept_error(generation, quality, str(exc))
            self._accept_finished(generation, quality)

    @input_io.using_state_inputs
    def submit(
        self,
        state,
        quality: str,
        ray_count: int,
        step_mm: float,
        *, section_request=None, particle_tuning=False,
    ) -> None:
        from temsim.optics.electron_gun.source_policy import require_physical_gun_source
        require_physical_gun_source(getattr(state, "electron_gun", None))
        if not is_tuning_quality(quality) and quality != "High accuracy":
            raise ValueError("Unknown calculation quality")
        from temsim.particle_section_io import normalise_section_request
        section_request = normalise_section_request(section_request, quality=quality)
        if particle_tuning and not is_tuning_quality(quality):
            raise ValueError("Particle live tuning requires Preview or Medium quality")
        from temsim.physics.optical_tuning import resolve_tuning_ray_count
        ray_count = resolve_tuning_ray_count(state,quality,ray_count)
        if quality == "High accuracy":
            from temsim.optics.model import State
            if isinstance(state, State):
                from temsim.instrument_snapshot import decode_instrument, encode_instrument
                # Preflight/signature helpers can refresh derived values. They
                # must not touch the live controls before capture or on failure.
                state = decode_instrument(encode_instrument(state))
                from temsim.physics.source_admission import admit_requested_wave_products
                admit_requested_wave_products(state)
        estimate = estimate_calculation_memory_bytes(
            state, quality, ray_count, step_mm
        )
        if (
            quality == "High accuracy"
            and estimate > HIGH_ACCURACY_MEMORY_BUDGET_BYTES
        ):
            wave_estimate = estimate_tem_wave_memory_bytes(state)
            wave_detail = (
                " Optional TEM wave imaging accounts for approximately "
                f"{format_memory_size(wave_estimate)} of this estimate."
                if wave_estimate > 0
                else ""
            )
            raise ValueError(
                "Requested calculation needs approximately "
                f"{format_memory_size(estimate)}, above the "
                f"{format_memory_size(HIGH_ACCURACY_MEMORY_BUDGET_BYTES)} "
                "application budget for a 32 GiB workstation."
                f"{wave_detail} Increase the integration step, or reduce the "
                "ray count, TEM wave grid, specimen thickness, or "
                "frozen-phonon configuration count."
            )
        # State contains immutable MappingProxyType values from the resolved
        # TOML assembly, so generic deepcopy cannot be used. Its canonical
        # persistence boundary produces an independent calculation snapshot.
        external_inputs = capture_external_input_identities(state)
        model_signature = state_model_signature(state)
        snapshot = self._calculation_snapshot(
            state, quality, ray_count, step_mm
        )
        request_signatures = calculation_signatures(snapshot)
        assert_external_input_inventory_unchanged(snapshot, external_inputs)
        request_key = request_signatures["request"]
        if (
            quality == "High accuracy"
            and self._running_high_key == request_key
            and self._request_active(self._running_high_generation)
        ):
            return

        calculation_manifest = None
        if quality == "High accuracy":
            try:
                calculation_manifest = capture_calculation_manifest(
                    snapshot, ray_count=ray_count, step_mm=step_mm,
                )
            except Exception as exc:
                raise ValueError(f"Could not capture the complete working point: {exc}") from exc

        generation = self._begin_request(quality)
        self._requests[generation]["section_request"] = section_request
        self._requests[generation]["particle_tuning"] = bool(particle_tuning)
        self._dispatch_prepared(
            snapshot, quality, ray_count, step_mm,
            model_signature=model_signature, request_signatures=request_signatures,
            generation=generation, estimate=estimate,
            external_inputs=external_inputs,
            calculation_manifest=calculation_manifest,
        )

    def _dispatch_prepared(
        self, snapshot, quality, ray_count, step_mm, *, model_signature,
        request_signatures, generation, estimate, already_started=False,
        external_inputs=None, calculation_manifest=None,
    ) -> None:
        """GUI-thread-only cache lookup and dispatch for either request path."""
        if quality == "High accuracy" and calculation_manifest is None:
            raise ValueError("A complete working point is required before high-accuracy dispatch")
        external_inputs = tuple(capture_external_input_identities(snapshot)
                                if external_inputs is None else external_inputs)
        assert_external_input_inventory_unchanged(snapshot, external_inputs)
        section_request = self._requests[generation].get("section_request")
        particle_tuning = self._requests[generation].get("particle_tuning", False) or section_request is not None
        if particle_tuning:
            from temsim.immutable_json import json_digest
            request_signatures = dict(request_signatures)
            request_signatures["particle_tuning"] = "physical-particle-live-v1"
            request_signatures["request"] = json_digest({"request": request_signatures["request"],
                "particle_tuning": request_signatures["particle_tuning"]})
        if section_request is not None:
            from temsim.particle_section_io import section_request_signatures
            request_signatures = section_request_signatures(request_signatures, section_request)
        self._request_input_guard = (generation, snapshot, external_inputs)
        self._requests[generation]["guard"] = self._request_input_guard
        request_key = request_signatures["request"]
        if quality == "High accuracy":
            self._high_requests[request_key] = generation
        cached = self._high_cache.get(request_key)
        if (quality == "High accuracy" and cached is not None
                and getattr(cached, "external_inputs", None) in (None, external_inputs)):
            self._cache_hits["high"] += 1
            self._high_cache.move_to_end(request_key)
            summary = summarise_calculation_result(cached)
            reused = set(cached.calculated_products) | set(
                cached.reused_products
            )
            for guarded_product in ("sample_region", "sample_downstream"):
                reused.discard(guarded_product)
                if guarded_product in summary.available_stage_keys:
                    reused.add(guarded_product)
            delivered = replace(
                cached,
                model_signature=model_signature,
                cache_hit=True,
                calculated_products=frozenset(),
                reused_products=frozenset(reused),
            )
            if not already_started:
                self.started.emit(quality)

            def deliver_cached() -> None:
                if not self._request_active(generation):
                    return
                self.progress_changed.emit(
                    quality, 1, 1, "Reusing completed high-accuracy result"
                )
                self._accept_result(generation, quality, delivered, 0.0)
                self._accept_finished(generation, quality)

            QTimer.singleShot(0, deliver_cached)
            return

        if is_tuning_quality(quality):
            tuning_key = (quality, request_key)
            cached = self._tuning_cache.get(tuning_key)
            if (cached is not None
                    and getattr(cached, "external_inputs", None) in (None, external_inputs)):
                self._cache_hits["tuning"] += 1
                delivered = replace(
                    cached, model_signature=model_signature, cache_hit=True,
                    calculated_products=frozenset(),
                )
                if not already_started:
                    self.started.emit(quality)

                def deliver_tuning_cache() -> None:
                    if not self._request_active(generation):
                        return
                    self._accept_result(generation, quality, delivered, 0.0)
                    self._accept_finished(generation, quality)

                QTimer.singleShot(0, deliver_tuning_cache)
                return
            self._cache_misses["tuning"] += 1
        else:
            self._cache_misses["high"] += 1

        if quality == "High accuracy":
            seed_entry = self._best_seed_entry(request_signatures)
            if section_request is not None:
                sections = [(key, result) for key, result in reversed(tuple(self._high_cache.items()))
                            if getattr(getattr(result, "simulation", None), "section_checkpoint", None) is not None]
                if sections:
                    candidate_entry = max(sections, key=lambda item:
                        self._section_seed_rank(item[1], request_signatures, model_signature))
                    if (seed_entry is None or self._section_seed_rank(candidate_entry[1], request_signatures, model_signature)
                            > self._section_seed_rank(seed_entry[1], request_signatures, model_signature)):
                        seed_entry = candidate_entry
                        self._high_cache.move_to_end(seed_entry[0])
            existing_result = self._make_room_for_calculation(
                estimate, seed_entry
            )
            if self._loaded_section_seeds:
                rank = (lambda result: self._section_seed_rank(result, request_signatures, model_signature)
                        if section_request is not None else self._seed_reuse_score(result, request_signatures))
                candidate = max(reversed(tuple(self._loaded_section_seeds.values())), key=rank)
                if existing_result is None or rank(candidate) > rank(existing_result):
                    existing_result = candidate
        else:
            existing_result = self._tuning_seeds.get(quality)
            if section_request is not None:
                existing_result = next((entry for (q, _key), entry in reversed(self._tuning_cache.items())
                    if q == quality and getattr(entry.simulation, "section_checkpoint", None) is not None),
                    existing_result)
            loaded = next((entry for (q, _key), entry in reversed(self._loaded_tuning_sections.items())
                           if q == quality), None)
            if loaded is not None:
                existing_sim = getattr(existing_result, "simulation", None)
                previous_z = (getattr(existing_sim, "metrics", {}) or {}).get("section_resumable_through_z_mm", -math.inf)
                loaded_z = loaded.simulation.metrics.get("section_resumable_through_z_mm",
                                                        loaded.simulation.metrics["section_target_z_mm"])
                if getattr(existing_sim, "section_checkpoint", None) is None or loaded_z >= previous_z:
                    existing_result = loaded
        if is_tuning_quality(quality):
            prepare_tuning_snapshot(snapshot, quality, particle_signals=particle_tuning)
            if particle_tuning:
                # Background preview preparation omits a working-set claim for
                # the old optical-only route. Physical particle previews also
                # own specimen descendants and must reserve solver memory.
                estimate = max(int(estimate), estimate_calculation_memory_bytes(
                    snapshot, quality, ray_count, step_mm))
        else:
            self._running_high_key = request_key
            self._running_high_generation = generation
        worker = CalculationWorker(
            generation,
            quality,
            snapshot,
            model_signature=model_signature,
            request_signatures=request_signatures,
            existing_result=existing_result,
            artifact_store=(
                self._artifact_store if quality == "High accuracy" else None
            ),
            calculation_manifest=calculation_manifest,
            external_inputs=external_inputs,
            allow_project_artifact_fallback=self._allow_project_artifact_fallback,
            artifact_cache_budget_bytes=self._artifact_cache_budget_bytes,
            section_request=section_request,
            particle_tuning=particle_tuning,
        )
        from temsim.gui.job_coordinator import ResourceClaim
        worker.resource_claim = ResourceClaim(working_bytes=int(estimate))
        worker.job_input_identity = request_key
        worker.cancel_event = self._requests[generation]["cancel"]
        worker.request_id = self._requests[generation]["request_id"]
        self._requests[generation]["identity"] = request_key
        worker.signals.result.connect(self._accept_result)
        worker.signals.error.connect(self._accept_error)
        worker.signals.progress.connect(self._accept_progress)
        worker.signals.finished.connect(self._accept_finished)
        if not already_started:
            self.started.emit(quality)
        if self._request_active(generation):
            job_id = self.pool.start(worker)
            if generation in self._requests:
                self._requests[generation]["job_id"] = job_id

    def invalidate_pending(self, *, include_explicit=False) -> None:
        """Live invalidation preserves explicit captured-state calculations."""
        self._cancel_live_requests()
        self._generation += 1
        self._request_input_guard = None
        if include_explicit:
            for generation, item in self._requests.items():
                self._trace_request(generation, "cancellation_request")
                item["cancel"].set()
            self._requests.clear()
            self._high_requests.clear()
            self._cancel_event.set()
            self.pool.clear()
            self._running_high_key = None
            self._running_high_generation = None

    def _accept_result(self, generation, quality, result, duration) -> None:
        if self._request_active(generation):
            try:
                guard = self._requests.get(generation, {}).get("guard", getattr(self, "_request_input_guard", None))
                if guard is not None and guard[0] == generation:
                    assert_external_input_inventory_unchanged(guard[1], guard[2])
                inputs = getattr(result, "external_inputs", None)
                if inputs is not None:
                    input_state = getattr(result, "state_snapshot", None)
                    if input_state is None and guard is not None and guard[0] == generation:
                        input_state = guard[1]
                    assert_external_input_inventory_unchanged(input_state, inputs)
            except (OSError, RuntimeError, ValueError) as exc:
                self._accept_error(generation, quality, str(exc))
                return
            if quality == "High accuracy":
                if isinstance(result, CalculationResult):
                    result.working_point_parent_id = self._requests.get(generation, {}).get("parent_id")
                self._cache_result(result)
            else:
                self._cache_tuning_result(quality, result)
            self._trace_request(generation, "publication", quality=quality)
            self.result_ready.emit(quality, result, duration)

    def _accept_error(self, generation, quality, message) -> None:
        if self._request_active(generation):
            self.failed.emit(quality, message)

    def _accept_progress(
        self,
        generation: int,
        quality: str,
        completed: int,
        total: int,
        stage: str,
    ) -> None:
        if self._request_active(generation):
            self.progress_changed.emit(
                quality,
                int(completed),
                int(total),
                str(stage),
            )

    def _accept_finished(self, generation, quality) -> None:
        owned = generation in self._requests
        active = self._request_active(generation)
        if active or owned:
            self._trace_request(generation, "request_terminal", quality=quality)
        self._finished_generation = max(self._finished_generation, generation)
        # Cancellation is also a terminal lifecycle: never retain its guard.
        if getattr(self, "_request_input_guard", None) is not None and self._request_input_guard[0] == generation:
            self._request_input_guard = None
        self._requests.pop(generation, None)
        self._high_requests = {key: value for key, value in self._high_requests.items() if value != generation}
        if quality == "High accuracy" and self._running_high_generation == generation:
            self._running_high_key = None
            self._running_high_generation = None
        if active or owned:
            self.finished.emit(quality)
