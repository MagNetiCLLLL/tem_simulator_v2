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
    capture_calculation_manifest,
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
    ) * rays * 4 * history_itemsize
    checkpoint_count = len(
        _column_checkpoint_planes(gun_start, sample_z, rays)
    )
    checkpoint_storage = checkpoint_count * rays * 4 * 8
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
        max(pre_history * rays, post_history * peak_post_rays) * 4 * 4
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


class PreparationWorker(QRunnable):
    """Prepare one captured request without blocking the Qt event thread."""

    def __init__(self, generation, request, cancel_event):
        super().__init__()
        self.generation = generation
        self.quality = request.quality
        self.request = request
        self.cancel_event = cancel_event
        self.signals = PreparationSignals()

    def run(self):
        try:
            prepared = self.request.prepare(self.cancel_event)
            if not self.cancel_event.is_set():
                self.signals.prepared.emit(self.generation, self.quality, prepared)
        except PreparationCancelled:
            return
        except Exception as exc:
            if not self.cancel_event.is_set():
                self.signals.error.emit(self.generation, self.quality, str(exc))
                self.signals.finished.emit(self.generation, self.quality)
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
        allow_project_artifact_fallback: bool = False,
        artifact_cache_budget_bytes: int = PERSISTENT_ARTIFACT_CACHE_BUDGET_BYTES,
    ) -> None:
        super().__init__()
        self.generation = generation
        self.quality = quality
        self.state = state
        self.model_signature = str(model_signature)
        self.request_signatures = dict(request_signatures or {})
        self.existing_result = existing_result
        self.artifact_store = artifact_store
        self.calculation_manifest = calculation_manifest
        self.allow_project_artifact_fallback = bool(
            allow_project_artifact_fallback and quality == "High accuracy"
        )
        self.artifact_cache_budget_bytes = int(artifact_cache_budget_bytes)
        self.signals = WorkerSignals()

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

    def run(self) -> None:
        started = perf_counter()
        try:
            cancelled = getattr(self, "cancel_event", None)
            if cancelled is not None and cancelled.is_set():
                return
            self.state.active_backend = "CPU"
            self.state._active_backends_used = set()
            if is_tuning_quality(self.quality):
                prepare_tuning_snapshot(self.state, self.quality)
                if cancelled is not None:
                    self.state._tuning_cancelled = cancelled.is_set
                layout = apply_physical_layout_to_state(self.state)
                simulation = run_ray_simulation(
                    self.state, resolved_layout=layout,
                    existing_simulation=getattr(self.existing_result, "simulation", None),
                    optical_only=True,
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
                result = calculate(self.state, **calculation_kwargs)
                if isinstance(result, CalculationResult):
                    result.model_signature = self.model_signature
                    self._persist_incident_seed(result)
            if cancelled is not None and cancelled.is_set():
                return
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
            # A delivered snapshot remains inspectable after the next request
            # cancels this worker's token; cancellation is not result state.
            if hasattr(self.state, "_tuning_cancelled"):
                delattr(self.state, "_tuning_cancelled")
            self.signals.finished.emit(self.generation, self.quality)

    def _report_progress(
        self, completed: int, total: int, stage: str
    ) -> None:
        self.signals.progress.emit(
            self.generation,
            self.quality,
            int(completed),
            int(total),
            str(stage),
        )


class CalculationController(QObject):
    started = Signal(str)
    result_ready = Signal(str, object, float)
    failed = Signal(str, str)
    progress_changed = Signal(str, int, int, str)
    finished = Signal(str)

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
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self._generation = 0
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
        self._trim_high_cache()
        self._trim_tuning_cache()

    def cache_statistics(self) -> dict[str, int | bool]:
        """Return cheap retention estimates, without reading arrays or disk."""

        store = self._artifact_store
        return {
            "high_budget_bytes": self._high_cache_budget_bytes,
            "high_used_bytes": self._high_cache_memory.total_bytes,
            "high_entries": len(self._high_cache),
            "high_limit": self._high_cache_limit,
            "high_hits": self._cache_hits["high"],
            "high_misses": self._cache_misses["high"],
            "tuning_budget_bytes": self._tuning_cache_budget_bytes,
            "tuning_used_bytes": self._tuning_cache_memory.total_bytes,
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
        budget = self._tuning_cache_budget_bytes if budget_bytes is None else max(0, int(budget_bytes))
        while self._tuning_cache and (
            len(self._tuning_cache) > self._tuning_cache_limit
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
        budget = (
            self._high_cache_budget_bytes
            if budget_bytes is None
            else max(0, int(budget_bytes))
        )
        while self._high_cache and (
            len(self._high_cache) > self._high_cache_limit
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
        if stem_requested:
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
        self._trim_tuning_cache(budget_bytes=max(0, available - self._high_cache_bytes()))
        available = max(0, available - self._tuning_cache_memory.total_bytes)
        seed_key = seed_entry[0] if seed_entry is not None else None
        self._trim_high_cache(
            budget_bytes=min(available, self._high_cache_budget_bytes),
            preserve_key=seed_key,
        )
        if self._high_cache_bytes() > available and seed_key is not None:
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

    def _begin_request(self) -> int:
        self._generation += 1
        self._cancel_event.set()
        self._cancel_event = Event()
        self._running_high_key = None
        self._running_high_generation = None
        self.pool.clear()
        return self._generation

    def submit_background(
        self, state, quality: str, ray_count: int, step_mm: float,
    ) -> None:
        """Capture Preview/Medium now; prepare their full request off-thread.

        High accuracy keeps its existing synchronous validation/deduplication
        contract. Only immutable installed geometry is shared with live State;
        edits immediately after this method returns affect a later request.
        """
        if quality == "High accuracy":
            return self.submit(state, quality, ray_count, step_mm)
        if not is_tuning_quality(quality):
            raise ValueError("Unknown calculation quality")
        if (
            int(ray_count) <= 0
            or not math.isfinite(float(step_mm))
            or float(step_mm) <= 0
        ):
            raise ValueError("Calculation resolution must be positive and finite")
        # Capture may fail before starting a lifecycle. Do not cancel a valid
        # current calculation or leave GUI progress active on such a failure.
        try:
            request = CapturedCalculationRequest.capture(
                state, quality, ray_count, step_mm,
            )
        except Exception as exc:
            raise ValueError(f"Could not capture calculation settings: {exc}") from exc
        generation = self._begin_request()
        worker = PreparationWorker(generation, request, self._cancel_event)
        worker.signals.prepared.connect(self._accept_prepared)
        worker.signals.error.connect(self._accept_error)
        worker.signals.finished.connect(self._accept_finished)
        self.started.emit(quality)
        if generation == self._generation:
            self.pool.start(worker)

    def _accept_prepared(self, generation, quality, prepared) -> None:
        if generation != self._generation or self._cancel_event.is_set():
            return
        try:
            self._dispatch_prepared(
                prepared.snapshot, quality, prepared.ray_count, prepared.step_mm,
                model_signature=prepared.model_signature,
                request_signatures=prepared.request_signatures,
                generation=generation, estimate=0, already_started=True,
            )
        except Exception as exc:
            self._accept_error(generation, quality, str(exc))
            self._accept_finished(generation, quality)

    def submit(
        self,
        state,
        quality: str,
        ray_count: int,
        step_mm: float,
    ) -> None:
        if not is_tuning_quality(quality) and quality != "High accuracy":
            raise ValueError("Unknown calculation quality")
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
        model_signature = state_model_signature(state)
        snapshot = self._calculation_snapshot(
            state, quality, ray_count, step_mm
        )
        request_signatures = calculation_signatures(snapshot)
        request_key = request_signatures["request"]
        if (
            quality == "High accuracy"
            and self._running_high_key == request_key
            and self._running_high_generation == self._generation
        ):
            return

        # Any accepted submission supersedes the previous worker.  Its queued
        # signals are ignored by generation, so its High-accuracy token must
        # be retired here as well; otherwise High -> Preview could leave a
        # stale key that blocks the next identical High request forever.
        generation = self._begin_request()
        self._dispatch_prepared(
            snapshot, quality, ray_count, step_mm,
            model_signature=model_signature, request_signatures=request_signatures,
            generation=generation, estimate=estimate,
        )

    def _dispatch_prepared(
        self, snapshot, quality, ray_count, step_mm, *, model_signature,
        request_signatures, generation, estimate, already_started=False,
    ) -> None:
        """GUI-thread-only cache lookup and dispatch for either request path."""
        request_key = request_signatures["request"]
        if quality == "High accuracy" and request_key in self._high_cache:
            self._cache_hits["high"] += 1
            cached = self._high_cache[request_key]
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
            self._cache_result(delivered)
            if not already_started:
                self.started.emit(quality)

            def deliver_cached() -> None:
                if generation != self._generation:
                    return
                self.progress_changed.emit(
                    quality, 1, 1, "Reusing completed high-accuracy result"
                )
                self.result_ready.emit(quality, delivered, 0.0)
                self._accept_finished(generation, quality)

            QTimer.singleShot(0, deliver_cached)
            return

        if is_tuning_quality(quality):
            tuning_key = (quality, request_key)
            cached = self._tuning_cache.get(tuning_key)
            if cached is not None:
                self._cache_hits["tuning"] += 1
                delivered = replace(
                    cached, model_signature=model_signature, cache_hit=True,
                    calculated_products=frozenset(),
                )
                self._cache_tuning_result(quality, delivered)
                if not already_started:
                    self.started.emit(quality)

                def deliver_tuning_cache() -> None:
                    if generation != self._generation:
                        return
                    self.result_ready.emit(quality, delivered, 0.0)
                    self._accept_finished(generation, quality)

                QTimer.singleShot(0, deliver_tuning_cache)
                return
            self._cache_misses["tuning"] += 1
        else:
            self._cache_misses["high"] += 1

        if quality == "High accuracy":
            seed_entry = self._best_seed_entry(request_signatures)
            existing_result = self._make_room_for_calculation(
                estimate, seed_entry
            )
        else:
            existing_result = self._tuning_seeds.get(quality)
        if is_tuning_quality(quality):
            prepare_tuning_snapshot(snapshot, quality)
        else:
            self._running_high_key = request_key
            self._running_high_generation = generation
        calculation_manifest = None
        if quality == "High accuracy" and (
            self._artifact_store is not None or self._allow_project_artifact_fallback
        ):
            try:
                calculation_manifest = capture_calculation_manifest(
                    snapshot,
                    ray_count=ray_count,
                    step_mm=step_mm,
                )
            except Exception:
                calculation_manifest = None
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
            allow_project_artifact_fallback=self._allow_project_artifact_fallback,
            artifact_cache_budget_bytes=self._artifact_cache_budget_bytes,
        )
        worker.cancel_event = self._cancel_event
        worker.signals.result.connect(self._accept_result)
        worker.signals.error.connect(self._accept_error)
        worker.signals.progress.connect(self._accept_progress)
        worker.signals.finished.connect(self._accept_finished)
        if not already_started:
            self.started.emit(quality)
        if generation == self._generation:
            self.pool.start(worker)

    def invalidate_pending(self) -> None:
        """Ignore queued/running results after the live state has changed."""

        self._generation += 1
        self._cancel_event.set()
        self.pool.clear()
        self._running_high_key = None
        self._running_high_generation = None

    def _accept_result(self, generation, quality, result, duration) -> None:
        if generation == self._generation:
            if quality == "High accuracy":
                self._cache_result(result)
            else:
                self._cache_tuning_result(quality, result)
            self.result_ready.emit(quality, result, duration)

    def _accept_error(self, generation, quality, message) -> None:
        if generation == self._generation:
            self.failed.emit(quality, message)

    def _accept_progress(
        self,
        generation: int,
        quality: str,
        completed: int,
        total: int,
        stage: str,
    ) -> None:
        if generation == self._generation:
            self.progress_changed.emit(
                quality,
                int(completed),
                int(total),
                str(stage),
            )

    def _accept_finished(self, generation, quality) -> None:
        if generation == self._generation:
            if quality == "High accuracy":
                self._running_high_key = None
                self._running_high_generation = None
            self.finished.emit(quality)
