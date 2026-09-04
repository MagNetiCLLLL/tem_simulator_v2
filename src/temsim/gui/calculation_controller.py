"""Single-worker asynchronous simulation controller."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import fields as dataclass_fields, is_dataclass, replace
import math
import sys
from time import perf_counter

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal

from temsim.calculation_cache import (
    calculation_signatures,
    matching_products,
    state_model_signature,
)
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.physics.all_lens_crossovers import detect_all_lens_crossovers
from temsim.physics.simulation import (
    MAX_VECTORIZED_POST_RAYS,
    _column_checkpoint_planes,
    run as run_ray_simulation,
)
from temsim.physics.recording_stop import determine_tem_stop_z
from temsim.physics.wave_imaging import estimate_tem_wave_memory_bytes
from temsim.physics.scan_geometry import (
    calculate_scan_geometry,
    calculate_scan_ray_paths,
)
from temsim.simulation_pipeline import (
    CalculationResult,
    aperture_stop_records,
    calculate,
    calculate_stem_scan_frame,
)


HIGH_ACCURACY_MEMORY_BUDGET_BYTES = 24 * 1024**3
HIGH_ACCURACY_RESULT_CACHE_BUDGET_BYTES = 6 * 1024**3
HIGH_ACCURACY_RESULT_CACHE_LIMIT = 4


def estimate_result_cache_bytes(*results: object) -> int:
    """Estimate retained bytes while counting shared array storage once.

    Segmented calculations deliberately share unchanged products between
    :class:`CalculationResult` instances. Summing per-result sizes would
    therefore substantially overstate cache use. Walking all cached roots
    together also handles NumPy views by charging their common root buffer
    only once.
    """

    seen_objects: set[int] = set()
    seen_buffers: set[int] = set()

    def retained_bytes(value: object) -> int:
        if value is None:
            return 0
        object_id = id(value)
        if object_id in seen_objects:
            return 0
        seen_objects.add(object_id)

        if isinstance(value, np.ndarray):
            root = value
            while isinstance(root.base, np.ndarray):
                root = root.base
            buffer_id = id(root)
            storage = 0
            if buffer_id not in seen_buffers:
                seen_buffers.add(buffer_id)
                storage = int(root.nbytes)
            header = int(sys.getsizeof(value))
            if bool(value.flags.owndata):
                # NumPy includes owned storage in __sizeof__; storage is
                # accounted separately so views and shared products dedupe.
                header = max(0, header - int(value.nbytes))
            return header + storage
        if isinstance(value, (str, bytes, bytearray)):
            return int(sys.getsizeof(value))
        if isinstance(value, Mapping):
            return int(sys.getsizeof(value)) + sum(
                retained_bytes(key) + retained_bytes(item)
                for key, item in value.items()
            )
        if isinstance(value, (tuple, list, set, frozenset)):
            return int(sys.getsizeof(value)) + sum(
                retained_bytes(item) for item in value
            )
        if is_dataclass(value) and not isinstance(value, type):
            return int(sys.getsizeof(value)) + sum(
                retained_bytes(getattr(value, field.name))
                for field in dataclass_fields(value)
            )
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            return int(sys.getsizeof(value)) + retained_bytes(attributes)
        return int(sys.getsizeof(value))

    return sum(retained_bytes(result) for result in results)


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
    specimen_mode = str(
        getattr(state.sample, "specimen_mode", "atomic")
    ).strip().lower()
    scattering_active = (
        bool(getattr(state.sample, "inserted", True))
        and bool(getattr(state.sample, "diffraction_enabled", True))
        and specimen_mode == "virtual"
    )
    if specimen_mode == "atomic" and bool(
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
    elif scattering_active:
        from temsim.specimen.virtual import virtual_scattering_branches

        branch_count = len(virtual_scattering_branches(state.sample))
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
    # X/TX/Y/TY are retained as float32 histories for the incident bundle and
    # every post-specimen branch.
    history = (
        pre_history + branch_count * post_history
    ) * rays * 4 * 4
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
        if quality != "Preview"
        else 0
    )
    return int(
        working
        + history
        + history_device_copy
        + checkpoint_peak
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
    ) -> None:
        super().__init__()
        self.generation = generation
        self.quality = quality
        self.state = state
        self.model_signature = str(model_signature)
        self.request_signatures = dict(request_signatures or {})
        self.existing_result = existing_result
        self.signals = WorkerSignals()

    def run(self) -> None:
        started = perf_counter()
        try:
            self.state.active_backend = "CPU"
            self.state._active_backends_used = set()
            if self.quality == "Preview":
                layout = apply_physical_layout_to_state(self.state)
                simulation = run_ray_simulation(
                    self.state, resolved_layout=layout
                )
                lens_crossovers = detect_all_lens_crossovers(
                    [simulation.incident, *simulation.branches.values()],
                    self.state.lenses,
                )
                scan_geometry = calculate_scan_geometry(self.state)
                result = CalculationResult(
                    simulation=simulation,
                    energy_filter=None,
                    state_snapshot=self.state,
                    layout=layout,
                    assembly=self.state._resolved_assembly,
                    scan_geometry=scan_geometry,
                    scan_ray_paths=calculate_scan_ray_paths(
                        self.state,
                        simulation,
                    ),
                    stem_scan=calculate_stem_scan_frame(
                        self.state,
                        simulation,
                    ),
                    lens_crossovers=tuple(lens_crossovers),
                    aperture_stops=aperture_stop_records(self.state),
                    model_signature=self.model_signature,
                    signatures=self.request_signatures,
                )
            else:
                calculation_kwargs = {
                    "progress_callback": self._report_progress,
                }
                if self.existing_result is not None:
                    calculation_kwargs["existing_result"] = self.existing_result
                result = calculate(self.state, **calculation_kwargs)
                if isinstance(result, CalculationResult):
                    result.model_signature = self.model_signature
            self.signals.result.emit(
                self.generation,
                self.quality,
                result,
                perf_counter() - started,
            )
        except Exception as exc:
            self.signals.error.emit(self.generation, self.quality, str(exc))
        finally:
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
    ) -> None:
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self._generation = 0
        self._high_cache: OrderedDict[str, CalculationResult] = OrderedDict()
        self._high_cache_limit = max(1, int(high_cache_limit))
        self._high_cache_budget_bytes = max(
            1, int(high_cache_budget_bytes)
        )
        self._running_high_key: str | None = None

    @staticmethod
    def _calculation_snapshot(state, quality, ray_count, step_mm):
        snapshot = type(state).from_dict(state.to_dict())
        emitter = getattr(snapshot.electron_gun, "emitter", None)
        if emitter is not None:
            emitter.ray_count = int(ray_count)
        else:
            snapshot.electron_gun.ray_count = int(ray_count)
        snapshot.step_mm = float(step_mm)
        snapshot.history_step_mm = max(
            float(step_mm), 2.0 if quality == "Preview" else 0.5
        )
        return snapshot

    def _cache_result(self, result: CalculationResult) -> None:
        signatures = getattr(result, "signatures", None) or {}
        key = str(signatures.get("request", ""))
        if not key:
            return
        if estimate_result_cache_bytes(result) > self._high_cache_budget_bytes:
            # The workspace still owns and displays this result. Do not let
            # one exceptionally large bundle erase several useful history
            # entries or remain pinned by the reuse cache as well.
            return
        self._high_cache[key] = result
        self._high_cache.move_to_end(key)
        self._trim_high_cache()

    def _high_cache_bytes(self) -> int:
        return estimate_result_cache_bytes(*self._high_cache.values())

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
            "scan": 4,
            "sample_region": 2,
        }
        return sum(
            weights.get(product, 1)
            for product in matching_products(
                getattr(result, "signatures", None), signatures
            )
        )

    def _best_seed_entry(
        self, signatures: dict[str, str]
    ) -> tuple[str, CalculationResult] | None:
        if not self._high_cache:
            return None
        key, result = max(
            reversed(tuple(self._high_cache.items())),
            key=lambda item: self._seed_reuse_score(item[1], signatures),
        )
        if self._seed_reuse_score(result, signatures) <= 0:
            return None
        self._high_cache.move_to_end(key)
        return key, result

    def _make_room_for_calculation(
        self,
        estimate_bytes: int,
        seed_entry: tuple[str, CalculationResult] | None,
    ) -> CalculationResult | None:
        available = max(
            0,
            HIGH_ACCURACY_MEMORY_BUDGET_BYTES - int(estimate_bytes),
        )
        seed_key = seed_entry[0] if seed_entry is not None else None
        self._trim_high_cache(
            budget_bytes=min(available, self._high_cache_budget_bytes),
            preserve_key=seed_key,
        )
        if self._high_cache_bytes() > available and seed_key is not None:
            # A cold calculation is safer than retaining a seed that would
            # push the estimated working set beyond the application budget.
            self._high_cache.pop(seed_key, None)
            seed_entry = None
            self._trim_high_cache(budget_bytes=available)
        return seed_entry[1] if seed_entry is not None else None

    def _best_seed(
        self, signatures: dict[str, str]
    ) -> CalculationResult | None:
        entry = self._best_seed_entry(signatures)
        return entry[1] if entry is not None else None

    def submit(
        self,
        state,
        quality: str,
        ray_count: int,
        step_mm: float,
    ) -> None:
        estimate = estimate_calculation_memory_bytes(
            state, quality, ray_count, step_mm
        )
        if (
            quality != "Preview"
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
        if quality != "Preview" and self._running_high_key == request_key:
            return

        self._generation += 1
        generation = self._generation
        self.pool.clear()
        if quality != "Preview" and request_key in self._high_cache:
            cached = self._high_cache[request_key]
            self._high_cache.move_to_end(request_key)
            delivered = replace(
                cached,
                model_signature=model_signature,
                cache_hit=True,
                calculated_products=frozenset(),
                reused_products=frozenset(
                    set(cached.calculated_products)
                    | set(cached.reused_products)
                    | (
                        {"sample_region"}
                        if cached.sample_region is not None
                        else set()
                    )
                ),
            )
            self._high_cache[request_key] = delivered
            self.started.emit(quality)

            def deliver_cached() -> None:
                if generation != self._generation:
                    return
                self.progress_changed.emit(
                    quality, 1, 1, "Reusing completed high-accuracy result"
                )
                self.result_ready.emit(quality, delivered, 0.0)
                self.finished.emit(quality)

            QTimer.singleShot(0, deliver_cached)
            return

        if quality != "Preview":
            seed_entry = self._best_seed_entry(request_signatures)
            existing_result = self._make_room_for_calculation(
                estimate, seed_entry
            )
        else:
            existing_result = None
        if quality == "Preview":
            snapshot.sample.wave_enabled = False
            snapshot.sample.stem_wave_enabled = False
            # Real specimens never receive display-only scattering branches.
            # Explicit Virtual interaction channels remain visible in Preview
            # as well as High accuracy when the user has enabled them.
            if str(snapshot.sample.specimen_mode).strip().lower() != "virtual":
                snapshot.sample.diffraction_enabled = False
        else:
            self._running_high_key = request_key
        worker = CalculationWorker(
            generation,
            quality,
            snapshot,
            model_signature=model_signature,
            request_signatures=request_signatures,
            existing_result=existing_result,
        )
        worker.signals.result.connect(self._accept_result)
        worker.signals.error.connect(self._accept_error)
        worker.signals.progress.connect(self._accept_progress)
        worker.signals.finished.connect(self._accept_finished)
        self.started.emit(quality)
        self.pool.start(worker)

    def invalidate_pending(self) -> None:
        """Ignore queued/running results after the live state has changed."""

        self._generation += 1
        self.pool.clear()
        self._running_high_key = None

    def _accept_result(self, generation, quality, result, duration) -> None:
        if generation == self._generation:
            if quality != "Preview":
                self._cache_result(result)
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
            if quality != "Preview":
                self._running_high_key = None
            self.finished.emit(quality)
