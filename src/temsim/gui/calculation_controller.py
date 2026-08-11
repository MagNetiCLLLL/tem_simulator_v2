"""Single-worker asynchronous simulation controller."""

from __future__ import annotations

import math
from time import perf_counter

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.physics.all_lens_crossovers import detect_all_lens_crossovers
from temsim.physics.simulation import run as run_ray_simulation
from temsim.physics.recording_stop import determine_tem_stop_z
from temsim.physics.scan_geometry import calculate_scan_geometry
from temsim.simulation_pipeline import (
    CalculationResult,
    aperture_stop_records,
    calculate,
    calculate_stem_scan_frame,
)


HIGH_ACCURACY_MEMORY_BUDGET_BYTES = 24 * 1024**3


def estimate_calculation_memory_bytes(
    state, quality: str, ray_count: int, step_mm: float
) -> int:
    """Conservatively estimate peak solver memory for one calculation.

    A 24 GiB application budget leaves approximately 8 GiB for Qt, Python,
    the operating system and allocator overhead on the supported 32 GiB
    workstation configuration.
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

    # Momentum, Larmor rate/gradient, X/Y focusing and temporary ufunc output
    # dominate the integration phase. Nine float64 matrices is deliberately
    # conservative across both the NumPy and Numba paths.
    working = max(pre_nodes, post_nodes) * rays * 8 * 9

    history_step = max(step, 2.0 if quality == "Preview" else 0.5)
    pre_history = int(math.ceil(pre_span / history_step)) + 2
    post_history = int(math.ceil(post_span / history_step)) + 2
    branch_count = (
        3 if bool(getattr(state.sample, "diffraction_enabled", True)) else 1
    )
    # X/TX/Y/TY are retained as float32 histories for the incident bundle and
    # every post-specimen branch.
    history = (
        pre_history + branch_count * post_history
    ) * rays * 4 * 4
    return int(working + history + 512 * 1024**2)


def format_memory_size(byte_count: int) -> str:
    return f"{float(byte_count) / 1024**3:.1f} GiB"


class WorkerSignals(QObject):
    result = Signal(int, str, object, float)
    error = Signal(int, str, str)
    finished = Signal(int, str)


class CalculationWorker(QRunnable):
    def __init__(self, generation: int, quality: str, state) -> None:
        super().__init__()
        self.generation = generation
        self.quality = quality
        self.state = state
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
                result = CalculationResult(
                    simulation=simulation,
                    energy_filter=None,
                    state_snapshot=self.state,
                    layout=layout,
                    assembly=self.state._resolved_assembly,
                    scan_geometry=calculate_scan_geometry(self.state),
                    stem_scan=calculate_stem_scan_frame(
                        self.state,
                        simulation,
                    ),
                    lens_crossovers=tuple(lens_crossovers),
                    aperture_stops=aperture_stop_records(self.state),
                )
            else:
                result = calculate(self.state)
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


class CalculationController(QObject):
    started = Signal(str)
    result_ready = Signal(str, object, float)
    failed = Signal(str, str)
    finished = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self._generation = 0

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
            raise ValueError(
                "Requested calculation needs approximately "
                f"{format_memory_size(estimate)}, above the "
                f"{format_memory_size(HIGH_ACCURACY_MEMORY_BUDGET_BYTES)} "
                "application budget for a 32 GiB workstation. Increase the "
                "integration step or reduce the ray count."
            )
        self._generation += 1
        generation = self._generation
        self.pool.clear()
        # State contains immutable MappingProxyType values from the resolved
        # TOML assembly, so generic deepcopy cannot be used. Its canonical
        # persistence boundary produces an independent calculation snapshot.
        snapshot = type(state).from_dict(state.to_dict())
        emitter = getattr(snapshot.electron_gun, "emitter", None)
        if emitter is not None:
            emitter.ray_count = int(ray_count)
        else:
            snapshot.electron_gun.ray_count = int(ray_count)
        snapshot.step_mm = float(step_mm)
        snapshot.history_step_mm = max(float(step_mm), 2.0 if quality == "Preview" else 0.5)
        if quality == "Preview":
            snapshot.sample.wave_enabled = False
            snapshot.sample.stem_wave_enabled = False
            # The interactive path is a direct-beam optical schematic. Full
            # diffraction branches are retained only while scanning because
            # HAADF/DF/BF pixels must be based on detector interception rather
            # than a display-only synthetic gradient.
            ac_scan = snapshot.ac_deflector
            snapshot.sample.diffraction_enabled = bool(
                ac_scan.enabled and ac_scan.scan_enabled
            )
        worker = CalculationWorker(generation, quality, snapshot)
        worker.signals.result.connect(self._accept_result)
        worker.signals.error.connect(self._accept_error)
        worker.signals.finished.connect(self._accept_finished)
        self.started.emit(quality)
        self.pool.start(worker)

    def invalidate_pending(self) -> None:
        """Ignore queued/running results after the live state has changed."""

        self._generation += 1
        self.pool.clear()

    def _accept_result(self, generation, quality, result, duration) -> None:
        if generation == self._generation:
            self.result_ready.emit(quality, result, duration)

    def _accept_error(self, generation, quality, message) -> None:
        if generation == self._generation:
            self.failed.emit(quality, message)

    def _accept_finished(self, generation, quality) -> None:
        if generation == self._generation:
            self.finished.emit(quality)
