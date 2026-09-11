"""Explicit development request: physical FEG tip -> column -> specimen -> sensor.

Observables are lazy views of the same executed complex state. A cache entry
is tied to the captured instrument, external model contents, source code and
numerics. There is no API for supplying a replacement downstream source.
"""
from collections import OrderedDict
from dataclasses import asdict, dataclass
import math

from temsim.detector.wave_readout import WaveReadoutOptions, read_wave_detector, _apply_recording_stop
from temsim.immutable_json import json_digest
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.electron_gun.tip_coherence import TipWaveNumerics
from temsim.physics.column_wave import _prepare_column, _propagate_column, _component_events, _propagate_column_segmented
from temsim.physics.gun_wave_transport import specimen_entrance_z_mm
from temsim.physics.specimen_wave_transport import _propagate_specimen
from temsim.physics.tip_gun_wave import GunWaveNumerics, build_tip_gun_checkpoint
from temsim.physics.wave_grid import WaveGridNumerics
from temsim.physics.wave_execution import WaveExecutionOptions, InelasticWaveNumerics
from temsim.physics.wave_checkpoint_store import ExecutedWaveStore


@dataclass(frozen=True)
class TipWaveRequest:
    stop: str = "detector"
    source: TipWaveNumerics = TipWaveNumerics()
    gun: GunWaveNumerics = GunWaveNumerics()
    column_step_mm: float = .5
    wave_grid: WaveGridNumerics = WaveGridNumerics()
    readout: WaveReadoutOptions = WaveReadoutOptions()
    detector_pixels: int = 512
    detector_key: str | None = None
    maximum_readout_bytes: int = 72*1024**3
    execution: WaveExecutionOptions = WaveExecutionOptions()
    inelastic: InelasticWaveNumerics = InelasticWaveNumerics()
    tip_time_s: float | None = None

    def validate(self):
        if self.stop not in ("gun_exit", "specimen_entrance", "specimen_exit", "detector"):
            raise ValueError("Unknown tip wave stopping stage")
        self.source.validate(); self.gun.validate(); self.readout.validate(); self.wave_grid.validate()
        self.execution.validate(); self.inelastic.validate()
        if self.tip_time_s is not None and (isinstance(self.tip_time_s, bool) or not math.isfinite(self.tip_time_s)):
            raise ValueError("Tip emission time must be finite")
        if self.detector_key is not None and (not isinstance(self.detector_key, str) or not self.detector_key):
            raise ValueError("Detector selection must be an installed component key")
        if isinstance(self.column_step_mm, bool) or not math.isfinite(self.column_step_mm) or self.column_step_mm <= 0:
            raise ValueError("Column step must be positive and finite")
        if isinstance(self.detector_pixels, bool) or not isinstance(self.detector_pixels, int) or not 2 <= self.detector_pixels <= 8192:
            raise ValueError("Detector pixels must be an integer from 2 to 8192")
        if isinstance(self.maximum_readout_bytes, bool) or not isinstance(self.maximum_readout_bytes, int) or self.maximum_readout_bytes <= 0:
            raise ValueError("Readout memory budget must be a positive integer")
        return self


@dataclass(frozen=True)
class TipWaveResult:
    checkpoint: object
    detector: object | None
    request: TipWaveRequest
    propagation_cache_hit: bool
    instrument_digest: str


_STAGES = OrderedDict()
_MAXIMUM_CACHE_BYTES = 8*1024**3


def _recording_planes(state, key):
    candidates = [d for d in (*getattr(state, "stem_detectors", ()),
                  getattr(state, "fluorescent_screen", None), getattr(state, "camera", None))
                  if d is not None and bool(d.inserted) and d.z_mm > state.sample.z_mm+state.sample.thickness_nm*.5e-6]
    candidates.sort(key=lambda d: d.z_mm)
    if len({d.key for d in candidates}) != len(candidates):
        raise ValueError("Duplicate physical detector identity")
    targets = [d for d in candidates if bool(getattr(d, "readout_enabled", True)) and (key is None or d.key == key)]
    if not targets:
        raise ValueError("Insert and enable the selected physical detector to record a wave result")
    target = targets[0]
    return tuple(d for d in candidates if d.z_mm <= target.z_mm), target


def _retain(key, result, maximum_bytes=_MAXIMUM_CACHE_BYTES):
    size = sum(m.plane.amplitude.nbytes for m in result.beam.modes)
    if size > maximum_bytes:
        return
    _STAGES[key] = (result, size)
    _STAGES.move_to_end(key)
    while len(_STAGES) > 8 or sum(v[1] for v in _STAGES.values()) > maximum_bytes:
        _STAGES.popitem(last=False)


def simulate_tip_wave(state, request=TipWaveRequest(), *, use_cache=True,
                      cancelled=lambda: False, progress_callback=None):
    """Compute selected stages and observations, preserving mandatory state.

This development product supports arrival-time scan coils and conditional
material-model inelastic waves. Existing ray, EDS and resolved EELS products
retain their established workflows and are not replaced by this entry point.
"""
    request.validate()
    if cancelled():
        raise InterruptedError("Tip wave request cancelled")
    snapshot = capture_instrument_snapshot(state)
    working = snapshot.restore()
    if bool(getattr(working, "equivalent_image_lenses_enabled", False)):
        raise ValueError("Only executed upstream caches may replace distributed optics")
    nano = getattr(working, "nanopulser", None)
    if nano is not None and nano.installed:
        raise ValueError("Installed nanopulser needs time-energy wavepacket propagation")
    # Every discrete upstream column event must have an owner, including a
    # component moved into the gun region by a custom assembly.
    gun_end = working.electron_gun.exit_plane_z_mm
    if _component_events(working, -1e-12, gun_end)[0]:
        raise ValueError("Column deflector events inside the gun need the joint accelerating operator")
    entrance = specimen_entrance_z_mm(working)
    planes, target = _recording_planes(working, request.detector_key) if request.stop == "detector" else ((), None)
    end = {"gun_exit": gun_end, "specimen_entrance": entrance,
           "specimen_exit": working.sample.z_mm+working.sample.thickness_nm*.5e-6,
           "detector": None if target is None else target.z_mm}[request.stop]
    if bool(getattr(working, "energy_filter_installed", False)):
        boundary = float(working.energy_filter.entrance_z_mm)
        if not math.isfinite(boundary):
            raise ValueError("Installed energy filter needs a finite physical entrance boundary")
        if end >= boundary:
            raise ValueError(f"The installed energy filter requires its coherent field operator at z={boundary:.9g} mm; "
                             f"the requested path to z={end:.9g} mm cannot skip it")
    if request.stop != "gun_exit":
        _prepare_column(working, gun_end, entrance, request.column_step_mm)
        if target is not None:
            _prepare_column(working, working.sample.z_mm+working.sample.thickness_nm*.5e-6,
                            target.z_mm, request.column_step_mm)
    propagation_id = json_digest({"instrument": snapshot.digest, "source": asdict(request.source),
                                 "gun": asdict(request.gun)})
    store = ExecutedWaveStore(request.execution.cache_directory, propagation_id, request.execution.maximum_disk_cache_bytes)
    if request.execution.segmented:
        _STAGES.clear()
        from temsim.physics.tip_gun_wave import _CACHE
        _CACHE.clear()
    cache_hit = False
    epoch = float(getattr(working, "simulation_time_s", 0.) if request.tip_time_s is None else request.tip_time_s)
    def stage(name, producer, *, inputs=()):
        nonlocal cache_hit
        key = (propagation_id, name, json_digest(inputs))
        if cancelled():
            raise InterruptedError("Tip wave request cancelled")
        if use_cache and request.execution.segmented:
            cached = store.get(store.key(*key))
            if cached is not None:
                cache_hit = True
                return cached
        if use_cache and not request.execution.segmented and key in _STAGES:
            _STAGES.move_to_end(key)
            cache_hit = True
            return _STAGES[key][0]
        result = producer()
        snapshot.restore()  # Verify source implementation and external bytes before caching.
        if cancelled():
            raise InterruptedError("Tip wave request cancelled")
        if request.execution.segmented:
            result = store.put(store.key(*key), result)
        elif use_cache:
            _retain(key, result, request.execution.maximum_ram_cache_bytes)
        return result
    def column(upstream, stop):
        nonlocal cache_hit
        if request.execution.segmented:
            result, hit = _propagate_column_segmented(working, upstream, stop, store=store,
                segment_steps=request.execution.segment_steps, maximum_step_mm=request.column_step_mm,
                grid_numerics=request.wave_grid, tip_time_s=epoch, verify=snapshot.restore,
                cancelled=cancelled, progress_callback=progress_callback, use_cache=use_cache)
            cache_hit |= hit
            return result
        return _propagate_column(working, upstream, stop, maximum_step_mm=request.column_step_mm,
            grid_numerics=request.wave_grid, tip_time_s=epoch, cancelled=cancelled, progress_callback=progress_callback)
    gun = stage("gun_exit", lambda: build_tip_gun_checkpoint(working.electron_gun, source_numerics=request.source,
        numerics=request.gun, _column_state=working, use_cache=use_cache and not request.execution.segmented,
        cancelled=cancelled, progress_callback=progress_callback))
    result = gun
    if request.stop != "gun_exit":
        result = column(result, entrance)
    del gun
    if request.stop in ("specimen_exit", "detector"):
        upstream = result
        from temsim.specimen.scene import SpecimenScene
        if SpecimenScene.from_state(working).is_vacuum:
            stop = working.sample.z_mm+working.sample.thickness_nm*.5e-6
            result = column(upstream, stop) if stop > upstream.plane_z_mm else upstream
        elif request.inelastic.method == "trajectories":
            from temsim.physics.inelastic_wave import _propagate_inelastic_specimen
            result = _propagate_inelastic_specimen(working, upstream, numerics=request.inelastic,
                store=store, maximum_step_mm=request.column_step_mm, grid_numerics=request.wave_grid,
                tip_time_s=epoch, cancelled=cancelled, progress_callback=progress_callback,
                verify=snapshot.restore, use_cache=use_cache)
        else:
            result = stage("specimen_exit", lambda: _propagate_specimen(working, upstream,
                maximum_checkpoint_bytes=request.gun.maximum_checkpoint_bytes,
                maximum_step_mm=request.column_step_mm, grid_numerics=request.wave_grid, tip_time_s=epoch,
                cancelled=cancelled, progress_callback=progress_callback),
                inputs=(upstream.digest, asdict(request.inelastic), request.column_step_mm, asdict(request.wave_grid), epoch))
        del upstream
    readout = None
    if target is not None:
        for plane in planes:
            upstream = result
            result = column(upstream, float(plane.z_mm))
            if plane.key != target.key:
                incident = result
                if request.execution.segmented:
                    from temsim.detector.wave_readout import _recording_stop_streamed
                    result = _recording_stop_streamed(incident, plane, store, verify=snapshot.restore,
                        cancelled=cancelled, use_cache=use_cache)
                else:
                    result = stage("detector_transmitted:"+plane.key, lambda: _apply_recording_stop(incident, plane), inputs=(incident.digest,))
        from temsim.detector.wave_readout import read_wave_detector_streamed
        reader = read_wave_detector_streamed if request.execution.segmented else read_wave_detector
        readout = reader(result, target, request.readout, pixels=request.detector_pixels,
                        maximum_bytes=request.maximum_readout_bytes)
    snapshot.restore()
    return TipWaveResult(result, readout, request, cache_hit, snapshot.digest)
