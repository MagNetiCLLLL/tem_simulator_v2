"""Optional observables of an incident mixed electron wave.

The complex wave is simulation state before detection. An intensity detector
does not directly measure its phase. Phase is mode-resolved, gauge-labelled,
masked at zeros, and never inferred from sqrt(sum of mode intensities).
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from temsim.immutable_json import freeze_json
from temsim.physics.wave_reference import AxialWaveReference


@dataclass(frozen=True)
class WaveReadoutOptions:
    intensity: bool = True
    phase: bool = False
    complex_amplitude: bool = False
    covariance: bool = False
    phase_relative_threshold: float = 1e-10

    def validate(self):
        for name in ("intensity", "phase", "complex_amplitude", "covariance"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"Readout {name} must be boolean")
        if not any((self.intensity, self.phase, self.complex_amplitude, self.covariance)):
            raise ValueError("Choose at least one detector observable")
        if not math.isfinite(self.phase_relative_threshold) or not 0 < self.phase_relative_threshold < 1:
            raise ValueError("Phase visibility threshold must be between zero and one")
        return self


def _frozen(array):
    if array is None:
        return None
    array = np.asarray(array)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _sensor_mask(x_m, y_m, detector):
    width = float(detector.outer_width_mm)*1e-3
    inner = float(getattr(detector, "inner_diameter_mm", 0.))*1e-3
    if not math.isfinite(width) or width <= 0 or not math.isfinite(inner) or not 0 <= inner <= width:
        raise ValueError("Detector dimensions must be finite with 0 <= inner <= outer")
    if hasattr(detector, "hit_mask"):
        # x/y here are sensor-centred components. The component owns its shape
        # and expects offsets in its coordinate arguments; re-add them only
        # for that API. The physical centre was already subtracted below.
        return np.asarray(detector.hit_mask(x_m*1e3+getattr(detector, "centre_offset_x_mm", 0.),
                                           y_m*1e3+getattr(detector, "centre_offset_y_mm", 0.)), dtype=bool)
    radius = np.hypot(x_m, y_m)
    geometry = str(detector.geometry)
    if geometry == "square":
        mask = np.maximum(abs(x_m), abs(y_m)) < width/2
    elif geometry in ("disk", "annulus"):
        mask = radius < width/2
    else:
        raise ValueError(f"Unsupported physical detector geometry {geometry}")
    return mask & (radius >= inner/2)


def _sensor_coordinates(wave, detector, frame):
    centre = np.array((getattr(detector, "centre_offset_x_mm", 0.), getattr(detector, "centre_offset_y_mm", 0.)))*1e-3
    return np.einsum("ij,jyx->iyx", frame.column_to_detector, wave.coordinates_m()-centre[:, None, None])


def _apply_recording_stop(checkpoint, detector):
    """Unobserved detectors still absorb their intercepted probability."""
    from dataclasses import replace
    from temsim.physics.first_order import detector_frame_from_component
    from temsim.physics.tip_gun_wave import TipGunCheckpoint
    from temsim.physics.wave_flux import BeamState
    frame = detector_frame_from_component(detector)
    modes, rows = [], []
    for mode in checkpoint.beam.modes:
        xy = _sensor_coordinates(mode.plane, detector, frame)
        hit = _sensor_mask(xy[0], xy[1], detector)
        amplitude = np.where(hit, 0j, mode.plane.amplitude)
        norm = float(np.sum(abs(amplitude)**2))
        output = replace(mode, plane=replace(mode.plane, amplitude=amplitude/math.sqrt(norm) if norm else amplitude),
                         weight_per_reference_electron=mode.weight_per_reference_electron*norm)
        modes.append(output)
        rows.append({"mode_id": mode.mode_id, "received_weight": mode.weight_per_reference_electron-output.weight_per_reference_electron,
                     "transmitted_weight": output.weight_per_reference_electron})
    return TipGunCheckpoint(BeamState(tuple(modes), checkpoint.beam.reference_plane), checkpoint.plane_z_mm,
        checkpoint.reference_current_a, {"schema": "executed-recording-stop-v1", "upstream": checkpoint.record,
        "upstream_digest": checkpoint.digest, "detector": detector.key, "modes": rows,
        "readout_enabled": bool(getattr(detector, "readout_enabled", True))})


@dataclass(frozen=True)
class ModeWaveReadout:
    mode_id: str
    energy_kev: float
    weight_per_tip_electron: float
    plane: object  # Complete immutable carrier representation; never discarded.
    phase_rad: np.ndarray | None = None
    phase_valid: np.ndarray | None = None
    complex_cell_amplitude: np.ndarray | None = None
    canonical_covariance: np.ndarray | None = None
    axial_reference: AxialWaveReference | None = None
    scattering_history: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "scattering_history", freeze_json(tuple(self.scattering_history)))
        for name in ("phase_rad", "phase_valid", "complex_cell_amplitude", "canonical_covariance"):
            object.__setattr__(self, name, _frozen(getattr(self, name)))


@dataclass(frozen=True)
class WaveDetectorReadout:
    modes: tuple[ModeWaveReadout, ...]
    x_mm: np.ndarray | None
    y_mm: np.ndarray | None
    optical_probability: np.ndarray | None
    detected_probability: np.ndarray | None
    record: object

    def __post_init__(self):
        if not isinstance(self.modes, _LazyModeReadouts):
            object.__setattr__(self, "modes", tuple(self.modes))
        for name in ("x_mm", "y_mm", "optical_probability", "detected_probability"):
            object.__setattr__(self, name, _frozen(getattr(self, name)))
        object.__setattr__(self, "record", freeze_json(self.record))


def read_wave_detector(checkpoint, detector, options=WaveReadoutOptions(), *, pixels=None,
                       maximum_bytes=512*1024**2):
    """Read an already propagated checkpoint without rerunning any optics.

Phase arrays use each mode's incident wave lattice; detector pixels integrate
probability and PSF. Phases of different spatial points are compared within a
mode. Incoherent mode phases cannot be compared or averaged across modes.
"""
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
    from temsim.physics.camera_wave import _deposit_mapped_probability
    from temsim.physics.first_order import detector_frame_from_component
    from temsim.detector.point_spread import DetectorPointSpread, apply_point_spread
    options.validate()
    if not math.isclose(checkpoint.plane_z_mm, detector.z_mm, abs_tol=1e-9, rel_tol=0):
        raise ValueError("Detector readout needs the wave at the physical detector plane")
    if not bool(getattr(detector, "inserted", False)):
        raise ValueError("Detector is not inserted")
    if not bool(getattr(detector, "readout_enabled", True)):
        raise ValueError("Selected detector readout is disabled; its physical absorption still applies")
    count = getattr(detector, "pixels", 512) if pixels is None else pixels
    if isinstance(count, bool) or not isinstance(count, int) or not 2 <= count <= 8192:
        raise ValueError("Detector grid must contain 2 to 8192 pixels per axis")
    required = sum(m.plane.amplitude.size for m in checkpoint.beam.modes)*(17*options.phase+16*options.complex_amplitude)
    required += count*count*64*options.intensity
    required += max(m.plane.amplitude.size for m in checkpoint.beam.modes)*192
    if required > maximum_bytes:
        raise ValueError(f"Selected detector outputs need approximately {required} bytes; deselect observables or increase the memory budget")
    from temsim.physics.wave_execution import check_available_memory
    check_available_memory(required)
    modes = []
    frame = detector_frame_from_component(detector)
    for mode in checkpoint.beam.modes:
        wave = mode.plane
        wavelength = float(wavelength_m(mode.energy_kev*1000))
        phase, valid, amplitude, covariance = None, None, None, None
        if options.phase or options.complex_amplitude:
            xy = wave.coordinates_m()-wave.origin_m[:, None, None]
            q = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
            tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
            carrier = (np.einsum("iyx,ij,jyx->yx", xy, q, xy)/2 + np.einsum("i,iyx->yx", tilt, xy))*2*np.pi/wavelength
            phasor = wave.amplitude*np.exp(1j*carrier)
            detector_xy = _sensor_coordinates(wave, detector, frame)
            sensor = _sensor_mask(detector_xy[0], detector_xy[1], detector)
            if options.complex_amplitude:
                amplitude = phasor*math.sqrt(mode.weight_per_reference_electron)*sensor
            if options.phase:
                probability = abs(wave.amplitude)**2
                valid = (probability > probability.max()*options.phase_relative_threshold) & (mode.weight_per_reference_electron > 0) & sensor
                phase = np.where(valid, np.angle(phasor), np.nan)
        if options.covariance and mode.weight_per_reference_electron > 0:
            covariance = wave.canonical_covariance(wavelength)
        modes.append(ModeWaveReadout(mode.mode_id, mode.energy_kev, mode.weight_per_reference_electron,
                                     wave, phase, valid, amplitude, covariance, mode.axial_reference, mode.scattering_history))
    x = y = optical = detected = None
    metadata = {"checkpoint_digest": checkpoint.digest, "plane_z_mm": checkpoint.plane_z_mm,
        "detector": detector.key, "tip_current_a": checkpoint.reference_current_a,
        "incident_weight": checkpoint.beam.total_weight,
        "phase_kind": "simulated incident stationary scalar wave, per coherent mode",
        "phase_reference": "inherited tip / continuous canonical gauge; longitudinal carrier factored out",
        "axial_references": [{"mode_id": mode.mode_id,
            "reference": None if mode.axial_reference is None else asdict(mode.axial_reference)} for mode in checkpoint.beam.modes],
        "axial_reference_kind": "elapsed axial flight time and spatial action integral p dz from tip; not a longitudinal wavepacket",
        "phase_grid": "each mode's affine incident-wave lattice, sensor acceptance masked, before pixel integration and PSF",
        "phase_sampling": "wrapped point samples plus exact retained analytic phase carriers; not an unwrapped reconstruction",
        "mixed_total_phase": "UNDEFINED; intensities sum, incoherent amplitudes do not",
        "hardware_phase_measurement": "NOT_SIMULATED; requires same-source reference interference or reconstruction",
        "detector_frame_status": frame.status,
        "probability_reference": checkpoint.beam.reference_plane}
    if options.intensity:
        width = float(detector.outer_width_mm)
        if not math.isfinite(width) or width <= 0:
            raise ValueError("Detector width must be finite and positive")
        x = (np.arange(count)-(count-1)/2)*width/count
        y = x.copy()
        optical = np.zeros((count, count))
        cell_area_m2 = (width/count*1e-3)**2
        for mode in checkpoint.beam.modes:
            xy = _sensor_coordinates(mode.plane, detector, frame)
            mask = _sensor_mask(xy[0], xy[1], detector)
            probability = mode.weight_per_reference_electron*abs(mode.plane.amplitude)**2*mask
            optical += _deposit_mapped_probability(probability, xy[0], xy[1], x*1e-3, y*1e-3)*cell_area_m2
        psf = DetectorPointSpread.from_component(detector)
        detected = apply_point_spread(optical, psf, pixel_size_x_mm=width/count, pixel_size_y_mm=width/count)
        xx, yy = np.meshgrid(x, y)
        sensor = _sensor_mask(xx*1e-3, yy*1e-3, detector)
        detected *= sensor
        metadata.update({"pixels": count, "optical_received_weight": float(optical.sum()),
            "response_weight": float(detected.sum()), "psf": asdict(psf),
            "pixel_integration": "conservative bilinear deposition of incident wave cell probabilities; refine wave and readout grids independently"})
    return WaveDetectorReadout(tuple(modes), x, y, optical, detected, metadata)


@dataclass(frozen=True)
class _LazyModeReadouts:
    """Native waves and optional per-mode arrays, loaded only on access."""
    checkpoint: object
    detector: object
    options: WaveReadoutOptions
    pixels: int
    maximum_bytes: int

    def __post_init__(self):
        from copy import deepcopy
        # Detach from the live component before a future lazy evaluation.
        object.__setattr__(self, "detector", deepcopy(self.detector))

    def __len__(self):
        return len(self.checkpoint.beam.modes)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        from dataclasses import replace
        from temsim.physics.tip_gun_wave import TipGunCheckpoint
        from temsim.physics.wave_flux import BeamState
        mode = self.checkpoint.beam.modes[index]
        if not any((self.options.phase, self.options.complex_amplitude, self.options.covariance)):
            return ModeWaveReadout(mode.mode_id, mode.energy_kev, mode.weight_per_reference_electron, mode.plane,
                axial_reference=mode.axial_reference, scattering_history=mode.scattering_history)
        local = TipGunCheckpoint(BeamState((mode,), self.checkpoint.beam.reference_plane), self.checkpoint.plane_z_mm,
            self.checkpoint.reference_current_a, {"parent": self.checkpoint.digest})
        return read_wave_detector(local, self.detector, replace(self.options, intensity=False), pixels=self.pixels,
                                  maximum_bytes=self.maximum_bytes).modes[0]


def read_wave_detector_streamed(checkpoint, detector, options=WaveReadoutOptions(), *, pixels=512,
                                maximum_bytes=72*1024**3):
    """Accumulate independent intensities; keep complex/phase modes lazy."""
    from temsim.physics.tip_gun_wave import TipGunCheckpoint
    from temsim.physics.wave_flux import BeamState
    options.validate()
    if isinstance(pixels, bool) or not isinstance(pixels, int) or not 2 <= pixels <= 8192:
        raise ValueError("Detector grid must contain 2 to 8192 pixels per axis")
    aggregate_bytes = 32*pixels*pixels if options.intensity else 0
    if aggregate_bytes >= maximum_bytes:
        raise ValueError("Detector aggregate exceeds readout memory budget")
    x = y = optical = detected = None
    metadata = None
    groups = {}
    # Even phase-only calls verify physical placement before returning a lazy view.
    if not math.isclose(checkpoint.plane_z_mm, detector.z_mm, abs_tol=1e-9, rel_tol=0) or not detector.inserted:
        raise ValueError("Detector readout needs the inserted physical detector plane")
    if not bool(getattr(detector, "readout_enabled", True)):
        raise ValueError("Selected detector readout is disabled")
    if options.intensity:
        for mode in checkpoint.beam.modes:
            local = TipGunCheckpoint(BeamState((mode,), checkpoint.beam.reference_plane), checkpoint.plane_z_mm,
                checkpoint.reference_current_a, {"parent": checkpoint.digest})
            result = read_wave_detector(local, detector, WaveReadoutOptions(), pixels=pixels,
                                       maximum_bytes=maximum_bytes-aggregate_bytes)
            if optical is None:
                x, y = result.x_mm, result.y_mm
                optical, detected = np.array(result.optical_probability), np.array(result.detected_probability)
                metadata = dict(result.record)
            else:
                optical += result.optical_probability
                detected += result.detected_probability
            if "/trajectory:" in mode.mode_id:
                group = mode.mode_id.rsplit("/trajectory:", 1)[0]
                values = groups.setdefault(group, [])
                values.append(float(result.detected_probability.sum()))
            del mode, local, result
    metadata = {} if metadata is None else metadata
    metadata.update(checkpoint_digest=checkpoint.digest, detector=detector.key, plane_z_mm=checkpoint.plane_z_mm,
        tip_current_a=checkpoint.reference_current_a, incident_weight=checkpoint.beam.total_weight,
        mode_readout_evaluation="streamed intensity; per-mode complex/phase/covariance evaluated on access",
        mixed_total_phase="UNDEFINED; independent source, phonon and environmental histories do not interfere",
        phase_reference="conditional mode envelope plus retained analytic carriers; axial action separate",
        hardware_phase_measurement="NOT_SIMULATED; simulated wave state only")
    if optical is not None:
        metadata.update(optical_received_weight=float(optical.sum()), response_weight=float(detected.sum()))
        metadata["inelastic_response_standard_error"] = (math.sqrt(sum(len(v)*float(np.var(v, ddof=1)) for v in groups.values()))
            if groups and all(len(v) > 1 for v in groups.values()) else None)
        metadata["statistical_error_scope"] = "conditional inelastic trajectory sampling only; excludes source/phonon/grid/dwell/model errors"
    # Mode-dependent reference metadata are read alongside their wave, never
    # copied from the first mode to pretend all energies share one trajectory.
    metadata.pop("axial_references", None)
    return WaveDetectorReadout(_LazyModeReadouts(checkpoint, detector, options, pixels, maximum_bytes-aggregate_bytes),
                               x, y, optical, detected, metadata)


def _recording_stop_streamed(checkpoint, detector, store, *, verify, cancelled, use_cache=True):
    from temsim.physics.tip_gun_wave import TipGunCheckpoint
    from temsim.physics.wave_flux import BeamState
    key = store.key("recording-stop", checkpoint.digest, detector.key)
    cached = store.get(key) if use_cache else None
    if cached is not None:
        return cached
    writer = store.writer(key, checkpoint.beam.reference_plane)
    rows = []
    try:
        for mode in checkpoint.beam.modes:
            if cancelled():
                raise InterruptedError("Detector interception cancelled")
            local = TipGunCheckpoint(BeamState((mode,), checkpoint.beam.reference_plane), checkpoint.plane_z_mm,
                checkpoint.reference_current_a, {"parent": checkpoint.digest})
            result = _apply_recording_stop(local, detector)
            writer.append(result.beam.modes[0])
            rows.extend(result.record["modes"])
            del mode, local, result
        verify()
        if cancelled():
            raise InterruptedError("Detector interception cancelled before commit")
        return writer.finish(checkpoint.plane_z_mm, checkpoint.reference_current_a,
            {"schema": "executed-recording-stop-stream-v1", "detector": detector.key,
             "upstream_digest": checkpoint.digest, "upstream": checkpoint.record, "modes": rows})
    finally:
        writer.abort()
