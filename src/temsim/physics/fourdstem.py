"""Streaming 4D-STEM storage, detector response, and virtual detectors.

The stored array order is ``(scan_y, scan_x, detector_y, detector_x)``.
Scan coordinates are specimen-plane micrometres and detector coordinates are
signed scattering angles in milliradians.  No ptychographic reconstruction is
performed here; this module only preserves calibrated diffraction frames and
forms reproducible detector observables.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType

import numpy as np
from scipy.ndimage import gaussian_filter

from temsim.physics.record_plane import (
    RecordPlanePlan,
    result_matches_plan,
    route_record_planes,
)


FOURDSTEM_FORMAT_VERSION = 2


class FourDSTEMCancelled(RuntimeError):
    """An explicitly cancelled capture retains its durable checkpoint."""


def _readonly(values, *, dtype=float) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(np.asarray(array))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(repr(values.shape).encode("ascii"))
        digest.update(values.view(np.uint8))
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@dataclass(frozen=True, slots=True)
class FourDSTEMCalibration:
    """Explicit calibrated scan and diffraction coordinates.

    All arrays are two-dimensional.  This permits rotated, sheared, or
    otherwise non-separable scan/detector calibrations without inventing a
    rectilinear approximation.
    """

    scan_x_um: np.ndarray
    scan_y_um: np.ndarray
    angle_x_mrad: np.ndarray
    angle_y_mrad: np.ndarray
    scan_times_s: np.ndarray | None = None

    def __post_init__(self) -> None:
        scan_x = _readonly(self.scan_x_um)
        scan_y = _readonly(self.scan_y_um)
        angle_x = _readonly(self.angle_x_mrad)
        angle_y = _readonly(self.angle_y_mrad)
        if scan_x.ndim != 2 or scan_x.shape != scan_y.shape or scan_x.size == 0:
            raise ValueError("4D-STEM scan X/Y coordinates must be matching non-empty 2-D arrays")
        if angle_x.ndim != 2 or angle_x.shape != angle_y.shape or angle_x.size == 0:
            raise ValueError(
                "4D-STEM detector-angle X/Y coordinates must be matching non-empty 2-D arrays"
            )
        if not all(np.all(np.isfinite(values)) for values in (scan_x, scan_y, angle_x, angle_y)):
            raise ValueError("4D-STEM coordinate arrays must be finite")
        object.__setattr__(self, "scan_x_um", scan_x)
        object.__setattr__(self, "scan_y_um", scan_y)
        object.__setattr__(self, "angle_x_mrad", angle_x)
        object.__setattr__(self, "angle_y_mrad", angle_y)
        if self.scan_times_s is not None:
            times = _readonly(self.scan_times_s)
            if times.shape != scan_x.shape or not np.all(np.isfinite(times)):
                raise ValueError("4D-STEM scan times must be finite and match the scan coordinates")
            object.__setattr__(self, "scan_times_s", times)

    @classmethod
    def from_rectilinear_axes(
        cls,
        scan_x_um,
        scan_y_um,
        angle_x_mrad,
        angle_y_mrad,
    ) -> "FourDSTEMCalibration":
        scan_x = np.asarray(scan_x_um, dtype=float)
        scan_y = np.asarray(scan_y_um, dtype=float)
        angle_x = np.asarray(angle_x_mrad, dtype=float)
        angle_y = np.asarray(angle_y_mrad, dtype=float)
        if any(axis.ndim != 1 or axis.size == 0 for axis in (scan_x, scan_y, angle_x, angle_y)):
            raise ValueError("Rectilinear 4D-STEM axes must be non-empty 1-D arrays")
        sx, sy = np.meshgrid(scan_x, scan_y, indexing="xy")
        ax, ay = np.meshgrid(angle_x, angle_y, indexing="xy")
        return cls(sx, sy, ax, ay)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.scan_x_um.shape + self.angle_x_mrad.shape

    @property
    def digest(self) -> str:
        return _array_digest(
            self.scan_x_um,
            self.scan_y_um,
            self.angle_x_mrad,
            self.angle_y_mrad,
            *((self.scan_times_s,) if self.scan_times_s is not None else ()),
        )

    def save(self, path: Path) -> None:
        with Path(path).open("wb") as stream:
            np.savez_compressed(
                stream,
                scan_x_um=self.scan_x_um,
                scan_y_um=self.scan_y_um,
                angle_x_mrad=self.angle_x_mrad,
                angle_y_mrad=self.angle_y_mrad,
                **({"scan_times_s": self.scan_times_s} if self.scan_times_s is not None else {}),
            )

    @classmethod
    def load(cls, path: Path) -> "FourDSTEMCalibration":
        with np.load(Path(path), allow_pickle=False) as values:
            return cls(
                values["scan_x_um"],
                values["scan_y_um"],
                values["angle_x_mrad"],
                values["angle_y_mrad"],
                values["scan_times_s"] if "scan_times_s" in values else None,
            )


@dataclass(frozen=True, slots=True)
class PixelatedDetectorResponse:
    """Adjustable pixel-detector forward response in electron/count units.

    The model order is quantum efficiency, optional charge-spread PSF, dark
    electrons, Poisson counting, Gaussian read noise, saturation, and gain.
    Defaults are explicitly ideal and do not claim an OEM calibration.
    """

    quantum_efficiency: float | np.ndarray = 1.0
    charge_spread_sigma_x_px: float = 0.0
    charge_spread_sigma_y_px: float = 0.0
    dark_electrons_per_pixel: float = 0.0
    read_noise_electrons_rms: float = 0.0
    saturation_electrons: float | None = None
    gain_counts_per_electron: float = 1.0
    offset_counts: float = 0.0
    poisson_enabled: bool = False
    seed: int | None = None
    status: str = "ideal_user_adjustable_not_oem_calibration"

    def __post_init__(self) -> None:
        efficiency = np.asarray(self.quantum_efficiency, dtype=float)
        if efficiency.ndim:
            object.__setattr__(self, "quantum_efficiency", _readonly(efficiency))
        else:
            object.__setattr__(self, "quantum_efficiency", float(efficiency))
        if self.seed is not None:
            seed = int(self.seed)
            if seed < 0:
                raise ValueError("Detector-response seed cannot be negative")
            object.__setattr__(self, "seed", seed)

    def validate(self, detector_shape: tuple[int, int] | None = None) -> "PixelatedDetectorResponse":
        efficiency = np.asarray(self.quantum_efficiency, dtype=float)
        if efficiency.ndim > 2:
            raise ValueError("Detector quantum efficiency must be scalar or 2-D")
        if detector_shape is not None and efficiency.ndim == 2 and efficiency.shape != detector_shape:
            raise ValueError("Detector quantum-efficiency map has the wrong shape")
        if not np.all(np.isfinite(efficiency)) or np.any(efficiency < 0.0) or np.any(efficiency > 1.0):
            raise ValueError("Detector quantum efficiency must lie between 0 and 1")
        for name in (
            "charge_spread_sigma_x_px",
            "charge_spread_sigma_y_px",
            "dark_electrons_per_pixel",
            "read_noise_electrons_rms",
            "offset_counts",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"Detector {name} must be finite")
            if name != "offset_counts" and value < 0.0:
                raise ValueError(f"Detector {name} cannot be negative")
        gain = float(self.gain_counts_per_electron)
        if not math.isfinite(gain) or gain <= 0.0:
            raise ValueError("Detector gain must be finite and positive")
        if self.saturation_electrons is not None:
            saturation = float(self.saturation_electrons)
            if not math.isfinite(saturation) or saturation <= 0.0:
                raise ValueError("Detector saturation must be finite and positive")
        if not str(self.status).strip():
            raise ValueError("Detector-response status must not be empty")
        return self

    def provenance(self) -> dict:
        efficiency = np.asarray(self.quantum_efficiency, dtype=float)
        efficiency_record = (
            float(efficiency)
            if efficiency.ndim == 0
            else {
                "shape": list(efficiency.shape),
                "sha256": _array_digest(efficiency),
                "minimum": float(np.min(efficiency)),
                "maximum": float(np.max(efficiency)),
            }
        )
        return {
            "model": "qe_charge_spread_dark_poisson_read_noise_saturation_gain_v1",
            "quantum_efficiency": efficiency_record,
            "charge_spread_sigma_px_xy": [
                float(self.charge_spread_sigma_x_px),
                float(self.charge_spread_sigma_y_px),
            ],
            "dark_electrons_per_pixel": float(self.dark_electrons_per_pixel),
            "read_noise_electrons_rms": float(self.read_noise_electrons_rms),
            "saturation_electrons": (
                None if self.saturation_electrons is None else float(self.saturation_electrons)
            ),
            "gain_counts_per_electron": float(self.gain_counts_per_electron),
            "offset_counts": float(self.offset_counts),
            "poisson_enabled": bool(self.poisson_enabled),
            "seed": self.seed,
            "status": str(self.status),
            "transport_neutral": self.transport_neutral,
        }

    @property
    def transport_neutral(self) -> bool:
        efficiency = np.asarray(self.quantum_efficiency, dtype=float)
        return bool(
            np.all(efficiency == 1.0)
            and self.charge_spread_sigma_x_px == 0.0
            and self.charge_spread_sigma_y_px == 0.0
            and self.dark_electrons_per_pixel == 0.0
            and self.read_noise_electrons_rms == 0.0
            and self.saturation_electrons is None
            and self.gain_counts_per_electron == 1.0
            and self.offset_counts == 0.0
            and not self.poisson_enabled
        )

    def apply(self, expected_electrons, *, rng: np.random.Generator | None = None) -> np.ndarray:
        values = np.asarray(expected_electrons, dtype=float)
        if values.ndim < 2:
            raise ValueError("Pixelated detector input must have at least two dimensions")
        detector_shape = values.shape[-2:]
        self.validate(detector_shape)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Expected detector electrons must be finite and non-negative")
        expected = values * np.asarray(self.quantum_efficiency, dtype=float)
        sigma_x = float(self.charge_spread_sigma_x_px)
        sigma_y = float(self.charge_spread_sigma_y_px)
        if sigma_x > 0.0 or sigma_y > 0.0:
            expected = gaussian_filter(
                expected,
                sigma=(0.0,) * (expected.ndim - 2) + (sigma_y, sigma_x),
                mode="constant",
                cval=0.0,
            )
        expected = np.maximum(expected, 0.0) + float(self.dark_electrons_per_pixel)
        generator = rng or np.random.default_rng(self.seed)
        if self.poisson_enabled:
            detected = generator.poisson(expected).astype(float)
        else:
            detected = expected.copy()
        read_noise = float(self.read_noise_electrons_rms)
        if read_noise > 0.0:
            detected += generator.normal(0.0, read_noise, size=detected.shape)
            detected = np.maximum(detected, 0.0)
        if self.saturation_electrons is not None:
            detected = np.minimum(detected, float(self.saturation_electrons))
        counts = (
            detected * float(self.gain_counts_per_electron)
            + float(self.offset_counts)
        )
        return np.asarray(counts, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class FourDSTEMArtifact:
    """Completed on-disk diffraction cube and its calibrated provenance."""

    path: Path
    metadata_path: Path
    calibration_path: Path
    calibration: FourDSTEMCalibration
    data: np.ndarray
    metadata: Mapping

    def __post_init__(self) -> None:
        if tuple(self.data.shape) != self.calibration.shape:
            raise ValueError("Stored 4D-STEM cube shape does not match calibration")


class FourDSTEMWriter:
    """Write detector frames directly to a disk-backed NPY array.

    The public target appears only after ``finalize``.  A partial data file,
    calibration file, metadata file, and completion bitmap permit explicit
    checkpointing and resumption without keeping the cube in RAM.
    """

    def __init__(
        self,
        path: str | Path,
        calibration: FourDSTEMCalibration,
        *,
        dtype=np.float32,
        response: PixelatedDetectorResponse | None = None,
        record_plane_plan: RecordPlanePlan | None = None,
        provenance: Mapping | None = None,
        overwrite: bool = False,
        resume: bool = False,
    ) -> None:
        self.path = Path(path)
        self.metadata_path = self.path.with_name(self.path.name + ".json")
        self.calibration_path = self.path.with_name(self.path.name + ".axes.npz")
        self.partial_path = self.path.with_name(self.path.name + ".partial")
        self.partial_metadata_path = self.metadata_path.with_name(self.metadata_path.name + ".partial")
        self.partial_calibration_path = self.calibration_path.with_name(self.calibration_path.name + ".partial")
        self.progress_path = self.path.with_name(self.path.name + ".progress.npy")
        self.calibration = calibration
        self.dtype = np.dtype(dtype)
        if self.dtype.kind not in {"f", "u", "i"}:
            raise ValueError("4D-STEM storage dtype must be numeric")
        self.response = response or PixelatedDetectorResponse()
        self.response.validate(calibration.angle_x_mrad.shape)
        self.record_plane_plan_fingerprint = (
            None if record_plane_plan is None else record_plane_plan.fingerprint
        )
        self.resolved_geometry_fingerprint = (
            None
            if record_plane_plan is None
            else record_plane_plan.resolved_geometry_fingerprint
        )
        try:
            self.provenance = json.loads(json.dumps(
                dict(provenance or {}),
                sort_keys=True,
                allow_nan=False,
            ))
        except (TypeError, ValueError) as error:
            raise ValueError("4D-STEM provenance must be finite JSON data") from error
        self._rng = np.random.default_rng(self.response.seed)
        self._closed = True  # failed admission must never checkpoint from __del__
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if resume:
            self._resume()
            return
        existing = tuple(
            candidate
            for candidate in (
                self.path,
                self.partial_path,
                self.metadata_path,
                self.partial_metadata_path,
                self.calibration_path,
                self.partial_calibration_path,
                self.progress_path,
            )
            if candidate.exists()
        )
        if existing and not overwrite:
            raise FileExistsError(f"4D-STEM output already exists: {existing[0]}")
        if overwrite:
            for candidate in existing:
                candidate.unlink()
        self._data = np.lib.format.open_memmap(
            self.partial_path,
            mode="w+",
            dtype=self.dtype,
            shape=calibration.shape,
        )
        self._completed = np.zeros(calibration.scan_x_um.shape, dtype=bool)
        self._frame_hashes = {}
        calibration.save(self.partial_calibration_path)
        self._closed = False
        self.checkpoint()

    def _base_metadata(self, status: str) -> dict:
        return {
            "format": "temsim_4dstem_npy",
            "format_version": FOURDSTEM_FORMAT_VERSION,
            "status": status,
            "shape": list(self.calibration.shape),
            "dtype": self.dtype.str,
            "axis_order": ["scan_y", "scan_x", "detector_y", "detector_x"],
            "scan_coordinate_unit": "um",
            "detector_coordinate": "signed_scattering_angle",
            "detector_coordinate_unit": "mrad",
            "calibration_sha256": self.calibration.digest,
            "record_plane_plan_fingerprint": self.record_plane_plan_fingerprint,
            "resolved_geometry_fingerprint": self.resolved_geometry_fingerprint,
            "detector_response": self.response.provenance(),
            "rng_state": self._rng.bit_generator.state,
            "completed_frames": int(np.count_nonzero(self._completed)),
            "total_frames": int(self._completed.size),
            "provenance": self.provenance,
            "completed_frame_sha256": dict(self._frame_hashes),
        }

    def _resume(self) -> None:
        required = (
            self.partial_metadata_path,
        )
        if not all(path.exists() for path in required):
            raise FileNotFoundError("A complete 4D-STEM partial checkpoint was not found")
        metadata = json.loads(self.partial_metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format_version") != FOURDSTEM_FORMAT_VERSION:
            raise ValueError("Legacy checkpoints lack frame integrity records; restart capture explicitly")
        if not self.partial_path.exists() and self.path.exists():
            os.replace(self.path, self.partial_path)
        if not self.partial_calibration_path.exists() and self.calibration_path.exists():
            os.replace(self.calibration_path, self.partial_calibration_path)
        disk_calibration = FourDSTEMCalibration.load(self.partial_calibration_path)
        if disk_calibration.digest != self.calibration.digest:
            raise ValueError("Cannot resume 4D-STEM output with different calibration")
        if tuple(metadata["shape"]) != self.calibration.shape:
            raise ValueError("Cannot resume 4D-STEM output with different shape")
        if np.dtype(metadata["dtype"]) != self.dtype:
            raise ValueError("Cannot resume 4D-STEM output with different dtype")
        if metadata.get("record_plane_plan_fingerprint") != self.record_plane_plan_fingerprint:
            raise ValueError("Cannot resume 4D-STEM output with a different record-plane plan")
        if metadata.get("resolved_geometry_fingerprint") != self.resolved_geometry_fingerprint:
            raise ValueError("Cannot resume 4D-STEM output with different resolved geometry")
        if metadata.get("detector_response") != self.response.provenance():
            raise ValueError("Cannot resume 4D-STEM output with different detector response")
        if metadata.get("provenance") != self.provenance:
            raise ValueError("Cannot resume 4D-STEM output from a different source calculation")
        try:
            self._rng.bit_generator.state = metadata["rng_state"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("4D-STEM checkpoint has an invalid detector RNG state") from error
        self._data = np.load(self.partial_path, mmap_mode="r+")
        if self._data.shape != self.calibration.shape or self._data.dtype != self.dtype:
            raise ValueError("4D-STEM checkpoint data header does not match calibration/dtype")
        self._completed = np.zeros(self.calibration.scan_x_um.shape, dtype=bool)
        self._frame_hashes = dict(metadata["completed_frame_sha256"])
        for index, checksum in self._frame_hashes.items():
            flat_index = int(index)
            if str(flat_index) != index or not 0 <= flat_index < self._completed.size:
                raise ValueError("4D-STEM completion index is outside raster")
            y, x = divmod(flat_index, self._completed.shape[1])
            if _array_digest(self._data[y, x]) != checksum:
                raise ValueError(f"4D-STEM frame checksum mismatch at ({y}, {x})")
            self._completed[y, x] = True
        if int(np.count_nonzero(self._completed)) != metadata["completed_frames"]:
            raise ValueError("4D-STEM checkpoint completion count does not match integrity record")
        self._closed = False

    @property
    def completed_frames(self) -> int:
        return int(np.count_nonzero(self._completed))

    @property
    def total_frames(self) -> int:
        return int(self._completed.size)

    def frame_completed(self, scan_y: int, scan_x: int) -> bool:
        """Whether this writer has accepted the frame (durable after checkpoint)."""

        y_index = int(scan_y)
        x_index = int(scan_x)
        if not (0 <= y_index < self._completed.shape[0] and 0 <= x_index < self._completed.shape[1]):
            raise IndexError("4D-STEM scan index is outside the calibrated raster")
        return bool(self._completed[y_index, x_index])

    def write_frame(self, scan_y: int, scan_x: int, expected_electrons) -> None:
        if self._closed:
            raise RuntimeError("4D-STEM writer is closed")
        y_index = int(scan_y)
        x_index = int(scan_x)
        if not (0 <= y_index < self._completed.shape[0] and 0 <= x_index < self._completed.shape[1]):
            raise IndexError("4D-STEM scan index is outside the calibrated raster")
        frame = np.asarray(expected_electrons, dtype=float)
        if self._completed[y_index, x_index]:
            raise ValueError("4D-STEM frame already completed; duplicate writes are not allowed")
        if frame.shape != self.calibration.angle_x_mrad.shape:
            raise ValueError("4D-STEM diffraction frame has the wrong detector shape")
        response = self.response.apply(frame, rng=self._rng)
        self._data[y_index, x_index] = response.astype(self.dtype, copy=False)
        index = y_index * self._completed.shape[1] + x_index
        self._frame_hashes[str(index)] = _array_digest(self._data[y_index, x_index])
        self._completed[y_index, x_index] = True

    def write_stream(self, frames: Iterable[tuple[int, int, np.ndarray]], *, checkpoint_every: int = 32) -> None:
        interval = int(checkpoint_every)
        if interval < 1:
            raise ValueError("4D-STEM checkpoint interval must be positive")
        since_checkpoint = 0
        for scan_y, scan_x, frame in frames:
            self.write_frame(scan_y, scan_x, frame)
            since_checkpoint += 1
            if since_checkpoint >= interval:
                self.checkpoint()
                since_checkpoint = 0
        if since_checkpoint:
            self.checkpoint()

    def checkpoint(self) -> None:
        if self._closed:
            raise RuntimeError("4D-STEM writer is closed")
        self._data.flush()
        with self.partial_path.open("r+b") as stream:
            os.fsync(stream.fileno())
        temporary_progress = self.progress_path.with_name(self.progress_path.name + ".tmp")
        with temporary_progress.open("wb") as stream:
            np.save(stream, self._completed, allow_pickle=False)
        os.replace(temporary_progress, self.progress_path)
        _atomic_json(self.partial_metadata_path, self._base_metadata("partial"))

    def _close_memmap(self) -> None:
        self._data.flush()
        mmap = getattr(self._data, "_mmap", None)
        if mmap is not None:
            mmap.close()
        self._closed = True

    def finalize(self) -> FourDSTEMArtifact:
        if self._closed:
            raise RuntimeError("4D-STEM writer is closed")
        if not np.all(self._completed):
            raise RuntimeError(
                f"Cannot finalize incomplete 4D-STEM cube: {self.completed_frames}/{self.total_frames} frames"
            )
        self.checkpoint()
        final_metadata = self._base_metadata("complete")
        self._close_memmap()
        os.replace(self.partial_calibration_path, self.calibration_path)
        os.replace(self.partial_path, self.path)
        _atomic_json(self.metadata_path, final_metadata)
        self.partial_metadata_path.unlink(missing_ok=True)
        self.progress_path.unlink(missing_ok=True)
        return open_fourdstem(self.path)

    def close(self, *, keep_partial: bool = True) -> None:
        if self._closed:
            return
        self.checkpoint()
        self._close_memmap()
        if not keep_partial:
            for path in (
                self.partial_path,
                self.partial_metadata_path,
                self.partial_calibration_path,
                self.progress_path,
            ):
                path.unlink(missing_ok=True)

    def __enter__(self) -> "FourDSTEMWriter":
        return self

    def __exit__(self, _exception_type, _exception, _traceback) -> None:
        self.close(keep_partial=True)

    def __del__(self) -> None:
        try:
            self.close(keep_partial=True)
        except Exception:
            # Interpreter shutdown and failed filesystems cannot be reported
            # safely from a finalizer; explicit close/finalize remains the
            # authoritative error-reporting path.
            pass


def open_fourdstem(path: str | Path, *, mmap_mode: str = "r") -> FourDSTEMArtifact:
    target = Path(path)
    metadata_path = target.with_name(target.name + ".json")
    calibration_path = target.with_name(target.name + ".axes.npz")
    if not target.exists() or not metadata_path.exists() or not calibration_path.exists():
        raise FileNotFoundError("Completed 4D-STEM data or sidecar files are missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") not in (1, FOURDSTEM_FORMAT_VERSION) or metadata.get("status") != "complete":
        raise ValueError("Unsupported or incomplete 4D-STEM artifact")
    calibration = FourDSTEMCalibration.load(calibration_path)
    if metadata.get("calibration_sha256") != calibration.digest:
        raise ValueError("4D-STEM calibration checksum mismatch")
    data = np.load(target, mmap_mode=mmap_mode, allow_pickle=False)
    if tuple(metadata.get("shape", ())) != tuple(data.shape):
        raise ValueError("4D-STEM data shape does not match metadata")
    if np.dtype(metadata.get("dtype")) != data.dtype:
        raise ValueError("4D-STEM data dtype does not match metadata")
    return FourDSTEMArtifact(
        path=target,
        metadata_path=metadata_path,
        calibration_path=calibration_path,
        calibration=calibration,
        data=data,
        metadata=MappingProxyType(metadata),
    )


class FourDSTEMCaptureSink:
    """Adapter from normalised STEM diffraction frames to an out-of-core cube.

    ``begin`` is called once the wave solver has established its exact scan and
    FFT-angle coordinates.  ``write_frame`` receives one configuration-averaged
    probability pattern at a time.  Invalid/aliased reciprocal pixels are set
    to zero, so their missing probability remains visible as a sum below one.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        electrons_per_frame: float | np.ndarray,
        response: PixelatedDetectorResponse | None = None,
        store_raw_probability: bool = False,
        record_plane_plan: RecordPlanePlan | None = None,
        dtype=np.float32,
        overwrite: bool = False,
        resume: bool = False,
        checkpoint_every: int = 32,
        provenance: Mapping | None = None,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> None:
        electrons = np.asarray(electrons_per_frame, dtype=float)
        if electrons.ndim > 2 or not np.all(np.isfinite(electrons)) or np.any(electrons < 0.0):
            raise ValueError("4D-STEM electrons per frame must be finite, non-negative, and scalar/2-D")
        interval = int(checkpoint_every)
        if interval < 1:
            raise ValueError("4D-STEM checkpoint interval must be positive")
        self.path = Path(path)
        self.electrons_per_frame = electrons
        self.response = response or PixelatedDetectorResponse()
        self.store_raw_probability = bool(store_raw_probability)
        self.record_plane_plan = record_plane_plan
        self.dtype = np.dtype(dtype)
        self.overwrite = bool(overwrite)
        self.resume = bool(resume)
        self.checkpoint_every = interval
        self.provenance = dict(provenance or {})
        self.cancellation_requested = cancellation_requested
        self.writer: FourDSTEMWriter | None = None
        self.calibration: FourDSTEMCalibration | None = None
        self.valid_reciprocal_mask: np.ndarray | None = None
        self.artifact: FourDSTEMArtifact | None = None
        self._frames_since_checkpoint = 0

    def begin(
        self,
        calibration: FourDSTEMCalibration,
        valid_reciprocal_mask,
        *,
        maximum_isotropic_angle_mrad: float,
    ) -> None:
        if self.writer is not None or self.artifact is not None:
            raise RuntimeError("4D-STEM capture sink was already started")
        valid = np.asarray(valid_reciprocal_mask, dtype=bool)
        if valid.shape != calibration.angle_x_mrad.shape:
            raise ValueError("4D-STEM valid-reciprocal mask has the wrong shape")
        if self.electrons_per_frame.ndim == 2 and self.electrons_per_frame.shape != calibration.scan_x_um.shape:
            raise ValueError("4D-STEM electron-count map has the wrong scan shape")
        maximum_angle = float(maximum_isotropic_angle_mrad)
        if not math.isfinite(maximum_angle) or maximum_angle <= 0.0:
            raise ValueError("Maximum isotropic diffraction angle must be finite and positive")
        self.calibration = calibration
        self.valid_reciprocal_mask = _readonly(valid, dtype=bool)
        electrons = np.asarray(self.electrons_per_frame, dtype=float)
        capture_provenance = {
            **self.provenance,
            "input_frame_quantity": "configuration-averaged diffraction probability",
            "stored_frame_quantity": (
                "configuration-averaged diffraction probability"
                if self.store_raw_probability
                else "detector counts after dose and response"
            ),
            "dose_and_detector_response": (
                "deferred_to_derived_products"
                if self.store_raw_probability
                else "baked_before_storage"
            ),
            "invalid_reciprocal_pixels": "zeroed_without_renormalisation",
            "valid_reciprocal_mask_sha256": _array_digest(valid),
            "maximum_isotropic_angle_mrad": maximum_angle,
        }
        self.writer = FourDSTEMWriter(
            self.path,
            calibration,
            dtype=self.dtype,
            response=(
                PixelatedDetectorResponse(
                    status="transport_neutral_raw_diffraction_probability"
                )
                if self.store_raw_probability
                else self.response
            ),
            record_plane_plan=self.record_plane_plan,
            provenance=capture_provenance,
            overwrite=self.overwrite,
            resume=self.resume,
        )

    def write_frame(self, scan_y: int, scan_x: int, diffraction_probability) -> None:
        self.check_cancelled()
        if self.writer is None or self.calibration is None or self.valid_reciprocal_mask is None:
            raise RuntimeError("4D-STEM capture sink has not been started")
        if self.writer.frame_completed(scan_y, scan_x):
            return
        probability = np.asarray(diffraction_probability, dtype=float)
        if probability.shape != self.valid_reciprocal_mask.shape:
            raise ValueError("4D-STEM diffraction probability has the wrong shape")
        if not np.all(np.isfinite(probability)) or np.any(probability < 0.0):
            raise ValueError("4D-STEM diffraction probability must be finite and non-negative")
        total = float(np.sum(probability, dtype=np.float64))
        if total > 1.0 + 2e-5:
            raise ValueError(f"4D-STEM diffraction exceeds its pre-specimen reference probability; got {total:.9g}")
        stored = np.where(self.valid_reciprocal_mask, probability, 0.0)
        if not self.store_raw_probability:
            electrons = (
                float(self.electrons_per_frame)
                if self.electrons_per_frame.ndim == 0
                else float(
                    self.electrons_per_frame[int(scan_y), int(scan_x)]
                )
            )
            stored = stored * electrons
        self.writer.write_frame(scan_y, scan_x, stored)
        self._frames_since_checkpoint += 1
        if self._frames_since_checkpoint >= self.checkpoint_every:
            self.writer.checkpoint()
            self._frames_since_checkpoint = 0

    def check_cancelled(self):
        if self.cancellation_requested is not None and self.cancellation_requested():
            self.close_partial()
            raise FourDSTEMCancelled("4D-STEM capture cancelled; durable partial retained")

    def finish(self) -> FourDSTEMArtifact:
        if self.writer is None:
            raise RuntimeError("4D-STEM capture sink has not been started")
        self.artifact = self.writer.finalize()
        self.writer = None
        return self.artifact

    def close_partial(self) -> None:
        if self.writer is not None:
            self.writer.close(keep_partial=True)
            self.writer = None

    def __del__(self) -> None:
        try:
            self.close_partial()
        except Exception:
            pass


@dataclass(frozen=True, slots=True)
class VirtualDetector:
    """Arbitrary fractional collection mask on the calibrated detector grid."""

    key: str
    weights: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        if not str(self.key).strip():
            raise ValueError("Virtual-detector key must not be empty")
        weights = _readonly(self.weights)
        if weights.ndim != 2 or weights.size == 0:
            raise ValueError("Virtual-detector weights must be a non-empty 2-D array")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or np.any(weights > 1.0):
            raise ValueError("Virtual-detector weights must lie between 0 and 1")
        object.__setattr__(self, "weights", weights)

    @classmethod
    def from_function(
        cls,
        key: str,
        calibration: FourDSTEMCalibration,
        function: Callable[[np.ndarray, np.ndarray], np.ndarray],
        *,
        label: str = "",
    ) -> "VirtualDetector":
        values = function(calibration.angle_x_mrad, calibration.angle_y_mrad)
        return cls(key=key, weights=np.asarray(values, dtype=float), label=label)


def annular_virtual_detector(
    key: str,
    calibration: FourDSTEMCalibration,
    inner_mrad: float,
    outer_mrad: float,
    *,
    centre_mrad: tuple[float, float] = (0.0, 0.0),
    label: str = "",
) -> VirtualDetector:
    inner = float(inner_mrad)
    outer = float(outer_mrad)
    if not math.isfinite(inner) or not math.isfinite(outer) or inner < 0.0 or outer <= inner:
        raise ValueError("Virtual annulus requires 0 <= inner < outer")
    cx, cy = (float(value) for value in centre_mrad)
    radius = np.hypot(calibration.angle_x_mrad - cx, calibration.angle_y_mrad - cy)
    return VirtualDetector(key, ((radius >= inner) & (radius <= outer)).astype(float), label)


def integrate_virtual_detectors(
    cube: FourDSTEMArtifact | np.ndarray,
    detectors: Sequence[VirtualDetector],
    *,
    chunk_scan_points: int = 32,
    response: PixelatedDetectorResponse | None = None,
    electrons_per_frame: float | np.ndarray = 1.0,
) -> Mapping[str, np.ndarray]:
    """Integrate masks, optionally applying dose/response as a derived layer."""

    data = cube.data if isinstance(cube, FourDSTEMArtifact) else np.asarray(cube)
    if data.ndim != 4:
        raise ValueError("4D-STEM cube must have axes (scan_y, scan_x, detector_y, detector_x)")
    points = int(chunk_scan_points)
    if points < 1:
        raise ValueError("4D-STEM integration chunk size must be positive")
    detectors = tuple(detectors)
    if len({detector.key for detector in detectors}) != len(detectors):
        raise ValueError("Virtual-detector keys must be unique")
    for detector in detectors:
        if detector.weights.shape != data.shape[-2:]:
            raise ValueError(f"{detector.key}: virtual-detector mask has the wrong shape")
    output = {
        detector.key: np.zeros(data.shape[:2], dtype=np.float64)
        for detector in detectors
    }
    if not detectors:
        return MappingProxyType(output)
    dose = np.asarray(electrons_per_frame, dtype=float)
    if (
        dose.ndim > 2
        or (dose.ndim == 2 and dose.shape != data.shape[:2])
        or not np.all(np.isfinite(dose))
        or np.any(dose < 0.0)
    ):
        raise ValueError(
            "Virtual-detector electron dose must be finite, non-negative, "
            "and scalar or match the scan grid"
        )
    detector_response = response
    if detector_response is not None:
        detector_response.validate(data.shape[-2:])
    response_rng = (
        None
        if detector_response is None
        else np.random.default_rng(detector_response.seed)
    )
    stacked = np.stack([detector.weights for detector in detectors], axis=0)
    flat_data = data.reshape((-1,) + data.shape[-2:])
    flat_dose = (
        None if dose.ndim == 0 else dose.reshape(-1)
    )
    flat_output = {
        key: values.reshape(-1)
        for key, values in output.items()
    }
    for start in range(0, flat_data.shape[0], points):
        stop = min(start + points, flat_data.shape[0])
        block = np.asarray(flat_data[start:stop], dtype=float)
        if detector_response is not None:
            block = block * (
                float(dose)
                if flat_dose is None
                else flat_dose[start:stop, None, None]
            )
            block = detector_response.apply(block, rng=response_rng)
        integrated = np.einsum("pij,kij->kp", block, stacked, optimize=True)
        for index, detector in enumerate(detectors):
            flat_output[detector.key][start:stop] = integrated[index]
    return MappingProxyType(output)


@dataclass(frozen=True, slots=True)
class RuntimeDetectorImages:
    """Sequential physical detector images formed from a calibrated 4D cube."""

    plan_fingerprint: str
    calibration_fingerprint: str
    images: Mapping[str, np.ndarray]
    surviving_weight: np.ndarray
    balance_error: float
    capture_plan_fingerprint: str | None = None
    plan_changed_since_capture: bool = False


def integrate_runtime_recording_planes(
    cube: FourDSTEMArtifact | np.ndarray,
    calibration: FourDSTEMCalibration | None,
    plan: RecordPlanePlan,
    *,
    chunk_scan_points: int = 1,
    allow_detector_response_weighting: bool = False,
) -> RuntimeDetectorImages:
    """Apply full mixed-plane geometry and sequential stops to every 4D pixel.

    Physical routing normally requires an ideal expected-electron cube.
    Applying a pixel-detector PSF/noise first would reinterpret readout counts
    as electron trajectories.  That is rejected unless the caller explicitly
    requests a response-weighted diagnostic.
    """

    if isinstance(cube, FourDSTEMArtifact):
        data = cube.data
        active_calibration = cube.calibration
        stored_plan = cube.metadata.get("record_plane_plan_fingerprint")
        response_record = cube.metadata.get("detector_response", {})
        neutral_response = bool(
            response_record.get("quantum_efficiency") == 1.0
            and response_record.get("charge_spread_sigma_px_xy") == [0.0, 0.0]
            and response_record.get("dark_electrons_per_pixel") == 0.0
            and response_record.get("read_noise_electrons_rms") == 0.0
            and response_record.get("saturation_electrons") is None
            and response_record.get("gain_counts_per_electron") == 1.0
            and response_record.get("offset_counts") == 0.0
            and response_record.get("poisson_enabled") is False
        )
        if not neutral_response and not bool(allow_detector_response_weighting):
            raise ValueError(
                "Physical record-plane routing requires a transport-neutral "
                "4D-STEM cube; detector response belongs after interception"
            )
        if calibration is not None and calibration.digest != active_calibration.digest:
            raise ValueError("Explicit and stored 4D-STEM calibrations differ")
    else:
        data = np.asarray(cube)
        stored_plan = None
        if calibration is None:
            raise ValueError("An in-memory 4D-STEM cube requires explicit calibration")
        active_calibration = calibration
    if tuple(data.shape) != active_calibration.shape:
        raise ValueError("4D-STEM cube shape does not match its calibration")
    if plan.time_dependent_deflection and active_calibration.scan_times_s is None:
        raise ValueError("Dynamic recording deflection requires a 4D-STEM cube with recorded scan times")
    if plan.scan_times_s is not None:
        if active_calibration.scan_times_s is None or not np.array_equal(
            plan.scan_times_s, active_calibration.scan_times_s,
        ):
            raise ValueError("Record-plane plan scan times differ from the 4D-STEM calibration")
    elif plan.time_dependent_deflection:
        raise ValueError("Build the dynamic record-plane plan with the cube's recorded scan times")
    points = int(chunk_scan_points)
    if points < 1:
        raise ValueError("Runtime detector integration chunk size must be positive")
    detector_keys = tuple(plane.key for plane in plan.planes if plane.kind == "detector")
    images = {key: np.zeros(data.shape[:2], dtype=np.float64) for key in detector_keys}
    surviving = np.zeros(data.shape[:2], dtype=np.float64)
    total_balance_error = 0.0
    angle = np.stack(
        (
            active_calibration.angle_x_mrad * 1.0e-3,
            active_calibration.angle_y_mrad * 1.0e-3,
        ),
        axis=-1,
    )[None, ...]
    flat_data = data.reshape((-1,) + data.shape[-2:])
    flat_scan_x = active_calibration.scan_x_um.reshape(-1)
    flat_scan_y = active_calibration.scan_y_um.reshape(-1)
    flat_images = {key: values.reshape(-1) for key, values in images.items()}
    flat_surviving = surviving.reshape(-1)
    for start in range(0, flat_data.shape[0], points):
        stop = min(start + points, flat_data.shape[0])
        block = np.asarray(flat_data[start:stop], dtype=float)
        if not np.all(np.isfinite(block)) or np.any(block < 0.0):
            raise ValueError("4D-STEM detector values must be finite and non-negative")
        position = np.stack(
            (
                flat_scan_x[start:stop] * 1.0e-6,
                flat_scan_y[start:stop] * 1.0e-6,
            ),
            axis=-1,
        )[:, None, None, :]
        routed = route_record_planes(
            plan, position, angle, weights=block, scan_slice=slice(start, stop),
        )
        if not result_matches_plan(routed, plan):
            raise RuntimeError("Record-plane result lost its plan provenance")
        for interaction in routed.interactions:
            if interaction.plane.kind != "detector":
                continue
            signal = np.where(interaction.signal_mask, block, 0.0)
            flat_images[interaction.plane.key][start:stop] = np.sum(
                signal,
                axis=(-2, -1),
                dtype=np.float64,
            )
        flat_surviving[start:stop] = np.sum(
            np.where(routed.surviving_mask, block, 0.0),
            axis=(-2, -1),
            dtype=np.float64,
        )
        total_balance_error += float(routed.balance_error)
    return RuntimeDetectorImages(
        plan_fingerprint=plan.fingerprint,
        calibration_fingerprint=active_calibration.digest,
        images=MappingProxyType(images),
        surviving_weight=surviving,
        balance_error=total_balance_error,
        capture_plan_fingerprint=stored_plan,
        plan_changed_since_capture=bool(
            stored_plan is not None and stored_plan != plan.fingerprint
        ),
    )


def runtime_detector_images_match_plan(
    result: RuntimeDetectorImages,
    plan: RecordPlanePlan,
) -> bool:
    """Reject a cached physical integration after any plan/geometry change."""

    return bool(result.plan_fingerprint == plan.fingerprint)
