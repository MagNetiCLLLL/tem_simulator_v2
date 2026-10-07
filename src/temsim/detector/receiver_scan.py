"""Bounded first-order scan previews from a captured physical receiver image.

This module does not acquire electrons. It averages translated copies of the
particles already intercepted in one executed result. Scan-dependent material
interactions and upstream clipping require new transport and are not inferred.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from temsim.detector.receiver_image import ReceiverImage, receiver_image
from temsim.optics.shared_deflectors import shared_channel_enabled


# These describe the executed static seed and its receiver, not a newly
# acquired exposure. Exclude seed current, dose, exposure and collected totals:
# none of those are measurements of the translated scan preview.
_STATIC_SEED_PROVENANCE = (
    "model", "capture_workflow", "capture_request_identity",
    "capture_manifest_identity", "capture_model_signature", "particle_source",
    "section_target_z_mm", "section_physics_scope", "projector_mode",
    "receiver_key", "receiver_z_mm", "receiver_width_mm", "receiver_inserted",
    "physical_area_extent_mm", "axis_convention", "probability_convention",
    "point_spread_model", "point_spread_status", "point_spread_source", "coherence",
)


def _readonly(values):
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _frozen_metadata(value):
    """Detach captured metadata, including any nested mutable values."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _frozen_metadata(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_frozen_metadata(item) for item in value)
    if isinstance(value, np.ndarray):
        result = np.array(value, copy=True)
        result.setflags(write=False)
        return result
    return value


@dataclass(frozen=True)
class ReceiverScanPreview:
    """One retained raster preview; all trajectories use ``array[row, column]``.

    ``trajectory_*`` are calibrated deflections from the undeflected reference,
    not absolute beam centroids. ``displacement_*`` subtract the deflection at
    the static particle result's captured time. No physical dose is claimed.
    """

    key: str
    status: str
    detail: str
    image: ReceiverImage | None = None
    trajectory_x_mm: np.ndarray | None = None
    trajectory_y_mm: np.ndarray | None = None
    displacement_x_mm: np.ndarray | None = None
    displacement_y_mm: np.ndarray | None = None
    times_s: np.ndarray | None = None
    dwell_weights: np.ndarray | None = None
    baseline_offset_mm: tuple[float, float] | None = None
    metrics: Mapping | None = None


def _active(component):
    return (component is not None and shared_channel_enabled(component)
            and bool(getattr(component, "scan_enabled", False)))


def _axis_weights(indices, count):
    """Number of actual equal-dwell pixels represented by each sampled pixel."""
    indices = np.asarray(indices, dtype=int)
    if (len(indices) < 2 or np.any(np.diff(indices) <= 0)
            or indices[0] < 0 or indices[-1] >= count):
        raise ValueError("The retained raster needs ordered, distinct sampled rows and columns")
    boundaries = np.r_[0, (indices[:-1] + indices[1:]) // 2 + 1, count]
    return np.diff(boundaries) / float(count)


def _captured_raster(result, key):
    state = getattr(result, "state_snapshot", None)
    geometry = getattr(result, "scan_geometry", None)
    if state is None or geometry is None:
        raise ValueError("No captured scan geometry. Calculate imaging to retain one raster first.")
    unavailable = getattr(geometry, "unavailable_planes", {})
    if key in unavailable:
        raise ValueError(str(unavailable[key]))
    coordinates = getattr(geometry, "plane_positions_um", {}).get(key)
    if coordinates is None:
        raise ValueError("The selected receiver has no captured scan trajectory")
    ac, descan = getattr(state, "ac_deflector", None), getattr(state, "descan_deflector", None)
    ac_on, descan_on = _active(ac), _active(descan)
    if not (ac_on or descan_on):
        raise ValueError("The captured AC and Descan rasters are both off")
    if (ac_on != bool(geometry.ac_enabled) or descan_on != bool(geometry.descan_enabled)):
        raise ValueError("Captured scan geometry and particle settings disagree")
    driver = ac if ac_on else descan
    nx, ny = int(driver.scan_pixels_x), int(driver.scan_lines)
    period = float(driver.scan_frame_period_s)
    if (nx < 2 or ny < 2 or not math.isfinite(period) or period <= 0
            or nx != int(geometry.requested_pixels_x) or ny != int(geometry.requested_pixels_y)):
        raise ValueError("Captured raster dimensions or frame period are invalid")
    if ac_on and descan_on and (
            int(ac.scan_pixels_x) != int(descan.scan_pixels_x)
            or int(ac.scan_lines) != int(descan.scan_lines)
            or not math.isclose(float(ac.scan_frame_period_s), float(descan.scan_frame_period_s),
                                rel_tol=1e-12, abs_tol=0.)):
        raise ValueError("Preview requires AC and Descan to share the captured raster clock")
    times = np.asarray(geometry.times_s, dtype=float)
    xy = np.stack(coordinates, axis=-1).astype(float) * 1e-3
    if (times.ndim != 2 or min(times.shape, default=0) < 2 or max(times.shape) > 128
            or xy.shape != times.shape + (2,)
            or not np.all(np.isfinite(times)) or not np.all(np.isfinite(xy))):
        raise ValueError("Captured scan coordinates must match a finite 2-D preview raster (at most 128 per axis)")
    row_ids = np.floor(times[:, 0] / period * ny).astype(int)
    col_ids = np.rint((times[0] / period * ny - row_ids[0]) * nx - .5).astype(int)
    expected_times = (row_ids[:, None] + (col_ids[None, :] + .5) / nx) / ny * period
    if not np.allclose(times, expected_times, rtol=0., atol=period*1e-11):
        raise ValueError("Captured times do not describe one pixel-centre raster")
    weights = _axis_weights(row_ids, ny)[:, None] * _axis_weights(col_ids, nx)[None, :]

    # A first-order plane map is affine in the two physical raster factors.
    # Fit the retained 2-D map, then evaluate the true captured time, including
    # frame wrap and the unsampled left edge at t=0. Never treat the first
    # retained pixel centre as the static calculation's baseline.
    factors = np.asarray([driver.scan_factors(float(t)) for t in times.ravel()], dtype=float)
    design = np.column_stack((np.ones(times.size), factors))
    coefficients, _, rank, _ = np.linalg.lstsq(design, xy.reshape(-1, 2), rcond=None)
    if rank != 3:
        raise ValueError("Captured raster does not span both scan axes")
    residual = float(np.max(np.abs(design @ coefficients - xy.reshape(-1, 2))))
    if residual > max(1e-12, float(np.max(np.abs(xy))) * 1e-8):
        raise ValueError("Captured trajectory is not a supported first-order affine raster")
    baseline_time = float(getattr(state, "simulation_time_s", 0.))
    if not math.isfinite(baseline_time):
        raise ValueError("Captured particle time must be finite")
    baseline_factors = np.asarray(driver.scan_factors(baseline_time), dtype=float)
    if baseline_factors.shape != (2,) or not np.all(np.isfinite(baseline_factors)):
        raise ValueError("Captured raster baseline is invalid")
    baseline = np.r_[1., baseline_factors] @ coefficients
    return state, geometry, times, xy, xy-baseline, weights, baseline, {
        "frame_period_s": period, "baseline_time_s": baseline_time,
        "sampled_raster_shape": tuple(times.shape), "physical_raster_shape": (ny, nx),
        "decimated_raster": times.shape != (ny, nx), "affine_fit_residual_mm": residual,
        "trajectory_coordinates": "calibrated deflection relative to undeflected reference",
        "plane_role": getattr(geometry, "plane_roles", {}).get(key, "unclassified"),
        "descan_enabled": descan_on,
    }


def _shift_kernel(shifts, weights, shape, pixel_mm):
    """Bilinear subpixel translation kernel with zero-filled finite support."""
    ny, nx = shape
    kernel = np.zeros((2*ny-1, 2*nx-1), dtype=float)
    offsets = np.asarray(shifts, dtype=float).reshape(-1, 2) / np.asarray(pixel_mm)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    # Far-away shifts cannot overlap the finite sensor. Reject them before
    # conversion to integers, which also avoids overflow for extreme optics.
    close = (np.abs(offsets[:, 0]) < nx) & (np.abs(offsets[:, 1]) < ny)
    offsets, weights = offsets[close], weights[close]
    lower = np.floor(offsets).astype(int)
    fractions = offsets-lower
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        x, y = lower[:, 0]+dx+nx-1, lower[:, 1]+dy+ny-1
        contribution = weights * (fractions[:, 0] if dx else 1-fractions[:, 0])
        contribution *= fractions[:, 1] if dy else 1-fractions[:, 1]
        valid = (x >= 0) & (x < kernel.shape[1]) & (y >= 0) & (y < kernel.shape[0])
        np.add.at(kernel, (y[valid], x[valid]), contribution[valid])
    return kernel


def _shift_image(values, shift, pixel_mm):
    """Translate a pair of images without an FFT or periodic wraparound."""
    ny, nx = values.shape[-2:]
    offsets = np.asarray(shift, dtype=float) / np.asarray(pixel_mm)
    moved = np.zeros_like(values)
    if abs(offsets[0]) >= nx or abs(offsets[1]) >= ny:
        return moved
    lower = np.floor(offsets).astype(int)
    fractions = offsets - lower
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        ix, iy = int(lower[0]+dx), int(lower[1]+dy)
        weight = (fractions[0] if dx else 1-fractions[0])
        weight *= fractions[1] if dy else 1-fractions[1]
        if weight == 0. or abs(ix) >= nx or abs(iy) >= ny:
            continue
        x0, x1 = max(0, ix), min(nx, nx+ix)
        y0, y1 = max(0, iy), min(ny, ny+iy)
        moved[..., y0:y1, x0:x1] += weight * values[..., y0-iy:y1-iy, x0-ix:x1-ix]
    return moved


@dataclass(frozen=True)
class PreparedReceiverScan:
    """One reusable geometric preview seed; no transport or timer is started.

    Trajectories and captured metadata are detached, read-only snapshots.
    Only the seed transforms and one complete frame are cached, so advancing
    playback never retains a history of images. ``base_image`` is the physical
    seed image, for locating its captured centroid without another histogram.
    """

    key: str
    status: str
    detail: str
    base_image: ReceiverImage | None = None
    trajectory_x_mm: np.ndarray | None = None
    trajectory_y_mm: np.ndarray | None = None
    displacement_x_mm: np.ndarray | None = None
    displacement_y_mm: np.ndarray | None = None
    times_s: np.ndarray | None = None
    dwell_weights: np.ndarray | None = None
    baseline_offset_mm: tuple[float, float] | None = None
    metrics: Mapping = field(default_factory=lambda: MappingProxyType({}))
    _values: np.ndarray | None = field(default=None, repr=False)
    _shifts: np.ndarray | None = field(default=None, repr=False)
    _stationary: np.ndarray | None = field(default=None, repr=False)
    _active: np.ndarray | None = field(default=None, repr=False)
    _pixel_mm: tuple[float, float] | None = field(default=None, repr=False)
    _exposure_s: float | None = field(default=None, repr=False)
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def _averaged(self, end):
        """Dwell-weighted prefix, using a bounded convolution grid."""
        if end == self.times_s.size and "full_frame" in self._cache:
            return self._cache["full_frame"]
        weights = self.dwell_weights.reshape(-1)[:end]
        stationary = self._stationary[:end]
        stationary_weight = float(np.sum(weights[stationary]))
        # Exact accepted hits can occupy circular-edge bins whose centres are
        # outside the sensor. Their stationary contribution must not be clipped.
        values = stationary_weight * self._values
        if np.any(~stationary):
            shape = self._values.shape[-2:]
            kernel = _shift_kernel(self._shifts[:end][~stationary], weights[~stationary],
                                   shape, self._pixel_mm)
            if np.any(kernel):
                from scipy.fft import irfftn, next_fast_len, rfftn
                if "seed_spectrum" not in self._cache:
                    fft_shape = tuple(next_fast_len(3*size-2, real=True) for size in shape)
                    spectrum = rfftn(self._values, s=fft_shape, axes=(-2, -1))
                    self._cache["seed_spectrum"] = (fft_shape, spectrum)
                fft_shape, spectrum = self._cache["seed_spectrum"]
                moved = irfftn(spectrum * rfftn(kernel, s=fft_shape),
                               s=fft_shape, axes=(-2, -1))
                ny, nx = shape
                moved = moved[..., ny-1:2*ny-1, nx-1:2*nx-1]
                values += np.where(self._active, np.maximum(moved, 0.), 0.)
        if end == self.times_s.size:
            values.setflags(write=False)
            self._cache["full_frame"] = (values, stationary_weight)
        return values, stationary_weight

    def preview(self, scan_index=None, *, accumulate=False):
        """Return a full frame, unit-brightness spot, or accumulated prefix.

        ``None`` integrates the full retained frame. An integer selects one
        flattened raster position; ``accumulate=True`` instead integrates all
        positions through that index, using their original full-frame weights.
        ``exposure_fraction`` always reports the represented fraction of frame
        dwell. A selected spot keeps unit image weight for visibility; prefix
        images keep full-frame normalization and grow with represented dwell.
        """
        if not isinstance(accumulate, (bool, np.bool_)):
            raise ValueError("Scan accumulation must be a boolean")
        if self.times_s is not None and scan_index is not None and (
                isinstance(scan_index, bool) or not isinstance(scan_index, (int, np.integer))
                or not 0 <= scan_index < self.times_s.size):
            raise ValueError("Scan position index is outside the retained raster")
        common = dict(trajectory_x_mm=self.trajectory_x_mm, trajectory_y_mm=self.trajectory_y_mm,
            displacement_x_mm=self.displacement_x_mm, displacement_y_mm=self.displacement_y_mm,
            times_s=self.times_s, dwell_weights=self.dwell_weights,
            baseline_offset_mm=self.baseline_offset_mm)
        if self.status != "GEOMETRIC_PREVIEW":
            return ReceiverScanPreview(self.key, self.status, self.detail, metrics=self.metrics, **common)
        index = None if scan_index is None else int(scan_index)
        weights = self.dwell_weights.reshape(-1)
        if index is None or accumulate:
            end = self.times_s.size if index is None else index+1
            values, stationary_weight = self._averaged(end)
            fraction = 1. if end == self.times_s.size else float(np.sum(weights[:end]))
            weighting = "full-frame dwell weights; no prefix renormalization"
        else:
            stationary_weight = float(self._stationary[index])
            values = (self._values if stationary_weight else
                      np.where(self._active, _shift_image(self._values, self._shifts[index],
                                                        self._pixel_mm), 0.))
            fraction = float(weights[index])
            weighting = "selected-position unit weight; exposure_fraction reports represented dwell"
        metrics = dict(self.metrics)
        metrics.update({"scan_index": index, "accumulate": bool(accumulate),
            "exposure_fraction": fraction, "probability_weighting": weighting,
            "preview_response_probability": float(np.sum(values[1])),
            "stationary_dwell_weight": stationary_weight})
        title = ("Frame build-up preview" if accumulate else
                 "Single scan preview" if index is None else "Scan position preview")
        detail = (title + " — first-order geometry, not an acquired exposure. Translates only particles "
            "already intercepted at the captured time. Scan-dependent specimen scattering and upstream "
            "clipping are not recalculated; previously missed particles cannot reappear. Physical detector "
            "counts and current are unavailable for this preview. Moved bins use approximate pixel-centre "
            "sensor acceptance; stationary exact-hit bins retain their captured weights.")
        if accumulate:
            detail += " Build-up retains full-frame normalization; signal grows with represented dwell."
        if metrics["decimated_raster"]:
            detail += " The retained raster is decimated; dwell-weighted quadrature approximates the full frame."
        image = ReceiverImage(key=self.base_image.key, name=self.base_image.name,
            status="GEOMETRIC_PREVIEW", detail=detail, ideal_probability=values[0],
            response_probability=values[1], x_mm=self.base_image.x_mm, y_mm=self.base_image.y_mm,
            exposure_s=self._exposure_s, expected_electrons=None, current_pa=None, metrics=metrics)
        immutable_metrics = _frozen_metadata(metrics)
        object.__setattr__(image, "metrics", immutable_metrics)
        return ReceiverScanPreview(self.key, "GEOMETRIC_PREVIEW", detail, image=image,
                                   metrics=immutable_metrics, **common)


def prepare_receiver_scan(result, key, *, pixels=192, exposure_s=None):
    """Validate captured geometry and sample its physical receiver image once."""
    key = str(key)
    if isinstance(pixels, bool) or not isinstance(pixels, (int, np.integer)) or not 16 <= pixels <= 512:
        raise ValueError("Scan-preview image needs 16 to 512 pixels per axis")
    if exposure_s is not None:
        exposure_s = float(exposure_s)
        if not math.isfinite(exposure_s) or exposure_s <= 0:
            raise ValueError("Preview exposure must be positive and finite")
    try:
        state, geometry, times, xy, shifts, weights, baseline, metrics = _captured_raster(result, key)
    except (AttributeError, TypeError, ValueError) as error:
        return PreparedReceiverScan(key, "UNAVAILABLE", str(error))
    common = dict(
        trajectory_x_mm=_readonly(xy[..., 0]), trajectory_y_mm=_readonly(xy[..., 1]),
        displacement_x_mm=_readonly(shifts[..., 0]), displacement_y_mm=_readonly(shifts[..., 1]),
        times_s=_readonly(times), dwell_weights=_readonly(weights),
        baseline_offset_mm=tuple(float(v) for v in baseline),
    )
    exposure = float(metrics["frame_period_s"] if exposure_s is None else exposure_s)
    base = receiver_image(result, key, pixels=int(pixels), exposure_s=exposure)
    if base.status != "AVAILABLE":
        return PreparedReceiverScan(key, base.status,
            "Geometric trajectory only; no usable physical receiver image. " + base.detail,
            base_image=base, metrics=_frozen_metadata(metrics), **common)
    metrics.update({"static_seed_" + name: base.metrics[name]
                    for name in _STATIC_SEED_PROVENANCE if name in base.metrics})
    ideal, response = np.asarray(base.ideal_probability), np.asarray(base.response_probability)
    x, y = np.asarray(base.x_mm), np.asarray(base.y_mm)
    if (ideal.shape != (len(y), len(x)) or response.shape != ideal.shape
            or len(x) < 2 or len(y) < 2):
        raise ValueError("Static receiver image axes and probabilities disagree")
    pixel_mm = (float(x[1]-x[0]), float(y[1]-y[0]))
    if (min(pixel_mm) <= 0 or not np.all(np.isfinite(pixel_mm))
            or not np.allclose(np.diff(x), pixel_mm[0])
            or not np.allclose(np.diff(y), pixel_mm[1])):
        raise ValueError("Static receiver image requires uniform increasing axes")
    rounding_mm = 64*np.finfo(float).eps*max(float(np.max(np.abs(xy))),
                                           float(np.max(np.abs(baseline))), 1e-12)
    stationary = np.all(np.abs(shifts.reshape(-1, 2)) <= rounding_mm, axis=1)
    stationary.setflags(write=False)
    plane = next(p for p in state.recording_planes if str(p.key) == key)
    xx, yy = np.meshgrid(x, y)
    active = np.array(plane.hit_mask(xx, yy), dtype=bool, copy=True)
    active.setflags(write=False)
    metrics.update({"model": "first_order_receiver_scan_preview", "actual_exposure": False,
        "scan_dependent_material_calculated": False, "scan_dependent_upstream_clipping_calculated": False,
        "static_accepted_probability": float(np.sum(ideal)),
        "surface_binning": "unmoved exact-hit bins retained; moved bins use conservative sensor-centre acceptance",
        "sampling": "nearest sampled raster cell dwell quadrature"})
    return PreparedReceiverScan(key, "GEOMETRIC_PREVIEW", "Prepared captured geometric scan preview.",
        base_image=base, metrics=_frozen_metadata(metrics), _values=_readonly(np.stack((ideal, response))),
        _shifts=_readonly(shifts.reshape(-1, 2)), _stationary=stationary, _active=active,
        _pixel_mm=pixel_mm, _exposure_s=exposure, **common)


def receiver_scan_preview(result, key, *, pixels=192, exposure_s=None, scan_index=None,
                          accumulate=False):
    """Return one geometric preview; use ``prepare_receiver_scan`` for playback.

    No particle/material transport, timer, physical counts or current are
    created by the preview. Existing single-frame and position calls keep
    their probability weighting; accumulation uses full-frame dwell weights.
    """
    return prepare_receiver_scan(result, key, pixels=pixels, exposure_s=exposure_s).preview(
        scan_index, accumulate=accumulate)
