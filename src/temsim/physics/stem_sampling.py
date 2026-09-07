"""Conservative STEM detector sampling checks; never change signal values.

Angles are mrad, real-space sampling is Angstrom, and physical transfers use
metres/radian. Bounds include the complete raster and affine detector offsets.
They deliberately ignore upstream stops: a clipped detector is not proof of
adequate wave sampling. Full coverage is necessary, not a convergence test.
"""
from __future__ import annotations

import math
import numpy as np


def detector_angular_bounds(detectors, *, positions_m, record_plane_plan=None,
                            detector_center_shifts_mrad=None):
    """Return conservative radial acceptance bounds over all scan positions."""
    planes = {} if record_plane_plan is None else {
        plane.key: (index, plane, transfer)
        for index, (plane, transfer) in enumerate(zip(record_plane_plan.planes, record_plane_plan.transfers))
        if plane.kind == "detector"
    }
    bounds = {}
    for detector in detectors:
        key = detector.key
        if key in planes:
            index, plane, transfer = planes[key]
            matrix = np.asarray(transfer.j_diff_m_per_rad, dtype=float)
            centre_m = np.array((plane.offset_x_mm, plane.offset_y_mm)) * 1e-3
            displaced = (centre_m - np.asarray(transfer.position_offset_m)
                         - np.asarray(positions_m) @ np.asarray(transfer.j_img).T)
            if getattr(record_plane_plan, "scan_position_offsets_m", ()):
                offsets = record_plane_plan.scan_position_offsets_m[index].reshape(-1, 2)
                if offsets.shape != displaced.shape:
                    raise ValueError("Record-plane sampling offsets must match the scan positions")
                displaced = displaced - offsets
            geometry = plane.geometry
            inner_m = plane.inner_diameter_mm * 0.5e-3 if geometry == "annulus" else 0.0
            outer_m = plane.outer_width_mm * 0.5e-3
        elif hasattr(detector, "sample_to_detector_m_per_rad"):
            matrix = np.asarray(detector.sample_to_detector_m_per_rad, dtype=float)
            physical = detector.detector
            geometry = physical.geometry
            inner_m = physical.inner_diameter_mm * 0.5e-3 if geometry == "annulus" else 0.0
            outer_m = physical.outer_width_mm * 0.5e-3
            centre_m = np.array((getattr(physical, "centre_offset_x_mm", 0.0),
                                 getattr(physical, "centre_offset_y_mm", 0.0))) * 1e-3
            displaced = centre_m[None, :]
        else:
            shift = (detector_center_shifts_mrad or {}).get(key)
            distance = 0.0 if shift is None else float(np.max(np.hypot(*shift)))
            bounds[key] = (max(0.0, detector.inner_mrad - distance),
                           detector.outer_mrad + distance)
            continue
        singular = np.linalg.svd(matrix, compute_uv=False)
        if singular[-1] <= 1e-15:
            # At an image-conjugate plane there is no finite angular aperture.
            bounds[key] = (0.0, math.inf)
            continue
        if geometry in {"square", "rectangle", "camera"}:
            outer_m *= math.sqrt(2.0)
        centres = np.linalg.solve(matrix, np.asarray(displaced).reshape(-1, 2).T).T
        distance = np.linalg.norm(centres, axis=-1)
        extra = (detector_center_shifts_mrad or {}).get(key)
        # The record-plane route already owns its affine offsets. Do not add
        # the legacy descan diagnostic a second time.
        shift_mrad = 0.0 if key in planes or extra is None else float(np.max(np.hypot(*extra)))
        outer = outer_m / singular[-1] * 1e3
        inner = inner_m / singular[0] * 1e3
        lower = max(0.0, inner - float(np.max(distance)) * 1e3 - shift_mrad,
                    float(np.min(distance)) * 1e3 - outer - shift_mrad)
        upper = outer + float(np.max(distance)) * 1e3 + shift_mrad
        bounds[key] = (lower, upper)
    return bounds


def detector_sampling_report(bounds_mrad, *, maximum_angle_mrad,
                             wavelength_angstrom, requested_fov_angstrom,
                             requested_grid_pixels, bandwidth_fraction,
                             probe_semiangle_mrad, potential_storage_bytes=0):
    """JSON-compatible diagnostics and a grid proposal, not an automatic resize.

    A theta-radius <= sin(FFT angular cutoff) is conservatively contained by
    the reciprocal disk even for diagonal angles. Two extra pixels per side
    protect against grid rounding by a commensurate atomistic cell.
    """
    maximum = float(maximum_angle_mrad)
    wavelength = float(wavelength_angstrom)
    fov = float(requested_fov_angstrom)
    bandwidth = float(bandwidth_fraction)
    if not (math.isfinite(maximum) and 0 < maximum < 1571
            and math.isfinite(wavelength) and wavelength > 0
            and math.isfinite(fov) and fov > 0 and 0 < bandwidth <= 1
            and math.isfinite(probe_semiangle_mrad) and probe_semiangle_mrad >= 0
            and int(requested_grid_pixels) >= 16):
        raise ValueError("Invalid STEM wave-grid sampling inputs")
    supported = math.sin(maximum * 1e-3) * 1e3
    rows = {}
    for key, (lower, upper) in bounds_mrad.items():
        if not (math.isfinite(lower) and lower >= 0 and upper > lower):
            raise ValueError(f"{key}: invalid angular acceptance bounds")
        if not math.isfinite(upper):
            status = "unknown"
        elif upper <= supported:
            status = "full"
        elif lower >= maximum:
            status = "outside"
        else:
            status = "partial"
        rows[key] = {
            "status": status,
            "required_inner_mrad": float(lower),
            "required_outer_mrad": float(upper) if math.isfinite(upper) else None,
            "sampled_outer_mrad": min(float(upper), maximum),
            "overlaps_illumination_disk": float(lower) < float(probe_semiangle_mrad),
        }
    complete = all(row["status"] == "full" for row in rows.values())
    required = max([float(probe_semiangle_mrad)] + [value[1] for value in bounds_mrad.values()])
    pixels = None
    if math.isfinite(required) and required < 1570:
        raw = math.ceil(2 * fov * required * 1e-3 / (wavelength * bandwidth)) + 4
        pixels = max(32, int(requested_grid_pixels), int(math.ceil(raw / 32) * 32))
    factor = None if pixels is None else (pixels / int(requested_grid_pixels)) ** 2
    return {
        "version": 1,
        "scope": "conservative full-raster angular bounds; before sequential stops",
        "maximum_simulated_angle_mrad": maximum,
        "probe_semiangle_mrad": float(probe_semiangle_mrad),
        "illumination_covered": float(probe_semiangle_mrad) <= supported,
        "detectors": rows,
        "coverage_complete": complete and float(probe_semiangle_mrad) <= supported,
        "recommended_grid_pixels": pixels,
        "requested_grid_pixels": int(requested_grid_pixels),
        "grid_area_factor": factor,
        "estimated_potential_bytes": None if factor is None else math.ceil(potential_storage_bytes * factor),
        "remedy": "Increase Sample wave Grid; preserve FOV, detector geometry and lens strengths. Recheck after calculation.",
    }


def frame_sampling_report(metrics):
    """Read current diagnostics, or explicitly flag older unchecked wave caches."""
    report = metrics.get("detector_sampling")
    if report is not None:
        return report
    if metrics.get("model") in {"multislice_angle_resolved", "thin_phase_angle_resolved"}:
        return {"coverage_complete": False, "detectors": {}, "legacy_unchecked": True}
    return None
