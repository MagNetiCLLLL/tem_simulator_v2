"""Conservative annular DF sizing from an immutable recording-plane map.

This proposes geometry only.  It does not change a plan, move a detector, tune
the column, or compensate for upstream stops.  In particular, an upstream
HAADF may intercept part of the proposed DF angular band; production routing
must retain that sequential shadow.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from temsim.physics.record_plane import RecordPlanePlan


@dataclass(frozen=True, slots=True)
class DarkFieldGeometryProposal:
    """Dimensions in mm and conservative angles relative to the probe chief.

    Camera lengths are the minimum/maximum singular values of ``J_diff`` in
    m/rad (numerically metres per radian), not a nominal preset camera length.
    Angular bounds describe possible geometric acceptance before other stops;
    they do not promise full azimuthal transmission or nonzero specimen signal.
    """

    status: str
    detail: str
    inner_diameter_mm: float | None = None
    outer_width_mm: float | None = None
    angular_inner_mrad: float | None = None
    angular_outer_mrad: float | None = None
    direct_disk_max_radius_mm: float | None = None
    direct_disk_clearance_mm: float | None = None
    target_inner_mrad: float | None = None
    target_outer_mrad: float | None = None
    maximum_affine_shift_mm: float | None = None
    camera_length_min_m: float | None = None
    camera_length_max_m: float | None = None

    @property
    def supported(self) -> bool:
        return self.status == "proposed"


def _positive_finite(value, name: str, *, allow_zero: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or (number < 0 if allow_zero else number <= 0):
        raise ValueError(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}")
    return number


def propose_dark_field_geometry(
    plan: RecordPlanePlan,
    scan_positions_m,
    probe_semiangle_mrad: float,
    *,
    target_inner_mrad: float | None = None,
    target_outer_mrad: float | None = None,
    detector_key: str = "df",
    chamber_inner_diameter_mm: float | None = None,
    probe_center_mrad=(0.0, 0.0),
) -> DarkFieldGeometryProposal:
    """Size a centred circular annulus to exclude the complete supplied disk.

    ``scan_positions_m`` contains physical specimen coordinates, including the
    specimen chief position and raster displacement exactly once.  The angular
    chief is supplied separately in mrad.  For a timed plan, positions must
    contain the complete raster in the same C order as ``plan.scan_times_s``;
    both flattened ``(N, 2)`` and raster ``(ny, nx, 2)`` arrays are accepted.

    For each scan point the detector-relative displacement is
    ``d = J_img*r + static_offset + timed_offset + J_diff*chief - centre``.
    If D=max(norm(d)) and s_min/s_max are the singular values of J_diff, setting
    ``R_in=D+s_max*theta_inner`` and ``R_out=s_min*theta_outer-D`` guarantees
    acceptance lies within the requested angular bounds, even for anisotropic
    maps.  Large anisotropy/offsets can make such a circular annulus impossible.

    The default inner angle is alpha + max(5 mrad, 20% alpha).  Preserve the
    existing outer diameter when it leaves at least 5 mrad of radial width;
    otherwise use inner + 20 mrad.  Explicit target bounds are never silently
    clipped to a chamber or to another detector's acceptance.

    Input errors raise ValueError.  Valid but infeasible or incomplete optical
    configurations return ``status='unsupported'`` with a reason.  This bound
    applies to the supplied first-order map and probe disk, not unmodelled
    higher-order ray aberrations or a probe percentile outside that disk.
    """

    alpha = _positive_finite(probe_semiangle_mrad, "Probe semiangle", allow_zero=True)
    positions = np.asarray(scan_positions_m, dtype=float)
    if positions.shape[-1:] != (2,) or not positions.size or not np.all(np.isfinite(positions)):
        raise ValueError("Scan positions must be a non-empty finite array ending in dimension 2")
    positions = positions.reshape(-1, 2)
    chief = np.asarray(probe_center_mrad, dtype=float)
    if chief.shape != (2,) or not np.all(np.isfinite(chief)):
        raise ValueError("Probe angular centre must contain two finite values in mrad")
    chamber = None if chamber_inner_diameter_mm is None else _positive_finite(
        chamber_inner_diameter_mm, "Chamber inner diameter"
    )
    minimum_inner = alpha + max(5.0, 0.2 * alpha)
    inner = minimum_inner if target_inner_mrad is None else _positive_finite(
        target_inner_mrad, "Target inner angle"
    )
    outer = None if target_outer_mrad is None else _positive_finite(
        target_outer_mrad, "Target outer angle"
    )
    if inner < minimum_inner - 1.0e-12:
        return DarkFieldGeometryProposal(
            "unsupported", f"Target inner angle must be at least {minimum_inner:.6g} mrad "
            "to exclude the supplied direct disk with the required margin."
        )
    if outer is not None and outer <= inner:
        return DarkFieldGeometryProposal("unsupported", "Target outer angle must exceed the inner angle.")
    found = next((i for i, plane in enumerate(plan.planes) if plane.key == detector_key), None)
    if found is None:
        return DarkFieldGeometryProposal("unsupported", f"Inserted detector {detector_key!r} is absent from the recording plan.")
    plane, transfer = plan.planes[found], plan.transfers[found]
    if plane.kind != "detector" or plane.geometry.lower() != "annulus":
        return DarkFieldGeometryProposal("unsupported", "DF sizing requires an existing annular detector.")
    if plan.time_dependent_deflection and plan.scan_times_s is None:
        return DarkFieldGeometryProposal("unsupported", "Time-dependent deflection requires a recording plan with the full raster times.")
    if plan.scan_times_s is not None and plan.scan_times_s.size != len(positions):
        return DarkFieldGeometryProposal("unsupported", "Scan positions must match every point in the recording plan's timed raster.")

    singular = np.linalg.svd(transfer.j_diff_m_per_rad, compute_uv=False)
    s_max, s_min = float(singular[0]), float(singular[-1])
    if s_max <= 0 or s_min <= s_max * 1.0e-12:
        return DarkFieldGeometryProposal("unsupported", "The current DF angular map is singular; annular angular sizing is unavailable.")
    centre_m = np.array((plane.offset_x_mm, plane.offset_y_mm)) * 1.0e-3
    displacements = (
        positions @ transfer.j_img.T
        + np.asarray(transfer.position_offset_m)
        + transfer.j_diff_m_per_rad @ (chief * 1.0e-3)
        - centre_m
    )
    if plan.scan_position_offsets_m:
        displacements = displacements + plan.scan_position_offsets_m[found].reshape(-1, 2)
    displacement = float(np.max(np.linalg.norm(displacements, axis=-1)))
    r_inner = displacement + s_max * inner * 1.0e-3
    preserved_outer = False
    if outer is None:
        existing_radius = plane.outer_width_mm * 0.5e-3
        if existing_radius - r_inner >= s_max * 5.0e-3:
            # Express the existing physical radius using the same conservative
            # angular envelope as an explicitly requested outer angle.
            outer = (existing_radius + displacement) / s_min * 1.0e3
            preserved_outer = True
        else:
            outer = inner + 20.0
    r_outer = s_min * outer * 1.0e-3 - displacement
    if not math.isfinite(r_inner + r_outer) or r_outer <= r_inner:
        return DarkFieldGeometryProposal(
            "unsupported", "The current angular anisotropy and scan/beam displacement leave no positive circular DF width within the requested angular band."
        )
    exterior_radius_mm = (r_outer + float(np.linalg.norm(centre_m))) * 1.0e3
    if chamber is not None and exterior_radius_mm > chamber / 2.0 + 1.0e-9:
        return DarkFieldGeometryProposal(
            "unsupported", f"Proposed DF requires {2.0 * exterior_radius_mm:.6g} mm of chamber diameter including detector offset, exceeding the supplied {chamber:.6g} mm inner diameter."
        )
    direct_radius = displacement + s_max * alpha * 1.0e-3
    return DarkFieldGeometryProposal(
        status="proposed",
        detail=(
            "The supplied direct disk is excluded at every supplied scan position using the current signed first-order map, beam centre and deflection offsets. "
            + ("The existing outer diameter is retained. " if preserved_outer else "The requested/default angular band determines the outer diameter. ")
            + "Upstream apertures and HAADF retain their actual sequential shadow; these bounds do not guarantee signal or full azimuthal transmission."
            + (" No mechanical chamber bound was supplied." if chamber is None else " The supplied mechanical chamber bound is satisfied.")
        ),
        inner_diameter_mm=2.0e3 * r_inner,
        outer_width_mm=2.0e3 * r_outer,
        angular_inner_mrad=(r_inner - displacement) / s_max * 1.0e3,
        angular_outer_mrad=(r_outer + displacement) / s_min * 1.0e3,
        direct_disk_max_radius_mm=direct_radius * 1.0e3,
        direct_disk_clearance_mm=(r_inner - direct_radius) * 1.0e3,
        target_inner_mrad=inner,
        target_outer_mrad=outer,
        maximum_affine_shift_mm=displacement * 1.0e3,
        camera_length_min_m=s_min,
        camera_length_max_m=s_max,
    )
