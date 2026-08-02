"""Mechanical, field and trajectory views of the Energy Filter."""

import math

import numpy as np
from matplotlib.patches import Arc, Circle, Rectangle

from temsim.optics.energy_filter_sector import (
    magnetic_field_from_energy_filter,
    sector_from_energy_filter,
)


def _draw_m12_glyph(ax, centre_axes, element):
    """Small transverse mechanical view with twelve independent poles."""

    cx, cy = centre_axes
    ax.add_patch(Circle(
        (cx, cy),
        0.055,
        transform=ax.transAxes,
        fill=False,
        edgecolor="#6a1b9a",
        linewidth=1.0,
        clip_on=False,
    ))
    for angle in element.pole_angles_rad:
        ax.add_patch(Circle(
            (
                cx + 0.042 * math.cos(float(angle)),
                cy + 0.042 * math.sin(float(angle)),
            ),
            0.006,
            transform=ax.transAxes,
            facecolor="#8e24aa",
            edgecolor="none",
            clip_on=False,
        ))
    ax.text(
        cx,
        cy - 0.072,
        element.name,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6,
    )


def _draw_field_map(ax, energy_filter, sector):
    radius_mm = float(energy_filter.prism_radius_mm)
    entrance_mm = float(energy_filter.prism_entrance_s_mm)
    x_min = min(0.0, entrance_mm - 25.0)
    x_max = sector.exit_point_m[0] * 1.0e3 + 45.0
    z_min = sector.exit_point_m[2] * 1.0e3 - 45.0
    z_max = 25.0
    x = np.linspace(x_min, x_max, 150)
    z = np.linspace(z_min, z_max, 150)
    xx, zz = np.meshgrid(x, z)
    positions = np.stack(
        (xx * 1.0e-3, np.zeros_like(xx), zz * 1.0e-3),
        axis=-1,
    )
    field = magnetic_field_from_energy_filter(
        energy_filter
    ).field_at_global_positions_t(positions)
    magnitude = np.linalg.norm(field, axis=-1)
    nonzero = magnitude[magnitude > 1.0e-12]
    if nonzero.size:
        upper = max(float(np.percentile(nonzero, 98)), 1.0e-9)
        ax.contourf(
            xx,
            zz,
            np.clip(magnitude, 0.0, upper),
            levels=np.linspace(0.08 * upper, upper, 9),
            cmap="Blues",
            alpha=0.24,
            antialiased=True,
        )


def draw_energy_filter(ax, state, result):
    ax.clear()
    energy_filter = getattr(state, "energy_filter", None)
    ax.set_title(
        "Energy Filter: finite soft-edge fields + relativistic trajectories",
        fontsize=9,
    )
    ax.set_xlabel("global X / diagram u (mm)")
    ax.set_ylabel("global Z / diagram v (mm)")
    ax.grid(alpha=0.18)
    if energy_filter is None or not energy_filter.enabled:
        ax.text(
            0.5,
            0.5,
            "Energy Filter system is not installed",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return

    sector = sector_from_energy_filter(energy_filter)
    _draw_field_map(ax, energy_filter, sector)
    entrance_u = sector.entrance_point_m[0] * 1.0e3
    exit_u = sector.exit_point_m[0] * 1.0e3
    exit_v = sector.exit_point_m[2] * 1.0e3
    radius = float(energy_filter.prism_radius_mm)
    centre = (
        sector.centre_m[0] * 1.0e3,
        sector.centre_m[2] * 1.0e3,
    )
    bend_deg = math.degrees(float(sector.bend_angle_rad))
    for radial_offset, width in (
        (-0.5 * energy_filter.pole_gap_mm, 1.2),
        (0.0, 4.0),
        (0.5 * energy_filter.pole_gap_mm, 1.2),
    ):
        diameter = 2.0 * (radius + radial_offset)
        ax.add_patch(Arc(
            centre,
            diameter,
            diameter,
            theta1=0.0,
            theta2=bend_deg,
            color="#0d47a1",
            lw=width,
            alpha=0.35 if width > 2 else 0.8,
        ))

    ax.plot([0.0, entrance_u], [0.0, 0.0], color="#78909c", lw=1)
    downstream_end = (
        sector.exit_point_m
        + sector.exit_tangent
        * (
            float(energy_filter.output_detector_d_mm)
            + float(energy_filter.eels_plane_offset_mm)
        )
        * 1.0e-3
    )
    ax.plot(
        [exit_u, downstream_end[0] * 1.0e3],
        [exit_v, downstream_end[2] * 1.0e3],
        color="#78909c",
        lw=1,
    )
    ax.vlines(
        0.0,
        -0.5 * energy_filter.entrance_aperture_mm,
        0.5 * energy_filter.entrance_aperture_mm,
        color="#263238",
        lw=5,
    )

    for element in (
        energy_filter.entrance_m12,
        energy_filter.exit_m12,
    ):
        origin = element.frame.origin_m * 1.0e3
        ax.scatter(
            [origin[0]],
            [origin[2]],
            s=95,
            marker="s",
            facecolor="#8e24aa",
            edgecolor="#4a148c",
            alpha=0.55,
            zorder=5,
        )
        ax.annotate(
            element.name,
            (origin[0], origin[2]),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=6,
        )

    outgoing_s = sector.exit_tangent
    outgoing_x = sector.exit_frame.rotation_local_to_global[:, 0]

    def plane_segment(distance_mm, half_width_mm, **kwargs):
        centre_point = (
            sector.exit_point_m
            + outgoing_s * float(distance_mm) * 1.0e-3
        )
        endpoints = np.stack((
            centre_point - outgoing_x * half_width_mm * 1.0e-3,
            centre_point + outgoing_x * half_width_mm * 1.0e-3,
        ))
        ax.plot(
            endpoints[:, 0] * 1.0e3,
            endpoints[:, 2] * 1.0e3,
            **kwargs,
        )
        return centre_point

    slit = energy_filter.energy_slit
    slit_point = plane_segment(
        slit.distance_from_sector_exit_m * 1.0e3,
        25.0,
        color="#e91e63",
        lw=4 if slit.inserted else 1.5,
        ls="-" if slit.inserted else ":",
    )
    ax.annotate(
        (
            f"physical slit {'INSERTED' if slit.inserted else 'retracted'}\n"
            f"gap {slit.gap_m * 1e6:.3g} µm = "
            f"{slit.derived_width_ev:.3g} eV"
        ),
        (slit_point[0] * 1.0e3, slit_point[2] * 1.0e3),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=6.3,
        color="#ad1457",
    )
    camera_point = plane_segment(
        energy_filter.output_detector_d_mm,
        0.5 * energy_filter.output_detector_width_mm,
        color=(
            "#c62828"
            if energy_filter.output_detector_inserted
            else "#616161"
        ),
        lw=6 if energy_filter.output_detector_inserted else 1.5,
        ls="-" if energy_filter.output_detector_inserted else ":",
    )
    ax.annotate(
        "Camera "
        + (
            "INSERTED"
            if energy_filter.output_detector_inserted
            else "retracted"
        ),
        (camera_point[0] * 1.0e3, camera_point[2] * 1.0e3),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=6.3,
    )
    eels_point = plane_segment(
        (
            energy_filter.output_detector_d_mm
            + energy_filter.eels_plane_offset_mm
        ),
        30.0,
        color="#1565c0",
        lw=1.2,
        ls="--",
    )
    ax.annotate(
        "EELS spectrometer plane",
        (eels_point[0] * 1.0e3, eels_point[2] * 1.0e3),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=6.3,
        color="#1565c0",
    )

    if result is not None:
        for u, v, colour in zip(
            result.paths_u_mm,
            result.paths_v_mm,
            result.colours,
        ):
            ax.plot(u, v, color=colour, lw=0.55, alpha=0.68)
        ax.text(
            0.01,
            0.01,
            (
                f"TEM rays through entrance: "
                f"{result.entrance_ray_count}\n{result.status}"
            ),
            transform=ax.transAxes,
            fontsize=7,
            va="bottom",
        )
    ax.text(
        0.02,
        0.97,
        (
            f"sector B = {float(energy_filter.sector_field_t):.6g} T\n"
            f"R = {radius:.3g} mm, bend = {bend_deg:.3g}°"
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=6.5,
    )
    _draw_m12_glyph(ax, (0.72, 0.90), energy_filter.entrance_m12)
    _draw_m12_glyph(ax, (0.87, 0.90), energy_filter.exit_m12)
    ax.set_aspect("auto")
    ax.margins(0.08)


def draw_slit_plane_dispersion(ax, state, result):
    ax.clear()
    ax.set_title("Physical energy slit and slit-plane diagnostics", fontsize=9)
    ax.set_xlabel("dispersive transverse offset (µm)", fontsize=8)
    ax.set_ylabel("distance from slit plane (mm)", fontsize=8)
    ax.grid(alpha=0.18)
    ax.tick_params(labelsize=7)
    energy_filter = getattr(state, "energy_filter", None)
    if (
        energy_filter is None
        or not energy_filter.enabled
        or result is None
        or not result.paths_u_mm
    ):
        ax.text(
            0.5,
            0.5,
            "No energy-filter trajectories",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7,
        )
        return

    sector = sector_from_energy_filter(energy_filter)
    exit_point_mm = sector.exit_point_m * 1.0e3
    outgoing_s = sector.exit_tangent
    outgoing_x = sector.exit_frame.rotation_local_to_global[:, 0]
    window_mm = 18.0
    all_transverse = []
    for u, v in zip(result.paths_u_mm, result.paths_v_mm):
        points = np.column_stack((u, np.zeros(len(u)), v))
        delta = points - exit_point_mm
        along = delta @ outgoing_s
        transverse = delta @ outgoing_x * 1.0e3
        local = along - energy_filter.slit_d_mm
        keep = np.abs(local) <= window_mm
        if np.any(keep):
            ax.plot(
                transverse[keep],
                local[keep],
                lw=0.65,
                alpha=0.7,
            )
            all_transverse.extend(transverse[keep])

    slit = energy_filter.energy_slit
    lower = slit.lower_blade_edge_m * 1.0e6
    upper = slit.upper_blade_edge_m * 1.0e6
    extent = max(
        1.3 * max(abs(lower), abs(upper), 25.0),
        max((abs(value) for value in all_transverse), default=0.0) * 1.1,
    )
    if slit.inserted:
        ax.hlines(0.0, -extent, lower, color="#e91e63", lw=5)
        ax.hlines(0.0, upper, extent, color="#e91e63", lw=5)
    else:
        ax.axhline(0.0, color="#e91e63", lw=1.2, ls=":")
    ax.axvline(lower, color="#ad1457", lw=0.7, ls="--")
    ax.axvline(upper, color="#ad1457", lw=0.7, ls="--")
    metrics = (
        result.metrics
        if getattr(result, "metrics", None) is not None
        else getattr(energy_filter, "_last_slit_metrics", None)
    )
    validation = getattr(
        energy_filter,
        "_last_integrator_validation",
        None,
    )
    if validation is not None:
        ax.scatter(
            [validation.boris_dispersive_um],
            [0.0],
            s=28,
            marker="o",
            facecolors="none",
            edgecolors="#263238",
            linewidths=1.0,
            label="Boris reference",
            zorder=8,
        )
        ax.scatter(
            [validation.adaptive_dispersive_um],
            [0.0],
            s=32,
            marker="x",
            color="#2e7d32",
            linewidths=1.2,
            label="adaptive DOP853",
            zorder=9,
        )
        ax.legend(loc="upper left", fontsize=6, framealpha=0.8)
    text = (
        f"physical gap {slit.gap_m * 1e6:.4g} µm | "
        f"request {slit.requested_centre_loss_ev:.4g} ± "
        f"{0.5 * slit.requested_width_ev:.4g} eV"
    )
    if metrics is not None:
        text += "\n" + metrics.summary()
    if validation is not None:
        text += (
            "\nBoris − DOP853: "
            f"{validation.dispersive_error_nm:+.4g} nm, "
            f"{validation.direction_error_urad:.4g} µrad"
        )
    ax.text(
        0.98,
        0.04,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#880e4f",
    )
    ax.set_xlim(-extent, extent)
    ax.set_ylim(window_mm, -window_mm)


def draw_physical_energy_filter(ax, state, result):
    return draw_energy_filter(ax, state, result)
