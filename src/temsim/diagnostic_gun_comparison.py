"""Measurements for a bounded gun-domain comparison, never a transport model.

All coordinates are SI XYZ with +Z downstream. Plane values are chronological
first-crossing interpolation of accepted states, not new integrated events.
Neither a completed comparison nor fixture tolerances establish equivalence
of the two field solutions or qualification of the microscope chain.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


REFERENCE_CHECKS = {
    "energy_invariant_absolute_ev": 1e-6,
    "curved_launch_potential_absolute_v": 1e-3,
    "field_linear_residual": 1e-9,
    "basis": "tests/test_magnetic_test_particle.py and tests/test_closed_gun_field.py; conservative reference-fixture checks, not a full-gun qualification",
    "domain_equivalence_tolerance": None,
    "domain_equivalence_status": "NOT_ESTABLISHED; differences require review against independent step/grid refinements",
}


def emission_samples(gun, count=193, sample_count=3):
    """Select deterministic members of the actual original emission population.

    The closest-to-axis, largest-angle and largest-position members exercise
    distinct source inputs without moving a particle to a downstream source.
    IDs and original weights are retained; the selected subset is not
    renormalised or represented as the complete emission population.
    """
    emitted = gun.emit(count)
    size = len(emitted.ray_id)
    if not 1 <= sample_count <= size:
        raise ValueError("Sample count must fit the emitted population")
    if hasattr(emitted, "surface_normal"):
        position = np.array(emitted.surface_position_m, copy=True)
        direction = np.array(emitted.surface_direction, copy=True)
    else:
        position = np.column_stack((emitted.x_m, emitted.y_m, np.zeros(size)))
        direction = np.column_stack((emitted.tx_rad, emitted.ty_rad, np.ones(size)))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    energy = float(gun.emitter.emission_energy_ev) + emitted.energy_offset_ev
    if gun.emitter.surface_model is not None:
        raise ValueError("This bounded comparison covers existing flat/continuous-curvature particles only")
    radius = np.linalg.norm(position[:, :2], axis=1)
    angle = np.arctan2(np.linalg.norm(direction[:, :2], axis=1), direction[:, 2])
    candidates = [int(np.argmin(radius)), int(np.argmax(angle)), int(np.argmax(radius))]
    candidates += list(np.argsort(-angle, kind="stable"))
    selected = list(dict.fromkeys(candidates))[:sample_count]
    rows = []
    for index in selected:
        rows.append({"ray_id": int(emitted.ray_id[index]), "original_index": int(index),
                     "original_weight": float(emitted.weight[index]),
                     "position_m": position[index].tolist(), "direction": direction[index].tolist(),
                     "kinetic_energy_ev": float(energy[index]),
                     "polar_angle_deg": float(np.degrees(angle[index])),
                     "azimuth_angle_deg": float(np.degrees(np.arctan2(direction[index, 1], direction[index, 0])))})
    digest = hashlib.sha256()
    for value in (emitted.ray_id, emitted.weight, position, direction, energy):
        array = np.ascontiguousarray(value)
        digest.update(str((array.shape, array.dtype.str)).encode("ascii"))
        digest.update(array.tobytes())
    return {"emitted_count": size, "selected_count": len(rows), "source_sha256": digest.hexdigest(),
            "emitted_weight": float(np.sum(emitted.weight)),
            "selected_weight": sum(row["original_weight"] for row in rows), "samples": rows}


def field_difference(first, second, points):
    """Report local differences without hiding near-wall errors in a peak norm."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Field comparison requires nonempty finite XYZ points")
    for provider in (first, second):
        if np.any(points[:, 2] < provider.z[0]) or np.any(points[:, 2] > provider.z[-1]):
            raise ValueError("Field comparison point lies outside the common axial domain")
        if np.any(np.hypot(points[:, 0], points[:, 1]) > provider.r[-1]):
            raise ValueError("Field comparison point lies outside the common radial domain")
    p0, e0 = (np.asarray(a, dtype=float) for a in first.interpolate(points))
    p1, e1 = (np.asarray(a, dtype=float) for a in second.interpolate(points))
    if p0.shape != (len(points),) or p1.shape != p0.shape or e0.shape != points.shape or e1.shape != points.shape:
        raise ValueError("Field comparison requires one potential and XYZ electric vector per point")
    if any(not np.isfinite(a).all() for a in (p0, e0, p1, e1)):
        raise ValueError("Field comparison cannot accept nonfinite values")
    difference = np.linalg.norm(e1-e0, axis=1)
    reference = np.linalg.norm(e0, axis=1)
    rows = []
    for index, point in enumerate(points):
        rows.append({"position_m": point.tolist(), "potential_difference_v": float(p1[index]-p0[index]),
                     "electric_difference_v_per_m": (e1[index]-e0[index]).tolist(),
                     "electric_difference_norm_v_per_m": float(difference[index]),
                     "reference_electric_norm_v_per_m": float(reference[index]),
                     "local_relative_electric_difference": float(difference[index]/reference[index]) if reference[index] > 0 else None,
                     "reference_potential_v": float(p0[index]),
                     "local_relative_potential_difference": float(abs(p1[index]-p0[index])/abs(p0[index])) if p0[index] != 0 else None})
    return {"maximum_absolute_potential_difference_v": float(np.max(abs(p1-p0))),
            "maximum_electric_difference_v_per_m": float(np.max(difference)),
            "points": rows}


def first_plane_crossing(trajectory, z_m):
    """The first forward crossing, preserving chronology on reflected paths."""
    position = np.asarray(trajectory.positions_m)
    if not len(position) or not np.isfinite(z_m):
        raise ValueError("A finite plane and nonempty executed trajectory are required")
    hit = np.flatnonzero((position[:-1, 2] <= z_m) & (position[1:, 2] >= z_m)
                         & (position[1:, 2] > position[:-1, 2]))
    terminal_contact = None
    if position[0, 2] == z_m:
        index, fraction = 0, 0.
    elif not len(hit):
        # A boundary event can end a few representable floats inside its
        # plane. Do not mistake that roundoff for lost population. This is
        # restricted to a completed forward terminal contact, not a generic
        # tolerance for interpolating/extrapolating unvisited planes.
        reason = str(getattr(trajectory, "reason", ""))
        roundoff = 64.*np.spacing(max(1., abs(float(z_m)), abs(float(position[-1, 2]))))
        delta_z = float(position[-1, 2])-float(z_m)
        contact = (reason == "domain_exit" or reason.startswith(("hardware:", "aperture:")))
        if (not bool(getattr(trajectory, "completed", False)) or not contact or len(position) < 2
                or position[-1, 2] <= position[-2, 2] or not -roundoff <= delta_z <= 0.):
            return None
        index, fraction = len(position)-1, 0.
        terminal_contact = {"terminal_delta_z_m": delta_z, "terminal_roundoff_window_m": float(roundoff)}
    else:
        index = int(hit[0])
        fraction = (z_m-position[index, 2])/(position[index+1, 2]-position[index, 2])

    def interpolate(values):
        if fraction == 0:
            return np.asarray(values[index])
        return (1-fraction)*values[index] + fraction*values[index+1]

    direction = interpolate(trajectory.directions)
    norm = np.linalg.norm(direction)
    direction = (direction/norm).tolist() if norm > 0 else None
    return {"position_m": interpolate(position).tolist(), "direction": direction,
            "kinetic_energy_ev": float(interpolate(trajectory.kinetic_energy_ev)),
            "flight_time_s": float(interpolate(trajectory.time_s)),
            "sample_basis": ("completed forward terminal contact within 64 float64 spacings; actual endpoint retained"
                             if terminal_contact else "first forward crossing interpolated between accepted float64 states"),
            **(terminal_contact or {})}


def trajectory_measurements(trajectory, planes, apertures=()):
    invariant = np.asarray(trajectory.kinetic_energy_ev)-np.asarray(trajectory.electrostatic_potential_v)
    drift = float(np.max(abs(invariant-invariant[0])))
    clearances = []
    for aperture in apertures:
        crossing = first_plane_crossing(trajectory, float(aperture.z_mm)*1e-3)
        if crossing is None:
            clearances.append({"key": aperture.key, "reached": False, "signed_clearance_m": None})
            continue
        xy = np.asarray(crossing["position_m"][:2])
        offset = np.array((aperture.offset_x_mm, aperture.offset_y_mm))*1e-3
        clearance = float(aperture.radius_mm)*1e-3-float(np.linalg.norm(xy-offset))
        clearances.append({"key": aperture.key, "reached": True, "signed_clearance_m": clearance})
    return {"reason": trajectory.reason, "completed": bool(trajectory.completed), "steps": int(trajectory.steps),
            "terminal_position_m": trajectory.positions_m[-1].tolist(),
            "initial_invariant_ev": float(invariant[0]), "maximum_invariant_drift_ev": drift,
            "reference_fixture_invariant_check": drift <= REFERENCE_CHECKS["energy_invariant_absolute_ev"],
            "planes": {str(name): first_plane_crossing(trajectory, float(z)) for name, z in planes},
            "aperture_clearances": clearances}


def compare_populations(first, second):
    """Match original IDs; loss of an entire population is not zero difference."""
    if not first or not second:
        raise ValueError("Population comparison requires nonempty original populations")
    left = {row["ray_id"]: row for row in first}
    right = {row["ray_id"]: row for row in second}
    if len(left) != len(first) or len(right) != len(second) or left.keys() != right.keys():
        raise ValueError("Population comparison requires unique identical original ray IDs")
    rows = []
    for ray_id in left:
        a, b = left[ray_id], right[ray_id]
        if a["original_weight"] != b["original_weight"]:
            raise ValueError("Population comparison must preserve original weights")
        if a["planes"].keys() != b["planes"].keys():
            raise ValueError("Population comparison requires identical observation planes")
        for name, first_plane in a["planes"].items():
            second_plane = b["planes"][name]
            row = {"ray_id": ray_id, "plane": name, "first_reached": first_plane is not None,
                   "second_reached": second_plane is not None}
            if first_plane is not None and second_plane is not None:
                angle = None
                if first_plane["direction"] is not None and second_plane["direction"] is not None:
                    d0, d1 = np.array(first_plane["direction"]), np.array(second_plane["direction"])
                    angle = float(np.arctan2(np.linalg.norm(np.cross(d0, d1)), np.dot(d0, d1)))
                row.update(xy_difference_m=float(np.linalg.norm(np.array(first_plane["position_m"][:2])-second_plane["position_m"][:2])),
                           direction_difference_rad=angle,
                           kinetic_energy_difference_ev=abs(first_plane["kinetic_energy_ev"]-second_plane["kinetic_energy_ev"]),
                           flight_time_difference_s=abs(first_plane["flight_time_s"]-second_plane["flight_time_s"]))
            rows.append(row)
    return {"identical_ids_and_weights": True,
            "same_terminal_reasons": all(left[k]["reason"] == right[k]["reason"] for k in left),
            "first_stop_counts": dict(Counter(row["reason"] for row in first)),
            "second_stop_counts": dict(Counter(row["reason"] for row in second)),
            "plane_populations": {name: {label: {"count": sum(row["planes"][name] is not None for row in data),
                "original_weight": sum(row["original_weight"] for row in data if row["planes"][name] is not None)}
                for label, data in (("first", first), ("second", second))} for name in next(iter(left.values()))["planes"]},
            "per_particle_plane_differences": rows}
