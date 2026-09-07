"""Relativistic batch tracing from an electron source to its exit aperture."""

from __future__ import annotations

import math

import numpy as np

from temsim.component_keys import FEG_MONOCHROMATOR_SLIT
from temsim.optics.electron_gun.base import (
    GunEqualTimeHistory,
    GunExitBundle,
    GunPlaneArrival,
    GunTraceResult,
)
from temsim.physics.relativistic_lorentz import (
    RelativisticPhaseSpace,
    boris_step,
    momentum_from_kinetic_energy_ev,
    velocity_from_momentum_m_per_s,
)


def trace_feg_to_exit(gun, count=None) -> GunTraceResult:
    emitted = gun.emit(count)
    n = emitted.x_m.size
    direction = np.column_stack((
        emitted.tx_rad,
        emitted.ty_rad,
        np.ones(n, dtype=float),
    ))
    launch_energy = (
        float(gun.emitter.emission_energy_ev) + emitted.energy_offset_ev
    )
    if np.any(~np.isfinite(launch_energy)) or np.any(launch_energy <= 0.0):
        raise ValueError(
            f"{gun.display_name} emitter produced non-positive launch "
            "kinetic energy. Check its energy-spread parameters."
        )
    position = np.column_stack((
        emitted.x_m,
        emitted.y_m,
        np.zeros(n, dtype=float),
    ))
    momentum = momentum_from_kinetic_energy_ev(launch_energy, direction)
    phase = RelativisticPhaseSpace(position, momentum)
    alive = np.ones(n, dtype=bool)
    completed = np.zeros(n, dtype=bool)
    dpa_passed = np.zeros(n, dtype=bool)
    slit_passed = np.zeros(n, dtype=bool)
    slit_reached = np.zeros(n, dtype=bool)
    slit_x_m = np.full(n, np.nan, dtype=float)
    blocked_z = np.full(n, np.nan, dtype=float)
    blocked_key = [""] * n
    exit_position = np.full((n, 3), np.nan, dtype=float)
    exit_momentum = np.full((n, 3), np.nan, dtype=float)
    dpa_arrival_time = np.full(n, np.nan, dtype=float)
    dpa_arrival_x = np.full(n, np.nan, dtype=float)
    dpa_arrival_y = np.full(n, np.nan, dtype=float)
    slit_arrival_time = np.full(n, np.nan, dtype=float)
    slit_arrival_y = np.full(n, np.nan, dtype=float)
    exit_arrival_time = np.full(n, np.nan, dtype=float)
    exit_arrival_x = np.full(n, np.nan, dtype=float)
    exit_arrival_y = np.full(n, np.nan, dtype=float)

    history_position = [position.copy()]
    history_momentum = [momentum.copy()]
    history_time = [0.0]
    history_alive = [alive.copy()]
    history_completed = [completed.copy()]
    history_stride = max(
        1, int(round(float(gun.history_step_mm) / gun.trace_step_mm))
    )
    exit_z_m = gun.exit_plane_z_mm * 1e-3
    dpa_z_m = gun.dpa_aperture.z_mm * 1e-3
    monochromator_installed = bool(
        getattr(gun, "monochromator_installed", False)
    )
    slit = (
        gun.monochromator.slit
        if monochromator_installed
        else None
    )
    slit_plane = gun.c1_aperture if slit is not None else None
    slit_z_m = (
        slit_plane.z_mm * 1.0e-3
        if slit_plane is not None
        else None
    )
    maximum_steps = int(np.ceil(gun.exit_plane_z_mm / gun.trace_step_mm)) * 8

    for step_index in range(maximum_steps):
        active = alive & ~completed
        if not np.any(active):
            break
        velocity = velocity_from_momentum_m_per_s(
            phase.momentum_kg_m_per_s[active]
        )
        forward = velocity[:, 2] > 0.0
        if not np.all(forward):
            indices = np.flatnonzero(active)[~forward]
            alive[indices] = False
            for index in indices:
                blocked_z[index] = phase.position_m[index, 2] * 1000.0
                blocked_key[index] = (
                    "feg_backstream"
                    if gun.type_key == "cold_feg"
                    else f"{gun.type_key}_backstream"
                )
            active = alive & ~completed
            if not np.any(active):
                break
            velocity = velocity_from_momentum_m_per_s(
                phase.momentum_kg_m_per_s[active]
            )
        active_z_mm = float(np.max(phase.position_m[active, 2])) * 1000.0
        step_m = gun.integration_step_mm_at(active_z_mm) * 1e-3
        dt = step_m / max(float(np.max(velocity[:, 2])), 1.0)
        previous_position = phase.position_m.copy()
        previous_momentum = phase.momentum_kg_m_per_s.copy()
        previous_time_s = float(phase.time_s)
        advanced = boris_step(
            phase,
            dt,
            gun.magnetic_field,
            electric_field=gun.electric_field,
        )
        new_position = phase.position_m.copy()
        new_momentum = phase.momentum_kg_m_per_s.copy()
        new_position[active] = advanced.position_m[active]
        new_momentum[active] = _enforce_static_field_energy(
            gun,
            advanced.position_m[active],
            advanced.momentum_kg_m_per_s[active],
            launch_energy[active],
        )
        phase = RelativisticPhaseSpace(
            new_position, new_momentum, phase.time_s + dt
        )

        _clip_body_bores(
            gun, previous_position, phase.position_m,
            alive, completed, blocked_z, blocked_key,
        )
        _resolve_aperture_crossing(
            gun.dpa_aperture,
            dpa_z_m,
            previous_position,
            previous_momentum,
            phase.position_m,
            phase.momentum_kg_m_per_s,
            alive,
            completed,
            blocked_z,
            blocked_key,
            passed=dpa_passed,
            previous_time_s=previous_time_s,
            new_time_s=float(phase.time_s),
            arrival_time_s=dpa_arrival_time,
            arrival_x_m=dpa_arrival_x,
            arrival_y_m=dpa_arrival_y,
        )
        if slit is not None:
            _resolve_monochromator_slit_crossing(
                slit,
                slit_plane,
                slit_z_m,
                previous_position,
                phase.position_m,
                alive,
                completed,
                blocked_z,
                blocked_key,
                passed=slit_passed,
                reached=slit_reached,
                slit_x_m=slit_x_m,
                previous_time_s=previous_time_s,
                new_time_s=float(phase.time_s),
                arrival_time_s=slit_arrival_time,
                arrival_y_m=slit_arrival_y,
            )
        _resolve_exit_crossing(
            gun.c1_aperture,
            exit_z_m,
            previous_position,
            previous_momentum,
            phase.position_m,
            phase.momentum_kg_m_per_s,
            alive,
            completed,
            blocked_z,
            blocked_key,
            exit_position,
            exit_momentum,
            previous_time_s=previous_time_s,
            new_time_s=float(phase.time_s),
            arrival_time_s=exit_arrival_time,
            arrival_x_m=exit_arrival_x,
            arrival_y_m=exit_arrival_y,
        )
        if (
            step_index % history_stride == 0
            or not np.any(alive & ~completed)
        ):
            snapshot_position = phase.position_m.copy()
            snapshot_momentum = phase.momentum_kg_m_per_s.copy()
            if np.any(completed):
                snapshot_position[completed] = exit_position[completed]
                snapshot_momentum[completed] = exit_momentum[completed]
            history_position.append(snapshot_position)
            history_momentum.append(snapshot_momentum)
            history_time.append(float(phase.time_s))
            history_alive.append(alive.copy())
            history_completed.append(completed.copy())
    else:
        raise RuntimeError(
            f"{gun.display_name} trace did not reach its exit aperture "
            "within the step limit."
        )

    passed_c1 = alive & completed
    if np.any(passed_c1):
        exit_momentum[passed_c1] = _enforce_static_field_energy(
            gun,
            exit_position[passed_c1],
            exit_momentum[passed_c1],
            launch_energy[passed_c1],
        )
    history_position_array = np.asarray(history_position)
    history_momentum_array = np.asarray(history_momentum)
    pz = exit_momentum[:, 2]
    tx = np.divide(
        exit_momentum[:, 0], pz,
        out=np.zeros(n, dtype=float), where=passed_c1,
    )
    ty = np.divide(
        exit_momentum[:, 1], pz,
        out=np.zeros(n, dtype=float), where=passed_c1,
    )
    history_pz = history_momentum_array[..., 2]
    history_tx = np.divide(
        history_momentum_array[..., 0],
        history_pz,
        out=np.zeros_like(history_pz),
        where=np.abs(history_pz) > 0.0,
    )
    history_ty = np.divide(
        history_momentum_array[..., 1],
        history_pz,
        out=np.zeros_like(history_pz),
        where=np.abs(history_pz) > 0.0,
    )
    common_z, path_x, path_y, path_tx, path_ty = _resample_gun_paths(
        gun,
        history_position_array,
        history_tx,
        history_ty,
        exit_position,
        tx,
        ty,
        passed_c1,
        blocked_z,
        slit_plane,
    )
    equal_time_history = GunEqualTimeHistory(
        time_s=np.asarray(history_time, dtype=float),
        z_mm=history_position_array[..., 2] * 1000.0,
        x_m=history_position_array[..., 0],
        y_m=history_position_array[..., 1],
        tx_rad=history_tx,
        ty_rad=history_ty,
        alive=np.asarray(history_alive, dtype=bool),
        completed=np.asarray(history_completed, dtype=bool),
    )
    plane_arrivals = [
        GunPlaneArrival(
            key=str(gun.dpa_aperture.key),
            name=str(gun.dpa_aperture.name),
            z_mm=float(gun.dpa_aperture.z_mm),
            time_s=dpa_arrival_time,
            x_m=dpa_arrival_x,
            y_m=dpa_arrival_y,
            reached=np.isfinite(dpa_arrival_time),
            transmitted=dpa_passed.copy(),
        )
    ]
    if slit is not None:
        plane_arrivals.append(GunPlaneArrival(
            key=FEG_MONOCHROMATOR_SLIT,
            name=str(slit.name),
            z_mm=float(slit_plane.z_mm),
            time_s=slit_arrival_time,
            x_m=slit_x_m.copy(),
            y_m=slit_arrival_y,
            reached=slit_reached.copy(),
            transmitted=slit_passed.copy(),
        ))
    exit_is_c1_plane = math.isclose(
        float(gun.c1_aperture.z_mm),
        float(gun.exit_plane_z_mm),
        abs_tol=1.0e-9,
    )
    plane_arrivals.append(GunPlaneArrival(
        key=(
            str(gun.c1_aperture.key)
            if exit_is_c1_plane
            else f"{gun.c1_aperture.key}:exit"
        ),
        name=(
            str(gun.c1_aperture.name)
            if exit_is_c1_plane
            else f"{gun.display_name} Exit"
        ),
        z_mm=float(gun.exit_plane_z_mm),
        time_s=exit_arrival_time,
        x_m=exit_arrival_x,
        y_m=exit_arrival_y,
        reached=np.isfinite(exit_arrival_time),
        transmitted=passed_c1.copy(),
    ))
    # Static fields preserve each emitted energy offset exactly.  Keep that
    # invariant directly instead of subtracting two ~300 keV float values.
    energy_offset = emitted.energy_offset_ev.copy()
    current = gun.emitted_current_a
    monochromator_current = None
    slit_dispersion = None
    if slit is not None:
        monochromator_current = (
            current * float(np.sum(emitted.weight[slit_passed]))
        )
        indices = np.flatnonzero(slit_reached)
        if (
            indices.size >= 2
            and np.ptp(energy_offset[indices]) > 0.0
        ):
            slit_dispersion = float(
                np.polyfit(
                    energy_offset[indices],
                    slit_x_m[indices] * 1.0e6,
                    1,
                )[0]
            )
    output_fwhm = _weighted_fwhm_from_standard_deviation(
        energy_offset[passed_c1],
        emitted.weight[passed_c1],
    )
    return GunTraceResult(
        z_mm=common_z,
        x_m=path_x,
        y_m=path_y,
        tx_rad=path_tx,
        ty_rad=path_ty,
        exit_bundle=GunExitBundle(
            x_m=np.nan_to_num(exit_position[:, 0]),
            y_m=np.nan_to_num(exit_position[:, 1]),
            tx_rad=tx,
            ty_rad=ty,
            energy_offset_ev=energy_offset,
            weight=emitted.weight,
            ray_id=emitted.ray_id,
            alive=passed_c1,
        ),
        blocked_z_mm=blocked_z,
        blocked_key=tuple(blocked_key),
        emitted_current_a=current,
        dpa_transmitted_current_a=(
            current * float(np.sum(emitted.weight[dpa_passed]))
        ),
        c1_transmitted_current_a=(
            current * float(np.sum(emitted.weight[passed_c1]))
        ),
        monochromator_transmitted_current_a=monochromator_current,
        output_energy_fwhm_ev=output_fwhm,
        slit_dispersion_um_per_ev=slit_dispersion,
        slit_x_m=(slit_x_m if slit is not None else None),
        slit_reached=(slit_reached if slit is not None else None),
        equal_time_history=equal_time_history,
        plane_arrivals=tuple(plane_arrivals),
    )


def _resample_gun_paths(
    gun,
    equal_time_position,
    equal_time_tx,
    equal_time_ty,
    exit_position,
    exit_tx,
    exit_ty,
    passed_exit,
    blocked_z_mm,
    slit_plane,
):
    """Return ray paths on one strict, shared axial grid.

    The Boris solver advances all electrons at common laboratory times, so
    their instantaneous Z coordinates differ. Those equal-time snapshots must
    not be plotted against one scalar Z value. Each ray is instead interpolated
    independently onto this shared plane grid; the original equal-time data is
    retained separately on ``GunTraceResult``.
    """

    exit_z = float(gun.exit_plane_z_mm)
    history_step = float(gun.history_step_mm)
    axis_values = list(np.arange(0.0, exit_z, history_step, dtype=float))
    axis_values.extend((0.0, exit_z, float(gun.dpa_aperture.z_mm)))
    if slit_plane is not None:
        axis_values.append(float(slit_plane.z_mm))
    for component in getattr(gun, "components", ()):
        for attribute in ("z_mm", "upper_z_mm", "lower_z_mm"):
            value = getattr(component, attribute, None)
            if value is not None and 0.0 <= float(value) <= exit_z:
                axis_values.append(float(value))
    common_z = np.unique(np.asarray(axis_values, dtype=float))

    raw_z = np.asarray(equal_time_position[..., 2], dtype=float) * 1000.0
    raw_values = (
        np.asarray(equal_time_position[..., 0], dtype=float),
        np.asarray(equal_time_position[..., 1], dtype=float),
        np.asarray(equal_time_tx, dtype=float),
        np.asarray(equal_time_ty, dtype=float),
    )
    ray_count = raw_z.shape[1]
    outputs = [
        np.empty((common_z.size, ray_count), dtype=float)
        for _ in raw_values
    ]

    for ray in range(ray_count):
        if bool(passed_exit[ray]):
            limit = exit_z
            endpoint = (
                float(exit_position[ray, 0]),
                float(exit_position[ray, 1]),
                float(exit_tx[ray]),
                float(exit_ty[ray]),
            )
        else:
            blocked = float(blocked_z_mm[ray])
            limit = blocked if math.isfinite(blocked) else float(np.nanmax(raw_z[:, ray]))
            limit = min(max(limit, 0.0), exit_z)
            endpoint = None

        finite = np.isfinite(raw_z[:, ray])
        indices = np.flatnonzero(finite)
        keep = []
        previous_z = -math.inf
        for index in indices:
            value = float(raw_z[index, ray])
            if value > previous_z + 1.0e-12:
                keep.append(int(index))
                previous_z = value
        if not keep:
            raise RuntimeError("Electron-gun trace contains no finite path samples")

        monotonic_z = raw_z[keep, ray]
        if endpoint is None:
            endpoint = tuple(
                float(np.interp(limit, monotonic_z, values[keep, ray]))
                for values in raw_values
            )
        before = monotonic_z < limit - 1.0e-12
        path_z = monotonic_z[before]
        if path_z.size == 0 or path_z[0] > 1.0e-12:
            path_z = np.r_[0.0, path_z]
            start_values = tuple(float(values[keep[0], ray]) for values in raw_values)
        else:
            start_values = None
        path_z = np.r_[path_z, limit]

        for output, values, end_value, start_value in zip(
            outputs,
            raw_values,
            endpoint,
            start_values or (None,) * len(raw_values),
        ):
            path_values = values[keep, ray][before]
            if start_value is not None:
                path_values = np.r_[start_value, path_values]
            path_values = np.r_[path_values, end_value]
            output[:, ray] = np.interp(
                common_z, path_z, path_values,
                left=path_values[0], right=end_value,
            )
    return common_z, *outputs


def _weighted_fwhm_from_standard_deviation(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    if values.size < 2 or total <= 0.0:
        return 0.0
    mean = float(np.sum(values * weights) / total)
    variance = float(
        np.sum(weights * (values - mean) ** 2) / total
    )
    return 2.354820045 * math.sqrt(max(0.0, variance))


def _resolve_monochromator_slit_crossing(
    slit,
    slit_plane,
    plane_z_m,
    previous_position,
    new_position,
    alive,
    completed,
    blocked_z,
    blocked_key,
    *,
    passed,
    reached,
    slit_x_m,
    previous_time_s,
    new_time_s,
    arrival_time_s,
    arrival_y_m,
):
    candidates = (
        alive
        & ~completed
        & ~reached
        & (previous_position[:, 2] <= plane_z_m)
        & (new_position[:, 2] >= plane_z_m)
    )
    indices = np.flatnonzero(candidates)
    if not indices.size:
        return
    fraction = _crossing_fraction(
        previous_position[indices, 2],
        new_position[indices, 2],
        plane_z_m,
    )
    positions = previous_position[indices] + fraction[:, None] * (
        new_position[indices] - previous_position[indices]
    )
    reached[indices] = True
    slit_x_m[indices] = positions[:, 0]
    arrival_y_m[indices] = positions[:, 1]
    arrival_time_s[indices] = previous_time_s + fraction * (
        new_time_s - previous_time_s
    )
    transmitted = slit.transmission_mask(
        positions[:, 0], positions[:, 1]
    )
    accepted = indices[transmitted]
    passed[accepted] = True
    rejected = indices[~transmitted]
    alive[rejected] = False
    for index in rejected:
        blocked_z[index] = slit_plane.z_mm
        blocked_key[index] = slit_plane.key


def _enforce_static_field_energy(gun, position, momentum, launch_energy_ev):
    """Preserve K - e*phi for an electron in the static gun fields."""

    potential_v = gun.electric_field.potential_v_at_global_positions(position)
    target_energy_ev = np.maximum(
        np.asarray(launch_energy_ev, dtype=float) + potential_v,
        1.0e-6,
    )
    direction = np.asarray(momentum, dtype=float)
    return momentum_from_kinetic_energy_ev(target_energy_ev, direction)


def _crossing_fraction(old_z, new_z, plane_z):
    denominator = new_z - old_z
    return np.divide(
        plane_z - old_z,
        denominator,
        out=np.zeros_like(old_z),
        where=np.abs(denominator) > 0.0,
    )


def _resolve_aperture_crossing(
    aperture,
    plane_z_m,
    previous_position,
    previous_momentum,
    new_position,
    new_momentum,
    alive,
    completed,
    blocked_z,
    blocked_key,
    *,
    passed,
    previous_time_s,
    new_time_s,
    arrival_time_s,
    arrival_x_m,
    arrival_y_m,
):
    candidates = (
        alive
        & ~completed
        & ~passed
        & (previous_position[:, 2] <= plane_z_m)
        & (new_position[:, 2] >= plane_z_m)
    )
    indices = np.flatnonzero(candidates)
    if not indices.size:
        return
    fraction = _crossing_fraction(
        previous_position[indices, 2],
        new_position[indices, 2],
        plane_z_m,
    )
    xy = previous_position[indices, :2] + fraction[:, None] * (
        new_position[indices, :2] - previous_position[indices, :2]
    )
    arrival_x_m[indices] = xy[:, 0]
    arrival_y_m[indices] = xy[:, 1]
    arrival_time_s[indices] = previous_time_s + fraction * (
        new_time_s - previous_time_s
    )
    transmitted = aperture.transmission_mask(
        xy[:, 0] * 1000.0, xy[:, 1] * 1000.0
    )
    passed[indices[transmitted]] = True
    rejected = indices[~transmitted]
    alive[rejected] = False
    for index in rejected:
        blocked_z[index] = aperture.z_mm
        blocked_key[index] = aperture.key


def _resolve_exit_crossing(
    aperture,
    plane_z_m,
    previous_position,
    previous_momentum,
    new_position,
    new_momentum,
    alive,
    completed,
    blocked_z,
    blocked_key,
    exit_position,
    exit_momentum,
    *,
    previous_time_s,
    new_time_s,
    arrival_time_s,
    arrival_x_m,
    arrival_y_m,
):
    candidates = (
        alive
        & ~completed
        & (previous_position[:, 2] <= plane_z_m)
        & (new_position[:, 2] >= plane_z_m)
    )
    indices = np.flatnonzero(candidates)
    if not indices.size:
        return
    fraction = _crossing_fraction(
        previous_position[indices, 2],
        new_position[indices, 2],
        plane_z_m,
    )
    positions = previous_position[indices] + fraction[:, None] * (
        new_position[indices] - previous_position[indices]
    )
    momenta = previous_momentum[indices] + fraction[:, None] * (
        new_momentum[indices] - previous_momentum[indices]
    )
    transmitted = aperture.transmission_mask(
        positions[:, 0] * 1000.0, positions[:, 1] * 1000.0
    )
    arrival_time_s[indices] = previous_time_s + fraction * (
        new_time_s - previous_time_s
    )
    arrival_x_m[indices] = positions[:, 0]
    arrival_y_m[indices] = positions[:, 1]
    accepted = indices[transmitted]
    exit_position[accepted] = positions[transmitted]
    exit_momentum[accepted] = momenta[transmitted]
    completed[accepted] = True
    rejected = indices[~transmitted]
    alive[rejected] = False
    for index in rejected:
        blocked_z[index] = aperture.z_mm
        blocked_key[index] = aperture.key


def _clip_body_bores(
    gun,
    previous_position,
    new_position,
    alive,
    completed,
    blocked_z,
    blocked_key,
):
    active = np.flatnonzero(alive & ~completed)
    if not active.size:
        return
    z_mm = new_position[active, 2] * 1000.0
    radius_mm = np.hypot(
        new_position[active, 0], new_position[active, 1]
    ) * 1000.0
    for component in gun.bore_components:
        half_length = 0.5 * component.mechanical_length_mm
        inside = np.abs(
            z_mm - component.mechanical_center_from_tip_mm
        ) <= half_length
        outside = radius_mm > (
            0.5 * component.mechanical_clear_bore_diameter_mm
        )
        rejected_local = np.flatnonzero(inside & outside)
        for local_index in rejected_local:
            index = active[local_index]
            if alive[index]:
                alive[index] = False
                blocked_z[index] = z_mm[local_index]
                blocked_key[index] = component.key
