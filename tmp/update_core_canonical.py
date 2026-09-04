from pathlib import Path
p=Path("src/temsim/physics/core.py")
s=p.read_text()
a=s.index("try:\n    from numba import cuda, njit, prange")
b=s.index("\ndef fields(",a)
s=s[:a]+"from temsim.physics.ray_integrator import (\n    NUMBA_AVAILABLE,\n    parallel_rk4 as _parallel_rk4,\n    vectorised_rk4 as _vectorised_rk4,\n    cuda_rk4 as _cuda_rk4,\n)\n\n"+s[b:]
a=s.index("@njit(cache=True,parallel=True,fastmath=True)")
b=s.index("def _record_active_backend(",a)
s=s[:a]+s[b:]
s=s.replace("    hex_skew_m3: np.ndarray\n", "    hex_skew_m3: np.ndarray\n    midpoint_magnetic_t: np.ndarray\n    midpoint_sx_m2: np.ndarray\n    midpoint_sy_m2: np.ndarray\n    midpoint_hex_normal_m3: np.ndarray\n    midpoint_hex_skew_m3: np.ndarray\n",1)
s=s.replace("    exact_z_mm.extend(float(value) for value in save_z_mm)\n", "    exact_z_mm.extend(float(value) for value in save_z_mm)\n    # Impulsive actions must occur at their physical planes, independent of\n    # the requested integration step or an unrelated observation plane.\n    exact_z_mm.extend(float(event[0]) for event in events)\n    if include_spherical_aberration:\n        exact_z_mm.extend(\n            float(lens.z_mm) for lens in state.lenses\n            if bool(getattr(lens, 'enabled', True))\n            and spherical_aberration_mm(lens, state.beam_voltage_kv)\n        )\n",1)
s=s.replace("    step_m=np.ascontiguousarray(step_mm*1e-3,np.float64)\n", "    step_m=np.ascontiguousarray(step_mm*1e-3,np.float64)\n    midpoint_z_mm = 0.5 * (zfull[:-1] + zfull[1:])\n",1)
s=s.replace("        magnetic,sx,sy=fields(zfull,state)\n", "        magnetic,sx,sy=fields(zfull,state)\n        midpoint_magnetic, midpoint_sx, midpoint_sy = fields(\n            midpoint_z_mm, state\n        )\n",1)
s=s.replace("        hex_normal, hex_skew = hexapole_field_components(zfull, state)\n", "        hex_normal, hex_skew = hexapole_field_components(zfull, state)\n        midpoint_hex_normal, midpoint_hex_skew = hexapole_field_components(\n            midpoint_z_mm, state\n        )\n",1)
s=s.replace("        hex_skew = np.zeros(len(zfull), np.float64)\n", "        hex_skew = np.zeros(len(zfull), np.float64)\n        midpoint_hex_normal = np.zeros(len(midpoint_z_mm), np.float64)\n        midpoint_hex_skew = np.zeros(len(midpoint_z_mm), np.float64)\n",1)
s=s.replace("        FIELD_SIGMA_CUTOFF,\n    ))", "        FIELD_SIGMA_CUTOFF,\n        'canonical-rk4-exact-midpoints-v1',\n    ))",1)
s=s.replace("        zfull, step_m, magnetic, sx, sy, hex_normal, hex_skew,\n", "        zfull, step_m, magnetic, sx, sy, hex_normal, hex_skew,\n        midpoint_magnetic, midpoint_sx, midpoint_sy,\n        midpoint_hex_normal, midpoint_hex_skew,\n",1)
s=s.replace("        hex_skew_m3=_frozen_array(hex_skew),\n", "        hex_skew_m3=_frozen_array(hex_skew),\n        midpoint_magnetic_t=_frozen_array(midpoint_magnetic),\n        midpoint_sx_m2=_frozen_array(midpoint_sx),\n        midpoint_sy_m2=_frozen_array(midpoint_sy),\n        midpoint_hex_normal_m3=_frozen_array(midpoint_hex_normal),\n        midpoint_hex_skew_m3=_frozen_array(midpoint_hex_skew),\n",1)
s=s.replace("    changed = np.flatnonzero(~equal)\n", "    # Midpoint i belongs to the interval leaving node i.  A changed\n    # midpoint invalidates that interval even when its endpoint fields match.\n    for name in (\n        'midpoint_magnetic_t', 'midpoint_sx_m2', 'midpoint_sy_m2',\n        'midpoint_hex_normal_m3', 'midpoint_hex_skew_m3',\n    ):\n        old, new = getattr(previous, name), getattr(current, name)\n        interval_count = min(count, old.size, new.size)\n        equal[:interval_count] &= old[:interval_count] == new[:interval_count]\n    changed = np.flatnonzero(~equal)\n",1)
s=s.replace("        # Bz at node i affects the numerical derivative and the RK4 interval\n        # ending at i.  Keep a two-node safety halo before the first change.\n", "        # Preserve the preceding interval when an endpoint or its midpoint\n        # changes.  The extra node keeps existing checkpoint selection safe.\n",1)
a=s.index("    magnetic = np.asarray(plan.magnetic_t[start_index:]")
b=s.index("    cs_kick = np.array(",a)
s=s[:a]+'''    def stages(node_name, midpoint_name):
        nodes = np.asarray(getattr(plan, node_name)[start_index:])
        midpoint = np.asarray(getattr(plan, midpoint_name)[start_index:])
        return interleaved_rk4_values(nodes, midpoint)

    magnetic = stages("magnetic_t", "midpoint_magnetic_t")
    sx = stages("sx_m2", "midpoint_sx_m2")
    sy = stages("sy_m2", "midpoint_sy_m2")
    hex_normal = stages("hex_normal_m3", "midpoint_hex_normal_m3")
    hex_skew = stages("hex_skew_m3", "midpoint_hex_skew_m3")
'''+s[b:]
a=s.index("    gun_index=np.int64(-1)")
b=s.index("    X,TX,Y,TY,CX,CTX,CY,CTY=outputs",a)
s=s[:a]+'''    backend, fallback_reason = choose_ray_backend(
        getattr(state, "acceleration_backend", "Auto"),
        acceleration_enabled=bool(getattr(state, "acceleration_enabled", False)),
        ray_count=arrays[0].size,
    )
    # Post-gun momentum is constant along Z, including with an energy spread.
    # Keep this factor separate on every backend; no (Z, ray) coefficient
    # matrices or finite-difference magnetic derivatives are necessary.
    momentum_at_start = momentum_profile(state, zfull[:1], energy_offset_ev)
    if momentum_at_start.ndim == 1:
        inverse_momentum = np.full(
            arrays[0].size, 1.0 / float(momentum_at_start[0]), dtype=np.float64
        )
    else:
        inverse_momentum = np.ascontiguousarray(
            1.0 / momentum_at_start[0], dtype=np.float64
        )
    larmor_axis = np.ascontiguousarray((-E) * magnetic / 2.0)
    inputs = (
        sx, sy, hex_normal, hex_skew, larmor_axis, inverse_momentum,
        cs_kick, thin_power, thin_rotation, step_m, *arrays, kickx, kicky,
        save, checkpoint_index,
    )
    if backend == BACKEND_CUDA:
        try:
            outputs = _cuda_rk4(*inputs)
        except Exception as exc:
            backend = BACKEND_NUMBA if NUMBA_AVAILABLE else BACKEND_CPU
            fallback_reason = f"CUDA error: {exc}"
            outputs = (
                _parallel_rk4(*inputs) if backend == BACKEND_NUMBA
                else _vectorised_rk4(*inputs)
            )
    elif backend == BACKEND_NUMBA:
        outputs = _parallel_rk4(*inputs)
    else:
        outputs = _vectorised_rk4(*inputs)
    _record_active_backend(state, backend, fallback_reason)
'''+s[b:]
marker="def build_propagation_plan("
a=s.index(marker)
s=s[:a]+'''def interleaved_rk4_values(nodes, midpoints):
    """Pack true endpoint/midpoint samples without interpolating fields."""
    node_values = np.asarray(nodes, dtype=np.float64)
    midpoint_values = np.asarray(midpoints, dtype=np.float64)
    if midpoint_values.shape != (max(node_values.size - 1, 0),):
        raise ValueError("RK4 stages require one midpoint per node interval")
    result = np.empty(max(2 * node_values.size - 1, 0), dtype=np.float64)
    result[::2] = node_values
    result[1::2] = midpoint_values
    return result


'''+s[a:]
p.write_text(s,encoding="utf-8")
