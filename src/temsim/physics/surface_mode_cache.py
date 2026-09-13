"""Internal completed-energy cache; a single energy is never a full source.

Only the joint solver supplies the executed result and dependency identity.
Near-tip complex fields, every boundary trace and phase coordinates remain
with the gun-exit mode. No read-only development export is imported here.
"""
from temsim.immutable_json import json_digest, freeze_json
from temsim.physics.surface_wave import SurfaceWaveMode, _immutable


HISTORY_FIELDS = frozenset(("z_nm", "width_curvature", "quartic_phase_per_nm4", "coefficients",
    "covariant_derivatives", "chart_derivatives", "near_radius_nm", "near_z_nm", "near_potential_rise_v"))
SCHEMA = "completed-joint-surface-energy-v2"


def mode_key(store, identity):
    return store.key(SCHEMA, identity)


def restore_mode(store, identity):
    key = mode_key(store, identity)
    cached = store.get(key)
    if cached is None:
        return None
    record = cached.record
    if (record.get("schema") != SCHEMA
            or record.get("executed_identity_digest") != json_digest(identity)
            or len(cached.beam.modes) != 1):
        raise ValueError("Completed tip-energy cache does not match its executed dependencies")
    arrays = store.auxiliary_arrays(key)
    if arrays is None or set(arrays) != HISTORY_FIELDS | {"near_amplitude"}:
        raise ValueError("Completed tip-energy cache lost mandatory complex history")
    mode = cached.beam.modes[0]
    if mode.mode_id != f"surface-energy:{identity['mode_index']}":
        raise ValueError("Completed tip-energy mode identity changed")
    near = SurfaceWaveMode(identity["energy_ev"], identity["mixture_weight"],
        _immutable(arrays.pop("near_amplitude")), freeze_json(record["mode_record"]["near_flux"]))
    if near.amplitude.shape != arrays["near_z_nm"].shape:
        raise ValueError("Completed tip-energy cache has incompatible near-tip geometry")
    return mode, near, record["mode_record"], record["radial_payload"], arrays


def preserve_mode(store, identity, mode, near, record, radial, history, *, z_mm, current_a, verify):
    if set(history) != HISTORY_FIELDS:
        raise ValueError("A completed tip-energy checkpoint requires every complex boundary state")
    writer = store.writer(mode_key(store, identity), mode.reference_plane)
    try:
        writer.append(mode)
        writer.append_auxiliary("near_amplitude", near.amplitude)
        for name in sorted(history):
            writer.append_auxiliary(name, history[name])
        verify()  # Never commit an energy calculated from changed source inputs.
        writer.finish(z_mm, current_a, {"schema": SCHEMA,
            "executed_identity_digest": json_digest(identity), "mode_record": record,
            "radial_payload": radial,
            "scope": "One executed energy only; complete mixture and physical convergence still required"})
    finally:
        writer.abort()
