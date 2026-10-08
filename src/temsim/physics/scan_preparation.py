"""Resolve captured scan controls once before particle/wave field execution.

The caller owns a private state restored from a verified immutable request.
Only calibration-owned AC/Descan values are cached/copied. Cache keys are the
complete request digest, including source implementation and external inputs;
these records never replace the source, lenses, geometry or a beam state.
"""
from collections import OrderedDict
from copy import deepcopy
from threading import Lock

from temsim.immutable_json import json_digest
from temsim.optics.shared_deflectors import shared_channel_enabled


_PREPARED = OrderedDict()
_PREPARED_LOCK = Lock()
_MAXIMUM_PREPARED = 32
_DERIVED_PUBLIC = frozenset({"lower_coil_gain", "scan_amplitude_x_mrad", "scan_amplitude_y_mrad"})
_RASTER = ("scan_frame_period_s", "scan_pixels_x", "scan_lines", "scan_pixel_size_nm")


def _components(state):
    return state.ac_deflector, state.descan_deflector


def _active_keys(state, stop=None):
    if stop is not None:
        from temsim.physics.instrument_magnetic import column_dipole_fields
        consumed = {key for coil in column_dipole_fields(state)
                    if coil.field_support_mm[0] < float(stop) for key in coil.drive_keys}
    return frozenset(component.key for component in _components(state)
        if shared_channel_enabled(component) and component.scan_enabled
        and (stop is None or component.key in consumed))


def _resolved_values(state):
    result = []
    for index, component in enumerate(_components(state)):
        values = {name: deepcopy(value) for name, value in vars(component).items()
                  if name in _DERIVED_PUBLIC or name.startswith(("_pure_shift_", "_image_plane_", "_scan_"))}
        if index == 1:
            values.update((name, getattr(component, name)) for name in _RASTER)
        result.append(values)
    return tuple(result)


def _install_values(state, values):
    for component, saved in zip(_components(state), values):
        for name, value in saved.items():
            object.__setattr__(component, name, deepcopy(value))


def copy_resolved_scan_drives(source, target):
    """Transfer only the calibration outputs from an executed particle state."""
    _install_values(target, _resolved_values(source))


def scan_drive_identity(state, *, emission_time_s=None):
    """Identity of actual commands, couplings, timing and physical coil hosts."""
    rows = []
    for component in _components(state):
        rows.append(dict(controls=component.to_dict(),
            enabled=shared_channel_enabled(component),
            physical_host_key=str(getattr(component, "_physical_host_key", component.key)),
            coil_planes_mm=(component.upper_z_mm, component.lower_z_mm),
            effective_thickness_mm=component.effective_thickness_mm,
            coil_kick_matrices=component.coil_kick_matrices(),
            scan_command_matrix_mrad=component.scan_command_matrix_mrad))
    epoch = (getattr(state, "simulation_time_s", 0.)
             if emission_time_s is None else emission_time_s)
    return json_digest({"emission_time_s": float(epoch), "scan_drives": rows})


def remember_resolved_scan_drives(state, request_identity, *, active_keys=None):
    """Register verified execution outputs, including a particle-to-wave handoff."""
    active = _active_keys(state) if active_keys is None else frozenset(active_keys)
    key = (str(request_identity), active)
    values = _resolved_values(state)
    with _PREPARED_LOCK:
        _PREPARED[key] = values
        _PREPARED.move_to_end(key)
        while len(_PREPARED) > _MAXIMUM_PREPARED:
            _PREPARED.popitem(last=False)


def prepare_scan_drives(state, request_identity, *, observation_stop_z_mm=None,
                        session_cache=None):
    """Prepare consumed finite scan fields on a private verified worker state.

    Automatic mode solves; held mode restores its record and rescales the FOV.
    Upstream-only observations do not demand unrelated downstream calibration.
    Repeated observations/dwells for the same captured request reuse outputs.
    """
    active = _active_keys(state, observation_stop_z_mm)
    if not active:
        return
    identity = str(request_identity)
    # A scan/session retains its resolved execution even if other workers
    # evict the bounded global cache between dwell or observation requests.
    if session_cache is not None:
        for key, values in tuple(session_cache.items()):
            if key[0] == identity and active <= key[1]:
                _install_values(state, values)
                return
    saved = None
    with _PREPARED_LOCK:
        for key in reversed(_PREPARED):
            if key[0] == identity and active <= key[1]:
                saved = key, _PREPARED[key]
                _PREPARED.move_to_end(key)
                break
    if saved is not None:
        key, values = saved
        # Stored values are private immutable-by-convention copies. Only
        # install deep copies, outside the bookkeeping lock.
        _install_values(state, values)
        if session_cache is not None:
            session_cache[key] = values
        return
    from temsim.physics.scan_geometry import calibrate_scan_system
    calibrate_scan_system(state, observation_stop_z_mm=observation_stop_z_mm)
    remember_resolved_scan_drives(state, request_identity, active_keys=active)
    if session_cache is not None:
        session_cache[(identity, active)] = _resolved_values(state)
