"""Local drive derivatives of the actual finite AC/Descan magnetic providers.

The perturbation is a physical foil's integrated nominal angular command,
in radians. Its captured finite length, registration, host identity and time
are retained. No extra centre-plane impulse is added. These are local chief
responses, not source-beam or finite-distribution qualification.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from temsim import input_io
from temsim.instrument_snapshot import decode_instrument, encode_instrument
from temsim.physics.core import build_propagation_plan, execute_propagation_plan, electron
from temsim.physics.instrument_magnetic import column_dipole_fields


@dataclass(frozen=True)
class FiniteScanResponse:
    z_mm: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    reference: np.ndarray
    source_z_mm: float
    coil_support_mm: tuple[tuple[float, float], tuple[float, float]]
    physical_host_key: str
    perturbation_rad: float

    def at(self, z_mm):
        match = np.flatnonzero(self.z_mm == float(z_mm))
        if len(match) != 1:
            raise ValueError("Finite scan response requires an explicitly captured plane")
        i = int(match[0])
        return self.upper[i], self.lower[i], self.reference[i]


def _private_reference(state, component_key):
    private = decode_instrument(encode_instrument(state))
    # A few local derivative rays are CPU diagnostics, independently of the
    # user's potentially GPU-required full-beam execution policy.
    from temsim.physics.core import NUMBA_AVAILABLE
    private.acceleration_enabled = bool(NUMBA_AVAILABLE)
    private.acceleration_backend = "Numba CPU" if NUMBA_AVAILABLE else "CPU"
    # Freeze an affine reference with static alignment retained and raster /
    # wobble disabled. No first-source transport or per-electron arrival-time
    # solve is implied: other drives use the captured simulation time.
    components = (*getattr(private, "deflectors", ()),
                  *getattr(private, "corrector_elements", ()))
    pair = next((c for c in components if str(c.key) == str(component_key)), None)
    if pair is None:
        raise ValueError(f"Unknown finite scan pair: {component_key}")
    for item in components:
        if str(item.key) in ("ac_deflector", "descan_deflector"):
            item.scan_enabled = False
            item.wobble_enabled = False
    pair.enabled = True
    host = getattr(pair, "_physical_host", None)
    if host is not None:
        host.enabled = True
    return private, pair


def _shift_coil(field, centre_mm):
    delta = (float(centre_mm) - float(field.event_z_mm))*1e-3
    return replace(field, lower_m=field.lower_m+delta, upper_m=field.upper_m+delta,
                   event_z_mm=float(centre_mm))


def _perturb_coil(field, axis, delta, charge):
    length = float(field.upper_m-field.lower_m)
    scale = field.reference_momentum / charge / length
    if axis == 0:
        return replace(field, by_t=field.by_t-scale*delta,
                       event_dx_rad=field.event_dx_rad+delta)
    return replace(field, bx_t=field.bx_t+scale*delta,
                   event_dy_rad=field.event_dy_rad+delta)


@input_io.using_state_inputs
def finite_scan_responses(state, component, target_z_values_mm, *,
                          upper_z_mm=None, lower_z_mm=None,
                          maximum_step_mm=None, perturbation_rad=1e-6):
    """Return two full 4x2 finite-coil derivatives at explicit checkpoints.

    Phase-space order is x/y metres then mechanical x/y slopes in radians.
    Both physical coils are differentiated about the same affine chief,
    launched on axis at the upstream finite-support boundary. Static drives,
    all active lens fields, and posed coil vector fields enter the production
    transport plan. The caller's state is never changed.

    Optional centres are diagnostic native-axis candidate locations. Their
    full registered field supports are retained; mechanical clearance is not
    certified. Observation inside a coil captures its partial finite response.
    """
    targets = np.asarray(sorted({float(z) for z in target_z_values_mm}), dtype=float)
    h = float(perturbation_rad)
    if not len(targets) or not np.isfinite(targets).all():
        raise ValueError("Finite scan observations must be nonempty and finite")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("Finite scan perturbation must be finite and positive")
    if maximum_step_mm is not None and (not np.isfinite(maximum_step_mm) or maximum_step_mm <= 0):
        raise ValueError("Finite scan maximum step must be finite and positive")
    private, pair = _private_reference(state, str(component.key))
    captured = list(column_dipole_fields(private))
    host_key = str(getattr(pair, "_physical_host_key", pair.key))
    indices = [i for i, c in enumerate(captured) if c.key.rsplit(":", 1)[0] == host_key]
    if len(indices) != 2:
        raise ValueError(f"{host_key}: finite scan response requires exactly two physical coils")
    indices.sort(key=lambda i: int(captured[i].key.rsplit(":", 1)[1]))
    for index, centre in zip(indices, (upper_z_mm, lower_z_mm)):
        if centre is not None:
            if not np.isfinite(centre):
                raise ValueError("Finite scan candidate centres must be finite")
            captured[index] = _shift_coil(captured[index], centre)
    supports = tuple(tuple(map(float, captured[i].field_support_mm)) for i in indices)
    # Descan and AC must use the same neutral-raster affine chief, otherwise
    # static AC steering through a nonlinear/posed lens is lost at the lower
    # pair. Start at the AC entrance when it is upstream, never at the source.
    ac_supports = [c.field_support_mm[0] for c in captured if c.key.startswith("ac_deflector:")]
    source = min([s[0] for s in supports] + ac_supports)
    if source <= 0:
        raise ValueError("Finite scan coil support must follow the source plane")
    stop = float(targets[-1])
    reference = np.zeros((len(targets), 4))
    derivatives = np.zeros((2, len(targets), 4, 2))
    downstream = targets[targets > source]
    if not len(downstream):
        return FiniteScanResponse(targets, derivatives[0], derivatives[1], reference,
                                  source, supports, host_key, h)
    charge = float(electron(private)[0])
    zero = np.zeros(1)

    def execute(coils):
        relevant = tuple(c for c in coils
                         if c.field_support_mm[0] < stop and c.field_support_mm[1] > source)
        events = tuple((c.event_z_mm, c.event_dx_rad, c.event_dy_rad) for c in relevant)
        plan = build_propagation_plan(private, source, stop, events,
            include_spherical_aberration=False, include_hexapole=True,
            save_z_mm=downstream, checkpoint_z_mm=downstream,
            maximum_step_mm=maximum_step_mm, _diagnostic_dipole_fields=relevant)
        points = execute_propagation_plan(private, plan, zero, zero, zero, zero)[-1]
        result = np.zeros((len(targets), 4))
        for i, z in enumerate(targets):
            if z <= source:
                continue
            matches = np.flatnonzero(np.asarray(points.z_mm) == z)
            if len(matches) != 1:
                raise ValueError("Finite scan transport lost an exact observation checkpoint")
            j = int(matches[0])
            result[i] = (points.x_m[j,0], points.y_m[j,0], points.tx_rad[j,0], points.ty_rad[j,0])
        if not np.isfinite(result).all():
            raise ValueError("Finite scan reference/perturbation left the transport domain")
        return result

    reference = execute(captured)
    for coil_index, physical_index in enumerate(indices):
        for axis in range(2):
            positive, negative = list(captured), list(captured)
            positive[physical_index] = _perturb_coil(captured[physical_index], axis, h, charge)
            negative[physical_index] = _perturb_coil(captured[physical_index], axis, -h, charge)
            derivatives[coil_index,:,:,axis] = (execute(positive)-execute(negative))/(2*h)
    return FiniteScanResponse(targets, derivatives[0], derivatives[1], reference,
                              source, supports, host_key, h)
