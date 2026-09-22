from temsim.physics.acceleration import momentum_profile
from temsim.component_keys import CONDENSER_LENS_KEYS
from temsim.optics.equivalent_image_lenses import (
    IMAGE_LENS_KEYS,
    equivalent_image_events,
    equivalent_image_lenses_enabled,
)
"""Parallel ray propagator with downsampled history.

Uses Numba across rays when available. The three-Gaussian fields and physical-axis
stigmators are unchanged. Full integration uses state.step_mm; only stored plotting
history is downsampled to reduce memory pressure.
"""
import math
from dataclasses import dataclass
import hashlib
from time import perf_counter
import numpy as np
from temsim.simulation_modes import is_ideal, mode_key

from temsim.optics.lens_focal_length import focal_length_mm
from temsim.optics.magnetic_lens_aberration import spherical_aberration_mm
from temsim.physics.compute_backend import (
    BACKEND_CPU,
    BACKEND_CUDA,
    BACKEND_NUMBA,
    choose_ray_backend,
    gpu_retry_reason, GPUExecutionError, normalise_backend,
)
from temsim.physics.lens_field_provider import (
    runtime_axial_magnetic_field_t,
    FrozenMappedField, active_mapped_providers,
)
from temsim.physics.ray_integrator import (
    NUMBA_AVAILABLE,
    parallel_rk4 as _parallel_rk4,
    serial_rk4 as _serial_rk4,
    vectorised_rk4 as _vectorised_rk4,
    cuda_rk4 as _cuda_rk4,
)

E=1.602176634e-19
M=9.1093837015e-31
C=299792458.0
H=6.62607015e-34
# This is now a solver boundary, not only a drawing range.  At 7 sigma a
# Gaussian amplitude is 2.29e-11 of its peak and its omitted two-sided
# integral is about 2.56e-12.  The explicit boundary makes upstream cache
# independence testable while keeping the truncation below ray-solver error.
FIELD_SIGMA_CUTOFF = 7.0


@dataclass(frozen=True, slots=True)
class PropagationCheckpoints:
    """Full-precision ray phase space at selected axial planes.

    Arrays use metres for X/Y, radians for TX/TY and have shape
    ``(checkpoint, ray)``.  They are kept separate from the float32 plotting
    history so resuming a high-accuracy integration does not add an extra
    history-quantisation error. Optional flight_time_s has the same shape;
    NaN denotes a missing upstream time, not a newly assigned time origin.
    """

    z_mm: np.ndarray
    x_m: np.ndarray
    tx_rad: np.ndarray
    y_m: np.ndarray
    ty_rad: np.ndarray
    flight_time_s: np.ndarray | None = None

    def __post_init__(self):
        z = np.asarray(self.z_mm)
        shape = np.shape(self.x_m)
        if z.ndim != 1 or len(shape) != 2 or shape[0] != z.size:
            raise ValueError("Checkpoint planes and phase-space shapes do not match")
        if not np.all(np.isfinite(z)) or np.any(np.diff(z) <= 0):
            raise ValueError("Checkpoint planes must be finite and strictly increasing")
        for name in ("z_mm", "x_m", "tx_rad", "y_m", "ty_rad"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if name != "z_mm" and value.shape != shape:
                raise ValueError("Checkpoint phase-space arrays do not align")
            # Immutable bytes, not only a reversible NumPy writeable flag.
            frozen = np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)
            object.__setattr__(self, name, frozen)
        if self.flight_time_s is not None:
            time = np.asarray(self.flight_time_s, dtype=np.float64)
            if time.shape != shape or np.any(np.isinf(time)) or np.any(time < 0.):
                raise ValueError("Checkpoint flight times must match rays and be non-negative or NaN")
            frozen = np.frombuffer(time.tobytes(order="C"), dtype=time.dtype).reshape(time.shape)
            object.__setattr__(self, "flight_time_s", frozen)


@dataclass(frozen=True, slots=True)
class AxialPropagationPlan:
    """Immutable, full-axis optical coefficients for restartable propagation."""

    z_mm: np.ndarray
    step_m: np.ndarray
    magnetic_t: np.ndarray
    sx_m2: np.ndarray
    sy_m2: np.ndarray
    sxy_m2: np.ndarray
    hex_normal_m3: np.ndarray
    hex_skew_m3: np.ndarray
    midpoint_magnetic_t: np.ndarray
    midpoint_sx_m2: np.ndarray
    midpoint_sy_m2: np.ndarray
    midpoint_sxy_m2: np.ndarray
    midpoint_hex_normal_m3: np.ndarray
    midpoint_hex_skew_m3: np.ndarray
    cs_kick_m3: np.ndarray
    thin_power_m1: np.ndarray
    thin_rotation_rad: np.ndarray
    kick_x_rad: np.ndarray
    kick_y_rad: np.ndarray
    save_index: np.ndarray
    checkpoint_index: np.ndarray
    solver_signature: str
    signature: str
    mapped_fields: tuple = ()


def _frozen_array(values, dtype=np.float64):
    result = np.ascontiguousarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


def _support_mask(z_mm, provider):
    """Return the solver's explicit finite-support mask for one field."""

    if hasattr(provider, "field_support_mm"):
        try:
            lower, upper = provider.field_support_mm(FIELD_SIGMA_CUTOFF)
        except TypeError:
            lower, upper = provider.field_support_mm()
    else:
        if not (
            hasattr(provider, "effective_length_mm")
            or hasattr(provider, "length_mm")
        ):
            return np.ones_like(np.asarray(z_mm), dtype=bool)
        centre = float(getattr(provider, "z_mm"))
        length = float(
            getattr(
                provider,
                "effective_length_mm",
                getattr(provider, "length_mm", 0.0),
            )
        )
        sigma = max(abs(length) / 2.355, 1e-12)
        lower = centre - FIELD_SIGMA_CUTOFF * sigma
        upper = centre + FIELD_SIGMA_CUTOFF * sigma
    z = np.asarray(z_mm, dtype=float)
    return (z >= float(lower)) & (z <= float(upper))


def electron(state):
    kinetic = E * state.beam_voltage_kv * 1000.0
    rest = M * C * C
    momentum = math.sqrt(kinetic * kinetic + 2.0 * kinetic * rest) / C
    # Charge stays signed so magnetic image rotation retains its handedness.
    return -E, momentum, H / momentum * 1.0e9


class _LegacyAxialFieldProvider:
    """Stable adapter for pre-provider Gaussian lens objects."""

    def __init__(self, source):
        self.source = source
        self.lens = source
        self.key = source.key

    def magnetic_field_t(self, values):
        values = np.asarray(values, dtype=float)
        result = np.zeros_like(values)
        for term in self.source.gaussian:
            result += (
                self.source.scale()
                * term.amplitude
                * np.exp(
                    -0.5
                    * (
                        (
                            values
                            - (
                                self.source.z_mm
                                + term.offset * self.source.a_mm
                            )
                        )
                        / (term.sigma * self.source.a_mm)
                    )
                    ** 2
                )
            )
        return result

    def field_support_mm(self, *args):
        return self.source.field_support_mm(*args)


def fields(z,state, *, exclude_mapped_keys=()):
    if hasattr(state,"sync_objective"): state.sync_objective()
    z=np.asarray(z,float)
    magnetic=np.zeros_like(z)
    sx=np.zeros_like(z)
    sy=np.zeros_like(z)
    equivalent_image = bool(
        getattr(state, "_using_equivalent_image_propagation", False)
    )
    post_sample_only = (
        equivalent_image
        and z.size > 0
        and float(z[0]) >= float(state.sample.z_mm)
    )
    for lens in state.lenses:
        if not getattr(lens,"enabled",True): continue
        if lens.key in exclude_mapped_keys:
            continue
        polarity = int(getattr(lens, "polarity", 1))
        if polarity not in (-1, 1):
            raise ValueError(
                f"{getattr(lens, 'name', 'Round lens')} polarity must be +1 or -1."
            )
        if lens.key in CONDENSER_LENS_KEYS:
            native_provider = state.condenser_system[lens.key]
            contribution, provider = runtime_axial_magnetic_field_t(
                state, lens.key, native_provider, z
            )
            magnetic += np.where(
                _support_mask(z, provider), contribution, 0.0
            )
            continue
        if hasattr(lens, "magnetic_field_t"):
            contribution, provider = runtime_axial_magnetic_field_t(
                state, lens.key, lens, z
            )
            if equivalent_image and lens.key in IMAGE_LENS_KEYS:
                if post_sample_only:
                    continue
                contribution = np.where(
                    z < float(state.sample.z_mm), contribution, 0.0
                )
            magnetic += np.where(
                _support_mask(z, provider), contribution, 0.0
            )
            continue
        # Legacy Gaussian-only lens objects are adapted to the same runtime
        # provider boundary so imported maps cannot bypass geometry checks.
        native_provider = _LegacyAxialFieldProvider(lens)
        contribution, provider = runtime_axial_magnetic_field_t(
            state, lens.key, native_provider, z
        )
        magnetic += np.where(
            _support_mask(z, provider), contribution, 0.0
        )
    sx, sy = multipole_focusing_fields(z, state)
    return magnetic, sx, sy


def multipole_focusing_fields(z, state):
    """Shared continuous quadrupole coefficients, without querying round lenses."""
    z = np.asarray(z, float)
    sx, sy = np.zeros_like(z), np.zeros_like(z)
    for stig in state.stigmators:
        if not stig.enabled: continue
        qx, qy, _ = stig.quadrupole_tensor_m2(z)
        mask = _support_mask(z, stig)
        sx += np.where(mask, qx, 0.0)
        sy += np.where(mask, qy, 0.0)
    for component in getattr(state, "corrector_elements", []):
        if not getattr(component, "enabled", False):
            continue
        if not hasattr(component, "quadrupole_strength_m2"):
            continue
        q = component.quadrupole_strength_m2(z)
        mask = _support_mask(z, component)
        sx += np.where(mask, q, 0.0)
        sy -= np.where(mask, q, 0.0)
    return sx, sy

def skew_quadrupole_field(z, state):
    """Off-diagonal focusing coefficient in the same frame as fields()."""
    z = np.asarray(z, float)
    skew = np.zeros_like(z)
    for stig in state.stigmators:
        if stig.enabled:
            _, _, value = stig.quadrupole_tensor_m2(z)
            skew += np.where(_support_mask(z, stig), value, 0.0)
    return skew

def bz(z,state): return fields(z,state)[0]


def hexapole_field(z, state):
    """Return the summed normal hexapole coefficient for compatibility."""

    return hexapole_field_components(z, state)[0]


def hexapole_field_components(z, state):
    """Sum continuous normal and skew hexapole coefficients."""

    z = np.asarray(z, float)
    normal = np.zeros_like(z)
    skew = np.zeros_like(z)
    if is_ideal(state):
        return normal, skew
    for component in getattr(state, "corrector_elements", []):
        if not getattr(component, "enabled", False):
            continue
        if hasattr(component, "hexapole_strength_components_m3"):
            component_normal, component_skew = (
                component.hexapole_strength_components_m3(z)
            )
            mask = _support_mask(z, component)
            normal += np.where(mask, component_normal, 0.0)
            skew += np.where(mask, component_skew, 0.0)
            continue
        if hasattr(component, "hexapole_strength_m3"):
            contribution = component.hexapole_strength_m3(z)
            normal += np.where(
                _support_mask(z, component), contribution, 0.0
            )
    return normal, skew


def larmor_coefficients_m1(magnetic_t, momentum, z_mm):
    """Return signed Larmor rate and its axial derivative.

    The coordinates use a right-handed frame with electrons travelling along
    +Z.  ``polarity=+1`` means +Bz and positive rotation follows the right-hand
    rule about +Z.  The rate is ``-q Bz / (2 pz)``; the coupled laboratory-frame
    equations below use ``g = q Bz / (2 pz) = -d(phi)/dz``.
    """

    magnetic = np.asarray(magnetic_t, dtype=float)
    momentum = np.asarray(momentum, dtype=float)
    z_m = np.asarray(z_mm, dtype=float) * 1.0e-3
    g = (-E) * magnetic[:, None] / (2.0 * momentum)
    if g.ndim == 1:
        g = g[:, None]
    if len(z_m) < 2:
        gradient = np.zeros_like(g)
    else:
        gradient = np.gradient(g, z_m, axis=0, edge_order=1)
    return (
        np.ascontiguousarray(g, dtype=np.float64),
        np.ascontiguousarray(gradient, dtype=np.float64),
    )


def spherical_aberration_kick_m3(z_mm, state):
    """Return thin, rotationally symmetric third-order lens-kick strengths.

    A positive conventional ``Cs`` increases the inward deflection of marginal
    rays.  For an equivalent focal length ``f`` the kick is
    ``delta slope = -(Cs/f**4) * r**2 * r_vector``.  It therefore reproduces
    the standard transverse blur magnitude ``Cs * alpha**3`` at the paraxial
    focal plane for a collimated bundle.  Field-derived aberration must not be
    combined with this calibrated preview model.
    """

    z = np.asarray(z_mm, dtype=float)
    result = np.zeros(z.size, dtype=np.float64)
    if z.size == 0 or is_ideal(state):
        return result
    boundary_tolerance = 32.0 * np.finfo(np.float64).eps * max(1., abs(float(z[0])), abs(float(z[-1])))
    mapped_keys = {provider.lens_key for provider in active_mapped_providers(state)}
    for lens in getattr(state, "lenses", ()):
        if not bool(getattr(lens, "enabled", True)):
            continue
        if lens.key in mapped_keys:
            # The imported spatial field supplies its own ray aberrations.
            # Do not add the native Gaussian lens's calibrated Cs again.
            continue
        cs_mm = spherical_aberration_mm(lens, state.beam_voltage_kv)
        if cs_mm is None or float(cs_mm) == 0.0:
            continue
        lens_z = float(getattr(lens, "z_mm"))
        if lens_z < float(z[0]) - boundary_tolerance or lens_z > float(z[-1]) + boundary_tolerance:
            continue
        try:
            focal_mm = float(focal_length_mm(lens, state.beam_voltage_kv))
        except Exception:
            continue
        if not math.isfinite(focal_mm) or focal_mm <= 0.0:
            continue
        index = int(np.argmin(np.abs(z - lens_z)))
        cs_m = float(cs_mm) * 1.0e-3
        focal_m = focal_mm * 1.0e-3
        result[index] += cs_m / focal_m**4
    return result


def _record_active_backend(state, backend, fallback_reason=None):
    """Accumulate backends without letting two-ray diagnostics hide CUDA."""
    used = getattr(state, "_active_backends_used", None)
    if used is None:
        used = set()
        state._active_backends_used = used
    used.add(str(backend))
    priority = (BACKEND_CUDA, BACKEND_NUMBA, BACKEND_CPU)
    primary = next(name for name in priority if name in used)
    auxiliaries = sorted(name for name in used if name != primary)
    label = primary
    if auxiliaries:
        label += f" (+ {', '.join(auxiliaries)} auxiliaries)"
    if fallback_reason:
        label += f" (fallback: {fallback_reason})"
    state.active_backend = label


def _endpoint_exact_axial_grid(z0, z1, maximum_step_mm):
    """Return a forward grid whose endpoints are physically exact.

    Full intervals retain ``maximum_step_mm`` so calibrated thin kicks and
    continuous corrector fields keep their established sampling phase.  Only
    the final interval is shortened when necessary.  RK4 therefore reaches
    interfaces such as the specimen plane without crossing or stopping short.
    """

    start = float(z0)
    stop = float(z1)
    maximum_step = float(maximum_step_mm)
    if not math.isfinite(start) or not math.isfinite(stop):
        raise ValueError("Propagation Z limits must be finite")
    if not math.isfinite(maximum_step) or maximum_step <= 0.0:
        raise ValueError("Propagation step must be finite and positive")
    span = stop - start
    if span < 0.0:
        raise ValueError("Propagation stop Z must not precede start Z")
    if span == 0.0:
        return (
            np.array([start], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    quotient = span / maximum_step
    nearest_integer = int(round(quotient))
    # Subtracting two large absolute Z coordinates can leave an exact
    # step-multiple a few ulps away from its integer quotient.  Scale the
    # tolerance with the absolute coordinate magnitudes as well as the span;
    # otherwise a spurious ~1e-13 mm terminal interval makes np.gradient
    # numerically singular at an exact optical plane.
    quotient_tolerance = 64.0 * np.finfo(np.float64).eps * max(
        1.0,
        abs(quotient),
        abs(start) / maximum_step,
        abs(stop) / maximum_step,
    )
    if (
        nearest_integer >= 1
        and abs(quotient - nearest_integer) <= quotient_tolerance
    ):
        full_interval_count = nearest_integer
        grid = start + maximum_step * np.arange(
            full_interval_count + 1, dtype=np.float64
        )
    else:
        full_interval_count = int(math.floor(quotient))
        grid = start + maximum_step * np.arange(
            full_interval_count + 1, dtype=np.float64
        )
        grid = np.append(grid, stop)

    grid[0] = start
    grid[-1] = stop
    intervals = np.diff(grid)
    step_tolerance = 32.0 * np.finfo(np.float64).eps * max(
        1.0, abs(start), abs(stop), abs(maximum_step)
    )
    if (
        np.any(intervals <= 0.0)
        or np.any(intervals > maximum_step + step_tolerance)
    ):
        raise RuntimeError("Failed to construct an endpoint-exact axial grid")
    return grid, intervals


def _nearest_axial_grid_index(value, grid):
    """Return the nearest valid index on a monotonic axial grid."""

    size = len(grid)
    if size <= 1:
        return np.int64(0)
    requested = float(value)
    upper = int(np.searchsorted(grid, requested, side="left"))
    if upper <= 0:
        return np.int64(0)
    if upper >= size:
        return np.int64(size - 1)
    lower = upper - 1
    if requested - float(grid[lower]) <= float(grid[upper]) - requested:
        return np.int64(lower)
    return np.int64(upper)


def _piecewise_endpoint_exact_axial_grid(
    z0, z1, maximum_step_mm, interior_z_mm=()
):
    """Build a step-bounded grid containing every optical event plane."""

    start = float(z0)
    stop = float(z1)
    boundaries = [
        start,
        *sorted({
            float(value)
            for value in interior_z_mm
            if start < float(value) < stop
        }),
        stop,
    ]
    segments = [
        _endpoint_exact_axial_grid(left, right, maximum_step_mm)[0]
        for left, right in zip(boundaries[:-1], boundaries[1:])
    ]
    grid = np.concatenate([
        segment if index == 0 else segment[1:]
        for index, segment in enumerate(segments)
    ])
    return grid, np.diff(grid)

def interleaved_rk4_values(nodes, midpoints):
    """Pack true endpoint/midpoint samples without interpolating fields."""
    node_values = np.asarray(nodes, dtype=np.float64)
    midpoint_values = np.asarray(midpoints, dtype=np.float64)
    if midpoint_values.shape != (max(node_values.size - 1, 0),):
        raise ValueError("RK4 stages require one midpoint per node interval")
    result = np.empty(max(2 * node_values.size - 1, 0), dtype=np.float64)
    result[::2] = node_values
    result[1::2] = midpoint_values
    return result


def build_propagation_plan(
    state, z0, z1, events=(), *, include_spherical_aberration=True,
    include_hexapole=True, save_z_mm=(), checkpoint_z_mm=(),
    maximum_step_mm=None, particle_medium=False, medium_energy_ev=None,
):
    """Build the single global axial plan used by full and resumed traces."""

    events = tuple(events)
    save_z_mm = tuple(save_z_mm)
    # Physical clipping must never depend on the sparse plotting history.
    # Keep each active aperture at its exact plane in every caller, including
    # full-column runs and Direct Alignment. Adding a saved plane is not a
    # second aperture action; the existing clipping owner still applies it.
    save_z_mm += tuple(
        float(aperture.z_mm) for aperture in getattr(state, "apertures", ())
        if bool(getattr(aperture, "enabled", True))
        and bool(getattr(aperture, "installed", True))
        and float(z0) <= float(aperture.z_mm) <= float(z1)
    )
    if is_ideal(state):
        include_spherical_aberration = include_hexapole = False
    nanopulser = getattr(state, "nanopulser", None)
    if nanopulser is not None and bool(nanopulser.installed):
        nanopulser.validate()
        events += tuple(
            event for event in nanopulser.kick_events(state.beam_voltage_kv)
            if float(z0) <= float(event[0]) <= float(z1)
        )
        # Stop interception must use the exact aperture plane even when the
        # GUI requests a coarse drawing-history interval. Keeping this in the
        # common plan also covers direct-alignment and calibration traces.
        if float(z0) <= float(nanopulser.stop_z_mm) <= float(z1):
            save_z_mm += (float(nanopulser.stop_z_mm),)
    requested_step=float(state.step_mm)
    if maximum_step_mm is not None:
        requested_step=min(requested_step,float(maximum_step_mm))
    reduce_image_maps = (equivalent_image_lenses_enabled(state)
                         and float(z0) >= float(state.sample.z_mm))
    # A span crossing the specimen uses distributed fields throughout. Thin
    # image events are only valid in a post-specimen-only segment, where the
    # corresponding distributed image fields are also excluded below.
    image_lens_events = (equivalent_image_events(state,float(z0),float(z1))
                         if reduce_image_maps else ())
    mapped_fields = tuple(
        FrozenMappedField.from_provider(provider)
        for provider in active_mapped_providers(state)
        if not (reduce_image_maps and provider.lens_key in IMAGE_LENS_KEYS)
        and provider.field_support_mm()[0] <= float(z1)
        and provider.field_support_mm()[1] >= float(z0)
    )
    mapped_keys = {item.lens_key for item in mapped_fields}
    exact_z_mm = [event.z_mm for event in image_lens_events]
    if particle_medium:
        from temsim.physics.residual_medium import medium_grid_nodes
        exact_z_mm.extend(medium_grid_nodes(state, z0, z1, medium_energy_ev))
        for segment in getattr(getattr(state, "_resolved_assembly", None), "vacuum_bore_segments", ()):
            exact_z_mm.extend(v for v in (segment.start_z_mm, segment.end_z_mm) if z0 < v < z1)
        exact_z_mm.extend(float(p.z_mm) for p in getattr(state, "recording_planes", ())
                          if z0 <= float(p.z_mm) <= z1)
        save_z_mm += tuple(float(v) for v in exact_z_mm)
    for item in mapped_fields:
        lower, upper = item.field_map.field_support_mm
        lower, upper = max(lower, float(z0)), min(upper, float(z1))
        if upper <= lower:
            continue
        local_step = item.maximum_step_mm(requested_step, electron(state)[1])
        exact_z_mm.extend(np.linspace(lower, upper,
            max(1, int(math.ceil((upper-lower)/local_step)))+1))
    exact_z_mm.extend(float(value) for value in save_z_mm)
    # Impulsive actions must occur at their physical planes, independent of
    # the requested integration step or an unrelated observation plane.
    exact_z_mm.extend(float(event[0]) for event in events)
    if include_spherical_aberration:
        exact_z_mm.extend(
            float(lens.z_mm) for lens in state.lenses
            if bool(getattr(lens, 'enabled', True))
            and spherical_aberration_mm(lens, state.beam_voltage_kv)
        )
    if particle_medium and ((z1-z0)/requested_step+len(exact_z_mm)+1 > state.vacuum_map.max_transport_nodes):
        raise ValueError("Particle / vacuum integration exceeds the configured node budget")
    zfull,step_mm=_piecewise_endpoint_exact_axial_grid(
        z0,z1,requested_step,
        exact_z_mm,
    )
    step_m=np.ascontiguousarray(step_mm*1e-3,np.float64)
    midpoint_z_mm = 0.5 * (zfull[:-1] + zfull[1:])
    grid_start=float(zfull[0])
    had_equivalent_propagation_flag = hasattr(
        state, "_using_equivalent_image_propagation"
    )
    previous_equivalent_propagation_flag = bool(
        getattr(state, "_using_equivalent_image_propagation", False)
    )
    state._using_equivalent_image_propagation = (
        equivalent_image_lenses_enabled(state)
        and float(zfull[0]) >= float(state.sample.z_mm)
    )
    try:
        field_options = {"exclude_mapped_keys": mapped_keys} if mapped_keys else {}
        magnetic,sx,sy=fields(zfull,state, **field_options)
        midpoint_magnetic, midpoint_sx, midpoint_sy = fields(
            midpoint_z_mm, state, **field_options
        )
        sxy = skew_quadrupole_field(zfull, state)
        midpoint_sxy = skew_quadrupole_field(midpoint_z_mm, state)
    finally:
        if had_equivalent_propagation_flag:
            state._using_equivalent_image_propagation = (
                previous_equivalent_propagation_flag
            )
        else:
            delattr(state, "_using_equivalent_image_propagation")
    if include_hexapole:
        hex_normal, hex_skew = hexapole_field_components(zfull, state)
        midpoint_hex_normal, midpoint_hex_skew = hexapole_field_components(
            midpoint_z_mm, state
        )
    else:
        hex_normal = np.zeros(len(zfull), np.float64)
        hex_skew = np.zeros(len(zfull), np.float64)
        midpoint_hex_normal = np.zeros(len(midpoint_z_mm), np.float64)
        midpoint_hex_skew = np.zeros(len(midpoint_z_mm), np.float64)
    hex_normal=np.ascontiguousarray(hex_normal,np.float64)
    hex_skew=np.ascontiguousarray(hex_skew,np.float64)
    cs_kick=np.ascontiguousarray(
        spherical_aberration_kick_m3(zfull,state)
        if include_spherical_aberration else np.zeros(len(zfull)),
        np.float64,
    )
    thin_power=np.zeros(len(zfull),np.float64)
    thin_rotation=np.zeros(len(zfull),np.float64)
    for lens_event in image_lens_events:
        index=_nearest_axial_grid_index(lens_event.z_mm,zfull)
        thin_power[index]+=float(lens_event.power_m1)
        thin_rotation[index]+=float(lens_event.rotation_rad)
    thin_power=np.ascontiguousarray(thin_power,np.float64)
    thin_rotation=np.ascontiguousarray(thin_rotation,np.float64)
    kickx=np.zeros(len(zfull),np.float64);kicky=np.zeros(len(zfull),np.float64)
    for ze,dx,dy in sorted(events):
        idx=_nearest_axial_grid_index(ze,zfull);kickx[idx]+=dx;kicky[idx]+=dy
    history_step=max(requested_step,float(getattr(state,"history_step_mm",2.0)))
    stride=max(1,int(round(history_step/requested_step)))
    save=np.arange(0,len(zfull),stride,dtype=np.int64)
    observation_z = getattr(state, "observation_plane_z_mm", None)
    if observation_z is not None and grid_start <= float(observation_z) <= float(zfull[-1]):
        observation_index = _nearest_axial_grid_index(observation_z, zfull)
        save = np.unique(np.r_[save, observation_index])
    requested_save_indices = [
        _nearest_axial_grid_index(value, zfull)
        for value in save_z_mm
        if grid_start <= float(value) <= float(zfull[-1])
    ]
    if requested_save_indices:
        save = np.unique(np.r_[save, requested_save_indices])
    checkpoint_indices = np.unique(np.asarray([
        _nearest_axial_grid_index(value, zfull)
        for value in checkpoint_z_mm
        if grid_start <= float(value) <= float(zfull[-1])
    ], dtype=np.int64))
    if save[-1] != len(zfull)-1: save=np.r_[save,np.int64(len(zfull)-1)]
    digest = hashlib.sha256()
    digest.update(f"field-cutoff={FIELD_SIGMA_CUTOFF:.17g}".encode("ascii"))
    solver_signature = repr((
        float(getattr(state, "beam_voltage_kv")),
        float(requested_step),
        float(getattr(state, "history_step_mm", requested_step)),
        bool(getattr(state, "acceleration_enabled", False)),
        str(getattr(state, "acceleration_backend", "Auto")),
        FIELD_SIGMA_CUTOFF,
        'canonical-rk4-quadrupole-tensor-tof-v4',
        mode_key(state),
        state.vacuum_map.signature() if particle_medium else "optical-map-no-medium",
    ))
    digest.update(solver_signature.encode("utf-8"))
    for item in mapped_fields:
        digest.update(item.fingerprint.encode("ascii"))
    for values in (
        zfull, step_m, magnetic, sx, sy, sxy, hex_normal, hex_skew,
        midpoint_magnetic, midpoint_sx, midpoint_sy, midpoint_sxy,
        midpoint_hex_normal, midpoint_hex_skew,
        cs_kick, thin_power, thin_rotation, kickx, kicky, save,
        checkpoint_indices,
    ):
        contiguous = np.ascontiguousarray(values, dtype=np.float64)
        digest.update(contiguous.shape.__repr__().encode("ascii"))
        digest.update(contiguous.tobytes())
    return AxialPropagationPlan(
        z_mm=_frozen_array(zfull),
        step_m=_frozen_array(step_m),
        magnetic_t=_frozen_array(magnetic),
        sx_m2=_frozen_array(sx),
        sy_m2=_frozen_array(sy),
        sxy_m2=_frozen_array(sxy),
        hex_normal_m3=_frozen_array(hex_normal),
        hex_skew_m3=_frozen_array(hex_skew),
        midpoint_magnetic_t=_frozen_array(midpoint_magnetic),
        midpoint_sx_m2=_frozen_array(midpoint_sx),
        midpoint_sy_m2=_frozen_array(midpoint_sy),
        midpoint_sxy_m2=_frozen_array(midpoint_sxy),
        midpoint_hex_normal_m3=_frozen_array(midpoint_hex_normal),
        midpoint_hex_skew_m3=_frozen_array(midpoint_hex_skew),
        cs_kick_m3=_frozen_array(cs_kick),
        thin_power_m1=_frozen_array(thin_power),
        thin_rotation_rad=_frozen_array(thin_rotation),
        kick_x_rad=_frozen_array(kickx),
        kick_y_rad=_frozen_array(kicky),
        save_index=_frozen_array(save, np.int64),
        checkpoint_index=_frozen_array(checkpoint_indices, np.int64),
        solver_signature=solver_signature,
        signature=digest.hexdigest(),
        mapped_fields=mapped_fields,
    )


def propagation_plan_common_prefix_nodes(previous, current):
    """Return count of leading nodes whose actual optical actions are equal."""

    if previous is None or current is None:
        return 0
    if previous.solver_signature != current.solver_signature:
        return 0
    old_z = np.asarray(previous.z_mm)
    new_z = np.asarray(current.z_mm)
    count = min(old_z.size, new_z.size)
    if count == 0:
        return 0
    arrays = (
        (old_z, new_z),
        (previous.magnetic_t, current.magnetic_t),
        (previous.sx_m2, current.sx_m2),
        (previous.sy_m2, current.sy_m2),
        (previous.sxy_m2, current.sxy_m2),
        (previous.hex_normal_m3, current.hex_normal_m3),
        (previous.hex_skew_m3, current.hex_skew_m3),
        (previous.cs_kick_m3, current.cs_kick_m3),
        (previous.thin_power_m1, current.thin_power_m1),
        (previous.thin_rotation_rad, current.thin_rotation_rad),
        (previous.kick_x_rad, current.kick_x_rad),
        (previous.kick_y_rad, current.kick_y_rad),
    )
    equal = np.ones(count, dtype=bool)
    old_maps = {item.lens_key: item for item in previous.mapped_fields}
    new_maps = {item.lens_key: item for item in current.mapped_fields}
    for key in old_maps.keys() | new_maps.keys():
        old, new = old_maps.get(key), new_maps.get(key)
        if old is not None and new is not None and old.fingerprint == new.fingerprint:
            continue
        boundary = min(item.field_map.field_support_mm[0]
                       for item in (old,new) if item is not None)
        equal &= new_z[:count] < boundary
    for old, new in arrays:
        equal &= np.asarray(old[:count]) == np.asarray(new[:count])
    # Midpoint i belongs to the interval leaving node i.  A changed
    # midpoint invalidates that interval even when its endpoint fields match.
    for name in (
        'midpoint_magnetic_t', 'midpoint_sx_m2', 'midpoint_sy_m2',
        'midpoint_sxy_m2',
        'midpoint_hex_normal_m3', 'midpoint_hex_skew_m3',
    ):
        old, new = getattr(previous, name), getattr(current, name)
        interval_count = min(count, old.size, new.size)
        equal[:interval_count] &= old[:interval_count] == new[:interval_count]
    changed = np.flatnonzero(~equal)
    if changed.size:
        # Preserve the preceding interval when an endpoint or its midpoint
        # changes.  The extra node keeps existing checkpoint selection safe.
        return max(0, int(changed[0]) - 2)
    if old_z.size != new_z.size:
        return max(0, count - 2)
    return count


def execute_propagation_plan(
    state, plan, x, tx, y, ty, energy_offset_ev=None, *, start_index=0,
    include_initial_plane_kicks=True,
    defer_nonfinite_until_clipping=False,
    medium_transport=None,
    initial_time_s=None, return_flight_times=False,
):
    """Execute a complete plan or resume it from an after-action checkpoint.

    With return_flight_times, append float64 saved times after the five ray
    history arrays and before checkpoints. Resuming passes the checkpoint's
    time row explicitly; absent upstream times remain NaN throughout.
    """

    from temsim.physics.optical_tuning import check_tuning_cancelled
    check_tuning_cancelled(state)

    start_index = int(start_index)
    if not 0 <= start_index < len(plan.z_mm):
        raise ValueError("Propagation-plan start index is out of range")
    zfull = np.asarray(plan.z_mm[start_index:], dtype=np.float64)
    step_m = np.ascontiguousarray(plan.step_m[start_index:], np.float64)
    def stages(node_name, midpoint_name):
        nodes = np.asarray(getattr(plan, node_name)[start_index:])
        midpoint = np.asarray(getattr(plan, midpoint_name)[start_index:])
        return interleaved_rk4_values(nodes, midpoint)

    magnetic = stages("magnetic_t", "midpoint_magnetic_t")
    sx = stages("sx_m2", "midpoint_sx_m2")
    sy = stages("sy_m2", "midpoint_sy_m2")
    sxy = stages("sxy_m2", "midpoint_sxy_m2")
    hex_normal = stages("hex_normal_m3", "midpoint_hex_normal_m3")
    hex_skew = stages("hex_skew_m3", "midpoint_hex_skew_m3")
    cs_kick = np.array(plan.cs_kick_m3[start_index:], dtype=np.float64)
    thin_power = np.array(plan.thin_power_m1[start_index:], dtype=np.float64)
    thin_rotation = np.array(
        plan.thin_rotation_rad[start_index:], dtype=np.float64
    )
    kickx = np.array(plan.kick_x_rad[start_index:], dtype=np.float64)
    kicky = np.array(plan.kick_y_rad[start_index:], dtype=np.float64)
    if not bool(include_initial_plane_kicks):
        cs_kick[0]=0.0
        thin_power[0]=0.0
        thin_rotation[0]=0.0
        kickx[0]=0.0
        kicky[0]=0.0
    arrays=[np.array(a,dtype=np.float64,order="C",copy=True) for a in (x,tx,y,ty)]
    global_save = np.asarray(plan.save_index, dtype=np.int64)
    save = np.ascontiguousarray(
        global_save[global_save >= start_index] - start_index, np.int64
    )
    global_checkpoints = np.asarray(plan.checkpoint_index, dtype=np.int64)
    checkpoint_index = np.ascontiguousarray(
        global_checkpoints[global_checkpoints >= start_index] - start_index,
        np.int64,
    )
    backend, fallback_reason = choose_ray_backend(
        getattr(state, "acceleration_backend", "Auto"),
        acceleration_enabled=bool(getattr(state, "acceleration_enabled", False)),
        ray_count=arrays[0].size,
    )
    # Post-gun momentum is constant along Z, including with an energy spread.
    # Keep this factor separate on every backend; no (Z, ray) coefficient
    # matrices or finite-difference magnetic derivatives are necessary.
    # Ideal column optics is evaluated at the reference energy. The caller's
    # energy array remains unchanged for scattering, EDS and EELS bookkeeping.
    momentum_at_start = momentum_profile(state, zfull[:1], None if is_ideal(state) else energy_offset_ev)
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
        save, checkpoint_index, sxy,
    )
    timing = {}
    if return_flight_times:
        initial = (np.full(arrays[0].size, np.nan, dtype=np.float64)
                   if initial_time_s is None else np.asarray(initial_time_s, dtype=np.float64))
        if initial.shape != (arrays[0].size,) or np.any(np.isinf(initial)) or np.any(initial < 0.):
            raise ValueError("Initial flight times must be one non-negative or NaN value per ray")
        # Acceleration has already executed in the gun. The present column
        # model has constant energy; use actual per-particle energy even when
        # Ideal optics evaluates its trajectory at the reference energy.
        actual_momentum = np.asarray(momentum_profile(state, zfull[:1], energy_offset_ev)[0])
        actual_momentum = np.broadcast_to(actual_momentum, (arrays[0].size,))
        inverse_speed = np.sqrt(M*M+(actual_momentum/C)**2)/actual_momentum
        timing = dict(initial_time_s=np.ascontiguousarray(initial),
                      inverse_speed=np.ascontiguousarray(inverse_speed))
    policy = normalise_backend(getattr(state, "acceleration_backend", "Auto")).lower().replace(" ", "_")
    tuning = bool(getattr(state, "_optical_tuning", False) or getattr(state, "_particle_tuning", False))
    workload = None
    if medium_transport is None and not plan.mapped_fields and not tuning:
        from temsim.physics.ray_device_cache import STAGE_COSTS, measured_workload
        workload = measured_workload((*inputs, *timing.values()) if timing else inputs)
        if policy == "auto" and getattr(state, "acceleration_enabled", False):
            from temsim.physics.compute_backend import cuda_capability
            eligible = [BACKEND_CPU]
            if NUMBA_AVAILABLE:
                eligible.append(BACKEND_NUMBA)
            if backend == BACKEND_CUDA or (BACKEND_CUDA in STAGE_COSTS.rows.get(workload, {}) and cuda_capability().available):
                eligible.append(BACKEND_CUDA)
            backend, measured_reason = STAGE_COSTS.choose(workload, eligible, backend)
            fallback_reason = measured_reason or fallback_reason
    transport_started = perf_counter()
    retried = False
    if policy == "require_gpu" and (medium_transport is not None or plan.mapped_fields):
        raise GPUExecutionError("unsupported_stage", "Requested column transport requires the existing CPU vector-field or residual-medium solver")
    if medium_transport is not None and not plan.mapped_fields:
        outputs = _vectorised_rk4(*inputs, step_operator=medium_transport, **timing)
        backend, fallback_reason = BACKEND_CPU, "classical residual-medium collisions between optical steps"
    elif plan.mapped_fields:
        from temsim.physics.vector_field_transport import vector_map_rk4
        backend, fallback_reason = BACKEND_CPU, "imported vector-field RK4"
        outputs = vector_map_rk4(*inputs, z_mm=zfull, mapped_fields=plan.mapped_fields,
                                defer_nonfinite_until_clipping=defer_nonfinite_until_clipping,
                                step_operator=medium_transport, **timing)
    elif (tuning and NUMBA_AVAILABLE
          and getattr(state, "acceleration_enabled", False)
          and getattr(state, "acceleration_backend", "Auto") == "Auto"):
        try:
            outputs = _serial_rk4(*inputs, **timing)
            backend = BACKEND_NUMBA
            state._tuning_kernel = "serial_numba"
        except Exception as exc:
            backend, fallback_reason = BACKEND_CPU, f"Tuning JIT unavailable: {exc}"
            outputs = _vectorised_rk4(*inputs, **timing)
    elif backend == BACKEND_CUDA:
        try:
            outputs = _cuda_rk4(*inputs, **timing)
        except Exception as exc:
            # Input, physics and cancellation errors must keep their original
            # meaning; only declared accelerator failures can use a CPU retry.
            from temsim.physics.backend_execution import record_backend
            from temsim.physics.compute_backend import gpu_failure_evidence
            record_backend("column", getattr(state, "acceleration_backend", "Auto"), BACKEND_CUDA,
                           outcome="raised", **gpu_failure_evidence(exc, stage="column_transport"))
            fallback_reason = gpu_retry_reason(exc, policy, stage="column_transport")
            retried = True
            backend = BACKEND_NUMBA if NUMBA_AVAILABLE else BACKEND_CPU
            outputs = (
                _parallel_rk4(*inputs, **timing) if backend == BACKEND_NUMBA
                else _vectorised_rk4(*inputs, **timing)
            )
    elif backend == BACKEND_NUMBA:
        outputs = _parallel_rk4(*inputs, **timing)
    else:
        outputs = _vectorised_rk4(*inputs, **timing)
    check_tuning_cancelled(state)
    if workload is not None and not retried:
        STAGE_COSTS.record(workload, backend, perf_counter() - transport_started)
    if backend == BACKEND_CUDA:
        from temsim.physics.ray_device_cache import last_device_receipt
        state._last_ray_device_receipt = last_device_receipt()
    _record_active_backend(state, backend, fallback_reason)
    from temsim.physics.backend_execution import record_backend
    record_backend("column", getattr(state, "acceleration_backend", "Auto"), backend,
                   reason=fallback_reason or "", retried=retried)
    X,TX,Y,TY,CX,CTX,CY,CTY=outputs[:8]
    checkpoints = PropagationCheckpoints(
        z_mm=_frozen_array(zfull[checkpoint_index]),
        x_m=_frozen_array(CX),
        tx_rad=_frozen_array(CTX),
        y_m=_frozen_array(CY),
        ty_rad=_frozen_array(CTY),
        flight_time_s=outputs[9] if return_flight_times else None,
    )
    if return_flight_times:
        return zfull[save],X,TX,Y,TY,outputs[8],checkpoints
    return zfull[save],X,TX,Y,TY,checkpoints


def propagate(
    state,z0,z1,x,tx,y,ty,events=(),energy_offset_ev=None,
    *,include_spherical_aberration=True,include_hexapole=True,
    save_z_mm=(),include_initial_plane_kicks=True,
    checkpoint_z_mm=(),return_checkpoints=False,maximum_step_mm=None,
    defer_nonfinite_until_clipping=False,
    particle_medium=False, medium_alive=None, medium_stream=2, medium_output=None,
    initial_time_s=None, return_flight_times=False,
):
    plan = build_propagation_plan(
        state,z0,z1,events,
        include_spherical_aberration=include_spherical_aberration,
        include_hexapole=include_hexapole,
        save_z_mm=save_z_mm,
        checkpoint_z_mm=checkpoint_z_mm,
        maximum_step_mm=maximum_step_mm,
        particle_medium=particle_medium,
        medium_energy_ev=(state.beam_voltage_kv*1000+np.asarray(0 if energy_offset_ev is None else energy_offset_ev)),
    )
    if particle_medium and len(plan.z_mm) > state.vacuum_map.max_transport_nodes:
        raise ValueError("Particle / vacuum integration exceeds the configured node budget")
    transport = None
    if particle_medium and state.vacuum_map.enabled:
        from temsim.physics.residual_medium import ColumnMediumTransport
        transport = ColumnMediumTransport(state, plan, len(x),
            state.beam_voltage_kv*1000+np.asarray(0 if energy_offset_ev is None else energy_offset_ev),
            alive=medium_alive, stream=medium_stream)
        if medium_output is not None:
            medium_output.append(transport)
    result = execute_propagation_plan(
        state,plan,x,tx,y,ty,energy_offset_ev,
        include_initial_plane_kicks=include_initial_plane_kicks,
        defer_nonfinite_until_clipping=defer_nonfinite_until_clipping,
        medium_transport=transport,
        initial_time_s=initial_time_s, return_flight_times=return_flight_times,
    )
    return result if return_checkpoints else result[:6 if return_flight_times else 5]

def transfer(state,z0,z1):
    if active_mapped_providers(state):
        from temsim.physics.first_order import trace_transverse_transfer
        matrix = trace_transverse_transfer(state, z0, z1).matrix
        return matrix[np.ix_((0, 2), (0, 2))]
    _,x,tx,_,_=propagate(
        state,z0,z1,
        np.array([1.,0.]),np.array([0.,1.]),np.zeros(2),np.zeros(2),
        include_spherical_aberration=False,include_hexapole=False,
    )
    return np.array([[x[-1,0],x[-1,1]],[tx[-1,0],tx[-1,1]]],float)


def complex_transfer(state, z0, z1):
    """Return the first-order, Larmor-coupled transfer in complex form."""

    if active_mapped_providers(state):
        from temsim.physics.first_order import trace_transverse_transfer
        matrix = trace_transverse_transfer(state, z0, z1).matrix
        result = np.empty((2, 2), dtype=np.complex128)
        for i in range(2):
            for j in range(2):
                block = matrix[2*i:2*i+2, 2*j:2*j+2]
                expected = np.array(((block[0,0], -block[1,0]),
                                     (block[1,0], block[0,0])))
                if not np.allclose(block, expected, rtol=1e-5, atol=1e-9):
                    raise ValueError("Non-axisymmetric map requires the full 4x4 transverse transfer")
                result[i,j] = block[0,0] + 1j*block[1,0]
        return result

    _, x, tx, y, ty = propagate(
        state, z0, z1,
        np.array([1.0, 0.0]), np.array([0.0, 1.0]),
        np.zeros(2), np.zeros(2),
        include_spherical_aberration=False,
        include_hexapole=False,
    )
    return np.array(
        [
            [x[-1, 0] + 1j * y[-1, 0], x[-1, 1] + 1j * y[-1, 1]],
            [
                tx[-1, 0] + 1j * ty[-1, 0],
                tx[-1, 1] + 1j * ty[-1, 1],
            ],
        ],
        dtype=np.complex128,
    )
