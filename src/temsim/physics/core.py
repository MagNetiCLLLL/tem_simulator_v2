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
    active_mapped_providers, active_vector_providers, freeze_vector_provider,
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
    kinetic_energy_ev: np.ndarray | None = None

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
        if self.kinetic_energy_ev is not None:
            energy = np.asarray(self.kinetic_energy_ev, dtype=np.float64)
            if energy.shape != shape or np.any(np.isinf(energy)) or np.any(energy <= 0.):
                raise ValueError("Checkpoint kinetic energies must match rays and be positive or NaN")
            frozen = np.frombuffer(energy.tobytes(order="C"), dtype=energy.dtype).reshape(energy.shape)
            object.__setattr__(self, "kinetic_energy_ev", frozen)


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
    dipole_bx_t: np.ndarray
    dipole_by_t: np.ndarray
    reference_momentum_kg_m_s: float
    mapped_fields: tuple = ()
    electric_field: object | None = None
    electric_field_identity: str | None = None
    electric_reference_invariant_ev: float | None = None
    posed_spherical_kicks: tuple = ()


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
    sx, sy = multipole_focusing_fields(z, state, exclude_mapped_keys=exclude_mapped_keys)
    from temsim.physics.instrument_magnetic import gun_paraxial_fields
    _, _, gun_sx, gun_sy, _ = gun_paraxial_fields(state, z)
    sx += gun_sx
    sy += gun_sy
    return magnetic, sx, sy


def multipole_focusing_fields(z, state, *, exclude_mapped_keys=()):
    """Shared continuous quadrupole coefficients, without querying round lenses."""
    z = np.asarray(z, float)
    sx, sy = np.zeros_like(z), np.zeros_like(z)
    for stig in state.stigmators:
        if not stig.enabled: continue
        if getattr(stig, "key", None) in exclude_mapped_keys: continue
        qx, qy, _ = stig.quadrupole_tensor_m2(z)
        mask = _support_mask(z, stig)
        sx += np.where(mask, qx, 0.0)
        sy += np.where(mask, qy, 0.0)
    for component in getattr(state, "corrector_elements", []):
        if getattr(component, "key", None) in exclude_mapped_keys: continue
        if not getattr(component, "enabled", False):
            continue
        if not hasattr(component, "quadrupole_strength_m2"):
            continue
        q = component.quadrupole_strength_m2(z)
        mask = _support_mask(z, component)
        sx += np.where(mask, q, 0.0)
        sy -= np.where(mask, q, 0.0)
    return sx, sy

def skew_quadrupole_field(z, state, *, exclude_mapped_keys=()):
    """Off-diagonal focusing coefficient in the same frame as fields()."""
    z = np.asarray(z, float)
    skew = np.zeros_like(z)
    for stig in state.stigmators:
        if getattr(stig, "key", None) in exclude_mapped_keys: continue
        if stig.enabled:
            _, _, value = stig.quadrupole_tensor_m2(z)
            skew += np.where(_support_mask(z, stig), value, 0.0)
    from temsim.physics.instrument_magnetic import gun_paraxial_fields
    skew += gun_paraxial_fields(state, z)[4]
    return skew

def bz(z,state): return fields(z,state)[0]


def hexapole_field(z, state):
    """Return the summed normal hexapole coefficient for compatibility."""

    return hexapole_field_components(z, state)[0]


def hexapole_field_components(z, state, *, exclude_mapped_keys=()):
    """Sum continuous normal and skew hexapole coefficients."""

    z = np.asarray(z, float)
    normal = np.zeros_like(z)
    skew = np.zeros_like(z)
    if is_ideal(state):
        return normal, skew
    for component in getattr(state, "corrector_elements", []):
        if getattr(component, "key", None) in exclude_mapped_keys: continue
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
    from temsim.lens_pose import has_lens_pose
    for lens in getattr(state, "lenses", ()):
        if not bool(getattr(lens, "enabled", True)):
            continue
        if lens.key in mapped_keys:
            # The imported spatial field supplies its own ray aberrations.
            # Do not add the native Gaussian lens's calibrated Cs again.
            continue
        if has_lens_pose(state, lens.key):
            # The posed equivalent Cs impulse is applied in its lens frame.
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
    _diagnostic_dipole_fields=None,
):
    """Build the single global axial plan used by full and resumed traces."""

    events = tuple(events)
    electric_field = None
    electric_reference = None
    if getattr(state, "electron_gun", None) is not None:
        from temsim.physics.instrument_electric import capture_instrument_electric_field
        electric_field = capture_instrument_electric_field(state)
        gun = electric_field.gun_snapshot
        reference_point = np.array([[0., 0., float(gun.exit_plane_z_mm)*1e-3]])
        reference_phi = float(electric_field.potential_rise_v_at_global_positions(reference_point)[0])
        electric_reference = float(state.beam_voltage_kv)*1000.-reference_phi
    from temsim.physics.instrument_magnetic import (
        column_dipole_fields, gun_magnetic_support_edges_mm, gun_paraxial_fields,
    )
    # Only consume declared magnetic steering events. Electrostatic blanking,
    # calibration probes and other independently supplied actions remain kicks.
    remaining_events = list(events)
    dipoles = []
    # A local drive Jacobian supplies frozen +/- perturbations of these same
    # physical providers. Matching still consumes each event exactly once, so
    # the diagnostic cannot also apply its coil as a centre-plane kick.
    captured_dipoles = (column_dipole_fields(state) if _diagnostic_dipole_fields is None
                        else tuple(_diagnostic_dipole_fields)) if events else ()
    for field in captured_dipoles:
        event = (field.event_z_mm, field.event_dx_rad, field.event_dy_rad)
        for index, supplied in enumerate(remaining_events):
            if tuple(supplied) == event:
                remaining_events.pop(index)
                dipoles.append(field)
                break
        else:
            if getattr(field, "registration", None) is not None and any(float(row[0]) == field.event_z_mm for row in remaining_events):
                raise ValueError(f"{field.key}: posed deflector event override does not match the captured physical coil drive")
    events = tuple(remaining_events)
    save_z_mm = tuple(save_z_mm)
    checkpoint_z_mm = tuple(checkpoint_z_mm)
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
        freeze_vector_provider(provider)
        for provider in active_vector_providers(state)
        if not (reduce_image_maps and provider.lens_key in IMAGE_LENS_KEYS)
        and provider.field_support_mm()[0] <= float(z1)
        and provider.field_support_mm()[1] >= float(z0)
    )
    from temsim.physics.posed_column_fields import capture_posed_column_fields
    posed_columns = capture_posed_column_fields(state, include_hexapole=include_hexapole, dipoles=dipoles)
    mapped_fields += tuple(field for field in posed_columns
                           if field.field_support_mm[0] <= float(z1) and field.field_support_mm[1] >= float(z0))
    # A translated component outside this segment must not reappear at its
    # old axial location through the scalar channel. Exclude all placed
    # components, while only retaining actual overlapping vector providers.
    posed_keys = {field.component_key for field in posed_columns}
    mapped_keys = {item.lens_key for item in mapped_fields} | posed_keys
    posed_coil_keys = {field.lens_key for field in posed_columns if field.event_z_mm is not None}
    dipoles = [field for field in dipoles if field.key not in posed_coil_keys]
    from temsim.physics.posed_aberrations import posed_spherical_kicks
    posed_cs = posed_spherical_kicks(state, float(z0), float(z1)) if include_spherical_aberration else ()
    exact_z_mm = [event.z_mm for event in (*image_lens_events, *posed_cs)]
    if electric_field is not None:
        electric_z = getattr(electric_field.base_field, "z", ())
        exact_z_mm.extend(float(value)*1e3 for value in electric_z if z0 < float(value)*1e3 < z1)
    gun_edges_mm = gun_magnetic_support_edges_mm(state)
    gun_in_span = (bool(gun_edges_mm) and max(gun_edges_mm) > float(z0)
                   and min(gun_edges_mm) < float(z1))
    if gun_in_span:
        exact_z_mm.extend(gun_edges_mm)
    for field in dipoles:
        exact_z_mm.extend((field.lower_m * 1e3, field.upper_m * 1e3))
    if particle_medium:
        from temsim.physics.residual_medium import medium_grid_nodes
        exact_z_mm.extend(medium_grid_nodes(state, z0, z1, medium_energy_ev))
        for segment in getattr(getattr(state, "_resolved_assembly", None), "vacuum_bore_segments", ()):
            exact_z_mm.extend(v for v in (segment.start_z_mm, segment.end_z_mm) if z0 < v < z1)
        exact_z_mm.extend(float(p.z_mm) for p in getattr(state, "recording_planes", ())
                          if z0 <= float(p.z_mm) <= z1)
        save_z_mm += tuple(float(v) for v in exact_z_mm)
    for item in mapped_fields:
        lower, upper = item.field_support_mm
        lower, upper = max(lower, float(z0)), min(upper, float(z1))
        if upper <= lower:
            continue
        local_step = item.maximum_step_mm(requested_step, electron(state)[1])
        exact_z_mm.extend(np.linspace(lower, upper,
            max(1, int(math.ceil((upper-lower)/local_step)))+1))
    exact_z_mm.extend(float(value) for value in save_z_mm)
    # A resumable state belongs to its requested plane, not the nearest
    # plotting/integration node (which can even merge distinct checkpoints).
    exact_z_mm.extend(float(value) for value in checkpoint_z_mm)
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
    # A uniform finite coil is discontinuous at its ends. All three stage
    # values belong to this interval's interior, never a shared boundary node.
    # Since every coil edge is an exact node, its midpoint gives the exact
    # constant field throughout each open interval, including overlapping coils.
    dipole_points = np.zeros((len(midpoint_z_mm), 3), dtype=np.float64)
    dipole_points[:, 2] = midpoint_z_mm * 1e-3
    dipole_values = np.zeros_like(dipole_points)
    for field in dipoles:
        dipole_values += field.field_at_global_positions_t(dipole_points)
    dipole_bx = np.repeat(dipole_values[:, 0], 3)
    dipole_by = np.repeat(dipole_values[:, 1], 3)
    # Smooth gun coils can extend past the nominal gun exit. Sample the
    # original providers at the same one-sided RK stages, preserving their
    # soft edges rather than converting them to another centre-plane kick.
    if gun_in_span:
        gun_stages_mm = np.column_stack((
            np.nextafter(zfull[:-1], zfull[1:]), midpoint_z_mm,
            np.nextafter(zfull[1:], zfull[:-1]),
        )).reshape(-1)
        gun_bx, gun_by, _, _, _ = gun_paraxial_fields(state, gun_stages_mm)
        dipole_bx += gun_bx
        dipole_by += gun_by
    reference_momentum = electron(state)[1]
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
        sxy = skew_quadrupole_field(zfull, state, **field_options)
        midpoint_sxy = skew_quadrupole_field(midpoint_z_mm, state, **field_options)
    finally:
        if had_equivalent_propagation_flag:
            state._using_equivalent_image_propagation = (
                previous_equivalent_propagation_flag
            )
        else:
            delattr(state, "_using_equivalent_image_propagation")
    if include_hexapole:
        hex_normal, hex_skew = hexapole_field_components(zfull, state, exclude_mapped_keys=mapped_keys)
        midpoint_hex_normal, midpoint_hex_skew = hexapole_field_components(
            midpoint_z_mm, state, exclude_mapped_keys=mapped_keys
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
        'canonical-rk4-shared-electric-magnetic-energy-v7-posed-cs',
        mode_key(state),
        state.vacuum_map.signature() if particle_medium else "optical-map-no-medium",
    ))
    digest.update(solver_signature.encode("utf-8"))
    if electric_field is not None:
        digest.update(str(electric_field.numerical_identity).encode("utf-8"))
        digest.update(repr(electric_reference).encode("ascii"))
        if electric_field.numerical_identity is None:
            # An unidentified provider can execute, but never establish cache
            # equivalence just because two absent identities compare equal.
            from uuid import uuid4
            digest.update(uuid4().bytes)
    for item in mapped_fields:
        digest.update(item.fingerprint.encode("ascii"))
    digest.update(repr(posed_cs).encode("utf-8"))
    for values in (
        zfull, step_m, magnetic, sx, sy, sxy, hex_normal, hex_skew,
        midpoint_magnetic, midpoint_sx, midpoint_sy, midpoint_sxy,
        midpoint_hex_normal, midpoint_hex_skew,
        cs_kick, thin_power, thin_rotation, kickx, kicky, save,
        checkpoint_indices, dipole_bx, dipole_by, np.asarray((reference_momentum,)),
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
        dipole_bx_t=_frozen_array(dipole_bx),
        dipole_by_t=_frozen_array(dipole_by),
        reference_momentum_kg_m_s=reference_momentum,
        mapped_fields=mapped_fields,
        posed_spherical_kicks=posed_cs,
        electric_field=electric_field,
        electric_field_identity=None if electric_field is None else electric_field.numerical_identity,
        electric_reference_invariant_ev=electric_reference,
    )


def propagation_plan_common_prefix_nodes(previous, current):
    """Return count of leading nodes whose actual optical actions are equal."""

    if previous is None or current is None:
        return 0
    if any(plan.electric_field is not None and plan.electric_field_identity is None
           for plan in (previous, current)):
        return 0
    if previous.solver_signature != current.solver_signature:
        return 0
    if previous.reference_momentum_kg_m_s != current.reference_momentum_kg_m_s:
        return 0
    if (previous.electric_field_identity != current.electric_field_identity
            or previous.electric_reference_invariant_ev != current.electric_reference_invariant_ev):
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
        boundary = min(item.field_support_mm[0]
                       for item in (old,new) if item is not None)
        equal &= new_z[:count] < boundary
    old_cs = {item.lens_key: item for item in previous.posed_spherical_kicks}
    new_cs = {item.lens_key: item for item in current.posed_spherical_kicks}
    for key in old_cs.keys() | new_cs.keys():
        before, after = old_cs.get(key), new_cs.get(key)
        if before != after:
            boundary = min(item.z_mm for item in (before, after) if item is not None)
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
    for name in ("dipole_bx_t", "dipole_by_t"):
        old = getattr(previous, name).reshape(-1, 3)
        new = getattr(current, name).reshape(-1, 3)
        interval_count = min(count, len(old), len(new))
        equal[:interval_count] &= np.all(old[:interval_count] == new[:interval_count], axis=1)
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
    initial_kinetic_energy_ev=None, energy_output=None,
):
    """Execute a complete plan or resume it from an after-action checkpoint.

    With return_flight_times, append float64 saved times after the five ray
    history arrays and before checkpoints. Resuming passes the checkpoint's
    time row explicitly; absent upstream times remain NaN throughout.
    """

    from temsim.physics.optical_tuning import check_tuning_cancelled
    check_tuning_cancelled(state)
    if (plan.electric_field_identity is not None or plan.electric_reference_invariant_ev is not None) and plan.electric_field is None:
        raise ValueError("This archived plan needs its captured electric field reconstructed before execution")
    from temsim.physics.electrostatic_column_transport import active_electric_field
    electric_field = active_electric_field(plan)
    if electric_field is not None:
        # This kernel currently uploads its own immutable electric buffers;
        # an older magnetic-kernel residency receipt must not describe it.
        state._last_ray_device_receipt = None
    if electric_field is not None and int(start_index) > 0 and initial_kinetic_energy_ev is None:
        raise ValueError("Resuming an electric column plan requires the checkpoint's actual kinetic energy")

    start_index = int(start_index)
    if not 0 <= start_index < len(plan.z_mm):
        raise ValueError("Propagation-plan start index is out of range")
    zfull = np.asarray(plan.z_mm[start_index:], dtype=np.float64)
    from temsim.physics.posed_aberrations import kick_indices
    posed_cs = kick_indices(plan.posed_spherical_kicks, zfull,
                            include_initial=bool(include_initial_plane_kicks))
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
    actual_initial_energy = np.broadcast_to(np.asarray(
        (float(state.beam_voltage_kv)*1000.+np.asarray(0. if energy_offset_ev is None else energy_offset_ev))
        if initial_kinetic_energy_ev is None else initial_kinetic_energy_ev,
        dtype=np.float64), arrays[0].shape).copy()
    initial_phase_finite = np.all(np.isfinite(arrays), axis=0)
    invalid_energy = ~np.isfinite(actual_initial_energy) | (actual_initial_energy <= 0.)
    if np.any(invalid_energy & (initial_phase_finite | (not bool(defer_nonfinite_until_clipping)))):
        raise ValueError("Column entrance kinetic energies must be positive and finite")
    actual_initial_energy[invalid_energy] = np.nan
    actual_offsets = actual_initial_energy-float(state.beam_voltage_kv)*1000.
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
    # Constant-field specialisation retains the original compact magnetic
    # ABI. With E present, the dimensional canonical kernel recomputes p and
    # speed from the conserved potential invariant at every actual RK stage.
    optical_offsets = None if is_ideal(state) else actual_offsets
    if (electric_field is None and plan.electric_field is not None and is_ideal(state)
            and plan.electric_reference_invariant_ev is not None):
        # Exact constant-potential continuation still belongs to the same
        # fixed gun-exit reference; starting a new segment cannot reset it.
        potential = float(np.asarray(plan.electric_field.potential_rise_v_at_global_positions(
            np.asarray(((0., 0., float(zfull[0])*1e-3),))))[0])
        optical_energy = float(plan.electric_reference_invariant_ev)+potential
        if not np.isfinite(optical_energy) or optical_energy <= 0.:
            raise ValueError("Ideal reference kinetic energy must remain positive and finite")
        optical_offsets = np.full(arrays[0].size, optical_energy-float(state.beam_voltage_kv)*1000.)
    momentum_at_start = momentum_profile(state, zfull[:1], optical_offsets)
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
        np.ascontiguousarray(plan.dipole_bx_t[3*start_index:]),
        np.ascontiguousarray(plan.dipole_by_t[3*start_index:]),
        np.asarray((plan.reference_momentum_kg_m_s,), dtype=np.float64),
    )
    timing = {}
    if return_flight_times:
        initial = (np.full(arrays[0].size, np.nan, dtype=np.float64)
                   if initial_time_s is None else np.asarray(initial_time_s, dtype=np.float64))
        if initial.shape != (arrays[0].size,) or np.any(np.isinf(initial)) or np.any(initial < 0.):
            raise ValueError("Initial flight times must be one non-negative or NaN value per ray")
        # Used only by the proven constant-potential specialisation. The
        # electric kernel integrates time using the varying actual energy.
        actual_momentum = np.asarray(momentum_profile(state, zfull[:1], actual_offsets)[0])
        actual_momentum = np.broadcast_to(actual_momentum, (arrays[0].size,))
        inverse_speed = np.sqrt(M*M+(actual_momentum/C)**2)/actual_momentum
        timing = dict(initial_time_s=np.ascontiguousarray(initial),
                      inverse_speed=np.ascontiguousarray(inverse_speed))
    policy = normalise_backend(getattr(state, "acceleration_backend", "Auto")).lower().replace(" ", "_")
    tuning = bool(getattr(state, "_optical_tuning", False) or getattr(state, "_particle_tuning", False))
    workload = None
    if electric_field is None and medium_transport is None and not plan.mapped_fields and not posed_cs and not tuning:
        from temsim.physics.ray_device_cache import STAGE_COSTS, measured_workload
        workload = measured_workload((*inputs, *timing.values()) if timing else inputs)
        if (policy == "auto" and backend != BACKEND_CUDA
                and getattr(state, "acceleration_enabled", False)):
            # CPU cost history must not override Auto's available-GPU choice.
            eligible = [BACKEND_CPU]
            if NUMBA_AVAILABLE:
                eligible.append(BACKEND_NUMBA)
            backend, measured_reason = STAGE_COSTS.choose(workload, eligible, backend)
            fallback_reason = "; ".join(filter(None, (fallback_reason, measured_reason))) or None
    transport_started = perf_counter()
    from temsim.physics.transport_progress import report_transport_progress
    report_transport_progress(
        f"Column | {backend} | {arrays[0].size:,} electrons | "
        f"Z {zfull[0]:.3f} to {zfull[-1]:.3f} mm | {len(zfull)-1:,} integration steps")
    retried = False
    if policy == "require_gpu" and (medium_transport is not None or plan.mapped_fields or posed_cs):
        raise GPUExecutionError("unsupported_stage", "Requested column transport requires the existing CPU vector-field or residual-medium solver")
    if electric_field is not None:
        from temsim.physics.electrostatic_column_transport import electrostatic_column_rk4
        electric_options = dict(z_mm=zfull, electric_field=electric_field,
            initial_kinetic_energy_ev=actual_initial_energy,
            optical_reference_invariant_ev=(plan.electric_reference_invariant_ev if is_ideal(state) else None),
            initial_time_s=timing.get("initial_time_s"), policy=policy,
            mapped_fields=plan.mapped_fields, step_operator=medium_transport,
            posed_spherical_kicks=posed_cs,
            defer_nonfinite_until_clipping=defer_nonfinite_until_clipping,
            serial=bool(tuning or arrays[0].size <= 16),
            cancel_check=lambda: check_tuning_cancelled(state))
        # A local Jacobian has only nine particles, but traversing thousands
        # of E-field nodes is still expensive in Python. Include axial work,
        # not only ray count, while preserving explicit CPU/disabled choices.
        electric_work = max(0, len(zfull)-1)*arrays[0].size
        if (NUMBA_AVAILABLE and getattr(state, "acceleration_enabled", False)
                and policy == "auto" and backend == BACKEND_CPU
                and (tuning or electric_work >= 4096)):
            backend = BACKEND_NUMBA
        try:
            outputs, backend, electric_reason = electrostatic_column_rk4(inputs, backend=backend, **electric_options)
            fallback_reason = electric_reason or fallback_reason
        except Exception as exc:
            if backend != BACKEND_CUDA:
                raise
            fallback_reason = gpu_retry_reason(exc, policy, stage="electric_column_transport")
            retried = True
            outputs, backend, _ = electrostatic_column_rk4(inputs,
                backend=BACKEND_NUMBA if NUMBA_AVAILABLE else BACKEND_CPU, **electric_options)
    elif medium_transport is not None and not plan.mapped_fields and not posed_cs:
        outputs = _vectorised_rk4(*inputs, step_operator=medium_transport, **timing)
        backend, fallback_reason = BACKEND_CPU, "classical residual-medium collisions between optical steps"
    elif plan.mapped_fields or posed_cs:
        from temsim.physics.vector_field_transport import vector_map_rk4
        backend, fallback_reason = BACKEND_CPU, "positioned magnetic vector-field RK4"
        outputs = vector_map_rk4(*inputs, z_mm=zfull, mapped_fields=plan.mapped_fields,
                                posed_spherical_kicks=posed_cs,
                                defer_nonfinite_until_clipping=defer_nonfinite_until_clipping,
                                step_operator=medium_transport, **timing)
    elif (tuning and NUMBA_AVAILABLE and backend == BACKEND_NUMBA
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
    if backend == BACKEND_CUDA and electric_field is None:
        from temsim.physics.ray_device_cache import last_device_receipt
        state._last_ray_device_receipt = last_device_receipt()
    _record_active_backend(state, backend, fallback_reason)
    from temsim.physics.backend_execution import record_backend
    record_backend("column", getattr(state, "acceleration_backend", "Auto"), backend,
                   reason=fallback_reason or "", retried=retried)
    X,TX,Y,TY,CX,CTX,CY,CTY=outputs[:8]
    saved_energy = (outputs[10] if electric_field is not None else
                    np.broadcast_to(actual_initial_energy, (len(save), len(actual_initial_energy))).copy())
    checkpoint_energy = (outputs[11] if electric_field is not None else
                         np.broadcast_to(actual_initial_energy, np.shape(CX)).copy())
    if energy_output is not None:
        energy_output.append(_frozen_array(saved_energy))
    checkpoints = PropagationCheckpoints(
        z_mm=_frozen_array(zfull[checkpoint_index]),
        x_m=_frozen_array(CX),
        tx_rad=_frozen_array(CTX),
        y_m=_frozen_array(CY),
        ty_rad=_frozen_array(CTY),
        flight_time_s=outputs[9] if return_flight_times else None,
        kinetic_energy_ev=checkpoint_energy,
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
    initial_kinetic_energy_ev=None, energy_output=None,
):
    entrance_energy_ev = (state.beam_voltage_kv*1000
                         + np.asarray(0 if energy_offset_ev is None else energy_offset_ev)
                         if initial_kinetic_energy_ev is None
                         else np.asarray(initial_kinetic_energy_ev))
    plan = build_propagation_plan(
        state,z0,z1,events,
        include_spherical_aberration=include_spherical_aberration,
        include_hexapole=include_hexapole,
        save_z_mm=save_z_mm,
        checkpoint_z_mm=checkpoint_z_mm,
        maximum_step_mm=maximum_step_mm,
        particle_medium=particle_medium,
        medium_energy_ev=entrance_energy_ev,
    )
    if particle_medium and len(plan.z_mm) > state.vacuum_map.max_transport_nodes:
        raise ValueError("Particle / vacuum integration exceeds the configured node budget")
    transport = None
    if particle_medium and state.vacuum_map.enabled:
        from temsim.physics.residual_medium import ColumnMediumTransport
        transport = ColumnMediumTransport(state, plan, len(x),
            entrance_energy_ev,
            alive=medium_alive, stream=medium_stream)
        if medium_output is not None:
            medium_output.append(transport)
    result = execute_propagation_plan(
        state,plan,x,tx,y,ty,energy_offset_ev,
        include_initial_plane_kicks=include_initial_plane_kicks,
        defer_nonfinite_until_clipping=defer_nonfinite_until_clipping,
        medium_transport=transport,
        initial_time_s=initial_time_s, return_flight_times=return_flight_times,
        initial_kinetic_energy_ev=initial_kinetic_energy_ev, energy_output=energy_output,
    )
    return result if return_checkpoints else result[:6 if return_flight_times else 5]

def transfer(state,z0,z1):
    if active_vector_providers(state) or getattr(state, "electron_gun", None) is not None:
        from temsim.physics.first_order import trace_transverse_transfer
        matrix = trace_transverse_transfer(state, z0, z1).matrix
        return matrix[np.ix_((0, 2), (0, 2))]
    checkpoints=propagate(
        state,z0,z1,
        np.array([0.,1.,0.]),np.array([0.,0.,1.]),np.zeros(3),np.zeros(3),
        include_spherical_aberration=False,include_hexapole=False,
        checkpoint_z_mm=(z1,),return_checkpoints=True,
    )[-1]
    # A finite gun magnet may extend into this span and displace the origin.
    # The transfer matrix is the linear part, with that affine orbit removed.
    x,tx=checkpoints.x_m[-1],checkpoints.tx_rad[-1]
    return np.array([x[1:]-x[0],tx[1:]-tx[0]],float)


def complex_transfer(state, z0, z1):
    """Return the first-order, Larmor-coupled transfer in complex form."""

    if active_vector_providers(state) or getattr(state, "electron_gun", None) is not None:
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

    checkpoints = propagate(
        state, z0, z1,
        np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]),
        np.zeros(3), np.zeros(3),
        include_spherical_aberration=False,
        include_hexapole=False,
        checkpoint_z_mm=(z1,), return_checkpoints=True,
    )[-1]
    position = checkpoints.x_m[-1] + 1j*checkpoints.y_m[-1]
    slope = checkpoints.tx_rad[-1] + 1j*checkpoints.ty_rad[-1]
    return np.asarray([position[1:]-position[0], slope[1:]-slope[0]],
                      dtype=np.complex128)
