"""Cached first-order image conjugates of any supported straight-column plane.

This is a nominal zero-loss optical observer, not a particle or wave restart.
The atlas uses the same captured fields and canonical observer as the specimen
diagnostic. Common-source maps have mechanical output slopes. Their quotient
therefore maps mechanical coordinates at the new reference to mechanical
coordinates at the target. The local source gauge is then restored so A uses
selected-reference canonical momentum; this source shear leaves B unchanged.
Image conjugacy tests the whole 2x2 B block, not det(B) or a beam crossover.

Positions are metres, mechanical slopes radians, and public Z values mm.
Cached position maps are interpolated with their actual slope derivatives.
A coarser Hermite interpolant supplies an interpolation error estimate; this
is a resolution diagnostic, not a rigorous bound on physical/model error.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline
from scipy.optimize import minimize_scalar

from temsim.cpu_resources import NumericalJobCancelled, numerical_job
from temsim.gui.ray_extent_data import completed_ray_extent
from temsim.optics.direct_alignment import (canonical_transfers, _canonical_source_basis,
    _active_column_electric_field, _column_reference_momentum)
from temsim.physics.core import fields, E
from temsim.physics.first_order import linear_map_properties
from temsim.physics.scan_geometry import IMAGE_CONJUGACY_TOLERANCE_M_PER_RAD
from temsim.physics.selected_plane import _observer_state


_LENGTH_M = 1.0e-3  # only a conditioning scale; it does not change the optics
_MAXIMUM_CONDITION = 1.0e12
_MAXIMUM_NODES = 24001
_MAXIMUM_CANDIDATES = 96
_SELF_EXCLUSION_MM = 1.0e-5


def _check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise NumericalJobCancelled("Conjugate-plane search superseded or cancelled")


def _readonly(values):
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _scaled_maps(matrices):
    # x/L, y/L, theta_x, theta_y on both ends, avoiding condition numbers
    # that depend on a caller spelling length in metres instead of mm.
    scale = np.asarray((1.0/_LENGTH_M, 1.0/_LENGTH_M, 1.0, 1.0))
    return np.asarray(matrices)*scale[:, None]/scale[None, :]


def _physical_map(matrix):
    scale = np.asarray((1.0/_LENGTH_M, 1.0/_LENGTH_M, 1.0, 1.0))
    return np.asarray(matrix)/scale[:, None]*scale[None, :]


@dataclass(frozen=True, slots=True)
class ConjugateCandidate:
    z_mm: float
    kind: str
    residual_m_per_rad: float
    magnifications: tuple[float, float]
    rotation_deg: float | None
    mirrored: bool
    detail: str


@dataclass(frozen=True, slots=True)
class ConjugateSearch:
    reference_z_mm: float
    candidates: tuple[ConjugateCandidate, ...]
    detail: str
    lower_z_mm: float
    upper_z_mm: float


@dataclass(frozen=True, slots=True)
class ConjugateAtlas:
    """Small immutable maps; result identity/invalidation belongs to the caller.

    ``matrices`` must share the same input basis and reference orbit. Output
    coordinates must be column positions and mechanical dx/dz, dy/dz slopes.
    The constructor also admits analytic maps for independent verification.
    """
    z_mm: np.ndarray
    matrices: np.ndarray
    detail: str = "Nominal zero-loss, paraxial optical map; no material or clipping qualification."
    image_tolerance_m_per_rad: float = IMAGE_CONJUGACY_TOLERANCE_M_PER_RAD
    source_g_m1: np.ndarray | None = None
    _fine: object = field(init=False, repr=False, compare=False)
    _coarse: object = field(init=False, repr=False, compare=False)
    _scaled: np.ndarray = field(init=False, repr=False, compare=False)
    _g: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        z = np.asarray(self.z_mm, dtype=float)
        matrices = np.asarray(self.matrices, dtype=float)
        if (z.ndim != 1 or len(z) < 3 or len(z) > _MAXIMUM_NODES
                or not np.all(np.isfinite(z)) or np.any(np.diff(z) <= 0.0)):
            raise ValueError("Conjugate atlas requires 3–24001 finite, increasing Z nodes")
        if matrices.shape != (len(z), 4, 4) or not np.all(np.isfinite(matrices)):
            raise ValueError("Conjugate atlas requires finite 4x4 maps at every node")
        if (not math.isfinite(self.image_tolerance_m_per_rad)
                or self.image_tolerance_m_per_rad <= 0.0):
            raise ValueError("Image-conjugacy tolerance must be positive and finite")
        # Scaled position derivative d(x/L)/dz_mm = theta * 1e-3/L.
        scaled = _scaled_maps(matrices)
        derivative = scaled[:, 2:, :]*(1e-3/_LENGTH_M)
        fine = CubicHermiteSpline(z, scaled[:, :2, :], derivative, axis=0,
                                 extrapolate=False)
        coarse_indices = np.unique(np.r_[np.arange(0, len(z), 2), len(z)-1])
        coarse = CubicHermiteSpline(z[coarse_indices], scaled[coarse_indices, :2, :],
            derivative[coarse_indices], axis=0, extrapolate=False)
        object.__setattr__(self, "z_mm", _readonly(z))
        object.__setattr__(self, "matrices", _readonly(matrices))
        object.__setattr__(self, "_scaled", _readonly(scaled))
        object.__setattr__(self, "_fine", fine)
        object.__setattr__(self, "_coarse", coarse)
        g = np.zeros(len(z)) if self.source_g_m1 is None else np.asarray(self.source_g_m1, dtype=float)
        if g.shape != z.shape or not np.all(np.isfinite(g)):
            raise ValueError("Canonical source gauge requires one finite Larmor rate per atlas node")
        object.__setattr__(self, "source_g_m1", _readonly(g))
        object.__setattr__(self, "_g", CubicSpline(z, g, extrapolate=False))

    @property
    def lower_z_mm(self):
        return float(self.z_mm[0])

    @property
    def upper_z_mm(self):
        return float(self.z_mm[-1])

    def _at(self, z, *, coarse=False):
        spline = self._coarse if coarse else self._fine
        value = np.concatenate((spline(z), spline(z, 1)*(_LENGTH_M/1e-3)), axis=-2)
        return value

    def transfer_matrix(self, reference_z_mm, target_z_mm):
        """Selected-plane canonical input map, with mechanical target output.

        An upstream map expresses a reciprocal optical relationship. It does
        not transport actual electrons backwards or undo interception/scatter.
        """
        for z in (reference_z_mm, target_z_mm):
            if not math.isfinite(z) or not self.lower_z_mm <= z <= self.upper_z_mm:
                raise ValueError("Requested reference/target is outside the supported atlas")
        reference = self._at(reference_z_mm)
        _require_conditioned(reference)
        basis = _scaled_maps(_canonical_source_basis(float(self._g(reference_z_mm))))
        return _physical_map(self._at(target_z_mm) @ np.linalg.solve(reference, basis))


def _require_conditioned(matrix):
    condition = float(np.linalg.cond(matrix))
    if not math.isfinite(condition) or condition > _MAXIMUM_CONDITION:
        raise ValueError("Reference optical map is ill-conditioned; conjugate positions are unavailable")


def build_conjugate_atlas(result, *, cancelled=None) -> ConjugateAtlas:
    """Build once per captured result; changing selected Z never retraces fields."""
    _check_cancelled(cancelled)
    snapshot = getattr(result, "state_snapshot", None)
    extent = completed_ray_extent(result)
    start, stop = extent.get("start_z_mm"), extent.get("completed_z_mm")
    if snapshot is None or start is None or stop is None:
        raise ValueError("No captured optical state with verified executed Z range is available")
    start, stop = float(start), float(stop)
    gun = getattr(snapshot, "electron_gun", None)
    if gun is not None:
        exit_z = float(gun.exit_plane_z_mm)
        if not math.isfinite(exit_z):
            raise ValueError("Captured gun exit is not finite")
        start = max(start, exit_z)
    if bool(getattr(snapshot, "energy_filter_installed", False)):
        entrance = float(snapshot.energy_filter.entrance_z_mm)
        if not math.isfinite(entrance):
            raise ValueError("Captured energy-filter entrance is not finite")
        stop = min(stop, float(np.nextafter(entrance, -math.inf)))
    if not math.isfinite(start) or not math.isfinite(stop) or stop <= start:
        raise ValueError("No executed paraxial straight-column interval after the gun exit is available")
    state = _observer_state(snapshot)
    # The captured ray run's backend receipt is not evidence of this observer's
    # backend. Preserve the global preference, but start a new local receipt.
    state._active_backends_used = set()
    if not math.isfinite(float(state.beam_voltage_kv)) or state.beam_voltage_kv <= 0.0:
        raise ValueError("Captured nominal beam energy must be positive")
    # Quarter-mm storage is separate from the <=0.025-mm optical integration.
    # Even the full 3-m column needs only ~1.5 MB for the stored 4x4 matrices.
    count = max(3, int(math.ceil((stop-start)/0.25))+1)
    if count > _MAXIMUM_NODES:
        raise ValueError("Conjugate atlas exceeds its bounded 6-m straight-column domain")
    z = np.linspace(start, stop, count)
    # Resolve unusually narrow installed field profiles locally, without
    # spending the entire column budget on the smallest lens half-width.
    local_nodes = []
    for lens in getattr(state, "lenses", ()):
        centre, half_width = getattr(lens, "z_mm", None), getattr(lens, "a_mm", None)
        if not getattr(lens, "enabled", True) or centre is None or half_width is None:
            continue
        centre, half_width = float(centre), float(half_width)
        if not math.isfinite(half_width) or half_width <= 0.0:
            continue
        left, right = max(start, centre-5*half_width), min(stop, centre+5*half_width)
        if right > left and half_width/12.0 < 0.25:
            local_nodes.extend(np.linspace(left, right, int(math.ceil((right-left)/(half_width/12.0)))+1))
    if local_nodes:
        z = np.unique(np.r_[z, local_nodes])
    reference_z = float(getattr(getattr(state, "sample", None), "z_mm", math.nan))
    if start < reference_z < stop:
        z = np.unique(np.r_[z, reference_z])
    if len(z) > _MAXIMUM_NODES:
        raise ValueError("Conjugate atlas exceeds its node budget")

    def solver_cancelled():
        _check_cancelled(cancelled)
        return False

    state._tuning_cancelled = solver_cancelled
    with numerical_job(1, cancelled=cancelled):
        _check_cancelled(cancelled)
        maps = canonical_transfers(state, start, z)
        electric_field = _active_column_electric_field(state, start, stop)
        magnetic = fields(z, state)[0]
        if electric_field is None:
            momenta = _column_reference_momentum(state, start)
        else:
            momenta = np.asarray([_column_reference_momentum(state, value, electric_field) for value in z])
        source_g = -E*np.asarray(magnetic)/(2.0*momenta)
    _check_cancelled(cancelled)
    matrices = np.stack([maps[float(value)].matrix for value in z])
    backends = sorted(str(value) for value in getattr(state, "_active_backends_used", ()))
    backend_text = ", ".join(backends) if backends else "CPU first-order matrices"
    detail = (f"Captured nominal zero-loss paraxial optics | {start:.9g}–{stop:.9g} mm | "
        f"{len(z)} cached nodes | {backend_text}. "
        "Uses installed lens, stigmator and deflector fields. Near-tip/gun propagation and curved "
        "energy-filter coordinates are excluded. Conjugacy does not guarantee particle transmission, "
        "absence of aberrations or material coherence. Upstream entries describe reciprocal optical "
        "relationships, not backward electron propagation or virtual-image continuations.")
    return ConjugateAtlas(z, matrices, detail, source_g_m1=source_g)


def _minimum_indices(values):
    values = np.asarray(values)
    # Strict on at least one side: a constant nonzero map has no isolated focus.
    return np.flatnonzero((values[1:-1] <= values[:-2])
        & (values[1:-1] <= values[2:])
        & ((values[1:-1] < values[:-2]) | (values[1:-1] < values[2:])))+1


def find_conjugate_planes(atlas: ConjugateAtlas, reference_z_mm, *, cancelled=None) -> ConjugateSearch:
    """Find isolated 2-D image minima and distinct 1-D foci without field work.

    Classification is numerical: ``image`` requires both singular values of B
    plus the interpolation estimate to fit the common image tolerance.
    ``line_focus`` means only one transverse combination is focused. Other
    isolated closest approaches are explicitly ``approximate``. The reference
    plane's identity map is excluded, even if it lies between stored nodes.
    """
    _check_cancelled(cancelled)
    reference_z = float(reference_z_mm)
    if (not math.isfinite(reference_z)
            or not atlas.lower_z_mm <= reference_z <= atlas.upper_z_mm):
        raise ValueError("Selected reference is outside the supported post-gun straight-column range")
    reference = atlas._at(reference_z)
    _require_conditioned(reference)
    condition = float(np.linalg.cond(reference))
    source_basis = _scaled_maps(_canonical_source_basis(float(atlas._g(reference_z))))
    inverse = np.linalg.solve(reference, source_basis)
    solve_residual = float(np.linalg.norm(reference @ inverse-source_basis, ord=2)) / max(
        float(np.linalg.norm(source_basis, ord=2)), np.finfo(float).tiny)
    coarse_reference = atlas._at(reference_z, coarse=True)
    _require_conditioned(coarse_reference)
    coarse_inverse = np.linalg.solve(coarse_reference, source_basis)
    sampled = atlas._scaled @ inverse
    singular = np.linalg.svd(sampled[:, :2, 2:]*_LENGTH_M, compute_uv=False)
    tolerance = atlas.image_tolerance_m_per_rad
    nodes = atlas.z_mm
    queries = [(int(index), 0) for index in _minimum_indices(singular[:, 0])]
    queries += [(int(index), 1) for index in _minimum_indices(singular[:, 1])]
    # A real conjugate can coincide with the supported-domain boundary.
    for index in (0, len(nodes)-1):
        for axis in (0, 1):
            # Merely being within the broad image tolerance at an endpoint
            # does not establish a focus (notably in the reference's drift).
            # Admit boundary roots only at near-zero numerical residual.
            if singular[index, axis] <= min(1e-10, tolerance*1e-4):
                queries.append((index, axis))
    # Bounded input has a finite lens count, but pathological ripple must not
    # turn slider motion into unbounded optimizations.
    truncated = len(queries) > _MAXIMUM_CANDIDATES*4
    queries = sorted(queries, key=lambda pair: singular[pair[0], pair[1]])[:_MAXIMUM_CANDIDATES*4]
    raw = []

    def blocks(z):
        matrix = atlas._at(z) @ inverse
        return matrix[:2, :2], matrix[:2, 2:]*_LENGTH_M

    def residual(z, axis):
        return float(np.linalg.svd(blocks(z)[1], compute_uv=False)[axis])

    for index, axis in queries:
        _check_cancelled(cancelled)
        if index in (0, len(nodes)-1):
            z = float(nodes[index])
        else:
            left, right = float(nodes[index-1]), float(nodes[index+1])
            midpoint, radius = 0.5*(left+right), 0.5*(right-left)
            optimized = minimize_scalar(lambda value: residual(midpoint+radius*value, axis),
                bounds=(-1.0, 1.0), method="bounded",
                options={"xatol": 1e-10, "maxiter": 64})
            if not optimized.success or not math.isfinite(optimized.fun):
                continue
            z = float(midpoint+radius*optimized.x)
        if abs(z-reference_z) <= _SELF_EXCLUSION_MM:
            continue
        a, b = blocks(z)
        b_singular = np.linalg.svd(b, compute_uv=False)
        # Compare two cached spatial resolutions, including re-basing error.
        coarse_b = (atlas._at(z, coarse=True) @ coarse_inverse)[:2, 2:]*_LENGTH_M
        interpolation_error = float(np.linalg.norm(b-coarse_b, ord=2))
        # Roundoff in the rebasing step is separately exposed as uncertainty.
        # Multiplication can subtract large prefix-map terms even when final B
        # is tiny. Estimate that cancellation from absolute products, not B
        # alone; include sensitivity to the reference solve's backward error.
        target_position = atlas._at(z)[:2]
        product_scale = float(np.linalg.norm(
            np.abs(target_position) @ np.abs(inverse[:, 2:]), ord=2))*_LENGTH_M
        roundoff = (8.0*np.finfo(float).eps*product_scale
            + condition*(8.0*np.finfo(float).eps+solve_residual)
            * max(float(b_singular[0]), _LENGTH_M))
        uncertainty = interpolation_error+roundoff
        properties = linear_map_properties(a)
        if b_singular[0]+uncertainty <= tolerance and properties.rank == 2:
            kind = "image"
        elif b_singular[1]+uncertainty <= tolerance and b_singular[0] > tolerance:
            kind = "line_focus"
        elif axis == 0:
            kind = "approximate"
        else:
            continue
        detail = (f"B singular values {b_singular[0]:.6g}, {b_singular[1]:.6g} m/rad; "
            f"cached-resolution/roundoff estimate {uncertainty:.3g} m/rad; "
            f"image tolerance {tolerance:.3g} m/rad. "
            + ("Only one transverse combination is focused; not a two-dimensional image."
               if kind == "line_focus" else
               "Closest local approach; the full image condition is not resolved within tolerance."
               if kind == "approximate" else
               "Two-dimensional first-order image condition is within the stated tolerance."))
        if properties.rank < 2:
            detail += (f" Position map A has rank {properties.rank}; a nonsingular "
                       "two-dimensional image mapping is not established.")
        raw.append(ConjugateCandidate(z, kind, float(b_singular[0]),
            properties.singular_values, properties.orientation_deg,
            properties.mirrored, detail))
    # The two singular-value searches can land on the same round-lens image.
    # Merge only numerical duplicates, not nearby astigmatic focal planes.
    raw.sort(key=lambda item: (item.z_mm, item.residual_m_per_rad))
    candidates = []
    for candidate in raw:
        same_basin = False
        if candidates and candidate.kind == candidates[-1].kind == "image":
            previous = candidates[-1]
            nearest = int(np.clip(np.searchsorted(nodes, candidate.z_mm), 1, len(nodes)-1))
            same_basin = (candidate.z_mm-previous.z_mm < nodes[nearest]-nodes[nearest-1]
                and residual(0.5*(candidate.z_mm+previous.z_mm), 0) <= tolerance)
        if candidates and (abs(candidate.z_mm-candidates[-1].z_mm) < 1e-4 or same_basin):
            previous = candidates[-1]
            rank = {"image": 0, "line_focus": 1, "approximate": 2}
            if (rank[candidate.kind], candidate.residual_m_per_rad) < (
                    rank[previous.kind], previous.residual_m_per_rad):
                candidates[-1] = candidate
        else:
            candidates.append(candidate)
    if len(candidates) > _MAXIMUM_CANDIDATES:
        candidates = sorted(candidates, key=lambda item: item.residual_m_per_rad)[:_MAXIMUM_CANDIDATES]
        candidates.sort(key=lambda item: item.z_mm)
        truncated = True
    detail = atlas.detail + " Search excludes the reference plane itself; a lens need not add an image plane."
    if not candidates:
        detail += " No other isolated image/line-focus minimum was found in the supported interval."
    if truncated:
        detail += " Candidate budget reached; this is not an exhaustive list."
    _check_cancelled(cancelled)
    return ConjugateSearch(reference_z, tuple(candidates), detail,
                           atlas.lower_z_mm, atlas.upper_z_mm)
