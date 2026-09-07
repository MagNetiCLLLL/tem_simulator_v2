"""User-level coupled Direct Alignment controls.

These controls solve live first-order optics; they do not add field sources or
replace the editable low-level lens percentages.  Requested targets and their
provenance live in ``configs/operating_modes/catalog.toml``.  A solve is
transactional: an unreachable target never changes the microscope state.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import brentq, least_squares

from temsim.component_keys import (
    CONDENSER_LENS_2,
    CONDENSER_LENS_3,
    DIFFRACTION_LENS,
    INTERMEDIATE_LENS,
    OBJECTIVE_LENS,
    PROJECTOR_LENS_1,
    PROJECTOR_LENS_2,
    STEM_DIFFRACTION_REFERENCE_PLANE,
)
from temsim.operating_modes import (
    DirectAlignmentDefinition,
    direct_alignment_by_key,
)
from temsim.physics.beam_statistics import (
    TransverseBeamStatistics,
    transverse_beam_statistics,
)
from temsim.physics.aperture_clipping import clip_segment
from temsim.physics.column_wall import clip_column_wall
from temsim.physics.core import E, fields, propagate, interleaved_rk4_values
from temsim.physics.ray_integrator import _canonical_step_numba
from temsim.physics.first_order import (
    TransverseTransfer,
    trace_transverse_transfer,
    trace_transverse_transfers,
)
from temsim.physics.lens_field_provider import active_mapped_providers
from temsim.physics.recording_stop import tem_projection_reference_plane
from temsim.optics.equivalent_image_lenses import (
    equivalent_image_calibrations,
    equivalent_image_transfer_matrix,
    equivalent_image_maps_supported,
)
from temsim.optics.direct_alignment_precalibration import (
    interpolated_precalculated_seed,
    interpolated_nanoprobe_seed,
    precalculated_alignment_points,
)

try:
    from numba import njit
except Exception:  # pragma: no cover - Numba is a required project dependency.
    def njit(*_args, **_kwargs):
        def decorate(function):
            return function
        return decorate


NANOPROBE_CONVERGENCE = "nanoprobe_convergence"
MICROPROBE_ILLUMINATION = "microprobe_illumination"
IMAGE_MAGNIFICATION = "image_magnification"
DIFFRACTION_CAMERA_LENGTH = "diffraction_camera_length"

CONDENSER_KEYS = (CONDENSER_LENS_2, CONDENSER_LENS_3)
PROJECTOR_KEYS = (
    DIFFRACTION_LENS,
    INTERMEDIATE_LENS,
    PROJECTOR_LENS_1,
    PROJECTOR_LENS_2,
)
IMAGE_KEYS = (OBJECTIVE_LENS, *PROJECTOR_KEYS)


@dataclass(frozen=True, slots=True)
class DirectAlignmentMeasurement:
    key: str
    value: float
    unit: str
    constraint_value: float
    constraint_unit: str
    convergence_95_mrad: float | None = None
    convergence_99_mrad: float | None = None
    illumination_diameter_95_um: float | None = None
    relay_error_um: float | None = None
    diffraction_conjugacy_residual: float | None = None


@dataclass(frozen=True, slots=True)
class DirectAlignmentResult:
    key: str
    success: bool
    requested: float
    achieved: float
    unit: str
    constraint_value: float
    constraint_unit: str
    strengths: dict[str, float]
    iterations: int
    validation_step_mm: float
    numerical_spread: float
    message: str
    convergence_95_mrad: float | None = None
    convergence_99_mrad: float | None = None
    illumination_diameter_95_um: float | None = None
    relay_error_um: float | None = None
    diffraction_conjugacy_residual: float | None = None
    target_plane_key: str | None = None
    target_plane_z_mm: float | None = None
    field_calibration_statuses: tuple[str, ...] = ()
    candidate_strengths: dict[str, float] | None = None
    candidate_limit_fractions: dict[str, float] | None = None
    state_updates: dict[str, float] | None = None


def diffraction_reference_plane(state):
    """Return the detector-independent camera-length calibration plane.

    Camera length belongs to the coupled projector system and is calibrated at
    the TOML-owned main fluorescent-screen plane.  Axially ordered HAADF, DF
    and BF detectors keep their own physical Z and full transfer; their
    insertion/readout state must never choose or alter the projector setting.
    """

    if str(getattr(state, "illumination_mode", "")).upper() != "STEM":
        recording_plane = tem_projection_reference_plane(state)
        return str(recording_plane.key), float(recording_plane.z_mm)
    screen = getattr(state, "fluorescent_screen", None)
    if screen is None or not math.isfinite(float(screen.z_mm)):
        raise ValueError("The main-screen camera-length reference is absent")
    reference_z_mm = float(screen.z_mm)
    return STEM_DIFFRACTION_REFERENCE_PLANE, reference_z_mm


def _canonical_source_basis(larmor_rate_m1: float) -> np.ndarray:
    """Map canonical/Larmor specimen slopes to mechanical ray slopes.

    The specimen lies inside the objective axial field.  At such a plane the
    independent diffraction coordinate is the canonical (Larmor-frame)
    slope, not a mechanical slope with position held fixed.  For
    ``w = x + i y`` and ``g = q Bz / (2 p)``, ``u' = w' + i g w``; therefore a
    pure specimen-position basis has ``theta_x = g y`` and
    ``theta_y = -g x``.  Omitting this basis change makes exact diffraction
    conjugacy mathematically unreachable whenever ``g != 0``.
    """

    g = float(larmor_rate_m1)
    basis = np.eye(4, dtype=float)
    basis[2, 1] = g
    basis[3, 0] = -g
    return basis


def diffraction_transfer(
    state, target_z_mm: float, *, stable_axisymmetric: bool = True
) -> TransverseTransfer:
    """Return the stable specimen-canonical diffraction transfer.

    Post-specimen projector optics are axisymmetric in the configured model,
    so the Larmor-frame scalar equation avoids the severe step-size error of
    integrating fast magnetic rotation and ``dBz/dz`` separately.  A future
    deliberately enabled post-specimen quadrupole falls back to the general
    laboratory-frame tracer plus the same canonical input-basis transform.
    """

    source_z_mm = float(state.sample.z_mm)
    target_z_mm = float(target_z_mm)
    if not stable_axisymmetric or active_mapped_providers(state):
        raw = trace_transverse_transfer(
            state,
            source_z_mm,
            target_z_mm,
            maximum_step_mm=0.025,
        )
        source_field_t = float(
            fields(np.asarray((source_z_mm,)), state)[0][0]
        )
        momentum = _electron_momentum_kg_m_s(state.beam_voltage_kv)
        source_g_m1 = -E * source_field_t / (2.0 * momentum)
        matrix = raw.matrix @ _canonical_source_basis(source_g_m1)
        return TransverseTransfer(
            source_z_mm=source_z_mm,
            target_z_mm=target_z_mm,
            j_img=matrix[:2, :2],
            j_diff_m_per_rad=matrix[:2, 2:],
            k_img_rad_per_m=matrix[2:, :2],
            k_diff=matrix[2:, 2:],
            position_offset_m=raw.position_offset_m,
            angle_offset_rad=raw.angle_offset_rad,
        )
    # Match the independently required production-validation resolution so
    # GUI diagnostics cannot regress to a visibly different coarse-step plane.
    step_mm = min(max(float(state.step_mm), 1.0e-6), 0.025)
    z_mm = _piecewise_endpoint_exact_grid(
        source_z_mm, target_z_mm, step_mm
    )
    stage_z_mm = interleaved_rk4_values(
        z_mm, 0.5 * (z_mm[:-1] + z_mm[1:])
    )
    magnetic_t, sx_m2, sy_m2 = fields(stage_z_mm, state)
    momentum = _electron_momentum_kg_m_s(state.beam_voltage_kv)
    g = np.ascontiguousarray(-E * magnetic_t / (2.0 * momentum))
    if (
        np.max(np.abs(sx_m2), initial=0.0) > 1.0e-15
        or np.max(np.abs(sy_m2), initial=0.0) > 1.0e-15
    ):
        raw = trace_transverse_transfer(
            state,
            source_z_mm,
            target_z_mm,
            maximum_step_mm=0.025,
        )
        matrix = raw.matrix @ _canonical_source_basis(g[0])
    else:
        z_m = np.ascontiguousarray(z_mm * 1.0e-3)
        radial = _rk4_axisymmetric_larmor_matrix(g, z_m)
        phase = float(np.sum(
            (g[:-2:2] + 4.0 * g[1::2] + g[2::2])
            * np.diff(z_m) / 6.0
        ))
        target_g = float(g[-1])

        def complex_map(value: complex) -> np.ndarray:
            return np.asarray((
                (value.real, -value.imag),
                (value.imag, value.real),
            ))

        rotation = complex(math.cos(-phase), math.sin(-phase))
        a, b = float(radial[0, 0]), float(radial[0, 1])
        c, d = float(radial[1, 0]), float(radial[1, 1])
        matrix = np.block([
            [
                complex_map(rotation * a),
                complex_map(rotation * b),
            ],
            [
                complex_map(rotation * complex(c, -target_g * a)),
                complex_map(rotation * complex(d, -target_g * b)),
            ],
        ])
    return TransverseTransfer(
        source_z_mm=source_z_mm,
        target_z_mm=target_z_mm,
        j_img=matrix[:2, :2],
        j_diff_m_per_rad=matrix[:2, 2:],
        k_img_rad_per_m=matrix[2:, :2],
        k_diff=matrix[2:, 2:],
    )


def projector_field_calibration_rows(state):
    """Return inspectable D/I/P1/P2 field calibration and provenance."""

    lenses = _lens_map(state)
    return tuple({
        "key": key,
        "maximum_peak_field_t": float(lenses[key].b0_t),
        "field_half_width_mm": float(lenses[key].a_mm),
        "maximum_excitation_percent": float(lenses[key].max_percent),
        "status": str(getattr(
            lenses[key], "field_calibration_status", "untracked"
        )),
        "source": str(getattr(
            lenses[key], "field_calibration_source", ""
        )),
    } for key in PROJECTOR_KEYS)


def diffraction_focus_depth_diagnostic(
    state,
    *,
    tolerance: float = 1.0e-3,
):
    """Estimate the local axial interval satisfying the A-block tolerance.

    The estimate propagates the target-plane transfer through a field-free
    local drift, ``A(dz) = A + dz K_img``. It is therefore a conjugacy depth,
    not specimen depth of field and not an OEM focus specification.
    """

    reference_key, target_z_mm = diffraction_reference_plane(state)
    transfer = diffraction_transfer(state, target_z_mm)
    a_block = np.asarray(transfer.j_img, dtype=float)
    derivative_per_m = np.asarray(transfer.k_img_rad_per_m, dtype=float)
    denominator = float(np.sum(derivative_per_m * derivative_per_m))
    if denominator <= 1.0e-30:
        return {
            "reference_plane_key": reference_key,
            "target_z_mm": target_z_mm,
            "tolerance": float(tolerance),
            "best_focus_offset_mm": math.nan,
            "best_residual": float(np.linalg.norm(a_block, ord=2)),
            "full_depth_mm": 0.0,
            "model": "local_field_free_first_order_drift",
        }
    best_m = -float(np.sum(a_block * derivative_per_m)) / denominator

    def residual(dz_m: float) -> float:
        return float(np.linalg.norm(
            a_block + float(dz_m) * derivative_per_m, ord=2
        ))

    best_residual = residual(best_m)
    full_depth_m = 0.0
    tolerance = float(tolerance)
    if best_residual <= tolerance:
        initial = max(
            tolerance
            / max(float(np.linalg.norm(derivative_per_m, ord=2)), 1.0e-30),
            1.0e-12,
        )

        def find_edge(direction: float) -> float:
            step = initial
            for _ in range(80):
                candidate = best_m + direction * step
                if residual(candidate) > tolerance:
                    return brentq(
                        lambda value: residual(value) - tolerance,
                        best_m,
                        candidate,
                    )
                step *= 2.0
            return best_m

        lower = find_edge(-1.0)
        upper = find_edge(1.0)
        full_depth_m = max(upper - lower, 0.0)
    return {
        "reference_plane_key": reference_key,
        "target_z_mm": target_z_mm,
        "tolerance": tolerance,
        "best_focus_offset_mm": best_m * 1.0e3,
        "best_residual": best_residual,
        "full_depth_mm": full_depth_m * 1.0e3,
        "model": "local_field_free_first_order_drift",
    }


def _endpoint_exact_grid(
    start_z_mm: float, end_z_mm: float, step_mm: float
) -> np.ndarray:
    start = float(start_z_mm)
    end = float(end_z_mm)
    step = float(step_mm)
    if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
        raise ValueError("First-order model requires increasing finite Z limits")
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("First-order model step must be finite and positive")
    intervals = max(1, int(math.ceil((end - start) / step)))
    return np.linspace(start, end, intervals + 1, dtype=float)


def _piecewise_endpoint_exact_grid(
    start_z_mm: float,
    end_z_mm: float,
    step_mm: float,
    interior_z_mm=(),
) -> np.ndarray:
    """Return a step-bounded grid which contains every requested plane."""

    start = float(start_z_mm)
    end = float(end_z_mm)
    points = sorted({
        float(value)
        for value in interior_z_mm
        if start < float(value) < end
    })
    boundaries = [start, *points, end]
    segments = [
        _endpoint_exact_grid(left, right, step_mm)
        for left, right in zip(boundaries[:-1], boundaries[1:])
    ]
    return np.concatenate([
        segment if index == 0 else segment[1:]
        for index, segment in enumerate(segments)
    ])


@njit(cache=True)
def _rk4_transfer_matrices(g, sx, sy, z_m, capture_indices):
    """Map laboratory slopes with the production canonical RK4 stages.

    Coefficients interleave exact endpoints and midpoints.  Keeping the
    affine-free four bases in slope coordinates at each node also handles
    source and capture planes inside a magnetic lens correctly.
    """
    captured = np.empty((capture_indices.size, 4, 4), dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    capture = 0
    if capture_indices.size and capture_indices[0] == 0:
        captured[0] = matrix
        capture = 1
    for index in range(z_m.size - 1):
        h = z_m[index + 1] - z_m[index]
        a, b, c = 2 * index, 2 * index + 1, 2 * index + 2
        for column in range(4):
            x, tx, y, ty = _canonical_step_numba(
                matrix[0, column], matrix[2, column],
                matrix[1, column], matrix[3, column], h,
                g[a], g[b], g[c],
                sx[a], sx[b], sx[c], sy[a], sy[b], sy[c],
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            )
            matrix[0, column], matrix[1, column] = x, y
            matrix[2, column], matrix[3, column] = tx, ty
        while capture < capture_indices.size and capture_indices[capture] == index + 1:
            captured[capture] = matrix
            capture += 1
    return captured


@njit(cache=True)
def _rk4_transfer_matrix(g, sx, sy, z_m):
    return _rk4_transfer_matrices(
        g, sx, sy, z_m, np.asarray((z_m.size - 1,), dtype=np.int64)
    )[0]


@njit(cache=True)
def _rk4_axisymmetric_larmor_matrix(g, z_m):
    """Integrate ``u'' + g**2 u = 0`` for an axisymmetric magnetic field.

    Working in the Larmor frame removes the axial-field derivative and the
    fast laboratory-frame rotation.  This is both the physically natural
    canonical basis at a specimen inside the objective field and a much more
    stable optimiser model for strongly overlapping projector fields.
    """

    a, b, c, d = 1.0, 0.0, 0.0, 1.0
    for index in range(z_m.size - 1):
        h = z_m[index + 1] - z_m[index]
        q0 = g[2 * index] * g[2 * index]
        qm = g[2 * index + 1] * g[2 * index + 1]
        q1 = g[2 * index + 2] * g[2 * index + 2]

        a1, b1, c1, d1 = c, d, -q0 * a, -q0 * b
        aa = a + 0.5 * h * a1
        bb = b + 0.5 * h * b1
        cc = c + 0.5 * h * c1
        dd = d + 0.5 * h * d1
        a2, b2, c2, d2 = cc, dd, -qm * aa, -qm * bb
        aa = a + 0.5 * h * a2
        bb = b + 0.5 * h * b2
        cc = c + 0.5 * h * c2
        dd = d + 0.5 * h * d2
        a3, b3, c3, d3 = cc, dd, -qm * aa, -qm * bb
        aa = a + h * a3
        bb = b + h * b3
        cc = c + h * c3
        dd = d + h * d3
        a4, b4, c4, d4 = cc, dd, -q1 * aa, -q1 * bb

        a += h * (a1 + 2.0 * a2 + 2.0 * a3 + a4) / 6.0
        b += h * (b1 + 2.0 * b2 + 2.0 * b3 + b4) / 6.0
        c += h * (c1 + 2.0 * c2 + 2.0 * c3 + c4) / 6.0
        d += h * (d1 + 2.0 * d2 + 2.0 * d3 + d4) / 6.0
    return np.asarray(((a, b), (c, d)), dtype=np.float64)


def _electron_momentum_kg_m_s(voltage_kv: float) -> float:
    electron_mass_kg = 9.1093837015e-31
    speed_of_light_m_s = 299792458.0
    kinetic_j = E * float(voltage_kv) * 1000.0
    rest_j = electron_mass_kg * speed_of_light_m_s**2
    return math.sqrt(kinetic_j**2 + 2.0 * kinetic_j * rest_j) / (
        speed_of_light_m_s
    )


def _lens_map(state) -> dict[str, object]:
    return {str(lens.key): lens for lens in state.lenses}


class _LiveFirstOrderModel:
    """Cached first-order map with selected round-lens fields variable."""

    def __init__(
        self,
        state,
        source_z_mm: float,
        target_z_mm: float,
        variable_keys: tuple[str, ...],
        *,
        step_mm: float,
        capture_z_mm=(),
    ) -> None:
        self.state = state
        self.variable_keys = tuple(variable_keys)
        self.vector_maps = bool(active_mapped_providers(state))
        self.maximum_step_mm = float(step_mm)
        self.z_mm = _piecewise_endpoint_exact_grid(
            source_z_mm, target_z_mm, step_mm, capture_z_mm
        )
        self.z_m = self.z_mm * 1.0e-3
        self.stage_z_mm = interleaved_rk4_values(
            self.z_mm, 0.5 * (self.z_mm[:-1] + self.z_mm[1:])
        )
        lenses = _lens_map(state)
        try:
            self.lenses = tuple(lenses[key] for key in self.variable_keys)
        except KeyError as exc:
            raise ValueError(
                f"Direct Alignment is missing lens {exc.args[0]!r}"
            ) from exc
        if any(not bool(getattr(lens, "enabled", True)) for lens in self.lenses):
            raise ValueError("Every coupled Direct Alignment lens must be enabled")
        self.upper = np.asarray(
            [float(lens.max_percent) for lens in self.lenses], dtype=float
        )
        if np.any(self.upper <= 0.0):
            raise ValueError("Coupled lens limits must be positive")

        if self.vector_maps:
            # Imported fields must use the production XYZ solver, not the
            # on-axis scalar profiles used by the analytic fast optimiser.
            return

        original = np.asarray(
            [float(lens.percent) for lens in self.lenses], dtype=float
        )
        try:
            for lens in self.lenses:
                lens.percent = 0.0
            fixed_b, sx, sy = fields(self.stage_z_mm, state)
            profiles = []
            for lens, maximum in zip(self.lenses, self.upper):
                lens.percent = float(maximum)
                maximum_b = fields(self.stage_z_mm, state)[0]
                profiles.append(
                    (maximum_b - fixed_b) * (100.0 / float(maximum))
                )
                lens.percent = 0.0
        finally:
            for lens, value in zip(self.lenses, original):
                lens.percent = float(value)
        self.fixed_b_t = np.ascontiguousarray(fixed_b, dtype=np.float64)
        self.unit_profiles_t = np.ascontiguousarray(
            np.vstack(profiles), dtype=np.float64
        )
        self.sx_m2 = np.ascontiguousarray(sx, dtype=np.float64)
        self.sy_m2 = np.ascontiguousarray(sy, dtype=np.float64)
        momentum = _electron_momentum_kg_m_s(state.beam_voltage_kv)
        self.field_to_g_m1 = -E / (2.0 * momentum)
        # Compile the capture kernel before the first optimiser callback.
        _rk4_transfer_matrices(
            np.zeros(3), np.zeros(3), np.zeros(3),
            np.array((0.0, 1.0)), np.array((1,), dtype=np.int64),
        )
        _rk4_axisymmetric_larmor_matrix(
            np.zeros(3), np.array((0.0, 1.0))
        )

    def _field_arrays(self, vector) -> np.ndarray:
        values = np.asarray(vector, dtype=float)
        if values.shape != (len(self.variable_keys),):
            raise ValueError("Coupled lens vector has the wrong shape")
        magnetic = self.fixed_b_t + np.tensordot(
            values / 100.0, self.unit_profiles_t, axes=(0, 0)
        )
        g = np.ascontiguousarray(
            self.field_to_g_m1 * magnetic, dtype=np.float64
        )
        return g

    @contextmanager
    def _mapped_candidate(self, vector):
        values = np.asarray(vector, dtype=float)
        if values.shape != (len(self.lenses),) or not np.all(np.isfinite(values)):
            raise ValueError("Coupled lens vector has the wrong shape or values")
        original = [float(lens.percent) for lens in self.lenses]
        had_flag = hasattr(self.state, "equivalent_image_lenses_enabled")
        old_flag = getattr(self.state, "equivalent_image_lenses_enabled", False)
        try:
            for lens, value in zip(self.lenses, values):
                lens.percent = float(value)
            self.state.equivalent_image_lenses_enabled = False
            yield
        finally:
            for lens, value in zip(self.lenses, original):
                lens.percent = value
            if had_flag:
                self.state.equivalent_image_lenses_enabled = old_flag
            else:
                delattr(self.state, "equivalent_image_lenses_enabled")

    def _mapped_transfers(self, vector, targets):
        with self._mapped_candidate(vector):
            return trace_transverse_transfers(
                self.state, self.z_mm[0], targets,
                maximum_step_mm=self.maximum_step_mm,
            )

    def rays_at(self, vector, source_rays, targets):
        """Capture actual nonlinear mapped trajectories for condenser stops."""
        targets = tuple(float(z) for z in targets)
        with self._mapped_candidate(vector):
            z, x, tx, y, ty = propagate(
                self.state, self.z_mm[0], max(targets),
                source_rays[0], source_rays[2], source_rays[1], source_rays[3],
                save_z_mm=targets, maximum_step_mm=self.maximum_step_mm,
            )
        indices = [int(np.argmin(abs(z-target))) for target in targets]
        return np.asarray([(x[i], y[i], tx[i], ty[i]) for i in indices])

    def matrix(self, vector) -> np.ndarray:
        if self.vector_maps:
            target = float(self.z_mm[-1])
            return self._mapped_transfers(vector, (target,))[target].matrix
        g = self._field_arrays(vector)
        return _rk4_transfer_matrix(
            g, self.sx_m2, self.sy_m2, self.z_m
        )

    def canonical_position_blocks(
        self, vector
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return specimen-canonical A/B blocks from the full float64 map.

        Wide camera-length searches contain narrow, strongly rotating
        branches.  The scalar Larmor reduction is useful diagnostically, but
        on those branches its discretised A block can select a different root
        from the production laboratory-frame 4x4 transfer.  Direct Alignment
        therefore optimises the same complete canonical map that production
        validation checks.
        """

        if self.vector_maps:
            matrix = self.matrix(vector)
            with self._mapped_candidate(vector):
                source_b = fields((self.z_mm[0],), self.state)[0][0]
            momentum = _electron_momentum_kg_m_s(self.state.beam_voltage_kv)
            canonical = matrix @ _canonical_source_basis(-E*source_b/(2*momentum))
            return canonical[:2, :2], canonical[:2, 2:]
        g = self._field_arrays(vector)
        matrix = self.matrix(vector)
        canonical = matrix @ _canonical_source_basis(g[0])
        return canonical[:2, :2], canonical[:2, 2:]

    def matrices_at(self, vector, z_mm) -> np.ndarray:
        """Return source-to-plane maps without reintegrating the column."""

        requested = np.asarray(tuple(z_mm), dtype=float)
        indices = np.searchsorted(self.z_mm, requested)
        if np.any(indices >= self.z_mm.size) or not np.allclose(
            self.z_mm[indices], requested, rtol=0.0, atol=1.0e-9
        ):
            raise ValueError("Requested capture plane is not on the model grid")
        if np.any(np.diff(indices) < 0):
            raise ValueError("Capture planes must be ordered along +Z")
        if self.vector_maps:
            transfers = self._mapped_transfers(vector, requested)
            return np.asarray([transfers[float(z)].matrix for z in requested])
        g = self._field_arrays(vector)
        return _rk4_transfer_matrices(
            g,
            self.sx_m2,
            self.sy_m2,
            self.z_m,
            np.ascontiguousarray(indices, dtype=np.int64),
        )


class _CondenserMeasurementModel:
    def __init__(self, state, *, step_mm: float) -> None:
        self.state = state
        self.source_z_mm = float(state.electron_gun.exit_plane_z_mm)
        self.sample_z_mm = float(state.sample.z_mm)
        apertures = tuple(state.apertures)
        nanopulser = getattr(state, "nanopulser", None)
        if nanopulser is not None and bool(nanopulser.installed):
            # Match production's permanent stop in the transmitted state;
            # otherwise a small NanoPulser pupil is absent from the solve.
            apertures += (nanopulser.aperture,)
        gun_trace = state.electron_gun.trace_to_exit()
        emitted = gun_trace.exit_bundle
        self.source_rays = np.vstack((
            emitted.x_m,
            emitted.y_m,
            emitted.tx_rad,
            emitted.ty_rad,
        )).astype(float, copy=False)
        self.source_alive = np.asarray(emitted.alive, dtype=bool)
        emitted_weights = getattr(emitted, "weight", None)
        self.source_weights = (
            np.ones(self.source_alive.size, dtype=float)
            if emitted_weights is None
            else np.asarray(emitted_weights, dtype=float)
        )
        self.sample_model = _LiveFirstOrderModel(
            state,
            self.source_z_mm,
            self.sample_z_mm,
            CONDENSER_KEYS,
            step_mm=step_mm,
            capture_z_mm=(
                float(aperture.z_mm)
                for aperture in apertures
                if bool(getattr(aperture, "enabled", True))
                and bool(getattr(aperture, "installed", True))
                and self.source_z_mm
                < float(aperture.z_mm)
                < self.sample_z_mm
            ),
        )
        self.apertures = tuple(
            aperture
            for aperture in sorted(apertures, key=lambda item: item.z_mm)
            if bool(getattr(aperture, "enabled", True))
            and bool(getattr(aperture, "installed", True))
            and self.source_z_mm < float(aperture.z_mm) < self.sample_z_mm
        )
        self.upper = self.sample_model.upper

    def measure(self, vector) -> TransverseBeamStatistics:
        alive = self.source_alive.copy()
        capture_planes = [
            *(float(aperture.z_mm) for aperture in self.apertures),
            self.sample_z_mm,
        ]
        if self.sample_model.vector_maps:
            captured = self.sample_model.rays_at(vector, self.source_rays, capture_planes)
        else:
            matrices = self.sample_model.matrices_at(vector, capture_planes)
            captured = matrices @ self.source_rays
        for aperture, rays in zip(self.apertures, captured[:-1]):
            x_mm = rays[0] * 1.0e3
            y_mm = rays[1] * 1.0e3
            if hasattr(aperture, "transmission_mask"):
                passed = np.asarray(
                    aperture.transmission_mask(x_mm, y_mm), dtype=bool
                )
            else:
                radius_mm = max(0.0, float(aperture.radius_mm))
                passed = np.hypot(
                    x_mm - float(aperture.offset_x_mm),
                    y_mm - float(aperture.offset_y_mm),
                ) <= radius_mm
            alive &= passed
        sample_rays = captured[-1]
        return transverse_beam_statistics(
            sample_rays[0],
            sample_rays[1],
            sample_rays[2],
            sample_rays[3],
            alive=alive,
            weights=self.source_weights,
        )


class _EquivalentImageFirstOrderModel:
    """Fast D(z)/L(f) model used by coordinated five-lens image presets."""

    def __init__(self, state, source_z_mm: float, target_z_mm: float) -> None:
        self.source_z_mm = float(source_z_mm)
        self.target_z_mm = float(target_z_mm)
        self.calibrations = equivalent_image_calibrations(
            state, self.source_z_mm, self.target_z_mm
        )
        if tuple(item.key for item in self.calibrations) != IMAGE_KEYS:
            raise ValueError("Equivalent image-lens calibration order is invalid")
        self.upper = np.asarray(
            [item.maximum_percent for item in self.calibrations], dtype=float
        )

    def matrix(self, vector) -> np.ndarray:
        return equivalent_image_transfer_matrix(
            self.calibrations,
            vector,
            self.source_z_mm,
            self.target_z_mm,
        )


class _ProjectorMeasurementModel:
    def __init__(
        self, state, definition: DirectAlignmentDefinition, *, step_mm: float
    ) -> None:
        self.state = state
        self.definition = definition
        stop_z_mm = float(tem_projection_reference_plane(state).z_mm)
        if definition.key == IMAGE_MAGNIFICATION:
            # Image presets are a coordinated five-lens solve.  There need
            # not be an isolated real image between every pair of lenses, so
            # the authoritative condition is the complete sample-to-recording
            # transfer B=0, with total signed magnification in A.
            self.plane_z_mm = None
            self.variable_keys = IMAGE_KEYS
            if equivalent_image_maps_supported(state):
                self.sample_model = _EquivalentImageFirstOrderModel(
                    state, float(state.sample.z_mm), stop_z_mm,
                )
            else:
                self.sample_model = _LiveFirstOrderModel(
                    state, float(state.sample.z_mm), stop_z_mm,
                    self.variable_keys, step_mm=step_mm,
                )
        else:
            self.reference_plane_key, target_z_mm = (
                diffraction_reference_plane(state)
            )
            self.plane_z_mm = float(target_z_mm)
            self.variable_keys = PROJECTOR_KEYS
            self.sample_model = _LiveFirstOrderModel(
                state,
                float(state.sample.z_mm),
                self.plane_z_mm,
                self.variable_keys,
                step_mm=step_mm,
            )
        self.upper = self.sample_model.upper

    @staticmethod
    def _isotropic_scale(block: np.ndarray) -> float:
        return math.sqrt(abs(float(np.linalg.det(block))))

    def measure(
        self, vector
    ) -> tuple[DirectAlignmentMeasurement, np.ndarray]:
        if self.definition.key == IMAGE_MAGNIFICATION:
            sample_matrix = self.sample_model.matrix(vector)
            value = self._isotropic_scale(sample_matrix[:2, :2])
            relay_block = sample_matrix[:2, 2:]
            relay_error_m = float(np.linalg.norm(relay_block, ord=2))
            return (
                DirectAlignmentMeasurement(
                    key=self.definition.key,
                    value=value,
                    unit=self.definition.unit,
                    constraint_value=relay_error_m * 1.0e6,
                    constraint_unit="um",
                    relay_error_um=relay_error_m * 1.0e6,
                ),
                relay_block,
            )

        conjugacy_block, diffraction_block = (
            self.sample_model.canonical_position_blocks(vector)
        )
        value = self._isotropic_scale(diffraction_block)
        conjugacy_residual = float(
            np.linalg.norm(conjugacy_block, ord=2)
        )
        return (
            DirectAlignmentMeasurement(
                key=self.definition.key,
                value=value,
                unit=self.definition.unit,
                constraint_value=conjugacy_residual,
                constraint_unit="dimensionless",
                diffraction_conjugacy_residual=conjugacy_residual,
            ),
            conjugacy_block,
        )


def _get_vector(state, keys: tuple[str, ...]) -> np.ndarray:
    lenses = _lens_map(state)
    return np.asarray([float(lenses[key].percent) for key in keys], dtype=float)


def _set_vector(state, keys: tuple[str, ...], vector) -> None:
    lenses = _lens_map(state)
    for key, value in zip(keys, np.asarray(vector, dtype=float)):
        lenses[key].percent = float(value)


def _mode_matches(state, definition: DirectAlignmentDefinition) -> bool:
    active_modes = set()
    illumination = str(state.illumination_mode).upper()
    if illumination == "STEM":
        active_modes.add("nano_probe")
    elif illumination == "TEM":
        active_modes.add("micro_probe")
    projector = str(state.projector_mode).lower()
    if projector in {"image", "imaging"}:
        active_modes.add("imaging")
    elif projector == "diffraction":
        active_modes.add("diffraction")
    return bool(active_modes.intersection(definition.active_mode_keys))


def _target_number(
    definition: DirectAlignmentDefinition, name: str, default: float
) -> float:
    return float(definition.targets.get(name, default))


def _image_preset_seeds(
    definition: DirectAlignmentDefinition,
    target: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> list[np.ndarray]:
    """Return same-branch TOML preset seeds, nearest target first."""

    points = precalculated_alignment_points(definition)
    lm_maximum = _target_number(
        definition, "lm_maximum_magnification", 1000.0
    )
    branch = "lm" if float(target) <= lm_maximum else "normal"
    points = tuple(point for point in points if point.branch == branch)
    points = tuple(sorted(
        points,
        key=lambda point: abs(math.log(point.target / float(target))),
    ))
    seeds = []
    interpolated = interpolated_precalculated_seed(
        definition, target, lower, upper
    )
    if interpolated is not None:
        seeds.append(interpolated)
    for point in points:
        vector = np.clip(np.asarray(point.strengths), lower, upper)
        if not any(np.allclose(vector, seed) for seed in seeds):
            seeds.append(vector)
    return seeds


def _diffraction_preset_seeds(
    definition: DirectAlignmentDefinition,
    target: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> list[np.ndarray]:
    """Return detector-independent camera-length seeds for the projector."""

    points = precalculated_alignment_points(definition)
    points = tuple(sorted(
        points,
        key=lambda point: abs(math.log(point.target / float(target))),
    ))
    seeds = []
    interpolated = interpolated_precalculated_seed(
        definition, target, lower, upper
    )
    if interpolated is not None:
        seeds.append(interpolated)
    for point in points:
        vector = np.clip(np.asarray(point.strengths), lower, upper)
        if not any(np.allclose(vector, seed) for seed in seeds):
            seeds.append(vector)
    return seeds


def _projector_bounds(
    model: _ProjectorMeasurementModel,
    definition: DirectAlignmentDefinition,
    target: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.zeros_like(model.upper)
    upper = model.upper.copy()
    if definition.key != IMAGE_MAGNIFICATION:
        return lower, upper
    lm_maximum = _target_number(
        definition, "lm_maximum_magnification", 1000.0
    )
    if float(target) <= lm_maximum:
        upper[0] = min(
            upper[0],
            _target_number(
                definition, "lm_objective_max_percent", 0.001
            ),
        )
    else:
        lower[0] = min(
            upper[0],
            _target_number(
                definition, "normal_objective_min_percent", 5.0
            ),
        )
    return lower, upper


def _condenser_measurement(
    definition: DirectAlignmentDefinition,
    statistics: TransverseBeamStatistics,
) -> DirectAlignmentMeasurement:
    if definition.key == NANOPROBE_CONVERGENCE:
        return DirectAlignmentMeasurement(
            key=definition.key,
            value=statistics.convergence_95_mrad,
            unit=definition.unit,
            constraint_value=statistics.waist_offset_m * 1.0e3,
            constraint_unit="mm",
            convergence_95_mrad=statistics.convergence_95_mrad,
            convergence_99_mrad=statistics.convergence_99_mrad,
            illumination_diameter_95_um=(
                statistics.illumination_diameter_95_um
            ),
        )
    return DirectAlignmentMeasurement(
        key=definition.key,
        value=statistics.illumination_diameter_95_um,
        unit=definition.unit,
        constraint_value=statistics.radial_wavefront_curvature_per_m,
        constraint_unit="1/m",
        convergence_95_mrad=statistics.convergence_95_mrad,
        convergence_99_mrad=statistics.convergence_99_mrad,
        illumination_diameter_95_um=statistics.illumination_diameter_95_um,
    )


def _deterministic_seeds(
    initial: np.ndarray, upper: np.ndarray, *, projector: bool
) -> list[np.ndarray]:
    seeds = [initial.copy()]
    if projector:
        seeds.extend(
            np.minimum(np.full(initial.shape, level), upper)
            for level in (15, 35, 55, 75, 95)
        )
    else:
        for delta_0, delta_1 in (
            (-8, -4), (-8, 4), (-4, -2), (-4, 2),
            (4, -2), (4, 2), (8, -4), (8, 4),
        ):
            seeds.append(
                np.clip(initial + (delta_0, delta_1), 0.0, upper)
            )
    return [np.asarray(seed, dtype=float) for seed in seeds]


def _optimise_condenser(
    state,
    definition: DirectAlignmentDefinition,
    target: float,
) -> tuple[np.ndarray, int]:
    initial = _get_vector(state, CONDENSER_KEYS)
    optimiser_step = _target_number(
        definition, "optimiser_step_mm", 0.1
    )
    model = _CondenserMeasurementModel(state, step_mm=optimiser_step)
    optimisation_lower = np.zeros(initial.size, dtype=float)
    optimisation_upper = model.upper.copy()
    if definition.key == NANOPROBE_CONVERGENCE:
        constraint_scale = _target_number(
            definition,
            "optimiser_waist_scale_mm",
            _target_number(definition, "maximum_waist_offset_mm", 0.002),
        )
    else:
        constraint_scale = _target_number(
            definition, "maximum_curvature_per_m", 25.0
        )
        maximum_angle = _target_number(
            definition, "maximum_convergence_mrad", 0.5
        )
        # The validated Microprobe branch reaches the configured area range
        # without driving the default C2 setting above the requested 30-70%
        # operating window.  The underlying low-level control remains free to
        # use the full TOML-rated field.
        optimisation_lower[0] = _target_number(
            definition, "c2_minimum_percent", 0.0
        )
        optimisation_upper[0] = min(
            _target_number(definition, "c2_maximum_percent", 70.0),
            optimisation_upper[0],
        )
        optimisation_lower[1] = _target_number(
            definition, "c3_minimum_percent", 0.0
        )
        optimisation_upper[1] = min(
            _target_number(definition, "c3_maximum_percent", 100.0),
            optimisation_upper[1],
        )
        if np.any(optimisation_lower >= optimisation_upper):
            raise ValueError("Microprobe C2/C3 operating bounds are invalid")

    def residual(vector):
        statistics = model.measure(vector)
        measurement = _condenser_measurement(definition, statistics)
        primary = math.log(max(measurement.value, 1.0e-15) / target)
        constraint = measurement.constraint_value / constraint_scale
        values = [primary, constraint]
        if definition.key == MICROPROBE_ILLUMINATION:
            values.append(
                max(0.0, statistics.convergence_95_mrad - maximum_angle)
                / max(maximum_angle * 0.1, 0.01)
            )
        regularisation = 1.0e-4 * (vector - initial) / np.maximum(
            np.abs(initial), 25.0
        )
        return np.r_[values, regularisation]

    candidate_seeds = _deterministic_seeds(
        np.clip(initial, optimisation_lower, optimisation_upper),
        optimisation_upper,
        projector=False,
    )
    if definition.key == NANOPROBE_CONVERGENCE:
        warm_seed = interpolated_nanoprobe_seed(
            definition,
            target,
            float(state.condenser_aperture_2.diameter_um),
            np.zeros_like(initial),
            optimisation_upper,
        )
        if warm_seed is not None:
            candidate_seeds = [warm_seed] + [
                seed
                for seed in candidate_seeds
                if not np.allclose(seed, warm_seed)
            ]
    # Microprobe illuminated area has a much wider C2 solution curve than the
    # local preset neighbourhood.  Rank a small deterministic coarse grid and
    # refine only its best points; this keeps the GUI solve both global enough
    # and interactive.
    if definition.key == MICROPROBE_ILLUMINATION:
        coarse = []
        for c2 in np.linspace(
            optimisation_lower[0], optimisation_upper[0], 9
        ):
            def curvature_residual(c3):
                statistics = model.measure((c2, float(c3[0])))
                return np.asarray((
                    statistics.radial_wavefront_curvature_per_m
                    / constraint_scale,
                ))

            c3_solution = least_squares(
                curvature_residual,
                np.asarray((np.clip(
                    initial[1],
                    optimisation_lower[1],
                    optimisation_upper[1],
                ),)),
                bounds=(
                    np.asarray((optimisation_lower[1],)),
                    np.asarray((optimisation_upper[1],)),
                ),
                max_nfev=30,
                diff_step=2.0e-3,
            )
            vector = np.asarray((c2, c3_solution.x[0]), dtype=float)
            values = residual(vector)
            headroom = np.maximum(0.0, np.abs(vector - 50.0) - 20.0)
            score = float(values[:3] @ values[:3]) + 1.0e-6 * float(
                headroom @ headroom
            )
            coarse.append((score, vector))
        coarse.sort(key=lambda item: item[0])
        candidate_seeds = [initial.copy()]
        for _cost, vector in coarse:
            if not any(np.allclose(vector, seed) for seed in candidate_seeds):
                candidate_seeds.append(vector)
            if len(candidate_seeds) == 6:
                break

        candidate_seeds = [
            np.clip(seed, optimisation_lower, optimisation_upper)
            for seed in candidate_seeds
        ]

    best_vector = initial.copy()
    best_cost = math.inf
    iterations = 0
    for seed in candidate_seeds:
        solution = least_squares(
            residual,
            seed,
            bounds=(optimisation_lower, optimisation_upper),
            max_nfev=70,
            diff_step=2.0e-3,
            x_scale="jac",
        )
        iterations += int(solution.nfev)
        primary_count = 3 if definition.key == MICROPROBE_ILLUMINATION else 2
        cost = float(
            residual(solution.x)[:primary_count]
            @ residual(solution.x)[:primary_count]
        )
        if definition.key == MICROPROBE_ILLUMINATION:
            headroom = np.maximum(
                0.0, np.abs(solution.x - 50.0) - 20.0
            )
            cost += 1.0e-6 * float(headroom @ headroom)
        if cost < best_cost:
            best_cost = cost
            best_vector = solution.x.copy()
        measurement = _condenser_measurement(
            definition, model.measure(solution.x)
        )
        relative_error = abs(math.log(max(measurement.value, 1.0e-15) / target))
        if definition.key == NANOPROBE_CONVERGENCE:
            constraint_ok = abs(measurement.constraint_value) <= constraint_scale
        else:
            constraint_ok = (
                abs(measurement.constraint_value) <= constraint_scale
                and float(
                    math.inf
                    if measurement.convergence_95_mrad is None
                    else measurement.convergence_95_mrad
                )
                <= maximum_angle
                and float(
                    math.inf
                    if measurement.convergence_99_mrad is None
                    else measurement.convergence_99_mrad
                )
                <= _target_number(
                    definition, "maximum_convergence_99_mrad", 0.5
                )
            )
        if (
            relative_error
            <= _target_number(definition, "maximum_relative_error", 0.03) * 0.5
            and constraint_ok
        ):
            break
    return best_vector, iterations


def _projector_continuation_targets(
    model: _ProjectorMeasurementModel,
    initial: np.ndarray,
    target: float,
    definition: DirectAlignmentDefinition,
) -> tuple[float, ...]:
    """Split a large optical-scale jump into bounded logarithmic steps."""

    current = float(model.measure(initial)[0].value)
    requested = float(target)
    if not (
        math.isfinite(current)
        and current > 0.0
        and math.isfinite(requested)
        and requested > 0.0
    ):
        return (requested,)
    maximum_ratio = max(
        1.1,
        _target_number(definition, "maximum_continuation_ratio", 2.0),
    )
    maximum_stages = max(
        1,
        int(round(
            _target_number(
                definition, "maximum_continuation_stages", 8.0
            )
        )),
    )
    logarithmic_span = abs(math.log(requested / current))
    stage_count = min(
        maximum_stages,
        max(1, int(math.ceil(logarithmic_span / math.log(maximum_ratio)))),
    )
    return tuple(
        float(value)
        for value in np.geomspace(current, requested, stage_count + 1)[1:]
    )


def _solve_projector_stage(
    model: _ProjectorMeasurementModel,
    definition: DirectAlignmentDefinition,
    target: float,
    initial: np.ndarray,
    *,
    allow_global_fallback: bool,
) -> tuple[np.ndarray, int]:
    """Solve one nearby target, falling back to deterministic global seeds."""

    constraint_scale = (
        _target_number(
            definition, "maximum_diffraction_conjugacy_residual", 1.0e-3
        )
        if definition.key == DIFFRACTION_CAMERA_LENGTH
        else _target_number(
            definition, "maximum_relay_error_um", 20.0
        ) * 1.0e-6
    )
    maximum_relative_error = _target_number(
        definition, "maximum_relative_error", 0.03
    )

    lower, upper = _projector_bounds(
        model, definition, float(target)
    )
    reference = np.clip(
        np.asarray(initial, dtype=float), lower, upper
    )

    def residual(vector):
        measurement, relay_block = model.measure(vector)
        primary = (
            math.log(max(measurement.value, 1.0e-15) / target)
            / maximum_relative_error
        )
        regularisation = 1.0e-4 * (vector - reference) / np.maximum(
            np.abs(reference), 25.0
        )
        return np.r_[
            primary,
            np.asarray(relay_block, dtype=float).ravel() / constraint_scale,
            regularisation,
        ]

    best_vector = reference.copy()
    best_cost = math.inf
    iterations = 0
    seeds = [reference]
    if allow_global_fallback:
        if definition.key == IMAGE_MAGNIFICATION:
            seeds.extend(
                seed
                for seed in _image_preset_seeds(
                    definition, target, lower, upper
                )
                if not any(np.allclose(seed, item) for item in seeds)
            )
        elif definition.key == DIFFRACTION_CAMERA_LENGTH:
            seeds.extend(
                seed
                for seed in _diffraction_preset_seeds(
                    definition, target, lower, upper
                )
                if not any(np.allclose(seed, item) for item in seeds)
            )
        seeds.extend(
            np.clip(seed, lower, upper)
            for seed in _deterministic_seeds(
                reference, model.upper, projector=True
            )[1:]
            if not any(np.allclose(seed, item) for item in seeds)
        )
    for seed in seeds:
        solution = least_squares(
            residual,
            seed,
            bounds=(lower, upper),
            max_nfev=100,
            diff_step=2.0e-3,
            x_scale="jac",
        )
        iterations += int(solution.nfev)
        if not np.all(np.isfinite(solution.x)):
            continue
        measurement, relay_block = model.measure(solution.x)
        relative_error = abs(
            math.log(max(measurement.value, 1.0e-15) / target)
        )
        cost = (
            (relative_error / maximum_relative_error) ** 2
            + (
                float(np.linalg.norm(relay_block, ord=2)) / constraint_scale
            ) ** 2
        )
        if cost < best_cost:
            best_cost = cost
            best_vector = solution.x.copy()
        if (
            relative_error <= maximum_relative_error * 0.5
            and float(np.linalg.norm(relay_block, ord=2))
            <= constraint_scale * 0.25
        ):
            break
    return best_vector, iterations


def _optimise_projector(
    state,
    definition: DirectAlignmentDefinition,
    target: float,
    *,
    initial_vector: np.ndarray | None = None,
    step_mm: float | None = None,
    continuation: bool = True,
    allow_global_fallback: bool = True,
) -> tuple[np.ndarray, int]:
    """Solve the mode-specific coupled lens set on a live transfer matrix."""

    keys = (
        IMAGE_KEYS
        if definition.key == IMAGE_MAGNIFICATION
        else PROJECTOR_KEYS
    )
    initial = (
        _get_vector(state, keys)
        if initial_vector is None
        else np.asarray(initial_vector, dtype=float).copy()
    )
    optimiser_step = (
        _target_number(definition, "optimiser_step_mm", 0.1)
        if step_mm is None
        else float(step_mm)
    )
    model = _ProjectorMeasurementModel(
        state, definition, step_mm=optimiser_step
    )
    lower, upper = _projector_bounds(model, definition, float(target))
    vector = np.clip(initial, lower, upper)
    if initial_vector is None and allow_global_fallback:
        warm_seed = interpolated_precalculated_seed(
            definition, target, lower, upper
        )
        if warm_seed is not None:
            constraint_scale = (
                _target_number(
                    definition,
                    "maximum_diffraction_conjugacy_residual",
                    1.0e-3,
                )
                if definition.key == DIFFRACTION_CAMERA_LENGTH
                else _target_number(
                    definition, "maximum_relay_error_um", 20.0
                ) * 1.0e-6
            )

            def warm_start_score(candidate: np.ndarray) -> float:
                measurement, constraint = model.measure(candidate)
                primary = abs(math.log(
                    max(measurement.value, 1.0e-15) / float(target)
                ))
                conjugacy = float(
                    np.linalg.norm(constraint, ord=2)
                ) / max(constraint_scale, 1.0e-15)
                return primary * primary + conjugacy * conjugacy

            if warm_start_score(warm_seed) < warm_start_score(vector):
                vector = warm_seed
    stage_targets = (
        _projector_continuation_targets(
            model, vector, float(target), definition
        )
        if continuation and definition.key != IMAGE_MAGNIFICATION
        else (float(target),)
    )
    iterations = 0
    for index, stage_target in enumerate(stage_targets):
        vector, stage_iterations = _solve_projector_stage(
            model,
            definition,
            stage_target,
            vector,
            allow_global_fallback=(
                allow_global_fallback and index == len(stage_targets) - 1
            ),
        )
        iterations += stage_iterations
    return vector, iterations


def _validate_projector(
    state,
    definition: DirectAlignmentDefinition,
    vector: np.ndarray,
    step_mm: float,
) -> DirectAlignmentMeasurement:
    model = _ProjectorMeasurementModel(
        state, definition, step_mm=step_mm
    )
    return model.measure(vector)[0]


def _pre_sample_kick_events(state) -> tuple[tuple[float, float, float], ...]:
    """Collect the same upstream affine kicks used by ``simulation.run``."""

    sample_z_mm = float(state.sample.z_mm)
    events: list[tuple[float, float, float]] = []
    for deflector in state.deflectors:
        if not bool(getattr(deflector, "enabled", False)):
            continue
        if hasattr(deflector, "kick_events"):
            pairs = deflector.kick_events()
        else:
            pairs = (
                (
                    deflector.upper_z_mm,
                    deflector.upper_x_mrad * 1.0e-3,
                    deflector.upper_y_mrad * 1.0e-3,
                ),
                (
                    deflector.lower_z_mm,
                    deflector.lower_x_mrad * 1.0e-3,
                    deflector.lower_y_mrad * 1.0e-3,
                ),
            )
        for event in pairs:
            if float(event[0]) <= sample_z_mm:
                events.append(tuple(float(value) for value in event))

    for component in getattr(state, "corrector_elements", ()):
        if not bool(getattr(component, "enabled", False)):
            continue
        if not hasattr(component, "kick_events"):
            continue
        try:
            pairs = component.kick_events(
                time_s=float(getattr(state, "simulation_time_s", 0.0))
            )
        except TypeError:
            pairs = component.kick_events()
        for event in pairs:
            if float(event[0]) <= sample_z_mm:
                events.append(tuple(float(value) for value in event))
    return tuple(events)


@contextmanager
def _production_validation_state(
    state,
    keys: tuple[str, ...],
    vector: np.ndarray,
    step_mm: float,
):
    """Temporarily install a candidate on the deterministic CPU path."""

    original_vector = _get_vector(state, keys)
    original_step = float(state.step_mm)
    original_acceleration_enabled = bool(state.acceleration_enabled)
    original_acceleration_backend = str(state.acceleration_backend)
    original_active_backend = str(state.active_backend)
    original_equivalent_image_lenses = bool(
        getattr(state, "equivalent_image_lenses_enabled", False)
    )
    had_used_backends = hasattr(state, "_active_backends_used")
    original_used_backends = set(
        getattr(state, "_active_backends_used", set())
    )
    try:
        _set_vector(state, keys, vector)
        state.step_mm = float(step_mm)
        state.acceleration_enabled = False
        state.acceleration_backend = "CPU"
        state.active_backend = "CPU"
        if tuple(keys) == IMAGE_KEYS:
            state.equivalent_image_lenses_enabled = True
        state._active_backends_used = set()
        yield
    finally:
        _set_vector(state, keys, original_vector)
        state.step_mm = original_step
        state.acceleration_enabled = original_acceleration_enabled
        state.acceleration_backend = original_acceleration_backend
        state.active_backend = original_active_backend
        state.equivalent_image_lenses_enabled = (
            original_equivalent_image_lenses
        )
        if had_used_backends:
            state._active_backends_used = original_used_backends
        elif hasattr(state, "_active_backends_used"):
            delattr(state, "_active_backends_used")


def _validate_condenser_production(
    state,
    definition: DirectAlignmentDefinition,
    vector: np.ndarray,
    step_mm: float,
) -> DirectAlignmentMeasurement:
    """Validate with the full nonlinear ray path and all physical clipping."""

    with _production_validation_state(
        state, CONDENSER_KEYS, vector, step_mm
    ):
        gun_trace = state.electron_gun.trace_to_exit()
        emitted = gun_trace.exit_bundle
        source_z_mm = float(state.electron_gun.exit_plane_z_mm)
        sample_z_mm = float(state.sample.z_mm)
        aperture_z_mm = tuple(
            float(aperture.z_mm)
            for aperture in state.apertures
            if bool(getattr(aperture, "enabled", False))
            and bool(getattr(aperture, "installed", True))
            and source_z_mm <= float(aperture.z_mm) <= sample_z_mm
        )
        z_mm, x_m, tx_rad, y_m, ty_rad = propagate(
            state,
            source_z_mm,
            sample_z_mm,
            emitted.x_m,
            emitted.tx_rad,
            emitted.y_m,
            emitted.ty_rad,
            events=_pre_sample_kick_events(state),
            energy_offset_ev=emitted.energy_offset_ev,
            save_z_mm=aperture_z_mm,
        )
        alive = np.asarray(emitted.alive, dtype=bool).copy()
        blocked_z_mm = np.asarray(gun_trace.blocked_z_mm, dtype=float).copy()
        blocked_key = list(gun_trace.blocked_key)
        alive, blocked_z_mm, blocked_key = clip_segment(
            state,
            z_mm,
            x_m,
            y_m,
            alive,
            blocked_z_mm,
            blocked_key,
        )
        alive, _blocked_z_mm, _blocked_key = clip_column_wall(
            state,
            z_mm,
            x_m,
            y_m,
            alive,
            blocked_z_mm,
            blocked_key,
        )
        statistics = transverse_beam_statistics(
            x_m[-1],
            y_m[-1],
            tx_rad[-1],
            ty_rad[-1],
            alive=alive,
            weights=getattr(emitted, "weight", None),
        )
    return _condenser_measurement(definition, statistics)


def _refine_nanoprobe_production_focus(
    state,
    definition: DirectAlignmentDefinition,
    target: float,
    vector: np.ndarray,
    step_mm: float,
) -> tuple[np.ndarray, int]:
    """Polish the production focus without discarding an acceptable pupil.

    A clipped current-weighted angular quantile is not a smooth lens response.
    When the first-order candidate already has an acceptable convergence,
    preserve C2 and solve the local C3 focus first.  A coupled Newton step is
    only a fallback, and must not replace a better production-validated point.
    The final Direct Alignment angle, waist and step-spread gates still apply.
    """
    if definition.key != NANOPROBE_CONVERGENCE:
        return np.asarray(vector, dtype=float), 0
    initial = np.asarray(vector, dtype=float).copy()
    lenses = _lens_map(state)
    upper = np.asarray(
        [float(lenses[key].max_percent) for key in CONDENSER_KEYS], dtype=float
    )
    angle_tolerance = _target_number(definition, "maximum_relative_error", 0.03)
    waist_tolerance = _target_number(definition, "maximum_waist_offset_mm", 0.002)
    measurements: dict[tuple[float, ...], DirectAlignmentMeasurement] = {}

    def measure(candidate):
        key = tuple(float(value) for value in candidate)
        if key not in measurements:
            measurements[key] = _validate_condenser_production(
                state, definition, candidate, step_mm
            )
        return measurements[key]

    def angle_error(measured):
        return abs(math.log(max(measured.value, 1.0e-15) / float(target)))

    def acceptable(measured):
        return (
            angle_error(measured) <= angle_tolerance
            and abs(measured.constraint_value) <= waist_tolerance
        )

    def score(candidate):
        measured = measure(candidate)
        # Every valid point ranks ahead of every invalid one.
        return max(
            angle_error(measured) / angle_tolerance,
            abs(measured.constraint_value) / waist_tolerance,
        )

    def polish_c3(vector):
        candidate = np.asarray(vector, dtype=float).copy()

        def waist_at_c3(percent):
            probe = candidate.copy()
            probe[1] = float(percent)
            return measure(probe).constraint_value

        centre = float(candidate[1])
        if abs(waist_at_c3(centre)) <= 1.0e-12:
            return candidate
        for half_width in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
            lower = max(0.0, centre - half_width)
            higher = min(float(upper[1]), centre + half_width)
            lower_waist, upper_waist = waist_at_c3(lower), waist_at_c3(higher)
            if lower_waist == 0.0:
                candidate[1] = lower
                return candidate
            if upper_waist == 0.0:
                candidate[1] = higher
                return candidate
            if lower_waist * upper_waist < 0.0:
                candidate[1] = brentq(
                    waist_at_c3, lower, higher, xtol=1.0e-7, rtol=1.0e-10
                )
                return candidate
        return candidate

    base = measure(initial)
    if acceptable(base):
        return initial, len(measurements)
    candidates = [initial]
    if angle_error(base) <= angle_tolerance:
        focused = polish_c3(initial)
        candidates.append(focused)
        if acceptable(measure(focused)):
            return focused, len(measurements)

    residual = np.asarray((
        math.log(max(base.value, 1.0e-15) / float(target)),
        base.constraint_value,
    ))
    perturbation = 1.0e-3
    jacobian = np.empty((2, 2), dtype=float)
    for index in range(2):
        shifted = initial.copy()
        shifted[index] = min(shifted[index] + perturbation, upper[index])
        actual_step = shifted[index] - initial[index]
        if actual_step <= 0.0:
            return min(candidates, key=score), len(measurements)
        measured = measure(shifted)
        shifted_residual = np.asarray((
            math.log(max(measured.value, 1.0e-15) / float(target)),
            measured.constraint_value,
        ))
        jacobian[:, index] = (shifted_residual - residual) / actual_step
    try:
        correction = np.linalg.solve(jacobian, -residual)
    except np.linalg.LinAlgError:
        return min(candidates, key=score), len(measurements)
    if not np.all(np.isfinite(correction)):
        return min(candidates, key=score), len(measurements)
    correction = np.clip(correction, -2.0, 2.0)
    candidate = np.clip(initial + correction, 0.0, upper)
    candidates.append(candidate)
    candidates.append(polish_c3(candidate))
    best = min(candidates, key=score)
    return best, len(measurements)


def _validate_projector_production(
    state,
    definition: DirectAlignmentDefinition,
    vector: np.ndarray,
    step_mm: float,
) -> DirectAlignmentMeasurement:
    """Validate using the production full transverse-transfer tracer."""

    keys = (
        IMAGE_KEYS
        if definition.key == IMAGE_MAGNIFICATION
        else PROJECTOR_KEYS
    )
    with _production_validation_state(
        state, keys, vector, step_mm
    ):
        target_z_mm = (
            float(tem_projection_reference_plane(state).z_mm)
            if definition.key == IMAGE_MAGNIFICATION
            else diffraction_reference_plane(state)[1]
        )
        sample_transfer = (
            trace_transverse_transfer(
                state, float(state.sample.z_mm), target_z_mm
            )
            if definition.key == IMAGE_MAGNIFICATION
            else diffraction_transfer(
                state, target_z_mm, stable_axisymmetric=False
            )
        )
        if definition.key == IMAGE_MAGNIFICATION:
            block = sample_transfer.j_img
            constraint_block = sample_transfer.j_diff_m_per_rad
            constraint_value = float(
                np.linalg.norm(constraint_block, ord=2)
            ) * 1.0e6
            constraint_unit = "um"
            relay_error_um = constraint_value
            conjugacy_residual = None
        else:
            block = sample_transfer.j_diff_m_per_rad
            constraint_block = sample_transfer.j_img
            constraint_value = float(
                np.linalg.norm(constraint_block, ord=2)
            )
            constraint_unit = "dimensionless"
            relay_error_um = None
            conjugacy_residual = constraint_value
        value = math.sqrt(abs(float(np.linalg.det(block))))
    return DirectAlignmentMeasurement(
        key=definition.key,
        value=value,
        unit=definition.unit,
        constraint_value=constraint_value,
        constraint_unit=constraint_unit,
        relay_error_um=relay_error_um,
        diffraction_conjugacy_residual=conjugacy_residual,
    )


def _solve_direct_alignment(
    state,
    key: str,
    target: float,
    *,
    definition: DirectAlignmentDefinition | None = None,
) -> DirectAlignmentResult:
    """Solve and transactionally apply one Direct Alignment target."""

    definition = definition or direct_alignment_by_key(key)
    if definition.key != str(key):
        raise ValueError("Direct Alignment definition key does not match")
    requested = float(target)
    if not math.isfinite(requested) or not (
        definition.minimum <= requested <= definition.maximum
    ):
        raise ValueError(
            f"{definition.name} must be between {definition.minimum:g} and "
            f"{definition.maximum:g} {definition.unit}"
        )
    if not _mode_matches(state, definition):
        active_modes = " or ".join(definition.active_mode_keys)
        raise ValueError(
            f"{definition.name} is only active in {active_modes} mode"
        )

    if definition.family == "condenser":
        keys = CONDENSER_KEYS
    elif definition.key == IMAGE_MAGNIFICATION:
        keys = IMAGE_KEYS
    else:
        keys = PROJECTOR_KEYS
    initial = _get_vector(state, keys)
    initial_equivalent_image_lenses = bool(
        getattr(state, "equivalent_image_lenses_enabled", False)
    )
    validation_step = _target_number(
        definition, "validation_step_mm", 0.05
    )
    optimiser_step = _target_number(
        definition, "optimiser_step_mm", 0.1
    )
    try:
        if definition.family == "condenser":
            current_is_acceptable = False
            if definition.key == NANOPROBE_CONVERGENCE:
                coarse = _validate_condenser_production(
                    state, definition, initial, optimiser_step
                )
                fine = _validate_condenser_production(
                    state, definition, initial, validation_step
                )
                current_relative_error = abs(math.log(
                    max(fine.value, 1.0e-15) / requested
                ))
                current_spread = abs(fine.value - coarse.value) / max(
                    abs(fine.value), 1.0e-15
                )
                current_is_acceptable = (
                    current_relative_error
                    <= _target_number(
                        definition, "maximum_relative_error", 0.03
                    )
                    and abs(fine.constraint_value)
                    <= _target_number(
                        definition, "maximum_waist_offset_mm", 0.002
                    )
                    and current_spread
                    <= _target_number(
                        definition, "maximum_numerical_spread", 0.01
                    )
                )
            if current_is_acceptable:
                candidate = initial.copy()
                iterations = 0
            else:
                candidate, iterations = _optimise_condenser(
                    state, definition, requested
                )
                candidate, refinement_iterations = (
                    _refine_nanoprobe_production_focus(
                        state,
                        definition,
                        requested,
                        candidate,
                        validation_step,
                    )
                )
                iterations += refinement_iterations
                # Compare integration resolutions with the same production
                # ray model. A paraxial-vs-production discrepancy is model
                # error, not numerical spread after aperture clipping.
                coarse = _validate_condenser_production(
                    state, definition, candidate, optimiser_step
                )
                fine = _validate_condenser_production(
                    state, definition, candidate, validation_step
                )
        else:
            candidate, iterations = _optimise_projector(
                state, definition, requested
            )
            if not math.isclose(
                optimiser_step,
                validation_step,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                candidate, refinement_iterations = _optimise_projector(
                    state,
                    definition,
                    requested,
                    initial_vector=candidate,
                    step_mm=validation_step,
                    continuation=False,
                    allow_global_fallback=False,
                )
                iterations += refinement_iterations
            coarse = _validate_projector(
                state, definition, candidate, optimiser_step
            )
            fine = _validate_projector_production(
                state, definition, candidate, validation_step
            )
        relative_error = abs(
            math.log(max(fine.value, 1.0e-15) / requested)
        )
        maximum_relative_error = _target_number(
            definition, "maximum_relative_error", 0.03
        )
        if definition.key == NANOPROBE_CONVERGENCE:
            constraint_ok = abs(fine.constraint_value) <= _target_number(
                definition, "maximum_waist_offset_mm", 0.002
            )
        elif definition.key == MICROPROBE_ILLUMINATION:
            constraint_ok = (
                abs(fine.constraint_value)
                <= _target_number(
                    definition, "maximum_curvature_per_m", 25.0
                )
                and float(
                    math.inf
                    if fine.convergence_95_mrad is None
                    else fine.convergence_95_mrad
                )
                <= _target_number(
                    definition, "maximum_convergence_mrad", 0.5
                )
                and float(
                    math.inf
                    if fine.convergence_99_mrad is None
                    else fine.convergence_99_mrad
                )
                <= _target_number(
                    definition, "maximum_convergence_99_mrad", 0.5
                )
            )
        elif definition.key == IMAGE_MAGNIFICATION:
            constraint_ok = (
                float(
                    math.inf
                    if fine.relay_error_um is None
                    else fine.relay_error_um
                )
                <= _target_number(
                    definition, "maximum_relay_error_um", 20.0
                )
            )
        else:
            constraint_ok = (
                float(
                    math.inf
                    if fine.diffraction_conjugacy_residual is None
                    else fine.diffraction_conjugacy_residual
                )
                <= _target_number(
                    definition,
                    "maximum_diffraction_conjugacy_residual",
                    1.0e-3,
                )
            )
        numerical_spread = abs(fine.value - coarse.value) / max(
            abs(fine.value), 1.0e-15
        )
        maximum_numerical_spread = _target_number(
            definition, "maximum_numerical_spread", 0.01
        )
        numerically_stable = numerical_spread <= maximum_numerical_spread
        success = (
            relative_error <= maximum_relative_error
            and constraint_ok
            and numerically_stable
        )
        if success:
            _set_vector(state, keys, candidate)
            if definition.key == IMAGE_MAGNIFICATION:
                state.equivalent_image_lenses_enabled = True
            committed = candidate
        else:
            _set_vector(state, keys, initial)
            state.equivalent_image_lenses_enabled = (
                initial_equivalent_image_lenses
            )
            committed = initial
        strengths = {
            lens_key: float(value)
            for lens_key, value in zip(keys, committed)
        }
        constraint_label = {
            NANOPROBE_CONVERGENCE: "waist offset",
            MICROPROBE_ILLUMINATION: "wavefront curvature",
            IMAGE_MAGNIFICATION: "sample-image residual",
            DIFFRACTION_CAMERA_LENGTH: "sample-diffraction residual",
        }[definition.key]
        if definition.key == DIFFRACTION_CAMERA_LENGTH:
            target_plane_key, _target_z_mm = diffraction_reference_plane(state)
        else:
            target_plane_key = None
            _target_z_mm = None
        message = (
            f"Requested {requested:.6g} {definition.unit}; achieved "
            f"{fine.value:.6g} {definition.unit}; {constraint_label} "
            f"{fine.constraint_value:.6g} {fine.constraint_unit}; "
            f"validated at {validation_step:g} mm."
        )
        if not success:
            message += (
                " Target is not reachable with the current field limits and "
                "conjugate constraint; previous lens values were restored."
            )
            lenses = _lens_map(state)
            active_limits = []
            for lens_key, value in zip(keys, candidate):
                upper = float(lenses[lens_key].max_percent)
                tolerance = max(1.0e-3, upper * 1.0e-5)
                if float(value) <= tolerance:
                    active_limits.append(
                        f"{lens_key}=lower limit ({float(value):.6g}%)"
                    )
                elif upper - float(value) <= tolerance:
                    active_limits.append(
                        f"{lens_key}=upper limit "
                        f"({float(value):.6g}/{upper:.6g}%)"
                    )
            if active_limits:
                message += " Limiting candidate: " + ", ".join(
                    active_limits
                ) + "."
            if not numerically_stable:
                message += (
                    " The optimiser/validation observable spread exceeded "
                    f"{maximum_numerical_spread:.3g}."
                )
            if definition.key == DIFFRACTION_CAMERA_LENGTH:
                statuses = sorted({
                    row["status"]
                    for row in projector_field_calibration_rows(state)
                })
                message += (
                    " Projector field calibration status: "
                    + ", ".join(statuses)
                    + "; update the recording-system TOML from measured or "
                    "manufacturer field calibration before expanding limits."
                )
        return DirectAlignmentResult(
            key=definition.key,
            success=success,
            requested=requested,
            achieved=fine.value,
            unit=definition.unit,
            constraint_value=fine.constraint_value,
            constraint_unit=fine.constraint_unit,
            strengths=strengths,
            iterations=iterations,
            validation_step_mm=validation_step,
            numerical_spread=numerical_spread,
            message=message,
            convergence_95_mrad=fine.convergence_95_mrad,
            convergence_99_mrad=fine.convergence_99_mrad,
            illumination_diameter_95_um=(
                fine.illumination_diameter_95_um
            ),
            relay_error_um=fine.relay_error_um,
            diffraction_conjugacy_residual=(
                fine.diffraction_conjugacy_residual
            ),
            target_plane_key=(
                target_plane_key
            ),
            target_plane_z_mm=(
                float(_target_z_mm) if _target_z_mm is not None else None
            ),
            field_calibration_statuses=tuple(sorted({
                row["status"]
                for row in projector_field_calibration_rows(state)
            })) if definition.family == "projector" else (),
            candidate_strengths={
                lens_key: float(value)
                for lens_key, value in zip(keys, candidate)
            },
            candidate_limit_fractions={
                lens_key: float(value)
                / max(float(_lens_map(state)[lens_key].max_percent), 1.0e-15)
                for lens_key, value in zip(keys, candidate)
            },
            state_updates={},
        )
    except Exception:
        _set_vector(state, keys, initial)
        state.equivalent_image_lenses_enabled = (
            initial_equivalent_image_lenses
        )
        raise


def apply_direct_alignment(
    state,
    key: str,
    target: float,
    *,
    definition: DirectAlignmentDefinition | None = None,
) -> DirectAlignmentResult:
    """Solve transmitted-beam optics while preserving both blanking gates.

    Condenser solves on the GUI worker snapshot temporarily open its ordinary
    and optional blankers. Actual beam measurements and image calculation keep
    their configured gate states and still report a blank specimen correctly.
    """
    definition = definition or direct_alignment_by_key(key)
    if definition.family != "condenser":
        return _solve_direct_alignment(state, key, target, definition=definition)

    from temsim.optics.calibration_beam import transmitted_calibration_beam

    with transmitted_calibration_beam(state):
        return _solve_direct_alignment(state, key, target, definition=definition)
