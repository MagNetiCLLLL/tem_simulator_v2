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
from scipy.optimize import least_squares

from temsim.component_keys import (
    CONDENSER_LENS_2,
    CONDENSER_LENS_3,
    DIFFRACTION_LENS,
    INTERMEDIATE_LENS,
    OBJECTIVE_LENS,
    PROJECTOR_LENS_1,
    PROJECTOR_LENS_2,
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
from temsim.physics.core import E, fields, propagate
from temsim.physics.first_order import trace_transverse_transfer
from temsim.physics.recording_stop import determine_tem_stop_z
from temsim.optics.equivalent_image_lenses import (
    equivalent_image_calibrations,
    equivalent_image_transfer_matrix,
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
    illumination_diameter_95_um: float | None = None
    relay_error_um: float | None = None


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
    illumination_diameter_95_um: float | None = None
    relay_error_um: float | None = None


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
def _rk4_transfer_matrix(g, dg, sx, sy, z_m):
    """Integrate the coupled laboratory-frame 4x4 paraxial map."""

    matrix = np.eye(4, dtype=np.float64)
    for index in range(z_m.size - 1):
        h = z_m[index + 1] - z_m[index]
        g0 = g[index]
        g1 = g[index + 1]
        gm = 0.5 * (g0 + g1)
        dg0 = dg[index]
        dg1 = dg[index + 1]
        dgm = 0.5 * (dg0 + dg1)
        sx0 = sx[index]
        sx1 = sx[index + 1]
        sxm = 0.5 * (sx0 + sx1)
        sy0 = sy[index]
        sy1 = sy[index + 1]
        sym = 0.5 * (sy0 + sy1)

        system0 = np.array((
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (-sx0, dg0, 0.0, 2.0 * g0),
            (-dg0, -sy0, -2.0 * g0, 0.0),
        ))
        systemm = np.array((
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (-sxm, dgm, 0.0, 2.0 * gm),
            (-dgm, -sym, -2.0 * gm, 0.0),
        ))
        system1 = np.array((
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (-sx1, dg1, 0.0, 2.0 * g1),
            (-dg1, -sy1, -2.0 * g1, 0.0),
        ))
        k1 = system0 @ matrix
        k2 = systemm @ (matrix + 0.5 * h * k1)
        k3 = systemm @ (matrix + 0.5 * h * k2)
        k4 = system1 @ (matrix + h * k3)
        matrix = matrix + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return matrix


@njit(cache=True)
def _rk4_transfer_matrices(g, dg, sx, sy, z_m, capture_indices):
    """Integrate once and retain maps at selected grid indices."""

    captured = np.empty((capture_indices.size, 4, 4), dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    capture = 0
    if capture_indices.size and capture_indices[0] == 0:
        captured[0] = matrix
        capture = 1
    for index in range(z_m.size - 1):
        h = z_m[index + 1] - z_m[index]
        g0 = g[index]
        g1 = g[index + 1]
        gm = 0.5 * (g0 + g1)
        dg0 = dg[index]
        dg1 = dg[index + 1]
        dgm = 0.5 * (dg0 + dg1)
        sx0 = sx[index]
        sx1 = sx[index + 1]
        sxm = 0.5 * (sx0 + sx1)
        sy0 = sy[index]
        sy1 = sy[index + 1]
        sym = 0.5 * (sy0 + sy1)

        system0 = np.array((
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (-sx0, dg0, 0.0, 2.0 * g0),
            (-dg0, -sy0, -2.0 * g0, 0.0),
        ))
        systemm = np.array((
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (-sxm, dgm, 0.0, 2.0 * gm),
            (-dgm, -sym, -2.0 * gm, 0.0),
        ))
        system1 = np.array((
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (-sx1, dg1, 0.0, 2.0 * g1),
            (-dg1, -sy1, -2.0 * g1, 0.0),
        ))
        k1 = system0 @ matrix
        k2 = systemm @ (matrix + 0.5 * h * k1)
        k3 = systemm @ (matrix + 0.5 * h * k2)
        k4 = system1 @ (matrix + h * k3)
        matrix = matrix + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        while (
            capture < capture_indices.size
            and capture_indices[capture] == index + 1
        ):
            captured[capture] = matrix
            capture += 1
    return captured


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
        self.z_mm = _piecewise_endpoint_exact_grid(
            source_z_mm, target_z_mm, step_mm, capture_z_mm
        )
        self.z_m = self.z_mm * 1.0e-3
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

        original = np.asarray(
            [float(lens.percent) for lens in self.lenses], dtype=float
        )
        try:
            for lens in self.lenses:
                lens.percent = 0.0
            fixed_b, sx, sy = fields(self.z_mm, state)
            profiles = []
            for lens, maximum in zip(self.lenses, self.upper):
                lens.percent = float(maximum)
                maximum_b = fields(self.z_mm, state)[0]
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
            np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2),
            np.array((0.0, 1.0)), np.array((1,), dtype=np.int64),
        )

    def _field_arrays(self, vector) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(vector, dtype=float)
        if values.shape != (len(self.variable_keys),):
            raise ValueError("Coupled lens vector has the wrong shape")
        magnetic = self.fixed_b_t + np.tensordot(
            values / 100.0, self.unit_profiles_t, axes=(0, 0)
        )
        g = np.ascontiguousarray(
            self.field_to_g_m1 * magnetic, dtype=np.float64
        )
        dg = np.ascontiguousarray(
            np.gradient(g, self.z_m, edge_order=1), dtype=np.float64
        )
        return g, dg

    def matrix(self, vector) -> np.ndarray:
        g, dg = self._field_arrays(vector)
        return _rk4_transfer_matrix(
            g, dg, self.sx_m2, self.sy_m2, self.z_m
        )

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
        g, dg = self._field_arrays(vector)
        return _rk4_transfer_matrices(
            g,
            dg,
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
                for aperture in state.apertures
                if bool(getattr(aperture, "enabled", True))
                and bool(getattr(aperture, "installed", True))
                and self.source_z_mm
                < float(aperture.z_mm)
                < self.sample_z_mm
            ),
        )
        self.apertures = tuple(
            aperture
            for aperture in sorted(state.apertures, key=lambda item: item.z_mm)
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
        matrices = self.sample_model.matrices_at(vector, capture_planes)
        for aperture, matrix in zip(self.apertures, matrices[:-1]):
            rays = matrix @ self.source_rays
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
        sample_rays = matrices[-1] @ self.source_rays
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
        stop_z_mm = float(determine_tem_stop_z(state))
        if definition.key == IMAGE_MAGNIFICATION:
            # Image presets are a coordinated five-lens solve.  There need
            # not be an isolated real image between every pair of lenses, so
            # the authoritative condition is the complete sample-to-recording
            # transfer B=0, with total signed magnification in A.
            self.plane_z_mm = None
            self.variable_keys = IMAGE_KEYS
            self.sample_model = _EquivalentImageFirstOrderModel(
                state,
                float(state.sample.z_mm),
                stop_z_mm,
            )
        else:
            objective = state.objective_lens
            plane_z_mm = objective.back_focal_plane_z_mm(
                state.beam_voltage_kv, state.sample
            )
            if plane_z_mm is None or not math.isfinite(float(plane_z_mm)):
                raise ValueError(
                    "The active Objective conjugate plane is undefined"
                )
            self.plane_z_mm = float(plane_z_mm)
            self.variable_keys = PROJECTOR_KEYS
            capture_z_mm = (self.plane_z_mm,)
            self.sample_model = _LiveFirstOrderModel(
                state,
                float(state.sample.z_mm),
                stop_z_mm,
                self.variable_keys,
                step_mm=step_mm,
                capture_z_mm=capture_z_mm,
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

        sample_to_plane, sample_matrix = self.sample_model.matrices_at(
            vector, (self.plane_z_mm, self.sample_model.z_mm[-1])
        )
        plane_matrix = np.linalg.solve(
            sample_to_plane.T, sample_matrix.T
        ).T
        value = self._isotropic_scale(sample_matrix[:2, 2:])
        relay_block = plane_matrix[:2, 2:]
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


def _get_vector(state, keys: tuple[str, ...]) -> np.ndarray:
    lenses = _lens_map(state)
    return np.asarray([float(lenses[key].percent) for key in keys], dtype=float)


def _set_vector(state, keys: tuple[str, ...], vector) -> None:
    lenses = _lens_map(state)
    for key, value in zip(keys, np.asarray(vector, dtype=float)):
        lenses[key].percent = float(value)


def _mode_matches(state, definition: DirectAlignmentDefinition) -> bool:
    if definition.mode_key == "nano_probe":
        return str(state.illumination_mode).upper() == "STEM"
    if definition.mode_key == "micro_probe":
        return str(state.illumination_mode).upper() == "TEM"
    if definition.mode_key == "imaging":
        return str(state.projector_mode).lower() == "image"
    if definition.mode_key == "diffraction":
        return str(state.projector_mode).lower() == "diffraction"
    return False


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

    raw_targets = definition.targets.get("preset_magnifications", ())
    raw_vectors = definition.targets.get("preset_vectors", ())
    try:
        targets = np.asarray(raw_targets, dtype=float)
        vectors = np.asarray(raw_vectors, dtype=float)
    except (TypeError, ValueError):
        return []
    if (
        targets.ndim != 1
        or vectors.shape != (targets.size, lower.size)
        or targets.size == 0
        or np.any(~np.isfinite(targets))
        or np.any(targets <= 0.0)
        or np.any(~np.isfinite(vectors))
    ):
        raise ValueError("Image preset seed table is invalid")
    lm_maximum = _target_number(
        definition, "lm_maximum_magnification", 1000.0
    )
    same_branch = (targets <= lm_maximum) == (float(target) <= lm_maximum)
    indices = np.flatnonzero(same_branch)
    indices = indices[
        np.argsort(np.abs(np.log(targets[indices] / float(target))))
    ]
    return [
        np.clip(vectors[index], lower, upper)
        for index in indices
    ]


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
    optimisation_upper = model.upper.copy()
    if definition.key == NANOPROBE_CONVERGENCE:
        constraint_scale = _target_number(
            definition, "maximum_waist_offset_mm", 0.002
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
        optimisation_upper[0] = min(70.0, optimisation_upper[0])

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
        np.minimum(initial, optimisation_upper),
        optimisation_upper,
        projector=False,
    )
    # Microprobe illuminated area has a much wider C2 solution curve than the
    # local preset neighbourhood.  Rank a small deterministic coarse grid and
    # refine only its best points; this keeps the GUI solve both global enough
    # and interactive.
    if definition.key == MICROPROBE_ILLUMINATION:
        coarse = []
        for c2 in np.linspace(20.0, optimisation_upper[0], 9):
            def curvature_residual(c3):
                statistics = model.measure((c2, float(c3[0])))
                return np.asarray((
                    statistics.radial_wavefront_curvature_per_m
                    / constraint_scale,
                ))

            c3_solution = least_squares(
                curvature_residual,
                np.asarray((initial[1],)),
                bounds=(
                    np.zeros(1),
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

    best_vector = initial.copy()
    best_cost = math.inf
    iterations = 0
    for seed in candidate_seeds:
        solution = least_squares(
            residual,
            seed,
            bounds=(np.zeros(initial.size), optimisation_upper),
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

    relay_scale_m = _target_number(
        definition, "maximum_relay_error_um", 20.0
    ) * 1.0e-6
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
            np.asarray(relay_block, dtype=float).ravel() / relay_scale_m,
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
                float(np.linalg.norm(relay_block, ord=2)) / relay_scale_m
            ) ** 2
        )
        if cost < best_cost:
            best_cost = cost
            best_vector = solution.x.copy()
        if (
            relative_error <= maximum_relative_error * 0.5
            and float(
                math.inf
                if measurement.relay_error_um is None
                else measurement.relay_error_um
            )
            <= relay_scale_m * 1.0e6 * 0.25
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
    vector = np.clip(initial, 0.0, model.upper)
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


def _validate_condenser(
    state,
    definition: DirectAlignmentDefinition,
    vector: np.ndarray,
    step_mm: float,
) -> DirectAlignmentMeasurement:
    model = _CondenserMeasurementModel(state, step_mm=step_mm)
    return _condenser_measurement(definition, model.measure(vector))


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
        if definition.key == IMAGE_MAGNIFICATION:
            plane_z_mm = None
        else:
            objective = state.objective_lens
            plane_z_mm = objective.back_focal_plane_z_mm(
                state.beam_voltage_kv, state.sample
            )
            if plane_z_mm is None or not math.isfinite(float(plane_z_mm)):
                raise ValueError(
                    "The active Objective conjugate plane is undefined"
                )
        stop_z_mm = float(determine_tem_stop_z(state))
        sample_transfer = trace_transverse_transfer(
            state, float(state.sample.z_mm), stop_z_mm
        )
        if definition.key == IMAGE_MAGNIFICATION:
            block = sample_transfer.j_img
            relay_block = sample_transfer.j_diff_m_per_rad
        else:
            block = sample_transfer.j_diff_m_per_rad
            plane_transfer = trace_transverse_transfer(
                state, float(plane_z_mm), stop_z_mm
            )
            relay_block = plane_transfer.j_diff_m_per_rad
        value = math.sqrt(abs(float(np.linalg.det(block))))
        relay_error_um = float(
            np.linalg.norm(relay_block, ord=2)
        ) * 1.0e6
    return DirectAlignmentMeasurement(
        key=definition.key,
        value=value,
        unit=definition.unit,
        constraint_value=relay_error_um,
        constraint_unit="um",
        relay_error_um=relay_error_um,
    )


def apply_direct_alignment(
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
        raise ValueError(
            f"{definition.name} is only active in {definition.mode_key} mode"
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
            candidate, iterations = _optimise_condenser(
                state, definition, requested
            )
            coarse = _validate_condenser(
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
            )
        else:
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
            DIFFRACTION_CAMERA_LENGTH: "BFP relay residual",
        }[definition.key]
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
            illumination_diameter_95_um=(
                fine.illumination_diameter_95_um
            ),
            relay_error_um=fine.relay_error_um,
        )
    except Exception:
        _set_vector(state, keys, initial)
        state.equivalent_image_lenses_enabled = (
            initial_equivalent_image_lenses
        )
        raise
