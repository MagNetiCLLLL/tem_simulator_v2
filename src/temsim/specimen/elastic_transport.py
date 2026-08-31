"""Event-driven elastic electron trajectories for finite TEM specimens.

The transport kernel is deliberately separate from the EDS ionisation model.
It generates weighted material paths that EDS can consume, while retaining a
small representative set of three-dimensional trajectories for inspection.

The present offline kernel uses the relativistic screened-Rutherford total
cross section and angular distribution described by Demers et al. for
100--300 keV STEM Monte Carlo work.  It is a provisional fallback, not an
ELSEPA-equivalent calculation; the public cross-section functions form the
replacement boundary for a future licensed/user-supplied ELSEPA table.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import xraylib

from temsim.detector.eds_signal import (
    EDSMaterial,
    ElectronTrackSegment,
    material_from_sample,
    material_from_support_grid,
)
from temsim.specimen.envelope import (
    canonical_sample_envelope_shape,
    envelope_contains_xy,
)
from temsim.specimen.support import SupportGrid, resolve_support_grid


RUTHERFORD_REFERENCE_URL = (
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC3165039/"
)
NIST_ELSEPA_REFERENCE_URL = (
    "https://www.nist.gov/publications/elsepa-dirac-partial-wave-calculation-"
    "elastic-scattering-electrons-and-positrons-atoms"
)
RUTHERFORD_MODEL_NAME = "relativistic_screened_rutherford_demers_2010"


def screened_rutherford_parameter(
    atomic_number: int, electron_energy_ev: float
) -> float:
    """Return the dimensionless screening parameter from Demers eq. 11."""

    z = int(atomic_number)
    energy_kev = float(electron_energy_ev) * 1.0e-3
    if not 1 <= z <= 99:
        raise ValueError("Elastic scattering requires Z=1..99")
    if not math.isfinite(energy_kev) or energy_kev <= 0.0:
        raise ValueError("Electron energy must be finite and positive")
    return 3.4e-3 * z ** (2.0 / 3.0) / energy_kev


def screened_rutherford_total_cross_section_cm2(
    atomic_number: int, electron_energy_ev: float
) -> float:
    """Relativistic screened-Rutherford total elastic cross section.

    Energy is accepted in eV and converted to keV for the published formula.
    The result is cm^2 per atom.  The source explicitly warns that this
    approximation becomes imprecise for elements heavier than Z=30.
    """

    z = int(atomic_number)
    energy_kev = float(electron_energy_ev) * 1.0e-3
    delta = screened_rutherford_parameter(z, electron_energy_ev)
    rest_energy_kev = 511.0
    return (
        5.21e-21
        * (z / energy_kev) ** 2
        * 4.0
        * math.pi
        / (delta * (1.0 + delta))
        * (energy_kev + rest_energy_kev)
        / (energy_kev + 2.0 * rest_energy_kev)
    )


def screened_rutherford_angle_cdf(
    theta_rad: float, atomic_number: int, electron_energy_ev: float
) -> float:
    """CDF of the normalized screened-Rutherford polar-angle law."""

    theta = float(theta_rad)
    if not math.isfinite(theta) or not 0.0 <= theta <= math.pi:
        raise ValueError("Elastic scattering angle must be in [0, pi]")
    delta = screened_rutherford_parameter(atomic_number, electron_energy_ev)
    u = math.sin(0.5 * theta) ** 2
    return (1.0 + delta) * u / (u + delta)


def screened_rutherford_differential_cross_section_cm2_sr(
    theta_rad: float, atomic_number: int, electron_energy_ev: float
) -> float:
    """Azimuthally symmetric differential elastic cross section."""

    theta = float(theta_rad)
    if not math.isfinite(theta) or not 0.0 <= theta <= math.pi:
        raise ValueError("Elastic scattering angle must be in [0, pi]")
    delta = screened_rutherford_parameter(atomic_number, electron_energy_ev)
    u = math.sin(0.5 * theta) ** 2
    probability_density_u = delta * (1.0 + delta) / (u + delta) ** 2
    return (
        screened_rutherford_total_cross_section_cm2(
            atomic_number, electron_energy_ev
        )
        * probability_density_u
        / (4.0 * math.pi)
    )


def sample_screened_rutherford_angle(
    rng: np.random.Generator,
    atomic_number: int,
    electron_energy_ev: float,
) -> tuple[float, float]:
    """Draw polar and azimuthal angles by analytical inverse transform."""

    delta = screened_rutherford_parameter(atomic_number, electron_energy_ev)
    random_cdf = float(rng.random())
    u = random_cdf * delta / (1.0 + delta - random_cdf)
    theta = 2.0 * math.asin(math.sqrt(min(max(u, 0.0), 1.0)))
    phi = 2.0 * math.pi * float(rng.random())
    return theta, phi


def elastic_scattering_rates_nm_inverse(
    material: EDSMaterial, electron_energy_ev: float
) -> tuple[tuple[int, float], ...]:
    """Macroscopic elastic rate by element in nm^-1."""

    rows = []
    for atomic_number, _fraction in material.mass_fractions:
        rate_cm_inverse = (
            material.atom_number_density_cm3(atomic_number)
            * screened_rutherford_total_cross_section_cm2(
                atomic_number, electron_energy_ev
            )
        )
        rows.append((atomic_number, rate_cm_inverse * 1.0e-7))
    return tuple(rows)


def elastic_mean_free_path_nm(
    material: EDSMaterial, electron_energy_ev: float
) -> float:
    """Compound/mixture mean free path using mass-fraction mixing."""

    total_rate = sum(
        rate
        for _atomic_number, rate in elastic_scattering_rates_nm_inverse(
            material, electron_energy_ev
        )
    )
    return math.inf if total_rate <= 0.0 else 1.0 / total_rate


def rotate_direction_after_scatter(
    direction: Iterable[float], theta_rad: float, phi_rad: float
) -> np.ndarray:
    """Rotate a unit flight direction by local polar/azimuthal angles."""

    forward = np.asarray(tuple(direction), dtype=float)
    if forward.shape != (3,) or not np.all(np.isfinite(forward)):
        raise ValueError("Electron direction must be a finite 3-vector")
    norm = float(np.linalg.norm(forward))
    if norm <= 0.0:
        raise ValueError("Electron direction cannot be zero")
    forward /= norm
    theta = float(theta_rad)
    phi = float(phi_rad)
    if not all(math.isfinite(value) for value in (theta, phi)):
        raise ValueError("Scattering angles must be finite")
    if not 0.0 <= theta <= math.pi:
        raise ValueError("Polar scattering angle must be in [0, pi]")
    reference = (
        np.asarray((0.0, 0.0, 1.0))
        if abs(float(forward[2])) < 0.9
        else np.asarray((1.0, 0.0, 0.0))
    )
    transverse_x = np.cross(reference, forward)
    transverse_x /= np.linalg.norm(transverse_x)
    transverse_y = np.cross(forward, transverse_x)
    result = (
        math.cos(theta) * forward
        + math.sin(theta)
        * (
            math.cos(phi) * transverse_x
            + math.sin(phi) * transverse_y
        )
    )
    result /= np.linalg.norm(result)
    return result


@dataclass(frozen=True, slots=True)
class ElasticScatterEvent:
    position_nm: tuple[float, float, float]
    source_key: str
    material_key: str
    atomic_number: int
    theta_rad: float
    phi_rad: float


@dataclass(frozen=True, slots=True)
class IncidentElectronRay:
    """One electron history sampled by the upstream column calculation.

    ``position_xy_nm`` is the position at the physical sample plane. The
    direction is a unit vector in the column coordinate system and therefore
    retains both incident tilt and accumulated Larmor rotation. ``weight`` is
    conditional on the electron having reached the sample plane.
    """

    source_ray_index: int
    position_xy_nm: tuple[float, float]
    direction: tuple[float, float, float]
    kinetic_energy_ev: float
    weight: float

    def __post_init__(self) -> None:
        position = np.asarray(self.position_xy_nm, dtype=float)
        direction = np.asarray(self.direction, dtype=float)
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            raise ValueError("Incident electron position must be a finite 2-vector")
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("Incident electron direction must be a finite 3-vector")
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0 or not math.isclose(norm, 1.0, rel_tol=1.0e-10):
            raise ValueError("Incident electron direction must be a unit vector")
        if (
            int(self.source_ray_index) < 0
            or not math.isfinite(float(self.kinetic_energy_ev))
            or float(self.kinetic_energy_ev) <= 0.0
            or not math.isfinite(float(self.weight))
            or float(self.weight) < 0.0
        ):
            raise ValueError("Incident electron index, energy or weight is invalid")


@dataclass(frozen=True, slots=True)
class IncidentRayBundle:
    """The exact upstream ray histories that survive to the sample plane."""

    rays: tuple[IncidentElectronRay, ...]
    emitted_ray_count: int
    reaching_ray_count: int
    surviving_fraction: float
    original_centroid_nm: tuple[float, float]
    target_centroid_nm: tuple[float, float]
    chief_angle_mrad: tuple[float, float]
    energy_range_ev: tuple[float, float]
    boundary_z_mm: float | None = None


@dataclass(frozen=True, slots=True)
class ElasticTrajectory:
    points_nm: np.ndarray
    events: tuple[ElasticScatterEvent, ...]
    outcome: str
    total_material_path_nm: float
    source_ray_index: int = 0
    incident_weight: float = 1.0
    initial_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    initial_energy_ev: float = 0.0

    def __post_init__(self) -> None:
        points = np.asarray(self.points_nm, dtype=float)
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or not np.all(np.isfinite(points))
        ):
            raise ValueError("Elastic trajectory points must be finite N by 3")
        points.setflags(write=False)
        object.__setattr__(self, "points_nm", points)


def incident_rays_from_simulation(
    state,
    simulation,
    *,
    target_x_nm: float | None = None,
    target_y_nm: float | None = None,
    boundary_z_mm: float | None = None,
) -> IncidentRayBundle:
    """Extract weighted phase space at one upstream/sample boundary.

    Every ray that survives the gun, apertures and column wall to the physical
    sample plane is used once. A point acquisition translates the weighted
    beam centroid to its requested scan coordinate without changing the
    calculated spread, direction, rotation, energy or current weight.
    """

    if simulation is None or getattr(simulation, "incident", None) is None:
        raise ValueError(
            "Elastic EDS transport requires a completed column calculation."
        )
    branch = simulation.incident
    final_alive = np.asarray(branch.alive, dtype=bool)
    if final_alive.ndim != 1:
        raise ValueError("Incident sample-plane survival mask must be one-dimensional")
    emitted_count = int(final_alive.size)
    has_axial_history = hasattr(branch, "z")
    branch_z = np.asarray(
        branch.z if has_axial_history else (float(state.sample.z_mm),),
        dtype=float,
    )
    requested_z = (
        float(state.sample.z_mm)
        if boundary_z_mm is None
        else float(boundary_z_mm)
    )
    if (
        branch_z.ndim != 1
        or branch_z.size < 1
        or not np.all(np.isfinite(branch_z))
        or np.any(np.diff(branch_z) < 0.0)
        or not math.isfinite(requested_z)
        or requested_z < float(branch_z[0]) - 1.0e-9
        or requested_z > float(branch_z[-1]) + 1.0e-9
    ):
        raise ValueError("Requested incident boundary is outside the ray history")
    requested_z = min(max(requested_z, float(branch_z[0])), float(branch_z[-1]))
    if hasattr(branch, "blocked_z"):
        blocked_z = np.asarray(branch.blocked_z, dtype=float)
        if blocked_z.shape != final_alive.shape:
            raise ValueError(
                "Incident blocking coordinates do not match the ray bundle"
            )
        reaches_boundary = np.isnan(blocked_z) | (
            blocked_z > requested_z + 1.0e-9
        )
    elif boundary_z_mm is None:
        # Compact synthetic/test bundles historically expose only their final
        # sample-plane mask and one phase-space row.
        reaches_boundary = final_alive.copy()
    else:
        raise ValueError(
            "An upstream incident boundary requires axial history and block Z data"
        )
    indices = np.flatnonzero(reaches_boundary)
    if indices.size == 0:
        raise ValueError("No electron rays reach the requested incident boundary.")

    def sample_row(values, label: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or array.shape[1] != emitted_count:
            raise ValueError(f"Incident {label} array does not match the ray bundle")
        if branch_z.size == 1 or math.isclose(
            requested_z, float(branch_z[-1]), abs_tol=1.0e-12
        ):
            row = np.asarray(array[-1, indices], dtype=float)
        else:
            hi = int(np.searchsorted(branch_z, requested_z, side="left"))
            if hi == 0:
                row = np.asarray(array[0, indices], dtype=float)
            elif math.isclose(
                requested_z, float(branch_z[hi]), abs_tol=1.0e-12
            ):
                row = np.asarray(array[hi, indices], dtype=float)
            else:
                lo = hi - 1
                fraction = (requested_z - branch_z[lo]) / (
                    branch_z[hi] - branch_z[lo]
                )
                row = np.asarray(
                    array[lo, indices]
                    + fraction * (array[hi, indices] - array[lo, indices]),
                    dtype=float,
                )
        if not np.all(np.isfinite(row)):
            raise ValueError(f"Incident {label} contains NaN or infinity")
        return row

    x_nm = sample_row(branch.x, "X position") * 1.0e9
    y_nm = sample_row(branch.y, "Y position") * 1.0e9
    tx = sample_row(branch.tx, "X slope")
    ty = sample_row(branch.ty, "Y slope")
    energy_offset = np.asarray(branch.energy_offset_ev, dtype=float)
    if energy_offset.shape != final_alive.shape or not np.all(
        np.isfinite(energy_offset[indices])
    ):
        raise ValueError("Incident energy offsets do not match the ray bundle")

    raw_weight = getattr(branch, "ray_weight", None)
    if raw_weight is None:
        weights = np.full(emitted_count, 1.0 / emitted_count, dtype=float)
    else:
        weights = np.asarray(raw_weight, dtype=float)
        if (
            weights.shape != final_alive.shape
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or float(weights.sum()) <= 0.0
        ):
            raise ValueError("Incident ray weights must be finite and non-negative")
        weights = weights / float(weights.sum())
    surviving_fraction = float(weights[indices].sum())
    if surviving_fraction <= 0.0:
        raise ValueError("No positive source current reaches the sample plane")
    conditional = weights[indices] / surviving_fraction

    original_centroid = (
        float(np.sum(conditional * x_nm)),
        float(np.sum(conditional * y_nm)),
    )
    target_centroid = (
        original_centroid[0] if target_x_nm is None else float(target_x_nm),
        original_centroid[1] if target_y_nm is None else float(target_y_nm),
    )
    if not all(math.isfinite(value) for value in target_centroid):
        raise ValueError("EDS target coordinate must be finite")
    x_nm = x_nm + target_centroid[0] - original_centroid[0]
    y_nm = y_nm + target_centroid[1] - original_centroid[1]

    energy_ev = float(state.beam_voltage_kv) * 1000.0 + energy_offset[indices]
    if not np.all(np.isfinite(energy_ev)) or np.any(energy_ev <= 0.0):
        raise ValueError("Incident kinetic energies must be finite and positive")
    direction_rows = np.column_stack((tx, ty, np.ones_like(tx)))
    direction_rows /= np.linalg.norm(direction_rows, axis=1)[:, None]
    rays = tuple(
        IncidentElectronRay(
            source_ray_index=int(source_index),
            position_xy_nm=(float(x_value), float(y_value)),
            direction=tuple(float(value) for value in direction),
            kinetic_energy_ev=float(energy),
            weight=float(weight),
        )
        for source_index, x_value, y_value, direction, energy, weight in zip(
            indices,
            x_nm,
            y_nm,
            direction_rows,
            energy_ev,
            conditional,
            strict=True,
        )
    )
    return IncidentRayBundle(
        rays=rays,
        emitted_ray_count=emitted_count,
        reaching_ray_count=len(rays),
        surviving_fraction=surviving_fraction,
        original_centroid_nm=original_centroid,
        target_centroid_nm=target_centroid,
        chief_angle_mrad=(
            float(np.sum(conditional * tx) * 1.0e3),
            float(np.sum(conditional * ty) * 1.0e3),
        ),
        energy_range_ev=(float(np.min(energy_ev)), float(np.max(energy_ev))),
        boundary_z_mm=requested_z,
    )


@dataclass(frozen=True, slots=True)
class ElasticMaterialFlight:
    """One stored straight flight that is known to be inside material."""

    start_nm: tuple[float, float, float]
    end_nm: tuple[float, float, float]
    source_key: str
    material_key: str
    history: str
    source_ray_index: int
    electron_energy_ev: float
    electron_weight: float


@dataclass(frozen=True, slots=True)
class ElasticTerminalBundle:
    """Terminal state of every simulated electron history."""

    source_ray_index: np.ndarray
    position_nm: np.ndarray
    direction: np.ndarray
    kinetic_energy_ev: np.ndarray
    weight: np.ndarray
    outcome: tuple[str, ...]
    event_count: np.ndarray
    has_scattered: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.outcome)
        arrays = {
            "source_ray_index": np.asarray(self.source_ray_index, dtype=np.int64),
            "position_nm": np.asarray(self.position_nm, dtype=float),
            "direction": np.asarray(self.direction, dtype=float),
            "kinetic_energy_ev": np.asarray(self.kinetic_energy_ev, dtype=float),
            "weight": np.asarray(self.weight, dtype=float),
            "event_count": np.asarray(self.event_count, dtype=np.int64),
            "has_scattered": np.asarray(self.has_scattered, dtype=bool),
        }
        expected_shapes = {
            "source_ray_index": (count,),
            "position_nm": (count, 3),
            "direction": (count, 3),
            "kinetic_energy_ev": (count,),
            "weight": (count,),
            "event_count": (count,),
            "has_scattered": (count,),
        }
        for name, array in arrays.items():
            if array.shape != expected_shapes[name]:
                raise ValueError(f"Elastic terminal {name} has an invalid shape")
            if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
                raise ValueError(f"Elastic terminal {name} contains NaN or infinity")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if (
            np.any(arrays["source_ray_index"] < 0)
            or np.any(arrays["kinetic_energy_ev"] <= 0.0)
            or np.any(arrays["weight"] < 0.0)
            or np.any(arrays["event_count"] < 0)
            or not np.allclose(
                np.linalg.norm(arrays["direction"], axis=1),
                1.0,
                rtol=1.0e-10,
                atol=1.0e-12,
            )
        ):
            raise ValueError("Elastic terminal electron values are invalid")


@dataclass(frozen=True, slots=True)
class ElasticTransportResult:
    eds_tracks: tuple[ElectronTrackSegment, ...]
    trajectories: tuple[ElasticTrajectory, ...]
    metrics: dict[str, object]
    material_flights: tuple[ElasticMaterialFlight, ...] = ()
    terminal_electrons: ElasticTerminalBundle | None = None


@dataclass(frozen=True, slots=True)
class _MaterialRegion:
    source_key: str
    material: EDSMaterial
    emitting_layer_thickness_nm: float


class ElasticTransportGeometry:
    """Finite rectangular specimen plus an immediately downstream grid.

    The support-grid plane is placed in contact with the specimen's downstream
    face.  Its square openings, crossed bars, annular rim and circular outer
    boundary are all evaluated in continuous 3-D, including lateral crossings
    caused by elastic deflection.
    """

    def __init__(
        self,
        *,
        inserted: bool,
        sample_material: EDSMaterial | None,
        sample_centre_xy_nm: tuple[float, float],
        sample_size_xy_nm: tuple[float, float],
        sample_thickness_nm: float,
        support_grid: SupportGrid,
        support_material: EDSMaterial | None,
        support_offset_xy_um: tuple[float, float],
        support_rotation_deg: float,
        sample_envelope_shape: str = "rectangle",
    ) -> None:
        self.inserted = bool(inserted)
        self.sample_material = sample_material if self.inserted else None
        self.sample_centre_xy_nm = tuple(
            float(value) for value in sample_centre_xy_nm
        )
        self.sample_size_xy_nm = tuple(float(value) for value in sample_size_xy_nm)
        self.sample_envelope_shape = canonical_sample_envelope_shape(
            sample_envelope_shape
        )
        self.sample_thickness_nm = max(float(sample_thickness_nm), 0.0)
        self.support_grid = support_grid
        self.support_material = support_material if self.inserted else None
        self.support_offset_xy_um = tuple(
            float(value) for value in support_offset_xy_um
        )
        self.support_rotation_deg = float(support_rotation_deg)
        positive_dimensions = tuple(
            value
            for value in (
                *self.sample_size_xy_nm,
                self.sample_thickness_nm,
                self.support_grid.foil_thickness_um * 1000.0,
            )
            if value > 0.0
        )
        minimum_dimension = min(positive_dimensions, default=1.0)
        self.epsilon_nm = max(1.0e-12, min(1.0e-5, minimum_dimension * 1.0e-6))

    @classmethod
    def from_state(cls, state) -> "ElasticTransportGeometry":
        sample = state.sample
        grid = resolve_support_grid(
            str(getattr(sample, "eds_support_material_key", "vacuum")),
            str(getattr(sample, "eds_support_mesh_key", "square_200")),
        )
        return cls(
            inserted=bool(getattr(sample, "inserted", True)),
            sample_material=material_from_sample(state),
            sample_centre_xy_nm=(
                float(getattr(sample, "centre_x_nm", 0.0)),
                float(getattr(sample, "centre_y_nm", 0.0)),
            ),
            sample_size_xy_nm=(
                float(getattr(sample, "size_x_nm", 0.0)),
                float(getattr(sample, "size_y_nm", 0.0)),
            ),
            sample_thickness_nm=float(getattr(sample, "thickness_nm", 0.0)),
            support_grid=grid,
            support_material=material_from_support_grid(grid),
            support_offset_xy_um=(
                float(getattr(sample, "eds_support_offset_x_um", 0.0)),
                float(getattr(sample, "eds_support_offset_y_um", 0.0)),
            ),
            support_rotation_deg=float(
                getattr(sample, "eds_support_rotation_deg", 0.0)
            ),
            sample_envelope_shape=str(
                getattr(sample, "envelope_shape", "rectangle")
            ),
        )

    @property
    def support_top_nm(self) -> float:
        return self.sample_thickness_nm

    @property
    def support_bottom_nm(self) -> float:
        return self.support_top_nm + self.support_grid.foil_thickness_um * 1000.0

    @property
    def transport_span_nm(self) -> float:
        return max(
            self.support_grid.outer_diameter_mm * 1000.0 * 1000.0,
            self.support_bottom_nm,
            *self.sample_size_xy_nm,
            1.0,
        )

    def region_at(self, position_nm: Iterable[float]) -> _MaterialRegion | None:
        position = np.asarray(tuple(position_nm), dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("Transport position must be a finite 3-vector")
        x_nm, y_nm, z_nm = (float(value) for value in position)
        centre_x, centre_y = self.sample_centre_xy_nm
        if (
            self.sample_material is not None
            and 0.0 <= z_nm <= self.sample_thickness_nm
            and envelope_contains_xy(
                self.sample_envelope_shape,
                x_nm,
                y_nm,
                centre_xy_nm=(centre_x, centre_y),
                size_xy_nm=self.sample_size_xy_nm,
            )
        ):
            return _MaterialRegion(
                "sample",
                self.sample_material,
                self.sample_thickness_nm,
            )
        if (
            self.support_material is not None
            and self.support_top_nm <= z_nm <= self.support_bottom_nm
        ):
            region = self.support_grid.region_at_nm(
                x_nm,
                y_nm,
                offset_x_um=self.support_offset_xy_um[0],
                offset_y_um=self.support_offset_xy_um[1],
                rotation_deg=self.support_rotation_deg,
            )
            if region in {"bar", "rim"}:
                return _MaterialRegion(
                    f"support:{region}",
                    self.support_material,
                    self.support_grid.foil_thickness_um * 1000.0,
                )
        return None

    @staticmethod
    def _region_identity(region: _MaterialRegion | None) -> tuple[str, str] | None:
        if region is None:
            return None
        return region.source_key, region.material.key

    def _surface_candidates(
        self,
        position: np.ndarray,
        direction: np.ndarray,
        maximum_distance_nm: float,
    ) -> list[float]:
        epsilon = self.epsilon_nm
        maximum = float(maximum_distance_nm)
        candidates: list[float] = []

        def add_plane(axis: int, value: float) -> None:
            component = float(direction[axis])
            if abs(component) <= 1.0e-15:
                return
            distance = (float(value) - float(position[axis])) / component
            if epsilon < distance <= maximum + epsilon:
                candidates.append(distance)

        if self.sample_material is not None:
            centre_x, centre_y = self.sample_centre_xy_nm
            if self.sample_envelope_shape == "rectangle":
                half_x = 0.5 * self.sample_size_xy_nm[0]
                half_y = 0.5 * self.sample_size_xy_nm[1]
                for value in (centre_x - half_x, centre_x + half_x):
                    add_plane(0, value)
                for value in (centre_y - half_y, centre_y + half_y):
                    add_plane(1, value)
            else:
                radius_x = 0.5 * self.sample_size_xy_nm[0]
                radius_y = 0.5 * self.sample_size_xy_nm[1]
                relative_x = float(position[0]) - centre_x
                relative_y = float(position[1]) - centre_y
                direction_x = float(direction[0])
                direction_y = float(direction[1])
                quadratic = (
                    (direction_x / radius_x) ** 2
                    + (direction_y / radius_y) ** 2
                )
                if quadratic > 1.0e-30:
                    linear = 2.0 * (
                        relative_x * direction_x / radius_x**2
                        + relative_y * direction_y / radius_y**2
                    )
                    constant = (
                        (relative_x / radius_x) ** 2
                        + (relative_y / radius_y) ** 2
                        - 1.0
                    )
                    discriminant = linear**2 - 4.0 * quadratic * constant
                    if discriminant >= 0.0:
                        root = math.sqrt(max(discriminant, 0.0))
                        for distance in (
                            (-linear - root) / (2.0 * quadratic),
                            (-linear + root) / (2.0 * quadratic),
                        ):
                            if epsilon < distance <= maximum + epsilon:
                                candidates.append(distance)
            add_plane(2, 0.0)
            add_plane(2, self.sample_thickness_nm)

        if self.support_material is None:
            return candidates

        add_plane(2, self.support_top_nm)
        add_plane(2, self.support_bottom_nm)
        offset_nm = np.asarray(self.support_offset_xy_um, dtype=float) * 1000.0
        relative_xy = position[:2] - offset_nm
        direction_xy = direction[:2]
        quadratic = float(direction_xy @ direction_xy)
        if quadratic > 1.0e-30:
            for radius_nm in (
                0.5 * self.support_grid.outer_diameter_mm * 1.0e6,
                0.5 * self.support_grid.outer_diameter_mm * 1.0e6
                - self.support_grid.rim_width_um * 1000.0,
            ):
                linear = 2.0 * float(relative_xy @ direction_xy)
                constant = float(relative_xy @ relative_xy) - radius_nm**2
                discriminant = linear * linear - 4.0 * quadratic * constant
                if discriminant < 0.0:
                    continue
                root = math.sqrt(max(discriminant, 0.0))
                for distance in (
                    (-linear - root) / (2.0 * quadratic),
                    (-linear + root) / (2.0 * quadratic),
                ):
                    if epsilon < distance <= maximum + epsilon:
                        candidates.append(distance)

        # Vertical mesh sidewalls.  Generate only surfaces intersected within
        # the requested flight; material flights normally span far less than a
        # pitch, while a vacuum opening changes region at its first sidewall.
        angle = math.radians(self.support_rotation_deg)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        local_position = np.asarray(
            (
                cosine * relative_xy[0] + sine * relative_xy[1],
                -sine * relative_xy[0] + cosine * relative_xy[1],
            ),
            dtype=float,
        )
        local_direction = np.asarray(
            (
                cosine * direction[0] + sine * direction[1],
                -sine * direction[0] + cosine * direction[1],
            ),
            dtype=float,
        )
        period_nm = self.support_grid.mesh.pitch_um * 1000.0
        half_hole_nm = 0.5 * self.support_grid.mesh.hole_width_um * 1000.0
        for coordinate, component in zip(
            local_position, local_direction, strict=True
        ):
            if abs(float(component)) <= 1.0e-15:
                continue
            endpoint = float(coordinate + component * maximum)
            lower = min(float(coordinate), endpoint)
            upper = max(float(coordinate), endpoint)
            for phase in (-half_hole_nm, half_hole_nm):
                first = math.ceil((lower - phase) / period_nm)
                last = math.floor((upper - phase) / period_nm)
                # A nearly horizontal ray can cross many cells.  The finite
                # circular grid bounds useful crossings; the guard is only a
                # defence against pathological caller-supplied path limits.
                if last - first > 4096:
                    last = first + 4096
                for index in range(first, last + 1):
                    boundary = phase + index * period_nm
                    distance = (boundary - float(coordinate)) / float(component)
                    if epsilon < distance <= maximum + epsilon:
                        candidates.append(distance)
        return candidates

    def next_region_boundary_distance_nm(
        self,
        position_nm: Iterable[float],
        direction: Iterable[float],
        current_region: _MaterialRegion | None,
        *,
        maximum_distance_nm: float,
    ) -> float | None:
        """Return the first surface at which the material identity changes."""

        position = np.asarray(tuple(position_nm), dtype=float)
        flight = np.asarray(tuple(direction), dtype=float)
        maximum = float(maximum_distance_nm)
        if (
            position.shape != (3,)
            or flight.shape != (3,)
            or not np.all(np.isfinite(position))
            or not np.all(np.isfinite(flight))
            or not math.isfinite(maximum)
            or maximum <= 0.0
        ):
            raise ValueError("Boundary query requires finite position, direction and range")
        flight_norm = float(np.linalg.norm(flight))
        if flight_norm <= 0.0:
            raise ValueError("Boundary query direction cannot be zero")
        flight /= flight_norm
        candidates = sorted(
            self._surface_candidates(position, flight, maximum)
        )
        if not candidates:
            return None
        current_identity = self._region_identity(current_region)
        grouped: list[float] = []
        for distance in candidates:
            if not grouped or not math.isclose(
                distance,
                grouped[-1],
                rel_tol=1.0e-11,
                abs_tol=4.0 * self.epsilon_nm,
            ):
                grouped.append(distance)
        for distance in grouped:
            probe_distance = min(
                distance + 4.0 * self.epsilon_nm,
                maximum + 4.0 * self.epsilon_nm,
            )
            after = self.region_at(position + flight * probe_distance)
            if self._region_identity(after) != current_identity:
                return min(distance, maximum)
        return None


def _choose_scatterer(
    rng: np.random.Generator,
    rates: tuple[tuple[int, float], ...],
) -> int:
    total = sum(rate for _atomic_number, rate in rates)
    if total <= 0.0:
        raise ValueError("Elastic scattering material has zero event rate")
    threshold = float(rng.random()) * total
    cumulative = 0.0
    for atomic_number, rate in rates:
        cumulative += rate
        if threshold <= cumulative:
            return atomic_number
    return rates[-1][0]


def _history_for_region(
    region: _MaterialRegion, has_scattered: bool
) -> str:
    if has_scattered:
        return "elastic_scattered"
    if region.source_key.startswith("support:"):
        return "straight_primary_after_sample"
    return "straight_primary"


def _terminal_outcome(direction: np.ndarray) -> str:
    if float(direction[2]) > 1.0e-12:
        return "transmitted"
    if float(direction[2]) < -1.0e-12:
        return "backscattered"
    return "lateral_escape"


def simulate_elastic_point_transport(
    state,
    *,
    incident_rays: Iterable[IncidentElectronRay],
    seed: int | None = None,
    maximum_events_per_trajectory: int | None = None,
    stored_trajectory_count: int = 256,
) -> ElasticTransportResult:
    """Trace the calculated incident ray bundle through sample and support.

    Elastic collisions change direction but not kinetic energy.  The returned
    EDS tracks preserve each source ray's conditional current weight and
    kinetic energy. The history count equals the number of upstream rays that
    actually reached the physical sample plane.
    """

    sample = state.sample
    rays = tuple(incident_rays)
    count = len(rays)
    random_seed = int(
        getattr(sample, "eds_elastic_seed", 0) if seed is None else seed
    )
    maximum_events = int(
        getattr(sample, "eds_elastic_max_events", 10_000)
        if maximum_events_per_trajectory is None
        else maximum_events_per_trajectory
    )
    stored_count = int(stored_trajectory_count)
    if count <= 0:
        raise ValueError("Elastic transport requires at least one incident ray")
    if random_seed < 0:
        raise ValueError("Elastic trajectory seed cannot be negative")
    if maximum_events <= 0:
        raise ValueError("Elastic maximum event count must be positive")
    if stored_count < 0:
        raise ValueError("Stored elastic trajectory count cannot be negative")
    input_weights = np.asarray([ray.weight for ray in rays], dtype=float)
    if (
        not np.all(np.isfinite(input_weights))
        or np.any(input_weights < 0.0)
        or float(input_weights.sum()) <= 0.0
    ):
        raise ValueError("Incident ray weights must be finite and positive in total")
    input_weights /= float(input_weights.sum())
    geometry = ElasticTransportGeometry.from_state(state)
    rng = np.random.default_rng(random_seed)
    maximum_path_nm = 8.0 * geometry.transport_span_nm
    tracks: list[ElectronTrackSegment] = []
    outcome_counts = {
        "transmitted": 0,
        "backscattered": 0,
        "lateral_escape": 0,
        "event_limit": 0,
        "path_limit": 0,
    }
    total_events = 0
    weighted_total_events = 0.0
    weighted_total_material_path = 0.0
    source_path_sums: dict[str, float] = {}
    stored_trajectories: list[ElasticTrajectory] = []
    stored_material_flights: list[ElasticMaterialFlight] = []
    terminal_source_indices: list[int] = []
    terminal_positions: list[np.ndarray] = []
    terminal_directions: list[np.ndarray] = []
    terminal_energies: list[float] = []
    terminal_weights: list[float] = []
    terminal_outcomes: list[str] = []
    terminal_event_counts: list[int] = []
    terminal_scattered: list[bool] = []
    encountered_atomic_numbers: set[int] = set()
    outcome_weights = {key: 0.0 for key in outcome_counts}

    for trajectory_index, (ray, ray_weight) in enumerate(
        zip(rays, input_weights, strict=True)
    ):
        direction = np.asarray(ray.direction, dtype=float)
        position = np.asarray(
            (
                float(ray.position_xy_nm[0]),
                float(ray.position_xy_nm[1]),
                -8.0 * geometry.epsilon_nm,
            ),
            dtype=float,
        )
        energy_ev = float(ray.kinetic_energy_ev)
        points = [position.copy()] if trajectory_index < stored_count else None
        events: list[ElasticScatterEvent] | None = (
            [] if trajectory_index < stored_count else None
        )
        event_count = 0
        travelled_path = 0.0
        trajectory_material_path = 0.0
        trajectory_paths: dict[
            tuple[str, str, str, float], tuple[EDSMaterial, float]
        ] = {}
        has_scattered = False
        outcome = "path_limit"
        while travelled_path < maximum_path_nm:
            region = geometry.region_at(position)
            if region is None:
                remaining = maximum_path_nm - travelled_path
                boundary = geometry.next_region_boundary_distance_nm(
                    position,
                    direction,
                    None,
                    maximum_distance_nm=remaining,
                )
                if boundary is None:
                    outcome = _terminal_outcome(direction)
                    break
                boundary_point = position + direction * boundary
                travelled_path += boundary
                if points is not None:
                    points.append(boundary_point.copy())
                position = boundary_point + direction * geometry.epsilon_nm
                continue

            rates = elastic_scattering_rates_nm_inverse(region.material, energy_ev)
            total_rate = sum(rate for _atomic_number, rate in rates)
            if total_rate <= 0.0:
                raise ValueError(
                    f"Material {region.material.key} has no elastic scattering rate"
                )
            free_path = float(rng.exponential(1.0 / total_rate))
            remaining = maximum_path_nm - travelled_path
            flight_limit = min(free_path, remaining)
            boundary = geometry.next_region_boundary_distance_nm(
                position,
                direction,
                region,
                maximum_distance_nm=flight_limit,
            )
            distance = boundary if boundary is not None else flight_limit
            history = _history_for_region(region, has_scattered)
            key = (
                region.source_key,
                region.material.key,
                history,
                region.emitting_layer_thickness_nm,
            )
            old_material, old_path = trajectory_paths.get(
                key, (region.material, 0.0)
            )
            trajectory_paths[key] = (old_material, old_path + distance)
            trajectory_material_path += distance
            source_path_sums[region.source_key] = (
                source_path_sums.get(region.source_key, 0.0)
                + float(ray_weight) * distance
            )
            flight_start = position.copy()
            endpoint = position + direction * distance
            if points is not None and distance > 0.0:
                stored_material_flights.append(
                    ElasticMaterialFlight(
                        start_nm=tuple(float(value) for value in flight_start),
                        end_nm=tuple(float(value) for value in endpoint),
                        source_key=region.source_key,
                        material_key=region.material.key,
                        history=history,
                        source_ray_index=int(ray.source_ray_index),
                        electron_energy_ev=energy_ev,
                        electron_weight=float(ray_weight),
                    )
                )
            travelled_path += distance
            if points is not None:
                points.append(endpoint.copy())
            position = endpoint
            if boundary is not None:
                position = endpoint + direction * geometry.epsilon_nm
                continue
            if free_path > remaining:
                outcome = "path_limit"
                break
            atomic_number = _choose_scatterer(rng, rates)
            encountered_atomic_numbers.add(atomic_number)
            theta, phi = sample_screened_rutherford_angle(
                rng, atomic_number, energy_ev
            )
            if events is not None:
                events.append(
                    ElasticScatterEvent(
                        position_nm=tuple(float(value) for value in position),
                        source_key=region.source_key,
                        material_key=region.material.key,
                        atomic_number=atomic_number,
                        theta_rad=theta,
                        phi_rad=phi,
                    )
                )
            direction = rotate_direction_after_scatter(direction, theta, phi)
            has_scattered = True
            event_count += 1
            total_events += 1
            if event_count >= maximum_events:
                outcome = "event_limit"
                break
        outcome_counts[outcome] += 1
        outcome_weights[outcome] += float(ray_weight)
        terminal_source_indices.append(int(ray.source_ray_index))
        terminal_positions.append(position.copy())
        terminal_directions.append(direction.copy())
        terminal_energies.append(energy_ev)
        terminal_weights.append(float(ray_weight))
        terminal_outcomes.append(outcome)
        terminal_event_counts.append(event_count)
        terminal_scattered.append(has_scattered)
        weighted_total_events += float(ray_weight) * event_count
        weighted_total_material_path += (
            float(ray_weight) * trajectory_material_path
        )
        for (
            source_key,
            _material_key,
            history,
            layer_thickness,
        ), (material, path_length) in trajectory_paths.items():
            if path_length <= 0.0 or ray_weight <= 0.0:
                continue
            tracks.append(
                ElectronTrackSegment(
                    source_key=source_key,
                    material=material,
                    path_length_nm=path_length,
                    electron_energy_ev=energy_ev,
                    electron_weight=float(ray_weight),
                    emitting_layer_thickness_nm=layer_thickness,
                    history=history,
                )
            )
        if points is not None and events is not None:
            stored_trajectories.append(
                ElasticTrajectory(
                    points_nm=np.asarray(points, dtype=float),
                    events=tuple(events),
                    outcome=outcome,
                    total_material_path_nm=trajectory_material_path,
                    source_ray_index=int(ray.source_ray_index),
                    incident_weight=float(ray_weight),
                    initial_direction=tuple(
                        float(value) for value in ray.direction
                    ),
                    initial_energy_ev=energy_ev,
                )
            )

    energies = np.asarray([ray.kinetic_energy_ev for ray in rays], dtype=float)
    directions = np.asarray([ray.direction for ray in rays], dtype=float)
    metrics: dict[str, object] = {
        "elastic_trajectory_generation": True,
        "electron_transport_model": RUTHERFORD_MODEL_NAME,
        "elastic_cross_section_reference": RUTHERFORD_REFERENCE_URL,
        "recommended_high_accuracy_model": "ELSEPA Dirac partial-wave",
        "recommended_high_accuracy_reference": NIST_ELSEPA_REFERENCE_URL,
        "trajectory_count": count,
        "trajectory_seed": random_seed,
        "stored_trajectory_count": len(stored_trajectories),
        "maximum_events_per_trajectory": maximum_events,
        "total_elastic_events": total_events,
        "mean_elastic_events_per_trajectory": weighted_total_events,
        "unweighted_mean_elastic_events_per_trajectory": total_events / count,
        "mean_material_path_nm": weighted_total_material_path,
        "mean_material_path_by_source_nm": {
            key: value for key, value in sorted(source_path_sums.items())
        },
        "outcome_counts": dict(outcome_counts),
        "outcome_weight_fractions": dict(outcome_weights),
        "transmitted_fraction": outcome_weights["transmitted"],
        "backscattered_fraction": outcome_weights["backscattered"],
        "lateral_escape_fraction": outcome_weights["lateral_escape"],
        "event_limit_fraction": outcome_weights["event_limit"],
        "path_limit_fraction": outcome_weights["path_limit"],
        "initial_beam_model": "calculated_surviving_sample_plane_phase_space",
        "incident_energy_range_ev": (
            float(np.min(energies)), float(np.max(energies))
        ),
        "incident_chief_direction": tuple(
            float(value)
            for value in np.sum(input_weights[:, None] * directions, axis=0)
        ),
        "incident_weights_normalised": True,
        "elastic_energy_loss_included": False,
        "nuclear_recoil_included": False,
        "inelastic_angular_deflection_included": False,
        "support_contact_assumption": (
            "support grid begins at specimen downstream face"
        ),
        "finite_sample_envelope": True,
        "sample_envelope_shape": geometry.sample_envelope_shape,
        "continuous_square_mesh_sidewalls": True,
        "rutherford_heavy_element_warning": any(
            atomic_number > 30 for atomic_number in encountered_atomic_numbers
        ),
        "rutherford_stated_energy_range_kev": (100.0, 300.0),
        "energy_within_stated_range": bool(
            np.all((energies >= 100_000.0) & (energies <= 300_000.0))
        ),
    }
    return ElasticTransportResult(
        eds_tracks=tuple(tracks),
        trajectories=tuple(stored_trajectories),
        metrics=metrics,
        material_flights=tuple(stored_material_flights),
        terminal_electrons=ElasticTerminalBundle(
            source_ray_index=np.asarray(terminal_source_indices, dtype=np.int64),
            position_nm=np.asarray(terminal_positions, dtype=float),
            direction=np.asarray(terminal_directions, dtype=float),
            kinetic_energy_ev=np.asarray(terminal_energies, dtype=float),
            weight=np.asarray(terminal_weights, dtype=float),
            outcome=tuple(terminal_outcomes),
            event_count=np.asarray(terminal_event_counts, dtype=np.int64),
            has_scattered=np.asarray(terminal_scattered, dtype=bool),
        ),
    )
