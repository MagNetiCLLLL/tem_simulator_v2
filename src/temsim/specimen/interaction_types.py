"""Shared contracts for specimen-local electron and signal calculations.

The specimen coordinate system is right handed.  Electrons enter along +Z;
positions in this module use nanometres, directions are dimensionless unit
vectors, kinetic energies use electronvolts and statistical weights are
source-normalised dimensionless probabilities.

This module intentionally contains no solver.  Coherent multislice,
stochastic inelastic branches and particle Monte Carlo remain separate
numerical methods, but they exchange data and report conservation through the
same immutable contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math
import numpy as np


class SpecimenObservable(StrEnum):
    """Physical observables that may be requested independently."""

    COHERENT_ELASTIC_WAVE = "coherent_elastic_wave"
    STOCHASTIC_INELASTIC = "stochastic_inelastic"
    ELASTIC_TRANSPORT = "elastic_transport"
    CHARACTERISTIC_X_RAY = "characteristic_x_ray"


class InteractionProcess(StrEnum):
    """Stable process identifiers for the future per-event ledger."""

    COHERENT_ELASTIC = "coherent_elastic"
    ELASTIC_SCATTER = "elastic_scatter"
    PLASMON_LOSS = "plasmon_loss"
    CORE_IONISATION = "core_ionisation"
    OTHER_INELASTIC = "other_inelastic"
    ABSORPTION = "absorption"
    RADIATIVE_RELAXATION = "radiative_relaxation"
    CHARACTERISTIC_X_RAY = "characteristic_x_ray"
    AUGER_RELAXATION = "auger_relaxation"
    UNRESOLVED_RELAXATION = "unresolved_relaxation"


@dataclass(frozen=True, slots=True)
class IncidentElectronRay:
    """One weighted electron at a specimen-region entrance plane.

    ``position_xy_nm`` is expressed in the laboratory specimen plane.
    ``direction`` is a right-handed ``(x, y, z)`` unit vector and therefore
    retains incident tilt and accumulated column rotation.
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
            raise ValueError(
                "Incident electron position must be a finite 2-vector"
            )
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError(
                "Incident electron direction must be a finite 3-vector"
            )
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0 or not math.isclose(
            norm, 1.0, rel_tol=1.0e-10
        ):
            raise ValueError(
                "Incident electron direction must be a unit vector"
            )
        if (
            int(self.source_ray_index) < 0
            or not math.isfinite(float(self.kinetic_energy_ev))
            or float(self.kinetic_energy_ev) <= 0.0
            or not math.isfinite(float(self.weight))
            or float(self.weight) < 0.0
        ):
            raise ValueError(
                "Incident electron index, energy or weight is invalid"
            )


@dataclass(frozen=True, slots=True)
class IncidentRayBundle:
    """Calculated phase space reaching one specimen-region boundary."""

    rays: tuple[IncidentElectronRay, ...]
    emitted_ray_count: int
    reaching_ray_count: int
    surviving_fraction: float
    original_centroid_nm: tuple[float, float]
    target_centroid_nm: tuple[float, float]
    chief_angle_mrad: tuple[float, float]
    energy_range_ev: tuple[float, float]
    boundary_z_mm: float | None = None

    def __post_init__(self) -> None:
        rays = tuple(self.rays)
        object.__setattr__(self, "rays", rays)
        emitted = int(self.emitted_ray_count)
        reaching = int(self.reaching_ray_count)
        survival = float(self.surviving_fraction)
        if emitted < 0 or reaching < 0 or reaching > emitted:
            raise ValueError("Incident ray counts are inconsistent")
        if reaching != len(rays):
            raise ValueError(
                "Incident reaching-ray count must match the ray tuple"
            )
        if not math.isfinite(survival) or not 0.0 <= survival <= 1.0:
            raise ValueError("Incident surviving fraction must be in [0, 1]")
        for name, values in (
            ("original centroid", self.original_centroid_nm),
            ("target centroid", self.target_centroid_nm),
            ("chief angle", self.chief_angle_mrad),
            ("energy range", self.energy_range_ev),
        ):
            array = np.asarray(values, dtype=float)
            if array.shape != (2,) or not np.all(np.isfinite(array)):
                raise ValueError(f"Incident {name} must contain two values")
        if self.energy_range_ev[0] <= 0.0:
            raise ValueError("Incident energy range must be positive")
        if self.energy_range_ev[1] < self.energy_range_ev[0]:
            raise ValueError("Incident energy range is reversed")
        if self.boundary_z_mm is not None and not math.isfinite(
            float(self.boundary_z_mm)
        ):
            raise ValueError("Incident boundary Z must be finite")
        if rays and not math.isclose(
            sum(float(ray.weight) for ray in rays),
            1.0,
            rel_tol=0.0,
            abs_tol=2.0e-12,
        ):
            raise ValueError(
                "Incident conditional ray weights must sum to one"
            )


@dataclass(frozen=True, slots=True)
class InteractionEvent:
    """One physical event record in the unified specimen ledger.

    ``position_nm`` is in the specimen-local frame.  The two
    ``deflection_rad`` values are tangent-plane components about the incoming
    direction, not necessarily laboratory X/Y angles.  Weighted representative
    events report their source-normalised expected occurrence separately from
    the parent electron's statistical weight.
    """

    process: InteractionProcess
    position_nm: tuple[float, float, float]
    electron_weight: float
    incident_energy_ev: float
    energy_transfer_ev: float = 0.0
    emitted_energy_ev: float | None = 0.0
    expected_occurrences_per_source_electron: float = 1.0
    event_id: str = ""
    parent_event_id: str | None = None
    parent_electron_index: int | None = None
    deflection_rad: tuple[float, float] = (0.0, 0.0)
    source_key: str = "sample"
    material_key: str = ""
    atomic_number: int | None = None
    transition: str = ""
    model: str = ""
    provenance: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "process", InteractionProcess(self.process)
        )
        position = np.asarray(self.position_nm, dtype=float)
        deflection = np.asarray(self.deflection_rad, dtype=float)
        values = (
            float(self.electron_weight),
            float(self.incident_energy_ev),
            float(self.energy_transfer_ev),
            float(self.expected_occurrences_per_source_electron),
        )
        emitted_energy = (
            None
            if self.emitted_energy_ev is None
            else float(self.emitted_energy_ev)
        )
        if not str(self.event_id).strip():
            raise ValueError("Interaction event ID cannot be empty")
        if self.parent_event_id == self.event_id:
            raise ValueError("Interaction event cannot be its own parent")
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("Interaction position must be a finite 3-vector")
        if deflection.shape != (2,) or not np.all(np.isfinite(deflection)):
            raise ValueError("Interaction deflection must be a finite 2-vector")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Interaction weights and energies must be finite")
        if (
            values[0] < 0.0
            or values[1] <= 0.0
            or values[2] < 0.0
            or values[3] < 0.0
        ):
            raise ValueError("Interaction weights and energies are invalid")
        if values[2] > values[1]:
            raise ValueError("Interaction cannot transfer more than incident energy")
        if emitted_energy is not None and (
            not math.isfinite(emitted_energy) or emitted_energy < 0.0
        ):
            raise ValueError("Interaction emitted energy is invalid")
        if emitted_energy is not None and emitted_energy > values[1]:
            raise ValueError("Interaction cannot emit more than incident energy")
        if self.parent_electron_index is not None and int(
            self.parent_electron_index
        ) < 0:
            raise ValueError("Interaction parent index cannot be negative")
        if self.atomic_number is not None and not 1 <= int(
            self.atomic_number
        ) <= 99:
            raise ValueError("Interaction atomic number must be Z=1..99")


@dataclass(frozen=True, slots=True)
class SpecimenModelCoupling:
    """Accounting relationship between one solver and physical processes.

    Only records marked ``contributes_to_exclusive_electron_budget`` may enter
    the specimen population-probability sum. Coherent intensity, independent
    particle trajectories and derived X-ray signals remain non-additive.
    """

    solver: str
    processes: tuple[InteractionProcess, ...]
    representation: str
    event_resolution: str
    energy_coupling: str
    contributes_to_exclusive_electron_budget: bool
    double_count_rule: str

    def __post_init__(self) -> None:
        processes = tuple(InteractionProcess(value) for value in self.processes)
        object.__setattr__(self, "processes", processes)
        if not processes:
            raise ValueError("Specimen model coupling needs a physical process")
        if len(set(processes)) != len(processes):
            raise ValueError("Specimen model coupling processes must be unique")
        for value in (
            self.solver,
            self.representation,
            self.event_resolution,
            self.energy_coupling,
            self.double_count_rule,
        ):
            if not str(value).strip():
                raise ValueError("Specimen model coupling metadata cannot be empty")


@dataclass(frozen=True, slots=True)
class ConservationCheck:
    """One auditable conservation equation with explicit units."""

    name: str
    quantity: str
    unit: str
    input_value: float
    output_channels: tuple[tuple[str, float], ...]
    absolute_tolerance: float = 2.0e-12

    def __post_init__(self) -> None:
        input_value = float(self.input_value)
        tolerance = float(self.absolute_tolerance)
        channels = tuple(
            (str(key), float(value))
            for key, value in self.output_channels
        )
        object.__setattr__(self, "output_channels", channels)
        if not self.name or not self.quantity or not self.unit:
            raise ValueError("Conservation metadata cannot be empty")
        if not math.isfinite(input_value) or input_value < 0.0:
            raise ValueError("Conservation input must be finite and non-negative")
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("Conservation tolerance must be finite and non-negative")
        keys = [key for key, _value in channels]
        if len(keys) != len(set(keys)):
            raise ValueError("Conservation channel names must be unique")
        if any(
            not math.isfinite(value) or value < 0.0
            for _key, value in channels
        ):
            raise ValueError(
                "Conservation outputs must be finite and non-negative"
            )

    @property
    def output_value(self) -> float:
        return float(sum(value for _key, value in self.output_channels))

    @property
    def residual(self) -> float:
        return self.output_value - float(self.input_value)

    @property
    def conserved(self) -> bool:
        return abs(self.residual) <= float(self.absolute_tolerance)

    def require_conserved(self) -> None:
        if not self.conserved:
            raise RuntimeError(
                f"{self.name} failed {self.quantity} conservation: "
                f"input={self.input_value:.17g} {self.unit}, "
                f"output={self.output_value:.17g} {self.unit}, "
                f"residual={self.residual:.17g} {self.unit}, "
                f"tolerance={self.absolute_tolerance:.17g} {self.unit}."
            )


@dataclass(frozen=True, slots=True)
class SpecimenInteractionRequest:
    """Explicit calculation request; unrequested observables never run.

    For point observables, the engine resolves each omitted coordinate to the
    sample's scan origin and records the resulting explicit acquisition point.
    This is separate from a raw incident-boundary extraction, which may retain
    the unshifted beam centroid.
    """

    observables: frozenset[SpecimenObservable] = field(default_factory=frozenset)
    point_x_nm: float | None = None
    point_y_nm: float | None = None
    dwell_time_s: float | None = None
    incident_electrons: float | None = None

    def __post_init__(self) -> None:
        observables = frozenset(
            SpecimenObservable(value) for value in self.observables
        )
        object.__setattr__(self, "observables", observables)
        for name, value in (
            ("point X", self.point_x_nm),
            ("point Y", self.point_y_nm),
            ("dwell time", self.dwell_time_s),
            ("incident electrons", self.incident_electrons),
        ):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"Specimen interaction {name} must be finite")
        if self.dwell_time_s is not None and self.dwell_time_s <= 0.0:
            raise ValueError("Specimen interaction dwell time must be positive")
        if self.incident_electrons is not None and self.incident_electrons < 0.0:
            raise ValueError(
                "Specimen interaction incident-electron count cannot be negative"
            )

    @classmethod
    def tem_wave(cls) -> "SpecimenInteractionRequest":
        return cls(
            observables=frozenset(
                (SpecimenObservable.COHERENT_ELASTIC_WAVE,)
            )
        )

    @classmethod
    def eds_point(
        cls,
        *,
        x_nm: float | None = None,
        y_nm: float | None = None,
        dwell_time_s: float | None = None,
        incident_electrons: float | None = None,
    ) -> "SpecimenInteractionRequest":
        return cls(
            observables=frozenset(
                (SpecimenObservable.CHARACTERISTIC_X_RAY,)
            ),
            point_x_nm=x_nm,
            point_y_nm=y_nm,
            dwell_time_s=dwell_time_s,
            incident_electrons=incident_electrons,
        )

    @classmethod
    def elastic_point(
        cls,
        *,
        x_nm: float | None = None,
        y_nm: float | None = None,
    ) -> "SpecimenInteractionRequest":
        return cls(
            observables=frozenset(
                (SpecimenObservable.ELASTIC_TRANSPORT,)
            ),
            point_x_nm=x_nm,
            point_y_nm=y_nm,
        )


@dataclass(frozen=True, slots=True)
class SpecimenInteractionResult:
    """Unified typed envelope around current specimen solver outputs."""

    request: SpecimenInteractionRequest
    completed_observables: frozenset[SpecimenObservable]
    scene: object | None = None
    incident_bundle: IncidentRayBundle | None = None
    wave_imaging: object | None = None
    inelastic_distribution: object | None = None
    elastic_transport: object | None = None
    eds_spectrum: object | None = None
    events: tuple[InteractionEvent, ...] = ()
    couplings: tuple[SpecimenModelCoupling, ...] = ()
    conservation: tuple[ConservationCheck, ...] = ()
    metrics: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        completed = frozenset(
            SpecimenObservable(value) for value in self.completed_observables
        )
        object.__setattr__(self, "completed_observables", completed)
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "couplings", tuple(self.couplings))
        object.__setattr__(self, "conservation", tuple(self.conservation))
        object.__setattr__(self, "metrics", dict(self.metrics))
        missing = self.request.observables - completed
        if missing:
            labels = ", ".join(sorted(value.value for value in missing))
            raise ValueError(
                f"Requested specimen observables were not completed: {labels}"
            )
        for check in self.conservation:
            check.require_conserved()
        exclusive_budgets = sum(
            bool(record.contributes_to_exclusive_electron_budget)
            for record in self.couplings
        )
        if exclusive_budgets > 1:
            raise ValueError(
                "Only one solver may own the exclusive electron population budget"
            )


def retain_specimen_observables(
    result: SpecimenInteractionResult | None,
    observables: frozenset[SpecimenObservable],
) -> SpecimenInteractionResult | None:
    """Keep only explicitly compatible products from a shared result.

    Cache invalidation is deliberately expressed in terms of physical
    observables.  Derived event ledgers and conservation summaries are rebuilt
    by :func:`run_specimen_interactions` on the next augmentation rather than
    being carried across a changed dependency signature.
    """

    if result is None:
        return None
    retained = frozenset(result.completed_observables & observables)
    request = SpecimenInteractionRequest(
        observables=retained,
        point_x_nm=result.request.point_x_nm,
        point_y_nm=result.request.point_y_nm,
        dwell_time_s=result.request.dwell_time_s,
        incident_electrons=result.request.incident_electrons,
    )
    return SpecimenInteractionResult(
        request=request,
        completed_observables=retained,
        scene=result.scene,
        incident_bundle=(
            result.incident_bundle
            if SpecimenObservable.ELASTIC_TRANSPORT in retained
            else None
        ),
        wave_imaging=(
            result.wave_imaging
            if SpecimenObservable.COHERENT_ELASTIC_WAVE in retained
            else None
        ),
        inelastic_distribution=(
            result.inelastic_distribution
            if SpecimenObservable.STOCHASTIC_INELASTIC in retained
            else None
        ),
        elastic_transport=(
            result.elastic_transport
            if SpecimenObservable.ELASTIC_TRANSPORT in retained
            else None
        ),
        eds_spectrum=(
            result.eds_spectrum
            if SpecimenObservable.CHARACTERISTIC_X_RAY in retained
            else None
        ),
        metrics={
            "contract_version": result.metrics.get("contract_version", 5),
            "dependency_signatures": dict(
                result.metrics.get("dependency_signatures", {})
            ),
            "retained_observables_after_dependency_check": tuple(
                sorted(value.value for value in retained)
            ),
        },
    )
