"""Manual, bounded specimen-region transport and downstream handoff.

The global column calculation owns deterministic electron optics up to the
sample boundary.  This module consumes that cached phase space, runs only the
finite specimen/support elastic and characteristic-EDS model, and reinjects
forward terminal electrons into the ordinary post-sample column propagator.

Explicit photon paths are Monte-Carlo representatives of the already
calculated characteristic line population.  Because public detector data do
not include active-face dimensions or sensor distance, detector interception
uses an azimuth-partitioned elevation band whose *aggregate* spherical area is
exactly the configured solid angle.  It is an angular acceptance surrogate,
not a mechanical sensor-face intersection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from temsim.detector.eds_geometry import EDSDetectorArrayGeometry
from temsim.detector.eds_signal import EDSSpectrum
from temsim.physics.simulation import Branch
from temsim.specimen.downstream_transport import build_geometric_specimen_exit
from temsim.specimen.elastic_transport import incident_rays_from_simulation
from temsim.specimen.interaction_engine import run_specimen_interactions
from temsim.specimen.interaction_types import (
    SpecimenInteractionRequest,
    SpecimenInteractionResult,
)
from temsim.specimen.scene import SpecimenScene
from temsim.specimen.axial_field_transport import (
    advance_in_uniform_axial_field,
    axial_field_polyline,
    axial_rotation_rate_rad_per_nm,
    sample_axial_field_diagnostic,
)


@dataclass(frozen=True, slots=True)
class SampleRegionElectronPath:
    positions_mm: np.ndarray
    kind: str
    weight: float
    kinetic_energy_ev: float | None
    provenance: str
    downstream_eligible: bool

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_mm, dtype=float)
        if (
            positions.ndim != 2
            or positions.shape[1:] != (3,)
            or positions.shape[0] < 2
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError("Sample-region electron path must be finite N by 3")
        positions.setflags(write=False)
        object.__setattr__(self, "positions_mm", positions)


@dataclass(frozen=True, slots=True)
class SampleRegionPhotonPath:
    positions_mm: np.ndarray
    direction: tuple[float, float, float]
    energy_ev: float
    emitted_weight: float
    detected_weight: float
    source_key: str
    transition: str
    detected: bool
    detector_segment: int | None
    provenance: str

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_mm, dtype=float)
        direction = np.asarray(self.direction, dtype=float)
        if (
            positions.shape != (2, 3)
            or not np.all(np.isfinite(positions))
            or direction.shape != (3,)
            or not np.all(np.isfinite(direction))
            or not math.isclose(float(np.linalg.norm(direction)), 1.0, rel_tol=1e-10)
        ):
            raise ValueError("Sample-region photon geometry is invalid")
        positions.setflags(write=False)
        object.__setattr__(self, "positions_mm", positions)


@dataclass(frozen=True, slots=True)
class SampleRegionResult:
    entry_z_mm: float
    exit_z_mm: float
    electron_paths: tuple[SampleRegionElectronPath, ...]
    photon_paths: tuple[SampleRegionPhotonPath, ...]
    downstream_branches: tuple[Branch, ...]
    spectrum: EDSSpectrum
    interactions: SpecimenInteractionResult
    metrics: dict[str, object]


def _isotropic_directions(
    rng: np.random.Generator, count: int
) -> np.ndarray:
    """Sample unit directions uniformly in solid angle, not uniformly in theta."""

    cosine = rng.uniform(-1.0, 1.0, int(count))
    azimuth = rng.uniform(0.0, 2.0 * math.pi, int(count))
    radial = np.sqrt(np.maximum(0.0, 1.0 - cosine * cosine))
    return np.column_stack(
        (radial * np.cos(azimuth), radial * np.sin(azimuth), cosine)
    )


def _angular_acceptance(
    direction: np.ndarray,
    geometry: EDSDetectorArrayGeometry,
    solid_angle_sr: float,
) -> tuple[bool, int | None, float]:
    """Evaluate the exact-area angular-band surrogate for one photon."""

    takeoff = math.radians(float(geometry.takeoff_angle_deg))
    argument = float(solid_angle_sr) / (4.0 * math.pi * math.cos(takeoff))
    if not 0.0 < argument <= 1.0:
        raise ValueError(
            "Configured EDS solid angle cannot form the documented angular band"
        )
    half_width = math.asin(argument)
    if (
        takeoff - half_width < -0.5 * math.pi
        or takeoff + half_width > 0.5 * math.pi
    ):
        raise ValueError(
            "Configured EDS angular band crosses a spherical elevation pole"
        )
    radial = math.hypot(float(direction[0]), float(direction[1]))
    elevation = math.atan2(-float(direction[2]), radial)
    accepted = abs(elevation - takeoff) <= half_width
    if not accepted:
        return False, None, math.degrees(half_width)
    azimuth = math.degrees(math.atan2(direction[1], direction[0])) % 360.0
    separations = [
        abs((azimuth - centre + 180.0) % 360.0 - 180.0)
        for centre in geometry.azimuth_centers_deg
    ]
    return True, int(np.argmin(separations)), math.degrees(half_width)


def _material_origin_mm(flight, fraction: float, sample_z_mm: float) -> np.ndarray:
    start = np.asarray(flight.start_nm, dtype=float)
    end = np.asarray(flight.end_nm, dtype=float)
    local_nm = start + float(fraction) * (end - start)
    return np.array(
        (
            local_nm[0] * 1.0e-6,
            local_nm[1] * 1.0e-6,
            float(sample_z_mm) + local_nm[2] * 1.0e-6,
        ),
        dtype=float,
    )


def _photon_paths(
    state,
    spectrum: EDSSpectrum,
    elastic_result,
    geometry: EDSDetectorArrayGeometry,
    *,
    count: int,
    rng: np.random.Generator,
    display_length_mm: float,
):
    lines = tuple(
        line for line in spectrum.lines if line.expected_emitted_photons > 0.0
    )
    if count <= 0 or not lines:
        return (), 0.0
    emitted = np.asarray(
        [line.expected_emitted_photons for line in lines], dtype=float
    )
    total_emitted = float(np.sum(emitted))
    line_indices = rng.choice(
        len(lines), size=int(count), replace=True, p=emitted / total_emitted
    )
    directions = _isotropic_directions(rng, int(count))
    solid_angle = (
        geometry.analytical_holder_solid_angle_sr
        if str(getattr(state.sample, "eds_solid_angle_mode", "installed_holder"))
        == "installed_holder"
        else geometry.minimum_unshadowed_solid_angle_sr
    )
    efficiency = float(getattr(state.sample, "eds_detector_efficiency", 1.0))
    flights = tuple(elastic_result.material_flights)
    rows = []
    band_half_width_deg = 0.0
    emitted_weight = total_emitted / int(count)
    for line_index, direction in zip(line_indices, directions, strict=True):
        line = lines[int(line_index)]
        source_flights = tuple(
            flight for flight in flights if flight.source_key == line.source_key
        )
        if source_flights:
            path_weights = np.asarray(
                [
                    np.linalg.norm(
                        np.asarray(flight.end_nm) - np.asarray(flight.start_nm)
                    )
                    * float(flight.electron_weight)
                    for flight in source_flights
                ],
                dtype=float,
            )
            if float(np.sum(path_weights)) > 0.0:
                flight = source_flights[
                    int(rng.choice(len(source_flights), p=path_weights / path_weights.sum()))
                ]
            else:
                flight = source_flights[int(rng.integers(0, len(source_flights)))]
            origin = _material_origin_mm(
                flight, float(rng.random()), float(state.sample.z_mm)
            )
        else:
            origin = np.array(
                (
                    float(state.sample.centre_x_nm) * 1.0e-6,
                    float(state.sample.centre_y_nm) * 1.0e-6,
                    float(state.sample.z_mm),
                )
            )
        accepted, segment, band_half_width_deg = _angular_acceptance(
            direction, geometry, solid_angle
        )
        detected_weight = (
            emitted_weight
            * float(line.self_absorption_transmission)
            * efficiency
            if accepted
            else 0.0
        )
        rows.append(
            SampleRegionPhotonPath(
                positions_mm=np.vstack(
                    (origin, origin + direction * float(display_length_mm))
                ),
                direction=tuple(float(value) for value in direction),
                energy_ev=float(line.energy_ev),
                emitted_weight=emitted_weight,
                detected_weight=detected_weight,
                source_key=str(line.source_key),
                transition=str(line.transition),
                detected=bool(accepted),
                detector_segment=segment,
                provenance=(
                    "isotropic characteristic photon; exact aggregate-solid-angle "
                    "angular-band surrogate; display endpoint is not a sensor face"
                ),
            )
        )
    return tuple(rows), band_half_width_deg


def _electron_paths(
    state,
    simulation,
    elastic_result,
    *,
    entry_z_mm: float,
    secondary_count: int,
    rng: np.random.Generator,
):
    scene = SpecimenScene.from_state(state)
    target_x_nm = float(getattr(state.sample, "scan_origin_x_nm", 0.0))
    target_y_nm = float(getattr(state.sample, "scan_origin_y_nm", 0.0))
    sample_bundle = incident_rays_from_simulation(
        state,
        simulation,
        target_x_nm=target_x_nm,
        target_y_nm=target_y_nm,
    )
    entry_bundle = incident_rays_from_simulation(
        state,
        simulation,
        boundary_z_mm=entry_z_mm,
    )
    field_diagnostic = sample_axial_field_diagnostic(state)
    rows: list[SampleRegionElectronPath] = []
    entry_local_z_nm = (
        float(entry_z_mm) - float(state.sample.z_mm)
    ) * 1.0e6
    for sample_ray in sample_bundle.rays:
        sample_direction = np.asarray(sample_ray.direction, dtype=float)
        rotation_rate = axial_rotation_rate_rad_per_nm(
            field_diagnostic.total_field_t,
            sample_ray.kinetic_energy_ev,
        )
        reference_position = np.asarray(
            (*sample_ray.position_xy_nm, 0.0), dtype=float
        )
        top_position, top_direction = advance_in_uniform_axial_field(
            reference_position,
            sample_direction,
            scene.sample_top_nm / float(sample_direction[2]),
            rotation_rate_rad_per_nm=rotation_rate,
        )
        top_to_entry_length = (
            (entry_local_z_nm - scene.sample_top_nm)
            / float(top_direction[2])
        )
        top_to_entry, _entry_direction = axial_field_polyline(
            top_position,
            top_direction,
            top_to_entry_length,
            rotation_rate_rad_per_nm=rotation_rate,
        )
        entry_to_top = top_to_entry[::-1]
        global_mm = entry_to_top.copy()
        global_mm[:, :2] *= 1.0e-6
        global_mm[:, 2] = (
            float(state.sample.z_mm) + entry_to_top[:, 2] * 1.0e-6
        )
        rows.append(
            SampleRegionElectronPath(
                positions_mm=global_mm,
                kind="boundary_input",
                weight=float(sample_ray.weight),
                kinetic_energy_ev=float(sample_ray.kinetic_energy_ev),
                provenance=(
                    "cached sample-plane phase space; local-uniform solver Bz "
                    "helical boundary transport"
                ),
                downstream_eligible=True,
            )
        )
    for trajectory in elastic_result.trajectories:
        local = np.asarray(trajectory.points_nm, dtype=float)
        if local.shape[0] < 2:
            continue
        global_mm = np.column_stack(
            (
                local[:, 0] * 1.0e-6,
                local[:, 1] * 1.0e-6,
                float(state.sample.z_mm) + local[:, 2] * 1.0e-6,
            )
        )
        if trajectory.outcome == "backscattered":
            kind = "backscattered"
        elif len(trajectory.events):
            kind = "elastic_rutherford"
        else:
            kind = "primary_material"
        rows.append(
            SampleRegionElectronPath(
                positions_mm=global_mm,
                kind=kind,
                weight=float(trajectory.incident_weight),
                kinetic_energy_ev=float(trajectory.initial_energy_ev),
                provenance="screened-Rutherford elastic material transport",
                downstream_eligible=trajectory.outcome == "transmitted",
            )
        )

    flights = tuple(elastic_result.material_flights)
    if flights and secondary_count > 0:
        directions = _isotropic_directions(rng, int(secondary_count))
        for direction in directions:
            flight = flights[int(rng.integers(0, len(flights)))]
            origin = _material_origin_mm(
                flight, float(rng.random()), float(state.sample.z_mm)
            )
            # This is intentionally a local marker.  No unvalidated yield or
            # energy distribution is introduced into charge/current budgets.
            endpoint = origin + direction * 1.0e-4
            rows.append(
                SampleRegionElectronPath(
                    positions_mm=np.vstack((origin, endpoint)),
                    kind="secondary_candidate",
                    weight=0.0,
                    kinetic_energy_ev=None,
                    provenance=(
                        "qualitative isotropic ionisation-secondary marker; "
                        "no quantitative yield or energy model"
                    ),
                    downstream_eligible=False,
                )
            )
    return tuple(rows), entry_bundle, sample_bundle


def simulate_sample_region(
    state,
    calculation_result,
    detector_geometry: EDSDetectorArrayGeometry,
    *,
    upstream_distance_um: float = 50.0,
    downstream_distance_um: float = 50.0,
    photon_path_count: int = 128,
    secondary_path_count: int = 48,
    seed: int = 0,
    existing_interactions: SpecimenInteractionResult | None = None,
) -> SampleRegionResult:
    """Build one explicit sample-local view from shared physical results.

    EDS and elastic transport are enriched only when they are missing from the
    compatible interaction envelope.  Boundary placement, path subsampling and
    photon display directions remain view construction and never rerun the
    underlying specimen physics.
    """

    upstream_um = float(upstream_distance_um)
    downstream_um = float(downstream_distance_um)
    if not bool(getattr(state.sample, "inserted", False)):
        raise ValueError("Insert a specimen before sample-region transport")
    if not bool(getattr(state.sample, "eds_enabled", False)):
        raise ValueError("Enable explicit EDS acquisition first")
    if not (upstream_um > 0.0 and downstream_um > 0.0):
        raise ValueError("Sample-region boundary distances must be positive")
    if int(photon_path_count) < 0 or int(secondary_path_count) < 0:
        raise ValueError("Displayed photon/electron path counts cannot be negative")
    simulation = getattr(calculation_result, "simulation", None)
    if simulation is None:
        raise ValueError("A completed global column calculation is required")
    entry_z_mm = float(state.sample.z_mm) - upstream_um * 1.0e-3
    exit_z_mm = float(state.sample.z_mm) + downstream_um * 1.0e-3
    rng = np.random.default_rng(int(seed))

    interactions = run_specimen_interactions(
        state,
        simulation,
        SpecimenInteractionRequest.eds_point(),
        detector_geometry=detector_geometry,
        existing_result=existing_interactions,
    )
    spectrum = interactions.eds_spectrum
    if spectrum is None:
        raise RuntimeError("Specimen interaction engine returned no EDS spectrum")
    elastic = spectrum.elastic_transport
    if elastic is None:
        raise ValueError(
            "Sample-region transport requires Elastic Monte Carlo electron paths"
        )
    electron_paths, entry_bundle, sample_bundle = _electron_paths(
        state,
        simulation,
        elastic,
        entry_z_mm=entry_z_mm,
        secondary_count=int(secondary_path_count),
        rng=rng,
    )
    photon_paths, acceptance_half_width_deg = _photon_paths(
        state,
        spectrum,
        elastic,
        detector_geometry,
        count=int(photon_path_count),
        rng=rng,
        display_length_mm=max(10.0, 4.0 * downstream_um * 1.0e-3),
    )
    downstream = build_geometric_specimen_exit(
        state,
        simulation,
        elastic,
        interactions.inelastic_distribution,
        save_z_mm=(exit_z_mm,),
    )
    downstream_branches = downstream.branches
    downstream_metrics = downstream.metrics
    wave_result = getattr(calculation_result, "wave_imaging", None)
    metrics: dict[str, object] = {
        "entry_z_mm": entry_z_mm,
        "exit_z_mm": exit_z_mm,
        "entry_ray_count": entry_bundle.reaching_ray_count,
        "sample_ray_count": sample_bundle.reaching_ray_count,
        "photon_path_count": len(photon_paths),
        "secondary_marker_count": sum(
            path.kind == "secondary_candidate" for path in electron_paths
        ),
        "xray_direction_sampling": "uniform cos(theta), uniform azimuth",
        "xray_detector_acceptance": (
            "azimuth-partitioned elevation band with exact aggregate solid angle"
        ),
        "xray_acceptance_half_width_deg": acceptance_half_width_deg,
        "xray_sensor_face_intersection": False,
        "xray_display_endpoint_is_schematic": True,
        "secondary_electron_model": (
            "qualitative local marker only; no yield or energy distribution"
        ),
        "secondary_electrons_in_downstream_column": False,
        "rutherford_relation": "Rutherford is the active elastic-scattering approximation",
        "channeling_model": (
            "coherent wave/multislice result available; not reclassified as particles"
            if wave_result is not None
            else "unavailable without a wave/multislice specimen result"
        ),
        "channeling_double_counted_in_elastic_mc": False,
        "automatic_recalculation": False,
        "sample_axial_field_t": elastic.metrics.get("sample_axial_field_t"),
        "sample_objective_field_t": elastic.metrics.get(
            "sample_objective_field_t"
        ),
        "sample_field_transport_model": elastic.metrics.get(
            "sample_field_transport_model"
        ),
        "sample_field_geometry_material_coupled": elastic.metrics.get(
            "sample_field_geometry_material_coupled"
        ),
        "shared_specimen_result_reused": bool(
            interactions.metrics.get("existing_result_reused", False)
        ),
        "specimen_observables_calculated_for_this_view": tuple(
            interactions.metrics.get("calculated_observables_this_call", ())
        ),
        **downstream_metrics,
    }
    return SampleRegionResult(
        entry_z_mm=entry_z_mm,
        exit_z_mm=exit_z_mm,
        electron_paths=electron_paths,
        photon_paths=photon_paths,
        downstream_branches=downstream_branches,
        spectrum=spectrum,
        interactions=interactions,
        metrics=metrics,
    )
