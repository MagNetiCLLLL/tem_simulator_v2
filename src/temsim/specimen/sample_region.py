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

from temsim.specimen.vector_field_transport import SpecimenFieldTransport

from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass, replace
import math

import numpy as np

from temsim.detector.eds_geometry import EDSDetectorArrayGeometry
from temsim.detector.eds_signal import EDSSpectrum
from temsim.physics.simulation import Branch
from temsim.specimen.downstream_transport import (
    GeometricSpecimenExit,
    build_geometric_specimen_exit,
    validated_geometric_specimen_exit,
)
from temsim.specimen.elastic_transport import incident_rays_from_simulation
from temsim.specimen.interaction_engine import run_specimen_interactions
from temsim.specimen.interaction_types import (
    SpecimenInteractionRequest,
    SpecimenInteractionResult,
)
from temsim.specimen.scene import SpecimenScene
from temsim.specimen.axial_field_transport import (
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
    specimen_exit: GeometricSpecimenExit | None = None
    photon_transport: object | None = None


_DOWNSTREAM_METRIC_KEYS = frozenset({
    "model",
    "coupling_approximation",
    "sample_incident_source_probability",
    "elastic_forward_conditional_probability",
    "elastic_nontransmitted_conditional_probability",
    "inelastic_tracked_conditional_probability",
    "inelastic_absorbed_conditional_probability",
    "tracked_downstream_source_probability",
    "inelastic_absorbed_source_probability",
    "elastic_nontransmitted_source_probability",
    "pre_sample_lost_source_probability",
    "downstream_branch_weight_sum",
    "downstream_forward_weight",
    "downstream_source_ray_count",
    "exit_plane_source_probability",
    "exit_plane_weight",
    "post_sample_reinjection_plane_z_mm",
    "material_terminal_z_collapsed_to_reference_plane",
    "terminal_state_projection_model",
    "terminal_state_projected_to_sample_reference_plane",
    "terminal_state_projection_preserves_free_flight_line",
    "source_probability_conservation_error",
    "source_probability_conserved",
    "geometric_reference_point_only",
    "pixel_resolved_specimen_contrast",
    "used_by_wave_multislice",
    "sample_downstream_signature",
    "downstream_reprojection_only",
})


def validated_sample_region_exit(
    sample_region: SampleRegionResult,
    expected_signature: str,
) -> GeometricSpecimenExit | None:
    """Recover a validated first-class or legacy downstream checkpoint."""

    checkpoint = validated_geometric_specimen_exit(
        getattr(sample_region, "specimen_exit", None),
        expected_signature,
    )
    if checkpoint is not None:
        return checkpoint
    metrics = dict(getattr(sample_region, "metrics", {}) or {})
    if str(metrics.get("sample_downstream_signature", "")) != str(
        expected_signature
    ):
        return None
    downstream_metrics = {
        key: value
        for key, value in metrics.items()
        if key in _DOWNSTREAM_METRIC_KEYS
    }
    return validated_geometric_specimen_exit(
        GeometricSpecimenExit(
            sample_region.downstream_branches,
            downstream_metrics,
            dependency_signature=expected_signature,
        ),
        expected_signature,
    )


def bind_sample_region_downstream(
    sample_region: SampleRegionResult,
    checkpoint: GeometricSpecimenExit,
    interactions: SpecimenInteractionResult,
    *,
    expected_signature: str,
    wave_imaging=None,
) -> SampleRegionResult:
    """Attach one validated downstream checkpoint to cached local paths."""

    signature = str(expected_signature)
    if validated_geometric_specimen_exit(checkpoint, signature) is None:
        raise ValueError(
            "A complete specimen-exit checkpoint for the current downstream "
            "dependencies is required"
        )
    interaction_signatures = dict(
        getattr(interactions, "metrics", {}) or {}
    ).get("dependency_signatures", {})
    if not isinstance(interaction_signatures, Mapping) or str(
        interaction_signatures.get("sample_downstream", "")
    ) != signature:
        raise ValueError(
            "Specimen interactions and downstream checkpoint must have the "
            "same current dependency signature"
        )
    spectrum = getattr(interactions, "eds_spectrum", None)
    if spectrum is None:
        raise ValueError("Current EDS interactions are required")
    elastic = getattr(interactions, "elastic_transport", None)
    if elastic is None or getattr(spectrum, "elastic_transport", None) is not elastic:
        raise ValueError("EDS and geometric transport must share one elastic result")
    metrics = {
        key: value
        for key, value in dict(sample_region.metrics).items()
        if key not in _DOWNSTREAM_METRIC_KEYS
    }
    metrics.update(checkpoint.metrics)
    metrics["channeling_model"] = (
        "coherent wave/multislice result available; not reclassified as particles"
        if wave_imaging is not None
        else "unavailable without a wave/multislice specimen result"
    )
    return replace(
        sample_region,
        downstream_branches=checkpoint.branches,
        spectrum=spectrum,
        interactions=interactions,
        metrics=metrics,
        specimen_exit=checkpoint,
    )


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
    field_transport = SpecimenFieldTransport(state)
    rows: list[SampleRegionElectronPath] = []
    entry_local_z_nm = (
        float(entry_z_mm) - float(state.sample.z_mm)
    ) * 1.0e6
    for sample_ray in sample_bundle.rays:
        sample_direction = np.asarray(sample_ray.direction, dtype=float)
        reference_position = np.asarray(
            (*sample_ray.position_xy_nm, 0.0), dtype=float
        )
        top_position, top_direction = field_transport.to_plane(
            reference_position,
            sample_direction,
            scene.sample_top_nm,
            energy_ev=sample_ray.kinetic_energy_ev,
        )
        top_to_entry, _entry_direction = field_transport.plane_polyline(
            top_position,
            top_direction,
            entry_local_z_nm,
            energy_ev=sample_ray.kinetic_energy_ev,
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
                    "cached sample-plane phase space; shared vector-field "
                    "boundary transport"
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


def reproject_sample_region_downstream(
    state,
    simulation,
    sample_region: SampleRegionResult,
    interactions: SpecimenInteractionResult,
    *,
    wave_imaging=None,
    dependency_signatures: dict[str, str] | None = None,
) -> SampleRegionResult:
    """Rebuild only post-specimen branches from cached local sample paths.

    ``electron_paths`` and ``photon_paths`` are immutable sample-local view
    products.  D/I/P, projector-mode or recording-plane edits do not alter
    them, so this function shares those arrays by reference and repeats only
    the ordinary downstream propagation and clipping step.
    """

    if sample_region is None or interactions is None:
        raise ValueError("Cached sample-local paths and interactions are required")
    spectrum = getattr(interactions, "eds_spectrum", None)
    elastic = getattr(interactions, "elastic_transport", None)
    if spectrum is None or elastic is None:
        raise ValueError(
            "Downstream sample transport requires current EDS and elastic results"
        )
    if getattr(spectrum, "elastic_transport", None) is not elastic:
        raise ValueError("EDS and downstream transport must share one elastic result")
    signatures = dependency_signatures or interactions.metrics.get(
        "dependency_signatures", {}
    )
    downstream_signature = str(signatures.get("sample_downstream", ""))
    downstream = build_geometric_specimen_exit(
        state,
        simulation,
        elastic,
        interactions.inelastic_distribution,
        save_z_mm=(float(sample_region.exit_z_mm),),
        dependency_signature=downstream_signature,
    )
    rebound = bind_sample_region_downstream(
        sample_region,
        downstream,
        interactions,
        expected_signature=downstream_signature,
        wave_imaging=wave_imaging,
    )
    metrics = dict(rebound.metrics)
    metrics["downstream_reprojection_only"] = True
    metrics["sample_region_signature"] = str(
        signatures.get("sample_region", "")
    )
    return replace(
        rebound,
        metrics=metrics,
    )


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
    # Region bounds and display sampling are explicit arguments.  Mirror them
    # into a shallow calculation context so the stored sample_region identity
    # describes the paths actually built, without mutating the cached global
    # state or duplicating its large/immutable assembly objects.
    calculation_state = copy(state)
    calculation_state.sample = copy(state.sample)
    calculation_state.sample.sample_region_upstream_distance_um = upstream_um
    calculation_state.sample.sample_region_downstream_distance_um = downstream_um
    calculation_state.sample.sample_region_photon_path_count = int(
        photon_path_count
    )
    calculation_state.sample.sample_region_secondary_path_count = int(
        secondary_path_count
    )
    calculation_state.sample.sample_region_seed = int(seed)
    state = calculation_state
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
    elastic = getattr(interactions, "elastic_transport", None)
    if elastic is None:
        raise ValueError(
            "Sample-region transport requires Elastic Monte Carlo electron paths"
        )
    if spectrum.elastic_transport is not elastic:
        raise ValueError("EDS and sample-region transport must share one elastic result")
    electron_paths, entry_bundle, sample_bundle = _electron_paths(
        state,
        simulation,
        elastic,
        entry_z_mm=entry_z_mm,
        secondary_count=int(secondary_path_count),
        rng=rng,
    )
    from temsim.detector.eds_photon_transport import (
        photons_from_sample_region_paths,
        transport_eds_photons,
    )

    photon_transport = getattr(spectrum, "photon_transport", None)
    shared_spectrum_photon_transport = photon_transport is not None
    if shared_spectrum_photon_transport:
        photon_paths = ()
        solid_angle = (
            detector_geometry.analytical_holder_solid_angle_sr
            if str(
                getattr(
                    state.sample,
                    "eds_solid_angle_mode",
                    "installed_holder",
                )
            )
            != "unshadowed"
            else detector_geometry.minimum_unshadowed_solid_angle_sr
        )
        takeoff = math.radians(detector_geometry.takeoff_angle_deg)
        acceptance_half_width_deg = math.degrees(math.asin(
            solid_angle / (4.0 * math.pi * math.cos(takeoff))
        ))
        xray_direction_sampling = (
            "shared deterministic main-spectrum detector quadrature"
        )
    else:
        photon_paths, acceptance_half_width_deg = _photon_paths(
            state,
            spectrum,
            elastic,
            detector_geometry,
            count=int(photon_path_count),
            rng=rng,
            display_length_mm=max(
                10.0, 4.0 * downstream_um * 1.0e-3
            ),
        )
        xray_direction_sampling = "uniform cos(theta), uniform azimuth"
    if photon_transport is None:
        photon_transport = transport_eds_photons(
            state,
            photons_from_sample_region_paths(photon_paths),
            detector_geometry,
            use_analytical_holder_solid_angle=(
                str(
                    getattr(
                        state.sample,
                        "eds_solid_angle_mode",
                        "installed_holder",
                    )
                )
                != "unshadowed"
            ),
            aggregate_detector_efficiency=float(
                getattr(state.sample, "eds_detector_efficiency", 1.0)
            ),
        )
    dependency_signatures = interactions.metrics.get(
        "dependency_signatures", {}
    )
    downstream_signature = str(
        dependency_signatures.get("sample_downstream", "")
    )
    downstream = validated_geometric_specimen_exit(
        getattr(calculation_result, "specimen_exit", None),
        downstream_signature,
    )
    if downstream is None:
        downstream = build_geometric_specimen_exit(
            state,
            simulation,
            elastic,
            interactions.inelastic_distribution,
            save_z_mm=(exit_z_mm,),
            dependency_signature=downstream_signature,
        )
    downstream_branches = downstream.branches
    downstream_metrics = downstream.metrics
    wave_result = getattr(calculation_result, "wave_imaging", None)
    metrics: dict[str, object] = {
        "entry_z_mm": entry_z_mm,
        "exit_z_mm": exit_z_mm,
        "entry_ray_count": entry_bundle.reaching_ray_count,
        "sample_ray_count": sample_bundle.reaching_ray_count,
        "photon_path_count": (
            len(tuple(getattr(photon_transport, "paths", ())))
            if shared_spectrum_photon_transport
            else len(photon_paths)
        ),
        "secondary_marker_count": sum(
            path.kind == "secondary_candidate" for path in electron_paths
        ),
        "xray_direction_sampling": xray_direction_sampling,
        "xray_detector_acceptance": (
            "azimuth-partitioned elevation band with exact aggregate solid angle"
        ),
        "xray_acceptance_half_width_deg": acceptance_half_width_deg,
        "xray_sensor_face_intersection": bool(
            photon_transport.geometry_complete
        ),
        "xray_display_endpoint_is_schematic": not bool(
            photon_transport.geometry_complete
        ),
        "eds_photon_transport_model": photon_transport.metrics["model"],
        "eds_photon_transport_shared_with_spectrum": (
            shared_spectrum_photon_transport
        ),
        "eds_photon_detector_geometry_complete": (
            photon_transport.geometry_complete
        ),
        "eds_photon_expected_weight_per_segment": (
            photon_transport.expected_detected_weight_per_segment
        ),
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
        "sample_region_signature": str(
            interactions.metrics.get("dependency_signatures", {}).get(
                "sample_region", ""
            )
        ),
        "sample_downstream_signature": str(
            interactions.metrics.get("dependency_signatures", {}).get(
                "sample_downstream", ""
            )
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
        specimen_exit=downstream,
        photon_transport=photon_transport,
    )
