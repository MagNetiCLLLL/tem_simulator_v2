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
from temsim.detector.eds_signal import EDSSpectrum, simulate_eds_point
from temsim.physics.column_wall import clip_column_wall
from temsim.physics.core import propagate
from temsim.physics.recording_clipping import clip_recording_planes
from temsim.physics.recording_stop import determine_tem_stop_z
from temsim.physics.simulation import Branch, RAY_INTERACTION_COLOURS
from temsim.specimen.elastic_transport import incident_rays_from_simulation


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


def _post_sample_events(state) -> tuple[tuple[float, float, float], ...]:
    events: list[tuple[float, float, float]] = []
    for collection_name in ("deflectors", "corrector_elements"):
        for component in getattr(state, collection_name, ()):
            if not bool(getattr(component, "enabled", False)):
                continue
            if not hasattr(component, "kick_events"):
                continue
            try:
                component_events = component.kick_events(
                    time_s=float(getattr(state, "simulation_time_s", 0.0))
                )
            except TypeError:
                component_events = component.kick_events()
            events.extend(
                (float(z), float(dx), float(dy))
                for z, dx, dy in component_events
                if float(z) > float(state.sample.z_mm)
            )
    return tuple(sorted(events))


def _downstream_branches(state, elastic_result, exit_z_mm: float):
    terminal = elastic_result.terminal_electrons
    if terminal is None or len(terminal.outcome) == 0:
        return (), {"downstream_forward_weight": 0.0, "exit_plane_weight": 0.0}
    directions = np.asarray(terminal.direction, dtype=float)
    outcomes = np.asarray(terminal.outcome, dtype=object)
    eligible = (outcomes == "transmitted") & (directions[:, 2] > 1.0e-12)
    if not np.any(eligible):
        return (), {"downstream_forward_weight": 0.0, "exit_plane_weight": 0.0}

    positions_nm = np.asarray(terminal.position_nm, dtype=float)[eligible]
    direction = directions[eligible]
    weights = np.asarray(terminal.weight, dtype=float)[eligible]
    energies = np.asarray(terminal.kinetic_energy_ev, dtype=float)[eligible]
    scattered = np.asarray(terminal.has_scattered, dtype=bool)[eligible]
    source_indices = np.asarray(terminal.source_ray_index, dtype=np.int64)[eligible]

    # The material kernel resolves nanometre-scale geometry about the axial
    # sample reference.  Its terminal transverse state is reinjected at that
    # reference plane; the ordinary propagator then supplies the deterministic
    # objective field and every downstream optical element.
    slopes = direction[:, :2] / direction[:, 2, None]
    stop_z_mm = float(determine_tem_stop_z(state))
    z, x, tx, y, ty = propagate(
        state,
        float(state.sample.z_mm),
        stop_z_mm,
        positions_nm[:, 0] * 1.0e-9,
        slopes[:, 0],
        positions_nm[:, 1] * 1.0e-9,
        slopes[:, 1],
        _post_sample_events(state),
        energies - float(state.beam_voltage_kv) * 1000.0,
        save_z_mm=(float(exit_z_mm),),
    )
    alive = np.ones(len(weights), dtype=bool)
    blocked = np.full(len(weights), np.nan, dtype=float)
    keys = [""] * len(weights)
    alive, blocked, keys = clip_recording_planes(
        state, z, x, y, alive, blocked, keys
    )
    alive, blocked, keys = clip_column_wall(state, z, x, y, alive, blocked, keys)
    exit_reached = np.isnan(blocked) | (blocked > float(exit_z_mm) + 1.0e-9)

    branches = []
    for is_scattered, name, kind in (
        (False, "sample_region_primary", "sample_region_primary"),
        (True, "sample_region_elastic", "sample_region_elastic"),
    ):
        mask = scattered == is_scattered
        if not np.any(mask):
            continue
        branch_weights = weights[mask]
        weight_sum = float(np.sum(branch_weights))
        ray_weight = (
            branch_weights / weight_sum if weight_sum > 0.0 else branch_weights
        )
        branches.append(
            Branch(
                name=name,
                colour=RAY_INTERACTION_COLOURS[kind],
                z=z,
                x=x[:, mask],
                y=y[:, mask],
                tx=tx[:, mask],
                ty=ty[:, mask],
                alive=alive[mask],
                blocked_z=blocked[mask],
                blocked_key=[key for key, keep in zip(keys, mask, strict=True) if keep],
                weight=weight_sum,
                energy_offset_ev=(
                    energies[mask] - float(state.beam_voltage_kv) * 1000.0
                ),
                ray_weight=ray_weight,
                interaction_kind=kind,
            )
        )
    return tuple(branches), {
        "downstream_forward_weight": float(np.sum(weights)),
        "exit_plane_weight": float(np.sum(weights[exit_reached])),
        "downstream_source_ray_count": int(source_indices.size),
        "post_sample_reinjection_plane_z_mm": float(state.sample.z_mm),
        "material_terminal_z_collapsed_to_reference_plane": True,
    }


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
    scan_translation_nm = (
        sample_bundle.target_centroid_nm[0] - sample_bundle.original_centroid_nm[0],
        sample_bundle.target_centroid_nm[1] - sample_bundle.original_centroid_nm[1],
    )
    sample_by_index = {ray.source_ray_index: ray for ray in sample_bundle.rays}
    rows: list[SampleRegionElectronPath] = []
    for ray in entry_bundle.rays:
        sample_ray = sample_by_index.get(ray.source_ray_index)
        if sample_ray is None:
            continue
        rows.append(
            SampleRegionElectronPath(
                positions_mm=np.array(
                    (
                        (
                            (ray.position_xy_nm[0] + scan_translation_nm[0])
                            * 1.0e-6,
                            (ray.position_xy_nm[1] + scan_translation_nm[1])
                            * 1.0e-6,
                            entry_z_mm,
                        ),
                        (
                            sample_ray.position_xy_nm[0] * 1.0e-6,
                            sample_ray.position_xy_nm[1] * 1.0e-6,
                            float(state.sample.z_mm),
                        ),
                    )
                ),
                kind="boundary_input",
                weight=float(sample_ray.weight),
                kinetic_energy_ev=float(sample_ray.kinetic_energy_ev),
                provenance="cached global-column phase space",
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
) -> SampleRegionResult:
    """Run one explicit sample-local calculation from the cached column result."""

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

    spectrum = simulate_eds_point(
        state, detector_geometry, simulation=simulation
    )
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
    downstream_branches, downstream_metrics = _downstream_branches(
        state, elastic, exit_z_mm
    )
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
        **downstream_metrics,
    }
    return SampleRegionResult(
        entry_z_mm=entry_z_mm,
        exit_z_mm=exit_z_mm,
        electron_paths=electron_paths,
        photon_paths=photon_paths,
        downstream_branches=downstream_branches,
        spectrum=spectrum,
        metrics=metrics,
    )
