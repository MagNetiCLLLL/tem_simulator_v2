"""EDS-only importance quadrature for an under-sampled finite specimen.

The original ray bundle is a discrete approximation to a continuous source.
Here a weighted Gaussian *kernel mixture*, not a single Gaussian beam, models
its sample-plane density. Scott's bandwidth is an explicit modelling choice:
it can smooth unresolved structure and does not establish physical overlap
where the available rays cannot resolve the beam. This module never replaces
the original elastic/downstream electron population.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.constants import c, physical_constants
from scipy.special import logsumexp
from scipy.stats import qmc

from temsim.specimen.interaction_types import IncidentElectronRay
from temsim.specimen.scene import SpecimenScene


@dataclass(frozen=True, slots=True)
class OverlapSamplingPlan:
    rays: tuple[IncidentElectronRay, ...]
    parent_source_ray_indices: tuple[int, ...]
    metrics: dict[str, object]


def _candidate_intersections(positions, slopes, half_size, thickness, shape, pad):
    """Finite straight-flight intersections, expanded by a field bound."""
    half = np.asarray(half_size, dtype=float) + float(pad)
    zlo = np.full(len(positions), -0.5 * thickness)
    zhi = np.full(len(positions), 0.5 * thickness)
    if shape == "rectangle":
        keep = np.ones(len(positions), dtype=bool)
        for axis in range(2):
            moving = np.abs(slopes[:, axis]) > 1.0e-15
            keep &= moving | (np.abs(positions[:, axis]) <= half[axis])
            speed = np.where(moving, slopes[:, axis], 1.0)
            first = (-half[axis] - positions[:, axis]) / speed
            second = (half[axis] - positions[:, axis]) / speed
            zlo = np.where(moving, np.maximum(zlo, np.minimum(first, second)), zlo)
            zhi = np.where(moving, np.minimum(zhi, np.maximum(first, second)), zhi)
        return keep & (zhi > zlo)
    # The shared disk envelope uses an ellipse when its two diameters differ.
    offset = positions / half
    velocity = slopes / half
    square_speed = np.sum(velocity * velocity, axis=1)
    closest_z = np.clip(
        -np.sum(offset * velocity, axis=1) / np.maximum(square_speed, 1.0e-300),
        zlo, zhi,
    )
    return np.sum((offset + velocity * closest_z[:, None]) ** 2, axis=1) < 1.0


def build_overlap_sampling_plan(state, incident_bundle, original_transport):
    """Return absolute conditional-current quadrature weights or a fallback.

    ``rays.weight`` sums to the probability mass represented by the candidate
    domain, not one. If a caller normalizes these rays for auxiliary elastic
    transport, it must multiply every resulting material-track weight by that
    original mass before EDS ionisation. Incident current remains source
    current times the original sample-plane survival fraction. Parent source
    IDs are retained separately; quadrature IDs are outside the source range.
    """
    sample = state.sample
    metrics = {
        "eds_overlap_sampling_status": "unavailable",
        "eds_overlap_sampling_detail": "",
        "eds_overlap_sampling_point_count": 0,
        "eds_overlap_probability_mass": 0.0,
        "eds_overlap_sampling_model": "weighted_ray_gaussian_kde_scott_sobol",
        "eds_overlap_sampling_scope": "EDS material integration only; original elastic/downstream rays unchanged",
        "eds_overlap_weight_reference": "conditional current reaching the original specimen plane; not renormalized to the hit subset",
        "eds_overlap_probability_semantics": "candidate material-intersection domain mass; auxiliary transport determines actual material paths",
        "eds_overlap_sampling_limitations": (
            "Scott KDE smooths unresolved source structure and has Gaussian tails; "
            "it is not a reconstructed coherent wave or proof of physical overlap. "
            "Parent-conditioned direction and energy are retained without local "
            "phase-space interpolation. Restricted to small thin specimens with "
            "small incident slopes and approximately constant weak local fields."
        ),
    }

    def fallback(status, detail):
        metrics["eds_overlap_sampling_status"] = status
        metrics["eds_overlap_sampling_detail"] = detail
        return OverlapSamplingPlan((), (), metrics)

    if not bool(getattr(sample, "eds_overlap_sampling_enabled", True)):
        return fallback("disabled", "EDS overlap quadrature is disabled.")
    if not bool(getattr(sample, "inserted", True)) or float(sample.thickness_nm) <= 0.0:
        return fallback("no_material", "A retracted or zero-thickness specimen creates no additional EDS material paths.")
    scene = SpecimenScene.from_state(state, include_eds_materials=True)
    if scene.sample_material is None or scene.matter_thickness_nm <= 0.0:
        return fallback("no_material", "No active real specimen material is available for overlap integration.")
    if not scene.support_grid.material.is_vacuum:
        return fallback("unsupported_support", "Overlap quadrature currently requires a vacuum support; sample/support coupled histories retain the original MC estimate.")
    transport_metrics = getattr(original_transport, "metrics", {})
    original_hits = transport_metrics.get("sample_hit_trajectory_count")
    if original_hits is None:
        return fallback("missing_hit_diagnostic", "Original sample-hit statistics are required before selecting EDS overlap integration.")
    metrics["eds_overlap_original_sample_hit_count"] = int(original_hits)
    if int(original_hits) >= 32:
        return fallback("adequate_original_hits", "At least 32 original trajectories already intersected the specimen; retain their elastic material estimate.")

    points_value = getattr(sample, "eds_overlap_sampling_points", 256)
    count = int(points_value)
    if count != points_value or not 32 <= count <= 4096:
        return fallback("invalid_point_count", "EDS overlap point count must be an integer in 32..4096.")
    metrics["eds_overlap_sampling_requested_points"] = count
    source_rows = sorted(
        (ray for ray in incident_bundle.rays if ray.weight > 0.0),
        key=lambda ray: ray.source_ray_index,
    )
    if len(source_rows) < 32:
        return fallback("insufficient_effective_rays", "At least 32 positive-weight original rays are required for a two-dimensional density estimate.")
    weights = np.asarray([ray.weight for ray in source_rows], dtype=float)
    if not np.all(np.isfinite(weights)) or float(weights.sum()) <= 0.0:
        return fallback("invalid_source_weights", "Original conditional ray weights must be finite and positive.")
    weights /= float(weights.sum())
    effective_count = 1.0 / float(np.sum(weights * weights))
    metrics["eds_overlap_effective_original_ray_count"] = effective_count
    if effective_count < 32.0 - 1.0e-10:
        return fallback("insufficient_effective_rays", "Unequal source weights leave fewer than 32 effective rays; retain the original MC estimate.")
    positions = np.asarray([ray.position_xy_nm for ray in source_rows], dtype=float)
    directions = np.asarray([ray.direction for ray in source_rows], dtype=float)
    energies = np.asarray([ray.kinetic_energy_ev for ray in source_rows], dtype=float)
    if (not np.all(np.isfinite(positions)) or not np.all(np.isfinite(directions))
            or not np.all(np.isfinite(energies)) or np.any(energies <= 0.0)
            or np.any(directions[:, 2] <= 0.0)):
        return fallback("invalid_incident_phase_space", "Overlap integration requires finite forward-going source rays with positive energies.")
    centroid = np.sum(weights[:, None] * positions, axis=0)
    centred = positions - centroid
    covariance = (centred.T @ (weights[:, None] * centred)) / (1.0 - float(np.sum(weights * weights)))
    eigenvalues = np.linalg.eigvalsh(covariance)
    if (not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] <= 0.0
            or eigenvalues[0] <= max(eigenvalues[1] * 1.0e-12, 1.0e-24)):
        return fallback("degenerate_covariance", "The incident positions do not resolve a nondegenerate two-dimensional density; no artificial bandwidth is inserted.")
    narrow_sigma = math.sqrt(float(eigenvalues[0]))
    size = np.asarray(scene.size_xy_nm, dtype=float)
    thickness = float(scene.matter_thickness_nm)
    metrics["eds_overlap_source_covariance_nm2"] = covariance.tolist()
    metrics["eds_overlap_source_centroid_nm"] = centroid.tolist()
    metrics["eds_overlap_sample_to_narrow_sigma_ratio"] = float(np.max(size) / narrow_sigma)
    if float(np.max(size)) > 0.5 * narrow_sigma:
        return fallback("sample_not_small_relative_to_beam", "The specimen is not small relative to the narrow principal beam width; retain the original resolved/focused-beam material paths.")
    slopes = directions[:, :2] / directions[:, 2, None]
    maximum_slope = float(np.max(np.linalg.norm(slopes, axis=1)))
    metrics["eds_overlap_maximum_incident_slope"] = maximum_slope
    if (thickness > float(np.min(size)) or maximum_slope > 0.1
            or thickness * maximum_slope > 0.1 * float(np.min(size))):
        return fallback("unsupported_thickness_or_angle", "The current integration-domain bound requires thickness <= the minimum specimen diameter, slopes <= 0.1, and transverse flight <= 10% of that diameter.")

    field = transport_metrics.get("sample_axial_field_t")
    faces = transport_metrics.get("sample_face_fields_t")
    field_source = str(transport_metrics.get("sample_field_source", ""))
    if (bool(transport_metrics.get("sample_field_geometry_material_coupled", False))
            or "measured/FEM" in field_source):
        return fallback("unsupported_mapped_field", "The current side-entry bound covers analytic axial fields only; measured/FEM grids, including arbitrary imported 3D fields, retain the original vector-field MC estimate.")
    from temsim.physics.core import multipole_focusing_fields, hexapole_field_components
    local_z_mm = float(sample.z_mm) + np.asarray((-0.5, 0.0, 0.5)) * thickness * 1.0e-6
    multipole_values = (*multipole_focusing_fields(local_z_mm, state),
                        *hexapole_field_components(local_z_mm, state))
    if any(np.any(np.asarray(values) != 0.0) for values in multipole_values):
        return fallback("unsupported_local_multipole", "Active specimen-local quadrupole/hexapole fields are outside the analytic axial side-entry bound; retain the original vector-field MC estimate.")
    if field is None or faces is None:
        return fallback("missing_field_diagnostic", "Original local-field diagnostics are required to bound curved side-entry paths.")
    fields = np.asarray((float(field), *faces), dtype=float)
    if fields.shape != (3,) or not np.all(np.isfinite(fields)):
        return fallback("invalid_field_diagnostic", "The local field must have finite centre and two-face values.")
    maximum_field = float(np.max(np.abs(fields)))
    variation = float(np.max(np.abs(fields - fields[0])))
    rest_energy_ev = physical_constants["electron mass energy equivalent in MeV"][0] * 1.0e6
    minimum_rigidity_tm = math.sqrt(float(energies.min()) * (float(energies.min()) + 2.0 * rest_energy_ev)) / c
    bend_angle = maximum_field * thickness * 1.0e-9 / minimum_rigidity_tm
    metrics["eds_overlap_local_field_turn_bound_rad"] = bend_angle
    if variation > max(1.0e-12, 0.01 * maximum_field) or bend_angle > 1.0e-4:
        return fallback("unsupported_local_field", "Local centre/face fields vary by more than 1%, or the estimated thin-slab turn exceeds 1e-4 rad; retain the original field-resolved MC estimate.")
    curvature_pad = 0.5 * bend_angle * thickness + 1.0e-9 * float(np.min(size))
    side_pad = 0.5 * thickness * np.max(np.abs(slopes), axis=0) + curvature_pad
    centre = np.asarray(scene.centre_xy_nm, dtype=float)
    lower = centre - 0.5 * size - side_pad
    upper = centre + 0.5 * size + side_pad
    area = float(np.prod(upper - lower))
    bandwidth_factor = effective_count ** (-1.0 / 6.0)
    kernel_covariance = covariance * bandwidth_factor**2
    inverse_cholesky = np.linalg.inv(np.linalg.cholesky(kernel_covariance))
    transformed_sources = centred @ inverse_cholesky.T
    log_normalizer = math.log(2.0 * math.pi) + 0.5 * float(np.linalg.slogdet(kernel_covariance)[1])
    seed = int(getattr(sample, "eds_elastic_seed", 0))
    if seed < 0:
        return fallback("invalid_seed", "EDS overlap quadrature requires a non-negative elastic seed.")
    unit_nodes = qmc.Sobol(d=3, scramble=True, seed=seed).random_base2((count - 1).bit_length())[:count]
    nodes = lower + unit_nodes[:, :2] * (upper - lower)
    densities = np.zeros(count)
    parent_indices = np.zeros(count, dtype=int)
    log_weights = np.log(weights)
    for start in range(0, count, 64):
        stop = min(start + 64, count)
        transformed_nodes = (nodes[start:stop] - centroid) @ inverse_cholesky.T
        delta = transformed_nodes[:, None, :] - transformed_sources[None, :, :]
        log_components = -0.5 * np.sum(delta * delta, axis=2) + log_weights[None, :]
        log_density = logsumexp(log_components, axis=1)
        densities[start:stop] = np.exp(log_density - log_normalizer)
        posterior = np.exp(log_components - log_density[:, None])
        cumulative = np.cumsum(posterior, axis=1)
        parent_indices[start:stop] = np.minimum(
            np.sum(cumulative < unit_nodes[start:stop, 2, None], axis=1), len(source_rows) - 1
        )
    absolute_weights = densities * (area / count)
    candidates = _candidate_intersections(
        nodes - centre, slopes[parent_indices], 0.5 * size,
        thickness, scene.envelope_shape, curvature_pad,
    ) & (absolute_weights > 0.0)
    selected = np.flatnonzero(candidates)
    mass = math.fsum(float(absolute_weights[index]) for index in selected)
    metrics.update({
        "eds_overlap_sampling_seed": seed,
        "eds_overlap_scott_bandwidth_factor": bandwidth_factor,
        "eds_overlap_kernel_covariance_nm2": kernel_covariance.tolist(),
        "eds_overlap_domain_bounds_nm": [float(lower[0]), float(upper[0]), float(lower[1]), float(upper[1])],
        "eds_overlap_side_pad_nm": side_pad.tolist(),
        "eds_overlap_domain_area_nm2": area,
        "eds_overlap_probability_mass": mass,
        "eds_overlap_quadrature_id_namespace": "IDs at or above emitted_ray_count; not original electron identities",
    })
    if not math.isfinite(mass) or mass > 1.0:
        return fallback("invalid_probability_mass", "The estimated conditional mass is not finite or exceeds one; it is rejected without clipping or renormalization.")
    if mass <= 0.0:
        return fallback("no_resolved_overlap", "The estimated density has no representable mass in the candidate material domain; no positive signal is fabricated.")
    metrics["eds_overlap_quadrature_effective_point_count"] = (
        1.0 / math.fsum((float(absolute_weights[index]) / mass)**2 for index in selected)
    )
    rays = tuple(
        IncidentElectronRay(
            source_ray_index=int(incident_bundle.emitted_ray_count) + int(index),
            position_xy_nm=tuple(float(value) for value in nodes[index]),
            direction=source_rows[parent_indices[index]].direction,
            kinetic_energy_ev=source_rows[parent_indices[index]].kinetic_energy_ev,
            weight=float(absolute_weights[index]),
        )
        for index in selected
    )
    parents = tuple(int(source_rows[parent_indices[index]].source_ray_index) for index in selected)
    metrics["eds_overlap_sampling_point_count"] = len(rays)
    metrics["eds_overlap_sampling_status"] = "active"
    metrics["eds_overlap_sampling_detail"] = (
        f"EDS-only weighted Scott-KDE/Sobol material integration: {len(rays)} candidate rays, "
        f"conditional mass {mass:.6g}, {effective_count:.6g} effective original rays. "
        "Absolute weights are retained; original elastic/downstream electrons are unchanged."
    )
    return OverlapSamplingPlan(rays, parents, metrics)
