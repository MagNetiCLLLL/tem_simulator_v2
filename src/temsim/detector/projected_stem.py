"""CIF-dependent particle STEM readout with cached physical response paths.

The projected-atom probabilities are observables of the executed incident
beam. They never replace its particle state or the material/EDS checkpoint.
One downstream response basis is reused across an affine scan; it is not a
full re-execution of nonlinear optics or stochastic vacuum at every pixel.
"""
from __future__ import annotations

import numpy as np

from temsim.optics.shared_deflectors import shared_channel_enabled

from temsim.physics.optical_tuning import check_tuning_cancelled
from temsim.physics.scan_geometry import paired_kick_response
from temsim.specimen.elastic_transport import incident_rays_from_simulation
from temsim.specimen.projected_scattering import projected_optical_depth


def _prepared_planes(state, response, planes):
    """Compact response samples; hardware losses survive raster readout."""
    from temsim.detector.stem_signal import _interpolate, _normalised_ray_weights

    branches = response.branches
    weights = np.concatenate([
        _normalised_ray_weights(branch) * float(branch.weight) for branch in branches
    ]) if branches else np.empty(0)
    output = {}
    for plane in planes:
        positions, reaches = [], []
        for branch in branches:
            if plane.z_mm < branch.z[0] - 1e-9 or plane.z_mm > branch.z[-1] + 1e-9:
                raise ValueError("A projected STEM response has not reached a recording plane.")
            positions.append(np.column_stack((
                _interpolate(branch.x, branch.z, plane.z_mm),
                _interpolate(branch.y, branch.z, plane.z_mm))) * 1e3)
            blocked = np.asarray(branch.blocked_z)
            # Response transport omits recording absorption only; the loop
            # below applies every inserted detector in its physical order.
            # An aperture, wall or medium stop is never revived here.
            reaches.append(np.isnan(blocked) | (blocked > plane.z_mm + 1e-9))
        output[plane.key] = (
            np.vstack(positions) if positions else np.empty((0, 2)),
            np.concatenate(reaches) if reaches else np.empty(0, dtype=bool),
        )
    return weights, output


def acquire_projected_stem_scan(
    simulation, state, inserted, *, scan_x_um, scan_y_um, kick_grid_mrad,
    baseline_scan_mrad, scan_times_s, baseline_descan_scan_mrad, sample_response,
    real_interactions=None, progress_callback=None,
):
    from temsim.detector.projected_response import build_projected_responses
    from temsim.detector.stem_signal import (
        DetectorSignal, _current_values, _stem_result, collection_angle,
    )

    check_tuning_cancelled(state)
    incident = incident_rays_from_simulation(state, simulation)
    baseline_um = np.asarray(sample_response) @ (np.asarray(baseline_scan_mrad) * 1e-3) * 1e3
    # The captured beam already contains the reference-time scan kick. Replace
    # that kick by each pixel command, rather than adding it twice.
    x_um = np.asarray(scan_x_um) + incident.original_centroid_nm[0] * 1e-3 - baseline_um[0]
    y_um = np.asarray(scan_y_um) + incident.original_centroid_nm[1] * 1e-3 - baseline_um[1]
    origin_mm = np.array((state.sample.scan_origin_x_nm, state.sample.scan_origin_y_nm)) * 1e-6
    origin_command = np.zeros(2)
    if np.any(origin_mm):
        response = np.asarray(sample_response)
        if not np.all(np.isfinite(response)) or np.linalg.cond(response) > 1e10:
            raise ValueError("The scan calibration cannot place the requested scan origin; recalibrate the scan coils.")
        origin_command = np.linalg.solve(response, origin_mm)
    projection = projected_optical_depth(state, simulation, x_um, y_um)
    stop = max(float(plane.z_mm) for plane in inserted)
    responses = build_projected_responses(
        state, simulation, projection, real_interactions,
        stop_z_mm=stop, progress_callback=progress_callback,
    )
    probabilities = {
        "vacuum": projection.vacuum_fraction,
        "direct": projection.direct_material_fraction,
        **{f"Z{element.atomic_number}": element.scattered_fraction for element in projection.elements},
    }
    probability_sum = sum(probabilities.values())
    if not np.allclose(probability_sum, 1.0, rtol=0, atol=2e-12):
        raise ValueError("Projected elastic/vacuum probabilities do not conserve incident current.")
    planes = sorted((plane for plane in state.recording_planes
                     if plane.inserted and state.sample.z_mm <= plane.z_mm <= stop),
                    key=lambda plane: float(plane.z_mm))
    prepared = {key: _prepared_planes(state, response, planes)
                for key, response in responses.items()}
    domain_masks = {
        key: (np.concatenate([np.asarray(branch.blocked_key) == "projected_field_domain"
                              for branch in response.branches])
              if response.branches else np.empty(0, dtype=bool))
        for key, response in responses.items()
    }
    plane_responses = {
        plane.key: (1e3 * paired_kick_response(state, state.ac_deflector, plane.z_mm),
                    1e3 * paired_kick_response(state, state.descan_deflector, plane.z_mm))
        for plane in planes
    }
    images = {plane.key: np.zeros_like(x_um) for plane in inserted}
    absorbed = np.zeros_like(x_um)
    truncated = np.zeros_like(x_um)
    survival = float(projection.surviving_fraction)
    for key, response in responses.items():
        absorbed += survival * probabilities[key] * float(
            response.metrics["inelastic_absorbed_source_probability"])
    if progress_callback:
        progress_callback(0, x_um.shape[0], "Integrating CIF-dependent physical detector signals")
    for row in range(x_um.shape[0]):
        check_tuning_cancelled(state)
        for col in range(x_um.shape[1]):
            scan_delta = (kick_grid_mrad[row, col] - baseline_scan_mrad) * 1e-3 + origin_command
            descan = state.descan_deflector
            descan_command = (descan.scan_kick_mrad(float(scan_times_s[row, col]))
                              if shared_channel_enabled(descan) else (0., 0.))
            descan_delta = (np.asarray(descan_command) - baseline_descan_scan_mrad) * 1e-3
            for key, (weights, plane_data) in prepared.items():
                pixel_weight = survival * float(probabilities[key][row, col])
                if pixel_weight <= 0:
                    continue
                available = np.ones(len(weights), dtype=bool)
                for plane in planes:
                    position, reaches = plane_data[plane.key]
                    scan_response, descan_response = plane_responses[plane.key]
                    shifted = position + scan_response @ scan_delta + descan_response @ descan_delta
                    hit = available & reaches & plane.hit_mask(shifted[:, 0], shifted[:, 1])
                    if plane.key in images:
                        images[plane.key][row, col] += pixel_weight * float(weights[hit].sum())
                    available[hit] = False
                truncated[row, col] += pixel_weight * float(weights[available & domain_masks[key]].sum())
        if progress_callback:
            progress_callback(row + 1, x_um.shape[0], f"Projected STEM row {row + 1}/{x_um.shape[0]}")
    collected = sum(images.values(), np.zeros_like(x_um))
    remainder = 1.0 - collected - absorbed - truncated
    if np.any(remainder < -2e-12):
        raise ValueError("Projected STEM readout counted more than the emitted current.")
    signals = {}
    for detector in inserted:
        fraction = float(images[detector.key].mean())
        simulated, current, electrons = _current_values(state, fraction)
        signals[detector.key] = DetectorSignal(detector.key, detector.name, fraction,
                                              simulated, current, electrons, collection_angle(state, detector))
    metadata = {
        "model": "projected_atomic_scattering", "quantitative_model": False,
        "pixel_resolved_specimen_contrast": True, "coherent_imaging": False,
        "projected_scattering": dict(projection.metrics),
        "response_transport": {key: dict(value.metrics) for key, value in responses.items()},
        "probe_sigma_principal_nm": projection.metrics["probe_sigma_principal_nm"],
        "maximum_plural_event_fraction": projection.metrics["maximum_plural_event_fraction"],
        "physical_detector_masks": True, "sequential_detector_interception": True,
        "post_sample_lens_transport_applied": True,
        "detector_signal_requires_physical_intersection": True,
        "sample_incident_source_probability": survival,
        "source_probability_conservation_error": float(np.max(np.abs(collected + absorbed + truncated + np.maximum(remainder, 0) - 1))),
        "maximum_numerical_truncation_fraction": float(truncated.max()),
        "scan_pixels_x": x_um.shape[1], "scan_pixels_y": x_um.shape[0],
        "scan_pixel_size_nm": float(state.ac_deflector.scan_pixel_size_nm),
        "scan_field_of_view_x_nm": float(state.ac_deflector.scan_field_of_view_x_nm),
        "scan_field_of_view_y_nm": float(state.ac_deflector.scan_field_of_view_y_nm),
        "descan_applied": bool(shared_channel_enabled(state.descan_deflector) and state.descan_deflector.scan_enabled),
        "model_limitation": (
            "Independent-atom, thin projected scattering with an executed Gaussian probe. "
            "HAADF estimates incoherent atomic contrast. BF/DF show particle redistribution, "
            "not Bragg interference, phase contrast, channeling or quantitative multislice agreement."
        ),
        "approximation_notes": (
            "At least one elastic event is represented by one effective angular draw; plural angular scattering is not resolved.",
            "Cached downstream trajectories include lenses, deflectors, apertures, walls and optional vacuum at the reference scan position; raster displacement uses the local affine response.",
            "Raster positions do not resample nonlinear clipping or stochastic vacuum collisions at each pixel.",
            "The atomic intensity closure does not replace material particle trajectories, EDS histories or resumable checkpoints.",
            "The executed probe width is retained. A broad probe or coarse pixel sampling can remove atomic contrast.",
        ),
    }
    return _stem_result(state, simulation, x_um, y_um, images, signals, metadata,
                        uncollected_fraction=np.maximum(remainder, 0), absorbed_fraction=absorbed,
                        truncated_fraction=truncated)
