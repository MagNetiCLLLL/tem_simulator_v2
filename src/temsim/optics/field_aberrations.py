"""Finite-aperture field-ray fits with independent holdout and energy checks.

First-order focus is measured for diagnostics but excluded from the returned
wave correction: illumination / camera transport already contains that focus.
Unmapped round lenses cannot supply missing high-order field information.
"""

import copy
import numpy as np

from temsim.optics.aberration_basis import (
    ABERRATION_SCHEMA, CARTESIAN_NAMES, POWERS, UNIMPLEMENTED_TERMS,
    coefficient_fields, polynomial_basis, predict_displacement_m,
)


def _basis(u):
    return polynomial_basis(u)


def fit_wave_gradient(angles_rad, displacement_m, semiangle_rad):
    """Recover fifteen length coefficients from minus wavefront gradients.

    RMS is per Cartesian component, in m; conditioning is of the pupil-scaled
    design matrix, not of coefficients with unlike powers of angle.
    """
    scale = float(semiangle_rad)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Aberration fit semi-angle must be positive")
    angles = np.asarray(angles_rad, float)
    gradients = polynomial_basis(angles / scale, gradient=True)
    displacement = np.asarray(displacement_m, float)
    if displacement.shape != angles.shape or not np.isfinite(displacement).all():
        raise ValueError("Ray displacements must be a matching finite N by 2 array in m")
    if np.any(np.linalg.norm(angles, axis=1) > scale * (1+1e-12)):
        raise ValueError("Fit rays extend outside the declared semi-angle")
    design = np.concatenate((gradients[:, 0], gradients[:, 1]))
    target = -displacement.T.ravel() * scale
    coefficients, _, rank, singular = np.linalg.lstsq(design, target, rcond=1e-10)
    if rank != design.shape[1]:
        raise ValueError("Aberration ray fit is rank deficient")
    rms = float(np.sqrt(np.mean((design @ coefficients - target)**2)) / scale)
    return coefficients / scale**POWERS * 1e3, rms, float(singular[0] / singular[-1])


def fit_options(options, state_step_mm):
    """Validate saved / GUI numerical settings without running any solver."""
    alpha = float(options.get("fit_semiangle_mrad", 10.0)) * 1e-3
    requested_step = float(options.get("fit_step_mm", .025))
    step = min(float(state_step_mm), requested_step)
    steps = np.asarray(options.get("fit_energy_relative_steps", (.002, .001, .0005)), float)
    if not (0 < alpha <= .1):
        raise ValueError("Field aberration fit requires a semi-angle in (0, 100] mrad")
    if not (np.isfinite(state_step_mm) and state_step_mm > 0
            and np.isfinite(requested_step) and requested_step > 0):
        raise ValueError("Field fit step must be finite and positive in mm")
    if (steps.ndim != 1 or not 3 <= len(steps) <= 6 or not np.isfinite(steps).all()
            or np.any(steps < 1e-7) or np.any(steps > .01) or np.any(np.diff(steps) >= 0)):
        raise ValueError("Cc requires 3–6 decreasing relative energy steps in [1e-7, 0.01]")
    return alpha, step, steps


def pupil_rays(alpha, *, holdout=False):
    # Distinct angles AND radii; holdout rays never enter the least-squares fit.
    count = 23 if holdout else 24
    phi = (np.arange(count) + (.37 if holdout else 0)) * (2*np.pi/count)
    radii = np.linspace(.11, .97, 7) if holdout else np.linspace(.15, 1, 8)
    return alpha * np.column_stack(((radii[:, None]*np.cos(phi)).ravel(),
                                    (radii[:, None]*np.sin(phi)).ravel()))


def field_grid_evidence(mapped):
    return [{"lens_key": provider.lens_key, "map_type": provider.field_map.map_type,
             "shape": [len(axis) for axis in provider.field_map.axes_m],
             "spacing_range_m": [[float(np.min(np.diff(axis))), float(np.max(np.diff(axis)))]
                                 for axis in provider.field_map.axes_m],
             "extent_m": [[float(axis[0]), float(axis[-1])] for axis in provider.field_map.axes_m],
             "content_fingerprint": provider.field_map.content_fingerprint,
             "geometry_fingerprint": provider.field_map.geometry_fingerprint,
             "source_kind": provider.field_map.provenance.kind,
             "source_sha256": provider.field_map.provenance.source_sha256,
             "source_note": provider.field_map.provenance.source_note,
             "excitation_scaling": provider.excitation_scaling} for provider in mapped]


def derive_field_aberrations(state, system):
    from temsim.optics.aberrations import EffectiveAberrationSet, SYSTEM_COEFFICIENT_ROWS
    from temsim.physics.core import build_propagation_plan, execute_propagation_plan
    from temsim.physics.first_order import trace_transverse_transfer
    from temsim.physics.illumination import state_at_energy
    from temsim.physics.lens_field_provider import active_mapped_providers
    if system not in {"probe", "image"}:
        raise ValueError("Aberration system must be probe or image")
    options = getattr(state, f"{system}_aberrations", {})
    alpha, step, energy_steps = fit_options(options, float(state.step_mm))
    mapped = active_mapped_providers(state)
    work = copy.copy(state)
    work.equivalent_image_lenses_enabled = False
    work.step_mm = step
    if system == "probe":
        z0, z1 = float(state.electron_gun.exit_plane_z_mm), float(state.sample.z_mm)
    else:
        z0, z1 = float(state.sample.z_mm), float(state.objective_image_plane_z_mm)
    if not (np.isfinite([z0, z1]).all() and z1 > z0):
        raise ValueError("Field aberration fit requires an ordered physical reference interval")
    mapped = tuple(provider for provider in mapped
                   if provider.field_support_mm()[1] >= z0 and provider.field_support_mm()[0] <= z1)
    if not mapped:
        raise ValueError("Field-derived aberrations require at least one registered or generated field map in the reference interval")
    transfer = trace_transverse_transfer(work, z0, z1, maximum_step_mm=step)
    training, holdout = pupil_rays(alpha), pupil_rays(alpha, holdout=True)
    pupil = np.vstack((training, holdout))
    reference = np.column_stack((np.zeros_like(pupil), pupil))
    if system == "probe":
        inputs = np.linalg.solve(transfer.matrix, reference.T).T
        normalise = np.eye(2)
    else:
        inputs = reference
        if np.linalg.cond(transfer.j_img) > 1e8:
            raise ValueError("Objective image reference is singular for an aberration fit")
        normalise = np.linalg.inv(transfer.j_img)
    # Keep initial rays and nominal reference FIXED at every energy perturbation.
    inputs = np.vstack((np.zeros((1, 4)), inputs))
    integration_grids = []

    def intercepts(snapshot):
        plan = build_propagation_plan(snapshot, z0, z1, include_spherical_aberration=False,
                                      include_hexapole=True, save_z_mm=(z1,))
        integration_grids.append({"energy_kev": float(snapshot.beam_voltage_kv),
                                  "interval_count": len(plan.step_m),
                                  "actual_step_range_mm": [float(np.min(plan.step_m)*1e3), float(np.max(plan.step_m)*1e3)]})
        trace = execute_propagation_plan(snapshot, plan, inputs[:, 0], inputs[:, 2], inputs[:, 1], inputs[:, 3])
        positions = np.column_stack((trace[1][-1], trace[3][-1]))
        if not np.isfinite(positions).all():
            raise ValueError("Nonfinite field-fit ray intercepts; reduce the aperture or inspect field support")
        return (positions[1:] - positions[0]) @ normalise.T

    linear = ((transfer.matrix @ inputs[1:].T).T[:, :2]) @ normalise.T
    residual = intercepts(work) - linear
    coefficients, rms, condition = fit_wave_gradient(training, residual[:len(training)], alpha)
    validation_error = predict_displacement_m(holdout, coefficients) - residual[len(training):]
    cc_values = []
    for delta in energy_steps:
        positions = [intercepts(state_at_energy(work, state.beam_voltage_kv * (1 + sign*delta)))
                     for sign in (-1, 1)]
        derivative = (positions[1] - positions[0]) / (2*delta)
        # Separate chromatic first-order focus from higher-order angular terms.
        cc_fit, _, _ = fit_wave_gradient(training, derivative[:len(training)], alpha)
        cc_values.append(float(cc_fit[0]))
    cc_floor, cc_relative_target = 1e-6, .01
    comparisons = [{"relative_energy_step": float(delta), "absolute_change_mm": abs(value-previous),
                    "acceptance_bound_mm": cc_floor + cc_relative_target*abs(value),
                    "status": "PASS" if abs(value-previous) <= cc_floor + cc_relative_target*abs(value) else "FAIL"}
                   for delta, previous, value in zip(energy_steps[1:], cc_values, cc_values[1:])]
    cc_evidence = {"relative_energy_steps": energy_steps.tolist(), "cc_mm": cc_values,
                   "fixed_hardware": True, "fixed_initial_rays_and_reference": True,
                   "absolute_floor_mm": cc_floor, "relative_target": cc_relative_target,
                   "comparisons": comparisons,
                   "status": "PASS" if all(r["status"] == "PASS" for r in comparisons[-2:]) else "INCONCLUSIVE"}
    active_keys = {item.lens_key for item in mapped}

    def intersects(lens):
        try:
            start, end = lens.field_support_mm(state.beam_voltage_kv)
        except (AttributeError, TypeError):
            start = end = float(lens.z_mm)
        return end >= z0 and start <= z1

    unfitted = tuple(lens.key for lens in state.lenses
                     if lens.enabled and lens.key not in active_keys and intersects(lens))
    support = "partial_field_support" if unfitted else "mapped_round_lens_support"
    values = {"reference_plane": "sample/probe" if system == "probe" else "objective image",
              "correction_state": "field-derived", **coefficient_fields(coefficients, include_focus=False),
              "cc_mm": cc_values[-1], "status": support,
              "source": "production field rays; no manual coefficients or empirical Cs kicks"}
    result = EffectiveAberrationSet(**values).validate()
    status = {term: "field ray fit" for term, _, _ in SYSTEM_COEFFICIENT_ROWS}
    status.update(C1="first-order transport; not added again", Cc="fixed-hardware energy difference; " + cc_evidence["status"])
    diagnostics = {"schema": ABERRATION_SCHEMA, "source": result.source, "coefficient_status": status,
                   "beam_energy_kev": float(state.beam_voltage_kv),
                   "hardware_operating_point": [
                       {"key": item.key, **{field: getattr(item, field) for field in
                           ("z_mm", "enabled", "percent", "polarity", "strength_m2", "strength_m3",
                            "orientation_rad", "kick_x_mrad", "kick_y_mrad") if hasattr(item, field)}}
                       for item in (*state.lenses, *getattr(state, "corrector_elements", ()))],
                   "inferred_coefficients": tuple(term for term, _, _ in SYSTEM_COEFFICIENT_ROWS if term != "C1"),
                   "unmeasured_coefficients": (), "unimplemented_terms": UNIMPLEMENTED_TERMS,
                   "fit_rms_m": rms, "fit_condition_number": condition,
                   "fit_semiangle_mrad": alpha*1e3, "fit_step_mm": step,
                   "integration_grids": integration_grids,
                   "reference_interval_mm": [z0, z1], "training_ray_count": len(training),
                   "holdout_ray_count": len(holdout), "holdout_independent": True,
                   "validation_range_mrad": [float(np.linalg.norm(holdout, axis=1).min()*1e3),
                                              float(np.linalg.norm(holdout, axis=1).max()*1e3)],
                   "holdout_rms_m": float(np.sqrt(np.mean(validation_error**2))),
                   "holdout_max_error_m": float(np.linalg.norm(validation_error, axis=1).max()),
                   "cartesian_coefficient_names": CARTESIAN_NAMES, "cartesian_coefficients_mm": coefficients.tolist(),
                   "transport_focus_removed_mm": float(coefficients[0]),
                   "field_support_status": support, "unmapped_round_lenses": unfitted,
                   "field_grids": field_grid_evidence(mapped), "chromatic_convergence": cc_evidence,
                   "convergence_status": "NOT_ASSESSED; refine semi-angle, ray step and independently generated field grids",
                   "diagnostic_scope": "Finite-aperture gradient fit with independent holdout; incomplete high-order basis. Unmapped round-lens high-order fields are not inferred. Cc varies energy at fixed hardware and reference planes; no autofocus.",
                   "correction_comparison_available": False,
                   "ray_error_rms_m": float(np.sqrt(np.mean(residual[:len(training)]**2))),
                   "ray_error_rms_unit": "m"}
    return result, result, diagnostics
