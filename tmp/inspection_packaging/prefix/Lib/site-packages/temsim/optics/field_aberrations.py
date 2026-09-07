"""Fit a wave-aberration gradient to production-field ray intercept errors.

This is a finite-aperture least-squares approximation, not an OEM calibration.
First-order focus is removed because the ray-derived illumination and camera
LCT already include it. No native empirical Cs kick or manual coefficient is
added during this fit. Analytic paraxial providers cannot supply missing
round-lens high-order information; diagnostics expose their presence.
"""

import copy
import numpy as np


def _basis(u):
    x, y = u[:, 0], u[:, 1]
    q, r2 = x + 1j*y, x*x + y*y
    return np.column_stack((r2/2, (q*q).real/2, (q*q).imag/2,
                            r2*x/3, r2*y/3, (q**3).real/3, (q**3).imag/3,
                            r2*r2/4, r2*(q*q).real/4, r2*(q*q).imag/4,
                            (q**4).real/4, (q**4).imag/4, r2**3/6))


def fit_wave_gradient(angles_rad, displacement_m, semiangle_rad):
    """Recover conventional length coefficients from minus wavefront gradients."""
    scale = float(semiangle_rad)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Aberration fit semi-angle must be positive")
    u = np.asarray(angles_rad, float) / scale
    h = 1e-5
    gradients = []
    for axis in range(2):
        step = np.zeros(2)
        step[axis] = h
        gradients.append((_basis(u + step) - _basis(u - step)) / (2*h))
    design = np.vstack(gradients)
    target = -np.asarray(displacement_m, float).T.ravel() * scale
    coefficients, _, rank, singular = np.linalg.lstsq(design, target, rcond=1e-10)
    if rank != design.shape[1]:
        raise ValueError("Aberration ray fit is rank deficient")
    rms = float(np.sqrt(np.mean((design @ coefficients - target)**2)) / scale)
    orders = np.array((2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 6))
    return coefficients / scale**orders * 1e3, rms, float(singular[0] / singular[-1])


def derive_field_aberrations(state, system):
    from temsim.optics.aberrations import EffectiveAberrationSet, SYSTEM_COEFFICIENT_ROWS
    from temsim.physics.core import propagate
    from temsim.physics.first_order import trace_transverse_transfer
    from temsim.physics.lens_field_provider import active_mapped_providers
    options = getattr(state, f"{system}_aberrations", {})
    alpha = float(options.get("fit_semiangle_mrad", 10.0)) * 1e-3
    if not (0 < alpha <= .1):
        raise ValueError("Field aberration fit requires a semi-angle in (0, 100] mrad")
    mapped = active_mapped_providers(state)
    if not mapped:
        raise ValueError("Field-derived aberrations require at least one registered or generated field map")
    work = copy.copy(state)
    work.equivalent_image_lenses_enabled = False
    work.step_mm = min(float(state.step_mm), .025)
    if system == "probe":
        z0, z1 = float(state.electron_gun.exit_plane_z_mm), float(state.sample.z_mm)
    else:
        z0, z1 = float(state.sample.z_mm), float(state.objective_image_plane_z_mm)
    if z1 <= z0:
        raise ValueError("Field aberration fit requires an ordered physical reference interval")
    transfer = trace_transverse_transfer(work, z0, z1, maximum_step_mm=.025)
    phi = np.arange(24) * (2*np.pi/24)
    radii = np.linspace(.15, 1, 8)
    pupil = alpha * np.column_stack(((radii[:, None]*np.cos(phi)).ravel(), (radii[:, None]*np.sin(phi)).ravel()))
    reference = np.column_stack((np.zeros_like(pupil), pupil))
    if system == "probe":
        inputs = np.linalg.solve(transfer.matrix, reference.T).T
        normalise = np.eye(2)
    else:
        inputs = reference
        if np.linalg.cond(transfer.j_img) > 1e8:
            raise ValueError("Objective image reference is singular for an aberration fit")
        normalise = np.linalg.inv(transfer.j_img)
    # Include the zero ray to remove field-induced translation, not beam tilt.
    inputs = np.vstack((np.zeros((1, 4)), inputs))
    trace = propagate(work, z0, z1, inputs[:, 0], inputs[:, 2], inputs[:, 1], inputs[:, 3],
                      include_spherical_aberration=False, include_hexapole=True, save_z_mm=(z1,))
    positions = np.column_stack((trace[1][-1], trace[3][-1]))
    linear = (transfer.matrix @ inputs.T).T[:, :2]
    residual = (positions[1:] - positions[0] - linear[1:]) @ normalise.T
    coefficients, rms, condition = fit_wave_gradient(pupil, residual, alpha)
    energy_delta = 1e-3
    chromatic_positions = []
    for sign in (-1, 1):
        shifted = copy.copy(work)
        shifted.beam_voltage_kv = state.beam_voltage_kv * (1 + sign * energy_delta)
        shifted_trace = propagate(shifted, z0, z1, inputs[:, 0], inputs[:, 2], inputs[:, 1], inputs[:, 3],
                                  include_spherical_aberration=False, include_hexapole=True, save_z_mm=(z1,))
        values_at_plane = np.column_stack((shifted_trace[1][-1], shifted_trace[3][-1]))
        chromatic_positions.append((values_at_plane[1:] - values_at_plane[0]) @ normalise.T)
    chromatic_derivative = (chromatic_positions[1] - chromatic_positions[0]) / (2 * energy_delta)
    cc_mm = -float(np.sum(chromatic_derivative * pupil) / np.sum(pupil*pupil)) * 1e3
    values = {"reference_plane": "sample/probe" if system == "probe" else "objective image",
              "correction_state": "field-derived", "c3_mm": float(coefficients[7]), "c5_mm": float(coefficients[12]), "cc_mm": cc_mm,
              "status": "finite_aperture_field_ray_fit", "source": "production field rays; no manual coefficients or empirical Cs kicks"}
    for name, index, harmonic in (("a1", 1, 2), ("b2", 3, 1), ("a2", 5, 3), ("s3", 8, 2), ("a3", 10, 4)):
        complex_value = complex(coefficients[index], coefficients[index + 1])
        values[f"{name}_mm"] = abs(complex_value)
        values[f"{name}_azimuth_deg"] = np.degrees(np.angle(complex_value)) / harmonic
    active_keys = {item.lens_key for item in mapped}
    unfitted = tuple(lens.key for lens in state.lenses if lens.enabled and lens.key not in active_keys
                    and z0 <= float(lens.z_mm) <= z1)
    result = EffectiveAberrationSet(**values).validate()
    status = {term: "field ray fit" for term, _, _ in SYSTEM_COEFFICIENT_ROWS}
    status.update(C1="first-order transport; not added again", Cc="energy-perturbed field ray fit")
    diagnostics = {"source": result.source, "coefficient_status": status,
                   "inferred_coefficients": ("A1", "B2", "A2", "C3", "S3", "A3", "C5", "Cc"),
                   "unmeasured_coefficients": (), "fit_rms_m": rms, "fit_condition_number": condition,
                   "fit_semiangle_mrad": alpha*1e3, "unmapped_round_lenses": unfitted,
                   "diagnostic_scope": "Finite-aperture fit; high-order fields of unmapped round lenses are not inferred; Cc uses a symmetric 0.1% energy perturbation",
                   "correction_comparison_available": False,
                   "ray_error_rms_m": float(np.sqrt(np.mean(residual**2))),
                   "ray_error_rms_unit": "m"}
    return result, result, diagnostics
