"""Centre-anchored classical tip emission with unchanged local distributions.

The projected truncated Gaussian, local slopes, energies and current are held
fixed. The separate conductor adapter supplies this surface to the electrode
field solve; this module only constructs the emitted particle state.
"""
from dataclasses import replace

import numpy as np

MODEL = "analytic-tip-centred-curvature-v3"
FWHM_TO_SIGMA = 2.354820045


def support_radius_nm(emitter):
    return 3.0 * float(emitter.virtual_source_fwhm_nm) / FWHM_TO_SIGMA


def validate_curvature(emitter):
    if emitter.curvature_model != MODEL:
        raise ValueError("Unsupported continuous tip geometry model")
    curvature = float(emitter.curvature_nm_inv)
    if not np.isfinite(curvature) or curvature < 0:
        raise ValueError("Tip curvature must be finite and non-negative (nm^-1)")
    if curvature == 0:
        return
    if curvature and (emitter.surface_model is not None or emitter.coherence is not None):
        raise ValueError("Continuous curvature requires classical tip emission")
    # A graph over the projected disk must stay short of the hemisphere rim.
    if curvature * support_radius_nm(emitter) > 0.95:
        raise ValueError("Tip curvature times emitting support radius must not exceed 0.95")
    cutoff = max(0.0, float(emitter.angular_cutoff_mrad))*1e-3 if emitter.angular_rms_mrad > 0 else 0.0
    if curvature * support_radius_nm(emitter) >= 1/np.hypot(1.0, cutoff):
        raise ValueError("Tip curvature and local angular support must emit downstream")


def curve_bundle(bundle, emitter):
    """Bend upstream about the fixed apex and rotate local emission directions."""
    validate_curvature(emitter)
    k = emitter.curvature_nm_inv  # do geometry in nm, transport in metres
    if k == 0:
        return bundle  # preserve the flat source and its numerical path exactly
    x, y = bundle.x_m*1e9, bundle.y_m*1e9
    nx, ny = k*x, k*y
    nz = np.sqrt(1.0 - nx*nx - ny*ny)
    normal = np.column_stack((nx, ny, nz))
    # The tip centre/apex remains (0, 0, 0); every off-axis point bends upstream.
    # Rationalized spherical sag avoids cancellation as curvature tends to zero.
    z = -k*(x*x + y*y)/(1.0 + nz)
    tx = np.column_stack((1-nx*nx/(1+nz), -nx*ny/(1+nz), -nx))
    ty = np.column_stack((-nx*ny/(1+nz), 1-ny*ny/(1+nz), -ny))
    direction = normal + bundle.tx_rad[:, None]*tx + bundle.ty_rad[:, None]*ty
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    # Public slopes remain a projection; tracing consumes the full direction.
    if np.any(direction[:, 2] <= 0):
        raise ValueError("Tip curvature and local angular support must emit downstream")
    result = replace(bundle, tx_rad=direction[:, 0]/direction[:, 2],
                     ty_rad=direction[:, 1]/direction[:, 2])
    object.__setattr__(result, "surface_position_m", np.column_stack((bundle.x_m, bundle.y_m, z*1e-9)))
    object.__setattr__(result, "surface_direction", direction)
    object.__setattr__(result, "surface_normal", normal)
    object.__setattr__(result, "emission_geometry_model", emitter.curvature_model)
    return result
