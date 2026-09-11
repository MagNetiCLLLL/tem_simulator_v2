"""Bounded ray-only tuning; never a specimen signal calculation.

X/Y are metres, slopes radians and axial positions millimetres. Medium
tuning traces interior source quadrature plus zero-current support probes.
An envelope is a display guide, not a reconstructed electron distribution.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TuningProfile:
    quality: str
    rays: int
    step_mm: float
    boundary_probes: int = 0


TUNING_PROFILES = {
    "Preview": TuningProfile("Preview", 49, 1.0),
    "Medium": TuningProfile("Medium", 193, .25, 33),
}


def is_tuning_quality(quality):
    return str(quality) in TUNING_PROFILES


def check_tuning_cancelled(state):
    callback = getattr(state, "_tuning_cancelled", None)
    if callback is not None and callback():
        raise RuntimeError("Superseded optical tuning request")


def prepare_tuning_snapshot(state, quality):
    profile = TUNING_PROFILES[quality]
    state._optical_tuning = True
    state._tuning_quality = quality
    emitter = getattr(state.electron_gun, "emitter", None)
    if emitter is not None:
        emitter._tuning_boundary_probes = profile.boundary_probes
    # Static user offsets still act. Time-dependent raster calibration,
    # multislice, inelastic branches and spectra are not tuning products.
    state.ac_deflector.scan_enabled = False
    state.descan_deflector.scan_enabled = False
    state.sample.wave_enabled = False
    state.sample.stem_wave_enabled = False
    state.sample.diffraction_enabled = False


def add_source_support_probes(bundle, emitter):
    count = int(getattr(emitter, "_tuning_boundary_probes", 0))
    if not count:
        return bundle
    if count != 33 or bundle.x_m.size <= count:
        raise ValueError("Medium tuning needs interior samples and 33 support probes")
    from dataclasses import replace
    interior = bundle.x_m.size - count
    arrays = {name: np.array(getattr(bundle, name), copy=True)
              for name in ("x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev", "weight")}
    # 8 azimuths x 4 source-position/angle orientations sample the 4-D
    # source support. They are not guaranteed global envelope extrema.
    phi = np.tile(np.arange(8) * (2*np.pi/8), 4)
    phase = np.repeat(np.arange(4) * (np.pi/2), 8)
    radius = 3 * emitter.virtual_source_fwhm_nm / 2.354820045 * 1e-9
    angle = emitter.angular_cutoff_mrad * 1e-3
    for name, values in (("x_m", radius*np.cos(phi)), ("y_m", radius*np.sin(phi)),
                         ("tx_rad", angle*np.cos(phi+phase)), ("ty_rad", angle*np.sin(phi+phase))):
        arrays[name][interior:-1] = values
        arrays[name][-1] = 0
    arrays["energy_offset_ev"][interior:] = 0
    arrays["weight"][:interior] = 1/interior
    arrays["weight"][interior:] = 0
    return replace(bundle, **arrays)


def tuning_metrics(state, incident):
    from temsim.physics.beam_statistics import branch_sample_statistics
    finite = np.isfinite(incident.x).all() and np.isfinite(incident.y).all()
    if not finite:
        raise ValueError("Non-finite tuning trajectory; reduce excitation or use a finer calculation")
    beam = branch_sample_statistics(incident) if np.any(incident.alive & (incident.ray_weight > 0)) else None
    return {
        "mode": state.projector_mode,
        "optical_tuning": True,
        "tuning_quality": getattr(state, "_tuning_quality", "requested_resolution"),
        "tuning_model": "sampled_field_rays_without_specimen_interactions",
        "sample_scattering_applied": False,
        "sample_scattering_model": "omitted_for_optical_tuning",
        "sample_inserted": bool(state.sample.inserted),
        "specimen_mode": state.sample.specimen_mode,
        "sample_beam_surviving_rays": int(np.count_nonzero(incident.alive)),
        "sample_beam_surviving_fraction": beam.surviving_fraction if beam else 0.,
        "sample_convergence_95_mrad": beam.convergence_95_mrad if beam else float("nan"),
        "sample_convergence_99_mrad": beam.convergence_99_mrad if beam else float("nan"),
        "sample_illumination_diameter_95_um": beam.illumination_diameter_95_um if beam else float("nan"),
        "sample_wavefront_curvature_per_m": beam.radial_wavefront_curvature_per_m if beam else float("nan"),
        "sample_waist_offset_mm": beam.waist_offset_m * 1e3 if beam else float("nan"),
        "branch_weights_are_absolute": True,
        "support_probe_count": int(np.count_nonzero(incident.ray_weight == 0)),
    }


def projected_support(branch, angle_deg=0.):
    """Sampled min/max guide, truncated at each ray's physical stop."""
    a = np.deg2rad(angle_deg)
    projected = (branch.x*np.cos(a) + branch.y*np.sin(a)) * 1e3
    blocked = np.asarray(branch.blocked_z)
    visible = (~np.isfinite(blocked)[None, :] | (branch.z[:, None] <= blocked[None, :]))
    valid = visible & np.isfinite(projected)
    lower = np.min(np.where(valid, projected, np.inf), axis=1)
    upper = np.max(np.where(valid, projected, -np.inf), axis=1)
    empty = ~np.any(valid, axis=1)
    lower[empty] = upper[empty] = np.nan
    return lower, upper
