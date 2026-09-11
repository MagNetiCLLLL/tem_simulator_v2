"""Honest admission for the gun-to-specimen coherent source-chain migration.

Tip mutual intensity and a development quadratic gun operator are available.
Column/specimen/detector, timed scan and conditional material-wave development
stages now exist; general-field and full-chain qualification remain open. Neither
a tip parameter label nor a development checkpoint qualifies a full image.
"""
from __future__ import annotations

import numpy as np

from temsim.immutable_json import freeze_json

PLANCK_J_S = 6.62607015e-34  # Exact SI definition.


class UnsupportedWaveSource(ValueError):
    code = "GUN_PHASE_CHAIN_UNAVAILABLE"


def gun_phase_readiness(state):
    gun = state.electron_gun
    from temsim.optics.electron_gun.source_policy import require_physical_gun_source, UnsupportedSourceModel
    try:
        require_physical_gun_source(gun)
    except UnsupportedSourceModel as error:
        return freeze_json({"status": "UNAVAILABLE", "code": error.code,
            "gun_model": str(gun.type_key), "reason": str(error),
            "coherent_source": "PROHIBITED_CUSTOM_EXIT_SOURCE",
            "gun_phase_transfer": "NOT_COMPUTED", "validation_status": "NOT_RUN"})
    from temsim.optics.electron_gun.tip_coherence import TipCoherence
    configured = isinstance(getattr(gun.emitter, "coherence", None), TipCoherence)
    return freeze_json({
        "status": "NOT_COMPUTED", "code": UnsupportedWaveSource.code,
        "gun_model": str(gun.type_key), "particle_source": "gun emission and relativistic Lorentz transport",
        "coherent_source": "TIP_GAUSSIAN_SCHELL_CONFIGURED" if configured else "TIP_COHERENCE_NOT_CONFIGURED",
        "gun_phase_transfer": "DEVELOPMENT_QUADRATIC_OPERATOR_AVAILABLE_NOT_COMPUTED",
        "stationary_zero_loss_pipeline": "DEVELOPMENT_AVAILABLE_NOT_FULL_PRODUCT_ACCEPTANCE",
        "dynamic_scan_and_inelastic_waves": "DEVELOPMENT_ARRIVAL_TIME_COILS_AND_CONDITIONAL_MATERIAL_TRAJECTORIES",
        "validation_status": "NOT_RUN",
        "missing": (["Explicit tip coherence parameters"] if not configured else []) +
                   ["General non-paraxial/imported-field gun phase operators",
                    "Converged physical-tip column/specimen/scanned-detector benchmarks",
                    "Longitudinal pulse wavepackets and atomic inelastic transition potentials",
                    "Energy-filter wave integration after upstream completion"],
    })


def require_gun_wave_source(state, *, product):
    """Closed production boundary until the full requested product is qualified.

    Do not accept arbitrary private attributes, metadata or caller-supplied
    arrays as proof. This deliberately cannot be enabled with origin='gun'.
    Ray tracing, EDS particle transport and historical viewing remain available.
    """
    readiness = gun_phase_readiness(state)
    if readiness["code"] == "TIP_ORIGIN_REQUIRED":
        raise UnsupportedWaveSource(f"{product}: {readiness['reason']}")
    raise UnsupportedWaveSource(
        f"{product}: gun-to-specimen coherent phase is unavailable for {readiness['gun_model']}. "
        "The existing ray-conditioned pupil is historical/reduced-order only. "
        "Tip coherence and a development quadratic gun operator are available, but are not full image admission. "
        "Development timed scan coils and conditional material-model inelastic waves are connected; full image qualification remains open. "
        "An effective exit source is not permitted."
    )


def admit_requested_wave_products(state):
    """Reject unsupported new wave work before heavy preparation/cache reuse."""
    from temsim.physics.wave_imaging import tem_wave_imaging_enabled
    from temsim.optics.electron_gun.source_policy import require_physical_gun_source
    require_physical_gun_source(state.electron_gun)
    sample = state.sample
    scan = getattr(state, "ac_deflector", None)
    if tem_wave_imaging_enabled(state) or (scan is not None and scan.enabled and scan.scan_enabled
                                         and bool(getattr(sample, "stem_wave_enabled", False))):
        require_gun_wave_source(state, product="Wave imaging")


def launch_emittance_audit(gun, *, count=4096):
    """Audit a possible literal quantum interpretation of the CLASSICAL source.

    This is not a coherence reconstruction. For each Cartesian pair, a quantum
    covariance must satisfy sqrt(var(x)var(px)-cov(x,px)^2) >= hbar/2.
    The ratio below is a necessary test, not sufficient quantum validation.
    The launch plane is z=0 of tracing.trace_feg_to_exit. No lens is adjusted.
    """
    from temsim.physics.relativistic_lorentz import momentum_from_kinetic_energy_ev
    emitted = gun.emit(count)
    energy = float(gun.emitter.emission_energy_ev) + emitted.energy_offset_ev
    direction = np.column_stack((emitted.tx_rad, emitted.ty_rad, np.ones(energy.size)))
    momentum = momentum_from_kinetic_energy_ev(energy, direction)
    weight = np.asarray(emitted.weight, float)
    weight = weight / weight.sum()
    values = np.column_stack((emitted.x_m, emitted.y_m, momentum[:, 0], momentum[:, 1]))
    mean = weight @ values
    centred = values - mean
    covariance = (centred.T * weight) @ centred
    minimum_action = PLANCK_J_S / (4*np.pi)
    action = [float(np.sqrt(max(0., covariance[k, k]*covariance[k+2, k+2]-covariance[k, k+2]**2))) for k in (0, 1)]
    return {"schema": "classical-launch-emittance-audit-v1", "sample_count": int(count),
            "reference_plane": "gun launch z=0 mm", "energy_min_ev": float(energy.min()), "energy_max_ev": float(energy.max()),
            "action_rms_j_s": action, "quantum_minimum_action_j_s": minimum_action,
            "quantum_bound_ratio_xy": [v/minimum_action for v in action],
            "necessary_quantum_covariance_condition": "PASS" if min(action) >= minimum_action else "FAIL",
            "scope": "Classical model interpretation audit only; does not create or validate coherent modes"}
