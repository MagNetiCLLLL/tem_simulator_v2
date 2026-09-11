"""Honest admission for the gun-to-specimen coherent source-chain migration.

The currently registered guns implement classical particle emission and Lorentz
transport. Neither an origin label nor a fitted specimen pupil supplies their
missing mutual intensity and coherent gun transfer. This module never promotes
such data to a production wave checkpoint.
"""
from __future__ import annotations

import numpy as np

from temsim.immutable_json import freeze_json

PLANCK_J_S = 6.62607015e-34  # Exact SI definition.


class UnsupportedWaveSource(ValueError):
    code = "GUN_PHASE_CHAIN_UNAVAILABLE"


def gun_phase_readiness(state):
    gun = state.electron_gun
    if getattr(gun, "source_representation", "classical_particles") == "effective_gaussian_schell":
        from temsim.optics.electron_gun.effective_source import generate_gun_emission
        try:
            source = generate_gun_emission(gun)
        except (ValueError, TypeError) as error:
            return freeze_json({"status": "UNAVAILABLE", "code": "GUN_SOURCE_SETUP_REQUIRED",
                "gun_model": str(gun.type_key), "reason": str(error),
                "coherent_source": "UNAVAILABLE", "gun_phase_transfer": "NOT_COMPUTED",
                "validation_status": "NOT_RUN"})
        return freeze_json({"status": "PARTIAL", "code": "GUN_IMAGE_ADAPTER_NOT_INTEGRATED",
            "gun_model": source.source_parameters.model_id, "source_id": source.digest,
            "coherent_source": "AVAILABLE_EFFECTIVE_EXIT_MODEL",
            "gun_phase_transfer": "QUADRATIC_COMPONENT_DOMAIN_ONLY",
            "validation_status": "PARTIAL_NUMERICAL_TESTS",
            "missing": ["Production TEM/STEM specimen and detector adapters",
                        "Complete-domain, scan and independent specimen validation"]})
    return freeze_json({
        "status": "NOT_COMPUTED", "code": UnsupportedWaveSource.code,
        "gun_model": str(gun.type_key), "particle_source": "gun emission and relativistic Lorentz transport",
        "coherent_source": "UNAVAILABLE", "gun_phase_transfer": "UNAVAILABLE",
        "validation_status": "NOT_RUN",
        "missing": ["Gun-owned mutual-intensity/coherent-mode model",
                    "Validated gun-to-column phase and canonical transport",
                    "Validated component phase operators through the specimen entrance"],
    })


def require_gun_wave_source(state, *, product):
    """Closed production boundary until an actual producer is implemented.

    Do not accept arbitrary private attributes, metadata or caller-supplied
    arrays as proof. This deliberately cannot be enabled with origin='gun'.
    Ray tracing, EDS particle transport and historical viewing remain available.
    """
    readiness = gun_phase_readiness(state)
    if readiness["code"] == "GUN_SOURCE_SETUP_REQUIRED":
        raise UnsupportedWaveSource(f"{product}: {readiness['reason']}")
    if readiness["status"] == "PARTIAL":
        raise UnsupportedWaveSource(
            f"{product}: the versioned gun source and quadratic column propagation are available, "
            "but the new TEM/STEM adapters are not integrated yet. No legacy specimen pupil is used as a substitute.")
    raise UnsupportedWaveSource(
        f"{product}: gun-to-specimen coherent phase is unavailable for {readiness['gun_model']}. "
        "The existing ray-conditioned pupil is historical/reduced-order only. "
        "Ray calculations and existing images remain available; a gun-owned phase model is required for new wave images."
    )


def admit_requested_wave_products(state):
    """Reject unsupported new wave work before heavy preparation/cache reuse."""
    from temsim.physics.wave_imaging import tem_wave_imaging_enabled
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
