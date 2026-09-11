"""Exclusive bookkeeping when a mode leaves the coherent zero-loss branch.

The first competing event is identifiable from the existing material rates.
This ledger does not supply an outgoing inelastic wave or assign its phase;
later scattering/absorption of those populations still needs transport.
"""
from dataclasses import asdict, replace
import math

import numpy as np


def _inverse_mfp(value):
    if math.isnan(value) or value <= 0:
        raise ValueError("Material mean free paths must be positive or infinite")
    return 0. if math.isinf(value) else 1./value


def _attenuate_zero_loss(mode, inside, thickness_nm, distribution):
    if not math.isfinite(thickness_nm) or thickness_nm < 0:
        raise ValueError("Material path length must be finite and non-negative")
    inside = np.asarray(inside, dtype=bool)
    if inside.shape != mode.plane.amplitude.shape:
        raise ValueError("Material occupancy must match the incoming wave")
    rates = {key: _inverse_mfp(value) for key, value in (
        ("real_plasmon", distribution.plasmon_mean_free_path_nm),
        ("real_ionisation", distribution.ionisation_mean_free_path_nm),
        ("real_other_inelastic", distribution.other_mean_free_path_nm),
        ("effective_absorption", distribution.absorption_mean_free_path_nm))}
    inelastic_rate = sum(rates[key] for key in rates if key != "effective_absorption")
    if not math.isclose(inelastic_rate, _inverse_mfp(distribution.total_inelastic_mean_free_path_nm), rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("Material first-event rates do not sum to the total inelastic rate")
    total_rate = sum(rates.values())
    optical_depth = inside*(total_rate*thickness_nm)
    probability = abs(mode.plane.amplitude)**2
    input_norm = float(probability.sum())
    removed = (mode.weight_per_reference_electron*float(np.sum(probability*(-np.expm1(-optical_depth))))/input_norm
               if input_norm else 0.)
    amplitude = mode.plane.amplitude*np.exp(-optical_depth/2)
    norm = float(np.sum(abs(amplitude)**2))
    if norm == 0:
        removed = mode.weight_per_reference_electron
    output = replace(mode, plane=replace(mode.plane, amplitude=amplitude/math.sqrt(norm) if norm else amplitude),
                     weight_per_reference_electron=max(0., mode.weight_per_reference_electron-removed))
    channels = {c.key: c for c in distribution.channels}
    rows = []
    for key, rate in rates.items():
        channel = channels.get(key)
        weight = removed*rate/total_rate if total_rate else 0.
        rows.append({"kind": key, "weight": weight, "inverse_mfp_per_nm": rate,
            "representative_loss_ev": None if channel is None else channel.energy_loss_ev,
            "characteristic_angle_mrad": None if channel is None else channel.characteristic_angle_mrad,
            "status": "REMOVED_FROM_TRACKED_POPULATION" if key == "effective_absorption" else "OUTGOING_WAVE_NOT_COMPUTED",
            "phase": "UNDEFINED"})
    record = {"parent_mode_id": mode.mode_id, "input_energy_kev": mode.energy_kev,
        "axial_reference": None if mode.axial_reference is None else asdict(mode.axial_reference),
        "input_weight": mode.weight_per_reference_electron, "zero_loss_weight": output.weight_per_reference_electron,
        "removed_weight": removed, "events": rows,
        "partition": "first event leaving the zero-loss branch; competing material hazards",
        "scope": "not final energy-loss populations; subsequent inelastic scattering and absorption not computed"}
    if not math.isclose(output.weight_per_reference_electron+sum(r["weight"] for r in rows),
                        mode.weight_per_reference_electron, rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError("Specimen first-event ledger did not conserve electron probability")
    return output, record
