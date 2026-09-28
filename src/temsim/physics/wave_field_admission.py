"""Explicit limits of the paused coherent transport implementation."""

import numpy as np


def require_supported_wave_dipoles(state, start_z_mm, stop_z_mm, plan):
    """Reject finite magnetic drives before allocating or transporting waves.

    Inspect the physical supports as well as the executed plan. Historical
    wave event builders use centre-plane clipping and can omit a coil whose
    finite field still overlaps the requested segment.
    """
    from temsim.physics.instrument_magnetic import column_dipole_fields
    from temsim.physics.electrostatic_column_transport import active_electric_field
    if active_electric_field(plan) is not None:
        raise ValueError(
            "Paused wave transport does not support the captured distributed "
            "electric column field; it cannot be silently omitted"
        )

    low, high = float(start_z_mm)*1e-3, float(stop_z_mm)*1e-3
    active_column = any(
        (coil.bx_t != 0. or coil.by_t != 0.)
        and coil.lower_m < high and coil.upper_m > low
        for coil in column_dipole_fields(state)
    )
    if active_column or np.any(plan.dipole_bx_t) or np.any(plan.dipole_by_t):
        raise ValueError(
            "Finite magnetic dipole wave transport is not implemented; "
            "coherent development is paused and these fields cannot be omitted"
        )
