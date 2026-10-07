"""Explicit operator limits of the coherent development implementation."""

import numpy as np


class UnsupportedWaveElectricField(ValueError):
    """A ray-capable provider without this wave model's scalar expansion."""


def sample_wave_electric(field, z_mm):
    """Second-order transverse expansion of the actual static scalar field.

    Potentials remain in volts and derivatives use metres. The supported
    axisymmetric providers interpolate regularly in r^2 close to the axis;
    differentiation inside their first radial cell reads that same law. The
    analytic Wien contribution is affine. Imported non-polynomial providers
    are not admitted by pretending that a few samples define their operator.
    """
    from temsim.physics.instrument_electric import InstrumentElectricField
    from temsim.physics.closed_gun_field import ClosedGunField
    from temsim.physics.planar_gun_field import PlanarGunField
    from temsim.physics.grounded_tip_field import GroundedTipField
    from temsim.optics.electron_gun.monochromator import CombinedElectricField, AnalyticWienField
    if type(field) is InstrumentElectricField:
        provider, base = field.provider, field.base_field
    else:
        provider = field
        base = field.base_field if type(field) is CombinedElectricField else field
    if type(base) not in (ClosedGunField, PlanarGunField, GroundedTipField):
        raise UnsupportedWaveElectricField("Non-polynomial electric maps need their resolved wave Hamiltonian")
    if provider is not base and type(provider) is not CombinedElectricField:
        raise UnsupportedWaveElectricField("Unknown composed electric provider cannot define a coherent operator")
    wien = getattr(provider, "wien_field", None)
    if wien is not None and type(wien) is not AnalyticWienField:
        raise UnsupportedWaveElectricField("Imported Wien electric maps need their resolved wave Hamiltonian")
    z = np.asarray(z_mm, dtype=float)
    if z.ndim != 1 or not np.all(np.isfinite(z)):
        raise ValueError("Wave electric sampling needs finite axial positions in mm")
    points = np.zeros((len(z), 3))
    points[:, 2] = z*1e-3
    potential_query = getattr(field, "potential_rise_v_at_global_positions", None)
    if potential_query is None:
        potential_query = field.potential_v_at_global_positions
    phi = np.asarray(potential_query(points), float)
    electric0 = np.asarray(field.field_at_global_positions_v_per_m(points), float)
    first_cell = float(np.asarray(base.r)[1])
    delta = min(first_cell*.25, 1e-6)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("Wave electric field has no regular resolved axis cell")
    hessian = np.empty((len(z), 2, 2))
    for axis in range(2):
        plus, minus = points.copy(), points.copy()
        plus[:, axis], minus[:, axis] = delta, -delta
        hessian[:, :, axis] = -(field.field_at_global_positions_v_per_m(plus)[:, :2]
            -field.field_at_global_positions_v_per_m(minus)[:, :2])/(2*delta)
    if not np.all(np.isfinite((phi,))) or not np.all(np.isfinite(electric0)) or not np.all(np.isfinite(hessian)):
        raise ValueError("Captured electric field has non-finite wave coefficients")
    if not np.allclose(hessian, hessian.transpose(0, 2, 1), rtol=1e-10, atol=1e-8):
        raise ValueError("Captured electric field is not a scalar symmetric wave Hessian")
    return phi, -electric0[:, :2], hessian


def sample_spherical_electric(field, z_mm):
    """Local scalar-potential coefficients for the common tilted Cs budget.

    These describe the captured axis expansion already used by the wave
    column. The event does not propagate an additional electric field.
    """
    z = np.atleast_1d(np.asarray(z_mm, float))
    _, transverse_gradient, hessian = sample_wave_electric(field, z)
    points = np.zeros((len(z), 3))
    points[:, 2] = z*1e-3
    gradient = -np.asarray(field.field_at_global_positions_v_per_m(points), float)
    gradient[:, :2] = transverse_gradient
    if not np.isfinite(gradient).all():
        raise ValueError("Captured electric field has non-finite Cs coefficients")
    return gradient, hessian


def require_supported_column_wave_fields(state, start_z_mm, stop_z_mm, plan):
    """Admission only for the column solver that executes E and dipole B.

    The older radial solver still uses the rejecting guard below. Keeping a
    separate capable entry point prevents an admission change from silently
    enabling omitted forces in another caller.
    """
    from temsim.physics.instrument_magnetic import column_dipole_fields
    field = getattr(plan, "electric_field", None)
    if field is not None:
        sample_wave_electric(field, np.asarray((start_z_mm, stop_z_mm)))
    low, high = float(start_z_mm), float(stop_z_mm)
    active = any((coil.bx_t != 0. or coil.by_t != 0.)
        and coil.field_support_mm[0] < high and coil.field_support_mm[1] > low for coil in column_dipole_fields(state))
    if active and not hasattr(plan, "dipole_bx_t"):
        raise ValueError("The shared wave plan is missing finite magnetic dipoles")


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
            "Wave development does not support the captured distributed "
            "electric column field; it cannot be silently omitted"
        )

    low, high = float(start_z_mm), float(stop_z_mm)
    active_column = any(
        (coil.bx_t != 0. or coil.by_t != 0.)
        and coil.field_support_mm[0] < high and coil.field_support_mm[1] > low
        for coil in column_dipole_fields(state)
    )
    if active_column or np.any(plan.dipole_bx_t) or np.any(plan.dipole_by_t):
        raise ValueError(
            "Finite magnetic dipole wave transport is not implemented; "
            "these captured fields cannot be omitted"
        )
