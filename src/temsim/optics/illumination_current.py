"""Explicit source-flux setting and physical pupil/current acceptance.

The existing total-flux control multiplies the emitted current, not the
survivor weights. Selection is performed once on a detached fitted state;
independent validation keeps that setting fixed. No emission distribution,
trajectory, physical loss or downstream source is changed here.
"""
from dataclasses import dataclass
import math
from pathlib import Path
from temsim import input_io
import tomllib

from temsim.physics.beam_current import (
    column_current_limit_percent, effective_source_current_pa,
)


@dataclass(frozen=True)
class CurrentApertureLimits:
    minimum_current_pa: float
    maximum_current_pa: float
    preferred_current_pa: float
    minimum_c2_diameter_um: float
    maximum_c2_diameter_um: float


def load_current_aperture_limits(path=None):
    from temsim.paths import OPERATING_MODE_CONFIG_ROOT
    path = Path(path) if path is not None else OPERATING_MODE_CONFIG_ROOT / 'illumination_targets.toml'
    with input_io.open_input(path) as stream:
        document = tomllib.load(stream)
    if document.get('schema') != 'assembly-illumination-targets-v1':
        raise ValueError('Unsupported illumination target schema')
    limits = CurrentApertureLimits(**document['current_and_aperture'])
    if (not all(math.isfinite(v) and v > 0 for v in vars(limits).values())
            or not limits.minimum_current_pa <= limits.preferred_current_pa <= limits.maximum_current_pa
            or limits.minimum_c2_diameter_um > limits.maximum_c2_diameter_um):
        raise ValueError('Current and aperture limits must be finite, positive and ordered')
    return limits


def aperture_gate(state, *, limits=None):
    limits = limits or load_current_aperture_limits()
    aperture = state.condenser_aperture_2
    diameter = float(aperture.diameter_mm) * 1e3
    enabled = bool(aperture.enabled and getattr(aperture, 'installed', True))
    return dict(diameter_um=diameter, enabled=enabled,
        minimum_um=limits.minimum_c2_diameter_um, maximum_um=limits.maximum_c2_diameter_um,
        passed=enabled and math.isfinite(diameter)
        and limits.minimum_c2_diameter_um <= diameter <= limits.maximum_c2_diameter_um)


def _transmission(measurement):
    fraction = float(measurement.statistics.surviving_fraction)
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Measured source-to-specimen transmission must lie between zero and one')
    return fraction


def current_gate(state, measurement, *, limits=None):
    limits = limits or load_current_aperture_limits()
    fraction = _transmission(measurement)
    source_pa = effective_source_current_pa(state)
    current = source_pa * fraction
    return dict(source_to_specimen_fraction=fraction,
        column_current_limit_percent=column_current_limit_percent(state),
        effective_source_current_pa=source_pa, specimen_current_pa=current,
        minimum_pa=limits.minimum_current_pa, maximum_pa=limits.maximum_current_pa,
        passed=math.isfinite(current) and limits.minimum_current_pa <= current <= limits.maximum_current_pa)


def set_calibration_flux(state, measurement, *, limits=None):
    """Set an explicitly authorised source ceiling, never amplify survivors.

    Call only on a detached calibration state. A source that cannot deliver
    the minimum current at 100% remains infeasible at this measured setting.
    The returned audit distinguishes physical transmission from flux control.
    """
    limits = limits or load_current_aperture_limits()
    if state.electron_gun.source_representation != 'classical_particles':
        raise ValueError('Default flux calibration requires classical tip emission')
    if not aperture_gate(state, limits=limits)['passed']:
        raise ValueError('The physical C2 aperture must be enabled with diameter 20-250 um')
    # Also validate the source policy and pre-existing control before mutation.
    effective_source_current_pa(state)
    fraction = _transmission(measurement)
    available = float(state.electron_gun.emitted_current_a) * 1e12 * fraction
    if not math.isfinite(available) or available < limits.minimum_current_pa:
        raise ValueError('Insufficient physical specimen current even at 100% source flux')
    chosen = min(limits.preferred_current_pa, available)
    old = column_current_limit_percent(state)
    state.column_current_limit_percent = min(100., 100. * chosen / available)
    audit = current_gate(state, measurement, limits=limits)
    audit.update(previous_limit_percent=old, available_at_full_flux_pa=available,
                 requested_setpoint_pa=chosen, policy='EXPLICIT_UPSTREAM_TOTAL_FLUX')
    return audit
