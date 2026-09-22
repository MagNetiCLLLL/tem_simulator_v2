"""Bounded geometric twofold beam-shape alignment with executed tip rays.

Circular covariance at one plane is not a wave-aberration measurement. Keep
that distinction explicit; all lens, source, aperture and current settings stay
fixed, and the observed current/support/size gates must pass independently.
"""
from dataclasses import asdict, dataclass
import math
import numpy as np
from temsim.immutable_json import json_digest
from temsim.operating_modes import DirectAlignmentDefinition

KEY = "condenser_twofold_shape"
OBSERVABLES = ("shape_normal", "shape_skew")
DEFINITION = DirectAlignmentDefinition(key=KEY, name="Condenser twofold beam shape", family="beam",
    mode_key="nano_probe", unit="scaled residual", minimum=0., maximum=0., default_value=0.,
    devices=("condenser_stigmator.strength_x_percent", "condenser_stigmator.strength_y_percent"),
    observable="weighted_twofold_position_covariance_at_specimen_entrance",
    constraint="physical_current_support_size_rank_and_independent_steps",
    calibration_status="Local measured authority required; not wave A1 qualification",
    calibration_reference="Executed classical tip-origin beam at specimen entrance",
    targets={}, applies_to_modes=("nano_probe", "micro_probe"), state_parameters=())


@dataclass(frozen=True)
class StigmatorAlignmentOptions:
    shape_tolerance: float = .01
    maximum_strength_percent: float = 100.
    maximum_diameter_nm: float = 10.
    maximum_evaluations: int = 24
    minimum_effective_samples: float = 16.
    minimum_current_pa: float = 2.
    targets: tuple = (0., 0.)

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in (
                self.shape_tolerance, self.maximum_strength_percent, self.maximum_diameter_nm)):
            raise ValueError("Shape tolerance, numerical strength bound and size limit must be positive")
        if self.shape_tolerance >= 1 or self.minimum_effective_samples < 3:
            raise ValueError("Shape tolerance must be below one with at least three effective samples")
        if not math.isfinite(self.minimum_effective_samples) or not math.isfinite(self.minimum_current_pa) or self.minimum_current_pa < 0:
            raise ValueError("Invalid support or current limit")
        if type(self.maximum_evaluations) is not int or not 8 <= self.maximum_evaluations <= 256:
            raise ValueError("Specify 8 to 256 search trials")
        if tuple(self.targets) != (0., 0.):
            raise ValueError("This task targets zero twofold position anisotropy")

    def to_dict(self):
        return asdict(self)

    @property
    def digest(self):
        return json_digest(self.to_dict())


def component(state):
    return next((s for s in state.stigmators if s.key == "condenser_stigmator"), None)


def capability(state):
    if state.electron_gun.source_representation != "classical_particles":
        return False, "Classical tip particles required; coherent development remains paused"
    stig = component(state)
    if stig is None or not stig.enabled:
        return False, "Condenser stigmator is absent or disabled"
    if stig.field_model != "normal_skew":
        return False, "Unsupported condenser stigmator field model; normal_skew is required"
    if state.vacuum_map.enabled:
        return False, "Incident alignment observer does not yet support active vacuum scattering"
    if not state.electron_gun.exit_plane_z_mm < stig.z_mm < state.sample.upper_surface_z_mm:
        return False, "Condenser stigmator must precede the specimen entrance"
    return True, "Two geometric shape components; measured rank, beam size and current gates required. Not a wave A1 measurement."


def initial_values(state):
    stig = component(state)
    return stig.strength_x_percent, stig.strength_y_percent


def allowed_state(request, controls):
    if request.registry_digest != json_digest(asdict(DEFINITION)) or set(controls) != set(DEFINITION.devices):
        raise ValueError("Stigmator request does not match registered physical controls")
    if not isinstance(request.options, StigmatorAlignmentOptions):
        raise TypeError("Expected captured stigmator targets and numerical limits")
    state = request.start_snapshot.restore()
    available, reason = capability(state)
    if not available:
        raise ValueError(reason)
    for key, value in controls.items():
        if not math.isfinite(value) or abs(value) > request.options.maximum_strength_percent:
            raise ValueError("Stigmator control is outside declared numerical bounds")
        setattr(component(state), key.split(".")[1], float(value))
    return state


def shape_metrics(arrays):
    from temsim.physics.phase_space_statistics import weighted_phase_space_statistics
    stats = weighted_phase_space_statistics(arrays)
    covariance = stats["covariance"]
    xx, yy, xy = covariance[0, 0], covariance[2, 2], covariance[0, 2]
    total = xx + yy
    if total <= 0:
        raise ValueError("Degenerate transverse beam cannot qualify a twofold correction")
    return dict(shape_normal=float((xx-yy)/total), shape_skew=float(2*xy/total))


def solve_candidate(request, *, cancelled):
    import sys
    from temsim.beam_alignment import solve_candidate as solve
    return solve(request, cancelled=cancelled, problem=sys.modules[__name__])
