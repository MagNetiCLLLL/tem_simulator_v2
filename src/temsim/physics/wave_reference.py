"""Carried axial reference of a stationary mode, distinct from a wavepacket.

These are elapsed axial flight time and spatial carrier action integral p dz
from the physical tip. They are not a pulse envelope, a temporal coherence
model, or the total time-dependent action integral (p dz - E dt).
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AxialWaveReference:
    flight_time_s: float
    longitudinal_action_j_s: float

    def __post_init__(self):
        for value in (self.flight_time_s, self.longitudinal_action_j_s):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("Axial wave references must be finite and non-negative")

    def advance(self, flight_time_s, longitudinal_action_j_s):
        increment = AxialWaveReference(flight_time_s, longitudinal_action_j_s)
        return AxialWaveReference(self.flight_time_s+increment.flight_time_s,
                                  self.longitudinal_action_j_s+increment.longitudinal_action_j_s)
