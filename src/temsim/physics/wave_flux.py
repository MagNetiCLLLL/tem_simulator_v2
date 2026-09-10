"""Explicit boundary between legacy wave shapes and electron probability.

PlaneWave amplitudes remain cell probability amplitudes. A WaveMode factors
that probability into a unit shape and an absolute weight at a named reference
plane. This factorisation never restores electrons lost in a mask.
"""
from dataclasses import asdict, dataclass, replace
import math

import numpy as np

from temsim.physics.multiplane_wave import PlaneWave


TEM_REFERENCE_PLANE = "specimen_entrance_conditional_zero_loss"
FLUX_RTOL = 1e-10
FLUX_ATOL = 1e-14
FLOAT32_FLUX_RTOL = 2e-5  # Accumulated single-precision specimen/FFT operations.


def check_lossless_norm(before: float, after: float, *, context: str, rtol=FLUX_RTOL) -> None:
    if (not math.isfinite(before) or not math.isfinite(after)
            or not math.isclose(before, after, rel_tol=rtol, abs_tol=FLUX_ATOL)):
        raise ValueError(f"{context}: lossless wave norm changed from {before} to {after}; refine sampling or inspect propagation")


@dataclass(frozen=True)
class WaveMode:
    plane: PlaneWave
    weight_per_reference_electron: float
    reference_plane: str
    mode_id: str
    energy_kev: float

    def __post_init__(self):
        if (not math.isfinite(self.weight_per_reference_electron)
                or self.weight_per_reference_electron < 0):
            raise ValueError("Wave-mode weight must be finite and non-negative")
        if not self.reference_plane or not self.mode_id:
            raise ValueError("Wave modes need an explicit reference plane and identity")
        if not math.isfinite(self.energy_kev) or self.energy_kev <= 0:
            raise ValueError("Wave-mode energy must be finite and positive")
        norm = self.plane.probability
        if self.weight_per_reference_electron == 0 and norm == 0:
            return
        check_lossless_norm(1., norm, context="Unit wave-mode representation")

    @classmethod
    def from_legacy(cls, amplitude, x_angstrom, y_angstrom, *, reference_discrete_norm,
                    prior_weight=1., reference_plane=TEM_REFERENCE_PLANE,
                    mode_id="coherent:0", energy_kev):
        """Convert a legacy shape using its PRE-LOSS incident norm.

        ``prior_weight`` is a mode/configuration probability, not a second
        column-transmission correction. The input may legitimately be zero.
        """
        if not math.isfinite(reference_discrete_norm) or reference_discrete_norm <= 0:
            raise ValueError("The pre-loss reference norm must be finite and positive")
        if not math.isfinite(prior_weight) or prior_weight < 0:
            raise ValueError("Mode prior must be finite and non-negative")
        x, y = np.asarray(x_angstrom, float) * 1e-10, np.asarray(y_angstrom, float) * 1e-10
        for axis in (x, y):
            if (axis.ndim != 1 or len(axis) < 2 or not np.all(np.isfinite(axis))
                    or axis[1] <= axis[0]
                    or not np.allclose(np.diff(axis), axis[1]-axis[0], rtol=1e-10, atol=1e-25)):
                raise ValueError("Wave axes must be finite, increasing and uniformly sampled")
        a = np.asarray(amplitude, complex)
        if a.shape != (len(y), len(x)) or not np.all(np.isfinite(a)):
            raise ValueError("Wave shape/values do not match the finite sampling grid")
        norm = float(np.sum(np.abs(a)**2))
        shape = a / math.sqrt(norm) if norm > 0 else np.zeros_like(a)
        shape.setflags(write=False)
        plane = PlaneWave(shape, np.diag((x[1]-x[0], y[1]-y[0])),
                          np.array((x[len(x)//2], y[len(y)//2])))
        return cls(plane, prior_weight * norm / reference_discrete_norm,
                   reference_plane, mode_id, energy_kev)

    def weighted_density_amplitude(self):
        area = abs(float(np.linalg.det(self.plane.basis_m)))
        if not math.isfinite(area) or area <= 0:
            raise ValueError("Wave cell area must be finite and positive")
        return self.plane.amplitude * math.sqrt(self.weight_per_reference_electron / area)


@dataclass(frozen=True)
class BeamState:
    """An incoherent mixture of declared coherent modes at a named plane."""
    modes: tuple[WaveMode, ...]
    reference_plane: str = TEM_REFERENCE_PLANE

    def __post_init__(self):
        if not self.modes or any(m.reference_plane != self.reference_plane for m in self.modes):
            raise ValueError("Beam modes must share one declared electron reference")
        if len({m.mode_id for m in self.modes}) != len(self.modes):
            raise ValueError("Beam mode IDs must be unique")

    @property
    def total_weight(self):
        return sum(m.weight_per_reference_electron for m in self.modes)

    def cell_probabilities(self):
        """Add intensities on an identical affine lattice, never amplitudes."""
        first = self.modes[0].plane
        result = np.zeros_like(first.amplitude, dtype=float)
        for mode in self.modes:
            p = mode.plane
            if (p.amplitude.shape != first.amplitude.shape or not np.array_equal(p.basis_m, first.basis_m)
                    or not np.array_equal(p.origin_m, first.origin_m)):
                raise ValueError("Mixed-mode intensities need a common physical grid; resample explicitly")
            result += mode.weight_per_reference_electron * np.abs(p.amplitude)**2
        return result

    def propagate(self, transfer_for_energy):
        """Reuse the norm-checked LCT and each mode's wavelength and map."""
        from types import SimpleNamespace
        from temsim.physics.core import electron
        from temsim.physics.multiplane_wave import propagate_plane_wave
        modes = []
        for mode in self.modes:
            matrix, offset = transfer_for_energy(mode.energy_kev)
            wavelength = electron(SimpleNamespace(beam_voltage_kv=mode.energy_kev))[2] * 1e-9
            modes.append(replace(mode, plane=propagate_plane_wave(mode.plane, matrix, offset, wavelength)))
        return BeamState(tuple(modes), self.reference_plane)


@dataclass(frozen=True)
class FluxEntry:
    reference_plane: str
    node_id: str
    physical_element_id: str | None
    input_weight: float
    output_weight: float
    loss_category: str
    branch_id: str
    condition: str = "conditional_zero_loss"
    exclusive: bool = True
    relative_tolerance: float = FLUX_RTOL
    parameters: dict | None = None

    def __post_init__(self):
        if (not all(math.isfinite(v) and v >= 0 for v in (self.input_weight, self.output_weight))
                or self.output_weight > self.input_weight * (1 + self.relative_tolerance) + FLUX_ATOL):
            raise ValueError(f"Invalid electron flux at {self.node_id}: {self.input_weight} -> {self.output_weight}")

    def to_dict(self):
        return {**asdict(self), "lost_weight": max(0., self.input_weight - self.output_weight),
                "positive_numerical_residual": max(0., self.output_weight - self.input_weight)}


class FluxLedger:
    """One electron branch; physical IDs may execute only once in it."""
    def __init__(self, *, branch_id, reference_plane=TEM_REFERENCE_PLANE, relative_tolerance=FLUX_RTOL):
        self.branch_id = branch_id
        self.reference_plane = reference_plane
        self.entries = []
        self._physical_ids = set()
        self.relative_tolerance = relative_tolerance

    def record(self, node_id, before, after, category, *, physical_element_id=None, parameters=None):
        if physical_element_id is not None and physical_element_id in self._physical_ids:
            raise ValueError(f"Physical element executed twice in {self.branch_id}: {physical_element_id}")
        if self.entries and not math.isclose(self.entries[-1].output_weight, before,
                                             rel_tol=self.relative_tolerance, abs_tol=FLUX_ATOL):
            raise ValueError(f"Discontinuous electron flux at {node_id}; a prior loss or weight was reset")
        entry = FluxEntry(self.reference_plane, node_id, physical_element_id,
                          float(before), float(after), category, self.branch_id,
                          relative_tolerance=self.relative_tolerance, parameters=parameters)
        self.entries.append(entry)
        if physical_element_id is not None:
            self._physical_ids.add(physical_element_id)
        return entry

    def weighted_rows(self, prior):
        return tuple(replace(row, input_weight=row.input_weight * prior,
                             output_weight=row.output_weight * prior).to_dict() for row in self.entries)


def expected_electron_counts(probability, *, reference_current_a, exposure_s):
    """Counts using current AT THE SAME reference plane as the probability.

    TEM conditional-zero-loss output needs that branch's current. Source
    current or a column transmission is deliberately not inferred here.
    """
    from temsim.physics.core import E
    p = np.asarray(probability, float)
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("Pixel probability must be finite and non-negative")
    if not all(math.isfinite(v) and v >= 0 for v in (reference_current_a, exposure_s)):
        raise ValueError("Reference current and exposure must be finite and non-negative")
    return p * (reference_current_a * exposure_s / abs(E))
