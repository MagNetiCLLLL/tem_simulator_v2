"""Round-column propagation with the complete radial complex field.

FFTLog's logarithmic Hankel transform reduces an exactly axisymmetric wave
from a square lattice to one radial axis. No Cs term is switched off, and
this is not an incoherent ray envelope. A non-axisymmetric operation ends
this representation before it is applied by the 2-D column operator.
"""
from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.fft import fhtoffset
from scipy.signal import resample

from temsim.physics.wave_grid import WaveSamplingError


class RadialDomainError(ValueError):
    """The Hankel interval wraps a represented field; expand before solving."""


@dataclass(frozen=True)
class RadialWave:
    radius_m: np.ndarray
    amplitude: np.ndarray  # density amplitude m^-1; curvature is factored
    curvature_m1: float = 0.

    @property
    def dln(self):
        return float(np.log(self.radius_m[1]/self.radius_m[0]))

    @property
    def probability(self):
        return float(2*np.pi*self.dln*np.sum(abs(self.radius_m*self.amplitude)**2))


def _complex_fht(value, dln, offset, *, inverse=False):
    from temsim.physics.radial_fftlog import complex_hankel
    return complex_hankel(value, dln, offset, inverse=inverse)


def _phase_sampling(amplitude, phase, label):
    # The argument is the log-coordinate FLUX amplitude (r*psi or k*psi_k).
    # Only the sampling check excludes at most 1e-12 probability at either
    # endpoint. All complex samples are retained, including those tails.
    # Relative point amplitude alone makes harmless FFT roundoff at enormous
    # reciprocal radii demand an unbounded grid.
    mass = abs(amplitude)**2
    total = mass.sum()
    # Sum each tail from its small end. Subtracting a cumulative sum from
    # one loses the low-probability tail on million-point complex grids.
    occupied = ((np.cumsum(mass) > total*1e-12)
                & (np.cumsum(mass[::-1])[::-1] > total*1e-12))
    increments = abs(np.diff(phase))[occupied[:-1] & occupied[1:]]
    largest = float(increments.max(initial=0.))
    if largest > .5*np.pi:
        raise WaveSamplingError(f"Radial {label} phase needs refinement: {largest:.6g} rad per sample",
                               largest/(.5*np.pi))


def _hankel_grid(wave):
    """Place the reciprocal log interval from the represented field moments.

    Offset is a numerical transform-grid choice, not a source width or a
    fitted phase. A reciprocal interval with an arbitrary zero offset can
    wrap a low-k component into the high-k end even for a Gaussian.
    """
    probability = wave.probability
    position2 = 2*np.pi*wave.dln*np.sum(abs(wave.radius_m**2*wave.amplitude)**2)/probability
    shell = abs(wave.radius_m*wave.amplitude)**2
    # FFTLog is L2 accurate but division by r amplifies roundoff very close
    # to the axis. Do not use that noise to place the numerical reciprocal
    # interval. This mask affects ONLY this arbitrary grid-placement metric;
    # every complex sample is still transformed, propagated and counted.
    metric = ((np.cumsum(shell) > shell.sum()*1e-10)
              & (np.cumsum(shell[::-1])[::-1] > shell.sum()*1e-10))
    momentum2 = 2*np.pi*wave.dln*np.sum(abs(np.gradient(wave.amplitude, wave.dln)[metric])**2)/probability
    if min(position2, momentum2) <= 0 or not np.all(np.isfinite((position2, momentum2))):
        raise ValueError("Radial transform needs finite spatial and spectral extents")
    # The quartic phase generates a long physical high-k tail. Centring on
    # its RMS alone wraps that tail into the low-k end of the periodic FFTLog
    # interval. Reserve a wider reciprocal domain; do not filter the tail.
    initial = np.log(wave.radius_m[0]*wave.radius_m[-1]*math.sqrt(momentum2/position2))
    offset = fhtoffset(wave.dln, 0., initial=float(initial))
    return offset, np.exp(offset)/wave.radius_m[::-1]


def expand_radial_domain(wave, factor=16., *, maximum_samples=4_194_304):
    """Zero-pad the finite represented flux field, retaining all old samples.

    This is an expanded numerical interval, not a claim to recover unknown
    detail outside an unconverged physical domain. Independent domain checks
    remain required. No old coefficient, phase or probability is discarded.
    """
    from scipy.fft import next_fast_len
    extra = math.ceil(math.log(factor)/wave.dln)
    count = next_fast_len(len(wave.radius_m)+2*extra)
    if count > maximum_samples:
        raise ValueError(f"Expanded radial domain needs {count} samples, above {maximum_samples}")
    radius = wave.radius_m[0]*np.exp((np.arange(count)-extra)*wave.dln)
    amplitude = np.zeros(count, complex)
    amplitude[extra:extra+len(wave.amplitude)] = wave.amplitude
    return RadialWave(radius, amplitude, wave.curvature_m1)


def _check_hankel_edges(value):
    # A finite mass at the high-k boundary is evidence of spectral wrapping.
    # Use probability in a boundary strip rather than one point's amplitude.
    strip = max(1, len(value)//1000)
    mass = abs(value)**2
    if max(mass[:strip].sum(), mass[-strip:].sum()) > 1e-12*mass.sum():
        raise RadialDomainError("The reciprocal Hankel interval reaches non-negligible boundary amplitude; expand the input domain")


def refine_radial(wave, scale, *, maximum_samples=4_194_304):
    count = 2**math.ceil(math.log2(max(len(wave.radius_m)+1, len(wave.radius_m)*scale*1.1)))
    if count > maximum_samples:
        raise ValueError(f"Radial wave needs {count} samples, above its {maximum_samples} budget")
    old = wave.probability
    radius = wave.radius_m[0]*np.exp(np.arange(count)*wave.dln*len(wave.radius_m)/count)
    # Fourier interpolation of the log-coordinate flux amplitude. The
    # negative Nyquist component is retained by scipy's periodic resampler.
    value = resample(wave.radius_m*wave.amplitude, count)
    result = RadialWave(radius, value/radius, wave.curvature_m1)
    if not math.isclose(old, result.probability, rel_tol=1e-9, abs_tol=1e-14):
        raise ValueError("Radial grid refinement changed represented probability")
    return result


class RadialExecution:
    """Refine by replaying executed operators, never a clipped interpolant.

    ``load`` evaluates the stored upstream complex expansion on a new grid;
    it may not introduce or fit an independent source. Operators must capture
    immutable parameters because they can be replayed after a later failure.
    Fourier interpolation is useful for smooth mathematical fixtures, but a
    hard aperture produces Gibbs tails if its already clipped array is used
    as the input to a subsequent finer calculation.
    """

    def __init__(self, load, samples, maximum_samples, *, cancelled=lambda: False,
                 report=None):
        self.load = load
        self.samples = samples
        self.maximum_samples = maximum_samples
        self.cancelled = cancelled
        self.report = report
        self.operations = []
        self.ledger = []
        self.refinements = []
        while True:
            if self.cancelled():
                raise InterruptedError("Loading the executed radial wave cancelled")
            try:
                self.wave = load(self.samples)
                break
            except WaveSamplingError as error:
                self._refine_samples(error, "input_replay")

    def _refine_samples(self, error, kind):
        if not math.isfinite(error.required_scale) or error.required_scale <= 0:
            raise ValueError("Radial replay requested an invalid refinement factor") from error
        count = 2**math.ceil(math.log2(max(self.samples+1, self.samples*error.required_scale*1.1)))
        if count > self.maximum_samples:
            raise ValueError(f"Replaying the executed radial chain needs {count} initial samples, "
                             f"above its {self.maximum_samples} budget; {error}") from error
        self._record(self.samples, count, str(error), kind)
        self.samples = count

    def _execute(self, value, operator):
        while True:
            if self.cancelled():
                raise InterruptedError("Round-column wave cancelled")
            try:
                return operator(value)
            except RadialDomainError as error:
                before = len(value.radius_m)
                value = expand_radial_domain(value, maximum_samples=self.maximum_samples)
                self._record(before, len(value.radius_m), str(error), "domain_expansion")

    def _record(self, before, after, reason, kind):
        self.refinements.append({"from": before, "to": after, "reason": reason, "kind": kind})
        if self.report:
            self.report(before, after, kind)

    def apply(self, operator, *, metadata=None):
        operations = (*self.operations, (operator, metadata))
        replay = False
        while True:
            try:
                if replay:
                    value = self.load(self.samples)
                    ledger = []
                    for previous, identity in operations:
                        before = value.probability
                        value = self._execute(value, previous)
                        if identity is not None:
                            ledger.append({**identity, "incoming": before, "outgoing": value.probability})
                else:
                    value = self._execute(self.wave, operator)
                    ledger = list(self.ledger)
                    if metadata is not None:
                        ledger.append({**metadata, "incoming": self.wave.probability, "outgoing": value.probability})
                self.wave = value
                self.ledger = ledger
                self.operations.append((operator, metadata))
                return value
            except WaveSamplingError as error:
                self._refine_samples(error, "upstream_replay")
                replay = True


def _radial_stop(wave, radius_m):
    return replace(wave, amplitude=np.where(wave.radius_m <= radius_m, wave.amplitude, 0j))


def round_map(matrix):
    a, b, c, d = matrix[:2, :2], matrix[:2, 2:], matrix[2:, :2], matrix[2:, 2:]
    block = max((a, b, c, d), key=lambda value: np.linalg.norm(value))
    scale = np.linalg.norm(block)/math.sqrt(2)
    if scale == 0:
        raise ValueError("Empty round canonical map")
    rotation = block/scale
    if not np.allclose(rotation.T@rotation, np.eye(2), atol=1e-9) or np.linalg.det(rotation) < 0:
        raise ValueError("A non-axisymmetric canonical map requires a 2-D wave")
    result = tuple(float(np.trace(rotation.T@value)/2) for value in (a, b, c, d))
    if any(not np.allclose(value, factor*rotation, rtol=1e-9, atol=1e-13*max(1., abs(factor)))
           for value, factor in zip((a, b, c, d), result)):
        raise ValueError("A non-axisymmetric canonical map requires a 2-D wave")
    return result


def propagate_radial(wave, matrix, wavelength_m, *, reference_phase_rad, reference_length_m=1e-3):
    """The same continuous metaplectic lift as the Cartesian column path."""
    from temsim.physics.canonical_action import drift_gaussian_phase
    a, b, c, d = round_map(matrix)
    k0 = 2*np.pi/wavelength_m
    effective_a = a+b*wave.curvature_m1
    before = wave.probability
    if before == 0:
        return wave
    # Prefer the direct Hankel chart when the spectral drift would spread
    # beyond its input log interval. This is a representation choice, not
    # a replacement beam or an omission of diffraction near a crossover.
    offset, frequency = _hankel_grid(wave)
    spectral_to_spatial = np.exp(offset)/(wave.radius_m[0]*wave.radius_m[-1])
    direct = b != 0 and abs(effective_a) < abs(b)*spectral_to_spatial/k0
    if not direct and abs(effective_a) > 1e-8:
        drift = b/effective_a
        transformed = _complex_fht(wave.radius_m*wave.amplitude, wave.dln, offset)
        _check_hankel_edges(transformed)
        phase = -.5*drift/k0*frequency**2
        _phase_sampling(transformed, phase, "Fresnel")
        outgoing = _complex_fht(transformed*np.exp(1j*phase), wave.dln, offset, inverse=True)
        amplitude = outgoing/wave.radius_m/abs(effective_a)
        radius = wave.radius_m*abs(effective_a)
        curvature = (c+d*wave.curvature_m1)/effective_a
        chart = drift_gaussian_phase(np.eye(2)*drift,
            np.eye(2)*(1j/reference_length_m-wave.curvature_m1))
    else:
        if b == 0:
            raise ValueError("Singular radial canonical chart")
        phase = .5*k0*(a/b+wave.curvature_m1)*wave.radius_m**2
        _phase_sampling(wave.radius_m*wave.amplitude, phase, "Fourier input")
        offset, frequency = _hankel_grid(replace(wave, amplitude=wave.amplitude*np.exp(1j*phase)))
        transformed = _complex_fht(wave.radius_m*wave.amplitude*np.exp(1j*phase), wave.dln, offset)
        _check_hankel_edges(transformed)
        amplitude = k0/(1j*b)*transformed/frequency
        radius = abs(b)/k0*frequency
        curvature = d/b
        chart = -math.atan2(b/reference_length_m, a)
    result = RadialWave(radius, amplitude*np.exp(1j*(reference_phase_rad-chart)), curvature)
    if not math.isclose(before, result.probability, rel_tol=1e-9, abs_tol=1e-14):
        raise ValueError("Round canonical propagation changed represented current")
    return result


def radial_cs(wave, wavelength_m, strength_m3):
    if strength_m3 == 0 or wave.probability == 0:
        return wave
    phase = -(2*np.pi/wavelength_m)*strength_m3*wave.radius_m**4/4
    _phase_sampling(wave.radius_m*wave.amplitude, phase, "spherical aberration")
    return replace(wave, amplitude=wave.amplitude*np.exp(1j*phase))


def from_executed_gun(mode, payload, samples=65536):
    """Read the alternative representation of the same executed gun mode."""
    from temsim.physics.quartic_radial_phase import radial_envelope
    if payload["mode_id"] != mode.mode_id:
        raise ValueError("Radial payload belongs to another executed gun mode")
    coefficients = np.asarray(payload["coefficients_real"])+1j*np.asarray(payload["coefficients_imag"])
    width = payload["width_nm"]
    radius = np.geomspace(width*1e-10, width*math.sqrt(4*len(coefficients)+160), samples, endpoint=False)
    # The coefficients are the executed upstream state. Chunk only their
    # evaluation: a denser downstream grid must not allocate N_modes*N_pixels
    # complex values at once or change the expansion.
    amplitude = np.empty(samples, complex)
    chunk = max(1, 1_048_576//len(coefficients))
    for start in range(0, samples, chunk):
        stop = min(start+chunk, samples)
        amplitude[start:stop] = radial_envelope(radius[start:stop], width, coefficients)*1e9
    quartic = float(payload.get("quartic_phase_per_nm4", 0.))
    if not math.isfinite(quartic):
        raise ValueError("Executed gun quartic phase must be finite")
    if quartic:
        phase = quartic*radius**4
        _phase_sampling(radius*amplitude, phase, "executed gun quartic")
        amplitude *= np.exp(1j*phase)
    wave = RadialWave(radius*1e-9, amplitude, payload["curvature_m1"])
    if not math.isclose(wave.probability, mode.plane.probability, rel_tol=1e-8, abs_tol=1e-13):
        raise ValueError("Executed radial and Cartesian gun representations disagree in probability")
    return wave


def _round_column_prefix(state, checkpoint, stop_z_mm, *, maximum_step_mm=.5,
                        initial_samples=65536, maximum_samples=4_194_304,
                        cancelled=lambda: False, progress_callback=None):
    """Execute the maximal round prefix; return before the first 2-D effect."""
    from temsim.physics.column_wave import _prepare_column, _column_transports
    from temsim.physics.canonical_action import CanonicalPath
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
    from temsim.simulation_modes import is_ideal
    plan, radii, stops, owners = _prepare_column(state, checkpoint.plane_z_mm, stop_z_mm, maximum_step_mm)
    end_index = len(plan.z_mm)-1
    for name in ("midpoint_sx_m2", "midpoint_sy_m2", "midpoint_hex_normal_m3", "midpoint_hex_skew_m3"):
        active = np.flatnonzero(getattr(plan, name))
        if len(active):
            end_index = min(end_index, int(active[0]))
    for name in ("kick_x_rad", "kick_y_rad"):
        active = np.flatnonzero(getattr(plan, name))
        if len(active):
            end_index = min(end_index, max(0, int(active[0])-1))
    for row in owners:
        if row["dynamic"]:
            index = int(np.searchsorted(plan.z_mm, row["z_mm"]))
            end_index = min(end_index, max(0, index-1))
    for index, apertures in stops.items():
        if any(getattr(a, "offset_x_mm", 0.) or getattr(a, "offset_y_mm", 0.) for a in apertures):
            end_index = min(end_index, max(0, index-1))
    if end_index == 0:
        return checkpoint.plane_z_mm, (), {"reason": "No round prefix before a non-axisymmetric operation"}
    payloads = {row["mode_id"]: row for row in checkpoint.record["radial_output_modes"]}
    result, records = [], []
    for index, mode in enumerate(checkpoint.beam.modes):
        i = 0  # Input-grid replay can report before the first column step.
        def report(before, after, kind):
            if progress_callback:
                progress_callback(i, end_index, f"Radial {kind}: {before} -> {after}")
        execution = RadialExecution(lambda samples: from_executed_gun(mode, payloads[mode.mode_id], samples),
            initial_samples, maximum_samples, cancelled=cancelled, report=report)
        wave = execution.wave
        energy = state.beam_voltage_kv if is_ideal(state) else mode.energy_kev
        wavelength = float(wavelength_m(mode.energy_kev*1000))
        pending = CanonicalPath(1e-3)
        # Conservative extent of the represented position/frequency grids.
        # If this bound reaches a wall, execute and clip at that exact node.
        # Otherwise grouping the known quadratic factors skips no interaction.
        spectral_extent = float(_hankel_grid(wave)[1][-1]) if wave.probability else 0.
        for i, path in enumerate(_column_transports(plan, energy)):
            if i >= end_index:
                break
            if cancelled():
                raise InterruptedError("Round-column wave cancelled")
            j = i+1
            try:
                pending.append(path.matrix)
            except ValueError as error:
                if "phase path is undersampled" not in str(error):
                    raise
                from types import SimpleNamespace
                from temsim.physics.core import electron
                from temsim.physics.column_wave import _linear_factor
                charge, momentum, _ = electron(SimpleNamespace(beam_voltage_kv=energy))
                g = charge*plan.midpoint_magnetic_t[i]/(2*momentum)
                rotation = np.array(((0., g), (-g, 0.)))
                generator = np.block([[rotation, np.eye(2)], [-np.eye(2)*g*g, rotation]])
                _linear_factor(pending, generator, float(plan.step_m[i]))
            a, b, _, _ = round_map(pending.matrix)
            represented_extent = abs(a+b*wave.curvature_m1)*wave.radius_m[-1]+abs(b)*spectral_extent*wavelength/(2*np.pi)
            must_execute = (j == end_index or plan.cs_kick_m3[j] != 0 or j in stops
                            or represented_extent >= radii[j]*1e-3)
            if not must_execute:
                continue
            wave = execution.apply(lambda w, matrix=pending.matrix.copy(), phase=pending.reference_phase_rad,
                length=pending.reference_length_m: propagate_radial(w, matrix, wavelength,
                    reference_phase_rad=phase, reference_length_m=length))
            pending = CanonicalPath(1e-3)
            if plan.cs_kick_m3[j] != 0:
                wave = execution.apply(lambda w, strength=plan.cs_kick_m3[j]: radial_cs(w, wavelength, strength))
            radius = radii[j]*1e-3
            if math.isfinite(radius):
                wave = execution.apply(lambda w, r=radius: _radial_stop(w, r),
                    metadata={"component": "column_wall", "z_mm": float(plan.z_mm[j])})
            for aperture in stops.get(j, ()):
                wave = execution.apply(lambda w, r=aperture.radius_mm*1e-3: _radial_stop(w, r),
                    metadata={"component": aperture.key, "z_mm": float(plan.z_mm[j])})
            spectral_extent = float(_hankel_grid(wave)[1][-1]) if wave.probability else 0.
            if progress_callback and i % 64 == 0:
                progress_callback(index*end_index+i, len(checkpoint.beam.modes)*end_index,
                                  f"Round column wave at {plan.z_mm[j]:.6g} mm")
        result.append((mode, wave))
        records.append({"mode_id": mode.mode_id, "refinements": execution.refinements, "losses": execution.ledger,
                        "input_fraction": mode.weight_per_reference_electron,
                        "output_fraction": mode.weight_per_reference_electron*wave.probability})
    return float(plan.z_mm[end_index]), tuple(result), {"schema": "executed-round-column-prefix-v1",
        "upstream_digest": checkpoint.digest, "plan_signature": plan.signature,
        "steps": end_index, "modes": records,
        "representation": "Full radial complex field and analytic quadratic phase; all Cs retained",
        "convergence": "Requires radial domain/grid and axial refinement; current balance is not accuracy"}


def round_column_prefix(state, checkpoint, stop_z_mm, *, backend="cpu", **numerics):
    from temsim.physics.radial_fftlog import hankel_backend
    with hankel_backend(backend):
        z, modes, record = _round_column_prefix(state, checkpoint, stop_z_mm, **numerics)
    return z, modes, {**record, "radial_fft_backend": backend,
                      "precision": "complex128; unchanged physical phase and sampling checks"}
