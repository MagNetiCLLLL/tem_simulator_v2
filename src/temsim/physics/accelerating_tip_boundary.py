"""Executed planar Robin tip through a weak-transverse accelerating near field.

Each sampled angular channel solves the two-way scalar relativistic orbital
equation, including evanescent intervals. The captured axial potential is never
replaced by an entered exit energy. A bounded small transverse-potential
approximation is explicit and checked; magnetic fields or interceptions in
this local domain are rejected. The subsequent gun retains its installed
electric/magnetic focusing and masks. This is an opt-in development method,
not a tunnelling model, a positive near-field ray distribution or a claim of
complete microscope qualification.
"""
from dataclasses import asdict, dataclass, replace
import math

import numpy as np
from scipy.constants import c, e, hbar, m_e

from temsim.immutable_json import freeze_json, json_digest
from temsim.physics.planar_tip_boundary import _frozen, _guard_hardware, _wave_number_squared


@dataclass(frozen=True)
class AcceleratingTipNumerics:
    # For the startup planar field, 4 um gives >=2.624 eV even at its .01 eV
    # energy floor. The source's analytic high-angle tail is then ~2.1e-11
    # at the 1% generator-error boundary. The 10 um numerical square retains
    # the executed spectral-transform tails; it is not the 5 nm source width.
    # Its worst-case transverse phase estimate is checked against 1e-3 rad
    # (<0.06 degrees) below, independently for every energy mode.
    end_z_nm: float = 4_000.
    transverse_radius_um: float = 10.
    maximum_step_nm: float = 50.
    maximum_fractional_energy_change: float = .02
    maximum_steps: int = 8192
    maximum_transverse_phase_error_rad: float = 1e-3
    maximum_paraxial_generator_error: float = .01
    angular_tail_tolerance: float = 1e-8
    maximum_working_bytes: int = 1024**3

    def validate(self):
        for name in ("end_z_nm", "transverse_radius_um", "maximum_step_nm",
                     "maximum_transverse_phase_error_rad"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("maximum_fractional_energy_change", "maximum_paraxial_generator_error",
                     "angular_tail_tolerance"):
            if not math.isfinite(getattr(self, name)) or not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must lie strictly between zero and one")
        if type(self.maximum_steps) is not int or not 1 <= self.maximum_steps <= 1_000_000:
            raise ValueError("maximum_steps must be a positive bounded integer")
        if type(self.maximum_working_bytes) is not int or self.maximum_working_bytes <= 0:
            raise ValueError("maximum_working_bytes must be a positive integer")
        return self


@dataclass(frozen=True)
class ExecutedAcceleratingTip:
    """Computed continuation state, with normal derivative and provenance.

    Construction is internal to ``execute_accelerating_tip``. This development
    result is admitted only as an internal executed prefix of the captured
    gun. Its complex traces are retained with the full checkpoint; a digest
    alone is insufficient and no arbitrary downstream input is accepted.
    """
    modes: tuple
    spectral_amplitudes: tuple
    spectral_derivatives_per_m: tuple
    logarithmic_transmissions: tuple
    original_energies_ev: tuple
    plane_z_mm: float
    reference_current_a: float
    record: object

    def __post_init__(self):
        for name in ("spectral_amplitudes", "spectral_derivatives_per_m", "logarithmic_transmissions"):
            object.__setattr__(self, name, tuple(_frozen(a) for a in getattr(self, name)))
        object.__setattr__(self, "record", freeze_json(self.record))

    @property
    def digest(self):
        from hashlib import sha256
        digest = sha256(json_digest(self.record).encode())
        for values in (*self.spectral_amplitudes, *self.spectral_derivatives_per_m,
                       *self.logarithmic_transmissions):
            digest.update(values.tobytes())
        for mode in self.modes:
            for name in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
                values = getattr(mode.plane, name)
                digest.update(name.encode())
                if values is not None:
                    digest.update(str(values.shape).encode())
                    digest.update(values.tobytes())
            digest.update(json_digest({"weight": mode.weight_per_reference_electron,
                "energy_kev": mode.energy_kev, "mode_id": mode.mode_id,
                "reference_plane": mode.reference_plane,
                "axial_reference": asdict(mode.axial_reference)}).encode())
        return digest.hexdigest()


def scalar_robin_transport(axial_k2_m2, widths_m, exit_k2_m2, transverse_k2_m2,
                           normal_k_per_m, *, cancelled=lambda: False):
    """All angular channels, two directions, no evanescent truncation.

    Returns LOG total-field transmission and source reflection relative to the
    same incident Robin reservoir amplitude. Cumulative evanescent amplitude
    stays logarithmic, including channels too small for a floating amplitude.
    Individual unresolved slabs are rejected. Slabs use only
    decaying exponentials, never exponentially growing transfer matrices.
    """
    axial = np.asarray(axial_k2_m2, float)
    widths = np.asarray(widths_m, float)
    transverse = np.asarray(transverse_k2_m2, float)
    kappa = normal_k_per_m
    if (axial.ndim != 1 or widths.shape != axial.shape or not len(axial)
            or not np.isfinite(axial).all() or np.any(axial <= 0)
            or not np.isfinite(widths).all() or np.any(widths <= 0)
            or transverse.ndim != 2 or not np.isfinite(transverse).all() or np.any(transverse < 0)
            or not math.isfinite(exit_k2_m2) or exit_k2_m2 <= 0
            or not math.isfinite(kappa) or kappa <= 0):
        raise ValueError("Scalar near-tip layers require positive finite energies, widths and admittance")
    exit_k = np.sqrt((exit_k2_m2-transverse).astype(complex))
    reflection = (kappa-exit_k)/(kappa+exit_k)
    log_transmission = np.log(1+reflection)
    worst, lowest_log, mixed_layers = 0., 0., 0
    for index in range(len(widths)-1, -1, -1):
        if cancelled():
            raise InterruptedError("Non-paraxial angular transport cancelled")
        values = axial[index]-transverse
        k = np.sqrt(values.astype(complex))
        phase = k*widths[index]
        attenuation = abs(phase.imag)
        pos, neg = np.exp(1j*phase-attenuation), np.exp(-1j*phase-attenuation)
        cosine = (pos+neg)/2
        sine_over_k = np.empty_like(k)
        small = abs(phase) < 1e-4
        sine_over_k[small] = widths[index]*np.exp(-attenuation[small])*(1-phase[small]**2/6+phase[small]**4/120)
        np.divide(pos-neg, 2j*k, out=sine_over_k, where=~small)
        denominator = cosine-.5j*(kappa+values/kappa)*sine_over_k
        r = .5j*(values/kappa-kappa)*sine_over_k/denominator
        log_t = -attenuation-np.log(denominator)
        lowest_log = min(lowest_log, float(log_t.real.min()))
        if lowest_log < math.log(np.finfo(float).tiny):
            raise ValueError("Near-tip slab needs finer axial steps to retain evanescent transmission")
        t = np.exp(log_t)
        error = max(float(np.max(abs(abs(r)**2+abs(t)**2-1))),
                    float(np.max(abs(2*np.real(r.conj()*t)))))
        worst = max(worst, error)
        if worst > 1e-9:
            raise ValueError("Near-tip angular slab failed current conservation")
        denominator = 1-r*reflection
        if not np.isfinite(denominator).all() or np.any(denominator == 0):
            raise ValueError("Near-tip reflected load is singular or non-finite")
        log_gain = log_t-np.log(denominator)
        gain = np.exp(log_gain)
        log_transmission += log_gain
        reflection = r+t*reflection*gain
        if not np.isfinite(reflection).all() or not np.isfinite(log_transmission).all():
            raise ValueError("Near-tip reflected load became non-finite")
        mixed_layers += int(np.any(values < 0))
    return log_transmission, reflection, exit_k, {
        "maximum_slab_current_residual": worst, "minimum_slab_log_transmission": lowest_log,
        "layers_with_evanescent_channels": mixed_layers,
        "minimum_complete_log_transmission": float(log_transmission.real.min()),
        "logarithmic_channels_below_float_range": int(np.count_nonzero(log_transmission.real < math.log(np.finfo(float).tiny))),
        "method": "two-way reflection embedding of exact scalar constant-potential slabs",
    }


def _near_field_domain(state, electric, magnetic, end_m, radius_m):
    from temsim.physics.closed_gun_field import ClosedGunField
    if type(electric.base_field) is not ClosedGunField:
        raise ValueError("This accelerating boundary needs the captured closed planar gun field; curved-tip fields need their own boundary solver")
    cathode = electric.base_field.request["boundary_conditions"]["cathode"]
    if cathode["type"] != "planar_equipotential":
        raise ValueError("Near-tip stationary energy components require an equipotential emitting boundary")
    wien = getattr(electric.provider, "wien_field", None)
    if wien is not None:
        from temsim.optics.electron_gun.monochromator import AnalyticWienField
        if type(wien) is not AnalyticWienField:
            raise ValueError("An imported electric provider needs a verified near-tip domain")
        lower, upper = wien.element.field_support_mm
        if wien.element.enabled and lower <= end_m*1e3 and upper >= 0:
            raise ValueError("A transverse electric element overlaps the near-tip domain; a coupled solve is required")
    hardware = _guard_hardware(state, end_m*1e3, math.sqrt(2)*radius_m*1e3)
    for source in magnetic._sources:
        if not source.known_zero and source.bounds_m[0, 2] <= end_m and source.bounds_m[1, 2] >= 0:
            raise ValueError("An installed magnetic field overlaps the near-tip domain; its vector-potential operator is required")
    return hardware


def _spectral_to_plane(coefficients, kxy, input_basis, origin, wavelength, quadratic_phase,
                       grid_numerics, cancelled):
    """Inverse Fourier transform with its large quadratic phase kept exact.

    The dual coordinates u=k/k0 * L are a numerical representation of an
    executed field, not a launch plane. The -pi/2 removes the 2-D inverse-
    Fourier metaplectic prefactor; the axial action is carried separately.
    """
    from temsim.physics.multiplane_wave import PlaneWave, propagate_plane_wave
    k0 = 2*np.pi/wavelength
    length = 1e-3
    ny, nx = coefficients.shape
    dual_basis = length*wavelength*np.linalg.inv(input_basis).T@np.diag((1/nx, 1/ny))
    envelope = coefficients*np.exp(-1j*quadratic_phase*np.sum(kxy**2, axis=0))
    # The original spatial offset is the Fourier phase exp(-i k.x0).
    dual = PlaneWave(envelope, dual_basis, np.zeros(2),
        np.eye(2)*(2*quadratic_phase*k0/length**2), -np.asarray(origin)/length)
    matrix = np.block([[np.zeros((2, 2)), -length*np.eye(2)],
                       [np.eye(2)/length, np.zeros((2, 2))]])
    result = propagate_plane_wave(dual, matrix, np.zeros(4), wavelength,
        reference_phase_rad=math.pi/2, reference_length_m=length,
        grid_numerics=grid_numerics, cancelled=cancelled)
    return replace(result, amplitude=-1j*result.amplitude)


def execute_accelerating_tip(state, emission, electric, *, numerics=AcceleratingTipNumerics(),
                             grid_numerics=None, cancelled=lambda: False, progress_callback=None):
    """Run the captured tip boundary, retaining source weights and flux losses.

    ``emission`` must have been generated from this captured state's active
    tip; equality is checked before execution. This explicit development
    entry requires the same captured electric provider. No arbitrary
    downstream wave is accepted and no full-gun result is published.
    """
    from temsim.calculation_manifest import solver_source_identity
    from temsim.instrument_snapshot import encode_instrument
    from temsim.optics.electron_gun.tip_coherence import generate_tip_boundary_emission, wavelength_m
    from temsim.physics.instrument_magnetic import capture_instrument_magnetic_field
    from temsim.physics.instrument_electric import capture_instrument_electric_field
    from temsim.physics.tip_gun_wave import GunWaveNumerics, _momentum_velocity, _refine_energy_grid
    from temsim.physics.wave_grid import WaveGridNumerics
    from temsim.physics.wave_reference import AxialWaveReference
    numerics.validate()
    if cancelled():
        raise InterruptedError("Accelerating tip boundary cancelled")
    expected = generate_tip_boundary_emission(state.electron_gun, emission.numerics)
    if expected.digest != emission.digest:
        raise ValueError("Near-tip boundary does not belong to the captured physical source")
    canonical_electric = capture_instrument_electric_field(state)
    identity_fields = ("physical_identity", "numerical_identity", "request_identity")
    if any(getattr(electric, name, None) is None
           or getattr(electric, name, None) != getattr(canonical_electric, name)
           for name in identity_fields):
        raise ValueError("Near-tip electric field does not belong to the captured instrument inputs")
    graph_id, implementation = json_digest(encode_instrument(state)), solver_source_identity()
    magnetic = capture_instrument_magnetic_field(state)
    end_m, radius_m = numerics.end_z_nm*1e-9, numerics.transverse_radius_um*1e-6
    hardware = _near_field_domain(state, electric, magnetic, end_m, radius_m)
    grid_numerics = grid_numerics or WaveGridNumerics(maximum_working_bytes=numerics.maximum_working_bytes)
    start_potential = float(electric.potential_rise_v_at_global_positions(np.zeros((1, 3)))[0])
    tip_radii = np.r_[0., electric.base_field.r[(electric.base_field.r > 0)
                    & (electric.base_field.r < math.sqrt(2)*radius_m)], math.sqrt(2)*radius_m]
    tip_positions = np.zeros((len(tip_radii), 3)); tip_positions[:, 0] = tip_radii
    tip_phi = electric.potential_rise_v_at_global_positions(tip_positions)
    if not np.isfinite(tip_phi).all() or np.any(tip_phi != start_potential):
        raise ValueError("The emitting boundary is not equipotential; its position-energy correlation requires a different source boundary")
    steps = math.ceil(numerics.end_z_nm/numerics.maximum_step_nm)
    if steps > numerics.maximum_steps:
        raise ValueError("Near-tip axial grid exceeds maximum_steps")
    base_grid = np.linspace(0., end_m*1e3, steps+1)
    nodes = np.asarray(electric.base_field.z)*1e3
    base_grid = np.unique(np.r_[base_grid, nodes[(nodes > 0) & (nodes < base_grid[-1])]])
    minimum = min(row["energy_ev"] for row in emission.record["energy_modes"])
    edges = _refine_energy_grid(base_grid, electric, minimum, GunWaveNumerics(
        max_steps=numerics.maximum_steps, maximum_fractional_energy_change=numerics.maximum_fractional_energy_change), cancelled)
    mids, dz = (edges[:-1]+edges[1:])*0.5e-3, np.diff(edges)*1e-3
    probes = np.zeros((len(mids)+1, 3)); probes[:, 2] = np.r_[mids, end_m]
    phi = electric.potential_rise_v_at_global_positions(probes)-start_potential
    # ClosedGunField interpolates in r^2; check every radial grid knot and
    # the box corner. This bounds its transverse variation at each midpoint.
    radii = np.r_[0., electric.base_field.r[(electric.base_field.r > 0)
                 & (electric.base_field.r < math.sqrt(2)*radius_m)], math.sqrt(2)*radius_m]
    radial_probes = np.repeat(probes[:, None, :], len(radii), axis=1)
    radial_probes[..., 0] = radii
    variation = np.max(abs(electric.potential_rise_v_at_global_positions(radial_probes)
                          -start_potential-phi[:, None]), axis=1)
    modes, amplitudes, derivatives, logs, energies, rows = [], [], [], [], [], []
    for index, mode in enumerate(emission.modes()):
        if cancelled():
            raise InterruptedError("Accelerating tip boundary cancelled")
        original_energy = mode.energy_kev*1e3
        kinetic = original_energy+phi
        _, velocity = _momentum_velocity(kinetic[:-1])
        transverse_error = float(np.sum(e/(hbar*velocity)*variation[:-1]*dz))
        if transverse_error > numerics.maximum_transverse_phase_error_rad:
            raise ValueError(f"Near-tip transverse electric phase estimate {transverse_error:.6g} exceeds its explicit numerical budget; a coupled transverse solve is required")
        full = mode.plane.full_amplitude(float(wavelength_m(original_energy)))
        ny, nx = full.shape
        grid_numerics.check(full.shape)
        fy, fx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(ny)), np.fft.fftshift(np.fft.fftfreq(nx)), indexing="ij")
        kxy = 2*np.pi*np.einsum("ij,jyx->iyx", np.linalg.inv(mode.plane.basis_m).T, np.stack((fx, fy)))
        transverse = np.sum(kxy*kxy, axis=0)
        source = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(full), norm="ortho"))
        normal_k = math.sqrt(float(_wave_number_squared(original_energy)))
        axial_k2 = _wave_number_squared(kinetic)
        log_transmitted, reflected, exit_k, record = scalar_robin_transport(
            axial_k2[:-1], dz, float(axial_k2[-1]), transverse, normal_k, cancelled=cancelled)
        incident = source/math.sqrt(normal_k)
        # Floating fields cannot represent amplitudes below tiny. Keep the
        # exact computed logarithmic transfer alongside the input spectrum;
        # no such channel is erased from the retained boundary state.
        transmitted = np.exp(log_transmitted)
        field = transmitted*incident
        derivative = 1j*exit_k*field
        flux = exit_k.real*abs(field)**2
        if not np.isfinite(field).all() or not np.isfinite(derivative).all() or not np.isfinite(flux).all():
            raise ValueError("Near-tip output field or normal current became non-finite")
        returned = float(normal_k*np.sum(abs(reflected*incident)**2))
        output = float(flux.sum())
        input_flux = float(np.sum(abs(source)**2))
        balance = abs(input_flux-returned-output)
        if balance > 1e-8*input_flux or output <= 0:
            raise ValueError("Near-tip source/reflection/output current did not close")
        q2 = transverse/axial_k2[-1]
        errors = np.where(q2 < 1, .5*(1-np.sqrt(np.maximum(0., 1-q2))), np.inf)
        incompatible = float(np.sum(flux[errors > numerics.maximum_paraxial_generator_error]))
        evanescent_norm = float(np.sum(abs(field[exit_k.real == 0])**2))
        total_trace_norm = float(np.sum(abs(field)**2))
        if (incompatible > numerics.angular_tail_tolerance*output
                or evanescent_norm > numerics.angular_tail_tolerance*total_trace_norm):
            raise ValueError("Executed near-tip field is not yet admissible for paraxial continuation; increase its executed endpoint")
        momentum, _ = _momentum_velocity(kinetic[:-1])
        action = float(np.sum(momentum*dz))
        elapsed = float(np.sum(dz/velocity))
        quadratic_phase = -.5*float(np.sum(dz/np.sqrt(axial_k2[:-1])))
        # Unit flux shape is a factorisation. The weight below retains all
        # reflected current; it never restores output to the incident weight.
        spectral_flux = np.sqrt(exit_k.real)*field/math.sqrt(output)*np.exp(-1j*action/hbar)
        plane = _spectral_to_plane(spectral_flux, kxy, mode.plane.basis_m,
            mode.plane.origin_m, float(wavelength_m(float(kinetic[-1]))), quadratic_phase,
            grid_numerics, cancelled)
        positions = plane.coordinates_m()
        outside = float(np.sum(abs(plane.amplitude[(abs(positions[0]) > radius_m)
                         | (abs(positions[1]) > radius_m)])**2))
        if outside > numerics.angular_tail_tolerance:
            raise ValueError(f"Executed near-tip field exceeds the checked transverse domain (outside fraction {outside:.6g}, radius {radius_m*1e6:.6g} um, transverse phase estimate {transverse_error:.6g} rad); increase its numerical radius")
        modes.append(replace(mode, plane=plane, energy_kev=float(kinetic[-1])*1e-3,
            weight_per_reference_electron=mode.weight_per_reference_electron*output,
            axial_reference=AxialWaveReference(elapsed, action)))
        amplitudes.append(field); derivatives.append(derivative); logs.append(log_transmitted); energies.append(original_energy)
        rows.append({"mode_id": mode.mode_id, "tip_energy_ev": original_energy,
            "endpoint_axial_energy_ev": float(kinetic[-1]), "injected_fraction": input_flux,
            "reflected_fraction": returned, "transmitted_fraction": output,
            "balance_residual": balance, "paraxial_incompatible_fraction": incompatible,
            "retained_evanescent_trace_fraction": evanescent_norm/max(total_trace_norm, 1e-300),
            "transverse_phase_error_estimate_rad": transverse_error,
            "outside_checked_domain_fraction": outside, **record})
        if progress_callback is not None:
            progress_callback(index+1, emission.record["mode_count"], "Executed non-paraxial accelerating tip boundary")
    if graph_id != json_digest(encode_instrument(state)) or implementation != solver_source_identity():
        raise ValueError("Captured source, fields or solver changed during near-tip execution")
    return ExecutedAcceleratingTip(tuple(modes), tuple(amplitudes), tuple(derivatives), tuple(logs), tuple(energies),
        end_m*1e3, emission.reference_current_a, {
            "schema": "executed-accelerating-planar-tip-v1", "tip_emission_id": emission.digest,
            "tip_emission": emission.record, "instrument_input_identity": graph_id,
            "electric_physical_identity": electric.physical_identity,
            "electric_numerical_identity": electric.numerical_identity,
            "magnetic_identity": magnetic.numerical_identity, "hardware_signature": hardware,
            "implementation": implementation, "numerics": asdict(numerics),
            "wave_grid_numerics": asdict(grid_numerics),
            "source_numerics": asdict(emission.numerics), "z_edges_mm": edges.tolist(),
            "modes": rows, "termination": "outgoing local load; verified small-angle continuation",
            "transverse_approximation": "axial potential plus a recorded first-order transverse phase error estimate; not a rigorous coupled scattering bound",
            "reservoir": "normal Robin admittance k(E); common per-energy injection normalisation",
            "particle_interpretation": "near-field evanescent modes have no positive classical particle representation",
            "validation_status": "DEVELOPMENT_NOT_FULL_TEM_STEM_ACCEPTANCE",
            "gun_source_admission": "NOT_A_GUN_CHECKPOINT", "full_chain_status": "INCOMPLETE",
        })
