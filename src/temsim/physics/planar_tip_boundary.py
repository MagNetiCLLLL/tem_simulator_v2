"""Explicit development solve of a planar tip's driven near field.

This is a Robin reservoir at the *physical tip*, not a downstream source.
The stationary scalar, spinless Klein--Gordon orbital equation retains both
axial directions and evanescent transverse channels. The reservoir has the
same normal admittance k(E) for all modes of one energy: their occupation
weights are not changed by separate transmitted-current normalisation.

The adapter captures the actual instrument electric field. It accepts only a
bounded electrostatic domain whose box fits inside all hardware and contains
no magnetic operator. Its far face has an explicit local outgoing load. That
is a DEVELOPMENT termination, not a solved load from the remainder of the gun;
this result cannot be admitted as a TipGunCheckpoint or full-chain result.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.constants import c, e, hbar, m_e
from scipy.linalg import solve

from temsim.immutable_json import freeze_json, json_digest
from temsim.physics.electrostatic_wave_channels import potential_matrix_v
from temsim.physics.scattering_load import hermitian_slab, outgoing_load


@dataclass(frozen=True)
class PlanarTipBoundaryNumerics:
    end_z_nm: float = 2.
    axial_slices: int = 4
    spectral_pixels: int = 24
    potential_quadrature_factor: int = 2
    source_spectral_tail_tolerance: float = 1e-6
    maximum_working_bytes: int = 2*1024**3

    def validate(self):
        if not math.isfinite(self.end_z_nm) or self.end_z_nm <= 0:
            raise ValueError("Near-tip endpoint must be positive and finite in nm")
        for name, lower, upper in (("axial_slices", 1, 4096),
                                   ("spectral_pixels", 2, 64),
                                   ("potential_quadrature_factor", 2, 16)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
        if not math.isfinite(self.source_spectral_tail_tolerance) or not 0 < self.source_spectral_tail_tolerance < 1:
            raise ValueError("Source spectral tail tolerance must lie in (0, 1)")
        if type(self.maximum_working_bytes) is not int or self.maximum_working_bytes <= 0:
            raise ValueError("Near-tip memory budget must be a positive integer")
        return self


def _frozen(array):
    value = np.asarray(array)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class DrivenBoundarySolution:
    """Complex traces and covariant normal derivatives, in the same chart.

    Coefficients use unit incident current in inverse-length chart units:
    Im(f* Df) is a fraction of incident current. Multiplication by hbar/m
    cancels between numerator and denominator. This is never an L2 flux rule.
    """
    amplitude: np.ndarray
    derivative_per_m: np.ndarray
    injected_amplitude: np.ndarray
    reflected_amplitude: np.ndarray
    current_fraction: np.ndarray
    record: object

    def __post_init__(self):
        for name in ("amplitude", "derivative_per_m", "injected_amplitude",
                     "reflected_amplitude", "current_fraction"):
            object.__setattr__(self, name, _frozen(getattr(self, name)))
        object.__setattr__(self, "record", freeze_json(self.record))


def solve_robin_load(load, boundary_coefficients, normal_wave_number_per_m):
    """Couple an unchanged reservoir shape to a COMPLETE supplied right load.

    At the left face Df + i*k_n*f = 2*i*k_n*a. ``boundary_coefficients``
    retains its input L2 norm; no omitted support or physical loss is restored.
    k_n is the physical injection admittance, independent of the load's
    numerical current chart kappa. Both directions are solved simultaneously.
    """
    shape = np.asarray(boundary_coefficients, complex)
    k = normal_wave_number_per_m
    if (shape.ndim != 1 or not np.isfinite(shape).all()
            or not math.isfinite(k) or k <= 0
            or load.input_admittance.shape != (len(shape), len(shape))):
        raise ValueError("Finite reservoir coefficients and positive normal admittance are required")
    incident = shape/math.sqrt(k)
    total = solve(load.input_admittance+1j*k*np.eye(len(shape)), 2j*k*incident,
                  check_finite=False)
    amplitude, derivative = load.propagate(total)
    reflected = total-incident
    injected = float(k*np.vdot(incident, incident).real)
    returned = float(k*np.vdot(reflected, reflected).real)
    current = np.imag(np.sum(amplitude.conj()*derivative, axis=1))
    imbalance = float(abs(injected-returned-current[-1]))
    drift = float(np.max(abs(current-current[0])))
    tolerance = 1e-8*max(injected, 1e-30)
    if (not np.isfinite(current).all() or current.min() < -tolerance
            or imbalance > tolerance or drift > tolerance):
        raise ValueError("Driven boundary did not conserve incident, reflected and transmitted current")
    return DrivenBoundarySolution(amplitude, derivative, incident, reflected, current, {
        "injected_current_fraction": injected, "reflected_current_fraction": returned,
        "transmitted_current_fraction": float(current[-1]),
        "current_balance_residual": imbalance, "interior_current_residual": drift,
        "reservoir_normal_wave_number_per_m": float(k),
        "normalisation": "common normal-admittance flux per energy; no output normalisation",
    })


def _wave_number_squared(energy_ev):
    kinetic = np.asarray(energy_ev)
    return 2*m_e*e/hbar**2*kinetic*(1+kinetic/(2*m_e*c*c/e))


def _retained_source(mode, pixels, tolerance):
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
    plane = mode.plane
    full = plane.full_amplitude(float(wavelength_m(mode.energy_kev*1e3)))
    if pixels > min(full.shape):
        raise ValueError("Retained source channels exceed the emitted source grid")
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(full), norm="ortho"))
    slices = tuple(slice(n//2-pixels//2, n//2-pixels//2+pixels) for n in full.shape)
    retained = spectrum[slices]
    missing = max(0., float(np.sum(abs(spectrum)**2)-np.sum(abs(retained)**2)))
    if missing > tolerance:
        raise ValueError(f"Near-tip source spectral support is unresolved ({missing:.6g}); increase spectral_pixels")
    # This changes the sampling pitch, never the physical transverse period.
    basis = plane.basis_m @ np.diag((full.shape[1]/pixels, full.shape[0]/pixels))
    return retained.ravel(), basis, missing


def _guard_hardware(state, stop_mm, corner_radius_mm):
    from temsim.physics.column_wave import _prepare_column
    gun = state.electron_gun
    if stop_mm >= gun.exit_plane_z_mm:
        raise ValueError("The local boundary termination is not a full gun-exit load")
    if bool(getattr(state, "energy_filter_installed", False)) and state.energy_filter.entrance_z_mm <= stop_mm:
        raise ValueError("The local tip domain reaches the energy filter; it cannot be bypassed")
    plan, radii, stops, owners = _prepare_column(state, 0., stop_mm, stop_mm/8)
    if stops or owners or np.any(radii <= corner_radius_mm):
        raise ValueError("An installed column interception/field requires a joint near-tip operator")
    if plan.mapped_fields:
        raise ValueError("Mapped fields require a joint magnetic near-tip operator")
    for part in gun.bore_components:
        low = part.mechanical_center_from_tip_mm-.5*part.mechanical_length_mm
        high = part.mechanical_center_from_tip_mm+.5*part.mechanical_length_mm
        if high >= 0 and low <= stop_mm and corner_radius_mm >= .5*part.mechanical_clear_bore_diameter_mm:
            raise ValueError(f"Near-tip box intersects {part.key}; its absorption cannot be omitted")
    for aperture in (gun.dpa_aperture, gun.c1_aperture):
        if bool(aperture.enabled) and 0 <= aperture.z_mm <= stop_mm:
            raise ValueError(f"Near-tip domain reaches {aperture.key}; its mask cannot be omitted")
    return plan.signature


def build_planar_tip_boundary(state, numerics=PlanarTipBoundaryNumerics(), *,
                              source_numerics=None, cancelled=lambda: False,
                              progress_callback=None):
    """Execute physical Gaussian tip modes in the captured local electric field.

    An explicit research entry point. It does not enable a source, change its
    size/energy, produce a gun-exit cache, or bypass the downstream gun.
    The periodic transverse box and local outgoing termination must be
    converged independently before using this kernel in a full coupled solve.
    """
    from temsim.calculation_manifest import solver_source_identity
    from temsim.instrument_snapshot import encode_instrument
    from temsim.optics.electron_gun.tip_coherence import (
        TipWaveNumerics, describe_physical_tip_source, generate_tip_boundary_emission,
    )
    from temsim.physics.instrument_electric import capture_instrument_electric_field
    from temsim.physics.instrument_magnetic import capture_instrument_magnetic_field
    from temsim.physics.wave_execution import check_available_memory
    numerics.validate()
    if cancelled():
        raise InterruptedError("Near-tip boundary calculation cancelled")
    graph = encode_instrument(state)
    physical_id = json_digest(graph)
    implementation = solver_source_identity()
    captured = deepcopy(state)
    gun = captured.electron_gun
    emitted = generate_tip_boundary_emission(gun, source_numerics or TipWaveNumerics())
    # The tip component's mechanical body centre is behind the emitting end;
    # it is not the emission-plane coordinate.
    if float(emitted.record["plane_z_mm"]) != 0:
        raise ValueError("The present planar boundary requires the declared physical tip emission plane Z=0")
    count, layers = numerics.spectral_pixels**2, numerics.axial_slices
    required = 16*(12*layers+48)*count**2
    if required > numerics.maximum_working_bytes:
        raise MemoryError(f"Near-tip two-way operator needs approximately {required} bytes")
    check_available_memory(required)
    electric = capture_instrument_electric_field(captured)
    magnetic = capture_instrument_magnetic_field(captured)
    edges = np.linspace(0., numerics.end_z_nm*1e-9, layers+1)
    source_phi = float(electric.potential_rise_v_at_global_positions(np.zeros((1, 3)))[0])
    pixels = numerics.spectral_pixels
    quadrature = pixels*numerics.potential_quadrature_factor
    yy, xx = np.meshgrid(np.arange(quadrature)-quadrature//2,
                         np.arange(quadrature)-quadrature//2, indexing="ij")
    fy, fx = np.meshgrid((np.arange(pixels)-pixels//2)/pixels,
                         (np.arange(pixels)-pixels//2)/pixels, indexing="ij")
    results, records = [], []
    for mode_index, mode in enumerate(emitted.modes()):
        if cancelled():
            raise InterruptedError("Near-tip boundary calculation cancelled")
        coefficients, basis, omitted = _retained_source(mode, pixels, numerics.source_spectral_tail_tolerance)
        xy = mode.plane.origin_m[:, None, None]+np.einsum("ij,jyx->iyx",
            basis/numerics.potential_quadrature_factor, np.stack((xx, yy)))
        positions = np.zeros((quadrature, quadrature, 3))
        positions[..., :2] = np.moveaxis(xy, 0, -1)
        extent = float(np.max(np.hypot(xy[0], xy[1])))
        hardware_id = _guard_hardware(captured, edges[-1]*1e3, extent*1e3)
        wave_vectors = 2*np.pi*np.einsum("ij,jyx->iyx", np.linalg.inv(basis).T, np.stack((fx, fy)))
        kxy2 = np.sum(wave_vectors**2, axis=0).ravel()
        energy = mode.energy_kev*1e3
        k_in = math.sqrt(float(_wave_number_squared(energy)))
        operators, exit_q, kinetic_ranges = [], None, []
        evanescent = []
        for index, z in enumerate(np.r_[(edges[1:]+edges[:-1])/2, edges[-1]]):
            if cancelled():
                raise InterruptedError("Near-tip boundary calculation cancelled")
            positions[..., 2] = z
            if np.any(magnetic.field_at_global_positions_t(positions)):
                raise ValueError("A magnetic field reaches the near-tip box; this electrostatic solve cannot omit it")
            potential = electric.potential_rise_v_at_global_positions(positions)
            kinetic = energy+potential-source_phi
            if not np.isfinite(kinetic).all() or kinetic.min() <= 0:
                raise ValueError("The scalar near-tip domain reaches a non-positive kinetic energy boundary")
            q = potential_matrix_v(_wave_number_squared(kinetic), (pixels, pixels),
                                   maximum_working_bytes=numerics.maximum_working_bytes)
            q[np.diag_indices(count)] -= kxy2
            kinetic_ranges.append((float(kinetic.min()), float(kinetic.max())))
            if index < layers:
                blocks, row = hermitian_slab(q, np.zeros_like(q), edges[index+1]-edges[index], k_in)
                operators.append(blocks)
                evanescent.append(row["evanescent_channels"])
            else:
                exit_q = q
        load = outgoing_load(operators, exit_q, k_in, cancelled=cancelled)
        solution = solve_robin_load(load, coefficients, k_in)
        results.append(solution)
        records.append({"mode_id": mode.mode_id, "energy_ev": energy,
            "incident_mode_weight": mode.weight_per_reference_electron,
            "source_spectral_omission": omitted, "basis_m": basis.tolist(),
            "origin_m": mode.plane.origin_m.tolist(), "hardware_signature": hardware_id,
            "local_kinetic_ranges_ev": kinetic_ranges, "evanescent_channels_per_slab": evanescent,
            **dict(solution.record)})
        if progress_callback is not None:
            progress_callback(mode_index+1, emitted.record["mode_count"], "Executed driven near-tip mode")
    if json_digest(encode_instrument(state)) != physical_id or solver_source_identity() != implementation:
        raise ValueError("Physical inputs or implementation changed during the near-tip calculation")
    return tuple(results), freeze_json({
        "schema": "planar-tip-driven-boundary-development-v1", "source": emitted.record,
        "physical_source_identity": json_digest(describe_physical_tip_source(gun.emitter)),
        "instrument_input_identity": physical_id, "implementation": implementation,
        "electric_physical_identity": electric.physical_identity,
        "electric_numerical_identity": electric.numerical_identity,
        "magnetic_identity": magnetic.numerical_identity,
        "numerics": asdict(numerics), "source_numerics": asdict(emitted.numerics),
        "z_edges_m": edges.tolist(), "modes": records,
        "reference_current_a": emitted.reference_current_a,
        "termination": "local constant outgoing half-space; not the complete downstream gun load",
        "scope": "bounded electrostatic planar-tip development solve; periodic transverse box",
        "gun_source_admission": "NOT_A_GUN_CHECKPOINT", "full_chain_status": "INCOMPLETE",
    })
