"""Sample the installed gun's electric potential into coupled transverse modes.

Field preparation only: no particles are emitted and no gun-exit state/cache
is published. These matrices own the electrostatic contribution, not magnetic
deflectors/stigmators, the Wien magnetic field, bores, or apertures. A complete
gun operator must compose those operations before claiming a physical output.
The existing analytic radial-order-r^2 potential is not a curved-tip solution.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.constants import e, hbar, m_e

from temsim.immutable_json import freeze_json, json_digest
from temsim.physics.wave_execution import check_available_memory


@dataclass(frozen=True)
class ElectrostaticChannelLayers:
    q_m2: np.ndarray
    widths_m: np.ndarray
    exit_q_m2: np.ndarray
    record: object


def potential_matrix_v(potential_v, wave_shape, *, maximum_working_bytes=72*1024**3):
    """Non-circular Fourier-Galerkin matrix of an independently sampled field.

    Grids share a centred physical period. Flattened channels are ordered
    ky then kx; Fourier sign is exp(-i*k.x). At least 2x potential quadrature
    resolves distinct retained frequency differences. Converge it separately.
    """
    phi = np.asarray(potential_v)
    if (len(wave_shape) != 2 or any(type(n) is not int or n < 2 for n in wave_shape)
            or phi.ndim != 2 or np.iscomplexobj(phi) or not np.all(np.isfinite(phi))
            or any(m < 2*n for m, n in zip(phi.shape, wave_shape))):
        raise ValueError("Real potential quadrature must have at least twice the wave samples per axis")
    ny, nx = wave_shape
    required = 128*(ny*nx)**2+48*phi.size
    if type(maximum_working_bytes) is not int or maximum_working_bytes <= 0:
        raise ValueError("Potential matrix memory budget must be a positive integer")
    if required > maximum_working_bytes:
        raise MemoryError(f"Potential matrix needs approximately {required} bytes")
    check_available_memory(required)
    ky, kx = np.meshgrid(np.arange(ny)-ny//2, np.arange(nx)-nx//2, indexing="ij")
    ky, kx = ky.ravel(), kx.ravel()
    coefficients = np.fft.fft2(np.fft.ifftshift(phi), norm="forward")
    return coefficients[(ky[:, None]-ky[None, :]) % phi.shape[0],
                        (kx[:, None]-kx[None, :]) % phi.shape[1]]


def sample_feg_electrostatic_channels(gun, z_edges_mm, emission_energy_ev,
        wave_shape, basis_m, origin_m, *, quadrature_factor=2,
        maximum_channels=1024, maximum_working_bytes=72*1024**3,
        cancelled=lambda: False):
    """Read ALL installed electrostatic providers without mutating controls.

    emission_energy_ev is an energy component at the physical tip; downstream
    kinetic gain is read from the gun field, never entered as a new source.
    This internal preparation API is not a public image/source admission route.
    The nonrelativistic kinetic operator admits positive tip energies <=1 keV
    and local E+phi-phi_tip in [-1, +1] keV (negative values are barriers).
    The final sample is only a boundary-matching matrix, not an assertion that
    the rest of the physical gun is a uniform outgoing medium.
    """
    from temsim.optics.electron_gun.field_emission import FieldEmissionGun
    from temsim.optics.electron_gun.electrostatic import FegElectrostaticField
    from temsim.optics.electron_gun.monochromator import AnalyticWienField, CombinedElectricField
    from temsim.instrument_snapshot import encode_instrument
    from temsim.calculation_manifest import solver_source_identity
    if type(gun) is not FieldEmissionGun:
        raise ValueError("This field adapter requires the existing physical FEG assembly")
    implementation = solver_source_identity()
    # Exact graph restoration does not apply presets or normalize live fields.
    graph = encode_instrument(gun)
    # decode_instrument only accepts a State root; detach this gun by deepcopy
    # instead, retaining the graph for a complete, geometry-inclusive identity.
    from copy import deepcopy
    working = deepcopy(gun)
    if json_digest(encode_instrument(working)) != json_digest(graph):
        raise ValueError("Electrostatic input capture changed the gun")
    working.validate()
    if json_digest(encode_instrument(working)) != json_digest(graph):
        raise ValueError("Gun validation changed captured settings; resolve the assembly before field sampling")
    if working.monochromator_installed and type(working.monochromator.field_provider) is not AnalyticWienField:
        raise ValueError("Imported Wien provider needs explicit electrostatic channel support; it cannot be skipped")
    field = working.electric_field
    if type(field) not in (FegElectrostaticField, CombinedElectricField):
        raise ValueError("Unsupported gun electric field provider")
    # Detach numerical inputs too: a live editor must not change the lattice
    # halfway through sampling while the returned record describes a new one.
    edges, basis, origin = (np.array(value, copy=True)
                            for value in (z_edges_mm, basis_m, origin_m))
    if (edges.ndim != 1 or len(edges) < 2 or np.iscomplexobj(edges)
            or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0)
            or edges[0] < 0 or edges[-1] > working.exit_plane_z_mm):
        raise ValueError("Axial sampling edges must be ordered within the physical gun")
    if (basis.shape != (2, 2) or origin.shape != (2,) or np.iscomplexobj(basis) or np.iscomplexobj(origin)
            or not np.all(np.isfinite(basis)) or not np.all(np.isfinite(origin))
            or np.linalg.det(basis) == 0):
        raise ValueError("A finite nonsingular SI transverse lattice is required")
    if (len(wave_shape) != 2 or any(type(n) is not int or n < 2 for n in wave_shape)
            or type(quadrature_factor) is not int or not 2 <= quadrature_factor <= 16):
        raise ValueError("Wave shape and potential quadrature factor are invalid")
    wave_shape = tuple(wave_shape)
    n, layers = math.prod(wave_shape), len(edges)-1
    if type(maximum_channels) is not int or maximum_channels < n:
        raise ValueError("Electrostatic channel count exceeds the numerical budget")
    if type(maximum_working_bytes) is not int or maximum_working_bytes <= 0:
        raise ValueError("Electrostatic memory budget must be a positive integer")
    required = 16*(3*layers+20)*n*n+256*n*quadrature_factor**2
    if required > maximum_working_bytes:
        raise MemoryError(f"Electrostatic matrices need approximately {required} bytes")
    check_available_memory(required)
    if cancelled():
        raise InterruptedError("Electrostatic channel preparation cancelled")
    if isinstance(emission_energy_ev, bool) or not math.isfinite(emission_energy_ev) or not 0 < emission_energy_ev <= 1000:
        raise ValueError("The nonrelativistic field adapter requires tip energies in (0, 1000] eV")
    ny, nx = wave_shape
    my, mx = ny*quadrature_factor, nx*quadrature_factor
    yy, xx = np.meshgrid(np.arange(my)-my//2, np.arange(mx)-mx//2, indexing="ij")
    xy = origin[:, None, None]+np.einsum("ij,jyx->iyx", basis/quadrature_factor, np.stack((xx, yy)))
    positions = np.zeros((my, mx, 3))
    positions[..., :2] = np.moveaxis(xy, 0, -1)
    fy, fx = np.meshgrid((np.arange(ny)-ny//2)/ny, (np.arange(nx)-nx//2)/nx, indexing="ij")
    kxy = 2*np.pi*np.einsum("ij,jyx->iyx", np.linalg.inv(basis).T, np.stack((fx, fy)))
    transverse_squared = (kxy*kxy).sum(axis=0).ravel()
    phi_tip = float(field.potential_v_at_global_positions(np.zeros((1, 3)))[0])
    coefficient = 2*m_e*e/hbar**2
    q, ranges = [], []
    sample_z = np.r_[(edges[:-1]+edges[1:])/2, edges[-1]]
    for z in sample_z:
        if cancelled():
            raise InterruptedError("Electrostatic channel preparation cancelled")
        positions[..., 2] = z*1e-3
        phi = np.asarray(field.potential_v_at_global_positions(positions))
        if phi.shape != positions.shape[:-1]:
            raise ValueError("Gun potential provider must return one value per sampled position")
        kinetic = emission_energy_ev+phi-phi_tip
        if not np.all(np.isfinite(kinetic)) or abs(kinetic).max() > 1000:
            raise ValueError("Local electrostatic energy exceeds the nonrelativistic field adapter domain; no accelerating stage was bypassed")
        matrix = coefficient*potential_matrix_v(kinetic, wave_shape, maximum_working_bytes=maximum_working_bytes)
        matrix[np.diag_indices(n)] -= transverse_squared
        q.append(matrix)
        ranges.append((float(kinetic.min()), float(kinetic.max())))
    if json_digest(encode_instrument(gun)) != json_digest(graph):
        raise ValueError("Gun settings changed during electric-field preparation")
    if solver_source_identity() != implementation:
        raise ValueError("Solver implementation changed during electric-field preparation")
    if cancelled():
        raise InterruptedError("Electrostatic channel preparation cancelled")
    arrays = (np.asarray(q[:-1]), np.diff(edges)*1e-3, q[-1])
    frozen = [np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape) for a in arrays]
    return ElectrostaticChannelLayers(*frozen, freeze_json({
        "schema": "feg-electrostatic-channel-layers-v1", "gun_graph": graph,
        "gun_input_identity": json_digest(graph), "implementation": implementation,
        "z_edges_mm": edges.tolist(), "sample_z_mm": sample_z.tolist(),
        "tip_component_energy_ev": float(emission_energy_ev), "tip_potential_v": phi_tip,
        "wave_shape": wave_shape, "basis_m": basis.tolist(), "origin_m": origin.tolist(),
        "potential_quadrature_factor": quadrature_factor, "local_kinetic_ranges_ev": ranges,
        "electrostatic_owners": [working.extractor.key, working.electrostatic_lens.key, working.accelerator.key]
            + ([working.monochromator.wien.key] if working.monochromator_installed else []),
        "scope": "electric potential contribution only, not complete gun propagation",
        "missing_operations": "magnetic coupling, absorbing material/bores/apertures, curved-tip field and flux boundary",
        "gun_source_admission": "NOT_A_GUN_CHECKPOINT"}))
