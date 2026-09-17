"""Executed coherent tip near field, not a complete gun/column checkpoint.

Stationary nonrelativistic scalar Schrodinger equation in an axisymmetric
vacuum domain with the *curved tip* as its driven boundary. Linear triangular
FEM uses nm coordinates and the cylindrical measure 2*pi*r dr dz. The driven
Robin reservoir injects prescribed incoming current; reflected current returns
to that reservoir. Top/side Robin ports are first-order absorbing boundaries,
NOT exact transparent boundaries or a downstream electron source.

The electric potential is sampled from the actual grounded electrode solve.
No paraxial expansion, ray phase reconstruction, tunnelling-current prediction,
magnetic omission, source-width fitting or outgoing renormalisation is used.
"""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.constants import e, hbar, m_e
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from temsim.immutable_json import freeze_json, json_digest

# k^2 [nm^-2] per kinetic energy [eV].
KINETIC_NM2_PER_EV = 2*m_e*e/hbar**2*1e-18


@dataclass(frozen=True)
class SurfaceWaveNumerics:
    radial_nodes: int = 97
    axial_nodes: int = 193
    outer_radius_factor: float = 1.5
    exit_height_nm: float = 2.0
    energy_samples: int = 3
    maximum_nodes: int = 1_000_000
    maximum_working_bytes: int = 8*1024**3
    linear_tolerance: float = 1e-9
    flux_tolerance: float = 1e-8
    element_order: int = 1
    joint_radial_phase: bool = True

    def validate(self):
        for name, lo, hi in (("radial_nodes", 3, 4097), ("axial_nodes", 3, 4097),
                             ("energy_samples", 1, 64), ("maximum_nodes", 9, 10_000_000)):
            value = getattr(self, name)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
        for name in ("outer_radius_factor", "exit_height_nm", "linear_tolerance", "flux_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.outer_radius_factor <= 1 or max(self.linear_tolerance, self.flux_tolerance) > 1e-4:
            raise ValueError("The radial boundary must enclose the patch and tolerances must be <= 1e-4")
        if type(self.maximum_working_bytes) is not int or self.maximum_working_bytes <= 0:
            raise ValueError("Near-field memory budget must be a positive integer")
        if type(self.element_order) is not int or self.element_order not in (1, 2):
            raise ValueError("Wave element order must be 1 or 2")
        if self.element_order == 2 and (self.radial_nodes % 2 != 1 or self.axial_nodes % 2 != 1):
            raise ValueError("Quadratic wave elements require odd radial and axial node counts")
        if type(self.joint_radial_phase) is not bool:
            raise ValueError("Joint radial phase coordinates must be explicitly enabled or disabled")
        return self


def _immutable(array):
    a = np.asarray(array)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True)
class SurfaceWaveMode:
    energy_ev: float
    weight: float
    amplitude: np.ndarray  # (radial nodes, axial nodes), complex flux-normalised field
    flux: object
    phase_reference: str = "tip-apex incoming reservoir phase = 0 rad"


@dataclass(frozen=True)
class SurfaceWaveCheckpoint:
    radius_nm: np.ndarray
    z_nm: np.ndarray
    potential_rise_v: np.ndarray
    modes: tuple[SurfaceWaveMode, ...]
    reference_current_a: float
    record: object

    @property
    def digest(self):
        from hashlib import sha256
        content = sha256()
        for a in (self.radius_nm, self.z_nm, self.potential_rise_v, *(m.amplitude for m in self.modes)):
            content.update(a.tobytes())
        return json_digest({"record": self.record, "arrays": content.hexdigest(),
            "reference_current_a": self.reference_current_a,
            "modes": [{"energy_ev": m.energy_ev, "weight": m.weight,
                       "phase_reference": m.phase_reference, "flux": m.flux} for m in self.modes]})

    @property
    def outgoing_current_a(self):
        return self.reference_current_a*sum(m.weight*m.flux["top"] for m in self.modes)

    @property
    def density(self):
        """Incoherent energy sum of |psi|^2, not an aggregate complex field."""
        return sum(m.weight*abs(m.amplitude)**2 for m in self.modes)


def _cancel(callback):
    if callback():
        raise InterruptedError("Coherent tip near-field calculation cancelled")


def structured_mesh(radius_nm, bottom_nm, top_nm, axial_nodes):
    r, bottom = np.asarray(radius_nm, float), np.asarray(bottom_nm, float)
    if (r.ndim != 1 or r.size < 3 or r[0] != 0 or np.any(np.diff(r) <= 0)
            or not np.all(np.isfinite(r)) or type(axial_nodes) is not int or axial_nodes < 3
            or bottom.shape != r.shape or not np.all(np.isfinite(bottom))
            or not math.isfinite(top_nm) or np.any(bottom >= top_nm)):
        raise ValueError("Invalid curved cylindrical wave domain")
    z = bottom[:, None]+(top_nm-bottom[:, None])*np.linspace(0., 1., axial_nodes)
    points = np.column_stack((np.broadcast_to(r[:, None], z.shape).ravel(), z.ravel()))
    ids = np.arange(points.shape[0]).reshape(z.shape)
    a, b, c_, d = (ids[:-1, :-1].ravel(), ids[1:, :-1].ravel(),
                   ids[1:, 1:].ravel(), ids[:-1, 1:].ravel())
    triangles = np.vstack((np.column_stack((a, b, c_)), np.column_stack((a, c_, d))))
    edges = {"source": np.column_stack((ids[:-1, 0], ids[1:, 0])),
             "top": np.column_stack((ids[:-1, -1], ids[1:, -1])),
             "side": np.column_stack((ids[-1, :-1], ids[-1, 1:]))}
    return points, triangles, edges, z


def fem_volume(points, triangles, potential_rise_v):
    """Return stiffness, unit-energy mass and potential mass; all real symmetric.

    Tensor Gauss quadrature under a Duffy transform integrates the P1 radial
    measure and nodal potential products. No diagonal mass-lumping is used.
    """
    if triangles.shape[1] == 6:
        from temsim.physics.quadratic_axisymmetric_fem import volume
        return volume(points, triangles, potential_rise_v)
    p = points[triangles]
    determinant = ((p[:, 1, 0]-p[:, 0, 0])*(p[:, 2, 1]-p[:, 0, 1])
                   -(p[:, 2, 0]-p[:, 0, 0])*(p[:, 1, 1]-p[:, 0, 1]))
    if np.any(determinant <= 0):
        raise ValueError("Wave mesh has inverted or degenerate triangles")
    grad = np.stack((p[:, [1, 2, 0], 1]-p[:, [2, 0, 1], 1],
                     p[:, [2, 0, 1], 0]-p[:, [1, 2, 0], 0]), axis=-1)/determinant[:, None, None]
    stiffness = np.einsum("tij,tkj->tik", grad, grad)*(math.pi*determinant*p[:, :, 0].mean(axis=1))[:, None, None]
    mass, potential_mass = np.zeros_like(stiffness), np.zeros_like(stiffness)
    nodes, weights = np.polynomial.legendre.leggauss(4)
    nodes, weights = (nodes+1)/2, weights/2
    potential = np.asarray(potential_rise_v)[triangles]
    for u, wu in zip(nodes, weights):
        for v, wv in zip(nodes, weights):
            basis = np.array([1-u, u*(1-v), u*v])
            factor = 2*math.pi*determinant*u*wu*wv*(p[:, :, 0]@basis)
            shape = np.outer(basis, basis)
            mass += factor[:, None, None]*shape
            potential_mass += (factor*(potential@basis))[:, None, None]*shape
    rows = np.broadcast_to(triangles[:, :, None], stiffness.shape).ravel()
    cols = np.broadcast_to(triangles[:, None, :], stiffness.shape).ravel()
    shape = (len(points), len(points))
    return tuple(coo_matrix((v.ravel(), (rows, cols)), shape=shape).tocsc()
                 for v in (stiffness, mass, potential_mass))


def robin_port(points, edges, kinetic_ev):
    """Positive local Sommerfeld flux matrix, with cylindrical surface measure."""
    if edges.shape[1] == 3:
        from temsim.physics.quadratic_axisymmetric_fem import boundary
        return boundary(points, edges, np.sqrt(KINETIC_NM2_PER_EV*np.asarray(kinetic_ev)))
    p = points[edges]
    length = np.linalg.norm(p[:, 1]-p[:, 0], axis=1)
    energy = np.asarray(kinetic_ev)[edges]
    if np.any(energy <= 0) or not np.all(np.isfinite(energy)):
        raise ValueError("The local Robin port requires positive real kinetic energy")
    matrix = np.zeros((len(edges), 2, 2))
    nodes, weights = np.polynomial.legendre.leggauss(5)
    for u, w in zip((nodes+1)/2, weights/2):
        basis = np.array([1-u, u])
        k = np.sqrt(KINETIC_NM2_PER_EV*(energy@basis))
        factor = 2*math.pi*length*w*(p[:, :, 0]@basis)*k
        matrix += factor[:, None, None]*np.outer(basis, basis)
    rows = np.broadcast_to(edges[:, :, None], matrix.shape).ravel()
    cols = np.broadcast_to(edges[:, None, :], matrix.shape).ravel()
    return coo_matrix((matrix.ravel(), (rows, cols)), shape=(len(points), len(points))).tocsc()


def solve_driven_wave(points, triangles, edges, potential, energy_ev, incident_amplitude,
                      *, matrices=None, linear_tolerance=1e-9, flux_tolerance=1e-8,
                      top_load=None, cancelled=lambda: False):
    """One fixed-energy *incoming* reservoir mode, including reflected flux.

    Boundary: dn psi = i*k*psi - 2*i*k*a at the tip; dn psi=i*k*psi
    on top/side. Incoming a is normalised BEFORE solving. No output norm is
    forced. Since the volume is Hermitian, R+T+side=1 is a discrete identity;
    it is not evidence that the finite domain/Robin approximation is converged.
    """
    _cancel(cancelled)
    potential = np.asarray(potential, float)
    if (not math.isfinite(energy_ev) or energy_ev <= 0 or np.any(~np.isfinite(potential))
            or np.min(energy_ev+potential) <= 0 or np.max(energy_ev+potential) > 1000):
        raise ValueError("Near-field scalar nonrelativistic solver requires kinetic energies in (0, 1000] eV")
    k, m, p = fem_volume(points, triangles, potential) if matrices is None else matrices
    ports = {key: robin_port(points, value, energy_ev+potential) for key, value in edges.items()}
    if top_load is not None:
        # Weak Dirichlet-to-Neumann load: dn psi = Y psi at the top. Its
        # anti-Hermitian part is the outgoing flux form. Reflections from the
        # appended domain therefore alter the solved tip field itself.
        top_load = top_load.tocsc()
        if top_load.shape != k.shape or not np.all(np.isfinite(top_load.data)):
            raise ValueError("The coupled top load must match the finite near-field matrix")
        ports["top"] = (top_load-top_load.conj().T)/(2j)
    a = np.asarray(incident_amplitude, complex).copy()
    if a.shape != (len(points),) or not np.all(np.isfinite(a)):
        raise ValueError("Provide a finite incoming surface amplitude per mesh node")
    input_flux = float(np.vdot(a, ports["source"]@a).real)
    if not math.isfinite(input_flux) or input_flux <= 0:
        raise ValueError("The prescribed surface mode carries no incoming reservoir flux")
    a /= math.sqrt(input_flux)
    matrix = (k-KINETIC_NM2_PER_EV*(energy_ev*m+p)).astype(complex)
    for name, port in ports.items():
        matrix -= top_load if name == "top" and top_load is not None else 1j*port
    rhs = -2j*(ports["source"]@a)
    _cancel(cancelled)
    factor = splu(matrix.tocsc())
    _cancel(cancelled)  # SuperLU itself has no cooperative cancellation hook
    psi = factor.solve(rhs)
    error = np.linalg.norm(matrix@psi-rhs)/max(np.linalg.norm(rhs), np.finfo(float).tiny)
    if not np.all(np.isfinite(psi)) or not math.isfinite(error) or error > linear_tolerance:
        raise ValueError(f"Near-field linear residual {error:.3g} exceeds {linear_tolerance:.3g}")
    reflected = psi-a
    flux = {"incoming": 1., "reflected": float(np.vdot(reflected, ports["source"]@reflected).real),
            "top": float(np.vdot(psi, ports["top"]@psi).real),
            "side": float(np.vdot(psi, ports["side"]@psi).real), "linear_residual": float(error)}
    flux["balance_error"] = abs(1-flux["reflected"]-flux["top"]-flux["side"])
    if flux["balance_error"] > flux_tolerance or min(flux[key] for key in ("reflected", "top", "side")) < -flux_tolerance:
        raise ValueError(f"Near-field flux balance failed: {flux}")
    _cancel(cancelled)
    return psi, freeze_json(flux)


_CACHE = OrderedDict()


def compute_surface_wave(gun, numerics=SurfaceWaveNumerics(), *, use_cache=True,
                         cancelled=lambda: False, progress_callback=None):
    return _compute_surface_wave(gun, numerics, use_cache=use_cache,
        cancelled=cancelled, progress_callback=progress_callback)


def prepare_surface_problem(gun, numerics=SurfaceWaveNumerics(), *, cancelled=lambda: False):
    """Prepare the SAME physical curved reservoir for a jointly solved load.

    No wave is launched or cached at an artificial downstream source plane.
    The caller must supply and solve its complete downstream boundary load.
    """
    return _compute_surface_wave(gun, numerics, use_cache=False,
        cancelled=cancelled, progress_callback=None, problem_only=True)


def guard_surface_column(state, numerics, *, prepare_column=None):
    """Check the actual column over the whole curved near-tip domain."""
    if prepare_column is None:
        from temsim.physics.column_wave import _prepare_column as prepare_column
    model = state.electron_gun.emitter.surface_model
    radius = model.geometry.apex_radius_nm
    extent = radius*math.sin(math.radians(model.emission.cap_half_angle_deg))*numerics.outer_radius_factor
    if extent >= radius:
        raise ValueError("Near-field radial extent exceeds the spherical tip")
    start = -extent**2/(radius+math.sqrt(radius**2-extent**2))*1e-6
    stop = numerics.exit_height_nm*1e-6
    if bool(getattr(state, "energy_filter_installed", False)) and state.energy_filter.entrance_z_mm <= stop:
        raise ValueError("The near-field request reaches the installed energy filter; it cannot bypass its operator")
    plan, radii, stops, owners = prepare_column(state, start, stop, (stop-start)/32)
    if stops or owners or np.any(radii < extent*1e-6):
        raise ValueError("A column aperture, deflector or wall intersects the near-field domain; a joint boundary is required")
    names = ("magnetic_t", "sx_m2", "sy_m2", "hex_normal_m3", "hex_skew_m3",
             "midpoint_magnetic_t", "midpoint_sx_m2", "midpoint_sy_m2", "midpoint_sxy_m2",
             "midpoint_hex_normal_m3", "midpoint_hex_skew_m3", "cs_kick_m3",
             "thin_power_m1", "thin_rotation_rad", "kick_x_rad", "kick_y_rad")
    if plan.mapped_fields or any(np.any(getattr(plan, name)) for name in names):
        raise ValueError("A column field reaches the coherent tip near field; the scalar electrostatic operator cannot omit it")
    return plan.signature


def _compute_surface_wave(gun, numerics, *, use_cache, cancelled, progress_callback, problem_only=False):
    """Compute only the tip-adjacent segment from physical inputs, never an exit source."""
    from temsim.calculation_manifest import solver_source_identity
    from temsim.instrument_snapshot import encode_instrument
    from temsim.physics.grounded_tip_field import grounded_field, field_request
    from temsim.physics.wave_execution import check_available_memory
    numerics.validate()
    _cancel(cancelled)
    working = deepcopy(gun)
    working.validate()
    model = getattr(working.emitter, "surface_model", None)
    if model is None or model.coherence is None:
        raise ValueError("Explicit coherent surface parameters are required at the physical tip")
    source = model.coherence.validate()
    energies, weights = source.energy_quadrature(numerics.energy_samples)
    geometry = model.geometry
    extent = geometry.apex_radius_nm*math.sin(math.radians(model.emission.cap_half_angle_deg))*numerics.outer_radius_factor
    if extent >= geometry.apex_radius_nm*math.cos(math.radians(geometry.cone_half_angle_deg)):
        raise ValueError("This near-field mesh must stay within the spherical apex; a full-cone wave domain is not implemented")
    count = numerics.radial_nodes*numerics.axial_nodes
    estimated = count*(16000+len(energies)*16)
    if count > numerics.maximum_nodes or estimated > numerics.maximum_working_bytes:
        raise MemoryError(f"Coherent near field needs {count} nodes and an estimated {estimated} working bytes; increase the declared budget or revise numerical resolution")
    check_available_memory(estimated)
    boundary = field_request(working)
    if numerics.exit_height_nm*1e-9 >= min(row[1] for row in boundary["rings"]):
        raise ValueError("Near-field domain reaches an electrode; that conductor cannot be skipped")
    implementation = solver_source_identity()
    identity = json_digest({"gun": encode_instrument(working), "numerics": asdict(numerics),
                            "field": boundary, "implementation": implementation})
    if use_cache and identity in _CACHE:
        _cancel(cancelled)
        _CACHE.move_to_end(identity)
        return _CACHE[identity]
    r = np.linspace(0, extent, numerics.radial_nodes)
    # Stable cap height, avoiding R-sqrt(R^2-r^2) cancellation near the axis.
    bottom = -r*r/(geometry.apex_radius_nm+np.sqrt(geometry.apex_radius_nm**2-r*r))
    for aperture in (working.dpa_aperture, working.c1_aperture):
        if aperture.enabled and bottom.min()*1e-6 <= aperture.z_mm <= numerics.exit_height_nm*1e-6:
            raise ValueError("Near-field domain reaches a gun aperture; its wave boundary cannot be skipped")
    mesher = structured_mesh
    if numerics.element_order == 2:
        from temsim.physics.quadratic_axisymmetric_fem import mesh
        mesher = mesh
    points, triangles, edges, z = mesher(r, bottom, numerics.exit_height_nm, numerics.axial_nodes)
    xyz = np.column_stack((points[:, 0], np.zeros(len(points)), points[:, 1]))*1e-9
    # The scalar axisymmetric operator cannot hide a magnetic contribution.
    for azimuth in (0., math.pi/2):
        rotated = xyz.copy()
        rotated[:, 0], rotated[:, 1] = xyz[:, 0]*math.cos(azimuth), xyz[:, 0]*math.sin(azimuth)
        magnetic = working.magnetic_field.field_at_global_positions_t(rotated)
        if np.any(magnetic != 0):
            raise ValueError("A magnetic field is present in the tip wave domain; an axisymmetric scalar electrostatic solver cannot omit it")
    field = grounded_field(working)
    potential = field.potential_rise_v_at_global_positions(xyz).copy()
    surface_ids = np.arange(numerics.radial_nodes)*numerics.axial_nodes
    # Boundary values are the actual conductor voltage, not a local energy
    # subtraction or a shifted source. Interior nodal interpolation is refined
    # independently from the electrode solve.
    potential[surface_ids] = 0.
    if working.monochromator_installed:
        extra = working.monochromator.field_provider.field_at_global_positions_v_per_m(xyz)
        if np.any(extra != 0):
            raise ValueError("An installed Wien field reaches the near-field domain and needs the joint electric/magnetic wave operator")
    matrices = fem_volume(points, triangles, potential)
    arc = np.arcsin(r/geometry.apex_radius_nm)/math.radians(model.emission.cap_half_angle_deg)
    envelope = np.where(arc < 1., np.cos(.5*math.pi*np.minimum(arc, 1.))**2, 0.)
    drive = np.zeros(len(points), complex)
    drive[surface_ids] = envelope*np.exp(1j*source.edge_phase_rad*arc**2)
    if problem_only:
        return {"points": points, "triangles": triangles, "edges": edges, "z": z,
            "radius_nm": r, "potential": potential, "drive": drive, "matrices": matrices,
            "energies": energies, "weights": weights, "model": model, "identity": identity,
            "implementation": implementation, "grounded_field": boundary}
    modes = []
    for index, (energy, weight) in enumerate(zip(energies, weights)):
        _cancel(cancelled)
        if progress_callback is not None:
            progress_callback(index, len(energies), "Coherent tip near field")
        psi, flux = solve_driven_wave(points, triangles, edges, potential, float(energy), drive,
            matrices=matrices, linear_tolerance=numerics.linear_tolerance,
            flux_tolerance=numerics.flux_tolerance, cancelled=cancelled)
        modes.append(SurfaceWaveMode(float(energy), float(weight), _immutable(psi.reshape(z.shape)), flux))
    _cancel(cancelled)
    if solver_source_identity() != implementation or json_digest(encode_instrument(gun)) != json_digest(encode_instrument(working)):
        raise RuntimeError("Coherent surface inputs changed during execution; result not published")
    result = SurfaceWaveCheckpoint(_immutable(r), _immutable(z), _immutable(potential.reshape(z.shape)),
        tuple(modes), model.current_na*1e-9, freeze_json({
            "schema": "executed-coherent-tip-near-field-v1", "identity": identity,
            "source": model.to_dict(), "grounded_field": boundary, "implementation": implementation,
            "numerics": asdict(numerics), "estimated_working_bytes": estimated,
            "equation": "axisymmetric stationary scalar Schrodinger; nonrelativistic vacuum, no space charge",
            "phase": "Per-energy complex fields; different energies are incoherent; no aggregate phase",
            "normalisation": "Unit incoming tip-reservoir flux before solving; no outgoing renormalisation",
            "ports": "Driven local Robin tip reservoir; local Sommerfeld top and side, approximate not exact transparent boundaries",
            "scope": "Executed tip near field ONLY; not a gun exit, TEM/STEM image or independent downstream source",
            "pending": "Full gun magnetic/relativistic transport, domain/port qualification and column/specimen/detector connection",
            "convergence": "Requires separate wave mesh, electrode mesh, energy quadrature and outer-boundary refinement"}))
    if use_cache:
        _CACHE[identity] = result
        while len(_CACHE) > 4 or sum(sum(m.amplitude.nbytes for m in v.modes) for v in _CACHE.values()) > numerics.maximum_working_bytes//4:
            _CACHE.popitem(last=False)
    if progress_callback is not None:
        progress_callback(len(energies), len(energies), "Coherent tip near field complete; gun propagation not yet connected")
    return result
