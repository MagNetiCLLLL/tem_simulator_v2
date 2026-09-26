"""Opt-in vacuum electrostatics for an explicitly idealised planar cathode.

The cathode is the entire numerical entrance plane at z=0, not the emitter's
nanometre emission patch or its mechanical cone envelope. All coordinates are
metres relative to the original tip; potential is a rise from that cathode in
volts, and E=-grad(Phi) is V/m. This diagnostic is never selected by the gun's
normal field provider. It includes reference annular extractor, gun-lens and
accelerator electrodes in one Laplace solve, with no artificial handoff plane.

Space charge, image charge and insulating materials are absent. Apertures and
magnetic components still require their normal particle transport operations;
this scalar field does not transport particles or certify that full chain.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np
import scipy

from temsim.cpu_resources import numerical_job
from temsim.physics.axisymmetric_cut_field import AxisymmetricCutField


SCHEMA = "diagnostic-planar-cathode-annular-gun-v1"
MAX_VERTICES = 1_500_000


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def request_digest(request):
    """Identity of the complete consumed static-field request."""
    return hashlib.sha256(_json_bytes(request)).hexdigest()


def _implementation_hash():
    paths = [Path(__file__), Path(inspect.getfile(AxisymmetricCutField)),
             Path(inspect.getfile(inspect.unwrap(numerical_job)))]
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def planar_field_request(gun, *, cathode_boundary, cells_per_bore=8,
                         outer_factor=2., exit_extension_mm=0.):
    """Capture electrode inputs after explicit admission of the planar model.

    Source current, emission samples and energy spread do not enter a vacuum
    Laplace solve. Analytic soft edges, fitted amplitudes and field offsets are
    likewise not electrode geometry. The cache intentionally excludes them.
    """
    if cathode_boundary != "planar_equipotential":
        raise ValueError("Explicit cathode_boundary='planar_equipotential' is required")
    emitter = gun.emitter
    if (getattr(emitter, "surface_model", None) is not None
            or float(getattr(emitter, "curvature_nm_inv", 0.)) != 0.):
        raise ValueError("The planar diagnostic requires a flat source; curved geometry cannot be omitted")
    if (getattr(emitter, "coherence", None) is not None
            or getattr(gun, "source_representation", "classical_particles") != "classical_particles"):
        raise ValueError("The planar diagnostic supports classical tip emission only; coherent work is paused")
    if gun.monochromator_installed:
        raise ValueError("An installed monochromator cannot be bypassed by the axisymmetric planar diagnostic")
    if type(cells_per_bore) is not int or not 4 <= cells_per_bore <= 64:
        raise ValueError("cells_per_bore must be an integer in [4, 64]")
    outer_factor, exit_extension_mm = float(outer_factor), float(exit_extension_mm)
    if not math.isfinite(outer_factor) or outer_factor <= 1.:
        raise ValueError("outer_factor must be finite and exceed one")
    if not math.isfinite(exit_extension_mm) or exit_extension_mm < 0.:
        raise ValueError("exit_extension_mm must be finite and non-negative")
    ht = float(gun.accelerator.high_tension_kv)*1000.
    ext = float(gun.extractor.voltage_kv)*1000.
    gun_exit = float(gun.exit_plane_z_mm)*1e-3
    end = gun_exit + exit_extension_mm*1e-3
    if not all(map(math.isfinite, (ht, ext, gun_exit, end))) or not 0 <= ext < ht:
        raise ValueError("Extraction voltage must lie between zero and finite positive high tension")
    if not 0 < gun_exit <= end:
        raise ValueError("The complete gun exit must lie inside the planar diagnostic domain")
    rings = []

    def add_ring(key, center_mm, thickness_mm, inner_diameter_mm, outer_diameter_mm, potential):
        center, half = float(center_mm)*1e-3, float(thickness_mm)*.5e-3
        inner, outer = float(inner_diameter_mm)*.5e-3, float(outer_diameter_mm)*.5e-3
        potential = float(potential)
        if (not all(map(math.isfinite, (center, half, inner, outer, potential)))
                or not 0 < inner < outer or half <= 0):
            raise ValueError(f"Invalid annular electrode geometry or potential: {key}")
        start, stop = center-half, center+half
        if not 0 < start < stop < end:
            raise ValueError(f"All complete electrodes must lie between cathode and exit: {key}")
        rings.append({"key": key, "start_m": start, "stop_m": stop,
                      "inner_m": inner, "outer_m": outer,
                      "potential_rise_v": potential})

    lens = gun.electrostatic_lens
    lens_v = lens.potential_rise_from_tip_v(ext/1000., ht/1000.)
    for key, component, potential in (("extractor", gun.extractor, ext),
                                     ("electrostatic_lens", lens, lens_v)):
        add_ring(key, component.mechanical_center_from_tip_mm,
                 component.mechanical_length_mm,
                 component.mechanical_clear_bore_diameter_mm,
                 component.mechanical_outer_diameter_mm, potential)
    accelerator = gun.accelerator
    thickness = getattr(accelerator, "_electrode_thickness_mm", None)
    if thickness is None:
        raise ValueError("Actual accelerator electrode thickness is required")
    previous_center, previous_fraction = -math.inf, 0.
    for index, stage in enumerate(accelerator.stages):
        center, fraction = float(stage.center_from_tip_mm), float(stage.voltage_fraction)
        if (not math.isfinite(center) or not math.isfinite(fraction)
                or center <= previous_center or not previous_fraction < fraction <= 1.):
            raise ValueError("Accelerator positions and voltage fractions must increase strictly")
        add_ring(f"accelerator:{index}", center, thickness,
                 accelerator.mechanical_clear_bore_diameter_mm,
                 accelerator.mechanical_outer_diameter_mm, ext+fraction*(ht-ext))
        previous_center, previous_fraction = center, fraction
    if previous_fraction != 1.:
        raise ValueError("The final accelerator stage must have voltage fraction exactly one")
    for index, left in enumerate(rings):
        for right in rings[index+1:]:
            if (max(left["start_m"], right["start_m"]) <= min(left["stop_m"], right["stop_m"])
                    and max(left["inner_m"], right["inner_m"]) <= min(left["outer_m"], right["outer_m"])):
                raise ValueError("Overlapping or touching annular electrodes")
    outer_radius = max(row["outer_m"] for row in rings)*outer_factor
    if not math.isfinite(outer_radius):
        raise ValueError("The radial domain must be finite")
    return {
        "schema": SCHEMA,
        "potential_reference": "rise_relative_to_original_tip_v",
        "source_admission": "flat_classical_tip_only",
        "rings": rings, "high_tension_v": ht, "extraction_v": ext,
        "domain": {"entrance_m": 0., "exit_m": end,
                   "gun_exit_m": gun_exit, "outer_radius_m": outer_radius},
        "boundary_conditions": {
            "cathode": {"type": cathode_boundary, "rise_v": 0.,
                        "extent": "entire_radial_domain_at_z_zero"},
            "exit": {"type": "uniform_dirichlet", "rise_v": ht},
            "radial_outer": "natural_zero_normal_derivative",
            "axis": "axisymmetric_regular", "metal": "annular_dirichlet"},
        "limitations": [
            "The idealised full-domain planar cathode is not the emitter's nanometre emission patch or cone envelope.",
            "Reference annuli use component bore/outer dimensions and thickness, not independently verified electrode contours.",
            "No space charge, image charge, insulator or tunnelling-current calculation.",
            "Aperture interception and magnetic transport remain separate required particle operations.",
            "A small linear residual does not certify mesh or boundary convergence or a complete gun."],
        "numerics": {"cells_per_bore": cells_per_bore,
                     "outer_radius_factor": outer_factor,
                     "exit_extension_mm": exit_extension_mm,
                     "mesh": "electrode-local-axial-radial-feature-graded-v1",
                     "interpolation": "bilinear-radius-squared-and-z-v1",
                     "linear_residual_tolerance": 1e-9,
                     "maximum_vertices": MAX_VERTICES},
        "implementation_sha256": _implementation_hash(),
        "libraries": {"numpy": np.__version__, "scipy": scipy.__version__},
    }


def _axis_nodes(boundaries, bands, coarse_step, limit):
    """Exact feature boundaries with locally bounded, graded interval widths."""
    boundaries = sorted(set(float(v) for v in boundaries))
    nodes = [boundaries[0]]
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        midpoint = .5*(left+right)
        active = [step for start, stop, step in bands if start <= midpoint <= stop]
        if active:
            count = max(1, math.ceil((right-left)/min(active)))
            if len(nodes)+count > limit:
                raise ValueError("Planar diagnostic mesh exceeds its vertex budget")
            nodes.extend(np.linspace(left, right, count+1)[1:])
        else:
            position = left
            while position < right:
                distance = min(position-left, right-position)
                width = coarse_step + .2*distance
                following = right if right-position <= 1.25*width else position+width
                if following <= position:
                    raise ValueError("Unresolvable planar diagnostic mesh interval")
                nodes.append(following)
                position = following
                if len(nodes) > limit:
                    raise ValueError("Planar diagnostic mesh exceeds its vertex budget")
    return np.asarray(nodes, dtype=float)


def mesh_axes(request):
    """Resolve each bore and electrode face, with no nanometre emission mesh."""
    rings, numerics, domain = request["rings"], request["numerics"], request["domain"]
    n = numerics["cells_per_bore"]
    maximum = min(MAX_VERTICES, int(numerics["maximum_vertices"]))
    start, end, rmax = domain["entrance_m"], domain["exit_m"], domain["outer_radius_m"]
    radial_boundaries, axial_boundaries = {0., rmax}, {start, end}
    radial_bands, axial_bands = [], []
    for ring in rings:
        inner, outer = ring["inner_m"], ring["outer_m"]
        step = inner/n
        for radius in (inner, outer):
            left, right = max(0., radius-inner), min(rmax, radius+inner)
            radial_boundaries.update((left, radius, right))
            radial_bands.append((left, right, step))
        left, right = max(start, ring["start_m"]-4*inner), min(end, ring["stop_m"]+4*inner)
        axial_boundaries.update((left, ring["start_m"], ring["stop_m"], right))
        axial_bands.append((left, right, step))
    coarse_step = max(row["inner_m"] for row in rings)/n
    r = _axis_nodes(radial_boundaries, radial_bands, coarse_step, maximum//3)
    z = _axis_nodes(axial_boundaries, axial_bands, coarse_step, maximum//len(r))
    if len(r)*len(z) > maximum:
        raise ValueError(f"Planar diagnostic grid exceeds vertex budget: {len(r)} x {len(z)}")
    if len(r) < 3 or len(z) < 3 or np.any(np.diff(r) <= 0) or np.any(np.diff(z) <= 0):
        raise ValueError("Invalid planar diagnostic mesh")
    return r, z


class PlanarGunField:
    """Bilinear scalar potential in (r squared, z); E is its exact gradient.

    This class also accepts independently prepared arrays for small analytic
    validation fixtures. Building an instrument field requires explicit source
    and boundary admission through :func:`build_planar_gun_field`.
    """

    def __init__(self, request, r, z, voltage, report=None):
        self.request = json.loads(_json_bytes(request))
        self.r, self.z = np.array(r, dtype=float, copy=True), np.array(z, dtype=float, copy=True)
        self.voltage = np.array(voltage, dtype=float, copy=True)
        if (self.r.ndim != 1 or self.z.ndim != 1 or len(self.r) < 2 or len(self.z) < 2
                or self.r[0] != 0 or np.any(np.diff(self.r) <= 0)
                or np.any(np.diff(self.z) <= 0) or self.voltage.shape != (len(self.r), len(self.z))
                or not all(np.isfinite(a).all() for a in (self.r, self.z, self.voltage))):
            raise ValueError("Invalid planar diagnostic field arrays")
        self.report = json.loads(_json_bytes(report or {}))
        self.report.update({"schema": request.get("schema", SCHEMA),
                            "potential_reference": "rise_relative_to_original_tip_v",
                            "grid_shape": [len(self.r), len(self.z)]})
        for array in (self.r, self.z, self.voltage):
            array.setflags(write=False)

    def interpolate(self, positions):
        """Return (potential rise V, E V/m) at finite xyz positions in metres."""
        p = np.asarray(positions, dtype=float)
        if p.shape[-1:] != (3,) or not np.isfinite(p).all():
            raise ValueError("Field positions must be finite xyz metres")
        shape = p.shape[:-1]
        q = p.reshape(-1, 3)
        if len(q) == 1:
            # A virtual electron makes many individual queries. Keep exactly
            # the same bilinear (r²,z) field and boundary convention without
            # allocating array indices and masks for a one-point batch.
            x, y, z = q[0]
            radius = np.hypot(x, y)
            if radius > self.r[-1] or z < self.z[0] or z > self.z[-1]:
                raise ValueError("Position outside planar diagnostic field; no extrapolation")
            i = min(max(int(np.searchsorted(self.r, radius, side="right"))-1, 0), len(self.r)-2)
            j = min(max(int(np.searchsorted(self.z, z, side="right"))-1, 0), len(self.z)-2)
            # Array ``**2`` uses multiplication. Scalar NumPy ``**2`` may
            # call pow and differ by an ulp, so retain the array arithmetic.
            lower_square = self.r[i]*self.r[i]
            ds, dz = self.r[i+1]*self.r[i+1]-lower_square, self.z[j+1]-self.z[j]
            u, v = (radius*radius-lower_square)/ds, (z-self.z[j])/dz
            a, b = self.voltage[i, j], self.voltage[i+1, j]
            c, d = self.voltage[i, j+1], self.voltage[i+1, j+1]
            bottom, top = a+u*(b-a), c+u*(d-c)
            potential = bottom+v*(top-bottom)
            derivative_s = ((1-v)*(b-a)+v*(d-c))/ds
            electric = np.array((-2*x*derivative_s, -2*y*derivative_s, -(top-bottom)/dz))
            return np.asarray(potential).reshape(shape), electric.reshape(p.shape)
        radius, z = np.hypot(q[:, 0], q[:, 1]), q[:, 2]
        if np.any(radius > self.r[-1]) or np.any(z < self.z[0]) or np.any(z > self.z[-1]):
            raise ValueError("Position outside planar diagnostic field; no extrapolation")
        i = np.clip(np.searchsorted(self.r, radius, side="right")-1, 0, len(self.r)-2)
        j = np.clip(np.searchsorted(self.z, z, side="right")-1, 0, len(self.z)-2)
        ds, dz = self.r[i+1]**2-self.r[i]**2, self.z[j+1]-self.z[j]
        u, v = (radius**2-self.r[i]**2)/ds, (z-self.z[j])/dz
        a, b = self.voltage[i, j], self.voltage[i+1, j]
        c, d = self.voltage[i, j+1], self.voltage[i+1, j+1]
        bottom, top = a+u*(b-a), c+u*(d-c)
        potential = bottom+v*(top-bottom)
        derivative_s = ((1-v)*(b-a)+v*(d-c))/ds
        electric = np.empty_like(q)
        electric[:, :2] = -2*q[:, :2]*derivative_s[:, None]
        electric[:, 2] = -(top-bottom)/dz
        return potential.reshape(shape), electric.reshape(p.shape)

    def potential_v_at_global_positions(self, positions):
        """Potential rise relative to the tip, not a ground-referenced voltage."""
        return self.interpolate(positions)[0]

    def potential_rise_v_at_global_positions(self, positions):
        return self.interpolate(positions)[0]

    def field_at_global_positions_v_per_m(self, positions):
        return self.interpolate(positions)[1]


def _array_checksum(r, z, voltage):
    digest = hashlib.sha256()
    for name, array in (("r", r), ("z", z), ("voltage", voltage)):
        array = np.ascontiguousarray(array, dtype="<f8")
        digest.update(_json_bytes({"name": name, "shape": list(array.shape), "dtype": "<f8"}))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _cache_paths(request, cache_dir):
    directory, key = Path(cache_dir), request_digest(request)
    return directory/(key+".npz"), directory/(key+".json")


def save_cached_field(field, cache_dir):
    """Save exact float64 field arrays plus a checked dependency manifest."""
    data_path, manifest_path = _cache_paths(field.request, cache_dir)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_data = temporary_manifest = None
    try:
        with tempfile.NamedTemporaryFile(dir=data_path.parent, suffix=".npz.tmp", delete=False) as stream:
            temporary_data = Path(stream.name)
            np.savez_compressed(stream, r=field.r, z=field.z, voltage=field.voltage)
        manifest = {
            "request": field.request, "request_sha256": request_digest(field.request),
            "array_sha256": _array_checksum(field.r, field.z, field.voltage),
            "archive_sha256": hashlib.sha256(temporary_data.read_bytes()).hexdigest(),
            "report": field.report, "saved_utc": datetime.now(timezone.utc).isoformat()}
        with tempfile.NamedTemporaryFile(dir=data_path.parent, suffix=".json.tmp", delete=False) as stream:
            temporary_manifest = Path(stream.name)
            stream.write(_json_bytes(manifest))
        temporary_data.replace(data_path)
        temporary_manifest.replace(manifest_path)
    finally:
        for path in (temporary_data, temporary_manifest):
            if path is not None:
                path.unlink(missing_ok=True)
    field.report["cache_path"] = str(data_path.resolve())
    field.report["cache_bytes"] = data_path.stat().st_size+manifest_path.stat().st_size
    return data_path


def load_cached_field(request, cache_dir):
    """Return a lossless field hit, None on absence, or reject corrupt data."""
    started = time.perf_counter()
    data_path, manifest_path = _cache_paths(request, cache_dir)
    if not data_path.exists() and not manifest_path.exists():
        return None
    if not data_path.exists() or not manifest_path.exists():
        raise ValueError("Incomplete planar diagnostic field cache")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("report"), dict):
        raise ValueError("Invalid planar diagnostic field cache manifest")
    if manifest.get("request") != request or manifest.get("request_sha256") != request_digest(request):
        raise ValueError("Planar diagnostic field cache request mismatch")
    if manifest.get("archive_sha256") != hashlib.sha256(data_path.read_bytes()).hexdigest():
        raise ValueError("Planar diagnostic field archive checksum mismatch")
    with np.load(data_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"r", "z", "voltage"}:
            raise ValueError("Unexpected planar diagnostic field archive members")
        r, z, voltage = arrays["r"], arrays["z"], arrays["voltage"]
    if manifest.get("array_sha256") != _array_checksum(r, z, voltage):
        raise ValueError("Planar diagnostic field array checksum mismatch")
    field = PlanarGunField(request, r, z, voltage, manifest.get("report"))
    field.report.update({"cache_hit": True, "load_seconds": time.perf_counter()-started,
                         "cache_path": str(data_path.resolve()),
                         "cache_bytes": data_path.stat().st_size+manifest_path.stat().st_size})
    return field


def build_planar_gun_field(gun, *, cathode_boundary, cells_per_bore=8,
                           outer_factor=2., exit_extension_mm=0., cache_dir=None):
    """Build the admitted diagnostic field; persistent caching is opt-in.

    The single-thread numerical-job context also serializes field builds with
    other in-process numerical workers. No particle count or source emission
    distribution is consumed because charge does not feed back on this field.
    """
    request = planar_field_request(gun, cathode_boundary=cathode_boundary,
        cells_per_bore=cells_per_bore, outer_factor=outer_factor,
        exit_extension_mm=exit_extension_mm)
    with numerical_job(requested=1) as cpu:
        if cache_dir is not None:
            cached = load_cached_field(request, cache_dir)
            if cached is not None:
                return cached
        started = time.perf_counter()
        r, z = mesh_axes(request)
        rr, zz = np.meshgrid(r, z, indexing="ij")
        fixed = np.zeros(rr.shape, dtype=bool)
        voltage = np.zeros(rr.shape)
        fixed[:, 0], fixed[:, -1] = True, True
        voltage[:, -1] = request["high_tension_v"]
        for ring in request["rings"]:
            metal = ((zz >= ring["start_m"]) & (zz <= ring["stop_m"])
                     & (rr >= ring["inner_m"]) & (rr <= ring["outer_m"]))
            if not np.any(metal) or np.any(metal & fixed):
                raise ValueError("Unresolved or overlapping planar diagnostic electrode")
            fixed |= metal
            voltage[metal] = ring["potential_rise_v"]
        solved = AxisymmetricCutField(r, z, fixed, voltage, geometry=None,
            tolerance=request["numerics"]["linear_residual_tolerance"])
        report = {"cache_hit": False, "load_seconds": 0.,
                  "solve_seconds": time.perf_counter()-started,
                  "linear_residual": float(solved.residual),
                  "elements": int(solved.element_count), "vertices": int(solved.vertex_count),
                  "minimum_radial_cell_m": float(np.min(np.diff(r))),
                  "minimum_axial_cell_m": float(np.min(np.diff(z))),
                  "maximum_axial_cell_m": float(np.max(np.diff(z))),
                  "request_sha256": request_digest(request),
                  "cpu_resources": cpu.to_dict(),
                  "limitations": request["limitations"],
                  "boundary_conditions": request["boundary_conditions"]}
        field = PlanarGunField(request, r, z, solved.nodal_voltage, report)
        if cache_dir is not None:
            save_cached_field(field, cache_dir)
        return field
