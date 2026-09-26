"""Joint electrode field for unchanged continuous-curvature tip emission.

The existing source coordinates and local kinetic energies are never moved or
replaced. The curved conductor is represented by conforming linear elements;
potential at its exact analytical surface therefore has a discretisation error
which is reported rather than forced to zero. Electric force differentiates
that same represented scalar potential, including the regular axial core.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from collections import OrderedDict
from pathlib import Path
from threading import RLock
import tempfile
import time

import numpy as np

from temsim.cpu_resources import numerical_job
from temsim.physics.axisymmetric_cut_field import AxisymmetricCutField
from temsim.physics.axis_regular_potential import AxisRegularPotential
from temsim.physics.closed_gun_field import closed_field_request, mesh_axes as closed_mesh_axes
from temsim.physics.continuous_curvature_conductor import ContinuousCurvatureConductor
from temsim.physics.grounded_tip_field import merge_axis_nodes
from temsim.physics.planar_gun_field import MAX_VERTICES, request_digest


SCHEMA = "continuous-curvature-closed-gun-field-v1"
_MEMORY_FIELDS = OrderedDict()
_MEMORY_LOCK = RLock()
_FEM_ARRAYS = ("r", "z", "lookup", "cut_cells", "nodal_voltage", "origin", "inverse",
               "gradient", "phi0", "boundary_s", "boundary_z")


def continuous_field_request(gun, *, liner_segments=None, cells_per_bore=None,
                             outer_factor=2.0, exit_extension_mm=None,
                             apex_cells_per_radius=32, tip_nodes=128,
                             axis_core_fraction=0.01):
    request = closed_field_request(gun, liner_segments=liner_segments,
        cells_per_bore=cells_per_bore, outer_factor=outer_factor,
        exit_extension_mm=exit_extension_mm)
    if request.get("source_admission") != "continuous_curvature_classical_tip":
        raise ValueError("Continuous gun electrostatics requires the existing nonzero-curvature emitter")
    if (type(apex_cells_per_radius) is not int or not 8 <= apex_cells_per_radius <= 256
            or type(tip_nodes) is not int or not 32 <= tip_nodes <= 512
            or not np.isfinite(axis_core_fraction) or not 0 <= axis_core_fraction <= 0.05):
        raise ValueError("Invalid continuous gun mesh or axial regularisation budget")
    geometry = ContinuousCurvatureConductor(**request["cathode_geometry"]).validate()
    request.update(schema=SCHEMA, model_status="continuous_curvature_conductor_with_grounded_outlet")
    request["numerics"].update(apex_cells_per_radius=apex_cells_per_radius,
        tip_nodes=tip_nodes, axis_core_fraction=float(axis_core_fraction),
        interpolation="conforming_cut_cells_with_same_potential_regular_axis")
    request["limitations"].append(
        "Original analytical emission positions are unchanged; nonzero launch potential from the chordal conductor representation is a mesh error, not an energy reset.")
    request["boundary_conditions"]["cathode_domain_intersection"] = (
        "Conductor portions intersecting the back or radial numerical boundary remain at cathode potential; the numerical domain need not enclose the whole shank.")
    paths = [Path(__file__), Path(inspect.getfile(ContinuousCurvatureConductor)),
             Path(inspect.getfile(AxisymmetricCutField)), Path(inspect.getfile(AxisRegularPotential))]
    from temsim.physics import axis_field_interpolation
    paths.append(Path(axis_field_interpolation.__file__))
    request["implementation_sha256"].update(
        {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    return request


def mesh_axes(request):
    geometry = ContinuousCurvatureConductor(**request["cathode_geometry"]).validate()
    r, z = closed_mesh_axes(request)
    radius = geometry.apex_radius_nm * 1e-9
    support = geometry.emission_support_radius_nm * 1e-9
    minimum = min(radius, support if support > 0 else radius) / request["numerics"]["apex_cells_per_radius"]
    count = request["numerics"]["tip_nodes"]

    def geometric_axis(end, nodes):
        if end <= minimum:
            return np.linspace(0.0, end, max(3, nodes))
        return np.r_[0.0, np.geomspace(minimum, end, nodes - 1)]

    r = merge_axis_nodes(r, geometric_axis(r[-1], count), boundaries=[0.0, r[-1]])
    z = merge_axis_nodes(z, np.r_[-geometric_axis(-geometry.back_z_m, max(32, count // 2))[::-1],
                                   geometric_axis(z[-1], count)],
                         boundaries=[geometry.back_z_m, 0.0, z[-1]])
    if len(r) * len(z) > MAX_VERTICES:
        raise ValueError("Continuous gun mesh exceeds its vertex budget")
    return r, z


def _boundary_arrays(request, r, z, geometry):
    rr, zz = np.meshgrid(r, z, indexing="ij")
    fixed = (zz <= 0.0) & (rr <= geometry.radius_m(zz))
    values = np.zeros(rr.shape)
    fixed[:, -1] = True
    values[:, -1] = request["high_tension_v"]
    for row in request["rings"]:
        metal = ((zz >= row["start_m"]) & (zz <= row["stop_m"])
                 & (rr >= row["inner_m"]) & (rr <= row["outer_m"]))
        if not metal.any() or np.any(metal & fixed & (values != row["potential_rise_v"])):
            raise ValueError("Unresolved or electrically conflicting continuous-gun conductor")
        fixed |= metal
        values[metal] = row["potential_rise_v"]
    return fixed, values


class ContinuousGunField:
    def __init__(self, request, fem, report):
        self.request = json.loads(json.dumps(request, allow_nan=False))
        self.geometry = ContinuousCurvatureConductor(**request["cathode_geometry"]).validate()
        self._fem = fem
        self.r, self.z, self.voltage = fem.r, fem.z, fem.nodal_voltage
        self._regular = AxisRegularPotential(fem,
            bore_radius_m=min(row["inner_m"] for row in request["rings"]),
            fraction=request["numerics"]["axis_core_fraction"])
        self.report = dict(report)
        self.report.update(request_sha256=request_digest(request), schema=SCHEMA,
                           potential_reference="rise_relative_to_original_tip_v",
                           grid_shape=[len(self.r), len(self.z)],
                           launch_positions="unchanged_analytical_emission_surface")

    @property
    def domain(self):
        return dict(self.request["domain"])

    @property
    def support_bounds_m(self):
        return float(self.z[0]), float(self.z[-1])

    def interpolate(self, positions):
        points = np.asarray(positions, dtype=float)
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError("Continuous gun positions must be finite xyz metres")
        shape = points.shape
        potential, electric = self._regular.interpolate(points.reshape(-1, 3))
        return potential.reshape(shape[:-1]), electric.reshape(shape)

    def potential_v_at_global_positions(self, positions):
        return self.interpolate(positions)[0]

    potential_rise_v_at_global_positions = potential_v_at_global_positions

    def field_at_global_positions_v_per_m(self, positions):
        return self.interpolate(positions)[1]

    def tip_material_mask(self, positions):
        points = np.asarray(positions, dtype=float)
        radius = np.hypot(points[..., 0], points[..., 1])
        surface = self.geometry.surface_z_m(radius)
        # Original points exactly on the analytic cap are launches, not returns.
        tolerance = 16 * np.spacing(np.maximum(np.abs(surface), np.abs(points[..., 2])))
        return (points[..., 2] >= self.geometry.back_z_m) & (points[..., 2] < surface - tolerance)

    def launch_boundary_report(self, positions):
        points = np.asarray(positions, dtype=float)
        radius = np.hypot(points[:, 0], points[:, 1])
        represented = self._fem.surface_z(radius)
        potential = self.potential_v_at_global_positions(points)
        return {"maximum_surface_representation_error_m": float(np.max(np.abs(points[:, 2] - represented), initial=0)),
                "maximum_launch_potential_error_v": float(np.max(np.abs(potential), initial=0)),
                "launch_positions_changed": False}


def _cache_paths(request, directory):
    path = Path(directory) / request_digest(request)
    return path.with_suffix(".npz"), path.with_suffix(".json")


def _load_cached(request, directory):
    archive, manifest = _cache_paths(request, directory)
    if not archive.exists() and not manifest.exists():
        return None
    if not archive.exists() or not manifest.exists():
        raise ValueError("Incomplete continuous gun field cache")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("request") != request or data.get("archive_sha256") != hashlib.sha256(archive.read_bytes()).hexdigest():
        raise ValueError("Continuous gun field cache identity or checksum mismatch")
    fem = object.__new__(AxisymmetricCutField)
    with np.load(archive, allow_pickle=False) as arrays:
        if set(arrays.files) != set(_FEM_ARRAYS):
            raise ValueError("Unexpected continuous gun field cache arrays")
        for key in _FEM_ARRAYS:
            value = arrays[key]
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite continuous gun field cache")
            value.setflags(write=False)
            setattr(fem, key, value)
    for key in ("element_count", "vertex_count", "residual"):
        setattr(fem, key, data["fem"][key])
    return ContinuousGunField(request, fem, {**data["report"], "cache_hit": True})


def _save_cached(field, directory):
    archive, manifest = _cache_paths(field.request, directory)
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary_data = temporary_manifest = None
    try:
        with tempfile.NamedTemporaryFile(dir=archive.parent, suffix=".npz.tmp", delete=False) as stream:
            temporary_data = Path(stream.name)
            np.savez_compressed(stream, **{key: getattr(field._fem, key) for key in _FEM_ARRAYS})
        payload = {"request": field.request, "report": field.report,
                   "archive_sha256": hashlib.sha256(temporary_data.read_bytes()).hexdigest(),
                   "fem": {key: getattr(field._fem, key) for key in ("element_count", "vertex_count", "residual")}}
        with tempfile.NamedTemporaryFile(dir=archive.parent, suffix=".json.tmp", delete=False) as stream:
            temporary_manifest = Path(stream.name)
            stream.write(json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8"))
        temporary_data.replace(archive)
        temporary_manifest.replace(manifest)
    finally:
        for path in (temporary_data, temporary_manifest):
            if path is not None:
                path.unlink(missing_ok=True)


def build_continuous_gun_field(gun, *, cache_dir=None, **options):
    request = continuous_field_request(gun, **options)
    with numerical_job(requested=1) as cpu:
        if cache_dir is not None:
            cached = _load_cached(request, cache_dir)
            if cached is not None:
                return cached
        started = time.perf_counter()
        geometry = ContinuousCurvatureConductor(**request["cathode_geometry"]).validate()
        r, z = mesh_axes(request)
        fixed, values = _boundary_arrays(request, r, z, geometry)
        fem = AxisymmetricCutField(r, z, fixed, values, geometry=geometry,
            tolerance=request["numerics"]["linear_residual_tolerance"])
        field = ContinuousGunField(request, fem, {
            "cache_hit": False, "solve_seconds": time.perf_counter() - started,
            "linear_residual": float(fem.residual), "elements": int(fem.element_count),
            "vertices": int(fem.vertex_count), "cpu_resources": cpu.to_dict(),
            "boundary_conditions": request["boundary_conditions"], "limitations": request["limitations"]})
        if cache_dir is not None:
            _save_cached(field, cache_dir)
        return field


def continuous_field(gun):
    request = continuous_field_request(gun)
    digest = request_digest(request)
    cached = getattr(gun, "_continuous_gun_field", None)
    if cached is not None and cached.report.get("request_sha256") == digest:
        return cached
    with _MEMORY_LOCK:
        field = _MEMORY_FIELDS.get(digest)
        if field is None:
            from temsim.physics.closed_gun_field import field_cache_directory, _trim_disk_cache
            directory = field_cache_directory() / "continuous"
            field = build_continuous_gun_field(gun, cache_dir=directory)
            _MEMORY_FIELDS[digest] = field
            _trim_disk_cache(directory, digest)
        _MEMORY_FIELDS.move_to_end(digest)
        while len(_MEMORY_FIELDS) > 4:
            _MEMORY_FIELDS.popitem(last=False)
        gun._continuous_gun_field = field
        return field
