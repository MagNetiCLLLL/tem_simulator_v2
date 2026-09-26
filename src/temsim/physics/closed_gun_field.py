"""Electrically closed scalar field for the classical electron gun.

All voltages are rises from the original cathode.  The simulator explicitly
grounds its existing vacuum liner downstream of the accelerator envelope,
including the connected column tube.  This is a model electrical assignment,
not a measured instrument connection.  Magnetic forces and interception stay
with their ordinary transport components; this module does not launch rays.

The numerical downstream plane cuts a continuing, physical grounded tube.
Moving that numerical plane never moves the stored physical liner endpoints.
No potential/energy reset or analytic-field extrapolation is performed.
"""
from __future__ import annotations

import hashlib
import inspect
import math
import os
from collections import OrderedDict
from pathlib import Path
from threading import RLock
import time
from types import SimpleNamespace

import numpy as np

from temsim.cpu_resources import numerical_job
from temsim.physics.axisymmetric_cut_field import AxisymmetricCutField
from temsim.physics.grounded_tip_field import merge_axis_nodes
from temsim.physics.planar_gun_field import (
    MAX_VERTICES, PlanarGunField, load_cached_field,
    _axis_nodes, planar_field_request, request_digest,
    save_cached_field,
)


SCHEMA = "closed-classical-gun-electrostatics-v1"
DEFAULT_CELLS_PER_BORE = 8
DEFAULT_EXIT_EXTENSION_MM = 100.
MAX_MEMORY_FIELDS = 4
MAX_DISK_CACHE_BYTES = 256*1024*1024
_MEMORY_FIELDS = OrderedDict()
_MEMORY_LOCK = RLock()


def _value(row, name):
    return row[name] if isinstance(row, dict) else getattr(row, name)


def _physical_liner_rows(gun, rows):
    """Capture the contiguous physical liner; do not truncate at solve exit."""
    accelerator = gun.accelerator
    start = (float(accelerator.mechanical_center_from_tip_mm)
             + .5*float(accelerator.mechanical_length_mm))
    if not math.isfinite(start) or start <= 0:
        raise ValueError("The grounded outlet needs a finite accelerator mechanical end")
    if rows is None:
        rows = getattr(gun, "_grounded_outlet_liner_segments", None)
    if rows is None:
        raise ValueError("Resolved physical outlet liner geometry is required for closed gun electrostatics")
    parsed = []
    for index, row in enumerate(rows):
        a, b, inner, outer = (float(_value(row, name)) for name in (
            "start_z_mm", "end_z_mm", "inner_diameter_mm", "outer_diameter_mm"))
        if not all(map(math.isfinite, (a, b, inner, outer))) or not a < b or not 0 < inner < outer:
            raise ValueError("Invalid physical grounded outlet liner geometry")
        if b <= start:
            continue
        parsed.append({"key": f"grounded_outlet:{index}",
            "start_m": max(start, a)*1e-3, "stop_m": b*1e-3,
            "inner_m": inner*.5e-3, "outer_m": outer*.5e-3})
    parsed.sort(key=lambda row: row["start_m"])
    if not parsed or not math.isclose(parsed[0]["start_m"], start*1e-3, abs_tol=1e-14):
        raise ValueError("Grounded liner must start at the accelerator mechanical end")
    for left, right in zip(parsed[:-1], parsed[1:]):
        if not math.isclose(left["stop_m"], right["start_m"], rel_tol=0., abs_tol=1e-14):
            raise ValueError("Grounded outlet liner must be contiguous and non-overlapping")
        # Preserve an exact shared face despite mm-to-m roundoff.
        right["start_m"] = left["stop_m"]
    return parsed


def closed_field_request(gun, *, liner_segments=None,
                         cells_per_bore=None,
                         outer_factor=2., exit_extension_mm=None):
    """Build a cheap, complete static-field identity from resolved mechanics.

    An installed monochromator is admitted only as part of the ordinary gun:
    its electric and magnetic providers must still be composed by that gun.
    Emission current/count/angles do not affect this charge-free field cache.
    """
    if liner_segments is None:
        from temsim.physics.gun_field_environment import ensure_gun_field_environment
        ensure_gun_field_environment(gun)
    if cells_per_bore is None:
        cells_per_bore = getattr(gun, "_gun_field_cells_per_bore", DEFAULT_CELLS_PER_BORE)
    if exit_extension_mm is None:
        exit_extension_mm = getattr(gun, "_gun_field_exit_extension_mm", DEFAULT_EXIT_EXTENSION_MM)
    # Reuse geometry validation without treating the monochromator's independent
    # transverse fields as axisymmetric annular electrodes.  They are not part
    # of this scalar base field and remain required in the composed provider.
    conductor = None
    source_admission = gun.emitter
    if (getattr(gun.emitter, "surface_model", None) is None
            and float(getattr(gun.emitter, "curvature_nm_inv", 0.)) != 0.):
        from temsim.physics.continuous_curvature_conductor import continuous_curvature_conductor
        conductor = continuous_curvature_conductor(gun.emitter)
        # This namespace admits the shared electrode validator, not a physical
        # source. The true curved conductor replaces the cathode boundary below;
        # flat-field construction explicitly rejects that resulting request.
        source_admission = SimpleNamespace(surface_model=None, curvature_nm_inv=0.,
            coherence=getattr(gun.emitter, "coherence", None))
    base = SimpleNamespace(emitter=source_admission, extractor=gun.extractor,
        electrostatic_lens=gun.electrostatic_lens, accelerator=gun.accelerator,
        exit_plane_z_mm=gun.exit_plane_z_mm, monochromator_installed=False,
        source_representation=getattr(gun, "source_representation", "classical_particles"))
    request = planar_field_request(base, cathode_boundary="planar_equipotential",
        cells_per_bore=cells_per_bore, outer_factor=outer_factor,
        exit_extension_mm=exit_extension_mm)
    if bool(getattr(gun, "monochromator_installed", False)):
        if getattr(gun, "_wien_housing_voltage_reference", None) != "gun_lens":
            raise ValueError("Installed velocity selector needs an explicit gun-lens-referenced housing boundary")
        housing = gun.monochromator.wien
        center = float(housing.mechanical_center_from_tip_mm)*1e-3
        half = float(housing.mechanical_length_mm)*.5e-3
        inner = float(housing.mechanical_clear_bore_diameter_mm)*.5e-3
        outer = float(housing.mechanical_outer_diameter_mm)*.5e-3
        if (not all(map(math.isfinite, (center, half, inner, outer)))
                or not 0 < inner < outer or half <= 0
                or not 0 < center-half < center+half < request["domain"]["gun_exit_m"]):
            raise ValueError("Invalid common-potential velocity-selector housing geometry")
        potential = next(row["potential_rise_v"] for row in request["rings"]
                         if row["key"] == "electrostatic_lens")
        request["rings"].append({"key": "wien_common_potential_housing",
            "start_m": center-half, "stop_m": center+half,
            "inner_m": inner, "outer_m": outer, "potential_rise_v": potential})
        request["wien_housing_boundary"] = {
            "voltage_reference": "gun_lens", "potential_rise_v": potential,
            "assignment": "explicit_simulator_electrical_model_design",
            "scope": "axisymmetric_common_bias_shell_only_transverse_Wien_fields_remain_required"}
    physical = _physical_liner_rows(gun, liner_segments)
    end, ht = request["domain"]["exit_m"], request["high_tension_v"]
    if physical[0]["start_m"] >= request["domain"]["gun_exit_m"]:
        raise ValueError("The physical grounded outlet must begin before the gun exit")
    if physical[-1]["stop_m"] < end-1e-14:
        raise ValueError("The numerical field domain must end inside the continuing physical grounded liner")
    grounded = []
    for row in physical:
        if row["start_m"] >= end:
            break
        grounded.append({**row, "stop_m": min(row["stop_m"], end), "potential_rise_v": ht})
    # A stepped, continuous tube includes the radial face between adjacent
    # radii.  These zero-thickness equipotential faces join existing shells;
    # they do not change the electron-accessible bore on either side.
    joints = []
    for index, (left, right) in enumerate(zip(physical[:-1], physical[1:])):
        face = left["stop_m"]
        if face > end:
            break
        if (left["inner_m"], left["outer_m"]) != (right["inner_m"], right["outer_m"]):
            joints.append({"key": f"grounded_outlet_joint:{index}",
                "start_m": face, "stop_m": face,
                "inner_m": min(left["inner_m"], right["inner_m"]),
                "outer_m": max(left["outer_m"], right["outer_m"]),
                "potential_rise_v": ht})
    request["rings"].extend(grounded+joints)
    if max(row["outer_m"] for row in request["rings"]) >= request["domain"]["outer_radius_m"]:
        raise ValueError("Grounded physical liner exceeds the padded radial field domain")
    request.update({"schema": SCHEMA,
        "model_status": "classical_planar_cathode_with_explicit_grounded_outlet",
        "grounded_liner": physical,
        "ground_assignment": {
            "reference": "simulator_electrical_model_design",
            "ground_potential_rise_v": ht,
            "start": "accelerator_mechanical_downstream_face",
            "extent": "connected_resolved_vacuum_liner_through_downstream_column_tube",
            "step_joints": "zero_thickness_grounded_radial_faces",
            "upstream_apertures": "not_assigned_electrical_potentials"},
        "required_composition": "ordinary_apertures_magnetic_components_and_installed_monochromator"})
    request["boundary_conditions"]["exit"]["location"] = "numerical_cut_inside_continuing_grounded_liner"
    request["boundary_conditions"]["grounded_liner"] = "physical_shells_and_radial_step_faces_at_ground"
    request["numerics"]["mesh"] = "electrode-local-and-grounded-liner-face-graded-v1"
    request["numerics"]["grounded_liner_face_refinement_bores"] = 2.
    request["numerics"]["grounded_liner_axial_cells_per_bore"] = cells_per_bore/2
    request["limitations"] = [
        "The flat source uses an idealised full-domain planar cathode, not a nanometre emission patch.",
        "Annular electrode contours use current mechanical bore, thickness and outer dimensions.",
        "Grounding the downstream liner is an explicit simulator electrical design, not measured wiring.",
        "Radial joints between stepped liner shells are idealised zero-thickness grounded faces.",
        "No space charge, image charge, insulators or tunnelling-current prediction.",
        "Magnetic controls, aperture absorption and any installed monochromator remain separate required operations.",
        "Field evaluation is strict inside the solved domain; no analytic fallback or energy reset."]
    request["boundary_conditions"]["cathode"]["material_extension"] = "z_below_zero_is_constant_potential_conductor_for_interception"
    if conductor is not None:
        request["source_admission"] = "continuous_curvature_classical_tip"
        request["model_status"] = "classical_curved_cathode_with_explicit_grounded_outlet"
        request["cathode_geometry"] = conductor.to_dict()
        request["domain"]["entrance_m"] = -conductor.shank_length_um*1e-6
        request["boundary_conditions"]["cathode"] = {
            "type": "finite_curved_conductor", "rise_v": 0.,
            "geometry": conductor.to_dict(), "source_positions": "original_unmodified_emission"}
        request["boundary_conditions"]["back"] = "natural_zero_normal_derivative_outside_conductor"
        request["limitations"][0] = "Continuous curvature uses the original spherical cap and tangent-cone conductor; emitted positions are not moved."
        request["implementation_sha256"]["continuous_curvature_conductor.py"] = hashlib.sha256(
            Path(inspect.getfile(type(conductor))).read_bytes()).hexdigest()
    paths = (Path(__file__), Path(inspect.getfile(merge_axis_nodes)))
    request["implementation_sha256"].update({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    return request


def mesh_axes(request):
    """Keep physical faces exactly while removing numerical node aliases."""
    domain = request["domain"]
    start, end, rmax = domain["entrance_m"], domain["exit_m"], domain["outer_radius_m"]
    n = request["numerics"]["cells_per_bore"]
    radial_boundaries, axial_boundaries = {0., rmax}, {start, end, domain["gun_exit_m"]}
    radial_bands, axial_bands = [], []
    for row in request["rings"]:
        inner, outer = row["inner_m"], row["outer_m"]
        step = inner/n
        for radius in (inner, outer):
            left, right = max(0., radius-inner), min(rmax, radius+inner)
            if row["key"].startswith("grounded_outlet"):
                # Only the vacuum side needs additional shell-face nodes;
                # nodes within the conductor have prescribed constant voltage.
                if radius == inner:
                    right = radius
                else:
                    left = radius
            radial_boundaries.update((left, radius, right))
            radial_bands.append((left, right, step))
        axial_boundaries.update((row["start_m"], row["stop_m"]))
        if row["key"].startswith("grounded_outlet"):
            # A long uniform conductor is refined at its faces, not uniformly
            # along field-free internal drift. Both ends and radial bore remain
            # exact. The free outer field still receives a graded axial grid.
            intervals = [(max(start, face-2*inner), min(end, face+2*inner))
                for face in (row["start_m"], row["stop_m"])]
        else:
            intervals = [(max(start, row["start_m"]-4*inner), min(end, row["stop_m"]+4*inner))]
        for left, right in intervals:
            axial_boundaries.update((left, right))
            axial_step = 2*step if row["key"].startswith("grounded_outlet") else step
            axial_bands.append((left, right, axial_step))
    coarse_step = max(row["inner_m"] for row in request["rings"])/n
    r = _axis_nodes(radial_boundaries, radial_bands, coarse_step, MAX_VERTICES//3)
    z = _axis_nodes(axial_boundaries, axial_bands, coarse_step, MAX_VERTICES//len(r))
    rfaces = [0., domain["outer_radius_m"],
        *[row[name] for row in request["rings"] for name in ("inner_m", "outer_m")]]
    zfaces = [domain["entrance_m"], domain["exit_m"], domain["gun_exit_m"],
        *[row[name] for row in request["rings"] for name in ("start_m", "stop_m")]]
    r, z = merge_axis_nodes(r, boundaries=rfaces), merge_axis_nodes(z, boundaries=zfaces)
    if len(r)*len(z) > MAX_VERTICES:
        raise ValueError("Closed gun field exceeds its vertex budget")
    return r, z


def electrode_boundary_arrays(request, r, z):
    """Dirichlet data for distinct electrodes and connected grounded shells."""
    rr, zz = np.meshgrid(r, z, indexing="ij")
    fixed, values = np.zeros(rr.shape, dtype=bool), np.zeros(rr.shape)
    fixed[:, 0], fixed[:, -1] = True, True
    values[:, -1] = request["high_tension_v"]
    for row in request["rings"]:
        metal = ((zz >= row["start_m"]) & (zz <= row["stop_m"])
                 & (rr >= row["inner_m"]) & (rr <= row["outer_m"]))
        if not metal.any() or np.any(metal & fixed & (values != row["potential_rise_v"])):
            raise ValueError("Unresolved or electrically conflicting closed gun conductor")
        fixed |= metal
        values[metal] = row["potential_rise_v"]
    return fixed, values


class ClosedGunField(PlanarGunField):
    """One scalar potential and its gradient; no downstream field substitution."""

    @property
    def domain(self):
        return dict(self.request["domain"])

    @property
    def support_bounds_m(self):
        return float(self.z[0]), float(self.z[-1])

    def interpolate(self, positions):
        """Evaluate vacuum or the flat cathode's constant-potential interior.

        Returning particles can enter this conductor during an integration
        trial so the trajectory code can locate its physical interception.
        This extension applies only below the cathode; downstream/radial
        vacuum outside the solved domain still raises an error.
        """
        p = np.asarray(positions, dtype=float)
        if p.shape[-1:] != (3,) or not np.isfinite(p).all():
            raise ValueError("Field positions must be finite xyz metres")
        metal = p[..., 2] < 0.
        if not np.any(metal):
            return super().interpolate(p)
        q = p.copy()
        q[..., 2] = np.maximum(q[..., 2], 0.)
        potential, electric = super().interpolate(q)
        potential = np.where(metal, 0., potential)
        electric = np.where(metal[..., None], 0., electric)
        return potential, electric

    def tip_material_mask(self, positions):
        p = np.asarray(positions, dtype=float)
        if p.shape[-1:] != (3,) or not np.isfinite(p).all():
            raise ValueError("Cathode positions must be finite xyz metres")
        return p[..., 2] < 0.

    def conductor_material_mask(self, positions):
        """Identify finite conductor volumes, for interception diagnostics.

        Transport must still intersect paths with physical surfaces; testing
        endpoints alone cannot catch a particle crossing a thin conductor.
        """
        p = np.asarray(positions, dtype=float)
        if p.shape[-1:] != (3,) or not np.isfinite(p).all():
            raise ValueError("Conductor positions must be finite xyz metres")
        radial, z = np.hypot(p[..., 0], p[..., 1]), p[..., 2]
        mask = z < self.z[0]
        for row in self.request["rings"]:
            mask = mask | ((z >= row["start_m"]) & (z <= row["stop_m"])
                & (radial >= row["inner_m"]) & (radial <= row["outer_m"]))
        return mask


def build_closed_gun_field(gun, *, liner_segments=None,
                           cells_per_bore=None, outer_factor=2.,
                           exit_extension_mm=None, cache_dir=None):
    """Solve or load the complete, dependency-bound scalar gun field."""
    request = closed_field_request(gun, liner_segments=liner_segments,
        cells_per_bore=cells_per_bore, outer_factor=outer_factor,
        exit_extension_mm=exit_extension_mm)
    return _build_request_field(request, cache_dir)


def _build_request_field(request, cache_dir):
    if request["source_admission"] != "flat_classical_tip_only":
        raise ValueError("A curved cathode requires the continuous conductor field solver; no planar substitution")
    with numerical_job(requested=1) as cpu:
        if cache_dir is not None:
            cached = load_cached_field(request, cache_dir)
            if cached is not None:
                return ClosedGunField(request, cached.r, cached.z, cached.voltage, cached.report)
        started = time.perf_counter()
        r, z = mesh_axes(request)
        fixed, voltage = electrode_boundary_arrays(request, r, z)
        solved = AxisymmetricCutField(r, z, fixed, voltage, geometry=None,
            tolerance=request["numerics"]["linear_residual_tolerance"])
        report = {"cache_hit": False, "load_seconds": 0.,
            "solve_seconds": time.perf_counter()-started,
            "linear_residual": float(solved.residual),
            "elements": int(solved.element_count), "vertices": int(solved.vertex_count),
            "minimum_radial_cell_m": float(np.min(np.diff(r))),
            "minimum_axial_cell_m": float(np.min(np.diff(z))),
            "maximum_axial_cell_m": float(np.max(np.diff(z))),
            "request_sha256": request_digest(request), "cpu_resources": cpu.to_dict(),
            "model_status": request["model_status"],
            "limitations": request["limitations"],
            "boundary_conditions": request["boundary_conditions"]}
        field = ClosedGunField(request, r, z, solved.nodal_voltage, report)
        if cache_dir is not None:
            save_cached_field(field, cache_dir)
        return field


def field_cache_directory():
    """Generated fields only; no particle archives or user input records."""
    base = Path(os.environ.get("LOCALAPPDATA", Path.home()/".cache"))
    return base/"TEMSimulator"/"cache"/"gun-fields"


def _trim_disk_cache(directory, keep_digest):
    """Bound only this module's content-addressed generated archive pairs."""
    directory = Path(directory)
    entries = []
    for data in directory.glob("*.npz"):
        stem = data.stem
        if len(stem) != 64 or any(c not in "0123456789abcdef" for c in stem):
            continue
        manifest = data.with_suffix(".json")
        if manifest.is_file():
            entries.append((max(data.stat().st_mtime, manifest.stat().st_mtime),
                data.stat().st_size+manifest.stat().st_size, data, manifest))
    total = sum(row[1] for row in entries)
    for _, size, data, manifest in sorted(entries):
        if total <= MAX_DISK_CACHE_BYTES:
            break
        if data.stem != keep_digest:
            data.unlink(missing_ok=True)
            manifest.unlink(missing_ok=True)
            total -= size


def closed_field(gun):
    """Production field accessor with four-entry memory and bounded disk cache."""
    request = closed_field_request(gun)
    digest = request_digest(request)
    cached = getattr(gun, "_closed_gun_field", None)
    if cached is not None and cached.report.get("request_sha256") == digest:
        return cached
    with _MEMORY_LOCK:
        field = _MEMORY_FIELDS.get(digest)
        if field is not None:
            _MEMORY_FIELDS.move_to_end(digest)
            gun._closed_gun_field = field
            return field
    # Never hold the memory lock while waiting for CPU admission: calculation
    # workers may already own that job lock before requesting the same field.
    with numerical_job(requested=1):
        with _MEMORY_LOCK:
            field = _MEMORY_FIELDS.get(digest)
            if field is None:
                directory = field_cache_directory()
                field = _build_request_field(request, directory)
                _MEMORY_FIELDS[digest] = field
                _trim_disk_cache(directory, digest)
            _MEMORY_FIELDS.move_to_end(digest)
            while len(_MEMORY_FIELDS) > MAX_MEMORY_FIELDS:
                _MEMORY_FIELDS.popitem(last=False)
            gun._closed_gun_field = field
            return field
