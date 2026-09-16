"""Read-only view of the analytic tip's active launch geometry.

These open surfaces are GUI guides, never CAD solids, material regions or new
source inputs. Use the emission operator itself to preserve historical models.
"""
from dataclasses import replace
from types import SimpleNamespace
import math

import numpy as np

from temsim.optics.electron_gun.tip_curvature import curve_bundle, support_radius_nm


def emission_view_values(emitter):
    if emitter.surface_model is not None or emitter.coherence is not None:
        return None
    return {name: getattr(emitter, name) for name in (
        "curvature_nm_inv", "curvature_model", "virtual_source_fwhm_nm",
        "angular_rms_mrad", "angular_cutoff_mrad")}


def analytic_emission(part, runtime):
    if (not part.get("tip_particle_model") or part.get("mechanical_only")
            or part.get("mechanical_part_role") == "custom_mechanical_copy"
            or (runtime or {}).get("tip_surface_model") is not None):
        return None
    values = (runtime or {}).get("tip_analytic_emission")
    return SimpleNamespace(**values, surface_model=None, coherence=None) if values is not None else None


def _positions(emitter, xy_nm):
    from temsim.optics.electron_gun.base import EmissionBundle
    n = len(xy_nm)
    bundle = EmissionBundle(xy_nm[:, 0]*1e-9, xy_nm[:, 1]*1e-9,
                           np.zeros(n), np.zeros(n), np.zeros(n),
                           np.full(n, 1/max(n, 1)), np.arange(n))
    emitted = curve_bundle(bundle, emitter)
    positions = getattr(emitted, "surface_position_m",
                        np.column_stack((bundle.x_m, bundle.y_m, np.zeros(n))))
    normals = getattr(emitted, "surface_normal", np.tile([0., 0., 1.], (n, 1)))
    return positions, normals


def emission_dimensions(part, runtime):
    emitter = analytic_emission(part, runtime)
    if emitter is None:
        return None
    radius = support_radius_nm(emitter)
    positions, _ = _positions(emitter, np.array([[0., 0.], [radius, 0.]]))
    return {"curvature_nm_inv": emitter.curvature_nm_inv,
            "emission_radius_nm": 1/emitter.curvature_nm_inv if emitter.curvature_nm_inv else math.inf,
            "virtual_source_fwhm_nm": emitter.virtual_source_fwhm_nm,
            "emission_support_diameter_nm": 2*radius,
            "emission_depth_nm": max(0.0, float(-positions[-1, 2]*1e9))}


def emission_parameter_paths(key):
    return (("runtime", key, "curvature_nm_inv"),
            ("runtime", key, "virtual_source_fwhm_nm"),
            *(("derived", key, name) for name in (
                "emission_radius_nm", "emission_support_diameter_nm", "emission_depth_nm")))


def emission_display_meshes(part, runtime, *, angular_segments=64):
    """Return selectable launch surface and normal guides in module-local mm."""
    from temsim.part_model_3d import TriangleMesh
    from temsim.optics.electron_gun.tip_assembly import tip_apex_z_mm
    emitter = analytic_emission(part, runtime)
    if emitter is None or support_radius_nm(emitter) == 0:
        return ()
    count, rings = angular_segments, 16
    support = support_radius_nm(emitter)
    phi = np.arange(count)*2*np.pi/count
    xy = np.vstack((np.zeros((1, 2)), *(
        np.column_stack((np.cos(phi), np.sin(phi)))*r
        for r in np.linspace(0., support, rings+1)[1:])))
    positions, _ = _positions(emitter, xy)
    positions = positions*1000
    positions[:, 2] += tip_apex_z_mm(part)
    faces = [(0, 1+j, 1+(j+1)%count) for j in range(count)]
    for ring in range(rings-1):
        inner, outer = 1+ring*count, 1+(ring+1)*count
        for j in range(count):
            k = (j+1)%count
            faces.extend(((inner+j, outer+j, outer+k), (inner+j, outer+k, inner+k)))
    faces = np.asarray(faces, dtype=np.int64)
    key, paths = part["key"], emission_parameter_paths(part["key"])
    boundary = positions[np.r_[np.arange(1+(rings-1)*count, 1+rings*count), 1+(rings-1)*count]]
    mesh = TriangleMesh(positions, faces, key, region="emitting_cap", color=(1., .72, .12, 1.),
        material_class="Launch surface (display only)",
        description="Active analytic-tip emitting surface; not a material body",
        face_groups=np.full(len(faces), "emitting_cap"),
        surfaces={"emitting_cap": {"label": "Active emitting surface", "kind": "emission",
                                  "parameter_paths": paths}},
        edges=({"id": "emission_boundary", "vertices": boundary,
                "label": "Projected emission support boundary", "parameter_paths": paths},))
    points, normals = _positions(emitter, np.array([[0., 0.], [support, 0.], [-support, 0.],
                                                   [0., support], [0., -support]]))
    points *= 1000
    points[:, 2] += tip_apex_z_mm(part)
    ends = points+normals*(.4*support*1e-6)
    guides = TriangleMesh(np.vstack((points, ends)), np.empty((0, 3), dtype=np.int64), key,
        region="emission_normals", color=(.2, .84, .6, 1.), material_class="Direction guides",
        description="Local emission normals, not propagated rays", wireframe=True,
        edges=tuple({"id": f"normal_{i}", "vertices": np.array([start, end]),
                     "parameter_paths": ()} for i, (start, end) in enumerate(zip(points, ends))))
    return mesh, guides


def emission_note(part, runtime):
    dims = emission_dimensions(part, runtime)
    if dims is None:
        return None
    radius = "flat (R = ∞)" if not dims["curvature_nm_inv"] else f"R {dims['emission_radius_nm']:.6g} nm"
    from temsim.optics.electron_gun.tip_curvature import ANGLE_ONLY_MODEL, LEGACY_MODEL
    model = analytic_emission(part, runtime).curvature_model
    historical = ("Historical angle-only launch plane; curvature defines emission axes. " if model == ANGLE_ONLY_MODEL else
                  "Historical sag model. " if model == LEGACY_MODEL else "")
    return (historical + f"Active emission: κ {dims['curvature_nm_inv']:.6g} nm⁻¹ · {radius} · "
            f"support diameter {dims['emission_support_diameter_nm']:.6g} nm · "
            f"edge depth {dims['emission_depth_nm']:.6g} nm. Gold: launch surface; green: local emission axes. "
            "Equal XYZ scale. Reference body and electrode field are unchanged.")


def emission_parameter_information(part, path, runtime, meaning, impact):
    """Scope Use annotations to the actual analytic source selected in the UI."""
    if analytic_emission(part, runtime) is None:
        return meaning, impact
    from temsim.optics.electron_gun.tip_assembly import PART_FIELDS
    if path[0] == "parts" and path[2] in PART_FIELDS:
        detail = ("Saved reference for the separate electrode-field tip model. "
                  "This value does not control the current analytic emitting surface or re-solve its electric field.")
        return replace(meaning, description=detail), replace(impact, status="inactive", active=False,
            label="Reference tip definition", detail=detail, effects=("display",), affected_results=("3D reference body",))
    if tuple(path) in emission_parameter_paths(part["key"]):
        detail = ("Current analytic-tip launch geometry. Radius is 1/curvature (infinite at zero); "
                  "support is the projected three-sigma disk. Geometry is read from the same emission operator as particle tracing. "
                  "Change curvature or source FWHM in Tip model / emission or the operating controls.")
        return replace(meaning, description=detail, category="operating",
            category_label="Derived emission geometry" if path[0] == "derived" else "Operating value / limit",
            source_kind="runtime", source_label="Current emission geometry", source_note=detail), replace(impact, status="active", active=True,
            label="Active emission geometry" if path[0] == "runtime" else "Derived from active emission",
            detail=detail, effects=("display", "operating"), affected_results=("3D emitting surface", "Particle launch",))
    return meaning, impact
