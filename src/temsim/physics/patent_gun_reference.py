"""Opt-in proportional electrode study, referenced to US8803411B2 Fig. 2.

This is a bounded downstream reference, NOT a reconstruction of the complete
patent gun. The original tip emission stays at z=0; its electrical boundary is
an explicitly chosen ideal full-domain planar cathode. The patent extractor's
upstream enclosure is incompatible with that plane and is not reconstructed.
Its downstream lip still supplies extraction. Control, four shaped intermediate
electrodes and a grounded terminal constitute SIX divider nodes, not six extra
acceleration electrodes. This provider never changes the normal gun model.

Coordinates were manually estimated from the patent drawing, not measured
hardware. The source-to-terminal span is normalised to the current mechanical
accelerator END position, not its length. A paired flat variant removes only
the four intermediate electrodes' return lips, holding their webs and voltages
fixed. All potentials are rises from the original tip (V), positions are m,
and the Cartesian electric field is -grad(Phi) in V/m.
"""
from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path
import time

import numpy as np
import scipy

from temsim.cpu_resources import numerical_job
from temsim.physics.axisymmetric_cut_field import AxisymmetricCutField
from temsim.physics.grounded_tip_field import merge_axis_nodes
from temsim.physics.planar_gun_field import (
    PlanarGunField, load_cached_field, mesh_axes as _graded_mesh_axes, request_digest,
    save_cached_field,
)


SCHEMA = "diagnostic-proportional-electrodes-planar-cathode-v1"
PDF_URL = "https://patentimages.storage.googleapis.com/b2/de/20/4b372cd1334732/US8803411.pdf"
PATENT_URL = "https://patents.google.com/patent/US8803411B2/en"
MAX_VERTICES = 1_500_000

# Approximate PDF-page points (1/72 inch in the source PDF), BEFORE conversion
# to a reference apparatus. y increases downwards. r is the estimated distance
# from the drawn axis. Manual uncertainty is about 2--3 page points; no physical
# dimensional accuracy is implied. Piece tuples: name, r_min, r_max, y_min,y_max.
SOURCE_APEX_Y = 310.
TERMINAL_DOWNSTREAM_Y = 560.
EXTRACTOR_LIP = (
    ("lip_upper", 6., 60., 312., 314.),
    ("lip_middle", 10., 60., 314., 318.),
    ("lip_lower", 21., 60., 318., 322.),
)
CONTROL_CONTOUR = (
    ("web", 18., 59., 349., 357.),
    ("outer_return", 50., 59., 340., 349.),
    ("inner_lip", 10., 18., 356., 362.),
)
INTERMEDIATE_WEB_CENTRES_Y = (393., 433., 473., 513.)
TERMINAL_CONTOUR = (("terminal", 13., 86., 550., 560.),)


def _implementation_hash():
    paths = [Path(__file__), Path(inspect.getfile(PlanarGunField)),
             Path(inspect.getfile(AxisymmetricCutField)),
             Path(inspect.getfile(merge_axis_nodes)),
             Path(inspect.getfile(inspect.unwrap(numerical_job)))]
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def patent_reference_request(gun, *, cathode_boundary, contour="stepped",
                             cells_per_bore=8, outer_factor=2., exit_extension_mm=0.):
    """Capture only the physics and numerical inputs consumed by this study.

    Existing extractor/lens voltage settings define V1/V2 relative to the tip.
    The five equal divider drops span V2 to high tension. The current gun's
    accelerator stage list and its analytic field-shaping parameters are not
    used: this explicitly selected reference has its own electrode topology.
    """
    if cathode_boundary != "planar_equipotential":
        raise ValueError("Explicit cathode_boundary='planar_equipotential' is required")
    emitter = gun.emitter
    if (getattr(emitter, "surface_model", None) is not None
            or float(getattr(emitter, "curvature_nm_inv", 0.)) != 0.):
        raise ValueError("The proportional reference requires a flat source; curved geometry cannot be omitted")
    if (getattr(emitter, "coherence", None) is not None
            or getattr(gun, "source_representation", "classical_particles") != "classical_particles"):
        raise ValueError("This reference supports classical tip emission only; coherent work is paused")
    if gun.monochromator_installed:
        raise ValueError("An installed monochromator cannot be bypassed by this reference")
    if contour not in ("stepped", "flat"):
        raise ValueError("contour must be 'stepped' or 'flat'")
    if type(cells_per_bore) is not int or not 4 <= cells_per_bore <= 64:
        raise ValueError("cells_per_bore must be an integer in [4, 64]")
    outer_factor, exit_extension_mm = float(outer_factor), float(exit_extension_mm)
    if not math.isfinite(outer_factor) or outer_factor <= 1.:
        raise ValueError("outer_factor must be finite and exceed one")
    if not math.isfinite(exit_extension_mm) or exit_extension_mm < 0.:
        raise ValueError("exit_extension_mm must be finite and non-negative")
    ht = float(gun.accelerator.high_tension_kv)*1000.
    extraction = float(gun.extractor.voltage_kv)*1000.
    control = float(gun.electrostatic_lens.potential_rise_from_tip_v(extraction/1000., ht/1000.))
    if (not all(map(math.isfinite, (ht, extraction, control)))
            or not 0 <= extraction < ht or not 0 <= control < ht):
        raise ValueError("Extraction and control potentials must be non-negative and below finite high tension")
    accelerator_center = float(gun.accelerator.mechanical_center_from_tip_mm)*1e-3
    accelerator_length = float(gun.accelerator.mechanical_length_mm)*1e-3
    terminal_end = accelerator_center + accelerator_length/2
    gun_exit = float(gun.exit_plane_z_mm)*1e-3
    end = gun_exit + exit_extension_mm*1e-3
    if (not all(map(math.isfinite, (accelerator_center, accelerator_length, terminal_end, gun_exit, end)))
            or accelerator_length <= 0 or not 0 < terminal_end < gun_exit <= end):
        raise ValueError("The positive reference terminal must precede the complete gun exit")
    scale = terminal_end/(TERMINAL_DOWNSTREAM_Y-SOURCE_APEX_Y)
    nodes, rings = [], []

    def add_node(key, potential, reference_y, digitized, divider_index=None):
        pieces = []
        for name, inner, outer, start, stop in digitized:
            piece = {"key": key, "piece": name,
                     "inner_m": inner*scale, "outer_m": outer*scale,
                     "start_m": (start-SOURCE_APEX_Y)*scale,
                     "stop_m": (stop-SOURCE_APEX_Y)*scale,
                     "potential_rise_v": float(potential)}
            if not 0 < piece["inner_m"] < piece["outer_m"] or not 0 < piece["start_m"] < piece["stop_m"] < end:
                raise ValueError("Invalid proportional reference contour")
            pieces.append(piece)
        nodes.append({"key": key, "reference_z_m": (reference_y-SOURCE_APEX_Y)*scale,
                      "potential_rise_v": float(potential), "divider_index": divider_index,
                      "minimum_bore_radius_m": min(p["inner_m"] for p in pieces),
                      "digitized_pieces_page_points": [list(row) for row in digitized],
                      "pieces": pieces})
        rings.extend(pieces)

    add_node("extractor", extraction, 316., EXTRACTOR_LIP)
    add_node("control", control, 353., CONTROL_CONTOUR, 0)
    for index, center in enumerate(INTERMEDIATE_WEB_CENTRES_Y, start=1):
        # The web and its opening are identical in both paired cases. Removing
        # only the return lips changes their field contribution, not voltages,
        # web locations or minimum bores.
        pieces = [("web", 18., 59., center-4., center+4.)]
        if contour == "stepped":
            pieces.extend((("outer_return", 50., 59., center-12., center-4.),
                           ("inner_return", 18., 28., center+4., center+22.)))
        add_node(f"accelerator:{index}", control+(ht-control)*index/5., center, pieces, index)
    add_node("terminal", ht, 555., TERMINAL_CONTOUR, 5)
    # Different electrical nodes may not meet. Pieces belonging to one node
    # deliberately share edges and are solved as one equipotential conductor.
    for i, left in enumerate(rings):
        for right in rings[i+1:]:
            if left["key"] == right["key"]:
                continue
            if (max(left["start_m"], right["start_m"]) <= min(left["stop_m"], right["stop_m"])
                    and max(left["inner_m"], right["inner_m"]) <= min(left["outer_m"], right["outer_m"])):
                raise ValueError("Different proportional electrode nodes overlap")
    outer_radius = outer_factor*max(row["outer_m"] for row in rings)
    if not math.isfinite(outer_radius):
        raise ValueError("The radial reference domain must be finite")
    return {
        "schema": SCHEMA, "potential_reference": "rise_relative_to_original_tip_v",
        "model_status": "bounded_proportional_downstream_reference_not_complete_patent_gun",
        "source_admission": "unchanged_flat_classical_tip_emission",
        "reference": {"publication": "US8803411B2", "figure": "2", "pdf_page": 4,
                      "drawing_sheet": "2 of 9", "patent_url": PATENT_URL, "pdf_url": PDF_URL,
                      "drawing_role": "conventional_six_node_accelerating_tube_example",
                      "digitization": "manual_rounded_axisymmetric_rectangular_approximation",
                      "page_coordinate_units": "PDF points, not physical dimensions",
                      "manual_coordinate_uncertainty_page_points": 3.,
                      "source_apex_y_page_points": SOURCE_APEX_Y,
                      "terminal_downstream_y_page_points": TERMINAL_DOWNSTREAM_Y},
        "normalization": {"method": "source_to_terminal_span_matches_current_accelerator_mechanical_end",
                          "metres_per_page_point": scale, "source_z_m": 0.,
                          "terminal_end_m": terminal_end,
                          "target_is_accelerator_end_not_accelerator_length": True},
        "contour": contour, "electrode_nodes": nodes, "rings": rings,
        "high_tension_v": ht, "extraction_v": extraction, "control_v": control,
        "voltage_network": {"reference": "all_potential_rises_relative_to_original_tip",
                            "divider_nodes": [node["key"] for node in nodes if node["divider_index"] is not None],
                            "equal_resistor_intervals": 5,
                            "voltage_drop_per_interval_v": (ht-control)/5.,
                            "control_setting": "current_lens_potential_rise_from_tip_v_no_analytic_scale",
                            "grounded_node": "terminal"},
        "domain": {"entrance_m": 0., "exit_m": end, "gun_exit_m": gun_exit,
                   "outer_radius_m": outer_radius},
        "boundary_conditions": {
            "cathode": {"type": cathode_boundary, "rise_v": 0.,
                        "extent": "entire_radial_domain_at_z_zero"},
            "terminal": {"type": "finite_annular_dirichlet", "rise_v": ht,
                         "downstream_face_m": terminal_end, "ground_potential_v": 0.},
            "exit": {"type": "uniform_dirichlet", "rise_v": ht,
                     "status": "artificial_far_boundary_requires_extension_check"},
            "radial_outer": "natural_zero_normal_derivative", "axis": "axisymmetric_regular",
            "metal": "same_node_rectangular_unions_at_dirichlet_potential"},
        "limitations": [
            "The full-domain planar cathode is an explicit idealization, not a reconstruction of the drawn tip or emission patch.",
            "Only the downstream extractor lip is reconstructed; its upstream enclosure conflicts with the planar cathode and is omitted explicitly.",
            "Drawing proportions are approximate manual estimates without a calibrated dimensional scale.",
            "The current-geometry comparison also changes bores, positions and the divider topology; it does not isolate electrode shape.",
            "The stepped-versus-flat pair removes only intermediate return lips; all electrical node voltages, web locations and minimum bores are held fixed.",
            "Insulators and external housings have no assigned electrical properties in this vacuum Laplace reference.",
            "The finite grounded terminal is explicit; no grounded downstream tube or zero-field drift is invented.",
            "Original emission, magnetic components and aperture interception remain separate required transport operations.",
            "No space charge, image charge, tunnelling-current prediction or coherent propagation.",
            "No full gun qualification follows from a small linear residual or this reference comparison."],
        "numerics": {"cells_per_bore": cells_per_bore, "outer_radius_factor": outer_factor,
                     "exit_extension_mm": exit_extension_mm,
                     "mesh": "electrode-local-graded-physical-boundary-ulp-merge-v1",
                     "interpolation": "bilinear-radius-squared-and-z-v1",
                     "linear_residual_tolerance": 1e-9, "maximum_vertices": MAX_VERTICES},
        "implementation_sha256": _implementation_hash(),
        "libraries": {"numpy": np.__version__, "scipy": scipy.__version__},
    }


def mesh_axes(request):
    """Reuse electrode-local meshing and remove only floating-point aliases."""
    r, z = _graded_mesh_axes(request)
    domain = request["domain"]
    radial_faces = [0., domain["outer_radius_m"],
                    *[p[name] for p in request["rings"] for name in ("inner_m", "outer_m")]]
    axial_faces = [domain["entrance_m"], domain["exit_m"],
                   *[p[name] for p in request["rings"] for name in ("start_m", "stop_m")]]
    return merge_axis_nodes(r, boundaries=radial_faces), merge_axis_nodes(z, boundaries=axial_faces)


def electrode_boundary_arrays(request, r, z):
    """Assemble fixed voltages, preserving unions of one conductor's pieces."""
    rr, zz = np.meshgrid(r, z, indexing="ij")
    fixed = np.zeros(rr.shape, dtype=bool)
    values = np.zeros(rr.shape)
    fixed[:, 0], fixed[:, -1] = True, True
    values[:, -1] = request["high_tension_v"]
    for piece in request["rings"]:
        metal = ((zz >= piece["start_m"]) & (zz <= piece["stop_m"])
                 & (rr >= piece["inner_m"]) & (rr <= piece["outer_m"]))
        if not metal.any() or np.any(metal & fixed & (values != piece["potential_rise_v"])):
            raise ValueError("Unresolved or electrically conflicting proportional reference electrode")
        fixed |= metal
        values[metal] = piece["potential_rise_v"]
    return fixed, values


def build_patent_gun_field(gun, *, cathode_boundary, contour="stepped",
                           cells_per_bore=8, outer_factor=2., exit_extension_mm=0., cache_dir=None):
    """Build only the selected reference; do not modify or install it on gun."""
    request = patent_reference_request(gun, cathode_boundary=cathode_boundary,
        contour=contour, cells_per_bore=cells_per_bore,
        outer_factor=outer_factor, exit_extension_mm=exit_extension_mm)
    with numerical_job(requested=1) as cpu:
        if cache_dir is not None:
            cached = load_cached_field(request, cache_dir)
            if cached is not None:
                return cached
        started = time.perf_counter()
        r, z = mesh_axes(request)
        fixed, values = electrode_boundary_arrays(request, r, z)
        solved = AxisymmetricCutField(r, z, fixed, values, geometry=None,
            tolerance=request["numerics"]["linear_residual_tolerance"])
        report = {"cache_hit": False, "load_seconds": 0.,
                  "solve_seconds": time.perf_counter()-started,
                  "linear_residual": float(solved.residual),
                  "elements": int(solved.element_count), "vertices": int(solved.vertex_count),
                  "minimum_radial_cell_m": float(np.min(np.diff(r))),
                  "minimum_axial_cell_m": float(np.min(np.diff(z))),
                  "maximum_axial_cell_m": float(np.max(np.diff(z))),
                  "request_sha256": request_digest(request), "cpu_resources": cpu.to_dict(),
                  "model_status": request["model_status"], "contour": contour,
                  "limitations": request["limitations"],
                  "boundary_conditions": request["boundary_conditions"]}
        field = PlanarGunField(request, r, z, solved.nodal_voltage, report)
        if cache_dir is not None:
            save_cached_field(field, cache_dir)
        return field
