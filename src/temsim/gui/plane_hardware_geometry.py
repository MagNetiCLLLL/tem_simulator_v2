"""Bounded presentation geometry at a captured particle observation plane.

All coordinates are laboratory X/Y millimetres. These outlines describe the
same clear openings and absorbing masks as the saved calculation; they do not
run transport, apply clipping, or consult currently edited instrument inputs.
Optical masks are thin planes, rather than their mechanical carrier envelopes.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass, replace
import math
from numbers import Integral
from types import SimpleNamespace


PLANE_TOLERANCE_MM = 1.0e-9
_WALL_TOLERANCE_MM = 1.0e-12

Point = tuple[float, float]
Polyline = tuple[Point, ...]


@dataclass(frozen=True, slots=True)
class PlaneHardwareOutline:
    """One physical boundary, using immutable bounded display polylines."""

    key: str
    name: str
    kind: str
    role: str
    z_mm: float
    polylines_mm: tuple[Polyline, ...]
    description: str


def _number(value, label: str, *, minimum=None, positive=False) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{label} must be a finite number in millimetres")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a finite number in millimetres") from error
    if (not math.isfinite(result) or (minimum is not None and result < minimum)
            or (positive and result <= 0.)):
        raise ValueError(f"{label} has invalid physical geometry")
    return result


def _circle(radius: float, x: float, y: float, segments: int) -> Polyline:
    points = tuple((x + radius * math.cos(2. * math.pi * index / segments),
                    y + radius * math.sin(2. * math.pi * index / segments))
                   for index in range(segments))
    return (*points, points[0])


def _outline(key, name, kind, role, z, paths, description):
    return PlaneHardwareOutline(str(key), str(name), kind, role, z,
                                tuple(paths), description)


def _column_wall(result, z: float, segments: int):
    active = []
    assembly = getattr(result, "assembly", None)
    for row in getattr(assembly, "vacuum_bore_segments", ()):
        start = _number(row.start_z_mm, "Vacuum bore start Z")
        stop = _number(row.end_z_mm, "Vacuum bore end Z")
        if stop < start:
            raise ValueError("Vacuum bore end must follow its start")
        if not start - _WALL_TOLERANCE_MM <= z <= stop + _WALL_TOLERANCE_MM:
            continue
        radius = .5 * _number(row.inner_diameter_mm, "Vacuum bore diameter", positive=True)
        active.append((radius, row))
    if not active:
        return ()
    # At a shared axial face, the narrower adjoining cylinder is the radial
    # shoulder. Do not show a wider, falsely transmitting bore at that face.
    radius, row = min(active, key=lambda value: value[0])
    return (_outline(getattr(row, "key", "column_wall"),
                     getattr(row, "name", "Column vacuum wall"), "column", "wall", z,
                     (_circle(radius, 0., 0., segments),),
                     f"Column vacuum wall; clear diameter {2. * radius:.9g} mm. "
                     "At or outside this radius rays contact the wall."),)


def _gun_bores(state, z: float, segments: int, *, project_upstream=False):
    rows = []
    gun = getattr(state, "electron_gun", None)
    for component in getattr(gun, "bore_components", ()):
        if not bool(getattr(component, "installed", True)):
            continue
        center = _number(component.mechanical_center_from_tip_mm, "Gun body centre Z")
        length = _number(component.mechanical_length_mm, "Gun body length", positive=True)
        start, stop = center - .5 * length, center + .5 * length
        if (start > z + _WALL_TOLERANCE_MM if project_upstream else
                not start - _WALL_TOLERANCE_MM <= z <= stop + _WALL_TOLERANCE_MM):
            continue
        radius = .5 * _number(component.mechanical_clear_bore_diameter_mm,
                             "Gun body clear bore diameter", positive=True)
        rows.append(_outline(component.key, getattr(component, "name", component.key),
                             "gun_bore", "wall", start if project_upstream else z,
                             (_circle(radius, 0., 0., segments),),
                             f"Gun body clear bore; diameter {2. * radius:.9g} mm. "
                             "Body interception remains physical when its field is disabled."
                             + (f" Body Z range {start:.12g}\u2013{stop:.12g} mm." if project_upstream else "")))
    return tuple(rows)


def _apertures(result, z: float, segments: int, *, project_upstream=False):
    rows = []
    for record in getattr(result, "aperture_stops", ()):
        if not isinstance(record, Mapping):
            raise ValueError("Captured aperture geometry must be a mapping")
        if not bool(record.get("enabled", True)) or not bool(record.get("installed", True)):
            continue
        reference = _number(record.get("z_mm"), "Aperture optical Z")
        if (reference > z + PLANE_TOLERANCE_MM if project_upstream else
                abs(reference - z) > PLANE_TOLERANCE_MM):
            continue
        key = record.get("key", "aperture")
        name = record.get("name", key)
        shape = record.get("shape")
        if shape == "circular":
            radius = .5 * _number(record.get("diameter_mm"), "Aperture opening diameter", minimum=0.)
            x = _number(record.get("offset_x_mm", 0.), "Aperture X offset")
            y = _number(record.get("offset_y_mm", 0.), "Aperture Y offset")
            paths = (_circle(radius, x, y, segments),)
            description = (f"Circular aperture opening; diameter {2. * radius:.9g} mm; "
                           f"X/Y offset {x:.9g}/{y:.9g} mm. "
                           + ("The opening is closed; all rays are blocked." if radius == 0.
                              else "The interior transmits; the exterior blocks."))
        elif shape == "two_blade_slit":
            radius = .5 * _number(record.get("bore_diameter_mm"), "Slit bore diameter", positive=True)
            paths = [_circle(radius, 0., 0., segments)]
            if bool(record.get("slit_inserted", False)):
                gap = _number(record.get("slit_gap_mm"), "Slit opening gap", minimum=0.)
                center = _number(record.get("slit_centre_x_mm"), "Slit X centre")
                # Finite blade edges end on the circular bore. No invented
                # carrier width or infinitely long blade crosses the view.
                for x in sorted(set((center - .5 * gap, center + .5 * gap))):
                    if abs(x) <= radius:
                        half_height = math.sqrt(max(0., radius * radius - x * x))
                        paths.append(((x, -half_height), (x, half_height)))
                closed = gap == 0. or center - .5 * gap >= radius or center + .5 * gap <= -radius
                description = (f"Two-blade slit; bore diameter {2. * radius:.9g} mm; "
                               f"X-directed gap {gap:.9g} mm centred at {center:.9g} mm. "
                               + ("No transmitting opening remains." if closed else
                                  "Only the gap inside the circular bore transmits; blades and bore exterior block."))
            else:
                description = (f"Slit blades retracted; the circular bore of diameter "
                               f"{2. * radius:.9g} mm still blocks its exterior.")
        else:
            raise ValueError(f"Unsupported captured aperture shape: {shape!r}")
        rows.append(_outline(key, name, "aperture", "opening", reference if project_upstream else z, paths,
                             description + f" Optical stop Z {reference:.12g} mm."))
    return tuple(rows)


def _detectors(state, z: float, segments: int, *, project_upstream=False):
    rows = []
    for detector in getattr(state, "recording_planes", ()):
        if not bool(getattr(detector, "inserted", False)):
            continue
        reference = _number(detector.z_mm, "Detector optical Z")
        if (reference > z + PLANE_TOLERANCE_MM if project_upstream else
                abs(reference - z) > PLANE_TOLERANCE_MM):
            continue
        geometry = str(detector.geometry).lower()
        outer = .5 * _number(detector.outer_width_mm, "Detector outer width", positive=True)
        if geometry in {"square", "rectangle", "camera"}:
            # Camera hit_mask is lab-axis aligned. Detector orientation and
            # flips affect readout coordinates, not this physical stop mask.
            paths = (((-outer, -outer), (outer, -outer), (outer, outer),
                      (-outer, outer), (-outer, -outer)),)
            description = f"Square detector; width {2. * outer:.9g} mm; its interior absorbs."
        elif geometry in {"disk", "annulus", "annular", "ring"}:
            x = _number(getattr(detector, "centre_offset_x_mm", 0.), "Detector X offset")
            y = _number(getattr(detector, "centre_offset_y_mm", 0.), "Detector Y offset")
            paths = [_circle(outer, x, y, segments)]
            if geometry != "disk":
                inner = .5 * _number(detector.inner_diameter_mm, "Detector inner diameter", minimum=0.)
                if inner >= outer:
                    raise ValueError("Detector inner diameter must be smaller than its outer diameter")
                if inner > 0.:
                    paths.append(_circle(inner, x, y, segments))
                description = (f"Annular detector; inner/outer diameter {2. * inner:.9g}/{2. * outer:.9g} mm; "
                               + ("the annular band absorbs and the central hole transmits." if inner > 0.
                                  else "the zero-diameter hole leaves the full interior absorbing."))
            else:
                description = f"Disk detector; diameter {2. * outer:.9g} mm; its interior absorbs."
            description += f" X/Y centre {x:.9g}/{y:.9g} mm."
        else:
            raise ValueError(f"Unsupported captured detector shape: {geometry!r}")
        if not bool(getattr(detector, "readout_enabled", True)):
            description += " Readout is disabled; physical absorption is still active."
        rows.append(_outline(detector.key, getattr(detector, "name", detector.key),
                             "detector", "absorbing", reference if project_upstream else z, paths,
                             description + f" Recording stop Z {reference:.12g} mm."))
    return tuple(rows)


def _projected_column_walls(result, z: float, segments: int):
    """Deduplicate repeated segments of one captured body with one diameter."""
    grouped = {}
    assembly = getattr(result, "assembly", None)
    for row in getattr(assembly, "vacuum_bore_segments", ()):
        start = _number(row.start_z_mm, "Vacuum bore start Z")
        stop = _number(row.end_z_mm, "Vacuum bore end Z")
        if stop < start:
            raise ValueError("Vacuum bore end must follow its start")
        if start > z + _WALL_TOLERANCE_MM:
            continue
        radius = .5 * _number(row.inner_diameter_mm, "Vacuum bore diameter", positive=True)
        key = str(getattr(row, "key", "column_wall"))
        name = str(getattr(row, "name", "Column vacuum wall"))
        grouped.setdefault((key, name, radius), []).append((start, stop))
    outlines = []
    for (key, name, radius), ranges in grouped.items():
        # Keep disjoint intervals explicit. A long carrier or a missing tube
        # segment does not become an invented continuous physical restriction.
        ranges = sorted(set(ranges))
        positions = "; ".join(f"{start:.12g}\u2013{stop:.12g}" for start, stop in ranges)
        label = ("Column vacuum wall" if name == "Column vacuum wall" or ":" in name
                 else f"Column vacuum wall \u2014 {name}")
        outlines.append(_outline(
            key, label, "column", "wall", ranges[0][0],
            (_circle(radius, 0., 0., segments),),
            f"Column vacuum wall; clear diameter {2. * radius:.9g} mm. "
            f"Captured body {name}; key {key}; Z range(s) {positions} mm. "
            "At or outside this radius rays contact the wall at its own Z."
        ))
    return tuple(outlines)


def hardware_geometry_snapshot(state, assembly=None):
    """Capture lightweight edited geometry for an explicitly labelled preview.

    This is display geometry, never an executed beam or continuation state.
    Only small component records are copied; no particle history, field or
    whole-instrument snapshot is allocated and no transport is performed.
    """
    from temsim.simulation_pipeline import aperture_stop_records

    gun = getattr(state, "electron_gun", None)
    geometry_state = SimpleNamespace(
        recording_planes=tuple(copy(row) for row in getattr(state, "recording_planes", ())),
        electron_gun=SimpleNamespace(
            bore_components=tuple(copy(row) for row in getattr(gun, "bore_components", ())),
        ),
    )
    return SimpleNamespace(
        state_snapshot=geometry_state,
        aperture_stops=aperture_stop_records(state),
        assembly=getattr(state, "_resolved_assembly", None) if assembly is None else assembly,
    )


def _request_geometry(z_mm, circle_segments):
    z = _number(z_mm, "Selected plane Z")
    if isinstance(circle_segments, bool) or not isinstance(circle_segments, Integral) or not 16 <= circle_segments <= 1024:
        raise ValueError("Circle display sampling must be an integer from 16 to 1024")
    return z, int(circle_segments)


def plane_hardware_outlines(result, z_mm, *, coordinate_frame="column",
                            circle_segments=128) -> tuple[PlaneHardwareOutline, ...]:
    """Describe physical cutoff boundaries at one selected captured plane.

    Positions stay in lab millimetres; the GUI applies its shared U/V rotation
    and display-unit conversion. Bent filter coordinates cannot be described
    by a straight-column section. No absent geometry is replaced by defaults.
    """
    z, segments = _request_geometry(z_mm, circle_segments)
    if result is None or coordinate_frame != "column":
        return ()
    state = getattr(result, "state_snapshot", None)
    return (*_column_wall(result, z, segments),
            *_gun_bores(state, z, segments),
            *_apertures(result, z, segments),
            *_detectors(state, z, segments))


def projected_hardware_outlines(result, z_mm, *, coordinate_frame="column",
                                circle_segments=128) -> tuple[PlaneHardwareOutline, ...]:
    """Look upstream along the column from selected Z without transporting masks.

    The polylines retain the hardware's laboratory X/Y geometry and actual Z;
    they are axial projections of restrictions encountered earlier on the
    path, not effective acceptance at the selected plane. Lenses and ray slopes
    prevent identifying clipping by comparing these outlines to current dots.
    Components downstream of selected Z are excluded. Bent filter coordinates
    cannot use straight-column projected boundaries.
    """
    z, segments = _request_geometry(z_mm, circle_segments)
    if result is None or coordinate_frame != "column":
        return ()
    state = getattr(result, "state_snapshot", None)
    outlines = (*_projected_column_walls(result, z, segments),
                *_gun_bores(state, z, segments, project_upstream=True),
                *_apertures(result, z, segments, project_upstream=True),
                *_detectors(state, z, segments, project_upstream=True))
    return tuple(replace(row, description=(
        f"Axial projection looking upstream from Z {z:.12g} mm. "
        + row.description
        + " This is hardware at its own Z, not an effective cutoff at the selected plane."
    )) for row in outlines)
