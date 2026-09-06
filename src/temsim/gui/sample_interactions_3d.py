"""Interactive 3-D view of cached specimen-local interactions.

This module is deliberately a renderer.  It never launches electron transport,
multislice, EDS generation, or downstream column propagation.  The page consumes
the shared :class:`SpecimenInteractionResult` produced by High accuracy and, when
the user explicitly requests it, the already bounded ``SampleRegionResult``.

Electron paths on this page end at a specimen-region boundary.  They are not
detector counts: electron detector signals are assigned only after the same
sample-exit states have passed through the downstream column and physically
intersected an inserted detector.
"""

from __future__ import annotations

from temsim.specimen.vector_field_transport import SpecimenFieldTransport

from dataclasses import dataclass
import math
import os

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QVector3D
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from temsim.specimen.scene import SpecimenScene
from temsim.specimen.axial_field_transport import (
    sample_axial_field_diagnostic,
)


try:
    if os.environ.get("TEMSIM_DISABLE_OPENGL", "").strip() == "1":
        raise ImportError("OpenGL disabled by TEMSIM_DISABLE_OPENGL")
    import pyqtgraph.opengl as gl
except Exception as _opengl_import_error:  # pragma: no cover - platform dependent
    gl = None
    OPENGL_IMPORT_ERROR = str(_opengl_import_error)
else:
    OPENGL_IMPORT_ERROR = None


PATH_STYLES = {
    "incident": ("Incident electrons", "#67e8f9"),
    "primary": ("Unscattered / primary electrons", "#4ade80"),
    "elastic": ("Elastically scattered electrons", "#fb7185"),
    "backscattered": ("Backscattered electrons", "#f97316"),
    "downstream_primary": ("Primary electrons leaving the sample", "#a3e635"),
    "downstream_elastic": ("Scattered electrons leaving the sample", "#facc15"),
    "xray_generated": ("Generated characteristic X-rays", "#f472b6"),
    "xray_detected": ("X-rays inside EDS acceptance", "#22d3ee"),
}

EVENT_STYLES = {
    "elastic_event": ("Elastic-scattering sites", "#fda4af"),
    "inelastic_event": ("Inelastic / vacancy sites", "#c084fc"),
    "relaxation_event": ("Relaxation / X-ray sites", "#fde047"),
}

LEGEND_LABELS = {
    "incident": "Incident",
    "primary": "Primary",
    "elastic": "Elastic",
    "backscattered": "Backscatter",
    "downstream_primary": "Primary exit",
    "downstream_elastic": "Scattered exit",
    "xray_generated": "X-rays",
    "xray_detected": "EDS accepted",
    "elastic_event": "Elastic sites",
    "inelastic_event": "Vacancy sites",
    "relaxation_event": "X-ray sites",
}

SIGNAL_GROUPS = (
    (
        "Electron trajectories",
        (
            "incident",
            "primary",
            "elastic",
            "backscattered",
            "downstream_primary",
            "downstream_elastic",
        ),
    ),
    ("Characteristic X-rays", ("xray_generated", "xray_detected")),
    (
        "Interaction sites",
        ("elastic_event", "inelastic_event", "relaxation_event"),
    ),
)


@dataclass(frozen=True, slots=True)
class ScenePath:
    """One immutable path in specimen-local nanometres."""

    positions_nm: np.ndarray
    category: str
    provenance: str = ""

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_nm, dtype=float)
        if (
            positions.ndim != 2
            or positions.shape[1:] != (3,)
            or positions.shape[0] < 2
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError("A 3-D scene path must be a finite N by 3 array")
        positions.setflags(write=False)
        object.__setattr__(self, "positions_nm", positions)


@dataclass(frozen=True, slots=True)
class SceneEvents:
    """One category of specimen-local event markers."""

    positions_nm: np.ndarray
    category: str

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_nm, dtype=float)
        if (
            positions.ndim != 2
            or positions.shape[1:] != (3,)
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError("3-D event positions must be a finite N by 3 array")
        positions.setflags(write=False)
        object.__setattr__(self, "positions_nm", positions)


@dataclass(frozen=True, slots=True)
class SampleInteractionScene:
    """Render-ready, calculation-free snapshot of specimen interactions."""

    paths: tuple[ScenePath, ...]
    events: tuple[SceneEvents, ...]
    material_bounds_nm: tuple[np.ndarray, np.ndarray]
    region_bounds_nm: tuple[np.ndarray, np.ndarray]
    sample_thickness_nm: float
    coherent_wave_available: bool
    has_bounded_result: bool
    specimen_mode: str
    specimen_source_key: str
    specimen_is_vacuum: bool
    sample_envelope_shape: str
    sample_centre_xy_nm: tuple[float, float]
    sample_size_xy_nm: tuple[float, float]
    sample_bounds_nm: tuple[np.ndarray, np.ndarray]
    virtual_region_outlines_nm: tuple[np.ndarray, ...]
    sample_axial_field_t: float
    sample_objective_field_t: float
    sample_field_face_variation_t: float
    sample_field_transport_model: str
    sample_field_geometry_material_coupled: bool


def _bounds(points, fallback_half_nm: float) -> tuple[np.ndarray, np.ndarray]:
    arrays = [np.asarray(value, dtype=float) for value in points if np.size(value)]
    if arrays:
        joined = np.concatenate(arrays, axis=0)
        lower = np.min(joined, axis=0)
        upper = np.max(joined, axis=0)
    else:
        half = max(float(fallback_half_nm), 1.0)
        lower = np.array((-half, -half, -half), dtype=float)
        upper = np.array((half, half, half), dtype=float)
    span = np.maximum(upper - lower, 1.0e-6)
    padding = np.maximum(0.08 * span, 0.5)
    return lower - padding, upper + padding


def _global_mm_to_local_nm(positions_mm, sample_z_mm: float) -> np.ndarray:
    values = np.asarray(positions_mm, dtype=float)
    local = values.copy()
    local[:, :2] *= 1.0e6
    local[:, 2] = (local[:, 2] - float(sample_z_mm)) * 1.0e6
    return local


def _gl_display_positions(positions_nm) -> np.ndarray:
    """Map physical local coordinates to the TEM-style 3-D display frame.

    The physical model remains right-handed with +Z downstream.  PyQtGraph's
    OpenGL world draws +Z upward in the default side view, so the renderer
    reflects only its displayed Z coordinate to make downstream screen-down.
    """

    displayed = np.asarray(positions_nm, dtype=float).copy()
    if displayed.shape[-1:] != (3,):
        raise ValueError("Displayed specimen positions need a final XYZ axis")
    displayed[..., 2] *= -1.0
    return displayed


def _gl_display_bounds(
    bounds_nm: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ordered OpenGL bounds after the display-only Z reflection."""

    lower, upper = (np.asarray(value, dtype=float) for value in bounds_nm)
    display_lower = lower.copy()
    display_upper = upper.copy()
    display_lower[2] = -upper[2]
    display_upper[2] = -lower[2]
    return display_lower, display_upper


def _clip_photon_path(
    positions_nm: np.ndarray,
    maximum_length_nm: float,
) -> np.ndarray:
    """Clip only the schematic photon endpoint; preserve origin/direction."""

    start = np.asarray(positions_nm[0], dtype=float)
    vector = np.asarray(positions_nm[-1], dtype=float) - start
    length = float(np.linalg.norm(vector))
    if length <= 0.0 or length <= maximum_length_nm:
        return np.asarray((start, positions_nm[-1]), dtype=float)
    return np.asarray((start, start + vector * (maximum_length_nm / length)))


def _event_category(process) -> str:
    key = str(getattr(process, "value", process))
    if key == "elastic_scatter":
        return "elastic_event"
    if key in {
        "radiative_relaxation",
        "characteristic_x_ray",
        "auger_relaxation",
        "unresolved_relaxation",
    }:
        return "relaxation_event"
    return "inelastic_event"


def _downstream_scene_paths(
    sample_region_result,
    sample_z_mm: float,
    *,
    maximum_paths: int = 192,
) -> list[ScenePath]:
    """Extract exact cached branch histories only as far as the chosen exit."""

    candidates: list[tuple[object, int]] = []
    for branch in tuple(getattr(sample_region_result, "downstream_branches", ())):
        x = np.asarray(getattr(branch, "x", ()), dtype=float)
        if x.ndim != 2:
            continue
        candidates.extend((branch, index) for index in range(x.shape[1]))
    if len(candidates) > int(maximum_paths):
        indices = np.linspace(
            0, len(candidates) - 1, int(maximum_paths), dtype=int
        )
        candidates = [candidates[index] for index in indices]

    exit_z_mm = float(sample_region_result.exit_z_mm)
    rows: list[ScenePath] = []
    for branch, ray_index in candidates:
        z = np.asarray(branch.z, dtype=float)
        x = np.asarray(branch.x, dtype=float)[:, ray_index]
        y = np.asarray(branch.y, dtype=float)[:, ray_index]
        mask = (
            (z >= float(sample_z_mm) - 1.0e-12)
            & (z <= exit_z_mm + 1.0e-12)
            & np.isfinite(x)
            & np.isfinite(y)
        )
        if np.count_nonzero(mask) < 2:
            continue
        points = np.column_stack(
            (
                x[mask] * 1.0e9,
                y[mask] * 1.0e9,
                (z[mask] - float(sample_z_mm)) * 1.0e6,
            )
        )
        kind = str(getattr(branch, "interaction_kind", ""))
        category = (
            "downstream_elastic"
            if kind == "sample_region_elastic"
            else "downstream_primary"
        )
        rows.append(ScenePath(points, category, "cached downstream branch"))
    return rows


def _sample_model_bounds(
    scene: SpecimenScene,
) -> tuple[np.ndarray, np.ndarray]:
    centre_x, centre_y = scene.centre_xy_nm
    size_x, size_y = scene.size_xy_nm
    half_z = 0.5 * scene.interacting_thickness_nm
    lower = np.asarray(
        (centre_x - 0.5 * size_x, centre_y - 0.5 * size_y, -half_z),
        dtype=float,
    )
    upper = np.asarray(
        (centre_x + 0.5 * size_x, centre_y + 0.5 * size_y, half_z),
        dtype=float,
    )
    return lower, upper


def _sample_model_outline(scene: SampleInteractionScene) -> np.ndarray:
    centre_x, centre_y = scene.sample_centre_xy_nm
    size_x, size_y = scene.sample_size_xy_nm
    if scene.sample_envelope_shape == "disk":
        angle = np.linspace(0.0, 2.0 * math.pi, 129)
        return np.column_stack(
            (
                centre_x + 0.5 * size_x * np.cos(angle),
                centre_y + 0.5 * size_y * np.sin(angle),
                np.zeros_like(angle),
            )
        )
    return np.asarray(
        (
            (centre_x - 0.5 * size_x, centre_y - 0.5 * size_y, 0.0),
            (centre_x + 0.5 * size_x, centre_y - 0.5 * size_y, 0.0),
            (centre_x + 0.5 * size_x, centre_y + 0.5 * size_y, 0.0),
            (centre_x - 0.5 * size_x, centre_y + 0.5 * size_y, 0.0),
            (centre_x - 0.5 * size_x, centre_y - 0.5 * size_y, 0.0),
        ),
        dtype=float,
    )


def _virtual_region_outlines(sample) -> tuple[np.ndarray, ...]:
    rows = []
    for raw in tuple(getattr(sample, "virtual_regions", ()) or ()):
        if not isinstance(raw, dict) or not bool(raw.get("enabled", True)):
            continue
        kind = str(raw.get("kind", "rectangle")).strip().lower()
        centre_x = float(raw.get("centre_x_nm", 0.0))
        centre_y = float(raw.get("centre_y_nm", 0.0))
        size_x = float(raw.get("size_x_nm", getattr(sample, "size_x_nm", 0.0)))
        size_y = float(raw.get("size_y_nm", getattr(sample, "size_y_nm", 0.0)))
        angle = math.radians(float(raw.get("rotation_deg", 0.0)))
        if min(size_x, size_y) <= 0.0:
            continue
        if kind == "ellipse":
            phase = np.linspace(0.0, 2.0 * math.pi, 97)
            local = np.column_stack(
                (0.5 * size_x * np.cos(phase), 0.5 * size_y * np.sin(phase))
            )
        else:
            local = np.asarray(
                (
                    (-0.5 * size_x, -0.5 * size_y),
                    (0.5 * size_x, -0.5 * size_y),
                    (0.5 * size_x, 0.5 * size_y),
                    (-0.5 * size_x, 0.5 * size_y),
                    (-0.5 * size_x, -0.5 * size_y),
                ),
                dtype=float,
            )
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation = np.asarray(((cosine, -sine), (sine, cosine)))
        xy = local @ rotation.T + np.asarray((centre_x, centre_y))
        rows.append(np.column_stack((xy, np.zeros(xy.shape[0]))))
    return tuple(rows)


def _sample_boundary_paths(
    calculation_result,
    scene: SpecimenScene,
    *,
    maximum_paths: int = 160,
) -> list[ScenePath]:
    """Expose cached sample-input/output states without assigning detectors.

    The low-level ray calculation has already propagated every output branch
    through the downstream column.  For this specimen-local page we display
    short field-integrated segments from the cached sample-plane phase space
    and the same registered vector fields as the global solver.
    """

    simulation = getattr(calculation_result, "simulation", None)
    incident = getattr(simulation, "incident", None)
    branches = tuple(getattr(simulation, "branches", {}).values())
    if incident is None or not branches:
        return []

    state = getattr(calculation_result, "state_snapshot", None)
    if state is None:
        return []
    field_diagnostic = sample_axial_field_diagnostic(state)
    field_transport = SpecimenFieldTransport(state)
    length_nm = max(4.0 * float(scene.interacting_thickness_nm), 100.0)
    alive = np.asarray(getattr(incident, "alive", ()), dtype=bool)
    ray_count = int(alive.size)
    valid_indices = np.flatnonzero(alive)
    if valid_indices.size == 0:
        return []

    incident_budget = min(max(int(maximum_paths) // 3, 1), valid_indices.size)
    incident_indices = valid_indices[
        np.linspace(0, valid_indices.size - 1, incident_budget, dtype=int)
    ]
    rows: list[ScenePath] = []
    for index in incident_indices:
        x_nm = float(incident.x[-1, index]) * 1.0e9
        y_nm = float(incident.y[-1, index]) * 1.0e9
        tx = float(incident.tx[-1, index])
        ty = float(incident.ty[-1, index])
        direction = np.asarray((tx, ty, 1.0), dtype=float)
        direction /= np.linalg.norm(direction)
        energy_ev = float(state.beam_voltage_kv) * 1000.0 + float(
            incident.energy_offset_ev[index]
        )
        points, _ = field_transport.plane_polyline(
            (x_nm, y_nm, 0.0),
            direction,
            -length_nm,
            energy_ev=energy_ev,
        )
        points = points[::-1]
        if np.all(np.isfinite(points)):
            rows.append(
                ScenePath(
                    points,
                    "incident",
                    "cached specimen-input state; shared vector field",
                )
            )

    remaining = max(int(maximum_paths) - len(rows), 1)
    branch_indices = np.arange(len(branches), dtype=int)
    if branch_indices.size > remaining:
        branch_indices = branch_indices[
            np.linspace(0, branch_indices.size - 1, remaining, dtype=int)
        ]
    per_branch = max(1, remaining // max(branch_indices.size, 1))
    for branch_index in branch_indices:
        branch = branches[int(branch_index)]
        branch_x = np.asarray(getattr(branch, "x", ()), dtype=float)
        if branch_x.ndim != 2 or branch_x.shape[1] != ray_count:
            continue
        candidates = valid_indices
        if candidates.size > per_branch:
            candidates = candidates[
                np.linspace(0, candidates.size - 1, per_branch, dtype=int)
            ]
        kind = str(getattr(branch, "interaction_kind", "")).lower()
        direct = kind in {
            "transmitted",
            "vacuum",
            "virtual_interactions_disabled",
        } or str(getattr(branch, "name", "")) == "000"
        category = "downstream_primary" if direct else "downstream_elastic"
        for index in candidates:
            x_nm = float(branch.x[0, index]) * 1.0e9
            y_nm = float(branch.y[0, index]) * 1.0e9
            tx = float(branch.tx[0, index])
            ty = float(branch.ty[0, index])
            direction = np.asarray((tx, ty, 1.0), dtype=float)
            direction /= np.linalg.norm(direction)
            energy_ev = float(state.beam_voltage_kv) * 1000.0 + float(
                branch.energy_offset_ev[index]
            )
            points, _ = field_transport.plane_polyline(
                (x_nm, y_nm, 0.0),
                direction,
                length_nm,
                energy_ev=energy_ev,
            )
            if np.all(np.isfinite(points)):
                rows.append(
                    ScenePath(
                        points,
                        category,
                        "cached specimen-exit state; shared vector field; "
                        "detector not assigned",
                    )
                )
    return rows


def build_sample_interaction_scene(
    calculation_result,
    sample_region_result=None,
) -> SampleInteractionScene:
    """Convert cached solver outputs into a physically scaled 3-D scene."""

    state = getattr(calculation_result, "state_snapshot", None)
    sample = getattr(state, "sample", None)
    sample_z_mm = float(getattr(sample, "z_mm", 0.0))
    if sample is None and sample_region_result is not None:
        sample_z_mm = 0.5 * (
            float(sample_region_result.entry_z_mm)
            + float(sample_region_result.exit_z_mm)
        )
    thickness_nm = max(float(getattr(sample, "thickness_nm", 0.0)), 0.0)
    interactions = getattr(calculation_result, "specimen_interactions", None)
    if sample_region_result is not None:
        interactions = getattr(sample_region_result, "interactions", interactions)
    specimen_scene = getattr(interactions, "scene", None)
    if specimen_scene is None:
        if state is None or sample is None:
            raise ValueError(
                "A cached state snapshot is required for the sample view."
            )
        specimen_scene = SpecimenScene.from_state(state)
    field_diagnostic = sample_axial_field_diagnostic(state)

    paths: list[ScenePath] = []
    if sample_region_result is not None:
        category_by_kind = {
            "boundary_input": "incident",
            "primary_material": "primary",
            "elastic_rutherford": "elastic",
            "backscattered": "backscattered",
        }
        for path in tuple(getattr(sample_region_result, "electron_paths", ())):
            category = category_by_kind.get(str(getattr(path, "kind", "")))
            # Secondary electrons are intentionally absent: the project has no
            # validated secondary yield/energy/escape model.
            if category is None:
                continue
            paths.append(
                ScenePath(
                    _global_mm_to_local_nm(path.positions_mm, sample_z_mm),
                    category,
                    str(getattr(path, "provenance", "")),
                )
            )
        paths.extend(
            _downstream_scene_paths(sample_region_result, sample_z_mm)
        )
        boundary_length_nm = max(
            abs(float(sample_region_result.entry_z_mm) - sample_z_mm),
            abs(float(sample_region_result.exit_z_mm) - sample_z_mm),
        ) * 1.0e6
        photon_display_length_nm = max(boundary_length_nm, 100.0)
        photon_transport = getattr(
            sample_region_result, "photon_transport", None
        )
        transport_paths = tuple(getattr(photon_transport, "paths", ()))
        if transport_paths:
            for transport_path in transport_paths:
                photon = transport_path.photon
                origin_mm = np.asarray(photon.origin_mm, dtype=float)
                hit_mm = getattr(transport_path, "detector_hit_mm", None)
                terminal_status = str(
                    getattr(transport_path, "terminal_status", "")
                )
                if hit_mm is not None:
                    endpoint_mm = np.asarray(hit_mm, dtype=float)
                    endpoint_status = "sourced detector-face intersection"
                else:
                    hard_intercepts = tuple(
                        interval
                        for interval in tuple(
                            getattr(transport_path, "material_intervals", ())
                        )
                        if bool(getattr(interval, "hard_shadow", False))
                    )
                    if hard_intercepts:
                        distance_mm = min(
                            float(interval.entry_distance_mm)
                            for interval in hard_intercepts
                        )
                        endpoint_mm = origin_mm + distance_mm * np.asarray(
                            photon.direction, dtype=float
                        )
                        endpoint_status = "physical shadow intercept"
                    else:
                        endpoint_mm = origin_mm + (
                            photon_display_length_nm * 1.0e-6
                            * np.asarray(photon.direction, dtype=float)
                        )
                        endpoint_status = (
                            "aggregate direction; detector face unavailable"
                        )
                local = _global_mm_to_local_nm(
                    np.asarray((origin_mm, endpoint_mm)), sample_z_mm
                )
                category = (
                    "xray_detected"
                    if float(getattr(transport_path, "detected_weight", 0.0))
                    > 0.0
                    else "xray_generated"
                )
                paths.append(ScenePath(
                    local,
                    category,
                    f"{getattr(transport_path, 'transport_mode', '')}; "
                    f"{terminal_status}; {endpoint_status}",
                ))
        else:
            # Compatibility with older cached results.  These endpoints are
            # clipped direction representatives, not physical sensor hits.
            for path in tuple(
                getattr(sample_region_result, "photon_paths", ())
            ):
                local = _global_mm_to_local_nm(
                    path.positions_mm, sample_z_mm
                )
                local = _clip_photon_path(local, photon_display_length_nm)
                category = (
                    "xray_detected" if bool(path.detected)
                    else "xray_generated"
                )
                paths.append(ScenePath(
                    local,
                    category,
                    str(getattr(path, "provenance", "")),
                ))
    elif interactions is not None:
        elastic = getattr(interactions, "elastic_transport", None)
        trajectories = tuple(getattr(elastic, "trajectories", ()))
        for trajectory in trajectories:
            points = np.asarray(trajectory.points_nm, dtype=float)
            if points.shape[0] < 2:
                continue
            outcome = str(getattr(trajectory, "outcome", ""))
            if outcome == "backscattered":
                category = "backscattered"
            elif tuple(getattr(trajectory, "events", ())):
                category = "elastic"
            else:
                category = "primary"
            paths.append(
                ScenePath(points, category, "cached elastic Monte Carlo")
            )

        bundle = getattr(interactions, "incident_bundle", None)
        rays = tuple(getattr(bundle, "rays", ()))
        if len(rays) > 128:
            indices = np.linspace(0, len(rays) - 1, 128, dtype=int)
            rays = tuple(rays[index] for index in indices)
        upstream_length_nm = max(4.0 * thickness_nm, 100.0)
        field_transport = SpecimenFieldTransport(state)
        top_z_nm = -0.5 * thickness_nm
        for ray in rays:
            direction = np.asarray(ray.direction, dtype=float)
            if direction.shape != (3,) or direction[2] <= 1.0e-12:
                continue
            sample_xy = np.asarray(ray.position_xy_nm, dtype=float)
            reference_to_top, top_direction = field_transport.plane_polyline(
                (*sample_xy, 0.0),
                direction,
                top_z_nm,
                energy_ev=ray.kinetic_energy_ev,
            )
            top_position = reference_to_top[-1]
            top_to_start, _ = field_transport.plane_polyline(
                top_position,
                top_direction,
                top_z_nm - upstream_length_nm,
                energy_ev=ray.kinetic_energy_ev,
            )
            paths.append(
                ScenePath(
                    top_to_start[::-1],
                    "incident",
                    "cached sample-plane phase space; shared vector field",
                )
            )

    if not paths:
        paths.extend(_sample_boundary_paths(calculation_result, specimen_scene))

    event_groups: dict[str, list[tuple[float, float, float]]] = {
        key: [] for key in EVENT_STYLES
    }
    for event in tuple(getattr(interactions, "events", ())):
        position = tuple(float(value) for value in event.position_nm)
        if len(position) == 3 and all(math.isfinite(value) for value in position):
            event_groups[_event_category(event.process)].append(position)
    events = tuple(
        SceneEvents(np.asarray(values, dtype=float).reshape((-1, 3)), category)
        for category, values in event_groups.items()
        if values
    )

    material_categories = {"primary", "elastic", "backscattered"}
    material_points = [
        path.positions_nm for path in paths if path.category in material_categories
    ] + [event.positions_nm for event in events]
    if not material_points:
        material_points = [
            path.positions_nm
            for path in paths
            if path.category not in {"xray_generated", "xray_detected"}
        ]
    material_bounds = _bounds(
        material_points,
        max(0.5 * thickness_nm, 10.0),
    )
    region_bounds = _bounds(
        [path.positions_nm for path in paths]
        + [event.positions_nm for event in events],
        max(0.5 * thickness_nm, 100.0),
    )
    sample_bounds = _sample_model_bounds(specimen_scene)
    return SampleInteractionScene(
        paths=tuple(paths),
        events=events,
        material_bounds_nm=material_bounds,
        region_bounds_nm=region_bounds,
        sample_thickness_nm=thickness_nm,
        coherent_wave_available=getattr(calculation_result, "wave_imaging", None)
        is not None,
        has_bounded_result=sample_region_result is not None,
        specimen_mode=specimen_scene.mode,
        specimen_source_key=specimen_scene.source_key,
        specimen_is_vacuum=specimen_scene.is_vacuum,
        sample_envelope_shape=specimen_scene.envelope_shape,
        sample_centre_xy_nm=specimen_scene.centre_xy_nm,
        sample_size_xy_nm=specimen_scene.size_xy_nm,
        sample_bounds_nm=sample_bounds,
        virtual_region_outlines_nm=(
            _virtual_region_outlines(sample)
            if (
                specimen_scene.mode == "virtual"
                and not specimen_scene.is_vacuum
                and sample is not None
            )
            else ()
        ),
        sample_axial_field_t=field_diagnostic.total_field_t,
        sample_objective_field_t=field_diagnostic.objective_field_t,
        sample_field_face_variation_t=field_diagnostic.face_variation_t,
        sample_field_transport_model=field_diagnostic.transport_model,
        sample_field_geometry_material_coupled=(
            field_diagnostic.geometry_material_coupled
        ),
    )


def _focus_diagnostic_text(calculation_result) -> str:
    """Describe the already-calculated sample-plane focus without retracing."""

    stem_scan = getattr(calculation_result, "stem_scan", None)
    stem_metrics = getattr(stem_scan, "metrics", {}) or {}
    if "probe_effective_defocus_nm" in stem_metrics:
        effective = float(stem_metrics["probe_effective_defocus_nm"])
        waist = float(stem_metrics.get("probe_ray_waist_offset_nm", 0.0))
        configured = float(
            stem_metrics.get("probe_configured_defocus_nm", 0.0)
        )
        return (
            f"STEM effective probe defocus {effective:+.6g} nm "
            f"(traced waist {waist:+.6g} nm; additional C1 "
            f"{configured:+.6g} nm)."
        )

    wave = getattr(calculation_result, "wave_imaging", None)
    wave_metrics = getattr(wave, "metrics", {}) or {}
    if "waist_offset_m" in wave_metrics:
        waist_nm = float(wave_metrics["waist_offset_m"]) * 1.0e9
        curvature = float(
            wave_metrics.get("radial_wavefront_curvature_per_m", 0.0)
        )
        return (
            f"TEM illumination waist offset {waist_nm:+.6g} nm; "
            f"sample-plane radial wavefront curvature {curvature:+.6g} m⁻¹."
        )
    return ""


def _beam_model_diagnostic_text(calculation_result) -> str:
    """Summarise cached beam coordinates without launching another trace."""

    state = getattr(calculation_result, "state_snapshot", None)
    simulation = getattr(calculation_result, "simulation", None)
    incident = getattr(simulation, "incident", None)
    if state is None or incident is None:
        return ""
    alive = np.asarray(getattr(incident, "alive", ()), dtype=bool)
    energy_offsets = np.asarray(
        getattr(incident, "energy_offset_ev", ()), dtype=float
    )
    energy_text = ""
    if energy_offsets.size and np.all(np.isfinite(energy_offsets)):
        energy_text = (
            f"; ΔE {float(np.min(energy_offsets)):+.4g} to "
            f"{float(np.max(energy_offsets)):+.4g} eV"
        )
    emitter = getattr(getattr(state, "electron_gun", None), "emitter", None)
    source_parts = []
    for attribute, label, unit in (
        ("virtual_source_fwhm_nm", "source FWHM", "nm"),
        ("angular_rms_mrad", "angular RMS", "mrad"),
        ("angular_cutoff_mrad", "angular cutoff", "mrad"),
        ("energy_spread_fwhm_ev", "energy FWHM", "eV"),
    ):
        value = getattr(emitter, attribute, None)
        if value is not None and math.isfinite(float(value)):
            source_parts.append(f"{label} {float(value):.4g} {unit}")
    source_text = "; ".join(source_parts)
    if source_text:
        source_text = "; " + source_text
    return (
        f"Beam cache: {int(np.count_nonzero(alive)):,}/{alive.size:,} rays at "
        f"{float(state.beam_voltage_kv):.6g} kV; each carries x, y, θx, θy, "
        f"ΔE and current weight{energy_text}{source_text}."
    )


def _aberration_scope_text(calculation_result) -> str:
    state = getattr(calculation_result, "state_snapshot", None)
    if state is None:
        return ""
    chromatic = (
        "enabled" if bool(getattr(state, "chromatic_aberration_enabled", False))
        else "disabled"
    )
    return (
        "Aberration scope: geometric rays include continuous round-lens, "
        "stigmator/corrector multipole fields and each lens C3(Cs) once; "
        f"objective Cc energy-dependent defocus is {chromatic}. Coherent "
        "TEM/STEM wave imaging applies C1, A1, B2, A2, C3, S3, A3 and C5; "
        "Cc is handled through energy-dependent defocus/temporal coherence, "
        "not as a single-energy phase term."
    )


class SampleInteractions3DPage(QWidget):
    """Rotate and inspect cached interactions near the active sample."""

    sample_region_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("sampleInteractions3DPage")
        self._calculation_result = None
        self._sample_region_result = None
        self._scene: SampleInteractionScene | None = None
        self._rendered_interactions = None
        self._items = []
        self._fit_scope = "material"
        self._signal_filter_guard = False
        self._hidden_scene = None
        self._hidden_view_state = None
        self._pending_view_restore = None
        self._view_states = {}
        self.parameters_panel = None

        self.summary = QLabel(
            "Run High accuracy to populate cached specimen trajectories."
        )
        self.summary.setObjectName("sampleInteractions3DSummary")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.summary.setStyleSheet("color: #94a3b8; font-weight: 600;")

        self.signal_filter = QPushButton()
        self.signal_filter.setObjectName("sampleInteractionsSignalFilter")
        self.signal_filter.setToolTip(
            "Choose individual cached electron, X-ray and interaction-site "
            "categories to display. Filtering never reruns the calculation."
        )
        self.signal_menu = QMenu(self.signal_filter)
        self.signal_actions: dict[str, QCheckBox] = {}
        self._signal_widget_actions: list[QWidgetAction] = []
        self.show_all_signals = self.signal_menu.addAction("Show all signals")
        self.hide_all_signals = self.signal_menu.addAction("Hide all signals")
        self.show_all_signals.triggered.connect(
            lambda: self._set_all_signal_visibility(True)
        )
        self.hide_all_signals.triggered.connect(
            lambda: self._set_all_signal_visibility(False)
        )
        self.signal_menu.addSeparator()
        all_styles = {**PATH_STYLES, **EVENT_STYLES}
        for group_index, (group_name, categories) in enumerate(SIGNAL_GROUPS):
            if group_index:
                self.signal_menu.addSeparator()
            heading = self.signal_menu.addAction(group_name)
            heading.setEnabled(False)
            for category in categories:
                label, _colour = all_styles[category]
                toggle = QCheckBox(label)
                toggle.setObjectName(
                    f"sampleInteractionsSignal_{category}"
                )
                toggle.setChecked(True)
                toggle.toggled.connect(self._signal_visibility_changed)
                widget_action = QWidgetAction(self.signal_menu)
                widget_action.setDefaultWidget(toggle)
                self.signal_menu.addAction(widget_action)
                self._signal_widget_actions.append(widget_action)
                self.signal_actions[category] = toggle
        self.signal_filter.setMenu(self.signal_menu)
        self._update_signal_filter_text()

        self.context_toggle = QCheckBox("Sample-plane context")
        self.context_toggle.setChecked(True)
        self.context_toggle.toggled.connect(
            lambda _checked=False: self._redraw(refit=False)
        )

        self.calculate_paths = QPushButton("Calculate detailed paths + X-rays")
        self.calculate_paths.setObjectName("sampleInteractions3DCalculate")
        self.calculate_paths.setEnabled(False)
        self.calculate_paths.setToolTip(
            "Explicitly calculate only missing bounded sample/EDS observables; "
            "cached High-accuracy interactions are reused."
        )
        self.calculate_paths.clicked.connect(self.sample_region_requested.emit)
        self.fit_material = QPushButton("Fit material")
        self.fit_material.setToolTip(
            "Fit the nanometre-scale material trajectories and event sites."
        )
        self.fit_region = QPushButton("Fit interaction region")
        self.fit_region.setToolTip(
            "Fit the complete cached entry/exit region and clipped display "
            "length of X-ray direction lines."
        )
        self.fit_sample = QPushButton("Fit user sample")
        self.fit_sample.setToolTip(
            "Fit the complete user-defined sample envelope. This may be much "
            "larger than the nanometre-scale interaction region."
        )
        self.fit_material.clicked.connect(lambda: self._fit("material"))
        self.fit_region.clicked.connect(lambda: self._fit("region"))
        self.fit_sample.clicked.connect(lambda: self._fit("sample"))

        visibility_controls = QHBoxLayout()
        visibility_controls.addWidget(self.signal_filter)
        visibility_controls.addWidget(self.context_toggle)
        visibility_controls.addStretch(1)
        action_controls = QGridLayout()
        for index, widget in enumerate((
            self.calculate_paths,
            self.fit_material,
            self.fit_region,
            self.fit_sample,
        )):
            action_controls.addWidget(widget, index // 2, index % 2)
        action_controls.setColumnStretch(2, 1)

        self.legend = QLabel(self._legend_html())
        self.legend.setObjectName("sampleInteractions3DLegend")
        self.legend.setTextFormat(Qt.TextFormat.RichText)
        self.legend.setWordWrap(True)
        self.legend.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.legend.setToolTip(
            "\n".join(
                f"{LEGEND_LABELS[category]}: {label}"
                for category, (label, _colour) in (
                    *PATH_STYLES.items(),
                    *EVENT_STYLES.items(),
                )
            )
        )

        self.opengl_available = False
        self.opengl_detail = OPENGL_IMPORT_ERROR
        platform_name = str(QGuiApplication.platformName()).lower()
        platform_supports_gl = platform_name not in {"offscreen", "minimal"}
        if gl is not None and platform_supports_gl:
            try:
                self.view = gl.GLViewWidget(self)
                self.view.setObjectName("sampleInteractionsOpenGlView")
                self.view.setBackgroundColor(QColor("#050816"))
                self.opengl_available = True
                self.opengl_detail = "pyqtgraph.opengl / PyOpenGL"
            except Exception as exc:  # pragma: no cover - driver dependent
                self.opengl_detail = f"OpenGL initialisation failed: {exc}"
                self.view = self._fallback_plot()
        else:
            if not platform_supports_gl:
                self.opengl_detail = (
                    f"Qt platform {platform_name!r} has no supported OpenGL widget"
                )
            self.view = self._fallback_plot()
        self.view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.view.setToolTip(
            "Physical path order is upstream -Z to downstream +Z. The 3-D "
            "renderer maps physical +Z downward in the default side view, so "
            "the incident beam travels top-to-bottom. Orbiting changes only "
            "the camera."
        )

        self._gl_view = self.view if self.opengl_available else None
        self._projection_view = (
            self._fallback_plot() if self.opengl_available else self.view
        )
        self._view_mode = "3d" if self.opengl_available else "xz"
        self.view_stack = QStackedWidget()
        self.view_stack.setObjectName("sampleInteractionViewStack")
        if self._gl_view is not None:
            self.view_stack.addWidget(self._gl_view)
        self.view_stack.addWidget(self._projection_view)
        self.view_stack.setCurrentWidget(self.view)

        view_controls = QHBoxLayout()
        view_controls.addWidget(QLabel("View"))
        self.view_buttons = {}
        self.view_button_group = QButtonGroup(self)
        self.view_button_group.setExclusive(True)
        for mode, label in (("3d", "3D"), ("xz", "X-Z"), ("yz", "Y-Z")):
            button = QPushButton(label)
            button.setObjectName(f"sampleInteractionView_{mode}")
            button.setCheckable(True)
            button.setChecked(mode == self._view_mode)
            button.setEnabled(mode != "3d" or self.opengl_available)
            button.setToolTip(
                "Rotate the cached 3-D scene. Requires OpenGL."
                if mode == "3d" else
                f"Orthogonal {label} projection of the same cached scene; +Z downward."
            )
            button.clicked.connect(
                lambda _checked=False, selected=mode: self.set_view_mode(selected)
            )
            self.view_button_group.addButton(button)
            self.view_buttons[mode] = button
            view_controls.addWidget(button)
        view_controls.addStretch(1)
        self.parameters_toggle = QPushButton("Parameters")
        self.parameters_toggle.setObjectName("sampleInteractionParametersToggle")
        self.parameters_toggle.setCheckable(True)
        self.parameters_toggle.setEnabled(False)
        self.parameters_toggle.setToolTip("Show sample transport and EDS settings.")
        self.parameters_toggle.toggled.connect(self._set_parameters_visible)
        view_controls.addWidget(self.parameters_toggle)

        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.content_splitter.setObjectName("sampleInteractionContentSplitter")
        self.content_splitter.setHandleWidth(7)
        self.content_splitter.addWidget(self.view_stack)
        self.content_splitter.setStretchFactor(0, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(self.summary)
        layout.addLayout(visibility_controls)
        layout.addLayout(action_controls)
        layout.addLayout(view_controls)
        layout.addWidget(self.legend)
        layout.addWidget(self.content_splitter, 1)

    @property
    def _using_gl(self) -> bool:
        return self.opengl_available and self._view_mode == "3d"

    @property
    def _projection_axis(self) -> int:
        return 1 if self._view_mode == "yz" else 0

    def set_parameters_widget(self, widget: QWidget) -> None:
        """Host the existing controls, without duplicating settings or solvers."""
        if self.parameters_panel is not None:
            raise ValueError("Sample interaction parameters are already installed")
        self.parameters_panel = widget
        self.content_splitter.addWidget(widget)
        self.content_splitter.setStretchFactor(1, 0)
        self.content_splitter.setSizes((900, 380))
        widget.hide()
        self.parameters_toggle.setEnabled(True)

    def _set_parameters_visible(self, visible: bool) -> None:
        if self.parameters_panel is not None:
            self.parameters_panel.setVisible(visible)

    def set_view_mode(self, mode: str) -> None:
        """Change only the renderer; never build a new interaction scene."""
        if mode not in {"3d", "xz", "yz"}:
            raise ValueError(f"Unknown sample view: {mode}")
        if mode == "3d" and not self.opengl_available:
            return
        if mode == self._view_mode:
            return
        previous_mode = self._view_mode
        previous_view = self._capture_view_state()
        self._view_states[previous_mode] = previous_view
        self._pending_view_restore = None
        self._clear_items()
        self._view_mode = mode
        self.view = self._gl_view if self._using_gl else self._projection_view
        self.view_stack.setCurrentWidget(self.view)
        self.view_buttons[mode].setChecked(True)
        self._redraw(refit=False)
        saved = self._view_states.get(mode)
        if saved is None and previous_mode != "3d" and mode != "3d":
            saved = {**previous_view, "mode": mode}
        if saved is not None:
            self._restore_view_state(saved)
        else:
            self._apply_fit()

    def _legend_html(self) -> str:
        visible = {
            category
            for category, toggle in self.signal_actions.items()
            if toggle.isChecked()
        }
        entries = [
            f'<span style="color:{colour}">●</span> {LEGEND_LABELS[category]}'
            for category, (label, colour) in (
                *PATH_STYLES.items(),
                *EVENT_STYLES.items(),
            )
            if category in visible
        ]
        if not entries:
            return '<span style="color:#94a3b8">No signal types selected.</span>'
        return "&nbsp;&nbsp; ".join(entries)

    @property
    def visible_signal_categories(self) -> frozenset[str]:
        return frozenset(
            category
            for category, toggle in self.signal_actions.items()
            if toggle.isChecked()
        )

    def _update_signal_filter_text(self) -> None:
        visible_count = sum(
            toggle.isChecked() for toggle in self.signal_actions.values()
        )
        self.signal_filter.setText(
            f"Visible signals: {visible_count}/{len(self.signal_actions)}"
        )

    def _signal_visibility_changed(self, _checked=False) -> None:
        if self._signal_filter_guard:
            return
        self._update_signal_filter_text()
        if hasattr(self, "legend"):
            self.legend.setText(self._legend_html())
        self._redraw(refit=False)

    def _set_all_signal_visibility(self, visible: bool) -> None:
        self._signal_filter_guard = True
        try:
            for toggle in self.signal_actions.values():
                toggle.setChecked(bool(visible))
        finally:
            self._signal_filter_guard = False
        self._signal_visibility_changed()

    @property
    def scene_snapshot(self) -> SampleInteractionScene | None:
        return self._scene

    def _capture_view_state(self):
        """Capture display state only; no specimen data are copied or rebuilt."""

        if self._using_gl:
            options = self.view.opts
            centre = options.get("center", QVector3D())
            return {
                "kind": "opengl",
                "mode": self._view_mode,
                "center": (
                    float(centre.x()),
                    float(centre.y()),
                    float(centre.z()),
                ),
                "distance": float(options.get("distance", 10.0)),
                "fov": float(options.get("fov", 60.0)),
                "elevation": float(options.get("elevation", 30.0)),
                "azimuth": float(options.get("azimuth", 45.0)),
            }
        ranges = self.view.getViewBox().viewRange()
        return {
            "kind": "fallback-2d",
            "mode": self._view_mode,
            "x_range": tuple(float(value) for value in ranges[0]),
            "y_range": tuple(float(value) for value in ranges[1]),
        }

    def _restore_view_state(self, state) -> None:
        if not state:
            return
        if state.get("mode", self._view_mode) != self._view_mode:
            return
        if self._using_gl and state.get("kind") == "opengl":
            self.view.opts["center"] = QVector3D(*state["center"])
            self.view.opts["fov"] = float(state["fov"])
            self.view.setCameraPosition(
                distance=float(state["distance"]),
                elevation=float(state["elevation"]),
                azimuth=float(state["azimuth"]),
            )
            self.view.update()
        elif not self._using_gl and state.get("kind") == "fallback-2d":
            self.view.setRange(
                xRange=state["x_range"],
                yRange=state["y_range"],
                padding=0.0,
            )

    def hideEvent(self, event) -> None:
        self._hidden_scene = self._scene
        self._hidden_view_state = self._capture_view_state()
        self._pending_view_restore = None
        super().hideEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if (
            self._scene is not None
            and self._hidden_scene is self._scene
            and self._hidden_view_state is not None
        ):
            self._pending_view_restore = (
                self._hidden_scene,
                self._hidden_view_state,
            )
            QTimer.singleShot(0, self._restore_cached_view_after_show)

    def _restore_cached_view_after_show(self) -> None:
        pending = self._pending_view_restore
        self._pending_view_restore = None
        if pending is None or not self.isVisible():
            return
        scene, view_state = pending
        if scene is not self._scene:
            return
        # A QOpenGLWidget context may be recreated after its tab was hidden.
        # Re-add display items from the exact same immutable scene snapshot;
        # never rebuild or resample the scientific calculation here.
        if self._using_gl:
            self._redraw(refit=False)
        self._restore_view_state(view_state)

    def _fallback_plot(self):
        plot = pg.PlotWidget(background="#050816")
        plot.setObjectName("sampleInteractionsFallback2DView")
        plot.setLabel("bottom", "Local X", units="nm")
        plot.setLabel("left", "Local Z (+ downstream ↓)", units="nm")
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.getViewBox().setAspectLocked(True)
        plot.getViewBox().invertY(True)
        return plot

    def display_result(self, calculation_result) -> None:
        changed = calculation_result is not self._calculation_result
        current_interactions = getattr(
            calculation_result, "specimen_interactions", None
        )
        interactions_changed = (
            current_interactions is not self._rendered_interactions
        )
        if changed:
            self._sample_region_result = None
        self._calculation_result = calculation_result
        if not changed and not interactions_changed:
            return
        self._refresh_scene()

    def mark_result_stale(self) -> None:
        """Retain the last complete 3-D scene as an explicitly stale view."""

        if self._calculation_result is None:
            return
        self.calculate_paths.setEnabled(False)
        self.summary.setText(
            "Previous sample-interaction scene retained | inputs changed"
        )
        self.summary.setToolTip(
            "Run High accuracy to bind the scene to the current microscope "
            "and specimen state. Rotation and signal filtering remain display-only."
        )

    def set_sample_region_result(self, result) -> None:
        if result is self._sample_region_result:
            return
        self._sample_region_result = result
        self._refresh_scene()

    def _refresh_scene(self) -> None:
        if self._calculation_result is None:
            self._scene = None
            self._rendered_interactions = None
            self._clear_items()
            self.summary.setText(
                "Run High accuracy to populate cached specimen trajectories."
            )
            return
        self._scene = build_sample_interaction_scene(
            self._calculation_result,
            self._sample_region_result,
        )
        self._rendered_interactions = getattr(
            self._calculation_result, "specimen_interactions", None
        )
        scene = self._scene
        state = getattr(self._calculation_result, "state_snapshot", None)
        sample = getattr(state, "sample", None)
        detailed_available = bool(
            sample is not None
            and not scene.specimen_is_vacuum
            and bool(getattr(sample, "eds_enabled", False))
        )
        self.calculate_paths.setEnabled(detailed_available)
        electron_count = sum(
            path.category not in {"xray_generated", "xray_detected"}
            for path in scene.paths
        )
        xray_count = sum(
            path.category in {"xray_generated", "xray_detected"}
            for path in scene.paths
        )
        event_count = sum(len(group.positions_nm) for group in scene.events)
        specimen_label = (
            "Vacuum reference"
            if scene.specimen_is_vacuum
            else "Virtual TOML sample"
            if scene.specimen_mode == "virtual"
            else "Real imported CIF sample"
        )
        parts = [
            (
                "Active specimen: "
                + specimen_label
                + f" ({scene.specimen_source_key}); "
                + f"{scene.sample_envelope_shape} "
                + f"{scene.sample_size_xy_nm[0]:.6g} × "
                + f"{scene.sample_size_xy_nm[1]:.6g} × "
                + f"{scene.sample_thickness_nm:.6g} nm."
            ),
            f"Cached local view: {electron_count:,} electron paths, "
            f"{event_count:,} interaction sites, {xray_count:,} X-ray paths."
        ]
        focus_text = _focus_diagnostic_text(self._calculation_result)
        if focus_text:
            parts.append(focus_text)
        beam_text = _beam_model_diagnostic_text(self._calculation_result)
        if beam_text:
            parts.append(beam_text)
        parts.append(
            f"Specimen field: total Bz {scene.sample_axial_field_t:+.6g} T; "
            f"Objective contribution {scene.sample_objective_field_t:+.6g} T; "
            f"face-to-face ΔBz {scene.sample_field_face_variation_t:.3g} T. "
            "Finite specimen flights use shared vector-field transport "
            "with no magnetic energy change."
        )
        aberration_text = _aberration_scope_text(self._calculation_result)
        if aberration_text:
            parts.append(aberration_text)
        if scene.coherent_wave_available:
            parts.append(
                "Coherent diffraction/channeling is retained as a wave result "
                "and is not misdrawn as a classical particle track."
            )
        if scene.virtual_region_outlines_nm:
            parts.append(
                f"The view includes {len(scene.virtual_region_outlines_nm)} "
                "enabled user-defined virtual-density region(s)."
            )
        if not scene.has_bounded_result and detailed_available:
            parts.append(
                "Use Calculate detailed paths + X-rays for the explicit "
                "entry/exit region and EDS photon directions."
            )
        elif scene.specimen_is_vacuum:
            parts.append(
                "Vacuum preserves the user reference plane but generates no "
                "sample scattering, ionisation, or characteristic X-rays."
            )
        photon_transport = getattr(
            self._sample_region_result, "photon_transport", None
        )
        if photon_transport is not None:
            metrics = dict(getattr(photon_transport, "metrics", {}) or {})
            geometry_label = (
                "exact detector faces"
                if bool(getattr(photon_transport, "geometry_complete", False))
                else "aggregate solid angle"
            )
            hit_count = int(metrics.get("detector_hit_count", 0))
            blocked_count = int(metrics.get("blocked_photon_count", 0))
            specimen_count = int(metrics.get("specimen_intersection_count", 0))
            mean_transmission = float(
                metrics.get("mean_specimen_transmission", 1.0)
            )
            parts.append(
                f"EDS photon transport: {geometry_label}; {hit_count} hit, "
                f"{blocked_count} shadowed; specimen self-absorption crossed "
                f"by {specimen_count} path segment(s), mean transmission "
                f"{mean_transmission:.4g}."
            )
        parts.append(
            "Electron lines here are specimen-input/output states, not "
            "HAADF/DF/BF counts. Electron signal is counted only after "
            "downstream lens transport and physical detector intersection."
        )
        parts.append(
            "Coordinates are specimen-local nm and path order is upstream "
            "-Z to downstream +Z. The display maps physical +Z downward in "
            "the default side view, so the incident beam appears top-to-bottom; "
            "rotating, zooming, filtering, and fitting only redraw this cache."
        )
        detail_text = " ".join(parts)
        self.summary.setText(
            f"{specimen_label} ({scene.specimen_source_key}) | "
            f"{scene.sample_envelope_shape} "
            f"{scene.sample_size_xy_nm[0]:.6g} × "
            f"{scene.sample_size_xy_nm[1]:.6g} × "
            f"{scene.sample_thickness_nm:.6g} nm | "
            f"{electron_count:,} electron paths · {event_count:,} sites · "
            f"{xray_count:,} X-rays"
            + (
                " · "
                + (
                    "exact-face"
                    if bool(getattr(photon_transport, "geometry_complete", False))
                    else "aggregate"
                )
                + f" · {int(dict(getattr(photon_transport, 'metrics', {}) or {}).get('detector_hit_count', 0))} hit"
                + f" · {int(dict(getattr(photon_transport, 'metrics', {}) or {}).get('blocked_photon_count', 0))} blocked"
                if photon_transport is not None
                else ""
            )
        )
        self.summary.setToolTip(
            detail_text
            + "\n\n"
            "The user-sample outline uses the configured finite envelope; use "
            "Fit user sample to see its complete edge. Exact detector-face "
            "and pole-shadow endpoints are retained when sourced geometry is "
            "available; aggregate-only X-ray directions remain clipped. "
            "X-rays are not deflected by magnetic "
            "lenses. The plotted axial field uses the same runtime provider "
            "as full-column propagation; its measured/FEM or provisional "
            "status is listed above. Secondary-electron paths are not shown "
            "because no validated yield/energy/escape model is implemented."
        )
        self._redraw()

    def _clear_items(self) -> None:
        if self._using_gl:
            for item in self._items:
                try:
                    self.view.removeItem(item)
                except Exception:
                    pass
        else:
            self.view.clear()
        self._items = []

    @staticmethod
    def _rgba(colour: str, alpha: float = 1.0):
        value = QColor(colour)
        return (
            value.redF(),
            value.greenF(),
            value.blueF(),
            float(alpha),
        )

    def _redraw(self, _checked=False, *, refit: bool = True) -> None:
        self._clear_items()
        scene = self._scene
        if scene is None:
            return
        if self._using_gl:
            self._draw_gl(scene)
        else:
            self._draw_2d(scene)
        if refit:
            self._apply_fit()

    def _path_visible(self, category: str) -> bool:
        toggle = self.signal_actions.get(category)
        return toggle is None or toggle.isChecked()

    def _event_visible(self, category: str) -> bool:
        toggle = self.signal_actions.get(category)
        return toggle is None or toggle.isChecked()

    def _draw_gl(self, scene: SampleInteractionScene) -> None:
        for path in scene.paths:
            if not self._path_visible(path.category):
                continue
            _label, colour = PATH_STYLES[path.category]
            item = gl.GLLinePlotItem(
                pos=_gl_display_positions(path.positions_nm),
                color=self._rgba(colour, 0.92),
                width=1.5,
                antialias=True,
                mode="line_strip",
            )
            self.view.addItem(item)
            self._items.append(item)
        for group in scene.events:
            if not self._event_visible(group.category):
                continue
            _label, colour = EVENT_STYLES[group.category]
            item = gl.GLScatterPlotItem(
                pos=_gl_display_positions(group.positions_nm),
                color=self._rgba(colour, 0.95),
                size=6.0,
                pxMode=True,
            )
            self.view.addItem(item)
            self._items.append(item)
        if self.context_toggle.isChecked():
            plane = _sample_model_outline(scene)
            item = gl.GLLinePlotItem(
                pos=_gl_display_positions(plane),
                color=self._rgba("#ffffff", 0.55),
                width=1.3,
                antialias=True,
                mode="line_strip",
            )
            self.view.addItem(item)
            self._items.append(item)
            for outline in scene.virtual_region_outlines_nm:
                item = gl.GLLinePlotItem(
                    pos=_gl_display_positions(outline),
                    color=self._rgba("#a78bfa", 0.75),
                    width=1.2,
                    antialias=True,
                    mode="line_strip",
                )
                self.view.addItem(item)
                self._items.append(item)

    def _draw_2d(self, scene: SampleInteractionScene) -> None:
        axis = self._projection_axis
        self.view.setLabel("bottom", f"Local {'Y' if axis else 'X'}", units="nm")
        for path in scene.paths:
            if not self._path_visible(path.category):
                continue
            _label, colour = PATH_STYLES[path.category]
            self.view.plot(
                path.positions_nm[:, axis],
                path.positions_nm[:, 2],
                pen=pg.mkPen(colour, width=1.4),
            )
        for group in scene.events:
            if not self._event_visible(group.category):
                continue
            _label, colour = EVENT_STYLES[group.category]
            item = pg.ScatterPlotItem(
                x=group.positions_nm[:, axis],
                y=group.positions_nm[:, 2],
                size=6,
                pen=pg.mkPen(colour),
                brush=pg.mkBrush(colour),
            )
            self.view.addItem(item)
        if self.context_toggle.isChecked():
            outline = _sample_model_outline(scene)
            self.view.plot(
                outline[:, axis],
                outline[:, 2],
                pen=pg.mkPen("#ffffff", width=1.2),
            )
            for region_outline in scene.virtual_region_outlines_nm:
                self.view.plot(
                    region_outline[:, axis],
                    region_outline[:, 2],
                    pen=pg.mkPen("#a78bfa", width=1.1),
                )

    def _active_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self._scene is None:
            return _bounds((), 100.0)
        if self._fit_scope == "sample":
            return self._scene.sample_bounds_nm
        if self._fit_scope == "region":
            return self._scene.region_bounds_nm
        return self._scene.material_bounds_nm

    def _fit(self, scope: str) -> None:
        self._fit_scope = (
            scope
            if scope in {"material", "region", "sample"}
            else "material"
        )
        self._redraw()

    def _apply_fit(self) -> None:
        if self._scene is None:
            return
        lower, upper = self._active_bounds()
        span = max(float(np.max(upper - lower)), 1.0)
        if self._using_gl:
            display_lower, display_upper = _gl_display_bounds((lower, upper))
            centre = 0.5 * (display_lower + display_upper)
            self.view.opts["center"] = QVector3D(*centre)
            self.view.setCameraPosition(
                distance=2.1 * span,
                elevation=18.0,
                azimuth=-45.0,
            )
            self.view.update()
        else:
            self.view.setRange(
                xRange=(
                    float(lower[self._projection_axis]),
                    float(upper[self._projection_axis]),
                ),
                yRange=(float(lower[2]), float(upper[2])),
                padding=0.0,
            )
