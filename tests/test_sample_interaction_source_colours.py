"""Display lineage regressions; no detector or scattering physics is changed."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtGui import QColor

from temsim.gui.sample_interactions_3d import (
    PATH_STYLES,
    SampleInteractions3DPage,
    ScenePath,
    _downstream_scene_paths,
    _sample_boundary_paths,
    build_sample_interaction_scene,
)
from temsim.optics.column import default_state
from temsim.specimen.downstream_transport import GeometricSpecimenExit
from temsim.specimen.sample_region import _electron_paths
from temsim.specimen.scene import SpecimenScene


def _result():
    state = default_state()
    state.sample.z_mm = 1.5
    state.sample.thickness_nm = 10.0
    ids = np.asarray((71, 19, 42), dtype=np.int64)
    angles = np.asarray((0.2, 1.3, 5.8))
    incident = SimpleNamespace(
        name="incident",
        z=np.asarray((1.499, 1.5)),
        x=np.asarray(((0.0, 1.0, -1.0), (0.0, 2.0, -2.0))) * 1e-9,
        y=np.asarray(((1.0, -1.0, 1.0), (2.0, -2.0, 2.0))) * 1e-9,
        tx=np.zeros((2, 3)),
        ty=np.zeros((2, 3)),
        alive=np.ones(3, dtype=bool),
        blocked_z=np.full(3, np.nan),
        energy_offset_ev=np.zeros(3),
        ray_weight=np.ones(3),
        source_ray_id=ids,
        source_azimuth_rad=angles,
    )
    trajectory = SimpleNamespace(
        source_ray_index=2,
        points_nm=np.asarray(((-2.0, 2.0, -5.0), (-1.0, 2.0, 5.0))),
        events=(object(),),
        outcome="transmitted",
        incident_weight=0.25,
        initial_energy_ev=300000.0,
    )
    ray = SimpleNamespace(
        source_ray_index=2,
        position_xy_nm=(-2.0, 2.0),
        direction=(0.0, 0.0, 1.0),
        kinetic_energy_ev=300000.0,
    )
    elastic = SimpleNamespace(trajectories=(trajectory,), material_flights=())
    interactions = SimpleNamespace(
        elastic_transport=elastic,
        incident_bundle=SimpleNamespace(rays=(ray,)),
        events=(),
    )
    simulation = SimpleNamespace(incident=incident, branches={}, gun_trace=None)
    return SimpleNamespace(
        state_snapshot=state,
        simulation=simulation,
        specimen_interactions=interactions,
        wave_imaging=None,
        signatures={"sample_downstream": "current"},
    )


def _compact_branch(result):
    incident = result.simulation.incident
    return SimpleNamespace(
        name="compacted",
        interaction_kind="sample_region_elastic",
        z=np.asarray((1.5, 1.50005)),
        x=np.asarray(((-2.0, 0.0), (4.0, 3.0))) * 1e-9,
        y=np.zeros((2, 2)),
        tx=np.full((2, 2), 0.001),
        ty=np.zeros((2, 2)),
        energy_offset_ev=np.asarray((-10.0, 0.0)),
        ray_weight=np.ones(2),
        weight=0.5,
        source_ray_id=incident.source_ray_id[[2, 0]],
        source_azimuth_rad=incident.source_azimuth_rad[[2, 0]],
    )


def _checkpoint(branches=(), *, signature="current"):
    weight = sum(branch.weight for branch in branches)
    return GeometricSpecimenExit(
        tuple(branches),
        {
            "tracked_downstream_source_probability": weight,
            "inelastic_absorbed_source_probability": 1.0 - weight,
        },
        dependency_signature=signature,
    )


def test_scene_keeps_source_identity_through_internal_scatter_and_compact_exit():
    result = _result()
    result.specimen_exit = _checkpoint((_compact_branch(result),))
    scene = build_sample_interaction_scene(result)

    elastic, incident, *downstream = scene.paths
    assert elastic.category == "elastic"
    assert incident.category == "incident"
    assert elastic.source_ray_index == incident.source_ray_index == 2
    assert elastic.source_ray_id == incident.source_ray_id == 42
    assert elastic.source_azimuth_rad == incident.source_azimuth_rad == 5.8
    assert [path.source_ray_id for path in downstream] == [42, 71]
    assert [path.source_azimuth_rad for path in downstream] == [5.8, 0.2]
    with pytest.raises(FrozenInstanceError):
        elastic.source_ray_id = 7


def test_bounded_branch_histories_keep_compacted_source_order():
    result = _result()
    region = SimpleNamespace(
        downstream_branches=(_compact_branch(result),), exit_z_mm=1.50005
    )
    paths = _downstream_scene_paths(region, 1.5, simulation=result.simulation)
    assert [path.source_ray_id for path in paths] == [42, 71]
    assert [path.source_azimuth_rad for path in paths] == [5.8, 0.2]
    np.testing.assert_allclose(paths[0].positions_nm[:, 0], (-2.0, 4.0))


def test_empty_physical_exit_does_not_restore_reference_electrons():
    result = _result()
    result.simulation.branches["reference"] = _compact_branch(result)
    result.specimen_exit = _checkpoint()
    scene = SpecimenScene.from_state(result.state_snapshot)
    assert _sample_boundary_paths(result, scene, include_incident=False) == []


def test_sample_region_paths_keep_source_id_not_incident_column_number():
    result = _result()
    paths, *_ = _electron_paths(
        result.state_snapshot,
        result.simulation,
        result.specimen_interactions.elastic_transport,
        entry_z_mm=1.499,
        secondary_count=0,
        rng=np.random.default_rng(14),
    )
    incoming = {path.source_ray_index: path for path in paths if path.kind == "boundary_input"}
    scattered = next(path for path in paths if path.kind == "elastic_rutherford")
    assert scattered.source_ray_index == 2
    assert scattered.source_ray_id == incoming[2].source_ray_id == 42
    assert scattered.source_azimuth_rad == incoming[2].source_azimuth_rad == 5.8
    region = SimpleNamespace(
        electron_paths=paths,
        interactions=result.specimen_interactions,
        downstream_branches=(_compact_branch(result),),
        photon_paths=(),
        entry_z_mm=1.499,
        exit_z_mm=1.50005,
    )
    region.specimen_exit = _checkpoint(region.downstream_branches)
    scene = build_sample_interaction_scene(result, region)
    scatter = next(path for path in scene.paths if path.category == "elastic")
    outgoing = next(path for path in scene.paths if path.category == "downstream_elastic")
    assert scatter.source_ray_index == 2
    assert scatter.source_ray_id == outgoing.source_ray_id == 42
    assert scatter.source_azimuth_rad == outgoing.source_azimuth_rad == 5.8
    assert scene.downstream_source == "Specimen exit"


def test_colour_mode_changes_only_rendering_and_preserves_camera(qtbot, monkeypatch):
    import temsim.gui.sample_interactions_3d as view_module

    result = _result()
    page = SampleInteractions3DPage()
    qtbot.addWidget(page)
    page.display_result(result)
    page.set_view_mode("xz")
    page.view.setRange(xRange=(-20, 20), yRange=(-150, 25), padding=0)
    before = np.asarray(page.view.viewRange())
    scene = page.scene_snapshot
    original = tuple(path.positions_nm.copy() for path in scene.paths)
    expected = QColor.fromHsvF(5.8 / (2 * np.pi), 0.88, 1.0)
    assert page.colour_by.currentText() == "Source position"
    assert page._path_colour(scene.paths[0]) == expected
    assert page._path_colour(scene.paths[1]) == expected
    assert 'color:#fb7185' not in page.legend.text()
    assert 'color:#94a3b8' not in page.legend.text()

    def forbidden(*args, **kwargs):
        raise AssertionError("Colour selection must not rebuild or recalculate")

    monkeypatch.setattr(view_module, "build_sample_interaction_scene", forbidden)
    monkeypatch.setattr(view_module, "SpecimenFieldTransport", forbidden)
    requests = []
    page.sample_region_requested.connect(lambda: requests.append(True))
    page.colour_by.setCurrentIndex(1)
    assert page._path_colour(scene.paths[0]) == QColor(PATH_STYLES["elastic"][1])
    page.colour_by.setCurrentIndex(0)
    assert page.scene_snapshot is scene
    assert not requests
    np.testing.assert_allclose(page.view.viewRange(), before)
    for path, points in zip(scene.paths, original, strict=True):
        np.testing.assert_array_equal(path.positions_nm, points)


def test_unknown_source_stays_neutral_and_photons_keep_category_colour(qtbot):
    page = SampleInteractions3DPage()
    qtbot.addWidget(page)
    positions = np.asarray(((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)))
    unknown = ScenePath(positions, "elastic")
    photon = ScenePath(positions, "xray_generated")
    assert page._path_colour(unknown) == QColor("#94a3b8")
    assert page._path_colour(photon) == QColor(PATH_STYLES["xray_generated"][1])
    assert photon.source_ray_id == -1
    assert np.isnan(photon.source_azimuth_rad)


def test_boundary_rejects_stale_exit_and_matches_shared_optical_reference():
    result = _result()
    stale = _compact_branch(result)
    result.specimen_exit = _checkpoint((stale,), signature="stale")
    reference = _compact_branch(result)
    reference.source_ray_id = np.asarray((19, 19))
    reference.source_azimuth_rad = np.asarray((1.3, 1.3))
    result.simulation.branches = {"reference": reference}
    paths = _sample_boundary_paths(
        result, SpecimenScene.from_state(result.state_snapshot),
        include_incident=False,
    )
    assert [path.source_ray_id for path in paths] == [19, 19]
    assert all("Optical reference" in path.provenance for path in paths)


def test_bounded_scene_rejects_stale_exit_and_keeps_local_electron_lineage():
    result = _result()
    stale = _compact_branch(result)
    region = SimpleNamespace(
        electron_paths=(),
        interactions=result.specimen_interactions,
        downstream_branches=(stale,),
        specimen_exit=_checkpoint((stale,), signature="stale"),
        metrics={"sample_downstream_signature": "stale"},
        photon_paths=(),
        entry_z_mm=1.499,
        exit_z_mm=1.50005,
    )
    scene = build_sample_interaction_scene(result, region)
    assert scene.downstream_source == "Optical reference"
    assert not any(path.category.startswith("downstream") for path in scene.paths)
