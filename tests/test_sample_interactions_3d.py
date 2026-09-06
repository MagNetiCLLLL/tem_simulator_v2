from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtGui import QVector3D
from PySide6.QtWidgets import QTabWidget, QWidget

from temsim.gui.sample_interactions_3d import (
    SampleInteractions3DPage,
    _gl_display_bounds,
    _gl_display_positions,
    build_sample_interaction_scene,
)
from temsim.optics.column import default_state


def _calculation_result():
    state = default_state()
    state.sample.z_mm = 1.5
    state.sample.thickness_nm = 10.0
    incident_bundle = SimpleNamespace(
        rays=(
            SimpleNamespace(
                position_xy_nm=(2.0, -3.0),
                direction=(0.0, 0.0, 1.0),
                kinetic_energy_ev=300_000.0,
            ),
        )
    )
    elastic = SimpleNamespace(
        trajectories=(
            SimpleNamespace(
                points_nm=np.asarray(((-1.0, 0.0, -5.0), (3.0, 2.0, 5.0))),
                events=(object(),),
                outcome="transmitted",
            ),
        )
    )
    interactions = SimpleNamespace(
        incident_bundle=incident_bundle,
        elastic_transport=elastic,
        events=(
            SimpleNamespace(
                process="elastic_scatter",
                position_nm=(1.0, 2.0, 0.0),
            ),
            SimpleNamespace(
                process="core_ionisation",
                position_nm=(-1.0, 1.0, 2.0),
            ),
        ),
    )
    return SimpleNamespace(
        state_snapshot=state,
        simulation=None,
        specimen_interactions=interactions,
        wave_imaging=object(),
    )


def test_high_accuracy_cache_builds_local_3d_scene_without_new_physics():
    scene = build_sample_interaction_scene(_calculation_result())

    categories = [path.category for path in scene.paths]
    assert categories == ["elastic", "incident"]
    assert scene.paths[0].positions_nm[-1] == pytest.approx((3.0, 2.0, 5.0))
    assert scene.paths[1].positions_nm[[0, -1], 2] == pytest.approx(
        (-105.0, -5.0)
    )
    assert {group.category for group in scene.events} == {
        "elastic_event",
        "inelastic_event",
    }
    assert scene.coherent_wave_available
    assert not scene.has_bounded_result


def test_xz_yz_views_reproject_one_scene_without_calculation(qtbot, monkeypatch):
    import temsim.gui.sample_interactions_3d as view_module

    page = SampleInteractions3DPage()
    qtbot.addWidget(page)
    page.resize(900, 700)
    page.show()
    page.display_result(_calculation_result())
    cached = page.scene_snapshot
    originals = tuple(path.positions_nm.copy() for path in cached.paths)
    def forbidden(*args, **kwargs):
        raise AssertionError("View changes must not rebuild the scene or physics")
    monkeypatch.setattr(view_module, "build_sample_interaction_scene", forbidden)
    monkeypatch.setattr(view_module, "SpecimenFieldTransport", forbidden)

    page.set_view_mode("xz")
    assert page.view.listDataItems()[0].xData == pytest.approx(originals[0][:, 0])
    page.view.setRange(xRange=(-50, 50), yRange=(-80, 80), padding=0)
    previous_range = np.asarray(page.view.viewRange())
    page.view_buttons["yz"].click()
    assert page._view_mode == "yz"
    assert page.view.listDataItems()[0].xData == pytest.approx(originals[0][:, 1])
    assert page.view.listDataItems()[0].yData == pytest.approx(originals[0][:, 2])
    assert page.view.getAxis("bottom").labelText == "Local Y"
    assert page.view.getViewBox().state["yInverted"] is True
    np.testing.assert_allclose(page.view.viewRange(), previous_range)
    page.fit_material.click()
    page.view_buttons["xz"].click()
    np.testing.assert_allclose(page.view.viewRange(), previous_range)
    assert page.scene_snapshot is cached
    for path, original in zip(cached.paths, originals, strict=True):
        np.testing.assert_array_equal(path.positions_nm, original)


def test_3d_projection_roundtrip_preserves_camera_without_opengl_execution(qtbot, monkeypatch):
    """Renderer API fixture only: no desktop/GPU visual validation is claimed."""
    import temsim.gui.sample_interactions_3d as view_module

    class FakeGLView(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.opts = {"center": QVector3D(), "distance": 100.0,
                         "fov": 60.0, "elevation": 18.0, "azimuth": -45.0}
            self.items = []

        def setBackgroundColor(self, colour):
            pass

        def setCameraPosition(self, **values):
            self.opts.update(values)

        def addItem(self, item):
            self.items.append(item)

        def removeItem(self, item):
            self.items.remove(item)

    monkeypatch.setattr(view_module, "gl", SimpleNamespace(
        GLViewWidget=FakeGLView,
        GLLinePlotItem=lambda **kwargs: SimpleNamespace(**kwargs),
        GLScatterPlotItem=lambda **kwargs: SimpleNamespace(**kwargs),
    ))
    monkeypatch.setattr(view_module, "QGuiApplication", SimpleNamespace(
        platformName=lambda: "renderer-api-fixture",
    ))
    page = SampleInteractions3DPage()
    qtbot.addWidget(page)
    page.display_result(_calculation_result())
    scene = page.scene_snapshot
    page.view.setCameraPosition(distance=543.0, elevation=27.0, azimuth=12.0)
    before = page._capture_view_state()
    for mode in ("xz", "yz", "3d"):
        page.view_buttons[mode].click()
        assert page.scene_snapshot is scene
    assert page._capture_view_state() == before
    assert len(page._gl_view.items) == len(page._items)


def test_3d_display_maps_physical_downstream_z_toward_screen_down():
    physical = np.asarray(((1.0, 2.0, -5.0), (3.0, 4.0, 7.0)))

    displayed = _gl_display_positions(physical)
    display_lower, display_upper = _gl_display_bounds((
        np.asarray((-2.0, -3.0, -5.0)),
        np.asarray((4.0, 6.0, 7.0)),
    ))

    np.testing.assert_allclose(displayed, ((1.0, 2.0, 5.0), (3.0, 4.0, -7.0)))
    np.testing.assert_allclose(physical[:, 2], (-5.0, 7.0))
    np.testing.assert_allclose(display_lower, (-2.0, -3.0, -7.0))
    np.testing.assert_allclose(display_upper, (4.0, 6.0, 5.0))


def test_bounded_scene_uses_global_to_local_coordinates_and_skips_secondaries():
    calculation = _calculation_result()
    interactions = calculation.specimen_interactions
    region = SimpleNamespace(
        entry_z_mm=1.49995,
        exit_z_mm=1.50005,
        interactions=interactions,
        downstream_branches=(),
        electron_paths=(
            SimpleNamespace(
                positions_mm=np.asarray(((1.0e-6, 0.0, 1.49995), (2.0e-6, 0.0, 1.5))),
                kind="boundary_input",
                provenance="cached phase space",
            ),
            SimpleNamespace(
                positions_mm=np.asarray(((0.0, 0.0, 1.5), (0.0, 0.0, 1.500001))),
                kind="secondary_candidate",
                provenance="unvalidated marker",
            ),
        ),
        photon_paths=(
            SimpleNamespace(
                positions_mm=np.asarray(((0.0, 0.0, 1.5), (10.0, 0.0, 1.5))),
                detected=True,
                provenance="isotropic characteristic photon",
            ),
        ),
    )

    scene = build_sample_interaction_scene(calculation, region)

    assert [path.category for path in scene.paths] == [
        "incident",
        "xray_detected",
    ]
    np.testing.assert_allclose(
        scene.paths[0].positions_nm,
        ((1.0, 0.0, -50.0), (2.0, 0.0, 0.0)),
    )
    assert np.linalg.norm(np.diff(scene.paths[1].positions_nm, axis=0)) == (
        pytest.approx(100.0)
    )
    assert scene.has_bounded_result


def test_bounded_scene_prefers_v2_photon_transport_endpoints_and_metrics(qtbot):
    calculation = _calculation_result()
    interactions = calculation.specimen_interactions
    photon = SimpleNamespace(
        origin_mm=(0.0, 0.0, 1.5),
        direction=(1.0, 0.0, 0.0),
        energy_ev=1_740.0,
    )
    exact = SimpleNamespace(
        photon=photon,
        detector_hit_mm=(0.001, 0.0, 1.5),
        material_intervals=(),
        detected_weight=0.2,
        terminal_status="detected",
        transport_mode="sourced_planar_segment_intersection",
    )
    blocked_photon = SimpleNamespace(
        origin_mm=(0.0, 0.0, 1.5),
        direction=(0.0, 1.0, 0.0),
        energy_ev=1_740.0,
    )
    interval = SimpleNamespace(
        entry_distance_mm=0.0002,
        hard_shadow=True,
    )
    blocked = SimpleNamespace(
        photon=blocked_photon,
        detector_hit_mm=None,
        material_intervals=(interval,),
        detected_weight=0.0,
        terminal_status="blocked_by:objective_upper_pole",
        transport_mode="sourced_planar_segment_intersection",
    )
    transport = SimpleNamespace(
        paths=(exact, blocked),
        geometry_complete=True,
        metrics={
            "detector_hit_count": 1,
            "blocked_photon_count": 1,
            "specimen_intersection_count": 2,
            "mean_specimen_transmission": 0.75,
        },
    )
    region = SimpleNamespace(
        entry_z_mm=1.49995,
        exit_z_mm=1.50005,
        interactions=interactions,
        downstream_branches=(),
        electron_paths=(),
        photon_transport=transport,
        photon_paths=(SimpleNamespace(
            positions_mm=np.asarray(((0.0, 0.0, 1.5), (10.0, 0.0, 1.5))),
            detected=False,
            provenance="legacy path must not be drawn",
        ),),
    )

    scene = build_sample_interaction_scene(calculation, region)
    xray_paths = tuple(
        path for path in scene.paths if path.category.startswith("xray_")
    )
    assert len(xray_paths) == 2
    np.testing.assert_allclose(
        xray_paths[0].positions_nm[-1], (1_000.0, 0.0, 0.0)
    )
    np.testing.assert_allclose(
        xray_paths[1].positions_nm[-1], (0.0, 200.0, 0.0)
    )

    page = SampleInteractions3DPage()
    qtbot.addWidget(page)
    page.display_result(calculation)
    page.set_sample_region_result(region)
    assert "exact-face" in page.summary.text()
    assert "1 hit" in page.summary.text()
    assert "1 blocked" in page.summary.text()
    assert "mean transmission 0.75" in page.summary.toolTip()


def test_3d_page_exposes_cached_view_and_explicit_calculation_request(qtbot):
    page = SampleInteractions3DPage()
    qtbot.addWidget(page)
    requests = []
    page.sample_region_requested.connect(lambda: requests.append(True))

    page.display_result(_calculation_result())

    assert page.scene_snapshot is not None
    assert page.calculate_paths.isEnabled()
    assert "2 electron paths" in page.summary.text()
    assert len(page.summary.text()) < 180
    assert "Cached local view" in page.summary.toolTip()
    assert "not HAADF/DF/BF counts" in page.summary.toolTip()
    assert "physical +Z downward" in page.summary.toolTip()
    assert "only redraw this cache" in page.summary.toolTip()
    if not page.opengl_available:
        assert page.view.getViewBox().state["yInverted"] is True
    calculation = _calculation_result()
    calculation.state_snapshot.sample.eds_enabled = True
    page.display_result(calculation)
    page.calculate_paths.click()
    assert requests == [True]


def test_3d_page_filters_each_cached_signal_category_without_recalculation(
    qtbot,
):
    page = SampleInteractions3DPage()
    qtbot.addWidget(page)
    page.display_result(_calculation_result())
    cached_scene = page.scene_snapshot

    assert len(page.signal_actions) == 11
    assert len(page.visible_signal_categories) == 11
    assert page.signal_filter.text() == "Visible signals: 11/11"

    page.signal_actions["elastic"].setChecked(False)
    page.signal_actions["inelastic_event"].setChecked(False)

    assert page.scene_snapshot is cached_scene
    assert "elastic" not in page.visible_signal_categories
    assert "inelastic_event" not in page.visible_signal_categories
    assert page.signal_filter.text() == "Visible signals: 9/11"
    assert " Elastic&nbsp;&nbsp;" not in page.legend.text()
    assert "Vacancy sites" not in page.legend.text()
    assert "Incident" in page.legend.text()
    assert "Elastically scattered electrons" in page.legend.toolTip()

    page.hide_all_signals.trigger()
    assert not page.visible_signal_categories
    assert "No signal types selected" in page.legend.text()
    assert page.context_toggle.isChecked()

    page.show_all_signals.trigger()
    assert len(page.visible_signal_categories) == 11
    assert page.scene_snapshot is cached_scene


def test_3d_tab_round_trip_preserves_cached_scene_and_view(qtbot):
    tabs = QTabWidget()
    page = SampleInteractions3DPage()
    tabs.addTab(page, "Sample Interactions 3D")
    tabs.addTab(QWidget(), "EDS")
    qtbot.addWidget(tabs)
    tabs.resize(1000, 700)
    tabs.show()
    calculation = _calculation_result()
    page.display_result(calculation)
    qtbot.wait(20)
    cached_scene = page.scene_snapshot
    page.display_result(calculation)
    assert page.scene_snapshot is cached_scene

    if page.opengl_available:
        page.view.opts["center"] = QVector3D(11.0, -7.0, 3.0)
        page.view.setCameraPosition(
            distance=321.0,
            elevation=67.0,
            azimuth=123.0,
        )
        expected = page._capture_view_state()
    else:
        page.view.setRange(
            xRange=(-17.0, 29.0),
            yRange=(-31.0, 13.0),
            padding=0.0,
        )
        expected = page._capture_view_state()

    tabs.setCurrentIndex(1)
    tabs.setCurrentIndex(0)
    qtbot.wait(20)
    qtbot.waitUntil(lambda: page._pending_view_restore is None)

    assert page.scene_snapshot is cached_scene
    restored = page._capture_view_state()
    assert restored["kind"] == expected["kind"]
    if page.opengl_available:
        assert restored["center"] == pytest.approx(expected["center"])
        assert restored["distance"] == pytest.approx(expected["distance"])
        assert restored["elevation"] == pytest.approx(expected["elevation"])
        assert restored["azimuth"] == pytest.approx(expected["azimuth"])
    else:
        for axis in ("x_range", "y_range"):
            restored_range = np.asarray(restored[axis])
            expected_range = np.asarray(expected[axis])
            assert float(np.mean(restored_range)) == pytest.approx(
                float(np.mean(expected_range)), abs=0.2
            )
            assert float(np.ptp(restored_range)) == pytest.approx(
                float(np.ptp(expected_range)), rel=0.01
            )


def test_virtual_and_vacuum_scenes_use_the_active_user_selection():
    state = default_state()
    state.sample.specimen_mode = "virtual"
    state.sample.specimen_preset_key = "si_110"
    state.sample.virtual_regions = [
        {
            "kind": "ellipse",
            "enabled": True,
            "density": 0.5,
            "centre_x_nm": 12.0,
            "centre_y_nm": -5.0,
            "size_x_nm": 40.0,
            "size_y_nm": 20.0,
            "rotation_deg": 30.0,
        }
    ]
    incident = SimpleNamespace(
        x=np.asarray(((0.0,), (0.0,))),
        y=np.asarray(((0.0,), (0.0,))),
        tx=np.asarray(((0.0,), (0.0,))),
        ty=np.asarray(((0.0,), (0.0,))),
        alive=np.asarray((True,)),
        energy_offset_ev=np.asarray((0.0,)),
    )
    branches = {
        "000": SimpleNamespace(
            name="000",
            interaction_kind="transmitted",
            x=np.asarray(((0.0,), (0.0,))),
            y=np.asarray(((0.0,), (0.0,))),
            tx=np.asarray(((0.0,), (0.0,))),
            ty=np.asarray(((0.0,), (0.0,))),
            energy_offset_ev=np.asarray((0.0,)),
        ),
        "virtual_+g": SimpleNamespace(
            name="virtual_+g",
            interaction_kind="diffraction_spots",
            x=np.asarray(((0.0,), (0.0,))),
            y=np.asarray(((0.0,), (0.0,))),
            tx=np.asarray(((5.0e-3,), (5.0e-3,))),
            ty=np.asarray(((0.0,), (0.0,))),
            energy_offset_ev=np.asarray((0.0,)),
        ),
    }
    calculation = SimpleNamespace(
        state_snapshot=state,
        simulation=SimpleNamespace(incident=incident, branches=branches),
        specimen_interactions=None,
        wave_imaging=None,
        stem_scan=None,
    )

    virtual = build_sample_interaction_scene(calculation)

    assert virtual.specimen_mode == "virtual"
    assert virtual.specimen_source_key == "preset:si_110"
    assert not virtual.specimen_is_vacuum
    assert len(virtual.virtual_region_outlines_nm) == 1
    assert {path.category for path in virtual.paths} == {
        "incident",
        "downstream_primary",
        "downstream_elastic",
    }
    assert all(
        "detector not assigned" in path.provenance
        for path in virtual.paths
        if path.category.startswith("downstream_")
    )

    state.sample.specimen_preset_key = "vacuum"
    calculation.simulation.branches = {"000": branches["000"]}
    vacuum = build_sample_interaction_scene(calculation)

    assert vacuum.specimen_is_vacuum
    assert vacuum.specimen_source_key == "preset:vacuum"
    assert not vacuum.virtual_region_outlines_nm
    assert {path.category for path in vacuum.paths} == {
        "incident",
        "downstream_primary",
    }
