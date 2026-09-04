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
