from types import SimpleNamespace

import numpy as np
import pytest

from temsim.gui.sample_interactions_3d import (
    SampleInteractions3DPage,
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
    assert "Cached local view" in page.summary.text()
    assert "not HAADF/DF/BF counts" in page.summary.text()
    assert "only redraw this cache" in page.summary.text()
    calculation = _calculation_result()
    calculation.state_snapshot.sample.eds_enabled = True
    page.display_result(calculation)
    page.calculate_paths.click()
    assert requests == [True]


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
