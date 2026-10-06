"""Cached page publications preserve single-frame ray presentation only."""

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from temsim.gui.visualization import VisualizationWorkspace


@pytest.fixture
def publication(monkeypatch):
    """Exercise the real publication and offset code without building a column."""
    frame = object()
    response = np.ones((2, 2, 2))
    paths = SimpleNamespace(
        baseline_ac_command_mrad=np.zeros(2),
        baseline_descan_command_mrad=np.zeros(2),
        responses_m_per_rad={"incident": (response, np.zeros_like(response))},
    )
    events = []
    command = Mock(side_effect=lambda time_s: (events.append("offset") or (time_s, 0.)))
    result = SimpleNamespace(
        simulation=SimpleNamespace(metrics={}),
        state_snapshot=SimpleNamespace(
            ac_deflector=SimpleNamespace(scan_kick_mrad=command),
            descan_deflector=SimpleNamespace(enabled=False, scan_enabled=False),
        ),
        stem_scan=frame, scan_ray_paths=paths, specimen_exit=None,
    )
    view = SimpleNamespace(
        _last_result=object(), _high_accuracy_result=None,
        _ray_display_cache={}, _ray_bundle_records=[],
        _scan_ray_paths=paths, _scan_playback_active=False,
        _scan_playback_time_s=15.5/16.,
        _scan_ray_offsets_m={"incident": np.full((2, 2), -99.)},
    )
    for name in (
        "_ray_flight_time_colours", "transverse_beam", "result_readout",
        "hardware_tuning", "transport_adjustment_readout", "ray_source_status",
        "interactive_calculation", "selected_plane_readout", "probe_aberrations",
        "image_aberrations", "optical_transfer", "energy_filter", "sample_page",
        "sample_interactions_3d", "eds_page", "wave_imaging",
        "_refresh_ray_calculation_extent", "_publish_optional_ray_panels",
        "_refresh_visible_ray_panels", "_display_ray_scope_products",
        "_update_sample_region_control_availability", "_update_projection_text",
    ):
        setattr(view, name, Mock())
    view.scan_control = Mock(_stem_frame=frame)
    for name in ("_prepare_scan_ray_playback", "_scan_playback_time_changed"):
        setattr(view, name, MethodType(getattr(VisualizationWorkspace, name), view))
    curve = Mock()

    def draw(*_args, **_kwargs):
        events.append("draw")
        assert view._scan_ray_offsets_m == {}  # Never reuse old-result offsets.
        view._ray_bundle_records = [(curve, "incident")]

    view._draw_ray_diagram = draw
    view._ray_record_lines = lambda key: (
        np.arange(2), view._scan_ray_offsets_m.get(key, np.zeros((2, 2)))[:, 0])
    monkeypatch.setattr("temsim.gui.visualization.sample_illumination_absent", lambda *_args: False)
    monkeypatch.setattr("temsim.gui.visualization.downstream_display_branches",
                        lambda current: ((), "Specimen exit" if current.specimen_exit else "Optical reference"))
    return view, result, command, curve, events


@pytest.mark.parametrize("active", (False, True))
@pytest.mark.parametrize("scope", ("rays", "energy_filter"))
def test_cached_frame_publication_restores_ray_time_and_activity_after_drawing(publication, active, scope):
    view, result, command, curve, events = publication
    view._scan_playback_active = active
    time_s = view._scan_playback_time_s
    arrays = [array.copy() for pair in result.scan_ray_paths.responses_m_per_rad.values() for array in pair]

    VisualizationWorkspace.display_result(view, result, "High accuracy", calculation_scope=scope)

    assert view._last_result is result
    assert view._scan_playback_time_s == time_s
    assert view._scan_playback_active is active
    assert events == ["draw", "offset"]
    command.assert_called_once_with(time_s)
    np.testing.assert_allclose(view._scan_ray_offsets_m["incident"], time_s*1e-3)
    np.testing.assert_allclose(curve.setData.call_args.args[1], time_s*1e-3)
    for array, original in zip(
        (array for pair in result.scan_ray_paths.responses_m_per_rad.values() for array in pair), arrays,
        strict=True,
    ):
        np.testing.assert_array_equal(array, original)


@pytest.mark.parametrize("change", ("explicit", "frame", "paths", "no_frame"))
def test_new_scan_publication_resets_ray_presentation(publication, change):
    view, result, command, curve, events = publication
    view._scan_playback_active = True
    scope = "stem" if change == "explicit" else "rays"
    if change == "frame":
        result.stem_scan = object()
    elif change == "paths":
        result.scan_ray_paths = SimpleNamespace(**vars(result.scan_ray_paths))
    elif change == "no_frame":
        result.stem_scan = None

    VisualizationWorkspace.display_result(view, result, "High accuracy", calculation_scope=scope)

    assert view._scan_playback_time_s is None
    assert not view._scan_playback_active
    assert view._scan_ray_offsets_m == {}
    assert events == ["draw"]
    command.assert_not_called()
    curve.setData.assert_not_called()


def test_cached_frame_with_detailed_specimen_exit_retains_time_without_optical_offsets(publication):
    view, result, command, curve, events = publication
    result.specimen_exit = object()
    time_s = view._scan_playback_time_s

    VisualizationWorkspace.display_result(view, result, "High accuracy", calculation_scope="rays")

    assert view._scan_playback_time_s == time_s
    assert not view._scan_playback_active
    assert view._scan_ray_offsets_m == {}
    assert events == ["draw"]
    command.assert_not_called()
    curve.setData.assert_not_called()
