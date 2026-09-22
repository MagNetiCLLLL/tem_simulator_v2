"""Display provenance using synthetic executed records; no particle solver."""
from types import SimpleNamespace

import numpy as np
import pytest

from test_beam_analysis_modes import make_result, view


def rotation_result(selected_ids, *, reorder_record=False, missing_record=False):
    ids = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    positions = np.array([[2., 0., 0.], [1., 3., 0.], [-2., 1., 0.],
                          [-1., -2., 0.], [3., -1., 0.]]) * 1e-9
    selected_ids = np.asarray(selected_ids, dtype=np.int64)
    selected = np.searchsorted(ids, selected_ids)
    result = make_result(len(selected))
    branch = result.simulation.incident
    branch.source_ray_id = selected_ids
    branch.source_azimuth_rad = np.mod(np.arctan2(positions[selected, 1], positions[selected, 0]), 2*np.pi)
    source = positions[selected, 0] + 1j * positions[selected, 1]
    downstream = source * np.exp(1j*np.deg2rad(37.)) * 1e4 + (2e-4 - 3e-4j)
    # Deliberately unrelated gun-exit positions must never supply tip data.
    gun_exit = source * np.exp(-1j*np.deg2rad(68.)) * 2e4
    branch.x = np.stack((gun_exit.real, downstream.real))
    branch.y = np.stack((gun_exit.imag, downstream.imag))
    if not missing_record:
        order = np.array([3, 0, 4, 1, 2]) if reorder_record else np.arange(5)
        result.simulation.gun_trace = SimpleNamespace(emission_reference={
            "ray_id": ids[order], "position_m": positions[order],
            "direction": np.tile([0., 0., 1.], (5, 1)),
            "normal": np.tile([0., 0., 1.], (5, 1)),
        })
    return result


@pytest.mark.parametrize("ids,reordered", [
    ([10, 20, 30, 40, 50], False),
    ([40, 10, 30], False),
    ([40, 10, 30], True),
])
def test_rotation_uses_recorded_tip_positions_by_id_after_compaction(view, ids, reordered):
    result = rotation_result(ids, reorder_record=reordered)
    view.display_result(result)
    view.focus_z(1.)
    for projection in (0., 90.):
        view.set_projection_angle(projection)
        rotation = view.summary.text().split("pattern rotation ")[1].split(" deg")[0]
        assert float(rotation) == pytest.approx(37., abs=1e-9)
        assert "bundle centroid" not in view.summary.toolTip()


@pytest.mark.parametrize("missing", ["record", "ids"])
def test_rotation_is_unavailable_without_matching_tip_identity(view, missing):
    result = rotation_result([40, 10, 30], missing_record=missing == "record")
    if missing == "ids":
        result.simulation.incident.source_ray_id += 1000
    view.display_result(result)
    view.focus_z(1.)
    assert "pattern rotation unavailable" in view.summary.text()
