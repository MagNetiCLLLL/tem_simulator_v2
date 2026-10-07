"""Reception images preserve actual first stops, capture state and electron dose."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.component_keys import CAMERA, FLUORESCENT_SCREEN
from temsim.detector.receiver_image import available_receivers, receiver_image


def _plane(key, z, *, disk=False):
    return SimpleNamespace(
        key=key, name=key, z_mm=z, inserted=True, outer_width_mm=2.0,
        point_spread_model="none", point_spread_sigma_x_mm=0.0,
        point_spread_sigma_y_mm=0.0, point_spread_rotation_deg=0.0,
        point_spread_status="provisional_model_parameter", point_spread_source="unit test",
        hit_mask=(lambda x, y: np.hypot(x, y) <= 1.0) if disk else
                 (lambda x, y: (np.abs(x) <= 1.0) & (np.abs(y) <= 1.0)))


@pytest.fixture
def capture(monkeypatch):
    monkeypatch.setattr("temsim.physics.beam_current.effective_source_current_pa", lambda state: 2.0)
    camera, screen = _plane(CAMERA, 3.0), _plane(FLUORESCENT_SCREEN, 2.0, disk=True)
    weights = np.array([0.2, 0.3, 0.5])
    branch = SimpleNamespace(
        z=np.array([1.0, 2.0, 3.0]), x=np.tile([0.0, 0.0004, -0.0004], (3, 1)),
        y=np.zeros((3, 3)), alive=np.zeros(3, dtype=bool), weight=1.0,
        ray_weight=weights.copy(), blocked_key=[CAMERA, FLUORESCENT_SCREEN, "wall"],
        blocked_z=np.array([3.0, 2.0, 1.5]))
    incident = SimpleNamespace(z=np.array([0.0, 1.0]), alive=np.ones(3, dtype=bool), ray_weight=weights.copy())
    state = SimpleNamespace(recording_planes=[screen, camera],
                            sample=SimpleNamespace(z_mm=1.0, specimen_mode="vacuum", inserted=False),
                            electron_gun=SimpleNamespace(ray_count=3), projector_mode="image")
    return SimpleNamespace(state_snapshot=state, specimen_exit=None, workflow="particle",
                           simulation=SimpleNamespace(incident=incident, branches={"post": branch},
                               metrics={"section_target_z_mm": 3.0}))


def test_receivers_include_retracted_hardware_but_exclude_uninstalled(capture):
    state = capture.state_snapshot
    state.recording_planes[0].inserted = False
    state.recording_planes.append(SimpleNamespace(key="haadf"))
    assert [p.key for p in available_receivers(state)] == [FLUORESCENT_SCREEN, CAMERA]
    state.recording_planes[0]._layout_installed = False
    assert [p.key for p in available_receivers(state)] == [CAMERA]
    state.recording_planes[1].installed = False
    assert available_receivers(state) == ()


def test_first_hit_probability_and_exposure_are_not_numerical_ray_counts(capture):
    before = deepcopy(capture)
    image = receiver_image(capture, CAMERA, pixels=16, exposure_s=0.002)
    assert image.status == "AVAILABLE"
    assert image.ideal_probability.sum() == pytest.approx(0.2)
    assert image.response_probability.sum() == pytest.approx(0.2)
    assert image.current_pa == pytest.approx(0.4)
    assert image.expected_electrons.sum() == pytest.approx(0.4e-12 / 1.602176634e-19 * 0.002)
    assert image.ideal_probability.shape == (16, 16)
    assert image.x_mm[[0, -1]].tolist() == pytest.approx([-0.9375, 0.9375])
    for values in (image.ideal_probability, image.response_probability, image.expected_electrons, image.x_mm):
        assert not values.flags.writeable
    original, after = before.simulation.branches["post"], capture.simulation.branches["post"]
    assert np.array_equal(original.x, after.x)
    assert original.blocked_key == after.blocked_key
    capture.state_snapshot.electron_gun.ray_count = 9000
    repeated = receiver_image(capture, CAMERA, pixels=16, exposure_s=0.002)
    assert np.array_equal(repeated.expected_electrons, image.expected_electrons)


def test_receiver_switch_obeys_upstream_first_interception(capture):
    assert receiver_image(capture, FLUORESCENT_SCREEN, pixels=16).ideal_probability.sum() == pytest.approx(0.3)
    branch = capture.simulation.branches["post"]
    branch.blocked_key = [FLUORESCENT_SCREEN] * 3
    branch.blocked_z[:] = 2.0
    camera = receiver_image(capture, CAMERA, pixels=16)
    assert camera.status == "AVAILABLE" and camera.current_pa == 0.0
    assert not camera.expected_electrons.any()


def test_material_exit_is_preferred_over_optical_display_branch(capture):
    branch = deepcopy(capture.simulation.branches["post"])
    branch.ray_weight = np.array([0.4, 0.1, 0.5])
    capture.specimen_exit = SimpleNamespace(branches=(branch,))
    image = receiver_image(capture, CAMERA, pixels=16)
    assert image.ideal_probability.sum() == pytest.approx(0.4)
    assert image.metrics["particle_source"] == "specimen_exit"


@pytest.mark.parametrize("mode,status", [
    ("retracted", "NOT_INSERTED"), ("readout_disabled", "READOUT_DISABLED"),
    ("unreached", "NOT_REACHED"), ("optical", "NOT_CALCULATED"),
    ("rays", "NOT_CALCULATED"), ("missing", "NOT_CALCULATED"),
    ("missing_material", "NOT_CALCULATED"), ("missing_branches", "NOT_CALCULATED"),
])
def test_unavailable_is_distinct_from_a_calculated_zero(capture, mode, status):
    camera = capture.state_snapshot.recording_planes[1]
    if mode == "retracted":
        camera.inserted = False
    elif mode == "readout_disabled":
        camera.readout_enabled = False
    elif mode == "unreached":
        capture.simulation.metrics["section_target_z_mm"] = 2.5
    elif mode == "optical":
        capture.simulation.metrics["optical_tuning"] = True
    elif mode == "rays":
        capture.workflow = "rays"
    elif mode == "missing":
        capture.simulation = None
    elif mode == "missing_branches":
        capture.simulation.branches = {}
    elif mode == "missing_material":
        capture.state_snapshot.sample = SimpleNamespace(
            z_mm=1.0, specimen_mode="atomic", inserted=True, thickness_nm=5.0, cif_path="captured.cif")
    image = receiver_image(capture, CAMERA, pixels=16)
    assert image.status == status
    assert image.expected_electrons is None and image.current_pa is None


def test_no_downstream_population_with_no_illumination_is_a_valid_zero(capture):
    capture.simulation.branches = {}
    capture.simulation.incident.alive[:] = False
    image = receiver_image(capture, CAMERA, pixels=16)
    assert image.status == "AVAILABLE" and not image.expected_electrons.any()


def test_incomplete_surviving_branch_cannot_be_extrapolated(capture):
    branch = capture.simulation.branches["post"]
    branch.z = branch.z[:2]
    branch.x, branch.y = branch.x[:2], branch.y[:2]
    branch.alive[0] = True
    image = receiver_image(capture, CAMERA, pixels=16)
    assert image.status == "NOT_REACHED"


def test_physical_plane_coordinates_are_interpolated_only_inside_history(capture):
    branch = capture.simulation.branches["post"]
    branch.z = np.array([1.0, 2.5, 3.5])
    branch.x[:, 0] = [0.0, 0.0, 0.0005]
    image = receiver_image(capture, CAMERA, pixels=16)
    row, col = np.unravel_index(image.ideal_probability.argmax(), image.ideal_probability.shape)
    assert image.x_mm[col] == pytest.approx(0.3125)
    assert image.y_mm[row] == pytest.approx(0.0625)


def test_screen_response_spreads_and_is_masked_to_physical_disk(capture):
    screen = capture.state_snapshot.recording_planes[0]
    screen.point_spread_model = "gaussian"
    screen.point_spread_sigma_x_mm = screen.point_spread_sigma_y_mm = 0.2
    branch = capture.simulation.branches["post"]
    branch.x[:, 1] = 0.00095
    image = receiver_image(capture, FLUORESCENT_SCREEN, pixels=64, exposure_s=None)
    active = screen.hit_mask(image.x_mm[None, :], image.y_mm[:, None])
    assert np.all(image.response_probability[~active] == 0.0)
    assert 0.0 < image.response_probability.sum() < image.ideal_probability.sum()
    assert image.expected_electrons is None


def test_no_psf_preserves_real_hit_in_a_pixel_crossing_screen_edge(capture):
    branch = capture.simulation.branches["post"]
    branch.x[:, 1] = branch.y[:, 1] = 0.0007
    image = receiver_image(capture, FLUORESCENT_SCREEN, pixels=4)
    assert image.response_probability.sum() == pytest.approx(0.3)


@pytest.mark.parametrize("corruption,message", [
    ("wrong_z", "stop position"), ("outside", "outside"), ("nan", "finite positions"),
    ("weights", "exceeds"),
])
def test_corrupt_hit_records_are_not_presented_as_physical_reception(capture, corruption, message):
    branch = capture.simulation.branches["post"]
    if corruption == "wrong_z":
        branch.blocked_z[0] = 2.9
    elif corruption == "outside":
        branch.x[:, 0] = 0.002
    elif corruption == "nan":
        branch.x[:, 0] = np.nan
    elif corruption == "weights":
        branch.ray_weight[0] = 2.0
    with pytest.raises(ValueError, match=message):
        receiver_image(capture, CAMERA, pixels=16)


@pytest.mark.parametrize("kwargs", [{"exposure_s": -1}, {"exposure_s": np.nan},
                                    {"pixels": 1}, {"pixels": 2.5},
                                    {"extent_mm": (-1, 1, -1, 1)}])
def test_invalid_display_or_dose_requests_fail_explicitly(capture, kwargs):
    with pytest.raises(ValueError):
        receiver_image(capture, CAMERA, **kwargs)


def test_unknown_receiver_cannot_be_created_from_a_virtual_plane(capture):
    with pytest.raises(ValueError, match="Unknown or uninstalled"):
        receiver_image(capture, "virtual_camera")
