from types import SimpleNamespace

import numpy as np
import pytest

from temsim.detector import receiver_scan
from temsim.detector.receiver_image import ReceiverImage


class Raster:
    def __init__(self, nx=4, ny=2, *, enabled=True):
        self.key = "test_scan"
        self.enabled = enabled
        self.scan_enabled = enabled
        self.scan_pixels_x = nx
        self.scan_lines = ny
        self.scan_frame_period_s = 1.

    def scan_factors(self, time):
        line = (float(time) % 1.) * self.scan_lines
        row = min(int(line), self.scan_lines-1)
        return 2*(line-row)-1, 2*(row+.5)/self.scan_lines-1


def captured(*, matrix=((1000., 0.), (0., 1000.)), time=0., nx=4, ny=2,
             columns=None, rows=None, descan=False):
    driver = Raster(nx, ny)
    columns = np.arange(nx) if columns is None else np.asarray(columns)
    rows = np.arange(ny) if rows is None else np.asarray(rows)
    times = (rows[:, None] + (columns[None, :] + .5)/nx)/ny
    factors = np.asarray([driver.scan_factors(t) for t in times.flat]).reshape(times.shape+(2,))
    xy = factors @ np.asarray(matrix).T
    plane = SimpleNamespace(key="camera", hit_mask=lambda x, y: (abs(x) <= 8.) & (abs(y) <= 8.))
    state = SimpleNamespace(ac_deflector=driver, descan_deflector=Raster(nx, ny, enabled=descan),
                            simulation_time_s=time, recording_planes=[plane])
    geometry = SimpleNamespace(times_s=times, plane_positions_um={"camera": (xy[..., 0], xy[..., 1])},
        ac_enabled=True, descan_enabled=descan, requested_pixels_x=nx, requested_pixels_y=ny,
        unavailable_planes={}, plane_roles={"camera": "image"})
    return SimpleNamespace(state_snapshot=state, scan_geometry=geometry)


def static_image(*, status="AVAILABLE", zero=False):
    image = np.zeros((16, 16))
    if not zero:
        image[8, 8] = .25
    return ReceiverImage(key="camera", name="Camera", status=status, detail="captured physical hits",
        ideal_probability=image, response_probability=image.copy(),
        x_mm=np.arange(16)-7.5, y_mm=np.arange(16)-7.5, exposure_s=1.,
        expected_electrons=image*100., current_pa=1., metrics={})


@pytest.fixture
def physical_image(monkeypatch):
    image = static_image()
    monkeypatch.setattr(receiver_scan, "receiver_image", lambda *a, **k: image)
    return image


def centroid(image):
    weights = image.response_probability
    return np.array((np.sum(weights*image.x_mm[None, :]),
                     np.sum(weights*image.y_mm[:, None]))) / weights.sum()


def test_one_frame_preview_uses_captured_time_and_preserves_probability(physical_image):
    result = captured(time=.375)
    preview = receiver_scan.receiver_scan_preview(result, "camera", pixels=16)
    assert preview.status == "GEOMETRIC_PREVIEW"
    assert preview.baseline_offset_mm == pytest.approx((.5, -.5))
    assert preview.image.response_probability.sum() == pytest.approx(.25)
    assert centroid(preview.image) == pytest.approx((0., 1.))
    assert preview.image.expected_electrons is None
    assert preview.image.current_pa is None
    assert preview.metrics["actual_exposure"] is False
    assert "upstream clipping" in preview.detail
    assert not preview.times_s.flags.writeable
    assert not preview.image.response_probability.flags.writeable


def test_preview_preserves_static_seed_identity_geometry_and_psf_without_dose_claim(physical_image):
    physical_image.metrics.update({
        "capture_request_identity": "request-123", "capture_manifest_identity": "manifest-456",
        "capture_model_signature": "model-789", "receiver_z_mm": 2900., "receiver_width_mm": 16.,
        "physical_area_extent_mm": (-8., 8., -8., 8.), "axis_convention": "array[y,x]; mm",
        "point_spread_model": "none", "point_spread_status": "configured",
        "point_spread_source": "captured receiver", "source_current_pa": 20., "exposure_s": 4.,
        "intercepted_current_pa": 5., "intercepted_probability": .25,
    })
    preview = receiver_scan.receiver_scan_preview(captured(), "camera", pixels=16)
    metadata = preview.image.metrics
    assert metadata["static_seed_capture_request_identity"] == "request-123"
    assert metadata["static_seed_capture_manifest_identity"] == "manifest-456"
    assert metadata["static_seed_capture_model_signature"] == "model-789"
    assert metadata["static_seed_receiver_z_mm"] == 2900.
    assert metadata["static_seed_receiver_width_mm"] == 16.
    assert metadata["static_seed_physical_area_extent_mm"] == (-8., 8., -8., 8.)
    assert metadata["static_seed_axis_convention"] == "array[y,x]; mm"
    assert metadata["static_seed_point_spread_model"] == "none"
    assert metadata["static_seed_point_spread_source"] == "captured receiver"
    assert "static_seed_source_current_pa" not in metadata
    assert "static_seed_exposure_s" not in metadata
    assert "static_seed_intercepted_current_pa" not in metadata
    assert metadata["actual_exposure"] is False
    assert preview.image.status == "GEOMETRIC_PREVIEW"
    assert preview.image.current_pa is None
    assert preview.image.expected_electrons is None


def test_t_zero_is_left_edge_not_first_pixel_centre(physical_image):
    preview = receiver_scan.receiver_scan_preview(captured(), "camera", pixels=16, scan_index=0)
    assert preview.baseline_offset_mm == pytest.approx((-1., -.5))
    assert preview.displacement_x_mm[0, 0] == pytest.approx(.25)
    assert centroid(preview.image) == pytest.approx((.75, .5))
    assert preview.metrics["scan_index"] == 0
    assert preview.detail.startswith("Scan position preview")


def test_deskew_and_cross_axis_rotation_follow_captured_plane_map(physical_image):
    result = captured(matrix=((0., 2000.), (-1000., 500.)), time=.375)
    preview = receiver_scan.receiver_scan_preview(result, "camera", pixels=16)
    assert preview.baseline_offset_mm == pytest.approx((-1., -.75))
    assert centroid(preview.image) == pytest.approx((1.5, 1.25))
    assert preview.trajectory_x_mm[0, 0] == pytest.approx(-1.)
    assert preview.trajectory_y_mm[0, 0] == pytest.approx(.5)


def test_compensated_descan_preserves_stationary_spot(physical_image):
    result = captured(matrix=((0., 0.), (0., 0.)), descan=True)
    preview = receiver_scan.receiver_scan_preview(result, "camera", pixels=16)
    np.testing.assert_allclose(preview.image.response_probability, physical_image.response_probability, atol=1e-16)
    assert np.max(np.abs(preview.displacement_x_mm)) == 0.
    assert preview.metrics["descan_enabled"] is True


def test_stationary_circular_screen_preserves_exact_edge_hit_without_psf(monkeypatch):
    # This physical hit is inside the circular screen, although its coarse
    # histogram bin centre (5.5, 6.5) is outside radius 8. A stationary scan
    # must retain that already accepted electron, without reclipping its bin.
    hit_x, hit_y = 5.05, 6.05
    assert np.hypot(hit_x, hit_y) < 8.
    edges = np.arange(17)-8.
    image = np.histogram2d([hit_y], [hit_x], bins=(edges, edges), weights=[.25])[0]
    axis = np.arange(16)-7.5
    assert np.hypot(axis[13], axis[14]) > 8.
    base = ReceiverImage(key="flu_screen", name="Fluorescent screen", status="AVAILABLE", detail="Exact hits",
        ideal_probability=image, response_probability=image.copy(), x_mm=axis, y_mm=axis,
        exposure_s=1., expected_electrons=None, current_pa=1., metrics={"point_spread_model": "none"})
    monkeypatch.setattr(receiver_scan, "receiver_image", lambda *a, **k: base)
    result = captured(matrix=((0., 0.), (0., 0.)), descan=True)
    plane = result.state_snapshot.recording_planes[0]
    plane.key = "flu_screen"
    plane.hit_mask = lambda x, y: np.hypot(x, y) <= 8.
    result.scan_geometry.plane_positions_um["flu_screen"] = result.scan_geometry.plane_positions_um.pop("camera")
    preview = receiver_scan.receiver_scan_preview(result, "flu_screen", pixels=16)
    np.testing.assert_array_equal(preview.image.ideal_probability, image)
    np.testing.assert_array_equal(preview.image.response_probability, image)
    assert preview.image.response_probability.sum() == pytest.approx(.25)
    assert preview.metrics["stationary_dwell_weight"] == 1.


def test_decimated_raster_weights_represent_physical_pixel_dwell(physical_image):
    result = captured(nx=8, columns=[0, 3, 7])
    preview = receiver_scan.receiver_scan_preview(result, "camera", pixels=16)
    np.testing.assert_allclose(preview.dwell_weights, [[.125, .25, .125], [.125, .25, .125]])
    assert preview.dwell_weights.sum() == pytest.approx(1.)
    assert preview.metrics["decimated_raster"] is True
    assert "decimated" in preview.detail
    expected_shift = np.array((np.sum(preview.dwell_weights*preview.displacement_x_mm),
                               np.sum(preview.dwell_weights*preview.displacement_y_mm)))
    assert centroid(preview.image) == pytest.approx(centroid(physical_image)+expected_shift)


def test_frame_wrap_preserves_captured_baseline(physical_image):
    first = receiver_scan.receiver_scan_preview(captured(time=.375), "camera", pixels=16)
    later = receiver_scan.receiver_scan_preview(captured(time=3.375), "camera", pixels=16)
    np.testing.assert_allclose(first.image.response_probability, later.image.response_probability)


def test_shift_outside_sensor_loses_probability_without_wraparound(physical_image):
    result = captured(matrix=((40000., 0.), (0., 1000.)), time=.0625)
    preview = receiver_scan.receiver_scan_preview(result, "camera", pixels=16, scan_index=3)
    assert preview.image.response_probability.sum() == 0.


def test_zero_static_signal_does_not_invent_scan_hits(monkeypatch):
    monkeypatch.setattr(receiver_scan, "receiver_image", lambda *a, **k: static_image(zero=True))
    preview = receiver_scan.receiver_scan_preview(captured(), "camera", pixels=16)
    assert np.count_nonzero(preview.image.response_probability) == 0


@pytest.mark.parametrize("status", ["NOT_INSERTED", "NOT_REACHED", "NOT_CALCULATED", "READOUT_DISABLED"])
def test_unavailable_physical_receiver_keeps_only_geometry(monkeypatch, status):
    monkeypatch.setattr(receiver_scan, "receiver_image", lambda *a, **k: static_image(status=status))
    preview = receiver_scan.receiver_scan_preview(captured(), "camera", pixels=16)
    assert preview.status == status
    assert preview.image is None
    assert preview.times_s.shape == (2, 4)


def test_no_geometry_does_not_trigger_new_transport(physical_image):
    result = captured()
    result.scan_geometry = None
    preview = receiver_scan.receiver_scan_preview(result, "camera", pixels=16)
    assert preview.status == "UNAVAILABLE"
    assert preview.image is None
    assert preview.times_s is None
    assert "Calculate imaging" in preview.detail


def test_filter_transport_unavailable_stays_unavailable(physical_image):
    result = captured()
    result.scan_geometry.unavailable_planes["camera"] = "Energy-filter transported response unavailable"
    preview = receiver_scan.receiver_scan_preview(result, "camera", pixels=16)
    assert preview.status == "UNAVAILABLE"
    assert "Energy-filter" in preview.detail


@pytest.mark.parametrize("change", ["clock", "times", "nonlinear", "dimensions", "enabled"])
def test_mismatched_or_unsupported_captured_geometry_is_rejected(physical_image, change):
    result = captured(descan=True)
    if change == "clock":
        result.state_snapshot.descan_deflector.scan_frame_period_s = 2.
    elif change == "times":
        result.scan_geometry.times_s[0, 0] += .01
    elif change == "nonlinear":
        result.scan_geometry.plane_positions_um["camera"][0][0, 0] += 1.
    elif change == "dimensions":
        result.scan_geometry.requested_pixels_x = 7
    else:
        result.state_snapshot.ac_deflector.scan_enabled = False
    preview = receiver_scan.receiver_scan_preview(result, "camera", pixels=16)
    assert preview.status == "UNAVAILABLE"
    assert preview.image is None


@pytest.mark.parametrize("index", [-1, 8, True, 1.5])
def test_invalid_scan_position_rejected(physical_image, index):
    with pytest.raises(ValueError, match="position index"):
        receiver_scan.receiver_scan_preview(captured(), "camera", pixels=16, scan_index=index)


@pytest.mark.parametrize("exposure", [0., -1., float("nan"), float("inf")])
def test_invalid_exposure_rejected(physical_image, exposure):
    with pytest.raises(ValueError, match="exposure"):
        receiver_scan.receiver_scan_preview(captured(), "camera", pixels=16, exposure_s=exposure)


def test_prepared_playback_samples_geometry_and_physical_image_once(monkeypatch):
    calls = {"geometry": 0, "image": 0}
    original_geometry = receiver_scan._captured_raster
    base = static_image()

    def geometry(*args, **kwargs):
        calls["geometry"] += 1
        return original_geometry(*args, **kwargs)

    def image(*args, **kwargs):
        calls["image"] += 1
        return base

    monkeypatch.setattr(receiver_scan, "_captured_raster", geometry)
    monkeypatch.setattr(receiver_scan, "receiver_image", image)
    prepared = receiver_scan.prepare_receiver_scan(captured(), "camera", pixels=16)
    assert prepared.base_image is base
    for index in (0, 2, 7, 1, 7):
        prepared.preview(index)
        prepared.preview(index, accumulate=True)
    prepared.preview()
    assert calls == {"geometry": 1, "image": 1}


def test_prepared_moving_spot_uses_no_fft(physical_image, monkeypatch):
    import scipy.fft

    def unexpected_fft(*args, **kwargs):
        raise AssertionError("Single-position playback must not perform FFTs")

    monkeypatch.setattr(scipy.fft, "rfftn", unexpected_fft)
    prepared = receiver_scan.prepare_receiver_scan(captured(time=.375), "camera", pixels=16)
    for index in range(prepared.times_s.size):
        preview = prepared.preview(index)
        expected_shift = (prepared.displacement_x_mm.flat[index],
                          prepared.displacement_y_mm.flat[index])
        assert centroid(preview.image) == pytest.approx(centroid(physical_image)+expected_shift)
        assert preview.image.response_probability.sum() == pytest.approx(.25)
        assert preview.metrics["exposure_fraction"] == pytest.approx(.125)
        assert preview.metrics["accumulate"] is False


def test_accumulated_prefix_keeps_full_frame_dwell_normalization(physical_image):
    prepared = receiver_scan.prepare_receiver_scan(
        captured(nx=8, columns=[0, 3, 7]), "camera", pixels=16)
    expected = np.zeros_like(physical_image.response_probability)
    for index, weight in enumerate(prepared.dwell_weights.flat):
        expected += weight*prepared.preview(index).image.response_probability
        prefix = prepared.preview(index, accumulate=True)
        fraction = prepared.dwell_weights.ravel()[:index+1].sum()
        np.testing.assert_allclose(prefix.image.response_probability, expected, atol=1e-16)
        assert prefix.image.response_probability.sum() == pytest.approx(.25*fraction)
        assert prefix.metrics["exposure_fraction"] == pytest.approx(fraction)
        assert prefix.metrics["scan_index"] == index
        assert prefix.metrics["accumulate"] is True
        assert prefix.detail.startswith("Frame build-up preview")
        assert prefix.image.status == "GEOMETRIC_PREVIEW"
        assert prefix.image.expected_electrons is None
        assert prefix.image.current_pa is None
    full = prepared.preview()
    np.testing.assert_array_equal(prefix.image.response_probability, full.image.response_probability)
    assert full.metrics["exposure_fraction"] == 1.
    assert full.metrics["scan_index"] is None


def test_accumulated_preview_supports_seeking_backwards(physical_image):
    prepared = receiver_scan.prepare_receiver_scan(captured(), "camera", pixels=16)
    first = prepared.preview(1, accumulate=True)
    prepared.preview(6, accumulate=True)
    prepared.preview()
    repeated = prepared.preview(1, accumulate=True)
    np.testing.assert_array_equal(first.image.response_probability, repeated.image.response_probability)
    assert repeated.metrics["exposure_fraction"] == .25


def test_accumulation_reuses_seed_transforms(physical_image, monkeypatch):
    import scipy.fft
    original_fft = scipy.fft.rfftn
    seed_transforms = []

    def count_fft(values, *args, **kwargs):
        if values.ndim == 3:
            seed_transforms.append(values.shape)
        return original_fft(values, *args, **kwargs)

    monkeypatch.setattr(scipy.fft, "rfftn", count_fft)
    prepared = receiver_scan.prepare_receiver_scan(captured(), "camera", pixels=16)
    for index in (0, 3, 7, 2, 6, 7):
        prepared.preview(index, accumulate=True)
    assert len(seed_transforms) == 1


def test_prepared_metadata_and_trajectories_are_detached_and_readonly(physical_image):
    physical_image.metrics["coherence"] = {"description": ["captured"]}
    result = captured()
    prepared = receiver_scan.prepare_receiver_scan(result, "camera", pixels=16)
    original_times = prepared.times_s.copy()
    original_trajectory = prepared.trajectory_x_mm.copy()
    result.scan_geometry.times_s[:] = -1.
    result.scan_geometry.plane_positions_um["camera"][0][:] = -1.
    physical_image.metrics["coherence"]["description"][0] = "changed"
    first = prepared.preview(0)
    later = prepared.preview(2, accumulate=True)
    np.testing.assert_array_equal(prepared.times_s, original_times)
    np.testing.assert_array_equal(prepared.trajectory_x_mm, original_trajectory)
    assert first.times_s is later.times_s
    assert first.metrics["static_seed_coherence"]["description"] == ("captured",)
    with pytest.raises(ValueError):
        first.times_s[0, 0] = 10.
    with pytest.raises(TypeError):
        prepared.metrics["frame_period_s"] = 10.
    with pytest.raises(TypeError):
        first.image.metrics["actual_exposure"] = True
    with pytest.raises(TypeError):
        first.metrics["static_seed_coherence"]["description"] = "changed"


def test_stationary_edge_bin_build_up_is_exact_and_dwell_weighted(monkeypatch):
    axis = np.arange(16)-7.5
    image = np.zeros((16, 16))
    image[14, 13] = .25
    base = ReceiverImage(key="camera", name="Camera", status="AVAILABLE", detail="Exact hits",
        ideal_probability=image, response_probability=image.copy(), x_mm=axis, y_mm=axis,
        exposure_s=1., expected_electrons=None, current_pa=1., metrics={})
    monkeypatch.setattr(receiver_scan, "receiver_image", lambda *args, **kwargs: base)
    result = captured(matrix=((0., 0.), (0., 0.)), descan=True)
    result.state_snapshot.recording_planes[0].hit_mask = lambda x, y: np.hypot(x, y) <= 8.
    prepared = receiver_scan.prepare_receiver_scan(result, "camera", pixels=16)
    for index in range(prepared.times_s.size):
        prefix = prepared.preview(index, accumulate=True)
        np.testing.assert_array_equal(prefix.image.response_probability, image*((index+1)/8.))
        np.testing.assert_array_equal(prepared.preview(index).image.response_probability, image)


def test_preparation_preserves_unavailable_geometry_without_sampling(monkeypatch):
    def unexpected_sampling(*args, **kwargs):
        raise AssertionError("Missing geometry must not sample the receiver")

    monkeypatch.setattr(receiver_scan, "receiver_image", unexpected_sampling)
    result = captured()
    result.scan_geometry = None
    prepared = receiver_scan.prepare_receiver_scan(result, "camera", pixels=16)
    assert prepared.status == "UNAVAILABLE"
    assert prepared.base_image is None
    assert prepared.preview(0, accumulate=True).status == "UNAVAILABLE"
    assert "Calculate imaging" in prepared.detail


def test_compatibility_wrapper_accepts_accumulation(physical_image):
    preview = receiver_scan.receiver_scan_preview(
        captured(), "camera", pixels=16, scan_index=0, accumulate=True)
    assert preview.metrics["exposure_fraction"] == .125
    assert preview.image.response_probability.sum() == pytest.approx(.25/8.)
