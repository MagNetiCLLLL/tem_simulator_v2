"""Offline stream fakes only. No microscope access and no new exposure commands."""
from dataclasses import fields, replace
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace as NS

import numpy as np
import pytest
import tifffile

from temsim.recorder.backend import InstrumentRecorder
from temsim.recorder.records import CaptureRequest, digest
from temsim.recorder.snapshot import Cancelled, catalog, reading, serialise, utc_now


class Point:
    def __init__(self, x=0., y=0.):
        self.x, self.y = x, y


class Lens:
    name = "OBJECTIVE"
    value = 0.42
    value_raw = 42000.
    limits_raw = {"min": 0., "max": 100000.}


class FailingGainDetector:
    name = "HAADF"
    display_name = "Test annular detector"
    is_insertable = True
    insertion_state = "Inserted"

    @property
    def gain(self):
        raise RuntimeError("Gain is not readable on this detector")


class FakeStream:
    def __init__(self, client, camera=None):
        self.client, self.camera = client, camera

    def wait_for_next_frame(self, detector_type=None):
        name = self.camera if self.camera is not None else detector_type
        self.client.calls.append(("frame", name))
        if name in self.client.fail_channels:
            raise RuntimeError("No current frame available")
        started = utc_now()
        metadata = NS(
            metadata_as_xml="<metadata><pixel_size unit='m'>1e-10</pixel_size><dwell>0.0000025</dwell></metadata>",
            acquisition=NS(acquisition_id=f"frame-{len(self.client.calls)}",
                           acquisition_start_date_time=started, acquisition_date_time=utc_now()),
            binary_result=NS(detector=name, image_size=Point(128, 96), pixel_size=Point(1e-10, 1e-10)),
            scan_settings=NS(dwell_time=2.5e-6, scan_size=Point(128, 96), frame_time=0.03072),
            imaging_detectors=[NS(detector_name=name, detector_type=name, exposure_time=0.037,
                                  exposure_time_2=None, binning=Point(2, 2), readout_area=None, post_magnification=1.)],
        )
        image = NS(data=self.client.data.copy(), metadata=metadata)
        self.client.transform_frame(image)
        self.client.after_frame()
        return image


class FakeClient:
    def __init__(self):
        self.calls = []
        self.after_frame = lambda: None
        self.transform_frame = lambda image: None
        self.fail_channels = set()
        self.disconnect_error = False
        # Deliberately non-square: no square image-size restriction remains.
        self.data = np.arange(128 * 96, dtype=np.uint16).reshape(96, 128)
        self.service = NS(autoscript=NS(server=NS(is_offline=False, version="1.18.0"),
                                       client=NS(version="1.18.0")),
                          system=NS(name="Offline test double", serial_number="FAKE-001", version="test"))
        self.lens = Lens()
        self.optics = NS(optical_mode="Tem", projector_mode="Imaging", objective_lens_mode="HM",
                         magnification=NS(value={"nominal": 10000, "calibrated": 9998}),
                         camera_length=NS(value={"nominal": 0.5, "calibrated": 0.498}),
                         lenses=NS(get_available=lambda: ["OBJECTIVE"], get_lens=lambda name: self.lens),
                         deflectors=NS(beam_shift=Point(1e-9, -2e-9)))
        self.source = NS(acceleration_voltage=NS(value=300000.))
        self.specimen = NS(stage=NS(is_moving=False, position={"x": 0., "y": 0., "z": 0.}),
                           piezo_stage=NS(is_moving=False, position={"x": 0., "y": 0., "z": 0.}))
        self.scanner = FailingGainDetector()
        self.scanner.get_segments = lambda: [NS(name="ALL", detector_name="HAADF", is_enabled=True,
                                                gain=0.5, offset=0.01)]
        self.detectors = NS(camera_detectors=["BM-Ceta", "Flucam"], scanning_detectors=["HAADF", "DF", "BF"],
                            eds_detectors=[], eels_detectors=[], screen=NS(position="Inserted"),
                            get_camera_detector=lambda n: NS(name=n, display_name=n),
                            get_scanning_detector=self.scanning_detector)
        self.acquisition = NS(continuous_camera_acquisitions=[FakeStream(self, "BM-Ceta"), FakeStream(self, "Flucam")],
                              continuous_stem_acquisitions=[FakeStream(self)])
        # Any accidental hidden-default exposure or stream reconfiguration fails the offline test.
        def forbidden(*args, **kwargs):
            raise AssertionError("Recorder must not start an exposure or reconfigure an acquisition")
        for method in ("acquire_camera_image", "acquire_stem_images", "acquire_stem_image",
                       "start_acquire_continuous_camera_images", "start_acquire_stem_images_advanced"):
            setattr(self.acquisition, method, forbidden)

    def scanning_detector(self, name):
        if name == "HAADF":
            return self.scanner
        return NS(name=name, display_name=name, get_segments=lambda: [])

    def connect(self, host, port):
        self.calls.append(("connect", host, port))

    def disconnect(self):
        self.calls.append(("disconnect",))
        if self.disconnect_error:
            raise RuntimeError("Disconnect not confirmed")


@pytest.fixture
def connected():
    fake = FakeClient()
    backend = InstrumentRecorder(lambda: fake)
    backend.connect("offline-test-double.invalid")
    return backend, fake


@pytest.fixture
def request_data(tmp_path):
    return CaptureRequest(output=str(tmp_path), route="ceta")


def test_request_has_no_sample_assumptions_or_acquisition_defaults():
    assert {f.name for f in fields(CaptureRequest)} == {"output", "route"}


def test_catalog_is_packaged_and_has_required_optical_families():
    data = catalog()
    assert data["sdk_version"] == "1.18.0"
    paths = {p["path"] for p in data["properties"]}
    assert len(paths) == len(data["properties"])
    assert {"source.field_emission_gun.gun_lens_value", "optics.camera_length.value",
            "optics.magnification.value", "optics.stigmators_raw.objective_stigmator"} <= paths


def test_snapshot_keeps_raw_values_and_unavailable_readbacks(connected):
    backend, fake = connected
    result = backend.snapshot()
    rows = {r["path"]: r for r in result["readings"]}
    assert rows['optics.lenses.get_lens["OBJECTIVE"].value_raw']["value"] == 42000.
    assert rows["optics.deflectors.beam_shift"]["value"]["x"] == 1e-9
    gain = rows['detectors.get_scanning_detector["HAADF"].gain']
    assert gain["status"] == "unavailable" and "not readable" in gain["error"]
    assert "value" not in gain
    assert result["coverage"]["unavailable"] > 0
    assert len(fake.calls) == 1


def test_current_frame_bundle_is_lossless_and_metadata_linked(connected, request_data):
    backend, fake = connected
    result = backend.capture(request_data)
    path = Path(result["path"])
    manifest = json.loads((path / "manifest.json").read_text("utf-8"))
    assert manifest["schema"] == "temsim.instrument-record/3"
    assert manifest["status"] == "recorded_with_warnings"
    assert manifest["sample_identity"] is None
    assert "particle" not in manifest["request"]
    assert "Au" not in path.name
    item = manifest["images"][0]
    assert item["detector"] == "BM-Ceta"
    np.testing.assert_array_equal(np.load(path / item["path"]), fake.data)
    np.testing.assert_array_equal(tifffile.imread(path / "channel_001/image.tiff"), fake.data)
    metadata = json.loads((path / "channel_001/image_metadata.json").read_text())
    values = {r["path"]: r.get("value") for r in metadata["readings"]}
    assert values["scan_settings.dwell_time"] == 2.5e-6
    assert values["imaging_detectors[0].exposure_time"] == 0.037
    assert "0.0000025" in (path / "channel_001/vendor_metadata.xml").read_text()
    assert item["shape"] == [96, 128]
    assert item["timing"]["status"] == "timestamps_within_collection_window"
    for name, info in manifest["artifacts"].items():
        assert digest(path / name) == info["sha256"]
    row = json.loads((path.parent / "records.jsonl").read_text().splitlines()[0])
    assert row["record_id"] == manifest["record_id"] and row["images"] == manifest["images"]
    assert not any(c[0] in ("camera", "stem") for c in fake.calls)


@pytest.mark.parametrize("route,optical,projector", [
    ("ceta", "Tem", "Imaging"), ("ceta", "Tem", "Diffraction"),
    ("flucam", "Tem", "Imaging"), ("flucam", "Tem", "Diffraction"),
    ("flucam", "Stem", "Imaging"), ("flucam", "Stem", "Diffraction"),
    ("stem_all", "Stem", "Diffraction"),
])
def test_routes_preserve_actual_modes(connected, request_data, route, optical, projector):
    backend, fake = connected
    fake.optics.optical_mode, fake.optics.projector_mode = optical, projector
    result = backend.capture(replace(request_data, route=route))
    assert result["manifest"]["status"].startswith("recorded")
    assert (fake.optics.optical_mode, fake.optics.projector_mode) == (optical, projector)
    assert f"{optical}_{projector}" in result["path"]


@pytest.mark.parametrize("fault", ["mode", "identity", "offline", "no_stream", "no_detector"])
def test_refusal_without_exposure_or_fake_data(connected, request_data, fault):
    backend, fake = connected
    if fault == "mode":
        fake.optics.optical_mode = "Stem"
    elif fault == "identity":
        fake.service.system.serial_number = "OTHER"
    elif fault == "offline":
        fake.service.autoscript.server.is_offline = True
    elif fault == "no_stream":
        fake.acquisition.continuous_camera_acquisitions = []
    else:
        fake.detectors.camera_detectors = []
    with pytest.raises(RuntimeError):
        backend.capture(request_data)
    assert len(fake.calls) == 1 and not list(Path(request_data.output).iterdir())


def test_changed_mode_during_snapshot_is_not_silently_accepted(connected, request_data, monkeypatch):
    backend, fake = connected
    original = backend.snapshot
    def read(*args):
        result = original(*args)
        fake.optics.projector_mode = "Diffraction"
        return result
    monkeypatch.setattr(backend, "snapshot", read)
    result = backend.capture(request_data)
    assert result["manifest"]["status"] == "failed" and len(fake.calls) == 1


def test_all_stem_collects_each_channel_without_new_scan(connected, request_data):
    backend, fake = connected
    fake.optics.optical_mode = "Stem"
    result = backend.capture(replace(request_data, route="stem_all"))
    assert fake.calls[1:] == [("frame", "HAADF"), ("frame", "DF"), ("frame", "BF")]
    assert [i["detector"] for i in result["manifest"]["images"]] == ["HAADF", "DF", "BF"]
    assert result["manifest"]["channel_coverage"]["complete"]
    assert result["manifest"]["quality"]["simultaneous_channels"].startswith("not established")


def test_partial_stem_frame_failure_keeps_other_channels(connected, request_data):
    backend, fake = connected
    fake.optics.optical_mode = "Stem"
    fake.fail_channels.add("DF")
    result = backend.capture(replace(request_data, route="stem_all"))
    assert result["manifest"]["status"] == "partial_channels"
    assert result["manifest"]["channel_coverage"]["missing"] == ["DF"]
    assert [i["detector"] for i in result["manifest"]["images"]] == ["HAADF", "BF"]


def test_ceta_is_selected_by_frame_identity_not_default_camera(connected, request_data):
    backend, fake = connected
    fake.detectors.camera_detectors.append("EF-Ceta")
    fake.acquisition.continuous_camera_acquisitions = [FakeStream(fake, "Flucam"), FakeStream(fake, "EF-Ceta")]
    result = backend.capture(request_data)
    assert [i["detector"] for i in result["manifest"]["images"]] == ["EF-Ceta"]


def test_unidentified_camera_frame_is_never_assumed_ceta(connected, request_data):
    backend, fake = connected
    fake.acquisition.continuous_camera_acquisitions = [FakeStream(fake, "BM-Ceta")]
    fake.transform_frame = lambda image: setattr(image, "metadata", None)
    result = backend.capture(request_data)
    assert result["manifest"]["status"] == "partial_channels"
    assert result["manifest"]["images"][0]["detector"] is None
    assert "image.npy" in result["manifest"]["images"][0]["path"]


@pytest.mark.parametrize("timestamp", [None, "2026-09-12T10:00:00", "bad"])
def test_missing_or_timezone_unknown_frame_time_is_not_assumed_current(connected, request_data, timestamp):
    backend, fake = connected
    def transform(image):
        image.metadata.acquisition.acquisition_start_date_time = timestamp
        image.metadata.acquisition.acquisition_date_time = timestamp
    fake.transform_frame = transform
    result = backend.capture(request_data)
    assert result["manifest"]["status"] == "frame_state_unverified"


def test_old_frame_not_relabelled_with_current_state(connected, request_data):
    backend, fake = connected
    def transform(image):
        image.metadata.acquisition.acquisition_start_date_time = "2000-01-01T00:00:00+00:00"
        image.metadata.acquisition.acquisition_date_time = "2000-01-01T00:00:01+00:00"
    fake.transform_frame = transform
    result = backend.capture(request_data)
    assert result["manifest"]["status"] == "frame_state_mismatch"
    assert result["manifest"]["images"][0]["timing"]["vendor_start"].startswith("2000-")


def test_images_survive_failed_post_readbacks(connected, request_data, monkeypatch):
    backend, fake = connected
    original = backend.snapshot
    def read(*args):
        if any(call[0] == "frame" for call in fake.calls):
            raise RuntimeError("Network lost after frame")
        return original(*args)
    monkeypatch.setattr(backend, "snapshot", read)
    result = backend.capture(request_data)
    assert result["manifest"]["status"] == "image_saved_incomplete_metadata"
    assert (Path(result["path"]) / result["manifest"]["images"][0]["path"]).exists()


def test_stop_during_frame_preserves_pixels_and_post_snapshot(connected, request_data):
    backend, fake = connected
    cancel = Event()
    fake.after_frame = cancel.set
    result = backend.capture(request_data, cancel)
    assert len(result["manifest"]["images"]) == 1
    assert result["manifest"]["stop_requested"]
    assert "parameters_after.json" in result["manifest"]["artifacts"]
    assert len(fake.calls) == 2


def test_stop_before_frame_does_not_read_signal(connected, request_data):
    backend, fake = connected
    cancel = Event()
    result = backend.capture(request_data, cancel, lambda _: cancel.set())
    assert result["manifest"]["status"] == "cancelled_before_signal"
    assert len(fake.calls) == 1


def test_cancelled_before_start_has_no_record(connected, request_data):
    cancel = Event()
    cancel.set()
    with pytest.raises(Cancelled):
        connected[0].capture(request_data, cancel)


def test_guard_marks_optical_state_changes(connected, request_data):
    backend, fake = connected
    fake.after_frame = lambda: setattr(fake.lens, "value_raw", 43000.)
    result = backend.capture(request_data)
    assert result["manifest"]["status"] == "frame_state_unverified"
    assert "optics.lenses[OBJECTIVE].value_raw" in result["manifest"]["quality"]["changed_guard_fields"]


def test_repeated_records_never_overwrite(connected, request_data):
    backend, _ = connected
    a, b = backend.capture(request_data), backend.capture(request_data)
    assert a["path"] != b["path"]
    assert len((Path(request_data.output) / "records.jsonl").read_text().splitlines()) == 2


def test_no_current_frames_returns_failure_not_fake_pixels(connected, request_data):
    backend, fake = connected
    fake.fail_channels = {"BM-Ceta", "Flucam"}
    result = backend.capture(request_data)
    assert result["manifest"]["status"] == "failed"
    assert result["previews"] == [] and result["manifest"]["images"] == []


def test_connect_rejects_offline_or_unknown_version():
    for offline, version in ((True, "1.18.0"), (False, "1.19.0"), (None, "1.18.0")):
        fake = FakeClient()
        fake.service.autoscript.server.is_offline = offline
        fake.service.autoscript.server.version = version
        backend = InstrumentRecorder(lambda: fake)
        with pytest.raises(RuntimeError):
            backend.connect("fake.invalid")
        assert backend.client is None and fake.calls[-1] == ("disconnect",)


def test_disconnect_failure_keeps_handle(connected):
    backend, fake = connected
    fake.disconnect_error = True
    with pytest.raises(RuntimeError, match="not confirmed"):
        backend.disconnect()
    assert backend.client is fake


def test_serialisation_does_not_guess_unknown_sdk_objects():
    with pytest.raises(TypeError, match="Uncatalogued"):
        serialise(object())
    assert serialise(float("nan")) == {"nonfinite": "nan"}
    assert reading("bad_value", lambda: float("nan"))[0]["status"] == "partial"


def test_index_failure_does_not_destroy_bundle(connected, request_data):
    (Path(request_data.output) / "records.jsonl").mkdir()
    result = connected[0].capture(request_data)
    assert (Path(result["path"]) / "manifest.json").exists()
    assert "Index append failed" in result["manifest"]["warnings"][-1]


def test_failed_raw_write_never_claims_success(connected, request_data, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("Disk full")
    monkeypatch.setattr(np, "save", fail)
    result = connected[0].capture(request_data)
    assert result["manifest"]["status"] == "failed"
    assert any("Disk full" in w for w in result["manifest"]["warnings"])


def test_one_invalid_stem_frame_does_not_block_later_files(connected, request_data):
    backend, fake = connected
    fake.optics.optical_mode = "Stem"
    def transform(image):
        if image.metadata.binary_result.detector == "HAADF":
            image.data = np.array([])
    fake.transform_frame = transform
    result = backend.capture(replace(request_data, route="stem_all"))
    assert len(result["manifest"]["images"]) == 2
    assert [item["path"] for item in result["manifest"]["images"]] == [
        "channel_002/image.npy", "channel_003/image.npy"]


def test_unknown_named_detector_retains_unidentified_pixels(connected, request_data):
    backend, fake = connected
    fake.acquisition.continuous_camera_acquisitions = [FakeStream(fake, "BM-Ceta")]
    fake.transform_frame = lambda image: setattr(image.metadata.binary_result, "detector", "Unknown alias")
    result = backend.capture(request_data)
    assert result["manifest"]["status"] == "partial_channels"
    assert result["manifest"]["images"][0]["reported_detector"] == "Unknown alias"
    assert result["manifest"]["images"][0]["detector"] is None


def test_real_sdk_metadata_structures_offline(tmp_path):
    """Structure constructors only: no client, connection or hardware execution."""
    sdk = pytest.importorskip("autoscript_tem_microscope_client.structures")
    from temsim.recorder.frame_metadata import collect_frame_metadata, frame_timing
    from temsim.recorder.records import RecordBundle

    metadata = sdk.AdornedImageMetadata(
        acquisition=sdk.AdornedImageMetadataAcquisition(
            acquisition_id="offline-structure-fixture", acquisition_start_date_time="2026-09-12T00:00:01Z",
            acquisition_date_time="2026-09-12T00:00:02Z"),
        binary_result=sdk.AdornedImageMetadataBinaryResult(detector="BM-Ceta", image_size=(12, 6)),
        scan_settings=sdk.AdornedImageMetadataScanSettings(dwell_time=3e-6, scan_size=(12, 6)),
        imaging_detectors=[sdk.AdornedImageMetadataImagingDetector(
            detector_name="BM-Ceta", exposure_time=0.123, binning=(2, 2))],
    )
    image = sdk.AdornedImage(np.arange(72, dtype=np.uint16).reshape(6, 12), metadata)
    result = collect_frame_metadata(image)
    values = {r["path"]: r.get("value") for r in result["readings"]}
    assert values["scan_settings.dwell_time"] == 3e-6
    assert values["imaging_detectors[0].exposure_time"] == 0.123
    assert values["binary_result.image_size"]["x"] == 12
    timing = frame_timing(result, "2026-09-12T00:00:00+00:00", "2026-09-12T00:00:03+00:00")
    assert timing["status"] == "timestamps_within_collection_window"
    bundle = RecordBundle(CaptureRequest(str(tmp_path)), {}, {"optical_mode": "Tem", "projector_mode": "Imaging"})
    bundle.save_image(image, {"detector": "BM-Ceta"}, "channel_001")
    assert (bundle.path / "channel_001/vendor_metadata.xml").read_text("utf-8") == metadata.metadata_as_xml
