from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics.first_order import TransverseTransfer
from temsim.physics.fourdstem import (
    FourDSTEMCalibration,
    FourDSTEMWriter,
    FourDSTEMCaptureSink,
    PixelatedDetectorResponse,
    VirtualDetector,
    annular_virtual_detector,
    integrate_runtime_recording_planes,
    integrate_virtual_detectors,
    open_fourdstem,
    runtime_detector_images_match_plan,
)
from temsim.physics.fourdstem_workflow import (
    FourDSTEMRequest,
    derive_fourdstem_products,
    prepare_fourdstem_capture,
)
from temsim.physics.record_plane import PlaneStop, RecordPlanePlan
from temsim.physics.stem_wave_imaging import (
    AngularDetector,
    simulate_angle_resolved_stem,
)
from temsim.optics.column import default_state
from temsim.assembly_catalog import AssemblyCatalog


def _calibration(scan_shape=(2, 3), detector_shape=(3, 4)):
    scan_y, scan_x = scan_shape
    detector_y, detector_x = detector_shape
    return FourDSTEMCalibration.from_rectilinear_axes(
        np.linspace(-1.0, 1.0, scan_x),
        np.linspace(-2.0, 2.0, scan_y),
        np.linspace(-6.0, 6.0, detector_x),
        np.linspace(-4.0, 4.0, detector_y),
    )


def _transfer(z_mm, *, j_img=None, j_diff=None):
    return TransverseTransfer(
        0.0,
        float(z_mm),
        np.eye(2) if j_img is None else np.asarray(j_img, dtype=float),
        np.zeros((2, 2)) if j_diff is None else np.asarray(j_diff, dtype=float),
        np.zeros((2, 2)),
        np.eye(2),
    )


def _plan(planes, transfers, fingerprint="a" * 64):
    return RecordPlanePlan(
        source_z_mm=0.0,
        planes=tuple(planes),
        transfers=tuple(transfers),
        resolved_geometry_fingerprint="b" * 64,
        fingerprint=fingerprint,
    )


def test_detector_response_applies_calibrated_order_and_saturation():
    response = PixelatedDetectorResponse(
        quantum_efficiency=0.5,
        dark_electrons_per_pixel=1.0,
        saturation_electrons=4.0,
        gain_counts_per_electron=3.0,
        offset_counts=2.0,
        poisson_enabled=False,
        status="synthetic_unit_test",
    )
    expected = np.array(((0.0, 2.0), (6.0, 20.0)))

    counts = response.apply(expected)

    # min((expected * QE) + dark, saturation) * gain + offset
    assert counts == pytest.approx(np.array(((5.0, 8.0), (14.0, 14.0))))
    assert counts.dtype == np.float32


def test_detector_response_seed_makes_poisson_and_read_noise_reproducible():
    response = PixelatedDetectorResponse(
        poisson_enabled=True,
        read_noise_electrons_rms=0.25,
        seed=1234,
        status="synthetic_unit_test",
    )
    expected = np.full((16, 16), 5.0)

    first = response.apply(expected)
    second = response.apply(expected)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, expected.astype(np.float32))


def test_streaming_writer_finalizes_valid_memmap_and_preserves_provenance(tmp_path):
    calibration = _calibration()
    path = tmp_path / "scan.npy"
    writer = FourDSTEMWriter(
        path,
        calibration,
        record_plane_plan=_plan((), (), fingerprint="c" * 64),
        provenance={"source": "unit_test"},
    )
    frames = []
    for y_index in range(calibration.shape[0]):
        for x_index in range(calibration.shape[1]):
            frame = np.full(calibration.shape[-2:], 10 * y_index + x_index, dtype=float)
            frames.append((y_index, x_index, frame))
    writer.write_stream(frames, checkpoint_every=2)

    artifact = writer.finalize()

    assert isinstance(artifact.data, np.memmap)
    assert artifact.data.shape == calibration.shape
    assert artifact.data[1, 2, 0, 0] == pytest.approx(12.0)
    assert artifact.metadata["record_plane_plan_fingerprint"] == "c" * 64
    assert artifact.metadata["resolved_geometry_fingerprint"] == "b" * 64
    assert artifact.metadata["completed_frames"] == artifact.metadata["total_frames"]
    assert artifact.metadata["axis_order"] == [
        "scan_y", "scan_x", "detector_y", "detector_x"
    ]
    assert open_fourdstem(path).calibration.digest == calibration.digest
    assert not path.with_name(path.name + ".partial").exists()


def test_partial_stream_resumes_without_overwriting_completed_frames(tmp_path):
    calibration = _calibration(scan_shape=(2, 2), detector_shape=(2, 2))
    path = tmp_path / "resume.npy"
    response = PixelatedDetectorResponse(status="synthetic_unit_test")
    writer = FourDSTEMWriter(
        path,
        calibration,
        response=response,
        record_plane_plan=_plan((), (), fingerprint="d" * 64),
    )
    writer.write_frame(0, 0, np.full((2, 2), 7.0))
    writer.close(keep_partial=True)

    resumed = FourDSTEMWriter(
        path,
        calibration,
        response=response,
        record_plane_plan=_plan((), (), fingerprint="d" * 64),
        resume=True,
    )
    assert resumed.completed_frames == 1
    for y_index, x_index in ((0, 1), (1, 0), (1, 1)):
        resumed.write_frame(y_index, x_index, np.full((2, 2), y_index + x_index))
    artifact = resumed.finalize()

    assert np.all(artifact.data[0, 0] == 7.0)
    assert np.all(artifact.data[1, 1] == 2.0)


def test_resume_rejects_a_different_source_or_dose_provenance(tmp_path):
    calibration = _calibration(scan_shape=(1, 2), detector_shape=(2, 2))
    path = tmp_path / "source-mismatch.npy"
    writer = FourDSTEMWriter(
        path,
        calibration,
        provenance={"source_signature": "source-a", "electrons_per_frame": 10.0},
    )
    writer.write_frame(0, 0, np.ones((2, 2)))
    writer.close(keep_partial=True)

    with pytest.raises(ValueError, match="different source calculation"):
        FourDSTEMWriter(
            path,
            calibration,
            provenance={"source_signature": "source-b", "electrons_per_frame": 20.0},
            resume=True,
        )


def test_resume_preserves_detector_random_stream(tmp_path):
    calibration = _calibration(scan_shape=(1, 3), detector_shape=(3, 3))
    response = PixelatedDetectorResponse(
        poisson_enabled=True,
        read_noise_electrons_rms=0.2,
        seed=991,
        status="synthetic_unit_test",
    )
    expected = np.full((3, 3), 4.5)
    uninterrupted_path = tmp_path / "uninterrupted.npy"
    uninterrupted = FourDSTEMWriter(
        uninterrupted_path,
        calibration,
        response=response,
    )
    for x_index in range(3):
        uninterrupted.write_frame(0, x_index, expected)
    uninterrupted_data = np.asarray(uninterrupted.finalize().data).copy()

    resumed_path = tmp_path / "resumed.npy"
    first = FourDSTEMWriter(resumed_path, calibration, response=response)
    first.write_frame(0, 0, expected)
    first.close(keep_partial=True)
    second = FourDSTEMWriter(
        resumed_path,
        calibration,
        response=response,
        resume=True,
    )
    second.write_frame(0, 1, expected)
    second.write_frame(0, 2, expected)
    resumed_data = np.asarray(second.finalize().data).copy()

    assert np.array_equal(resumed_data, uninterrupted_data)


def test_arbitrary_virtual_detector_integration_is_chunk_invariant():
    calibration = _calibration(scan_shape=(5, 4), detector_shape=(3, 4))
    cube = np.arange(np.prod(calibration.shape), dtype=float).reshape(calibration.shape)
    checkerboard = VirtualDetector(
        "checkerboard",
        (np.indices(calibration.shape[-2:]).sum(axis=0) % 2).astype(float),
    )
    annulus = annular_virtual_detector("annulus", calibration, 2.0, 8.0)

    point_one = integrate_virtual_detectors(cube, (checkerboard, annulus), chunk_scan_points=1)
    point_seven = integrate_virtual_detectors(cube, (checkerboard, annulus), chunk_scan_points=7)

    for detector in (checkerboard, annulus):
        reference = np.einsum("yxij,ij->yx", cube, detector.weights)
        assert point_one[detector.key] == pytest.approx(reference)
        assert point_seven[detector.key] == pytest.approx(reference)


def test_runtime_detector_integration_uses_j_img_and_sequential_interception():
    # Scan positions are intentionally large enough for the image term to
    # move one probe outside the first detector.  An angle-only implementation
    # would incorrectly assign both scans to BF.
    calibration = FourDSTEMCalibration.from_rectilinear_axes(
        np.array((0.0, 2_000.0)),  # um -> 0 and 2 mm
        np.array((0.0,)),
        np.array((0.0,)),
        np.array((0.0,)),
    )
    cube = np.ones(calibration.shape)
    bf = PlaneStop(
        "bf", "BF", 1.0, "detector", "disk", outer_width_mm=2.0,
        readout_enabled=True,
    )
    camera = PlaneStop(
        "camera", "Camera", 2.0, "detector", "square", outer_width_mm=10.0,
        readout_enabled=True,
    )
    plan = _plan((bf, camera), (_transfer(1.0), _transfer(2.0)))

    result = integrate_runtime_recording_planes(
        cube,
        calibration,
        plan,
        chunk_scan_points=1,
    )

    assert result.images["bf"] == pytest.approx(np.array(((1.0, 0.0),)))
    assert result.images["camera"] == pytest.approx(np.array(((0.0, 1.0),)))
    assert result.surviving_weight == pytest.approx(np.array(((0.0, 0.0),)))
    assert result.balance_error == pytest.approx(0.0, abs=1.0e-15)


def test_virtual_detector_applies_dose_and_pixel_response_after_raw_cube():
    calibration = _calibration(scan_shape=(1, 2), detector_shape=(2, 2))
    cube = np.full(calibration.shape, 0.25, dtype=float)
    detector = VirtualDetector("all", np.ones((2, 2)))

    integrated = integrate_virtual_detectors(
        cube,
        (detector,),
        response=PixelatedDetectorResponse(quantum_efficiency=0.5),
        electrons_per_frame=np.asarray(((100.0, 200.0),)),
        chunk_scan_points=1,
    )

    assert integrated["all"] == pytest.approx(
        np.asarray(((50.0, 100.0),))
    )


def test_raw_cube_can_be_rerouted_but_cached_detector_images_reject_old_plan(tmp_path):
    calibration = _calibration(scan_shape=(1, 1), detector_shape=(1, 1))
    path = tmp_path / "geometry.npy"
    stored_plan = _plan((), (), fingerprint="1" * 64)
    writer = FourDSTEMWriter(path, calibration, record_plane_plan=stored_plan)
    writer.write_frame(0, 0, np.ones((1, 1)))
    artifact = writer.finalize()
    detector = PlaneStop(
        "camera", "Camera", 1.0, "detector", "square", outer_width_mm=10.0,
        readout_enabled=True,
    )
    different_plan = _plan((detector,), (_transfer(1.0),), fingerprint="2" * 64)

    rerouted = integrate_runtime_recording_planes(artifact, None, different_plan)

    assert rerouted.plan_fingerprint == different_plan.fingerprint
    assert rerouted.plan_changed_since_capture is True
    assert rerouted.capture_plan_fingerprint == stored_plan.fingerprint
    assert runtime_detector_images_match_plan(rerouted, different_plan) is True
    assert runtime_detector_images_match_plan(rerouted, stored_plan) is False


def test_physical_routing_rejects_cube_after_non_neutral_pixel_response(tmp_path):
    calibration = _calibration(scan_shape=(1, 1), detector_shape=(1, 1))
    detector = PlaneStop(
        "camera", "Camera", 1.0, "detector", "square", outer_width_mm=10.0,
        readout_enabled=True,
    )
    plan = _plan((detector,), (_transfer(1.0),))
    writer = FourDSTEMWriter(
        tmp_path / "response.npy",
        calibration,
        response=PixelatedDetectorResponse(
            quantum_efficiency=0.7,
            status="synthetic_unit_test",
        ),
    )
    writer.write_frame(0, 0, np.ones((1, 1)))
    artifact = writer.finalize()

    with pytest.raises(ValueError, match="detector response belongs after"):
        integrate_runtime_recording_planes(artifact, None, plan)


def test_stem_wave_solver_streams_configuration_averaged_diffraction_cube(tmp_path):
    state = default_state()
    state.acceleration_enabled = False
    state.illumination_mode = "STEM"
    state.sample.specimen_preset_key = "vacuum"
    state.sample.thickness_nm = 0.0
    state.sample.wave_grid_pixels = 32
    state.sample.wave_field_of_view_angstrom = 16.0
    state.sample.wave_multislice_enabled = False
    state.sample.wave_atomistic_enabled = False
    incident = SimpleNamespace(
        alive=np.ones(5, dtype=bool),
        ray_weight=np.asarray((0.6, 0.1, 0.1, 0.1, 0.1)),
        x=np.zeros((1, 5)),
        y=np.zeros((1, 5)),
        tx=np.asarray(((0.0, 2.0e-3, -2.0e-3, 0.0, 0.0),)),
        ty=np.asarray(((0.0, 0.0, 0.0, 2.0e-3, -2.0e-3),)),
    )
    scan_x = np.asarray(((0.0, 1.0e-4),))
    scan_y = np.zeros_like(scan_x)
    sink = FourDSTEMCaptureSink(
        tmp_path / "wave_cube.npy",
        electrons_per_frame=100.0,
        response=PixelatedDetectorResponse(status="synthetic_unit_test"),
    )

    result = simulate_angle_resolved_stem(
        state,
        SimpleNamespace(incident=incident),
        (AngularDetector("bf", 0.0, 10.0),),
        scan_x,
        scan_y,
        diffraction_sink=sink,
    )

    artifact = result.fourdstem_artifact
    assert artifact is not None
    assert artifact.data.shape[:2] == scan_x.shape
    assert artifact.data.shape[-2:] == (32, 32)
    # Invalid reciprocal pixels are deliberately zeroed without
    # renormalisation, so stored charge cannot exceed the incident dose.
    assert np.all(np.sum(artifact.data, axis=(-2, -1)) <= 100.0 + 1.0e-4)
    assert artifact.metadata["provenance"]["input_frame_quantity"] == (
        "configuration-averaged diffraction probability"
    )


def test_high_accuracy_adapter_separates_cube_and_downstream_plan_signatures(tmp_path):
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    state.acceleration_enabled = False
    state.illumination_mode = "STEM"
    state.ac_deflector.scan_pixels_x = 2
    state.ac_deflector.scan_lines = 1
    state.ac_deflector.scan_frame_period_s = 0.02
    state.sample.specimen_preset_key = "vacuum"
    state.sample.thickness_nm = 0.0
    state.sample.wave_grid_pixels = 32
    state.sample.wave_field_of_view_angstrom = 16.0
    state.sample.wave_multislice_enabled = False
    state.sample.wave_atomistic_enabled = False
    incident = SimpleNamespace(
        alive=np.asarray((True, True, False)),
        ray_weight=np.asarray((0.4, 0.4, 0.2)),
        x=np.zeros((1, 3)),
        y=np.zeros((1, 3)),
        tx=np.asarray(((0.0, 2.0e-3, -2.0e-3),)),
        ty=np.asarray(((0.0, 0.0, 0.0),)),
    )
    simulation = SimpleNamespace(incident=incident)
    prepared = prepare_fourdstem_capture(
        state,
        simulation,
        FourDSTEMRequest(
            tmp_path / "workflow.npy",
            frame_count=2,
            overwrite=True,
        ),
        maximum_step_mm=5.0,
    )
    assert prepared.incident_fraction == pytest.approx(0.8)
    assert prepared.dwell_time_s == pytest.approx(0.01)
    assert prepared.expected_electrons_per_frame > 0.0

    scan_x = np.asarray(((0.0, 1.0e-4),))
    scan_y = np.zeros_like(scan_x)
    wave = simulate_angle_resolved_stem(
        state,
        simulation,
        (AngularDetector("bf", 0.0, 10.0),),
        scan_x,
        scan_y,
        diffraction_sink=prepared.sink,
    )
    virtual = annular_virtual_detector(
        "central",
        wave.fourdstem_artifact.calibration,
        0.0,
        10.0,
    )
    products = derive_fourdstem_products(
        prepared,
        wave.fourdstem_artifact,
        virtual_detectors=(virtual,),
        chunk_scan_points=1,
    )
    assert products.virtual_detector_images["central"].shape == scan_x.shape
    assert abs(products.physical_recording.balance_error) <= 1.0e-7
    provenance = wave.fourdstem_artifact.metadata["provenance"]
    assert provenance["fourdstem_cube_state_signature"] == (
        prepared.cube_state_signature
    )
    assert provenance["record_plane_plan"]["fingerprint"] == (
        prepared.record_plane_plan.fingerprint
    )
    assert provenance["record_plane_plan"]["resolved_geometry_fingerprint"]
    assert all(
        len(transfer["matrix"]) == 4
        for transfer in provenance["record_plane_plan"]["transfers"]
    )

    original_cube_signature = prepared.cube_dependency_signature
    original_plan_signature = prepared.record_plane_plan.fingerprint
    state.projector_lens_p2.percent += 1.0
    changed = prepare_fourdstem_capture(
        state,
        simulation,
        FourDSTEMRequest(
            tmp_path / "changed.npy",
            frame_count=2,
            overwrite=True,
        ),
        maximum_step_mm=5.0,
    )
    assert changed.cube_dependency_signature == original_cube_signature
    assert changed.record_plane_plan.fingerprint != original_plan_signature
    changed.sink.close_partial()
