"""Small real CPU transport: retained gun -> screen/camera -> scan preview."""
import numpy as np

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.component_keys import CAMERA, FLUORESCENT_SCREEN
from temsim.cpu_resources import numerical_job
from temsim.detector.particle_readout import measure_particle_detectors
from temsim.detector.receiver_image import receiver_image
from temsim.detector.receiver_scan import receiver_scan_preview
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.simulation_pipeline import calculate


def test_cpu_receiver_reuses_real_gun_obeys_screen_and_captures_scan():
    state = default_state()
    AssemblyCatalog().apply(state, AssemblySelection("FEG", "C2", "No Energy Filter"))
    state.electron_gun.emitter.ray_count = 9
    state.step_mm = state.history_step_mm = 1.
    state.acceleration_backend = "Numba CPU"
    state.acceleration_enabled = True
    state.vacuum_map.enabled = False
    state.sample.specimen_mode = "vacuum"
    state.sample.inserted = False
    state.sample.eds_enabled = False
    state.sample.wave_enabled = state.sample.stem_wave_enabled = False
    state.sample.stem_image_enabled = False
    state.ac_deflector.enabled = state.ac_deflector.scan_enabled = True
    state.ac_deflector.wobble_enabled = False
    state.descan_deflector.scan_enabled = False
    for drive in (state.ac_deflector, state.descan_deflector):
        drive.scan_pixels_x = drive.scan_lines = 4
    for plane in state.recording_planes:
        plane.inserted = plane.key in {CAMERA, FLUORESCENT_SCREEN}
    with numerical_job(1):
        rays = calculate(state, workflow="rays")
        received = calculate(capture_instrument_snapshot(rays.state_snapshot).restore(),
                             workflow="receiver", existing_result=rays)
        rows = {row.key: row for row in measure_particle_detectors(received)}
        for key in (CAMERA, FLUORESCENT_SCREEN):
            image = receiver_image(received, key, pixels=64)
            assert image.status == "AVAILABLE"
            np.testing.assert_allclose(image.ideal_probability.sum(), rows[key].fraction, atol=1e-12)
        preview = receiver_scan_preview(received, FLUORESCENT_SCREEN, pixels=64)
        assert preview.status == "GEOMETRIC_PREVIEW", preview.detail
        assert preview.times_s.shape == (4, 4)
        assert preview.image.expected_electrons is None
        assert received.stem_scan is received.wave_imaging is None
        camera_state = capture_instrument_snapshot(received.state_snapshot).restore()
        next(p for p in camera_state.recording_planes if p.key == FLUORESCENT_SCREEN).inserted = False
        camera_result = calculate(camera_state, workflow="receiver", existing_result=received)
        assert receiver_image(camera_result, FLUORESCENT_SCREEN, pixels=64).status == "NOT_INSERTED"
        image = receiver_image(camera_result, CAMERA, pixels=64)
        assert image.status == "AVAILABLE"
        camera_row = next(row for row in measure_particle_detectors(camera_result) if row.key == CAMERA)
        np.testing.assert_allclose(image.ideal_probability.sum(), camera_row.fraction, atol=1e-12)
