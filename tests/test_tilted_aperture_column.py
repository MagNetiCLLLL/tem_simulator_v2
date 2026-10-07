"""Installed finite aperture plates: common ray contact and CPU/GPU wave loss.

These short field-free segments isolate actual aperture geometry, without
pretending a swept global-Z mask certifies oblique-boundary diffraction.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.physics.aperture_clipping import clip_segment, posed_aperture_registration
from temsim.physics.column_wave import _prepare_column, _propagate_column
from temsim.physics.core import execute_propagation_plan
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.posed_wave_aperture import PosedWaveAperture
from temsim.physics.wave_grid import WaveGridNumerics
from test_wave_detector_readout import checkpoint


def aperture_fixture(key="condenser_aperture_2", *, radius_mm=None):
    source = default_state()
    aperture = next(item for item in source.apertures if item.key == key)
    aperture.enabled = True
    if radius_mm is not None:
        aperture.radius_mm = radius_mm
    assembly = replace(source._resolved_assembly, parts=tuple(
        replace(part, data={**part.data, "rotation_y_mrad": 1.})
        if part.key == key else part for part in source._resolved_assembly.parts))
    state = SimpleNamespace(lenses=[], condenser_system=source.condenser_system,
        _resolved_assembly=assembly, stigmators=[], corrector_elements=[],
        deflectors=[], apertures=[aperture], recording_planes=[], simulation_mode="ideal",
        beam_voltage_kv=300., step_mm=.01, history_step_mm=.01,
        acceleration_enabled=False, acceleration_backend="CPU", projector_mode="diffraction",
        equivalent_image_lenses_enabled=False, sample=source.sample)
    plate = PosedWaveAperture.from_component(state, aperture,
                                           posed_aperture_registration(state, aperture))
    return state, aperture, plate


def aperture_checkpoint(start, *, origin=(0., 0.), uniform=False, pitch=5e-9, size=64, sigma=25e-9):
    axis = (np.arange(size)-size//2)*pitch
    xx, yy = np.meshgrid(axis, axis)
    amplitude = (np.ones((size, size), dtype=complex) if uniform else
                 np.exp(-(xx*xx+yy*yy)/(2*sigma**2)).astype(complex))
    amplitude /= np.linalg.norm(amplitude)
    plane = PlaneWave(amplitude, np.eye(2)*pitch, np.array(origin),
                      np.zeros((2, 2)), np.zeros(2))
    initial = checkpoint()
    return replace(initial, plane_z_mm=start,
        beam=replace(initial.beam, modes=(replace(initial.beam.modes[0], plane=plane),)))


def propagate_aperture(state, initial, stop, step=.01, backend="CPU"):
    state.step_mm = step
    prepared = _prepare_column(state, initial.plane_z_mm, stop, step)
    result = _propagate_column(state, initial, stop, _prepared=prepared,
        grid_numerics=WaveGridNumerics(compute_backend=backend,
            maximum_pixels=1024, maximum_working_bytes=512*1024**2,
            maximum_device_working_bytes=512*1024**2))
    assert result.record["modes"][0]["compute_backend"] == (
        "cupy" if backend == "Require GPU" else "numpy")
    return result, prepared


@pytest.mark.parametrize("key", ["condenser_aperture_2", "objective_aperture"])
def test_installed_tilted_plate_wave_loss_matches_ray_first_contact(key):
    state, aperture, plate = aperture_fixture(key)
    start, stop = aperture.z_mm-.22, aperture.z_mm+.22
    initial = aperture_checkpoint(start)
    blocked = replace(initial.beam.modes[0], mode_id="outside-hole", weight_per_reference_electron=.2,
        plane=replace(initial.beam.modes[0].plane, origin_m=np.array((1e-4, 0.))))
    initial = replace(initial, beam=replace(initial.beam, modes=(*initial.beam.modes, blocked)))
    result, prepared = propagate_aperture(state, initial, stop)
    plan = prepared[0]
    ray = execute_propagation_plan(state, plan, np.array((0., 1e-4)), np.zeros(2),
                                  np.zeros(2), np.zeros(2))
    alive, blocked_z, keys = clip_segment(state, ray[0], ray[1], ray[3])
    assert alive.tolist() == [True, False]
    assert keys == ["", key]
    # Independent plane equation at x=0.1 mm; contact is at the plate front,
    # not its nominal zero-thickness centre.
    rotation, offset = plate.registration.rotation_array, plate.registration.origin_array_m*1e3
    expected_z = offset[2]+(plate.local_start_z_mm-(.1-offset[0])*rotation[0, 2]
                            +offset[1]*rotation[1, 2])/rotation[2, 2]
    assert blocked_z[1] == pytest.approx(expected_z, abs=2e-10)
    assert abs(blocked_z[1]-aperture.z_mm) > .09
    assert result.beam.modes[0].weight_per_reference_electron == pytest.approx(.6, abs=1e-12)
    assert result.beam.modes[1].weight_per_reference_electron == 0.
    losses = result.record["modes"][1]["losses"]
    assert sum(row["lost_weight"] for row in losses if row["component"] == key) == pytest.approx(.2)
    assert initial.beam.total_weight == pytest.approx(.8)


def test_closed_tilted_plate_only_absorbs_local_slab_on_actual_gpu():
    from test_wave_device import _cuda
    _cuda()
    state, aperture, plate = aperture_fixture(radius_mm=0.)
    rotation, offset = plate.registration.rotation_array, plate.registration.origin_array_m*1e3
    face_z = offset[2]+(plate.local_start_z_mm+offset[0]*rotation[0, 2]
                       +offset[1]*rotation[1, 2])/rotation[2, 2]
    start = face_z-3e-5
    initial = aperture_checkpoint(start, origin=(2.5e-6, 0.), uniform=True, pitch=5e-6, size=8)
    result, _ = propagate_aperture(state, initial, face_z, step=.01, backend="Require GPU")
    xy = initial.beam.modes[0].plane.coordinates_m()
    # At this oblique front face exactly half the grid has entered material.
    expected = xy[0] < 0.
    np.testing.assert_array_equal(expected.sum(axis=1), np.full(8, 4))
    actual = result.beam.modes[0]
    assert actual.weight_per_reference_electron == pytest.approx(.3, abs=1e-12)
    np.testing.assert_allclose(abs(actual.plane.amplitude)**2*.3,
                               expected*abs(initial.beam.modes[0].plane.amplitude)**2*.6, atol=2e-14)
    x = np.tile(xy[0].ravel(), (2, 1))
    y = np.tile(xy[1].ravel(), (2, 1))
    alive, blocked_z, keys = clip_segment(state, np.array((start, face_z)), x, y)
    np.testing.assert_array_equal(alive, expected.ravel())
    assert all(key == aperture.key for key, remains in zip(keys, alive) if not remains)
    assert np.all(blocked_z[~alive] <= face_z)


def test_tilted_c2_finite_shoulder_diffraction_cpu_gpu_complex_parity():
    from test_wave_device import _cuda
    _cuda()
    state, aperture, plate = aperture_fixture()
    # A narrow beam near the downstream left rim: as Z increases, the
    # inclined finite wall crosses the beam. Most absorption occurs inside
    # the plate rather than at its nominal centre.
    theta = .001
    half_thickness = .5*(plate.local_end_z_mm-plate.local_start_z_mm)
    origin = ((-aperture.radius_mm*np.cos(theta)+half_thickness*np.sin(theta))*1e-3, 0.)
    start, stop = aperture.z_mm-.22, aperture.z_mm+.12
    initial = aperture_checkpoint(start, origin=origin)
    outputs = []
    for backend in ("CPU", "Require GPU"):
        output, _ = propagate_aperture(state, initial, stop, .01, backend)
        # Neither a clear beam nor complete absorption; retain absolute loss.
        assert .2 < output.beam.total_weight < .4
        losses = [row for row in output.record["modes"][0]["losses"]
                  if row["component"] == aperture.key and row["lost_weight"] > 0.]
        assert len(losses) >= 3
        assert sum(row["lost_weight"] for row in losses) == pytest.approx(
            initial.beam.total_weight-output.beam.total_weight, abs=2e-12)
        outputs.append(output.beam.modes[0])
    cpu, gpu = outputs
    assert gpu.weight_per_reference_electron == pytest.approx(cpu.weight_per_reference_electron, abs=2e-12)
    for name in ("basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        # Curvature is measured in inverse metres; roundoff in its zero
        # off-diagonal is not a 1e-15 metre displacement tolerance.
        np.testing.assert_allclose(getattr(cpu.plane, name), getattr(gpu.plane, name),
                                   rtol=2e-10, atol=1e-12 if name == "curvature_m1" else 1e-15)
    # Compare the complete factored representation: envelope and carrier
    # separately. This makes no claim that a hard-edge envelope plus carrier
    # can be materialized on this lattice without further refinement.
    np.testing.assert_allclose(cpu.plane.amplitude*np.sqrt(cpu.weight_per_reference_electron),
        gpu.plane.amplitude*np.sqrt(gpu.weight_per_reference_electron), rtol=2e-9, atol=2e-13)
    assert initial.beam.total_weight == .6


def test_tilted_c2_broad_beam_total_transmission_stabilizes_on_gpu():
    """Bounded axial check of total probability, not hard-edge image qualification."""
    from test_wave_device import _cuda
    _cuda()
    state, aperture, plate = aperture_fixture()
    half_thickness = .5*(plate.local_end_z_mm-plate.local_start_z_mm)
    origin = ((-aperture.radius_mm*np.cos(.001)+half_thickness*np.sin(.001))*1e-3, 0.)
    initial = aperture_checkpoint(aperture.z_mm-.22, origin=origin,
                                  pitch=20e-9, size=128, sigma=125e-9)
    weights = []
    for step in (.02, .01, .005):
        output, _ = propagate_aperture(state, initial, aperture.z_mm+.12, step, "Require GPU")
        assert .2 < output.beam.total_weight < .4
        weights.append(output.beam.total_weight)
    coarse_change, fine_change = abs(np.diff(weights))
    assert fine_change < .2*coarse_change
    assert fine_change < 2e-5  # Absolute electron probability; no fitted scale.
