"""Finite tilted plate masks and ray contacts use one physical geometry."""
from types import SimpleNamespace
from dataclasses import asdict

import numpy as np
import pytest

from temsim.lens_pose import rotation_matrix_mrad
from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.posed_wave_aperture import PosedWaveAperture


def _plate(angle=100., thickness=.4, radius=.03, hole_x=0.):
    rotation = rotation_matrix_mrad((0., angle, 0.))
    centre = np.array((0., 0., 10.))
    shift_mm = centre - rotation @ centre + np.array((.12, -.03, .2))
    registration = CoordinateRegistration(tuple(shift_mm*1e-3), tuple(map(tuple, rotation)))
    aperture = SimpleNamespace(key="objective_aperture", z_mm=10., radius_mm=radius,
        plate_thickness_mm=thickness, offset_x_mm=hole_x, offset_y_mm=0.)
    return PosedWaveAperture.from_component(SimpleNamespace(), aperture, registration)


def test_requires_real_declared_thickness_not_schematic_extent():
    aperture = SimpleNamespace(key="projection_chamber_dpa_aperture", z_mm=10., radius_mm=.1,
                               length_mm=2., mechanical_display_thickness_mm=.3)
    with pytest.raises(ValueError, match="declared positive plate_thickness_mm"):
        PosedWaveAperture.from_component(SimpleNamespace(), aperture, CoordinateRegistration())


def test_captured_manifest_thickness_and_hole_offset_are_applied_once():
    aperture = SimpleNamespace(key="objective_aperture", z_mm=10., radius_mm=.1,
                               offset_x_mm=.2, offset_y_mm=.3)
    part = SimpleNamespace(key=aperture.key, data={"plate_thickness_mm": .2})
    state = SimpleNamespace(_resolved_assembly=SimpleNamespace(parts=[part]))
    plate = PosedWaveAperture.from_component(state, aperture, CoordinateRegistration((.001, 0., 0.)))
    assert plate.local_start_z_mm == pytest.approx(9.9)
    assert plate.transmission_mask(np.array([1.2, .2]), np.array([.3, .3]), 10.).tolist() == [True, False]
    assert asdict(plate)["key"] == aperture.key


def test_zero_angle_finite_plate_matches_circular_opening():
    plate = _plate(angle=0., radius=.05)
    x, y = np.meshgrid(np.linspace(.03, .21, 17), np.linspace(-.12, .06, 17))
    expected = np.hypot(x-.12, y+.03) <= .05
    np.testing.assert_array_equal(plate.transmission_mask(x, y, 10.2), expected)
    np.testing.assert_array_equal(plate.transmission_mask(x, y, 11., previous_z_mm=9.), expected)
    assert np.all(plate.transmission_mask(x, y, 11.))


def test_sweep_catches_oblique_shoulder_despite_both_endpoints_outside_plate():
    plate = _plate()
    x, y = np.array([.12, .16]), np.array([-.03, -.03])
    assert np.all(plate.transmission_mask(x, y, 9.))
    assert np.all(plate.transmission_mask(x, y, 11.))
    mask = plate.transmission_mask(x, y, 11., previous_z_mm=9.)
    assert mask.tolist() == [True, False]
    start = np.column_stack((x, y, np.full(2, 9.)))
    end = np.column_stack((x, y, np.full(2, 11.)))
    contact = plate.first_contact_fraction(start, end)
    assert np.isinf(contact[0]) and 0. < contact[1] < 1.
    hit = start[1] + contact[1]*(end[1]-start[1])
    local_hit = (hit-plate.registration.origin_array_m*1e3) @ plate.registration.rotation_array
    assert local_hit[2] == pytest.approx(plate.local_start_z_mm)
    assert np.hypot(*local_hit[:2]) > plate.radius_mm


def test_ray_cylinder_exit_inside_plate_has_analytic_contact():
    plate = _plate()
    # Start on hole axis, then cross the cylindrical wall before the plate exit.
    local_start = np.array([[0., 0., 10.]])
    local_end = np.array([[.06, 0., 10.1]])
    rotation, origin = plate.registration.rotation_array, plate.registration.origin_array_m*1e3
    start, end = local_start @ rotation.T+origin, local_end @ rotation.T+origin
    assert plate.first_contact_fraction(start, end)[0] == pytest.approx(.5)


def test_swept_mask_is_partition_invariant_and_matches_independent_ray_chords():
    plate = _plate(angle=170., hole_x=.011)
    x, y = np.meshgrid(np.linspace(.03, .21, 51), np.linspace(-.12, .06, 39))
    expected = plate.transmission_mask(x, y, 11., previous_z_mm=9.)
    for count in (4, 37):
        accumulated = np.ones_like(expected)
        steps = np.linspace(9., 11., count+1)
        for low, high in zip(steps[:-1], steps[1:]):
            accumulated &= plate.transmission_mask(x, y, high, previous_z_mm=low)
        np.testing.assert_array_equal(accumulated, expected)
    start = np.stack((x, y, np.full_like(x, 9.)), axis=-1)
    end = np.stack((x, y, np.full_like(x, 11.)), axis=-1)
    np.testing.assert_array_equal(np.isinf(plate.first_contact_fraction(start, end)), expected)
    np.testing.assert_array_equal(plate.transmission_mask(x, y, 9., previous_z_mm=11.), expected)


def test_grid_clearance_never_proves_a_hidden_shoulder_clear():
    plate = _plate()
    assert not plate.clears_grid((8, 8), np.eye(2)*1e-5, (.00016, -.00003), 11., previous_z_mm=9.)
    # Clearance proof is deliberately conservative for a long prism: it
    # need not prove the full 9..11 interval clear outside the local slab.
    assert plate.clears_grid((8, 8), np.eye(2)*1e-7, (.00012, -.00003), 10.2, previous_z_mm=10.1)
    assert len(plate.event_z_mm()) == 6


def test_zero_radius_blocks_even_exact_axis_in_the_plate():
    plate = _plate(angle=0., radius=0.)
    assert not plate.transmission_mask(np.array([.12]), np.array([-.03]), 10.2)[0]


def test_gpu_uses_same_swept_geometry_and_contacts():
    cupy = pytest.importorskip("cupy")
    try:
        devices = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    if devices < 1:
        pytest.skip("No CUDA device")
    plate = _plate(angle=170., hole_x=.011)
    x, y = np.meshgrid(np.linspace(.03, .21, 51), np.linspace(-.12, .06, 39))
    expected = plate.transmission_mask(x, y, 11., previous_z_mm=9.)
    actual = plate.transmission_mask(cupy.asarray(x), cupy.asarray(y), 11., previous_z_mm=9., xp=cupy)
    np.testing.assert_array_equal(cupy.asnumpy(actual), expected)
    start = np.stack((x, y, np.full_like(x, 9.)), axis=-1)
    end = np.stack((x, y, np.full_like(x, 11.)), axis=-1)
    np.testing.assert_allclose(cupy.asnumpy(plate.first_contact_fraction(cupy.asarray(start), cupy.asarray(end))),
                               plate.first_contact_fraction(start, end), atol=1e-14)
