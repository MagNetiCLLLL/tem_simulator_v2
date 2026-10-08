"""Finite oblique bore geometry: analytic limits and shared ray intersections."""
from dataclasses import asdict, replace
from types import SimpleNamespace
import json

import numpy as np
import pytest

from temsim.physics.column_wall import _posed_bore_stop_fractions
from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.posed_wave_hardware import PosedWaveBore, placed_wave_bores


def _bore(tilt=.025, *, radius=1., length=1.):
    c, s = np.cos(tilt), np.sin(tilt)
    registration = CoordinateRegistration((.0002, -.0001, .01),
        ((c, 0., s), (0., 1., 0.), (-s, 0., c)))
    return PosedWaveBore("test_bore", -.5*length, .5*length, radius, registration)


def _ray_passes(bore, x, y, first, last):
    segment = SimpleNamespace(start_z_mm=bore.local_start_z_mm,
        end_z_mm=bore.local_end_z_mm, inner_diameter_mm=2*bore.radius_mm)
    starts = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, first)))
    ends = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, last)))
    return ~np.isfinite(_posed_bore_stop_fractions(starts, ends, segment,
                                                 bore.registration)).reshape(x.shape)


def test_zero_tilt_is_translated_circle_with_strict_wall_and_axial_shoulders():
    bore = _bore(0.)
    x = np.array([.2, 1.19, 1.2, 1.21])
    y = np.full(x.shape, -.1)
    np.testing.assert_array_equal(bore.transmission_mask(x, y, 10.), [True, True, False, False])
    assert np.all(bore.transmission_mask(x, y, 9.49))
    assert np.all(bore.transmission_mask(x, y, 10.51))
    np.testing.assert_array_equal(bore.transmission_mask(x, y, 9.5), [True, True, False, False])
    np.testing.assert_array_equal(bore.transmission_mask(x, y, 10.5), [True, True, False, False])


@pytest.mark.parametrize("tilt", [-.03, 0., .025])
@pytest.mark.parametrize("first,last", [(8., 12.), (9.9, 10.1), (12., 8.), (8., 8.5)])
def test_swept_mask_matches_exact_ray_slab_and_wall_intersection(tilt, first, last):
    bore = _bore(tilt)
    # Irrational-ish offsets deliberately avoid floating-point tangencies;
    # exact contact is covered separately by the analytic zero-angle test.
    x, y = np.meshgrid(np.linspace(-1.41, 1.53, 43), np.linspace(-1.33, 1.31, 39))
    actual = bore.transmission_mask(x, y, last, previous_z_mm=first)
    np.testing.assert_array_equal(actual, _ray_passes(bore, x, y, first, last))


def test_swept_mask_catches_tilted_thin_shoulder_between_clear_endpoints():
    bore = _bore(.025, length=.01)
    x, y = np.array([1.7]), np.array([-.1])
    assert bore.transmission_mask(x, y, 9.).item()
    assert bore.transmission_mask(x, y, 11.).item()
    assert not bore.transmission_mask(x, y, 11., previous_z_mm=9.).item()
    assert not _ray_passes(bore, x, y, 9., 11.).item()


def test_swept_absorption_is_invariant_under_subdivision_for_vertical_flow():
    bore = _bore(.07, radius=.2, length=.015)
    x, y = np.meshgrid(np.linspace(-.4, .8, 51), np.linspace(-.5, .3, 49))
    direct = bore.swept_transmission_mask(x, y, 9.5, 10.5)
    for count in (3, 16, 127):
        stepped = np.ones(x.shape, dtype=bool)
        axis = np.linspace(9.5, 10.5, count+1)
        for first, last in zip(axis[:-1], axis[1:]):
            stepped &= bore.swept_transmission_mask(x, y, first, last)
        np.testing.assert_array_equal(stepped, direct)


def test_interval_clearance_proof_never_skips_oblique_shoulder():
    bore = _bore(.025, length=.01)
    shape, basis = (9, 11), np.diag([1e-5, 1e-5])
    origin = np.array([.0017, -.0001])
    assert bore.clears_grid(shape, basis, origin, 9.)
    assert bore.clears_grid(shape, basis, origin, 11.)
    assert not bore.clears_grid(shape, basis, origin, 11., previous_z_mm=9.)
    assert bore.clears_grid(shape, basis, np.array([.0002, -.0001]), 11., previous_z_mm=9.)


def test_clearance_proof_covers_skew_grid_samples_and_intermediate_z():
    bore = _bore(.03, radius=.3)
    shape, basis, origin = (13, 19), np.array([[2e-5, 4e-6], [-3e-6, 1e-5]]), np.array([.0002, -.0001])
    assert bore.clears_grid(shape, basis, origin, 10.6, previous_z_mm=9.4)
    ny, nx = shape
    u, v = np.meshgrid(np.arange(nx)-nx//2, np.arange(ny)-ny//2)
    x = (origin[0]+basis[0, 0]*u+basis[0, 1]*v)*1e3
    y = (origin[1]+basis[1, 0]*u+basis[1, 1]*v)*1e3
    for z in np.linspace(9.4, 10.6, 41):
        assert np.all(bore.transmission_mask(x, y, z))


def test_linear_grouping_rejects_internal_contact_with_equal_clear_endpoints():
    """A turned trajectory can hit a wall and return between plan nodes.

    The example has x(s)=x0+0.012*s-12*s**2 with s in metres. Across
    1 mm its endpoints coincide but its midpoint is 3 um further out.
    Endpoint-grid tests alone therefore incorrectly admit grouping.
    """
    from temsim.physics.column_wave import _ColumnPath, _linear_factor, _linear_run
    bore = _bore(.0001, length=2.)
    shape = (16, 16)
    wave = SimpleNamespace(amplitude=np.ones(shape, dtype=complex),
        basis_m=np.eye(2)*5e-8, origin_m=np.array([1.198e-3, -.1e-3]),
        tilt_rad=np.array([.012, 0.]), curvature_m1=np.zeros((2, 2)))
    generator = np.zeros((4, 4))
    generator[0, 2] = generator[1, 3] = 1.
    force = np.array([0., 0., -24., 0.])
    path = _ColumnPath(1e-3, generator, force, .001)
    _linear_factor(path, generator, .001, force=force)
    output_origin = (path.matrix[:2, :2]@wave.origin_m
                     + path.matrix[:2, 2:]@wave.tilt_rad + path.offset[:2])
    np.testing.assert_allclose(output_origin, wave.origin_m, rtol=0., atol=1e-18)
    assert bore.clears_grid(shape, wave.basis_m, wave.origin_m, 10.5, previous_z_mm=9.5)
    assert bore.clears_grid(shape, wave.basis_m, output_origin, 10.5, previous_z_mm=9.5)
    assert not bore.transmission_mask(np.array([1.201]), np.array([-.1]), 10.).item()
    plan = SimpleNamespace(z_mm=np.array([9.5, 10.5]), step_m=np.array([.001]),
        midpoint_hex_normal_m3=np.zeros(1), midpoint_hex_skew_m3=np.zeros(1),
        cs_kick_m3=np.zeros(2))
    def run(stops):
        return _linear_run(wave, 2e-12, [path], 0, plan, np.full(2, np.inf),
                           stops, np.zeros(2), np.zeros(2))
    assert run({})[0] == 1  # All unrelated phase/grid gates admit this step.
    assert run({1: (bore,)}) == (0, None)


def test_axial_bounds_depend_on_grid_and_contain_every_local_face_crossing():
    bore = _bore(.04, length=.01)
    basis = np.diag([.0001, .0002])
    origin = np.array([.002, -.001])
    low, high = bore.grid_axial_bounds_mm((13, 17), basis, origin)
    r, t = bore.registration.rotation_array, bore.registration.origin_array_m*1e3
    for x in (1.2, 2., 2.8):
        for y in (-2.2, -1., .2):
            for local_z in (bore.local_start_z_mm, bore.local_end_z_mm):
                z = t[2]+(local_z-(x-t[0])*r[0, 2]-(y-t[1])*r[1, 2])/r[2, 2]
                assert low <= z <= high
    events = bore.event_z_mm()
    assert len(events) == 6 and tuple(sorted(events)) == events
    # Faces extend beyond the finite opening; its events are not sufficient
    # bounds for a larger, off-axis grid.
    assert low < min(events)


def test_geometric_step_guide_scales_with_pixel_size_and_tilt():
    basis = np.diag([1e-7, 2e-7])
    assert np.isinf(_bore(0.).maximum_step_mm(basis))
    guide = _bore(.025).maximum_step_mm(basis)
    assert guide == pytest.approx(.5e-4/np.tan(.025), rel=1e-14)
    assert _bore(.025).maximum_step_mm(2*basis) == pytest.approx(2*guide)


def test_geometry_identity_serializes_pose_and_changes_without_excitation():
    bore = _bore()
    assert json.loads(json.dumps(asdict(bore)))["key"] == "test_bore"
    assert bore.identity != replace(bore, radius_mm=.5).identity
    assert bore.identity != _bore(-.025).identity


def test_factory_recovers_independent_stationary_tube_and_unexcited_tilted_bores():
    from test_column_wave_transport import _unexcited_posed_column
    state = _unexcited_posed_column(rotation_y_mrad=1.)
    stationary, placed = placed_wave_bores(state, 1600., 1625.)
    assert stationary and placed
    assert any(bore.key == "objective_lens" for bore in placed)
    assert all(not lens.enabled for lens in state.lenses)
    # A finite global-Z bound cannot cull a transversely unbounded tilted
    # shoulder before its finite wave grid is known.
    assert placed_wave_bores(state, 2000., 2001.)[1] == placed


def test_real_cuda_bore_mask_matches_ray_contacts_and_cpu():
    cp = pytest.importorskip("cupy", reason="optional real CUDA verification requires CuPy")
    try:
        available = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA device unavailable")
    if not available:
        pytest.skip("CUDA device unavailable")
    bore = _bore(.027, radius=.5, length=.01)
    x, y = np.meshgrid(np.linspace(-.71, .91, 67), np.linspace(-.83, .79, 63))
    expected = _ray_passes(bore, x, y, 9., 11.)
    gpu = bore.transmission_mask(cp.asarray(x), cp.asarray(y), 11., previous_z_mm=9., xp=cp)
    np.testing.assert_array_equal(cp.asnumpy(gpu), expected)
    np.testing.assert_array_equal(cp.asnumpy(gpu), bore.transmission_mask(x, y, 11., previous_z_mm=9.))
