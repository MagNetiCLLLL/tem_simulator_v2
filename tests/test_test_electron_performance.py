"""Exact field/interception equivalence for virtual-electron hot-path reductions."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_test_electron_scene import fixture_scene

from temsim.optics import lens_focal_length as focal
from temsim.physics.closed_gun_field import ClosedGunField
from temsim.physics.planar_gun_field import PlanarGunField
from temsim.test_electron_scene import _Bore, _bore_intercept


@pytest.fixture
def nonuniform_field():
    r = np.array([0., 2e-9, 7e-7, 2e-4, 1e-3])
    z = np.array([0., 3e-9, 1e-6, 3e-4, .02])
    values = np.random.default_rng(193).uniform(-7e4, 3e5, (len(r), len(z)))
    return PlanarGunField({}, r, z, values)


def test_single_point_interpolation_matches_retained_batch_for_grid_faces_and_random_points(nonuniform_field):
    field = nonuniform_field
    rng = np.random.default_rng(501)
    radius = rng.uniform(0, field.r[-1], 300)
    angles = rng.uniform(0, 2*np.pi, len(radius))
    random_points = np.column_stack((radius*np.cos(angles), radius*np.sin(angles),
                                     rng.uniform(field.z[0], field.z[-1], len(radius))))
    faces = np.array([(r, 0., z) for r in field.r for z in field.z])
    points = np.concatenate((faces, random_points))
    potentials, fields = field.interpolate(points)
    for index, point in enumerate(points):
        for shape in ((3,), (1, 3), (1, 1, 3)):
            potential, electric = field.interpolate(point.reshape(shape))
            assert potential.shape == shape[:-1] and electric.shape == shape
            np.testing.assert_array_equal(potential.reshape(-1), potentials[index:index+1])
            np.testing.assert_array_equal(electric.reshape(1, 3), fields[index:index+1])


def test_single_point_bilinear_potential_and_exact_gradient():
    r, z = np.array([0., .001, .003]), np.array([0., .002, .02])
    ss, zz = np.meshgrid(r*r, z, indexing='ij')
    # Bilinear in r²,z, so this potential and its gradient are represented exactly.
    values = 2e5 + 3e7*ss - 4e6*zz + 6e8*ss*zz
    field = PlanarGunField({}, r, z, values)
    for point in (np.array([0., 0., 0.]), np.array([.0004, .0003, .004]), np.array([.003, 0., .02])):
        potential, electric = field.interpolate(point)
        x, y, height = point
        square = x*x+y*y
        assert float(potential) == pytest.approx(2e5+3e7*square-4e6*height+6e8*square*height, abs=1e-10)
        np.testing.assert_allclose(electric, [-2*x*(3e7+6e8*height), -2*y*(3e7+6e8*height),
                                            4e6-6e8*square], rtol=1e-13, atol=1e-8)


@pytest.mark.parametrize('point', [(1.001e-3, 0., 0.), (0., 0., -.001), (0., 0., .02001),
                                   (float('nan'), 0., 0.), (0., float('inf'), 0.)])
def test_single_point_retains_no_extrapolation_and_finite_checks(nonuniform_field, point):
    with pytest.raises(ValueError):
        nonuniform_field.interpolate(np.array(point))
    with pytest.raises(ValueError):
        nonuniform_field.interpolate(np.array([point, point]))


def test_closed_cathode_extension_still_matches_batch_and_does_not_extend_radially(nonuniform_field):
    reference = nonuniform_field
    field = ClosedGunField({}, reference.r, reference.z, reference.voltage)
    points = np.array([[0., 0., -1e-8], [.0002, .0001, -1e-5], [0., 0., 0.], [.0001, 0., .005]])
    potential, electric = field.interpolate(points)
    for index, point in enumerate(points):
        one_phi, one_e = field.interpolate(point[None, :])
        np.testing.assert_array_equal(one_phi, potential[index:index+1])
        np.testing.assert_array_equal(one_e, electric[index:index+1])
    np.testing.assert_array_equal(potential[:2], 0.)
    np.testing.assert_array_equal(electric[:2], 0.)
    with pytest.raises(ValueError):
        field.interpolate(np.array([[reference.r[-1]*1.01, 0., -.001]]))


def reference_stop(bores, start, end):
    hits = [(fraction, 'hardware:'+bore.key) for bore in bores
            if (fraction := _bore_intercept(start, end, bore)) is not None]
    return min(hits, key=lambda row: row[0]) if hits else None


def test_axial_broadphase_preserves_chronological_contacts_and_only_checks_overlaps(monkeypatch):
    bores = tuple(_Bore(f'wall-{index}', index*.001, index*.001+.0005, .0002, .0006) for index in range(240))
    bores += (_Bore('zero-thickness', .00575, .00575, .0001, .0003),)
    scene = replace(fixture_scene(monkeypatch), _bores=bores, _apertures=())
    assert not scene._bore_axial_bounds_m.flags.writeable
    random = np.random.default_rng(319)
    segments = list(zip(random.uniform([-.001, -.001, 0], [.001, .001, .02], (100, 3)),
                        random.uniform([-.001, -.001, 0], [.001, .001, .02], (100, 3))))
    segments += [(np.array([.0004, 0., .005]), np.array([.0004, 0., .0055])),
                 (np.array([.0004, 0., .0055]), np.array([.0004, 0., .005])),
                 (np.array([0., 0., .00525]), np.array([.001, 0., .00525])),
                 (np.array([.0002, 0., .0056]), np.array([.0002, 0., .0059]))]
    for start, end in segments:
        assert scene.diagnostic_segment_stop(start, end) == reference_stop(bores, start, end)
    calls = []
    def counted(start, end, bore):
        calls.append(bore.key)
        return _bore_intercept(start, end, bore)
    monkeypatch.setattr('temsim.test_electron_scene._bore_intercept', counted)
    scene.diagnostic_segment_stop((0., 0., .0051), (0., 0., .0052))
    assert calls == ['wall-5']
    empty = replace(scene, _bores=())
    assert empty.diagnostic_segment_stop((0., 0., 0.), (0., 0., .01)) is None


def test_peak_cache_is_exact_and_invalidates_every_consumed_input(monkeypatch):
    lens = SimpleNamespace(a_mm=1.7, normalise_profile_peak=False,
        gaussian=[SimpleNamespace(amplitude=.83, sigma=.62, offset=.2),
                  SimpleNamespace(amplitude=.17, sigma=.93, offset=-1.1)])
    raw = focal._unit_field_samples
    focal._unit_field_peak_cached.cache_clear()
    calls = []
    def counted(*args, **kwargs):
        calls.append(True)
        return raw(*args, **kwargs)
    monkeypatch.setattr(focal, '_unit_field_samples', counted)
    def check(value, samples=4001):
        expected = float(np.max(np.abs(raw(value, samples)[1])))
        assert focal.unit_field_peak(value, samples) == expected
        return expected
    check(lens)
    for _ in range(10):
        check(deepcopy(lens))
    assert len(calls) == 1
    for field, value in (('amplitude', .91), ('sigma', .58), ('offset', -.31)):
        changed = deepcopy(lens)
        setattr(changed.gaussian[0], field, value)
        check(changed)
    check(SimpleNamespace(**{**vars(lens), 'a_mm': 2.1}))
    check(SimpleNamespace(**{**vars(lens), 'normalise_profile_peak': True}))
    check(lens, 2001)
    assert len(calls) == 7
    # Translation/excitation do not enter unit-profile normalization.
    lens.z_mm, lens.percent, lens.b0_t = 42., 28., 3.2
    check(lens)
    assert len(calls) == 7
    assert focal.unit_field_peak(SimpleNamespace(gaussian=[])) == 0.
    assert focal._unit_field_peak_cached.cache_info().maxsize == 128


def test_peak_cache_preserves_numpy_scalar_precision_and_distinguishes_equal_typed_values():
    lenses = []
    for dtype in (np.float32, np.float64, float):
        lenses.append(SimpleNamespace(a_mm=dtype(1.7), normalise_profile_peak=False,
            gaussian=[SimpleNamespace(amplitude=dtype(.83), sigma=dtype(.62), offset=dtype(.2))]))
    for lens in lenses:
        expected = float(np.max(np.abs(focal._unit_field_samples(lens)[1])))
        assert focal.unit_field_peak(lens) == expected
