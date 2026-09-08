"""Gaussian material overlap belongs to the specimen, never to the raster."""
from types import SimpleNamespace
import math

import numpy as np
import pytest
from scipy.integrate import quad

from temsim.specimen.rutherford import finite_sample_gaussian_overlap


def _sample(**changes):
    values = dict(envelope_shape="disk", size_x_nm=10., size_y_nm=10.,
                  centre_x_nm=0., centre_y_nm=0.)
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("shape", ["disk", "rectangle"])
def test_same_probe_centre_is_independent_of_raster_size_spacing_and_order(shape):
    sample = _sample(envelope_shape=shape)
    assert finite_sample_gaussian_overlap(sample, 0., 0., probe_sigma_nm=.1) == 1.
    for count, span_nm in ((1, 0.), (3, .04), (17, .32), (65, 1.28), (65, 8.)):
        axis = np.linspace(-span_nm/2, span_nm/2, count) * 1e-3
        x, y = np.meshgrid(axis, axis)
        values = finite_sample_gaussian_overlap(sample, x, y, probe_sigma_nm=.1)
        assert values[count//2, count//2] == 1.
        np.testing.assert_allclose(values.ravel()[::-1], finite_sample_gaussian_overlap(
            sample, x.ravel()[::-1], y.ravel()[::-1], probe_sigma_nm=.1), atol=0, rtol=0)


def test_round_disk_matches_analytic_center_probability_and_rotational_symmetry():
    sample = _sample()
    sigma = 3.
    expected = 1 - math.exp(-5**2/(2*sigma**2))
    assert finite_sample_gaussian_overlap(sample, 0., 0., probe_sigma_nm=sigma) == pytest.approx(expected)
    phi = np.linspace(0., 2*np.pi, 33)
    values = finite_sample_gaussian_overlap(sample, .005*np.cos(phi), .005*np.sin(phi), probe_sigma_nm=.2)
    np.testing.assert_allclose(values, values[0], rtol=0, atol=1e-13)


@pytest.mark.parametrize("shape", ["disk", "rectangle"])
def test_true_sample_edge_is_smooth_and_far_outside_is_zero(shape):
    sample = _sample(envelope_shape=shape)
    x_nm = np.array([0., 4., 4.8, 5., 5.2, 6., 100.])
    values = finite_sample_gaussian_overlap(sample, x_nm*1e-3, np.zeros_like(x_nm), probe_sigma_nm=.1)
    assert values[0] == 1. and values[-1] == 0.
    assert np.all(np.diff(values) <= 0.)
    assert .49 < values[3] <= .5
    assert 0. < values[4] < .03


@pytest.mark.parametrize("shape,expected", [("disk", [1., 1., 0., 0.]), ("rectangle", [1., 1., 0., 1.])])
def test_zero_sigma_uses_the_real_offcentre_geometry(shape, expected):
    sample = _sample(envelope_shape=shape, centre_x_nm=20., centre_y_nm=-10.)
    x = np.array([20., 25., 25.001, 24.5])*1e-3
    y = np.array([-10., -10., -10., -5.5])*1e-3
    np.testing.assert_array_equal(finite_sample_gaussian_overlap(sample, x, y), expected)
    np.testing.assert_allclose(finite_sample_gaussian_overlap(sample, x, y, probe_sigma_nm=.2),
        finite_sample_gaussian_overlap(_sample(envelope_shape=shape), x-.020, y+.010, probe_sigma_nm=.2), atol=1e-13)


def test_rectangle_matches_independent_gaussian_integrals():
    sample = _sample(envelope_shape="rectangle", size_x_nm=4., size_y_nm=8., centre_x_nm=1., centre_y_nm=-2.)
    sigma = 1.3
    for px, py in ((1., -2.), (2.8, 1.5), (-5., 0.)):
        density = lambda value, mean: math.exp(-.5*((value-mean)/sigma)**2)/(math.sqrt(2*math.pi)*sigma)
        expected = quad(density, -1., 3., args=(px,), epsabs=1e-13)[0] * quad(density, -6., 2., args=(py,), epsabs=1e-13)[0]
        assert finite_sample_gaussian_overlap(sample, px*1e-3, py*1e-3, probe_sigma_nm=sigma) == pytest.approx(expected, abs=1e-12)


def test_unequal_disk_diameters_follow_ellipse_geometry_with_independent_area_integral():
    sample = _sample(size_x_nm=6., size_y_nm=2.)
    sigma, px, py = .8, 2., .3
    def angular_integral(phi):
        # Polar coordinates of the ellipse provide a reference different
        # from the production Cartesian normal-CDF conditional integral.
        def radial(r):
            x, y = 3*r*math.cos(phi), r*math.sin(phi)
            return 3*r*math.exp(-((x-px)**2+(y-py)**2)/(2*sigma**2))/(2*math.pi*sigma**2)
        return quad(radial, 0., 1., epsabs=1e-12)[0]
    expected = quad(angular_integral, 0., 2*math.pi, epsabs=1e-11)[0]
    actual = finite_sample_gaussian_overlap(sample, px*1e-3, py*1e-3, probe_sigma_nm=sigma)
    assert actual == pytest.approx(expected, abs=2e-10)


def test_macroscopic_disk_narrow_probe_boundary_remains_finite():
    # Avoid enormous noncentral chi-square parameters near an actual edge.
    # Curvature is negligible here, so the boundary tends to a half plane.
    sample = _sample(size_x_nm=3e6, size_y_nm=3e6)
    value = finite_sample_gaussian_overlap(sample, 1500., 0., probe_sigma_nm=.1)
    assert value == pytest.approx(.5, abs=1e-6)


@pytest.mark.parametrize("x,y", [(np.zeros(2), np.zeros(3)), ([], []), ([float("nan")], [0.]), ([0.], [float("inf")])])
def test_invalid_coordinate_shape_or_values_are_rejected(x, y):
    with pytest.raises(ValueError, match="matching non-empty finite"):
        finite_sample_gaussian_overlap(_sample(), x, y, probe_sigma_nm=.1)


@pytest.mark.parametrize("changes,sigma,match", [
    ({"size_x_nm": 0.}, .1, "sizes"), ({"size_y_nm": float("inf")}, .1, "sizes"),
    ({"centre_y_nm": float("nan")}, .1, "centre"), ({}, -1., "sigma"),
    ({}, float("nan"), "sigma"), ({}, float("inf"), "sigma"),
    ({"envelope_shape": "unknown"}, .1, "envelope shape")])
def test_invalid_geometry_or_sigma_is_rejected(changes, sigma, match):
    with pytest.raises(ValueError, match=match):
        finite_sample_gaussian_overlap(_sample(**changes), 0., 0., probe_sigma_nm=sigma)


def test_real_tail_uses_specimen_overlap_for_single_point_and_different_rasters(monkeypatch):
    from temsim.detector import stem_signal
    from temsim.physics.first_order import TransverseTransfer
    from temsim.physics.record_plane import PlaneStop, RecordPlanePlan
    sample = _sample(specimen_mode="atomic", real_high_angle_tail_enabled=True,
        real_tail_material_source="manual", real_tail_screening_source="manual",
        real_tail_atomic_number=14, real_tail_areal_density_atoms_nm2=250.,
        real_tail_screening_angle_mrad=5., real_tail_max_angle_mrad=100.)
    state = SimpleNamespace(sample=sample, beam_voltage_kv=200.)
    simulation = SimpleNamespace(incident=SimpleNamespace(alive=np.ones(1, dtype=bool),
        ray_weight=np.ones(1), x=np.zeros((1,1)), y=np.zeros((1,1)), tx=np.zeros((1,1)), ty=np.zeros((1,1))))
    monkeypatch.setattr("temsim.physics.wave_imaging.effective_sample_thickness_nm", lambda _: 5.)
    monkeypatch.setattr(stem_signal, "probe_state_from_simulation", lambda *_: SimpleNamespace(probe_sigma_nm=.1))
    monkeypatch.setattr(stem_signal, "measure_sample_current", lambda *_: SimpleNamespace(fraction=1.))
    plane = PlaneStop("haadf", "Test detector", 1., "detector", "disk", outer_width_mm=4., readout_enabled=True)
    transfer = TransverseTransfer(0., 1., np.eye(2), np.eye(2)*.001, np.zeros((2,2)), np.eye(2))
    plan = RecordPlanePlan(0., (plane,), (transfer,), "1"*64, "2"*64)
    reference = None
    for count, span_nm in ((1, 0.), (3, .04), (17, .32)):
        axis = np.linspace(-span_nm/2, span_nm/2, count)*1e-3
        x, y = np.meshgrid(axis, axis)
        images, lost, probability, metrics = stem_signal._real_high_angle_tail(
            simulation, state, [SimpleNamespace(key="haadf")], x, y, None, 50., plan)
        value = images["haadf"][count//2, count//2]
        reference = value if reference is None else reference
        assert value == pytest.approx(reference, abs=1e-15)
        np.testing.assert_allclose(images["haadf"] + lost, probability, atol=1e-15)
        assert metrics["minimum_sample_overlap"] == metrics["maximum_sample_overlap"] == 1.
        assert metrics["probe_sigma_nm"] == .1
