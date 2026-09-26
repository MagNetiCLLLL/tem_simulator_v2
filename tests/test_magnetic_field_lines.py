"""Field-line display tests use analytic fields, not electron transport."""

from dataclasses import dataclass

import numpy as np
import pytest

from temsim.magnetic_field_lines import build_field_lines, field_strength_fraction


@dataclass
class Scene:
    field: object
    bounds_m: object = ((-1., -1., -1.), (1., 1., 1.))
    seed_regions_m: object = None
    source_keys: tuple = ("analytic_fixture",)
    notes: tuple = ()
    mask: object = None

    def __post_init__(self):
        self.bounds_m = np.asarray(self.bounds_m)
        if self.seed_regions_m is None:
            self.seed_regions_m = (self.bounds_m,)

    def contains(self, points):
        valid = ((points >= self.bounds_m[0]) & (points <= self.bounds_m[1])).all(axis=1)
        if self.mask is not None:
            valid &= self.mask(points)
        return valid

    def field_at_global_positions_t(self, points):
        # Validate that a provider is never queried outside its support.
        assert np.all(self.contains(points))
        return self.field(points)


def uniform(vector, **kwargs):
    return Scene(lambda p: np.broadcast_to(vector, p.shape).copy(), **kwargs)


def test_shared_field_colour_fraction_has_a_fixed_logarithmic_scale():
    strengths = np.array((0., 1e-6, 1e-3, 1., 100.))
    fractions = field_strength_fraction(strengths, 1.)
    np.testing.assert_allclose(fractions, np.log1p(np.minimum(strengths, 1.) * 1e6) / np.log1p(1e6))
    np.testing.assert_array_equal(fractions[[0, -2, -1]], (0., 1., 1.))
    assert 0 < fractions[1] < fractions[2] < 1
    np.testing.assert_array_equal(field_strength_fraction(strengths, 1., gain=.5),
                                  field_strength_fraction(strengths, 2.))
    np.testing.assert_array_equal(field_strength_fraction(strengths, 1., gain=0.), np.zeros(5))


@pytest.mark.parametrize("strengths, reference, gain", [
    ((-1.,), 1., 1.), ((np.nan,), 1., 1.), ((np.inf,), 1., 1.),
    ((1.,), 0., 1.), ((1.,), np.inf, 1.), ((1.,), 1., -1.),
])
def test_field_colour_fraction_rejects_invalid_physical_strengths(strengths, reference, gain):
    with pytest.raises(ValueError):
        field_strength_fraction(strengths, reference, gain=gain)


@pytest.mark.parametrize("vector", [(0., 0., 1.), (0., 0., -1.), (1., 0., 0.), (0., -1., 0.)])
def test_uniform_field_segments_and_arrows_follow_positive_b(vector):
    scene = uniform(vector)
    result = build_field_lines(scene, reference_t=1., max_lines=33)
    assert result.line_count == result.seed_count == 33
    for pieces in (result.segments_m, result.direction_segments_m):
        delta = pieces[:, 1] - pieces[:, 0]
        assert np.all(np.sum(delta * vector, axis=1) > 0)
        assert np.allclose(np.cross(delta, vector), 0)
    assert np.allclose(result.strengths_t, 1.)
    axis = np.argmax(np.abs(vector))
    transverse = [i for i in range(3) if i != axis]
    # No duplicate lines produced by seeds stacked along the same field line.
    unique = np.unique(result.segments_m[:, 0, transverse], axis=0)
    assert len(unique) == result.line_count
    assert np.all(scene.contains(result.segments_m.reshape(-1, 3)))


def test_density_uses_fixed_absolute_reference_without_frame_renormalisation():
    counts = []
    for magnitude in (0., 0.01, 0.1, 0.3, 0.7, 1., 2.):
        result = build_field_lines(uniform((0., 0., magnitude)), reference_t=1., max_lines=200)
        counts.append(result.line_count)
        if result.line_count:
            assert np.allclose(result.strengths_t, magnitude)
    assert counts[0] == 0
    assert all(a < b for a, b in zip(counts[:5], counts[1:6]))
    assert counts[-2:] == [200, 200]
    low = build_field_lines(uniform((0., 0., .5)), reference_t=1., density=.5)
    high = build_field_lines(uniform((0., 0., .5)), reference_t=1., density=1.)
    changed_ref = build_field_lines(uniform((0., 0., .5)), reference_t=2., density=1.)
    assert low.line_count < high.line_count
    assert changed_ref.line_count == low.line_count


def test_log_density_retains_weak_fields_and_increases_across_six_decades():
    counts = []
    for magnitude in np.logspace(-6, 0, 7):
        result = build_field_lines(uniform((0., 0., magnitude)), reference_t=1., max_lines=200)
        counts.append(result.line_count)
        # Compression changes seed acceptance only, not the recorded tesla.
        np.testing.assert_allclose(result.strengths_t, magnitude)
    assert counts[0] > 0
    assert all(a < b for a, b in zip(counts, counts[1:]))
    assert counts[-1] == 200
    assert any("six-decade logarithmic" in note for note in result.notes)


def test_strong_lens_and_weak_transverse_regions_share_one_global_reference():
    regions = tuple(np.asarray(((-.001, -.001, z - .003), (.001, .001, z + .003)))
                    for z in (-.02, 0., .02))

    def mask(points):
        return np.any([((points >= r[0]) & (points <= r[1])).all(axis=1)
                       for r in regions], axis=0)

    def field(points, lens_scale=1.):
        b = np.zeros_like(points)
        lens = points[:, 2] < -.01
        deflector = points[:, 2] > .01
        stigmator = ~lens & ~deflector
        b[lens, 2] = .5 * lens_scale
        b[deflector, 0] = 2e-5
        b[deflector, 1] = -3e-5
        # Divergence-free ideal quadrupole in its declared local support.
        b[stigmator, 0] = .05 * points[stigmator, 0]
        b[stigmator, 1] = -.05 * points[stigmator, 1]
        return b

    scene = Scene(field, bounds_m=((-0.001, -.001, -.023), (.001, .001, .023)),
                  seed_regions_m=regions, mask=mask,
                  source_keys=("lens", "stigmator", "deflector"))
    result = build_field_lines(scene, reference_t=.5)
    centres = result.segments_m.mean(axis=1)
    for region in regions:
        assert np.any(((centres >= region[0]) & (centres <= region[1])).all(axis=1))

    deflector = centres[:, 2] > .01
    deflector_segments = result.segments_m[deflector]
    delta = np.diff(deflector_segments, axis=1)[:, 0]
    np.testing.assert_allclose(np.cross(delta, (2e-5, -3e-5, 0.)), 0., atol=1e-20)
    np.testing.assert_allclose(result.strengths_t[deflector], np.hypot(2e-5, 3e-5))

    # Changing a spatially disjoint lens does not renormalise, strengthen or
    # reseed the weak deflector. Its local field and reference are held fixed.
    scene.field = lambda p: field(p, lens_scale=10.)
    brighter_lens = build_field_lines(scene, reference_t=.5)
    other_deflector = brighter_lens.segments_m.mean(axis=1)[:, 2] > .01
    np.testing.assert_array_equal(deflector_segments, brighter_lens.segments_m[other_deflector])


def test_deterministic_bounded_geometry_and_read_only_arrays():
    scene = uniform((0., 0., 1.))
    a = build_field_lines(scene, reference_t=.5, max_lines=37, max_steps=2)
    b = build_field_lines(scene, reference_t=.5, max_lines=37, max_steps=2)
    assert len(a.segments_m) <= 2 * 37 * 2
    assert a.line_count <= 37
    np.testing.assert_array_equal(a.segments_m, b.segments_m)
    assert not a.segments_m.flags.writeable
    assert any("display step budget" in note for note in a.notes)
    assert build_field_lines(scene, reference_t=1., density=0).line_count == 0


def test_no_chord_connects_across_a_known_domain_gap():
    # With the 0.4 m step, both endpoints 0.4 and 0.8 are valid. A test only
    # at the RK midpoint (0.6) would miss the actual 0.45..0.55 map hole.
    scene = uniform((0., 0., 1.), mask=lambda p: (p[:, 2] < .45) | (p[:, 2] > .55))
    result = build_field_lines(scene, reference_t=1., max_lines=17)
    z = result.segments_m[..., 2]
    assert not np.any((z[:, 0] < .45) & (z[:, 1] > .55))
    assert np.max(z) <= .45


def test_nonfinite_or_zero_field_stops_without_invalid_output():
    def field(points):
        b = np.zeros_like(points)
        b[:, 2] = 1.
        b[points[:, 2] > .5] = np.nan
        return b

    result = build_field_lines(Scene(field), reference_t=1., max_lines=13)
    assert np.all(np.isfinite(result.segments_m))
    assert np.max(result.segments_m[..., 2]) <= .5
    result = build_field_lines(uniform((0., 0., 0.)), reference_t=1.)
    assert result.segments_m.shape == (0, 2, 3)
    assert result.direction_segments_m.shape == (0, 2, 3)
    assert result.strengths_t.shape == (0,)


def test_axisymmetric_lens_lines_bend_and_resolve_analytic_flux_tubes():
    # Divergence-free paraxial field Bz=1+z^2, Br=-r*z. Its integral curves
    # satisfy r*sqrt(1+z^2)=constant, with minimum radius at the stronger ends.
    def field(points):
        x, y, z = points.T
        return np.column_stack((-x * z, -y * z, 1. + z*z))

    scene = Scene(field, bounds_m=((-0.25, -0.25, -1.), (.25, .25, 1.)))
    result = build_field_lines(scene, reference_t=1., max_lines=41)
    points = result.segments_m
    radii = np.linalg.norm(points[..., :2], axis=2)
    invariant = radii * np.sqrt(1. + points[..., 2] ** 2)
    # RK2 display tolerance is tied to this known field, not particle tracing.
    relative_step_error = np.abs(invariant[:, 1] - invariant[:, 0]) / invariant[:, 0]
    assert np.max(relative_step_error) < .0005
    delta = points[:, 1] - points[:, 0]
    near_end = points[:, 0, 2] > .5
    assert np.any(near_end)
    assert np.all(np.sum(delta[near_end, :2] * points[near_end, 0, :2], axis=1) < 0)
    assert np.ptp(result.strengths_t) > .5


def test_zero_on_axis_quadrupole_still_has_off_axis_field_lines():
    scene = Scene(lambda p: np.column_stack((p[:, 0], -p[:, 1], np.zeros(len(p)))))
    result = build_field_lines(scene, reference_t=.5)
    assert result.line_count > 0
    assert np.all(result.strengths_t > 0)


def test_seeds_find_split_lens_lobes_instead_of_only_the_weak_box_centre():
    def field(points):
        x, y, z = points.T
        upper = np.exp(-.5 * ((z - .5) / .08) ** 2)
        lower = np.exp(-.5 * ((z + .5) / .08) ** 2)
        bz = upper + lower
        derivative = -(z - .5) / .08**2 * upper - (z + .5) / .08**2 * lower
        return np.column_stack((-.5 * x * derivative, -.5 * y * derivative, bz))

    scene = Scene(field, bounds_m=((-0.01, -0.01, -1.), (.01, .01, 1.)))
    assert np.linalg.norm(scene.field_at_global_positions_t(np.zeros((1, 3)))) < 1e-8
    result = build_field_lines(scene, reference_t=1., max_lines=100)
    assert result.line_count > 20


@pytest.mark.parametrize("kwargs", [
    {"reference_t": 0.}, {"reference_t": np.nan},
    {"reference_t": 1., "density": -1.}, {"reference_t": 1., "max_steps": 0},
    {"reference_t": 1., "max_lines": 0}, {"reference_t": 1., "max_lines": 1.5},
])
def test_bad_numeric_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        build_field_lines(uniform((0., 0., 1.)), **kwargs)
