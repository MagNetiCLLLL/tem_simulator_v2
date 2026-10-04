"""Independent electron arrivals from absolute, executed screen probabilities.

Small analytic distributions test statistics and flux loss; these tests do not
qualify propagation, a physical detector, or any individual electron path.
"""

import numpy as np
import pytest

from temsim.physics.electron_detection import (
    MAX_EMITTED_ELECTRONS, sample_electron_detections,
)


def test_two_bin_absolute_probabilities_keep_the_lost_electrons():
    # Two 2 um x 3 um bins: density integrates to P=(0.1, 0.2), not 1.
    emitted = 200_000
    probability = np.array([[.1], [.2]])
    result = sample_electron_detections(probability/6., ((-2., 2.), (3., 6.)),
                                       emitted_electrons=emitted, seed=481)
    assert result.emitted_count == emitted
    assert result.detection_probability == pytest.approx(.3)
    assert result.detected_count + result.lost_count == emitted
    assert result.detected_count == int(result.counts.sum()) == len(result.points_um)
    observed = np.r_[result.counts.ravel(), result.lost_count]
    expected_probability = np.array([.1, .2, .7])
    sigma = np.sqrt(emitted*expected_probability*(1.-expected_probability))
    assert np.all(np.abs(observed-emitted*expected_probability) < 6.*sigma)


def test_independent_modes_contribute_probabilities_despite_opposite_phases():
    from temsim.physics.multiplane_wave import PlaneWave
    from temsim.physics.wave_flux import BeamState, WaveMode

    probability = np.array([[.1, .2], [.3, .4]])
    modes = tuple(WaveMode(PlaneWave(sign*np.sqrt(probability), np.eye(2)*1e-6,
                                    np.zeros(2)), .3, "tip", str(index), 200.)
                  for index, sign in enumerate((1., -1.)))
    beam = BeamState(modes, "tip")
    # Each display cell is 1 square micrometre. Amplitude addition would
    # incorrectly cancel these independent modes and produce no arrivals.
    density = beam.cell_probabilities().T
    np.testing.assert_allclose(density, .6*probability.T, rtol=1e-14)
    result = sample_electron_detections(density, ((-1.5, .5), (-1.5, .5)),
                                       emitted_electrons=30_000, seed=73)
    assert result.detection_probability == pytest.approx(.6)
    assert abs(result.detected_count-18_000) < 6.*np.sqrt(30_000*.6*.4)
    assert result.lost_count > 0


def test_non_square_grid_keeps_xy_orientation_and_rectangular_bin_edges():
    density = np.zeros((2, 3))  # Display bins are X,Y, unlike native wave arrays.
    density[1, 0] = 1./12.
    result = sample_electron_detections(density, ((10., 16.), (-9., 3.)),
                                       emitted_electrons=4000, seed=5)
    assert result.counts.shape == (2, 3)
    assert result.counts[1, 0] == result.detected_count == 4000
    assert result.lost_count == 0
    x, y = result.points_um.T
    assert np.all((x >= 13.) & (x < 16.))
    assert np.all((y >= -9.) & (y < -5.))
    # A virtual hit is within its bin, not all at a grid centre.
    assert np.std(x) == pytest.approx(3./np.sqrt(12.), rel=.04)
    assert np.std(y) == pytest.approx(4./np.sqrt(12.), rel=.04)


def test_reported_histogram_agrees_with_returned_hit_coordinates():
    density = np.arange(1., 7.).reshape((2, 3))/100.
    bounds = np.array(((-1., 1.), (-2., 1.)))
    result = sample_electron_detections(density, bounds, emitted_electrons=20_000, seed=23)
    counts, _, _ = np.histogram2d(result.points_um[:, 0], result.points_um[:, 1],
                                 bins=density.shape, range=bounds)
    np.testing.assert_array_equal(counts, result.counts)


def test_same_seed_reproduces_hits_and_increasing_emissions_preserves_prefix():
    density = np.array([[.08, .17], [.1, .2]])
    bounds = ((-1., 1.), (-1., 1.))
    small = sample_electron_detections(density, bounds, emitted_electrons=67_001, seed=31)
    same = sample_electron_detections(density, bounds, emitted_electrons=67_001, seed=31)
    large = sample_electron_detections(density, bounds, emitted_electrons=100_003, seed=31)
    different = sample_electron_detections(density, bounds, emitted_electrons=67_001, seed=32)
    np.testing.assert_array_equal(same.points_um, small.points_um)
    np.testing.assert_array_equal(same.counts, small.counts)
    np.testing.assert_array_equal(large.points_um[:small.detected_count], small.points_um)
    assert np.all(large.counts >= small.counts)
    assert large.lost_count >= small.lost_count
    assert not np.array_equal(different.points_um, small.points_um)


@pytest.mark.parametrize("emitted", [0, 1234])
def test_absorbed_beam_has_no_virtual_hits(emitted):
    result = sample_electron_detections(np.zeros((2, 3)), ((0., 2.), (0., 3.)),
                                       emitted_electrons=emitted)
    assert result.points_um.shape == (0, 2)
    assert result.detected_count == result.detection_probability == 0
    assert result.lost_count == emitted
    assert not np.any(result.counts)


def test_zero_emission_still_reports_available_absolute_probability():
    result = sample_electron_detections([[.25]], ((0., 1.), (0., 1.)),
                                       emitted_electrons=0)
    assert result.points_um.shape == (0, 2)
    assert result.detection_probability == .25
    assert result.emitted_count == result.detected_count == result.lost_count == 0


def test_result_arrays_are_read_only_and_do_not_share_input_buffers():
    density = np.array([[.5]])
    bounds = np.array(((0., 1.), (0., 1.)))
    result = sample_electron_detections(density, bounds, emitted_electrons=10, seed=3)
    before = result.points_um.copy()
    density[:] = 0.
    bounds[:] = 0.
    np.testing.assert_array_equal(result.points_um, before)
    for array in (result.points_um, result.counts):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize("emitted", [-1, True, False, 1., np.nan, np.inf,
                                     MAX_EMITTED_ELECTRONS+1, "5"])
def test_invalid_emitted_count_is_rejected_without_truncation(emitted):
    with pytest.raises(ValueError, match="emitted_electrons"):
        sample_electron_detections([[1.]], ((0., 1.), (0., 1.)), emitted_electrons=emitted)


@pytest.mark.parametrize("seed", [-1, True, 2., np.nan, np.inf, "5", 2**64])
def test_invalid_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        sample_electron_detections([[1.]], ((0., 1.), (0., 1.)), emitted_electrons=10, seed=seed)


@pytest.mark.parametrize("density", [[], [1.], [[np.nan]], [[np.inf]], [[-.1]],
                                    [[1.+1j]], [[1.01]], [[1e308, 1e308]]])
def test_invalid_absolute_probability_is_rejected(density):
    with pytest.raises(ValueError):
        sample_electron_detections(density, ((0., 1.), (0., 1.)), emitted_electrons=10)


@pytest.mark.parametrize("bounds", [((1., 0.), (0., 1.)), ((0., 0.), (0., 1.)),
    ((0., np.inf), (0., 1.)), ((0., np.nan), (0., 1.)), (0., 1.),
    ((-1e308, 1e308), (0., 1.)), ((0., 1e200), (0., 1e200)),
    ((0., 1e-300), (0., 1e-300))])
def test_invalid_geometry_is_rejected(bounds):
    with pytest.raises(ValueError, match="bounds|area"):
        sample_electron_detections([[.1]], bounds, emitted_electrons=10)


def test_probability_above_one_is_only_tolerated_at_floating_point_roundoff():
    result = sample_electron_detections([[1.+1e-14]], ((0., 1.), (0., 1.)),
                                       emitted_electrons=100)
    assert result.detected_count == 100 and result.lost_count == 0
    assert result.detection_probability == 1.


def test_numpy_integer_counts_and_seeds_have_the_same_deterministic_contract():
    args = [[.5]], ((0., 1.), (0., 1.))
    result = sample_electron_detections(*args, emitted_electrons=np.int64(17), seed=np.uint64(29))
    reference = sample_electron_detections(*args, emitted_electrons=17, seed=29)
    np.testing.assert_array_equal(result.points_um, reference.points_um)
