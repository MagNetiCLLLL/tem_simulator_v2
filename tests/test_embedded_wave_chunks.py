"""Fixed grouping equivalence and full-operator checks, not gun acceptance."""
from dataclasses import replace
import math

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from temsim.physics.embedded_wave_chunks import refine_embedded_process_intervals
from temsim.physics.global_embedded_refinement import EmbeddedInterval
from temsim.physics.occupied_axial_refinement import OccupiedAxialRefinement, _Work, evaluate_interval
from temsim.physics.scattering_load import compose, hermitian_slab
from test_occupied_axial_refinement import varying


def tree():
    result = EmbeddedInterval(varying, 1., 2., hermitian_slab(*varying(.5), 1., 2., carrier_k=2.)[0])
    result.divisions = 256
    return result


@pytest.mark.parametrize("integrator", ["cf4", "cf6"])
def test_one_long_interval_uses_multiple_jobs_with_identical_serial_grouping(integrator):
    settings = OccupiedAxialRefinement(enabled=True, integrator=integrator, maximum_evaluations=20_000,
        workers=2, executor="process", strategy="global_embedded")
    incoming = [(np.array([.5, .3j]), np.array([.07j, .01]))]*2
    parallel, serial = {0: tree()}, tree()
    work = _Work(settings, lambda: False, None)
    single = _Work(replace(settings, workers=1), lambda: False, None)
    with threadpool_limits(1):
        serial.refine(incoming[0][0], incoming[1][1], math.inf, single)
        update = refine_embedded_process_intervals(parallel, incoming, work)
    np.testing.assert_array_equal(parallel[0].operator, serial.operator)
    assert update[0][1] == serial.refine(incoming[0][0], incoming[1][1], math.inf, single)[0]
    assert work.evaluations == single.evaluations == 512
    assert work.retained_bytes == single.retained_bytes == sum(b.nbytes for b in serial.operator)
    # Cached operator reuse computes the new incoming error, not its slabs.
    refine_embedded_process_intervals(parallel, incoming, work)
    assert work.evaluations == 512


def test_chunk_association_matches_original_full_serial_composition():
    settings = OccupiedAxialRefinement(enabled=True, integrator="cf6", maximum_evaluations=20_000)
    work = _Work(settings, lambda: False, None)
    grouped = tree()
    with threadpool_limits(1):
        grouped.refine(np.ones(2), np.ones(2)*.1j, math.inf, work)
        original = None
        for j in range(512):
            slab = evaluate_interval(varying, 1., 2., j/512, (j+1)/512, work)
            original = slab if original is None else compose(original, slab)
    np.testing.assert_allclose(grouped.operator, original, rtol=5e-13, atol=5e-13)


def test_chunk_failures_preserve_work_and_do_not_publish_partial_operator():
    settings = OccupiedAxialRefinement(enabled=True, workers=2, executor="process", maximum_evaluations=2)
    roots = {0: tree()}
    work = _Work(settings, lambda: False, None)
    inputs = [(np.ones(2), np.zeros(2))]*2
    with pytest.raises(ValueError, match="evaluation budget"):
        refine_embedded_process_intervals(roots, inputs, work)
    assert roots[0].operator is None
