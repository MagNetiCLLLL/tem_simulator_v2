"""Remove redundant numerical matrices, never propagated channels or states."""
import math

import numpy as np

from temsim.physics.occupied_axial_refinement import OccupiedAxialRefinement, _Work, apply_two_port
from temsim.physics.scattering_load import compose
from temsim.physics.spatial_embedded_mesh import initial_mesh, size
from test_occupied_axial_refinement import plan_and_boundary


def test_current_ports_match_full_composed_error_without_retaining_composition():
    plan, _, _, _ = plan_and_boundary()
    leaf, = initial_mesh(plan)
    work = _Work(OccupiedAxialRefinement(integrator="cf6"), lambda: False, None)
    rng = np.random.default_rng(712)
    for _ in range(5):
        left, right = rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2))
        error, _ = leaf.refine(left, right, math.inf, work)
        composed = compose(*leaf.halves)
        reference = np.linalg.norm(apply_two_port(tuple(a-b for a, b in zip(leaf.coarse, composed)),
                                                  left, right))*math.sqrt(leaf.kappa)
        np.testing.assert_allclose(error, reference, atol=3e-15, rtol=3e-12)
    assert work.evaluations == 2  # New inputs reuse complete half matrices only.
    assert leaf.operator is None
    assert work.retained_bytes == sum(size(half) for half in leaf.halves)
    assert work.retained_bytes == 2*size(leaf.coarse)  # Previously three full operators.
    children = leaf.split(work)
    assert all(child.owns_coarse for child in children)
    assert work.retained_bytes == sum(size(child.coarse) for child in children)
