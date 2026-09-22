"""An upstream section endpoint cannot supply specimen convergence colours."""
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from temsim.gui.visualization import VisualizationWorkspace


def display_and_branch(metrics):
    slopes = np.array([[-.001, 0., .001], [-.002, 0., .002]])
    branch = SimpleNamespace(
        tx=slopes, ty=np.zeros_like(slopes), z=np.array([450., 550.]),
        blocked_z=np.full(3, np.nan), ray_weight=np.ones(3) / 3.,
    )
    simulation = SimpleNamespace(incident=branch, metrics=metrics)
    display = SimpleNamespace(
        _last_result=SimpleNamespace(simulation=simulation),
        _branch_interaction_kind=lambda _branch: "incident",
    )
    display._sample_convergence_semiangles_mrad = MethodType(
        VisualizationWorkspace._sample_convergence_semiangles_mrad, display)
    return display, simulation, branch


def test_unreached_sample_has_no_convergence_even_with_finite_section_endpoint():
    display, simulation, branch = display_and_branch({
        "section_sample_reference_reached": False,
        "sample_convergence_99_mrad": float("nan"),
    })
    original = branch.tx.copy()
    full = display._sample_convergence_semiangles_mrad(branch)
    subset = display._sample_convergence_semiangles_mrad(branch, [2, 0])
    assert full.shape == (3,) and np.all(np.isnan(full))
    assert subset.shape == (2,) and np.all(np.isnan(subset))
    assert VisualizationWorkspace._convergence_reference_mrad(display, simulation) == 0.
    np.testing.assert_array_equal(branch.tx, original)


@pytest.mark.parametrize("metrics", [{}, {"section_sample_reference_reached": True}])
def test_legacy_or_reached_sample_keeps_existing_convergence_reader(metrics):
    display, _simulation, branch = display_and_branch(metrics)
    expected = np.arctan(np.array([.002, 0., .002])) * 1e3
    np.testing.assert_allclose(display._sample_convergence_semiangles_mrad(branch), expected)
    np.testing.assert_allclose(display._sample_convergence_semiangles_mrad(branch, [2, 0]), expected[[2, 0]])
