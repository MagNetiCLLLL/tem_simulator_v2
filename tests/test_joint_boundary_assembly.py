"""Consumed FEM/source/interface identity; no current calibration or source admission."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from temsim.physics.joint_boundary_assembly import assembly_identity, prepare_joint_assembly


def problem():
    return dict(points=np.zeros((5, 2)), triangles=np.array([[0, 1, 2]]), potential=np.ones(5),
        drive=np.ones(5, complex), edges={name: np.array([[0, 1]]) for name in ("source", "side", "top")},
        matrices=tuple(csc_matrix(np.eye(5)*scale) for scale in (1., 2., 3.)))


@pytest.mark.parametrize("name", ["points", "triangles", "potential", "drive", "source", "side", "top", "matrices"])
def test_reuse_key_binds_all_consumed_problem_arrays(name):
    first = problem()
    second = deepcopy(first)
    args = (.3, 4, 1.2, .1, 2., True)
    original = assembly_identity(first, *args)
    assert original == assembly_identity(second, *args)
    if name in second["edges"]:
        second["edges"][name][0, 0] += 1
    elif name == "matrices":
        second["matrices"][1].data[0] += .2
    else:
        second[name].flat[0] += 1
    assert original != assembly_identity(second, *args)


@pytest.mark.parametrize("index", range(6))
def test_reuse_key_binds_energy_and_entire_matching_basis(index):
    inputs = [.3, 4, 1.2, .1, 2., True]
    original = assembly_identity(problem(), *inputs)
    inputs[index] = False if index == 5 else inputs[index]+1
    assert original != assembly_identity(problem(), *inputs)


def test_no_assembly_allocation_after_cancellation_or_tiny_memory_budget():
    with pytest.raises(InterruptedError):
        prepare_joint_assembly(problem(), .3, 4, 1.2, .1, 2., True,
            maximum_working_bytes=1<<20, cancelled=lambda: True)
    with pytest.raises(MemoryError):
        prepare_joint_assembly(problem(), .3, 4, 1.2, .1, 2., True,
            maximum_working_bytes=1, cancelled=lambda: False)


def test_sparse_storage_format_is_part_of_consumed_operator_identity():
    first = problem()
    second = deepcopy(first)
    second["matrices"] = tuple(matrix.tocsr() for matrix in second["matrices"])
    assert assembly_identity(first, .3, 4, 1.2, .1, 2., True) != assembly_identity(second, .3, 4, 1.2, .1, 2., True)
