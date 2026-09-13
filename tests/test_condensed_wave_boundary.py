"""Exact full-state reconstruction, cache invalidation and independent sparse solves."""
import numpy as np
import pytest
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from temsim.physics.condensed_wave_boundary import solve_cached_boundary


def fixture():
    rng = np.random.default_rng(22)
    a = rng.normal(size=(80, 80))+1j*rng.normal(size=(80, 80))
    matrix = a.conj().T@a-2j*np.eye(80)
    rhs = rng.normal(size=80)+1j*rng.normal(size=80)
    return csc_matrix(matrix), rhs


def test_every_new_reflected_load_resolves_complete_complex_interior():
    matrix, rhs = fixture()
    cache, outputs = {}, []
    for i, strength in enumerate((.1, 3., .2)):
        load = (strength+.2j)*np.eye(8)
        actual, record = solve_cached_boundary(matrix, rhs, load, cache)
        direct = matrix.toarray()
        direct[-8:, -8:] -= load
        expected = spsolve(csc_matrix(direct), rhs)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
        assert record["cache_hit"] == (i != 0)
        assert record["linear_residual"] < 1e-12
        outputs.append(actual)
    assert np.linalg.norm(outputs[0]-outputs[1]) > .01


def test_physical_operator_or_drive_change_invalidates_condensation():
    matrix, rhs = fixture()
    cache = {}
    load = 1j*np.eye(8)
    solve_cached_boundary(matrix, rhs, load, cache)
    rhs[0] += .1j
    _, record = solve_cached_boundary(matrix, rhs, load, cache)
    assert not record["cache_hit"]
    matrix.data[0] += .1
    _, record = solve_cached_boundary(matrix, rhs, load, cache)
    assert not record["cache_hit"]
    assert len(cache) == 1


def test_memory_and_cancellation_cannot_return_an_incomplete_wave():
    matrix, rhs = fixture()
    with pytest.raises(MemoryError):
        solve_cached_boundary(matrix, rhs, 1j*np.eye(8), {}, maximum_working_bytes=1)
    with pytest.raises(InterruptedError):
        solve_cached_boundary(matrix, rhs, 1j*np.eye(8), {}, cancelled=lambda: True)


def test_actual_tip_and_full_round_gun_match_direct_boundary_solution(monkeypatch):
    from dataclasses import replace
    from threadpoolctl import threadpool_limits
    from temsim.optics.column import default_state
    from temsim.optics.electron_gun.tip_surface import load_tip_surface_reference, SurfaceCoherence
    from temsim.physics.surface_wave import SurfaceWaveNumerics, prepare_surface_problem
    from temsim.physics.radial_gun_wave import RadialGunNumerics, prepare_round_gun
    from temsim.physics.surface_gun_wave import solve_joint_boundary
    gun = default_state().electron_gun
    gun.emitter.surface_model = replace(load_tip_surface_reference(), coherence=SurfaceCoherence())
    surface = SurfaceWaveNumerics(element_order=2, radial_nodes=97, axial_nodes=65, outer_radius_factor=4.)
    with threadpool_limits(1):
        problem = prepare_surface_problem(gun, surface)
        energy = float(problem["energies"][0])
        plan = {}
        load, frame = prepare_round_gun(gun, surface.exit_height_nm, energy,
            RadialGunNumerics(radial_modes=8, potential_quadrature=32, relative_axial_step=.1, field_step_mm=.5),
            refinement_plan=plan)
        import cloudpickle
        indices = list(plan["samplers"])
        for index in (indices[0], indices[len(indices)//2], indices[-1]):
            sample = plan["samplers"][index][0]
            copied = cloudpickle.loads(cloudpickle.dumps(sample))
            for original, transferred in zip(sample(.37), copied(.37)):
                np.testing.assert_array_equal(original, transferred)
        direct = solve_joint_boundary(problem, energy, load, frame)
        cache, assembly_cache = {}, {}
        actual = solve_joint_boundary(problem, energy, load, frame, condensation_cache=cache, assembly_cache=assembly_cache)
        import temsim.physics.surface_wave as surface_module
        def no_reassembly(*args, **kwargs):
            raise AssertionError("Fixed FEM volume should not be reassembled for the same boundary input")
        with monkeypatch.context() as context:
            context.setattr(surface_module, "fem_volume", no_reassembly)
            repeated = solve_joint_boundary(problem, energy, load, frame,
                condensation_cache=cache, assembly_cache=assembly_cache)
        assert repeated[2]["boundary_linear_solve"]["assembly_cache_hit"]
        assert len(assembly_cache) == 1
        # The consumed input arrays, not the problem object's identity, bind reuse.
        changed = dict(problem, drive=problem["drive"]*np.exp(.1j))
        another = solve_joint_boundary(changed, energy, load, frame,
            condensation_cache=cache, assembly_cache=assembly_cache)
        assert not another[2]["boundary_linear_solve"]["assembly_cache_hit"]
        np.testing.assert_allclose(another[0], actual[0]*np.exp(.1j), atol=1e-10, rtol=1e-9)
    np.testing.assert_allclose(actual[0], direct[0], atol=1e-10, rtol=1e-9)
    np.testing.assert_allclose(actual[1], direct[1], atol=1e-10, rtol=1e-9)
    assert actual[2]["top"] == pytest.approx(direct[2]["top"], abs=1e-10)
    assert actual[2]["balance_error"] < 1e-8
    assert repeated[2]["boundary_linear_solve"]["cache_hit"]
    # This is one-mode solver equivalence at coarse numerics, NOT full-gun
    # convergence or a full source/image acceptance certificate.
