"""Near-axis force precision, scalar consistency and real-boundary checks."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics.axis_regular_potential import AxisRegularPotential
from temsim.physics.axisymmetric_cut_field import AxisymmetricCutField
from temsim.optics.electron_gun.tip_surface import load_tip_surface_reference


def nodal_fixture():
    # Exact harmonic potential sampled on a deliberately multiscale grid.
    # This tests interpolation conditioning independently of the FEM solver.
    f = object.__new__(AxisymmetricCutField)
    f.r = np.r_[0., np.geomspace(1e-9, .004, 80)]
    f.z = np.array([.001, .004, .008, .012, .02])
    rr, zz = np.meshgrid(f.r, f.z, indexing="ij")
    f.nodal_voltage = 100000.+2e5*zz+1e7*(rr**2-2*zz**2)
    f.cut_cells = np.zeros((len(f.r)-1, len(f.z)-1), bool)
    f.lookup = np.full((2*f.cut_cells.size, 2), -1)
    f.origin = f.gradient = np.empty((0, 2))
    f.inverse = np.empty((0, 2, 2))
    f.phi0 = np.empty(0)
    return f


def test_kilovolt_subtraction_cannot_erase_or_reverse_axis_focusing():
    raw = nodal_fixture()
    regular = AxisRegularPotential(raw, bore_radius_m=.002, fraction=.01)
    radii = np.geomspace(1e-12, 1e-6, 30)
    p = np.column_stack((radii, radii*0, radii*0+.01))
    _, old = raw.interpolate(p)
    _, new = regular.interpolate(p)
    expected = -2e7
    assert np.max(abs(old[:, 0]/radii/expected-1)) > .01  # reproduced defect
    np.testing.assert_allclose(new[:, 0]/radii, expected, rtol=2e-5)
    assert np.all(new[:, 0] < 0)


def test_force_differentiates_the_same_potential_and_is_rotationally_covariant():
    raw = nodal_fixture()
    regular = AxisRegularPotential(raw, bore_radius_m=.002, fraction=.01)
    p = np.array([[8e-6, 3e-6, .01], [1.1e-5, 1e-6, .006]])
    potential, electric = regular.interpolate(p)
    for axis in range(3):
        delta = np.eye(3)[axis]*1e-7
        derivative = (regular.interpolate(p+delta)[0]-regular.interpolate(p-delta)[0])/2e-7
        np.testing.assert_allclose(-derivative, electric[:, axis], rtol=2e-4, atol=.002)
    rotated = p[:, [1, 0, 2]]*np.array([-1, 1, 1])
    rp, re = regular.interpolate(rotated)
    np.testing.assert_allclose(rp, potential, atol=1e-10)
    np.testing.assert_allclose(re, electric[:, [1, 0, 2]]*[-1, 1, 1], atol=1e-8)


def test_core_and_axial_boundaries_preserve_potential_continuity():
    raw = nodal_fixture()
    regular = AxisRegularPotential(raw, bore_radius_m=.002, fraction=.01)
    for j in (1, 2):
        z = raw.z[j]
        r = regular.radius_m[j]
        p = np.array([[r, 0, z]])
        for axis in (0, 2):
            delta = np.eye(3)[axis]*1e-13
            a, _ = regular.interpolate(p-delta)
            b, _ = regular.interpolate(p+delta)
            assert abs(float((b-a)[0])) < 1e-7


def test_off_axis_and_disabled_interpolation_are_unchanged():
    raw = nodal_fixture()
    p = np.array([[.001, .0001, .006], [.003, 0., .01]])
    for fraction in (0., .01):
        new = AxisRegularPotential(raw, bore_radius_m=.002, fraction=fraction).interpolate(p)
        old = raw.interpolate(p)
        for a, b in zip(new, old):
            np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="fraction"):
        AxisRegularPotential(raw, bore_radius_m=.002, fraction=.2)


def test_harmonic_fourth_order_core_error_converges_with_support_radius():
    raw = nodal_fixture()
    raw.r = np.array([0., 1e-9, .001, .002, .004, 1.])
    raw.z = np.array([.05, .1, .2])
    rr, zz = np.meshgrid(raw.r, raw.z, indexing="ij")
    # Exact axisymmetric harmonic quartic: Laplacian(phi) = 0.
    raw.nodal_voltage = 5.+rr**4-8*rr**2*zz**2+(8/3)*zz**4
    raw.cut_cells = np.zeros((len(raw.r)-1, len(raw.z)-1), bool)
    raw.lookup = np.full((2*raw.cut_cells.size, 2), -1)
    point = np.array([[1e-7, 0., .1]])
    exact = 16*.1**2-4*point[0, 0]**2
    errors = []
    for fraction in (.04, .02, .01):
        regular = AxisRegularPotential(raw, bore_radius_m=1., fraction=fraction)
        coefficient = regular.interpolate(point)[1][0, 0]/point[0, 0]
        errors.append(abs(coefficient-exact))
    assert errors[0]/errors[1] > 3.9 and errors[1]/errors[2] > 3.9


@pytest.mark.parametrize("electrode_cells", [0, 8])
def test_real_gun_axis_limit_is_stable_and_tip_boundary_is_unchanged(electrode_cells):
    from temsim.optics.column import default_state
    from temsim.physics.grounded_tip_field import grounded_field
    gun = default_state().electron_gun
    model = load_tip_surface_reference()
    gun.emitter.surface_model = replace(model,
        field_numerics=replace(model.field_numerics, electrode_cells_per_bore=electrode_cells))
    field = grounded_field(gun)
    radii = np.geomspace(1e-12, 1e-6, 20)
    p = np.column_stack((radii, radii*0, radii*0+.2))
    _, electric = field._interpolate(p)
    coefficient = electric[:, 0]/radii
    np.testing.assert_allclose(coefficient, coefficient[0], rtol=1e-12)
    comparison = []
    for fraction in (.02, .01, .005):
        regular = AxisRegularPotential(field._fem,
            bore_radius_m=min(row[3] for row in field.request["rings"]), fraction=fraction)
        comparison.append(regular.interpolate(p)[1][0, 0]/radii[0])
    if electrode_cells == 0:
        # Replay the original defect's mesh. This was measured independently
        # before adding electrode refinement. It does not qualify other meshes:
        # the refined solve reaches a nodal precision floor at fraction .005.
        assert np.ptp(comparison)/abs(comparison[-1]) < .005
    geometry = gun.emitter.surface_model.geometry
    radius = geometry.apex_radius_nm*1e-9
    theta = np.deg2rad(np.arange(1., 10.))
    p = np.column_stack((radius*np.sin(theta), theta*0, radius*(np.cos(theta)-1)))
    represented, _ = field.surface_mesh_positions(p)
    a, b = field._regular.interpolate(represented), field._fem.interpolate(represented)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_numerical_option_roundtrips_and_invalidates_only_matching_cache_identity():
    from temsim.optics.column import default_state
    from temsim.physics.grounded_tip_field import field_request
    from temsim.instrument_snapshot import capture_instrument_snapshot
    state = default_state()
    gun = state.electron_gun
    gun.emitter.surface_model = load_tip_surface_reference()
    before = field_request(gun), gun._cache_key(193)
    original = gun.emitter.surface_model
    assert "axis_core_fraction" not in original.to_dict()["field_numerics"]
    gun.emitter.surface_model = replace(original,
        field_numerics=replace(original.field_numerics, axis_core_fraction=.005))
    assert before[0] != field_request(gun) and before[1] != gun._cache_key(193)
    restored = capture_instrument_snapshot(state).restore()
    assert restored.electron_gun._cache_key(193) == gun._cache_key(193)
    assert restored.electron_gun.emitter.surface_model.geometry == original.geometry
    assert restored.electron_gun.emitter.surface_model.emission == original.emission


def test_static_energy_step_with_regular_potential_preserves_work():
    from scipy.constants import m_e, c, e
    from temsim.physics.static_energy_lorentz import static_energy_step
    from temsim.physics.relativistic_lorentz import RelativisticPhaseSpace, momentum_from_kinetic_energy_ev
    regular = AxisRegularPotential(nodal_fixture(), bore_radius_m=.002, fraction=.01)
    field = SimpleNamespace(potential_v_at_global_positions=lambda p: regular.interpolate(p)[0],
        field_at_global_positions_v_per_m=lambda p: regular.interpolate(p)[1],
        field_at_global_positions_t=lambda p: np.zeros_like(p))
    start = RelativisticPhaseSpace(np.array([[8e-6, 3e-6, .01]]),
                                   momentum_from_kinetic_energy_ev([1000.], [[.1, .1, 1.]]))
    def conserved(phase):
        p2 = np.sum((phase.momentum_kg_m_per_s/(m_e*c))**2, axis=1)
        energy = m_e*c*c/e*p2/(np.sqrt(1+p2)+1)
        return energy-field.potential_v_at_global_positions(phase.position_m)
    result = static_energy_step(start, 1e-12, field, field)
    np.testing.assert_allclose(conserved(result), conserved(start), rtol=0, atol=1e-7)


def test_compiled_and_numpy_paths_agree_in_cut_cells_core_and_off_axis():
    from temsim.optics.column import default_state
    gun = default_state().electron_gun
    gun.emitter.surface_model = load_tip_surface_reference()
    field = gun.electric_field
    compiled = field._regular
    reference = AxisRegularPotential(field._fem, bore_radius_m=.002,
        fraction=compiled.fraction, compiled=False)
    rng = np.random.default_rng(12)
    radius = np.geomspace(1e-12,.1,300)
    phi = rng.uniform(0,2*np.pi,len(radius))
    z = rng.uniform(0,.45,len(radius))
    points = np.column_stack((radius*np.cos(phi),radius*np.sin(phi),z))
    angles = np.linspace(0,np.deg2rad(10),35)
    r = field.geometry.apex_radius_nm*1e-9
    tip = np.column_stack((r*np.sin(angles),angles*0,r*(np.cos(angles)-1)))
    tip,_ = field.surface_mesh_positions(tip)
    points = np.vstack((points,tip,tip+[0,0,1e-12]))
    expected = reference.interpolate(points)
    actual = compiled.interpolate(points)
    np.testing.assert_allclose(actual[0],expected[0],rtol=2e-14,atol=2e-10)
    np.testing.assert_allclose(actual[1],expected[1],rtol=2e-13,atol=2e-7)


def test_unavailable_compiler_uses_numpy_without_changing_physics(monkeypatch):
    from temsim.physics import axis_field_interpolation as module
    raw = nodal_fixture()
    points = np.array([[1e-8,0,.01],[.001,.0001,.006]])
    reference = AxisRegularPotential(raw,bore_radius_m=.002,fraction=.01,compiled=False)
    candidate = AxisRegularPotential(raw,bore_radius_m=.002,fraction=.01)
    monkeypatch.setattr(module,"compiled_evaluate",None)
    for a,b in zip(candidate.interpolate(points),reference.interpolate(points)):
        np.testing.assert_array_equal(a,b)
