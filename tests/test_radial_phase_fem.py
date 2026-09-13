"""Independent exact-wave checks for radial phase-factor finite elements."""
import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from temsim.physics.quadratic_axisymmetric_fem import mesh, volume
from temsim.physics.radial_phase_fem import radial_phase_terms, interface_gram


def plane_error(count, gamma):
    radius = np.linspace(0., 1., count)
    points, triangles, edges, z = mesh(radius, np.zeros(count), 1., count)
    stiffness, mass, _ = volume(points, triangles, np.zeros(len(points)))
    linear, square = radial_phase_terms(points, triangles)
    kz = .7
    operator = stiffness+1j*gamma*linear+gamma**2*square-kz**2*mass
    assert abs(operator-operator.conj().T).max() < 1e-12
    exact = np.exp(1j*(kz*points[:, 1]-.5*gamma*points[:, 0]**2))
    boundary = np.unique(np.r_[edges["source"].ravel(), edges["top"].ravel(), edges["side"].ravel()])
    interior = np.setdiff1d(np.arange(len(points)), boundary)
    solved = exact.copy()
    solved[interior] = spsolve(operator[interior][:, interior], -operator[interior][:, boundary]@exact[boundary])
    difference = solved-exact
    return float(np.sqrt(np.vdot(difference, mass@difference).real/np.vdot(exact, mass@exact).real))


@pytest.mark.parametrize("gamma", (2., 20.))
def test_radial_phase_coordinates_converge_to_the_same_physical_plane_wave(gamma):
    errors = [plane_error(count, gamma) for count in (17, 33, 65, 129)]
    assert all(fine < coarse/3 for coarse, fine in zip(errors, errors[1:]))
    assert errors[-1] < 5e-4


def test_factored_interface_does_not_interpolate_the_fast_carrier():
    from temsim.physics.radial_gun_wave import basis_values
    count = 257
    radius = np.linspace(0., 70., count)
    points, triangles, edges, z = mesh(radius, np.zeros(count), 2., 3)
    top = np.unique(edges["top"])
    direct = basis_values(points[top, 0], 3.5, .01, 30., 16)
    factored = basis_values(points[top, 0], 3.5, 0., 30., 16)
    error_direct = np.linalg.norm(interface_gram(points, edges["top"], direct)-np.eye(16), 2)
    error_factored = np.linalg.norm(interface_gram(points, edges["top"], factored)-np.eye(16), 2)
    assert error_factored < error_direct/100
    assert error_factored < .005


def test_missing_joined_face_is_rejected_without_rescaling_the_source():
    from dataclasses import replace
    from temsim.optics.column import default_state
    from temsim.optics.electron_gun.tip_surface import load_tip_surface_reference, SurfaceCoherence
    from temsim.physics.surface_wave import SurfaceWaveNumerics
    from temsim.physics.radial_gun_wave import RadialGunNumerics
    from temsim.physics.surface_gun_wave import validate_joint_radial_domain
    gun = default_state().electron_gun
    gun.emitter.surface_model = replace(load_tip_surface_reference(), coherence=SurfaceCoherence())
    source = gun.emitter.surface_model
    with pytest.raises(ValueError, match="outside the near-field face"):
        validate_joint_radial_domain(gun, SurfaceWaveNumerics(outer_radius_factor=1.5), RadialGunNumerics())
    assert validate_joint_radial_domain(gun, SurfaceWaveNumerics(outer_radius_factor=4.), RadialGunNumerics()) < 1e-8
    assert gun.emitter.surface_model == source
    with pytest.raises(ValueError, match="numerical Laguerre coordinate"):
        RadialGunNumerics(coordinate_width_over_cap_radius=float("nan")).validate()
