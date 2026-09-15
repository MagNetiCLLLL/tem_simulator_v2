"""Independent analytic and boundary checks for the conforming gun field."""
import numpy as np
import pytest

from temsim.physics.axisymmetric_cut_field import AxisymmetricCutField


def rectangular_field(count, analytic):
    r = np.linspace(0, 1, count)**1.2
    z = np.linspace(-1, 1, count)
    rr, zz = np.meshgrid(r,z,indexing="ij")
    fixed = np.zeros_like(rr,dtype=bool)
    fixed[-1,:] = fixed[:,0] = fixed[:,-1] = True
    return AxisymmetricCutField(r,z,fixed,analytic(rr,zz))


def test_axial_linear_potential_and_field():
    field = rectangular_field(18,lambda r,z: 4+3*z)
    p = np.array([[.11,.23,.17],[0,0,-.63],[.54,0,.41]])
    potential, electric = field.interpolate(p)
    np.testing.assert_allclose(potential,4+3*p[:,2],atol=2e-12)
    np.testing.assert_allclose(electric,np.tile([0,0,-3.],(3,1)),atol=3e-11)


def test_harmonic_quadratic_mesh_convergence():
    # Laplacian(r**2-2*z**2)=0 in axisymmetric three-dimensional space.
    p = np.array([[.17,.23,.37],[.28,0,-.61],[0,0,.24]])
    exact = p[:,0]**2+p[:,1]**2-2*p[:,2]**2
    errors = []
    for count in (17,33,65):
        field = rectangular_field(count,lambda r,z:r*r-2*z*z)
        potential, electric = field.interpolate(p)
        # Measure the supremum at each grid's axial cell midpoints. A fixed
        # handful of points can accidentally approach nodes under refinement.
        midpoints = .5*(field.z[:-1]+field.z[1:])
        probes = np.column_stack((np.zeros(len(midpoints)),np.zeros(len(midpoints)),midpoints))
        errors.append(np.max(np.abs(field.interpolate(probes)[0]+2*midpoints**2)))
        assert np.max(np.abs(potential-exact)) <= .5*np.max(np.diff(field.z))**2+1e-11
        delta = 1e-7
        for axis in range(3):
            offset = np.eye(3)[axis]*delta
            derivative = (field.interpolate(p+offset)[0]-field.interpolate(p-offset)[0])/(2*delta)
            np.testing.assert_allclose(-derivative,electric[:,axis],atol=2e-8)
    assert errors[1] < errors[0]/3
    assert errors[2] < errors[1]/3


def test_rejects_invalid_grid_and_outside_queries():
    field = rectangular_field(5,lambda r,z:z)
    with pytest.raises(ValueError,match="outside"):
        field.interpolate([[2,0,0]])
    with pytest.raises(ValueError,match="finite"):
        field.interpolate([[0,0,np.nan]])


def test_actual_tip_boundary_is_equipotential_and_nearly_normal():
    from temsim.optics.column import default_state
    gun = default_state().electron_gun
    field = gun.electric_field
    geometry = gun.emitter.surface_model.geometry
    radius = geometry.apex_radius_nm*1e-9
    angles = np.deg2rad(np.array([1.,3.,5.,7.,9.]))
    points = np.column_stack((radius*np.sin(angles),np.zeros(5),radius*(np.cos(angles)-1)))
    represented, shift = field.surface_mesh_positions(points)
    assert shift < .02e-9
    potential, electric = field._interpolate(represented)
    np.testing.assert_allclose(potential,0.,atol=2e-10)
    normals = np.column_stack((np.sin(angles),np.zeros(5),np.cos(angles)))
    tangent = np.linalg.norm(np.cross(electric,normals),axis=1)/np.linalg.norm(electric,axis=1)
    assert np.max(tangent) < .012
    assert np.all(electric[:,2] < 0)
