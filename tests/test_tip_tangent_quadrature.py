"""Numerical phase-space refinement must not become a narrowed/tilted source."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.special import ndtr

from temsim.optics.electron_gun.tip_sampling import stratified_tangent_momenta, tangent_cell_ids
from temsim.optics.electron_gun.tip_surface import emit_surface,load_tip_surface_reference,TipSurfaceModel


@pytest.mark.parametrize("centre",[0.,-1.5])
@pytest.mark.parametrize("halfwidth",[.1,.2])
@pytest.mark.parametrize("allocation",[(),(1,1,1,1,32,1,1,1,1)])
def test_every_box_retains_its_exact_gaussian_probability(centre,halfwidth,allocation):
    n = 117
    q,a,b,w = stratified_tangent_momenta(n,sigma_sqrt_ev=.3,radial_centre_sigma=centre,
                                       halfwidth_sigma=halfwidth,allocation=allocation)
    assert np.all(w > 0) and np.all((q>0)&(q<1))
    assert w.sum() == pytest.approx(1.,abs=2e-15)
    p1 = np.diff(np.r_[0.,ndtr(centre+np.array([-halfwidth,halfwidth])),1.])
    p2 = np.diff(np.r_[0.,ndtr(np.array([-halfwidth,halfwidth])),1.])
    for cell in range(9):
        assert w[tangent_cell_ids(n,allocation) == cell].sum() == pytest.approx((p1[:,None]*p2)[divmod(cell,3)])


def test_tangent_allocation_is_only_a_budget_and_survives_roundtrip():
    import json
    original = load_tip_surface_reference()
    allocation = (1,1,1,1,32,1,1,1,1)
    model = replace(original,emission=replace(original.emission,directions_per_position=72,
        spatial_sampling='apex_stratified_v1',angular_sampling='tangent_stratified_v2',
        angular_stratum_allocation=allocation))
    assert TipSurfaceModel.from_dict(json.loads(json.dumps(model.to_dict()))) == model
    assert model.current_na == original.current_na and model.geometry == original.geometry
    p,d,e,w = emit_surface(model,1296)
    assert w.sum() == pytest.approx(1.,abs=2e-15)
    assert np.all(w > 0) and np.all(e > 0)
    assert set(tangent_cell_ids(9,allocation)) == set(range(9))
    assert np.sum(tangent_cell_ids(72,allocation) == 4) > 50
    for invalid in ((1,)*8, (1,)*8+(0,), (1,)*8+(True,)):
        with pytest.raises(ValueError,match='allocation'):
            tangent_cell_ids(72,invalid)
    with pytest.raises(ValueError,match='allocation'):
        replace(model,emission=replace(model.emission,angular_sampling='uniform_cdf')).validate()
    # Numerical priorities cannot alter the physical field, but MUST invalidate
    # the executed particle cache even if total current is unchanged.
    from temsim.optics.column import default_state
    from temsim.physics.grounded_tip_field import field_request
    state = default_state()
    gun = state.electron_gun
    before_field,before_key = field_request(gun),gun._cache_key(1297)
    gun.emitter.surface_model = model
    assert field_request(gun) == before_field
    assert gun._cache_key(1297) != before_key


@pytest.mark.parametrize("centre",[0.,-1.5])
@pytest.mark.parametrize("allocation",[(),(1,1,1,1,32,1,1,1,1)])
def test_moved_refinement_box_does_not_change_zero_mean_energy_or_covariance(centre,allocation):
    # Equal minimum TAIL resolution for this whole-source moment diagnostic.
    # Allocating more central samples must not be sold as improved tail accuracy.
    fine_count = 20000*(sum(allocation) if allocation else 9)
    qc,_,_,wc = stratified_tangent_momenta(fine_count//2,sigma_sqrt_ev=.3,radial_centre_sigma=centre,allocation=allocation)
    coarse_normal_error = abs(wc@(-.2*np.log1p(-qc))-.2)
    q,a,b,w = stratified_tangent_momenta(fine_count,sigma_sqrt_ev=.3,radial_centre_sigma=centre,allocation=allocation)
    assert abs(w@a) < .0005 and abs(w@b) < .0005
    assert w@(a*a) == pytest.approx(.09,abs=.0002)
    assert w@(b*b) == pytest.approx(.09,abs=.0002)
    assert abs(w@(a*b)) < .0003
    normal = -.2*np.log1p(-q)
    assert w@normal == pytest.approx(.2,abs=.0002)
    assert abs(w@normal-.2) < coarse_normal_error
    assert abs(w@(normal*a)) < .0003


def test_surface_refinement_preserves_the_full_cap_current_and_outgoing_emission():
    original = load_tip_surface_reference()
    model = replace(original,emission=replace(original.emission,directions_per_position=72,
        spatial_sampling="apex_stratified_v1",angular_sampling="tangent_stratified_v1",angular_refinement_gain=80.))
    p,d,e,w = emit_surface(model,1296)
    assert np.unique(p,axis=0).shape[0] == 18
    assert w.sum() == pytest.approx(1.,abs=2e-15)
    assert np.all(w > 0) and np.all(e > 0) and np.isfinite(d).all()
    np.testing.assert_allclose(np.linalg.norm(d,axis=1),1.,rtol=1e-14)
    r = model.geometry.apex_radius_nm*1e-9
    normal = (p+np.array([0,0,r]))/r
    assert np.all(np.einsum('ij,ij->i',normal,d) > 0)
    u = -p[:,2]/(r*(1-np.cos(np.deg2rad(model.emission.cap_half_angle_deg))))
    edges = np.r_[0.,np.logspace(-8,0,9)]
    for lo,hi in zip(edges[:-1],edges[1:]):
        assert w[(u>=lo)&(u<hi)].sum() == pytest.approx(hi-lo,rel=1e-12)
    assert model.current_na == original.current_na
    assert model.geometry == original.geometry
    assert TipSurfaceModel.from_dict(model.to_dict()) == model


def test_legacy_defaults_are_unchanged_and_unsupported_refinement_is_rejected():
    model = load_tip_surface_reference()
    assert not any(key.startswith('angular_') for key in model.to_dict()['emission'])
    assert TipSurfaceModel.from_dict(model.to_dict()) == model
    for changes in ({'directions_per_position':8},{'maximum_angle_deg':80},
                    {'energy_distribution':'gamma'},{'tangential_mean_energy_ev':0}):
        with pytest.raises(ValueError,match='Tangent quadrature'):
            replace(model,emission=replace(model.emission,angular_sampling='tangent_stratified_v1',
                                          **changes)).validate()


def test_numerical_refinement_invalidates_trajectory_not_field_identity():
    from temsim.optics.column import default_state
    from temsim.physics.grounded_tip_field import field_request
    from temsim.instrument_snapshot import capture_instrument_snapshot
    state = default_state()
    gun = state.electron_gun
    old_field,old_key = field_request(gun),gun._cache_key(1297)
    model = gun.emitter.surface_model
    gun.emitter.surface_model = replace(model,emission=replace(model.emission,directions_per_position=72,
        angular_sampling='tangent_stratified_v1',angular_refinement_gain=80.))
    assert gun._cache_key(1297) != old_key
    assert field_request(gun) == old_field
    restored = capture_instrument_snapshot(state).restore()
    assert restored.electron_gun._cache_key(1297) == gun._cache_key(1297)


def test_unresolvable_bins_are_explicit_not_silently_clipped():
    with pytest.raises(ValueError,match='unresolved'):
        stratified_tangent_momenta(9,sigma_sqrt_ev=1.,radial_centre_sigma=-100.)


def test_v2_refines_local_energy_across_sites_and_v1_is_still_replayable():
    model = load_tip_surface_reference()
    model = replace(model,emission=replace(model.emission,directions_per_position=72,
        spatial_sampling='apex_stratified_v1',angular_sampling='tangent_stratified_v1'))
    p,d,e,w = emit_surface(model,1296)
    new = replace(model,emission=replace(model.emission,angular_sampling='tangent_stratified_v2'))
    pp,dd,ee,ww = emit_surface(new,1296)
    np.testing.assert_array_equal(pp,p)
    np.testing.assert_array_equal(ww,w)
    assert np.unique(ee).size > np.unique(e).size
    assert TipSurfaceModel.from_dict(new.to_dict()) == new
    p,d,e,w = emit_surface(new,90000)
    radius = new.geometry.apex_radius_nm*1e-9
    normals = (p+[0,0,radius])/radius
    normal_energy = e*np.einsum('ij,ij->i',d,normals)**2
    assert w@normal_energy == pytest.approx(.2,abs=.002)
    assert w@(e-normal_energy) == pytest.approx(.1,abs=.002)
