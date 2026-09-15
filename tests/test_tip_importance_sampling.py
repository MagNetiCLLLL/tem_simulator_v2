from dataclasses import replace

import numpy as np
import pytest

from temsim.optics.electron_gun.tip_sampling import stratified_cap_area
from temsim.optics.electron_gun.tip_surface import load_tip_surface_reference, emit_surface, TipSurfaceModel


def test_strata_preserve_full_area_current_including_outer_cap():
    for count in (9, 24, 96):
        u, phi, w = stratified_cap_area(count)
        assert w.sum() == pytest.approx(1., abs=1e-14)
        assert np.all(w > 0) and np.all(u > 0) and np.all(u < 1)
        edges = np.r_[0., np.logspace(-8, 0, 9)]
        for lo, hi in zip(edges[:-1], edges[1:]):
            assert w[(u >= lo) & (u < hi)].sum() == pytest.approx(hi-lo)


def test_adaptive_budget_preserves_each_full_area_probability():
    allocation = (1, 1, 1, 1, 1, 4, 24, 24, 1)
    edges = np.r_[0., np.logspace(-8, 0, 9)]
    for count in (9, 18, 72, 144):
        u, phi, w = stratified_cap_area(count, allocation)
        assert len(u) == count and len(phi) == count and np.all(w > 0)
        assert w.sum() == pytest.approx(1.)
        for lo, hi in zip(edges[:-1], edges[1:]):
            assert w[(u >= lo) & (u < hi)].sum() == pytest.approx(hi-lo)
        if count == 144:
            assert np.sum((u >= .001) & (u < .1)) > 100
    # Refinement must also converge for a diagnostic dominated by the outer
    # cap, not only for a narrow downstream aperture's accepted rays.
    uc, _, wc = stratified_cap_area(18000, allocation)
    u, phi, w = stratified_cap_area(72000, allocation)
    assert w@u == pytest.approx(.5, abs=.001)
    assert abs(w@u-.5) < abs(wc@uc-.5)
    assert abs(w@np.exp(2j*np.pi*phi)) < .01


def test_cap_allocation_roundtrip_and_invalid_omission():
    model = load_tip_surface_reference()
    assert "spatial_stratum_allocation" not in model.to_dict()["emission"]
    for allocation in ((1,)*8, (1,)*8+(0,), (1,)*8+(True,)):
        with pytest.raises(ValueError, match="allocation"):
            stratified_cap_area(144, allocation)
    model = replace(model, emission=replace(model.emission, spatial_sampling="apex_stratified_v1",
                    spatial_stratum_allocation=(1, 1, 1, 1, 1, 4, 24, 24, 1), directions_per_position=72))
    import json
    assert TipSurfaceModel.from_dict(json.loads(json.dumps(model.to_dict()))) == model
    p, d, e, w = emit_surface(model, 10368)
    assert w.sum() == pytest.approx(1.)
    assert model.current_na == load_tip_surface_reference().current_na


def test_weighted_area_moment_converges_to_uniform_law_not_apex_source():
    u, phi, w = stratified_cap_area(9000)
    assert w@u == pytest.approx(.5, abs=.001)
    assert w@(u*u) == pytest.approx(1/3, abs=.001)
    assert np.mean(u) < .1  # Importance density is not the physical density.
    assert abs(w@np.exp(2j*np.pi*phi)) < .005
    # Each annulus, not just the total cap, must cover every azimuth sector.
    for i in range(9):
        assert np.ptp(phi[i::9]) > .99


def test_source_roundtrip_and_multiple_directions_keep_physical_inputs():
    model = load_tip_surface_reference()
    model = replace(model, emission=replace(model.emission,
        directions_per_position=8, spatial_sampling="apex_stratified_v1"))
    assert TipSurfaceModel.from_dict(model.to_dict()) == model
    p, d, energy, w = emit_surface(model, 769)
    unique, inverse, sizes = np.unique(p, axis=0, return_inverse=True, return_counts=True)
    assert len(unique) == 96 and sizes.min() >= 8
    assert np.allclose(np.linalg.norm(d, axis=1), 1.) and np.all(energy > 0)
    assert w.sum() == pytest.approx(1.)
    assert model.current_na == pytest.approx(load_tip_surface_reference().current_na)
    for site in range(len(unique)):
        assert np.unique(d[inverse == site], axis=0).shape[0] >= 8


def test_uniform_default_remains_historically_serialisable():
    model = load_tip_surface_reference()
    assert "spatial_sampling" not in model.to_dict()["emission"]
    assert TipSurfaceModel.from_dict(model.to_dict()).emission.spatial_sampling == "uniform_area"
    with pytest.raises(ValueError, match="nine"):
        stratified_cap_area(8)
    with pytest.raises(ValueError, match="quadrature"):
        replace(model, emission=replace(model.emission, spatial_sampling="invented")).validate()


def test_sampling_invalidates_gun_cache_without_changing_field_or_physical_source():
    from temsim.optics.column import default_state
    from temsim.physics.grounded_tip_field import field_request
    from temsim.instrument_snapshot import capture_instrument_snapshot
    state = default_state()
    gun = state.electron_gun
    before = gun._cache_key(193)
    field = field_request(gun)
    model = gun.emitter.surface_model
    gun.emitter.surface_model = replace(model,
        emission=replace(model.emission, spatial_sampling="apex_stratified_v1"))
    assert gun._cache_key(193) != before
    assert field_request(gun) == field
    restored = capture_instrument_snapshot(state).restore()
    assert restored.electron_gun._cache_key(193) == gun._cache_key(193)
    assert restored.electron_gun.emitter.surface_model.current_na == model.current_na
