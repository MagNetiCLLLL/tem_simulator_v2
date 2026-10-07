"""Named source ownership and independently conserved intensity mixtures."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE, TipCoherence
from temsim.optics.electron_gun.tip_surface import SharedSurfaceCoherence, load_tip_surface_reference
from temsim.physics.coherent_inputs import source_settings_from_state
from temsim.physics.coherent_state_set import (
    common_state_identity, tip_geometry_identity, restore_tip_state,
    rebase_tip_state, combine_intensity_previews,
)
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.tip_gun_wave import TipGunCheckpoint
from temsim.physics.wave_flux import BeamState, WaveMode


@pytest.fixture
def state():
    value = default_state()
    emitter = value.electron_gun.emitter
    emitter.surface_model = None
    emitter.curvature_nm_inv = 0.
    emitter.virtual_source_fwhm_nm = 100.
    emitter.coherence = TipCoherence()
    return value


def test_common_identity_allows_only_declared_emission_values_and_preserves_live_state(state):
    original = capture_instrument_snapshot(state)
    changed = original.restore()
    emitter = changed.electron_gun.emitter
    emitter.virtual_source_fwhm_nm = 150.
    emitter.emission_energy_ev = .6
    emitter.minimum_kinetic_energy_ev = .02
    emitter.energy_spread_fwhm_ev = .2
    emitter.coherence = TipCoherence(.2, 10., 3., 5., 1., 2., .1, -.1)
    assert capture_instrument_snapshot(changed).digest != original.digest
    assert common_state_identity(changed) == common_state_identity(state)
    assert capture_instrument_snapshot(state).digest == original.digest
    restored = restore_tip_state(original, changed)
    assert source_settings_from_state(restored) == source_settings_from_state(state)
    assert restored is not state


@pytest.mark.parametrize("change", [
    lambda state: setattr(state.electron_gun.emitter, "tip_radius_nm", 110.),
    lambda state: setattr(state.electron_gun.emitter, "emission_current_na", 50.),
    lambda state: setattr(state.electron_gun.emitter, "ray_count", 42),
    lambda state: setattr(state.electron_gun.emitter, "young_decay_width_ev", .3),
    lambda state: setattr(state.lenses[0], "percent", 45.),
    lambda state: setattr(state.apertures[0], "enabled", False),
    lambda state: setattr(state.sample, "thickness_nm", 17.),
    lambda state: setattr(state, "new_physical_input", .125),
])
def test_common_identity_keeps_geometry_optics_numerics_and_unknown_values(state, change):
    original = capture_instrument_snapshot(state)
    changed = original.restore()
    change(changed)
    assert common_state_identity(changed) != common_state_identity(state)
    with pytest.raises(ValueError, match="changed"):
        restore_tip_state(original, changed)


def test_cursor_is_readout_only(state):
    before = common_state_identity(state)
    state.virtual_observation_z_mm = 1700.
    assert common_state_identity(state) == before


def test_surface_mean_width_and_derived_monoenergetic_law_are_state_inputs(state):
    model = load_tip_surface_reference()
    surface = replace(model, emission=replace(model.emission,
        flux_profile="cosine_cap", maximum_angle_deg=0.,
        energy_distribution="gamma", kinetic_mean_ev=.7, kinetic_sigma_ev=.12),
        coherence=SharedSurfaceCoherence()).validate()
    state.electron_gun.emitter.surface_model = surface
    state.electron_gun.emitter.coherence = None
    original = common_state_identity(state)
    state.electron_gun.emitter.surface_model = replace(surface,
        emission=replace(surface.emission, energy_distribution="monoenergetic",
                         kinetic_mean_ev=1.3, kinetic_sigma_ev=0.))
    assert common_state_identity(state) == original
    state.electron_gun.emitter.surface_model = replace(surface,
        geometry=replace(surface.geometry, apex_radius_nm=surface.geometry.apex_radius_nm*1.1))
    assert common_state_identity(state) != original


def test_explicit_rebase_keeps_saved_emission_and_current_optics_without_mutation(state):
    saved = capture_instrument_snapshot(state)
    state.lenses[0].percent += 1.
    state.electron_gun.emitter.virtual_source_fwhm_nm = 150.
    state.electron_gun.emitter.coherence = TipCoherence(offset_x_nm=2.)
    current = capture_instrument_snapshot(state)
    rebased = rebase_tip_state(saved, state)
    assert source_settings_from_state(rebased) == source_settings_from_state(saved.restore())
    assert rebased.lenses[0].percent == state.lenses[0].percent
    assert capture_instrument_snapshot(rebased).physical_digest != saved.physical_digest
    assert common_state_identity(rebased) == common_state_identity(state)
    assert capture_instrument_snapshot(state).digest == current.digest


def test_rebase_rejects_changed_tip_geometry_and_current(state):
    saved = capture_instrument_snapshot(state)
    state.electron_gun.emitter.tip_radius_nm += 1.
    with pytest.raises(ValueError, match="Tip geometry"):
        rebase_tip_state(saved, state)
    state.electron_gun.emitter.tip_radius_nm -= 1.
    state.electron_gun.emitter.emission_current_na += 1.
    with pytest.raises(ValueError, match="current"):
        rebase_tip_state(saved, state)


def test_particle_sample_count_does_not_change_metal_for_explicit_rebase(state):
    saved = capture_instrument_snapshot(state)
    original = tip_geometry_identity(state)
    state.electron_gun.emitter.ray_count += 37
    assert tip_geometry_identity(state) == original
    assert rebase_tip_state(saved, state).electron_gun.emitter.ray_count == state.electron_gun.emitter.ray_count


def _item(density, bounds=((0., 2.), (0., 2.)), *, z=100., current=1e-9, phase=1., weight=1.):
    density = np.asarray(density, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    # An actual independent complex member is retained, but never read or
    # added by the mixture. Phase can differ without changing probabilities.
    wave = PlaneWave(np.full((2, 2), complex(phase)/2), np.eye(2)*1e-9, np.zeros(2))
    mode = WaveMode(wave, 1., TIP_REFERENCE, "state", 300.)
    result = SimpleNamespace(checkpoint=TipGunCheckpoint(BeamState((mode,), TIP_REFERENCE), z, current, {}))
    probability = float(density.sum()*np.prod((bounds[:, 1]-bounds[:, 0])/density.shape))
    preview = SimpleNamespace(density=density, bounds_um=bounds, probability=probability,
        mode_count=1, retained_bytes=density.nbytes+bounds.nbytes,
        grid_shapes=(wave.amplitude.shape,), pair_token=None, comparison=None, comparison_error=None)
    return result, preview, weight


def _integral(preview):
    return preview.density.sum()*np.prod(np.diff(preview.bounds_um, axis=1)[:, 0]/preview.density.shape)


def test_single_state_returns_exact_existing_preview_without_rebinning():
    item = _item(((.1, .2), (.3, .4)), weight=3.)
    observation, preview = combine_intensity_previews([item])
    assert preview is item[1]
    assert observation.members == (item[0],)
    assert observation.relative_weights == (1.,)


def test_opposite_complex_phases_do_not_cancel_and_original_modes_are_retained():
    a = _item(np.full((2, 2), .25), phase=1.)
    b = _item(np.full((2, 2), .25), phase=-1., weight=3.)
    observation, preview = combine_intensity_previews([a, b])
    np.testing.assert_allclose(preview.density, .25, atol=0., rtol=0.)
    assert preview.probability == pytest.approx(1.)
    assert observation.relative_weights == (.25, .75)
    assert observation.members[0] is a[0] and observation.members[1] is b[0]
    assert not hasattr(observation.checkpoint, "beam")
    assert not preview.density.flags.writeable and not preview.bounds_um.flags.writeable


def test_distinct_nonsquare_domains_rebin_by_area_and_keep_xy_orientation():
    # A point-like displayed cell occupies x=[0,1], y=[0,1]. The second grid
    # extends the common view without putting probability in that quadrant.
    a = _item(((1., 0.), (0., 0.), (0., 0.)), ((0., 3.), (0., 2.)), weight=1.)
    b = _item(((0., 0., 0.), (0., 0., 0.)), ((0., 4.), (0., 3.)), weight=3.)
    observation, preview = combine_intensity_previews([a, b])
    assert preview.density.shape == (3, 3)
    np.testing.assert_allclose(preview.bounds_um, ((0., 4.), (0., 3.)))
    assert preview.probability == pytest.approx(.25)
    assert _integral(preview) == pytest.approx(.25)
    assert preview.density[0, 0] == pytest.approx(.25/(4/3))
    assert np.count_nonzero(preview.density) == 1


def test_different_grid_scales_conserve_weighted_integrated_probability():
    a = _item(np.arange(1, 7).reshape(2, 3)/100, ((-1., 2.), (-3., 2.)), weight=2.)
    b = _item(np.arange(1, 13).reshape(4, 3)/200, ((-5., 1.), (-2., 3.)), weight=3.)
    _, preview = combine_intensity_previews([a, b])
    expected = .4*a[1].probability+.6*b[1].probability
    assert preview.probability == pytest.approx(expected, rel=2e-14)
    assert _integral(preview) == pytest.approx(expected, rel=2e-14)


@pytest.mark.parametrize("weight", [-1., np.nan, np.inf, True, "1"])
def test_invalid_weights_are_rejected(weight):
    with pytest.raises(ValueError, match="weights"):
        combine_intensity_previews([_item(np.ones((2, 2)), weight=weight)])


def test_empty_zero_total_mixed_z_or_current_are_rejected():
    with pytest.raises(ValueError, match="at least one"):
        combine_intensity_previews([])
    with pytest.raises(ValueError, match="positive"):
        combine_intensity_previews([_item(np.ones((2, 2)), weight=0.)])
    a = _item(np.ones((2, 2)))
    with pytest.raises(ValueError, match="same Z"):
        combine_intensity_previews([a, _item(np.ones((2, 2)), z=np.nextafter(100., 101.))])
    with pytest.raises(ValueError, match="reference current"):
        combine_intensity_previews([a, _item(np.ones((2, 2)), current=2e-9)])


def test_large_weights_normalize_without_overflow():
    _, preview = combine_intensity_previews([
        _item(np.full((2, 2), .1), weight=1e308),
        _item(np.full((2, 2), .3), weight=1e308)])
    np.testing.assert_allclose(preview.density, .2)


def test_overlay_preserves_only_one_common_captured_magnetic_gauge():
    from dataclasses import replace
    from test_wave_plane_observables import _posed_gauge
    gauge = _posed_gauge()
    first, second = (_item(np.full((2, 2), .25), z=gauge.plane_z_mm) for _ in range(2))
    for row in (first, second):
        row[1].magnetic_gauge = gauge
        row[1].axial_bz_t = gauge.aligned_bz_t
    _, shared = combine_intensity_previews([first, second])
    assert shared.magnetic_gauge.fingerprint == gauge.fingerprint
    # Equal scalar fields are insufficient if a lens has moved. The mixture
    # can still display intensity but must not fabricate a shared flow gauge.
    field = gauge.fields[0]
    moved = replace(field.registration, origin_global_m=(0., 0., 1.5))
    second[1].magnetic_gauge = replace(gauge, fields=(replace(field, registration=moved),))
    _, mixed = combine_intensity_previews([first, second])
    assert mixed.magnetic_gauge is None and mixed.axial_bz_t is None
    assert mixed.probability == pytest.approx(shared.probability)
    _, excluded = combine_intensity_previews([first, (*second[:2], 0.)])
    assert excluded.magnetic_gauge.fingerprint == gauge.fingerprint
