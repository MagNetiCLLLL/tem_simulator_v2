"""Classical surface quadrature and launch lineage, not coherent acceptance."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.electron_gun.tip_surface import (
    emit_surface, surface_bundle, load_tip_surface_reference, TipSurfaceModel,
)
from temsim.optics.electron_gun.field_emission import FieldEmissionGun
from temsim import module_manifest
from temsim.optics.electron_gun.tip_assembly import model_from_part
from temsim.physics.ray_identity import emission_reference, emission_colour_values, source_identity


@pytest.mark.parametrize("count", [9, 49, 193, 1000, 15000])
def test_each_surface_site_has_multiple_local_directions_without_multiplying_flux(count):
    model = load_tip_surface_reference()
    positions, directions, energy, weights = emit_surface(model, count)
    sites, inverse, sizes = np.unique(positions, axis=0, return_inverse=True, return_counts=True)
    assert positions.shape == directions.shape == (count, 3)
    assert sizes.min() >= min(model.emission.directions_per_position, count)
    assert sizes.max()-sizes.min() <= 1
    assert np.all(weights > 0) and weights.sum() == pytest.approx(1., abs=1e-14)
    np.testing.assert_allclose(np.bincount(inverse, weights=weights), 1/len(sites), atol=1e-14)
    for site in range(min(len(sites), 30)):
        selected = inverse == site
        assert len(np.unique(directions[selected], axis=0)) == selected.sum()
        assert np.ptp(energy[selected]) > 0
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1., atol=1e-14)
    assert model.current_na == pytest.approx(100.)


def test_zero_spread_is_not_artificially_broadened_for_display():
    model = load_tip_surface_reference()
    model = replace(model, emission=replace(model.emission, maximum_angle_deg=0))
    position, direction, _, _ = emit_surface(model, 49)
    radius = model.geometry.apex_radius_nm*1e-9
    np.testing.assert_allclose(direction, (position+[0, 0, radius])/radius, atol=1e-14)


@pytest.mark.parametrize("value", [0, 257, 2.5, True, float("nan")])
def test_invalid_direction_budget_rejected(value):
    model = load_tip_surface_reference()
    with pytest.raises(ValueError, match="Directions per emission"):
        replace(model, emission=replace(model.emission, directions_per_position=value)).validate()


def test_old_sampling_is_explicit_and_new_budget_invalidates_transport_not_field():
    from temsim.physics.grounded_tip_field import field_request
    gun = FieldEmissionGun()
    gun.emitter.surface_model = model_from_part(module_manifest.part_data("gun/FEG.toml", "feg_tip"))
    payload = gun.emitter.surface_model.to_dict()
    payload["emission"].pop("directions_per_position")
    old = TipSurfaceModel.from_dict(payload)
    assert old.emission.directions_per_position == 1
    assert len(np.unique(emit_surface(old, 49)[0], axis=0)) == 49
    before_key, before_field = gun._cache_key(49), field_request(gun)
    gun.emitter.surface_model = replace(gun.emitter.surface_model,
        emission=replace(gun.emitter.surface_model.emission, directions_per_position=4))
    assert gun._cache_key(49) != before_key
    assert field_request(gun) == before_field


@pytest.mark.parametrize("count,probes", [(49, 1), (193, 33)])
def test_zero_current_support_probes_do_not_replace_partial_emission_sites(count, probes):
    model = load_tip_surface_reference()
    expected = emit_surface(model, count-probes)
    bundle = surface_bundle(model, count, support_probes=probes)
    for actual, value in zip((bundle.surface_position_m, bundle.surface_direction,
                             bundle.surface_energy_ev, bundle.weight), expected):
        np.testing.assert_array_equal(actual[:-probes], value)
    assert np.all(bundle.weight[-probes:] == 0)
    assert bundle.weight.sum() == pytest.approx(1)


def test_launch_lineage_uses_surface_not_resampled_plane_and_full_hemisphere():
    model = load_tip_surface_reference()
    bundle = surface_bundle(model, 49)
    reference = emission_reference(bundle, model)
    incident = SimpleNamespace(x=np.full((2, 49), 1.), y=np.full((2, 49), -2.))
    trace = SimpleNamespace(exit_bundle=SimpleNamespace(ray_id=bundle.ray_id), emission_reference=reference)
    ids, angles = source_identity(incident, trace)
    expected = np.mod(np.arctan2(bundle.y_m, bundle.x_m), 2*np.pi)
    np.testing.assert_allclose(angles, expected, atol=1e-14)
    simulation = SimpleNamespace(gun_trace=trace)
    selected = ids[[5, 1, 5, 0]]  # reordered, repeated scattered ancestors
    for mode in ("emission_direction", "emission_angle"):
        values = emission_colour_values(simulation, ids, mode)
        np.testing.assert_array_equal(emission_colour_values(simulation, selected, mode), values[[5, 1, 5, 0]])
        assert np.isnan(emission_colour_values(simulation, [-1, 999], mode)).all()
        assert np.isnan(emission_colour_values(SimpleNamespace(), ids, mode)).all()
    assert not reference["direction"].flags.writeable
    # Slopes alone invert azimuth for dz<0. The full original direction must win.
    trace.emission_reference = {"ray_id": np.array([12]), "direction": np.array([[.8, 0, -.6]]),
                                "normal": np.array([[1., 0, 0]])}
    assert emission_colour_values(simulation, [12], "emission_direction")[0] == 0.
    assert emission_colour_values(simulation, [12], "emission_angle")[0] == pytest.approx(np.arccos(.8))


def test_direction_budget_editor_is_draft_only(qtbot):
    from temsim.gui.gun_source_dialog import GunSourceDialog
    gun = FieldEmissionGun()
    gun.emitter.surface_model = model_from_part(module_manifest.part_data("gun/FEG.toml", "feg_tip"))
    dialog = GunSourceDialog(gun)
    qtbot.addWidget(dialog)
    dialog.directions_per_position.setValue(12)
    dialog.accept()
    assert dialog.value()["surface_model"].emission.directions_per_position == 12
    assert gun.emitter.surface_model.emission.directions_per_position == 8
