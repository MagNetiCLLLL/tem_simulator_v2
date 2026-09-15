"""Geometry reloads must not replace independent source or readout controls."""
from dataclasses import replace

import pytest

from temsim import module_manifest
from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.optics.electron_gun.field_emission import FieldEmissionGun
from temsim.optics.electron_gun.tip_assembly import apply_tip_part, derived_envelope, model_from_part


def changed_tip(**updates):
    part = dict(module_manifest.part_data("gun/FEG.toml", "feg_tip"))
    part.update(updates)
    part.update(derived_envelope(model_from_part(part)))
    return part


def test_tip_radius_reload_retains_prescribed_current_energy_cap_and_sampling():
    gun = FieldEmissionGun()
    original = model_from_part(changed_tip())
    emission = replace(original.emission, current_na=50., flux_electrons_per_nm2_s=None,
        normal_mean_energy_ev=.6, cap_half_angle_deg=5., directions_per_position=12,
        angular_sampling="tangent_stratified_v2", angular_refinement_gain=80.)
    numerics = replace(original.field_numerics, radial_nodes=640, axial_nodes=1280)
    gun.emitter.surface_model = replace(original, emission=emission, field_numerics=numerics)
    key = gun._cache_key(193)
    apply_tip_part(gun.emitter, changed_tip(tip_radius_nm=120.))
    model = gun.emitter.surface_model
    assert model.geometry.apex_radius_nm == 120.
    assert model.emission == emission
    assert model.field_numerics == numerics
    assert model.current_na == 50.
    assert gun._cache_key(193) != key
    # Applying the same geometry twice must be idempotent.
    apply_tip_part(gun.emitter, changed_tip(tip_radius_nm=120.))
    assert gun.emitter.surface_model == model


def test_geometry_reload_preserves_flux_density_not_derived_total_current():
    gun = FieldEmissionGun()
    original = model_from_part(changed_tip())
    emission = replace(original.emission, flux_electrons_per_nm2_s=2e8)
    gun.emitter.surface_model = replace(original, emission=emission)
    before = gun.emitted_current_a
    apply_tip_part(gun.emitter, changed_tip(tip_radius_nm=120.))
    assert gun.emitter.surface_model.emission == emission
    assert gun.emitted_current_a / before == pytest.approx(1.2**2)


def test_changed_tip_defaults_update_only_values_without_operating_overrides():
    gun = FieldEmissionGun()
    original = model_from_part(changed_tip())
    gun.emitter.surface_model = replace(original, emission=replace(
        original.emission, normal_mean_energy_ev=.6, cap_half_angle_deg=5.))
    apply_tip_part(gun.emitter, changed_tip(tip_radius_nm=120.,
        emission_normal_mean_energy_ev=.4, emission_cap_half_angle_deg=15.,
        emission_maximum_angle_deg=35., tip_field_radial_nodes=256))
    model = gun.emitter.surface_model
    assert model.emission.normal_mean_energy_ev == .6
    assert model.emission.cap_half_angle_deg == 5.
    assert model.emission.maximum_angle_deg == 35.
    assert model.field_numerics.radial_nodes == 256


@pytest.mark.parametrize("inserted", [False, True])
def test_same_assembly_reload_preserves_aperture_and_recording_controls(inserted):
    state = default_state()
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    catalog.apply(state, selection)
    for plane in state.recording_planes:
        if plane.key in {"camera", "flu_screen"}:
            plane.inserted = inserted
    state.condenser_aperture_3.radius_mm = .0123
    catalog.apply(state, selection, preserve_operating_parameters=True)
    assert state.condenser_aperture_3.radius_mm == .0123
    assert all(plane.inserted is inserted for plane in state.recording_planes
               if plane.key in {"camera", "flu_screen"})
    # Explicit assembly installation still owns its documented defaults.
    catalog.apply(state, selection, preserve_operating_parameters=False)
    assert state.condenser_aperture_3.radius_mm == state.condenser_aperture_3.maximum_radius_mm
    assert all(plane.inserted is (not state.energy_filter_installed)
               for plane in state.recording_planes if plane.key in {"camera", "flu_screen"})
