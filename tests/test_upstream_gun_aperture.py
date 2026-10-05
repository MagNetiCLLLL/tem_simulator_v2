"""Gun-aperture transport and prescribed vacuum boundaries without field solves."""

from dataclasses import replace

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.optics.electron_gun import tracing
from temsim.physics.radial_gun_wave import _mask_events
from temsim.physics.relativistic_lorentz import (
    momentum_from_kinetic_energy_ev,
    velocity_from_momentum_m_per_s,
)
from temsim.physics.tip_gun_wave import GunWaveNumerics, _axial_grid
from temsim.vacuum import (
    VacuumMap, bind_gun_environment, boundary_anchors,
    ensure_standalone_gun_environment, resolve_regions,
)


@pytest.fixture(params=("FEG", "FEG + Mono"))
def state(request):
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, replace(catalog.default_selection(), gun=request.param))
    return state


def test_gun_aperture_and_vacuum_boundary_follow_the_live_upstream_plane(state):
    gun = state.electron_gun
    aperture = gun.dpa_aperture
    gun_lens_end = gun.electrostatic_lens.mechanical_center_from_tip_mm + gun.electrostatic_lens.mechanical_length_mm / 2
    accelerator_start = gun.accelerator.mechanical_center_from_tip_mm - gun.accelerator.mechanical_length_mm / 2
    assert gun_lens_end < aperture.z_mm == 24.0 < accelerator_start
    pressure = [region.medium.pressure_mbar for region in state.vacuum_map.regions]

    for offset in (0.0, 0.25):
        aperture.field_center_offset_mm = offset
        anchors = boundary_anchors(state)
        assert anchors["gun_vacuum_boundary"] == aperture.z_mm
        assert anchors["gun_acceleration_start"] == accelerator_start
        rows = {row.key: row for row in resolve_regions(state, include_disabled=True)}
        assert rows["gun_tip"].end_z_mm == aperture.z_mm
        assert rows["gun_accelerator"].start_z_mm == aperture.z_mm
        assert rows["gun_accelerator"].end_z_mm == gun.exit_plane_z_mm

    # Hole size is an interception setting, not a pump/conductance calculation.
    aperture.radius_mm *= 0.5
    assert [region.medium.pressure_mbar for region in state.vacuum_map.regions] == pressure


def test_explicit_saved_accelerator_boundary_remains_unchanged(state):
    for region in state.vacuum_map.regions:
        if region.key == "gun_tip":
            region.end_anchor = "gun_acceleration_start"
        elif region.key == "gun_accelerator":
            region.start_anchor = "gun_acceleration_start"
    state.vacuum_map = VacuumMap.from_dict(state.vacuum_map.to_dict())
    rows = {row.key: row for row in resolve_regions(state, include_disabled=True)}
    assert rows["gun_tip"].end_z_mm == boundary_anchors(state)["gun_acceleration_start"]
    assert rows["gun_accelerator"].start_z_mm != state.electron_gun.dpa_aperture.z_mm


def test_thermionic_vacuum_boundary_keeps_existing_accelerator_contract():
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, replace(catalog.default_selection(), gun="Thermionic"))
    anchors = boundary_anchors(state)
    rows = {row.key: row for row in resolve_regions(state, include_disabled=True)}
    assert anchors["gun_vacuum_boundary"] == anchors["gun_acceleration_start"]
    assert rows["gun_tip"].end_z_mm == anchors["gun_acceleration_start"]
    assert rows["gun_accelerator"].start_z_mm == anchors["gun_acceleration_start"]


@pytest.mark.parametrize("gun_name", ["FEG", "FEG + Mono", "Thermionic"])
@pytest.mark.parametrize("legacy_boundary", [False, True])
def test_standalone_enabled_gun_map_keeps_both_pressure_regions(monkeypatch, gun_name, legacy_boundary):
    from temsim.optics.electron_gun import FieldEmissionGun, ThermionicGun

    gun = ThermionicGun() if gun_name == "Thermionic" else FieldEmissionGun()
    if gun_name == "FEG + Mono":
        gun.monochromator.installed = True
        gun.apply_manifest_geometry()
    config = VacuumMap.load()
    config.enabled = True
    if legacy_boundary:
        config.regions[0].end_anchor = "gun_acceleration_start"
        config.regions[1].start_anchor = "gun_acceleration_start"
    monkeypatch.setattr(VacuumMap, "load", classmethod(lambda cls: config))

    ensure_standalone_gun_environment(gun)

    rows = {row.key: row for row in gun._vacuum_regions}
    expected_boundary = (
        gun.accelerator.mechanical_center_from_tip_mm - gun.accelerator.mechanical_length_mm / 2
        if legacy_boundary or gun_name == "Thermionic" else gun.dpa_aperture.z_mm
    )
    assert set(rows) == {"gun_tip", "gun_accelerator"}
    assert rows["gun_tip"].start_z_mm <= 0.0
    assert rows["gun_tip"].end_z_mm == expected_boundary
    assert rows["gun_accelerator"].start_z_mm == expected_boundary
    assert rows["gun_accelerator"].end_z_mm == gun.exit_plane_z_mm
    assert rows["gun_tip"].medium.pressure_mbar == 3e-11
    assert rows["gun_accelerator"].medium.pressure_mbar == 2e-8


@pytest.mark.parametrize("gun_name", ["FEG", "FEG + Mono", "Thermionic"])
@pytest.mark.parametrize("enabled", [False, True])
def test_standalone_entry_preserves_already_bound_map(monkeypatch, gun_name, enabled):
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, replace(catalog.default_selection(), gun=gun_name))
    state.vacuum_map.enabled = enabled
    state.vacuum_map.seed = 319
    state.vacuum_map.regions[0].medium.pressure_mbar = 1e-10
    regions = bind_gun_environment(state)

    def unexpected_reload(cls):
        pytest.fail("A bound gun environment must not reload the default map")

    monkeypatch.setattr(VacuumMap, "load", classmethod(unexpected_reload))
    ensure_standalone_gun_environment(state.electron_gun)
    assert state.electron_gun._vacuum_regions is regions
    assert state.electron_gun._vacuum_seed == 319
    if enabled:
        assert regions[0].medium.pressure_mbar == 1e-10


def test_particles_cross_and_stop_at_upstream_gun_aperture(state):
    aperture = state.electron_gun.dpa_aperture
    # Two parallel rays immediately downstream of Gun Lens isolate the event.
    # Neither ray reaches the accelerator or requires an electrostatic solve.
    x = np.array([0.5, 1.25]) * aperture.radius_mm * 1e-3
    previous = np.column_stack((x, np.zeros(2), np.full(2, 0.023)))
    current = previous.copy()
    current[:, 2] = 0.025
    momentum = momentum_from_kinetic_energy_ev(
        np.full(2, 1000.0), np.tile([0.0, 0.0, 1.0], (2, 1))
    )
    duration = 0.002 / velocity_from_momentum_m_per_s(momentum)[0, 2]
    alive, completed, passed = np.ones(2, bool), np.zeros(2, bool), np.zeros(2, bool)
    blocked_z, arrival_time, arrival_x, arrival_y = (np.full(2, np.nan) for _ in range(4))
    blocked_key = ["", ""]
    tracing._resolve_aperture_crossing(
        aperture, aperture.z_mm * 1e-3, previous, momentum, current, momentum,
        alive, completed, blocked_z, blocked_key, passed=passed,
        previous_time_s=0.0, new_time_s=duration,
        arrival_time_s=arrival_time, arrival_x_m=arrival_x, arrival_y_m=arrival_y,
    )
    assert alive.tolist() == passed.tolist() == [True, False]
    assert blocked_z[1] == aperture.z_mm
    assert blocked_key[1] == aperture.key
    np.testing.assert_allclose(arrival_time, duration * 0.5)
    np.testing.assert_allclose(arrival_x, x)


def test_coherent_mask_schedule_and_cache_include_the_upstream_plane(state):
    gun = state.electron_gun
    aperture = gun.dpa_aperture
    grid, masks = _axial_grid(gun, GunWaveNumerics(field_step_mm=5.0, bore_step_mm=5.0))
    assert aperture.z_mm in grid and aperture.z_mm in masks
    events = _mask_events(gun, aperture.z_mm * 1e6)
    assert (aperture.key, aperture.radius_mm * 1e6, aperture.interaction_kind) in events
    for previous_plane_mm in (12.0, 120.0, 170.0):
        assert aperture.key not in {row[0] for row in _mask_events(gun, previous_plane_mm * 1e6)}
    original = gun._cache_key(2)
    aperture.field_center_offset_mm = 0.25
    assert gun._cache_key(2) != original
