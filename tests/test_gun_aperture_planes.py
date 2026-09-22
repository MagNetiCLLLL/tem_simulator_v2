"""A gun aperture is a physical stop, separate from the column handoff.

Straight-flight fixtures isolate event geometry. Full tip-origin gun tests
below additionally execute the existing extraction and accelerating fields.
"""
import json
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.electron_gun.aperture import GunAperture
from temsim.optics.electron_gun.field_emission import FieldEmissionGun
from temsim.optics.electron_gun.monochromator import MonochromatorSlit
from temsim.optics.electron_gun import tracing
from temsim.physics.relativistic_lorentz import (
    momentum_from_kinetic_energy_ev, velocity_from_momentum_m_per_s,
)


def aperture_at(z_mm, radius_mm=.6):
    return GunAperture('C1', 'feg_c1_aperture', z_mm, .1, 10., 6., .01,
                       radius_mm, 3.)


def flight():
    momenta = momentum_from_kinetic_energy_ev(np.full(2, 1000.),
                                            np.array([[1., 0., 1.], [.5, 0., 1.]]))
    velocity = velocity_from_momentum_m_per_s(momenta)
    duration = .002 / np.min(velocity[:, 2])
    return SimpleNamespace(
        old=np.zeros((2, 3)), new=velocity*duration, momenta=momenta,
        duration=duration, alive=np.ones(2, bool), completed=np.zeros(2, bool),
        blocked_z=np.full(2, np.nan), blocked_key=['', ''], passed=np.zeros(2, bool),
        time=np.full(2, np.nan), x=np.full(2, np.nan), y=np.full(2, np.nan),
    )


ZERO_ELECTRIC = SimpleNamespace(
    field_at_global_positions_v_per_m=lambda p: np.zeros_like(p),
    potential_v_at_global_positions=lambda p: np.zeros(len(p)),
)
ZERO_MAGNETIC = SimpleNamespace(field_at_global_positions_t=lambda p: np.zeros_like(p))


def clip_and_exit(aperture, *, curved=False, exit_z_mm=1.5):
    f = flight()
    extra = dict(electric=ZERO_ELECTRIC, magnetic=ZERO_MAGNETIC) if curved else {}
    tracing._resolve_aperture_crossing(aperture, aperture.z_mm*.001,
        f.old, f.momenta, f.new, f.momenta.copy(), f.alive, f.completed,
        f.blocked_z, f.blocked_key, passed=f.passed, previous_time_s=0.,
        new_time_s=f.duration, arrival_time_s=f.time, arrival_x_m=f.x,
        arrival_y_m=f.y, **extra)
    f.exit_position = np.full((2, 3), np.nan)
    f.exit_momentum = np.full((2, 3), np.nan)
    f.exit_time, f.exit_x, f.exit_y = (np.full(2, np.nan) for _ in range(3))
    resolver = tracing._resolve_surface_exit_crossing if curved else tracing._resolve_exit_crossing
    resolver(exit_z_mm*.001, f.old, f.momenta, f.new, f.momenta.copy(),
        f.alive, f.completed, f.exit_position, f.exit_momentum,
        previous_time_s=0., new_time_s=f.duration, arrival_time_s=f.exit_time,
        arrival_x_m=f.exit_x, arrival_y_m=f.exit_y, **extra)
    return f


@pytest.mark.parametrize('curved', [False, True])
def test_moving_c1_changes_acceptance_at_its_plane_not_at_exit(curved):
    earlier = clip_and_exit(aperture_at(.5), curved=curved)
    later = clip_and_exit(aperture_at(1.), curved=curved)
    assert earlier.completed.tolist() == [True, True]
    assert later.completed.tolist() == [False, True]
    assert later.blocked_z[0] == 1.
    assert later.blocked_key[0] == 'feg_c1_aperture'
    # The second ray fits C1 at 1 mm, then expands beyond that opening by the
    # handoff. There is no physical second C1 at the handoff to absorb it.
    assert later.x[1]*1000 == pytest.approx(.5)
    assert later.exit_x[1]*1000 == pytest.approx(.75)
    assert later.exit_time[1] > later.time[1]
    assert np.isnan(later.exit_time[0])
    np.testing.assert_allclose(later.exit_position[1, 2], .0015, atol=1e-15)


@pytest.mark.parametrize('curved', [False, True])
def test_coincident_aperture_and_exit_does_not_admit_blocked_rays(curved):
    result = clip_and_exit(aperture_at(1.5, radius_mm=1.), curved=curved)
    assert result.completed.tolist() == [False, True]
    assert result.blocked_z[0] == 1.5
    assert result.exit_time[1] == pytest.approx(result.time[1], abs=1e-18)


@pytest.mark.parametrize('curved', [False, True])
@pytest.mark.parametrize('inserted', [False, True])
def test_monochromator_slit_is_not_reapplied_at_the_exit(curved, inserted):
    aperture = aperture_at(1.)
    slit = MonochromatorSlit(gap_um=1200., maximum_gap_um=2000., inserted=inserted)
    aperture.bind_slit_profile(slit).select_slit_mode(True)
    result = clip_and_exit(aperture, curved=curved)
    assert result.completed.tolist() == ([False, True] if inserted else [True, True])
    assert result.exit_x[1]*1000 == pytest.approx(.75)
    # Removing blades never removes the mechanical clear bore.
    assert aperture.transmission_mask(np.array([0.]), np.array([3.1])).tolist() == [False]


def test_closed_openings_do_not_admit_a_finite_weight_axis_ray():
    aperture = aperture_at(1., radius_mm=0.)
    assert not aperture.transmission_mask(np.array([0.]), np.array([0.]))[0]
    slit = MonochromatorSlit(gap_um=0.)
    aperture.bind_slit_profile(slit).select_slit_mode(True)
    assert not aperture.transmission_mask(np.array([0.]), np.array([0.]))[0]


@pytest.mark.parametrize('position', [-1., 451., float('nan')])
def test_gun_rejects_an_aperture_outside_its_transport_interval(position):
    gun = FieldEmissionGun()
    gun.c1_aperture.field_center_offset_mm = position-gun.c1_aperture.mechanical_center_from_tip_mm
    with pytest.raises(ValueError, match='aperture'):
        gun.validate()


@pytest.mark.parametrize('curved', [False, True])
def test_full_tip_gun_keeps_c1_arrival_separate_and_does_not_clip_exit_again(curved):
    from temsim import module_manifest
    from temsim.optics.electron_gun.tip_assembly import model_from_part
    gun = FieldEmissionGun()
    if curved:
        gun.emitter.surface_model = model_from_part(module_manifest.part_data("gun/FEG.toml", "feg_tip"))
    gun.c1_aperture.radius_mm = 3.
    baseline = gun.trace_to_exit(9)
    c1 = next(p for p in baseline.plane_arrivals if p.key == gun.c1_aperture.key)
    handoff = baseline.plane_arrivals[-1]
    assert c1.z_mm == gun.c1_aperture.z_mm == 445.
    assert handoff.z_mm == gun.exit_plane_z_mm == 450.
    assert c1.key != handoff.key
    r_c1 = np.hypot(c1.x_m, c1.y_m)*1000
    r_exit = np.hypot(handoff.x_m, handoff.y_m)*1000
    # A real working point may be converging or diverging here. Choose an
    # opening that distinguishes the two physical planes in either case;
    # never require production defaults to put the exit after a waist.
    candidates = np.flatnonzero(handoff.transmitted & (np.abs(r_exit-r_c1) > 1e-8))
    assert candidates.size
    ray = candidates[-1]
    gun.c1_aperture.radius_mm = (r_exit[ray]+r_c1[ray])/2
    expected = baseline.exit_bundle.alive & (r_c1 <= gun.c1_aperture.radius_mm)
    assert expected[ray] != (r_exit[ray] <= gun.c1_aperture.radius_mm)
    limited = gun.trace_to_exit(9)
    assert limited is not baseline
    np.testing.assert_array_equal(limited.exit_bundle.alive, expected)
    if r_exit[ray] > r_c1[ray]:
        assert limited.exit_bundle.alive[ray]
        assert np.hypot(limited.exit_bundle.x_m[ray], limited.exit_bundle.y_m[ray])*1000 > gun.c1_aperture.radius_mm
    else:
        assert limited.blocked_key[ray] == gun.c1_aperture.key
        assert limited.blocked_z_mm[ray] == gun.c1_aperture.z_mm
    assert gun.trace_to_exit(9) is limited
    if curved:
        assert limited.surface_model_report['maximum_exit_energy_error_ev'] < 1e-3
    gun.c1_aperture.radius_mm = 0.
    gun.c1_aperture.mechanical_center_from_tip_mm = 440.
    moved = gun.trace_to_exit(9)
    stopped = np.array(moved.blocked_key) == gun.c1_aperture.key
    assert stopped.any()
    np.testing.assert_array_equal(moved.blocked_z_mm[stopped], 440.)
    assert not moved.exit_bundle.alive.any()
    assert moved.c1_transmitted_current_a == 0.
    assert not moved.plane_arrivals[-1].reached.any()


def test_old_gun_and_downstream_cache_identities_cannot_be_reused(monkeypatch):
    import temsim.calculation_cache as cache
    import temsim.optics.electron_gun.field_emission as feg
    from temsim.optics.column import default_state
    state = default_state()
    gun = state.electron_gun
    monkeypatch.setattr(feg, '_SHARED_TRACE_CACHE', OrderedDict())
    old_payload = json.loads(gun._cache_key(9))
    old_payload['trace_geometry_schema'] = 'executed-gun-components-v1'
    old_key = json.dumps(old_payload, sort_keys=True, separators=(',', ':'))
    old_result, new_result = object(), object()
    monkeypatch.setitem(feg._SHARED_TRACE_CACHE, old_key, old_result)
    monkeypatch.setattr(feg, 'trace_feg_to_exit', lambda *args: new_result)
    assert gun.trace_to_exit(9) is new_result
    current = cache.calculation_signatures(state)
    monkeypatch.setattr(cache, '_STAGE_INPUT_SCHEMA', 'live-lens-components-v3-tip-support')
    previous = cache.calculation_signatures(state)
    assert not {'incident', 'column', 'elastic', 'eds', 'stem', 'tem'} & cache.matching_products(previous, current)


def test_installed_monochromator_preserves_slit_and_bore_before_handoff():
    gun = FieldEmissionGun()
    gun.install_monochromator()
    gun.emitter.surface_model = None  # exercise the retained classical planar model
    gun.monochromator.slit.inserted = False
    opened = gun.trace_to_exit(9)
    slit = next(p for p in opened.plane_arrivals if p.key == 'feg_monochromator_slit')
    assert slit.z_mm == gun.c1_aperture.z_mm < gun.exit_plane_z_mm
    assert slit.transmitted.any() and opened.exit_bundle.alive.any()
    gun.monochromator.slit.inserted = True
    gun.monochromator.slit.gap_um = 0.
    closed = gun.trace_to_exit(9)
    assert closed.slit_reached.any()
    assert not closed.exit_bundle.alive.any()
    assert closed.monochromator_transmitted_current_a == 0.
    assert closed.c1_transmitted_current_a == 0.
    stopped = np.array(closed.blocked_key) == gun.c1_aperture.key
    np.testing.assert_array_equal(closed.blocked_z_mm[stopped], gun.c1_aperture.z_mm)


def test_thermionic_gun_uses_the_same_physical_c1_contract():
    from temsim.optics.electron_gun.thermionic import ThermionicGun
    gun = ThermionicGun()
    result = gun.trace_to_exit(9)
    c1 = next(p for p in result.plane_arrivals if p.key == gun.c1_aperture.key)
    assert c1.z_mm == gun.c1_aperture.z_mm
    assert result.plane_arrivals[-1].z_mm == gun.exit_plane_z_mm > c1.z_mm
    assert result.exit_bundle.alive.any()
    gun.c1_aperture.radius_mm = 0.
    blocked = gun.trace_to_exit(9)
    stopped = np.array(blocked.blocked_key) == gun.c1_aperture.key
    assert stopped.any() and not blocked.exit_bundle.alive.any()
    np.testing.assert_array_equal(blocked.blocked_z_mm[stopped], gun.c1_aperture.z_mm)
