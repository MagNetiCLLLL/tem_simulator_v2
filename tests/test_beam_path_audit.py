from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.beam_path_audit import gun_planes_coordinates
from temsim.optics.beam_path_audit import partition_surface_crossovers

from temsim.optics.beam_path_audit import (
    spot_measurement, gun_plane_coordinates, crossover_candidates,
    crossover_intervals, require_same_topology, emission_measurement, probe_qualification,
    optical_component_planes,
    uniform_cap_footprint,
)


def test_tip_footprint_uses_full_physical_cap_not_importance_sample_extent():
    from dataclasses import replace
    from temsim.optics.electron_gun.tip_surface import load_tip_surface_reference
    model = load_tip_surface_reference()
    result = uniform_cap_footprint(model)
    assert result['edge_diameter_nm'] == pytest.approx(34.72963553338607)
    # Integrate uniform emitting area independently with midpoint quadrature.
    u = (np.arange(100000)+.5)/100000
    cosine = 1-u*(1-np.cos(np.deg2rad(10)))
    r = 100*np.sqrt(1-cosine*cosine)
    assert result['rms_radius_nm'] == pytest.approx(np.sqrt(np.mean(r*r)),rel=1e-10)
    assert result['diameter95_nm'] == pytest.approx(2*np.quantile(r,.95),rel=1e-5)
    numerical = replace(model,emission=replace(model.emission,spatial_sampling='apex_stratified_v1',
        spatial_stratum_allocation=(1,1,1,1,1,4,24,24,1)))
    assert uniform_cap_footprint(numerical) == result


def test_spot_units_weighted_diameter_and_zero_current_support():
    phi = np.arange(8)*np.pi/4
    x, y = 2e-9*np.cos(phi), 2e-9*np.sin(phi)
    x, y = np.r_[x, 1.], np.r_[y, 1.]
    w = np.r_[np.ones(8)/8, 0.]
    s = spot_measurement("tip", None, (x, y, x*0, y*0), w, np.ones(9, bool))
    assert s.rays == 8
    assert s.rms_radius_nm == pytest.approx(2.)
    assert s.diameter95_nm == pytest.approx(4.)
    assert s.source_current_fraction == 1.
    empty = spot_measurement("blocked", 5., (x, y, x*0, y*0), w, np.zeros(9, bool))
    assert empty.rays == 0 and np.isnan(empty.diameter95_nm)


def test_emission_subpopulation_keeps_its_source_fraction():
    b = SimpleNamespace(x_m=np.array([-1.,1.,-10.,10.])*1e-9,
        y_m=np.zeros(4),weight=np.array([.1,.1,.4,.4]),
        surface_direction=np.tile([0.,0.,1.],(4,1)))
    r = emission_measurement(b,alive=[True,True,False,False])
    assert r.source_current_fraction == pytest.approx(.2)
    assert r.diameter95_nm == pytest.approx(2.)
    with pytest.raises(ValueError,match='population'):
        emission_measurement(b,alive=[True])


def test_spot_preserves_source_fraction_and_rejects_nonfinite_survivor():
    coords = np.arange(8, dtype=float)
    alive = np.arange(8) < 4
    s = spot_measurement("plane", 5., (coords,)*4, np.ones(8)/8, alive)
    assert s.source_current_fraction == .5
    coords[0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        spot_measurement("plane", 5., (coords,)*4, np.ones(8)/8, alive)


def test_curved_emission_never_flips_backward_directions_into_forward_rays():
    bundle = SimpleNamespace(x_m=np.zeros(10), y_m=np.zeros(10),
        weight=np.ones(10)/10, surface_direction=np.array([[0., 0., 1.]]*9+[[0., 0., -1.]]))
    s = emission_measurement(bundle)
    assert s.alpha95_mrad == pytest.approx(np.pi*1000)
    assert np.isnan(s.covariance_waist_offset_mm)


def test_gun_crossing_keeps_later_losses_but_uses_actual_exit_state():
    hist = SimpleNamespace(z_mm=np.array([[0., 0.], [1., 1.], [2., 2.-1e-15]]))
    hist.x_m = hist.z_mm*1e-3
    hist.y_m = hist.tx_rad = hist.ty_rad = hist.x_m*0
    e = SimpleNamespace(weight=np.array([.5, .5]), alive=np.array([False, True]),
                        x_m=np.array([.002, .002]), y_m=np.zeros(2),
                        tx_rad=np.zeros(2), ty_rad=np.zeros(2))
    trace = SimpleNamespace(z_mm=np.array([0., 2.]), equal_time_history=hist,
                            exit_bundle=e, blocked_z_mm=np.array([1., np.nan]))
    p, mask = gun_plane_coordinates(trace, .5)
    assert mask.all() and np.allclose(p[0], .0005)
    p, mask = gun_plane_coordinates(trace, 1.5)
    assert mask.tolist() == [False, True] and np.isnan(p[0, 0])
    p, mask = gun_plane_coordinates(trace, 2.)
    assert mask.tolist() == [False, True] and p[0, 1] == .002


def test_topology_preserves_count_order_and_component_interval():
    planes = [("C1", 1.), ("C2", 2.), ("aperture", 2.5), ("C3", 3.), ("sample", 4.)]
    assert require_same_topology([1.5, 2.2, 3.5], [1.1, 2.4, 3.1], planes)
    for changed in ([1.5, 3.5], [1.5, 2.2, 2.3, 3.5], [1.5, 2.7, 3.5], [1.5, 2.5, 3.5]):
        with pytest.raises(ValueError, match="topology changed"):
            require_same_topology([1.5, 2.2, 3.5], changed, planes)
    shared = planes+[("objective", 4.)]
    assert crossover_intervals([4.], shared) == (("on", ("objective", "sample")),)


def test_topology_never_uses_schematic_pole_parts_as_optical_boundaries():
    state = SimpleNamespace(electron_gun=SimpleNamespace(exit_plane_z_mm=0.), sample=SimpleNamespace(z_mm=4.),
        _resolved_optics_layout=[SimpleNamespace(key=key, kind=kind, optical_reference_plane_z_mm=z)
            for key, kind, z in [("C1", "magnetic_lens", 1.), ("C1_pole", "pole_piece", 1.5),
                                 ("C2", "magnetic_lens", 2.), ("sample", "sample", 4.)]])
    assert optical_component_planes(state) == [("gun_exit", 0.), ("C1", 1.), ("C2", 2.), ("sample", 4.)]


def test_terminal_focus_classification_never_hides_defocus_or_intermediate_waists():
    rows = [{"z_mm":z} for z in (500.,700.,1599.-32e-6,1599.-.03e-6)]
    intermediate, terminal = partition_surface_crossovers(rows,1599.)
    assert intermediate == rows[:3] and terminal == rows[3:]
    assert intermediate+terminal == rows
    assert len(partition_surface_crossovers([{"z_mm":1599.+.03e-6}],1599.)[1]) == 1
    with pytest.raises(ValueError):
        partition_surface_crossovers(rows,1599.,tolerance_nm=0)


def test_batched_gun_history_keeps_first_crossing_stops_and_exact_exit():
    z = np.array([[0.,0.,0.], [1.,1.,1.], [.5,2.,2.], [2.,3.,3.], [3.,4.,4.]])
    values = np.arange(15,dtype=float).reshape(5,3)
    history = SimpleNamespace(z_mm=z, x_m=values, y_m=2*values, tx_rad=3*values, ty_rad=4*values)
    e = SimpleNamespace(weight=np.array([.3,.3,.4]), alive=np.array([True,False,True]),
                        x_m=np.array([100.,101.,102.]), y_m=values[0], tx_rad=values[1], ty_rad=values[2])
    trace = SimpleNamespace(equal_time_history=history, z_mm=np.array([0.,4.]),
                            exit_bundle=e, blocked_z_mm=np.array([np.nan,2.5,np.nan]))
    planes = np.array([1.5,.5,1.5,4.,2.5])  # Order and duplicate requests are preserved.
    coordinates, masks = gun_planes_coordinates(trace,planes)
    for j,plane in enumerate(planes[:-2]):
        for ray in range(3):
            upper = np.flatnonzero(z[:,ray]>=plane)[0]
            lower = max(0,upper-1)
            fraction = (plane-z[lower,ray])/(z[upper,ray]-z[lower,ray])
            expected = values[lower,ray]+fraction*(values[upper,ray]-values[lower,ray])
            np.testing.assert_array_equal(coordinates[j,:,ray],np.arange(1,5)*expected)
    np.testing.assert_array_equal(coordinates[3,0],e.x_m)
    np.testing.assert_array_equal(masks[3],e.alive)
    assert not masks[4,1] and np.isnan(coordinates[4,:,1]).all()
    with pytest.raises(ValueError,match="Missing recorded"):
        gun_planes_coordinates(trace,[3.5])  # Ray 0 never reaches this interior plane.
    with pytest.raises(ValueError,match="precedes"):
        gun_planes_coordinates(trace,[-.1])


@pytest.mark.parametrize("bad", [[np.nan], [2., 1.], [2., 2.], [5.]])
def test_invalid_topology_positions(bad):
    with pytest.raises(ValueError):
        crossover_intervals(bad, [("tip", 0.), ("sample", 4.)])


def test_subgrid_crossover_uses_slopes_and_not_display_minimum():
    phi = np.arange(16)*2*np.pi/16
    tx, ty = .03*np.cos(phi), .03*np.sin(phi)
    z = np.array([10., 10.5])
    dz = (z[:, None]-10.234567)*1e-3
    cp = SimpleNamespace(z_mm=z, x_m=dz*tx+1e-9*np.sin(phi),
        y_m=dz*ty-1e-9*np.cos(phi), tx_rad=np.broadcast_to(tx, (2, 16)),
        ty_rad=np.broadcast_to(ty, (2, 16)))
    rows = crossover_candidates(cp, np.ones((2, 16), bool), np.ones(16)/16)
    assert len(rows) == 1
    assert rows[0]["z_mm"] == pytest.approx(10.234567, abs=1e-9)
    assert rows[0]["rms_radius_nm"] == pytest.approx(1., rel=1e-6)
    assert rows[0]["status"].startswith("INTERPOLATED")
    cp.tx_rad = -cp.tx_rad
    cp.ty_rad = -cp.ty_rad
    assert crossover_candidates(cp, np.ones((2, 16), bool), np.ones(16)/16) == []


def test_clipping_does_not_manufacture_a_crossover():
    # Five surviving divergent rays, plus five later-clipped convergent rays.
    a = np.linspace(-1., 1., 5)
    x = np.array([np.r_[a, 100*a], np.r_[2*a, a]])*1e-6
    slope = np.broadcast_to(np.r_[a, -100*a]*.001, (2, 10))
    cp = SimpleNamespace(z_mm=np.array([1., 2.]), x_m=x, y_m=x*0,
                         tx_rad=slope, ty_rad=x*0)
    masks = np.ones((2, 10), bool)
    masks[1, 5:] = False
    assert crossover_candidates(cp, masks, np.ones(10)/10) == []


def test_correct_angle_alone_cannot_qualify_a_micrometre_probe_or_changed_topology():
    from temsim.optics.surface_probe_focus import SurfaceFocusMeasurement
    from temsim.physics.beam_statistics import transverse_beam_statistics
    phi = np.arange(32)*2*np.pi/32
    stats = transverse_beam_statistics(2e-6*np.cos(phi), 2e-6*np.sin(phi),
        -.03*np.sin(phi), .03*np.cos(phi))
    m = SurfaceFocusMeasurement(4., stats, 0., .001, (2e-6,)*3, .025)
    result = probe_qualification(m, target_mrad=30., maximum_diameter95_nm=1.,
        reference_crossovers_mm=[1.5, 2.5], candidate_crossovers_mm=[1.5, 2.5, 3.5],
        component_planes=[("C1", 1.), ("C2", 2.), ("C3", 3.), ("sample", 4.)],
        sampling_converged=False)
    assert result["status"] == "FAIL"
    assert set(result["failures"]) == {"probe_diameter", "crossover_topology", "sampling_convergence"}


def test_crossover_refinement_reexecutes_physical_planes(monkeypatch):
    from temsim.optics import beam_path_audit as module
    phi = np.arange(16)*2*np.pi/16
    tx, ty = .03*np.cos(phi), .03*np.sin(phi)
    gun = SimpleNamespace(exit_bundle=SimpleNamespace(weight=np.ones(16)/16))
    calls = []

    def execute(state, planes, *, step_mm):
        z = np.asarray(planes)
        calls.append(z.copy())
        dz = (z[:, None]-10.234567)*1e-3
        cp = SimpleNamespace(z_mm=z, x_m=dz*tx+1e-9*np.sin(phi),
            y_m=dz*ty-1e-9*np.cos(phi), tx_rad=np.broadcast_to(tx, (len(z), 16)),
            ty_rad=np.broadcast_to(ty, (len(z), 16)))
        return gun, cp, np.ones((len(z), 16), bool)

    monkeypatch.setattr(module, "incident_checkpoints", execute)
    result = module.refine_crossover_candidate(None, {"bracket_mm": (10., 10.5)})
    assert result["z_mm"] == pytest.approx(10.234567, abs=1e-9)
    assert result["measurement"]["rms_radius_nm"] == pytest.approx(1.)
    assert result["evaluations"] >= 3 and len(calls) >= 2
