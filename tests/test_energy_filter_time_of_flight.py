"""Filter-boundary clocks use executed Boris crossings and upstream ancestry.

The straight, zero-field fixture is an analytical timing reference, not a
qualification of a complete curved-filter column or a replacement source.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.component_keys import ENERGY_FILTER_EFTEM_OUTPUT_PLANE
from temsim.optics import energy_filter_raytrace as tracing
from temsim.physics.relativistic_lorentz import (
    momentum_from_kinetic_energy_ev, velocity_from_momentum_m_per_s,
)


@pytest.fixture
def straight_filter(monkeypatch):
    sector = SimpleNamespace(
        exit_frame=SimpleNamespace(rotation_local_to_global=np.array(
            [[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])),
        exit_point_m=np.zeros(3), arc_length_m=.001,
        aperture_blocked_mask=lambda p: np.zeros(len(p), dtype=bool),
    )
    field = SimpleNamespace(sector=sector,
        field_at_global_positions_t=lambda p: np.zeros_like(p))
    slit = SimpleNamespace(key="energy_filter_slit", distance_from_sector_exit_m=.001,
        calibrated_dispersion_um_per_ev=1.,
        transmission_mask=lambda x, y: np.ones(x.shape, dtype=bool))
    energy_filter = SimpleNamespace(
        enabled=True, entrance_z_mm=15., maximum_trace_rays=64,
        alignment_x_mrad=0., alignment_y_mrad=0., ray_step_mm=.7,
        prism_entrance_s_mm=0., output_detector_d_mm=2., eels_plane_offset_mm=1.,
        energy_slit=slit, multipoles=(), fast_shutter_d_mm=1.5,
        fast_shutter=SimpleNamespace(enabled=False, open=True, key="energy_filter_shutter"),
        output_detector_inserted=False, output_detector_width_mm=1.,
        multi_eels_enabled=False, bias_tube=SimpleNamespace(enabled=False, offset_ev=0.),
        zebra_detector=SimpleNamespace(key="energy_filter_zebra", alignment_mode=False,
            recording_mask=lambda x, y: np.ones(x.shape, dtype=bool)),
        camera_deflector=SimpleNamespace(enabled=False),
    )
    state = SimpleNamespace(energy_filter=energy_filter, beam_voltage_kv=200.,
        energy_filter_entrance_aperture=SimpleNamespace(enabled=False, installed=True))
    monkeypatch.setattr(tracing, "ensure_energy_filter", lambda state: None)
    monkeypatch.setattr(tracing, "magnetic_field_from_energy_filter", lambda ef: field)
    monkeypatch.setattr(tracing, "measure_aperture_transmitted_current", lambda *args: object())
    monkeypatch.setattr(tracing, "source_current_pa", lambda state: 10.)
    return state


def test_field_free_clock_matches_relativistic_velocity_at_each_physical_plane(straight_filter):
    state = straight_filter
    tx, ty = np.array([0., .3]), np.array([0., -.2])
    offsets = np.array([0., -100_000.])
    batch = tracing.trace_energy_filter_batch(state, [.1, -.1], [.05, -.05], tx, ty, offsets)
    momentum = momentum_from_kinetic_energy_ev(state.beam_voltage_kv * 1000. + offsets,
        np.column_stack((np.ones(2), ty, tx)))
    vx = velocity_from_momentum_m_per_s(momentum)[:, 0]
    for distance, prefix in ((.001, "slit"), (.002, "output"), (.003, "eels")):
        times = getattr(batch, prefix + "_time_s")
        assert times.dtype == np.float64
        np.testing.assert_allclose(times, distance / vx, rtol=4e-15, atol=1e-25)
        np.testing.assert_allclose(getattr(batch, prefix + "_dispersive_m"),
            -(np.array([.1, -.1]) * .001 + tx * distance), atol=2e-18)
        np.testing.assert_allclose(getattr(batch, prefix + "_non_dispersive_m"),
            np.array([.05, -.05]) * .001 + ty * distance, atol=2e-18)
        np.testing.assert_allclose(getattr(batch, prefix + "_tx_rad"), -tx, atol=2e-16)
        np.testing.assert_allclose(getattr(batch, prefix + "_ty_rad"), ty, atol=2e-16)
    assert batch.time_s[0] == 0.
    assert batch.time_s.shape == (len(batch.positions_m),)
    assert np.all(np.diff(batch.time_s) > 0.)
    assert np.all(batch.eels_time_s <= batch.time_s[batch.stop_step])


def test_slit_absorption_has_a_slit_clock_but_no_later_arrivals(straight_filter):
    straight_filter.energy_filter.energy_slit.transmission_mask = lambda x, y: np.abs(x) < .0005
    batch = tracing.trace_energy_filter_batch(straight_filter, [0., 1.], [0., 0.],
        [0., 0.], [0., 0.], [0., 0.])
    np.testing.assert_array_equal(batch.reached_slit, [True, True])
    np.testing.assert_array_equal(batch.passed_slit, [True, False])
    assert np.all(np.isfinite(batch.slit_time_s))
    assert np.isfinite(batch.eels_time_s[0])
    assert np.isnan(batch.output_time_s[1]) and np.isnan(batch.eels_time_s[1])
    assert batch.stop_key[1] == "energy_filter_slit"


def test_output_camera_absorbs_only_intercepted_rays_and_eels_misses_still_arrive(straight_filter):
    ef = straight_filter.energy_filter
    ef.output_detector_inserted = True
    ef.zebra_detector.recording_mask = lambda x, y: np.zeros(x.shape, dtype=bool)
    batch = tracing.trace_energy_filter_batch(straight_filter, [0., 2.], [0., 0.],
        [0., 0.], [0., 0.], [0., 0.])
    assert np.all(np.isfinite(batch.output_time_s))
    assert batch.stop_key[0] == "energy_filter_output_detector"
    assert np.isnan(batch.eels_time_s[0])
    assert np.isfinite(batch.eels_time_s[1]) and batch.reached_eels[1]
    assert not batch.zebra_recorded[1]
    assert batch.stop_key[1] == "energy_filter_zebra_miss"


def test_stopping_at_slit_preserves_only_its_executed_clock(straight_filter):
    batch = tracing.trace_energy_filter_batch(straight_filter, [0.], [0.], [0.], [0.], [0.],
        stop_at_slit=True)
    assert np.isfinite(batch.slit_time_s[0])
    assert np.isnan(batch.output_time_s[0]) and np.isnan(batch.eels_time_s[0])


def test_eels_geometry_is_not_relabelled_as_bias_shifted_detector_coordinate(straight_filter):
    ef = straight_filter.energy_filter
    ef.multi_eels_enabled = True
    ef.bias_tube.enabled = True
    ef.bias_tube.offset_ev = 100.
    recorded_coordinate = []

    def record(x, y):
        recorded_coordinate.append(x.copy())
        return np.ones(x.shape, dtype=bool)

    ef.zebra_detector.recording_mask = record
    batch = tracing.trace_energy_filter_batch(straight_filter, [0.], [0.], [0.], [0.], [0.])
    np.testing.assert_array_equal(batch.eels_dispersive_m, [0.])
    np.testing.assert_allclose(recorded_coordinate[0], [-100e-6])
    assert np.isfinite(batch.eels_time_s[0])


def upstream_simulation(*, known_times=True):
    branch = SimpleNamespace(name="loss", colour="#fff", z=np.array([10., 20.]),
        x=np.zeros((2, 3)), y=np.zeros((2, 3)), tx=np.zeros((2, 3)), ty=np.zeros((2, 3)),
        blocked_z=np.full(3, np.nan), alive=np.ones(3, bool), weight=1.,
        ray_weight=np.array([.1, .2, .3]), energy_offset_ev=np.zeros(3),
        source_ray_id=np.array([8, 8, 9]), source_azimuth_rad=np.full(3, np.nan),
        flight_time_s=(np.array([[1., 5., np.nan], [3., 9., np.nan]]) * 1e-9
                       if known_times else None))
    return SimpleNamespace(incident=None, branches={"loss": branch},
        metrics={"branch_weights_are_absolute": True})


def test_published_arrivals_keep_ancestry_and_do_not_include_the_reference_ray(straight_filter, monkeypatch):
    simulation = upstream_simulation()
    calls = []
    original = tracing.trace_energy_filter_batch

    def counted(*args, **kwargs):
        batch = original(*args, **kwargs)
        calls.append(batch)
        return batch

    monkeypatch.setattr(tracing, "trace_energy_filter_batch", counted)
    result = tracing.simulate_energy_filter(straight_filter, simulation)
    assert len(calls) == 2  # Existing reference diagnostic plus physical population.
    assert len(result.paths_u_mm) == 4
    np.testing.assert_array_equal(result.source_ray_id, [8, 8, 9])
    np.testing.assert_array_equal(result.source_path_index, [0, 1, 2])
    assert result.source_branch == ("loss", "loss", "loss")
    np.testing.assert_array_equal(result.source_fraction, [.1, .2, .3])
    np.testing.assert_allclose(result.entrance_time_s, [2e-9, 7e-9, np.nan])
    assert [p.key for p in result.timed_planes] == [
        "energy_filter_slit", ENERGY_FILTER_EFTEM_OUTPUT_PLANE, "energy_filter_zebra"]
    for plane, prefix in zip(result.timed_planes, ("slit", "output", "eels")):
        assert plane.coordinate_frame == "sector_exit_local"
        assert plane.time_reference == "simultaneous_tip_emission"
        assert plane.time_s.shape == (3,)
        np.testing.assert_allclose(plane.time_s[:2], result.entrance_time_s[:2]
            + getattr(calls[-1], prefix + "_time_s")[:2], rtol=0., atol=0.)
        assert np.isnan(plane.time_s[2]) and plane.reached[2]
        assert plane.time_s[0] != plane.time_s[1]  # Same ancestor, distinct paths.


def test_unknown_upstream_clock_is_not_replaced_by_entrance_or_reference_zero(straight_filter):
    result = tracing.simulate_energy_filter(straight_filter, upstream_simulation(known_times=False))
    assert np.all(np.isnan(result.entrance_time_s))
    for plane in result.timed_planes:
        assert np.all(plane.reached)
        assert np.all(np.isnan(plane.time_s))


def test_representative_sampling_retains_selected_path_clocks_and_lineage():
    rays = [tracing.EntranceRay(0., 0., 0., 0., 0., "#fff", .2,
        source_ray_id=50+i, flight_time_s=(i+1)*1e-9, source_branch="elastic",
        source_path_index=i) for i in range(5)]
    selected, total = tracing._representative_entrance_rays(rays, 2)
    assert total == 5
    assert [r.source_ray_id for r in selected] == [50, 54]
    assert [r.source_path_index for r in selected] == [0, 4]
    assert [r.flight_time_s for r in selected] == [rays[0].flight_time_s, rays[4].flight_time_s]
    assert sum(r.source_fraction for r in selected) == pytest.approx(1.)
    assert all(r.source_branch == "elastic" for r in selected)


def test_explicit_source_id_survives_missing_source_azimuth(straight_filter):
    simulation = upstream_simulation()
    simulation.branches["loss"].source_azimuth_rad = None
    rays = tracing.extract_entrance_rays(straight_filter, simulation)
    assert [r.source_ray_id for r in rays] == [8, 8, 9]


def test_filter_entrance_excludes_prior_absorption_even_when_plane_clock_exists(straight_filter):
    from temsim.physics.flight_time import sample_flight_time
    simulation = upstream_simulation()
    branch = simulation.branches["loss"]
    branch.blocked_z = np.array([14., 15., 16.])
    assert np.isfinite(sample_flight_time(branch, 15.)[1])
    rays = tracing.extract_entrance_rays(straight_filter, simulation)
    assert len(rays) == 1 and rays[0].source_path_index == 2


def test_filter_uses_validated_exit_clock_not_known_optical_reference(straight_filter):
    from temsim.simulation_pipeline import _energy_filter_source_view
    from temsim.specimen.downstream_transport import GeometricSpecimenExit
    reference = upstream_simulation()
    detailed_branch = upstream_simulation(known_times=False).branches["loss"]
    checkpoint = GeometricSpecimenExit((detailed_branch,),
        {"tracked_downstream_source_probability": 1.,
         "inelastic_absorbed_source_probability": 0.}, dependency_signature="detailed-v1")
    view = _energy_filter_source_view(reference, checkpoint,
        {"column": "reference-v1", "sample_downstream": "detailed-v1"})
    result = tracing.simulate_energy_filter(straight_filter, view)
    assert result.entrance_provenance == "validated_specimen_exit"
    assert result.entrance_dependency_signature == "detailed-v1"
    assert np.all(np.isnan(result.entrance_time_s))
    for plane in result.timed_planes:
        assert np.all(plane.reached)
        assert np.all(np.isnan(plane.time_s))
    assert reference.branches["loss"].flight_time_s is not None
