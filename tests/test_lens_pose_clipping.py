"""Rigid lens hardware stops use the same local frame as its magnetic field."""
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

from temsim.physics.aperture_clipping import clip_segment
from temsim.physics.column_wall import clip_column_wall, _partition_vacuum_segments


def _state(*, offset_x=0., angle_y=0., bore_radius=1., aperture=False, tube_radius=None):
    row = dict(key="lens", mechanical_profile="magnetic_lens_assembly",
               local_center_z_mm=10., offset_x_mm=offset_x, rotation_y_mrad=angle_y)
    parts = [SimpleNamespace(key="lens", module_key="module", data=row, center_z_mm=10.)]
    segments = [SimpleNamespace(key="lens", start_z_mm=0., end_z_mm=20., inner_diameter_mm=2*bore_radius)]
    if tube_radius is not None:
        segments.append(SimpleNamespace(key="@vacuum_tube:stationary", start_z_mm=0., end_z_mm=20.,
                                       inner_diameter_mm=2*tube_radius))
    apertures = []
    if aperture:
        parts.append(SimpleNamespace(key="aperture", module_key="module", center_z_mm=12.,
            data=dict(key="aperture", parent_key="lens", local_center_z_mm=12.)))
        apertures.append(SimpleNamespace(key="aperture", enabled=True, installed=True,
            z_mm=12., radius_mm=.1, offset_x_mm=0., offset_y_mm=0.))
    return SimpleNamespace(_resolved_assembly=SimpleNamespace(parts=tuple(parts),
        vacuum_bore_segments=tuple(segments)), apertures=apertures, recording_planes=[],
        sample=SimpleNamespace(z_mm=8.))


def test_shifted_bore_moves_clearance_in_both_directions():
    state = _state(offset_x=.5)
    x = np.array([[1.25, -.75], [1.25, -.75]]) * 1e-3
    alive, stops, keys = clip_column_wall(state, [0., 20.], x, np.zeros_like(x))
    assert alive.tolist() == [True, False]
    assert np.isnan(stops[0]) and stops[1] == 0.
    assert keys == ["", "column_wall"]


def test_tilted_bore_finds_exact_wall_contact_between_saved_points():
    state = _state(angle_y=100., bore_radius=.25)
    x = np.zeros((2, 1))
    alive, stops, keys = clip_column_wall(state, [10., 20.], x, x)
    assert alive.tolist() == [False]
    assert stops[0] == pytest.approx(10.+.25/np.sin(.1))
    assert keys == ["column_wall"]


def test_stationary_vacuum_tube_does_not_follow_shifted_lens():
    state = _state(offset_x=.5, tube_radius=.75)
    x = np.full((2, 1), 1e-3)
    alive, stops, _ = clip_column_wall(state, [0., 20.], x, np.zeros_like(x))
    assert alive.tolist() == [False]
    assert stops[0] == 0.


@pytest.mark.parametrize("prior_stop", [np.nan, 14., 10.5])
def test_tilted_aperture_local_opening_and_global_stop_order(prior_stop):
    state = _state(angle_y=100., aperture=True)
    x = np.full((2, 1), 1e-3)
    alive, stops, keys = clip_segment(state, [10., 14.], x, np.zeros_like(x),
        alive=[not np.isfinite(prior_stop)], blocked_z=[prior_stop], blocked_key=["prior"])
    expected = 10.+(2.-np.sin(.1))/np.cos(.1)
    assert not alive[0]
    assert stops[0] == pytest.approx(min(prior_stop, expected) if np.isfinite(prior_stop) else expected)
    assert keys == ["prior" if prior_stop == 10.5 else "aperture"]


def test_post_sample_aperture_clipping_uses_same_tilted_plane():
    from temsim.physics.recording_clipping import clip_recording_planes
    state = _state(angle_y=100., aperture=True)
    x = np.full((2, 1), 1e-3)
    alive, stops, keys = clip_recording_planes(state, [10., 14.], x, np.zeros_like(x),
        np.array([True]), np.array([np.nan]), [""])
    assert not alive[0] and keys == ["aperture"]
    assert stops[0] == pytest.approx(10.+(2.-np.sin(.1))/np.cos(.1))


def test_medium_path_ends_at_posed_wall_before_scattering():
    from temsim.physics.residual_medium import ColumnMediumTransport
    state = _state(angle_y=100., bore_radius=.25)
    transport = object.__new__(ColumnMediumTransport)
    transport.state, transport.z = state, np.array([10., 20.])
    transport.alive = np.array([True])
    transport.blocked_z, transport.blocked_key = np.array([np.nan]), [""]
    transport.node_radius, transport.interval_radius = np.full(2, np.inf), np.full(1, np.inf)
    _, transport.placed_bores = _partition_vacuum_segments(state, state._resolved_assembly.vacuum_bore_segments)
    transport.plane_steps, transport.recording_keys = set(), set()
    transport.has_posed_apertures, transport.energies = False, np.array([200000.])
    ends = []
    def advance(start, end, direction, energy, **kwargs):
        ends.append(end.copy())
        return direction
    transport.advance = advance
    before, after = np.zeros((4, 1)), np.zeros((4, 1))
    transport(0, before, after)
    expected = 10.+.25/np.sin(.1)
    assert not transport.alive[0]
    assert transport.blocked_z[0] == pytest.approx(expected)
    assert ends[0][0, 2]*1000 == pytest.approx(expected)


def test_real_resolver_keeps_hidden_fixed_tube_when_narrower_condenser_bore_moves():
    from temsim.optics.column import default_state
    from temsim.column.module_assembly import _module_vacuum_segments
    from temsim.immutable_json import freeze_json
    state = default_state()
    assembly = state._resolved_assembly
    changed = tuple(replace(part, data=freeze_json({**part.data, **values}))
        if (values := ({"offset_x_mm": 1.} if part.key == "condenser_lens_1" else
                       {"vacuum_inner_diameter_mm": 5.} if part.key == "condenser_lens_1_lower_pole" else {}))
        else part for part in assembly.parts)
    reduced = []
    for module in assembly.modules:
        reference = next(part for part in changed if part.module_key == module.key)
        origin = reference.center_z_mm-float(reference.data["local_center_z_mm"])
        reduced.extend(_module_vacuum_segments(module, origin, changed))
    state._resolved_assembly = replace(assembly, parts=changed, vacuum_bore_segments=tuple(reduced))
    pole = next(part for part in changed if part.key == "condenser_lens_1_lower_pole")
    z = np.array([pole.start_z_mm+1., pole.start_z_mm+2.])
    nominal = next(segment for segment in reduced if segment.start_z_mm < z[0] < segment.end_z_mm)
    assert nominal.key == pole.key and nominal.inner_diameter_mm == 5.
    x = np.full((2, 1), 3.1e-3)
    # x=3.1 is inside the moved 2.5 mm-radius pole (local x=2.1), but
    # outside the fixed 2.88 mm-radius tube omitted by nominal reduction.
    alive, stops, _ = clip_column_wall(state, z, x, np.zeros_like(x))
    assert not alive[0] and stops[0] == pytest.approx(z[0])
    stationary, _ = _partition_vacuum_segments(state, reduced)
    tube = next(segment for segment in stationary
                if segment.key == "@vacuum_tube:c1_c2_to_upper_objective"
                and segment.start_z_mm < z[0] < segment.end_z_mm)
    assert tube.inner_diameter_mm == 5.76


@pytest.mark.parametrize("lens_radius, fixed_radius, shift, ray_x", [
    (1., 2., 2., 2.5),  # Narrowest moved bore had hidden a wider fixed bore.
    (2., 1., 3., 0.),   # Moved wider bore never owned the nominal profile.
])
def test_raw_overlapping_bores_survive_nominal_narrowest_reduction(lens_radius, fixed_radius, shift, ray_x):
    from temsim.column.module_assembly import _module_vacuum_segments
    def part(key, profile, radius, **fields):
        return SimpleNamespace(key=key, name=key, module_key="module", length_mm=20.,
            start_z_mm=0., center_z_mm=10., end_z_mm=20., data=dict(key=key,
                mechanical_profile=profile, local_center_z_mm=10.,
                vacuum_inner_diameter_mm=2*radius, **fields))
    parts = (part("lens", "magnetic_lens_assembly", lens_radius, offset_x_mm=shift),
             part("fixed", "vacuum_liner", fixed_radius))
    module = SimpleNamespace(key="module", entrance_z_mm=0., exit_z_mm=20., parts=parts,
        geometry={"vacuum_drift_inner_diameter_mm": 20.})
    reduced = _module_vacuum_segments(module, 0., parts)
    assert len(reduced) == 1
    state = SimpleNamespace(_resolved_assembly=SimpleNamespace(
        modules=(module,), parts=parts, vacuum_bore_segments=reduced))
    x = np.full((2, 1), ray_x*1e-3)
    alive, stops, _ = clip_column_wall(state, [5., 15.], x, np.zeros_like(x))
    assert not alive[0] and stops[0] == pytest.approx(5.)


def test_bore_partition_reuses_only_same_immutable_assembly_and_segment_tuple(monkeypatch):
    from temsim.optics.column import default_state
    from temsim.physics import column_wall
    state = default_state()
    segments = state._resolved_assembly.vacuum_bore_segments
    original = column_wall._partition_vacuum_segments_uncached
    calls = []
    def counted(*args):
        calls.append(None)
        return original(*args)
    monkeypatch.setattr(column_wall, "_partition_vacuum_segments_uncached", counted)
    first = _partition_vacuum_segments(state, segments)
    assert _partition_vacuum_segments(state, segments) is first
    assert len(calls) == 1
    state._resolved_assembly = replace(state._resolved_assembly)
    _partition_vacuum_segments(state, segments)
    assert len(calls) == 2
    _partition_vacuum_segments(state, (*segments,))
    assert len(calls) == 3
    mutable = _state(offset_x=.5)
    mutable_segments = mutable._resolved_assembly.vacuum_bore_segments
    _, before = _partition_vacuum_segments(mutable, mutable_segments)
    mutable._resolved_assembly.parts[0].data["offset_x_mm"] = 1.
    _, after = _partition_vacuum_segments(mutable, mutable_segments)
    assert before[0][1].origin_global_m != after[0][1].origin_global_m
    assert not hasattr(mutable, "_lens_pose_bore_cache")
