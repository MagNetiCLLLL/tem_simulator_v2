"""Finite-field recording fixtures, without microscope performance claims."""
from copy import copy
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.model import DeflectorPair
from temsim.physics import core, record_plane
from temsim.physics.first_order import trace_transverse_transfer
from temsim.physics.instrument_magnetic import active_column_events
from temsim.physics.record_plane import PlaneStop, build_record_plane_plan
from test_shared_column_magnetic_transport import state_with_coil


class OscillatingPair(DeflectorPair):
    scan_enabled = True

    def kick_events(self, time_s=0.):
        return ((self.upper_z_mm, self.upper_x_mrad*1e-3*(1.+np.sin(2.*np.pi*time_s)),
                 self.upper_y_mrad*1e-3),
                (self.lower_z_mm, self.lower_x_mrad*1e-3, self.lower_y_mrad*1e-3))


def _planes(monkeypatch, values):
    planes = tuple(PlaneStop(str(index), "Fixture detector", value, "detector", "disk",
        outer_width_mm=10., readout_enabled=True, non_blocking=True)
        for index, value in enumerate(values))
    monkeypatch.setattr(record_plane, "runtime_recording_stops", lambda *_args: planes)
    return planes


def _central_orbit(state, source, targets):
    points = core.propagate(state, source, targets[-1], *(np.zeros(1),)*4,
        events=active_column_events(state), include_spherical_aberration=False,
        include_hexapole=False, checkpoint_z_mm=targets, return_checkpoints=True)[-1]
    indices = np.searchsorted(points.z_mm, targets)
    return np.column_stack((points.x_m[indices, 0], points.y_m[indices, 0]))


def test_default_first_order_and_recording_keep_coincident_finite_coils(monkeypatch):
    state, _ = state_with_coil(step=.2)
    state.sample.z_mm = 0.
    state.deflectors.append(DeflectorPair("Second coil", "second", 5., 20.,
        .2, 0., 0., 0., thickness_mm=4.))
    transfer = trace_transverse_transfer(state, 0., 4.5)
    # Independent constant-force integral before either centre plane.
    expected = .5*(.0003/.002)*.0005**2 + .5*(.0002/.004)*.0015**2
    assert transfer.position_offset_m[0] == pytest.approx(expected, abs=1e-20)
    assert trace_transverse_transfer(state, 0., 4.5, events=()).position_offset_m == (0., 0.)
    _planes(monkeypatch, (4.5, 5.5))
    plan = build_record_plane_plan(state, source_z_mm=0.)
    assert plan.transfers[0].position_offset_m[0] == pytest.approx(expected, abs=1e-20)
    partial = trace_transverse_transfer(state, 4.5, 5.5)
    assert partial.position_offset_m[0] == pytest.approx(.5*(.0003/.002+.0002/.004)*.001**2,
                                                       abs=1e-20)
    from temsim.specimen.downstream_transport import _post_sample_events
    assert _post_sample_events(state) == active_column_events(state)


def test_scanned_finite_coil_response_crosses_overlapping_lens_without_pixel_retraces(monkeypatch):
    state, _ = state_with_coil(step=.1)
    state.sample.z_mm = 0.
    state.simulation_time_s = .13
    state.deflectors = [OscillatingPair("Driven coil", "driven", 5., 20.,
        .3, -.1, 0., 0., thickness_mm=4.)]
    state.lenses = [SimpleNamespace(key="fixture_lens", name="Fixture lens", enabled=True,
        z_mm=5., polarity=1, percent=100., max_percent=100.,
        magnetic_field_t=lambda z: .05*np.exp(-.5*((np.asarray(z)-5.)/2.)**2),
        field_support_mm=lambda *_args: (0., 10.))]
    targets = (4., 5., 6., 9.)
    _planes(monkeypatch, targets)
    times = np.linspace(0., 1., 64, endpoint=False)[None]
    execute = core.execute_propagation_plan
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return execute(*args, **kwargs)

    monkeypatch.setattr(core, "execute_propagation_plan", counted)
    plan = build_record_plane_plan(state, scan_times_s=times)
    assert len(calls) <= 3  # Reference plus finite B response; independent of 64 pixels.
    monkeypatch.setattr(core, "execute_propagation_plan", execute)
    for index in (0, 16, 32, 48):
        working = copy(state)
        working.simulation_time_s = float(times[0, index])
        actual = _central_orbit(working, 0., targets)
        projected = np.asarray([np.asarray(t.position_offset_m)+offset[0, index]
            for t, offset in zip(plan.transfers, plan.scan_position_offsets_m)])
        np.testing.assert_allclose(projected, actual, rtol=2e-12, atol=1e-17)


def test_mapped_recording_traces_central_orbit_at_each_captured_scan_time(monkeypatch):
    from test_vector_field_transport import _state, _map
    state = _state()
    state.simulation_time_s = .11
    state.deflectors = [OscillatingPair("Mapped drive", "mapped_drive", .5, 2.,
        .4, -.1, 0., 0., thickness_mm=.6)]
    _map(state, (0., 0., .05))
    targets = (.3, .6, 1.)
    _planes(monkeypatch, targets)
    times = np.asarray([[0., .25, .75]])
    plan = build_record_plane_plan(state, scan_times_s=times)
    for index, time in enumerate(times.ravel()):
        working = copy(state)
        working.simulation_time_s = float(time)
        actual = _central_orbit(working, 0., targets)
        projected = np.asarray([np.asarray(t.position_offset_m)+offset[0, index]
            for t, offset in zip(plan.transfers, plan.scan_position_offsets_m)])
        np.testing.assert_allclose(projected, actual, rtol=3e-12, atol=1e-17)
    assert "Local fixed Jacobian" in record_plane.record_plane_plan_provenance(plan)["scan_mapping_scope"]
