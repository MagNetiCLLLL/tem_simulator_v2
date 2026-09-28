"""Identity plumbing fixtures; no field solve, child process or particle run."""
from copy import deepcopy
from dataclasses import replace
import io
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.diagnostic_execution_identity import transport_context_identity, trajectory_execution_identity
from temsim.magnetic_test_particle import TestElectronSettings, TestElectronTrajectory, trace_test_electron
from temsim.optics.model import Aperture
from temsim.test_electron_execution import _scene_metadata, _send, _receive
from temsim.test_electron_scene import _Bore


def scene():
    bounds = np.array(((-.01, -.01, 0.), (.01, .01, .02)))
    result = SimpleNamespace(physical_identity="physical-fixture", numerical_identity="numerical-fixture",
        diagnostic_bounds_m=bounds, electric_bounds_m=bounds.copy(), bounds_m=bounds,
        _apertures=(Aperture("Test aperture", "test_aperture", 5., .1, enabled=True),),
        _bores=(_Bore("test_wall", 0., .02, .01),), _flat_cathode=True, _post_exit_ground=False,
        _unsupported_stops=(), initial_position_m=(0., 0., 0.), initial_energy_ev=.3,
        default_path_length_m=.02, notes=("Fixture only",), support_metadata=({"kind": "fixture"},))
    result.transport_identity = transport_context_identity(result)
    return result


def trajectory(reason="domain_exit"):
    points = np.array(((0., 0., 0.), (0., 0., .01)))
    return TestElectronTrajectory(points, np.tile([0., 0., 1.], (2, 1)), np.array([0., 1e-9]),
        np.array([0., .01]), 0., 0., reason, reason != "in_progress", 1, np.array([.3, .3]),
        np.zeros(2), np.ones(2), np.tile([0., 0., 1e-25], (2, 1)), ("Fixture only",))


@pytest.mark.parametrize("mutation", [
    lambda value: setattr(value, "numerical_identity", "other-grid"),
    lambda value: setattr(value._apertures[0], "radius_mm", .2),
    lambda value: setattr(value._apertures[0], "offset_x_mm", .01),
    lambda value: setattr(value, "_bores", (replace(value._bores[0], inner_m=.009),)),
    lambda value: setattr(value, "_unsupported_stops", ((.01, "unsupported_field:test"),)),
    lambda value: value.diagnostic_bounds_m.__setitem__((1, 2), .019),
])
def test_consumed_field_stop_and_support_changes_invalidate_transport_context(mutation):
    original = scene()
    changed = deepcopy(original)
    mutation(changed)
    assert transport_context_identity(changed) != original.transport_identity


def test_unknown_fields_or_custom_aperture_laws_do_not_gain_reproducible_identity():
    original = scene()
    original.numerical_identity = None
    assert transport_context_identity(original) is None
    original = scene()
    original._apertures[0].transmission_mask = lambda x, y: np.ones_like(x, dtype=bool)
    assert transport_context_identity(original) is None


def test_class_level_aperture_override_remains_unknown(monkeypatch):
    original = scene()
    assert original.transport_identity is not None
    monkeypatch.setattr(Aperture, "transmission_mask", lambda self, x, y: np.ones_like(x, dtype=bool), raising=False)
    assert transport_context_identity(original) is None


@pytest.mark.parametrize("change", [dict(kinetic_energy_ev=.31), dict(position_m=(1e-9, 0., 0.)),
    dict(polar_angle_deg=.1), dict(azimuth_angle_deg=45.), dict(max_path_length_m=.03),
    dict(step_m=2e-4), dict(max_steps=13000), dict(relative_tolerance=1e-5),
    dict(position_tolerance_m=2e-12)])
def test_every_initial_state_and_integration_control_changes_execution_identity(change):
    original = scene()
    settings = TestElectronSettings()
    assert trajectory_execution_identity(original, settings) != trajectory_execution_identity(original, replace(settings, **change))


def test_worker_tokens_and_display_names_do_not_replace_physical_identity():
    original = scene()
    first = _scene_metadata(original, "token-one", "process-one")
    second = _scene_metadata(original, "token-two", "process-two")
    assert first.token != second.token
    assert first.process_identity != second.process_identity
    assert first.physical_identity == second.physical_identity == original.physical_identity
    assert first.numerical_identity == second.numerical_identity == original.numerical_identity
    settings = TestElectronSettings()
    assert trajectory_execution_identity(first, settings) == trajectory_execution_identity(second, settings)
    renamed = replace(second, notes=("Renamed or recoloured display",))
    assert trajectory_execution_identity(first, settings) == trajectory_execution_identity(renamed, settings)
    assert trajectory_execution_identity(first, settings, use_compiled=False) != trajectory_execution_identity(first, settings, use_compiled=True)


def test_missing_context_remains_unknown_despite_live_ownership_token():
    original = scene()
    original.transport_identity = None
    remote = _scene_metadata(original, "live-token", "live-process")
    assert trajectory_execution_identity(remote, TestElectronSettings()) is None


def test_private_protocol_keeps_reproducible_metadata_separate_from_ownership():
    original = _scene_metadata(scene(), "private-token", "private-process")
    stream = io.BytesIO()
    _send(stream, (1, "ok", original))
    stream.seek(0)
    restored = _receive(stream)[2]
    assert restored == original
    assert restored.transport_identity == original.transport_identity
    assert restored.support_metadata == original.support_metadata


@pytest.mark.parametrize("electric", [False, True])
def test_progress_and_final_share_real_execution_identity_without_changing_arrays(monkeypatch, electric):
    original = scene()
    original.has_electric_field = electric
    original.diagnostic_fields_at_global_position = lambda value: None
    prefix, final = trajectory("in_progress"), trajectory()
    seen = []
    if electric:
        def fake_trace(captured, settings, cancelled, progress, interval, use_compiled):
            assert captured is original
            progress(prefix)
            return final
        monkeypatch.setattr("temsim.magnetic_test_particle._trace_electromagnetic_electron", fake_trace)
    else:
        def fake_trace(captured, settings, *, cancelled, progress, progress_interval_s):
            assert captured is original
            progress(prefix)
            return final
        monkeypatch.setattr("temsim.magnetic_test_particle._trace_magnetic_electron", fake_trace)
    settings = TestElectronSettings()
    result = trace_test_electron(original, settings, progress=seen.append, use_compiled=False)
    assert len(seen) == 1
    assert seen[0].reason == "in_progress" and result.reason == "domain_exit"
    for value in (seen[0], result):
        assert value.physical_field_identity == original.physical_identity
        assert value.numerical_field_identity == original.numerical_identity
        assert value.execution_identity == trajectory_execution_identity(original, settings, use_compiled=False)
    assert seen[0].positions_m is prefix.positions_m
    assert result.positions_m is final.positions_m
    assert prefix.execution_identity is None and final.execution_identity is None
