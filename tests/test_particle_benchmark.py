"""Benchmark receipt guards; all particle arrays here are synthetic fixtures."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.particle_benchmark import (
    OperationCounts, array_inventory, array_receipt, compare_arrays, compare_section_terminals, continuation_planes,
    require_exact_arrays, require_exact_prefix, section_terminal_arrays, source_arrays,
)


def _gun():
    values = np.array([1., 2.])
    bundle = SimpleNamespace(x_m=values*1e-9, y_m=values*2e-9,
        tx_rad=values*1e-3, ty_rad=values*2e-3, energy_offset_ev=values*.1,
        weight=np.array([.4, .6]), ray_id=np.array([11, 29], dtype=np.int64))
    gun = SimpleNamespace(emitter=SimpleNamespace(emission_energy_ev=1., surface_model=None),
                          emit=lambda count: bundle)
    source = source_arrays(gun, 2)
    exit_bundle = SimpleNamespace(**vars(bundle), alive=np.array([True, True]),
                                flight_time_s=values*1e-9)
    history = np.array([[0., 0.], [1e-6, 2e-6]])
    result = SimpleNamespace(z_mm=np.array([0., 450.]), x_m=history, y_m=history*2,
        tx_rad=history*.1, ty_rad=history*.2, flight_time_s=history*.001,
        blocked_z_mm=np.full(2, np.nan), blocked_key=("", ""), exit_bundle=exit_bundle,
        emission_reference={name.removeprefix("launch_"): value for name, value in source.items()
                            if name in {"launch_ray_id", "launch_position_m", "launch_direction", "launch_normal"}})
    return gun, result


def _section(z):
    _, gun = _gun()
    z = np.asarray(z, dtype=np.float64)
    values = z[:, None]*np.array([1e-9, 2e-9])
    checkpoints = SimpleNamespace(z_mm=z, x_m=values, y_m=values*2,
        tx_rad=values*.1, ty_rad=values*.2, flight_time_s=values*.001)
    branch = SimpleNamespace(source_ray_id=gun.exit_bundle.ray_id.copy(),
        ray_weight=gun.exit_bundle.weight.copy(), energy_offset_ev=gun.exit_bundle.energy_offset_ev.copy(),
        alive=np.array([True, True]), blocked_z=np.full(2, np.nan), blocked_key=("", ""))
    segment = SimpleNamespace(name="incident", checkpoints=checkpoints, branch=branch)
    checkpoint = SimpleNamespace(gun_dependency_signature="synthetic-upstream", gun_trace=gun, segments=(segment,))
    return SimpleNamespace(simulation=SimpleNamespace(section_checkpoint=checkpoint))


@pytest.mark.parametrize("field", ["ray_id", "weight", "energy_offset_ev", "x_m", "tx_rad"])
def test_source_identity_covers_ids_weights_energies_and_launch_coordinates(field):
    gun, _ = _gun()
    initial = source_arrays(gun, 2)
    changed = {name: value.copy() for name, value in initial.items()}
    changed[field][0] += 1
    assert array_inventory(initial) != array_inventory(changed)
    with pytest.raises(AssertionError, match=field):
        require_exact_arrays(initial, changed, "Synthetic source")


def test_array_identity_preserves_shape_dtype_signed_zero_and_nan_payloads():
    bits = np.array([0, 1 << 63, 0x7ff8000000000001], dtype=np.uint64)
    values = bits.view(np.float64)
    assert array_receipt(values) == array_receipt(values.copy())
    assert array_receipt(values) != array_receipt(values.reshape(1, 3))
    assert array_receipt(values) != array_receipt(bits)
    altered = values.copy()
    altered[1] = 0.
    report = compare_arrays({"values": values}, {"values": altered})
    assert not report["exact"]
    assert report["fields"]["values"]["maximum_finite_absolute_difference"] == 0.
    with pytest.raises(TypeError, match="plain arrays"):
        array_receipt(np.array([object()], dtype=object))


def test_large_integer_ids_do_not_lose_one_unit_in_comparison():
    first = np.array([2**62], dtype=np.int64)
    report = compare_arrays({"ids": first}, {"ids": first+1})
    assert not report["exact"]
    assert report["fields"]["ids"]["maximum_finite_absolute_difference"] == 1


def test_comparison_reports_deltas_and_missing_layout_without_tolerance_inflation():
    first = {"xy": np.array([1., np.nan, 2.])}
    actual = {"xy": np.array([1.25, np.nan, 2.])}
    assert compare_arrays(first, actual)["fields"]["xy"]["maximum_finite_absolute_difference"] == .25
    assert not compare_arrays(first, {"xy": np.array([1.])})["fields"]["xy"]["same_shape_dtype"]
    with pytest.raises(ValueError, match="different fields"):
        compare_arrays(first, {})


def test_continuation_checks_executed_prefix_and_source_ancestry():
    before, after = _section([450., 460.]), _section([450., 455., 460., 470.])
    report = require_exact_prefix(before, after)
    assert report["exact"] and report["segments"][0]["through_z_mm"] == 460.
    assert section_terminal_arrays(after)["z_mm"] == 470.
    after.simulation.section_checkpoint.segments[0].branch.ray_weight[0] += .01
    with pytest.raises(AssertionError, match="Retained source"):
        require_exact_prefix(before, after)


def test_direct_reference_uses_declared_numeric_guard_but_exact_identity_and_stops():
    expected, actual = _section([450., 470.]), _section([450., 470.])
    assert compare_section_terminals(expected, actual)["exact"]
    cp = actual.simulation.section_checkpoint.segments[0]
    cp.checkpoints.x_m[-1, 0] += 1e-15
    report = compare_section_terminals(expected, actual)
    assert not report["exact"] and report["accepted"]
    assert report["fields"]["x_m"]["atol"] == 1e-14
    cp.branch.alive[0] = False
    assert not compare_section_terminals(expected, actual)["accepted"]
    cp.branch.alive[0] = True
    cp.checkpoints.x_m[-1, 0] += 1e-10
    assert not compare_section_terminals(expected, actual)["accepted"]


@pytest.mark.parametrize("change", ["missing_plane", "clock", "gun_dependency"])
def test_continuation_does_not_accept_reconstructed_or_changed_prefix(change):
    before, after = _section([450., 460.]), _section([450., 460., 470.])
    checkpoint = after.simulation.section_checkpoint
    if change == "missing_plane":
        checkpoint.segments[0].checkpoints.z_mm[1] = 461.
    elif change == "clock":
        checkpoint.segments[0].checkpoints.flight_time_s[0, 0] += 1e-15
    else:
        checkpoint.gun_dependency_signature = "changed"
    with pytest.raises(AssertionError):
        require_exact_prefix(before, after)


def test_cutoffs_are_distinct_after_gun_and_strictly_before_sample(monkeypatch):
    monkeypatch.setattr("temsim.physics.particle_sections.section_limits", lambda state: (450., 3000.))
    state = SimpleNamespace(sample=SimpleNamespace(z_mm=1500.))
    assert continuation_planes(state) == (460., 470.)
    state.sample.z_mm = 450.6
    first, second = continuation_planes(state)
    assert 450. < first < second < 450.6
    with pytest.raises(ValueError, match="positive"):
        continuation_planes(state, 0.)
    state.sample.z_mm = 440.
    with pytest.raises(ValueError, match="positive"):
        continuation_planes(state)


def test_outer_counts_distinguish_execution_disk_load_and_cached_return():
    counts = OperationCounts()
    transport = counts.wrap("gun_transport_calls", lambda: object())
    source = counts.wrap("gun_request_calls", transport)
    source()
    cached = object()
    counts.wrap("gun_request_calls", lambda: cached)()
    counts.wrap("field_build_or_disk_load_calls", lambda: SimpleNamespace(report={"cache_hit": True}))()
    counts.wrap("field_build_or_disk_load_calls", lambda: SimpleNamespace(report={"cache_hit": False}))()
    assert counts.values["gun_transport_calls"] == 1
    assert counts.values["gun_request_calls"] == 2 and counts.values["gun_result_cache_hits"] == 1
    assert counts.values["field_disk_cache_hits"] == counts.values["field_builds"] == 1


def test_column_counter_records_real_resume_index_and_zero_interval_reuse():
    counts = OperationCounts()
    plan = SimpleNamespace(z_mm=np.array([450., 460., 470.]))
    operation = counts.wrap("column_transport_calls", lambda *args, **kwargs: None)
    operation(None, plan, start_index=1)
    operation(None, plan, start_index=2)
    assert counts.values["requested_column_intervals"] == 1
    assert counts.values["column_zero_interval_calls"] == 1
    assert [row["start_z_mm"] for row in counts.column_calls] == [460., 470.]


def test_failed_outer_call_does_not_count_successful_cache_hit():
    counts = OperationCounts()
    def failed():
        raise RuntimeError("Synthetic failure")
    with pytest.raises(RuntimeError, match="Synthetic"):
        counts.wrap("gun_request_calls", failed)()
    assert counts.values["gun_request_calls_failed"] == 1
    assert counts.values["gun_result_cache_hits"] == 0


def test_script_keeps_step_hooks_native_and_uses_owned_process_supervision():
    """Static integration guard; deliberately performs no benchmark execution."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts" / "benchmark_particle_stages.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    strings = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert "_analytic_step" not in strings and "_surface_step" not in strings
    assert "run_bounded" in text and "subprocess.Popen" not in text
    assert "cancelled.is_set" in text
    assert "default=\"gun\"" in text
    assert 'options.output.with_suffix(".artifacts").exists()' in text
    keywords = {node.arg for node in ast.walk(tree) if isinstance(node, ast.keyword)}
    assert "supervisor_reason" in keywords and "NOT_COMPLETE" in strings


@pytest.fixture
def installation_smoke(monkeypatch):
    import importlib.util
    import os
    for name in ("TEMSIM_CPU_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS",
                 "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "QT_QPA_PLATFORM"):
        # Loading this executable sets an explicit smoke budget. Restore the
        # caller's environment afterward; the test performs no transport.
        if name in os.environ:
            monkeypatch.setenv(name, os.environ[name])
        else:
            monkeypatch.delenv(name, raising=False)
    path = Path(__file__).resolve().parents[1] / "scripts/installation_diagnostic_smoke.py"
    spec = importlib.util.spec_from_file_location("installation_diagnostic_smoke_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _smoke_branch(start, stop):
    coordinates = np.zeros((2, 2))
    return SimpleNamespace(z=np.array([start, stop]), x=coordinates.copy(), y=coordinates.copy(),
                           tx=coordinates.copy(), ty=coordinates.copy())


def test_installed_column_endpoint_includes_downstream_branches(installation_smoke):
    simulation = SimpleNamespace(incident=_smoke_branch(0., 1599.2),
                                 branches={"optical": _smoke_branch(1599.2, 3026.4)})
    readout = installation_smoke.column_history_summary(simulation)
    assert readout == {"incident_endpoint_mm": 1599.2, "column_endpoint_mm": 3026.4, "branches_checked": 2}


@pytest.mark.parametrize("name", ["z", "x", "y", "tx", "ty"])
def test_installed_smoke_rejects_nonfinite_downstream_history(installation_smoke, name):
    downstream = _smoke_branch(1599.2, 3026.4)
    getattr(downstream, name).flat[-1] = np.nan
    simulation = SimpleNamespace(incident=_smoke_branch(0., 1599.2), branches={"optical": downstream})
    with pytest.raises(AssertionError, match="branch 1"):
        installation_smoke.column_history_summary(simulation)
