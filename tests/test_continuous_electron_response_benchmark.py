"""Synthetic benchmark accounting/Qt paint tests; no field solve or transport."""
from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np
import pytest


@pytest.fixture
def benchmark():
    path = Path(__file__).resolve().parents[1] / "scripts/validate_continuous_electron_response.py"
    spec = importlib.util.spec_from_file_location("continuous_response_benchmark_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("count", [0, 1, 19, 20, 24])
def test_p95_needs_twenty_actual_samples(benchmark, count):
    report = benchmark.latency_summary([.01] * count)
    assert report["samples"] == count
    assert (report["p95_ms"] is not None) == (count >= 20)
    assert report["status"] == ("MEASURED" if count >= 20 else "INSUFFICIENT_SAMPLES")


@pytest.mark.parametrize("values", [[float("nan")], [float("inf")], [-.1], [[1.]]])
def test_invalid_latency_not_silently_filtered(benchmark, values):
    with pytest.raises(ValueError):
        benchmark.latency_summary(values)


def test_paint_match_requires_time_generation_settings_and_current(benchmark):
    def row(stamp, generation=2, settings="new", state="current"):
        return {"ended": stamp, "tokens": (("electron-1", generation, settings, state, 4, 123),)}
    old = [row(9.), row(11., generation=1), row(12., settings="old"),
           row(13., state="previous"), row(14., state="in_progress")]
    kwargs = dict(started=10., key="electron-1", generation=2, settings="new")
    assert benchmark.matching_paint(old, **kwargs) is None
    actual = row(15.)
    assert benchmark.matching_paint(old + [actual], **kwargs) is actual


def test_display_tokens_keep_executed_older_settings(benchmark):
    terminal = SimpleNamespace(steps=7)
    progress = SimpleNamespace(steps=3)
    record = SimpleNamespace(key="electron-1", settings="latest", trajectory=terminal,
        trajectory_settings="accepted_old", trajectory_generation=4,
        progress_trajectory=progress, progress_key=(4, "sampled_old", "electron-1", 8))
    paths = [SimpleNamespace(key="electron-1", state="previous"),
             SimpleNamespace(key="electron-1/progress", state="previous")]
    values = benchmark.displayed_tokens(SimpleNamespace(records=(record,)), paths)
    assert values[0][:5] == ("electron-1", 4, "accepted_old", "previous", 7)
    assert values[1][:5] == ("electron-1", 4, "sampled_old", "previous", 3)


def test_terminal_inventory_preserves_exact_physical_arrays_and_strides(benchmark):
    from temsim.magnetic_test_particle import TestElectronTrajectory
    positions = np.arange(24, dtype=np.float64).reshape(8, 3)[::2]
    scalars = np.arange(4, dtype=np.float64)
    result = TestElectronTrajectory(
        positions_m=positions, directions=positions + 1., time_s=scalars,
        path_length_m=scalars + 2., energy_invariant_error_ev=0.,
        energy_invariant_relative_error=0., reason="path_limit", completed=True, steps=3,
        kinetic_energy_ev=scalars + 3., electrostatic_potential_v=scalars + 4.,
        speed_m_per_s=scalars + 5., momentum_kg_m_per_s=scalars + 6., notes=())
    before = positions.copy()
    inventory = benchmark.trajectory_array_inventory(result)
    assert set(inventory) == {"positions_m", "directions", "time_s", "path_length_m",
                              "kinetic_energy_ev", "electrostatic_potential_v",
                              "speed_m_per_s", "momentum_kg_m_per_s"}
    assert inventory["positions_m"] == {
        "shape": [4, 3], "dtype": positions.dtype.str, "nbytes": positions.nbytes,
        "sha256": hashlib.sha256(positions.tobytes(order="C")).hexdigest()}
    np.testing.assert_array_equal(positions, before)
    positions[0, 0] = np.nextafter(positions[0, 0], 1.)
    assert benchmark.trajectory_array_inventory(result)["positions_m"]["sha256"] != inventory["positions_m"]["sha256"]


def test_terminal_inventory_rejects_nonreproducible_objects(benchmark):
    @dataclass
    class Synthetic:
        array: np.ndarray

    with pytest.raises(TypeError, match="Object arrays"):
        benchmark.trajectory_array_inventory(Synthetic(np.array([object()])))


def test_reuse_counter_counts_successes_not_refresh_lookups(benchmark):
    probes = benchmark.Probes()
    record = SimpleNamespace(settings="same", accepted=False)
    controller = SimpleNamespace()

    def store(record, result, **_kwargs):
        record.accepted = True

    def reuse(record):
        if record.accepted:
            return False
        controller._store_result(record, object())
        return True

    controller._reuse_result, controller._store_result = reuse, store
    benchmark.instrument_controller(controller, probes)
    assert controller._reuse_result(record)
    for _ in range(12):
        assert not controller._reuse_result(record)
    assert probes.counts["exact_result_cache_hits"] == 1
    assert probes.counts["result_store_calls"] == 1
    assert probes.counts["current_result_stores"] == 1
    controller._store_result(record, object(), settings="previous")
    assert probes.counts["older_result_stores"] == 1
    assert probes.counts["exact_result_cache_hits"] == 1


def test_probes_preserve_failure_and_explicit_nested_accounting(benchmark):
    probes = benchmark.Probes()
    with pytest.raises(RuntimeError):
        with probes.stage("outer"):
            with probes.stage("inner"):
                raise RuntimeError("synthetic")
    assert [row["stage"] for row in probes.spans] == ["inner", "outer"]
    assert all(row["outcome"] == "raised" for row in probes.spans)
    assert all(row["accounting"] == "inclusive; not additive" for row in probes.spans)
    count = len(probes.events.snapshot())
    probes.enabled = False
    with probes.stage("disabled"):
        pass
    assert len(probes.events.snapshot()) == count


@pytest.mark.parametrize("profile_paint", [False, True])
def test_canvas_observes_actual_paint_and_cached_reprojection(benchmark, qtbot, profile_paint):
    from temsim.gui.test_electron_types import ElectronPath
    probes = benchmark.Probes()
    canvas = benchmark.benchmark_canvas_class()(probes, profile_paint=profile_paint)
    qtbot.addWidget(canvas)
    canvas.resize(640, 400)
    canvas.set_electron_mode(True)
    canvas.show()
    qtbot.waitUntil(lambda: bool(canvas.paints))
    assert canvas.profile_paint == profile_paint
    positions = np.array(((0., 0., 0.), (1e-6, 0., .001)))
    path = ElectronPath("electron-1", "Synthetic", "#ffd166", positions)
    tokens = (("electron-1", 1, "settings", "current", 1, 123),)
    started = time.perf_counter()
    canvas.display_tokens = tokens
    canvas.set_electron_paths((path,))
    assert benchmark.matching_paint(canvas.paints, started=started, key="electron-1",
                                   generation=1, settings="settings") is None
    qtbot.waitUntil(lambda: benchmark.matching_paint(canvas.paints, started=started,
        key="electron-1", generation=1, settings="settings") is not None)
    prior = probes.counts.copy()
    canvas.set_electron_paths((path,))
    assert probes.counts["projection_cache_hits"] > prior["projection_cache_hits"]
    assert probes.counts["projection_builds"] == prior["projection_builds"]
    revision = canvas._projection_revision
    canvas.set_projection_angle(45.)
    assert canvas._projection_revision > revision
    assert probes.counts["projection_builds"] > prior["projection_builds"]
    assert any(row["stage"] == "canvas_paint_including_field_raster" for row in probes.spans)
    fine_stages = {row["stage"] for row in probes.spans if row["stage"] in {
        "canvas_axes", "canvas_labels", "canvas_electron_markers", "canvas_footer"}}
    assert fine_stages == ({"canvas_axes", "canvas_labels", "canvas_electron_markers", "canvas_footer"}
                           if profile_paint else set())


@pytest.fixture
def fake_process_tree():
    registry = {}

    class MissingProcess(Exception):
        pass

    class Process:
        def __init__(self, pid, ppid, created=1., rss=100):
            self.pid, self.parent, self.created, self.rss = pid, ppid, created, rss
            self.members = []
            self.reads = self.tree_calls = self.parent_calls = 0
            registry[pid] = self

        def ppid(self):
            self.parent_calls += 1
            return self.parent

        def create_time(self):
            return self.created

        def is_running(self):
            return registry.get(self.pid) is self

        def children(self, *, recursive):
            assert recursive
            self.tree_calls += 1
            return list(self.members)

        def memory_info(self):
            assert self.is_running(), "Must not sample a retired process identity"
            self.reads += 1
            return SimpleNamespace(rss=self.rss)

    def lookup(pid):
        if pid not in registry:
            raise MissingProcess(pid)
        return registry[pid]

    return SimpleNamespace(registry=registry, make=Process,
                           api=SimpleNamespace(Process=lookup, NoSuchProcess=MissingProcess))


def test_rss_discovery_is_bounded_and_new_children_wait_for_next_discovery(benchmark, fake_process_tree):
    tree = fake_process_tree
    parent, owner = tree.make(1, 0), tree.make(2, 1)
    backend = SimpleNamespace(_process=SimpleNamespace(pid=2, poll=lambda: None))
    sampler = benchmark.ResidentMemory(backend)
    for sample in range(50):
        sampler._sample_once(tree.api, parent, sample * .02)
    assert owner.parent_calls == owner.tree_calls == 1
    assert owner.reads == 50
    child = tree.make(3, 2)
    owner.members.append(child)
    sampler._sample_once(tree.api, parent, .99)
    assert child.reads == 0
    sampler._sample_once(tree.api, parent, 1.)
    assert child.reads == 1
    assert sampler.result["tree_discoveries"] == 2
    del tree.registry[3]
    sampler._sample_once(tree.api, parent, 1.02)
    assert child.reads == 1
    assert sampler.result["retired_process_identities"] == 1


def test_rss_rejects_reused_owner_pid_and_drops_removed_owner(benchmark, fake_process_tree):
    tree = fake_process_tree
    parent, owner, child = tree.make(1, 0), tree.make(2, 1), tree.make(3, 2)
    owner.members.append(child)
    backend = SimpleNamespace(_process=SimpleNamespace(pid=2, poll=lambda: None))
    sampler = benchmark.ResidentMemory(backend)
    sampler._sample_once(tree.api, parent, 0.)
    replacement = tree.make(2, 1, created=2.)
    sampler._sample_once(tree.api, parent, .02)
    sampler._sample_once(tree.api, parent, 1.)
    assert owner.reads == child.reads == 1
    assert replacement.reads == 0
    assert sampler._known_processes == {}
    # A different owned process handle authorizes fresh identity discovery.
    backend._process = SimpleNamespace(pid=2, poll=lambda: None)
    sampler._sample_once(tree.api, parent, 1.02)
    assert replacement.reads == 1
    backend._process = None
    sampler._sample_once(tree.api, parent, 1.04)
    assert replacement.reads == 1
    assert sampler._known_processes == {}


def test_rss_never_adopts_an_unowned_process(benchmark, fake_process_tree):
    tree = fake_process_tree
    parent, outsider = tree.make(1, 0), tree.make(2, 77)
    backend = SimpleNamespace(_process=SimpleNamespace(pid=2, poll=lambda: None))
    sampler = benchmark.ResidentMemory(backend)
    sampler._sample_once(tree.api, parent, 0.)
    assert outsider.reads == outsider.tree_calls == 0
    assert sampler.result["owned_child_peak_bytes"] == 0


def test_probe_overhead_is_paired_and_restores_enabled(benchmark):
    probes = benchmark.Probes()
    calls = []

    def reproject():
        calls.append(probes.enabled)
        with probes.stage("electron_reproject"):
            pass

    report = benchmark.probe_overhead(SimpleNamespace(_rebuild_electron_projection=reproject),
                                      probes, repeats=3, calls=4)
    assert Counter(calls) == {False: 12, True: 12}
    assert probes.enabled
    assert len(report["repeats"]) == 3
    assert all("extra_seconds_per_call" in row for row in report["repeats"])


def test_supervisor_timeout_never_preserves_old_pass(benchmark, tmp_path, monkeypatch):
    def timeout(_command, **kwargs):
        assert kwargs["timeout"] == 30.
        benchmark._save(tmp_path / "report.json", {"status": "RUNNING", "case": "B06"})
        return SimpleNamespace(returncode=124)

    monkeypatch.setattr("temsim.validation_process.run_bounded", timeout)
    benchmark._save(tmp_path / "report.json", {"status": "PASS", "old": True})
    assert benchmark.main(["--output", str(tmp_path), "--budget-seconds", "30"]) == 1
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "TIMEOUT"
    assert report["owned_tree_termination_confirmed"]
    assert "old" not in report


@pytest.mark.parametrize("profile_paint", [False, True])
def test_supervisor_forwards_paint_profile_option_only_when_requested(benchmark, tmp_path, monkeypatch, profile_paint):
    def bounded(command, **_kwargs):
        assert ("--profile-paint" in command) == profile_paint
        benchmark._save(tmp_path / "report.json", {"status": "PASS", "case": "B06",
                        "primary_latency_evidence": not profile_paint})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("temsim.validation_process.run_bounded", bounded)
    args = ["--output", str(tmp_path), "--budget-seconds", "30"]
    if profile_paint:
        args.append("--profile-paint")
    assert benchmark.main(args) == 0
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["primary_latency_evidence"] == (not profile_paint)


def test_source_read_failure_still_saves_failure_without_process(benchmark, tmp_path, monkeypatch):
    def fail(_root):
        raise OSError("synthetic source read failure")

    monkeypatch.setitem(sys.modules, "validate_classical_scope",
                        SimpleNamespace(source_hashes=fail, git_metadata=lambda _: {}))
    args = SimpleNamespace(output=tmp_path, budget_seconds=30.)
    assert benchmark.worker(args) == 1
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAILED"
    assert "synthetic source read failure" in report["error"]
    assert report["owned_processes"] == []
    assert report["source_unchanged"] is False
