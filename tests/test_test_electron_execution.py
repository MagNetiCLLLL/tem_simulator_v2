"""Real-process isolation, unchanged numerical results and owned lifecycle."""
from dataclasses import fields, replace
import io
import pickle
from queue import Queue
import sys
from threading import Event, Thread, Timer
import time
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from temsim.cpu_resources import numerical_job
from temsim.magnetic_test_particle import TestElectronSettings, trace_test_electron
from temsim.test_electron_execution import (
    ElectronExecutionBackend, ElectronExecutionCancelled, ElectronExecutionError,
    RemoteElectronScene, _dumps,
)


@pytest.fixture(scope="module")
def captured():
    from temsim.optics.column import default_state
    from temsim.magnetic_field_scene import prepare_magnetic_scene
    from temsim.test_electron_scene import prepare_test_electron_scene
    with numerical_job(1):
        state = default_state()
        magnetic = prepare_magnetic_scene(state, z_limits_mm=(0., 3026.4))
        scene = prepare_test_electron_scene(state, magnetic, z_limits_mm=(0., 3026.4))
    settings = TestElectronSettings(kinetic_energy_ev=scene.initial_energy_ev,
        position_m=scene.initial_position_m, max_path_length_m=.01, step_m=.001, max_steps=20000)
    return state, magnetic, scene, settings


@pytest.fixture(scope="module")
def prepared(captured):
    backend = ElectronExecutionBackend()
    state, magnetic, _, _ = captured
    remote = backend.prepare(state, magnetic, z_limits_mm=(0., 3026.4))
    yield backend, remote
    backend.close()


def assert_same_result(actual, expected):
    for item in fields(expected):
        left, right = getattr(actual, item.name), getattr(expected, item.name)
        if isinstance(right, np.ndarray):
            assert np.array_equal(left, right), item.name
            assert not left.flags.writeable
        else:
            assert left == right, item.name


def test_constructor_is_lazy_and_close_is_idempotent():
    backend = ElectronExecutionBackend()
    assert backend.process_id is None
    backend.close()
    backend.close()
    with pytest.raises(ElectronExecutionCancelled):
        backend.start()


def test_cold_start_owns_actual_interpreter_pid_and_preserves_virtual_environment():
    backend = ElectronExecutionBackend()
    responses = []
    worker = Thread(target=lambda: responses.append(backend.start()))
    worker.start()
    try:
        worker.join(timeout=15.)
        assert not worker.is_alive(), "Cold numerical startup stalled"
        info, = responses
        assert info["process_id"] == backend.process_id
        assert info["python_executable"] == sys.executable
        assert info["python_prefix"] == sys.prefix
        assert info["cpu"]["numba_threads"] == info["cpu"]["blas_threads"] == 1
        process = backend._process
        backend.close()
        assert process.poll() is not None
    finally:
        backend.close()
        worker.join(timeout=2.)


def test_captured_mapping_proxy_graph_is_preserved():
    proxy = MappingProxyType({"upstream": {"voltage": 300.}})
    value = pickle.loads(_dumps((proxy, proxy)))
    assert isinstance(value[0], MappingProxyType)
    assert value[0] is value[1]
    with pytest.raises(TypeError):
        value[0]["source"] = "new"


def test_real_preparation_returns_metadata_and_reuses_one_process(prepared, captured):
    backend, remote = prepared
    _, _, local, settings = captured
    assert isinstance(remote, RemoteElectronScene)
    assert not hasattr(remote, "electric_provider")
    assert remote.initial_position_m == local.initial_position_m
    assert remote.initial_energy_ev == local.initial_energy_ev
    assert remote.default_path_length_m == local.default_path_length_m
    pid = backend.process_id
    with numerical_job(1):
        expected = trace_test_electron(local, settings)
    actual = backend.trace(remote, settings)
    assert_same_result(actual, expected)
    changed = replace(settings, polar_angle_deg=3., azimuth_angle_deg=40.)
    with numerical_job(1):
        expected = trace_test_electron(local, changed)
    assert_same_result(backend.trace(remote, changed), expected)
    assert backend.process_id == pid
    assert backend.owns_scene(remote)
    receipt = backend.start()
    assert receipt["cpu"]["numerical_thread_budget"] == 1
    assert receipt["process_id"] == backend.process_id


def test_cancelled_trace_retains_prepared_fields_and_accepts_next_electron(prepared, captured):
    backend, remote = prepared
    settings = captured[3]
    cancel = Event()
    timer = Timer(.1, cancel.set)
    pid = backend.process_id
    timer.start()
    started = time.perf_counter()
    try:
        with pytest.raises(ElectronExecutionCancelled):
            backend.trace(remote, replace(settings, max_path_length_m=3., step_m=1e-7,
                                           max_steps=200000), cancelled=cancel.is_set)
    finally:
        timer.cancel()
    assert time.perf_counter()-started < 2.
    assert backend.process_id == pid and backend.owns_scene(remote)
    assert backend.trace(remote, settings).reason == "path_limit"


def test_other_owner_cannot_use_scene(prepared, captured):
    other = ElectronExecutionBackend()
    try:
        with pytest.raises(ElectronExecutionError, match="capture the fields again"):
            other.trace(prepared[1], captured[3])
        assert other.process_id is None
    finally:
        other.close()


def test_shared_cpu_admission_can_cancel_before_process_start():
    backend = ElectronExecutionBackend()
    cancellation = Event()
    errors = []

    def waiting():
        try:
            backend.start(cancelled=cancellation.is_set)
        except Exception as exc:
            errors.append(exc)

    with numerical_job(1):
        thread = Thread(target=waiting)
        thread.start()
        time.sleep(.075)
        assert backend.process_id is None
        cancellation.set()
        thread.join(timeout=2.)
    backend.close()
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], ElectronExecutionCancelled)


def test_missing_magnetic_scene_is_prepared_in_child(captured):
    backend = ElectronExecutionBackend()
    try:
        remote = backend.prepare(captured[0], None, z_limits_mm=(0., 3026.4))
        with numerical_job(1):
            expected = trace_test_electron(captured[2], captured[3])
        assert_same_result(backend.trace(remote, captured[3]), expected)
    finally:
        backend.close()


def test_replaced_scene_handle_is_rejected(captured):
    backend = ElectronExecutionBackend()
    try:
        old = backend.install_prepared(captured[2])
        current = backend.install_prepared(captured[2])
        assert old.token != current.token
        with pytest.raises(ElectronExecutionError, match="replaced"):
            backend.trace(old, captured[3])
        assert backend.trace(current, captured[3]).reason == "path_limit"
    finally:
        backend.close()


def test_process_loss_is_explicit_and_recapture_restores_execution(captured):
    backend = ElectronExecutionBackend()
    try:
        old = backend.install_prepared(captured[2])
        process = backend._process
        process.terminate()
        process.wait(timeout=2.)
        with pytest.raises(ElectronExecutionError, match="capture the fields again"):
            backend.trace(old, captured[3])
        current = backend.install_prepared(captured[2])
        assert not backend.owns_scene(old)
        assert backend.owns_scene(current)
        assert backend.trace(current, captured[3]).reason == "path_limit"
    finally:
        backend.close()


def test_cancelled_preparation_discards_child_and_next_preparation_can_start(captured):
    backend = ElectronExecutionBackend()
    cancel = Event()
    timer = Timer(.075, cancel.set)
    timer.start()
    started = time.perf_counter()
    try:
        with pytest.raises(ElectronExecutionCancelled):
            backend.prepare(captured[0], captured[1], z_limits_mm=(0., 3026.4), cancelled=cancel.is_set)
        assert time.perf_counter()-started < 2.
        assert backend.process_id is None
        remote = backend.install_prepared(captured[2])
        assert backend.trace(remote, captured[3]).reason == "path_limit"
    finally:
        timer.cancel()
        backend.close()


def test_process_crash_during_work_returns_error_without_hanging(captured):
    backend = ElectronExecutionBackend()
    remote = backend.install_prepared(captured[2])
    process = backend._process
    timer = Timer(.1, process.terminate)
    timer.start()
    try:
        with pytest.raises(ElectronExecutionError, match="closed its output"):
            backend.trace(remote, replace(captured[3], max_path_length_m=3., step_m=1e-7, max_steps=200000))
        assert not backend.owns_scene(remote)
        current = backend.install_prepared(captured[2])
        assert backend.trace(current, captured[3]).reason == "path_limit"
    finally:
        timer.cancel()
        backend.close()


def test_close_interrupts_active_trace_without_orphan_process(captured):
    backend = ElectronExecutionBackend()
    remote = backend.install_prepared(captured[2])
    errors = []

    def trace():
        try:
            backend.trace(remote, replace(captured[3], max_path_length_m=3., step_m=1e-7, max_steps=200000))
        except Exception as exc:
            errors.append(exc)

    thread = Thread(target=trace)
    thread.start()
    time.sleep(.1)
    started = time.perf_counter()
    backend.close()
    thread.join(timeout=2.)
    assert time.perf_counter()-started < 2.
    assert not thread.is_alive() and backend.process_id is None
    assert errors and isinstance(errors[0], ElectronExecutionCancelled)


def test_real_streamed_prefixes_arrive_before_unchanged_final_result(prepared, captured):
    backend, remote = prepared
    # A deliberately resolved spatial budget keeps this real trace longer
    # than the 80 ms publication interval even on the compiled fast path.
    settings = replace(captured[3], max_path_length_m=.03, step_m=1e-6, max_steps=100000,
                       polar_angle_deg=2., azimuth_angle_deg=35.)
    progress = []
    final_returned = False

    def observe(prefix):
        assert not final_returned
        assert prefix.reason == "in_progress" and not prefix.completed
        for item in fields(prefix):
            array = getattr(prefix, item.name)
            if isinstance(array, np.ndarray):
                assert not array.flags.writeable, item.name
        progress.append(prefix)

    actual = backend.trace(remote, settings, progress=observe)
    final_returned = True
    with numerical_job(1):
        expected = trace_test_electron(captured[2], settings)
    assert_same_result(actual, expected)
    assert progress, "Actual E+B integration must report progress before completion"
    assert all(left.steps < right.steps for left, right in zip(progress, progress[1:]))
    for prefix in progress:
        count = len(prefix.positions_m)
        assert count <= len(actual.positions_m)
        for name in ("positions_m", "momentum_kg_m_per_s", "time_s", "path_length_m",
                     "kinetic_energy_ev", "electrostatic_potential_v"):
            assert np.array_equal(getattr(prefix, name), getattr(actual, name)[:count]), name


def test_cancel_during_real_progress_reuses_child_without_leaking_old_prefixes(prepared, captured):
    backend, remote = prepared
    cancelled = Event()
    seen = []
    pid = backend.process_id

    def cancel_on_prefix(prefix):
        seen.append(prefix)
        cancelled.set()

    with pytest.raises(ElectronExecutionCancelled):
        backend.trace(remote, replace(captured[3], max_path_length_m=3., step_m=1e-7, max_steps=200000),
                      cancelled=cancelled.is_set, progress=cancel_on_prefix)
    assert len(seen) == 1
    assert backend.process_id == pid and backend.owns_scene(remote)
    followup_progress = []
    changed = replace(captured[3], polar_angle_deg=7., azimuth_angle_deg=70.)
    actual = backend.trace(remote, changed, progress=followup_progress.append)
    with numerical_job(1):
        expected = trace_test_electron(captured[2], changed)
    assert_same_result(actual, expected)
    for prefix in followup_progress:
        assert np.array_equal(prefix.positions_m, actual.positions_m[:len(prefix.positions_m)])


@pytest.fixture
def protocol_backend(monkeypatch):
    """Private-protocol fixture; real child/physics behavior is tested above."""
    backend = ElectronExecutionBackend()
    backend._responses = Queue()
    backend._process = SimpleNamespace(stdin=io.BytesIO(), pid=-1, poll=lambda: None)
    stopped = []
    monkeypatch.setattr(backend, "_start", lambda: None)
    monkeypatch.setattr(backend, "_stop_process", lambda: stopped.append(True))
    yield backend, stopped
    backend.close()


def test_cancel_drains_queued_progress_and_terminal_before_next_request(protocol_backend):
    backend, _ = protocol_backend
    cancelled = Event()
    seen = []
    for response in ((1, "progress", "first"), (1, "progress", "queued"),
                     (1, "ok", "obsolete final"), (2, "ok", "new final")):
        backend._responses.put(response)

    def cancel(value):
        seen.append(value)
        cancelled.set()

    with pytest.raises(ElectronExecutionCancelled):
        backend._request(("trace",), cancelled=cancelled.is_set, progress=cancel)
    assert seen == ["first"]
    assert backend._request(("trace",)) == "new final"


def test_stale_progress_identity_is_rejected_before_observer(protocol_backend):
    backend, stopped = protocol_backend
    backend._responses.put((0, "progress", "stale"))
    seen = []
    with pytest.raises(ElectronExecutionError, match="identity"):
        backend._request(("trace",), progress=seen.append)
    assert not seen and stopped


def test_progress_consumer_failure_drains_terminal_and_preserves_next_request(protocol_backend):
    backend, stopped = protocol_backend
    for response in ((1, "progress", "first"), (1, "progress", "queued"),
                     (1, "cancelled", None), (2, "ok", "new result")):
        backend._responses.put(response)
    calls = []

    def invalid_observer(value):
        calls.append(value)
        raise ValueError("failed display observer")

    with pytest.raises(ElectronExecutionError, match="consumer failed") as exc:
        backend._request(("trace",), progress=invalid_observer)
    assert isinstance(exc.value.__cause__, ValueError)
    assert calls == ["first"] and not stopped
    assert backend._request(("trace",)) == "new result"


def test_child_error_after_progress_does_not_return_prefix_as_final(protocol_backend):
    backend, stopped = protocol_backend
    for response in ((1, "progress", "partial"), (1, "error", "actual child error"),
                     (2, "ok", "new result")):
        backend._responses.put(response)
    seen = []
    with pytest.raises(ElectronExecutionError, match="actual child error"):
        backend._request(("trace",), progress=seen.append)
    assert seen == ["partial"] and not stopped
    assert backend._request(("trace",)) == "new result"


def test_unrequested_progress_is_protocol_error(protocol_backend):
    backend, stopped = protocol_backend
    backend._responses.put((1, "progress", "unexpected"))
    with pytest.raises(ElectronExecutionError, match="Unrequested"):
        backend._request(("trace",))
    assert stopped


def test_unfinished_prefix_cannot_be_returned_as_accepted_terminal_result(prepared, captured, monkeypatch):
    backend, remote = prepared
    monkeypatch.setattr(backend, "_request", lambda *args, **kwargs: SimpleNamespace(reason="in_progress"))
    with pytest.raises(ElectronExecutionError, match="unfinished prefix as its final"):
        backend.trace(remote, captured[3])
