"""Bounded scheduling/thread masks, not a promise of 50% CPU utilisation."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, get_ident

import pytest

from temsim import cpu_resources as resources


@pytest.fixture
def clean_limits(monkeypatch):
    for name in ("TEMSIM_CPU_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OMP_THREAD_LIMIT"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("available,expected", [(32,16), (10,5), (3,1), (1,1)])
def test_half_available_logical_cpu_budget(monkeypatch, clean_limits, available, expected):
    monkeypatch.setattr(resources, "available_cpu_count", lambda: available)
    assert resources.numerical_thread_budget() == expected
    assert resources.numerical_thread_budget(100) == expected
    assert resources.numerical_thread_budget(1) == 1


@pytest.mark.parametrize("name", ["TEMSIM_CPU_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OMP_THREAD_LIMIT"])
def test_explicit_lower_environment_limit_is_never_raised(monkeypatch, clean_limits, name):
    monkeypatch.setattr(resources, "available_cpu_count", lambda: 32)
    monkeypatch.setenv(name, "3")
    assert resources.numerical_thread_budget() == 3


def test_available_count_uses_affinity_instead_of_machine_total(monkeypatch):
    monkeypatch.setattr(resources.os, "sched_getaffinity", lambda pid: {1, 4, 6}, raising=False)
    assert resources.available_cpu_count() == 3


def test_openmp_nested_environment_retains_its_lower_outer_limit(monkeypatch, clean_limits):
    monkeypatch.setattr(resources, "available_cpu_count", lambda: 32)
    monkeypatch.setenv("OMP_NUM_THREADS", "3,2")
    assert resources.numerical_thread_budget() == 3


def test_numba_mask_is_applied_inside_worker_and_nested_scope_cannot_expand():
    import numba
    from threadpoolctl import threadpool_info
    previous = numba.get_num_threads()
    target = min(4, resources.numerical_thread_budget())
    numba.set_num_threads(1)  # The caller mask must not be the only applied limit.
    main_thread = get_ident()
    def execute():
        with resources.numerical_job(target) as receipt:
            actual = numba.get_num_threads()
            with resources.numerical_job(100) as nested:
                assert numba.get_num_threads() <= target
                assert nested.numerical_thread_budget == receipt.numerical_thread_budget
            assert all(row["num_threads"] == 1 for row in threadpool_info() if row["user_api"] == "blas")
            return get_ident(), actual, receipt
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            thread, actual, receipt = executor.submit(execute).result(timeout=15)
        assert thread != main_thread and actual == target
        assert receipt.numba_threads == target
        assert numba.get_num_threads() == 1
    finally:
        numba.set_num_threads(previous)


def test_multiple_numerical_jobs_share_one_budget_and_waiting_job_can_cancel():
    entered, release, attempted, cancelled = Event(), Event(), Event(), Event()
    def first():
        with resources.numerical_job(1):
            entered.set()
            assert release.wait(10)
    def second():
        attempted.set()
        with resources.numerical_job(1, cancelled=cancelled.is_set):
            pytest.fail("Second numerical task entered before first released the budget")
    with ThreadPoolExecutor(max_workers=2) as executor:
        a = executor.submit(first)
        try:
            assert entered.wait(5)
            b = executor.submit(second)
            assert attempted.wait(5)
            cancelled.set()
            with pytest.raises(resources.NumericalJobCancelled):
                b.result(timeout=5)
        finally:
            release.set()
            a.result(timeout=5)
    with resources.numerical_job(1):
        pass  # Cancellation did not strand the common lease.


def test_qt_runner_enforces_budget_in_worker_not_gui_thread(qtbot):
    import numba
    from test_job_coordination import Worker
    from temsim.gui.job_coordinator import JobCoordinator, CoordinatedPool
    from threadpoolctl import threadpool_info
    coordinator = JobCoordinator(numerical_threads=2)
    pool = CoordinatedPool(coordinator=coordinator)
    rows = []
    gui_thread = get_ident()
    def operation():
        rows.append((get_ident(), numba.get_num_threads(), threadpool_info()))
    pool.start(Worker(operation))
    assert pool.waitForDone(15000)
    assert rows and rows[0][0] != gui_thread
    assert rows[0][1] <= min(2, resources.numerical_thread_budget())
    assert all(row["num_threads"] == 1 for row in rows[0][2] if row["user_api"] == "blas")
    events = coordinator.events.snapshot()
    receipt = next(row for row in events if row["event"] == "cpu_resources")
    assert receipt["blas_threads"] == 1 and receipt["numerical_jobs"] == 1


def test_failed_thread_mask_restore_does_not_strand_the_shared_budget(monkeypatch):
    import numba
    previous = numba.get_num_threads()
    original = numba.set_num_threads
    calls = []
    def fail_restore(count):
        calls.append(count)
        if len(calls) == 2:
            raise RuntimeError("controlled restore failure")
        original(count)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(numba, "set_num_threads", fail_restore)
            with pytest.raises(RuntimeError, match="controlled restore"):
                with resources.numerical_job(1):
                    pass
        def enter_from_another_thread():
            with resources.numerical_job(1):
                return True
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(enter_from_another_thread).result(timeout=5)
    finally:
        original(previous)


def test_fresh_import_honours_explicit_lower_limit_and_late_numba_import_is_safe():
    # A separate process verifies import ordering without mutating the running
    # test process's already-initialised Numba pool configuration.
    environment = dict(os.environ, TEMSIM_CPU_THREADS="2", NUMBA_NUM_THREADS="3")
    script = "import numba; import temsim; from temsim.cpu_resources import numerical_job;\nwith numerical_job() as r: print(r.numerical_thread_budget, numba.get_num_threads())"
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
        env=environment, capture_output=True, text=True, timeout=20, check=True)
    assert result.stdout.strip() == "2 2"


def test_existing_interval_executor_uses_shared_cap_and_serial_child_kernels(monkeypatch):
    from types import SimpleNamespace
    import numba
    from temsim.physics import occupied_axial_refinement as refinement
    # Scheduling-only fixture: no wave/source/field problem is constructed.
    calls, masks = [], []
    class Tree:
        operator = None
        def refine(self, left, right, budget, work):
            masks.append(numba.get_num_threads())
            return 0., False
    def executor(**kwargs):
        calls.append(kwargs["max_workers"])
        return ThreadPoolExecutor(**kwargs)
    monkeypatch.setattr(refinement, "ThreadPoolExecutor", executor)
    monkeypatch.setattr("temsim.physics.occupied_tree_certificate.compress_tree", lambda *args: None)
    work = SimpleNamespace(settings=SimpleNamespace(workers=32, executor="thread"),
        failure=None, abort=lambda error: None)
    cap = min(2, resources.numerical_thread_budget())
    with resources.numerical_job(cap):
        result = refinement._refine_intervals({i: Tree() for i in range(4)},
            [(None, None)] * 5, 1., work)
    assert len(result) == 4
    if cap > 1:
        assert calls == [cap] and masks == [1] * 4
    else:
        assert not calls and masks == [1] * 4
