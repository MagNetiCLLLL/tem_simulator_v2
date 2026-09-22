"""Archive dispatch/UI policy; real serialization is covered by section IO tests."""
from threading import Event, get_ident
from types import SimpleNamespace as NS

import numpy as np
import pytest
from PySide6.QtCore import QSettings, Qt

from temsim.gui import interactive_calculation as gui, calculation_controller as controller
from temsim.gui.main_window import MainWindow
from temsim.particle_section_io import section_archive_summary, section_file_fingerprint


@pytest.fixture
def page(qtbot, monkeypatch, tmp_path):
    monkeypatch.setattr(gui, "QSettings", lambda: QSettings(str(tmp_path / "ui.ini"), QSettings.IniFormat))
    widget = gui.InteractiveCalculationPage()
    qtbot.addWidget(widget)
    yield widget
    widget.shutdown()


@pytest.fixture
def completed():
    state = NS(electron_gun=NS(exit_plane_z_mm=450.), sample=NS(z_mm=1000.))
    checkpoint = NS(gun_dependency_signature="executed-source", gun_trace=NS(exit_bundle=NS(x_m=np.zeros(3))))
    return NS(simulation=NS(section_checkpoint=checkpoint, metrics={
        "section_target_z_mm": 1100., "section_component_keys": (), "section_full_path": True,
        "section_resumable_through_z_mm": 1100., "tuning_quality": "High accuracy",
        "section_reuse_reason": "upstream_inputs_or_model_changed", "section_resume_z_mm": 450.,
    }), state_snapshot=state, signatures={"request": "request"}, model_signature="current")


def test_archive_summary_is_selectable_and_distinguishes_pending_failure_and_saved(page, completed):
    page.expect_section_archive(completed)
    info = section_archive_summary(completed)
    page.set_section_archive_status(dict(info, status="saving"))
    assert "Saving completed" in page.section_archive_status.text()
    assert "Saved at" not in page.section_archive_status.text()
    page.set_section_archive_status(dict(info, status="failed", error="Disk full", operation="save"))
    assert "Saving failed: Disk full" in page.section_archive_status.text()
    assert "calculation remains available" in page.section_archive_status.text()
    saved = dict(info, status="saved", path="F:/local/section.temsection", saved_at_utc="2026-09-19T12:00:00+00:00")
    saved["file_io_seconds"] = 2.5
    saved["file_io_operation"] = "save"
    page.set_section_archive_status(saved)
    for text in ("Z = 1100 mm", "High accuracy", "3 emitted particles", "section.temsection", "2026-09-19T12:00", "did not match"):
        assert text in page.section_archive_status.text()
    assert page.section_archive_status.textInteractionFlags() & Qt.TextSelectableByMouse
    assert "Save operation: 2.500 s" in page.section_archive_status.text()
    page.set_section_archive_status(dict(saved, reused=True, file_io_reused_measurement=True))
    assert "Previous save operation: 2.500 s" in page.section_archive_status.text()
    page.set_section_archive_status(dict(saved, reused=True, file_io_operation="verify_existing"))
    assert "Existing archive verification operation: 2.500 s" in page.section_archive_status.text()
    page.invalidate_section_result()
    assert "Saved earlier" in page.section_archive_status.text()
    assert page.section_archive_load.isEnabled()  # Recovery is available before capture/live tuning.


def test_automatic_save_runs_off_thread_and_duplicate_cache_hit_does_not_write_again(qtbot, monkeypatch, tmp_path, completed):
    calls, states = [], []
    started, release = Event(), Event()
    gui_thread = get_ident()
    path = tmp_path / "saved.temsection"
    def write(result, directory, *, maximum_unpacked_bytes):
        assert maximum_unpacked_bytes == owner.section_file_pool.coordinator.ram_budget_bytes
        calls.append(get_ident())
        started.set()
        assert release.wait(5.)
        path.write_bytes(b"IO routing fixture; not a serialized physical result")
        return dict(section_archive_summary(result), path=str(path), saved_at_utc="2026-09-19T12:00:00+00:00",
                    _file_fingerprint=section_file_fingerprint(path), _package_digest="IO-routing-fixture")
    monkeypatch.setattr("temsim.particle_section_io.archive_section_result", write)
    owner = controller.CalculationController(persistent_cache_enabled=False, artifact_cache_root=tmp_path)
    owner.section_archive_changed.connect(states.append)
    try:
        identity = owner.archive_completed_section(completed)
        qtbot.waitUntil(started.is_set, timeout=5000)
        assert states[-1]["status"] == "saving"
        assert calls == [calls[0]] and calls[0] != gui_thread
        assert owner.archive_completed_section(completed) == identity
        release.set()
        qtbot.waitUntil(lambda: any(row["status"] == "saved" for row in states), timeout=5000)
        assert owner.section_file_pool.waitForDone(3000)
        owner.archive_completed_section(completed)
        assert len(calls) == 1
        assert states[-1]["reused"]
        assert states[-1]["file_io_seconds"] >= 0.
        assert states[-1]["file_io_reused_measurement"]
        assert states[-1]["file_io_operation"] == "save"
    finally:
        release.set()
        owner.section_file_pool.waitForDone(3000)


def test_background_save_failure_does_not_report_saved_or_discard_result(qtbot, monkeypatch, tmp_path, completed):
    monkeypatch.setattr("temsim.particle_section_io.archive_section_result",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("Read-only directory")))
    owner = controller.CalculationController(persistent_cache_enabled=False, artifact_cache_root=tmp_path)
    events = []
    owner.section_archive_changed.connect(events.append)
    owner.archive_completed_section(completed)
    qtbot.waitUntil(lambda: any(row["status"] == "failed" for row in events), timeout=5000)
    assert owner.section_file_pool.waitForDone(3000)
    assert not any(row["status"] == "saved" for row in events)
    assert completed.simulation.section_checkpoint is not None


@pytest.mark.parametrize("mismatch", ["model", "pending", "target", "no_checkpoint"])
def test_main_window_never_automatically_archives_stale_or_incomplete_result(page, completed, monkeypatch, mismatch):
    saved = []
    shell = NS(workspace=NS(interactive_calculation=page), state=object(), _interactive_preview_pending=False,
               calculations=NS(archive_completed_section=saved.append))
    monkeypatch.setattr("temsim.calculation_cache.state_model_signature", lambda state: "current")
    if mismatch == "model":
        completed.model_signature = "older"
    elif mismatch == "pending":
        shell._interactive_preview_pending = True
    elif mismatch == "target":
        monkeypatch.setattr(page, "segment_request", lambda: {"target_z_mm": 1050., "component_keys": ()})
    else:
        completed.simulation.section_checkpoint = None
    MainWindow._archive_matching_particle_result(shell, completed)
    assert saved == []


def test_normal_completed_high_accuracy_is_archived_with_section_checkbox_off(page, completed, monkeypatch):
    saved = []
    shell = NS(workspace=NS(interactive_calculation=page), state=object(), _interactive_preview_pending=False,
               calculations=NS(archive_completed_section=saved.append))
    monkeypatch.setattr("temsim.calculation_cache.state_model_signature", lambda state: "current")
    assert page.segment_request() is None and not page._live_mode
    MainWindow._archive_matching_particle_result(shell, completed)
    assert saved == [completed]


def test_final_button_passes_empty_participant_section_without_live_mode(page):
    submitted = []
    page.section_group.setChecked(True)
    page.section_z.setValue(1100.)
    shell = NS(workspace=NS(interactive_calculation=page), preview_timer=NS(stop=lambda: None),
               calculations=NS(submit_background=lambda *args, **kwargs: submitted.append((args, kwargs))),
               high_rays=NS(value=lambda: 5000), high_step=NS(value=lambda: .2), state=object())
    MainWindow.run_high_accuracy(shell)
    assert submitted[0][0][1:] == ("High accuracy", 5000, .2)
    assert submitted[0][1]["section_request"] == {"target_z_mm": 1100., "component_keys": ()}
    assert not page._live_mode


@pytest.mark.parametrize("quality", ["Preview", "Medium", "High accuracy"])
def test_loaded_restart_state_never_enters_complete_result_cache_and_shares_its_budget(
        qtbot, monkeypatch, completed, quality):
    # Exact integer sizes isolate admission policy from Python allocator noise.
    monkeypatch.setattr(controller, "retained_memory_inventory", lambda value: {id(value): 700})
    owner = controller.CalculationController(persistent_cache_enabled=False,
        high_cache_budget_bytes=1000, tuning_cache_budget_bytes=1000)
    complete = NS(signatures={"request": "completed"})
    if quality == "High accuracy":
        owner._cache_result(complete)
    else:
        owner._cache_tuning_result(quality, complete)
    completed.simulation.metrics["tuning_quality"] = quality
    completed.loaded_section_only = True
    owner.retain_section_seed(completed)
    # The restart archive has no complete EDS/STEM/filter products, even if its
    # request signature matches. It is offered only to the transport worker.
    assert not owner._high_cache and not owner._tuning_cache and not owner._tuning_seeds
    loaded = owner._loaded_section_seeds if quality == "High accuracy" else owner._loaded_tuning_sections
    assert tuple(loaded.values()) == (completed,)
    category = "high" if quality == "High accuracy" else "tuning"
    assert owner.cache_statistics()[f"{category}_used_bytes"] == 700
    owner.configure_cache(**{f"{category}_cache_budget_bytes": 0})
    assert not loaded
    assert owner.cache_statistics()[f"{category}_used_bytes"] == 0


@pytest.mark.parametrize("quality", ["Preview", "High accuracy"])
def test_loaded_and_completed_histories_share_entry_limit(qtbot, monkeypatch, completed, quality):
    from copy import deepcopy
    monkeypatch.setattr(controller, "retained_memory_inventory", lambda value: {id(value): 1})
    owner = controller.CalculationController(persistent_cache_enabled=False,
        high_cache_limit=3, tuning_cache_limit=3)
    completed.simulation.metrics["tuning_quality"] = quality
    for index in range(3):
        item = deepcopy(completed)
        item.signatures["request"] = str(index)
        owner.retain_section_seed(item)
    category = "high" if quality == "High accuracy" else "tuning"
    owner.configure_cache(**{f"{category}_cache_limit": 2})
    loaded = owner._loaded_section_seeds if quality == "High accuracy" else owner._loaded_tuning_sections
    assert len(loaded) == 2
    complete = NS(signatures={"request": "complete"})
    if quality == "High accuracy":
        owner._cache_result(complete)
        assert len(loaded) + len(owner._high_cache) <= 2
    else:
        owner._cache_tuning_result(quality, complete)
        assert len(loaded) + len(owner._tuning_cache) <= 2


def test_decoded_archive_rejected_by_memory_budget_reports_failure_to_waiting_page(
        qtbot, monkeypatch, page, completed, tmp_path):
    monkeypatch.setattr("temsim.working_point.WorkingPointArchiveIndex.read",
                        lambda *args, **kwargs: NS(unpacked_size_bytes=1024))
    owner = controller.CalculationController(persistent_cache_enabled=False,
                                              high_cache_budget_bytes=0)
    workers, states = [], []
    monkeypatch.setattr(owner.section_file_pool, "start", workers.append)
    owner.section_archive_changed.connect(page.set_section_archive_status)
    owner.section_archive_changed.connect(states.append)
    owner.load_section_archive(tmp_path / "executed.temsection")
    assert "Loading saved" in page.section_archive_status.text()
    token = workers[0].token
    path = tmp_path / "executed.temsection"
    path.write_bytes(b"Load admission routing fixture")
    info = dict(section_archive_summary(completed, path=path),
                _file_fingerprint=section_file_fingerprint(path), _package_digest="load-routing-fixture")
    assert info["identity"] != token
    # Decoding succeeded; actual controller admission rejects its retained size.
    owner._section_file_completed(token, {"result": completed, "info": info})
    assert "Loading failed:" in page.section_archive_status.text()
    assert "cache budget" in page.section_archive_status.text()
    assert states[-1]["identity"] == token and states[-1]["status"] == "failed"
    assert not owner._loaded_section_seeds
    owner._section_file_finished(token)


@pytest.mark.parametrize("scenario", ["newer_session", "better_signature", "deeper_archive", "multiple_archives"])
def test_high_section_worker_selects_matching_deepest_seed_without_reverting_to_loaded_origin(
        qtbot, monkeypatch, scenario):
    """Routing fixtures only; no particle integration or fabricated continuation."""
    from temsim.calculation_cache import calculation_signatures, state_model_signature
    from temsim.optics.column import default_state
    from temsim.simulation_pipeline import CalculationResult
    state = default_state()
    owner = controller.CalculationController(persistent_cache_enabled=False)
    signatures = calculation_signatures(owner._calculation_snapshot(state, "High accuracy", 49, 1.))
    model = state_model_signature(state)
    def seed(z, *, matched=True, name):
        checkpoint = NS(gun_dependency_signature="selection-fixture", gun_trace=NS(exit_bundle=NS(x_m=np.zeros(3))))
        simulation = NS(incident=object(), incident_plan=object(), gun_trace=object(),
            incident_checkpoints=NS(z_mm=np.asarray([450., z])), section_checkpoint=checkpoint,
            metrics={"tuning_quality":"High accuracy", "section_target_z_mm":z,
                     "section_resumable_through_z_mm":z, "section_component_keys":()})
        return CalculationResult(simulation=simulation, energy_filter=None, state_snapshot=state,
            model_signature=model, signatures={"request":name,
                "incident": signatures["incident"] if matched else "changed-upstream-inputs"})
    loaded = seed(600., name="loaded")
    current = seed(1000., name="current")
    expected = current
    if scenario == "better_signature":
        current = seed(1000., matched=False, name="current")
        expected = loaded
    elif scenario == "deeper_archive":
        loaded = seed(1100., name="loaded")
        expected = loaded
    elif scenario == "multiple_archives":
        current = seed(500., matched=False, name="current")
        expected = loaded
    owner.retain_section_seed(loaded)
    if scenario == "multiple_archives":
        owner.retain_section_seed(seed(1150., matched=False, name="newer-but-incompatible"))
    owner._cache_result(current)
    workers = []
    monkeypatch.setattr(owner.pool, "start", workers.append)
    owner.submit(state, "High accuracy", 49, 1., section_request={
        "target_z_mm":1200., "component_keys":()})
    assert len(workers) == 1
    assert workers[0].existing_result is expected
    owner.invalidate_pending(include_explicit=True)
