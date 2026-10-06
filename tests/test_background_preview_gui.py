"""Main-window routing and interaction during detached request preparation."""
from PySide6.QtCore import QSettings
import pytest


@pytest.fixture
def window(qtbot, monkeypatch, tmp_path, request):
    from temsim.gui import main_window as shell, interactive_calculation as gui, calculation_controller as controller
    settings = QSettings(str(tmp_path / "workspace.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(shell, "QSettings", lambda: settings)
    monkeypatch.setattr(gui, "QSettings", lambda: settings)
    monkeypatch.setattr(controller, "default_artifact_cache_root", lambda: tmp_path / "artifacts")
    monkeypatch.setattr(shell.MainWindow, "INITIAL_PREVIEW_DELAY_MS", 60000)
    if getattr(request, "param", None) is not None:
        state = shell.default_state()
        state.acceleration_backend, state.acceleration_enabled = request.param
        monkeypatch.setattr(shell, "default_state", lambda: state)
    instance = shell.MainWindow()
    instance.preview_timer.stop()
    qtbot.addWidget(instance)
    yield instance
    instance.preview_timer.stop()
    instance.calculations.invalidate_pending()
    instance.calculations.pool.waitForDone(3000)


def test_startup_preview_request_keeps_armed_stem_controls_uncomputed(window, monkeypatch):
    from PySide6.QtCore import QSignalBlocker
    from temsim.physics.optical_tuning import TUNING_PROFILES

    page = window.workspace.scan_control
    calls, acquisition_requests = [], []
    monkeypatch.setattr(window.calculations, "submit_background",
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(window.calculations, "submit",
                        lambda *args, **kwargs: pytest.fail("Startup must use detached preview preparation"))
    page.calculation_requested.connect(lambda: acquisition_requests.append(True))
    assert window.state.ac_deflector.enabled and window.state.ac_deflector.scan_enabled
    assert window.state.sample.stem_image_enabled
    assert page.ac_controls["scan_enabled"].isChecked() and page.image_enabled.isChecked()
    assert page._stem_frame is None
    assert all(item.image is None for item in page.detector_image_items.values())
    assert window.workspace._last_result is None
    assert window.workspace._high_accuracy_result is None
    assert not window.result_files.hold_automatic_preview

    # Exercise checkbox changes during binding without editing the live state.
    with QSignalBlocker(page.ac_controls["scan_enabled"]), QSignalBlocker(page.image_enabled):
        page.ac_controls["scan_enabled"].setChecked(False)
        page.image_enabled.setChecked(False)
    page.set_state(window.state)
    assert page.ac_controls["scan_enabled"].isChecked() and page.image_enabled.isChecked()
    assert not calls and not acquisition_requests

    # The fixture stops the delayed startup timer; deliver its signal directly
    # after replacing submission so this regression never executes transport.
    assert window.preview_timer.interval() == window.INITIAL_PREVIEW_DELAY_MS
    assert window.tuning_quality.currentData() == "Preview"
    window.preview_timer.timeout.emit()
    profile = TUNING_PROFILES["Preview"]
    assert calls == [((window.state, "Preview", profile.rays, profile.step_mm),
                     {"particle_tuning": True})]
    assert not acquisition_requests
    assert window.workspace._high_accuracy_result is None
    assert page._stem_frame is None
    assert all(item.image is None for item in page.detector_image_items.values())


@pytest.mark.parametrize("window", [("gpu", False), ("Require GPU", False)], indirect=True)
def test_backend_display_preserves_requested_policy_and_explicit_disabled_flag(window):
    requested = window.state.acceleration_backend
    assert requested in {"gpu", "Require GPU"}
    assert not window.state.acceleration_enabled
    assert window.compute_backend.currentData() == ("CUDA GPU" if requested == "gpu" else requested)
    window._sync_working_point_selectors()
    assert window.state.acceleration_backend == requested
    assert not window.state.acceleration_enabled


def test_toolbar_owns_persisted_particle_population_and_preview_remains_local(window, monkeypatch, tmp_path):
    from temsim.profile_io import save_profile, read_profile
    from temsim.gui.calculation_request import apply_request_numerics
    from temsim.instrument_snapshot import encode_instrument, decode_instrument
    from temsim.runtime_parameters import runtime_targets, editable_parameters
    assert window.high_rays.value() == window.state.electron_gun.ray_count == 3000
    assert window.high_step.value() == window.state.step_mm == 0.1
    changes = []
    monkeypatch.setattr(window, "schedule_preview", lambda parameter, **options: changes.append((parameter, options)))
    window.high_rays.setValue(4000)
    assert window.state.electron_gun.ray_count == 4000
    assert changes == [("ray_count", {"automatic": False})]
    window.high_step.setValue(0.02)
    assert window.high_step.value() == window.state.step_mm == 0.02
    assert changes[-1] == ("step_mm", {"automatic": False})
    snapshot = decode_instrument(encode_instrument(window.state))
    apply_request_numerics(snapshot, "Preview", 49, 1.)
    assert snapshot.electron_gun.ray_count == 49
    assert snapshot.step_mm == 1.
    assert window.high_step.value() == window.state.step_mm == 0.02
    assert window.state.electron_gun.ray_count == window.high_rays.value() == 4000
    path = tmp_path / "ray-count.toml"
    save_profile(path, window.state, window.selection)
    _, values = read_profile(path)
    assert values["feg_tip"]["ray_count"] == 4000
    assert values["simulation"]["step_mm"] == 0.02
    target = runtime_targets(window.state)["feg_tip"]
    assert "ray_count" in {p.name for p in editable_parameters(target)}
    window.parameter_panel._runtime_target = target
    assert not window.parameter_panel._runtime_parameters()
    target = runtime_targets(window.state)["simulation"]
    assert "step_mm" in {p.name for p in editable_parameters(target)}
    window.parameter_panel._runtime_target = target
    assert "step_mm" not in {p.name for p in window.parameter_panel._runtime_parameters()}
    assert "history_step_mm" in {p.name for p in window.parameter_panel._runtime_parameters()}


def test_loaded_ray_count_and_product_quadrature_are_displayed_without_mutation(window, monkeypatch):
    from temsim.optics.electron_gun.emitter import EmissionQuadrature
    changes = []
    monkeypatch.setattr(window, "schedule_preview", lambda *args, **kwargs: changes.append((args, kwargs)))
    emitter = window.state.electron_gun.emitter
    emitter.ray_count = 217
    window.state.step_mm = 2.5
    window._refresh_assembly_views()
    assert window.high_rays.value() == 217 and window.high_rays.isEnabled()
    assert emitter.ray_count == 217 and not changes
    assert window.high_step.value() == window.state.step_mm == 2.5
    emitter.quadrature = EmissionQuadrature(3, 5, 7)
    emitter.ray_count = emitter.quadrature.total
    window._sync_working_point_selectors()
    assert window.high_rays.value() == 105 and not window.high_rays.isEnabled()
    window.high_rays.setValue(1000)
    assert window.high_rays.value() == emitter.ray_count == 105
    assert not changes


@pytest.mark.parametrize("quality", ["Preview", "Medium"])
def test_toolbar_preview_uses_background_preparation(window, monkeypatch, quality):
    from temsim.physics.optical_tuning import TUNING_PROFILES
    assert window.state.electron_gun.emitter.surface_model is None
    assert window.state.electron_gun.emitter.coherence is None
    assert window.state.electron_gun.source_representation == "classical_particles"
    window.tuning_quality.setCurrentIndex(window.tuning_quality.findData(quality))
    window.preview_timer.stop()
    calls = []
    monkeypatch.setattr(window.calculations, "submit_background", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(window.calculations, "submit", lambda *args: pytest.fail("Foreground request preparation"))
    window.run_preview()
    profile = TUNING_PROFILES[quality]
    assert calls == [((window.state, quality, profile.rays, profile.step_mm), {"particle_tuning": True})]
    snapshot = window.calculations._calculation_snapshot(*calls[0][0])
    assert snapshot.electron_gun.emitter.surface_model is None
    snapshot.electron_gun.validate()


def test_tip_entry_points_share_physical_editor_and_invalidate_ray_display(window, monkeypatch):
    from temsim.gui.gun_source_dialog import GunSourceDialog
    from PySide6.QtWidgets import QPushButton
    previous = object()
    window.workspace._last_result = previous
    original = window.state.electron_gun._cache_key(49)
    page = window.workspace.physical_layout.model_editor
    applied = []
    matches = []
    monkeypatch.setattr(window, "apply_direct_alignment", lambda *args: matches.append(args))

    def apply(dialog):
        assert dialog.parentWidget() is page
        assert window.workspace.tabs.currentWidget() is window.workspace.physical_layout
        assert page._selected_key == "feg_tip" and page._mesh_records
        assert all(edit.isReadOnly() for edit in dialog.geometry_inputs.values())
        dialog.surface_enabled.setChecked(not dialog.surface_enabled.isChecked())
        dialog.accept()
        assert dialog.result() == dialog.DialogCode.Accepted, dialog.error.text()
        applied.append(dialog.value())
        return dialog.result()

    monkeypatch.setattr(GunSourceDialog, "exec", apply)
    window.workspace.model_inspector.tip_editor_requested.emit()
    window.preview_timer.stop()
    page._flush_runtime_refresh()
    assert window.state.electron_gun.emitter.surface_model is not None
    assert window.state.electron_gun._cache_key(49) != original
    assert window.workspace._last_result is previous
    assert "awaiting recalculation" in window.workspace.ray_source_status.text()
    assert "Curved tip" in window.workspace.ray_source_status.text()
    assert any("emitting_cap" in mesh["surfaces"] for mesh in page._mesh_records)
    assert not window.parameter_panel.quick_box.isHidden()
    button = next(button for button in window.parameter_panel.quick_box.findChildren(QPushButton)
                  if button.text() == "Open tip in Physical Layout…")
    button.click()
    window.preview_timer.stop()
    page._flush_runtime_refresh()
    assert len(applied) == 2
    assert matches == [("column_transport", .01)]
    assert window.state.electron_gun.emitter.surface_model is None
    assert "Flat tip" in window.workspace.ray_source_status.text()
    emitting = next(mesh for mesh in page._mesh_records if mesh["region"] == "emitting_cap")
    assert (emitting["vertices"][:, 2] == 0).all()
    assert window.workspace._last_result is previous


def test_tip_navigation_preserves_unsaved_geometry(window, monkeypatch):
    from temsim.gui.gun_source_dialog import GunSourceDialog
    window._reveal_physical_model("feg_tip", 0.)
    page = window.workspace.physical_layout.model_editor
    page._open_pending_part()
    page.session.set_dimension(("parts", "feg_tip", "tip_radius_nm"), 120.)
    before = window.state.electron_gun.to_dict()
    monkeypatch.setattr(GunSourceDialog, "exec", lambda _: pytest.fail("Cannot replace a geometry draft"))
    window._open_tip_editor()
    assert page.session.dirty
    assert page.session.part("feg_tip")["tip_radius_nm"] == 120.
    assert window.state.electron_gun.to_dict() == before
    assert "Save or revert" in page.status.text()


def test_capture_error_does_not_leave_live_queue_running(window, monkeypatch):
    def fail_capture(*_args, **_kwargs):
        raise ValueError("Controlled invalid capture")
    errors = []
    monkeypatch.setattr(window.calculations, "submit_background", fail_capture)
    monkeypatch.setattr(window, "_show_error", errors.append)
    old_plot = object()
    window.workspace._last_result = old_plot
    window._interactive_preview_pending = True
    window.run_preview()
    assert errors == ["Controlled invalid capture"]
    assert window._interactive_preview_generation is None
    assert not window._interactive_preview_pending
    assert not window.preview_timer.isActive()
    assert window.workspace._last_result is old_plot


@pytest.mark.parametrize("coherent", [False, True])
def test_rejected_surface_image_keeps_applied_source_and_previous_result(window, qtbot, monkeypatch, coherent):
    from types import SimpleNamespace
    from temsim.gui.gun_source_dialog import GunSourceDialog
    from temsim.instrument_snapshot import capture_instrument_snapshot

    def apply(dialog):
        dialog.surface_enabled.setChecked(True)
        dialog.surface_coherent.setChecked(coherent)
        dialog.match_transport.setChecked(False)  # Exercise image admission without a concurrent alignment.
        dialog.accept()
        assert dialog.result() == dialog.DialogCode.Accepted, dialog.error.text()
        return dialog.result()

    monkeypatch.setattr(GunSourceDialog, "exec", apply)
    window.workspace.model_inspector._edit_gun_source()
    window.preview_timer.stop()
    window.state.ac_deflector.enabled = True
    window.state.ac_deflector.scan_enabled = True
    window.state.sample.stem_wave_enabled = True
    before = capture_instrument_snapshot(window.state).digest
    previous = object()
    window.workspace._last_result = previous
    window.workspace._high_accuracy_result = SimpleNamespace(simulation=object())
    errors = []
    monkeypatch.setattr(window, "_show_error", errors.append)
    monkeypatch.setattr(window.calculations.pool, "start", lambda *_: pytest.fail("Unsupported imaging must not start a worker"))
    window.run_page_calculation("stem")
    assert len(errors) == 1 and "STEM wave imaging" in errors[0]
    assert "Source selection and previous results are unchanged" in errors[0]
    assert capture_instrument_snapshot(window.state).digest == before
    assert window.workspace._last_result is previous
    reopened = GunSourceDialog(window.state.electron_gun, instrument_state=window.state)
    qtbot.addWidget(reopened)
    assert reopened.surface_enabled.isChecked()
    assert reopened.surface_coherent.isChecked() == coherent


def test_high_toolbar_routes_to_background_without_foreground_memory_preparation(window, monkeypatch):
    from PySide6.QtWidgets import QPushButton

    assert window.high_rays.value() == 3000
    calls = []
    window._working_point_parent = "captured-parent-fixture"
    monkeypatch.setattr(window.calculations, "submit_background", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(window.calculations, "submit", lambda *_: pytest.fail("Foreground High preparation"))
    import temsim.gui.main_window as module
    monkeypatch.setattr(module, "estimate_calculation_memory_bytes", lambda *_: pytest.fail("Heavy memory estimation belongs in preparation worker"))
    window.findChild(QPushButton, "highAccuracyButton").click()
    assert len(calls) == 1 and calls[0][0][1] == "High accuracy"
    assert calls[0][0][2] == 3000
    assert calls[0][0][3] == window.high_step.value()
    assert calls[0][1] == {"parent_id": "captured-parent-fixture", "section_request": None,
                            "workflow": "rays", "existing_result": None}


@pytest.mark.parametrize("policy", ["CPU", "Auto", "CUDA GPU"])
def test_toolbar_owns_device_policy_for_ray_wave_and_page_requests(window, monkeypatch, policy):
    from types import SimpleNamespace
    from temsim.runtime_parameters import editable_parameters, runtime_targets
    synchronized = []
    monkeypatch.setattr(window.workspace.magnetic_field.field_lines, "set_compute_policy",
                        lambda *values: synchronized.append(values))
    previous = window.compute_backend.currentData()
    window.compute_backend.setCurrentIndex(window.compute_backend.findData(policy))
    assert window.state.acceleration_backend == policy
    assert window.state.acceleration_enabled is (policy != "CPU")
    if previous != policy:
        assert synchronized == [(policy, policy != "CPU")]
    assert not window.preview_timer.isActive()
    names = {item.name for item in editable_parameters(runtime_targets(window.state)["simulation"])}
    assert not {"acceleration_backend", "acceleration_enabled"}.intersection(names)
    assert not hasattr(window.workspace.scan_control, "stem_execution_policy")
    wave = window.workspace.coherent_beam._make_request().wave_grid
    assert wave.compute_backend == policy
    assert wave.acceleration_enabled is (policy != "CPU")
    calls = []
    monkeypatch.setattr(window.calculations, "submit_background",
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    window.run_high_accuracy()
    window.workspace._high_accuracy_result = SimpleNamespace(simulation=object())
    window.workspace.sample_page.calculation_bar.button.click()
    assert len(calls) == 2
    for args, _kwargs in calls:
        assert args[0].acceleration_backend == policy
        assert args[2] == 3000


def test_sample_edits_and_tab_switches_do_not_launch_calculation(window, monkeypatch, qtbot):
    calls = []
    monkeypatch.setattr(window.calculations, "submit_background", lambda *args, **kw: calls.append((args, kw)))
    window.workspace.sample_page.parameters_changed.emit("sample.thickness_nm")
    assert not window.preview_timer.isActive()
    for page in (window.workspace.sample_page, window.workspace.eds_page, window.workspace.scan_control):
        window.workspace.tabs.setCurrentWidget(page)
    qtbot.wait(40)
    assert not calls
    assert "Click Calculate" in window.status_label.text()


@pytest.mark.parametrize("page_name,workflow", [
    ("sample_page", "sample"), ("eds_page", "eds"), ("scan_control", "stem"),
])
def test_page_button_requires_completed_high_accuracy_then_reuses_beam(window, monkeypatch, page_name, workflow):
    from types import SimpleNamespace
    from PySide6.QtWidgets import QPushButton

    calls, errors = [], []
    page = getattr(window.workspace, page_name)
    window.workspace._high_accuracy_result = None
    monkeypatch.setattr(window, "_show_error", errors.append)
    monkeypatch.setattr(window.calculations, "submit_background", lambda *args, **kw: calls.append((args, kw)))
    page.calculation_bar.button.click()
    assert not calls and len(errors) == 1
    assert "Run high-accuracy once" in errors[-1]

    window.findChild(QPushButton, "highAccuracyButton").click()
    assert len(calls) == 1 and calls[0][1]["workflow"] == "rays"
    page.calculation_bar.button.click()
    assert len(calls) == 1 and len(errors) == 2  # Submission alone is not a completed beam.

    # Metadata-only completion fixture; physical prerequisite checks have
    # separate coverage in test_simulation_workflows.
    seed = SimpleNamespace(simulation=object())
    window.workspace._high_accuracy_result = seed
    page.calculation_bar.button.click()
    assert len(errors) == 2 and len(calls) == 2
    assert calls[1][1]["workflow"] == workflow
    assert calls[1][1]["existing_result"] is seed


def test_arming_single_stem_frame_uses_existing_particle_seed_once(window, monkeypatch):
    from types import SimpleNamespace

    calls, errors = [], []
    monkeypatch.setattr(window, "_show_error", errors.append)
    monkeypatch.setattr(window.calculations, "submit_background",
                        lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr("temsim.gui.scan_panel.calibrate_scan_system", lambda *_a, **_kw: None)
    page = window.workspace.scan_control
    window.state.ac_deflector.scan_enabled = False
    window.state.sample.stem_image_enabled = False
    page.set_state(window.state)
    assert calls == []
    window.workspace._high_accuracy_result = None
    page.ac_controls["scan_enabled"].setChecked(True)
    page.image_enabled.setChecked(True)
    assert calls == [] and len(errors) == 1
    assert "Run high-accuracy once" in errors[0]

    seed = SimpleNamespace(simulation=object())
    window.workspace._high_accuracy_result = seed
    page.image_enabled.setChecked(False)
    page.image_enabled.setChecked(True)
    assert len(calls) == 1
    assert calls[0]["workflow"] == "stem" and calls[0]["existing_result"] is seed
    page.image_enabled.setChecked(True)
    assert len(calls) == 1
    # A subsequent deliberate single scan uses the same established request path.
    page.calculate_button.click()
    assert len(calls) == 2 and calls[1]["workflow"] == "stem"
    assert calls[1]["existing_result"] is seed


@pytest.mark.parametrize("dispatch_attempt", range(5))
def test_live_edits_during_preparation_keep_one_frame_and_latest_pending(window, qtbot, monkeypatch, dispatch_attempt):
    from threading import Event
    from types import SimpleNamespace
    from PySide6.QtCore import QTimer
    from temsim.gui.calculation_controller import CalculationWorker
    from temsim.gui.calculation_request import CapturedCalculationRequest
    from temsim.interactive_calculation import CalculationRange, available_controls

    entered, release = Event(), Event()
    errors = []
    monkeypatch.setattr(window, "_show_error", errors.append)
    prepare = CapturedCalculationRequest.prepare
    preparations = []

    def held_prepare(request, cancel_event):
        preparations.append(request)
        if len(preparations) == 1:
            entered.set()
            if not release.wait(30):
                raise RuntimeError("Test preparation release timed out")
        return prepare(request, cancel_event)

    monkeypatch.setattr(CapturedCalculationRequest, "prepare", held_prepare)
    start = window.calculations.pool.start
    solver_jobs = []

    def dispatch(worker):
        if isinstance(worker, CalculationWorker):
            solver_jobs.append(worker)
        else:
            start(worker)

    monkeypatch.setattr(window.calculations.pool, "start", dispatch)
    old_high = SimpleNamespace(model_signature="retained complete result")
    window.workspace._high_accuracy_result = old_high
    window.workspace._high_accuracy_current = False
    control = next(c for c in available_controls(window.state) if c.key == "objective_lens" and c.field == "percent")
    axis = CalculationRange(control, 0., 100.)
    try:
        window._apply_interactive_tuning(((axis, 67.5),))
        window.preview_timer.stop()
        window.run_preview()
        try:
            qtbot.waitUntil(entered.is_set, timeout=5000)
        except Exception as exc:
            import faulthandler
            faulthandler.dump_traceback()
            coordinator = window.calculations.pool.coordinator
            raise AssertionError(f"Preparation did not start: errors={errors}; resources={coordinator.statistics()}; lifecycle={coordinator.diagnostic_snapshot(include_stacks=True)}") from exc
        generation = window.calculations.generation
        cancel_event = window.calculations._cancel_event
        assert window._interactive_preview_in_flight()
        assert "calculation" in window._progress_owners
        ticks = []
        QTimer.singleShot(0, lambda: ticks.append(True))
        qtbot.waitUntil(lambda: bool(ticks))
        for value in (67.6, 67.7, 68.25):
            window._apply_interactive_tuning(((axis, value),))
        assert window.calculations.generation == generation
        assert not cancel_event.is_set()
        assert window._interactive_preview_pending
        assert solver_jobs == []
        release.set()
        qtbot.waitUntil(lambda: len(solver_jobs) == 1 or bool(errors), timeout=20000)
        assert not errors
        first = solver_jobs[0]
        assert first.state.objective_lens.percent == pytest.approx(67.5)
        assert window.state.objective_lens.percent == pytest.approx(68.25)
        assert window._interactive_preview_in_flight(), "Preparation must not finish the full request"
        window.calculations._accept_finished(first.generation, first.quality)
        qtbot.waitUntil(lambda: len(solver_jobs) == 2 or bool(errors), timeout=20000)
        assert not errors
        second = solver_jobs[1]
        assert second.state.objective_lens.percent == pytest.approx(68.25)
        assert len(preparations) == 2
        window.calculations._accept_finished(second.generation, second.quality)
        assert not window._interactive_preview_pending
        assert "calculation" not in window._progress_owners
        assert window.workspace._high_accuracy_result is old_high
    finally:
        release.set()
        window.calculations.invalidate_pending()
