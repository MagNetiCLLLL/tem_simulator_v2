"""Real window source ownership and invalidation; no full-column wave solve."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton

from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.physics.coherent_inputs import TipEmissionSettings, source_settings_from_state, prepare_coherent_state
from test_working_point_restore_gui import window


def supported_settings(state):
    # Declared synthetic Gaussian boundary for software wiring, not a measured
    # FEG or a replacement startup default. Keep nonzero energy spread.
    return replace(source_settings_from_state(state), enabled=True,
                   tip_fwhm_nm=100., tip_mean_energy_ev=.3,
                   tip_minimum_energy_ev=.01, tip_energy_spread_fwhm_ev=.3,
                   tip_curvature_x_m1=120., tip_curvature_y_m1=-80.,
                   incoherent_angle_rms_mrad=.2)


def test_particle_toolbar_and_beam_page_have_separate_entries(window):
    workspace = window.workspace
    page = workspace.coherent_beam
    assert page.calculate_button.text() == "Calculate beam"
    assert page.compare_classical.isEnabledTo(page.advanced_body)
    assert not page.compare_classical.isChecked()
    assert workspace.tabs.tabText(workspace.tabs.indexOf(page)) == "Electron beam"
    assert workspace.ray_beam_tabs.tabText(
        workspace.ray_beam_tabs.indexOf(workspace.transverse_beam)) == "Classical rays"
    assert workspace.ray_beam_tabs.tabText(
        workspace.ray_beam_tabs.indexOf(workspace.coherent_ray_view)) == "Beam observation"
    toolbar_button = window.findChild(QPushButton, "highAccuracyButton")
    assert toolbar_button is not None and toolbar_button.text() == "Run high-accuracy once"
    assert window.findChild(QPushButton, "calculateBeamButton") is None
    assert window.classical_rays_action in window.simulation_menu.actions()
    assert window.classical_rays_action.text() == "Calculate classical rays"


def test_toolbar_high_accuracy_keeps_current_page_and_unapplied_tip_draft(window, monkeypatch):
    workspace = window.workspace
    page = workspace.coherent_beam
    before = capture_instrument_snapshot(window.state)
    revision = window._physical_revision
    page.source_enabled.setChecked(not page.source_enabled.isChecked())
    page.tip_offset_x.setValue(page.tip_offset_x.value() + .125)
    draft = page._settings()
    calls = []
    monkeypatch.setattr(page, "calculate", lambda: pytest.fail("Particle toolbar started a wave calculation"))
    monkeypatch.setattr(page, "source_applier", lambda _settings: pytest.fail("Toolbar applied a Tip draft"))
    monkeypatch.setattr(window, "_submit_high_accuracy", lambda **kwargs: calls.append(kwargs))
    workspace.tabs.setCurrentWidget(workspace.scanning_page)

    window.findChild(QPushButton, "highAccuracyButton").click()

    assert calls == [{"workflow": "rays"}]
    assert workspace.tabs.currentWidget() is workspace.scanning_page
    assert page._settings() == draft
    assert not page.compare_classical.isChecked()
    assert capture_instrument_snapshot(window.state).digest == before.digest
    assert window._physical_revision == revision


def test_classical_menu_keeps_explicit_ray_workflow(window, monkeypatch):
    submitted = []
    monkeypatch.setattr(window, "_submit_high_accuracy", lambda **kwargs: submitted.append(kwargs))
    window.classical_rays_action.trigger()
    assert submitted == [{"workflow": "rays"}]


def test_shared_projection_updates_beam_observation_without_calculating(window, monkeypatch):
    page = window.workspace.coherent_beam
    projected = []
    before = capture_instrument_snapshot(window.state)
    monkeypatch.setattr(page, "set_projection_angle", projected.append)
    monkeypatch.setattr(page, "calculate", lambda: pytest.fail("Projection started a beam calculation"))

    window.workspace._set_projection_angle(37.5)

    assert projected == [37.5]
    assert capture_instrument_snapshot(window.state).digest == before.digest


def test_shared_apply_invalidates_both_views_without_launching_jobs(window, monkeypatch):
    page = window.workspace.coherent_beam
    started = []
    monkeypatch.setattr(window.calculations, "submit_background", lambda *a, **k: started.append("ray"))
    monkeypatch.setattr(page, "_start_query", lambda: started.append("wave"))
    original = capture_instrument_snapshot(window.state)
    revision = window._physical_revision
    gun = window.state.electron_gun
    gun._trace_cache = object()
    gun._trace_cache_key = "previous source"
    settings = supported_settings(window.state)

    assert page.source_applier(settings) is window.state
    assert window._physical_revision == revision + 1
    assert window.workspace._ray_extent_stale
    assert page._stale
    assert gun._trace_cache is gun._trace_cache_key is None
    assert not window.preview_timer.isActive()
    assert started == []
    assert capture_instrument_snapshot(window.state).physical_digest != original.physical_digest
    captured = prepare_coherent_state(window.state)
    assert captured.electron_gun.to_dict() == gun.to_dict()
    assert page.source_enabled.isChecked()
    assert page.tip_fwhm.value() == settings.tip_fwhm_nm
    assert page.tip_energy_spread.value() == .3
    assert window._runtime_targets["feg_tip"].obj is gun.emitter
    assert captured.electron_gun.emit(65).x_m.tolist() == gun.emit(65).x_m.tolist()
    # Applying the same source must not invalidate a fresh result again.
    page.source_applier(source_settings_from_state(window.state))
    assert window._physical_revision == revision + 1


def test_unapplied_or_unsupported_draft_never_replaces_live_source(window, qtbot):
    page = window.workspace.coherent_beam
    before = capture_instrument_snapshot(window.state)
    revision = window._physical_revision
    page.source_enabled.setChecked(True)
    # The historical forward-only boundary remains unsupported at this scale.
    # The explicit driven boundary now executes the missing near-tip physics.
    page.tip_boundary.setCurrentIndex(page.tip_boundary.findData("forward_gaussian_schell"))
    page.tip_fwhm.setValue(5.)
    page.tip_energy.setValue(.3)
    page.tip_minimum_energy.setValue(.01)
    page.tip_energy_spread.setValue(.3)
    qtbot.mouseClick(page.calculate_button, Qt.MouseButton.LeftButton)
    assert page._worker is None
    assert capture_instrument_snapshot(window.state).digest == before.digest
    qtbot.mouseClick(page.apply_source_button, Qt.MouseButton.LeftButton)
    assert page._worker is None
    assert "source" in page.status.text().lower()
    assert capture_instrument_snapshot(window.state).digest == before.digest
    assert window._physical_revision == revision
    assert not window.preview_timer.isActive()


def test_default_driven_tip_applies_and_dispatches_same_instrument(window, qtbot, monkeypatch):
    page = window.workspace.coherent_beam
    started = []
    monkeypatch.setattr(page, "_start_query", lambda: started.append(page._captured))
    page.source_enabled.setChecked(True)
    assert page.tip_boundary.currentData() == "driven_gaussian_schell"
    qtbot.mouseClick(page.apply_source_button, Qt.MouseButton.LeftButton)
    emitter = window.state.electron_gun.emitter
    assert emitter.coherence.boundary_model == "driven_gaussian_schell"
    applied = source_settings_from_state(window.state)
    assert (applied.tip_fwhm_nm, applied.tip_mean_energy_ev,
            applied.tip_energy_spread_fwhm_ev) == (5., .3, .3)
    assert started == []
    qtbot.mouseClick(page.calculate_button, Qt.MouseButton.LeftButton)
    assert len(started) == 1
    assert not page._stale
    assert started[0].electron_gun.to_dict() == window.state.electron_gun.to_dict()
    assert "Source ready" in page.status.text()
    # This verifies the real window's publication/dispatch boundary. Numerical
    # propagation is independently exercised by the driven-gun tests.


def test_shared_phase_edit_invalidates_session_but_z_does_not_edit_source(window):
    page = window.workspace.coherent_beam
    page.source_applier(supported_settings(window.state))
    before = capture_instrument_snapshot(window.state)
    revision = window._physical_revision
    page.set_selected_z(window.state.sample.z_mm + 5.)
    assert capture_instrument_snapshot(window.state).digest == before.digest
    assert window._physical_revision == revision
    page.source_applier(replace(source_settings_from_state(window.state), tip_curvature_xy_m1=30.))
    assert page._stale
    assert window._physical_revision == revision + 1
    assert capture_instrument_snapshot(window.state).physical_digest != before.physical_digest
    assert page.tip_curvature_xy.value() == 30.


def test_invalid_shared_edit_is_atomic_in_main_window(window):
    page = window.workspace.coherent_beam
    before = capture_instrument_snapshot(window.state)
    revision = window._physical_revision
    with pytest.raises(ValueError):
        page.source_applier(replace(supported_settings(window.state), tip_fwhm_nm=-1.))
    assert capture_instrument_snapshot(window.state).digest == before.digest
    assert window._physical_revision == revision
    assert not window.preview_timer.isActive()


def particle_fixture(pair, workflow):
    """Executed-state metadata fixture; no particle propagation is simulated."""
    from temsim.gui.calculation_request import apply_request_numerics
    state = apply_request_numerics(pair._snapshot.restore(), "High accuracy",
                                  pair._ray_count, pair._step_mm)
    return SimpleNamespace(workflow=workflow, state_snapshot=state)


@pytest.mark.parametrize("with_specimen", [False, True])
def test_beam_button_with_classical_comparison_captures_once_and_continues_sample_before_wave(window, monkeypatch, with_specimen):
    import temsim.gui.paired_beam_controller as pairing
    page, pair = window.workspace.coherent_beam, window.paired_beams
    page.source_applier(supported_settings(window.state))
    # The interaction flag substitutes material routing only. This verifies
    # ownership/continuation, not a CIF calculation or numerical agreement.
    monkeypatch.setattr(pairing, "specimen_interactions_active", lambda _sample: with_specimen)
    submitted, waves, published = [], [], []
    monkeypatch.setattr(pair.calculations, "submit_background", lambda *a, **kw: submitted.append((a, kw)))
    monkeypatch.setattr(page, "start_captured_calculation", lambda *a, **kw: waves.append((a, kw)))
    def publish(*args, **kwargs):
        published.append((args, kwargs))
        window.workspace._last_result = args[1]
    monkeypatch.setattr(window, "_calculation_ready", publish)
    original = capture_instrument_snapshot(window.state)
    window.high_rays.setValue(3000)
    page.compare_classical.setChecked(True)
    page.calculate_button.click()
    assert len(submitted) == 1 and not waves
    assert submitted[0][1]["workflow"] == "rays"
    assert submitted[0][0][2] == 3000
    assert capture_instrument_snapshot(submitted[0][0][0]).digest == original.digest
    rays = particle_fixture(pair, "rays")
    pair._particle_ready("High accuracy", rays, .25)
    if with_specimen:
        assert not waves
        assert submitted[1][1] == {"workflow": "sample", "existing_result": rays}
        assert capture_instrument_snapshot(submitted[1][0][0]).digest == original.digest
        pair._particle_ready("High accuracy", particle_fixture(pair, "sample"), .5)
    assert len(waves) == len(published) == 1
    context = waves[0][1]["pair_context"]
    assert context.token == pair.token
    assert context.physical_identity == original.physical_digest
    assert context.instrument_identity == original.digest
    assert context.particle_ray_count == 3000
    assert capture_instrument_snapshot(waves[0][0][0]).digest == original.digest
    assert published[0][1]["paired_token"] == context.token
    assert capture_instrument_snapshot(window.state).digest == original.digest


def test_pair_rejects_physical_drift_before_wave_publication(window, monkeypatch):
    page, pair = window.workspace.coherent_beam, window.paired_beams
    page.source_applier(supported_settings(window.state))
    monkeypatch.setattr(pair.calculations, "submit_background", lambda *a, **kw: None)
    waves = []
    monkeypatch.setattr(page, "start_captured_calculation", lambda *a, **kw: waves.append(1))
    page.compare_classical.setChecked(True)
    page.calculate_button.click()
    old = particle_fixture(pair, "rays")
    window.state.electron_gun.emitter.emission_energy_ev += .1
    pair._particle_ready("High accuracy", old, .1)
    assert waves == [] and pair.token is None
    assert "inputs changed" in page.status.text().lower()


@pytest.mark.parametrize("barrier", ["loading", "hold", "signature", "not_published", "cancelled"])
def test_pair_requires_accepted_particle_publication_before_wave(window, monkeypatch, barrier):
    page, pair = window.workspace.coherent_beam, window.paired_beams
    page.source_applier(supported_settings(window.state))
    monkeypatch.setattr(pair.calculations, "submit_background", lambda *a, **kw: None)
    waves = []
    monkeypatch.setattr(page, "start_captured_calculation", lambda *a, **kw: waves.append(1))
    page.compare_classical.setChecked(True)
    page.calculate_button.click()
    result = particle_fixture(pair, "rays")
    previous = window.workspace._last_result
    if barrier == "loading":
        window.result_files.loading = True
    elif barrier == "hold":
        window.result_files.hold_automatic_preview = True
    elif barrier == "signature":
        # The real publication method keeps results with an earlier signature
        # out of Ray Diagram, even if their state metadata appears current.
        result.model_signature = "a different captured calculation"
    elif barrier == "not_published":
        monkeypatch.setattr(window, "_calculation_ready", lambda *a, **kw: None)
    else:
        def publish_then_cancel(*args, **kwargs):
            window.workspace._last_result = args[1]
            page.mark_inputs_stale()
        monkeypatch.setattr(window, "_calculation_ready", publish_then_cancel)
    pair._particle_ready("High accuracy", result, .1)
    assert waves == []
    assert pair.token is None
    assert page._pair_context is None
    assert page.comparison_table.isHidden()
    if barrier != "cancelled":
        assert window.workspace._last_result is previous
    assert "incomplete" in page.comparison_status.text().lower()


def test_pair_rejects_particle_snapshot_with_different_source(window, monkeypatch):
    page, pair = window.workspace.coherent_beam, window.paired_beams
    page.source_applier(supported_settings(window.state))
    monkeypatch.setattr(pair.calculations, "submit_background", lambda *a, **kw: None)
    waves = []
    monkeypatch.setattr(page, "start_captured_calculation", lambda *a, **kw: waves.append(1))
    page.compare_classical.setChecked(True)
    page.calculate_button.click()
    wrong = particle_fixture(pair, "rays")
    wrong.state_snapshot.electron_gun.emitter.emission_energy_ev += .1
    pair._particle_ready("High accuracy", wrong, .1)
    assert waves == [] and pair.token is None
    assert "particle source or optics" in page.status.text()


def test_cancelled_particle_generation_cannot_join_new_same_settings_pair(window, monkeypatch):
    page, pair = window.workspace.coherent_beam, window.paired_beams
    page.source_applier(supported_settings(window.state))
    # Run real controller request capture/generation, but do not run its queued
    # numerical or preparation workers.
    monkeypatch.setattr(pair.calculations.pool, "start", lambda worker: None)
    waves = []
    monkeypatch.setattr(page, "start_captured_calculation", lambda *a, **kw: waves.append(1))
    page.compare_classical.setChecked(True)
    page.calculate_button.click()
    old_generation, old_token = pair.calculations.generation, pair.token
    old_result = particle_fixture(pair, "rays")
    page.cancel()
    page.compare_classical.setChecked(True)
    page.calculate_button.click()
    assert pair.token != old_token
    new_token = pair.token
    pair.calculations._accept_result(old_generation, "High accuracy", old_result, .1)
    assert pair.token == new_token and waves == []


def test_paired_source_and_budget_drafts_cancel_pending_pair_without_launch(window, monkeypatch):
    page, pair = window.workspace.coherent_beam, window.paired_beams
    page.source_applier(supported_settings(window.state))
    submitted = []
    monkeypatch.setattr(pair.calculations, "submit_background", lambda *a, **kw: submitted.append(1))
    page.compare_classical.setChecked(True)
    page.calculate_button.click()
    assert pair.token is not None
    page.grid_pixels.setValue(page.grid_pixels.value()+32)
    assert pair.token is None
    assert submitted == [1]
    page.tip_offset_x.setValue(page.tip_offset_x.value()+.1)
    page.compare_classical.setChecked(True)
    page.calculate_button.click()
    assert submitted == [1]
    assert "Apply tip parameters" in page.status.text()


@pytest.mark.parametrize("stage", ["rays", "sample"])
def test_saved_state_metadata_keeps_pending_paired_particles_cancellable(window, monkeypatch, stage):
    import temsim.gui.paired_beam_controller as pairing

    page, pair = window.workspace.coherent_beam, window.paired_beams
    page.source_applier(supported_settings(window.state))
    page.state_list.add_button.click()
    assert len(page.state_list.entries()) == 1, page.status.text()
    assert page.state_list.display_mode() == "current"
    submitted = []
    monkeypatch.setattr(pair.calculations, "submit_background",
                        lambda *args, **kwargs: submitted.append((args, kwargs)))
    monkeypatch.setattr(pairing, "specimen_interactions_active", lambda _sample: stage == "sample")
    page.compare_classical.setChecked(True)
    page.calculate_button.click()
    assert pair.token is not None
    if stage == "sample":
        pair._particle_ready("High accuracy", particle_fixture(pair, "rays"), .1)
    assert pair._stage == stage
    token = pair.token
    assert len(submitted) == (2 if stage == "sample" else 1)
    assert page._worker is None and page.state_set.worker is None
    assert page.paired_busy_provider()
    page._update_cancel_enabled()
    assert page.cancel_button.isEnabled()
    pair._progress("High accuracy", 1, 10, "Pending particle segment")
    progress = page.status.text()
    page.state_list.table.cellWidget(0, 2).setValue(2.)
    page.state_list.table.item(0, 1).setText("Saved state during paired calculation")
    page.state_list.table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    assert pair.token == token
    assert pair._stage == stage
    assert page.status.text() == progress
    assert page.cancel_button.isEnabled()
    page.cancel_button.click()
    assert pair.token is None
    assert not page.paired_busy_provider()
    assert not page.cancel_button.isEnabled()


def test_pair_does_not_silently_disable_vacuum(window, monkeypatch):
    page, pair = window.workspace.coherent_beam, window.paired_beams
    page.source_applier(supported_settings(window.state))
    window.state.vacuum_map.enabled = True
    submitted = []
    monkeypatch.setattr(pair.calculations, "submit_background", lambda *a, **kw: submitted.append(1))
    page.compare_classical.setChecked(True)
    page.calculate_button.click()
    assert not submitted
    assert window.state.vacuum_map.enabled
    assert "ideal vacuum" in page.status.text()


def test_curved_shared_pair_keeps_one_tip_geometry_and_energy_owner(window, monkeypatch):
    """Real GUI capture/handoff; workers are intercepted, not fake wave data."""
    import temsim.gui.paired_beam_controller as pairing
    from temsim.optics.electron_gun.tip_surface import load_tip_surface_reference
    page, pair = window.workspace.coherent_beam, window.paired_beams
    original_surface = load_tip_surface_reference()
    window.state.electron_gun.emitter.surface_model = original_surface
    page.set_state(window.state)
    original = capture_instrument_snapshot(window.state)
    revision = window._physical_revision
    submitted, waves = [], []
    monkeypatch.setattr(pair.calculations, "submit_background", lambda *a, **kw: submitted.append((a, kw)))
    monkeypatch.setattr(window.calculations, "submit_background", lambda *_a, **_kw: pytest.fail("Source Apply launched calculation"))
    monkeypatch.setattr(page, "start_captured_calculation", lambda *a, **kw: waves.append((a, kw)))
    monkeypatch.setattr(pairing, "specimen_interactions_active", lambda _sample: False)
    def publish(*args, **_kwargs):
        window.workspace._last_result = args[1]
    monkeypatch.setattr(window, "_calculation_ready", publish)
    page.source_applier(TipEmissionSettings(enabled=True,
        surface_mean_energy_ev=.7, surface_energy_rms_ev=.12, surface_edge_phase_rad=0.))
    assert not submitted and not waves and not window.preview_timer.isActive()
    assert window._physical_revision == revision + 1
    active = window.state.electron_gun.emitter.surface_model
    assert active.geometry == original_surface.geometry
    assert active.field_numerics == original_surface.field_numerics
    assert active.emission.kinetic_mean_ev == .7
    assert active.emission.kinetic_sigma_ev == .12
    assert set(active.to_dict()["coherence"]) == {"model", "edge_phase_rad"}
    assert capture_instrument_snapshot(window.state).physical_digest != original.physical_digest
    captured = capture_instrument_snapshot(window.state)
    page.compare_classical.setChecked(True)
    page.calculate_button.click()
    assert len(submitted) == 1 and submitted[0][1]["workflow"] == "rays"
    particle_state = submitted[0][0][0]
    assert particle_state.electron_gun.emitter.surface_model.to_dict() == active.to_dict()
    pair._particle_ready("High accuracy", particle_fixture(pair, "rays"), .25)
    assert len(waves) == 1
    wave_state = waves[0][0][0]
    assert wave_state.electron_gun.emitter.surface_model.to_dict() == active.to_dict()
    assert capture_instrument_snapshot(wave_state).digest == captured.digest
    assert capture_instrument_snapshot(window.state).digest == captured.digest
    assert "Geometric-optics" in waves[0][0][2]["particle_representation"]


def test_historical_surface_pair_is_rejected_before_any_job_submission(window, monkeypatch):
    from temsim.optics.electron_gun.tip_surface import SurfaceCoherence, load_tip_surface_reference
    page, pair = window.workspace.coherent_beam, window.paired_beams
    saved = replace(load_tip_surface_reference(), coherence=SurfaceCoherence(
        mean_energy_ev=.4, energy_rms_ev=.05, edge_phase_rad=.2))
    window.state.electron_gun.emitter.surface_model = saved
    page.set_state(window.state)
    before = capture_instrument_snapshot(window.state)
    monkeypatch.setattr(pair.calculations, "submit_background", lambda *_a, **_kw: pytest.fail("Historical source started particles"))
    monkeypatch.setattr(page, "start_captured_calculation", lambda *_a, **_kw: pytest.fail("Unmatched historical wave started"))
    page.compare_classical.setChecked(True)
    page.calculate_button.click()
    assert pair.token is None and page._worker is None
    assert "no shared particle representation" in page.status.text()
    assert "Explicitly replace" in page.status.text()
    assert page.surface_mean.isReadOnly() and page.surface_rms.isReadOnly()
    assert capture_instrument_snapshot(window.state).digest == before.digest
    assert window.state.electron_gun.emitter.surface_model is saved
