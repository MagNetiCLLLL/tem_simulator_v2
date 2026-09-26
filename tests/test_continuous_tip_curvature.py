"""One particle emission law, continuous geometry, and executed gun transport."""
import numpy as np
import pytest

from temsim.optics.electron_gun.emitter import ColdFieldEmitter
from temsim.optics.electron_gun.field_emission import FieldEmissionGun, field_emission_gun_from_dict
from temsim.optics.electron_gun.tip_curvature import curve_bundle, support_radius_nm
from temsim.physics.ray_identity import emission_reference


def emitter():
    return ColdFieldEmitter(0, 1, 1)


@pytest.mark.parametrize("curvature", [1e-14, 1e-8, 1e-5, .01])
def test_projected_source_and_local_angles_are_preserved_when_surface_bends(curvature):
    source = emitter()
    flat = source.emit(1000)
    assert curve_bundle(flat, source) is flat
    current = source.emitted_current_a
    source.curvature_nm_inv = curvature
    bent = source.emit(1000)
    for field in ("x_m", "y_m", "weight", "ray_id", "energy_offset_ev"):
        np.testing.assert_array_equal(getattr(flat, field), getattr(bent, field))
    assert source.emitted_current_a == current
    np.testing.assert_array_equal(bent.surface_position_m[:, :2], emission_reference(flat)["position_m"][:, :2])
    assert np.all(bent.surface_position_m[:, 2] < 0)
    np.testing.assert_allclose(np.linalg.norm(bent.surface_normal, axis=1), 1, atol=1e-15)
    local_angle = np.arctan2(np.linalg.norm(np.cross(bent.surface_direction, bent.surface_normal), axis=1),
                           np.einsum("ij,ij->i", bent.surface_direction, bent.surface_normal))
    np.testing.assert_allclose(local_angle, np.arctan(np.hypot(flat.tx_rad, flat.ty_rad)), atol=2e-17)
    reference = emission_reference(bent)
    np.testing.assert_array_equal(reference["normal"], bent.surface_normal)
    np.testing.assert_array_equal(reference["position_m"], bent.surface_position_m)


def test_invalid_curvature_and_unsupported_model_combinations_are_explicit():
    source = emitter()
    for value in (-1, np.nan, np.inf):
        with pytest.raises(ValueError):
            source.curvature_nm_inv = value
    source.curvature_nm_inv = 1/support_radius_nm(source)
    with pytest.raises(ValueError, match="support radius"):
        source.validate()
    source.curvature_nm_inv = 1e-8
    from temsim.optics.electron_gun.tip_coherence import TipCoherence
    source.coherence = TipCoherence()
    with pytest.raises(ValueError, match="classical"):
        source.emit(49)


@pytest.mark.parametrize("curvature", [1e-14, 1e-8, .01])
def test_centre_is_fixed_and_sag_is_upstream_and_symmetric(curvature):
    from dataclasses import replace
    source = emitter()
    bundle = replace(source.emit(9), x_m=np.array([-2, 2, 0, 0, -1, 1, 0, 0, 0])*1e-9,
                     y_m=np.array([0, 0, -2, 2, 0, 0, -1, 1, 0])*1e-9)
    source.curvature_nm_inv = curvature
    bent = curve_bundle(bundle, source)
    # The centre is the apex, not the centroid of the curved surface.
    np.testing.assert_array_equal(bent.surface_position_m[-1], [0, 0, 0])
    np.testing.assert_array_equal(bent.surface_normal[-1], [0, 0, 1])
    z = bent.surface_position_m[:4, 2]*1e9
    assert np.all(z < 0)
    np.testing.assert_array_equal(z, np.full(4, z[0]))
    # Stable analytic sphere relation, z + k/2 (r^2 + z^2) = 0.
    np.testing.assert_allclose(z, -curvature/2*(4 + z*z), rtol=5e-15, atol=0)
    np.testing.assert_allclose(bent.surface_normal[:4, :2],
                               curvature*np.array([[-2, 0], [2, 0], [0, -2], [0, 2]]), atol=0)


@pytest.mark.parametrize("old_model", ["analytic-tip-curvature-v1", "analytic-tip-angle-only-v2"])
def test_historical_curvature_models_are_rejected_without_conversion(old_model):
    gun = FieldEmissionGun()
    gun.emitter.curvature_nm_inv = .01
    before = gun.to_dict()
    document = gun.to_dict()
    document["components"]["feg_tip"]["emission_geometry_model"] = old_model
    with pytest.raises(ValueError, match="Unsupported continuous tip geometry model"):
        field_emission_gun_from_dict(document)
    with pytest.raises(ValueError, match="Unsupported continuous tip geometry model"):
        gun.emitter.curvature_model = old_model
    assert gun.to_dict() == before


def test_current_profile_preserves_curvature_and_rejects_old_versions(tmp_path):
    from copy import deepcopy
    from temsim.optics.column import default_state
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.profile_io import save_profile, read_profile, apply_profile_values
    from temsim.optics.electron_gun.tip_curvature import MODEL
    state = default_state()
    state.electron_gun.emitter.curvature_nm_inv = .01
    path = tmp_path / "curvature.toml"
    save_profile(path, state, AssemblyCatalog().default_selection())
    _, values = read_profile(path)
    state.electron_gun.emitter.curvature_nm_inv = 0.
    apply_profile_values(state, values)
    assert state.electron_gun.emitter.curvature_nm_inv == .01
    assert state.electron_gun.emitter.curvature_model == MODEL
    before = state.to_dict()
    for version in (9, 10):
        old = deepcopy(values)
        old["__profile_format_version__"] = version
        with pytest.raises(ValueError, match="Unsupported operating-profile format"):
            apply_profile_values(state, old)
        assert state.to_dict() == before


def test_current_editor_changes_tip_geometry_only_after_accept(qtbot):
    from temsim.gui.gun_source_dialog import GunSourceDialog
    from temsim.optics.electron_gun.tip_curvature import MODEL
    from temsim.optics.electron_gun.tip_edit import candidate_tip_edit
    gun = FieldEmissionGun()
    dialog = GunSourceDialog(gun)
    qtbot.addWidget(dialog)
    dialog.inputs["curvature_nm_inv"].setText(".01")
    dialog.accept()
    assert dialog.result() == dialog.DialogCode.Accepted, dialog.error.text()
    result = candidate_tip_edit(gun, dialog.value())
    assert result.emitter.curvature_model == MODEL
    assert np.all(result.emit(49).surface_position_m[:, 2] < 0)
    assert gun.emitter.curvature_nm_inv == 0.


def test_serialization_snapshot_and_cache_identity_round_trip():
    from temsim.optics.column import default_state
    from temsim.instrument_snapshot import encode_instrument, decode_instrument
    from temsim.calculation_cache import calculation_signatures
    from temsim.runtime_parameters import runtime_targets, editable_parameters, validate_runtime_assignment
    state = default_state()
    gun = state.electron_gun
    before = gun.to_dict()
    key = gun._cache_key(49)
    signatures = calculation_signatures(state)
    target = runtime_targets(state)[gun.emitter.key]
    assert "curvature_nm_inv" in {p.name for p in editable_parameters(target)}
    assert validate_runtime_assignment(target, "curvature_nm_inv", 1e-8) == 1e-8
    gun.emitter.curvature_nm_inv = 1e-8
    assert gun._cache_key(49) != key
    changed = calculation_signatures(state)
    for stage in ("request", "column", "incident", "elastic", "wave_source", "stem", "eds", "sample_downstream"):
        assert changed[stage] != signatures[stage], stage
    restored = field_emission_gun_from_dict(gun.to_dict())
    assert restored.emitter.curvature_nm_inv == 1e-8
    assert restored.to_dict() == gun.to_dict()
    assert decode_instrument(encode_instrument(state)).electron_gun.emitter.curvature_nm_inv == 1e-8
    gun.emitter.curvature_nm_inv = 0
    assert gun.to_dict() == before
    assert gun._cache_key(49) == key
    assert calculation_signatures(state) == signatures


def test_live_tuning_is_transactional_and_does_not_change_optics():
    from temsim.optics.column import default_state
    from temsim.interactive_calculation import available_controls, CalculationRange, apply_live_tuning_values
    state = default_state()
    axis = CalculationRange(next(c for c in available_controls(state) if c.group == "source"), 0, 1e-5)
    before = state.electron_gun.to_dict()["components"]
    apply_live_tuning_values(state, ((axis, 1e-8),))
    assert state.electron_gun.emitter.curvature_nm_inv == 1e-8
    after = state.electron_gun.to_dict()["components"]
    assert {k: v for k, v in before.items() if k != "feg_tip"} == {k: v for k, v in after.items() if k != "feg_tip"}
    with pytest.raises(ValueError):
        apply_live_tuning_values(state, ((axis, 1),))
    assert state.electron_gun.emitter.curvature_nm_inv == 1e-8


def test_medium_boundary_probes_receive_same_geometry_mapping():
    source = emitter()
    source._tuning_boundary_probes = 33
    flat = source.emit(193)
    source.curvature_nm_inv = 1e-5
    bent = source.emit(193)
    np.testing.assert_array_equal(flat.weight, bent.weight)
    assert (bent.weight == 0).sum() == 33
    assert bent.surface_position_m.shape == (193, 3)
    assert bent.surface_normal[-1, 2] == 1
    np.testing.assert_array_equal(bent.surface_position_m[-1], [0, 0, 0])


def test_actual_gun_transport_is_continuous_near_flat_and_cache_reusable():
    from temsim.physics.gun_field_environment import ensure_gun_field_environment

    def execute(source):
        original = source.emit(49)
        reference = emission_reference(original)
        result = source.trace_to_exit(49)
        assert source.trace_to_exit(49) is result
        after = source.emit(49)
        for name in ("x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev", "weight", "ray_id",
                     "surface_position_m", "surface_normal", "surface_direction"):
            if hasattr(original, name):
                np.testing.assert_array_equal(getattr(after, name), getattr(original, name))
        for name in reference:
            np.testing.assert_array_equal(result.emission_reference[name], reference[name])
        np.testing.assert_array_equal(result.equal_time_history.z_mm[0],
                                      reference["position_m"][:, 2]*1000)
        return result

    def exit_xy(result):
        return np.column_stack((result.exit_bundle.x_m, result.exit_bundle.y_m))

    def separation(left, right):
        np.testing.assert_array_equal(left.exit_bundle.alive, right.exit_bundle.alive)
        return np.max(np.linalg.norm(exit_xy(left)-exit_xy(right), axis=1))

    gun = FieldEmissionGun()
    ensure_gun_field_environment(gun)
    gun._gun_field_cells_per_bore = 16
    flat = execute(gun)
    for k in (1e-8, 2e-8):
        gun.emitter.curvature_nm_inv = k
        result = execute(gun)
        assert result is not flat
        assert separation(result, flat) > 0
        assert np.all(result.equal_time_history.z_mm[0] < 0)
    # R=100/50 mm changes the macroscopic cathode boundary as well as launch
    # normals, so these two finite curvatures need not give an exactly 2x
    # response. Test the actual planar limit separately, with optics, source,
    # exit plane and transport budget held fixed.
    gun.emitter.curvature_nm_inv = 1e-14
    weak = execute(gun)
    coarse = FieldEmissionGun()
    ensure_gun_field_environment(coarse)
    coarse._gun_field_cells_per_bore = 8
    coarse_flat = execute(coarse)
    coarse.emitter.curvature_nm_inv = 1e-14
    coarse_weak = execute(coarse)
    fine_gap = separation(weak, flat)
    coarse_gap = separation(coarse_weak, coarse_flat)
    mesh_change = separation(coarse_flat, flat) + separation(coarse_weak, weak)
    # The two providers use different meshes/interpolation near the axis.
    # Bound their discrepancy by the measured combined 8->16 mesh change,
    # and require the discrepancy itself to fall under refinement. This is
    # a scoped discretisation comparison, not a fitted absolute tolerance.
    assert fine_gap < coarse_gap
    assert fine_gap <= mesh_change
    gun.emitter.curvature_nm_inv = 0
    assert gun.trace_to_exit(49) is flat


def test_curvature_slider_updates_while_held_down(qtbot):
    from PySide6.QtWidgets import QSlider
    from temsim.optics.column import default_state
    from temsim.gui.interactive_calculation import InteractiveCalculationPage
    from temsim.interactive_calculation import apply_live_tuning_values
    state = default_state()
    page = InteractiveCalculationPage()
    qtbot.addWidget(page)
    page.set_source(state)
    index = next(i for i in range(page.choice.count()) if page.choice.itemData(i).group == "source")
    page.choice.setCurrentIndex(index)
    page._add_range()
    page.start_live_tuning()
    assert page.live_widgets, page.status.text()
    received = []
    def apply(values):
        apply_live_tuning_values(state, values)
        received.append(state.electron_gun.emitter.curvature_nm_inv)
    page.tuning_changed.connect(apply)
    slider = page.live_table.findChild(QSlider)
    slider.setSliderDown(True)
    slider.setValue(10)
    qtbot.waitUntil(lambda: bool(received) and received[-1] == pytest.approx(1e-8), timeout=1000)
    assert slider.isSliderDown()
    slider.setValue(20)
    qtbot.waitUntil(lambda: received[-1] == pytest.approx(2e-8), timeout=1000)
    slider.setSliderDown(False)
    page.timer.stop()


def test_source_dialog_uses_one_continuous_geometry_parameter(qtbot):
    from temsim.gui.gun_source_dialog import GunSourceDialog
    from temsim.optics.electron_gun.tip_edit import candidate_tip_edit
    gun = FieldEmissionGun()
    dialog = GunSourceDialog(gun)
    qtbot.addWidget(dialog)
    dialog.inputs["curvature_nm_inv"].setText("1e-8")
    assert dialog.continuous_preview._continuous[0] == 1e-8
    dialog.continuous_preview.resize(500, 180)
    assert not dialog.continuous_preview.grab().isNull()
    dialog.accept()
    assert dialog.result() == dialog.DialogCode.Accepted, dialog.error.text()
    edited = candidate_tip_edit(gun, dialog.value())
    assert edited.emitter.curvature_nm_inv == 1e-8
    assert edited.emitter.surface_model is None and edited.emitter.coherence is None
    assert gun.emitter.curvature_nm_inv == 0


def test_profile_geometry_reload_and_partial_controls_preserve_explicit_choices(tmp_path):
    from temsim.optics.column import default_state
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.profile_io import save_profile, read_profile, apply_profile_values
    from temsim.optics.electron_gun.tip_assembly import model_from_part
    state = default_state()
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    assembly = catalog.apply(state, selection)
    tip = state.electron_gun.emitter
    tip.curvature_nm_inv = 1e-8
    path = tmp_path / "curvature.toml"
    save_profile(path, state, selection)
    _, values = read_profile(path)
    catalog.apply(state, selection, preserve_operating_parameters=True)
    assert tip.curvature_nm_inv == 1e-8
    tip.curvature_nm_inv = 0
    tip.surface_model = model_from_part(assembly.part("feg_tip").data)
    assert not apply_profile_values(state, values)
    assert tip.surface_model is None and tip.curvature_nm_inv == 1e-8
    values[tip.key].pop("curvature_nm_inv")
    apply_profile_values(state, values)
    assert tip.curvature_nm_inv == 1e-8
    tip.curvature_nm_inv = 1e-8
    catalog.apply(state, selection, preserve_operating_parameters=False)
    assert tip.curvature_nm_inv == 0


def test_full_preview_pipeline_uses_curvature_and_keeps_sample_signals_out():
    from temsim.optics.column import default_state
    from temsim.gui.calculation_controller import CalculationController, CalculationWorker
    from temsim.physics.optical_tuning import TUNING_PROFILES
    state = default_state()
    frames = []
    for k in (0, 1e-5):
        state.electron_gun.emitter.curvature_nm_inv = k
        profile = TUNING_PROFILES["Preview"]
        snapshot = CalculationController._calculation_snapshot(state, "Preview", profile.rays, profile.step_mm)
        worker = CalculationWorker(1, "Preview", snapshot)
        errors = []
        worker.signals.result.connect(lambda generation, quality, frame, duration: frames.append(frame))
        worker.signals.error.connect(lambda *args: errors.append(args))
        worker.run()
        assert not errors
    assert len(frames) == 2
    assert not np.array_equal(frames[0].simulation.incident.x, frames[1].simulation.incident.x)
    assert frames[1].state_snapshot.electron_gun.emitter.curvature_nm_inv == 1e-5
    assert frames[1].stem_scan is None and frames[1].wave_imaging is None


def test_main_window_curvature_edits_use_existing_preview_scheduler(qtbot, monkeypatch, tmp_path):
    from PySide6.QtCore import QSettings
    from temsim.gui import main_window
    monkeypatch.setattr(main_window, "QSettings", lambda: QSettings(str(tmp_path/"ui.ini"), QSettings.Format.IniFormat))
    monkeypatch.setattr(main_window.MainWindow, "schedule_preview", lambda *args: None)
    window = main_window.MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    calls = []
    monkeypatch.setattr(window, "schedule_preview", lambda *args: calls.append(args))
    window._capture_interactive_settings()
    page = window.workspace.interactive_calculation
    i = next(i for i in range(page.choice.count()) if page.choice.itemData(i).group == "source")
    page.choice.setCurrentIndex(i)
    page._add_range()
    page.start_live_tuning()
    page.timer.stop()
    value = next(iter(page.live_widgets.values()))
    value.setValue(1e-8)
    page._flush_live_tuning()
    assert window.state.electron_gun.emitter.curvature_nm_inv == 1e-8
    assert ("interactive_tuning",) in calls
    window._select_component_from_workspace("feg_tip")
    widget = window.parameter_panel._quick_widgets["curvature_nm_inv"]
    assert widget.value() == 1e-8
    window._reveal_physical_model("feg_tip", 0.)
    editor = window.workspace.physical_layout.model_editor
    if editor._pending_part is not None:
        editor._open_pending_part()
    editor._flush_runtime_refresh()
    vertices = next(m["vertices"].copy() for m in editor._mesh_records if m["region"] == "emitting_cap")
    widget = window.parameter_panel._quick_widgets["curvature_nm_inv"]
    widget.setValue(2e-8)
    assert window.state.electron_gun.emitter.curvature_nm_inv == 2e-8
    editor._flush_runtime_refresh()
    changed = next(m["vertices"] for m in editor._mesh_records if m["region"] == "emitting_cap")
    assert not np.array_equal(vertices, changed)
    assert changed[:, 2].max() == 0
    page.shutdown()
