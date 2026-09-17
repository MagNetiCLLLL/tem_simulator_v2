"""Bounded particle GUI/archive path; no coherent imaging qualification."""
from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtTest import QTest

from test_portable_inputs import portable_fixture, _remove_fixture_location
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.working_point import WorkingPointCheckpoint
from temsim.calculation_cache import calculation_signatures
from temsim.optics.beam_path_audit import incident_checkpoints


def test_bounded_application_compare_cancel_export_relocated_restore(portable_fixture, tmp_path, monkeypatch, qtbot):
    from temsim.gui import main_window as shell, interactive_calculation as gui, calculation_controller as controller
    from temsim.alignment_transaction import AlignmentRequest, solve_alignment_candidate, AlignmentCancelled
    from temsim.sampling_diagnostics import sampling_summary
    from temsim.alignment_constraints import ConstrainedAlignment
    portable, original, root = portable_fixture
    settings = QSettings(str(tmp_path / 'workflow.ini'), QSettings.Format.IniFormat)
    monkeypatch.setattr(shell, 'QSettings', lambda: settings)
    monkeypatch.setattr(gui, 'QSettings', lambda: settings)
    monkeypatch.setattr(controller, 'default_artifact_cache_root', lambda: tmp_path / 'cache')
    monkeypatch.setattr(shell.MainWindow, 'INITIAL_PREVIEW_DELAY_MS', 600000)
    errors = []
    monkeypatch.setattr(shell.MainWindow, '_show_error', lambda self, message: errors.append(message))
    window = shell.MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    window._restore_working_point(portable)
    assert not errors
    state = portable.compatible_state()
    plane = state.sample.upper_surface_z_mm

    def execute(point_state):
        snapshot = capture_instrument_snapshot(point_state)
        gun, cp, mask = incident_checkpoints(point_state, [plane], step_mm=1.)
        arrays = {name: getattr(cp, name)[-1] for name in ('x_m', 'y_m', 'tx_rad', 'ty_rad')}
        arrays.update(alive=mask[-1], weight=gun.exit_bundle.weight, energy_offset_ev=gun.exit_bundle.energy_offset_ev)
        return WorkingPointCheckpoint(snapshot, arrays, plane, calculation_signatures(point_state)['incident'],
            dict(validation_status='NOT_RUN', scope='Nine-ray incident audit; no specimen/detector or convergence qualification'))

    first = execute(state)
    readout = sampling_summary(first.arrays, plane_z_mm=plane, source_current_a=state.electron_gun.emitted_current_a)
    assert readout['emitted_samples'] == 9
    initial_signatures = calculation_signatures(state)
    next(lens for lens in state.lenses if lens.key == 'condenser_lens_2').percent += .1
    assert calculation_signatures(state)['incident'] != initial_signatures['incident']
    second = execute(state)
    window.working_points.add_checkpoint(first, label='Executed A')
    window.working_points._pin('A')
    window.working_points.add_checkpoint(second, label='Executed B')
    window.working_points._pin('B')
    window.working_points._compare_pins()
    assert 'read-only' in window.working_points.status.text()
    before_alignment = capture_instrument_snapshot(state).digest
    request = AlignmentRequest.capture(state, 'nanoprobe_convergence', 25., revision=0,
        options=ConstrainedAlignment((), {'condenser_lens_2': (0., 100.), 'condenser_lens_3': (0., 100.)}, maximum_evaluations=4))
    with pytest.raises(AlignmentCancelled):
        solve_alignment_candidate(request, cancelled=lambda: True)
    failed = solve_alignment_candidate(request)
    assert failed.status != 'READY_TO_APPLY'
    assert capture_instrument_snapshot(state).digest == before_alignment
    path = tmp_path / 'moved.temwp'
    first.write_package(path, mode='inputs_and_results')
    _remove_fixture_location(root, tmp_path)
    loaded = WorkingPointCheckpoint.read_package(path)
    window._restore_working_point(loaded)
    window.preview_timer.stop()
    assert not errors
    rerun = execute(loaded.compatible_state())
    for name, value in first.arrays.items():
        np.testing.assert_array_equal(rerun.arrays[name], value)
    assert not root.exists()
    assert window.working_points.shutdown()
    window.close()


def test_task_pages_at_current_display_scale(tmp_path, monkeypatch, qtbot):
    from temsim.gui import main_window as shell, interactive_calculation as gui, calculation_controller as controller
    settings = QSettings(str(tmp_path / 'scaled.ini'), QSettings.Format.IniFormat)
    monkeypatch.setattr(shell, 'QSettings', lambda: settings)
    monkeypatch.setattr(gui, 'QSettings', lambda: settings)
    monkeypatch.setattr(controller, 'default_artifact_cache_root', lambda: tmp_path / 'cache')
    monkeypatch.setattr(shell.MainWindow, 'INITIAL_PREVIEW_DELAY_MS', 600000)
    window = shell.MainWindow()
    from temsim.app import APPLICATION_STYLE
    window.setStyleSheet(APPLICATION_STYLE)
    qtbot.addWidget(window)
    window.preview_timer.stop()
    window.resize(1500, 920)
    window.show()
    # Qt's virtual 800px offscreen monitor otherwise clamps restored geometry.
    # Exercise the requested logical viewport explicitly after each selection.
    monkeypatch.setattr(window, 'restoreGeometry', lambda _value: True)
    snapshot = capture_instrument_snapshot(window.state).digest
    monkeypatch.setattr(window.calculations.pool, 'start', lambda *_: pytest.fail('Presentation started physics'))
    destination = Path('tmp/report-ui')
    destination.mkdir(exist_ok=True)
    for name in ('instrument', 'alignment', 'experiments', 'results'):
        window.workspace_layouts.select('task_' + name)
        window.resize(1500, 920)
        qtbot.wait(40)
        assert window.width() == 1500 and window.height() == 920
        assert capture_instrument_snapshot(window.state).digest == snapshot
        assert not window.preview_timer.isActive()
        scale = window.devicePixelRatioF()
        assert window.grab().save(str(destination / f'{name}-{scale:g}.png'))
    window.workspace_layouts.select('task_experiments')
    window.resize(1500, 920)
    page = window.workspace.design_explorer
    from PySide6.QtWidgets import QTabWidget
    result_tabs = page.experiment_tools.parentWidget().parentWidget()
    assert isinstance(result_tabs, QTabWidget)
    result_tabs.setCurrentWidget(page.geometry_editor)
    page.geometry_editor.mode.setCurrentIndex(1)
    window.workspace.page_scroll.ensureWidgetVisible(page.geometry_editor)
    qtbot.wait(40)
    assert window.grab().save(str(destination / f'geometry-{scale:g}.png'))
    assert page.geometry_editor.constraints.enabled.isChecked()
    button = window.workspace.design_explorer.experiment_tools.load_button
    button.setFocus()
    QTest.keyClick(button, Qt.Key.Key_Tab)
    window.close()
