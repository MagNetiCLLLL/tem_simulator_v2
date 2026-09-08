"""DF fitting reviews current optics and saves one real module transaction."""

from dataclasses import replace
from types import SimpleNamespace
import shutil

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QLabel, QPushButton

from temsim import module_manifest
from temsim.assembly_catalog import AssemblyCatalog
from temsim.calculation_cache import calculation_signatures
from temsim.detector.stem_signal import StemScanResult
from temsim.gui import main_window
from temsim.gui.scan_panel import ScanControlView
from temsim.manifest_editor import ManifestEditor
from temsim.optics.column import default_state
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.physics.stem_sampling import detector_sampling_report


def frame_for(state):
    report = detector_sampling_report(
        {"bf": (0., 1.), "df": (1., 7.), "haadf": (60., 120.)},
        maximum_angle_mrad=169., wavelength_angstrom=.019687,
        requested_fov_angstrom=40., requested_grid_pixels=1024,
        bandwidth_fraction=2 / 3, probe_semiangle_mrad=24.8,
    )
    x, y = np.meshgrid([.001, .002], [.003, .004])
    return StemScanResult(x, y, {key: np.ones((2, 2)) * .1 for key in ("haadf", "df", "bf")}, {},
                          metrics={"model": "multislice_angle_resolved", "detector_sampling": report,
                                   "sampling_state_signature": calculation_signatures(state)["stem"]})


def test_df_action_guards_bank_stale_paused_and_changed_camera_length(qtbot):
    view = ScanControlView()
    qtbot.addWidget(view)
    state = default_state()
    view.set_state(state)
    frame = frame_for(state)
    view._set_stem_frame(frame)
    requested, errors = [], []
    view.df_geometry_requested.connect(requested.append)
    view.error.connect(errors.append)
    assert view.exclude_direct_beam.isEnabled()
    view.exclude_direct_beam.click()
    assert requested == [frame]
    state.lenses[-1].percent += .001
    view._request_df_geometry()  # Slot guard also protects a not-yet-refreshed button.
    assert len(requested) == 1 and "camera length" in errors[-1]
    state.lenses[-1].percent -= .001
    view.set_bank_readout(SimpleNamespace(stem=frame))
    view.image_source.setCurrentIndex(view.image_source.findData("bank"))
    assert not view.exclude_direct_beam.isEnabled()
    view._request_df_geometry()
    assert len(requested) == 1
    view.image_source.setCurrentIndex(view.image_source.findData("current"))
    view.pause_image_refresh.setChecked(True)
    newer = frame_for(state)
    view._set_stem_frame(newer)
    assert not view.exclude_direct_beam.isEnabled()
    view._request_df_geometry()
    assert len(requested) == 1
    view.pause_image_refresh.setChecked(False)
    view.mark_stem_frame_stale()
    assert not view.exclude_direct_beam.isEnabled()


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    source_bytes = {path: path.read_bytes() for path in INSTRUMENT_CONFIG_ROOT.rglob("*.toml")}
    root = tmp_path / "instruments"
    shutil.copytree(INSTRUMENT_CONFIG_ROOT, root)
    monkeypatch.setattr(main_window, "AssemblyCatalog", lambda: AssemblyCatalog(root))
    monkeypatch.setattr(main_window, "ManifestEditor", lambda: ManifestEditor(root))
    monkeypatch.setattr(main_window, "QSettings", lambda: QSettings(str(tmp_path / "ui.ini"), QSettings.Format.IniFormat))
    monkeypatch.setattr(main_window.MainWindow, "schedule_preview", lambda *_: None)
    widget = main_window.MainWindow()
    qtbot.addWidget(widget)
    widget.preview_timer.stop()
    widget.errors_for_test = []
    monkeypatch.setattr(widget, "_show_error", widget.errors_for_test.append)
    frame = frame_for(widget.state)
    result = SimpleNamespace(stem_scan=frame, state_snapshot=widget.state, simulation=SimpleNamespace())
    widget.workspace._high_accuracy_result = result
    widget.workspace.scan_control._set_stem_frame(frame, state_snapshot=widget.state)
    # GUI transaction tests use an explicitly synthetic supported proposal;
    # numerical fitting and sequential interception have their own physics tests.
    proposal = SimpleNamespace(
        supported=True, detail="Synthetic fixture using current signed maps.",
        inner_diameter_mm=8., outer_width_mm=16., angular_inner_mrad=30., angular_outer_mrad=50.,
        direct_disk_max_radius_mm=3., direct_disk_clearance_mm=1.,
        camera_length_min_m=.8, camera_length_max_m=1.2,
    )
    widget.proposal_calls = []
    def propose(*args, **kwargs):
        widget.proposal_calls.append((args, kwargs))
        return proposal
    monkeypatch.setattr("temsim.physics.dark_field_geometry.propose_dark_field_geometry", propose)
    monkeypatch.setattr(widget, "_df_geometry_plan_inputs", lambda *_: (object(), np.zeros((2, 2, 2)), 24.8, (.1, .2)))
    yield widget, frame, root
    if widget._df_geometry_dialog is not None:
        widget._df_geometry_dialog.reject()
    assert all(path.read_bytes() == contents for path, contents in source_bytes.items())


def save_button(dialog):
    return dialog.findChild(QPushButton, "saveDfDimensions")


def test_review_saves_both_df_dimensions_in_actual_module_and_retains_old_frame(window):
    widget, frame, root = window
    part = widget.assembly.part("df")
    path = root / part.source_file
    before = {item["key"]: item for item in module_manifest.read_document(path)["parts"]}
    lens_values = [(item.key, item.percent) for item in widget.state.lenses]
    widget.assembly_panel.select_key("df")
    panel = widget.parameter_panel
    row = next(index for index, field in enumerate(panel._manifest_fields) if field.path[-1] == "vacuum_inner_diameter_mm")
    panel.manifest_table.item(row, 1).setText("unfinished draft")
    dialog = widget._review_df_geometry(frame)
    assert dialog is not None, widget.errors_for_test
    details = dialog.findChild(QLabel, "dfGeometryProposalDetails").text()
    assert part.source_file in details and "0.8–1.2 m" in details
    assert "Jimg" in details and "HAADF" in details
    assert widget.proposal_calls[0][1]["chamber_inner_diameter_mm"] == before["post_projector_detector_chamber"]["mechanical_inner_diameter_mm"]
    save_button(dialog).click()
    after = {item["key"]: item for item in module_manifest.read_document(path)["parts"]}
    assert after["df"]["inner_diameter_mm"] == 8.
    assert after["df"]["outer_width_mm"] == 16.
    assert after["haadf"] == before["haadf"]
    assert after["df"]["optical_reference_local_z_mm"] == before["df"]["optical_reference_local_z_mm"]
    assert [(item.key, item.percent) for item in widget.state.lenses] == lens_values
    view = widget.workspace.scan_control
    assert view._stem_frame is frame and view._stem_frame_stale
    assert "inputs changed" in view.image_model_notice.text()
    np.testing.assert_array_equal(view.detector_image_items["df"].image, frame.fractions["df"].T)
    assert panel.manifest_table.item(row, 1).text() == "unfinished draft"


@pytest.mark.parametrize("change", ["projector", "source_file", "save_failure"])
def test_review_rejects_stale_camera_length_external_file_and_failed_save(window, monkeypatch, change):
    widget, frame, root = window
    path = root / widget.assembly.part("df").source_file
    dialog = widget._review_df_geometry(frame)
    assert dialog is not None
    if change == "projector":
        widget.state.lenses[-1].percent += .01
    elif change == "source_file":
        path.write_bytes(path.read_bytes() + b"\n# external change during review\n")
    else:
        def fail(*_args):
            raise ValueError("Synthetic validation failure")
        monkeypatch.setattr(widget, "_save_geometry_updates_preserving_drafts", fail)
    before = path.read_bytes()
    save_button(dialog).click()
    assert path.read_bytes() == before
    assert dialog.isVisible()
    error = dialog.findChild(QLabel, "dfGeometryProposalError").text()
    assert "not saved" in error
    if change == "projector":
        assert "camera length" in error
    assert widget.workspace.scan_control._stem_frame is frame


def test_cancel_and_missing_chamber_never_save_or_guess_geometry(window):
    widget, frame, root = window
    path = root / widget.assembly.part("df").source_file
    before = path.read_bytes()
    dialog = widget._review_df_geometry(frame)
    dialog.reject()
    assert path.read_bytes() == before
    widget.assembly = replace(widget.assembly, parts=tuple(
        item for item in widget.assembly.parts if item.key != "post_projector_detector_chamber"))
    assert widget._review_df_geometry(frame) is None
    assert "chamber mechanical inner diameter" in widget.errors_for_test[-1]
    assert len(widget.proposal_calls) == 1


@pytest.mark.parametrize("has_simulation_time", [True, False])
def test_plan_inputs_include_beam_centre_baseline_origin_and_all_raster_times(monkeypatch, has_simulation_time):
    kick_times = []
    def kick(time):
        kick_times.append(time)
        return (1., 2.)
    ac = SimpleNamespace(scan_pixels_x=2, scan_lines=2, scan_frame_period_s=4.,
                         scan_kick_mrad=kick)
    state = SimpleNamespace(corrector_elements=[], ac_deflector=ac, sample=SimpleNamespace(z_mm=100.))
    if has_simulation_time:
        state.simulation_time_s = 1.
    report = {"coverage_complete": True, "detectors": {}, "probe_semiangle_mrad": 24.8}
    x, y = np.meshgrid([1., 2.], [3., 4.])  # Stored coordinates already include scan origin.
    frame = SimpleNamespace(scan_x_um=x, scan_y_um=y, metrics={"detector_sampling": report})
    statistics = dict(mean_x_m=10e-6, mean_y_m=20e-6, mean_tx_rad=.003, mean_ty_rad=-.004)
    monkeypatch.setattr("temsim.physics.scan_geometry.calibrate_scan_system", lambda _state: None)
    monkeypatch.setattr("temsim.physics.scan_geometry.paired_kick_response", lambda *_: np.diag([2., .5]))
    monkeypatch.setattr("temsim.physics.wave_imaging._weighted_ray_statistics", lambda _: statistics)
    plans = []
    def build(snapshot, *, scan_times_s):
        plans.append((snapshot, scan_times_s.copy()))
        return "plan"
    monkeypatch.setattr("temsim.physics.record_plane.build_record_plane_plan", build)
    result = SimpleNamespace(state_snapshot=state, simulation=SimpleNamespace(incident=object()))
    plan, positions, alpha, chief = main_window.MainWindow._df_geometry_plan_inputs(result, frame)
    assert plan == "plan" and plans[0][0] is not state
    assert kick_times == [1. if has_simulation_time else 0.]
    np.testing.assert_allclose(positions, np.stack((x, y), axis=-1) * 1e-6 + np.array([10e-6, 20e-6]) - np.array([.002, .001]))
    np.testing.assert_allclose(plans[0][1], [[.5, 1.5], [2.5, 3.5]])
    np.testing.assert_allclose(chief, [3., -4.])
    assert alpha == 24.8
