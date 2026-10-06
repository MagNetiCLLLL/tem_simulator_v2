"""Scan images keep physical coordinates and honest preview contrast."""

from types import SimpleNamespace

import numpy as np
import pytest

from temsim.detector.stem_signal import StemScanResult
from temsim.gui.scan_panel import ScanControlView
from temsim.optics.column import default_state


def _frame(*, pitch_nm=.02, model="geometric_detector_interception", material=False):
    coordinates = (np.arange(32) - 15.5) * pitch_nm * 1e-3
    x, y = np.meshgrid(coordinates, coordinates)
    bf = np.full((32, 32), 3531. / 15000)
    bf[:, 16:] = 3532. / 15000
    return StemScanResult(
        scan_x_um=x, scan_y_um=y,
        fractions={"haadf": np.full_like(bf, 16. / 15000),
                   "df": np.full_like(bf, 2423. / 15000), "bf": bf},
        detector_signals={},
        metrics={"model": model, "scan_pixel_size_nm": pitch_nm,
                 "scan_field_of_view_x_nm": 32 * pitch_nm,
                 "scan_field_of_view_y_nm": 32 * pitch_nm,
                 "finite_specimen_exit_used": material,
                 **({"detector_sampling": {"coverage_complete": True, "detectors": {}}}
                    if model == "multislice_angle_resolved" else {})},
    )


@pytest.fixture
def view(qtbot):
    widget = ScanControlView()
    qtbot.addWidget(widget)
    widget.set_state(default_state())
    widget.resize(1100, 700)
    widget.result_tabs.setCurrentIndex(1)
    widget.show()
    return widget


def test_picometre_scan_has_one_si_prefix_and_correct_tick_values(view, qtbot):
    frame = _frame()
    view._set_stem_frame(frame)
    plot = view.detector_image_views["bf"]
    plot.setRange(xRange=(-.00032, .00032), yRange=(-.00032, .00032), padding=0)
    for axis_name in ("bottom", "left"):
        axis = plot.getAxis(axis_name)
        qtbot.waitUntil(lambda: axis.labelUnitPrefix == "p")
        assert axis.labelUnits == "m"
        assert "pm" in axis.label.toPlainText()
        assert "µµm" not in axis.label.toPlainText()
        labels = axis.tickStrings([-.0003, 0., .0003], axis.scale * axis.autoSIPrefixScale, .0001)
        assert [float(value) for value in labels] == [-300., 0., 300.]
    item = view.detector_image_items["bf"]
    bounds = item.mapRectToParent(item.boundingRect())
    assert bounds.width() == pytest.approx(.00064)  # µm = 640 pm.
    assert "FOV 640 pm" in view.image_model_notice.text()
    # The geometry and 4D-STEM pages share the same µm data convention.
    for other in (view.sample_plot, view.downstream_plot, view.fourdstem_image_view):
        assert other.getAxis("bottom").labelUnits == "m"
        assert other.getAxis("bottom").scale == 1e-6


@pytest.mark.parametrize("model", ["geometric_detector_interception", "multislice_angle_resolved",
                                   "projected_atomic_scattering"])
def test_positive_constant_signal_is_rendered_midgray_and_zero_stays_black(view, model):
    frame = _frame(model=model)
    frame.fractions["df"].fill(0.)
    original = {key: values.copy() for key, values in frame.fractions.items()}
    view._set_stem_frame(frame)
    positive = view.detector_image_items["haadf"]
    positive.render()
    assert 126 <= positive.qimage.pixelColor(0, 0).red() <= 128
    assert "Constant signal: 0.00106666666667" in view.detector_contrast_labels["haadf"].text()
    assert "Uniform mid-gray" in view.detector_contrast_labels["haadf"].text()
    zero = view.detector_image_items["df"]
    zero.render()
    assert zero.qimage.pixelColor(0, 0).red() == 0
    assert "Constant signal: 0" in view.detector_contrast_labels["df"].text()
    for key in original:
        np.testing.assert_array_equal(frame.fractions[key], original[key])
        np.testing.assert_array_equal(view.detector_image_items[key].image, original[key].T)


def test_fixed_fraction_preserves_small_changes_without_stretching_and_auto_is_optional(view):
    frame = _frame(material=True)
    original = frame.fractions["bf"].copy()
    view._set_stem_frame(frame)
    item = view.detector_image_items["bf"]
    assert view.image_contrast_mode.currentData() == "auto"
    assert tuple(item.levels) == (3531. / 15000, 3532. / 15000)
    view.image_contrast_mode.setCurrentIndex(view.image_contrast_mode.findData("fixed_fraction"))
    assert tuple(item.levels) == (0., 1.)
    item.render()
    pixels = [item.qimage.pixelColor(x, y).red() for x, y in ((0, 0), (31, 31))]
    assert max(pixels) - min(pixels) <= 1
    assert min(pixels) > 0
    assert "Fixed fraction scale: 0–1" in view.detector_contrast_labels["bf"].text()
    assert "0.235466667" in view.detector_contrast_labels["bf"].text()
    assert "Material particle preview" in view.image_model_notice.text()
    assert "CIF atomic contrast not calculated" in view.image_model_notice.text()
    assert "selected CIF not used" not in view.image_model_notice.text()
    np.testing.assert_array_equal(frame.fractions["bf"], original)
    # The display choice also applies to historical wave frames.
    wave = _frame(model="multislice_angle_resolved")
    view._set_stem_frame(wave)
    assert tuple(item.levels) == (0., 1.)
    view.image_contrast_mode.setCurrentIndex(view.image_contrast_mode.findData("auto"))
    assert tuple(item.levels) == (3531. / 15000, 3532. / 15000)
    assert "Auto contrast" in view.detector_contrast_labels["bf"].text()


def test_projected_atom_notice_uses_its_limitations_and_does_not_offer_wave(view):
    frame = _frame(model="projected_atomic_scattering")
    frame.metrics.update({
        "model_limitation": "Finite angular quadrature limited to 200 mrad.",
        "approximation_notes": ("Independent atom contributions; no interference.",),
        "probe_sigma_principal_nm": (.025, .04),
        "maximum_plural_event_fraction": .0123,
        "maximum_numerical_truncation_fraction": .00456,
    })
    view._set_stem_frame(frame)
    assert "Projected-atom scattering approximation" in view.image_model_notice.text()
    assert "CIF atomic contrast not calculated" not in view.image_model_notice.text()
    assert "Finite angular quadrature limited to 200 mrad" in view.image_model_notice.toolTip()
    assert "no interference" in view.image_model_notice.toolTip()
    assert "channeling" in view.image_model_notice.toolTip()
    assert "Probe σ 0.025/0.04 nm" in view.image_model_notice.text()
    assert "plural ≤ 0.0123" in view.image_model_notice.text()
    assert "not fitted widths" in view.image_model_notice.toolTip()
    assert "not an acceptance threshold" in view.image_model_notice.toolTip()
    assert "unresolved ≤ 0.456%" in view.image_model_notice.text()
    assert "not physical absorption" in view.image_model_notice.toolTip()
    assert view.enable_wave_images.isHidden()
    assert not view._state.sample.stem_wave_enabled


def test_particle_model_selection_updates_only_its_owned_setting_and_waits_for_calculate(view):
    from copy import deepcopy

    assert view._state.sample.stem_particle_model == "projected_atoms"
    assert view.stem_particle_model.currentData() == "projected_atoms"
    assert view.stem_particle_model.itemText(1) == "Material particle paths"
    original = deepcopy(view._state.to_dict())
    changed, requested = [], []
    view.parameters_changed.connect(changed.append)
    view.calculation_requested.connect(lambda: requested.append(True))
    view.stem_particle_model.setCurrentIndex(view.stem_particle_model.findData("material_paths"))
    assert view._state.sample.stem_particle_model == "material_paths"
    assert changed == ["sample.stem_particle_model"]
    assert not requested
    assert "Inputs changed" in view.calculation_bar.status.text()
    updated = view._state.to_dict()
    updated["sample"]["stem_particle_model"] = original["sample"]["stem_particle_model"]
    assert updated == original
    view.set_state(view._state)
    assert view.stem_particle_model.currentData() == "material_paths"
    assert changed == ["sample.stem_particle_model"]
    assert not view._state.sample.stem_wave_enabled


def test_particle_model_does_not_silently_select_another_model_for_invalid_state(view):
    view._state.sample.stem_particle_model = "unsupported-model"
    changes = []
    view.parameters_changed.connect(changes.append)
    view.set_state(view._state)
    assert view.stem_particle_model.currentData() is None
    assert view._state.sample.stem_particle_model == "unsupported-model"
    assert not changes


def test_contrast_change_uses_retained_frame_until_replacement_without_requesting_work(view, monkeypatch):
    from copy import deepcopy

    frame = _frame(model="projected_atomic_scattering")
    view._set_stem_frame(frame)
    captured = view._stem_frame_context
    view._state.sample.thickness_nm += 1.
    view.set_state(view._state)
    view.mark_stem_frame_stale()
    new = _frame(model="projected_atomic_scattering")
    new.fractions["bf"] *= .5
    original = {key: values.copy() for key, values in frame.fractions.items()}
    state = deepcopy(view._state.to_dict())
    requests = []
    view.calculation_requested.connect(lambda: requests.append("calculate"))
    view.parameters_changed.connect(requests.append)
    monkeypatch.setattr("temsim.detector.stem_signal.acquire_stem_scan",
                        lambda *_a, **_k: pytest.fail("Display must not calculate"))
    view.image_contrast_mode.setCurrentIndex(view.image_contrast_mode.findData("fixed_fraction"))
    item = view.detector_image_items["bf"]
    assert tuple(item.levels) == (0., 1.)
    np.testing.assert_array_equal(item.image, frame.fractions["bf"].T)
    view.image_contrast_mode.setCurrentIndex(view.image_contrast_mode.findData("auto"))
    assert tuple(item.levels) == (float(original["bf"].min()), float(original["bf"].max()))
    assert "0.2354" in view.detector_contrast_labels["bf"].text()
    assert not requests and view._state.to_dict() == state
    assert view._stem_frame is frame and view._stem_frame_context is captured
    for key, values in original.items():
        np.testing.assert_array_equal(frame.fractions[key], values)
    view._set_stem_frame(new)
    assert view._stem_frame is new and not view._stem_frame_stale
    np.testing.assert_array_equal(item.image, new.fractions["bf"].T)
    assert tuple(item.levels) == (float(new.fractions["bf"].min()), float(new.fractions["bf"].max()))
    assert not requests and view._state.to_dict() == state


def test_fixed_fraction_constant_remains_uniform_at_its_actual_brightness(view):
    frame = _frame(model="projected_atomic_scattering")
    frame.fractions["bf"].fill(.25)
    view._set_stem_frame(frame)
    view.image_contrast_mode.setCurrentIndex(view.image_contrast_mode.findData("fixed_fraction"))
    item = view.detector_image_items["bf"]
    item.render()
    assert tuple(item.levels) == (0., 1.)
    assert item.qimage.pixelColor(0, 0).red() == item.qimage.pixelColor(31, 31).red()
    assert 62 <= item.qimage.pixelColor(0, 0).red() <= 64
    assert "Constant signal: 0.25" in view.detector_contrast_labels["bf"].text()
    assert "fixed fraction scale" in view.detector_contrast_labels["bf"].text()


def test_unqualified_wave_action_preserves_setting_and_retained_image(view, qtbot):
    view._state.sample.stem_wave_enabled = False
    view.set_state(view._state)
    assert "coherent imaging paused" in view.image_model_notice.text()
    frame = _frame()
    view._set_stem_frame(frame)
    view._state.sample.thickness_nm += 1.
    view.mark_stem_frame_stale()
    assert "selected CIF not used" in view.image_model_notice.text()
    qtbot.waitUntil(view.enable_wave_images.isVisible)
    changes = []
    view.parameters_changed.connect(changes.append)
    before = frame.fractions["bf"].copy()
    view.enable_wave_images.click()
    assert changes == []
    assert not view._state.sample.stem_wave_enabled and not view.wave_scan_enabled.isChecked()
    assert "not qualified" in view.wave_image_action_note.text()
    assert "Resume refresh" not in view.wave_image_action_note.text()
    assert view._stem_frame is frame and view._stem_frame_stale
    np.testing.assert_array_equal(view.detector_image_items["bf"].image, before.T)


def test_retained_and_bank_frames_keep_capture_context_until_new_frame_arrives(view):
    captured = default_state()
    captured.sample.specimen_mode = "atomic"
    captured.sample.cif_path = "capture-A.cif"
    captured.stem_detectors[0].z_mm = 1234.
    old = _frame()
    view.display_result(None, old, complete=True, state_snapshot=captured)
    view._state.sample.cif_path = "live-B.cif"
    view._state.sample.specimen_mode = "atomic"
    view._state.stem_detectors[0].z_mm = 9876.
    view.set_state(view._state)
    view.mark_stem_frame_stale()
    assert "capture-A.cif" in view.image_model_notice.toolTip()
    assert "Geometry preview only" in view.image_model_notice.text()
    assert "1234" in view.detector_geometry_labels[captured.stem_detectors[0].key].text()
    assert view._stem_frame is old and view._stem_image_rect(view._stem_frame).width() == pytest.approx(.00064)

    bank_state = default_state()
    bank_state.sample.specimen_mode = "atomic"
    bank_state.sample.cif_path = "bank-C.cif"
    bank = _frame(pitch_nm=.05, material=True)
    view.set_bank_readout(SimpleNamespace(stem=bank, state_snapshot=bank_state))
    bank_state.sample.cif_path = "changed-after-capture.cif"
    view.image_source.setCurrentIndex(view.image_source.findData("bank"))
    assert "bank-C.cif" in view.image_model_notice.toolTip()
    assert "Material particle preview" in view.image_model_notice.text()
    assert "FOV 1.6 nm" in view.image_model_notice.text()
    item = view.detector_image_items["bf"]
    assert item.mapRectToParent(item.boundingRect()).width() == pytest.approx(.0016)
    view.image_source.setCurrentIndex(view.image_source.findData("current"))
    assert "capture-A.cif" in view.image_model_notice.toolTip()
    assert "FOV 640 pm" in view.image_model_notice.text()
    view.image_source.setCurrentIndex(view.image_source.findData("bank"))
    new = _frame(pitch_nm=1., model="multislice_angle_resolved")
    view.display_result(None, new, complete=True, state_snapshot=view._state)
    assert view._stem_frame is new
    assert "bank-C.cif" in view.image_model_notice.toolTip()
    assert item.mapRectToParent(item.boundingRect()).width() == pytest.approx(.0016)
    view.image_source.setCurrentIndex(view.image_source.findData("current"))
    assert view._stem_frame_context.cif_path == "live-B.cif"
    assert "9876" in view.detector_geometry_labels[captured.stem_detectors[0].key].text()
    assert "Specimen image" in view.image_model_notice.text()
    assert "FOV 32 nm" in view.image_model_notice.text()
    assert view.enable_wave_images.isHidden()
    assert item.mapRectToParent(item.boundingRect()).width() == pytest.approx(.032)


def test_stale_notice_retains_displayed_model_information(view):
    frame = _frame(material=True)
    view._set_stem_frame(frame)
    view.mark_stem_frame_stale()
    assert "inputs changed" in view.image_model_notice.text()
    assert "Material particle preview" in view.image_model_notice.text()
    assert "CIF atomic contrast not calculated" in view.image_model_notice.text()
