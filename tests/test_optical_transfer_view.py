"""Visual optical-transfer regressions using cached synthetic Jacobians."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.diagnostics import OpticalTransferRecord
from temsim.gui import diagnostic_tabs
from temsim.gui.diagnostic_tabs import OpticalTransferView
from temsim.physics.first_order import (
    DetectorFrameCalibration,
    MECHANICAL_SLOPES,
    SPECIMEN_CANONICAL_MOMENTUM,
    TransverseTransfer,
    linear_map_properties,
)


def _rotation(degrees):
    angle = np.deg2rad(degrees)
    return np.array(((np.cos(angle), -np.sin(angle)),
                     (np.sin(angle), np.cos(angle))))


def _record(image_map, angular_map, *, key="camera", name="Camera"):
    image_map = np.asarray(image_map, dtype=float)
    angular_map = np.asarray(angular_map, dtype=float)
    transfer = TransverseTransfer(
        source_z_mm=10.0,
        target_z_mm=20.0,
        j_img=image_map,
        j_diff_m_per_rad=angular_map,
        k_img_rad_per_m=np.zeros((2, 2)),
        k_diff=np.eye(2),
        position_offset_m=(0.012, -0.034),
        input_basis=SPECIMEN_CANONICAL_MOMENTUM,
    )
    return OpticalTransferRecord(
        key=key,
        name=name,
        z_mm=20.0,
        plane_role="diffraction_reference",
        inserted=True,
        transfer=transfer,
        image_properties=linear_map_properties(image_map),
        diffraction_properties=linear_map_properties(angular_map),
        detector_frame=DetectorFrameCalibration(
            key=key,
            axis_rotation_deg=68.0,
            flip_x=True,
        ),
    )


def _result(*records, mode="diffraction"):
    return SimpleNamespace(
        state_snapshot=SimpleNamespace(projector_mode=mode, lenses=()),
        simulation=SimpleNamespace(
            optical_transfers=records,
            metrics={"lambda_nm": 0.002},
        ),
        assembly=None,
    )


def _outline(panel):
    x, y = panel.outline.getData()
    if x is None:
        return np.empty((2, 0))
    return np.vstack((x, y))


def _assert_mapped_circle(panel, map_in_plot_units):
    points = _outline(panel)
    assert points.shape[1] >= 4
    assert points[:, 0] == pytest.approx(map_in_plot_units[:, 0])
    unit_points = np.linalg.solve(map_in_plot_units, points)
    assert np.linalg.norm(unit_points, axis=0) == pytest.approx(
        np.ones(points.shape[1]), abs=1.0e-12,
    )
    signed_area = np.sum(
        points[0] * np.roll(points[1], -1)
        - points[1] * np.roll(points[0], -1)
    )
    assert np.sign(signed_area) == np.sign(np.linalg.det(map_in_plot_units))


def test_response_plots_preserve_rotation_mirroring_and_physical_units(qtbot):
    view = OpticalTransferView()
    qtbot.addWidget(view)
    image_map = _rotation(31.0) @ np.diag((4.0, 1.5))
    angular_map = _rotation(-23.0) @ np.diag((-0.006, 0.002))

    view.display_result(_result(_record(image_map, angular_map)))

    # A 1 nm displacement is plotted in nm; a 1 mrad canonical-angle change in um.
    # Neither detector-frame rotation/flips nor the affine ray offset applies.
    _assert_mapped_circle(view.overview.position_response, image_map)
    _assert_mapped_circle(view.overview.angle_response, angular_map * 1000.0)
    angular_text = (
        view.overview.angle_response.metrics.text()
        + view.overview.angle_response.status.text()
    ).lower()
    assert "mirror" in angular_text


def test_conjugacy_comes_from_matrices_not_mode_or_named_plane(qtbot):
    view = OpticalTransferView()
    qtbot.addWidget(view)
    identity = np.eye(2)
    zero = np.zeros((2, 2))

    for image_map, angular_map, expected in (
        (identity, zero, "image"),
        (zero, identity, "diffraction"),
        (identity, identity, "mixed"),
        (zero, zero, "degenerate"),
    ):
        record = _record(image_map, angular_map, name="Diffraction reference")
        view.display_result(_result(record, mode="diffraction"))
        assert expected in view.overview.plane_kind.text().lower()


@pytest.mark.parametrize("small_value", [0.0, 5.0e-16])
def test_zero_or_subthreshold_map_has_no_claimed_orientation(qtbot, small_value):
    view = OpticalTransferView()
    qtbot.addWidget(view)
    zero = small_value * np.eye(2)

    view.display_result(_result(_record(zero, zero)))

    for panel in (view.overview.position_response, view.overview.angle_response):
        points = _outline(panel)
        assert points.size > 0
        scale = 1000.0 if panel.angular else 1.0
        assert points[:, 0] == pytest.approx((small_value * scale, 0.0), abs=1.0e-30)
        text = (panel.metrics.text() + " " + panel.status.text()).lower()
        assert "unavailable" in text
        if small_value:
            assert "threshold" in text
            assert "collapses to the origin" not in text
        assert np.all(np.isfinite(panel.plot.getViewBox().viewRange()))


def test_target_changes_reuse_cached_records_and_keep_numeric_details(qtbot, monkeypatch):
    def forbidden_trace(_state):
        pytest.fail("Changing an optical-transfer display must reuse cached records")

    monkeypatch.setattr(diagnostic_tabs, "optical_transfer_records", forbidden_trace)
    view = OpticalTransferView()
    qtbot.addWidget(view)
    first = _record(np.eye(2), np.zeros((2, 2)), key="first")
    second = _record(7.0 * np.eye(2), 0.003 * np.eye(2), key="second")

    view.display_result(_result(first, second))

    assert view.display_tabs.tabText(view.display_tabs.currentIndex()) == "Overview"
    assert "J_img" in view.matrix_text.toPlainText()
    assert "J_diff" in view.matrix_text.toPlainText()
    assert "eta = p_perp(canonical) / p0" in view.matrix_text.toPlainText()
    first_tooltip = view.overview.plane_kind.toolTip()
    assert "Image plane" in first_tooltip
    view.target_plane.setCurrentIndex(view.target_plane.findData("second"))
    assert "Mixed plane" in view.overview.plane_kind.toolTip()
    _assert_mapped_circle(view.overview.position_response, second.transfer.j_img)
    _assert_mapped_circle(
        view.overview.angle_response, second.transfer.j_diff_m_per_rad * 1000.0,
    )
    view.target_plane.setCurrentIndex(view.target_plane.findData("first"))
    assert view.overview.plane_kind.toolTip() == first_tooltip
    _assert_mapped_circle(view.overview.position_response, first.transfer.j_img)


def test_legacy_optical_records_are_rebuilt_in_canonical_coordinates(qtbot, monkeypatch):
    view = OpticalTransferView()
    qtbot.addWidget(view)
    canonical = _record(3.0 * np.eye(2), np.zeros((2, 2)))
    old_record = _record(np.eye(2), 0.3 * np.eye(2))
    mechanical = replace(old_record.transfer, input_basis=MECHANICAL_SLOPES)
    untagged = SimpleNamespace(
        j_img=mechanical.j_img,
        j_diff_m_per_rad=mechanical.j_diff_m_per_rad,
        position_offset_m=mechanical.position_offset_m,
    )
    calls = []

    def rebuild(snapshot):
        calls.append(snapshot)
        return (canonical,)

    monkeypatch.setattr(diagnostic_tabs, "optical_transfer_records", rebuild)
    for index, old_transfer in enumerate((mechanical, untagged), start=1):
        result = _result(replace(old_record, transfer=old_transfer))
        view.display_result(result)
        assert len(calls) == index
        assert calls[-1] is result.state_snapshot
        assert "image" in view.overview.plane_kind.text().lower()
        _assert_mapped_circle(view.overview.position_response, canonical.transfer.j_img)
        view._display_selected_record()
        assert len(calls) == index
        assert result.simulation.optical_transfers[0].transfer is old_transfer


def test_missing_snapshot_clears_previous_visual_and_numeric_result(qtbot):
    view = OpticalTransferView()
    qtbot.addWidget(view)
    view.display_result(_result(_record(np.eye(2), np.eye(2))))
    assert _outline(view.overview.position_response).size > 0

    view.display_result(SimpleNamespace(state_snapshot=None))

    assert view.target_plane.count() == 0
    assert view.matrix_text.toPlainText() == ""
    assert not view.capture_current.isEnabled()
    assert _outline(view.overview.position_response).size == 0
    assert _outline(view.overview.angle_response).size == 0
    assert "Model only" in view.overview.plane_kind.toolTip()
