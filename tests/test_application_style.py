from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QCheckBox

from temsim.app import APPLICATION_STYLE, _CHECKBOX_ASSET_ROOT


CHECKBOX_ASSETS = (
    "checkbox_unchecked.svg",
    "checkbox_unchecked_hover.svg",
    "checkbox_unchecked_disabled.svg",
    "checkbox_checked.svg",
    "checkbox_checked_hover.svg",
    "checkbox_checked_disabled.svg",
    "checkbox_indeterminate.svg",
    "checkbox_indeterminate_hover.svg",
    "checkbox_indeterminate_disabled.svg",
)


def _render_checkbox(qtbot, state: Qt.CheckState) -> QImage:
    checkbox = QCheckBox()
    checkbox.setTristate(True)
    checkbox.setCheckState(state)
    checkbox.resize(32, 28)
    qtbot.addWidget(checkbox)
    checkbox.show()
    QApplication.processEvents()
    image = QImage(
        checkbox.size(),
        QImage.Format.Format_ARGB32_Premultiplied,
    )
    image.fill(QColor("#111827"))
    checkbox.render(image)
    return image


def _near_colour_count(image: QImage, colour: QColor, tolerance: int = 4) -> int:
    target = colour.getRgb()[:3]
    count = 0
    for y in range(image.height()):
        for x in range(image.width()):
            observed = image.pixelColor(x, y).getRgb()[:3]
            if all(
                abs(channel - expected) <= tolerance
                for channel, expected in zip(observed, target, strict=True)
            ):
                count += 1
    return count


def test_checkbox_style_assets_are_packaged_and_resolved():
    root = Path(_CHECKBOX_ASSET_ROOT)

    assert "__CHECKBOX_ASSET_ROOT__" not in APPLICATION_STYLE
    assert "QCheckBox::indicator:checked" in APPLICATION_STYLE
    assert all((root / name).is_file() for name in CHECKBOX_ASSETS)


def test_checkbox_checked_and_partial_states_have_distinct_high_contrast_fill(
    qtbot,
):
    application = QApplication.instance()
    previous_style = application.styleSheet()
    application.setStyleSheet(APPLICATION_STYLE)
    try:
        unchecked = _render_checkbox(qtbot, Qt.CheckState.Unchecked)
        checked = _render_checkbox(qtbot, Qt.CheckState.Checked)
        partial = _render_checkbox(qtbot, Qt.CheckState.PartiallyChecked)
    finally:
        application.setStyleSheet(previous_style)

    checked_blues = (QColor("#0284c7"), QColor("#0ea5e9"))
    partial_violets = (QColor("#7c3aed"), QColor("#8b5cf6"))
    unchecked_borders = (QColor("#f8fafc"), QColor("#7dd3fc"))
    assert sum(
        _near_colour_count(checked, colour) for colour in checked_blues
    ) >= 20
    assert sum(
        _near_colour_count(unchecked, colour) for colour in checked_blues
    ) == 0
    assert sum(
        _near_colour_count(partial, colour) for colour in partial_violets
    ) >= 20
    assert sum(
        _near_colour_count(unchecked, colour) for colour in unchecked_borders
    ) >= 10
