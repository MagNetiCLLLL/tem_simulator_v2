"""Keep scrolling separate from changing an input's value."""

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QSpinBox,
)


def _is_value_editor(widget):
    if isinstance(widget, (QAbstractSpinBox, QComboBox)):
        return True
    # A combo's popup list must still scroll normally.  Only include the
    # embedded text editor, not every descendant of the input widget.
    return isinstance(widget, QLineEdit) and isinstance(
        widget.parentWidget(), (QAbstractSpinBox, QComboBox)
    )


class _NumericWheelFilter(QObject):
    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Wheel:
            if _is_value_editor(watched):
                # An ignored event can continue to the enclosing scroll area;
                # returning True keeps it out of Qt's value-stepping handler.
                event.ignore()
                return True
        return False


def install_numeric_input_policy(application=None):
    """Protect native delegate/third-party editors as well as our own controls."""
    if application is None:
        application = QApplication.instance()
    if application is None:
        return
    if getattr(application, "_numeric_wheel_filter", None) is None:
        event_filter = _NumericWheelFilter(application)
        application.installEventFilter(event_filter)
        application._numeric_wheel_filter = event_filter


class _WheelSafeInput:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Panels are also constructed directly in embedded tools and tests,
        # where the application's usual entry point may never have run.
        install_numeric_input_policy()

    def wheelEvent(self, event):
        event.ignore()


class WheelSafeSpinBox(_WheelSafeInput, QSpinBox):
    """Integer input changed only by typing, keys or arrow buttons."""


class WheelSafeDoubleSpinBox(_WheelSafeInput, QDoubleSpinBox):
    """Floating-point input changed only by typing, keys or arrow buttons."""


class WheelSafeComboBox(_WheelSafeInput, QComboBox):
    """Selection input whose closed field does not respond to scrolling."""
