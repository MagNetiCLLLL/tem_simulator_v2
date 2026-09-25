"""Fixed display sizes for cached beam plots; no numerical model inputs."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QTimer, Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QGridLayout, QHBoxLayout, QLabel, QLayout,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from temsim.gui.input_policy import WheelSafeSpinBox


class TransversePlotLayout(QObject):
    """Size plot widgets in Qt logical pixels, independently of window space."""

    DEFAULTS = {"beam": (400, 360), "legend": (184, 184)}
    LABELS = {"beam": "Source and selected-plane plots", "legend": "Colour legend"}
    MINIMUMS = {"beam": (240, 160), "legend": (120, 120)}
    MAXIMUM = 2400

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self._sizes = {}
        self.dialog = None
        self.button = QPushButton("Plot sizes…")
        self.button.setObjectName("transversePlotSizesButton")
        self.button.setToolTip("Set one fixed width and height for both beam plots. Saved with the workspace layout.")
        self.button.clicked.connect(self.open_dialog)
        toolbar = QHBoxLayout()
        toolbar.addStretch(1)
        toolbar.addWidget(self.button)

        self.content = QWidget()
        self.content.setObjectName("transversePlotContent")
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        for panel in (owner.initial_beam_panel, owner.section_beam_panel):
            content_layout.addWidget(panel, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        content_layout.addStretch(1)
        self.scroll = QScrollArea()
        self.scroll.setObjectName("transversePlotScrollArea")
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumSize(0, 0)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll.setWidget(self.content)
        # Equal outer sizes also need equal axis gutters, otherwise the source
        # and selected-plane plotting rectangles still have different sizes.
        for plot in (owner.source_plot.plot, owner.plot):
            plot.getAxis("left").setWidth(76)
            plot.getAxis("bottom").setHeight(46)
            # AxisItem nudges the bottom title 5 px outside its own boundary.
            # Reserve that overhang within the fixed picture, including after
            # labels change between position and angular coordinates.
            plot.getPlotItem().layout.setContentsMargins(1, 1, 1, 8)
        layout = QVBoxLayout(owner)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(toolbar)
        layout.addWidget(self.scroll, 1)
        self._height_timer = QTimer(owner)
        self._height_timer.setSingleShot(True)
        self._height_timer.timeout.connect(self._refresh_panel_heights)
        self._height_panels = (owner.source_plot, owner.initial_beam_panel, owner.section_beam_panel)
        for panel in self._height_panels:
            panel.installEventFilter(self)
        self.set_state(None, emit=False)

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.LayoutRequest, QEvent.Type.Resize, QEvent.Type.Show):
            self._height_timer.start(0)
        return False

    def _refresh_panel_heights(self):
        # Qt's ordinary minimumSizeHint can be lower than heightForWidth for
        # wrapped captions. Enforce the latter so a scroll viewport cannot push
        # a caption upwards over its fixed-size plot. Work from inner to outer.
        changed = False
        for panel in self._height_panels:
            layout = panel.layout()
            height = layout.totalHeightForWidth(panel.width())
            if height >= 0 and panel.minimumHeight() != height:
                panel.setMinimumHeight(height)
                changed = True
        if changed:
            self.content.layout().invalidate()
            self.content.layout().activate()

    def state(self):
        return {key: list(size) for key, size in self._sizes.items()}

    @classmethod
    def _normalise(cls, state):
        values = state if isinstance(state, dict) else {}
        result = {}
        for key, default in cls.DEFAULTS.items():
            size = values.get(key)
            if (not isinstance(size, (list, tuple)) or len(size) != 2
                    or any(type(value) is not int or not lower <= value <= cls.MAXIMUM
                           for value, lower in zip(size, cls.MINIMUMS[key]))):
                size = default
            result[key] = tuple(size)
        return result

    def set_state(self, state, *, emit=True):
        sizes = self._normalise(state)
        if sizes == self._sizes:
            return
        self._sizes = sizes
        owner = self.owner
        for key, widget in (("beam", owner.source_plot.plot), ("beam", owner.plot),
                            ("legend", owner.angle_colour_wheel)):
            widget.setFixedSize(*sizes[key])
        # Text and controls wrap separately; their contents cannot resize the
        # pictures. A large legend is also reachable inside the local scroll area.
        panel_width = max(340, sizes["beam"][0] + 8, sizes["legend"][0] + 8)
        owner.initial_beam_panel.setFixedWidth(panel_width)
        owner.section_beam_panel.setFixedWidth(panel_width)
        owner.source_plot.layout().setAlignment(owner.source_plot.plot, Qt.AlignmentFlag.AlignLeft)
        owner.section_beam_panel.layout().setAlignment(owner.plot, Qt.AlignmentFlag.AlignLeft)
        self.content.layout().invalidate()
        self.content.layout().activate()
        self._height_timer.start(0)
        if emit:
            owner.plot_sizes_changed.emit()

    def open_dialog(self):
        if self.dialog is not None:
            self.dialog.raise_()
            self.dialog.activateWindow()
            return
        dialog = QDialog(self.owner)
        self.dialog = dialog
        dialog.setObjectName("transversePlotSizesDialog")
        dialog.setWindowTitle("Beam plot sizes")
        dialog.setMinimumWidth(430)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        layout = QVBoxLayout(dialog)
        note = QLabel("Both beam plots share one fixed width and height, including axes, in display pixels.\n"
                      "Scroll to see larger plots. Sizes are saved with the workspace layout.")
        note.setWordWrap(True)
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse
                                     | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        note.setToolTip("Display pixels are Qt logical pixels and follow the monitor's display scaling. "
                        "Plot size does not change physical coordinates or calculate new trajectories.")
        layout.addWidget(note)
        grid = QGridLayout()
        grid.addWidget(QLabel("Picture"), 0, 0)
        grid.addWidget(QLabel("Width (px)"), 0, 1)
        grid.addWidget(QLabel("Height (px)"), 0, 2)
        self.editors = {}
        for row, (key, size) in enumerate(self._sizes.items(), 1):
            grid.addWidget(QLabel(self.LABELS[key]), row, 0)
            editors = []
            for column, (axis, value, lower) in enumerate(zip(("Width", "Height"), size, self.MINIMUMS[key]), 1):
                editor = WheelSafeSpinBox()
                editor.setObjectName(f"transverse{key.title()}{axis}")
                editor.setRange(lower, self.MAXIMUM)
                editor.setValue(value)
                grid.addWidget(editor, row, column)
                editors.append(editor)
            self.editors[key] = editors
        layout.addLayout(grid)
        for label in dialog.findChildren(QLabel):
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse
                                         | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Apply
                                   | QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.RestoreDefaults)
        layout.addWidget(buttons)

        def apply():
            self.set_state({key: [editor.value() for editor in editors] for key, editors in self.editors.items()})

        def reset():
            for key, editors in self.editors.items():
                for editor, value in zip(editors, self.DEFAULTS[key]):
                    editor.setValue(value)

        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(apply)
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(reset)
        buttons.accepted.connect(apply)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        dialog.finished.connect(self._dialog_finished)
        dialog.open()

    def _dialog_finished(self, _result):
        self.dialog = None
        self.editors = {}
