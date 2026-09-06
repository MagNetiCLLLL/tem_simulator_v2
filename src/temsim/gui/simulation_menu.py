"""Compact fidelity menu; model changes are explicit, not accuracy presets."""

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QMenu

from temsim.simulation_modes import MODES, mode_key, linear_mode_issues, nonlinear_mode_issues


class SimulationMenu(QMenu):
    mode_requested = Signal(str)
    configure_requested = Signal()

    def __init__(self, parent=None):
        super().__init__("Simulation", parent)
        self.setObjectName("simulationMenu")
        self.setToolTipsVisible(True)
        self.setStyleSheet("""
            QMenu::item { padding: 5px 22px 5px 24px; }
            QMenu::item:disabled { color: #94a3b8; }
            QMenu::item:selected:enabled { background: #2563eb; color: #ffffff; }
        """)
        self.addSection("Column lens model")
        self.mode_group = QActionGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_actions = {}
        for mode in MODES:
            if mode.key == "custom":
                self.addSeparator()
            text = mode.label if mode.available else mode.label + " (not available)"
            action = QAction(text, self)
            action.setObjectName(f"simulationMode_{mode.key}")
            action.setCheckable(True)
            action.setData(mode.key)
            action.setEnabled(mode.available)
            action.setToolTip(mode.detail)
            action.triggered.connect(lambda checked, key=mode.key: self.mode_requested.emit(key) if checked else None)
            self.mode_group.addAction(action)
            self.addAction(action)
            self.mode_actions[mode.key] = action
        self.addSeparator()
        configure = self.addAction("Configure lens models...")
        configure.triggered.connect(self.configure_requested.emit)
        note = self.addAction("High accuracy changes numerical resolution only")
        note.setEnabled(False)

    def set_state(self, state):
        current = mode_key(state)
        for key, action in self.mode_actions.items():
            action.setChecked(key == current)
        # A retained linear shelf can be ready even when the current custom
        # configuration is different. This never runs the FEM solver.
        saved = getattr(state, "simulation_mode_profiles", {}).get("linear_geometry", {})
        descriptors = (state.lens_field_map_descriptors if current == "linear_geometry" else
                       saved.get("lens_field_map_descriptors", state.lens_field_map_descriptors))
        issues = linear_mode_issues(state, descriptors)
        action = self.mode_actions["linear_geometry"]
        action.setEnabled(not issues)
        action.setText("Linear Geometry Field (setup required)" if issues else "Linear Geometry Field")
        action.setToolTip(("Configure lens models first.\n" + "\n".join(issues[:5])) if issues else MODES[2].detail)
        saved = getattr(state, "simulation_mode_profiles", {}).get("nonlinear_material", {})
        descriptors = (state.lens_field_map_descriptors if current == "nonlinear_material" else
                       saved.get("lens_field_map_descriptors", state.lens_field_map_descriptors))
        issues = nonlinear_mode_issues(state, descriptors)
        action = self.mode_actions["nonlinear_material"]
        action.setEnabled(not issues)
        action.setText("Nonlinear Material Field (setup required)" if issues else "Nonlinear Material Field")
        action.setToolTip(("Configure B-H fields first.\n" + "\n".join(issues[:5])) if issues else MODES[3].detail)
