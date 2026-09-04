import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from temsim.gui.parameter_panel import ParameterPanel
from temsim.optics.column import default_state
from temsim.runtime_parameters import runtime_targets
app = QApplication([])
panel = ParameterPanel()
state = default_state()
targets = runtime_targets(state)
target = targets['condenser_lens_2']
panel.set_context('C2', target, None, (), None)
panel.resize(550, 700)
panel.show()
panel.activateWindow()
app.processEvents()
QTest.qWaitForWindowExposed(panel)
spin = panel.lens_excitation
spin.setFocus()
app.processEvents()
print('visible', spin.isVisible(), 'position', spin.mapTo(panel, spin.rect().center()), 'at', app.widgetAt(spin.mapToGlobal(spin.rect().center())))
before = spin.value()
QTest.wheelEvent(panel.windowHandle(), spin.mapTo(panel, spin.rect().center()), QPoint(0, -120))
app.processEvents()
print({'before': before, 'after': spin.value(), 'filter_installed': hasattr(app, '_numeric_wheel_filter'), 'scroll': panel.scroll_area.verticalScrollBar().value()})
