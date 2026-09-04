import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from types import SimpleNamespace
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QFontDatabase
from temsim.app import APPLICATION_STYLE
from temsim.gui.aberration_view import AberrationComparisonView
from temsim.optics.column import default_state

app = QApplication([])
font_id = QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
app.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0], 10))
app.setStyleSheet(APPLICATION_STYLE)
state = default_state()
triangle = np.exp(2j * np.pi * np.arange(3) / 3) * 1e-9
branch = SimpleNamespace(
    x=triangle.real[None, :], y=triangle.imag[None, :],
    tx=np.zeros((1, 3)), ty=np.zeros((1, 3)),
    alive=np.ones(3, dtype=bool), ray_weight=np.ones(3),
)
view = AberrationComparisonView(fixed_system='probe')
view.resize(1000, 650)
view.display_result(SimpleNamespace(state_snapshot=state, simulation=SimpleNamespace(incident=branch)))
view.show()
app.processEvents()
view.grab().save('tmp/probe_diagnostic_ui_smoke.png')
print(ascii(view.summary.text()))
