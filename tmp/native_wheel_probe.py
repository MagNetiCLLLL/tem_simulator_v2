import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QPoint, QObject, QEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QVBoxLayout, QWidget
class Watch(QObject):
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel:
            print('wheel', type(obj).__name__, event.isAccepted())
        return False
app=QApplication([])
watch=Watch(app); app.installEventFilter(watch)
panel=QWidget(); layout=QVBoxLayout(panel); spin=QDoubleSpinBox(); spin.setValue(35); layout.addWidget(spin)
panel.resize(500,300); panel.show(); panel.activateWindow(); app.processEvents(); QTest.qWaitForWindowExposed(panel)
spin.setFocus(); app.processEvents()
print('focus',app.focusWidget(), 'rect',spin.rect(), 'position',spin.mapTo(panel, spin.rect().center()),'at',app.widgetAt(spin.mapToGlobal(spin.rect().center())))
QTest.wheelEvent(panel.windowHandle(), spin.mapTo(panel, spin.rect().center()), QPoint(0,-120))
app.processEvents()
print('value',spin.value())
