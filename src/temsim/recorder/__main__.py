"""Run the recorder without constructing or calculating a simulator workspace."""
from __future__ import annotations

import sys


def run():
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication
    from temsim.gui.input_policy import install_numeric_input_policy
    from temsim.gui.instrument_recorder import InstrumentRecorderWindow

    QCoreApplication.setOrganizationName("TEM Simulator")
    QCoreApplication.setApplicationName("Instrument Recorder")
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    install_numeric_input_policy(app)
    window = InstrumentRecorderWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
