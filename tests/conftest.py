import os
from pathlib import Path
import sys

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp_cls():
    """Use the native Windows UI font when Qt's offscreen font DB is empty.

    PySide6 no longer bundles fonts for this plugin. The fallback displays
    boxes and inflates text controls, so it cannot validate the native layout.
    Font files are read from Windows only; no system or user settings change.
    """
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QApplication

    class TestApplication(QApplication):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if sys.platform != "win32" or self.platformName() != "offscreen":
                return
            fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
            for filename in ("segoeui.ttf", "segoeuib.ttf"):
                path = fonts / filename
                if path.is_file():
                    QFontDatabase.addApplicationFont(str(path))
            if "Segoe UI" in QFontDatabase.families():
                self.setFont(QFont("Segoe UI", 9))

    return TestApplication
