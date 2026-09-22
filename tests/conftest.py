import os
from pathlib import Path
import sys

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item):
    """Finish Qt's deferred widget deletion within the owning test.

    pytest-qt closes registered widgets with deleteLater, but processEvents
    alone does not flush DeferredDelete outside a running event loop. Without
    this, many hidden workspaces survive until an unrelated test calls waitUntil,
    which then spends its timeout destroying earlier tests' plots and editors.
    This hook runs inside pytest-qt's teardown exception-capture wrapper.
    """
    widgets = sys.modules.get("PySide6.QtWidgets")
    if widgets is not None and widgets.QApplication.instance() is not None:
        from PySide6.QtCore import QCoreApplication, QEvent

        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


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
