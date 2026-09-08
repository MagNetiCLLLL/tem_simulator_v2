"""Offscreen layout QA ONLY: synthetic frame/proposal, never a new simulation.

Reuses the dedicated GUI test fixture and copies the catalog into a temporary
directory. No Save button is invoked. The user's running app is not accessed.
"""
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import runpy

from pytest import MonkeyPatch
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest

from temsim.app import create_application
from temsim.gui.scan_panel import ScanControlView

root = Path(__file__).resolve().parents[2]
output = Path(__file__).resolve().parent
fixture_module = runpy.run_path(str(root / "tests/test_df_geometry_gui.py"))
app = create_application([])
for name in ("segoeui.ttf", "segoeuib.ttf", "segoeuii.ttf"):
    QFontDatabase.addApplicationFont(str(Path("C:/Windows/Fonts") / name))
app.setFont(QFont("Segoe UI", 9))

class WidgetCollector:
    def __init__(self):
        self.widgets = []
    def addWidget(self, widget):
        self.widgets.append(widget)

with TemporaryDirectory(prefix="df_layout_fixture_", dir=root / "tmp") as temporary:
    collector = WidgetCollector()
    with MonkeyPatch.context() as patches:
        generator = fixture_module["window"].__wrapped__(collector, Path(temporary), patches)
        window, frame, catalog_root = next(generator)
        panel = ScanControlView()
        panel.set_state(window.state)
        panel._set_stem_frame(frame, state_snapshot=window.state)
        panel.result_tabs.setCurrentIndex(1)
        panel.resize(1280, 760)
        panel.show()
        app.processEvents()
        QTest.qWait(150)
        action_path = output / "ui_action_layout_fixture.png"
        assert panel.grab().save(str(action_path))
        dialog = window._review_df_geometry(frame)
        assert dialog is not None, window.errors_for_test
        app.processEvents()
        QTest.qWait(150)
        review_path = output / "ui_review_layout_fixture.png"
        assert dialog.grab().save(str(review_path))
        details = {
            "purpose": "Layout-only synthetic GUI fixture, not computed microscope physics",
            "style": "Actual Fusion + APPLICATION_STYLE + Segoe UI 9",
            "catalog": "Temporary copy; no Save invoked; original catalog byte invariance checked by fixture",
            "frame": "Uniform synthetic 2x2 fractions and synthetic 1-7 mrad DF report",
            "proposal": "Synthetic ID 8 mm / OD 16 mm, Jdiff singular-axis range 0.8-1.2 m",
            "panel_size": [panel.width(), panel.height()],
            "dialog_size": [dialog.width(), dialog.height()],
            "outputs": [str(action_path), str(review_path)],
        }
        (output / "ui_layout_fixture.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
        print(json.dumps(details))
        dialog.reject()
        panel.close()
        try:
            next(generator)
        except StopIteration:
            pass
        window.close()
        app.processEvents()
