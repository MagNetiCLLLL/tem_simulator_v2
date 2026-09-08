"""Render the reported raw fractions in offscreen Qt, not a new CIF simulation."""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from pathlib import Path
import runpy
import numpy as np
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest

from temsim.app import create_application
from temsim.gui.scan_panel import ScanControlView
from temsim.optics.column import default_state

root = Path(__file__).resolve().parents[1]
make_frame = runpy.run_path(str(root / "tests/test_stem_image_presentation.py"))["_frame"]
app = create_application([])
for filename in ("segoeui.ttf", "segoeuib.ttf", "segoeuii.ttf"):
    QFontDatabase.addApplicationFont(str(Path("C:/Windows/Fonts") / filename))
app.setFont(QFont("Segoe UI", 9))
output = root / "outputs/eds_overlap_sampling"
output.mkdir(parents=True, exist_ok=True)
view = ScanControlView()
state = default_state()
state.sample.stem_wave_enabled = False  # Reconstruct the reported geometry-only setting.
view.set_state(state)
view.resize(1280, 780)
view.result_tabs.setCurrentIndex(1)
view.show()
for pitch_nm, filename in ((.02, "scan_preview_002nm.png"), (.05, "scan_preview_005nm.png")):
    for deflector in (state.ac_deflector, state.descan_deflector):
        deflector.scan_pixel_size_nm = pitch_nm
        deflector.scan_enabled = True
    view.set_state(state)
    frame = make_frame(pitch_nm=pitch_nm, material=True)
    row, column = np.indices((32, 32))
    frame.fractions["bf"][:] = (3531. + (row >= column)) / 15000.
    if pitch_nm == .05:
        frame.fractions["df"][:] = (2423. + (column < 5)) / 15000.
    view.display_result(None, frame, complete=True, state_snapshot=state)
    view.detector_playback_summary.setText(
        "Display verification: reconstructed ray-count fractions; not a new CIF acquisition"
    )
    app.processEvents()
    QTest.qWait(200)
    assert view.grab().save(str(output / filename))
    print(output / filename)
view.close()
