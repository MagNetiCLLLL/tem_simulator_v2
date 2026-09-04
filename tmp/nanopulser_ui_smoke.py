import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from dataclasses import replace
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QFontDatabase
from temsim.app import APPLICATION_STYLE
from temsim.gui.main_window import MainWindow
from temsim.simulation_pipeline import calculate

app = QApplication.instance() or QApplication([])
QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
app.setFont(QFont("Segoe UI", 9))
app.setStyleSheet(APPLICATION_STYLE)
window = MainWindow()
window.preview_timer.stop()
selection = replace(window.selection, beam_blanker="NanoPulser")
window.assembly = window.catalog.apply(window.state, selection)
window.selection = selection
window.state.nanopulser.blanked = True
window.state.electron_gun.emitter.ray_count = 49
window.state.step_mm = 0.1
window.state.history_step_mm = 2.0
window.state.acceleration_backend = "CPU"
window.assembly_panel.set_selection(selection)
window._refresh_assembly_views()
result = calculate(window.state)
window.workspace.display_result(result, "High accuracy")
window.workspace.jump_to_ray_position(window.state.sample.z_mm)
window._select_component_from_workspace("nanopulser_deflector")
window.resize(1600, 1000)
window.show()
window.preview_timer.stop()
app.processEvents()
path = Path("tmp/nanopulser_ui_smoke.png").resolve()
assert window.grab().save(str(path))
assert result.simulation.metrics["sample_surviving_current_pa"] == 0.0
assert "No incident current" in window.workspace.wave_imaging.summary.text()
assert window.blank_beam_button.isChecked()
print(path)
print("Full blanked pipeline and GUI passed")
window.close()
