"""Collect Python widget cycles on the Qt thread, including embedded panels."""
import gc
from collections import deque
from time import monotonic
from PySide6.QtCore import QObject, QTimer, Slot


class GuiGarbageCollector(QObject):
    def __init__(self, application):
        super().__init__(application)
        self._restore_enabled = gc.isenabled()
        self.history = deque(maxlen=32)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.collect_if_due)
        if self._restore_enabled:
            # Reference counting still releases ordinary arrays immediately.
            # Only cyclic collection moves to the GUI thread: a worker must
            # never finalize an unreachable Qt widget with GUI affinity.
            gc.disable()
            self.timer.start()
        application.aboutToQuit.connect(self.stop)

    @Slot()
    def collect_if_due(self):
        counts, thresholds = gc.get_count(), gc.get_threshold()
        due = [index for index in range(3) if thresholds[index] and counts[index] >= thresholds[index]]
        if due:
            generation = max(due)
            started = monotonic()
            collected = gc.collect(generation)
            self.history.append(dict(generation=generation, start_s=started,
                                     end_s=monotonic(), collected=collected))

    @Slot()
    def stop(self):
        self.timer.stop()
        if self._restore_enabled:
            gc.enable()
            self._restore_enabled = False


def install_gui_gc(application):
    if getattr(application, "_temsim_gui_gc", None) is None:
        application._temsim_gui_gc = GuiGarbageCollector(application)
