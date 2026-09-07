"""Modeless, presentation-only cache settings and on-demand statistics."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

from PySide6.QtCore import QByteArray, Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
)

from temsim.cache_preferences import (
    GIB, MIB, SETTINGS_ROOT, CachePreferences, default_cache_preferences,
    detect_total_memory_bytes, load_cache_preferences, save_cache_preferences,
    validate_cache_preferences,
)


class CacheSettingsDialog(QDialog):
    """Apply emits validated CachePreferences; it never starts or clears calculations.

    statistics_provider returns {"calculation": controller.cache_statistics(),
    "ray_display": workspace.ray_display_cache_info()}. It is called only when
    the dialog opens, the user refreshes statistics, or settings are applied.
    """

    preferencesChanged = Signal(object)

    def __init__(self, settings, statistics_provider: Callable[[], Mapping] | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Performance and cache")
        self.setObjectName("performanceCacheDialog")
        self.setModal(False)
        self.settings = settings
        self.statistics_provider = statistics_provider
        self.total_memory_bytes = detect_total_memory_bytes()
        self.preferences = load_cache_preferences(settings, self.total_memory_bytes)
        self.resize(580, 490)
        layout = QVBoxLayout(self)
        layout.addWidget(self._label("Cache limits retain reusable results. They do not preallocate memory."))
        group = QGroupBox("Retention limits")
        form = QFormLayout(group)
        self.high_cache = self._spin(" GiB", 0.001, 256.0, 3)
        self.tuning_cache = self._spin(" GiB", 0.001, 256.0, 3)
        self.ray_display_cache = self._spin(" MiB", 1.0, 256.0 * 1024, 1)
        self.prepared_specimen_cache = self._spin(" MiB", 1.0, 256.0 * 1024, 1)
        self.sample_display_cache = self._spin(" MiB", 1.0, 256.0 * 1024, 1)
        self.disk_cache = self._spin(" GiB", 0.001, 1024.0, 3)
        form.addRow("High-accuracy results", self.high_cache)
        form.addRow("Live-tuning results", self.tuning_cache)
        form.addRow("Ray display", self.ray_display_cache)
        form.addRow("Specimen potentials (RAM)", self.prepared_specimen_cache)
        form.addRow("Sample atom display (RAM)", self.sample_display_cache)
        form.addRow("Disk checkpoints", self.disk_cache)
        self.ram_summary = self._label("")
        form.addRow(self.ram_summary)
        layout.addWidget(group)
        note = self._label("Current plots and running workers use additional memory.\n"
                           "Disk cache stores restartable incident checkpoints, not full images.\n"
                           "Disk quota changes take effect on the next checkpoint write.")
        note.setToolTip("These are cache retention caps, not a limit on the whole application. "
                        "Physical settings and calculation results are not reset. "
                        "Memory entry limits also apply. A smaller disk quota is enforced on the "
                        "next checkpoint write. Disk usage is not scanned periodically.")
        layout.addWidget(note)
        stats_header = QHBoxLayout()
        stats_title = self._label("Retained cache data")
        stats_title.setWordWrap(False)
        stats_header.addWidget(stats_title)
        stats_header.addStretch()
        self.refresh_button = QPushButton("Refresh statistics")
        self.refresh_button.clicked.connect(self.refresh_statistics)
        stats_header.addWidget(self.refresh_button)
        layout.addLayout(stats_header)
        self.statistics = self._label("Not available")
        layout.addWidget(self.statistics)
        self.feedback = self._label("")
        layout.addWidget(self.feedback)
        layout.addStretch()
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Close
            | QDialogButtonBox.StandardButton.RestoreDefaults)
        self.buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply_preferences)
        self.buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(self.restore_defaults)
        self.buttons.rejected.connect(self.close)
        layout.addWidget(self.buttons)
        for control in (self.high_cache, self.tuning_cache, self.ray_display_cache,
                        self.prepared_specimen_cache, self.sample_display_cache, self.disk_cache):
            control.valueChanged.connect(self._update_summary)
        self._populate(self.preferences)
        geometry = settings.value(f"{SETTINGS_ROOT}/dialog_geometry")
        if isinstance(geometry, QByteArray):
            self.restoreGeometry(geometry)

    @staticmethod
    def _label(text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    @staticmethod
    def _spin(suffix: str, minimum: float, maximum: float, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(minimum, maximum)
        spin.setSuffix(suffix)
        spin.setSingleStep(0.25 if suffix == " GiB" else 64)
        return spin

    def _populate(self, prefs: CachePreferences) -> None:
        self.high_cache.setValue(prefs.high_cache_budget_bytes / GIB)
        self.tuning_cache.setValue(prefs.tuning_cache_budget_bytes / GIB)
        self.ray_display_cache.setValue(prefs.ray_display_cache_budget_bytes / MIB)
        self.prepared_specimen_cache.setValue(prefs.prepared_specimen_cache_budget_bytes / MIB)
        self.sample_display_cache.setValue(prefs.sample_display_cache_budget_bytes / MIB)
        self.disk_cache.setValue(prefs.disk_cache_budget_bytes / GIB)
        self._update_summary()

    def _draft(self) -> CachePreferences:
        return replace(
            self.preferences,
            high_cache_budget_bytes=round(self.high_cache.value() * GIB),
            tuning_cache_budget_bytes=round(self.tuning_cache.value() * GIB),
            ray_display_cache_budget_bytes=round(self.ray_display_cache.value() * MIB),
            prepared_specimen_cache_budget_bytes=round(self.prepared_specimen_cache.value() * MIB),
            sample_display_cache_budget_bytes=round(self.sample_display_cache.value() * MIB),
            disk_cache_budget_bytes=round(self.disk_cache.value() * GIB),
        )

    def _update_summary(self, *_args) -> None:
        used = self._draft().managed_ram_budget_bytes / GIB
        if self.total_memory_bytes is None:
            self.ram_summary.setText(f"Managed RAM limit: {used:.3g} GiB. Physical RAM unavailable.")
        else:
            total = self.total_memory_bytes / GIB
            self.ram_summary.setText(f"Managed RAM limit: {used:.3g} / {total / 2:.3g} GiB allowed ({total:.3g} GiB physical RAM).")

    def restore_defaults(self) -> None:
        self._populate(default_cache_preferences(self.total_memory_bytes))
        self.feedback.setText("Defaults selected. Apply to use them.")

    def set_preferences(self, prefs: CachePreferences) -> None:
        """Reflect externally applied preferences without persistence or signals."""
        validate_cache_preferences(prefs, self.total_memory_bytes)
        self.preferences = prefs
        self._populate(prefs)

    def apply_preferences(self) -> bool:
        try:
            prefs = self._draft()
            validate_cache_preferences(prefs, self.total_memory_bytes)
            save_cache_preferences(self.settings, prefs, self.total_memory_bytes)
        except (ValueError, OSError) as error:
            self.feedback.setText(str(error))
            return False
        self.preferences = prefs
        self.preferencesChanged.emit(prefs)
        self.feedback.setText("Cache limits applied. No calculation started.")
        self.refresh_statistics()
        return True

    def refresh_statistics(self) -> None:
        if self.statistics_provider is None:
            self.statistics.setText("Cache statistics unavailable.")
            return
        try:
            data = self.statistics_provider()
            calculation = data.get("calculation", {})
            display = data.get("ray_display", {})
            prepared = data.get("prepared_specimen", {})
            sample = data.get("sample_display", {})
            lines = []
            for title, values, prefix in (("High accuracy", calculation, "high_"),
                                           ("Live tuning", calculation, "tuning_"),
                                           ("Ray display", display, ""),
                                           ("Specimen potentials", prepared, ""),
                                           ("Sample atom display", sample, "")):
                retained = values.get(f"{prefix}used_bytes", 0) / MIB
                entries = values.get(f"{prefix}entries", 0)
                hits = values.get(f"{prefix}hits", 0)
                text = f"{title}: {retained:.1f} MiB | {entries} entries | {hits} hits"
                misses = values.get(f"{prefix}misses")
                if misses is not None:
                    requests = hits + misses
                    text += f" | {100.0 * hits / requests:.1f}% hit rate" if requests else " | not used"
                lines.append(text)
            if calculation.get("disk_enabled") is False:
                lines.append("Disk checkpoints: unavailable; memory caching remains active.")
            elif calculation.get("disk_enabled") is True:
                disk_limit = calculation.get("disk_budget_bytes", 0) / GIB
                lines.append(f"Disk checkpoints: enabled | {disk_limit:.3g} GiB retention limit")
            self.statistics.setText("\n".join(lines))
        except (TypeError, ValueError, AttributeError, KeyError, RuntimeError):
            self.statistics.setText("Cache statistics unavailable.")

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_statistics()

    def _save_geometry(self) -> None:
        self.settings.setValue(f"{SETTINGS_ROOT}/dialog_geometry", self.saveGeometry())
        self.settings.sync()

    def closeEvent(self, event):
        self._save_geometry()
        self._populate(self.preferences)
        super().closeEvent(event)

    def hideEvent(self, event):
        self._save_geometry()
        super().hideEvent(event)

    def reject(self) -> None:
        self._save_geometry()
        self._populate(self.preferences)
        super().reject()
