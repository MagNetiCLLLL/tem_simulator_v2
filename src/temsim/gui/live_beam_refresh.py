"""Coalesce Live tuning edits before refreshing an already calculated wave.

This schedules the existing Electron beam action, never another solver or
source definition. Particle frames keep their own scheduler. A wave refresh
needs both a quiet edit interval and a successfully published latest ray frame.
"""
from PySide6.QtCore import QObject, QTimer


class LiveBeamRefresh(QObject):
    SETTLE_MS = 300

    def __init__(self, page, parent=None, *, allowed=lambda: True):
        super().__init__(parent)
        self.page = page
        self.allowed = allowed
        self._token = None
        self._settled = False
        self._rays_ready = False
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._settle)
        page.cancel_button.clicked.connect(self.cancel)
        page.calculate_button.clicked.connect(self.cancel)

    def _identity(self):
        page = self.page
        return (page._generation, page.state_set.generation,
                page.state_list.display_mode())

    @property
    def pending(self):
        return self._token is not None

    def invalidate(self, mark_stale):
        """Invalidate normal displays, retaining only an eligible Live refresh."""
        page = self.page
        continuing = self.pending and self._token == self._identity()
        if page.state_set.viewing_states():
            calculated = page.state_set.active and page.result is not None
        else:
            calculated = (not page._stale and page.result is not None
                          and (page._session_ready or page._worker is not None))
        eligible = continuing or calculated
        # set_state cancels old wave work and preserves the previous image.
        # Draft edits, explicit Cancel, or other input changes alter identity.
        self.cancel()
        mark_stale()
        if eligible and not page._closed:
            self._token = self._identity()
            self.timer.start(self.SETTLE_MS)
            page.cancel_button.setEnabled(True)
            page.status.setText(
                "Live tuning: previous coherent image retained; waiting for "
                "edits to settle and the latest particle frame before updating.")

    def particle_ready(self):
        if self.pending:
            self._rays_ready = True
            # Run outside result publication, after its remaining slots finish.
            if self._settled:
                self.timer.start(0)

    def _settle(self):
        self._settled = True
        if not self.pending:
            return
        if self.page._closed or self._token != self._identity():
            self.cancel()
            return
        if not self._rays_ready:
            return
        if not self.allowed():
            self.cancel()
            return
        self.cancel()
        # Uses current Z, selected saved states, numerical budgets and the
        # single global backend. Source admission/draft checks remain intact.
        self.page.calculate()

    def cancel(self, *_args):
        self.timer.stop()
        self._token = None
        self._settled = self._rays_ready = False
        self.page._update_cancel_enabled()
