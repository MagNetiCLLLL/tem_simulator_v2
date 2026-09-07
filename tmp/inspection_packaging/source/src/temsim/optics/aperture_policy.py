"""Non-retractable stops keep their opening active when hardware is installed."""

from temsim.component_keys import FIXED_APERTURE_KEYS


class ApertureInsertionPolicy:
    """Keep legacy enabled fields serializable without permitting retraction.

    ``installed`` remains a separate topology gate: an absent optional branch
    must never clip the beam. Old snapshots with enabled=False are normalized.
    """

    @property
    def always_inserted(self):
        return getattr(self, "key", None) in FIXED_APERTURE_KEYS

    def __setattr__(self, name, value):
        if name == "enabled" and self.always_inserted:
            value = True
        super().__setattr__(name, value)
