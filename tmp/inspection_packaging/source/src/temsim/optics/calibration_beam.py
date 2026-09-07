"""Virtual transmitted illumination for optical solves on calculation state."""

from contextlib import contextmanager


@contextmanager
def transmitted_calibration_beam(state):
    """Temporarily open both blankers without changing alignment settings.

    A condenser setting describes the transmitted beam, independently of the
    current exposure gate.  GUI solves use a state snapshot; the real displayed
    illumination is still calculated with its original blanking flags.  Only
    those flags change here: gun tilt/shift values, NanoPulser voltage, hardware
    installation and physical stops retain their configured values.
    """
    saved = []
    if hasattr(state, "beam_blanked"):
        saved.append((state, "beam_blanked", state.beam_blanked))
    nanopulser = getattr(state, "nanopulser", None)
    if nanopulser is not None:
        saved.append((nanopulser, "blanked", nanopulser.blanked))
    try:
        for obj, field, _ in saved:
            setattr(obj, field, False)
        yield
    finally:
        for obj, field, blanked in reversed(saved):
            setattr(obj, field, blanked)
