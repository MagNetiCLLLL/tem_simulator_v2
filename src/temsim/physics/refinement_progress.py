"""Optional read-only round diagnostics, separate from source acceptance."""
from temsim.immutable_json import freeze_json


def emit_refinement_record(progress_callback, method, row):
    """A progress consumer may persist completed checks while physics runs.

    The observer receives immutable diagnostics, never a mutable wave, matrix
    or source. Missing observers are a no-op. Observer failures abort normally;
    they cannot mark a failed or interrupted calculation as accepted.
    """
    observer = getattr(progress_callback, "record_refinement", None)
    if observer is None:
        return
    if not callable(observer):
        raise TypeError("Refinement diagnostic observer must be callable")
    observer(method, freeze_json(row))
