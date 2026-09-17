"""Evidence identity checks; a saved PASS label is never qualification."""
from temsim.immutable_json import json_digest


def assess_evidence(checkpoint, report, *, implementation):
    if report.get("schema") != "incident-convergence-evidence-v1":
        return "HISTORICAL", "Unrecognized evidence schema; retained for inspection"
    if report.get("digest") != json_digest({k: v for k, v in report.items() if k != "digest"}):
        raise ValueError("Evidence checksum mismatch")
    if (report.get("checkpoint_id") != checkpoint.digest or report.get("snapshot_id") != checkpoint.snapshot.digest
            or report.get("implementation") != checkpoint.snapshot.implementation
            or report.get("implementation") != implementation):
        return "HISTORICAL", "Evidence input or implementation identities do not match"
    from temsim.working_point_archive import WorkingPointArchiveIndex
    if isinstance(checkpoint, WorkingPointArchiveIndex):
        return "UNVERIFIED_PRODUCTS", "Load retained data before admitting numerical evidence"
    if report.get("comparison", {}).get("status") not in {"UNRESOLVED", "STABLE_FOR_CHECKED_AXIS"}:
        raise ValueError("Unsupported convergence verdict")
    return "MATCHING_NUMERICAL_COMPARISON", "The named numerical comparison only; full numerical and physical qualification not established"
