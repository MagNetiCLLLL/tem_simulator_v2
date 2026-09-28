"""Explicit calculation intent, separate from saved physical instrument inputs."""

from temsim.immutable_json import json_digest


WORKFLOW_LABELS = {
    "full": "Complete calculation",
    "rays": "Ray Diagram",
    "sample": "Sample interactions",
    "sample_region": "Detailed sample paths and X-rays",
    "eds": "EDS spectrum",
    "stem": "STEM detector readout",
    "imaging": "Illuminating Image",
    "energy_filter": "Energy Filter",
}


def validate_workflow(workflow):
    if not isinstance(workflow, str) or workflow not in WORKFLOW_LABELS:
        raise ValueError(f"Unknown calculation workflow: {workflow!r}")
    return workflow


def workflow_signatures(signatures, workflow):
    """Scope a raw physical signature set; callers must not pass scoped keys.

    Page workflows keep a specimen-free optical column reference. Its branches
    cannot be reused as the full pipeline's material-loss quadrature. The exact
    incident identity remains shared across every workflow.
    """
    validate_workflow(workflow)
    result = dict(signatures)
    if workflow != "full":
        result["column"] = json_digest({"column": result["column"],
            "transport_scope": "optical-reference-without-specimen-v1"})
        result["workflow"] = workflow
        result["request"] = json_digest({
            "request": result["request"], "workflow": workflow,
            "workflow_schema": "explicit-page-calculation-v1",
        })
    return result


def admit_workflow(state, workflow):
    """Apply wave admission only to requested readouts; never enable a source."""
    validate_workflow(workflow)
    from temsim.optics.electron_gun.source_policy import require_physical_gun_source
    from temsim.physics.source_admission import (
        admit_requested_wave_products, require_gun_wave_source,
    )
    require_physical_gun_source(state.electron_gun)
    if workflow == "full":
        admit_requested_wave_products(state)
    elif workflow == "imaging":
        # This page currently implements wave imaging. A classical ray count
        # is not a substitute for its missing coherent tip-origin state.
        require_gun_wave_source(state, product="Illuminating Image")
    elif workflow == "stem" and bool(getattr(state.sample, "stem_wave_enabled", False)):
        require_gun_wave_source(state, product="STEM wave imaging")
