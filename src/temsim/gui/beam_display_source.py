"""Choose completed display trajectories without mutating physics results."""

from collections.abc import Mapping

from temsim.specimen.downstream_transport import validated_geometric_specimen_exit
from temsim.specimen.sample_region import validated_sample_region_exit


def downstream_display_branches(result):
    """Prefer a provenance-checked detailed exit, including an empty exit.

    A valid empty checkpoint means no forward electrons; falling back to an
    optical reference in that case would resurrect absorbed/backscattered rays.
    Legacy caches without a verified exit remain explicitly labelled references.
    """
    signatures = getattr(result, "signatures", {}) or {}
    if not isinstance(signatures, Mapping):
        signatures = {}
    signature = str(signatures.get("sample_downstream", ""))
    checkpoint = validated_geometric_specimen_exit(
        getattr(result, "specimen_exit", None), signature,
    )
    region = getattr(result, "sample_region", None)
    if checkpoint is None and region is not None and signature:
        try:
            checkpoint = validated_sample_region_exit(region, signature)
        except (AttributeError, TypeError, ValueError):
            # An incomplete legacy region has no trustworthy exit population.
            checkpoint = None
    if checkpoint is not None:
        return tuple(checkpoint.branches), "Specimen exit"
    simulation = getattr(result, "simulation", None)
    branches = tuple(getattr(simulation, "branches", {}).values())
    return branches, "Optical reference"
