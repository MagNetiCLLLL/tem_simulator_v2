"""Separate optical/control coordinates from independently modelled hardware.

The probe service-channel names establish controls, not separate mechanical
cartridges. Keep their calibrated coordinates in the optical model; consumers
of material geometry must not turn their legacy envelopes into solids.
"""

from temsim.component_keys import (
    PROBE_DP11_DEFLECTOR, PROBE_DP21_DEFLECTOR, PROBE_DP22_DEFLECTOR,
    PROBE_DPH1_DEFLECTOR, PROBE_DPH2_DEFLECTOR,
    PROBE_HPC_HEXAPOLE, PROBE_HPOL_HEXAPOLE,
    PROBE_QPC_QUADRUPOLE, PROBE_QPH1_QUADRUPOLE,
    PROBE_QPH2_QUADRUPOLE, PROBE_QPOL_QUADRUPOLE,
)


PROBE_CONTROL_CHANNEL_KEYS = frozenset({
    PROBE_DP11_DEFLECTOR, PROBE_DP21_DEFLECTOR, PROBE_DP22_DEFLECTOR,
    PROBE_DPH1_DEFLECTOR, PROBE_DPH2_DEFLECTOR,
    PROBE_HPC_HEXAPOLE, PROBE_HPOL_HEXAPOLE,
    PROBE_QPC_QUADRUPOLE, PROBE_QPH1_QUADRUPOLE,
    PROBE_QPH2_QUADRUPOLE, PROBE_QPOL_QUADRUPOLE,
})


def non_material_role(part_data, *, profile=None) -> str:
    """Return the non-material role, including compatibility for older files.

An empty result leaves existing physical/reference-surface handling intact.
In particular, a detector plane is not virtual merely because it is thin.
"""
    key = str(part_data.get("key", ""))
    role = str(part_data.get("layout_role", ""))
    if key == "probe_dp12_scan_deflector":
        return "virtual_reference"
    if key in PROBE_CONTROL_CHANNEL_KEYS:
        return "control_channel"
    if role in {"control_channel", "virtual_reference"}:
        return role
    shape = profile if profile is not None else part_data.get("mechanical_profile", "")
    if (shape in {"virtual_plane", "virtual_layout"}
            or part_data.get("virtual") or part_data.get("is_virtual")):
        return "virtual_reference"
    return ""


def representation_note(part_data, *, profile=None) -> str:
    role = non_material_role(part_data, profile=profile)
    if role == "control_channel":
        owner = str(part_data.get("layout_owner", "probe_corrector")).replace("_", " ")
        return (
            f"Control channel in {owner}. The marker shows its optical-model "
            "reference position. Its physical host and separate housing are "
            "unverified; no independent material body is defined."
        )
    if role == "virtual_reference":
        return "Virtual reference plane; no independent material body or thickness."
    return ""
