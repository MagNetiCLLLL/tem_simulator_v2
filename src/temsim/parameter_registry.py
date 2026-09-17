"""Small adapter-backed parameter registry; existing validators remain owners.

Reuse explanations can broaden existing invalidation, never narrow it. Unknown
graph changes invalidate the plan conservatively. No field or particle solve is
performed while classifying parameters or comparing dependency signatures.
"""
from dataclasses import dataclass
import re

from temsim.component_keys import APERTURE_KEYS
from temsim.parameter_semantics import parameter_unit


# Existing dynamic aliases/readbacks already have scoped cache owners. Anything
# else added to a public model object is conservatively an unclassified input.
_KNOWN_DYNAMIC = {
    "State": {"corrector_elements", "recording_planes", "energy_filter_installed",
              "image_corrector_installed", "probe_corrector_installed", "show_field_diagram",
              "objective_back_focal_plane_z_mm", "objective_image_plane_z_mm", "simulation_time_s"},
    "EnergyFilterSystem": {"dynamic_focus_quadrupole_bore_mm", "dynamic_focus_quadrupole_geometry_source",
        "dynamic_focus_quadrupole_geometry_status", "dynamic_focus_quadrupole_length_mm",
        "dynamic_focus_quadrupole_model_status", "dynamic_focus_quadrupole_outer_mm",
        "output_plane_geometry_source", "output_plane_geometry_status"},
    "Aperture": {"maximum_radius_mm"},
}
for _lens_type in ("CondenserLensState", "AdapterLensComponent", "Tl22LensComponent",
        "Tl21LensComponent", "Tl12LensComponent", "MiniCondenserComponent",
        "ObjectiveLensComponent", "DiffractionLensComponent", "IntermediateLensComponent",
        "ProjectorLensP1Component", "ProjectorLensP2Component"):
    _KNOWN_DYNAMIC[_lens_type] = {"field_polarity_source", "field_polarity_status",
                                "field_calibration_source", "field_calibration_status"}


def unmapped_public_inputs(state):
    """Exact extension identity used by real caches, not just UI explanations."""
    supplied = getattr(state, "_captured_unmapped_inputs", None)
    if supplied is not None:
        return supplied
    from collections.abc import Mapping
    from dataclasses import fields, is_dataclass
    from temsim.instrument_snapshot import _RUNTIME_NAMES, encode_instrument
    from temsim.immutable_json import json_digest
    visited, extensions = set(), {}
    assets = None

    def extension_identity(item):
        nonlocal assets
        if assets is None:
            from temsim.input_assets import INPUT_ASSETS
            assets = INPUT_ASSETS.capture()
        # Already-admitted immutable arrays contribute exact content references,
        # without undoing lightweight capture by creating a full inline hex copy.
        return json_digest(encode_instrument(item, asset_store=assets))

    def visit(value, path):
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, Mapping):
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
                visit(item, f"{path}[{key}]")
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
        elif is_dataclass(value) and not isinstance(value, type):
            declared = {field.name for field in fields(value)}
            attributes = {name: getattr(value, name) for name in declared}
            if hasattr(value, "__dict__"):
                attributes.update(vars(value))
            attributes = {name: item for name, item in attributes.items()
                if not name.startswith("_") and name not in _RUNTIME_NAMES}
            known = _KNOWN_DYNAMIC.get(type(value).__name__, set())
            for name, item in sorted(attributes.items()):
                if name not in declared and name not in known:
                    extensions[f"{path}.{name}"] = extension_identity(item)
                visit(item, f"{path}.{name}")
    try:
        visit(state, "instrument")
    finally:
        if assets is not None:
            assets.close()
    return extensions


@dataclass(frozen=True)
class ParameterDefinition:
    component: str
    name: str
    category: str
    unit: str
    label: str
    description: str
    sweep_eligible: bool = False
    adapter: str = "existing runtime validator"

    @property
    def identity(self):
        return f"{self.component}.{self.name}"

    def tooltip(self, *, enabled=True, surface_source=False):
        status = "Active in current model"
        if self.category == "presentation":
            status = "Display geometry only"
        elif self.category == "inverse_target":
            status = "Derived readout; explicit Direct Alignment required"
        elif not enabled or (surface_source and self.component == "feg_tip" and self.name != "ray_count"):
            status = "Stored but inactive"
        return (f"{self.label} | {self.category} | {self.unit or 'dimensionless'}\n{status}\n"
                f"{self.description}\nSweep: {'eligible through existing validator' if self.sweep_eligible else 'not in the validated sweep pilot'}")


def parameter_definition(component, name, *, lens=False):
    category, label, detail, sweep = None, name.replace("_", " "), "", False
    if component == "simulation":
        if name == "virtual_observation_z_mm":
            category, label, detail = "presentation", "Observation plane", "Read-only virtual observation; does not move a physical optical element."
        elif name in {"step_mm", "history_step_mm", "acceleration_backend"}:
            category, detail = "execution", "Numerical resolution/backend evidence remains part of the consumed stage identities."
    elif component == "feg_tip" and name in {"emission_current_na", "virtual_source_fwhm_nm", "angular_rms_mrad", "angular_cutoff_mrad", "energy_spread_fwhm_ev", "ray_count"}:
        category = "execution" if name == "ray_count" else "operating"
        label, detail = "Tip " + label, "Emission at the physical tip; extraction, acceleration, apertures and column transport remain required."
        sweep = name != "ray_count"
    elif component == "feg_accelerator" and name == "high_tension_kv":
        category, label, detail, sweep = "operating", "Gun accelerating voltage", "Changes the connected gun field and every dependent electron transport stage.", True
    elif lens:
        if name in {"percent", "polarity", "enabled", "cs_mm", "cc_mm"}:
            category, label, detail = "operating", "Lens " + label, "Changes the existing lens control. Dependencies follow complete field support, including overlap and shared circuits."
            sweep = name == "percent"
        elif name in {"z_mm", "a_mm", "b0_t", "gaussian"}:
            category, detail = "structural", "Geometry/calibration changes require the existing structural editor; component centres alone do not bound a field."
        elif name == "colour":
            category, detail = "presentation", "Display colour only; existing stage signatures still conservatively own reuse decisions."
    elif component in APERTURE_KEYS and name in {"radius_mm", "diameter_mm", "offset_x_mm", "offset_y_mm", "enabled"}:
        category, label, detail = "operating", "Aperture " + label, "Physical transmission/masking, independent of display and detector readout selections."
    elif component == "sample" and name == "thickness_nm":
        category, label, detail, sweep = "operating", "Specimen thickness", "Specimen interactions and surface positions remain physical dependencies.", True
    elif component == "inverse" and name in {"alpha95", "diameter95"}:
        category, detail = "inverse_target", "A requested measured quantity; cannot be assigned as a downstream source or raw lens parameter."
    if category is None:
        return None
    return ParameterDefinition(str(component), str(name), category, parameter_unit((name,)), label, detail, sweep)


def runtime_definition(target, name):
    return parameter_definition(target.key, name, lens=hasattr(target.obj, "percent") and hasattr(target.obj, "b0_t"))


def registered_sweep_eligibility(parsed):
    """Return a pilot decision or None to retain the established wider allowlist."""
    if len(parsed) == 2 and parsed[0][0] == "lenses" and parsed[0][1] is not None:
        definition = parameter_definition(parsed[0][1], parsed[1][0], lens=True)
    elif parsed == (("sample", None), ("thickness_nm", None)):
        definition = parameter_definition("sample", "thickness_nm")
    else:
        return None
    return None if definition is None else definition.sweep_eligible


def dependency_plan(before, after):
    """Read-only plan backed by existing signatures and the full captured diff."""
    from temsim.calculation_cache import calculation_signatures, explain_product_reuse, incident_field_dependencies
    from temsim.immutable_json import freeze_json
    from temsim.instrument_snapshot import decode_instrument
    from temsim.working_point import snapshot_changes
    left, right = decode_instrument(before.graph), decode_instrument(after.graph)
    old, new = calculation_signatures(left), calculation_signatures(right)
    legacy = explain_product_reuse(old, new)
    changes, unknown = [], []
    nodes = after.graph["nodes"]
    root_id = after.graph["root"]["ref"]
    lens_refs = {v["ref"] for v in nodes[root_id]["attributes"]["lenses"]["list"]}
    for path, old_value, new_value in snapshot_changes(before, after):
        match = re.fullmatch(r"/graph/nodes/(\d+)/attributes/([^/]+)", path)
        definition = None
        if match:
            node_id, name = int(match[1]), match[2]
            if node_id < len(nodes):
                key = "simulation" if node_id == root_id else nodes[node_id].get("attributes", {}).get("key")
                definition = parameter_definition(key, name, lens=node_id in lens_refs)
        if definition is None:
            unknown.append(path)
        changes.append(dict(path=path, before=old_value, after=new_value,
            parameter_id=None if definition is None else definition.identity,
            category="unmapped" if definition is None else definition.category,
            reason="Unmapped input; conservative invalidation" if definition is None else definition.label+" changed"))
    support = incident_field_dependencies(right)
    reasons = tuple(dict.fromkeys(row["reason"] for row in changes))
    stages = {}
    for key, row in legacy.items():
        reusable = row["reusable"] and not unknown
        stages[key] = dict(reusable=reusable, legacy_reusable=row["reusable"], previous_signature=old.get(key),
            signature=new[key], reasons=("Exact existing stage signature; captured changes are mapped",) if reusable else
            (reasons or (row["reason"],)))
    return freeze_json(dict(schema="parameter-dependency-plan-v1", before=before.digest, after=after.digest,
        changes=changes, stages=stages, unknown_paths=unknown, incident_field_support=support,
        policy="Read-only conservative adapter; no existing invalidation is narrowed; no solver is executed"))
