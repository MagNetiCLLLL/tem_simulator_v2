"""Explicit illumination-control patches over complete captured instruments.

These are operating-input transactions, never transport or qualification. The
objective lens is deliberately excluded: it is shared with image formation.
"""
from dataclasses import dataclass
from uuid import uuid4

from temsim.component_keys import CONDENSER_LENS_KEYS
from temsim.immutable_json import json_digest
from temsim.instrument_snapshot import capture_instrument_snapshot, encode_instrument
from temsim.runtime_parameters import runtime_targets, validate_runtime_assignment
from temsim.working_point import snapshot_changes


# Stable existing runtime IDs and setters; no source, geometry, insertion or
# downstream controls are inferred from a component's position in the column.
ILLUMINATION_CONTROLS = {
    **{key: ("percent", "polarity") for key in (*CONDENSER_LENS_KEYS,
        "adapter_lens", "probe_tl22_lens", "probe_tl21_lens", "probe_tl12_lens", "mini_condenser")},
    **{key: ("radius_mm", "offset_x_mm", "offset_y_mm") for key in
        ("condenser_aperture_2", "condenser_aperture_3")},
    "condenser_deflector": ("upper_x_mrad", "upper_y_mrad", "lower_x_mrad", "lower_y_mrad"),
    "condenser_stigmator": ("strength_x_percent", "strength_y_percent"),
    "probe_hp1_hexapole": ("strength_m3", "orientation_rad"),
    "probe_hp2_hexapole": ("strength_m3", "orientation_rad"),
}


def _model_identity(obj, excluded=()):
    from temsim.immutable_json import thaw_json
    graph = thaw_json(encode_instrument(obj))
    attrs = graph["nodes"][graph["root"]["ref"]]["attributes"]
    for name in excluded:
        attrs.pop(name, None)
    return json_digest(graph)


@dataclass(frozen=True)
class IlluminationPatch:
    request_id: str
    revision: int
    checkpoint: object
    before: object
    after: object
    controls: tuple
    exact_changes: tuple

    def replacement(self, state, *, revision):
        if revision != self.revision or capture_instrument_snapshot(state).digest != self.before.digest:
            raise ValueError("STALE: current inputs or revision changed after the illumination preview")
        # Rebuild through the same allowlist and compatibility checks at commit.
        verified = prepare_illumination_patch(state, self.checkpoint, revision=revision)
        if verified.after.digest != self.after.digest or verified.controls != self.controls:
            raise ValueError("Illumination patch no longer matches its declared controls")
        return verified.after.restore()


def prepare_illumination_patch(state, checkpoint, *, revision):
    from temsim.optics.electron_gun.source_policy import require_physical_gun_source
    require_physical_gun_source(state.electron_gun)
    before = capture_instrument_snapshot(state)
    donor = checkpoint.snapshot.restore()  # Historical inputs require explicit migration first.
    require_physical_gun_source(donor.electron_gun)
    if _model_identity(state.electron_gun) != _model_identity(donor.electron_gun):
        raise ValueError("Incompatible physical source or gun settings; illumination-only Apply preserves the source")
    if any(getattr(state, name, None) != getattr(donor, name, None)
           for name in ("illumination_mode", "simulation_mode")):
        raise ValueError("Incompatible operating mode; use an explicit full Restore or Fork")
    if getattr(state, "_resolved_assembly", None) is None or getattr(donor, "_resolved_assembly", None) is None:
        raise ValueError("Illumination Apply requires complete captured assemblies")
    if json_digest(encode_instrument(state._resolved_assembly)) != json_digest(encode_instrument(donor._resolved_assembly)):
        raise ValueError("Incompatible assembly or structural model")
    candidate = before.restore()
    targets, incoming = runtime_targets(candidate), runtime_targets(donor)
    controls = []
    for key, names in ILLUMINATION_CONTROLS.items():
        if (key in targets) != (key in incoming):
            raise ValueError(f"Incompatible illumination component: {key}")
        if key not in targets:
            continue
        target, source = targets[key], incoming[key]
        if _model_identity(target.obj, names) != _model_identity(source.obj, names):
            raise ValueError(f"Incompatible geometry, insertion or calibration: {key}")
        for name in names:
            old, value = getattr(target.obj, name), getattr(source.obj, name)
            if old == value:
                continue
            value = validate_runtime_assignment(target, name, value)
            setattr(target.obj, name, value)
            controls.append((key, name, old, value))
    after = capture_instrument_snapshot(candidate)
    changes = snapshot_changes(before, after)
    allowed = set()
    for index, node in enumerate(before.graph["nodes"]):
        attrs = node.get("attributes", {})
        key = attrs.get("key")
        for control_key, name, _, _ in controls:
            if key == control_key and name in attrs:
                allowed.add(f"/graph/nodes/{index}/attributes/{name}")
    if any(path not in allowed for path, _, _ in changes):
        raise ValueError("A runtime setter changed non-target inputs; no illumination controls applied")
    return IlluminationPatch(str(uuid4()), int(revision), checkpoint, before, after,
                             tuple(controls), tuple(changes))
