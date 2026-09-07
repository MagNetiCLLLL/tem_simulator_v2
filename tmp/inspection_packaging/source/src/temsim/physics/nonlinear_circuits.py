"""Joint nonlinear field routing: one field, one contribution to transport.

All configured B-H channels are solved together. Their display/control entries
remain separate, but only the first enabled entry carries the joint field.
No independently saturated solutions are superposed. Nearby active foreign
coils must join the solve instead of being silently ignored.
"""

from dataclasses import replace
from functools import lru_cache
import json
from types import SimpleNamespace

import numpy as np

from temsim.magnetic_materials import validate_bh_material

SOLVER = "axisymmetric_nonlinear_fem"


def nonlinear_state_fingerprint(state):
    """Read-only diagnostics identity; never assemble a mesh or solve fields."""
    from temsim.physics.lens_field_provider import _fingerprint, _part_geometry_data, _runtime_excitation, _GEOMETRY_ATTRIBUTES
    physical = tuple((p.key, p.start_z_mm, p.end_z_mm, _part_geometry_data(p))
                     for p in state._resolved_assembly.parts)
    lenses = tuple((lens.key, _runtime_excitation(lens),
                    tuple((name, float(getattr(lens, name))) for name in _GEOMETRY_ATTRIBUTES
                          if isinstance(getattr(lens, name, None), (float, int)))) for lens in state.lenses)
    return _fingerprint((physical, lenses, state.lens_field_map_descriptors))


def operator_settings(recipe):
    if recipe.get("solver") != SOLVER:
        raise ValueError("Nonlinear current channels require B-H recipes")
    settings = dict(solver=SOLVER, bh_material=validate_bh_material(recipe.get("bh_material")),
                material_bh_overrides={key: validate_bh_material(row)
                                       for key, row in recipe.get("material_bh_overrides", {}).items()},
                radial_nodes=int(recipe.get("radial_nodes", 40)), axial_nodes=int(recipe.get("axial_nodes", 80)),
                padding_factor=float(recipe.get("padding_factor", 2)),
                relative_tolerance=float(recipe.get("relative_tolerance", 1e-7)),
                max_iterations=int(recipe.get("max_iterations", 80)))
    if not (12 <= settings["radial_nodes"] <= 256 and 16 <= settings["axial_nodes"] <= 512
            and np.isfinite(settings["padding_factor"]) and 1.5 <= settings["padding_factor"] <= 10
            and np.isfinite(settings["relative_tolerance"]) and 1e-12 <= settings["relative_tolerance"] <= 1e-3
            and 1 <= settings["max_iterations"] <= 500):
        raise ValueError("Invalid B-H mesh, boundary padding, tolerance or iteration limit")
    for name in ("radial_nodes", "axial_nodes", "max_iterations"):
        if name in recipe and settings[name] != recipe[name]:
            raise ValueError(f"B-H {name} must be an integer")
    return settings


@lru_cache(maxsize=12)
def _joint_map(geometry_json, settings_json):
    from temsim.physics.axisymmetric_magnetostatics import solve_geometry_field_map
    return solve_geometry_field_map(SimpleNamespace(canonical_geometry_json=geometry_json), json.loads(settings_json))


def resolve_nonlinear_provider(state, key, native, binding, *, prepare_only=False):
    from temsim.component_keys import CONDENSER_LENS_KEYS
    from temsim.magnetic_circuits import MAGNETIC_BODIES, circuit_channels
    from temsim.physics.lens_field_provider import (
        lens_geometry_binding, _runtime_excitation, MappedLensFieldProvider, FieldMapError, _part_geometry_data, _fingerprint,
    )
    recipes = state.lens_field_map_descriptors
    lenses = {lens.key: lens for lens in state.lenses}
    keys = tuple(sorted(k for k, row in recipes.items() if row.get("solver") == SOLVER and k in lenses))
    operator = operator_settings(recipes[key])
    for channel in keys:
        if operator_settings(recipes[channel]) != operator:
            raise FieldMapError("Joint B-H channels require identical material, mesh and solver settings; coil ampere-turns may differ")
    by_key = {part.key: part for part in state._resolved_assembly.parts}
    for channel in keys:
        missing = set(circuit_channels(by_key, channel))-set(keys)
        if missing:
            raise FieldMapError("Configure every shared magnetic-circuit channel for B-H, including disabled coils: " + ", ".join(sorted(missing)))
    sources = tuple(state.condenser_system[k] if k in CONDENSER_LENS_KEYS else lenses[k] for k in keys)
    point = tuple(_runtime_excitation(source) for source in sources)
    guard_sources = tuple(state.lenses)
    guard_point = tuple(_runtime_excitation(source) for source in guard_sources)
    active = [k for k, (enabled, _, _) in zip(keys, point) if enabled]
    owner = active[0] if active else keys[0]
    bindings = [lens_geometry_binding(state, k, source) for k, source in zip(keys, sources)]
    physical_token = tuple((p.key, p.start_z_mm, p.end_z_mm, json.dumps(_part_geometry_data(p), sort_keys=True))
                           for p in by_key.values() if p.data.get("mechanical_profile") in MAGNETIC_BODIES | {"magnetic_excitation_coil", "magnetic_lens_assembly"})
    token = (keys, point, guard_point, physical_token, tuple(b.geometry_fingerprint for b in bindings),
             json.dumps({k: recipes[k] for k in keys}, sort_keys=True, allow_nan=False))
    cache = getattr(state, "_runtime_nonlinear_provider_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        state._runtime_nonlinear_provider_cache = cache
    cached = cache.get(key)
    if not prepare_only and cached is not None and cached[0] == token and cached[1].native_provider is native:
        cached[1].excitation_scale()
        return cached[1]
    parts, neighbours = {}, {}
    for item in bindings:
        assembly = json.loads(item.canonical_geometry_json)["lens_assembly"]
        parts.update({p["key"]: p for p in assembly["parts"]})
        neighbours.update({p["key"]: p for p in assembly.get("magnetostatic_neighbours", ())})
    used = [p for p in parts.values() if p["data"].get("mechanical_profile") in MAGNETIC_BODIES | {"magnetic_excitation_coil"}]
    if not used:
        raise FieldMapError("Nonlinear magnetic system has no field geometry")
    low, high = min(p["start_z_mm"] for p in used), max(p["end_z_mm"] for p in used)
    centre, half = .5*(low+high), .5*(high-low)*operator["padding_factor"]
    # A driven unmodelled channel inside the solve domain would also magnetize
    # these bodies. Require explicit inclusion; do not call it a joint solution.
    for part in by_key.values():
        if (part.data.get("mechanical_profile") == "magnetic_excitation_coil"
                and part.start_z_mm < centre+half and part.end_z_mm > centre-half):
            channel = part.data.get("field_source_key", part.parent_key)
            lens = lenses.get(channel)
            if channel not in keys and lens is not None and _runtime_excitation(lens)[0] and _runtime_excitation(lens)[1] != 0:
                raise FieldMapError(f"Active coil {channel} overlaps the nonlinear domain; include it in the B-H solve or explicitly disable it")
    # Rebuild passive neighbours over the combined domain, not just the individual
    # local domains. Their canonical data and split intervals come from bindings.
    from temsim.magnetic_geometry import objective_layer_intervals_mm
    for part in by_key.values():
        if part.key not in parts and part.data.get("mechanical_profile") in MAGNETIC_BODIES and part.start_z_mm < centre+half and part.end_z_mm > centre-half:
            data = _part_geometry_data(part)
            parent = by_key.get(part.parent_key)
            if parent is not None:
                intervals = objective_layer_intervals_mm(parent.data, parent.start_z_mm, data["mechanical_profile"])
                if intervals and data["mechanical_profile"] == "magnetic_lens_yoke":
                    data["material_intervals_mm"] = intervals
            neighbours[part.key] = dict(key=part.key, start_z_mm=part.start_z_mm, end_z_mm=part.end_z_mm, data=data)
    geometry = dict(lens_key="joint_bh", lens_assembly=dict(
        magnetic_circuit_topology="joint_nonlinear", magnetic_circuit_channels=keys,
        parts=[parts[k] for k in sorted(parts)],
        magnetostatic_neighbours=[neighbours[k] for k in sorted(neighbours) if k not in parts]))
    currents = {}
    for channel, (enabled, percent, polarity) in zip(keys, point):
        ni = float(recipes[channel]["ampere_turns"])
        if not np.isfinite(ni) or ni < 0:
            raise FieldMapError("Ampere-turns at 100% must be finite and nonnegative; use polarity for direction")
        currents[channel] = ni*percent*.01*polarity if enabled else 0.0
    settings = dict(operator, channel_ampere_turns=currents)
    if prepare_only:
        return geometry, settings, keys
    field = _joint_map(json.dumps(geometry, sort_keys=True, allow_nan=False), json.dumps(settings, sort_keys=True, allow_nan=False))
    # Per-lens interfaces must not multiply the same total field by N channels.
    components = field.components_t if key == owner else tuple(np.zeros_like(a) for a in field.components_t)
    mapped = replace(field, geometry_fingerprint=binding.geometry_fingerprint, components_t=components)
    provider = MappedLensFieldProvider(key, mapped, native, binding,
        model_status="joint_nonlinear_field" if key == owner else "included_in_joint_nonlinear_field",
        excitation_scaling="joint_nonlinear_operating_point", circuit_sources=guard_sources, circuit_operating_point=guard_point)
    cache[key] = token, provider
    diagnostics = getattr(state, "_field_provider_diagnostics", {})
    diagnostics[key] = dict(mode=provider.model_status, model_status=provider.model_status, joint_field_owner=owner,
                            channels=keys, source_note=field.provenance.source_note,
                            geometry_fingerprint=binding.geometry_fingerprint,
                            descriptor_fingerprint=_fingerprint(recipes[key]),
                            operating_point=guard_point,
                            joint_state_fingerprint=nonlinear_state_fingerprint(state),
                            source_sha256=field.provenance.source_sha256,
                            divergence_within_tolerance=field.validation.divergence_within_tolerance)
    state._field_provider_diagnostics = diagnostics
    return provider
