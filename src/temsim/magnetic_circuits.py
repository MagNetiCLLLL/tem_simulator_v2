"""Explicit magnetic circuits, independent of the number of optical controls.

All geometry remains TOML-owned. A circuit groups physical bodies and current
channels; it never creates another optical element. Legacy assemblies retain
their existing geometry and are reported as unverified, not OEM reconstructions.
"""

from dataclasses import dataclass
from collections.abc import Mapping
import math

import numpy as np

TOPOLOGIES = {
    "two_pole_single_gap": "Independent two-pole lens",
    "shared_pole_multi_gap": "Shared-pole compound lens",
    "monolithic_saturated_insert": "Monolithic saturation insert",
    "air_core": "Air-core coil",
}
MAGNETIC_BODIES = frozenset({
    "magnetic_lens_yoke", "magnetic_pole_piece",
})
# A photo-identified mixed-material carrier affects map identity, but is not
# silently filled with homogeneous iron by the material solver.
MAGNETIC_STRUCTURES = MAGNETIC_BODIES | {"c1_c2_pole_piece_cartridge"}
COIL = "magnetic_excitation_coil"
LENS = "magnetic_lens_assembly"
EVIDENCE_LEVELS = frozenset({
    "engineering_assumption", "published_design", "measured_component", "oem_drawing",
})
CUSTOM_MECHANICAL_ROLES = frozenset({"custom_mechanical", "custom_mechanical_copy"})


def part_data(part) -> dict:
    """Accept both manifest dictionaries and resolved immutable parts."""
    if isinstance(part, Mapping):
        return dict(part)
    return {**dict(part.data), "key": part.key, "parent_key": part.parent_key}


def is_custom_mechanical_part(part) -> bool:
    """Explicit CAD-only bodies; native mechanical coils still belong to FEM."""
    return part_data(part).get("mechanical_part_role") in CUSTOM_MECHANICAL_ROLES


def optical_owner(part, by_key) -> str | None:
    """Nearest optical ancestor; nested lenses own their own current channel."""
    row = part_data(part)
    seen = set()
    while row:
        if is_custom_mechanical_part(row):
            return None
        key = str(row.get("key", ""))
        if key in seen:
            raise ValueError(f"Cyclic magnetic-part ancestry: {key}")
        seen.add(key)
        if row.get("mechanical_profile") == LENS:
            return key
        parent = by_key.get(str(row.get("parent_key", "")))
        row = part_data(parent) if parent is not None else {}
    return None


def circuit_channels(by_key, lens_key: str) -> tuple[str, ...]:
    row = part_data(by_key[lens_key]) if lens_key in by_key else {}
    if is_custom_mechanical_part(row):
        return ()
    circuit = row.get("magnetic_circuit_id")
    if not circuit:
        return (lens_key,)
    return tuple(sorted(key for key, part in by_key.items()
                        if part_data(part).get("mechanical_profile") == LENS
                        and not is_custom_mechanical_part(part)
                        and part_data(part).get("magnetic_circuit_id") == circuit))


def belongs_to_circuit(part, lens_key: str, by_key, channels=None) -> bool:
    """One body can affect multiple maps without being duplicated in the column."""
    row = part_data(part)
    if is_custom_mechanical_part(row):
        return False
    channels = set(circuit_channels(by_key, lens_key)) if channels is None else channels
    return (optical_owner(part, by_key) in channels
            or bool(channels.intersection(row.get("magnetic_lens_keys", ()))))


def radial_profile_mm(data, length_mm: float) -> np.ndarray | None:
    """Return [offset Z, inner radius, outer radius] in mm, linear between knots.

    The profile spans the entire part. It is shared by rendering and the field
    material mask; it is not an extra drawing-only shape or an inferred B-H law.
    """
    values = data.get("magnetic_radial_profile_mm")
    if values is None:
        return None
    profile = np.asarray(values, dtype=float)
    if (profile.ndim != 2 or profile.shape[1] != 3 or len(profile) < 2
            or not np.all(np.isfinite(profile)) or np.any(np.diff(profile[:, 0]) <= 0)
            or not math.isclose(float(profile[0, 0]), 0, abs_tol=1e-9)
            or not math.isclose(float(profile[-1, 0]), length_mm, rel_tol=0, abs_tol=1e-9)
            or np.any(profile[:, 1] < 0) or np.any(profile[:, 2] <= profile[:, 1])):
        raise ValueError("Magnetic radial profile requires increasing Z from 0 to part length and 0 <= inner < outer radii (mm)")
    envelope = float(data["mechanical_outer_diameter_mm"]) / 2
    clear = float(data.get("vacuum_inner_diameter_mm", 0)) / 2
    if np.max(profile[:, 2]) > envelope + 1e-9 or np.min(profile[:, 1]) < clear - 1e-9:
        raise ValueError("Magnetic radial profile must fit its envelope and clear its vacuum bore")
    return profile


@dataclass(frozen=True)
class MagneticCircuit:
    key: str
    topology: str
    channels: tuple[str, ...]
    body_keys: tuple[str, ...]
    coil_keys: tuple[str, ...]
    evidence: str
    source: str


def circuit_inventory(parts) -> tuple[MagneticCircuit, ...]:
    """Read-only evidence for the inspector; no field solve or geometry changes."""
    parts = tuple(parts)
    by_key = {str(part_data(part)["key"]): part for part in parts}
    result = []
    seen = set()
    for key, part in by_key.items():
        row = part_data(part)
        if row.get("mechanical_profile") != LENS or is_custom_mechanical_part(row):
            continue
        circuit = str(row.get("magnetic_circuit_id", key))
        identity = (bool(row.get("magnetic_circuit_id")), circuit)
        if identity in seen:
            continue
        seen.add(identity)
        channels = set(circuit_channels(by_key, key))
        members = [part_data(p) for p in parts if belongs_to_circuit(p, key, by_key, channels)]
        result.append(MagneticCircuit(
            circuit, str(row.get("magnetic_circuit_topology", row.get("pole_piece_topology", "unspecified"))),
            circuit_channels(by_key, key),
            tuple(p["key"] for p in members if p.get("mechanical_profile") in MAGNETIC_BODIES),
            tuple(p["key"] for p in members if p.get("mechanical_profile") == COIL),
            str(row.get("magnetic_circuit_evidence", "engineering_assumption")),
            str(row.get("magnetic_circuit_source", "Legacy reconstruction; internal topology not verified")),
        ))
    return tuple(result)


def validate_circuit_declarations(parts) -> None:
    """Validate opt-in circuits without forcing a pair of poles per channel.

    Legacy manufacturing heuristics are retained for undeclared assemblies.
    This check establishes geometry/schema consistency, never OEM provenance.
    """
    rows = [part_data(part) for part in parts if not is_custom_mechanical_part(part)]
    by_key = {str(row["key"]): row for row in rows}
    lenses = {key for key, row in by_key.items() if row.get("mechanical_profile") == LENS}
    for row in rows:
        key = row["key"]
        references = row.get("magnetic_lens_keys", ())
        if (not isinstance(references, (list, tuple)) or any(not isinstance(k, str) for k in references)
                or len(set(references)) != len(references) or not set(references) <= lenses):
            raise ValueError(f"{key}: magnetic_lens_keys must reference distinct optical lens keys")
        if references and row.get("mechanical_profile") not in MAGNETIC_STRUCTURES:
            raise ValueError(f"{key}: only passive magnetic bodies may declare magnetic_lens_keys")
        if "field_source_key" in row:
            if row.get("mechanical_profile") != COIL or row["field_source_key"] not in lenses:
                raise ValueError(f"{key}: field_source_key must assign a coil to one optical lens")
            owner = optical_owner(row, by_key)
            if owner and row["field_source_key"] not in circuit_channels(by_key, owner):
                raise ValueError(f"{key}: coil owner must belong to its magnetic circuit")
        if "magnetic_radial_profile_mm" in row:
            if row.get("mechanical_profile") not in MAGNETIC_BODIES:
                raise ValueError(f"{key}: radial profiles describe passive magnetic bodies only")
            radial_profile_mm(row, float(row["length_mm"]))
        if "magnetic_circuit_topology" in row and not row.get("magnetic_circuit_id"):
            raise ValueError(f"{key}: magnetic_circuit_topology requires a circuit ID")
        if not row.get("magnetic_circuit_id"):
            continue
        if key not in lenses or not isinstance(row["magnetic_circuit_id"], str):
            raise ValueError(f"{key}: declare circuit IDs on optical lens parents only")
        if not row["magnetic_circuit_id"].strip():
            raise ValueError(f"{key}: circuit ID must not be blank")
        topology = row.get("magnetic_circuit_topology")
        if topology not in TOPOLOGIES:
            raise ValueError(f"{key}: unsupported magnetic circuit topology {topology}")
        if row.get("pole_piece_topology", topology) != topology:
            raise ValueError(f"{key}: legacy and explicit magnetic topologies disagree")
        if (row.get("magnetic_circuit_evidence") not in EVIDENCE_LEVELS
                or not str(row.get("magnetic_circuit_source", "")).strip()):
            raise ValueError(f"{key}: magnetic circuit requires evidence level and source")

    for circuit in circuit_inventory(rows):
        parents = [by_key[key] for key in circuit.channels]
        if not parents[0].get("magnetic_circuit_id"):
            continue
        if any(p.get("magnetic_circuit_topology") != circuit.topology for p in parents):
            raise ValueError(f"{circuit.key}: channels must declare the same circuit topology")
        if any(p.get("magnetic_circuit_evidence") != circuit.evidence for p in parents):
            raise ValueError(f"{circuit.key}: channels must declare the same circuit evidence level")
        members = [by_key[key] for key in (*circuit.body_keys, *circuit.coil_keys)]
        for member in members:
            length = float(member["length_mm"])
            inner = float(member.get("mechanical_inner_diameter_mm", member.get("mechanical_bore_diameter_mm", 0)))
            outer = float(member["mechanical_outer_diameter_mm"])
            if not all(math.isfinite(v) for v in (length, inner, outer)) or length <= 0 or not 0 <= inner < outer:
                raise ValueError(f"{member['key']}: magnetic body/coil dimensions are invalid")
        for channel in circuit.channels:
            coils = [by_key[k] for k in circuit.coil_keys
                     if by_key[k].get("field_source_key", optical_owner(by_key[k], by_key)) == channel]
            if not coils:
                raise ValueError(f"{channel}: no owned excitation coil in declared circuit")
        poles = [p for p in members if p.get("mechanical_profile") == "magnetic_pole_piece"]
        if circuit.topology == "air_core" and circuit.body_keys:
            raise ValueError(f"{circuit.key}: an air-core circuit cannot contain magnetic bodies")
        if circuit.topology == "two_pole_single_gap" and (len(circuit.channels) != 1 or len(poles) != 2):
            raise ValueError(f"{circuit.key}: independent two-pole circuit requires one channel and two poles")
        if circuit.topology == "two_pole_single_gap":
            upstream, downstream = sorted(poles, key=lambda p: p["local_start_z_mm"])
            gap = float(downstream["local_start_z_mm"]) - float(upstream["local_end_z_mm"])
            if gap <= 0 or not math.isclose(gap, float(parents[0].get("pole_gap_mm", gap)), rel_tol=0, abs_tol=1e-9):
                raise ValueError(f"{circuit.key}: pole faces must have the declared positive gap")
        if circuit.topology == "shared_pole_multi_gap" and (len(circuit.channels) < 2 or len(poles) < 3):
            raise ValueError(f"{circuit.key}: shared-pole circuit requires multiple channels and at least three poles")
        if circuit.topology == "monolithic_saturated_insert":
            inserts = [p for p in members if p.get("magnetic_part_role") == "saturating_insert"]
            if len(circuit.channels) != 1 or poles or len(inserts) != 1 or "magnetic_radial_profile_mm" not in inserts[0]:
                raise ValueError(f"{circuit.key}: monolithic circuit requires one channel, one profiled saturation insert and no independent poles")
