"""Exact working-point records, separate from migrating operating profiles.

No constructors, presets, layout resolvers or defaults run during restoration.
Only allow-listed model types are decoded; this is not pickle or an arbitrary
Python object loader. Object references preserve shared component ownership.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from functools import lru_cache
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from types import MappingProxyType
import math

import numpy as np

from temsim.immutable_json import freeze_json, json_digest, thaw_json

SNAPSHOT_SCHEMA = "complete-working-point-v1"
# Deliberately bounded to parameter/assembly models, never GUI, IO or workers.
_MODEL_MODULES = (
    "column.layout", "column.module_assembly",
    "detector.camera", "detector.fluorescent_screen", "detector.stem_detector",
    "optics.model", "optics.nanopulser", "optics.ac_deflector",
    "optics.beam_deflector", "optics.condenser_aperture", "optics.condenser_deflector",
    "optics.condenser_lens", "optics.condenser_stigmator", "optics.descan_deflector",
    "optics.diffraction_lens", "optics.diffraction_stigmator",
    "optics.electron_gun.alignment", "optics.electron_gun.aperture",
    "optics.electron_gun.electrostatic", "optics.electron_gun.emitter",
    "optics.electron_gun.effective_source",
    "optics.electron_gun.field_emission", "optics.electron_gun.thermionic",
    "optics.electron_gun.monochromator", "optics.energy_filter",
    "optics.energy_filter_detector", "optics.energy_filter_entrance_aperture",
    "optics.energy_filter_m12", "optics.energy_filter_slit",
    "optics.image_corrector", "optics.image_diffraction_deflector",
    "optics.intermediate_lens", "optics.mini_condenser", "optics.objective_aperture",
    "optics.objective_lens", "optics.objective_stigmator", "optics.probe_corrector",
    "optics.projector_lens_p1", "optics.projector_lens_p2",
    "optics.selected_area_aperture", "optics.twelve_pole_element",
    "physics.finite_multipole_field", "physics.multipole_field",
    "physics.lens_field_provider",
)
_PLAIN_TYPES = frozenset({
    "LayoutResult", "MultipoleField", "CondenserLensComponent",
    "CondenserSystem", "ProbeCorrectorSystem", "ImageCorrectorSystem",
})
# These are memoized products, not inputs. All public attributes are captured,
# including extra parameters attached by the assembly/calibration loaders.
_RUNTIME_NAMES = frozenset({
    "_trace_cache", "_trace_cache_key", "_active_backends_used",
    "_runtime_lens_field_provider_cache", "_field_provider_diagnostics",
    "_objective_plane_signature", "_equivalent_image_calibration_cache",
})


@lru_cache(maxsize=1)
def _model_types():
    result = {}
    for suffix in _MODEL_MODULES:
        module = import_module("temsim." + suffix)
        for name, cls in vars(module).items():
            if (isinstance(cls, type) and cls.__module__ == module.__name__
                    and (is_dataclass(cls) or name in _PLAIN_TYPES)):
                result[module.__name__ + ":" + name] = cls
    return result


def _attribute_names(value):
    names = set(vars(value)) if hasattr(value, "__dict__") else set()
    if is_dataclass(value):
        names.update(f.name for f in fields(value))
    if type(value).__name__ == "MagneticFieldMap":
        names.discard("_interpolators")
    return sorted(name for name in names if name not in _RUNTIME_NAMES)


def encode_instrument(state) -> Mapping:
    """Capture every supported parameter, including disabled hardware.

    Unknown model objects fail closed rather than being silently omitted.
    Arrays are stored as exact bytes, with dtype/shape; JSON is finite and
    floating-point values are never rounded to display precision.
    """
    nodes, seen = [], {}
    registry = _model_types()

    def encode(value):
        if isinstance(value, np.bool_):
            return bool(value)
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, (float, np.floating)):
            if not math.isfinite(value):
                return {"float": float(value).hex()}
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, Path):
            return {"path": str(value)}
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject or value.dtype.fields is not None:
                raise TypeError("Working-point arrays require a non-object, unstructured dtype")
            return {"array": value.tobytes().hex(), "dtype": value.dtype.str,
                    "shape": list(value.shape), "readonly": not value.flags.writeable}
        if isinstance(value, Mapping):
            return {"mapping": [[encode(k), encode(v)] for k, v in sorted(
                value.items(), key=lambda pair: repr(pair[0]))],
                "readonly": isinstance(value, MappingProxyType)}
        if type(value) in (tuple, list):
            return {type(value).__name__: [encode(v) for v in value]}
        if type(value) in (set, frozenset):
            return {type(value).__name__: [encode(v) for v in sorted(value, key=repr)]}
        type_key = type(value).__module__ + ":" + type(value).__name__
        if registry.get(type_key) is not type(value):
            raise TypeError(f"Unregistered working-point model: {type_key}")
        if id(value) in seen:
            return {"ref": seen[id(value)]}
        index = len(nodes)
        seen[id(value)] = index
        node = {"type": type_key}
        nodes.append(node)
        if isinstance(value, tuple):
            node["items"] = [encode(v) for v in value]
        else:
            node["fields"] = [f.name for f in fields(value)] if is_dataclass(value) else []
            node["attributes"] = {k: encode(getattr(value, k)) for k in _attribute_names(value)}
        return {"ref": index}

    root = encode(state)
    return freeze_json({"schema": SNAPSHOT_SCHEMA, "root": root, "nodes": nodes})


def decode_instrument(graph):
    """Return a detached editable state; never normalize saved controls."""
    if graph.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("Unsupported working-point schema; historical viewing only")
    registry, nodes, restored = _model_types(), graph["nodes"], {}

    def decode(value):
        if not isinstance(value, Mapping):
            return value
        if "ref" in value:
            index = value["ref"]
            if not isinstance(index, int) or not 0 <= index < len(nodes):
                raise ValueError("Invalid working-point object reference")
            if index in restored:
                return restored[index]
            node = nodes[index]
            cls = registry.get(node["type"])
            if cls is None:
                raise ValueError("Unregistered working-point model; historical viewing only")
            if issubclass(cls, tuple):
                result = tuple.__new__(cls, (decode(v) for v in node["items"]))
                restored[index] = result
                return result
            current_fields = [f.name for f in fields(cls)] if is_dataclass(cls) else []
            if list(node["fields"]) != current_fields:
                raise ValueError(f"Model schema changed for {node['type']}; explicit migration required")
            attributes = node["attributes"]
            required_fields = set(current_fields)
            if cls.__name__ == "MagneticFieldMap":
                required_fields.discard("_interpolators")
            if not required_fields.issubset(attributes):
                raise ValueError("Incomplete working-point model fields")
            if any(k.startswith("__") for k in attributes):
                raise ValueError("Invalid working-point attribute")
            result = object.__new__(cls)
            restored[index] = result
            for k, v in attributes.items():
                if isinstance(getattr(cls, k, None), property) or callable(getattr(cls, k, None)):
                    raise ValueError("Working-point attributes cannot invoke model methods or setters")
                if hasattr(result, "__dict__"):
                    result.__dict__[k] = decode(v)
                else:
                    object.__setattr__(result, k, decode(v))
            # Gun caches are deliberately absent, with their documented empty state.
            if cls.__module__ in {"temsim.optics.electron_gun.field_emission",
                                  "temsim.optics.electron_gun.thermionic"}:
                object.__setattr__(result, "_trace_cache", None)
                object.__setattr__(result, "_trace_cache_key", None)
            if cls.__name__ == "MagneticFieldMap":
                # Recreate only the interpolation accelerator from captured
                # grids/values; never regenerate the physical field or fit it.
                from scipy.interpolate import RegularGridInterpolator
                object.__setattr__(result, "_interpolators", tuple(
                    RegularGridInterpolator(result.axes_m, component, method="linear",
                                            bounds_error=False, fill_value=0.)
                    for component in result.components_t
                ))
            return result
        if "array" in value:
            dtype = np.dtype(value["dtype"])
            if dtype.hasobject or dtype.fields is not None:
                raise ValueError("Object and structured arrays are forbidden")
            array = np.frombuffer(bytes.fromhex(value["array"]), dtype=dtype).reshape(value["shape"])
            return array if value["readonly"] else array.copy()
        if "mapping" in value:
            result = {decode(k): decode(v) for k, v in value["mapping"]}
            return MappingProxyType(result) if value["readonly"] else result
        if "list" in value:
            return [decode(v) for v in value["list"]]
        if "tuple" in value:
            return tuple(decode(v) for v in value["tuple"])
        if "set" in value:
            return set(decode(v) for v in value["set"])
        if "frozenset" in value:
            return frozenset(decode(v) for v in value["frozenset"])
        if "path" in value:
            return Path(value["path"])
        if "float" in value:
            return float.fromhex(value["float"])
        raise ValueError("Invalid working-point value")

    result = decode(graph["root"])
    from temsim.optics.model import State
    if not isinstance(result, State):
        raise ValueError("Working-point root must be an instrument State")
    if json_digest(encode_instrument(result)) != json_digest(graph):
        raise ValueError("Working-point restoration did not preserve every captured value")
    return result


@dataclass(frozen=True, slots=True)
class InstrumentSnapshot:
    """Complete model graph and content-pinned external inputs.

    Native model field names retain units (mm, nm, radians, percent, etc.).
    Viewing to_dict() is detached and never applies this state to the UI.
    """
    graph: Mapping
    external_inputs: tuple = ()
    implementation: str = ""

    def __post_init__(self):
        object.__setattr__(self, "graph", freeze_json(self.graph))
        object.__setattr__(self, "external_inputs", freeze_json(self.external_inputs))
        for row in self.external_inputs:
            content = row["content_hex"]
            if content is not None and sha256(bytes.fromhex(content)).hexdigest() != row["sha256"]:
                raise ValueError("Archived model content checksum mismatch")

    @property
    def digest(self):
        return json_digest(self.identity_payload())

    @property
    def physical_digest(self):
        """Compare physics without changing the complete restoration payload.

        Only the explicitly read-only virtual observation plane is omitted.
        Unknown/new fields remain physical by default. Do not use display
        signatures to ignore source, geometry, model or numerical changes.
        """
        payload = thaw_json(freeze_json(self.identity_payload()))
        root = payload["graph"]["root"]["ref"]
        payload["graph"]["nodes"][root]["attributes"].pop("virtual_observation_z_mm", None)
        return json_digest(payload)

    def identity_payload(self):
        return {"graph": self.graph, "external_inputs": self.external_inputs,
                "implementation": self.implementation}

    def to_dict(self):
        return {**thaw_json(freeze_json(self.identity_payload())), "digest": self.digest}

    @classmethod
    def from_dict(cls, data):
        result = cls(data["graph"], tuple(data["external_inputs"]), data["implementation"])
        if result.digest != data["digest"]:
            raise ValueError("Working-point checksum mismatch")
        return result

    def restore(self):
        """Explicit restoration; changed/missing dependencies remain read-only.

        Embedded external bytes preserve evidence, but this first slice does
        not replace live files or redirect field loaders to an archive.
        """
        from temsim.calculation_manifest import solver_source_identity
        if self.implementation != solver_source_identity():
            raise ValueError("Solver implementation changed; historical viewing only")
        for row in self.external_inputs:
            try:
                content = Path(row["path"]).read_bytes()
            except OSError as exc:
                raise ValueError(f"Missing {row['role']}; historical viewing only") from exc
            if sha256(content).hexdigest() != row["sha256"]:
                raise ValueError(f"Changed {row['role']}; historical viewing only")
        result = decode_instrument(self.graph)
        from temsim.physics.illumination import illumination_config
        illumination_config(result)
        return result


def capture_instrument_snapshot(state) -> InstrumentSnapshot:
    from temsim.calculation_manifest import (
        capture_external_input_identities, solver_source_identity,
    )
    graph = encode_instrument(state)
    dependencies = []
    for identity in capture_external_input_identities(state):
        if not identity.available:
            dependencies.append({"role": identity.role, "path": identity.path,
                                 "sha256": "", "content_hex": None})
            continue
        content = Path(identity.path).read_bytes()
        if sha256(content).hexdigest() != identity.sha256:
            raise ValueError("External model changed during working-point capture")
        dependencies.append({"role": identity.role, "path": identity.path,
                             "sha256": identity.sha256, "content_hex": content.hex()})
    return InstrumentSnapshot(graph, tuple(dependencies), solver_source_identity())
