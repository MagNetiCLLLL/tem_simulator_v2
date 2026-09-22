"""Exact current working-point records, separate from operating profiles.

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
from temsim import input_io

SNAPSHOT_SCHEMA = "complete-working-point-v1"
ASSET_SNAPSHOT_SCHEMA = "complete-working-point-assets-v2"
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
    "optics.electron_gun.tip_coherence",
    "optics.electron_gun.tip_surface",
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
    "_tuning_cancelled",  # Worker cancellation callback, never a physical input.
    "_last_ray_device_receipt",  # Timing only; copied to result performance.
    "_archive_inputs", "_archive_resolver",  # Retained in the graph header; never encoded as a model object.
})
# Older pipeline revisions attached these outputs to State as well as keeping
# them in CalculationResult. They are not source/optics inputs or checkpoints.
_STATE_PRODUCT_NAMES = frozenset({"energy_filter_result", "all_lens_crossovers", "last_gun_waist_mm"})


@lru_cache(maxsize=1)
def _model_types():
    result = {}
    for suffix in (*_MODEL_MODULES, "vacuum"):
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


def encode_instrument(state, *, asset_store=None) -> Mapping:
    """Capture every supported parameter, including disabled hardware.

    Completed outputs remain in CalculationResult, not in this input graph.
    Unknown model objects fail closed rather than being silently omitted.
    Arrays are stored as exact bytes, with dtype/shape; JSON is finite and
    floating-point values are never rounded to display precision.
    """
    nodes, seen, array_aliases = [], {}, {}
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
            if asset_store is not None and value.nbytes >= 65536:
                alias = array_aliases.setdefault(id(value), len(array_aliases))
                return {"asset": asset_store.register(value), "dtype": value.dtype.str,
                        "shape": list(value.shape), "readonly": not value.flags.writeable, "alias": alias}
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
            names = _attribute_names(value)
            if type_key == "temsim.optics.model:State":
                names = [k for k in names if k not in _STATE_PRODUCT_NAMES]
            node["attributes"] = {k: encode(getattr(value, k)) for k in names}
        return {"ref": index}

    root = encode(state)
    graph = {"schema": ASSET_SNAPSHOT_SCHEMA if asset_store is not None else SNAPSHOT_SCHEMA,
             "root": root, "nodes": nodes}
    if input_io.archive_payload(state) is not None:
        graph["archived_inputs"] = input_io.archive_payload(state)
    return freeze_json(graph)


def decode_instrument(graph, *, assets=None):
    """Return a detached editable state; never normalize saved controls."""
    if graph.get("schema") not in {SNAPSHOT_SCHEMA, ASSET_SNAPSHOT_SCHEMA}:
        raise ValueError("Unsupported working-point schema; historical viewing only")
    if graph.get("schema") == ASSET_SNAPSHOT_SCHEMA and assets is None:
        raise ValueError("This working point requires its pinned input assets")
    registry, nodes, restored = _model_types(), graph["nodes"], {}
    array_aliases = {}

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
                raise ValueError(f"Unsupported model schema for {node['type']}; only current fields are accepted")
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
        if "asset" in value:
            if graph.get("schema") != ASSET_SNAPSHOT_SCHEMA or assets is None:
                raise ValueError("Unexpected input asset reference")
            alias = value["alias"]
            if type(alias) is not int or alias < 0:
                raise ValueError("Invalid input asset alias")
            if alias in array_aliases:
                descriptor, array = array_aliases[alias]
                if descriptor != value:
                    raise ValueError("Conflicting input asset alias")
                return array
            array = assets.array(value["asset"], value["dtype"], value["shape"], readonly=value["readonly"])
            array_aliases[alias] = (value, array)
            return array
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
    from temsim.optics.model import State, STATE_SCHEMA_VERSION
    if not isinstance(result, State):
        raise ValueError("Working-point root must be an instrument State")
    if type(result.schema_version) is not int or result.schema_version != STATE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported instrument state schema; expected {STATE_SCHEMA_VERSION}")
    if result.energy_filter.enabled is not bool(result.energy_filter_installed):
        raise ValueError("Energy filter participation must match its assembly installation")
    from temsim.optics.model import Stigmator
    from temsim.optics.ac_deflector import AcDeflectorComponent
    from temsim.optics.descan_deflector import DescanDeflectorComponent
    from temsim.component_keys import AC_DEFLECTOR, DESCAN_DEFLECTOR
    from temsim.optics.model import Sample
    from temsim.optics.electron_gun.emitter import ColdFieldEmitter
    from temsim.optics.electron_gun.tip_curvature import validate_curvature
    for component in restored.values():
        if isinstance(component, Sample) and any(
            name in vars(component) for name in (
                "specimen_rotation_x_deg", "specimen_rotation_y_deg", "specimen_rotation_z_deg",
                "diffraction_enabled", "g_inv_nm", "excitation_error_inv_nm",
                "rocking_width_inv_nm", "diffuse_broadening_mrad",
            )
        ):
            raise ValueError("Sample snapshot contains retired fields; use the current sample model and quaternion")
        if isinstance(component, ColdFieldEmitter):
            if component.curvature_nm_inv and "_tip_curvature_model" not in vars(component):
                raise ValueError("Curved tip snapshot requires an explicit current curvature model")
            validate_curvature(component)
        if isinstance(component, Stigmator) and component.field_model != "normal_skew":
            raise ValueError("Unsupported stigmator field model; expected normal_skew")
        if isinstance(component, AcDeflectorComponent) and component.key != AC_DEFLECTOR:
            raise ValueError(f"Scan component key must be {AC_DEFLECTOR}")
        if isinstance(component, DescanDeflectorComponent):
            if component.key != DESCAN_DEFLECTOR:
                raise ValueError(f"Descan component key must be {DESCAN_DEFLECTOR}")
            if not component.descan_target_key or component.descan_target_key == "legacy_image_reference":
                raise ValueError("Descan requires an explicit current component key")
    if "archived_inputs" in graph:
        input_io.bind_archive(result, graph["archived_inputs"])
    # Exact current graphs restore without injected defaults or aliases.
    reencoded = encode_instrument(result,
        asset_store=assets if graph.get("schema") == ASSET_SNAPSHOT_SCHEMA else None)
    if json_digest(reencoded) != json_digest(graph):
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

        Legacy records verify original files. Explicit portable copies resolve
        their complete, verified input inventory without writing live files.
        """
        from temsim.calculation_manifest import solver_source_identity
        if self.graph.get("schema") not in {SNAPSHOT_SCHEMA, ASSET_SNAPSHOT_SCHEMA}:
            raise ValueError("This record does not contain a restorable input graph; historical viewing only")
        if self.implementation != solver_source_identity():
            raise ValueError("Solver implementation changed; historical viewing only")
        result = decode_instrument(self.graph)
        from temsim.physics.illumination import illumination_config
        with input_io.input_scope(result, inherit=False) as resolver:
            if resolver is not None:
                resolver.assert_current_runtime()
            for row in self.external_inputs:
                try:
                    content = input_io.read_bytes(row["path"])
                except OSError as exc:
                    raise ValueError(f"Missing {row['role']}; historical viewing only") from exc
                if sha256(content).hexdigest() != row["sha256"]:
                    raise ValueError(f"Changed {row['role']}; historical viewing only")
            from temsim.calculation_manifest import capture_external_input_identities
            actual = {(row.role, row.path, row.sha256) for row in capture_external_input_identities(result)}
            expected = {(row["role"], row["path"], row["sha256"]) for row in self.external_inputs}
            if actual != expected:
                raise ValueError("External dependency inventory changed; capture current inputs before recalculating")
            illumination_config(result)
            result.electron_gun.validate()
        return result


@input_io.using_state_inputs
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
        content = input_io.read_bytes(identity.path)
        if sha256(content).hexdigest() != identity.sha256:
            raise ValueError("External model changed during working-point capture")
        dependencies.append({"role": identity.role, "path": identity.path,
                             "sha256": identity.sha256, "content_hex": content.hex()})
    return InstrumentSnapshot(graph, tuple(dependencies), solver_source_identity())
