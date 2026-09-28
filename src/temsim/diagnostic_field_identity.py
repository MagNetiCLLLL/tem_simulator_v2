"""Expose identities of existing solved fields without creating another cache.

Coordinates are metres from the original tip and potentials retain their
declared reference. Identity is provenance, not a claim of numerical accuracy.
Unknown providers or incomplete historical requests stay unknown.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata
from pathlib import Path

import numpy as np

from temsim.physics.planar_gun_field import request_digest
from temsim.physics.planar_gun_field import PlanarGunField
from temsim.physics.closed_gun_field import ClosedGunField
from temsim.physics.continuous_gun_field import ContinuousGunField
from temsim.physics.axis_regular_potential import AxisRegularPotential
from temsim.physics.axisymmetric_cut_field import AxisymmetricCutField
from temsim.physics.continuous_curvature_conductor import ContinuousCurvatureConductor


def _method_contract(cls, names):
    return tuple((name, getattr(cls, name)) for name in names)


_ELECTRIC_METHODS = {
    cls: _method_contract(cls, ("interpolate", "potential_v_at_global_positions",
                               "potential_rise_v_at_global_positions", "field_at_global_positions_v_per_m")
                          + (() if cls is PlanarGunField else ("tip_material_mask",)))
    for cls in (PlanarGunField, ClosedGunField, ContinuousGunField)
}
_REGULAR_METHODS = _method_contract(AxisRegularPotential, ("interpolate",))
_FEM_METHODS = _method_contract(AxisymmetricCutField, ("interpolate", "surface_z"))
_CONDUCTOR_METHODS = _method_contract(ContinuousCurvatureConductor, ("surface_z_m", "radius_m"))


def _methods_match(value, methods):
    return all(getattr(getattr(value, name, None), "__func__", getattr(value, name, None)) is original
               for name, original in methods)


def identity_digest(namespace: str, inputs) -> str:
    """Hash explicit JSON inputs using the existing field-request convention.

    Callers supply consumed quantities, never a mutable GUI object or repr.
    Nonfinite values and arbitrary objects are rejected by request_digest.
    """
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("Identity namespace must be a nonempty string")
    return request_digest({"namespace": namespace, "inputs": inputs})


def field_array_digest(array) -> str:
    """Bind finite numeric array values, shape and actual represented dtype."""
    value = np.asarray(array)
    if value.dtype.kind not in "biufc" or not np.isfinite(value).all():
        raise ValueError("Field identity requires finite numeric arrays")
    digest = hashlib.sha256()
    digest.update(request_digest({"shape": value.shape, "dtype": value.dtype.str}).encode("ascii"))
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ElectricFieldIdentity:
    status: str
    physical_id: str | None = None
    numerical_id: str | None = None
    request_id: str | None = None
    # Cylindrical vacuum interpolation support: (maximum radius, first Z, last Z).
    # Conductor interception extensions are not additional vacuum support.
    support_r_z_m: tuple[float, float, float] | None = None
    reason: str = ""

    def to_dict(self):
        return asdict(self)


def _physical_request(request):
    from temsim.physics.closed_gun_field import SCHEMA as closed_schema
    from temsim.physics.continuous_gun_field import SCHEMA as curved_schema
    from temsim.physics.planar_gun_field import SCHEMA as planar_schema

    schemas = (closed_schema, curved_schema, planar_schema)
    required = ("potential_reference", "source_admission", "rings", "high_tension_v",
                "extraction_v", "domain", "boundary_conditions", "numerics",
                "implementation_sha256", "libraries")
    if not isinstance(request, dict) or request.get("schema") not in schemas:
        raise ValueError("Electric provider request schema is unknown")
    if any(key not in request for key in required):
        raise ValueError("Electric provider has an incomplete historical field request")
    request_digest(request)  # Reject nonfinite/unsupported inputs, without normalising references.
    if not request["implementation_sha256"] or not request["numerics"] or not request["rings"]:
        raise ValueError("Electric provider lacks numerical implementation or electrode inputs")
    result = {key: deepcopy(value) for key, value in request.items()
              if key not in {"domain", "numerics", "implementation_sha256", "libraries",
                             "limitations", "model_status"}}
    if request["schema"] in (closed_schema, curved_schema):
        physical = request.get("grounded_liner")
        if not physical or not request.get("ground_assignment"):
            raise ValueError("Electric provider lacks the full physical grounded liner")
        # The solved rings contain clipped copies of the physical liner. Validate
        # that relationship before excluding those numerical cuts from physics.
        end, voltage = request["domain"]["exit_m"], request["high_tension_v"]
        generated = [{**row, "stop_m": min(row["stop_m"], end), "potential_rise_v": voltage}
                     for row in physical if row["start_m"] < end]
        for index, (left, right) in enumerate(zip(physical[:-1], physical[1:])):
            face = left["stop_m"]
            if face > end:
                break
            if (left["inner_m"], left["outer_m"]) != (right["inner_m"], right["outer_m"]):
                generated.append({"key": f"grounded_outlet_joint:{index}",
                                  "start_m": face, "stop_m": face,
                                  "inner_m": min(left["inner_m"], right["inner_m"]),
                                  "outer_m": max(left["outer_m"], right["outer_m"]),
                                  "potential_rise_v": voltage})
        captured = [row for row in request["rings"] if row["key"].startswith("grounded_outlet")]
        if captured != generated:
            raise ValueError("Electric field rings do not match the complete physical liner")
        result["rings"] = [row for row in result["rings"]
                           if not row["key"].startswith("grounded_outlet")]
    if request["schema"] == curved_schema:
        if not request.get("cathode_geometry"):
            raise ValueError("Electric provider lacks the curved conductor geometry")
        # Emitting support affects mesh refinement, not conductor shape. It
        # remains in the full numerical request, including admission checks.
        result["cathode_geometry"].pop("emission_support_radius_nm", None)
        result["boundary_conditions"]["cathode"]["geometry"].pop("emission_support_radius_nm", None)
    return result


def electric_request_physical_id(request) -> str | None:
    """Identify declared physical inputs; incomplete/unsupported requests return None."""
    try:
        return identity_digest("electric-physical-v1", _physical_request(request))
    except (TypeError, ValueError, KeyError, AttributeError):
        return None


def _field_arrays(provider):
    from temsim.physics.continuous_gun_field import _FEM_ARRAYS
    arrays = {name: getattr(provider, name) for name in ("r", "z", "voltage")}
    if type(provider) is ContinuousGunField:
        arrays.update({f"fem.{name}": getattr(provider._fem, name) for name in _FEM_ARRAYS})
        arrays.update({f"regular.{name}": getattr(provider._regular, name)
                       for name in ("radius_m", "s", "slope")})
    return arrays


def _current_interpolation_identity(provider_type):
    """Bind execution of old solved arrays to today's admitted evaluator code."""
    names = ["physics/planar_gun_field.py"]
    if provider_type is ClosedGunField:
        names.append("physics/closed_gun_field.py")
    elif provider_type is ContinuousGunField:
        names.extend(("physics/continuous_gun_field.py", "physics/axis_regular_potential.py",
                      "physics/axisymmetric_cut_field.py", "physics/continuous_curvature_conductor.py",
                      "physics/axis_field_interpolation.py"))
    root = Path(__file__).parent
    libraries = {"numpy": np.__version__}
    if provider_type is ContinuousGunField:
        for name in ("numba", "llvmlite"):
            try:
                libraries[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                libraries[name] = None
    return identity_digest("electric-interpolation-implementation-v1", {
        "files": {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names},
        "libraries": libraries})


def electric_field_identity(provider) -> ElectricFieldIdentity:
    """Bind a supported immutable field's request and actual interpolation data.

    This is intended once at scene preparation, not on every electron edit.
    Composed/custom fields need explicit component identities at their owner.
    """
    from temsim.physics.closed_gun_field import ClosedGunField, SCHEMA as closed_schema
    from temsim.physics.continuous_gun_field import ContinuousGunField, SCHEMA as curved_schema
    from temsim.physics.planar_gun_field import PlanarGunField, SCHEMA as planar_schema

    if type(provider) not in (ClosedGunField, ContinuousGunField, PlanarGunField):
        return ElectricFieldIdentity("unknown", reason="Unsupported electric provider identity")
    try:
        if not _methods_match(provider, _ELECTRIC_METHODS[type(provider)]):
            raise ValueError("Electric provider uses a custom field method")
        request = provider.request
        physical_id = identity_digest("electric-physical-v1", _physical_request(request))
        expected_schema = {ClosedGunField: closed_schema, ContinuousGunField: curved_schema,
                           PlanarGunField: planar_schema}[type(provider)]
        if request["schema"] != expected_schema:
            raise ValueError("Electric provider type disagrees with its field request schema")
        request_id = request_digest(request)
        if provider.report.get("request_sha256") != request_id:
            raise ValueError("Electric field report is missing or disagrees with its request identity")
        if provider.report.get("potential_reference") != request["potential_reference"]:
            raise ValueError("Electric provider disagrees with its declared potential reference")
        if type(provider) is ContinuousGunField:
            if (type(provider._regular) is not AxisRegularPotential
                    or type(provider._fem) is not AxisymmetricCutField
                    or type(provider.geometry) is not ContinuousCurvatureConductor
                    or not _methods_match(provider._regular, _REGULAR_METHODS)
                    or not _methods_match(provider._fem, _FEM_METHODS)
                    or not _methods_match(provider.geometry, _CONDUCTOR_METHODS)
                    or provider._regular.field is not provider._fem
                    or provider.r is not provider._fem.r or provider.z is not provider._fem.z
                    or provider.voltage is not provider._fem.nodal_voltage
                    or provider.geometry.to_dict() != request["cathode_geometry"]):
                raise ValueError("Electric provider uses custom or inconsistent interpolation/conductor data")
        arrays = _field_arrays(provider)
        if any(not isinstance(value, np.ndarray) or value.flags.writeable for value in arrays.values()):
            raise ValueError("Electric provider data are not immutable arrays")
        r, z = provider.r, provider.z
        if (r.ndim != 1 or z.ndim != 1 or len(r) < 2 or len(z) < 2
                or r[0] != 0. or np.any(np.diff(r) <= 0.) or np.any(np.diff(z) <= 0.)
                or provider.voltage.shape != (len(r), len(z))):
            raise ValueError("Electric field interpolation arrays have invalid shape or axes")
        support = (float(r[-1]), float(z[0]), float(z[-1]))
        domain = request["domain"]
        if support != (domain["outer_radius_m"], domain["entrance_m"], domain["exit_m"]):
            raise ValueError("Electric field arrays do not cover their declared numerical domain")
        numerical_id = identity_digest("electric-numerical-v1", {
            "request_sha256": request_id, "provider": type(provider).__name__,
            "support_r_z_m": support,
            "current_interpolation": _current_interpolation_identity(type(provider)),
            "regular_compiled_requested": (bool(provider._regular.compiled)
                                           if type(provider) is ContinuousGunField else None),
            "arrays": {name: field_array_digest(value) for name, value in arrays.items()},
        })
        return ElectricFieldIdentity("known", physical_id, numerical_id, request_id, support)
    except (TypeError, ValueError, KeyError, AttributeError, IndexError) as error:
        return ElectricFieldIdentity("unknown", reason=str(error))


def _reuse_request(request):
    """Allow only a different axial cut inside the same continuing conductor."""
    physical = _physical_request(request)
    if "grounded_liner" not in physical:
        return request  # The separate open planar prototype has no continuing liner.
    result = deepcopy(request)
    result["rings"] = physical["rings"]
    result["domain"].pop("exit_m")
    result["numerics"].pop("exit_extension_mm")
    return result


def can_reuse_electric_field(provider, requested_request, axial_bounds_m,
                            radial_limit_m=None) -> bool:
    """Admit an already solved covering field without relabelling its identity.

    Every numerical dependency other than the axial cut must match. A finer
    mesh or changed interpolation is not silently substituted. Callers must
    continue reporting ``electric_field_identity(provider)`` for reused data.
    No field is solved, cached, extrapolated or changed here.
    """
    identity = electric_field_identity(provider)
    if identity.status != "known":
        return False
    try:
        if electric_request_physical_id(requested_request) != identity.physical_id:
            return False
        if request_digest(_reuse_request(provider.request)) != request_digest(_reuse_request(requested_request)):
            return False
        low, high = (float(value) for value in axial_bounds_m)
        radius, first, last = identity.support_r_z_m
        requested_radius = (float(requested_request["domain"]["outer_radius_m"])
                            if radial_limit_m is None else float(radial_limit_m))
        return bool(np.isfinite((low, high, requested_radius)).all()
                    and low <= high and requested_radius >= 0.
                    and first <= low and high <= last and requested_radius <= radius)
    except (TypeError, ValueError, KeyError, AttributeError):
        return False
