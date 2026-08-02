"""Modular, replaceable electron-gun assemblies."""

from temsim.optics.electron_gun.base import (
    ElectronGunAssembly,
    EmissionBundle,
    GunEqualTimeFront,
    GunEqualTimeHistory,
    GunExitBundle,
    GunPlaneArrival,
    GunTraceResult,
    create_electron_gun,
    register_electron_gun,
    registered_electron_gun_types,
)
from temsim.optics.electron_gun.field_emission import (
    FieldEmissionGun,
    field_emission_gun_from_dict,
)
from temsim.optics.electron_gun.thermionic import (
    ThermionicEmitter,
    ThermionicGun,
    thermionic_gun_from_dict,
)
from temsim.optics.electron_gun.monochromator import (
    AnalyticWienField,
    FiniteWienElement,
    MonochromatorSlit,
    WienMonochromatorAssembly,
    monochromator_from_dict,
)
from temsim.optics.electron_gun.validation import trace_reference_ray_to_c1


register_electron_gun("cold_feg", field_emission_gun_from_dict)
register_electron_gun("thermionic", thermionic_gun_from_dict)

__all__ = [
    "ElectronGunAssembly",
    "EmissionBundle",
    "FieldEmissionGun",
    "GunEqualTimeFront",
    "GunEqualTimeHistory",
    "GunExitBundle",
    "GunPlaneArrival",
    "GunTraceResult",
    "AnalyticWienField",
    "FiniteWienElement",
    "MonochromatorSlit",
    "ThermionicEmitter",
    "ThermionicGun",
    "WienMonochromatorAssembly",
    "create_electron_gun",
    "field_emission_gun_from_dict",
    "monochromator_from_dict",
    "register_electron_gun",
    "registered_electron_gun_types",
    "thermionic_gun_from_dict",
    "trace_reference_ray_to_c1",
]
