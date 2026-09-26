"""Explicit field laws admitted by the virtual-electron compiled backend.

A derived or individually replaced provider is an arbitrary physical callback,
not evidence that its field is still Gaussian. Capture the original functions
and require both a listed concrete type and unchanged method identities.
"""
from temsim.physics.lens_field_provider import GeometryAwareAnalyticFieldProvider
from temsim.magnetic_field_scene import _EquivalentDeflectorField, _MultipoleField
from temsim.optics.condenser_lens import CondenserLensComponent
from temsim.optics.objective_lens import ObjectiveLensComponent
from temsim.optics.round_lens import RoundLensComponent, AnchoredRoundLensComponent
from temsim.optics.diffraction_lens import DiffractionLensComponent
from temsim.optics.intermediate_lens import IntermediateLensComponent
from temsim.optics.projector_lens_p1 import ProjectorLensP1Component
from temsim.optics.projector_lens_p2 import ProjectorLensP2Component
from temsim.optics.mini_condenser import MiniCondenserComponent
from temsim.optics.probe_corrector import (
    AdapterLensComponent, Tl22LensComponent, Tl21LensComponent, Tl12LensComponent,
    Qph2QuadrupoleComponent, QpcQuadrupoleComponent, Qph1QuadrupoleComponent, QpolQuadrupoleComponent,
    Hp2HexapoleComponent, HpcHexapoleComponent, Hp1HexapoleComponent, HpolHexapoleComponent,
)
from temsim.optics.image_corrector import (
    ImageCorrectorOlPostLensComponent, ImageCorrectorTl11LensComponent, ImageCorrectorTl12LensComponent,
    ImageCorrectorTl21LensComponent, ImageCorrectorTl22LensComponent, ImageCorrectorAdapterLensComponent,
    ImageCorrectorHpolHexapoleComponent, ImageCorrectorHp1HexapoleComponent, ImageCorrectorHp2HexapoleComponent,
    ImageCorrectorQpolQuadrupoleComponent, ImageCorrectorDstgQuadrupoleComponent,
)
from temsim.optics.model import Stigmator
from temsim.optics.condenser_stigmator import CondenserStigmatorComponent
from temsim.optics.objective_stigmator import ObjectiveStigmatorComponent
from temsim.optics.diffraction_stigmator import DiffractionStigmatorComponent
from temsim.optics.quadrupole import QuadrupoleComponent
from temsim.optics.hexapole import HexapoleComponent
from temsim.optics.electron_gun.alignment import GunDeflector, GunStigmator


def _matches(instance, methods):
    for name, original in methods:
        value = getattr(instance, name, None)
        if getattr(value, '__func__', value) is not original:
            return False
    return True


def _contract(cls, names):
    return tuple((name, getattr(cls, name)) for name in names)


_LENSES = {}
for family, base, concrete in (
    ('condenser', CondenserLensComponent, (CondenserLensComponent,)),
    ('round', RoundLensComponent, (RoundLensComponent, Tl22LensComponent, Tl21LensComponent,
        Tl12LensComponent, MiniCondenserComponent, ImageCorrectorOlPostLensComponent,
        ImageCorrectorTl11LensComponent, ImageCorrectorTl12LensComponent, ImageCorrectorTl21LensComponent,
        ImageCorrectorTl22LensComponent, ImageCorrectorAdapterLensComponent)),
    ('round', AdapterLensComponent, (AdapterLensComponent,)),
    ('normalized_round', AnchoredRoundLensComponent, (AnchoredRoundLensComponent,
        IntermediateLensComponent, ProjectorLensP1Component, ProjectorLensP2Component)),
    ('normalized_round', DiffractionLensComponent, (DiffractionLensComponent,)),
    ('objective', ObjectiveLensComponent, (ObjectiveLensComponent,)),
):
    names = ('magnetic_field_t', 'field_support_mm')
    if family == 'objective':
        names += ('unit_excitation_field_t', '_profile')
    methods = _contract(base, names)
    for cls in concrete:
        _LENSES[cls] = family, methods

_MULTIPOLES = {}
for base, concrete, names in (
    (Stigmator, (Stigmator, CondenserStigmatorComponent, ObjectiveStigmatorComponent,
                 DiffractionStigmatorComponent), ('quadrupole_tensor_m2',)),
    (QuadrupoleComponent, (QuadrupoleComponent, Qph2QuadrupoleComponent, QpcQuadrupoleComponent,
        Qph1QuadrupoleComponent, QpolQuadrupoleComponent, ImageCorrectorQpolQuadrupoleComponent,
        ImageCorrectorDstgQuadrupoleComponent), ('quadrupole_strength_m2',)),
    (HexapoleComponent, (HexapoleComponent, Hp2HexapoleComponent, HpcHexapoleComponent,
        Hp1HexapoleComponent, HpolHexapoleComponent, ImageCorrectorHpolHexapoleComponent,
        ImageCorrectorHp1HexapoleComponent, ImageCorrectorHp2HexapoleComponent),
        ('hexapole_strength_m3', 'hexapole_strength_components_m3')),
):
    for cls in concrete:
        _MULTIPOLES[cls] = _contract(base, names)

_PROVIDERS = {
    cls: _contract(cls, ('field_at_global_positions_t',))
    for cls in (GeometryAwareAnalyticFieldProvider, _EquivalentDeflectorField, _MultipoleField,
                GunDeflector, GunStigmator)
}
_PROVIDERS[GeometryAwareAnalyticFieldProvider] += _contract(
    GeometryAwareAnalyticFieldProvider, ('magnetic_field_t', 'field_support_mm'))


def lens_family(native):
    contract = _LENSES.get(type(native))
    return contract[0] if contract is not None and _matches(native, contract[1]) else None


def multipole_is_supported(component):
    contract = _MULTIPOLES.get(type(component))
    return contract is not None and _matches(component, contract)


def magnetic_provider_is_supported(provider):
    contract = _PROVIDERS.get(type(provider))
    return contract is not None and _matches(provider, contract)
