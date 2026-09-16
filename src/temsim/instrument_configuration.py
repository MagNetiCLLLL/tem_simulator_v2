"""Unit-level instrument choices resolved through the existing TOML assembler.

Legacy template names remain serialization adapters, not operator presets.
Checking geometry neither propagates electrons nor optimizes lens strengths.
"""
from dataclasses import dataclass, fields

from temsim.assembly_catalog import AssemblySelection
from temsim.instrument_snapshot import capture_instrument_snapshot


@dataclass(frozen=True)
class InstrumentUnits:
    source: str = "cold_feg"
    monochromator: bool = False
    beam_blanker: bool = False
    c3_lens: bool = True
    probe_corrector: bool = True
    image_corrector: bool = False
    energy_filter: bool = True

    def selection(self, catalog):
        if self.source not in {"cold_feg", "thermionic"}:
            raise ValueError("Select Cold FEG or Thermionic")
        for field in fields(self):
            if field.name != "source" and type(getattr(self, field.name)) is not bool:
                raise ValueError(f"{field.name} must be an explicit on/off choice")
        if self.monochromator and self.source != "cold_feg":
            raise ValueError("Monochromator requires Cold FEG")
        if not self.c3_lens and (self.probe_corrector or self.image_corrector):
            raise ValueError("Probe and image correctors require C3 lens")

        def match(options, **properties):
            found = [o for o in options if all(o.properties.get(k) == v for k, v in properties.items())]
            if len(found) != 1:
                raise ValueError(f"No unique supported assembly for {properties}")
            return found[0].name

        return catalog.normalise_selection(AssemblySelection(
            gun=match(catalog.guns, electron_gun="FEG" if self.source == "cold_feg" else "Thermionic",
                      monochromator=self.monochromator),
            column=match(catalog.columns, c3_lens=self.c3_lens,
                         probe_corrector=self.probe_corrector, image_corrector=self.image_corrector),
            recording=match(catalog.recording_systems, energy_filter=self.energy_filter),
            beam_blanker=match(catalog.beam_blankers, installed=self.beam_blanker)))

    @classmethod
    def from_selection(cls, catalog, selection):
        selection = catalog.normalise_selection(selection)
        gun = catalog._by_name(catalog.guns, selection.gun).properties
        column = catalog._by_name(catalog.columns, selection.column).properties
        recording = catalog._by_name(catalog.recording_systems, selection.recording).properties
        return cls(source="cold_feg" if gun["electron_gun"] == "FEG" else "thermionic",
                   monochromator=bool(gun["monochromator"]),
                   beam_blanker=selection.beam_blanker != "None",
                   c3_lens=bool(column["c3_lens"]), probe_corrector=bool(column["probe_corrector"]),
                   image_corrector=bool(column["image_corrector"]), energy_filter=bool(recording["energy_filter"]))


@dataclass(frozen=True)
class CheckedInstrumentAssembly:
    units: InstrumentUnits
    selection: AssemblySelection
    original: object
    candidate: object

    def restore_for(self, state):
        if capture_instrument_snapshot(state).physical_digest != self.original.physical_digest:
            raise ValueError("Instrument settings changed. Check the assembly again.")
        # Restore also verifies consumed files and solver identity. A changed
        # definition cannot be applied under a previous Check result.
        return self.candidate.restore()


def check_instrument_configuration(state, catalog, units):
    selection = units.selection(catalog)
    original = capture_instrument_snapshot(state)
    candidate = original.restore()
    current = catalog.selection_for_resolved(state._resolved_assembly)
    if selection != current:
        catalog.apply(candidate, selection, preserve_operating_parameters=True)
    # Preserve exact settings on an unchanged selection; do not reset a live
    # tip, lens or aperture merely by opening/checking the configuration window.
    candidate.electron_gun.validate()
    return CheckedInstrumentAssembly(units, selection, original, capture_instrument_snapshot(candidate))


def unit_for_component(assembly, key):
    """Mechanical children inherit the unit of their optical parent."""
    from temsim.component_keys import IMAGE_CORRECTOR_KEYS, PROBE_CORRECTOR_KEYS
    parts = {p.key: p for p in assembly.parts}
    visited = set()
    while key in parts and key not in visited:
        visited.add(key)
        part = parts[key]
        if key.startswith("feg_monochromator"):
            return "monochromator"
        if key == "condenser_lens_3":
            return "c3_lens"
        if key in PROBE_CORRECTOR_KEYS:
            return "probe_corrector"
        if key in IMAGE_CORRECTOR_KEYS:
            return "image_corrector"
        if key.startswith("energy_filter") or part.branch == "energy_filter":
            return "energy_filter"
        if part.parent_key:
            key = part.parent_key
            continue
        module_type = next((m.type for m in assembly.modules if m.key == part.module_key), "")
        return {"gun": "source", "beam_blanker": "beam_blanker"}.get(module_type)
    return None
