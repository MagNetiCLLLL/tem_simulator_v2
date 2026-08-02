"""Catalog-backed selection of complete TEM assemblies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.optics.corrector_structure import ensure_corrector_structure
from temsim.paths import INSTRUMENT_CONFIG_ROOT


@dataclass(frozen=True, slots=True)
class AssemblyOption:
    name: str
    file: str
    properties: dict[str, object]


@dataclass(frozen=True, slots=True)
class AssemblySelection:
    gun: str
    column: str
    recording: str


class AssemblyCatalog:
    def __init__(self, root: Path = INSTRUMENT_CONFIG_ROOT) -> None:
        self.root = Path(root).resolve()
        with (self.root / "catalog.toml").open("rb") as stream:
            document = tomllib.load(stream)
        self.guns = self._options(document["gun_variants"])
        self.columns = self._options(document["column_variants"])
        self.recording_systems = self._options(
            document["project_and_recording_system_variants"]
        )

    @staticmethod
    def _options(entries) -> tuple[AssemblyOption, ...]:
        return tuple(
            AssemblyOption(
                name=str(entry["name"]),
                file=str(entry["file"]),
                properties={
                    key: value
                    for key, value in entry.items()
                    if key not in {"name", "file"}
                },
            )
            for entry in entries
        )

    @staticmethod
    def _by_name(options, name: str) -> AssemblyOption:
        try:
            return next(option for option in options if option.name == name)
        except StopIteration as exc:
            raise ValueError(f"Unknown assembly option: {name}") from exc

    def default_selection(self) -> AssemblySelection:
        return AssemblySelection(
            gun="FEG",
            column="C3 + Probe Corrector",
            recording="Energy Filter",
        )

    def selected_paths(self, selection: AssemblySelection) -> dict[str, str]:
        return {
            "gun": self._by_name(self.guns, selection.gun).file,
            "column": self._by_name(self.columns, selection.column).file,
            "project_and_recording_system": self._by_name(
                self.recording_systems, selection.recording
            ).file,
        }

    def apply(self, state, selection: AssemblySelection):
        gun = self._by_name(self.guns, selection.gun)
        column = self._by_name(self.columns, selection.column)
        recording = self._by_name(self.recording_systems, selection.recording)

        gun_type = (
            "thermionic"
            if gun.properties["electron_gun"] == "Thermionic"
            else "cold_feg"
        )
        if state.electron_gun.type_key != gun_type:
            state.select_electron_gun(gun_type)
        state.monochromator_installed = bool(
            gun.properties.get("monochromator", False)
        )

        has_c3 = bool(column.properties["c3_lens"])
        has_probe = bool(column.properties["probe_corrector"])
        has_image = bool(column.properties["image_corrector"])
        if has_probe and has_image:
            corrector_mode = "double_corrector"
        elif has_probe:
            corrector_mode = "probe_corrector"
        elif has_image:
            corrector_mode = "image_corrector"
        else:
            corrector_mode = "no_corrector"
        state.corrector_mode = corrector_mode
        state.probe_corrector_installed = has_probe
        state.image_corrector_installed = has_image
        state.layout_c3_hardware = "three_condenser" if has_c3 else "two_condenser"
        state.layout_c3_excited = has_c3
        state.column_mode = "three_lens" if has_c3 else "two_lens_c3_off"

        has_filter = bool(recording.properties["energy_filter"])
        state.energy_filter_mode = "energy_filter" if has_filter else "no_energy_filter"
        state.energy_filter_installed = has_filter
        ensure_corrector_structure(state)
        apply_physical_layout_to_state(
            state, preserve_operating_parameters=False
        )
        return state._resolved_assembly
