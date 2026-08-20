"""Catalog-backed selection of complete TEM assemblies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

from temsim import module_manifest
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
        self._validate_document(document)
        self.guns = self._options(document["gun_variants"])
        self.columns = self._options(document["column_variants"])
        self._recording_system_modules = self._options(
            document["project_and_recording_system_variants"]
        )
        self.recording_systems = tuple(
            option
            for option in self._recording_system_modules
            if bool(option.properties.get("selectable", True))
        )

    def _validate_document(self, document) -> None:
        if int(document.get("format_version", 0)) != 1:
            raise ValueError("Unsupported instrument-catalog format")
        if document.get("coordinate_system") != "module_local_z_mm":
            raise ValueError("Invalid instrument-catalog coordinate system")

        expected = (
            ("gun", "gun_variants"),
            ("column", "column_variants"),
            (
                "project_and_recording_system",
                "project_and_recording_system_variants",
            ),
        )
        order = tuple(document.get("assembly", {}).get("order", ()))
        expected_order = tuple(module_type for module_type, _ in expected)
        if order != expected_order:
            raise ValueError(
                "Instrument catalog assembly order must be exactly "
                f"{expected_order}, found {order}"
            )

        selected_files: list[str] = []
        module_keys: list[str] = []
        for module_type, group in expected:
            entries = tuple(document.get(group, ()))
            if not entries:
                raise ValueError(f"Instrument catalog has no {group}")
            names = [str(entry["name"]) for entry in entries]
            files = [Path(str(entry["file"])).as_posix() for entry in entries]
            signatures = [
                tuple(sorted(
                    (str(key), repr(value))
                    for key, value in entry.items()
                    if key not in {"name", "file"}
                ))
                for entry in entries
            ]
            for label, values in (
                ("name", names),
                ("file", files),
                ("selection properties", signatures),
            ):
                if len(set(values)) != len(values):
                    raise ValueError(
                        f"Duplicate {label} in instrument catalog {group}"
                    )
            for relative in files:
                path = (self.root / relative).resolve()
                if not path.is_relative_to(self.root):
                    raise ValueError(
                        f"Instrument module escapes catalog root: {relative}"
                    )
                if not path.is_file():
                    raise ValueError(
                        f"Instrument catalog module does not exist: {relative}"
                    )
                module_document = module_manifest.read_document(path)
                module_manifest.validate_document(module_document)
                actual_type = str(module_document["module"]["type"])
                if actual_type != module_type:
                    raise ValueError(
                        f"Instrument module {relative} has type "
                        f"{actual_type!r}, expected {module_type!r}"
                    )
                module_keys.append(str(module_document["module"]["key"]))
            selected_files.extend(files)

        if len(set(selected_files)) != len(selected_files):
            raise ValueError("Instrument module file is listed more than once")
        if len(set(module_keys)) != len(module_keys):
            raise ValueError("Instrument module key is defined more than once")

        selectable_recordings = tuple(
            entry
            for entry in document["project_and_recording_system_variants"]
            if bool(entry.get("selectable", True))
        )
        if any(
            not isinstance(entry.get("selectable", True), bool)
            for entry in document["project_and_recording_system_variants"]
        ):
            raise ValueError(
                "Instrument recording-system selectable flags must be Boolean"
            )
        if len(selectable_recordings) != 1:
            raise ValueError(
                "Instrument catalog must expose exactly one installed "
                "recording system"
            )
        if not bool(selectable_recordings[0].get("energy_filter", False)):
            raise ValueError(
                "The installed recording system must include an Energy Filter"
            )
        disk_files = {
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*.toml")
            if path.name != "catalog.toml"
        }
        catalog_files = set(selected_files)
        if disk_files != catalog_files:
            raise ValueError(
                "Instrument TOML/catalog mismatch: "
                f"unlisted={sorted(disk_files - catalog_files)}, "
                f"missing={sorted(catalog_files - disk_files)}"
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
            recording=self.recording_systems[0].name,
        )

    def normalise_selection(
        self, selection: AssemblySelection
    ) -> AssemblySelection:
        """Return a selection using the permanently installed filter module."""

        # Accept known legacy selections so saved profiles migrate cleanly,
        # while retaining strict validation for malformed/unknown names.
        self._by_name(self._recording_system_modules, selection.recording)
        return AssemblySelection(
            gun=selection.gun,
            column=selection.column,
            recording=self.recording_systems[0].name,
        )

    def selected_paths(self, selection: AssemblySelection) -> dict[str, str]:
        selection = self.normalise_selection(selection)
        return {
            "gun": self._by_name(self.guns, selection.gun).file,
            "column": self._by_name(self.columns, selection.column).file,
            "project_and_recording_system": self._by_name(
                self.recording_systems, selection.recording
            ).file,
        }

    def apply(self, state, selection: AssemblySelection):
        selection = self.normalise_selection(selection)
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
        from temsim.optics.energy_filter import ensure_energy_filter
        ensure_energy_filter(state)
        state.energy_filter.enabled = has_filter
        ensure_corrector_structure(state)
        apply_physical_layout_to_state(
            state,
            preserve_operating_parameters=False,
            assembly_root=self.root,
        )
        from temsim.component_keys import (
            BRIGHT_FIELD_DETECTOR,
            CAMERA,
            FLUORESCENT_SCREEN,
        )
        from temsim.detector.recording_system import ensure_recording_system
        ensure_recording_system(state)
        for plane in state.recording_planes:
            if plane.key in {
                FLUORESCENT_SCREEN,
                BRIGHT_FIELD_DETECTOR,
                CAMERA,
            }:
                # These solid on-axis recording surfaces must retract before
                # rays can enter the post-column Energy Filter branch.
                plane.inserted = not has_filter
        state.condenser_aperture_3.radius_mm = (
            0.05
            if state.monochromator_installed
            else state.condenser_aperture_3.maximum_radius_mm
        )
        return state._resolved_assembly
