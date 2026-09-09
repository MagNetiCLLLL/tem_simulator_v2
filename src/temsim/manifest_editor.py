"""Safe generic editing and anchor auditing for module TOMLs."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Real
from pathlib import Path
from types import SimpleNamespace
import tomllib

from temsim import module_manifest
from temsim.column.module_assembly import resolve_module_assembly


@dataclass(frozen=True, slots=True)
class ManifestField:
    path: tuple[str, ...]
    label: str
    value: object
    editable: bool = True
    meaning: object | None = None


@dataclass(frozen=True, slots=True)
class ManifestTarget:
    module_path: str
    part_key: str | None = None


@dataclass(frozen=True, slots=True)
class AnchorRecord:
    module_key: str
    part_key: str
    name: str
    anchor: str
    start_z_mm: float
    center_z_mm: float
    end_z_mm: float
    optical_references_mm: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CatalogAudit:
    module_count: int
    part_definition_count: int
    logical_part_key_count: int
    variant_scoped_duplicate_count: int
    assembly_count: int
    resolved_part_authority_count: int


STRUCTURAL_READ_ONLY_FIELDS = frozenset({"key"})


def format_toml_value(value: object) -> str:
    return module_manifest._format_toml_value(value)


def parse_toml_value(text: str) -> object:
    try:
        return tomllib.loads(f"value = {text}\n")["value"]
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML value: {text}") from exc


def _validated_part_length(length):
    if isinstance(length, bool) or not isinstance(length, Real):
        raise ValueError("Part length_mm must be a finite non-negative number")
    try:
        length = float(length)
    except OverflowError as exc:
        raise ValueError("Part length_mm must be a finite non-negative number") from exc
    if not math.isfinite(length) or length < 0.0:
        raise ValueError("Part length_mm must be a finite non-negative number")
    return length


def resized_part_axial_coordinates(part, length, *, center_z_mm=None):
    """Resize one envelope about its fixed centre, preserving any asymmetry."""

    length = _validated_part_length(length)
    start = float(part["local_start_z_mm"])
    center = float(part["local_center_z_mm"])
    end = float(part["local_end_z_mm"])
    if not all(math.isfinite(value) for value in (start, center, end)) or not start <= center <= end:
        raise ValueError("Part axial coordinates must be finite with start <= center <= end")
    fraction = (center - start) / (end - start) if end > start else 0.5
    pivot = center if center_z_mm is None else float(center_z_mm)
    resized = {
        "local_start_z_mm": pivot - fraction * length,
        "local_end_z_mm": pivot + (1.0 - fraction) * length,
    }
    if not all(math.isfinite(value) for value in resized.values()):
        raise ValueError("Resized part axial coordinates must be finite")
    return resized


def _complete_part_length_updates(document, updates):
    """Supply redundant endpoints for a length edit; explicit endpoints win."""

    completed = dict(updates)
    parts = {str(part["key"]): part for part in document.get("parts", ())}
    for path, length in updates.items():
        if len(path) != 3 or path[0] != "parts" or path[2] != "length_mm":
            continue
        # Explicit endpoints suppress inference, never value validation. A
        # quoted number/bool must not slip through after an earlier valid edit.
        _validated_part_length(length)
        prefix = path[:2]
        if any(prefix + (field,) in updates for field in ("local_start_z_mm", "local_end_z_mm")):
            continue
        if path[1] not in parts:
            raise ValueError(f"Missing TOML part {path[1]!r}")
        endpoints = resized_part_axial_coordinates(
            parts[path[1]], length,
            center_z_mm=updates.get(prefix + ("local_center_z_mm",)),
        )
        completed.update({prefix + (field,): value for field, value in endpoints.items()})
    return completed


class ManifestEditor:
    def __init__(self, root: Path = module_manifest.MODULE_ROOT) -> None:
        self.root = Path(root).resolve()

    def fields(self, target: ManifestTarget) -> tuple[ManifestField, ...]:
        from temsim.parameter_semantics import describe_parameter

        document = module_manifest.read_document(self.root / target.module_path)
        by_key = {str(part["key"]): part for part in document.get("parts", ())}
        if target.part_key is not None:
            part = next(
                part
                for part in document["parts"]
                if str(part["key"]) == target.part_key
            )
            return tuple(
                ManifestField(
                    path=("parts", target.part_key, str(field)),
                    label=str(field),
                    value=value,
                    editable=str(field) not in STRUCTURAL_READ_ONLY_FIELDS,
                    meaning=describe_parameter(part, ("parts", target.part_key, str(field)), by_key=by_key),
                )
                for field, value in part.items()
            )

        fields: list[ManifestField] = []
        for section_name in ("module", "geometry"):
            for field, value in document.get(section_name, {}).items():
                fields.append(ManifestField(
                    path=(section_name, str(field)),
                    label=f"{section_name}.{field}",
                    value=value,
                    editable=str(field) not in STRUCTURAL_READ_ONLY_FIELDS,
                    meaning=describe_parameter(document, (section_name, str(field)), by_key=by_key),
                ))
        for port_name, port in document.get("ports", {}).items():
            for field, value in port.items():
                fields.append(ManifestField(
                    path=("ports", str(port_name), str(field)),
                    label=f"ports.{port_name}.{field}",
                    value=value,
                    editable=str(field) != "interface",
                    meaning=describe_parameter(document, ("ports", str(port_name), str(field)), by_key=by_key),
                ))
        return tuple(fields)

    def save(self, target: ManifestTarget, updates: dict[tuple[str, ...], object], configuration):
        if not updates:
            return
        from temsim.component_operations import PartChangeSet

        if isinstance(updates, PartChangeSet):
            from temsim.component_persistence import save_component_changes

            document = module_manifest.read_document(self.root / target.module_path)
            completed = _complete_part_length_updates(document, updates)
            return save_component_changes(
                self.root, target.module_path, replace(updates, fields=completed), configuration,
            )
        document = module_manifest.read_document(self.root / target.module_path)
        updates = _complete_part_length_updates(document, updates)
        originals = module_manifest.update_manifest_values(
            {target.module_path: updates}, root=self.root
        )
        try:
            self.validate_catalog()
            resolve_module_assembly(configuration, root=self.root)
        except Exception:
            module_manifest.restore_manifest_texts(originals, root=self.root)
            raise
        return originals

    def validate_catalog(self) -> CatalogAudit:
        from temsim.assembly_catalog import AssemblyCatalog

        # Validate catalog names, selection signatures, module paths/types,
        # and the one-to-one catalog/disk file set before enumerating builds.
        AssemblyCatalog(self.root)
        with (self.root / "catalog.toml").open("rb") as stream:
            catalog = tomllib.load(stream)
        module_paths = {
            str(entry["file"])
            for group in (
                "gun_variants",
                "beam_blanker_variants",
                "column_variants",
                "project_and_recording_system_variants",
            )
            for entry in catalog.get(group, ())
        }
        part_count = 0
        logical_part_keys = set()
        for module_path in module_paths:
            document = module_manifest.read_document(self.root / module_path)
            module_manifest.validate_document(document)
            part_count += len(document.get("parts", ()))
            logical_part_keys.update(
                str(part["key"])
                for part in document.get("parts", ())
            )

        assembly_count = 0
        resolved_part_authority_count = 0
        for gun in catalog["gun_variants"]:
            for column in catalog["column_variants"]:
                for recording in catalog[
                    "project_and_recording_system_variants"
                ]:
                    if not bool(recording.get("selectable", True)):
                        continue
                    if column["probe_corrector"] and column["image_corrector"]:
                        corrector = "double_corrector"
                    elif column["probe_corrector"]:
                        corrector = "probe_corrector"
                    elif column["image_corrector"]:
                        corrector = "image_corrector"
                    else:
                        corrector = "no_corrector"
                    configuration = SimpleNamespace(
                        electron_gun_type=(
                            "thermionic"
                            if gun["electron_gun"] == "Thermionic"
                            else "cold_feg"
                        ),
                        monochromator_installed=bool(gun["monochromator"]),
                        gun_components=(),
                        corrector=SimpleNamespace(value=corrector),
                        c3_hardware=SimpleNamespace(value=(
                            "three_condenser"
                            if column["c3_lens"] else "two_condenser"
                        )),
                        energy_filter_selected=bool(recording["energy_filter"]),
                    )
                    blanker_states = (
                        (False, True)
                        if catalog.get("beam_blanker_variants") else (False,)
                    )
                    for installed in blanker_states:
                        configuration.nanopulser_installed = installed
                        assembly = resolve_module_assembly(
                            configuration, root=self.root
                        )
                        authorities = assembly.part_authorities
                        if len(authorities) != len(assembly.parts):
                            raise ValueError(
                                "Resolved assembly has duplicate active part keys"
                            )
                        if len(set(authorities.values())) != len(assembly.parts):
                            raise ValueError(
                                "Resolved assembly reuses one TOML part definition"
                            )
                        resolved_part_authority_count += len(assembly.parts)
                        assembly_count += 1
        return CatalogAudit(
            module_count=len(module_paths),
            part_definition_count=part_count,
            logical_part_key_count=len(logical_part_keys),
            variant_scoped_duplicate_count=(
                part_count - len(logical_part_keys)
            ),
            assembly_count=assembly_count,
            resolved_part_authority_count=resolved_part_authority_count,
        )

    @staticmethod
    def anchor_records(assembly) -> tuple[AnchorRecord, ...]:
        records: list[AnchorRecord] = []
        for module in assembly.modules:
            module_parts = sorted(
                (
                    part for part in assembly.parts
                    if part.module_key == module.key
                ),
                key=lambda part: int(part.data["order"]),
            )
            previous_key = f"{module.key}:entrance"
            for part in module_parts:
                anchor = part.parent_key or previous_key
                references: list[float] = []
                local_center = float(part.data["local_center_z_mm"])
                for field, value in part.data.items():
                    if field == "optical_reference_local_z_mm":
                        references.append(
                            part.center_z_mm + float(value) - local_center
                        )
                    elif field == "interaction_centers_local_z_mm":
                        references.extend(
                            part.center_z_mm + float(item) - local_center
                            for item in value
                        )
                    elif field.endswith("_field_reference_local_z_mm"):
                        references.append(
                            part.center_z_mm + float(value) - local_center
                        )
                    elif field == "virtual_reference_local_z_mm":
                        references.append(
                            part.center_z_mm + float(value) - local_center
                        )
                records.append(AnchorRecord(
                    module_key=part.module_key,
                    part_key=part.key,
                    name=part.name,
                    anchor=str(anchor),
                    start_z_mm=float(part.start_z_mm),
                    center_z_mm=float(part.center_z_mm),
                    end_z_mm=float(part.end_z_mm),
                    optical_references_mm=tuple(references),
                ))
                if (
                    part.key.startswith("condenser_lens_")
                    and part.key.endswith("_pole")
                    and part.parent_key
                ):
                    # Pole pieces are children of a lens assembly and must not
                    # silently become the axial anchor for the next assembly.
                    previous_key = part.parent_key
                else:
                    previous_key = part.key
        return tuple(records)
