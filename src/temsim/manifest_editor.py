"""Safe generic editing and anchor auditing for module TOMLs."""

from __future__ import annotations

from dataclasses import dataclass
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
    assembly_count: int


STRUCTURAL_READ_ONLY_FIELDS = frozenset({"key"})


def format_toml_value(value: object) -> str:
    return module_manifest._format_toml_value(value)


def parse_toml_value(text: str) -> object:
    try:
        return tomllib.loads(f"value = {text}\n")["value"]
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML value: {text}") from exc


class ManifestEditor:
    def __init__(self, root: Path = module_manifest.MODULE_ROOT) -> None:
        self.root = Path(root).resolve()

    def fields(self, target: ManifestTarget) -> tuple[ManifestField, ...]:
        document = module_manifest.read_document(self.root / target.module_path)
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
                ))
        for port_name, port in document.get("ports", {}).items():
            for field, value in port.items():
                fields.append(ManifestField(
                    path=("ports", str(port_name), str(field)),
                    label=f"ports.{port_name}.{field}",
                    value=value,
                    editable=str(field) != "interface",
                ))
        return tuple(fields)

    def save(self, target: ManifestTarget, updates: dict[tuple[str, ...], object], configuration):
        if not updates:
            return
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
        with (self.root / "catalog.toml").open("rb") as stream:
            catalog = tomllib.load(stream)
        module_paths = {
            str(entry["file"])
            for group in (
                "gun_variants",
                "column_variants",
                "project_and_recording_system_variants",
            )
            for entry in catalog[group]
        }
        part_count = 0
        for module_path in module_paths:
            document = module_manifest.read_document(self.root / module_path)
            module_manifest.validate_document(document)
            part_count += len(document.get("parts", ()))

        assembly_count = 0
        for gun in catalog["gun_variants"]:
            for column in catalog["column_variants"]:
                for recording in catalog[
                    "project_and_recording_system_variants"
                ]:
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
                    resolve_module_assembly(configuration, root=self.root)
                    assembly_count += 1
        return CatalogAudit(
            module_count=len(module_paths),
            part_definition_count=part_count,
            assembly_count=assembly_count,
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
