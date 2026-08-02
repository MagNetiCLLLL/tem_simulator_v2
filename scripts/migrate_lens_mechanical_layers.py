"""Split every magnetic-lens envelope into mechanical-only radial layers."""

from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).parents[1]
INSTRUMENT_ROOT = ROOT / "configs" / "instruments"


def _part_span(lines: list[str], key: str) -> tuple[int, int]:
    starts = [
        index for index, line in enumerate(lines)
        if line.strip() == "[[parts]]"
    ]
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        if any(
            line.strip() == f'key = "{key}"'
            for line in lines[start:end]
        ):
            return start, end
    raise KeyError(key)


def _number(value: float) -> str:
    return f"{float(value):.12g}"


def _child_block(
    *,
    order: int,
    lens: dict,
    lens_key: str,
    suffix: str,
    label: str,
    profile: str,
    material: str,
    start: float,
    end: float,
    inner: float,
    outer: float,
    overlap_group: str,
    newline: str,
) -> str:
    center = 0.5 * (start + end)
    name = str(lens["name"])
    if name.endswith(" Assembly"):
        name = name[:-9]
    values = [
        "[[parts]]",
        f"order = {order}",
        f'key = "{lens_key}_{suffix}"',
        f'name = "{name} {label}"',
        f'branch = "{lens["branch"]}"',
        f"local_start_z_mm = {_number(start)}",
        f"local_center_z_mm = {_number(center)}",
        f"local_end_z_mm = {_number(end)}",
        f"length_mm = {_number(end - start)}",
        (
            "vacuum_inner_diameter_mm = "
            f'{_number(lens["vacuum_inner_diameter_mm"])}'
        ),
        f"mechanical_inner_diameter_mm = {_number(inner)}",
        f"mechanical_outer_diameter_mm = {_number(outer)}",
        f'mechanical_profile = "{profile}"',
        f'mechanical_part_role = "{suffix}"',
        f'material_class = "{material}"',
        "mechanical_only = true",
        f'mechanical_overlap_group = "{overlap_group}"',
        'mechanical_overlap_role = "member"',
        (
            'mechanical_overlap_reason = "Concentric independent mechanical '
            f'layer inside the {name} assembly."'
        ),
        f'parent_key = "{lens_key}"',
    ]
    if suffix == "excitation_coil":
        values.append(f'field_source_key = "{lens_key}"')
    values.append("")
    return newline.join(values)


def migrate(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    document = tomllib.loads(text)
    parts = document.get("parts", ())
    by_key = {str(part["key"]): part for part in parts}
    lens_keys = sorted({
        str(part["parent_key"])
        for part in parts
        if str(part.get("key", "")).endswith("_pole")
        and part.get("parent_key") in by_key
    })
    if not lens_keys:
        return False

    newline = "\r\n" if "\r\n" in text else "\n"
    changed = False
    lines = text.splitlines(keepends=True)
    pole_keys = sorted(
        str(part["key"])
        for part in parts
        if str(part.get("key", "")).endswith("_pole")
        and part.get("parent_key") in lens_keys
    )
    for pole_key in pole_keys:
        start, end = _part_span(lines, pole_key)
        profile_line = next((
            index for index in range(start, end)
            if lines[index].startswith("mechanical_profile = ")
        ), None)
        canonical = f'mechanical_profile = "magnetic_pole_piece"{newline}'
        if profile_line is None:
            lines[end:end] = [canonical]
            changed = True
        elif lines[profile_line] != canonical:
            lines[profile_line] = canonical
            changed = True
    for lens_key in lens_keys:
        start, end = _part_span(lines, lens_key)
        block = "".join(lines[start:end])
        additions = []
        if "mechanical_profile = " not in block:
            additions.append(
                f'mechanical_profile = "magnetic_lens_assembly"{newline}'
            )
        if "mechanical_part_role = " not in block:
            additions.append(
                f'mechanical_part_role = "optical_parent"{newline}'
            )
        if additions:
            lines[end:end] = additions
            changed = True
    text = "".join(lines)

    document = tomllib.loads(text)
    parts = document["parts"]
    by_key = {str(part["key"]): part for part in parts}
    next_order = max(int(part["order"]) for part in parts) + 1
    appended = []
    for lens_key in lens_keys:
        lens = by_key[lens_key]
        child_keys = (
            f"{lens_key}_housing",
            f"{lens_key}_yoke",
            f"{lens_key}_excitation_coil",
        )
        if all(key in by_key for key in child_keys):
            continue
        if any(key in by_key for key in child_keys):
            raise ValueError(f"Incomplete prior lens split for {lens_key} in {path}")
        lens_start = float(lens["local_start_z_mm"])
        lens_center = float(lens["local_center_z_mm"])
        lens_end = float(lens["local_end_z_mm"])
        lens_length = lens_end - lens_start
        parent_outer = float(lens["mechanical_outer_diameter_mm"])
        overlap_group = str(lens.get(
            "mechanical_overlap_group",
            f"{lens_key}_assembly",
        ))
        layers = (
            (
                "housing", "Housing", "magnetic_lens_housing",
                "non_magnetic_structural", lens_start, lens_end,
                0.94 * parent_outer, parent_outer,
            ),
            (
                "yoke", "Magnetic Yoke", "magnetic_lens_yoke",
                "soft_magnetic", lens_start, lens_end,
                0.76 * parent_outer, 0.92 * parent_outer,
            ),
            (
                "excitation_coil", "Excitation Coil",
                "magnetic_excitation_coil", "insulated_copper_winding",
                lens_center - 0.25 * lens_length,
                lens_center + 0.25 * lens_length,
                0.69 * parent_outer, 0.74 * parent_outer,
            ),
        )
        for (
            suffix, label, profile, material, start, end, inner, outer,
        ) in layers:
            appended.append(_child_block(
                order=next_order,
                lens=lens,
                lens_key=lens_key,
                suffix=suffix,
                label=label,
                profile=profile,
                material=material,
                start=start,
                end=end,
                inner=inner,
                outer=outer,
                overlap_group=overlap_group,
                newline=newline,
            ))
            next_order += 1
        changed = True
    if changed:
        if appended:
            text = text.rstrip() + newline * 2 + newline.join(appended).rstrip()
            text += newline
        path.write_text(text, encoding="utf-8", newline="")
    return changed


if __name__ == "__main__":
    for manifest in sorted(INSTRUMENT_ROOT.rglob("*.toml")):
        if migrate(manifest):
            print(manifest.relative_to(ROOT))
