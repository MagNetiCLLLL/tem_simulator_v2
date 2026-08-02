"""Add explicit two-pole mechanical children to round magnetic lenses."""

from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).parents[1]
COLUMN_ROOT = ROOT / "configs" / "instruments" / "column"

LENS_DEFAULTS = {
    "mini_condenser": (210.0, 20.0, 16.0),
    "adapter_lens": (180.0, 20.0, 14.0),
    "probe_tl22_lens": (180.0, 20.0, 12.0),
    "probe_tl21_lens": (180.0, 20.0, 12.0),
    "probe_tl12_lens": (180.0, 20.0, 14.0),
    "image_ol_post_lens": (280.0, 20.0, 20.0),
    "image_tl11_lens": (200.0, 13.0, 20.0),
    "image_tl21_lens": (170.0, 13.0, 20.0),
    "image_tl22_lens": (170.0, 13.0, 20.0),
    "image_adapter_lens": (160.0, 13.0, 20.0),
}


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


def _pole_block(
    *, order: int, lens: dict, lens_key: str, side: str,
    outer: float, bore: float, gap: float, overlap_group: str, newline: str,
) -> str:
    start = float(lens["local_start_z_mm"])
    center = float(lens["local_center_z_mm"])
    end = float(lens["local_end_z_mm"])
    if side == "upper":
        pole_start, pole_end = start, center - 0.5 * gap
    else:
        pole_start, pole_end = center + 0.5 * gap, end
    pole_center = 0.5 * (pole_start + pole_end)
    pole_length = pole_end - pole_start
    title = "Upper" if side == "upper" else "Lower"
    values = (
        "[[parts]]",
        f"order = {order}",
        f'key = "{lens_key}_{side}_pole"',
        f'name = "{lens["name"]} {title} Pole Piece"',
        f'branch = "{lens["branch"]}"',
        f"local_start_z_mm = {pole_start:g}",
        f"local_center_z_mm = {pole_center:g}",
        f"local_end_z_mm = {pole_end:g}",
        f"length_mm = {pole_length:g}",
        f'vacuum_inner_diameter_mm = {float(lens["vacuum_inner_diameter_mm"]):g}',
        f"mechanical_outer_diameter_mm = {0.67 * outer:g}",
        f"mechanical_tip_diameter_mm = {2.0 * bore:g}",
        f"mechanical_bore_diameter_mm = {bore:g}",
        'mechanical_profile = "cylindrical_shank_frustum_tip"',
        f'mechanical_overlap_group = "{overlap_group}"',
        'mechanical_overlap_role = "member"',
        f'mechanical_overlap_reason = "Independent pole piece nested inside the {lens["name"]} assembly."',
        f'parent_key = "{lens_key}"',
        "",
    )
    return newline.join(values)


def migrate(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    changed = False
    next_order = max(
        int(part["order"])
        for part in tomllib.loads(text)["parts"]
    ) + 1
    appended: list[str] = []
    for lens_key, defaults in LENS_DEFAULTS.items():
        document = tomllib.loads(text)
        by_key = {str(part["key"]): part for part in document["parts"]}
        lens = by_key.get(lens_key)
        if lens is None:
            continue
        overlap_group = str(lens.get(
            "mechanical_overlap_group",
            f"{lens_key}_assembly",
        ))
        if f"{lens_key}_upper_pole" in by_key:
            lines = text.splitlines(keepends=True)
            normalised = False
            for side in ("upper", "lower"):
                start, end = _part_span(lines, f"{lens_key}_{side}_pole")
                for index in range(start, end):
                    if lines[index].startswith("mechanical_overlap_group = "):
                        replacement = (
                            f'mechanical_overlap_group = "{overlap_group}"'
                            f"{newline}"
                        )
                        if lines[index] != replacement:
                            lines[index] = replacement
                            normalised = True
                        break
            if normalised:
                text = "".join(lines)
                changed = True
            continue
        outer = float(lens.get("mechanical_outer_diameter_mm", defaults[0]))
        bore = float(lens.get(
            "bore_diameter_mm",
            lens.get("mechanical_clear_bore_diameter_mm", defaults[1]),
        ))
        gap = float(lens.get("pole_gap_mm", defaults[2]))
        lines = text.splitlines(keepends=True)
        _start, end = _part_span(lines, lens_key)
        additions = []
        for field, value in (
            ("mechanical_outer_diameter_mm", f"{outer:g}"),
            ("bore_diameter_mm", f"{bore:g}"),
            ("pole_gap_mm", f"{gap:g}"),
            ("pole_piece_topology", '"two_pole_single_gap"'),
        ):
            if field not in lens:
                additions.append(f"{field} = {value}{newline}")
        lines[end:end] = additions
        text = "".join(lines)
        for side in ("upper", "lower"):
            appended.append(_pole_block(
                order=next_order,
                lens=lens,
                lens_key=lens_key,
                side=side,
                outer=outer,
                bore=bore,
                gap=gap,
                overlap_group=overlap_group,
                newline=newline,
            ))
            next_order += 1
        changed = True
    if changed:
        text = text.rstrip() + newline * 2 + newline.join(appended).rstrip() + newline
        path.write_text(text, encoding="utf-8", newline="")
    return changed


if __name__ == "__main__":
    for manifest in sorted(COLUMN_ROOT.glob("*.toml")):
        if migrate(manifest):
            print(manifest.relative_to(ROOT))
