"""Insert the current CEOS-style TL12/DPH1/DPH2 optical channels.

The existing component centres are deliberately left untouched.  TL12 shares
the existing DP12 module envelope, while DPH1 and DPH2 are alignment windings
nested in the corresponding principal-hexapole bodies.
"""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
TARGETS = (
    ROOT / "configs/instruments/column/C3_ImageCorrector.toml",
    ROOT
    / "configs/instruments/column/C3_ProbeCorrector_ImageCorrector.toml",
)


TL12_AND_DPH1 = '''[[parts]]
order = {tl12_order}
key = "image_tl12_lens"
name = "Image Corrector TL12"
branch = "image"
local_start_z_mm = {tl12_start}
local_center_z_mm = {dp12_center}
local_end_z_mm = {tl12_end}
length_mm = 1.0
vacuum_inner_diameter_mm = 5
optical_reference_local_z_mm = {dp12_center}
mechanical_outer_diameter_mm = 160
bore_diameter_mm = 13
pole_gap_mm = 1
pole_piece_topology = "single_effective_plane"
mechanical_profile = "integrated_magnetic_lens_channel"
mechanical_part_role = "optical_parent"
mechanical_overlap_group = "image_dp12_tl12_integrated_assembly"
mechanical_overlap_role = "member"
mechanical_overlap_reason = "TL12 is an optical thin-lens channel integrated in the existing DP12 module envelope; existing component centres remain unchanged."

[[parts]]
order = {dph1_order}
key = "image_dph1_deflector"
name = "Image Corrector DPH1 Deflector"
branch = "image"
local_start_z_mm = {dph1_start}
local_center_z_mm = {hp1_center}
local_end_z_mm = {dph1_end}
length_mm = 30.0
vacuum_inner_diameter_mm = 5
optical_reference_local_z_mm = {hp1_center}
parent_key = "image_hp1_hexapole"

'''


DPH2 = '''[[parts]]
order = {dph2_order}
key = "image_dph2_deflector"
name = "Image Corrector DPH2 Deflector"
branch = "image"
local_start_z_mm = {dph2_start}
local_center_z_mm = {hp2_center}
local_end_z_mm = {dph2_end}
length_mm = 30.0
vacuum_inner_diameter_mm = 5
optical_reference_local_z_mm = {hp2_center}
parent_key = "image_hp2_hexapole"

'''


def _part_value(text: str, key: str, field: str) -> float:
    pattern = re.compile(
        rf'(?ms)^\[\[parts\]\]\s+order\s*=\s*\d+\s+'
        rf'key\s*=\s*"{re.escape(key)}".*?^{field}\s*=\s*([-+0-9.eE]+)'
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"Missing {field} for {key}")
    return float(match.group(1))


def _part_order(text: str, key: str) -> int:
    pattern = re.compile(
        rf'(?ms)^\[\[parts\]\]\s+order\s*=\s*(\d+)\s+'
        rf'key\s*=\s*"{re.escape(key)}"'
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"Missing order for {key}")
    return int(match.group(1))


def _format(value: float) -> str:
    return f"{value:.12g}"


def migrate(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "image_tl12_lens" in text:
        return

    hp1_order = _part_order(text, "image_hp1_hexapole")
    hp2_order = _part_order(text, "image_hp2_hexapole")
    values = {
        "dp12_center": _part_value(
            text, "image_dp12_deflector", "local_center_z_mm"
        ),
        "hp1_center": _part_value(
            text, "image_hp1_hexapole", "local_center_z_mm"
        ),
        "hp2_center": _part_value(
            text, "image_hp2_hexapole", "local_center_z_mm"
        ),
        "tl12_order": hp1_order,
        "dph1_order": hp1_order + 1,
        "dph2_order": hp2_order + 2,
    }
    values.update(
        tl12_start=values["dp12_center"] - 0.5,
        tl12_end=values["dp12_center"] + 0.5,
        dph1_start=values["hp1_center"] - 15.0,
        dph1_end=values["hp1_center"] + 15.0,
        dph2_start=values["hp2_center"] - 15.0,
        dph2_end=values["hp2_center"] + 15.0,
    )
    formatted = {
        key: str(value) if key.endswith("_order") else _format(value)
        for key, value in values.items()
    }

    def shift_order(match: re.Match[str]) -> str:
        order = int(match.group(1))
        if order >= hp2_order:
            order += 3
        elif order >= hp1_order:
            order += 2
        return f"order = {order}"

    text = re.sub(r"^order\s*=\s*(\d+)\s*$", shift_order, text, flags=re.M)

    shifted_hp1_order = hp1_order + 2
    dp12_marker = (
        f'[[parts]]\norder = {shifted_hp1_order}\n'
        'key = "image_hp1_hexapole"'
    )
    if dp12_marker not in text:
        raise ValueError(f"Cannot find shifted HP1 insertion point in {path}")
    text = text.replace(
        dp12_marker,
        TL12_AND_DPH1.format(**formatted) + dp12_marker,
        1,
    )

    shifted_hp2_order = hp2_order + 3
    hp2_marker = (
        f'[[parts]]\norder = {shifted_hp2_order}\n'
        'key = "image_hp2_hexapole"'
    )
    if hp2_marker not in text:
        raise ValueError(f"Cannot find shifted HP2 insertion point in {path}")
    text = text.replace(
        hp2_marker,
        DPH2.format(**formatted) + hp2_marker,
        1,
    )

    dp12_reference = (
        f'optical_reference_local_z_mm = {formatted["dp12_center"]}\n'
    )
    dp12_overlap = (
        dp12_reference
        + 'mechanical_overlap_group = "image_dp12_tl12_integrated_assembly"\n'
        + 'mechanical_overlap_role = "container"\n'
        + 'mechanical_overlap_reason = "DP12 contains the latest-structure TL12 optical thin-lens channel without moving the existing DP12 centre."\n'
    )
    text = text.replace(dp12_reference, dp12_overlap, 1)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    for target in TARGETS:
        migrate(target)
        print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
