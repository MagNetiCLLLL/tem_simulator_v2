"""Move magnetic-lens Bz signs into the authoritative instrument TOMLs."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = (
    *sorted((ROOT / "configs" / "instruments" / "column").glob("*.toml")),
    *sorted(
        (ROOT / "configs" / "instruments" / "project_and_recording_system")
        .glob("*.toml")
    ),
)

FIELD_POLARITIES = {
    "condenser_lens_1": 1,
    "condenser_lens_2": 1,
    "condenser_lens_3": 1,
    "adapter_lens": 1,
    "probe_tl22_lens": -1,
    "probe_tl21_lens": 1,
    "probe_tl12_lens": 1,
    "mini_condenser": -1,
    "objective_lens": 1,
    "image_ol_post_lens": 1,
    "image_tl11_lens": -1,
    "image_tl12_lens": 1,
    "image_tl21_lens": 1,
    "image_tl22_lens": -1,
    "image_adapter_lens": 1,
    "diffraction_lens": 1,
    "intermediate_lens": 1,
    "projector_lens_1": 1,
    "projector_lens_2": 1,
}

CORRECTOR_LENSES = frozenset({
    "adapter_lens",
    "probe_tl22_lens",
    "probe_tl21_lens",
    "probe_tl12_lens",
    "image_ol_post_lens",
    "image_tl11_lens",
    "image_tl12_lens",
    "image_tl21_lens",
    "image_tl22_lens",
    "image_adapter_lens",
})

COMMON_SOURCE = (
    "Provisional common-column sign convention; physical coil direction is "
    "not confirmed by manufacturer service data."
)
CORRECTOR_SOURCE = (
    "Existing coupled-corrector sign convention; physical coil direction is "
    "not confirmed by manufacturer service data."
)
MINI_CONDENSER_SOURCE = (
    "Default STEM effective-field sign; Microprobe mode overrides it in "
    "operating_modes/catalog.toml; physical channel sign is not "
    "manufacturer-confirmed."
)

_KEY_RE = re.compile(r'^key\s*=\s*"([^"]+)"\s*$')
_POLARITY_FIELD_RE = re.compile(
    r"^(?:polarity|field_polarity|field_polarity_status|"
    r"field_polarity_source)\s*="
)


def _source_for(key: str) -> str:
    if key == "mini_condenser":
        return MINI_CONDENSER_SOURCE
    if key in CORRECTOR_LENSES:
        return CORRECTOR_SOURCE
    return COMMON_SOURCE


def migrate_text(text: str) -> tuple[str, int]:
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)
    starts = [
        index for index, line in enumerate(lines)
        if line.strip() == "[[parts]]"
    ]
    changed = 0
    for position in range(len(starts) - 1, -1, -1):
        start = starts[position]
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        key = None
        for line in lines[start:end]:
            match = _KEY_RE.match(line.strip())
            if match:
                key = match.group(1)
                break
        if key not in FIELD_POLARITIES:
            continue
        block = [
            line for line in lines[start:end]
            if not _POLARITY_FIELD_RE.match(line.strip())
        ]
        insert_at = next(
            (
                index + 1 for index, line in enumerate(block)
                if line.strip() == 'mechanical_part_role = "optical_parent"'
            ),
            None,
        )
        if insert_at is None:
            raise ValueError(f"Magnetic lens {key} has no optical-parent marker")
        fields = [
            f"field_polarity = {FIELD_POLARITIES[key]}{newline}",
            (
                "field_polarity_status = "
                f'"provisional_model_assumption"{newline}'
            ),
            f'field_polarity_source = "{_source_for(key)}"{newline}',
        ]
        migrated = block[:insert_at] + fields + block[insert_at:]
        if migrated != lines[start:end]:
            lines[start:end] = migrated
            changed += 1
    return "".join(lines), changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="write changes; without this flag the script only checks",
    )
    args = parser.parse_args()
    pending = []
    total = 0
    for path in MANIFESTS:
        original = path.read_text(encoding="utf-8")
        migrated, changed = migrate_text(original)
        if not changed:
            continue
        pending.append(path)
        total += changed
        if args.write:
            path.write_text(migrated, encoding="utf-8", newline="")
    if pending and not args.write:
        for path in pending:
            print(path.relative_to(ROOT))
        print(f"{total} magnetic-lens parts require migration")
        return 1
    print(f"migrated {total} magnetic-lens parts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
