"""Project and editable configuration paths."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def project_root() -> Path:
    """Return the root containing editable or wheel-installed configuration."""

    override = os.environ.get("TEMSIM_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    checkout_root = Path(__file__).resolve().parents[2]
    if (checkout_root / "configs").is_dir():
        return checkout_root
    installed_root = Path(sys.prefix).resolve()
    if (installed_root / "configs").is_dir():
        return installed_root
    return checkout_root


CONFIG_ROOT = project_root() / "configs"
INSTRUMENT_CONFIG_ROOT = CONFIG_ROOT / "instruments"
SPECIMEN_CONFIG_ROOT = CONFIG_ROOT / "specimens"
OPERATING_MODE_CONFIG_ROOT = CONFIG_ROOT / "operating_modes"
