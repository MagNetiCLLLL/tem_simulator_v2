"""Project-local defaults for real instrument records."""
from pathlib import Path

from temsim.paths import project_root


def prepare_output_directory(saved_output: str | None = None) -> tuple[Path, str | None]:
    """Create the default folder without replacing a user's chosen location."""
    default = project_root() / "instrument_records"
    saved = (saved_output or "").strip()
    selected = Path(saved).expanduser() if saved else default
    if selected == default:
        try:
            selected.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return selected, f"Cannot create default output folder: {exc}. Choose another output folder."
    return selected, None
