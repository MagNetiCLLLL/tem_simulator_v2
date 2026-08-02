from pathlib import Path
import shutil

import pytest

from temsim.column.state_layout import layout_configuration_from_state
from temsim.manifest_editor import ManifestEditor, ManifestTarget
from temsim.optics.column import default_state
from temsim.paths import INSTRUMENT_CONFIG_ROOT


def test_manifest_save_validates_and_invalid_edit_rolls_back(tmp_path: Path):
    root = tmp_path / "instruments"
    shutil.copytree(INSTRUMENT_CONFIG_ROOT, root)
    editor = ManifestEditor(root)
    target = ManifestTarget(
        "column/C3_ProbeCorrector.toml", "condenser_lens_1"
    )
    configuration = layout_configuration_from_state(default_state())

    editor.save(
        target,
        {("parts", "condenser_lens_1", "name"): "C1 Test Lens"},
        configuration,
    )
    saved_text = (root / target.module_path).read_text(encoding="utf-8")
    assert 'name = "C1 Test Lens"' in saved_text

    with pytest.raises(ValueError, match="length mismatch"):
        editor.save(
            target,
            {("parts", "condenser_lens_1", "length_mm"): 999.0},
            configuration,
        )

    assert (root / target.module_path).read_text(encoding="utf-8") == saved_text
