r"""Reproduce the operating-profile loss of estimated lens aberrations.

Run from the repository with:
    .venv\Scripts\python.exe tmp\project_inspection_repro_profile.py

Only a temporary profile is written; source and instrument TOMLs are unchanged.
"""

from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from temsim.assembly_catalog import AssemblyCatalog
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.optics.aberrations import intrinsic_lens_aberration_profile
from temsim.optics.column import default_state
from temsim.profile_io import apply_profile_values, read_profile, save_profile


def describe(state):
    lens = state.objective_lens
    profile = intrinsic_lens_aberration_profile(lens, state.beam_voltage_kv)
    return {
        "stored_cs_mm": lens.cs_mm,
        "stored_cc_mm": lens.cc_mm,
        "effective_cs_mm": profile.cs_mm,
        "effective_cc_mm": profile.cc_mm,
        "status": profile.status,
    }


def main():
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    state = default_state()
    catalog.apply(state, selection)

    # This is the same assignment as choosing the GUI's
    # "Focal-length estimate (provisional)" aberration model.
    state.objective_lens.cs_mm = None
    state.objective_lens.cc_mm = None
    before = describe(state)

    with tempfile.TemporaryDirectory(prefix="temsim-profile-review-") as directory:
        path = Path(directory) / "estimated-aberrations.toml"
        save_profile(path, state, selection)
        loaded_selection, values = read_profile(path)
        print("Serialized objective fields:", values["objective_lens"])

        # Reproduce MainWindow.open_profile's candidate-state loading sequence.
        restored = type(state).from_dict(state.to_dict())
        catalog.apply(restored, loaded_selection)
        skipped = apply_profile_values(restored, values)
        apply_physical_layout_to_state(restored, preserve_operating_parameters=True)

    after = describe(restored)
    print("Before:", before)
    print("After: ", after)
    print("Skipped fields:", skipped)
    assert before["stored_cs_mm"] is None and before["stored_cc_mm"] is None
    assert after["stored_cs_mm"] is not None and after["stored_cc_mm"] is not None
    assert before["status"] != after["status"]
    assert (before["effective_cs_mm"], before["effective_cc_mm"]) != (
        after["effective_cs_mm"], after["effective_cc_mm"]
    )
    print("REPRODUCED: saving and opening the profile changes aberration mode and values.")


if __name__ == "__main__":
    main()
