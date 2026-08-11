from pathlib import Path

import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.profile_io import apply_profile_values, read_profile, save_profile
from temsim.specimen.geometry import quaternion_from_euler_xyz_deg


def test_profile_v2_round_trips_sample_tables_and_quaternion(tmp_path: Path):
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    state = default_state()
    state.sample.specimen_orientation_quaternion_wxyz = (
        quaternion_from_euler_xyz_deg((12.0, -4.0, 33.0))
    )
    state.sample.zone_axis_uvw = (1, 1, 0)
    state.sample.in_plane_axis_uvw = (0, 0, 1)
    state.sample.virtual_interactions = [
        {
            "name": "absorbed",
            "kind": "absorption",
            "enabled": True,
            "probability": 0.2,
        }
    ]
    state.sample.virtual_regions = [
        {
            "name": "island",
            "kind": "ellipse",
            "enabled": True,
            "density": 0.7,
            "centre_x_nm": 2.0,
            "centre_y_nm": -1.0,
            "size_x_nm": 5.0,
            "size_y_nm": 4.0,
        }
    ]
    state.sample.wave_frozen_phonon_sigma_by_element_angstrom = {
        "Si": 0.075
    }
    path = tmp_path / "sample-v2.toml"

    save_profile(path, state, selection)
    loaded_selection, values = read_profile(path)
    restored = default_state()
    skipped = apply_profile_values(restored, values)

    assert loaded_selection == selection
    assert skipped == []
    assert restored.sample.specimen_orientation_quaternion_wxyz == pytest.approx(
        state.sample.specimen_orientation_quaternion_wxyz
    )
    assert restored.sample.zone_axis_uvw == (1, 1, 0)
    assert restored.sample.in_plane_axis_uvw == (0, 0, 1)
    assert restored.sample.virtual_interactions == state.sample.virtual_interactions
    assert restored.sample.virtual_regions == state.sample.virtual_regions
    assert restored.sample.wave_frozen_phonon_sigma_by_element_angstrom == {
        "Si": 0.075
    }
    assert "format_version = 2" in path.read_text(encoding="utf-8")


def test_profile_v1_is_read_and_legacy_virtual_weights_are_migrated(tmp_path: Path):
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    path = tmp_path / "sample-v1.toml"
    path.write_text(
        "\n".join(
            (
                "format_version = 1",
                "[assembly]",
                f'gun = "{selection.gun}"',
                f'column = "{selection.column}"',
                f'recording = "{selection.recording}"',
                "[devices.sample]",
                'specimen_mode = "virtual"',
                "virtual_diffraction_relative_weight = 0.5",
                "virtual_scattering_relative_weight = 0.25",
            )
        ),
        encoding="utf-8",
    )

    _selection, values = read_profile(path)
    state = default_state()
    skipped = apply_profile_values(state, values)

    assert skipped == []
    probabilities = [
        row["probability"] for row in state.sample.virtual_interactions
    ]
    assert sum(probabilities) < 1.0
    assert all(value >= 0.0 for value in probabilities)

