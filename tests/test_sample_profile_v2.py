from pathlib import Path
import tomllib

import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.profile_io import (
    PROFILE_FORMAT_VERSION,
    apply_profile_values,
    read_profile,
    save_profile,
)
from temsim.specimen.geometry import quaternion_from_euler_xyz_deg
from temsim.specimen.source import (
    active_cif_path,
    selected_reference_preset_key,
)


def test_profile_round_trips_real_sample_metadata_and_quaternion(tmp_path: Path):
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    state = default_state()
    state.sample.envelope_shape = "rectangle"
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
    state.sample.eds_support_material_key = "gold"
    state.sample.eds_support_mesh_key = "square_400_eoa"
    state.sample.eds_support_offset_x_um = 12.5
    state.sample.eds_support_rotation_deg = 17.0
    state.sample.eds_detector_efficiency = 0.83
    state.sample.eds_energy_resolution_fwhm_ev = 125.0
    state.sample.eds_poisson_enabled = True
    state.sample.eds_poisson_seed = 44
    state.sample.eds_transport_mode = "elastic_monte_carlo"
    state.sample.eds_elastic_seed = 45
    state.sample.eds_elastic_max_events = 1234
    state.sample.sample_region_upstream_distance_um = 72.5
    state.sample.sample_region_downstream_distance_um = 88.0
    state.sample.sample_region_photon_path_count = 77
    state.sample.sample_region_secondary_path_count = 19
    state.sample.sample_region_seed = 46
    path = tmp_path / "sample-v2.toml"

    save_profile(path, state, selection)
    loaded_selection, values = read_profile(path)
    restored = default_state()
    skipped = apply_profile_values(restored, values)

    assert loaded_selection == selection
    assert skipped is None
    assert restored.sample.envelope_shape == "rectangle"
    assert restored.sample.specimen_orientation_quaternion_wxyz == pytest.approx(
        state.sample.specimen_orientation_quaternion_wxyz
    )
    assert restored.sample.zone_axis_uvw == (1, 1, 0)
    assert restored.sample.in_plane_axis_uvw == (0, 0, 1)
    assert restored.sample.virtual_interactions == []
    assert restored.sample.virtual_regions == []
    saved = tomllib.loads(path.read_text(encoding="utf-8"))
    assert "virtual_interactions" not in saved["sample_model"]
    assert "virtual_regions" not in saved["sample_model"]
    assert restored.sample.wave_frozen_phonon_sigma_by_element_angstrom == {
        "Si": 0.075
    }
    assert restored.sample.eds_support_material_key == "gold"
    assert restored.sample.eds_support_mesh_key == "square_400_eoa"
    assert restored.sample.eds_support_offset_x_um == pytest.approx(12.5)
    assert restored.sample.eds_support_rotation_deg == pytest.approx(17.0)
    assert restored.sample.eds_detector_efficiency == pytest.approx(0.83)
    assert restored.sample.eds_energy_resolution_fwhm_ev == pytest.approx(
        125.0
    )
    assert restored.sample.eds_poisson_enabled is True
    assert restored.sample.eds_poisson_seed == 44
    assert restored.sample.eds_transport_mode == "elastic_monte_carlo"
    assert restored.sample.eds_elastic_seed == 45
    assert restored.sample.eds_elastic_max_events == 1234
    assert restored.sample.sample_region_upstream_distance_um == pytest.approx(
        72.5
    )
    assert restored.sample.sample_region_downstream_distance_um == pytest.approx(
        88.0
    )
    assert restored.sample.sample_region_photon_path_count == 77
    assert restored.sample.sample_region_secondary_path_count == 19
    assert restored.sample.sample_region_seed == 46
    assert tomllib.loads(path.read_text(encoding="utf-8"))["format_version"] == PROFILE_FORMAT_VERSION


def test_retired_eds_trajectory_count_is_rejected_without_partial_edits():
    state = default_state()
    seed = state.sample.eds_elastic_seed
    with pytest.raises(ValueError, match="Unknown operating-profile field"):
        apply_profile_values(state, {"sample": {
            "eds_elastic_seed": 73, "eds_elastic_trajectory_count": 19}})
    assert state.sample.eds_elastic_seed == seed


def test_profile_round_trips_mode_owned_structure_sources(tmp_path: Path):
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    state = default_state()
    state.sample.specimen_mode = "atomic"
    state.sample.specimen_preset_key = "si_110"
    state.sample.cif_path = "ideal-sample.cif"
    path = tmp_path / "cif-source.toml"

    save_profile(path, state, selection)
    _loaded_selection, values = read_profile(path)
    restored = default_state()
    skipped = apply_profile_values(restored, values)

    assert skipped is None
    assert restored.sample.specimen_mode == "atomic"
    assert not hasattr(restored.sample, "atomic_structure_source")
    assert restored.sample.specimen_preset_key == "si_110"
    assert restored.sample.cif_path == "ideal-sample.cif"
    assert active_cif_path(restored.sample) == "ideal-sample.cif"
    assert selected_reference_preset_key(restored.sample) == ""


def test_partial_cif_path_does_not_implicitly_switch_sample_mode():
    state = default_state()
    mode = state.sample.specimen_mode
    apply_profile_values(state, {"sample": {"cif_path": "sample.cif"}})
    assert state.sample.specimen_mode == mode
    assert state.sample.cif_path == "sample.cif"


def test_profile_rejects_unknown_retired_atomic_structure_source():
    state = default_state()

    with pytest.raises(ValueError, match="Unknown operating-profile field"):
        apply_profile_values(
            state,
            {"sample": {"atomic_structure_source": "both"}},
        )
