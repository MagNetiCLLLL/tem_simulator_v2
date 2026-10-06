"""Shared deflector channels expose operating controls, not a second geometry."""

from types import SimpleNamespace

import pytest
import tomli_w

from temsim.runtime_parameters import (
    RuntimeTarget, editable_parameters, validate_runtime_assignment,
)


GEOMETRY = {
    "z_mm": 200., "upper_z_mm": 187.5, "lower_z_mm": 212.5,
    "optical_reference_z_mm": 200., "mechanical_length_mm": 40.,
    "mechanical_coil_length_mm": 15., "mechanical_inter_coil_gap_mm": 10.,
    "mechanical_outer_diameter_mm": 54., "mechanical_clear_bore_diameter_mm": 20.,
    "effective_thickness_mm": 15., "optical_plane_separation_mm": 25.,
    "effective_aperture_radius_mm": 10., "maximum_kick_mrad": 100.,
}
OPERATING = {"enabled": True, "scan_enabled": False, "upper_coil_gain": .7,
             "scan_pixels_x": 32, "scan_lines": 24, "scan_pixel_size_nm": .1,
             "scan_frame_period_s": .5, "kick_x_mrad": .1, "kick_y_mrad": -.2,
             "descan_target_key": "selected_area_aperture"}


def test_descan_geometry_is_hidden_while_operating_controls_remain_available():
    target = RuntimeTarget("descan_deflector", "Descan", SimpleNamespace(**GEOMETRY, **OPERATING))
    editable = {item.name for item in editable_parameters(target)}
    assert set(GEOMETRY).isdisjoint(editable)
    assert set(OPERATING) <= editable
    for name, value in OPERATING.items():
        assert validate_runtime_assignment(target, name, value) == value


@pytest.mark.parametrize("name", GEOMETRY)
def test_direct_shared_geometry_assignments_are_rejected(name):
    target = RuntimeTarget("descan_deflector", "Descan", SimpleNamespace(**GEOMETRY))
    with pytest.raises(ValueError, match="image_diffraction_deflector"):
        validate_runtime_assignment(target, name, GEOMETRY[name])
    assert vars(target.obj) == GEOMETRY


def test_independent_scan_effective_thickness_retains_its_control():
    target = RuntimeTarget("ac_deflector", "Scan", SimpleNamespace(**GEOMETRY, **OPERATING))
    assert "effective_thickness_mm" in {item.name for item in editable_parameters(target)}
    assert validate_runtime_assignment(target, "effective_thickness_mm", 12.) == 12.


def test_old_profile_geometry_readouts_do_not_override_shared_host(tmp_path):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.column import default_state
    from temsim.profile_io import PROFILE_FORMAT_VERSION, apply_profile_values, read_profile, save_profile

    path = tmp_path / "profile.toml"
    path.write_text(tomli_w.dumps({
        "format_version": PROFILE_FORMAT_VERSION,
        "assembly": {"gun": "FEG", "column": "C3 + Probe Corrector", "recording": "No Energy Filter"},
        "devices": {"descan_deflector": {**OPERATING, "effective_thickness_mm": 999.,
                                            "optical_plane_separation_mm": 999.},
                    "ac_deflector": {"effective_thickness_mm": 12.}},
    }), encoding="utf-8")
    _, values = read_profile(path)
    assert values["descan_deflector"] == OPERATING
    assert values["ac_deflector"]["effective_thickness_mm"] == 12.
    state = default_state()
    before = state.descan_deflector.upper_z_mm, state.descan_deflector.lower_z_mm, state.descan_deflector.effective_thickness_mm
    apply_profile_values(state, values)
    assert (state.descan_deflector.upper_z_mm, state.descan_deflector.lower_z_mm,
            state.descan_deflector.effective_thickness_mm) == before
    for name, value in OPERATING.items():
        assert getattr(state.descan_deflector, name) == value
    rewritten = tmp_path / "current.toml"
    selection = AssemblyCatalog().selection_for_resolved(state._resolved_assembly)
    save_profile(rewritten, state, selection)
    import tomllib
    saved = tomllib.loads(rewritten.read_text(encoding="utf-8"))["devices"]
    assert "effective_thickness_mm" not in saved["descan_deflector"]
    assert "optical_plane_separation_mm" not in saved["descan_deflector"]
    assert saved["ac_deflector"]["effective_thickness_mm"] == 12.
    assert saved["descan_deflector"]["upper_coil_gain"] == OPERATING["upper_coil_gain"]
