from copy import deepcopy
import tomllib

import pytest
import tomli_w

from temsim.assembly_catalog import AssemblyCatalog
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.optics.aberrations import intrinsic_lens_aberration_profile
from temsim.optics.column import default_state
from temsim.profile_io import (
    PROFILE_FORMAT_VERSION,
    apply_profile_values,
    read_profile,
    save_profile,
)


def _load_like_gui(catalog, current_state, path):
    selection, values = read_profile(path)
    candidate = type(current_state).from_dict(current_state.to_dict())
    catalog.apply(candidate, selection)
    skipped = apply_profile_values(candidate, values)
    apply_physical_layout_to_state(candidate, preserve_operating_parameters=True)
    assert skipped == []
    return candidate


@pytest.mark.parametrize(
    "cs_mm, cc_mm", [(None, None), (0.85, 1.25), (None, 1.25), (0.85, None)]
)
def test_profile_preserves_aberration_mode_through_gui_assembly_reload(
    tmp_path, cs_mm, cc_mm
):
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    state = default_state()
    catalog.apply(state, selection)
    state.objective_lens.cs_mm = cs_mm
    state.objective_lens.cc_mm = cc_mm
    before = intrinsic_lens_aberration_profile(state.objective_lens, state.beam_voltage_kv)
    path = tmp_path / "aberrations.toml"
    save_profile(path, state, selection)

    # Loading must restore the saved mode even after editing the live lens.
    state.objective_lens.cs_mm = 8.0
    state.objective_lens.cc_mm = 9.0
    restored = _load_like_gui(catalog, state, path)
    after = intrinsic_lens_aberration_profile(restored.objective_lens, restored.beam_voltage_kv)

    assert restored.objective_lens.cs_mm == cs_mm
    assert restored.objective_lens.cc_mm == cc_mm
    assert after.status == before.status
    assert after.cs_mm == pytest.approx(before.cs_mm)
    assert after.cc_mm == pytest.approx(before.cc_mm)


def test_profile_restores_estimates_on_all_runtime_lens_types(tmp_path):
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    state = default_state()
    catalog.apply(state, selection)
    for lens in state.lenses:
        lens.cs_mm = None
        lens.cc_mm = None
    path = tmp_path / "all-estimated.toml"
    save_profile(path, state, selection)

    restored = _load_like_gui(catalog, state, path)

    assert all(lens.cs_mm is None and lens.cc_mm is None for lens in restored.lenses)
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    assert document["format_version"] == PROFILE_FORMAT_VERSION == 7


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_legacy_profile_keeps_omitted_coefficients_and_explicit_numbers(tmp_path, version):
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    path = tmp_path / "legacy.toml"
    path.write_text(
        tomli_w.dumps({
            "format_version": version,
            "assembly": {
                "gun": selection.gun,
                "column": selection.column,
                "recording": selection.recording,
            },
            "devices": {"objective_lens": {"cs_mm": 0.85}},
        }),
        encoding="utf-8",
    )
    state = default_state()
    catalog.apply(state, selection)
    original_cc_mm = state.objective_lens.cc_mm

    restored = _load_like_gui(catalog, state, path)

    assert restored.objective_lens.cs_mm == 0.85
    assert restored.objective_lens.cc_mm == original_cc_mm


@pytest.mark.parametrize(
    "none_values, devices, message",
    [
        ([], {}, "none_values must be a table"),
        ({"objective_lens": "cs_mm"}, {}, "array of names"),
        ({"objective_lens": [1]}, {}, "array of names"),
        ({"objective_lens": ["cs_mm", "cs_mm"]}, {}, "duplicate names"),
        (
            {"objective_lens": ["cs_mm"]},
            {"objective_lens": {"cs_mm": 1.0}},
            "conflicting none values",
        ),
    ],
)
def test_profile_rejects_malformed_or_conflicting_none_values(
    tmp_path, none_values, devices, message
):
    path = tmp_path / "invalid.toml"
    path.write_text(
        tomli_w.dumps({
            "format_version": 4,
            "assembly": {"gun": "FEG", "column": "C3 + Probe Corrector", "recording": "Energy Filter"},
            "devices": devices,
            "none_values": none_values,
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        read_profile(path)


@pytest.mark.parametrize(
    "invalid",
    [
        {"objective_lens": {"percent": None}},
        {"objective_lens": {"z_mm": None}},
        {"objective_lens": {"unknown": None}},
        {"sample": {"cs_mm": None}},
        {"unknown_device": {"cs_mm": None}},
        {"simulation": {"step_mm": 0.0}},
        {"__sample_model__": {"zone_axis_uvw": [0, 0, 0]}},
        {"__simulation_model__": {"mode": "unknown"}},
    ],
)
def test_optional_assignments_are_validated_before_any_state_changes(invalid):
    state = default_state()
    state.objective_lens.cs_mm = 1.7
    before = deepcopy(state.to_dict())
    values = {
        "objective_lens": {"cs_mm": None},
        "sample": {"centre_x_nm": 12.0},
    }
    for key, attributes in invalid.items():
        values.setdefault(key, {}).update(attributes)

    with pytest.raises(ValueError):
        apply_profile_values(state, values)

    assert state.to_dict() == before


def test_explicit_none_clears_a_numeric_optional_field_without_replacing_sample():
    state = default_state()
    sample = state.sample
    state.objective_lens.cs_mm = 1.7

    skipped = apply_profile_values(state, {"objective_lens": {"cs_mm": None}})

    assert state.objective_lens.cs_mm is None
    assert state.sample is sample
    assert skipped == []
