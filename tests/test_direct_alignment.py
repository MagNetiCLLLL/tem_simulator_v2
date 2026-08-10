import math

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.operating_modes import (
    apply_operating_mode_pair,
    direct_alignment_by_key,
    load_operating_mode_catalog,
)
from temsim.optics.column import default_state
from temsim.optics.direct_alignment import (
    _ProjectorMeasurementModel,
    _validate_condenser_production,
    apply_direct_alignment,
)
from temsim.physics.beam_statistics import transverse_beam_statistics
from temsim.physics.first_order import trace_transverse_transfer
from temsim.physics.recording_stop import determine_tem_stop_z


CONDENSER_KEYS = ("condenser_lens_2", "condenser_lens_3")
PROJECTOR_KEYS = (
    "diffraction_lens",
    "intermediate_lens",
    "projector_lens_1",
    "projector_lens_2",
)
IMAGE_KEYS = ("objective_lens", *PROJECTOR_KEYS)


@pytest.fixture(scope="module")
def assembled_state():
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    return state


def _state_copy(template):
    return type(template).from_dict(template.to_dict())


def _lens_values(state):
    return {lens.key: float(lens.percent) for lens in state.lenses}


def test_equivalent_image_lens_mode_round_trips_in_state(assembled_state):
    state = _state_copy(assembled_state)
    state.equivalent_image_lenses_enabled = True

    restored = _state_copy(state)

    assert restored.equivalent_image_lenses_enabled is True


def test_toml_defines_the_four_exact_direct_alignment_controls():
    catalog = load_operating_mode_catalog()
    definitions = {
        definition.key: definition
        for definition in catalog.direct_alignments
    }

    assert set(definitions) == {
        "nanoprobe_convergence",
        "microprobe_illumination",
        "image_magnification",
        "diffraction_camera_length",
    }
    expected = {
        "nanoprobe_convergence": (
            "nano_probe", "mrad", 20.0, 40.0, 30.0, CONDENSER_KEYS,
        ),
        "microprobe_illumination": (
            "micro_probe", "um", 0.5, 2.2, 2.0, CONDENSER_KEYS,
        ),
        "image_magnification": (
            "imaging", "x", 10.0, 1_000_000.0, 65.7, IMAGE_KEYS,
        ),
        "diffraction_camera_length": (
            "diffraction", "m", 0.01, 5.0, 0.05, PROJECTOR_KEYS,
        ),
    }
    for key, values in expected.items():
        definition = definitions[key]
        assert (
            definition.mode_key,
            definition.unit,
            definition.minimum,
            definition.maximum,
            definition.default_value,
            definition.devices,
        ) == values
        assert definition.calibration_status
        assert definition.calibration_reference
        assert float(definition.targets["maximum_numerical_spread"]) == 0.01
        if definition.family == "projector":
            assert float(
                definition.targets["maximum_continuation_ratio"]
            ) == 2.0
            assert int(
                definition.targets["maximum_continuation_stages"]
            ) == 8
    image = definitions["image_magnification"]
    assert image.constraint == "sample_to_recording_plane_B_zero"
    assert image.targets["preset_magnifications"] == [
        10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0,
    ]
    assert len(image.targets["preset_vectors"]) == 6


def test_beam_statistics_are_invariant_to_common_larmor_rotation():
    x_m = np.array((-2.0, -0.4, 0.8, 2.5, 1.1)) * 1.0e-6
    y_m = np.array((0.3, 1.2, -1.4, 0.7, -0.2)) * 1.0e-6
    tx_rad = np.array((-0.015, -0.006, 0.004, 0.018, 0.009))
    ty_rad = np.array((0.003, 0.011, -0.008, 0.006, -0.012))
    weights = np.array((0.08, 0.17, 0.31, 0.29, 0.15))
    reference = transverse_beam_statistics(
        x_m, y_m, tx_rad, ty_rad, weights=weights
    )

    angle = 0.731
    cosine = math.cos(angle)
    sine = math.sin(angle)

    def rotate(first, second):
        return (
            cosine * first - sine * second,
            sine * first + cosine * second,
        )

    rotated_x, rotated_y = rotate(x_m, y_m)
    rotated_tx, rotated_ty = rotate(tx_rad, ty_rad)
    rotated = transverse_beam_statistics(
        rotated_x,
        rotated_y,
        rotated_tx,
        rotated_ty,
        weights=weights,
    )

    invariant_fields = (
        "convergence_rms_rad",
        "convergence_95_rad",
        "convergence_99_rad",
        "convergence_edge_rad",
        "radius_rms_m",
        "radius_95_m",
        "radius_99_m",
        "radial_position_angle_covariance_m_rad",
        "radial_wavefront_curvature_per_m",
        "waist_offset_m",
    )
    for field in invariant_fields:
        assert getattr(rotated, field) == pytest.approx(
            getattr(reference, field), rel=2.0e-12, abs=2.0e-15
        )


def test_beam_statistics_use_current_weighted_95_percent_quantiles():
    statistics = transverse_beam_statistics(
        np.array((0.0, 1.0e-6, -1.0e-6)),
        np.zeros(3),
        np.array((0.0, 0.1, -0.1)),
        np.zeros(3),
        weights=np.array((0.96, 0.02, 0.02)),
    )

    assert statistics.convergence_95_rad == pytest.approx(0.0, abs=1.0e-15)
    assert statistics.convergence_99_rad == pytest.approx(math.atan(0.1))
    assert statistics.radius_95_m == pytest.approx(0.0, abs=1.0e-18)
    assert statistics.radius_99_m == pytest.approx(1.0e-6)


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (np.array((np.nan, 1.0)), "weights must be finite"),
        (np.array((np.inf, 1.0)), "weights must be finite"),
        (np.array((0.0, 0.0)), "positive total weight"),
        (np.array((-1.0, 1.0)), "weights must be non-negative"),
    ],
)
def test_beam_statistics_reject_invalid_current_weights(weights, message):
    with pytest.raises(ValueError, match=message):
        transverse_beam_statistics(
            np.array((0.0, 1.0e-6)),
            np.zeros(2),
            np.array((0.0, 0.01)),
            np.zeros(2),
            weights=weights,
        )


def test_cached_projector_map_matches_the_production_full_transverse_trace(
    assembled_state,
):
    state = _state_copy(assembled_state)
    apply_operating_mode_pair(state, "nano_probe", "imaging")
    state.step_mm = 0.1
    state.equivalent_image_lenses_enabled = True
    definition = direct_alignment_by_key("image_magnification")
    model = _ProjectorMeasurementModel(
        state, definition, step_mm=state.step_mm
    )
    values = np.asarray([
        next(lens for lens in state.lenses if lens.key == key).percent
        for key in IMAGE_KEYS
    ])

    cached = model.sample_model.matrix(values)
    production = trace_transverse_transfer(
        state, state.sample.z_mm, determine_tem_stop_z(state)
    ).matrix

    assert cached == pytest.approx(production, rel=1.0e-4, abs=1.0e-6)


def test_condenser_production_validation_uses_kicks_and_aperture_planes(
    assembled_state, monkeypatch
):
    import temsim.optics.direct_alignment as direct_alignment

    state = _state_copy(assembled_state)
    apply_operating_mode_pair(state, "nano_probe", "imaging")
    definition = direct_alignment_by_key("nanoprobe_convergence")
    vector = np.asarray([
        next(lens for lens in state.lenses if lens.key == key).percent
        for key in CONDENSER_KEYS
    ])
    state.condenser_aperture_3.radius_mm = 0.35
    had_used_backends = hasattr(state, "_active_backends_used")
    real_propagate = direct_alignment.propagate
    calls = []

    def observe_propagate(*args, **kwargs):
        calls.append({
            "events": tuple(kwargs.get("events", ())),
            "save_z_mm": tuple(kwargs.get("save_z_mm", ())),
        })
        return real_propagate(*args, **kwargs)

    monkeypatch.setattr(direct_alignment, "propagate", observe_propagate)
    _validate_condenser_production(
        state, definition, vector, step_mm=0.05
    )
    assert hasattr(state, "_active_backends_used") is had_used_backends
    assert state.condenser_aperture_2.z_mm in calls[-1]["save_z_mm"]
    assert state.condenser_aperture_3.z_mm in calls[-1]["save_z_mm"]

    before = (
        tuple(vector),
        state.step_mm,
        state.acceleration_enabled,
        state.acceleration_backend,
        state.active_backend,
    )
    state.condenser_deflector.upper_x_mrad = 5.0
    with pytest.raises(ValueError, match="No finite surviving rays"):
        _validate_condenser_production(
            state, definition, vector, step_mm=0.05
        )
    assert any(abs(event[1]) > 0.0 for event in calls[-1]["events"])
    assert (
        tuple(
            next(lens for lens in state.lenses if lens.key == key).percent
            for key in CONDENSER_KEYS
        ),
        state.step_mm,
        state.acceleration_enabled,
        state.acceleration_backend,
        state.active_backend,
    ) == before


def test_nanoprobe_30_mrad_commits_only_the_c2_c3_solution(assembled_state):
    state = _state_copy(assembled_state)
    apply_operating_mode_pair(state, "nano_probe", "imaging")
    before = _lens_values(state)

    result = apply_direct_alignment(state, "nanoprobe_convergence", 30.0)
    after = _lens_values(state)

    assert result.success
    assert result.achieved == pytest.approx(30.0, rel=0.03)
    assert set(result.strengths) == set(CONDENSER_KEYS)
    assert {
        key for key in before if after[key] != before[key]
    } == set(CONDENSER_KEYS)
    assert all(
        after[key] == pytest.approx(result.strengths[key])
        for key in CONDENSER_KEYS
    )
    definition = direct_alignment_by_key("nanoprobe_convergence")
    assert abs(result.constraint_value) <= float(
        definition.targets["maximum_waist_offset_mm"]
    )


@pytest.mark.parametrize(
    "target", (10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0)
)
def test_image_working_points_commit_the_five_lens_solution(
    assembled_state, target
):
    state = _state_copy(assembled_state)
    apply_operating_mode_pair(state, "nano_probe", "imaging")
    before = _lens_values(state)

    result = apply_direct_alignment(state, "image_magnification", target)
    after = _lens_values(state)

    assert result.success
    assert result.achieved == pytest.approx(target, rel=0.03)
    assert set(result.strengths) == set(IMAGE_KEYS)
    assert {
        key for key in before if after[key] != before[key]
    } == set(IMAGE_KEYS)
    assert all(
        after[key] == pytest.approx(result.strengths[key])
        for key in IMAGE_KEYS
    )
    definition = direct_alignment_by_key("image_magnification")
    assert result.constraint_value <= float(
        definition.targets["maximum_relay_error_um"]
    )
    assert state.equivalent_image_lenses_enabled
    if target <= 1000.0:
        assert after["objective_lens"] <= float(
            definition.targets["lm_objective_max_percent"]
        )
    else:
        assert after["objective_lens"] >= float(
            definition.targets["normal_objective_min_percent"]
        )


def test_microprobe_area_keeps_the_parallel_branch_and_c2_headroom(
    assembled_state,
):
    state = _state_copy(assembled_state)
    apply_operating_mode_pair(state, "micro_probe", "imaging")
    before = _lens_values(state)

    result = apply_direct_alignment(state, "microprobe_illumination", 2.0)
    after = _lens_values(state)

    assert result.success
    assert result.achieved == pytest.approx(2.0, rel=0.05)
    assert abs(result.constraint_value) <= 25.0
    assert result.convergence_95_mrad <= 0.5
    assert 30.0 <= after["condenser_lens_2"] <= 70.0
    assert {
        key for key in before if after[key] != before[key]
    } == set(CONDENSER_KEYS)


@pytest.mark.parametrize("target", (0.01, 0.05, 0.1, 0.5, 1.0, 2.0))
def test_camera_length_working_points_use_all_projector_lenses(
    assembled_state, target
):
    state = _state_copy(assembled_state)
    apply_operating_mode_pair(state, "nano_probe", "diffraction")
    before = _lens_values(state)

    result = apply_direct_alignment(
        state, "diffraction_camera_length", target
    )
    after = _lens_values(state)

    assert result.success
    assert result.achieved == pytest.approx(target, rel=0.03)
    assert result.relay_error_um <= 30.0
    assert result.validation_step_mm == pytest.approx(0.025)
    assert {
        key for key in before if after[key] != before[key]
    } == set(PROJECTOR_KEYS)


def test_wrong_mode_and_out_of_range_targets_do_not_change_lenses(
    assembled_state,
):
    state = _state_copy(assembled_state)
    apply_operating_mode_pair(state, "nano_probe", "imaging")
    before = _lens_values(state)

    with pytest.raises(ValueError, match="only active in micro_probe mode"):
        apply_direct_alignment(state, "microprobe_illumination", 2.0)
    assert _lens_values(state) == before

    with pytest.raises(ValueError, match="must be between 20 and 40 mrad"):
        apply_direct_alignment(state, "nanoprobe_convergence", 40.01)
    assert _lens_values(state) == before


def test_unreachable_image_target_restores_all_five_lenses(
    assembled_state,
):
    state = _state_copy(assembled_state)
    apply_operating_mode_pair(state, "nano_probe", "imaging")
    before = _lens_values(state)
    for lens in state.lenses:
        if lens.key in IMAGE_KEYS:
            lens.max_percent = 0.001

    result = apply_direct_alignment(
        state, "image_magnification", 1_000.0
    )

    assert not result.success
    assert _lens_values(state) == before
    assert result.strengths == {
        key: before[key] for key in IMAGE_KEYS
    }
    assert not state.equivalent_image_lenses_enabled
    assert "previous lens values were restored" in result.message
