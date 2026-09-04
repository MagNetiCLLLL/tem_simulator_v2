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
from temsim.physics.recording_stop import tem_camera_plane_z


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


def test_toml_defines_the_five_exact_direct_alignment_controls():
    catalog = load_operating_mode_catalog()
    definitions = {
        definition.key: definition
        for definition in catalog.direct_alignments
    }

    assert set(definitions) == {
        "spot_size_current_limit",
        "nanoprobe_convergence",
        "microprobe_illumination",
        "image_magnification",
        "diffraction_camera_length",
    }
    expected = {
        "spot_size_current_limit": (
            "micro_probe", "%", 0.1, 100.0, 100.0, (),
        ),
        "nanoprobe_convergence": (
            "nano_probe", "mrad", 3.0, 60.0, 30.0, CONDENSER_KEYS,
        ),
        "microprobe_illumination": (
            "micro_probe", "um", 0.75, 2.2, 2.0, CONDENSER_KEYS,
        ),
        "image_magnification": (
            "imaging", "x", 10.0, 1_000_000.0, 65.7, IMAGE_KEYS,
        ),
        "diffraction_camera_length": (
            "diffraction", "m", 0.005, 2.5, 0.05, PROJECTOR_KEYS,
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
            ) == (10 if key == "diffraction_camera_length" else 8)
    spot = definitions["spot_size_current_limit"]
    assert spot.active_mode_keys == ("micro_probe", "nano_probe")
    assert spot.state_parameters == ("column_current_limit_percent",)
    assert not spot.devices
    image = definitions["image_magnification"]
    assert image.constraint == "sample_to_recording_plane_B_zero"
    assert image.targets["preset_magnifications"] == [
        10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0,
    ]
    assert len(image.targets["preset_vectors"]) == 6
    diffraction = definitions["diffraction_camera_length"]
    assert diffraction.targets["preset_camera_lengths"] == [
        0.005, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
    ]
    assert np.asarray(
        diffraction.targets["preset_vectors"], dtype=float
    ).shape == (7, 4)


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


def test_beam_statistics_report_weighted_surviving_current_fraction():
    statistics = transverse_beam_statistics(
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        alive=np.asarray((True, False, True)),
        weights=np.asarray((0.15, 0.55, 0.30)),
    )

    assert statistics.surviving_rays == 2
    assert statistics.surviving_fraction == pytest.approx(0.45)


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
        state, state.sample.z_mm, tem_camera_plane_z(state)
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


@pytest.mark.parametrize(
    ("diameter_um", "target"),
    ((10.0, 3.0), (100.0, 30.0), (200.0, 60.0)),
)
def test_nanoprobe_range_commits_only_the_c2_c3_solution(
    assembled_state, diameter_um, target
):
    state = _state_copy(assembled_state)
    apply_operating_mode_pair(state, "nano_probe", "imaging")
    state.condenser_aperture_2.diameter_um = diameter_um
    before = _lens_values(state)

    result = apply_direct_alignment(state, "nanoprobe_convergence", target)
    after = _lens_values(state)

    assert result.success
    assert result.achieved == pytest.approx(target, rel=0.08)
    assert state.condenser_aperture_2.diameter_um == pytest.approx(diameter_um)
    assert set(result.strengths) == set(CONDENSER_KEYS)
    assert {
        key for key in before if after[key] != before[key]
    }.issubset(set(CONDENSER_KEYS))
    assert all(
        after[key] == pytest.approx(result.strengths[key])
        for key in CONDENSER_KEYS
    )
    definition = direct_alignment_by_key("nanoprobe_convergence")
    assert abs(result.constraint_value) <= float(
        definition.targets["maximum_waist_offset_mm"]
    )


@pytest.mark.parametrize("diameter_um", (20.0, 60.0, 140.0))
def test_nanoprobe_solve_uses_the_current_c2_aperture_diameter(
    assembled_state, diameter_um
):
    state = _state_copy(assembled_state)
    apply_operating_mode_pair(state, "nano_probe", "imaging")
    state.condenser_aperture_2.diameter_um = diameter_um

    target_mrad = 0.3 * diameter_um
    result = apply_direct_alignment(
        state, "nanoprobe_convergence", target_mrad
    )

    assert result.success
    assert result.achieved == pytest.approx(target_mrad, rel=0.08)
    assert state.condenser_aperture_2.diameter_um == pytest.approx(diameter_um)


def test_fixed_lens_aperture_scaling_matches_recalculated_toml_metrics(
    assembled_state,
):
    mode = next(
        item
        for item in load_operating_mode_catalog().modes
        if item.key == "nano_probe"
    )
    if mode.calibration_status.startswith("retained_not_recomputed_"):
        pytest.skip(
            "stored aperture-scaling metrics predate the C2 aperture move"
        )
    state = _state_copy(assembled_state)
    apply_operating_mode_pair(state, "nano_probe", "imaging")
    definition = direct_alignment_by_key("nanoprobe_convergence")
    lenses = {lens.key: lens for lens in state.lenses}
    vector = np.asarray([
        lenses[key].percent for key in CONDENSER_KEYS
    ])
    angles = []
    diameters_um = (10.0, 20.0, 40.0, 60.0, 80.0, 100.0,
                    120.0, 140.0, 160.0, 180.0, 200.0)
    for diameter_um in diameters_um:
        state.condenser_aperture_2.diameter_um = diameter_um
        measurement = _validate_condenser_production(
            state, definition, vector, step_mm=0.1
        )
        angles.append(measurement.value)

    assert np.all(np.diff(angles) > 0.0)
    errors_mrad = np.asarray(angles) - 0.3 * np.asarray(diameters_um)
    assert np.sqrt(np.mean(errors_mrad**2)) == pytest.approx(
        float(definition.targets["aperture_scaling_rms_error_mrad"]),
        rel=2.0e-7,
    )
    assert np.max(np.abs(errors_mrad)) == pytest.approx(
        float(definition.targets[
            "aperture_scaling_maximum_absolute_error_mrad"
        ]),
        rel=2.0e-7,
    )


def test_c2_aperture_plane_follows_the_shared_cartridge_before_c3(
    assembled_state,
):
    state = _state_copy(assembled_state)
    apply_operating_mode_pair(state, "nano_probe", "imaging")

    assert state.condenser_aperture_2.z_mm == pytest.approx(765.0)
    lenses = {lens.key: lens for lens in state.lenses}
    assert lenses["condenser_lens_2"].z_mm < state.condenser_aperture_2.z_mm
    assert state.condenser_aperture_2.z_mm < lenses["condenser_lens_3"].z_mm
    assert state.condenser_aperture_2.diameter_um == pytest.approx(100.0)


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


@pytest.mark.parametrize("target_um", (0.75, 1.0, 1.5, 2.0, 2.2))
def test_microprobe_area_keeps_the_parallel_branch_and_c2_c3_bounds(
    assembled_state, target_um,
):
    state = _state_copy(assembled_state)
    apply_operating_mode_pair(state, "micro_probe", "imaging")
    before = _lens_values(state)

    result = apply_direct_alignment(
        state, "microprobe_illumination", target_um
    )
    after = _lens_values(state)

    assert result.success
    assert result.achieved == pytest.approx(target_um, rel=0.05)
    assert abs(result.constraint_value) <= 25.0
    assert result.convergence_95_mrad <= 0.3
    assert result.convergence_99_mrad <= 0.5
    assert 10.0 <= after["condenser_lens_2"] <= 25.0
    assert 30.0 <= after["condenser_lens_3"] <= 40.0
    assert {
        key for key in before if after[key] != before[key]
    } == set(CONDENSER_KEYS)


@pytest.mark.parametrize("probe_mode,projector", (
    ("micro_probe", "imaging"),
    ("nano_probe", "diffraction"),
))
def test_spot_size_caps_physical_current_without_changing_rays_or_lenses(
    assembled_state, probe_mode, projector,
):
    state = _state_copy(assembled_state)
    apply_operating_mode_pair(state, probe_mode, projector)
    before_lenses = _lens_values(state)
    before_ray_count = state.electron_gun.ray_count

    result = apply_direct_alignment(
        state, "spot_size_current_limit", 37.5
    )

    assert result.success
    assert result.achieved == pytest.approx(37.5)
    assert result.strengths == {}
    assert result.state_updates == {
        "column_current_limit_percent": pytest.approx(37.5)
    }
    assert result.constraint_value == pytest.approx(
        state.electron_gun.emitted_current_a * 1.0e12 * 0.375
    )
    assert state.column_current_limit_percent == pytest.approx(37.5)
    assert state.electron_gun.ray_count == before_ray_count
    assert _lens_values(state) == before_lenses


def test_spot_size_current_limit_round_trips_and_scales_stem_and_eds(
    assembled_state,
):
    from temsim.detector.eds_signal import default_eds_incident_electrons
    from temsim.detector.stem_signal import source_current_pa

    state = _state_copy(assembled_state)
    state.column_current_limit_percent = 40.0
    restored = _state_copy(state)
    current_pa = state.electron_gun.emitted_current_a * 1.0e12 * 0.4

    assert restored.column_current_limit_percent == pytest.approx(40.0)
    assert source_current_pa(state) == pytest.approx(current_pa)
    assert default_eds_incident_electrons(
        state, 2.0e-6
    ) == pytest.approx(current_pa * 1.0e-12 * 2.0e-6 / 1.602176634e-19)


def test_camera_length_uses_main_screen_reference_and_commits(
    assembled_state,
):
    state = _state_copy(assembled_state)
    apply_operating_mode_pair(state, "nano_probe", "diffraction")
    before = _lens_values(state)
    result = apply_direct_alignment(
        state, "diffraction_camera_length", 0.05
    )
    after = _lens_values(state)

    assert result.success
    assert result.relay_error_um is None
    assert result.achieved == pytest.approx(0.05, rel=3.0e-2)
    assert result.diffraction_conjugacy_residual <= 1.0e-3
    assert result.target_plane_key == "stem_diffraction_reference_plane"
    assert result.target_plane_z_mm == pytest.approx(
        state.fluorescent_screen.z_mm
    )
    assert (
        state.haadf_detector.z_mm
        < state.dark_field_detector.z_mm
        < state.bright_field_detector.z_mm
    )
    assert result.field_calibration_statuses == (
        "provisional_non_oem_principle_model",
    )
    assert set(result.candidate_strengths) == set(PROJECTOR_KEYS)
    assert all(
        0.0 <= fraction <= 1.0
        for fraction in result.candidate_limit_fractions.values()
    )
    assert result.validation_step_mm == pytest.approx(0.025)
    assert after != before
    assert {
        key: after[key] for key in PROJECTOR_KEYS
    } == pytest.approx(result.strengths)
    assert "validated at 0.025 mm" in result.message


@pytest.mark.parametrize("target_m", (0.005, 2.5))
def test_microprobe_diffraction_camera_length_endpoints_commit(
    assembled_state, target_m
):
    state = _state_copy(assembled_state)
    apply_operating_mode_pair(state, "micro_probe", "diffraction")

    result = apply_direct_alignment(
        state, "diffraction_camera_length", target_m
    )

    assert result.success
    assert result.achieved == pytest.approx(target_m, rel=0.03)
    assert result.diffraction_conjugacy_residual <= 1.0e-3
    assert result.validation_step_mm == pytest.approx(0.025)


def test_canonical_diffraction_basis_removes_objective_field_position_term(
    assembled_state,
):
    from temsim.optics.direct_alignment import diffraction_transfer
    from temsim.physics.first_order import trace_transverse_transfer

    state = _state_copy(assembled_state)
    apply_operating_mode_pair(state, "nano_probe", "diffraction")
    result = apply_direct_alignment(
        state, "diffraction_camera_length", 0.05
    )
    assert result.success
    state.step_mm = 0.025

    raw = trace_transverse_transfer(
        state, state.sample.z_mm, state.fluorescent_screen.z_mm
    )
    canonical = diffraction_transfer(
        state,
        state.fluorescent_screen.z_mm,
        stable_axisymmetric=False,
    )

    assert np.linalg.norm(raw.j_img, ord=2) > 1.0
    assert np.linalg.norm(canonical.j_img, ord=2) <= 1.0e-3
    assert np.sqrt(abs(np.linalg.det(canonical.j_diff_m_per_rad))) == (
        pytest.approx(0.05, rel=3.0e-2)
    )


def test_tem_wave_diffraction_uses_canonical_transfer_at_inserted_screen(
    assembled_state,
):
    from temsim.physics.camera_wave import project_wave_to_recording_plane
    from temsim.physics.simulation import run

    state = _state_copy(assembled_state)
    apply_operating_mode_pair(state, "micro_probe", "diffraction")
    state.fluorescent_screen.inserted = True
    state.camera.inserted = False
    result = apply_direct_alignment(
        state, "diffraction_camera_length", 0.05
    )
    assert result.success
    axis = np.linspace(-8.0, 8.0, 32)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    wave = np.exp(-(xx * xx + yy * yy) / 16.0).astype(np.complex128)

    projection = project_wave_to_recording_plane(
        state,
        wave,
        axis,
        axis,
        0.0197,
        convergence_semiangle_rad=1.0e-4,
    )
    phase_shifted = project_wave_to_recording_plane(
        state,
        wave * np.exp(1j * 2.0 * np.pi * xx / 4.0),
        axis,
        axis,
        0.0197,
        convergence_semiangle_rad=1.0e-4,
    )

    assert projection.metrics["recording_plane_key"] == "flu_screen"
    assert projection.metrics["recording_plane_observable"] == (
        "diffraction_pattern"
    )
    assert projection.metrics["projector_transfer_input_basis"] == (
        "specimen_canonical_momentum"
    )
    assert projection.method == "collins_fft_linear_canonical_transform"
    assert projection.metrics[
        "camera_collected_zero_loss_relative_intensity"
    ] == pytest.approx(1.0, rel=1.0e-10)
    assert np.count_nonzero(projection.intensity) > 0
    assert not np.allclose(projection.intensity, phase_shifted.intensity)
    assert np.linalg.norm(
        np.asarray(projection.metrics["projector_image_map"]), ord=2
    ) <= 1.0e-3
    state.acceleration_enabled = False
    state.step_mm = 5.0
    state.history_step_mm = 5.0
    emitter = getattr(state.electron_gun, "emitter", None)
    if emitter is None:
        state.electron_gun.ray_count = 9
    else:
        emitter.ray_count = 9
    simulation = run(state)
    expected_length_m = np.sqrt(abs(np.linalg.det(
        np.asarray(
            projection.metrics["projector_diffraction_map_m_per_rad"]
        )
    )))

    assert simulation.metrics["transfer_analysis_plane_key"] == "flu_screen"
    assert simulation.metrics["transfer_analysis_plane_z_mm"] == pytest.approx(
        state.fluorescent_screen.z_mm
    )
    assert simulation.metrics["effective_camera_length_m"] == pytest.approx(
        expected_length_m, rel=1.0e-10
    )


def test_canonical_transfer_keeps_validation_step_with_stigmator_enabled(
    assembled_state,
):
    from temsim.optics.direct_alignment import diffraction_transfer

    state = _state_copy(assembled_state)
    apply_operating_mode_pair(state, "micro_probe", "diffraction")
    assert apply_direct_alignment(
        state, "diffraction_camera_length", 0.05
    ).success
    stigmator = next(
        component
        for component in state.stigmators
        if component.key == "diffraction_stigmator"
    )
    stigmator.strength_x_percent = 0.001
    stigmator.strength_y_percent = -0.0005
    state.step_mm = 0.5
    coarse_request = diffraction_transfer(
        state, state.fluorescent_screen.z_mm
    )
    state.step_mm = 0.025
    validation_request = diffraction_transfer(
        state, state.fluorescent_screen.z_mm
    )

    np.testing.assert_allclose(
        coarse_request.matrix,
        validation_request.matrix,
        rtol=0.0,
        atol=0.0,
    )


def test_axially_ordered_stem_channels_use_one_projector_state(assembled_state):
    from temsim.optics.direct_alignment import diffraction_transfer

    state = _state_copy(assembled_state)
    apply_operating_mode_pair(state, "nano_probe", "diffraction")
    result = apply_direct_alignment(
        state, "diffraction_camera_length", 0.05
    )

    assert result.success
    assert result.target_plane_key == "stem_diffraction_reference_plane"
    assert result.achieved == pytest.approx(0.05, rel=3.0e-2)
    assert result.diffraction_conjugacy_residual <= 1.0e-3
    positions = [detector.z_mm for detector in state.stem_detectors]
    assert positions == sorted(positions)
    assert len(set(positions)) == 3
    detector_lengths = []
    for detector in state.stem_detectors:
        transfer = diffraction_transfer(state, detector.z_mm)
        detector_lengths.append(
            np.sqrt(abs(np.linalg.det(transfer.j_diff_m_per_rad)))
        )
        assert np.all(np.isfinite(transfer.matrix))
    assert len(set(np.round(detector_lengths, 9))) == 3


def test_stem_channel_state_does_not_select_camera_length(assembled_state):
    from temsim.detector.stem_signal import collection_angle
    from temsim.optics.direct_alignment import (
        diffraction_reference_plane,
        diffraction_transfer,
    )

    state = _state_copy(assembled_state)
    apply_operating_mode_pair(state, "nano_probe", "diffraction")
    result = apply_direct_alignment(
        state, "diffraction_camera_length", 0.05
    )
    assert result.success

    expected_ranges = {
        "bf": (0.0, 10.3738),
        "df": (15.9894, 111.9256),
        "haadf": (60.1078, 330.5927),
    }
    assert all(
        detector.inserted and detector.readout_enabled
        for detector in state.stem_detectors
    )
    for detector in state.stem_detectors:
        angle = collection_angle(state, detector)
        assert angle.inner_mrad == pytest.approx(
            expected_ranges[detector.key][0], rel=3.0e-2, abs=0.05
        )
        assert angle.outer_mrad == pytest.approx(
            expected_ranges[detector.key][1], rel=3.0e-2, abs=0.05
        )

    reference_key, reference_z_mm = diffraction_reference_plane(state)
    reference_transfer = diffraction_transfer(state, reference_z_mm)
    for detector in state.stem_detectors:
        detector.inserted = False
        detector.readout_enabled = False
    toggled_key, toggled_z_mm = diffraction_reference_plane(state)
    toggled_transfer = diffraction_transfer(state, toggled_z_mm)

    assert toggled_key == reference_key
    assert toggled_z_mm == pytest.approx(reference_z_mm)
    assert toggled_transfer.j_img == pytest.approx(reference_transfer.j_img)
    assert toggled_transfer.j_diff_m_per_rad == pytest.approx(
        reference_transfer.j_diff_m_per_rad
    )


def test_diffraction_model_uses_the_main_screen_reference_plane(
    assembled_state,
):
    state = _state_copy(assembled_state)
    apply_operating_mode_pair(state, "nano_probe", "diffraction")
    model = _ProjectorMeasurementModel(
        state,
        direct_alignment_by_key("diffraction_camera_length"),
        step_mm=0.1,
    )

    assert model.reference_plane_key == "stem_diffraction_reference_plane"
    assert model.plane_z_mm == pytest.approx(state.fluorescent_screen.z_mm)
    vector = np.asarray([
        next(lens for lens in state.lenses if lens.key == key).percent
        for key in PROJECTOR_KEYS
    ])
    measurement, constraint = model.measure(vector)
    assert measurement.constraint_unit == "dimensionless"
    assert measurement.diffraction_conjugacy_residual == pytest.approx(
        np.linalg.norm(constraint, ord=2)
    )


def test_wrong_mode_and_out_of_range_targets_do_not_change_lenses(
    assembled_state,
):
    state = _state_copy(assembled_state)
    apply_operating_mode_pair(state, "nano_probe", "imaging")
    before = _lens_values(state)

    with pytest.raises(ValueError, match="only active in micro_probe mode"):
        apply_direct_alignment(state, "microprobe_illumination", 2.0)
    assert _lens_values(state) == before

    with pytest.raises(ValueError, match="must be between 3 and 60 mrad"):
        apply_direct_alignment(state, "nanoprobe_convergence", 60.01)
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
