from types import SimpleNamespace

import numpy as np
import pytest

from temsim.calculation_cache import calculation_signatures
from temsim.design_explorer import (
    ProductStatus,
    evaluate_product_statuses,
    summarise_calculation_result,
)
from temsim.gui.calculation_controller import CalculationController
from temsim.optics.column import default_state
from temsim.physics.fourdstem import integrate_runtime_recording_planes
from temsim.physics.record_plane import build_record_plane_plan
from temsim.simulation_pipeline import CalculationResult
from temsim.simulation_pipeline import calculate


def test_fourdstem_cache_scopes_separate_capture_mask_and_recording_geometry():
    state = default_state()
    baseline = calculation_signatures(state)

    state.projector_lens_p2.percent += 0.5
    downstream = calculation_signatures(state)
    assert downstream["fourdstem_cube"] == baseline["fourdstem_cube"]
    assert (
        downstream["fourdstem_virtual_detectors"]
        == baseline["fourdstem_virtual_detectors"]
    )
    assert (
        downstream["fourdstem_physical_recording"]
        != baseline["fourdstem_physical_recording"]
    )

    state.sample.stem_fourdstem_virtual_outer_mrad += 5.0
    virtual = calculation_signatures(state)
    assert virtual["fourdstem_cube"] == downstream["fourdstem_cube"]
    assert virtual["stem"] == downstream["stem"]
    assert (
        virtual["fourdstem_physical_recording"]
        == downstream["fourdstem_physical_recording"]
    )
    assert (
        virtual["fourdstem_virtual_detectors"]
        != downstream["fourdstem_virtual_detectors"]
    )

    state.sample.stem_fourdstem_quantum_efficiency = 0.75
    response = calculation_signatures(state)
    assert response["stem"] == virtual["stem"]
    assert response["fourdstem_cube"] == virtual["fourdstem_cube"]
    assert (
        response["fourdstem_physical_recording"]
        == virtual["fourdstem_physical_recording"]
    )
    assert (
        response["fourdstem_virtual_detectors"]
        != virtual["fourdstem_virtual_detectors"]
    )

    state.sample.stem_fourdstem_output_path = "another-output.npy"
    output_only = calculation_signatures(state)
    assert output_only["stem"] == response["stem"]
    for key in (
        "fourdstem_cube",
        "fourdstem_virtual_detectors",
        "fourdstem_physical_recording",
    ):
        assert output_only[key] == response[key]


def test_energy_filter_cache_retains_wave_inputs_only_for_eftem():
    eels = default_state()
    eels.energy_filter.operating_mode = "eels"
    before_eels = calculation_signatures(eels)
    eels.sample.wave_defocus_nm += 7.0
    eels.image_aberrations = {"a2_mm": 0.002, "a2_azimuth_deg": 17.0}
    after_eels = calculation_signatures(eels)
    assert after_eels["energy_filter"] == before_eels["energy_filter"]

    eftem_wave = default_state()
    eftem_wave.energy_filter.operating_mode = "eftem"
    before_wave = calculation_signatures(eftem_wave)
    eftem_wave.sample.wave_defocus_nm += 7.0
    after_wave = calculation_signatures(eftem_wave)
    assert after_wave["energy_filter"] != before_wave["energy_filter"]

    eftem_aberration = default_state()
    eftem_aberration.energy_filter.operating_mode = "eftem"
    before_aberration = calculation_signatures(eftem_aberration)
    eftem_aberration.image_aberrations = {
        "a2_mm": 0.002,
        "a2_azimuth_deg": 17.0,
    }
    after_aberration = calculation_signatures(eftem_aberration)
    assert (
        after_aberration["energy_filter"]
        != before_aberration["energy_filter"]
    )


def test_design_summary_tracks_cube_and_two_derived_products_independently():
    state = default_state()
    signatures = calculation_signatures(state)
    artifact = SimpleNamespace(
        metadata={
            "provenance": {
                "fourdstem_cube_state_signature": signatures[
                    "fourdstem_cube"
                ]
            }
        }
    )
    stem_scan = SimpleNamespace(
        fourdstem_artifact=artifact,
        metrics={
            "fourdstem_virtual_detectors_signature": signatures[
                "fourdstem_virtual_detectors"
            ],
            "fourdstem_physical_recording_signature": signatures[
                "fourdstem_physical_recording"
            ],
        },
    )
    result = CalculationResult(
        simulation=SimpleNamespace(incident=object(), metrics={}),
        energy_filter=None,
        stem_scan=stem_scan,
        signatures=dict(signatures),
        calculated_products=frozenset({"column", "stem"}),
    )
    summary = summarise_calculation_result(result)
    assert {
        "fourdstem_cube",
        "fourdstem_virtual_detectors",
        "fourdstem_physical_recording",
    } <= summary.available_stage_keys
    assert {
        "fourdstem_cube",
        "fourdstem_virtual_detectors",
        "fourdstem_physical_recording",
    } <= summary.calculated_products

    state.projector_lens_p2.percent += 0.5
    changed = calculation_signatures(state)
    statuses = {
        item.stage.key: item.status
        for item in evaluate_product_statuses(
            changed,
            summary,
            requested_stage_keys={
                "fourdstem_cube",
                "fourdstem_virtual_detectors",
                "fourdstem_physical_recording",
            },
        )
    }
    assert statuses["fourdstem_cube"] is ProductStatus.REUSABLE
    assert statuses["fourdstem_virtual_detectors"] is ProductStatus.REUSABLE
    assert (
        statuses["fourdstem_physical_recording"]
        is ProductStatus.RECALCULATE
    )


def test_design_stage_request_is_explicit_wave_stem_only():
    state = default_state()
    # This branch specifically tests an explicit user opt-out, independent of
    # the application's default request for STEM wave images.
    state.sample.stem_wave_enabled = False
    state.ac_deflector.enabled = True
    state.ac_deflector.scan_enabled = True
    state.ac_deflector.wobble_enabled = False
    state.sample.stem_fourdstem_enabled = True
    state.illumination_mode = "STEM"

    without_wave = CalculationController._requested_design_stage_keys(state)
    assert "fourdstem_cube" not in without_wave

    state.sample.stem_wave_enabled = True
    with_wave = CalculationController._requested_design_stage_keys(state)
    assert {
        "fourdstem_cube",
        "fourdstem_virtual_detectors",
        "fourdstem_physical_recording",
    } <= with_wave


def test_high_accuracy_reuses_raw_cube_for_response_and_dose_changes(tmp_path):
    state = default_state()
    state.acceleration_enabled = False
    state.illumination_mode = "STEM"
    state.energy_filter.enabled = False
    state.sample.eds_enabled = False
    state.sample.specimen_preset_key = "vacuum"
    state.sample.thickness_nm = 0.0
    state.sample.wave_grid_pixels = 32
    state.sample.wave_field_of_view_angstrom = 16.0
    # Controlled periodic, zero-thickness window for capture/reuse invariants.
    # Default-column focus is not under test: it may require a much larger
    # real-space window, so do not auto-expand this deliberately small fixture.
    state.sample.wave_probe_padding_factor = 0.0
    state.sample.wave_multislice_enabled = False
    state.sample.wave_atomistic_enabled = False
    state.sample.stem_wave_enabled = True
    state.sample.stem_fourdstem_enabled = True
    state.sample.stem_fourdstem_output_path = str(tmp_path / "raw.npy")
    state.sample.stem_fourdstem_overwrite = True
    state.ac_deflector.enabled = True
    state.ac_deflector.scan_enabled = True
    state.ac_deflector.wobble_enabled = False
    state.ac_deflector.scan_pixels_x = 2
    state.ac_deflector.scan_lines = 2
    state.ac_deflector.scan_frame_period_s = 0.01
    emitter = getattr(state.electron_gun, "emitter", None)
    if emitter is None:
        state.electron_gun.ray_count = 9
    else:
        emitter.ray_count = 9

    first = calculate(state)
    artifact = first.stem_scan.fourdstem_artifact
    assert artifact is not None
    assert artifact.metadata["provenance"]["stored_frame_quantity"] == (
        "configuration-averaged diffraction probability"
    )
    assert artifact.metadata["detector_response"]["transport_neutral"] is True
    assert np.all(
        np.sum(artifact.data, axis=(-2, -1)) <= 1.0 + 1.0e-5
    )
    assert first.stem_scan.metrics["post_sample_transport_model"] == (
        "full_signed_j_img_r_plus_j_diff_theta_sequential_stops"
    )
    physical = integrate_runtime_recording_planes(
        artifact,
        None,
        build_record_plane_plan(state),
        chunk_scan_points=2,
    )
    incident = first.stem_scan.metrics["incident_sample_fraction"]
    for detector_key, live_image in first.stem_scan.fractions.items():
        assert live_image == pytest.approx(
            physical.images[detector_key] * incident,
            abs=2.0e-6,
        )

    state.sample.stem_fourdstem_response_mode = "adjustable"
    state.sample.stem_fourdstem_quantum_efficiency = 0.6
    response_changed = calculate(state, existing_result=first)
    assert response_changed.stem_scan is first.stem_scan
    assert "fourdstem_cube" in response_changed.reused_products

    state.sample.stem_fourdstem_output_path = str(tmp_path / "renamed.npy")
    path_changed = calculate(state, existing_result=response_changed)
    assert path_changed.stem_scan is response_changed.stem_scan
    assert "fourdstem_cube" in path_changed.reused_products

    state.column_current_limit_percent = 40.0
    dose_changed = calculate(state, existing_result=path_changed)
    assert dose_changed.stem_scan is not path_changed.stem_scan
    assert (
        dose_changed.stem_scan.fourdstem_artifact
        is path_changed.stem_scan.fourdstem_artifact
    )
    assert dose_changed.stem_scan.metrics["dose_reweighted_without_transport"]
    assert "stem_transport" in dose_changed.reused_products
    assert "fourdstem_cube" in dose_changed.reused_products
