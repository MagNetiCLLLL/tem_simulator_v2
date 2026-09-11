"""Historical assumption superseded by HANDOFF T2-23: mapped field support must be proved."""
from temsim.optics.column import default_state
from temsim.calculation_cache import calculation_signatures
from temsim.simulation_pipeline import calculate


def test_projector_change_reprojects_saved_objective_wave_without_specimen_recalc():
    """Original pre-handoff expectation; not current gun-chain acceptance."""
    state = default_state()
    for detector in state.stem_detectors:
        detector.inserted = False
        detector.readout_enabled = False
    state.step_mm = 5.0
    state.history_step_mm = 5.0
    state.acceleration_enabled = False
    state.illumination_mode = "TEM"
    state.projector_mode = "image"
    state.sample.eds_enabled = False
    state.sample.wave_enabled = True
    state.sample.specimen_mode = "reference"
    state.sample.specimen_preset_key = "si_110"
    state.sample.thickness_nm = 2.0
    state.sample.wave_grid_pixels = 32
    state.sample.wave_field_of_view_angstrom = 16.0
    state.sample.wave_multislice_enabled = True
    state.sample.wave_atomistic_enabled = True
    state.fluorescent_screen.inserted = False
    state.camera.inserted = True
    emitter = getattr(state.electron_gun, "emitter", None)
    if emitter is None:
        state.electron_gun.ray_count = 9
    else:
        emitter.ray_count = 9
    previous = calculate(state)
    state.intermediate_lens.percent += 2.0
    updated = calculate(state, existing_result=previous)
    assert updated.wave_imaging is not previous.wave_imaging
    assert updated.wave_imaging.exit_wave is previous.wave_imaging.exit_wave
    assert updated.specimen_interactions.wave_imaging is updated.wave_imaging
    assert "wave_projection" in updated.calculated_products
    assert "wave" not in updated.calculated_products
    assert {"incident", "wave_source"} <= updated.reused_products
    assert updated.wave_imaging.metrics["projector_checkpoint_reused"] is True

def test_projector_field_map_descriptor_does_not_invalidate_upstream_products(
    tmp_path,
):
    first_map = tmp_path / "projector-first.npz"
    second_map = tmp_path / "projector-second.npz"
    first_map.write_bytes(b"first projector map")
    second_map.write_bytes(b"second projector map")
    state = default_state()
    state.lens_field_map_descriptors = {
        "projector_lens_1": {
            "source_path": str(first_map),
            "source_sha256": "first-descriptor",
            "geometry_fingerprint": "projector-geometry-a",
            "reference_excitation_percent": 100.0,
        }
    }
    before = calculation_signatures(state)

    state.lens_field_map_descriptors["projector_lens_1"] = {
        "source_path": str(second_map),
        "source_sha256": "second-descriptor",
        "geometry_fingerprint": "projector-geometry-b",
        "reference_excitation_percent": 80.0,
    }
    after = calculation_signatures(state)

    for product in (
        "incident",
        "wave_source",
        "elastic",
        "eds",
        "sample_region",
        "fourdstem_cube",
        "fourdstem_virtual_detectors",
    ):
        assert after[product] == before[product]
    assert after["request"] != before["request"]
    assert after["column"] != before["column"]
