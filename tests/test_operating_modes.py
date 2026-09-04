import math

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.optics.column import default_state
from temsim.operating_modes import apply_operating_mode_pair, mode_by_key
from temsim.physics.aperture_clipping import clip_segment
from temsim.physics.column_wall import clip_column_wall
from temsim.physics.core import complex_transfer, propagate
from temsim.physics.recording_stop import tem_camera_plane_z


def _state():
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    return state


def _sample_statistics(state):
    state.step_mm = 0.1
    emitted = state.electron_gun.trace_to_exit().exit_bundle
    z, x, tx, y, ty = propagate(
        state,
        state.electron_gun.exit_plane_z_mm,
        state.sample.z_mm,
        emitted.x_m,
        emitted.tx_rad,
        emitted.y_m,
        emitted.ty_rad,
        energy_offset_ev=emitted.energy_offset_ev,
    )
    alive = emitted.alive.copy()
    blocked = np.full(alive.size, np.nan)
    blocked_key = [""] * alive.size
    alive, blocked, blocked_key = clip_segment(
        state, z, x, y, alive, blocked, blocked_key
    )
    alive, blocked, blocked_key = clip_column_wall(
        state, z, x, y, alive, blocked, blocked_key
    )
    weights = np.maximum(np.asarray(emitted.weight)[alive], 0.0)
    weights /= np.sum(weights)

    def centred(values):
        selected = np.asarray(values, dtype=float)[alive]
        return selected - np.sum(weights * selected)

    angular_radius = np.hypot(centred(tx[-1]), centred(ty[-1]))
    spatial_radius = np.hypot(centred(x[-1]), centred(y[-1]))
    return {
        "surviving": int(np.count_nonzero(alive)),
        "semi_angle_mrad": float(np.quantile(angular_radius, 0.95) * 1e3),
        "rms_angle_mrad": math.sqrt(
            float(np.sum(weights * angular_radius**2))
        ) * 1e3,
        "rms_radius_nm": math.sqrt(
            float(np.sum(weights * spatial_radius**2))
        ) * 1e9,
    }


@pytest.mark.parametrize(
    ("mode_key", "minimum_mrad", "maximum_mrad", "diameter_um"),
    (
        ("micro_probe", 0.0, 0.3, 100.0),
        ("nano_probe", 3.0, 60.0, 100.0),
    ),
)
def test_probe_modes_reach_the_sample_angle_with_real_aperture_clipping(
    mode_key, minimum_mrad, maximum_mrad, diameter_um
):
    state = _state()
    apply_operating_mode_pair(state, mode_key, "imaging")

    statistics = _sample_statistics(state)

    assert minimum_mrad <= statistics["semi_angle_mrad"] <= maximum_mrad
    assert state.condenser_aperture_2.diameter_um == pytest.approx(diameter_um)
    assert state.condenser_aperture_3.diameter_um == pytest.approx(4000.0)
    transfer_lenses = {
        lens.key: lens.percent
        for lens in state.lenses
        if lens.key in {
            "probe_tl22_lens", "probe_tl21_lens", "probe_tl12_lens"
        }
    }
    assert set(transfer_lenses.values()) == {60.0}
    transfer_fields = {
        lens.key: lens.scale()
        for lens in state.lenses
        if lens.key in transfer_lenses
    }
    assert transfer_fields == pytest.approx({
        "probe_tl22_lens": 0.31809425,
        "probe_tl21_lens": 0.29864759,
        "probe_tl12_lens": 0.33,
    })
    if mode_key == "nano_probe":
        assert statistics["rms_radius_nm"] < 2.0


def test_imaging_mode_relays_the_selected_objective_image_plane():
    state = _state()
    apply_operating_mode_pair(state, "nano_probe", "imaging")
    state.step_mm = 0.1
    source_z = state.objective_lens.image_plane_z_mm(
        state.beam_voltage_kv, state.sample
    )

    matrix = complex_transfer(
        state, source_z, tem_camera_plane_z(state)
    )

    assert source_z is not None
    assert state.objective_image_plane_z_mm == pytest.approx(source_z)
    assert abs(matrix[0, 1]) < 1.0e-5
    assert abs(matrix[0, 0]) > 1.0


@pytest.mark.parametrize(
    "column",
    ("C2", "C3", "C3 + Probe Corrector"),
)
def test_stored_imaging_relay_is_valid_for_each_declared_column(column):
    state = default_state()
    AssemblyCatalog().apply(
        state,
        AssemblySelection("FEG", column, "Energy Filter"),
    )
    imaging = mode_by_key("imaging")
    lenses = {lens.key: lens for lens in state.lenses}
    lenses["objective_lens"].percent = 68.9801
    for key, values in imaging.devices.items():
        lenses[key].percent = float(values["percent"])
    state.sync_objective()
    state.step_mm = 0.1
    source_z = state.objective_lens.image_plane_z_mm(
        state.beam_voltage_kv, state.sample
    )

    matrix = complex_transfer(
        state, source_z, tem_camera_plane_z(state)
    )

    assert source_z is not None
    assert abs(matrix[0, 1]) < 1.0e-5
    assert abs(matrix[0, 0]) == pytest.approx(
        imaging.targets["achieved_plane_magnification"],
        rel=5.0e-4,
    )


def test_diffraction_mode_targets_the_main_screen_reference_plane():
    from temsim.optics.direct_alignment import diffraction_transfer

    state = _state()
    apply_operating_mode_pair(state, "nano_probe", "diffraction")
    transfer = diffraction_transfer(state, state.fluorescent_screen.z_mm)

    assert np.linalg.norm(transfer.j_img, ord=2) < 1.0e-3
    assert np.sqrt(abs(np.linalg.det(transfer.j_diff_m_per_rad))) == (
        pytest.approx(0.05, rel=3.0e-2)
    )


def test_probe_mode_presets_keep_tem_and_stem_recording_paths_exclusive():
    state = _state()

    apply_operating_mode_pair(state, "micro_probe", "imaging")
    assert all(not detector.inserted for detector in state.stem_detectors)
    assert all(
        not detector.readout_enabled for detector in state.stem_detectors
    )
    assert state.camera.inserted is True
    assert state.fluorescent_screen.inserted is False

    apply_operating_mode_pair(state, "nano_probe", "diffraction")
    assert all(detector.inserted for detector in state.stem_detectors)
    assert all(detector.readout_enabled for detector in state.stem_detectors)
    assert state.fluorescent_screen.inserted is False
    assert state.camera.inserted is False
