import math

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.operating_modes import apply_operating_mode_pair
from temsim.physics.aperture_clipping import clip_segment
from temsim.physics.column_wall import clip_column_wall
from temsim.physics.core import complex_transfer, propagate
from temsim.physics.recording_stop import determine_tem_stop_z


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
    ("mode_key", "minimum_mrad", "maximum_mrad", "aperture_um"),
    (
        ("micro_probe", 0.0, 0.5, 50.0),
        ("nano_probe", 20.0, 40.0, 100.0),
    ),
)
def test_probe_modes_reach_the_sample_angle_with_real_aperture_clipping(
    mode_key, minimum_mrad, maximum_mrad, aperture_um
):
    state = _state()
    apply_operating_mode_pair(state, mode_key, "imaging")

    statistics = _sample_statistics(state)

    assert minimum_mrad <= statistics["semi_angle_mrad"] <= maximum_mrad
    assert state.condenser_aperture_2.radius_um == pytest.approx(aperture_um)
    assert state.condenser_aperture_3.radius_um == pytest.approx(2000.0)
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


@pytest.mark.parametrize(
    ("projector_key", "plane_attribute", "maximum_residual_m"),
    (
        ("imaging", "image_plane_z_mm", 1.0e-5),
        ("diffraction", "back_focal_plane_z_mm", 2.0e-6),
    ),
)
def test_projector_modes_relay_the_selected_objective_plane(
    projector_key, plane_attribute, maximum_residual_m
):
    state = _state()
    apply_operating_mode_pair(state, "nano_probe", projector_key)
    state.step_mm = 0.1
    source_z = getattr(state.objective_lens, plane_attribute)(
        state.beam_voltage_kv, state.sample
    )
    cached_source_z = getattr(
        state,
        (
            "objective_image_plane_z_mm"
            if plane_attribute == "image_plane_z_mm"
            else "objective_back_focal_plane_z_mm"
        ),
    )

    matrix = complex_transfer(
        state, source_z, determine_tem_stop_z(state)
    )

    assert source_z is not None
    assert cached_source_z == pytest.approx(source_z)
    assert abs(matrix[0, 1]) < maximum_residual_m
    assert abs(matrix[0, 0]) > 1.0
