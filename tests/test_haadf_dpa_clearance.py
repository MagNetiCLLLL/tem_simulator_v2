"""The enlarged default DPA opens a real HAADF path without bypassing stops."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.physics.record_plane import build_record_plane_plan, route_record_planes


def test_nanoprobe_default_dpa_passes_wave_haadf_band_and_still_blocks_higher_angles():
    state = default_state()
    apply_operating_mode_pair(state, "nano_probe", "diffraction")
    plan = build_record_plane_plan(state)
    dpa = next(plane for plane in plan.planes if plane.key == "projection_chamber_dpa_aperture")
    assert dpa.radius_mm == pytest.approx(6.0)
    assert not plan.time_dependent_deflection

    # Include the current raster's extreme specimen positions and its centre.
    # All azimuths use the full signed transfer, including image-plane terms,
    # affine deflection and every other physically inserted downstream stop.
    driver = state.ac_deflector
    half_x = .5 * (driver.scan_pixels_x - 1) * driver.scan_pixel_size_nm * 1e-9
    half_y = .5 * (driver.scan_lines - 1) * driver.scan_pixel_size_nm * 1e-9
    positions = np.asarray([(x, y) for x in (-half_x, half_x) for y in (-half_y, half_y)] + [(0., 0.)])
    radial_mrad = np.r_[np.linspace(60.2, 168.8, 13), 185., 187., 200., 300.]
    phi = np.linspace(0., 2 * np.pi, 32, endpoint=False)
    directions = np.stack((np.cos(phi), np.sin(phi)), axis=-1)
    angles = radial_mrad[:, None, None] * 1e-3 * directions[None, :, :]

    result = route_record_planes(plan, positions[:, None, None, :], angles[None, ...])
    events = {event.plane.key: event for event in result.interactions}
    in_wave_band = radial_mrad <= 168.8
    above_stop_cutoff = radial_mrad >= 187.
    assert np.all(events["haadf"].signal_mask[:, in_wave_band, :])
    assert np.all(events["haadf"].signal_mask[:, radial_mrad == 185., :])
    assert not np.any(events["haadf"].signal_mask[:, above_stop_cutoff, :])
    assert np.all(events[dpa.key].intercepted_mask[:, above_stop_cutoff, :])
    assert result.balance_error == pytest.approx(0., abs=1e-12)

    # The former hardware bore is an immutable comparison snapshot. Keep all
    # lens maps and every other stop exactly as traced from the live preset.
    old_plan = replace(plan, planes=tuple(
        replace(plane, radius_mm=.1) if plane.key == dpa.key else plane
        for plane in plan.planes
    ))
    old = route_record_planes(old_plan, positions[:, None, None, :], angles[None, ...])
    old_events = {event.plane.key: event for event in old.interactions}
    assert not np.any(old_events["haadf"].signal_mask)
    assert np.all(old_events[dpa.key].intercepted_mask)
    assert old.balance_error == pytest.approx(0., abs=1e-12)
