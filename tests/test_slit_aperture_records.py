"""Captured aperture geometry and exact projections; no transport solve."""

from types import SimpleNamespace

import numpy as np
import pytest

from temsim.gui.visualization import VisualizationWorkspace
from temsim.optics.electron_gun.field_emission import FieldEmissionGun
from temsim.simulation_pipeline import aperture_stop_records


def _slit_record():
    gun = FieldEmissionGun()
    gun.install_monochromator()
    slit = gun.monochromator.slit
    slit.gap_um = 500.0
    slit.centre_offset_um = 300.0
    state = SimpleNamespace(electron_gun=gun, apertures=())
    return gun, state, next(record for record in aperture_stop_records(state)
                            if record["key"] == gun.c1_aperture.key)


def test_c1_record_captures_bound_slit_controls_without_circular_aliases():
    gun, state, record = _slit_record()
    assert record["shape"] == "two_blade_slit"
    assert record["slit_gap_mm"] == pytest.approx(0.5)
    assert record["slit_centre_x_mm"] == pytest.approx(0.3)
    assert record["bore_diameter_mm"] == gun.c1_aperture.mechanical_bore_diameter_mm
    assert record["slit_inserted"] is True
    assert "diameter_mm" not in record
    assert "offset_x_mm" not in record
    # Unused circular controls must not claim to change the physical slit.
    gun.c1_aperture.radius_mm = 0.012
    gun.c1_aperture.offset_x_mm = 1.0
    same = next(r for r in aperture_stop_records(state) if r["key"] == record["key"])
    assert same == record
    gun.monochromator.slit.gap_um = 200.0
    gun.monochromator.slit.centre_offset_um = -100.0
    changed = next(r for r in aperture_stop_records(state) if r["key"] == record["key"])
    assert (changed["slit_gap_mm"], changed["slit_centre_x_mm"]) == pytest.approx((0.2, -0.1))
    assert (record["slit_gap_mm"], record["slit_centre_x_mm"]) == pytest.approx((0.5, 0.3))


def test_ordinary_aperture_record_retains_actual_diameter_and_offsets():
    gun = FieldEmissionGun()
    gun.c1_aperture.radius_mm = 0.4
    gun.c1_aperture.offset_x_mm = 0.12
    gun.c1_aperture.offset_y_mm = -0.15
    state = SimpleNamespace(electron_gun=gun, apertures=())
    record = next(r for r in aperture_stop_records(state) if r["key"] == gun.c1_aperture.key)
    assert record["shape"] == "circular"
    assert (record["diameter_mm"], record["offset_x_mm"], record["offset_y_mm"]) == pytest.approx((0.8, 0.12, -0.15))
    assert "slit_gap_mm" not in record


@pytest.mark.parametrize("angle", [0.0, 37.0, 90.0, 145.0, 270.0])
def test_slit_projection_matches_real_bore_clipped_opening(angle):
    gun, _, record = _slit_record()
    view = SimpleNamespace(_projection_angle_deg=angle)
    centre, half_span = VisualizationWorkspace._project_aperture_opening(view, record)
    # Independently sample both bore arcs over the physical blade interval.
    x = np.linspace(0.05, 0.55, 20001)
    radius = gun.c1_aperture.mechanical_bore_diameter_mm / 2
    y = np.sqrt(radius ** 2 - x ** 2)
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    projected = np.concatenate((c * x + s * y, c * x - s * y))
    assert centre - half_span == pytest.approx(projected.min(), abs=1e-8)
    assert centre + half_span == pytest.approx(projected.max(), abs=1e-8)
    # Inside/outside cases exercise the same bound physical profile directly.
    assert gun.c1_aperture.transmission_mask(np.array([0.3, 0.7]), np.zeros(2)).tolist() == [True, False]


@pytest.mark.parametrize("inserted,gap,expected_half_span", [(False, 0.5, 3.0), (True, 0.0, 0.0)])
def test_slit_retraction_preserves_bore_and_zero_gap_is_closed(inserted, gap, expected_half_span):
    _, _, record = _slit_record()
    record.update(slit_inserted=inserted, slit_gap_mm=gap)
    view = SimpleNamespace(_projection_angle_deg=37.0)
    assert VisualizationWorkspace._project_aperture_opening(view, record) == pytest.approx((0.0, expected_half_span))


def test_unknown_aperture_shape_is_not_presented_as_a_circular_opening():
    with pytest.raises(ValueError, match="Unsupported aperture shape"):
        VisualizationWorkspace._project_aperture_opening(SimpleNamespace(), {"shape": "unknown"})
