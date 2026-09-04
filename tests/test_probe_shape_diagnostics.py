"""Actual probe shape must not be confused with a C3-only ring diagnostic."""

from types import SimpleNamespace

import numpy as np
import pytest

from temsim.gui.aberration_view import AberrationComparisonView
from temsim.optics import aberrations
from temsim.optics.column import default_state
from temsim.physics.beam_statistics import transverse_beam_statistics


def _statistics(z, *, weights=None, alive=None):
    z = np.asarray(z, dtype=complex)
    zeros = np.zeros(z.size)
    return transverse_beam_statistics(
        z.real, z.imag, zeros, zeros, weights=weights, alive=alive,
    )


def test_threefold_shape_is_detected_despite_circular_covariance():
    triangle = np.exp(2j * np.pi * np.arange(3) / 3.0) * 1.0e-9
    stats = _statistics(triangle)
    assert stats.twofold_moment == pytest.approx(0.0, abs=1.0e-14)
    assert stats.threefold_moment == pytest.approx(1.0)
    # Shape metrics must survive laboratory rotation and a translated centroid.
    transformed = triangle * 7.0 * np.exp(0.77j) + (2.0 - 5.0j) * 1.0e-9
    rotated = _statistics(transformed)
    assert rotated.twofold_moment == pytest.approx(stats.twofold_moment, abs=1.0e-14)
    assert rotated.threefold_moment == pytest.approx(stats.threefold_moment)


def test_round_and_elliptical_bundles_have_distinct_shape_moments():
    circle = np.exp(2j * np.pi * np.arange(48) / 48) * 1.0e-9
    circular = _statistics(circle)
    elliptical = _statistics(2.0 * circle.real + 1j * circle.imag)
    assert circular.twofold_moment == pytest.approx(0.0, abs=1.0e-14)
    assert circular.threefold_moment == pytest.approx(0.0, abs=1.0e-14)
    assert elliptical.twofold_moment == pytest.approx(3.0 / 5.0)
    assert elliptical.threefold_moment == pytest.approx(0.0, abs=1.0e-14)


def test_shape_moments_use_surviving_current_and_handle_a_point_probe():
    points = np.array([1.0 + 2.0j, -2.0 + 1.0j, 0.0 - 3.0j]) * 1.0e-9
    repeated = _statistics(np.repeat(points, [1, 2, 3]))
    weighted = _statistics(
        np.r_[points, 1.0 + 1.0j],
        weights=[1.0, 2.0, 3.0, 100.0], alive=[True, True, True, False],
    )
    assert weighted.twofold_moment == pytest.approx(repeated.twofold_moment)
    assert weighted.threefold_moment == pytest.approx(repeated.threefold_moment)
    zero_weight = _statistics(
        np.r_[points, 1.0e100 + 1.0e100j], weights=[1.0, 2.0, 3.0, 0.0],
    )
    assert zero_weight.threefold_moment == pytest.approx(repeated.threefold_moment)
    point = _statistics(np.zeros(5, dtype=complex))
    assert point.twofold_moment == point.threefold_moment == 0.0


@pytest.fixture
def diagnostic_state(monkeypatch):
    state = default_state()
    calls = []

    def compact_ring(candidate, system):
        calls.append(system)
        return 0.1, 2.0e-9, 1.0e-9, "distributed test reference fields"

    monkeypatch.setattr(aberrations, "_corrector_trace_ratio", compact_ring)
    return state, calls


def test_c3_diagnostic_marks_unmeasured_residuals_and_metre_units(diagnostic_state):
    state, _calls = diagnostic_state
    _before, after, diagnostic = aberrations.effective_aberration_comparison(state, "probe")
    assert diagnostic["inferred_coefficients"] == ("C3",)
    assert "A2" in diagnostic["unmeasured_coefficients"]
    assert diagnostic["ray_error_rms_unit"] == "m"
    # Numerical wave defaults remain compatible, without claiming they are measured.
    assert after.a2_mm == 0.0
    state.probe_aberrations["a2_mm"] = 0.00012
    _before, after, diagnostic = aberrations.effective_aberration_comparison(state, "probe")
    assert "A2" not in diagnostic["unmeasured_coefficients"]
    assert diagnostic["coefficient_status"]["A2"] == "configured; not fitted"
    assert after.a2_mm == 0.00012


@pytest.mark.parametrize("parameter", [
    "orientation", "quadrupole", "sample_z", "defocus", "step",
    "cc", "field_width", "polarity",
])
def test_optical_edits_invalidate_aberration_cache(diagnostic_state, parameter):
    state, calls = diagnostic_state
    initial = aberrations.effective_aberration_comparison(state, "probe")
    assert aberrations.effective_aberration_comparison(state, "probe") is initial
    assert len(calls) == 1
    if parameter == "orientation":
        next(item for item in state.corrector_elements if hasattr(item, "orientation_rad")).orientation_rad += 0.01
    elif parameter == "quadrupole":
        next(item for item in state.corrector_elements if hasattr(item, "strength_m2")).strength_m2 += 0.01
    elif parameter == "sample_z":
        state.sample.z_mm += 0.001
    elif parameter == "defocus":
        state.sample.wave_defocus_nm += 1.0
    elif parameter == "step":
        state.step_mm *= 0.5
    elif parameter == "cc":
        state.objective_lens.cc_mm += 0.1
    elif parameter == "field_width":
        state.lenses[0].a_mm *= 1.01
    elif parameter == "polarity":
        state.lenses[0].polarity *= -1
    changed = aberrations.effective_aberration_comparison(state, "probe")
    assert changed is not initial
    assert len(calls) == 2
    assert len(state._effective_aberration_cache) == 1


def test_probe_view_shows_measured_shape_without_false_a2_zero(qtbot, diagnostic_state):
    state, _calls = diagnostic_state
    triangle = np.exp(2j * np.pi * np.arange(3) / 3.0) * 1.0e-9
    branch = SimpleNamespace(
        x=triangle.real[None, :], y=triangle.imag[None, :],
        tx=np.zeros((1, 3)), ty=np.zeros((1, 3)),
        alive=np.ones(3, dtype=bool), ray_weight=np.ones(3),
    )
    view = AberrationComparisonView(fixed_system="probe")
    qtbot.addWidget(view)
    view.display_result(SimpleNamespace(
        state_snapshot=state, simulation=SimpleNamespace(incident=branch),
    ))
    view._refresh()
    assert "RMS radius 1 nm" in view.probe_summary.text()
    assert "D95 2 nm" in view.probe_summary.text()
    assert "threefold moment 1.0000" in view.probe_summary.text()
    assert "2 → 1 nm" in view.summary.text()
    a2_row = next(i for i, row in enumerate(aberrations.SYSTEM_COEFFICIENT_ROWS) if row[0] == "A2")
    assert view.table.item(a2_row, 3).text() == "—"
    assert "unmeasured" in view.table.item(a2_row, 6).text()
