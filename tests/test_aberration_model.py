import math

import numpy as np
import pytest
from PySide6.QtWidgets import QWidget

from temsim.optics.aberrations import (
    EffectiveAberrationSet,
    aberration_phase_rad,
    chromatic_defocus_mm,
    effective_aberration_comparison,
    intrinsic_lens_aberration_profile,
)
from temsim.optics.column import default_state
from temsim.gui.parameter_panel import ParameterPanel
from temsim.runtime_parameters import RuntimeTarget


def test_intrinsic_round_lens_profile_distinguishes_explicit_and_estimated():
    state = default_state()
    condenser = state.lenses[0]
    profile = intrinsic_lens_aberration_profile(
        condenser, state.beam_voltage_kv
    )
    assert condenser.cs_mm is None
    assert profile.cs_mm > 0.0
    assert profile.cc_mm > 0.0
    assert profile.status == "provisional principle model"
    assert "non-OEM" in profile.source

    objective = state.objective_lens
    explicit = intrinsic_lens_aberration_profile(
        objective, state.beam_voltage_kv
    )
    assert explicit.cs_mm == pytest.approx(1.2)
    assert explicit.cc_mm == pytest.approx(2.0)
    assert explicit.status == "configured"


def test_aberration_phase_has_exact_zero_and_c3_alpha_four_scaling():
    wavelength_angstrom = 0.025
    alpha = 0.01
    fx = np.array([alpha / wavelength_angstrom])
    fy = np.zeros_like(fx)
    zero = EffectiveAberrationSet("sample", "uncorrected")
    assert aberration_phase_rad(fx, fy, wavelength_angstrom, zero)[0] == 0.0

    c3 = EffectiveAberrationSet("sample", "uncorrected", c3_mm=1.2)
    phase = aberration_phase_rad(fx, fy, wavelength_angstrom, c3)[0]
    expected = (
        2.0
        * math.pi
        / (wavelength_angstrom * 1.0e-10)
        * 0.25
        * 1.2e-3
        * alpha**4
    )
    assert phase == pytest.approx(expected)
    half_alpha = aberration_phase_rad(
        fx * 0.5, fy, wavelength_angstrom, c3
    )[0]
    assert half_alpha / phase == pytest.approx(0.5**4)


def test_c5_and_oriented_a1_follow_order_and_azimuth():
    wavelength_angstrom = 0.025
    alpha = 0.02
    frequency = alpha / wavelength_angstrom
    c5 = EffectiveAberrationSet("sample", "uncorrected", c5_mm=4.0)
    full = aberration_phase_rad(
        np.array([frequency]), np.array([0.0]), wavelength_angstrom, c5
    )[0]
    half = aberration_phase_rad(
        np.array([frequency / 2.0]), np.array([0.0]), wavelength_angstrom, c5
    )[0]
    assert half / full == pytest.approx(0.5**6)

    a1 = EffectiveAberrationSet(
        "sample", "uncorrected", a1_mm=0.01, a1_azimuth_deg=45.0
    )
    along_axis = aberration_phase_rad(
        np.array([frequency / math.sqrt(2.0)]),
        np.array([frequency / math.sqrt(2.0)]),
        wavelength_angstrom,
        a1,
    )[0]
    across_axis = aberration_phase_rad(
        np.array([frequency / math.sqrt(2.0)]),
        np.array([-frequency / math.sqrt(2.0)]),
        wavelength_angstrom,
        a1,
    )[0]
    assert along_axis == pytest.approx(-across_axis)


def test_chromatic_defocus_uses_relative_energy_offset():
    assert chromatic_defocus_mm(2.0, 1.0, 200.0) == pytest.approx(1.0e-5)
    assert chromatic_defocus_mm(2.0, -1.0, 200.0) == pytest.approx(-1.0e-5)
    with pytest.raises(ValueError):
        chromatic_defocus_mm(-1.0, 1.0, 200.0)


def test_effective_coefficients_persist_and_inactive_corrector_is_identity():
    state = default_state()
    state.image_aberrations = {
        "a1_mm": 0.003,
        "a1_azimuth_deg": 31.0,
        "c5_mm": 4.0,
    }
    payload = state.to_dict()
    restored = type(state).from_dict(payload)
    assert restored.schema_version == 65
    assert restored.image_aberrations == state.image_aberrations
    before, after, diagnostics = effective_aberration_comparison(
        restored, "image"
    )
    for _term, value_name, angle_name in (
        ("A1", "a1_mm", "a1_azimuth_deg"),
        ("C3", "c3_mm", None),
        ("C5", "c5_mm", None),
    ):
        assert getattr(after, value_name) == getattr(before, value_name)
        if angle_name is not None:
            assert getattr(after, angle_name) == getattr(before, angle_name)
    assert diagnostics["c3_residual_ratio"] == 1.0


def test_active_probe_corrector_reduces_c3_ray_error():
    state = default_state()
    _before, _after, diagnostics = effective_aberration_comparison(
        state, "probe"
    )
    assert abs(diagnostics["c3_residual_ratio"]) < 0.2
    assert diagnostics["ray_error_rms_after"] < (
        0.2 * diagnostics["ray_error_rms_before"]
    )


def test_lens_panel_shows_estimate_instead_of_none_as_zero(qtbot):
    state = default_state()
    parent = QWidget()
    parent.state = state
    panel = ParameterPanel(parent)
    qtbot.addWidget(parent)
    qtbot.addWidget(panel)
    lens = state.lenses[0]
    target = RuntimeTarget(lens.key, lens.name, lens)
    panel.set_context(lens.name, target, None, (), None)
    assert lens.cs_mm is None
    assert panel.lens_cs.value() > 0.0
    assert panel.lens_cc.value() > 0.0
    assert panel.lens_aberration_model.currentData() == "estimate"
    assert "provisional" in panel.lens_aberration_provenance.text()

    panel.lens_aberration_model.setCurrentIndex(
        panel.lens_aberration_model.findData("explicit")
    )
    assert lens.cs_mm == pytest.approx(panel.lens_cs.value())
    assert lens.cc_mm == pytest.approx(panel.lens_cc.value())
