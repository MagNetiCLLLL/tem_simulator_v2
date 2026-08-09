from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics.core import electron, propagate
from temsim.optics.magnetic_lens_aberration import (
    DEFAULT_CS_TO_FOCAL_LENGTH_RATIO,
    spherical_aberration_mm,
)


class _RoundLens:
    key = "test_round_lens"
    name = "Test Round Lens"
    enabled = True
    z_mm = 0.0
    a_mm = 10.0
    percent = 100.0
    gaussian = ()
    cs_mm = None

    def __init__(self, *, peak_t=0.002, polarity=1, focal_mm=1000.0):
        self.b0_t = float(peak_t)
        self.polarity = int(polarity)
        self._focal_mm = float(focal_mm)

    def magnetic_field_t(self, z_mm):
        z = np.asarray(z_mm, dtype=float)
        return (
            self.polarity
            * self.b0_t
            * np.exp(-0.5 * (z / self.a_mm) ** 2)
        )

    def focal_length_for_voltage_mm(self, _voltage_kv):
        return self._focal_mm


def _state(lens):
    return SimpleNamespace(
        lenses=[lens],
        stigmators=[],
        corrector_elements=[],
        beam_voltage_kv=300.0,
        step_mm=0.05,
        history_step_mm=0.05,
        acceleration_enabled=False,
        acceleration_backend="CPU",
        active_backend="CPU",
    )


def _trace(lens, *, x_m=1.0e-4, tx_rad=0.0):
    state = _state(lens)
    result = propagate(
        state,
        -60.0,
        60.0,
        np.array([x_m]),
        np.array([tx_rad]),
        np.array([0.0]),
        np.array([0.0]),
    )
    return state, result


def test_round_lens_polarity_reverses_rotation_but_not_radial_focusing():
    positive_state, positive = _trace(_RoundLens(polarity=1))
    _negative_state, negative = _trace(_RoundLens(polarity=-1))
    z_mm, x_positive, _, y_positive, _ = positive
    _, x_negative, _, y_negative, _ = negative

    assert x_negative[-1, 0] == pytest.approx(x_positive[-1, 0], rel=1e-6)
    assert y_negative[-1, 0] == pytest.approx(-y_positive[-1, 0], rel=1e-6)
    assert np.hypot(x_negative[-1, 0], y_negative[-1, 0]) == pytest.approx(
        np.hypot(x_positive[-1, 0], y_positive[-1, 0]), rel=1e-6
    )

    charge_c, momentum, _ = electron(positive_state)
    field_t = positive_state.lenses[0].magnetic_field_t(z_mm)
    expected_rotation = -charge_c * np.trapezoid(
        field_t, z_mm * 1.0e-3
    ) / (2.0 * momentum)
    measured_rotation = np.arctan2(
        y_positive[-1, 0], x_positive[-1, 0]
    )
    assert measured_rotation == pytest.approx(expected_rotation, rel=2e-5)


def test_positive_spherical_aberration_adds_radially_inward_cubic_kick():
    lens = _RoundLens(peak_t=0.0, focal_mm=10.0)
    lens.cs_mm = 1.0
    state = _state(lens)
    state.step_mm = state.history_step_mm = 0.1
    radius_m = 1.0e-3

    _, _, tx, _, ty = propagate(
        state,
        -1.0,
        1.0,
        np.array([radius_m]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
    )

    coefficient_m3 = (lens.cs_mm * 1.0e-3) / (10.0e-3) ** 4
    assert tx[-1, 0] == pytest.approx(
        -coefficient_m3 * radius_m**3, rel=1e-6
    )
    assert ty[-1, 0] == pytest.approx(0.0, abs=1e-12)


def test_zero_cs_preserves_the_paraxial_ray_direction():
    lens = _RoundLens(peak_t=0.0, focal_mm=10.0)
    lens.cs_mm = 0.0
    _state_value, (_z, _x, tx, _y, ty) = _trace(
        lens, x_m=1.0e-3, tx_rad=2.0e-3
    )

    assert tx[-1, 0] == pytest.approx(2.0e-3)
    assert ty[-1, 0] == pytest.approx(0.0)


def test_unspecified_round_lens_cs_uses_positive_physical_estimate():
    lens = _RoundLens(peak_t=0.0, focal_mm=10.0)

    assert spherical_aberration_mm(lens, 300.0) == pytest.approx(
        DEFAULT_CS_TO_FOCAL_LENGTH_RATIO * 10.0
    )
