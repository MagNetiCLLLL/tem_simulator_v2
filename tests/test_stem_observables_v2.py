from types import SimpleNamespace

import numpy as np
import pytest

from temsim.detector.stem_signal import (
    ELEMENTARY_CHARGE_C,
    _stem_result,
    source_current_pa,
)
from temsim.optics.column import default_state


def _probe_simulation():
    count = 4
    branch = SimpleNamespace(
        x=np.asarray((np.zeros(count), np.asarray((-1.0, -0.5, 0.5, 1.0)) * 1.0e-9)),
        y=np.asarray((np.zeros(count), np.asarray((-0.5, 0.5, -0.5, 0.5)) * 1.0e-9)),
        tx=np.asarray((np.zeros(count), np.asarray((-2.0, -1.0, 1.0, 2.0)) * 1.0e-3)),
        ty=np.asarray((np.zeros(count), np.asarray((-1.0, 1.0, -1.0, 1.0)) * 1.0e-3)),
        alive=np.ones(count, dtype=bool),
        ray_weight=np.full(count, 0.25),
        energy_offset_ev=np.asarray((-0.2, -0.1, 0.1, 0.2)),
    )
    return SimpleNamespace(incident=branch)


def test_stem_result_reports_current_expected_electrons_and_seeded_poisson():
    state = default_state()
    state.ac_deflector.scan_frame_period_s = 0.04
    state.sample.stem_poisson_enabled = True
    state.sample.stem_poisson_seed = 123
    fractions = {"bf": np.full((2, 2), 0.25)}
    simulation = _probe_simulation()

    first = _stem_result(
        state,
        simulation,
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        fractions,
        {},
        {"model": "test"},
    )
    second = _stem_result(
        state,
        simulation,
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        fractions,
        {},
        {"model": "test"},
    )

    assert first.dwell_time_s == pytest.approx(0.01)
    assert first.current_pa["bf"] == pytest.approx(
        np.full((2, 2), 0.25 * source_current_pa(state))
    )
    assert first.expected_electrons["bf"] == pytest.approx(
        first.current_pa["bf"] * 1.0e-12 * 0.01 / ELEMENTARY_CHARGE_C
    )
    assert np.array_equal(first.poisson_counts["bf"], second.poisson_counts["bf"])
    assert first.probe_state.angular_covariance_mrad2[0][0] > 0.0
    assert first.probe_state.energy_bins


def test_poisson_is_optional_but_deterministic_observables_remain():
    state = default_state()
    state.sample.stem_poisson_enabled = False
    result = _stem_result(
        state,
        _probe_simulation(),
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        {"bf": np.asarray(((0.5,),))},
        {},
        {},
    )

    assert result.poisson_counts is None
    assert result.current_pa["bf"][0, 0] > 0.0
    assert result.expected_electrons["bf"][0, 0] > 0.0

