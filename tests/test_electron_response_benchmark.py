"""Benchmark admissions fail on changed physics; cache evidence uses real code."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.magnetic_test_particle import TestElectronSettings


@pytest.fixture(scope="module")
def benchmark():
    path = Path(__file__).resolve().parents[1]/"scripts/benchmark_test_electron_responsiveness.py"
    spec = importlib.util.spec_from_file_location("electron_response_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trajectory():
    return SimpleNamespace(reason="path_limit", completed=True,
        positions_m=np.zeros((2, 3)), directions=np.tile([0., 0., 1.], (2, 1)),
        momentum_kg_m_per_s=np.tile([0., 0., 1e-22], (2, 1)), time_s=np.array([0., 1e-9]),
        path_length_m=np.array([0., .01]), kinetic_energy_ev=np.full(2, 300000.),
        electrostatic_potential_v=np.full(2, 300000.), speed_m_per_s=np.full(2, 2.3e8))


def test_reference_receipt_rejects_changed_stop_and_path(benchmark):
    expected, actual = trajectory(), trajectory()
    assert benchmark.reference_comparison(expected, actual)["accepted"]
    actual.reason = "aperture:fixture"
    assert not benchmark.reference_comparison(expected, actual)["accepted"]
    actual.reason = expected.reason
    actual.positions_m[-1, 0] = 1e-5
    assert not benchmark.reference_comparison(expected, actual)["accepted"]
    actual.positions_m = np.zeros((3, 3))
    assert not benchmark.reference_comparison(expected, actual)["accepted"]


def test_exact_lookup_executes_real_controller_cache_without_worker(benchmark, qapp):
    result = trajectory()
    receipt = benchmark.exact_gui_lookup(SimpleNamespace(), TestElectronSettings(), result, 3)
    assert receipt["cache_hits"] == 3
    assert receipt["trajectory_executions"] == 0
    assert receipt["same_result_object"]


@pytest.mark.parametrize("case", ["B01", "B02", "B03", "B04"])
def test_bounded_cases_retain_physical_tip_and_gun_operations(benchmark, case):
    state, stop = benchmark.case_inputs(case)
    gun = state.electron_gun
    assert stop == gun.exit_plane_z_mm > gun.accelerator.optical_reference_from_tip_mm
    assert gun.extractor is not None and gun.electrostatic_lens is not None
    assert gun.emitter.curvature_nm_inv == (.01 if case == "B03" else 0.)
    assert state.monochromator_installed == (case == "B04")
    if case == "B02":
        assert gun.dpa_aperture.enabled
        assert gun.dpa_aperture.offset_x_mm > gun.dpa_aperture.radius_mm
