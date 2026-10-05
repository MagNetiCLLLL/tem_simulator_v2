"""Backend admission and retry control; no hardware-GPU equivalence claim."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics import compute_backend as backend


@pytest.mark.parametrize("ray_count,tuning,expected,field_route", [
    (3, False, "CPU", "electric"), (3, True, "Numba CPU", "electric"),
    (256, False, "Numba CPU", "electric"), (3, False, "CPU", "magnetic"),
])
def test_auto_missing_cuda_compiler_completes_column_on_cpu(
    monkeypatch, ray_count, tuning, expected, field_route,
):
    from numba import cuda
    from numba.cuda.cudadrv import driver
    from temsim.optics.column import default_state
    from temsim.physics import core
    from temsim.physics.backend_execution import backend_receipts

    monkeypatch.setattr(cuda, "is_available", lambda: False)
    monkeypatch.setattr(driver, "driver", SimpleNamespace(is_available=True))
    state = default_state()
    state.step_mm = 1.
    state.acceleration_enabled = True
    state.acceleration_backend = "CPU"
    state._particle_tuning = tuning
    start = state.electron_gun.exit_plane_z_mm
    plan = core.build_propagation_plan(state, start, start+1., checkpoint_z_mm=(start+1.,))
    if field_route == "magnetic":
        plan = replace(plan, electric_field=None, electric_field_identity=None,
                       electric_reference_invariant_ev=None)
        from temsim.physics.ray_device_cache import STAGE_COSTS
        monkeypatch.setattr(STAGE_COSTS, "choose", lambda *_: ("CPU", "measured CPU choice"))
    source = (np.linspace(-1e-8, 1e-8, ray_count), np.full(ray_count, 1e-5),
              np.linspace(1e-8, -1e-8, ray_count), np.full(ray_count, -1e-5))
    reference = core.execute_propagation_plan(state, plan, *source)
    state.acceleration_backend = "Auto"
    state._active_backends_used = set()
    rows = []
    with backend_receipts(rows):
        result = core.execute_propagation_plan(state, plan, *source)
    assert state.acceleration_backend == "Auto"
    assert state.active_backend.startswith(expected)
    assert "toolchain_unavailable" in state.active_backend
    assert rows[-1]["requested"] == "Auto" and rows[-1]["actual"] == expected
    assert rows[-1]["outcome"] == "returned"
    assert "toolchain_unavailable" in rows[-1]["reason"]
    if field_route == "magnetic":
        assert "measured CPU choice" in rows[-1]["reason"]
    assert result[0][-1] == start+1.
    for actual, cpu in zip(result[:5], reference[:5]):
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, cpu, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("policy", ["Require GPU", "require_gpu", " Require GPU "])
def test_strict_ray_gpu_never_claims_cpu_fallback(monkeypatch, policy):
    monkeypatch.setattr(backend, "cuda_capability", lambda: backend.BackendCapability(False, "fixture: no device"))
    with pytest.raises(backend.GPUExecutionError, match="no device"):
        backend.choose_ray_backend(policy, acceleration_enabled=True, ray_count=3)
    with pytest.raises(backend.GPUExecutionError, match="disabled"):
        backend.choose_ray_backend(policy, acceleration_enabled=False, ray_count=3)
    with pytest.raises(backend.GPUExecutionError, match="fixture oom"):
        backend.gpu_retry_reason(backend.GPUExecutionError("out_of_memory", "fixture oom"), policy)


def test_preference_and_explicit_cpu_remain_distinct(monkeypatch):
    monkeypatch.setattr(backend, "cuda_capability", lambda: backend.BackendCapability(False, "fixture: no device"))
    monkeypatch.setattr(backend, "numba_cpu_capability", lambda: backend.BackendCapability(True, "CPU"))
    actual, reason = backend.choose_ray_backend("Prefer GPU", acceleration_enabled=True, ray_count=3)
    assert actual == backend.BACKEND_NUMBA and "no device" in reason
    monkeypatch.setattr(backend, "cuda_capability", lambda: pytest.fail("CPU selection cannot query or use GPU"))
    assert backend.choose_ray_backend("CPU", acceleration_enabled=True, ray_count=10000) == (backend.BACKEND_CPU, None)


def test_numba_invalid_inputs_and_generic_errors_are_not_retryable():
    from numba.cuda.cudadrv.driver import CudaAPIError
    assert backend.gpu_failure_category(CudaAPIError(2, "fixture oom")) == "out_of_memory"
    for error in (ValueError("invalid optics"), RuntimeError("cancelled"), CudaAPIError(1, "invalid value"),
                  backend.GPUExecutionError("unsupported_physics", "fixture")):
        with pytest.raises(type(error)):
            backend.gpu_retry_reason(error, "Prefer GPU")
    with pytest.raises(backend.GPUExecutionError):
        backend.gpu_retry_reason(CudaAPIError(2, "fixture oom"), "Require GPU")


@pytest.mark.parametrize("kind", ["invalid", "resource", "strict_resource"])
@pytest.mark.parametrize("field_route", ["electric", "magnetic"])
def test_column_dispatch_retries_only_eligible_errors(monkeypatch, kind, field_route):
    from temsim.physics import core
    from temsim.physics import electrostatic_column_transport as electric_transport
    from temsim.optics.column import default_state
    state = default_state()
    state.step_mm = 1.
    state.acceleration_enabled = True
    state.acceleration_backend = "Require GPU" if kind == "strict_resource" else "Prefer GPU"
    start = state.electron_gun.exit_plane_z_mm
    plan = core.build_propagation_plan(state, start, start+1., checkpoint_z_mm=(start+1.,))
    if field_route == "magnetic":
        # Isolated magnetic-kernel dispatch fixture. Real gun-exit plans keep
        # their electric field and are covered by the other parameter value.
        plan = replace(plan, electric_field=None, electric_field_identity=None,
                       electric_reference_invariant_ev=None)
    else:
        assert electric_transport.active_electric_field(plan) is not None
    monkeypatch.setattr(core, "choose_ray_backend", lambda *_a, **_k: (backend.BACKEND_CUDA, None))
    monkeypatch.setattr(core, "NUMBA_AVAILABLE", False)
    failure = ValueError("invalid optics") if kind == "invalid" else backend.GPUExecutionError("out_of_memory", "fixture")
    calls = []
    if field_route == "electric":
        reference = electric_transport.electrostatic_column_rk4
        def electric_dispatch(inputs, *, backend, **options):
            if backend == core.BACKEND_CUDA:
                raise failure
            assert backend == core.BACKEND_CPU
            calls.append(True)
            return reference(inputs, backend=backend, **options)
        monkeypatch.setattr(electric_transport, "electrostatic_column_rk4", electric_dispatch)
    else:
        def fail(*_args, **_kwargs):
            raise failure
        monkeypatch.setattr(core, "_cuda_rk4", fail)
        reference = core._vectorised_rk4
        def cpu(*inputs, **options):
            calls.append(True)
            return reference(*inputs, **options)
        monkeypatch.setattr(core, "_vectorised_rk4", cpu)
    source = tuple(np.zeros(3) for _ in range(4))
    if kind != "resource":
        with pytest.raises(type(failure)):
            core.execute_propagation_plan(state, plan, *source)
        assert calls == []
    else:
        result = core.execute_propagation_plan(state, plan, *source)
        assert calls == [True] and state.active_backend.startswith(backend.BACKEND_CPU)
        assert "fallback: out_of_memory" in state.active_backend
        assert result[5].x_m.dtype == np.float64
