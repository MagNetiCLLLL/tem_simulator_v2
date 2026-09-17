"""Policy/ownership fixtures, NOT CuPy wave or physical GPU qualification."""
from types import SimpleNamespace
import sys

import pytest

from temsim.physics import compute_backend as backend
from temsim.physics.backend_execution import backend_receipts, classical_backend_preflight, record_backend
from temsim.physics.ray_device_cache import RayDeviceCache


def cupy_error(name, code=None, module="cupy_backends.cuda.api.runtime"):
    cls = type(name, (RuntimeError,), {"__module__": module})
    error = cls("controlled accelerator fixture")
    error.status = code
    return error


@pytest.mark.parametrize("name,code,module,category", [
    ("CUDARuntimeError", 2, "cupy_backends.cuda.api.runtime", "out_of_memory"),
    ("CUDADriverError", 100, "cupy_backends.cuda.api.driver", "unavailable"),
    ("OutOfMemoryError", None, "cupy.cuda.memory", "out_of_memory"),
    ("CompileException", 2, "cupy.cuda.compiler", "compilation_failure"),
    ("NVRTCError", 2, "cupy_backends.cuda.libs.nvrtc", "compilation_failure"),
    ("CUDARuntimeError", 700, "cupy_backends.cuda.api.runtime", "context_corrupted"),
    ("CUDARuntimeError", 701, "cupy_backends.cuda.api.runtime", "invalid_configuration"),
    ("CuFFTError", 2, "cupy.cuda.cufft", None),
    ("CUDARuntimeError", 999, "cupy_backends.cuda.api.runtime", None),
    ("OutOfMemoryError", 2, "user_code", None),
])
def test_cupy_taxonomy_uses_api_and_code_not_blanket_class(name, code, module, category):
    assert backend.gpu_failure_category(cupy_error(name, code, module)) == category


@pytest.mark.parametrize("code,category", [(2, "out_of_memory"), (100, "unavailable"),
    (201, "context_corrupted"), (700, "context_corrupted"), (702, "context_corrupted"),
    (709, "context_corrupted"), (710, "context_corrupted"), (719, "context_corrupted"),
    (701, "invalid_configuration"), (218, "compilation_failure"), (999, None)])
def test_numba_taxonomy(code, category):
    from numba.cuda.cudadrv.driver import CudaAPIError
    assert backend.gpu_failure_category(CudaAPIError(code, "fixture")) == category


@pytest.mark.parametrize("error", [ValueError("invalid optics"), FloatingPointError("nonfinite wave"),
    OSError("corrupt input file"), RuntimeError("cancelled"), cupy_error("CompileException", 2, "cupy.cuda.compiler"),
    cupy_error("NVRTCError", 2, "cupy_backends.cuda.libs.nvrtc"), cupy_error("CUDARuntimeError", 999)])
def test_prefer_never_masks_nonresource_error_even_if_cpu_would_succeed(error):
    with pytest.raises(type(error)) as caught:
        try:
            raise error
        except Exception as original:
            backend.gpu_retry_reason(original, "Prefer GPU")
        pytest.fail("CPU success path must be unreachable")
    assert caught.value is error
    assert caught.value.__traceback__ is not None


def test_retry_preserves_original_error_evidence_and_strict_cause():
    from temsim.job_events import JobEvents, traced_job
    trace = JobEvents()
    error = cupy_error("CUDARuntimeError", 2)
    with traced_job(trace, {"request_id": "fixture"}):
        try:
            raise error
        except Exception as original:
            reason = backend.gpu_retry_reason(original, "Prefer GPU", stage="column_transport")
    row = trace.snapshot()[0]
    assert "out_of_memory" in reason
    assert row["attempted_backend"] == "CUDA GPU"
    assert row["failure_stage"] == "column_transport"
    assert "test_retry_preserves_original" in row["traceback"]
    with pytest.raises(backend.GPUExecutionError) as caught:
        backend.gpu_retry_reason(error, "Require GPU")
    assert caught.value.__cause__ is error


def test_corrupt_context_clears_and_quarantines_until_explicit_new_process(monkeypatch):
    from temsim.physics import ray_device_cache
    cache = RayDeviceCache()
    cache.entry = {"fixture": True}
    cache.retained_bytes = 99
    monkeypatch.setattr(ray_device_cache, "DEVICE_CACHE", cache)
    error = cupy_error("CUDARuntimeError", 700)
    with pytest.raises(type(error)) as caught:
        backend.gpu_retry_reason(error, "Prefer GPU")
    assert caught.value is error
    assert cache.entry is None and cache.retained_bytes == 0
    cache.clear()  # Ordinary cache clearing is not CUDA recovery.
    with pytest.raises(backend.GPUExecutionError, match="quarantined"):
        backend.cuda_capability()
    with pytest.raises(backend.GPUExecutionError, match="quarantined"):
        cache.execute(None, None, None)
    assert backend.choose_ray_backend("CPU", acceleration_enabled=False, ray_count=1) == ("CPU", None)


@pytest.mark.parametrize("function,module", [(backend.cuda_capability, "numba.cuda"), (backend.cupy_capability, "cupy")])
def test_unknown_discovery_error_propagates(function, module, monkeypatch):
    def fail(*args):
        raise ValueError("broken capability input")
    if module == "numba.cuda":
        from numba import cuda
        monkeypatch.setattr(cuda, "is_available", fail)
    else:
        monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=fail))))
    with pytest.raises(ValueError, match="broken capability"):
        function()


def test_missing_optional_cupy_is_unavailable_not_hardware_pass(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    assert not backend.cupy_capability().available


def test_missing_numba_compiler_is_not_reported_as_absent_device(monkeypatch):
    from numba import cuda
    from numba.cuda.cudadrv import driver
    monkeypatch.setattr(cuda, "is_available", lambda: False)
    monkeypatch.setattr(driver, "driver", SimpleNamespace(is_available=True))
    with pytest.raises(backend.GPUExecutionError, match="compiler"):
        backend.cuda_capability()
    text = backend.capability_detail_for_display(backend.cuda_capability)
    assert "Capability check failed" in text and "toolchain_unavailable" in text


def test_wave_policy_does_not_ignore_disabled_acceleration(monkeypatch):
    monkeypatch.setattr(backend, "cupy_capability", lambda: pytest.fail("disabled GPU must not be queried"))
    assert backend.choose_wave_backend("Prefer GPU", acceleration_enabled=False, work_items=1)[0] == "NumPy CPU"
    with pytest.raises(backend.GPUExecutionError, match="disabled"):
        backend.choose_wave_backend("Require GPU", acceleration_enabled=False, work_items=1)


def test_shared_fft_utility_propagates_compile_error_without_cpu_success(monkeypatch):
    # Isolated 4x4 mathematics only; no coherent source or image-chain bypass.
    import numpy as np
    from temsim.physics import wave_fft
    error = cupy_error("CompileException", 2, "cupy.cuda.compiler")
    def fail():
        raise error
    monkeypatch.setattr(wave_fft, "cupy_module", fail)
    with pytest.raises(type(error)) as caught:
        wave_fft.form_tem_image(np.ones((4, 4), complex), np.ones((4, 4), complex), compute_backend="CuPy CUDA")
    assert caught.value is error


def test_new_selections_are_validated_without_changing_legacy_loading():
    assert backend.normalise_backend("gpu") == "CUDA GPU"
    assert backend.normalise_backend("historical_unknown") == "Auto"
    for value in ("gpu", "historical_unknown", None, " Require GPU"):
        with pytest.raises(ValueError):
            backend.validate_backend_selection(value)
    for value in backend.BACKEND_CHOICES:
        assert backend.validate_backend_selection(value) == value


def test_cpu_only_preflight_precedes_gun_and_does_not_mutate(monkeypatch):
    from temsim.optics.column import default_state
    from temsim.physics.simulation import run
    from temsim.optics.electron_gun import source
    state = default_state()
    state.acceleration_backend = "Require GPU"
    state.vacuum_map.enabled = True
    monkeypatch.setattr(source, "trace_source_to_exit", lambda *args: pytest.fail("preflight must precede source execution"))
    before = state.to_dict()
    with pytest.raises(backend.GPUExecutionError, match="CPU-only"):
        classical_backend_preflight(state)
    assert state.to_dict() == before
    with pytest.raises(backend.GPUExecutionError, match="CPU-only"):
        run(state, resolved_layout=object())


def test_mapped_column_is_rejected_before_loading_a_field_or_gun():
    state = SimpleNamespace(acceleration_backend="Require GPU", simulation_mode="custom",
        lens_field_map_descriptors={"lens": {"path": "must-not-load.npy"}},
        lenses=[SimpleNamespace(key="lens", enabled=True)])
    with pytest.raises(backend.GPUExecutionError, match="Vector-field transport is CPU-only"):
        classical_backend_preflight(state)
    state.acceleration_backend = "Prefer GPU"
    assert classical_backend_preflight(state)[1]["capability"] == "CPU"


def test_mixed_stages_and_requested_actual_backend_are_not_collapsed():
    from temsim.calculation_performance import calculation_performance_lines
    rows = []
    with backend_receipts(rows):
        record_backend("gun", "Require GPU", "CPU")
        record_backend("column", "Require GPU", "CUDA GPU")
        record_backend("energy_filter", "Require GPU", "CPU")
    result = SimpleNamespace(performance={"backend_stages": rows})
    lines = calculation_performance_lines(result)
    assert any("gun: requested Require GPU; actual CPU" in line for line in lines)
    assert any("column: requested Require GPU; actual CUDA GPU" in line for line in lines)
    assert any("energy_filter: requested Require GPU; actual CPU" in line for line in lines)


def test_completed_worker_publishes_backend_receipt_without_mutating_old_result(monkeypatch):
    from temsim.gui import calculation_controller as module
    from temsim.optics.column import default_state
    state = default_state()
    state.sample.wave_enabled = False
    seen, errors = [], []
    def calculate(snapshot, **kwargs):
        record_backend("gun", "Auto", "CPU")
        record_backend("column", "Auto", "Numba CPU")
        return module.CalculationResult(simulation=None, energy_filter=None)
    monkeypatch.setattr(module, "calculate", calculate)
    worker = module.CalculationWorker(1, "High accuracy", state)
    worker.signals.result.connect(lambda *args: seen.append(args[2]))
    worker.signals.error.connect(lambda *args: errors.append(args[-1]))
    worker.run()
    assert not errors
    assert len(seen) == 1
    assert [row["stage"] for row in seen[0].performance["backend_stages"]] == ["gun", "column"]
    assert "not hardware qualification" in seen[0].performance["backend_scope"]


def test_compatible_reset_rebuilds_but_illegal_access_is_not_recovered(monkeypatch):
    # A context reset fixture tests ownership, not actual CUDA recovery.
    from test_ray_device_residency import FakeCUDA, ReferenceKernel, inputs
    cache, cuda = RayDeviceCache(), FakeCUDA()
    cache.execute(cuda, ReferenceKernel(), inputs())
    cuda.context.memory_manager.allocations.clear()
    cache.execute(cuda, ReferenceKernel(), inputs())
    from temsim.physics.ray_device_cache import last_device_receipt
    assert not last_device_receipt()["plan_reused"]
    class BrokenKernel:
        def __getitem__(self, config):
            def run(*args):
                raise cupy_error("CUDARuntimeError", 700)
            return run
    with pytest.raises(RuntimeError):
        cache.execute(cuda, BrokenKernel(), inputs())
    assert cache.entry is None and last_device_receipt() is None
    with pytest.raises(backend.GPUExecutionError, match="quarantined"):
        cache.execute(cuda, ReferenceKernel(), inputs())
