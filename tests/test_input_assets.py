"""Pinned input assets, immutable ownership and complete request capture."""
from threading import Event

import numpy as np
import pytest

from temsim.input_assets import InputAssetStore, freeze_numeric_array
from temsim.instrument_snapshot import encode_instrument, decode_instrument
from temsim.immutable_json import json_digest
from temsim.optics.column import default_state


def test_source_mutation_cannot_change_admitted_asset_and_pins_survive_eviction():
    original = np.arange(32768, dtype=float)
    store = InputAssetStore(budget_bytes=2*original.nbytes)
    lease = store.capture()
    record = lease.register(original)
    array = lease.array(record, original.dtype.str, original.shape, readonly=True)
    original[:] = -1
    assert array[10] == 10
    with pytest.raises(ValueError):
        array.setflags(write=True)
    store.configure(0)
    assert store.statistics()["pinned_bytes"] == array.nbytes
    assert array[10] == 10
    lease.close()
    assert store.statistics()["resident_bytes"] == 0
    assert array[10] == 10  # Result still owns the immutable backing bytes.


def test_second_capture_of_admitted_immutable_array_does_not_rehash_or_copy(monkeypatch):
    import temsim.input_assets as module
    value = freeze_numeric_array(np.arange(65536, dtype=float))
    store = InputAssetStore(budget_bytes=value.nbytes*2)
    first = store.capture()
    identity = first.register(value)
    def forbidden(*args, **kwargs):
        pytest.fail("An unchanged admitted immutable array must not be hashed again")
    monkeypatch.setattr(module, "sha256", forbidden)
    second = store.capture()
    assert second.register(value) == identity
    assert store.statistics()["resident_bytes"] == value.nbytes
    first.close()
    assert store.statistics()["pinned_bytes"] == value.nbytes
    second.close()


def test_asset_graph_roundtrip_retains_complete_values_and_object_aliases():
    state = default_state()
    state.extra_input_array = freeze_numeric_array(np.arange(32768, dtype=float))
    state.extra_input_alias = state.extra_input_array
    store = InputAssetStore(budget_bytes=1024**2)
    lease = store.capture()
    graph = encode_instrument(state, asset_store=lease)
    assert graph["schema"] == "complete-working-point-assets-v2"
    root = graph["nodes"][graph["root"]["ref"]]["attributes"]
    assert "array" not in root["extra_input_array"]
    assert len(root["extra_input_array"]["asset"]) == 64
    restored = decode_instrument(graph, assets=lease)
    assert np.array_equal(restored.extra_input_array, state.extra_input_array)
    assert restored.extra_input_array is restored.extra_input_alias
    assert json_digest(encode_instrument(restored)) == json_digest(encode_instrument(state))
    assert restored.objective_lens is next(l for l in restored.lenses if l.key == "objective_lens")
    with pytest.raises(ValueError, match="asset"):
        decode_instrument(graph)
    lease.close()


def test_request_capture_does_not_decode_or_copy_large_registered_input(monkeypatch):
    from temsim.gui import calculation_request
    from temsim.instrument_snapshot import encode_instrument as original_encode
    state = default_state()
    state.extra_input_array = freeze_numeric_array(np.arange(32768, dtype=float))
    calls = []
    from temsim import instrument_snapshot
    extension_calls = []
    def guarded(value, **kwargs):
        if value is state.extra_input_array:
            extension_calls.append(kwargs)
            assert kwargs.get("asset_store") is not None, "Unknown-input guard must not inline the bulk array"
        return original_encode(value, **kwargs)
    monkeypatch.setattr(instrument_snapshot, "encode_instrument", guarded)
    def tracked(state, **kwargs):
        calls.append(kwargs)
        return original_encode(state, **kwargs)
    monkeypatch.setattr(calculation_request, "encode_instrument", tracked)
    monkeypatch.setattr(calculation_request, "decode_instrument", lambda *_args, **_kwargs: pytest.fail("Decode belongs in the worker"))
    request = calculation_request.CapturedCalculationRequest.capture(state, "High accuracy", 9, 1.)
    assert calls[0]["asset_store"] is request._input_assets
    assert extension_calls
    state.extra_input_array = np.zeros(10)
    state.sample.thickness_nm += 2
    restored = decode_instrument(request._instrument_graph, assets=request._input_assets)
    assert restored.extra_input_array.size == 32768
    assert restored.sample.thickness_nm != state.sample.thickness_nm
    request._input_assets.close()
