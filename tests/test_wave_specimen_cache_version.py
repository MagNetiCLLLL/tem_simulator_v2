"""Wave-model corrections reject stale products without losing incident work."""

import numpy as np
import pytest

import temsim.calculation_cache as cache
from temsim.artifact_store import ArtifactStore
from temsim.calculation_manifest import capture_calculation_manifest
from temsim.optics.column import default_state


_WAVE_PRODUCTS = {
    "wave", "wave_source", "stem", "stem_transport", "fourdstem_cube",
    "fourdstem_virtual_detectors", "fourdstem_physical_recording",
}


@pytest.mark.parametrize("filter_mode", ["eels", "eftem"])
def test_specimen_algorithm_versions_all_and_only_wave_products(monkeypatch, filter_mode):
    state = default_state()
    state.energy_filter.operating_mode = filter_mode
    current = cache.calculation_signatures(state)
    monkeypatch.setattr(cache, "_WAVE_SPECIMEN_SCHEMA", "legacy-wave-extent-atom-box")
    previous = cache.calculation_signatures(state)

    changed = {key for key in current if current[key] != previous[key]}
    expected = _WAVE_PRODUCTS | {"request"}
    if filter_mode == "eftem":
        expected |= {"energy_filter"}
    assert changed == expected

    reusable = cache.matching_products(previous, current)
    assert {
        "incident", "column", "elastic", "eds", "scan_geometry",
        "scan_ray_paths", "sample_region", "sample_downstream",
    } <= reusable
    assert not reusable.intersection(changed)


def test_complete_request_cannot_bypass_wave_algorithm_invalidation(monkeypatch):
    state = default_state()
    current = cache.calculation_signatures(state)
    legacy_request = cache._digest({
        "stem_recording_schema": cache._STEM_RECORDING_SCHEMA,
        "parameters": cache._state_payload(state),
    })
    assert current["request"] != legacy_request

    # Every independently versioned wave algorithm must also protect the
    # controller's complete-result fast path, not just product-level reuse.
    monkeypatch.setattr(cache, "_WAVE_COORDINATE_SCHEMA", "legacy-corner-origin")
    previous = cache.calculation_signatures(state)
    assert current["request"] != previous["request"]
    assert current["incident"] == previous["incident"]


def test_disk_wave_products_miss_but_unchanged_incident_artifact_survives(tmp_path, monkeypatch):
    state = default_state()
    with monkeypatch.context() as legacy:
        legacy.setattr(cache, "_WAVE_SPECIMEN_SCHEMA", "legacy-wave-extent-atom-box")
        previous = capture_calculation_manifest(state)
    current = capture_calculation_manifest(state)
    assert current.solver.digest == previous.solver.digest
    assert current.digest != previous.digest

    # Tiny numeric bundles exercise actual on-disk identity/readback without
    # running a physical ray or wave calculation or touching the user's cache.
    store = ArtifactStore(tmp_path / "artifacts", quota_bytes=1_000_000)
    values = np.arange(4, dtype=np.float64)
    for key in ("incident", "wave_source", "stem"):
        store.put_array_bundle(
            previous, product_key=key,
            dependency_signature=previous.calculation_signatures[key],
            arrays={"values": values},
        )
    reopened = ArtifactStore(tmp_path / "artifacts", quota_bytes=1_000_000)
    incident = reopened.get_array_bundle(
        current, product_key="incident",
        dependency_signature=current.calculation_signatures["incident"],
    )
    assert incident is not None
    np.testing.assert_array_equal(incident.arrays["values"], values)
    assert not incident.arrays["values"].flags.writeable
    for key in ("wave_source", "stem"):
        assert reopened.get_array_bundle(
            current, product_key=key,
            dependency_signature=current.calculation_signatures[key],
        ) is None
        # Old data remains available by its original scientific provenance.
        assert reopened.get_array_bundle(
            previous, product_key=key,
            dependency_signature=previous.calculation_signatures[key],
        ) is not None
