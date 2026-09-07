"""Small exact potential fixtures; no column, GPU or production image solve."""
from dataclasses import replace
from threading import Event
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.model import Sample
from temsim.physics import prepared_specimen_cache, wave_imaging
from temsim.physics.prepared_specimen_cache import (
    PreparedSpecimenCache,
    clear_prepared_specimen_cache,
    configure_prepared_specimen_cache,
    prepared_specimen_cache_info,
)
from temsim.specimen.presets import load_specimen_preset


@pytest.fixture(autouse=True)
def isolated_cache():
    original = prepared_specimen_cache_info()
    clear_prepared_specimen_cache()
    configure_prepared_specimen_cache(budget_bytes=2 * 1024**2, max_entries=8)
    yield
    clear_prepared_specimen_cache()
    configure_prepared_specimen_cache(
        budget_bytes=original["budget_bytes"], max_entries=original["max_entries"],
    )


def _state():
    sample = Sample()
    sample.specimen_mode = "virtual"
    sample.specimen_preset_key = "si_110"
    sample.envelope_shape = "rectangle"
    sample.wave_atomistic_enabled = False
    sample.wave_grid_pixels = 32
    sample.wave_field_of_view_angstrom = 16.0
    return SimpleNamespace(sample=sample, lenses=[SimpleNamespace(excitation=67.0)])


def _preset():
    return load_specimen_preset("si_110")


def _small_product():
    axis = np.arange(4, dtype=np.float64)
    potential = np.arange(16, dtype=np.float32).reshape(4, 4)
    return wave_imaging.PreparedSpecimen(
        axis, axis, (potential,), potential, None,
        {"potential_model": "synthetic_exact_test", "nested": {"values": [1, 2]}},
    )


def _assert_same_arrays(left, right):
    for name in ("x_angstrom", "y_angstrom", "mean_projected_potential_v_angstrom"):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
    for a, b in zip(left.potential_configurations_v_angstrom,
                    right.potential_configurations_v_angstrom, strict=True):
        np.testing.assert_array_equal(a, b)
    if left.slice_thicknesses_angstrom is None:
        assert right.slice_thicknesses_angstrom is None
    else:
        np.testing.assert_array_equal(left.slice_thicknesses_angstrom, right.slice_thicknesses_angstrom)


def test_actual_analytic_cold_hit_and_uncached_reference_are_exact():
    state, preset = _state(), _preset()
    expected = wave_imaging._prepare_specimen_potentials_uncached(state, preset)
    cold = wave_imaging.prepare_specimen_potentials(state, preset)
    hit = wave_imaging.prepare_specimen_potentials(state, preset)
    _assert_same_arrays(expected, cold)
    _assert_same_arrays(cold, hit)
    assert not cold.metrics["prepared_specimen_cache_hit"]
    assert hit.metrics["prepared_specimen_cache_hit"]
    assert hit.metrics["preparation_build_seconds"] == 0.0
    assert cold.metrics["preparation_seconds"] >= cold.metrics["preparation_build_seconds"] >= 0.0
    assert hit.metrics["preparation_seconds"] >= 0.0
    assert prepared_specimen_cache_info()["builds"] == 1
    assert prepared_specimen_cache_info()["hits"] == 1


def test_lens_beam_voltage_and_detector_response_are_not_potential_dependencies():
    state, preset = _state(), _preset()
    first = wave_imaging.prepare_specimen_potentials(state, preset)
    state.lenses[0].excitation = 42.0
    state.voltage_kv = 80.0
    state.sample.wave_additional_defocus_nm = 150.0
    state.sample.eds_energy_resolution_fwhm_ev = 150.0
    hit = wave_imaging.prepare_specimen_potentials(state, preset)
    assert hit.metrics["prepared_specimen_cache_hit"]
    _assert_same_arrays(first, hit)


@pytest.mark.parametrize("field,value", [
    ("thickness_nm", 12.0), ("size_x_nm", 5.0), ("centre_x_nm", 1.0),
    ("envelope_shape", "disk"), ("wave_grid_pixels", 64),
    ("wave_field_of_view_angstrom", 18.0), ("wave_slice_thickness_angstrom", 3.0),
    ("wave_frozen_phonon_enabled", True), ("wave_frozen_phonon_seed", 101),
    ("wave_frozen_phonon_configurations", 2), ("wave_frozen_phonon_sigma_angstrom", 0.02),
    ("wave_frozen_phonon_sigma_by_element_angstrom", {"Si": 0.08}),
    ("specimen_rotation_x_deg", 5.0), ("inserted", False),
])
def test_potential_input_change_invalidates(monkeypatch, field, value):
    monkeypatch.setattr(wave_imaging, "_prepare_specimen_potentials_uncached",
                        lambda *args, **kwargs: _small_product())
    state, preset = _state(), _preset()
    wave_imaging.prepare_specimen_potentials(state, preset)
    setattr(state.sample, field, value)
    changed = wave_imaging.prepare_specimen_potentials(state, preset)
    assert not changed.metrics["prepared_specimen_cache_hit"]
    assert prepared_specimen_cache_info()["builds"] == 2


@pytest.mark.parametrize("kwargs", [
    {"field_of_view_angstrom_override": 18.0},
    {"calculation_roi_centre_nm": (1.0, 0.0)},
    {"calculation_roi_bounds_nm": (-0.5, 0.5, -0.5, 0.5)},
])
def test_resolved_window_and_mask_policy_change_invalidates(monkeypatch, kwargs):
    monkeypatch.setattr(wave_imaging, "_prepare_specimen_potentials_uncached",
                        lambda *args, **kwargs: _small_product())
    state, preset = _state(), _preset()
    wave_imaging.prepare_specimen_potentials(state, preset)
    changed = wave_imaging.prepare_specimen_potentials(state, preset, **kwargs)
    assert not changed.metrics["prepared_specimen_cache_hit"]


def test_parsed_preset_content_not_only_name_is_in_key():
    state, preset = _state(), _preset()
    wave_imaging.prepare_specimen_potentials(state, preset)
    changed = replace(preset, columns=(replace(preset.columns[0], occupancy=0.5),))
    result = wave_imaging.prepare_specimen_potentials(state, changed)
    assert not result.metrics["prepared_specimen_cache_hit"]


def test_same_cif_path_same_size_modified_content_invalidates(tmp_path, monkeypatch):
    path = tmp_path / "input.cif"
    path.write_text("first", encoding="utf-8")
    monkeypatch.setattr(wave_imaging, "_prepare_specimen_potentials_uncached",
                        lambda *args, **kwargs: _small_product())
    state, preset = _state(), _preset()
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = str(path)
    first = wave_imaging.prepare_specimen_potentials(state, preset)
    path.write_text("other", encoding="utf-8")
    second = wave_imaging.prepare_specimen_potentials(state, preset)
    assert not second.metrics["prepared_specimen_cache_hit"]
    assert first.metrics["prepared_specimen_cache_key"] != second.metrics["prepared_specimen_cache_key"]


def test_cif_change_during_build_does_not_publish_wrong_key(tmp_path, monkeypatch):
    path = tmp_path / "input.cif"
    path.write_text("before", encoding="utf-8")
    def builder(*args, **kwargs):
        path.write_text("after!", encoding="utf-8")
        return _small_product()
    monkeypatch.setattr(wave_imaging, "_prepare_specimen_potentials_uncached", builder)
    state, preset = _state(), _preset()
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = str(path)
    wave_imaging.prepare_specimen_potentials(state, preset)
    assert prepared_specimen_cache_info()["entries"] == 0


def test_cached_arrays_immutable_metadata_detached_and_alias_counted_once():
    cache = PreparedSpecimenCache(budget_bytes=100_000)
    source = _small_product()
    first, hit, _, _ = cache.get_or_build("key", lambda: source)
    assert not hit
    first.metrics["nested"]["values"].append(9)
    first.metrics["stem_only"] = True
    source.mean_projected_potential_v_angstrom[:] = -9.0
    second, hit, _, _ = cache.get_or_build("key", lambda: pytest.fail("cache miss"))
    assert hit
    assert second.metrics["nested"]["values"] == [1, 2]
    assert "stem_only" not in second.metrics
    assert second.potential_configurations_v_angstrom[0] is second.mean_projected_potential_v_angstrom
    np.testing.assert_array_equal(second.mean_projected_potential_v_angstrom, np.arange(16).reshape(4, 4))
    for array in (second.x_angstrom, second.mean_projected_potential_v_angstrom):
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array.flat[0] = 9.0
    assert 0 < cache.info()["used_bytes"] <= 100_000


def test_lru_budget_reduction_zero_disable_and_existing_borrow_survives():
    cache = PreparedSpecimenCache(budget_bytes=100_000, max_entries=2)
    first, *_ = cache.get_or_build("a", _small_product)
    cache.get_or_build("b", _small_product)
    cache.get_or_build("a", lambda: pytest.fail("cache miss"))
    cache.get_or_build("c", _small_product)
    assert cache.info()["entries"] == 2
    assert cache.info()["evictions"] == 1
    assert cache.get_or_build("b", _small_product)[1] is False
    cache.configure(budget_bytes=0)
    assert cache.info()["entries"] == cache.info()["used_bytes"] == 0
    assert cache.get_or_build("a", _small_product)[1] is False
    assert cache.info()["entries"] == 0
    np.testing.assert_array_equal(first.mean_projected_potential_v_angstrom, np.arange(16).reshape(4, 4))


def test_oversized_value_and_failures_are_not_retained():
    cache = PreparedSpecimenCache(budget_bytes=1)
    assert cache.get_or_build("large", _small_product)[1] is False
    assert cache.info()["entries"] == cache.info()["used_bytes"] == 0
    def failure():
        raise RuntimeError("build failed")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="build failed"):
            cache.get_or_build("bad", failure)
    assert cache.info()["failures"] == 2
    assert cache.info()["entries"] == 0


def test_immutable_copy_memory_failure_returns_exact_result_to_owner_and_waiter(monkeypatch):
    cache = PreparedSpecimenCache(budget_bytes=100_000)
    expected = _small_product()
    source = _small_product()
    snapshot = prepared_specimen_cache._snapshot
    snapshot_modes = []
    entered, joined, release = Event(), Event(), Event()

    def fail_optional_copy(value, *, immutable_buffers):
        snapshot_modes.append(immutable_buffers)
        if immutable_buffers:
            raise MemoryError("Optional immutable buffer allocation failed")
        return snapshot(value, immutable_buffers=False)

    class JoinedFuture(Future):
        def result(self, timeout=None):
            joined.set()
            return super().result(timeout=timeout)

    def builder():
        entered.set()
        assert release.wait(5)
        return source

    monkeypatch.setattr(prepared_specimen_cache, "_snapshot", fail_optional_copy)
    monkeypatch.setattr(prepared_specimen_cache, "Future", JoinedFuture)
    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(cache.get_or_build, "a", builder)
        assert entered.wait(5)
        waiter = pool.submit(cache.get_or_build, "a", builder)
        try:
            assert joined.wait(5)
        finally:
            release.set()
        first, second = owner.result(timeout=5), waiter.result(timeout=5)

    assert snapshot_modes == [True, False]
    assert first[1] is False and second[1] is True
    assert first[2] >= 0.0 and second[2] == 0.0
    for product, _hit, _seconds, retained_bytes in (first, second):
        _assert_same_arrays(expected, product)
        assert retained_bytes == 0
        for array in prepared_specimen_cache._array_fields(product):
            if array is not None:
                assert not array.flags.writeable
        assert product.metrics == expected.metrics
    first[0].metrics["nested"]["values"].append(9)
    assert second[0].metrics == source.metrics == expected.metrics
    stats = cache.info()
    assert stats["entries"] == stats["used_bytes"] == stats["failures"] == 0
    assert stats["builds"] == stats["skipped"] == stats["waits"] == 1

    # The completed single-flight slot is removed even when no cache entry
    # could be retained; a later request can retry normally.
    assert cache.get_or_build("a", _small_product)[1] is False
    assert cache.info()["builds"] == cache.info()["skipped"] == 2


def test_physical_builder_memory_failure_is_not_optional_cache_failure(monkeypatch):
    cache = PreparedSpecimenCache(budget_bytes=100_000)

    def failure():
        raise MemoryError("Physical potential allocation failed")

    monkeypatch.setattr(
        prepared_specimen_cache, "_snapshot",
        lambda *args, **kwargs: pytest.fail("No cache snapshot should be attempted"),
    )
    with pytest.raises(MemoryError, match="Physical potential allocation failed"):
        cache.get_or_build("a", failure)
    assert cache.info()["failures"] == 1
    assert cache.info()["skipped"] == cache.info()["entries"] == 0


def test_atomistic_qualitative_fallback_is_not_cached():
    cache = PreparedSpecimenCache(budget_bytes=100_000)
    def fallback():
        value = _small_product()
        value.metrics.update(atomistic_requested=True, atomistic_applied=False,
                             atomistic_fallback_reason="Backend unavailable temporarily")
        return value
    for _ in range(2):
        assert cache.get_or_build("fallback", fallback)[1] is False
    assert cache.info()["entries"] == 0
    assert cache.info()["builds"] == 2


def test_invalid_sampling_keeps_original_actionable_diagnostic():
    state = _state()
    state.sample.wave_field_of_view_angstrom = float("inf")
    with pytest.raises(ValueError, match="Wave calculation FOV must be finite and positive"):
        wave_imaging.prepare_specimen_potentials(state, _preset())
    assert prepared_specimen_cache_info()["entries"] == 0


def test_same_key_concurrent_requests_share_single_build():
    cache = PreparedSpecimenCache(budget_bytes=100_000)
    entered, release = Event(), Event()
    def builder():
        entered.set()
        assert release.wait(5)
        return _small_product()
    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(cache.get_or_build, "a", builder)
        assert entered.wait(5)
        waiter = pool.submit(cache.get_or_build, "a", builder)
        release.set()
        first, second = owner.result(timeout=5), waiter.result(timeout=5)
    assert first[1] is False
    assert second[1] is True
    assert cache.info()["builds"] == 1
    assert first[0].metrics is not second[0].metrics


@pytest.mark.parametrize("action", ["clear", "reduce"])
def test_running_build_does_not_refill_cleared_or_reduced_cache(action):
    cache = PreparedSpecimenCache(budget_bytes=100_000)
    entered, release = Event(), Event()
    def builder():
        entered.set()
        assert release.wait(5)
        return _small_product()
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(cache.get_or_build, "a", builder)
        assert entered.wait(5)
        if action == "clear":
            cache.clear()
        else:
            cache.configure(budget_bytes=0)
        release.set()
        assert running.result(timeout=5)[0] is not None
    assert cache.info()["entries"] == cache.info()["used_bytes"] == 0


@pytest.mark.parametrize("kwargs", [{"budget_bytes": -1}, {"budget_bytes": True},
                                   {"budget_bytes": 1.5}, {"max_entries": 0}])
def test_cache_configuration_is_validated(kwargs):
    with pytest.raises(ValueError):
        PreparedSpecimenCache(**kwargs)
