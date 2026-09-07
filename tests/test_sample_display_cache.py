"""Exact bounded rendering-data reuse; no propagation calculation required."""

import os
from time import perf_counter

import numpy as np
import pytest

from temsim.specimen import geometry
from temsim.specimen.display_cache import (
    cached_cif_display,
    configure_sample_display_cache,
    sample_display_cache_info,
)


@pytest.fixture(autouse=True)
def clean_cache():
    previous = sample_display_cache_info()["budget_bytes"]
    configure_sample_display_cache(budget_bytes=0)
    configure_sample_display_cache(budget_bytes=128 * 1024**2)
    yield
    configure_sample_display_cache(budget_bytes=0)
    configure_sample_display_cache(budget_bytes=previous)


def _payload():
    return (np.arange(18, dtype=float).reshape(6, 3), np.full(6, 14),
            np.asarray([[0, 1], [1, 2]]), np.eye(3), (0., 0., 0.),
            (1., 1., 1.), False, ())


@pytest.fixture
def fake_preview(tmp_path, monkeypatch):
    path = tmp_path / "controlled.cif"
    path.write_text("first contents", encoding="utf-8")
    builds = []

    def build(*args, **kwargs):
        builds.append((args, kwargs))
        return _payload()

    monkeypatch.setattr(geometry, "_read_cif_preview_uncached", build)
    options = dict(requested_bounds_nm=(-.5, .5, -.5, .5), thickness_nm=1.,
                   specimen_size_xy_nm=(10., 10.), specimen_centre_xy_nm=(0., 0.),
                   maximum_atoms=2500, specimen_envelope_shape="disk")

    def read(**overrides):
        rotation = overrides.pop("rotation", np.eye(3))
        return geometry.read_cif_preview(str(path), rotation, **(options | overrides))

    return path, builds, read


def test_warm_preview_reuses_immutable_atoms_and_bonds(fake_preview):
    _path, builds, read = fake_preview
    before = sample_display_cache_info()
    first, second = read(), read()
    assert first is second
    assert len(builds) == 1
    for array in first[:4]:
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 99
        with pytest.raises(ValueError):
            array.setflags(write=True)
    info = sample_display_cache_info()
    assert info["hits"] - before["hits"] == 1
    assert info["misses"] - before["misses"] == 1
    assert info["entries"] == 1
    assert 0 < info["used_bytes"] <= info["budget_bytes"]


@pytest.mark.parametrize("changed", [
    dict(rotation=np.diag((-1., -1., 1.))),
    dict(requested_bounds_nm=(-.4, .5, -.5, .5)),
    dict(thickness_nm=2.), dict(specimen_size_xy_nm=(9., 9.)),
    dict(specimen_centre_xy_nm=(1., 0.)), dict(maximum_atoms=2600),
    dict(specimen_envelope_shape="rectangle"),
])
def test_every_geometry_dependency_invalidates_preview(fake_preview, changed):
    _path, builds, read = fake_preview
    first = read()
    assert read(**changed) is not first
    assert len(builds) == 2


def test_same_size_same_mtime_replacement_invalidates_and_deletion_never_hits(fake_preview):
    path, builds, read = fake_preview
    first = read()
    stat = path.stat()
    path.write_text("other contents", encoding="utf-8")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_size == stat.st_size
    assert read() is not first
    assert len(builds) == 2
    path.unlink()
    with pytest.raises(ValueError, match="CIF file does not exist"):
        read()
    assert len(builds) == 2


def test_file_changing_during_build_is_not_cached(fake_preview, monkeypatch):
    path, _builds, read = fake_preview

    def unstable(*_args, **_kwargs):
        path.write_text("replacement during build", encoding="utf-8")
        return _payload()

    monkeypatch.setattr(geometry, "_read_cif_preview_uncached", unstable)
    with pytest.raises(ValueError, match="changed while building"):
        read()
    assert sample_display_cache_info()["entries"] == 0


def test_budget_evicts_lru_and_zero_disables_without_changing_output(fake_preview):
    _path, builds, read = fake_preview
    first = read()
    size = sample_display_cache_info()["used_bytes"]
    configure_sample_display_cache(budget_bytes=size + 64)
    read(maximum_atoms=2600)
    assert sample_display_cache_info()["entries"] == 1
    read()
    assert len(builds) == 3
    configure_sample_display_cache(budget_bytes=0)
    assert sample_display_cache_info()["entries"] == 0
    for _ in range(2):
        new = read()
        for actual, expected in zip(new[:4], first[:4]):
            np.testing.assert_array_equal(actual, expected)
    assert len(builds) == 5
    assert sample_display_cache_info()["used_bytes"] == 0


def test_failed_and_oversize_builds_are_not_retained():
    configure_sample_display_cache(budget_bytes=1)
    cached_cif_display(("oversize",), _payload)
    assert sample_display_cache_info()["entries"] == 0
    with pytest.raises(RuntimeError, match="builder failed"):
        cached_cif_display(("error",), lambda: (_ for _ in ()).throw(RuntimeError("builder failed")))
    assert sample_display_cache_info()["entries"] == 0
    with pytest.raises(ValueError, match="non-negative"):
        configure_sample_display_cache(budget_bytes=-1)
    for invalid in (True, 1.5, "1024"):
        with pytest.raises(ValueError, match="non-negative integer"):
            configure_sample_display_cache(budget_bytes=invalid)


@pytest.mark.parametrize("budget", [0, 1, 128 * 1024**2])
def test_optional_cache_copy_memory_failure_keeps_successful_display(monkeypatch, budget):
    from temsim.specimen import display_cache
    configure_sample_display_cache(budget_bytes=budget)
    copy_attempts = []

    def exhausted(*_args, **_kwargs):
        copy_attempts.append(1)
        raise MemoryError("optional cache copy")

    monkeypatch.setattr(display_cache.np, "frombuffer", exhausted)
    built = _payload()
    actual = cached_cif_display(("memory pressure",), lambda: built)
    assert actual is built
    assert all(not item.flags.writeable for item in actual[:4])
    assert len(copy_attempts) == int(budget > 1)
    assert sample_display_cache_info()["entries"] == 0
    assert sample_display_cache_info()["used_bytes"] == 0


def test_builder_memory_failure_is_not_swallowed():
    def exhausted():
        raise MemoryError("atom generation failed")

    with pytest.raises(MemoryError, match="atom generation failed"):
        cached_cif_display(("failed build",), exhausted)
    assert sample_display_cache_info()["entries"] == 0


def test_actual_si_preview_cached_and_uncached_are_exact(tmp_path, monkeypatch):
    """Small synthetic Si display benchmark; excludes Qt painting and physics."""
    from temsim.specimen.atomistic import atomistic_capability
    if not atomistic_capability().available:
        pytest.skip("Atomistic CIF display backend unavailable")
    from ase.build import bulk
    from ase.io import write
    unit = bulk("Si", "diamond", a=5.43, cubic=True)
    path = tmp_path / "display-benchmark-si.cif"
    write(path, unit)
    quaternion = geometry.quaternion_from_zone_axes(np.asarray(unit.cell), (1, 1, 0), (1, -1, 0))
    rotation = geometry.quaternion_to_matrix(quaternion)
    options = dict(requested_bounds_nm=(-5., 5., -5., 5.), thickness_nm=10.,
                   specimen_size_xy_nm=(10., 10.), specimen_centre_xy_nm=(0., 0.),
                   maximum_atoms=2500, specimen_envelope_shape="disk")
    builds = []
    original = geometry._read_cif_preview_uncached

    def tracked(*args, **kwargs):
        builds.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(geometry, "_read_cif_preview_uncached", tracked)
    configure_sample_display_cache(budget_bytes=0)
    start = perf_counter()
    uncached = geometry.read_cif_preview(str(path), rotation, **options)
    off_seconds = perf_counter() - start
    configure_sample_display_cache(budget_bytes=128 * 1024**2)
    start = perf_counter()
    cold = geometry.read_cif_preview(str(path), rotation, **options)
    cold_seconds = perf_counter() - start
    start = perf_counter()
    for _ in range(5):
        warm = geometry.read_cif_preview(str(path), rotation, **options)
        assert warm is cold
    hit_seconds = (perf_counter() - start) / 5
    assert len(builds) == 2
    for expected, actual in zip(uncached[:4], warm[:4]):
        np.testing.assert_array_equal(actual, expected)
    assert warm[4:] == uncached[4:]
    print(f"Sample display data only: atoms={len(warm[1])}, bonds={len(warm[2])}; "
          f"cache off={off_seconds:.6f}s, cold={cold_seconds:.6f}s, "
          f"warm mean={hit_seconds:.6f}s (5 hits); capped window nm={warm[5]}; "
          "Qt/GL painting excluded")
