"""Lossless compact results preserve every retained field and executed restart."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
import json

import numpy as np
import pytest

from test_particle_section_io import executed_section
from temsim.particle_section_io import (
    _deduplicate_record_arrays, _pack_record_graph, _unpack_record_graph,
    _result_records, _same_executed_record, archive_section_result,
    checked_section_archive_info, load_section_result, save_section_result,
)


def test_deduplication_preserves_bit_patterns_shape_dtype_and_strided_data():
    bits = np.array([0, 1 << 63, 0x7ff8000000000001, 0x7ff8000000000002], dtype=np.uint64)
    floating = bits.view(np.float64)
    backing = np.arange(400_000, dtype='>f8').reshape(1000, 400)
    values = {'original': floating, 'copy': floating.copy(), 'bits': bits,
              'reshaped': floating.reshape(2, 2), 'strided': backing[:, ::2],
              'contiguous': backing[:, ::2].copy(), 'empty': np.empty((0, 4)),
              'empty_copy': np.empty((0, 4)), 'scalar': np.asarray(-0.)}
    arrays = {}
    graph = _pack_record_graph(values, arrays, deduplicate_arrays=True)
    restored = _unpack_record_graph(graph, arrays, maximum_unpacked_bytes=8*1024**2)
    for name, value in values.items():
        assert restored[name].dtype == value.dtype and restored[name].shape == value.shape
        assert restored[name].tobytes(order='C') == value.tobytes(order='C')
    assert restored['original'] is restored['copy']
    assert restored['strided'] is restored['contiguous']
    assert restored['empty'] is restored['empty_copy']
    assert restored['original'] is not restored['bits']
    assert restored['original'] is not restored['reshaped']


def test_hash_collision_does_not_merge_different_arrays(monkeypatch):
    class Collision:
        def update(self, value): pass
        def digest(self): return b'simulated hash collision'
    monkeypatch.setattr('hashlib.sha256', Collision)
    arrays = {'a': np.array([0., -0.]), 'b': np.array([-0., 0.]), 'c': np.array([0., -0.])}
    tree = {'mapping': {key: {'array': key} for key in arrays}}
    _deduplicate_record_arrays(tree, arrays)
    assert set(arrays) == {'a', 'b'}
    assert tree['mapping']['c']['array'] == 'a'
    assert tree['mapping']['b']['array'] == 'b'


def test_compact_result_preserves_complete_records_and_actual_continuation(
        executed_section, tmp_path, monkeypatch, record_property):
    from temsim.physics.particle_sections import run_particle_section
    from temsim.optics.electron_gun import source
    compact, fast = tmp_path / 'compact.temsection', tmp_path / 'fast.temsection'
    package = save_section_result(executed_section, compact)
    raw = save_section_result(executed_section, fast, compression=ZIP_STORED,
                              compresslevel=None, deduplicate_arrays=False)
    record_property('compact_bytes', compact.stat().st_size)
    record_property('uncompressed_bytes', fast.stat().st_size)
    record_property('compact_unique_arrays', len(package.arrays))
    record_property('uncompressed_unique_arrays', len(raw.arrays))
    assert len(package.arrays) < len(raw.arrays)
    assert compact.stat().st_size < fast.stat().st_size
    with ZipFile(compact) as archive:
        assert all(row.compress_type == ZIP_DEFLATED for row in archive.infolist())
        manifest = json.loads(archive.read('manifest.json'))
        assert manifest['metadata']['package_kind'] == 'optical-particle-section-package-v2'
    restored = load_section_result(compact)
    assert _same_executed_record(_result_records(restored), _result_records(executed_section))
    checked = checked_section_archive_info(executed_section, compact,
        maximum_unpacked_bytes=8*1024**3, expected_package_digest=package.digest, verify_payload=True)
    for info in (checked, restored.section_archive_info):
        assert info['compressed_size_bytes'] == compact.stat().st_size
        assert info['unpacked_size_bytes'] > info['compressed_size_bytes']
    monkeypatch.setattr(source, 'trace_source_to_exit',
                        lambda *args, **kwargs: pytest.fail('Saved physical gun must be reused'))
    target = restored.simulation.metrics['section_target_z_mm'] + 1.
    extended = run_particle_section(restored.state_snapshot, observation_stop_z_mm=target,
        tuning_component_keys=('projector_lens_1',), existing_simulation=restored.simulation)
    assert extended.metrics['section_gun_reused'] and extended.metrics['section_reused_prefix']
    assert extended.branches['000'].z[-1] == target


def test_automatic_archive_keeps_low_overhead_storage_and_actual_size(executed_section, tmp_path):
    info = archive_section_result(executed_section, tmp_path)
    path = Path(info['path'])
    with ZipFile(path) as archive:
        assert all(row.compress_type == ZIP_STORED for row in archive.infolist())
        assert info['unpacked_size_bytes'] == sum(row.file_size for row in archive.infolist())
    assert info['compressed_size_bytes'] == path.stat().st_size


@pytest.mark.parametrize('level', [1, 6])
def test_explicit_compression_level_keeps_package_content(executed_section, tmp_path, level):
    from temsim.working_point import WorkingPointCheckpoint
    path = tmp_path / f'level-{level}.temsection'
    expected = save_section_result(executed_section, path, compresslevel=level)
    restored = WorkingPointCheckpoint.read_package(path)
    assert restored.digest == expected.digest
    for name, value in expected.arrays.items():
        assert restored.arrays[name].tobytes() == value.tobytes()
