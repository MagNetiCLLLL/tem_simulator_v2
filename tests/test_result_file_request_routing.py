"""File-request ordering only; codec/transport correctness have real archive tests."""
from types import SimpleNamespace

import pytest

from temsim.gui.calculation_controller import CalculationController


@pytest.fixture
def routing(qapp, monkeypatch, tmp_path):
    owner = CalculationController(persistent_cache_enabled=False, artifact_cache_root=tmp_path)
    monkeypatch.setattr('temsim.working_point.WorkingPointArchiveIndex.read',
        lambda *_args, **_kwargs: SimpleNamespace(unpacked_size_bytes=1024, digest='a'*64))
    pending, loaded, retained, status = [], [], [], []
    monkeypatch.setattr(owner.section_file_pool, 'start', pending.append)
    monkeypatch.setattr(owner, '_section_archive_record_current', lambda _info: True)
    monkeypatch.setattr(owner, 'retain_section_seed', retained.append)
    owner.section_loaded.connect(lambda result, info: loaded.append((result, info)))
    owner.section_archive_changed.connect(status.append)
    return owner, pending, loaded, retained, status


def test_only_latest_open_can_publish_or_retain_its_result(routing, tmp_path):
    owner, pending, loaded, retained, status = routing
    first = owner.load_section_archive(tmp_path/'first.temresult')
    second = owner.load_section_archive(tmp_path/'second.temresult')
    assert first != second and pending[0].cancel_event.is_set()
    assert not pending[1].cancel_event.is_set()
    assert pending[1].expected_package_digest == 'a'*64
    assert status[-1]['operation_token'] == second
    owner._section_file_completed(first, {'result': 'old', 'info': {'identity': 'old'}})
    assert not loaded and not retained
    owner._section_file_failed(first, {'identity': first}, 'late failure')
    assert status[-1]['status'] == 'loading'
    owner._section_file_finished(first)
    assert owner._latest_section_load_token == second
    owner._section_file_completed(second, {'result': 'current', 'info': {'identity': 'current'}})
    assert retained == ['current']
    assert loaded == [('current', {'identity': 'current', 'operation_token': second})]
    owner._section_file_finished(second)
    assert owner._latest_section_load_token is None


def test_invalid_new_file_does_not_cancel_previous_valid_request(routing, monkeypatch, tmp_path):
    owner, pending, loaded, retained, status = routing
    first = owner.load_section_archive(tmp_path/'valid.temresult')
    def invalid(*_args, **_kwargs):
        raise ValueError('invalid package')
    monkeypatch.setattr('temsim.working_point.WorkingPointArchiveIndex.read', invalid)
    with pytest.raises(ValueError, match='invalid package'):
        owner.load_section_archive(tmp_path/'invalid.temresult')
    assert owner._latest_section_load_token == first
    assert not pending[0].cancel_event.is_set()
    owner._section_file_completed(first, {'result': 'valid', 'info': {'identity': 'valid'}})
    assert retained == ['valid']


def test_current_load_failure_keeps_operation_identity(routing, tmp_path):
    owner, pending, loaded, retained, status = routing
    token = owner.load_section_archive(tmp_path/'bad.temresult')
    owner._section_file_failed(token, {'identity': 'decoded-result'}, 'checksum failed')
    assert status[-1]['identity'] == token
    assert status[-1]['operation_token'] == token
    assert status[-1]['operation'] == 'load'
    assert status[-1]['status'] == 'failed'
    assert not retained and not loaded


def test_start_failure_releases_the_file_request(routing, monkeypatch, tmp_path):
    owner, pending, loaded, retained, status = routing
    def unavailable(_worker):
        raise RuntimeError('File worker unavailable')
    monkeypatch.setattr(owner.section_file_pool, 'start', unavailable)
    with pytest.raises(RuntimeError, match='unavailable'):
        owner.load_section_archive(tmp_path/'valid.temresult')
    assert owner._latest_section_load_token is None
    assert not owner._section_file_jobs and not owner._section_file_pending
