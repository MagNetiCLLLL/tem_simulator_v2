"""Small controller regressions; synthetic maps keep optical solves out of UI tests."""

from types import SimpleNamespace

import numpy as np
import pytest

from temsim.gui import conjugate_plane_panel as module


def _search(reference=10., z=30., kind="image"):
    candidate = SimpleNamespace(
        z_mm=z, kind=kind, residual_m_per_rad=1e-9,
        magnifications=(2., 1.99), rotation_deg=12., mirrored=True,
        detail="B block checked; finite particle transmission has not been evaluated.",
    )
    return SimpleNamespace(
        reference_z_mm=reference, candidates=(candidate,), detail="Captured canonical transfer.",
        lower_z_mm=1., upper_z_mm=100.,
    )


@pytest.fixture
def panel(qtbot, monkeypatch):
    widget = module.ConjugatePlanePanel()
    qtbot.addWidget(widget)
    widget.submitted = []
    monkeypatch.setattr(widget.pool, "start", widget.submitted.append)
    monkeypatch.setattr(widget.pool, "clear", lambda: None)
    monkeypatch.setattr(widget.pool, "waitForDone", lambda _timeout: True)
    yield widget
    widget.shutdown()


def _ready(panel, z=10.):
    panel.set_result(SimpleNamespace(name="captured"))
    panel.select_z(z)


def _solve_last(panel):
    panel.submitted[-1].run()


def test_cursor_visibility_and_reference_pinning_are_lazy(panel):
    _ready(panel)
    panel.show()
    panel.select_z(20.)
    panel.use_selected_z()
    panel.select_z(30.)
    assert panel._reference_z_mm == 20.
    assert not panel.submitted
    assert panel.find_button.isEnabled()
    assert panel.find_button.property("calculationAction") is True
    assert panel.maximumHeight() <= 260
    assert "selected 30" in panel.reference.text()


def test_first_find_pins_z_builds_once_and_reuses_atlas_and_exact_cache(panel, monkeypatch):
    calls = []
    atlas = object()
    monkeypatch.setattr(module, "build_conjugate_atlas", lambda result, cancelled: calls.append("build") or atlas)
    monkeypatch.setattr(module, "find_conjugate_planes", lambda received, z, cancelled: calls.append((received, z)) or _search(z))
    _ready(panel)
    panel.find_planes()
    _solve_last(panel)
    assert panel._reference_z_mm == 10.
    assert panel.table.rowCount() == 1
    assert panel._atlas is atlas
    panel.find_planes()
    assert len(panel.submitted) == 1
    panel.select_z(20.)
    panel.use_selected_z()
    panel.find_planes()
    _solve_last(panel)
    assert calls == ["build", (atlas, 10.), (atlas, 20.)]
    assert list(panel._cache) == [10., 20.]


def test_candidate_jump_preserves_reference_and_reports_theoretical_scope(panel, monkeypatch):
    monkeypatch.setattr(module, "build_conjugate_atlas", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(module, "find_conjugate_planes", lambda _atlas, z, **_kwargs: _search(z, z=5.))
    _ready(panel)
    panel.find_planes()
    _solve_last(panel)
    jumped = []
    panel.plane_selected.connect(jumped.append)
    panel.plane_selected.connect(panel.select_z)
    panel._row_clicked(0, 0)
    assert jumped == [5.]
    assert panel._selected_z_mm == 5.
    assert panel._reference_z_mm == 10.
    assert len(panel.submitted) == 1
    panel.table.setCurrentCell(0, 0)
    assert "Upstream reciprocal" in panel.details.toPlainText()
    assert "not a virtual image" in panel.details.toPlainText()
    assert "does not prove particles reach" in panel.details.toPlainText()
    assert panel.table.item(0, 1).text() == "-5"
    tooltip = panel.table.item(0, 2).toolTip()
    assert "Reference: pinned reference plane" in tooltip
    assert "rank(A) = 2" in tooltip
    assert "1e-09" not in tooltip
    panel.mark_stale()
    assert "previous result (stale)" in panel.table.item(0, 2).toolTip()


def test_cancelled_late_search_cannot_cache_but_completed_atlas_is_reusable(panel):
    _ready(panel)
    panel.find_planes()
    old = panel._worker
    panel.cancel()
    atlas = object()
    panel._atlas_ready(old.result_generation, atlas)
    panel._solved(old.generation, old.result_generation, _search())
    assert panel._atlas is atlas
    assert not panel._cache
    assert panel.table.rowCount() == 0
    assert old.event.is_set()
    panel._finished(old.generation)
    panel.find_planes()
    assert panel._worker.atlas is atlas


def test_republication_and_stale_inputs_reject_late_atlas_search_and_error(panel):
    _ready(panel)
    panel.find_planes()
    old = panel._worker
    # Even publishing the same result identity invalidates old captured work.
    panel.set_result(panel._result)
    panel._atlas_ready(old.result_generation, object())
    panel._solved(old.generation, old.result_generation, _search())
    panel._failed(old.generation, "old failure")
    assert panel._atlas is None
    assert not panel._cache
    assert "old failure" not in panel.status.text()
    panel._finished(old.generation)
    panel.find_planes()
    current = panel._worker
    panel.mark_stale()
    panel._atlas_ready(current.result_generation, object())
    panel._solved(current.generation, current.result_generation, _search())
    assert panel._atlas is None
    assert not panel._cache
    assert not panel.find_button.isEnabled()
    assert "inputs changed" in panel.status.text()


def test_pending_requests_are_serialized_and_only_latest_reference_runs(panel, monkeypatch):
    calls = []
    monkeypatch.setattr(module, "build_conjugate_atlas", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(module, "find_conjugate_planes", lambda _atlas, z, **_kwargs: calls.append(z) or _search(z))
    _ready(panel)
    panel.find_planes()
    first = panel._worker
    for z in (20., 40.):
        panel.select_z(z)
        panel.use_selected_z()
        panel.find_planes()
    assert len(panel.submitted) == 1
    assert panel._pending[-1] == 40.
    first.run()  # Cancelled before entry; releases the slot for latest request.
    assert len(panel.submitted) == 2
    _solve_last(panel)
    assert calls == [40.]
    assert panel._search.reference_z_mm == 40.
    assert list(panel._cache) == [40.]


def test_worker_cancel_during_build_never_publishes_incomplete_atlas(panel, monkeypatch):
    def build(*_args, **_kwargs):
        panel._worker.event.set()
        return object()

    monkeypatch.setattr(module, "build_conjugate_atlas", build)
    monkeypatch.setattr(module, "find_conjugate_planes", lambda *_args, **_kwargs: pytest.fail("Cancelled build must not search"))
    _ready(panel)
    panel.find_planes()
    _solve_last(panel)
    assert panel._atlas is None
    assert not panel._cache


def test_worker_failure_is_visible_and_retains_completed_atlas(panel, monkeypatch):
    atlas = object()
    monkeypatch.setattr(module, "build_conjugate_atlas", lambda *_args, **_kwargs: atlas)
    def fail(*_args, **_kwargs):
        raise ValueError("Reference outside supported column")
    monkeypatch.setattr(module, "find_conjugate_planes", fail)
    _ready(panel)
    panel.find_planes()
    _solve_last(panel)
    assert "Reference outside supported column" in panel.status.text()
    assert panel._atlas is atlas
    assert not panel._cache
    assert panel._worker is None


def test_search_signal_clears_markers_for_new_reference_stale_clear_and_shutdown(panel):
    seen = []
    panel.search_changed.connect(seen.append)
    _ready(panel)
    panel.use_selected_z()
    result = _search()
    panel._solved(panel._generation, panel._result_generation, result)
    assert seen[-1] is result
    panel.select_z(20.)
    panel.use_selected_z()
    assert seen[-1] is None
    panel.mark_stale()
    assert seen[-1] is None
    panel.clear()
    assert panel._result is None
    assert panel._selected_z_mm is None
    assert panel._reference_z_mm is None
    panel.shutdown()
    panel.set_result(SimpleNamespace())
    panel.select_z(10.)
    panel._atlas_ready(panel._result_generation, object())
    panel._solved(panel._generation, panel._result_generation, _search())
    assert panel._result is None and panel._atlas is None
    assert not panel._cache
    assert not panel.find_button.isEnabled()


@pytest.mark.parametrize("invalid", (None, True, "20", float("nan"), float("inf")))
def test_invalid_z_does_not_start_or_pin_work(panel, invalid):
    panel.set_result(SimpleNamespace())
    panel.select_z(invalid)
    panel.find_planes()
    assert not panel.submitted
    assert panel._reference_z_mm is None


def test_display_cache_bounded_and_uses_exact_reference_values(panel):
    _ready(panel)
    for z in range(panel.CACHE_LIMIT + 3):
        panel.select_z(float(z))
        panel.use_selected_z()
        panel._solved(panel._generation, panel._result_generation, _search(float(z)))
    assert len(panel._cache) == panel.CACHE_LIMIT
    assert 0. not in panel._cache
    near = float(z) + 1e-10
    panel.select_z(near)
    panel.use_selected_z()
    panel.find_planes()
    assert panel._worker.reference_z_mm == near


def test_coordinated_worker_delivers_results_and_reuses_atlas(qtbot, monkeypatch):
    calls = []
    atlas = object()
    monkeypatch.setattr(module, "build_conjugate_atlas", lambda *_args, **_kwargs: calls.append("build") or atlas)
    monkeypatch.setattr(module, "find_conjugate_planes", lambda received, z, **_kwargs: calls.append(z) or _search(z))
    widget = module.ConjugatePlanePanel()
    qtbot.addWidget(widget)
    try:
        _ready(widget)
        widget.find_planes()
        qtbot.waitUntil(lambda: widget._search is not None and widget._worker is None, timeout=10000)
        widget.select_z(20.)
        widget.use_selected_z()
        widget.find_planes()
        qtbot.waitUntil(lambda: widget._search is not None and widget._worker is None, timeout=10000)
        assert widget._search.reference_z_mm == 20.
        assert calls == ["build", 10., 20.]
        assert widget._atlas is atlas
        assert not widget.cancel_button.isEnabled()
    finally:
        assert widget.shutdown()


def _path_branch(name="incident", z=(0., 10.), stops=(np.nan, 5., np.nan, np.nan)):
    rows, columns = len(z), len(stops)
    return SimpleNamespace(
        name=name, z=np.asarray(z), x=np.zeros((rows, columns)), y=np.zeros((rows, columns)),
        blocked_z=np.asarray(stops), blocked_key=["aperture_a"] * columns,
        source_ray_id=np.arange(columns), ray_weight=np.array([1., 1., 0., 1.])[:columns], weight=1.,
    )


def _path_result(incident=None, branches=()):
    return SimpleNamespace(
        state_snapshot=SimpleNamespace(sample=SimpleNamespace(z_mm=10.), recording_planes=()),
        simulation=SimpleNamespace(
            incident=incident if incident is not None else _path_branch(),
            branches={str(i): branch for i, branch in enumerate(branches)},
        ),
        signatures={"sample_downstream": "valid"},
        aperture_stops=[{"key": "aperture_a", "name": "Aperture A"}],
    )


def test_recorded_context_counts_saved_paths_and_uses_actual_stop_positions():
    from temsim.gui.conjugate_plane_context import RecordedConjugateContext

    branch = _path_branch()
    branch.x[1, 3] = np.nan
    context = RecordedConjugateContext()
    context.set_result(_path_result(branch))
    detail = context.at(8.)
    assert "1 reach this Z" in detail
    assert "1 have a recorded stop before Z" in detail
    assert "1 unavailable" in detail
    assert "1 zero-weight support paths excluded" in detail
    assert "not electron counts or transmission percentages" in detail
    assert "Aperture A [aperture_a]" in detail
    assert "Z 5 mm" in detail
    assert "hardware" in detail
    assert "Source: Incident" in detail


def test_recorded_context_missing_history_is_unavailable_not_zero():
    from temsim.gui.conjugate_plane_context import RecordedConjugateContext

    context = RecordedConjugateContext()
    context.set_result(_path_result(SimpleNamespace(name="missing")))
    text = context.at(8.)
    assert "path reach unavailable" in text
    assert "0 reach" not in text
    context.set_result(_path_result())
    assert "Path reach unavailable: no retained downstream histories" in context.at(12.)


def test_recorded_context_uses_valid_detailed_exit_and_keeps_valid_empty_exit():
    from temsim.gui.conjugate_plane_context import RecordedConjugateContext
    from temsim.specimen.downstream_transport import GeometricSpecimenExit

    incident = _path_branch(stops=(np.nan,) * 4)
    optical = _path_branch(name="optical", z=(10., 30.), stops=(11., 11., 11., 11.))
    detailed = _path_branch(name="detailed", z=(10., 30.), stops=(15., np.nan, np.nan, np.nan))
    result = _path_result(incident, (optical,))
    result.specimen_exit = GeometricSpecimenExit(
        (detailed,), {"tracked_downstream_source_probability": 1., "inelastic_absorbed_source_probability": 0.}, "valid",
    )
    context = RecordedConjugateContext()
    context.set_result(result)
    detail = context.at(20.)
    assert "Source: Specimen exit" in detail
    assert "detailed: 2 reach" in detail
    assert "Z 15 mm" in detail
    assert "optical:" not in detail and "Z 11 mm" not in detail
    result.specimen_exit = GeometricSpecimenExit(
        (), {"tracked_downstream_source_probability": 0., "inelastic_absorbed_source_probability": 1.}, "valid",
    )
    context.set_result(result)
    detail = context.at(20.)
    assert "0 forward path representatives" in detail
    assert "no optical-reference fallback" in detail
    assert "optical:" not in detail and "Z 11 mm" not in detail


def test_recorded_context_queries_only_row_selection_and_caches_per_publication(panel, monkeypatch):
    import temsim.gui.conjugate_plane_context as context_module

    reads = []
    original = context_module._branch_counts
    monkeypatch.setattr(context_module, "_branch_counts", lambda branch, z: reads.append(z) or original(branch, z))
    result = _path_result()
    panel.set_result(result)
    panel.select_z(1.)
    panel.use_selected_z()
    panel._solved(panel._generation, panel._result_generation, _search(1., z=8.))
    for z in (2., 3., 4.):
        panel.select_z(z)
    assert reads == []
    panel.table.setCurrentCell(0, 0)
    assert reads == [8.]
    assert "Recorded path representatives" in panel.details.toPlainText()
    panel._row_changed(0, 0, 0, 0)
    assert reads == [8.]
    panel.set_result(result)
    panel.use_selected_z()
    panel._solved(panel._generation, panel._result_generation, _search(4., z=8.))
    panel.table.setCurrentCell(0, 0)
    assert reads == [8., 8.]


def test_recorded_context_does_not_treat_numerical_stops_as_hardware():
    from temsim.gui.conjugate_plane_context import RecordedConjugateContext

    branch = _path_branch()
    branch.blocked_key[1] = "field_domain"
    context = RecordedConjugateContext()
    context.set_result(_path_result(branch))
    assert "field_domain | numerical" in context.at(8.)
