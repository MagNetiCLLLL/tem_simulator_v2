"""Timing display uses current request metadata and cannot initiate physics."""
from types import SimpleNamespace
import pytest

from temsim.gui.calculation_timing_view import CalculationTimingView, timing_details


def result(**overrides):
    values = dict(performance={"pipeline_seconds": 4., "gun_seconds": .2,
        "stages": [{"stage": "Transporting emitted electrons to the selected section", "seconds": 3.}],
        "timing_scope": "Executed classical particle section"},
        simulation=SimpleNamespace(metrics={"section_gun_reused": True,
            "section_target_z_mm": 1200., "section_resume_z_mm": 1000.,
            "section_reused_prefix": True, "section_reuse_reason": "compatible_executed_prefix"}))
    values.update(overrides)
    return SimpleNamespace(**values)


def test_current_stage_and_included_gun_are_distinct_from_total():
    text = timing_details(result(), 4.5)
    assert "worker: 4.500 s" in text
    assert "Pipeline: 4.000 s" in text
    assert "3.000 s" in text
    assert "0.200 s (included in transport stage)" in text
    assert "Calculated through: Z = 1200" in text
    assert "Reused upstream through: Z = 1000" in text
    assert "excludes queue, GUI drawing and section-archive save" in text
    assert "includes internal cache IO" in text


def test_complete_reuse_suppresses_inherited_stage_and_gun_times():
    text = timing_details(result(cache_hit=True), .03)
    assert "worker: 0.030 s" in text
    assert "Complete result reused" in text
    assert "3.000" not in text and "4.000" not in text and "0.200" not in text
    assert "Reused upstream through" not in text


def test_historical_missing_and_invalid_times_are_not_reported_as_zero():
    assert "unavailable" in timing_details(SimpleNamespace())
    text = timing_details(SimpleNamespace(performance={"pipeline_seconds": float("nan")}), True)
    assert "unavailable" in text


def test_recalculation_does_not_claim_upstream_reuse():
    item = result()
    item.simulation.metrics.update(section_gun_reused=False, section_reused_prefix=False,
        section_reuse_reason="upstream_inputs_or_model_changed")
    text = timing_details(item)
    assert "calculated from tip emission" in text
    assert "solver version changed" in text
    assert "Reused upstream through" not in text


def test_reused_column_does_not_claim_its_historical_gun_work_is_current():
    item = result(reused_products={"column", "eds"})
    item.simulation.metrics.update(section_gun_reused=False,
        section_reuse_reason="upstream_inputs_or_model_changed")
    text = timing_details(item)
    assert "Reused products: electron column, EDS spectrum" in text
    assert "Electron gun:" not in text
    assert "0.200" not in text and "solver version changed" not in text
    assert "Reused upstream through" not in text  # Old partial-prefix metadata is not this request.


def test_normal_pipeline_column_checkpoint_reuse_has_current_extent():
    item = result(simulation=SimpleNamespace(metrics={"section_target_z_mm": 3026.4,
        "column_segment_cache": {"hit": True, "resume_z_mm": 1599.2}}))
    text = timing_details(item)
    assert "Reused upstream through: Z = 1599.2 mm" in text
    assert "Calculated through: Z = 3026.4 mm" in text


def test_display_keeps_previous_evidence_until_next_result(qtbot):
    view = CalculationTimingView()
    qtbot.addWidget(view)
    view.set_result(result())
    original = view.text.toPlainText()
    view.mark_stale()
    assert "previous result" in view.title()
    assert view.text.toPlainText() == original
    view.set_result(SimpleNamespace())
    assert "displayed result" in view.title()
    assert "unavailable" in view.text.toPlainText()
    assert view.text.isReadOnly()


@pytest.mark.parametrize("operation,automatic,reused", [("save", True, False), ("save", True, True),
                                                       ("save", False, False), ("load", False, False)])
def test_file_worker_records_executed_io_duration(monkeypatch, tmp_path, operation, automatic, reused, qapp):
    from temsim.gui import calculation_controller as module
    from temsim import particle_section_io as io
    ticks = iter((10., 12.5))
    monkeypatch.setattr(module, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(io, "archive_section_result", lambda *a, **k: {"identity": "fixture", "reused": reused})
    monkeypatch.setattr(io, "save_section_result", lambda *a, **k: SimpleNamespace(
        digest="timing-routing-fixture"))
    def checked_receipt(result, path, *, maximum_unpacked_bytes, expected_package_digest):
        # Timing-only routing fixture; real archive verification is covered by
        # executed-particle persistence tests, not fabricated by this stub.
        assert expected_package_digest == "timing-routing-fixture"
        assert maximum_unpacked_bytes == 8*1024**3
        return {"identity": "fixture", "path": str(path)}
    monkeypatch.setattr(io, "checked_section_archive_info", checked_receipt)
    monkeypatch.setattr(io, "load_section_result", lambda *a, **k: SimpleNamespace(
        section_archive_info={"identity": "fixture"}))
    worker = module.SectionFileWorker("fixture", operation, tmp_path / "fixture.temsection", automatic=automatic,
                                      maximum_unpacked_bytes=8*1024**3)
    outputs = []
    errors = []
    worker.signals.result.connect(lambda token, output: outputs.append(output))
    worker.signals.error.connect(errors.append)
    worker.run()
    assert not errors
    info = outputs[0]["info"] if operation == "load" else outputs[0]
    assert info["file_io_seconds"] == 2.5
    assert info["file_io_operation"] == ("verify_existing" if reused else operation)
    assert not info["file_io_reused_measurement"]
