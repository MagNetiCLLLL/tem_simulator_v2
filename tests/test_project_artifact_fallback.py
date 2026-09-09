"""Recovery-cache routing tests; no electron propagation is executed."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import temsim.gui.calculation_controller as controller_module
from temsim.gui.calculation_controller import CalculationController, CalculationWorker
from temsim.simulation_pipeline import CalculationResult


@pytest.fixture(autouse=True)
def _no_calculation(monkeypatch):
    monkeypatch.setattr(
        controller_module, "calculate",
        Mock(side_effect=AssertionError("This test must not calculate")),
    )


def _worker(primary=None, *, enabled=True, manifest=None, quality="High accuracy"):
    return CalculationWorker(
        1, quality, SimpleNamespace(name="requested state"),
        model_signature="requested model",
        artifact_store=primary,
        calculation_manifest=(
            manifest if manifest is not None else
            SimpleNamespace(calculation_signatures={"incident": "exact incident"}, external_inputs=())
        ),
        allow_project_artifact_fallback=enabled,
        artifact_cache_budget_bytes=123456,
    )


def _fallback(monkeypatch, tmp_path, *, seed=None, error=None):
    root = tmp_path / "recovery"
    root.mkdir()
    reader = Mock()
    reader.get_incident_simulation_seed = Mock(return_value=seed, side_effect=error)
    locator = Mock(return_value=root)
    factory = Mock(return_value=reader)
    monkeypatch.setattr(controller_module, "project_artifact_fallback_root", locator)
    monkeypatch.setattr(controller_module, "ArtifactStore", factory)
    return root, reader, locator, factory


def test_primary_hit_never_consults_project_fallback(monkeypatch, tmp_path):
    seed = object()
    primary = Mock()
    primary.get_incident_simulation_seed.return_value = seed
    _root, _reader, locator, factory = _fallback(monkeypatch, tmp_path)
    worker = _worker(primary)

    result = worker._load_persistent_incident_seed(None)

    assert result.simulation is seed
    assert result.state_snapshot is worker.state
    assert result.model_signature == worker.model_signature
    primary.get_incident_simulation_seed.assert_called_once_with(worker.calculation_manifest)
    locator.assert_not_called()
    factory.assert_not_called()


@pytest.mark.parametrize("primary_state", ["missing", "error", "unavailable"])
def test_fallback_uses_exact_manifest_once_and_retains_other_products(
    monkeypatch, tmp_path, primary_state,
):
    seed = object()
    root, reader, locator, factory = _fallback(monkeypatch, tmp_path, seed=seed)
    primary = None if primary_state == "unavailable" else Mock()
    if primary is not None:
        primary.get_incident_simulation_seed.return_value = None
        if primary_state == "error":
            primary.get_incident_simulation_seed.side_effect = OSError("virtualized path")
    worker = _worker(primary)
    existing = CalculationResult(
        simulation=object(), energy_filter=object(), wave_imaging=object(),
        state_snapshot=object(), signatures={"incident": "old", "wave": "retained"},
    )

    result = worker._load_persistent_incident_seed(existing)

    assert result is not existing
    assert result.simulation is seed
    assert result.wave_imaging is existing.wave_imaging
    assert result.energy_filter is existing.energy_filter
    assert result.state_snapshot is existing.state_snapshot
    assert result.signatures == {"incident": "exact incident", "wave": "retained"}
    assert existing.signatures == {"incident": "old", "wave": "retained"}
    locator.assert_called_once_with()
    factory.assert_called_once_with(root, quota_bytes=123456)
    reader.get_incident_simulation_seed.assert_called_once_with(worker.calculation_manifest)
    reader.put_incident_simulation_seed.assert_not_called()


@pytest.mark.parametrize("failure", ["missing_directory", "constructor", "read", "mismatch"])
def test_unavailable_or_nonmatching_fallback_is_safe_miss(monkeypatch, tmp_path, failure):
    _root, reader, locator, factory = _fallback(monkeypatch, tmp_path)
    if failure == "missing_directory":
        locator.return_value = None
    elif failure == "constructor":
        factory.side_effect = OSError("permission denied")
    elif failure == "read":
        reader.get_incident_simulation_seed.side_effect = ValueError("invalid checksum")
    worker = _worker()
    existing = CalculationResult(simulation=object(), energy_filter=object())

    assert worker._load_persistent_incident_seed(existing) is existing
    if failure == "missing_directory":
        factory.assert_not_called()
    else:
        assert factory.call_count == 1


def test_complete_memory_seed_takes_priority_over_both_disk_stores(monkeypatch, tmp_path):
    primary = Mock()
    _root, _reader, locator, factory = _fallback(monkeypatch, tmp_path)
    worker = _worker(primary)
    existing = CalculationResult(
        simulation=SimpleNamespace(
            incident_plan=object(), incident_checkpoints=object(), gun_trace=object(),
        ),
        energy_filter=None, signatures={"incident": "exact incident"},
    )

    assert worker._load_persistent_incident_seed(existing) is existing
    primary.get_incident_simulation_seed.assert_not_called()
    locator.assert_not_called()
    factory.assert_not_called()


@pytest.mark.parametrize("quality,enabled", [("High accuracy", False), ("Preview", True)])
def test_worker_disallowed_fallback_does_not_probe_disk(monkeypatch, tmp_path, quality, enabled):
    _root, _reader, locator, factory = _fallback(monkeypatch, tmp_path)
    worker = _worker(quality=quality, enabled=enabled)

    assert worker._load_persistent_incident_seed(None) is None
    locator.assert_not_called()
    factory.assert_not_called()


@pytest.mark.parametrize("options", [
    {"persistent_cache_enabled": False},
    {"artifact_store": Mock()},
    {"artifact_cache_root": "explicit-isolated-cache"},
])
def test_controller_explicit_configuration_disables_fallback(monkeypatch, qapp, options):
    factory = Mock()
    monkeypatch.setattr(controller_module, "ArtifactStore", factory)
    locator = Mock(side_effect=AssertionError("Do not inspect a project cache"))
    monkeypatch.setattr(controller_module, "project_artifact_fallback_root", locator)

    controller = CalculationController(**options)

    assert not controller._allow_project_artifact_fallback
    locator.assert_not_called()
    if options.get("persistent_cache_enabled") is False or "artifact_store" in options:
        factory.assert_not_called()


def test_default_primary_failure_still_dispatches_manifest_and_updated_fallback_quota(
    monkeypatch, qapp,
):
    monkeypatch.setattr(
        controller_module, "ArtifactStore", Mock(side_effect=OSError("primary unavailable")),
    )
    manifest = SimpleNamespace(calculation_signatures={"incident": "exact incident"})
    capture = Mock(return_value=manifest)
    monkeypatch.setattr(controller_module, "capture_calculation_manifest", capture)
    controller = CalculationController()
    controller.configure_cache(disk_cache_budget_bytes=654321)
    assert controller.artifact_store is None
    monkeypatch.setattr(controller, "_best_seed_entry", Mock(return_value=None))
    monkeypatch.setattr(controller, "_make_room_for_calculation", Mock(return_value=None))
    start = Mock()
    monkeypatch.setattr(controller.pool, "start", start)
    state = SimpleNamespace()

    controller._dispatch_prepared(
        state, "High accuracy", 3, 0.25,
        model_signature="model", request_signatures={"request": "request"},
        generation=controller.generation, estimate=0,
    )

    capture.assert_called_once_with(state, ray_count=3, step_mm=0.25)
    worker = start.call_args.args[0]
    assert worker.calculation_manifest is manifest
    assert worker.allow_project_artifact_fallback
    assert worker.artifact_cache_budget_bytes == 654321
    assert worker.artifact_store is None


def _checkout(monkeypatch, tmp_path):
    checkout = tmp_path / "checkout"
    source = checkout / "src" / "temsim" / "gui" / "calculation_controller.py"
    source.parent.mkdir(parents=True)
    source.touch()
    (checkout / "pyproject.toml").touch()
    (checkout / "configs").mkdir()
    monkeypatch.setattr(controller_module, "__file__", str(source))
    return checkout, checkout / "outputs" / "high_accuracy_artifacts"


def test_fallback_locator_only_returns_existing_current_checkout_cache(monkeypatch, tmp_path):
    checkout, candidate = _checkout(monkeypatch, tmp_path)
    monkeypatch.setenv("TEMSIM_PROJECT_ROOT", str(tmp_path / "different-project"))
    monkeypatch.chdir(tmp_path)

    assert controller_module.project_artifact_fallback_root() is None
    assert not (checkout / "outputs").exists()
    candidate.mkdir(parents=True)
    assert controller_module.project_artifact_fallback_root() == candidate.resolve()


def test_fallback_locator_rejects_escaping_resolved_directory(monkeypatch, tmp_path):
    _root, candidate = _checkout(monkeypatch, tmp_path)
    candidate.mkdir(parents=True)
    original_resolve = Path.resolve

    def resolved(path, *args, **kwargs):
        if path == candidate:
            return tmp_path / "other-project"
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolved)
    assert controller_module.project_artifact_fallback_root() is None


def test_fallback_locator_does_not_search_installed_package_data(monkeypatch, tmp_path):
    source = tmp_path / "site-packages" / "temsim" / "gui" / "calculation_controller.py"
    source.parent.mkdir(parents=True)
    source.touch()
    monkeypatch.setattr(controller_module, "__file__", str(source))

    assert controller_module.project_artifact_fallback_root() is None


def test_default_primary_path_is_unchanged(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert controller_module.default_artifact_cache_root() == (
        tmp_path / "TEM Simulator v2" / "high_accuracy_artifacts"
    )
