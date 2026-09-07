"""Offline verification of the cache utility; never run physical calculations."""

from copy import deepcopy
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[1] / "scripts/cache_high_accuracy.py"
    spec = importlib.util.spec_from_file_location("cache_high_accuracy_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class _Plan:
    z_mm: np.ndarray
    signature: str = "plan"


def _seed():
    def group(names):
        return SimpleNamespace(**{name: np.arange(3, dtype=float) for name in names.split()})
    return SimpleNamespace(
        incident=group("z x y tx ty alive blocked_z energy_offset_ev ray_weight"),
        incident_checkpoints=group("z_mm x_m tx_rad y_m ty_rad"),
        gun_trace=SimpleNamespace(exit_bundle=group("x_m y_m tx_rad ty_rad energy_offset_ev weight ray_id alive")),
        incident_plan=_Plan(np.arange(3, dtype=float)),
    )


def test_readback_checks_coordinates_weights_and_plan(cli):
    original = _seed()
    restored = deepcopy(original)
    checked = cli.verify_incident_seed(original, restored)
    assert {"incident.energy_offset_ev", "incident.ray_weight", "checkpoint.tx_rad",
            "gun_exit.weight", "gun_exit.ray_id", "plan.z_mm"} <= set(checked)


@pytest.mark.parametrize("group,field", [
    ("incident", "x"), ("incident", "energy_offset_ev"), ("incident", "ray_weight"),
    ("incident_checkpoints", "x_m"), ("incident_plan", "z_mm"),
])
def test_corrupt_retained_values_are_not_reported_as_verified(cli, group, field):
    original = _seed()
    restored = deepcopy(original)
    getattr(getattr(restored, group), field)[0] += 1
    with pytest.raises(RuntimeError, match="differs"):
        cli.verify_incident_seed(original, restored)


def test_missing_seed_is_not_success(cli):
    with pytest.raises(RuntimeError, match="no restart seed"):
        cli.verify_incident_seed(_seed(), None)


def test_verify_existing_never_repeats_physics(cli, tmp_path, monkeypatch):
    manifest = SimpleNamespace(digest="manifest", calculation_signatures={"request": "request"})
    monkeypatch.setattr(cli, "read_manifest", lambda *_: manifest)
    monkeypatch.setattr(cli, "ArtifactStore", lambda root, **_: SimpleNamespace(root=root))
    monkeypatch.setattr(cli.CalculationController, "load_persisted_incident_seed", lambda *_: _seed())
    monkeypatch.setattr(cli, "calculate", lambda *_args, **_kwargs: pytest.fail("Physics must not run"))
    assert cli.main(["--verify-existing", "--output", str(tmp_path),
                     "--cache-root", str(tmp_path / "cache")]) == 0
    report = json.loads((tmp_path / "verification.json").read_text(encoding="utf-8"))
    assert report["cache_readback_verified"]
    assert not report["physics_repeated"]


def test_verify_missing_cache_does_not_publish_success(cli, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "read_manifest", lambda *_: object())
    monkeypatch.setattr(cli, "ArtifactStore", lambda root, **_: SimpleNamespace(root=root))
    monkeypatch.setattr(cli.CalculationController, "load_persisted_incident_seed", lambda *_: None)
    monkeypatch.setattr(cli, "calculate", lambda *_args, **_kwargs: pytest.fail("Physics must not run"))
    with pytest.raises(RuntimeError, match="No matching incident seed"):
        cli.main(["--verify-existing", "--output", str(tmp_path),
                  "--cache-root", str(tmp_path / "cache")])
    assert not (tmp_path / "verification.json").exists()
