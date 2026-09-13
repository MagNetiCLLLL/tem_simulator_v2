"""Frozen reference-run plumbing; not physics acceptance."""
from pathlib import Path
import json

import pytest

from scripts.freeze_physics_run import freeze, inventory
from scripts.reuse_frozen_physics import verified_inputs


def checkout(root):
    for name, value in (("src/temsim/calculation_manifest.py", "original"),
                        ("configs/example.toml", "value = 1"),
                        ("scripts/inspect_surface_column.py", "driver")):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")


def test_snapshot_retains_exact_inputs_without_following_later_edits(tmp_path):
    root, copy = tmp_path/"source", tmp_path/"frozen"
    checkout(root)
    hashes = freeze(root, copy)
    assert inventory(root) == inventory(copy) == hashes
    (root/"configs/example.toml").write_text("value = 2", encoding="utf-8")
    assert inventory(copy) == hashes
    assert inventory(root) != hashes
    with pytest.raises(FileExistsError):
        freeze(root, copy)


def test_cannot_snapshot_inside_inputs_or_overwrite_source(tmp_path):
    checkout(tmp_path)
    for target in (tmp_path, tmp_path/"src/copy", tmp_path/"configs/copy"):
        with pytest.raises(ValueError, match="outside"):
            freeze(tmp_path, target)


def test_empty_checkout_is_not_a_frozen_solver(tmp_path):
    with pytest.raises(ValueError, match="complete"):
        freeze(tmp_path, tmp_path/"copy")


def test_comparisons_reuse_exact_frozen_paths_but_reject_changed_inputs(tmp_path):
    root, copy = tmp_path/"source", tmp_path/"frozen"
    checkout(root)
    expected = freeze(root, copy)
    receipt = {"schema": "frozen-physics-execution-v1", "snapshot_root": str(copy),
               "input_sha256": expected, "status": "RUNNING"}
    (copy/"execution.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert verified_inputs(copy) == expected
    (root/"configs/example.toml").write_text("unrelated live edit", encoding="utf-8")
    assert verified_inputs(copy) == expected
    (copy/"configs/example.toml").write_text("changed frozen input", encoding="utf-8")
    with pytest.raises(ValueError, match="inputs changed"):
        verified_inputs(copy)


def test_frozen_comparison_rejects_a_relocated_manifest(tmp_path):
    checkout(tmp_path)
    (tmp_path/"execution.json").write_text(json.dumps({
        "schema": "frozen-physics-execution-v1", "snapshot_root": str(tmp_path/"elsewhere"),
        "input_sha256": inventory(tmp_path)}), encoding="utf-8")
    with pytest.raises(ValueError, match="matching frozen"):
        verified_inputs(tmp_path)


def test_long_driver_retains_stdout_stderr_and_failure_without_overwrite(tmp_path, capsys):
    import os
    import sys
    from scripts.physics_driver_log import run_logged_driver
    log = tmp_path/"driver.log"
    code = run_logged_driver([sys.executable, "-c",
        "import sys; print('progress', flush=True); print('failure detail', file=sys.stderr); sys.exit(7)"],
        cwd=tmp_path, env=dict(os.environ), log_path=log)
    assert code == 7
    assert b"progress" in log.read_bytes() and b"failure detail" in log.read_bytes()
    assert "failure detail" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        run_logged_driver([sys.executable, "-c", "raise Exception('must not run')"],
            cwd=tmp_path, env=dict(os.environ), log_path=log)
