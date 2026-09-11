"""Failure receipt and option handling; no full microscope qualification."""
import importlib.util
import json
from pathlib import Path

import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.optics.electron_gun.tip_coherence import TipCoherence
from temsim.profile_io import save_profile


@pytest.fixture
def cli(tmp_path):
    spec = importlib.util.spec_from_file_location("tip_trace_cli_test", Path(__file__).parents[1]/"scripts"/"trace_tip_wave.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = default_state()
    state.electron_gun.emitter.coherence = TipCoherence()
    profile = tmp_path/"physical_tip.toml"
    catalog = AssemblyCatalog()
    save_profile(profile, state, catalog.default_selection())
    return module, profile


@pytest.mark.parametrize("error", [ValueError("unresolved test operator at z=650 mm"), InterruptedError("user cancelled")])
def test_failed_request_preserves_inputs_and_identity_without_a_wave_archive(cli, tmp_path, monkeypatch, error):
    module, profile = cli
    output = tmp_path/"failed"
    def fails(state, request, progress_callback):
        progress_callback(4, 9, "test progress")
        raise error
    monkeypatch.setattr(module, "simulate_tip_wave", fails)
    with pytest.raises(type(error), match=str(error)):
        module.main(["--profile", str(profile), "--output", str(output), "--wave-budget-mb", "64"])
    failure = json.loads((output/"failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == ("CANCELLED" if isinstance(error, InterruptedError) else "FAILED")
    assert failure["request"]["wave_grid"]["maximum_working_bytes"] == 64*1024**2
    assert failure["implementation_at_start"] == failure["implementation_at_failure"]
    assert failure["last_progress"]["progress"] == 4
    assert failure["requested_checkpoint_published"] is False
    assert (output/"request.toml").exists()
    assert not list(output.glob("*.npz"))
    assert not (output/"receipt.json").exists()


def test_describe_selects_numerics_without_transport_or_files(cli, tmp_path, monkeypatch, capsys):
    module, profile = cli
    output = tmp_path/"description"
    monkeypatch.setattr(module, "simulate_tip_wave", lambda *a, **kw: pytest.fail("description must not propagate"))
    module.main(["--profile", str(profile), "--output", str(output), "--describe", "--no-wave-refinement",
                 "--maximum-wave-grid", "512", "--observables", "phase", "covariance"])
    description = json.loads(capsys.readouterr().out)
    assert description["request"]["wave_grid"]["automatic_refinement"] is False
    assert description["request"]["wave_grid"]["maximum_pixels"] == 512
    assert description["request"]["readout"]["intensity"] is False
    assert not output.exists()


def test_large_machine_budgets_and_scan_numerics_are_explicit(cli, capsys):
    module, profile = cli
    module.main(["--profile", str(profile), "--describe", "--scan", "--scan-stride", "3",
                 "--dwell-samples", "2", "--segment-steps", "40", "--trajectories", "8"])
    description = json.loads(capsys.readouterr().out)
    request = description["request"]
    assert request["wave_grid"]["maximum_working_bytes"] == 72*1024**3
    assert request["execution"]["maximum_ram_cache_bytes"] == 8*1024**3
    assert request["execution"]["maximum_disk_cache_bytes"] == 192*1024**3
    assert request["execution"]["segment_steps"] == 40
    assert request["inelastic"]["trajectories_per_mode"] == 8
    assert description["scan_numerics"]["stride"] == 3


def test_npz_stream_does_not_retain_previous_arrays(cli, tmp_path):
    import weakref
    import gc
    import numpy as np
    module, _ = cli
    references = []
    def arrays():
        for i in range(3):
            value = np.full((32, 32), i+1j)
            references.append(weakref.ref(value))
            yield f"mode_{i}", value
            del value
    output = tmp_path/"stream.npz"
    module._stream_npz(output, arrays())
    gc.collect()
    assert all(ref() is None for ref in references)
    with np.load(output, allow_pickle=False) as archive:
        assert archive.files == ["mode_0", "mode_1", "mode_2"]
        assert np.all(archive["mode_2"] == 2+1j)


def test_scan_cli_streams_waves_completes_progress_and_exports_weighted_image(cli, tmp_path, monkeypatch):
    from dataclasses import replace
    import numpy as np
    from temsim.physics.tip_wave_pipeline import TipWaveResult
    from temsim.physics.tip_wave_scan import ScanWaveSample
    from temsim.detector.wave_readout import read_wave_detector
    from test_wave_detector_readout import checkpoint, detector
    module, profile = cli
    c, d = checkpoint(.4), detector()
    received = read_wave_detector(c, d)
    def scan(state, request, numerics, **kwargs):
        for i in range(2):
            yield ScanWaveSample(0, 0, i, i*.1, .5,
                TipWaveResult(c, received, replace(request, tip_time_s=i*.1), False, "independent-export-fixture"))
    monkeypatch.setattr(module, "simulate_tip_scan", scan)
    output = tmp_path/"scan"
    module.main(["--profile", str(profile), "--output", str(output), "--stop", "detector", "--scan", "--dwell-samples", "2"])
    assert json.loads((output/"scan_progress.json").read_text())["complete"]
    assert not (output/"failure.json").exists()
    for i in range(2):
        assert (output/f"row_0_column_0_dwell_{i}"/"detector_wave.npz").exists()
    with np.load(output/"scan_response.npz") as data:
        assert data["response_per_tip_electron"][0, 0] == pytest.approx(received.record["response_weight"])
        assert data["dwell_coverage"][0, 0] == 1.
