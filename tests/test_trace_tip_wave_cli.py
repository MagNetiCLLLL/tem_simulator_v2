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
                 "--maximum-wave-grid", "512", "--observables", "phase", "covariance",
                 "--specimen-phase-method", "galerkin", "--specimen-quadrature-factor", "4"])
    description = json.loads(capsys.readouterr().out)
    assert description["request"]["wave_grid"]["automatic_refinement"] is False
    assert description["request"]["wave_grid"]["maximum_pixels"] == 512
    assert description["request"]["wave_grid"]["specimen_phase_method"] == "galerkin"
    assert description["request"]["wave_grid"]["specimen_quadrature_factor"] == 4
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


def test_surface_gun_refinement_reaches_full_pipeline_request_without_changing_profile(cli, capsys):
    module, profile = cli
    original = profile.read_bytes()
    module.main(["--profile", str(profile), "--describe", "--stop", "detector",
        "--surface-element-order", "2", "--surface-radial-nodes", "769", "--surface-axial-nodes", "257",
        "--surface-radius-factor", "5", "--energy-samples", "3", "--grid", "2048",
        "--radial-modes", "64", "--radial-quadrature", "128", "--relative-axial-step", ".01",
        "--axial-integrator", "cf4", "--coordinate-blend", ".1", ".0001", ".001",
        "--wave-following-iterations", "1", "--wave-following-phase-order", "4",
        "--occupied-axial-tolerance", ".001", "--occupied-axial-evaluations", "1000000",
        "--occupied-axial-workers", "8", "--occupied-axial-executor", "process",
        "--occupied-axial-integrator", "cf6", "--occupied-budget-allocation", "initial_indicator",
        "--gun-budget-mb", "24576", "--field-step-mm", ".5"])
    request = json.loads(capsys.readouterr().out)["request"]
    radial = request["radial_gun"]
    assert request["surface"]["element_order"] == 2
    assert request["surface"]["radial_nodes"] == 769
    assert radial["radial_modes"] == 64
    assert radial["potential_quadrature"] == 128
    assert radial["coordinate_blend"]["target_width_over_cap_radius"] == .1
    assert radial["wave_following"]["phase_order"] == 4
    assert radial["occupied_refinement"]["integrator"] == "cf6"
    assert radial["occupied_refinement"]["error_budget_allocation"] == "initial_indicator"
    assert radial["occupied_refinement"]["maximum_evaluations"] == 1000000
    assert radial["field_step_mm"] == .5
    assert radial["maximum_working_bytes"] == radial["occupied_refinement"]["maximum_working_bytes"] == 24*1024**3
    assert profile.read_bytes() == original


def test_spatial_refinement_reaches_the_imaging_request_without_transport(cli, capsys, monkeypatch):
    module, profile = cli
    before = profile.read_bytes()
    monkeypatch.setattr(module, "simulate_tip_wave", lambda *a, **kw: pytest.fail("Description must not propagate"))
    module.main(["--profile", str(profile), "--describe", "--stop", "detector",
                 "--occupied-refinement-strategy", "spatial_embedded", "--occupied-axial-tolerance", ".001"])
    request = json.loads(capsys.readouterr().out)["request"]
    assert request["radial_gun"]["occupied_refinement"]["strategy"] == "spatial_embedded"
    assert profile.read_bytes() == before


def test_surface_description_does_not_call_historical_planar_source(cli, monkeypatch, capsys):
    from dataclasses import replace
    from temsim.optics.electron_gun.tip_surface import SurfaceCoherence, load_tip_surface_reference
    module, profile = cli
    state = default_state()
    state.electron_gun.emitter.surface_model = replace(load_tip_surface_reference(), coherence=SurfaceCoherence())
    save_profile(profile, state, AssemblyCatalog().default_selection())
    original = profile.read_bytes()
    monkeypatch.setattr(module, "generate_tip_emission", lambda *a: pytest.fail("No planar-source substitution"))
    module.main(["--profile", str(profile), "--describe", "--stop", "detector"])
    description = json.loads(capsys.readouterr().out)
    assert description["source_status"] == "CONFIGURED_NOT_PROPAGATED"
    assert description["tip"]["coherence"] is not None
    assert profile.read_bytes() == original


def test_round_column_budgets_reach_request_without_new_source_inputs(cli, capsys):
    module, profile = cli
    original = profile.read_bytes()
    module.main(["--profile", str(profile), "--describe", "--radial-column-backend", "cuda",
        "--radial-column-samples", "32768", "--maximum-radial-column-samples", "4194304",
        "--handoff-pixels", "512", "--handoff-complex-tolerance", "2e-6",
        "--handoff-current-tolerance", "2e-9"])
    request = json.loads(capsys.readouterr().out)["request"]
    assert request["radial_column"] == {"backend": "cuda", "initial_samples": 32768,
        "maximum_samples": 4194304, "initial_cartesian_pixels": 512,
        "complex_tolerance": 2e-6, "current_tolerance": 2e-9}
    assert profile.read_bytes() == original


@pytest.mark.parametrize("options", [
    ["--radial-modes", "64", "--radial-quadrature", "64"],
    ["--occupied-axial-tolerance", ".2"], ["--occupied-axial-workers", "0"],
    ["--coordinate-blend", ".1", ".01", ".001"],
])
def test_invalid_gun_refinement_is_rejected_before_execution(cli, options, monkeypatch):
    module, profile = cli
    monkeypatch.setattr(module, "simulate_tip_wave", lambda *a, **kw: pytest.fail("Invalid numerics must not execute"))
    with pytest.raises(ValueError):
        module.main(["--profile", str(profile), "--describe", *options])


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


def test_default_coherent_domain_failure_is_saved_without_changing_tip(cli, tmp_path):
    from temsim.optics.electron_gun.tip_source_domain import TipSourceDomainError
    from temsim.profile_io import read_profile
    module, profile = cli
    original = profile.read_bytes()
    output = tmp_path/"source_failure"
    with pytest.raises(TipSourceDomainError):
        module.main(["--profile", str(profile), "--output", str(output), "--stop", "gun_exit"])
    failure = json.loads((output/"failure.json").read_text())
    assert failure["status"] == "FAILED"
    assert failure["source_domain"]["minimum_evaluated_energy_ev"] == .01
    assert failure["source_domain"]["paraxial_generator_relative_error_budget"] == .01
    assert failure["requested_checkpoint_published"] is False
    assert not list(output.glob("*.npz"))
    assert profile.read_bytes() == original
    selection, before = read_profile(profile)
    restored_selection, after = read_profile(output/"request.toml")
    assert restored_selection == selection
    for section in before:
        if section != "sample":
            assert after[section] == before[section]
    # The sample orientation is reconstructed from its rotation matrix on
    # load; the Euler-angle round trip is not a bit-exact representation.
    for key, value in before["sample"].items():
        if key.startswith("specimen_rotation_"):
            assert after["sample"][key] == pytest.approx(value, abs=1e-12, rel=0.)
        else:
            assert after["sample"][key] == value


def test_describe_reports_unqualified_source_without_claiming_propagation(cli, capsys):
    module, profile = cli
    module.main(["--profile", str(profile), "--describe", "--stop", "detector", "--detectors", "haadf", "df", "bf"])
    data = json.loads(capsys.readouterr().out)
    assert data["request"]["detector_keys"] == ["haadf", "df", "bf"]
    assert data["status"] == "CONFIGURATION_ONLY_NOT_PROPAGATED"
    assert data["source_status"] == "UNAVAILABLE"
    assert "forward support" in data["error"]


def test_multichannel_scan_export_retains_native_modes_at_every_detector(cli, tmp_path, monkeypatch):
    """Export-format fixture only; these are not propagated microscope waves."""
    from dataclasses import replace
    import numpy as np
    from temsim.physics.tip_wave_pipeline import TipWaveResult
    from temsim.physics.tip_wave_scan import ScanWaveSample
    from temsim.detector.wave_readout import read_wave_detector
    from test_wave_detector_readout import checkpoint, detector
    module, profile = cli
    c = checkpoint(.4, two=True)
    readouts = []
    for index, key in enumerate(("haadf", "df", "bf")):
        d = detector()
        d.key = key
        local = replace(c, beam=replace(c.beam, modes=tuple(replace(m,
            plane=replace(m.plane, amplitude=m.plane.amplitude*np.exp(1j*.2*index))) for m in c.beam.modes)))
        readouts.append(read_wave_detector(local, d))
    def scan(state, request, numerics, **kwargs):
        assert request.detector_keys == ("haadf", "df", "bf")
        yield ScanWaveSample(0, 0, 0, .1, 1.,
            TipWaveResult(local, readouts[-1], request, False, "export-fixture", tuple(readouts)))
    monkeypatch.setattr(module, "simulate_tip_scan", scan)
    output = tmp_path/"bank_scan"
    module.main(["--profile", str(profile), "--output", str(output), "--scan", "--stop", "detector",
                 "--detectors", "haadf", "df", "bf"])
    directory = output/"row_0_column_0_dwell_0"
    receipt = json.loads((directory/"receipt.json").read_text())
    assert [row["detector"] for row in receipt["detectors"]] == ["haadf", "df", "bf"]
    for row, readout in zip(receipt["detectors"], readouts):
        with np.load(directory/row["wave_archive"], allow_pickle=False) as waves:
            for index, mode in enumerate(readout.modes):
                np.testing.assert_array_equal(waves[f"mode_{index}_amplitude"], mode.plane.amplitude)
                np.testing.assert_array_equal(waves[f"mode_{index}_curvature_m1"], mode.plane.curvature_m1)
                np.testing.assert_array_equal(waves[f"mode_{index}_tilt_rad"], mode.plane.tilt_rad)
        assert row["modes"][1]["mode_id"] == readout.modes[1].mode_id
    with np.load(output/"scan_response.npz", allow_pickle=False) as data:
        assert data["detector_keys"].tolist() == ["haadf", "df", "bf"]
        assert data["detector_response_per_tip_electron"].shape == (3, 1, 1)
    assert json.loads((output/"scan_progress.json").read_text())["complete"]
