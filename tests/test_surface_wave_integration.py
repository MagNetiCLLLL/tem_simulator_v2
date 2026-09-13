"""Actual grounded reference near-field integration; not full imaging acceptance."""
from dataclasses import replace
import json

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.optics.electron_gun.tip_surface import load_tip_surface_reference, SurfaceCoherence
from temsim.physics.surface_wave import SurfaceWaveNumerics
from temsim.physics.tip_wave_pipeline import TipWaveRequest, simulate_tip_wave


def configured():
    state = default_state()
    state.electron_gun.emitter.surface_model = replace(load_tip_surface_reference(),
        coherence=SurfaceCoherence(energy_rms_ev=0.))
    return state


REQUEST = TipWaveRequest(stop="tip_near_field",
    surface=SurfaceWaveNumerics(radial_nodes=17, axial_nodes=33, energy_samples=1))


def test_pipeline_preserves_phase_cache_and_instrument_identity():
    state = configured()
    result = simulate_tip_wave(state, REQUEST)
    repeated = simulate_tip_wave(state, REQUEST)
    assert repeated.propagation_cache_hit
    assert repeated.checkpoint.digest == result.checkpoint.digest
    assert result.detector is None and not result.detector_readouts
    assert "column_guard_plan" in result.checkpoint.record
    state.electron_gun.extractor.voltage_kv += .1
    changed = simulate_tip_wave(state, REQUEST)
    assert not changed.propagation_cache_hit
    assert changed.instrument_digest != result.instrument_digest
    assert changed.checkpoint.digest != result.checkpoint.digest
    assert np.iscomplexobj(changed.checkpoint.modes[0].amplitude)


def test_full_image_is_not_admitted_by_the_existence_of_a_near_field():
    from temsim.physics.source_admission import gun_phase_readiness, require_gun_wave_source
    state = configured()
    assert gun_phase_readiness(state)["coherent_source"] == "COHERENT_SURFACE_NEAR_FIELD_AVAILABLE"
    with pytest.raises(ValueError, match="not yet qualified"):
        require_gun_wave_source(state, product="STEM image")
    assert gun_phase_readiness(state)["gun_phase_transfer"] == "DEVELOPMENT_TWO_WAY_ROUND_GUN_AVAILABLE_NOT_QUALIFIED"


def test_column_aperture_and_filter_moved_to_tip_cannot_be_skipped():
    state = configured()
    aperture = state.apertures[0]
    aperture.apply_manifest_geometry(z_mm=1e-6)
    aperture.enabled = True
    if hasattr(aperture, "inserted"):
        aperture.inserted = True
    with pytest.raises(ValueError, match="column aperture"):
        simulate_tip_wave(state, REQUEST)
    state = configured()
    state.energy_filter.entrance_z_mm = 1e-6
    with pytest.raises(ValueError, match="energy filter"):
        simulate_tip_wave(state, REQUEST)


def test_shared_column_field_is_checked_before_cached_wave_reuse(monkeypatch):
    import temsim.physics.tip_wave_pipeline as pipeline
    state = configured()
    simulate_tip_wave(state, REQUEST)
    original = pipeline._prepare_column
    def magnetic(*args, **kwargs):
        plan, radii, stops, owners = original(*args, **kwargs)
        return replace(plan, magnetic_t=np.ones_like(plan.magnetic_t)*.01), radii, stops, owners
    monkeypatch.setattr(pipeline, "_prepare_column", magnetic)
    with pytest.raises(ValueError, match="column field"):
        simulate_tip_wave(state, REQUEST)


def test_profile_and_snapshot_roundtrip_without_converting_classical_source(tmp_path):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.profile_io import read_profile, save_profile, apply_profile_values
    from temsim.instrument_snapshot import capture_instrument_snapshot
    state = configured()
    catalog = AssemblyCatalog()
    path = tmp_path/"coherent.toml"
    save_profile(path, state, catalog.default_selection())
    _, values = read_profile(path)
    restored = default_state()
    assert not apply_profile_values(restored, values)
    assert restored.electron_gun.emitter.surface_model == state.electron_gun.emitter.surface_model
    snapshot = capture_instrument_snapshot(restored)
    assert snapshot.restore().electron_gun.emitter.surface_model.coherence == SurfaceCoherence(energy_rms_ev=0.)
    classical = replace(state.electron_gun.emitter.surface_model, coherence=None)
    state.electron_gun.emitter.surface_model = classical
    save_profile(tmp_path/"classical.toml", state, catalog.default_selection())
    _, values = read_profile(tmp_path/"classical.toml")
    apply_profile_values(restored, values)
    assert restored.electron_gun.emitter.surface_model.coherence is None


def test_archive_keeps_complex_mode_and_does_not_overwrite(tmp_path):
    from temsim.physics.surface_wave_export import export_surface_wave, draw_surface_wave
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    result = simulate_tip_wave(configured(), REQUEST)
    path = tmp_path/"wave.npz"
    receipt = export_surface_wave(result, path)
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["mode_0_amplitude"], result.checkpoint.modes[0].amplitude)
        assert json.loads(str(archive["receipt_json"]))["scope"] == "TIP_NEAR_FIELD_ONLY_NOT_TEM_STEM"
    assert receipt["modes"][0]["phase_reference"]
    mode = result.checkpoint.modes[0]
    reweighted = replace(result.checkpoint, modes=(replace(mode, weight=.5),))
    assert reweighted.digest != result.checkpoint.digest
    with pytest.raises(FileExistsError):
        export_surface_wave(result, path)
    figure = Figure(figsize=(10, 5)); FigureCanvasAgg(figure)
    draw_surface_wave(figure, result.checkpoint)
    figure.savefig(tmp_path/"near.png")
    assert (tmp_path/"near.png").stat().st_size > 1000


def test_surface_editor_hides_duplicate_classical_energies(qtbot):
    from temsim.gui.gun_source_dialog import GunSourceDialog
    state = configured()
    dialog = GunSourceDialog(state.electron_gun, instrument_state=state)
    qtbot.addWidget(dialog)
    assert dialog.surface_coherent.isChecked()
    assert dialog.surface_inputs["normal_mean_energy_ev"].isHidden()
    assert not dialog.quantum_inputs["mean_energy_ev"].isHidden()
    assert dialog.near_field_button.isEnabled()
    dialog.quantum_inputs["mean_energy_ev"].setText("0.4")
    dialog.surface_inputs["normal_mean_energy_ev"].setText("invalid inactive draft")
    dialog.accept()
    assert dialog.value()["surface_model"].coherence.mean_energy_ev == .4
    assert state.electron_gun.emitter.surface_model.coherence.mean_energy_ev == .3


def test_viewer_close_cancels_without_destroying_a_running_thread(qtbot):
    from temsim.gui.surface_wave_dialog import SurfaceWaveDialog
    dialog = SurfaceWaveDialog(configured())
    qtbot.addWidget(dialog)
    dialog._start()
    dialog.close()
    qtbot.waitUntil(lambda: dialog.worker is None, timeout=30000)
    assert dialog._closing


@pytest.mark.parametrize("version", [8, 9])
def test_surface_profile_versions_preserve_explicit_model(tmp_path, version):
    import tomllib
    import tomli_w
    from temsim.profile_io import save_profile, read_profile, apply_profile_values
    from temsim.assembly_catalog import AssemblyCatalog
    state = configured()
    path = tmp_path/"profile.toml"
    save_profile(path, state, AssemblyCatalog().default_selection())
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    document["format_version"] = version
    if version == 8:
        # Historical grounded surface profile, before a coherent reservoir.
        document["gun_source_model"]["surface_model"].pop("coherence")
    path.write_text(tomli_w.dumps(document), encoding="utf-8")
    target = configured()
    assert not apply_profile_values(target, read_profile(path)[1])
    assert (target.electron_gun.emitter.surface_model.coherence is None) == (version == 8)


def test_viewer_runs_without_replacing_instrument_state_and_keeps_failed_result(qtbot):
    from temsim.gui.surface_wave_dialog import SurfaceWaveDialog
    state = configured()
    dialog = SurfaceWaveDialog(state)
    qtbot.addWidget(dialog)
    dialog.radial.setValue(17); dialog.axial.setValue(33); dialog.energies.setValue(1)
    dialog._start()
    qtbot.waitUntil(lambda: dialog.worker is None, timeout=30000)
    assert dialog.calculation is not None
    prior = dialog.calculation
    assert state.electron_gun.emitter.surface_model.coherence.mean_energy_ev == .3
    dialog.extent.setValue(10.)  # declared domain cannot fit on the sphere
    dialog._start()
    qtbot.waitUntil(lambda: dialog.worker is None, timeout=30000)
    assert dialog.calculation is prior
    assert "Previous field retained" in dialog.status.text()
    dialog.close()


def test_cli_executes_and_exports_coherent_near_field(tmp_path):
    import importlib.util
    from pathlib import Path
    from temsim.profile_io import save_profile
    from temsim.assembly_catalog import AssemblyCatalog
    path = Path(__file__).parents[1]/"scripts"/"trace_tip_wave.py"
    spec = importlib.util.spec_from_file_location("surface_cli", path)
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    profile = tmp_path/"source.toml"
    save_profile(profile, configured(), AssemblyCatalog().default_selection())
    output = tmp_path/"result"
    assert cli.main(["--profile", str(profile), "--output", str(output), "--stop", "tip_near_field",
        "--surface-radial-nodes", "17", "--surface-axial-nodes", "33", "--energy-samples", "1"]) == 0
    assert (output/"tip_near_field.npz").exists()
    receipt = json.loads((output/"receipt.json").read_text())
    assert receipt["full_tem_stem"] == "NOT_CONNECTED_NOT_VALIDATED"


@pytest.mark.parametrize("mean,rms", [(.3,.001), (.3,.1), (.3,.6)])
def test_gamma_energy_rule_retains_actual_inputs_including_narrow_and_broad_spectra(mean, rms):
    energies, weights = SurfaceCoherence(mean_energy_ev=mean, energy_rms_ev=rms).energy_quadrature(7)
    assert energies.min() > 0
    assert weights@energies == pytest.approx(mean, rel=1e-12)
    assert np.sqrt(weights@(energies-mean)**2) == pytest.approx(rms, rel=1e-12)
