"""Analytical clipping, bounded banks, and isolated Qt range planning."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import temsim.interactive_calculation as interactive
from temsim.interactive_calculation import (
    RangeControl, CalculationRange, InteractivePlan, available_controls,
    build_bank, read_bank, prepare_replay, BankPoint, InteractiveBank,
)
from temsim.optics.model import Aperture
from temsim.detector.recording_system import RecordingPlane
from temsim.physics.simulation import Branch, Simulation
from temsim.simulation_pipeline import CalculationResult


@pytest.fixture
def toy(monkeypatch):
    """Analytical field-free rays; real production clipping, no microscope claim."""
    s = SimpleNamespace(
        lenses=[SimpleNamespace(key="D", name="D lens", percent=50.0, enabled=True, installed=True)],
        apertures=[Aperture("Test aperture", "test_aperture", 1, .5, enabled=True)],
        recording_planes=[RecordingPlane("camera", "Camera", 2, "disk", 4)],
        electron_gun=SimpleNamespace(ray_count=4),
        sample=SimpleNamespace(z_mm=0, stem_fourdstem_enabled=False, stem_wave_enabled=False),
        ac_deflector=SimpleNamespace(enabled=False, scan_enabled=False),
        illumination_mode="STEM", step_mm=1, column_inner_diameter_mm=100,
    )
    monkeypatch.setattr(interactive, "detached_state", deepcopy)
    monkeypatch.setattr(interactive, "external_model_signature", lambda state: "unchanged")
    monkeypatch.setattr(interactive, "calculation_signatures", lambda state: {"request": str(state.lenses[0].percent)})
    import temsim.gui.calculation_controller as controller
    monkeypatch.setattr(controller, "estimate_calculation_memory_bytes", lambda *a: 1024)
    import temsim.physics.beam_current as current
    monkeypatch.setattr(current, "effective_source_current_pa", lambda state: 100.0)
    return s


def result_for(s):
    x = np.tile(np.array([0, .25, .75, 1.5]) * 1e-3, (3, 1))
    weights = np.array([.1, .2, .3, .4])
    b = Branch("beam", (1, 1, 1), np.array([0., 1., 2.]), x, np.zeros_like(x),
               np.zeros_like(x), np.zeros_like(x), np.zeros(4, bool), np.ones(4),
               ["old_aperture"] * 4, 1., np.zeros(4), weights)
    incident = replace(b, alive=np.array([True, True, True, False]),
                       blocked_z=np.array([np.nan, np.nan, np.nan, -1]),
                       blocked_key=["", "", "", "upstream_stop"])
    sim = Simulation(incident, {"beam": b}, {"branch_weights_are_absolute": True})
    return CalculationResult(sim, None, s, signatures={"request": str(s.lenses[0].percent)})


def aperture_plan(s):
    c = next(c for c in available_controls(s) if c.group == "aperture" and c.field == "diameter_mm")
    return InteractivePlan((CalculationRange(c, .1, 2., 3),), 10**7)


def test_ranges_are_explicit_finite_unique_and_bounded(toy):
    c = available_controls(toy)[0]
    with pytest.raises(ValueError, match="at least one"):
        InteractivePlan((), 1024)
    for lo, hi in ((1, 1), (2, 1), (np.nan, 2), (1, np.inf)):
        with pytest.raises(ValueError):
            CalculationRange(c, lo, hi)
    a = CalculationRange(c, 20, 80, 17)
    with pytest.raises(ValueError, match="only one"):
        InteractivePlan((a, a), 1024)
    b = CalculationRange(replace(c, key="P1"), 20, 80, 17)
    with pytest.raises(ValueError, match="289"):
        InteractivePlan((a, b), 1024)


def test_shrink_and_reopen_preserve_weights_and_upstream_stops(toy):
    result = result_for(toy)
    plan = aperture_plan(toy)
    bank = InteractiveBank(plan, toy, "unchanged", (BankPoint((), result, prepare_replay(result)),), 1)
    key = plan.ranges[0].control.identity
    smaller = read_bank(bank, {key: .2})
    assert smaller.detector_fractions["camera"] == pytest.approx(.1)
    assert smaller.detector_current_pa["camera"] == pytest.approx(10)
    wider = read_bank(bank, {key: 2.})
    assert wider.detector_fractions["camera"] == pytest.approx(.6)
    assert wider.detector_current_pa["camera"] == pytest.approx(60)
    # Not 100 pA: upstream-lost electrons must never be resurrected.
    assert toy.apertures[0].diameter_mm == 1.
    np.testing.assert_array_equal(result.simulation.branches["beam"].blocked_z, np.ones(4))
    with pytest.raises(ValueError, match="outside"):
        read_bank(bank, {key: 2.01})
    with pytest.raises(ValueError, match="exactly"):
        read_bank(bank, {})


def test_first_detector_wins_and_readout_does_not_run_the_solver(toy, monkeypatch):
    toy.apertures[0].radius_mm = 2
    toy.recording_planes.insert(0, RecordingPlane("screen", "Screen", 1.5, "disk", .6))
    c = next(c for c in available_controls(toy) if c.key == "screen" and c.field == "outer_width_mm")
    plan = InteractivePlan((CalculationRange(c, .1, 2),), 10**7)
    result = result_for(toy)
    bank = InteractiveBank(plan, toy, "unchanged", (BankPoint((), result, prepare_replay(result)),), 1)
    monkeypatch.setattr(interactive, "calculate", lambda *a, **k: pytest.fail("Unexpected propagation"))
    r = read_bank(bank, {c.identity: .6})
    assert r.detector_fractions["screen"] == pytest.approx(.3)
    assert r.detector_fractions["camera"] == pytest.approx(.3)
    r = read_bank(bank, {c.identity: .1})
    assert r.detector_fractions["screen"] == pytest.approx(.1)
    assert r.detector_fractions["camera"] == pytest.approx(.5)


def test_wall_stops_survive_reopening(toy):
    toy.column_inner_diameter_mm = .8
    result = result_for(toy)
    plan = aperture_plan(toy)
    bank = InteractiveBank(plan, toy, "unchanged", (BankPoint((), result, prepare_replay(result)),), 1)
    r = read_bank(bank, {plan.ranges[0].control.identity: 2.})
    assert r.detector_fractions["camera"] == pytest.approx(.3)


def test_detector_z_replays_retained_coordinates_without_changing_source(toy, monkeypatch):
    from temsim.detector.camera import CameraDetectorComponent
    toy.selected_area_aperture = SimpleNamespace(z_mm=.5)
    toy.apertures[0].radius_mm = 2
    toy.recording_planes = [CameraDetectorComponent(z_mm=2, outer_width_mm=1,
                                                  optical_reference_downstream_of_anchor_mm=1.5)]
    result = result_for(toy)
    branch = result.simulation.branches["beam"]
    # Analytical divergence: x(z) doubles between the two observation planes.
    branch.x[:] *= branch.z[:, None] / 2
    control = next(c for c in available_controls(toy) if c.group == "detector" and c.field == "z_mm")
    plan = InteractivePlan((CalculationRange(control, 1, 2),), 10**7)
    bank = InteractiveBank(plan, toy, "unchanged", (BankPoint((), result, prepare_replay(result)),), 1)
    monkeypatch.setattr(interactive, "calculate", lambda *a, **k: pytest.fail("Unexpected propagation"))
    near = read_bank(bank, {control.identity: 1})
    far = read_bank(bank, {control.identity: 2})
    assert near.detector_fractions["camera"] == pytest.approx(.6)
    assert far.detector_fractions["camera"] == pytest.approx(.3)
    assert toy.recording_planes[0].z_mm == 2


def test_zero_diameter_blocks_even_the_exactly_on_axis_electron(toy):
    result = result_for(toy)
    plan = aperture_plan(toy)
    plan = replace(plan, ranges=(replace(plan.ranges[0], minimum=0),))
    bank = InteractiveBank(plan, toy, "unchanged", (BankPoint((), result, prepare_replay(result)),), 1)
    r = read_bank(bank, {plan.ranges[0].control.identity: 0})
    assert r.detector_fractions["camera"] == 0


def test_optical_bank_is_detached_and_exact_nodes_only(toy):
    c = available_controls(toy)[0]
    plan = InteractivePlan((CalculationRange(c, 40, 60, 3),), 10**7)
    calls = []
    def calculator(s, **kwargs):
        calls.append((s.lenses[0].percent, kwargs["existing_result"]))
        return result_for(s)
    bank = build_bank(toy, plan, calculator=calculator)
    assert [p.coordinates for p in bank.points] == [(40.,), (50.,), (60.,)]
    assert len(calls) == 3
    assert toy.lenses[0].percent == 50
    with pytest.raises(ValueError, match="precomputed"):
        read_bank(bank, {c.identity: 45})
    read_bank(bank, {c.identity: 40})
    assert len(calls) == 3


def test_build_cancellation_budget_and_external_change_do_not_return_partial_bank(toy, monkeypatch):
    plan = aperture_plan(toy)
    with pytest.raises(interactive.InteractiveCancelled):
        build_bank(toy, plan, cancelled=lambda: True, calculator=lambda s, **k: result_for(s))
    with pytest.raises(ValueError, match="cache budget"):
        build_bank(toy, replace(plan, cache_budget_bytes=1), calculator=lambda s, **k: result_for(s))
    def changed(s, **kwargs):
        monkeypatch.setattr(interactive, "external_model_signature", lambda s: "changed")
        return result_for(s)
    with pytest.raises(ValueError, match="External model changed"):
        build_bank(toy, plan, calculator=changed)


def test_readout_rejects_changed_external_inputs(toy, monkeypatch):
    result = result_for(toy)
    plan = aperture_plan(toy)
    bank = InteractiveBank(plan, toy, "unchanged", (BankPoint((), result, prepare_replay(result)),), 1)
    monkeypatch.setattr(interactive, "external_model_signature", lambda s: "changed")
    with pytest.raises(ValueError, match="external model"):
        read_bank(bank, {plan.ranges[0].control.identity: 1.})


def test_memory_cube_budget_completion_and_raw_probability():
    from temsim.physics.diffraction_memory import MemoryDiffractionSink
    from temsim.physics.fourdstem import FourDSTEMCalibration
    angles_x, angles_y = np.meshgrid([-1., 1.], [-1., 1.])
    calibration = FourDSTEMCalibration(np.zeros((1, 1)), np.zeros((1, 1)), angles_x, angles_y)
    sink = MemoryDiffractionSink(1, "signature")
    with pytest.raises(ValueError, match="requires"):
        sink.begin(calibration, np.ones((2, 2), bool), maximum_isotropic_angle_mrad=1)
    sink = MemoryDiffractionSink(1024, "signature")
    sink.begin(calibration, [[True, False], [True, True]], maximum_isotropic_angle_mrad=1)
    with pytest.raises(ValueError, match="Incomplete"):
        sink.finish()
    sink.write_frame(0, 0, np.ones((2, 2)) / 4)
    artifact = sink.finish()
    assert artifact.path is None
    assert artifact.data.sum() == pytest.approx(.75)
    assert not artifact.data.flags.writeable


def test_pre_aperture_wave_checkpoint_reopens_pupil_without_specimen_solver(monkeypatch):
    import temsim.physics.wave_imaging as wave
    n = 16
    axis = np.arange(n, dtype=float)
    unapertured = np.exp(2j * np.pi * np.arange(n)[None, :] * 3/n) * np.ones((n, 1))
    checkpoint = wave.ProjectorWaveCheckpoint((np.zeros((n, n), complex),), axis, axis, .02, .01,
                                              (unapertured,), .001)
    result = wave.WaveImagingResult("test", "test", axis, axis, np.zeros((n,n)), unapertured,
                                   np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)),
                                   axis, axis, axis, axis, {}, checkpoint)
    state = SimpleNamespace(objective_aperture=SimpleNamespace(enabled=True, radius_mm=1, z_mm=1),
                            sample=SimpleNamespace(z_mm=0))
    def project(s, cp):
        intensity = abs(cp.objective_wave_configurations[0])**2
        return intensity, intensity, np.zeros_like(intensity), SimpleNamespace(x_mm=axis, y_mm=axis, metrics={})
    monkeypatch.setattr(wave, "_project_objective_configurations", project)
    opened = wave.reproject_wave_image(state, result)
    np.testing.assert_allclose(opened.camera_electron_optical_intensity, 1, atol=1e-12)
    assert not np.any(result.camera_electron_optical_intensity)
    state.objective_aperture.radius_mm = .001
    closed = wave.reproject_wave_image(state, opened)
    assert closed.camera_electron_optical_intensity.max() < 1e-25


def test_page_requires_range_endpoints_and_does_not_auto_calculate(toy):
    from PySide6.QtWidgets import QApplication
    from temsim.gui.interactive_calculation import InteractiveCalculationPage
    app = QApplication.instance() or QApplication([])
    page = InteractiveCalculationPage()
    page.set_source(toy)
    assert not page.busy
    page._add_range()
    with pytest.raises(ValueError, match="required"):
        page.plan()
    page.ranges.cellWidget(0, 1).setText("40")
    page.ranges.cellWidget(0, 2).setText("60")
    assert page.plan().point_count == 5
    assert not page.busy
    page.shutdown()
    page.deleteLater()
    app.processEvents()


def small_real_state():
    from temsim.optics.column import default_state
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.gui.calculation_controller import CalculationController
    s = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(s, catalog.default_selection())
    s.sample.eds_enabled = False
    s.sample.sample_region_enabled = False
    s.sample.wave_enabled = False
    s.sample.stem_wave_enabled = False
    s.sample.diffraction_enabled = False
    s.ac_deflector.scan_enabled = False
    s.descan_deflector.scan_enabled = False
    s.acceleration_enabled = False
    return CalculationController._calculation_snapshot(s, "High accuracy", 9, 5.)


def test_production_bank_reuses_incident_and_never_mutates_source():
    from temsim.component_keys import PROJECTOR_LENS_2
    s = small_real_state()
    before = s.to_dict()
    control = next(c for c in available_controls(s) if c.key == PROJECTOR_LENS_2)
    plan = InteractivePlan((CalculationRange(control, 10, 11, 2),), 100 * 1024**2)
    bank = build_bank(s, plan)
    assert len(bank.points) == 2
    assert "incident" in bank.points[1].result.reused_products
    assert s.to_dict() == before
    assert bank.retained_bytes < plan.cache_budget_bytes
    r = read_bank(bank, {control.identity: 10.})
    assert all(np.isfinite(list(r.detector_current_pa.values())))


def test_production_tem_aperture_replay_agrees_with_fresh_projection():
    from temsim.physics.wave_imaging import simulate_wave_image
    s = small_real_state()
    s.illumination_mode = "TEM"
    s.projector_mode = "image"
    for d in s.stem_detectors:
        d.inserted = False
    s.fluorescent_screen.inserted = False
    s.camera.inserted = True
    s.sample.specimen_mode = "virtual"
    s.sample.specimen_preset_key = "si_110"
    s.sample.wave_enabled = True
    s.sample.wave_grid_pixels = 32
    s.sample.wave_field_of_view_angstrom = 16.
    s.sample.wave_multislice_enabled = False
    s.sample.wave_atomistic_enabled = False
    s.objective_aperture.enabled = True
    c = next(c for c in available_controls(s) if c.key == s.objective_aperture.key and c.field == "diameter_mm")
    plan = InteractivePlan((CalculationRange(c, .02, .06),), 200 * 1024**2)
    bank = build_bank(s, plan)
    replay = read_bank(bank, {c.identity: .04})
    assert replay.wave is not None
    fresh_state = interactive.detached_state(bank.points[0].result.state_snapshot)
    fresh_state.objective_aperture.diameter_mm = .04
    fresh = simulate_wave_image(fresh_state, bank.points[0].result.simulation)
    np.testing.assert_allclose(replay.wave.camera_electron_optical_intensity,
                               fresh.camera_electron_optical_intensity, rtol=2e-6, atol=1e-10)


def test_production_stem_captures_ram_cube_and_reintegrates_without_multislice(monkeypatch):
    import temsim.physics.stem_wave_imaging as wave
    import temsim.detector.stem_signal as signal
    from temsim.component_keys import PROJECTOR_LENS_2
    from temsim.physics.diffraction_memory import recollect_stem
    s = small_real_state()
    s.illumination_mode = "STEM"
    s.ac_deflector.enabled = True
    s.ac_deflector.scan_enabled = True
    s.ac_deflector.scan_pixels_x = 2
    s.ac_deflector.scan_lines = 2
    s.sample.specimen_mode = "virtual"
    s.sample.specimen_preset_key = "si_110"
    s.sample.stem_wave_enabled = True
    s.sample.wave_grid_pixels = 32
    s.sample.wave_field_of_view_angstrom = 16.
    s.sample.wave_multislice_enabled = False
    s.sample.wave_atomistic_enabled = False
    s.sample.stem_rutherford_tail_enabled = False
    for d in s.stem_detectors:
        d.inserted = True
        d.readout_enabled = True
    c = next(c for c in available_controls(s) if c.key == s.stem_detectors[-1].key
             and c.field == "outer_width_mm")
    lens = next(control for control in available_controls(s) if control.key == PROJECTOR_LENS_2)
    plan = InteractivePlan((CalculationRange(c, c.current / 2, c.current),
                            CalculationRange(lens, 10., 11., 2)), 300 * 1024**2)
    original = signal.simulate_angle_resolved_stem
    calls = []
    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(signal, "simulate_angle_resolved_stem", counted)
    bank = build_bank(s, plan)
    assert len(calls) == 1
    assert "fourdstem_cube" in bank.points[1].result.reused_products
    frame = bank.points[0].result.stem_scan
    assert frame is not None
    assert frame.fourdstem_artifact is not None
    assert frame.fourdstem_artifact.path is None
    assert not s.sample.stem_fourdstem_enabled
    replay = recollect_stem(bank.points[0].result.state_snapshot, frame)
    for key, values in frame.fractions.items():
        np.testing.assert_allclose(replay.fractions[key], values, rtol=5e-6, atol=1e-8)
    monkeypatch.setattr(wave, "simulate_angle_resolved_stem", lambda *a, **k: pytest.fail("Repeated multislice"))
    changed = read_bank(bank, {c.identity: c.current / 2, lens.identity: 10.})
    assert changed.stem is not None
    assert changed.stem.metrics["interactive_cube_reused"]


def test_controller_failure_and_cancel_preserve_previous_complete_bank(qtbot, monkeypatch):
    import temsim.gui.interactive_controller as module
    controller = module.InteractiveController()
    old = object()
    controller.bank = old
    def fail(*args, **kwargs):
        raise ValueError("test failure")
    monkeypatch.setattr(module, "build_bank", fail)
    with qtbot.waitSignal(controller.failed):
        controller.build(None, None)
    qtbot.waitUntil(lambda: not controller.busy)
    assert controller.bank is old
    # Late result from a cancelled generation must not replace the bank.
    controller._event.set()
    controller._result(controller._generation, "build", object())
    assert controller.bank is old
    controller.shutdown()
