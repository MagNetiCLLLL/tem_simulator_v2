"""Analytical clipping, bounded banks, and isolated Qt range planning."""
from copy import deepcopy
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

import temsim.interactive_calculation as interactive
from temsim.interactive_calculation import (
    RangeControl, CalculationRange, InteractivePlan, available_controls,
    build_bank, read_bank, prepare_replay, BankPoint, InteractiveBank,
)
from temsim.optics.model import Aperture
from temsim.physics.simulation import Branch, Simulation
from temsim.simulation_pipeline import CalculationResult


@dataclass
class RecordingPlane:
    """Test-only finite absorbing plane for isolated clipping mathematics."""
    key: str
    name: str
    z_mm: float
    geometry: str
    outer_width_mm: float
    inner_diameter_mm: float = 0.
    inserted: bool = True


@pytest.fixture
def toy(monkeypatch):
    """Analytical field-free rays; real production clipping, no microscope claim."""
    s = SimpleNamespace(
        lenses=[SimpleNamespace(key="D", name="D lens", percent=50.0, enabled=True, installed=True)],
        apertures=[Aperture("Test aperture", "test_aperture", 1, .5, enabled=True)],
        recording_planes=[RecordingPlane("camera", "Camera", 2, "disk", 4)],
        electron_gun=SimpleNamespace(ray_count=4, exit_plane_z_mm=0.),
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


def test_legacy_wave_checkpoint_requires_explicit_pre_loss_reference():
    # Exercise the local projection contract before any wave execution.
    # Amplitude-only checkpoints cannot reconstruct missing incident flux.
    import temsim.physics.wave_imaging as wave
    n = 16
    axis = np.arange(n, dtype=float)
    amplitude = np.ones((n, n), complex)
    checkpoint = wave.ProjectorWaveCheckpoint((amplitude,), axis, axis, .02, .01,
                                               (amplitude,), .001)
    result = wave.WaveImagingResult("test", "test", axis, axis, np.zeros((n,n)), amplitude,
                                   np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)),
                                   axis, axis, axis, axis, {}, checkpoint)
    from temsim.optics.column import default_state
    state = default_state()
    with pytest.raises(ValueError, match="pre-loss reference norm"):
        wave._project_objective_configurations(state, checkpoint)


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
    s.ac_deflector.scan_enabled = False
    s.descan_deflector.scan_enabled = False
    s.acceleration_enabled = False
    return CalculationController._calculation_snapshot(s, "High accuracy", 9, 5.)


def test_production_bank_reuses_incident_and_never_mutates_source(monkeypatch):
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
    outputs = tuple(p.result.energy_filter for p in bank.points)
    assert all(output is None for output in outputs)  # Explicit default: no filter installed.
    assert all(not hasattr(p.result.state_snapshot, "energy_filter_result") for p in bank.points)
    monkeypatch.setattr(interactive, "calculate", lambda *a, **k: pytest.fail("Repeated propagation on readout"))
    r = read_bank(bank, {control.identity: 10.})
    assert all(np.isfinite(list(r.detector_current_pa.values())))
    read_bank(bank, {control.identity: 11.})



def test_production_tem_aperture_replay_agrees_with_fresh_projection_requires_qualified_tip_source(monkeypatch):
    from temsim.optics.column import default_state
    from temsim.physics.source_admission import UnsupportedWaveSource
    from temsim.simulation_pipeline import calculate
    import temsim.simulation_pipeline as pipeline
    state = default_state()
    state.illumination_mode = "TEM"
    from temsim.component_keys import STEM_DETECTOR_KEYS
    for detector in state.recording_planes:
        if detector.key in STEM_DETECTOR_KEYS:
            detector.inserted = False
    from temsim.physics.wave_imaging import tem_wave_imaging_enabled
    state.sample.wave_enabled = True
    state.sample.stem_wave_enabled = False
    state.sample.stem_fourdstem_enabled = False
    assert tem_wave_imaging_enabled(state)
    before = state.to_dict()
    monkeypatch.setattr(pipeline, "run", lambda *a, **k: pytest.fail("Unqualified source must not start transport"))
    with pytest.raises(UnsupportedWaveSource):
        calculate(state)
    assert state.to_dict() == before


def test_production_stem_captures_ram_cube_and_reintegrates_without_multislice_requires_qualified_tip_source(monkeypatch):
    from temsim.optics.column import default_state
    from temsim.physics.source_admission import UnsupportedWaveSource
    from temsim.simulation_pipeline import calculate
    import temsim.simulation_pipeline as pipeline
    state = default_state()
    state.illumination_mode = "STEM"
    state.ac_deflector.wobble_enabled = False
    state.ac_deflector.scan_enabled = True
    state.sample.wave_enabled = True
    state.sample.stem_wave_enabled = True
    state.sample.stem_fourdstem_enabled = False
    before = state.to_dict()
    monkeypatch.setattr(pipeline, "run", lambda *a, **k: pytest.fail("Unqualified source must not start transport"))
    with pytest.raises(UnsupportedWaveSource):
        calculate(state)
    assert state.to_dict() == before


def test_controller_failure_and_cancel_preserve_previous_complete_bank(qtbot, monkeypatch):
    import temsim.gui.interactive_controller as module
    controller = module.InteractiveController()
    old = object()
    controller.bank = old
    def fail(*args, **kwargs):
        raise ValueError("test failure")
    monkeypatch.setattr(module, "build_bank", fail)
    state = small_real_state()
    control = next(c for c in available_controls(state) if c.field == "percent")
    plan = InteractivePlan((CalculationRange(control, 10, 11, 2),), 100 * 1024**2)
    with qtbot.waitSignal(controller.failed):
        controller.build(state, plan)
    qtbot.waitUntil(lambda: not controller.busy)
    assert controller.bank is old
    # Late result from a cancelled generation must not replace the bank.
    controller._event.set()
    controller._result(controller._generation, "build", object())
    assert controller.bank is old
    controller.shutdown()
