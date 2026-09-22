"""Cancellation through ordinary entry points with bounded particle kernels.

The incident bundle is supplied locally to isolate material/worker boundaries;
these are not full tip-to-detector scientific qualification calculations.
"""
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from temsim import simulation_pipeline as pipeline
from temsim.gui.calculation_controller import CalculationWorker
from temsim.optics.column import default_state


def _point_request(monkeypatch, *, eds=False):
    state = default_state()
    state.sample.wave_enabled = False
    state.sample.stem_wave_enabled = False
    state.sample.eds_enabled = eds
    state.ac_deflector.scan_enabled = False
    state.descan_deflector.scan_enabled = False
    count = 8
    branch = SimpleNamespace(
        alive=np.ones(count, dtype=bool),
        x=np.zeros((1, count)), y=np.zeros((1, count)),
        tx=np.zeros((1, count)), ty=np.zeros((1, count)),
        energy_offset_ev=np.zeros(count), ray_weight=np.full(count, 1 / count),
    )
    simulation = pipeline.Simulation(
        incident=branch, branches={},
        metrics={"sample_beam_surviving_fraction": 1.0},
    )
    signatures = pipeline.calculation_signatures(state)
    previous = pipeline.CalculationResult(
        simulation=simulation, energy_filter=None,
        signatures={key: signatures[key] for key in ("column", "incident", "energy_filter")},
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("Cancelled material work reached downstream transport")

    monkeypatch.setattr(pipeline, "build_geometric_specimen_exit", forbidden)
    return state, previous


@pytest.mark.parametrize("stage,label", [
    ("elastic", "Preparing elastic specimen histories"),
    ("eds", "EDS ionisation tracks 0/"),
])
@pytest.mark.parametrize("with_display_callback", [False, True])
def test_ordinary_pipeline_stops_real_material_kernel(
    monkeypatch, stage, label, with_display_callback,
):
    state, previous = _point_request(monkeypatch, eds=stage == "eds")
    token = Event()
    state._tuning_cancelled = token.is_set
    events = []

    def progress(done, total, message):
        events.append(message)
        if label in message:
            token.set()

    if not with_display_callback:
        # A non-GUI caller can supply the state token without a display
        # callback. Request cancellation on entering the real target kernel.
        module = ("temsim.specimen.elastic_transport" if stage == "elastic"
                  else "temsim.detector.eds_signal")
        name = ("simulate_elastic_point_transport" if stage == "elastic"
                else "simulate_eds_tracks")
        from importlib import import_module
        owner = import_module(module)
        original = getattr(owner, name)

        def cancel_at_entry(*args, **kwargs):
            token.set()
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, cancel_at_entry)

    with pytest.raises(RuntimeError, match="Superseded optical tuning request"):
        pipeline.calculate(
            state, existing_result=previous,
            progress_callback=progress if with_display_callback else None,
        )
    assert token.is_set()
    if with_display_callback:
        assert label in events[-1]
    assert previous.specimen_interactions is None
    assert previous.simulation.metrics == {"sample_beam_surviving_fraction": 1.0}


def test_cancelled_ordinary_entry_does_not_prepare_state(monkeypatch):
    state = default_state()
    state._tuning_cancelled = lambda: True

    def forbidden(*_args):
        pytest.fail("Already-cancelled work must not prepare a new calculation")

    monkeypatch.setattr("temsim.physics.illumination.illumination_config", forbidden)
    with pytest.raises(RuntimeError, match="Superseded optical tuning request"):
        pipeline.calculate(state)


@pytest.mark.parametrize("cancel_during_signal", [False, True])
def test_worker_progress_aborts_before_returning_to_solver(cancel_during_signal):
    worker = CalculationWorker(7, "High accuracy", default_state())
    seen = []

    def progress(*event):
        seen.append(event)
        worker.cancel_event.set()

    worker.signals.progress.connect(progress)
    if not cancel_during_signal:
        worker.cancel_event.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        worker._report_progress(1, 8, "Elastic specimen history 1/8")
    assert len(seen) == int(cancel_during_signal)


def test_cancelled_material_worker_emits_finished_without_publication(monkeypatch):
    state, previous = _point_request(monkeypatch)
    worker = CalculationWorker(9, "High accuracy", state, existing_result=previous)
    results, errors, finished, labels = [], [], [], []
    worker.signals.result.connect(lambda *event: results.append(event))
    worker.signals.error.connect(lambda *event: errors.append(event))
    worker.signals.finished.connect(lambda *event: finished.append(event))

    def progress(_generation, _quality, _done, _total, label):
        labels.append(label)
        if "Preparing elastic specimen histories" in label:
            worker.cancel_event.set()

    worker.signals.progress.connect(progress)
    monkeypatch.setattr(worker, "_persist_incident_seed",
                        lambda *_args: results.append("partial persistence"))
    worker.run()
    assert worker.cancel_event.is_set()
    assert "Preparing elastic specimen histories" in labels[-1]
    assert results == errors == []
    assert finished == [(9, "High accuracy")]
    assert not hasattr(state, "_tuning_cancelled")


@pytest.mark.parametrize("cancel_mode", ["display", "state_only", "none"])
def test_scan_entry_stops_geometric_rows_at_cancellation_boundary(monkeypatch, cancel_mode):
    """Real geometric row integration with a supplied one-ray optical fixture."""
    from temsim.detector import stem_signal
    from temsim.vacuum import VacuumMap

    token = Event()
    hits, progress_events = [], []

    def hit_mask(x, y):
        hits.append(1)
        if cancel_mode == "state_only" and len(hits) == 2:
            token.set()
        return np.ones_like(x, dtype=bool)

    detector = SimpleNamespace(key="local_detector", name="Detector", inserted=True,
                               readout_enabled=True)
    state = SimpleNamespace(
        _tuning_cancelled=token.is_set, simulation_time_s=0.0, vacuum_map=VacuumMap(),
        sample=SimpleNamespace(specimen_mode="reference", inserted=False, z_mm=1.0,
                               stem_image_enabled=True, stem_wave_enabled=False),
        ac_deflector=SimpleNamespace(
            enabled=True, scan_enabled=True, scan_pixels_x=2, scan_lines=3,
            scan_command_matrix_mrad=np.eye(2), scan_frame_period_s=1.0,
            scan_pixel_size_nm=1.0, scan_field_of_view_x_nm=2.0,
            scan_field_of_view_y_nm=3.0, scan_kick_mrad=lambda _time: (0.0, 0.0)),
        descan_deflector=SimpleNamespace(enabled=False, scan_enabled=False),
        stem_detectors=(detector,),
        recording_planes=(SimpleNamespace(key=detector.key, inserted=True, z_mm=2.0,
                                         hit_mask=hit_mask),),
    )
    branch = SimpleNamespace(z=np.array([1.0, 2.0]), x=np.zeros((2, 1)),
                             y=np.zeros((2, 1)), blocked_z=np.array([np.nan]),
                             blocked_key=[""], ray_weight=np.array([1.0]), weight=1.0)
    simulation = SimpleNamespace(incident=object(), branches={"transmitted": branch})
    monkeypatch.setattr(stem_signal, "sample_illumination_absent", lambda *_: False)
    monkeypatch.setattr(stem_signal, "paired_kick_response", lambda *_: np.zeros((2, 2)))
    monkeypatch.setattr(stem_signal, "raster_sample_grid",
                        lambda *_, **__: (np.zeros((3, 2)),) * 3)
    monkeypatch.setattr(stem_signal, "physical_angular_detectors", lambda *_: ((), {}))
    monkeypatch.setattr(stem_signal, "_current_values", lambda *_: (1.0, 1.0, 1.0))
    monkeypatch.setattr(stem_signal, "collection_angle", lambda *_: None)
    monkeypatch.setattr(stem_signal, "_stem_result",
                        lambda _state, _sim, _x, _y, images, _signals, _metrics, **_: images)
    monkeypatch.setattr(stem_signal, "_readout_view", lambda result, **_: result)

    def progress(done, total, label):
        progress_events.append((done, total, label))
        if cancel_mode == "display" and label == "STEM detector row 1/3":
            token.set()

    def run():
        return pipeline.calculate_stem_scan_frame(
            state, simulation, scan_calibrated=True,
            progress_callback=progress if cancel_mode != "state_only" else None)

    if cancel_mode == "none":
        result = run()
        assert len(hits) == 6
        np.testing.assert_array_equal(result[detector.key], np.ones((3, 2)))
        assert progress_events[-1] == (3, 3, "STEM detector row 3/3")
    else:
        with pytest.raises(RuntimeError, match="Superseded optical tuning request"):
            run()
        assert token.is_set()
        assert len(hits) == 2, "Cancellation must stop before the second detector row"
