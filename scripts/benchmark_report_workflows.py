"""Bounded CPU product reuse, resumable experiments and isolated UI timings.

Run alone with the project interpreter. Uses the production classical pipeline;
never enables waves or changes saved settings. Results are scalar receipts.
"""
import argparse
import json
import os
from pathlib import Path
import platform
from statistics import median
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import perf_counter
from unittest.mock import patch

import numpy as np
import psutil
from numba import set_num_threads
from threadpoolctl import threadpool_limits


def measure(function):
    process = psutil.Process()
    baseline = process.memory_info().rss
    peaks = [baseline]
    done = Event()
    def sample():
        while not done.wait(.02):
            peaks.append(process.memory_info().rss)
    monitor = Thread(target=sample, daemon=True)
    monitor.start()
    start = perf_counter()
    try:
        value = function()
        peaks.append(process.memory_info().rss)
        return value, dict(wall_s=perf_counter()-start, rss_before_bytes=baseline,
            sampled_peak_rss_bytes=max(peaks), rss_after_bytes=peaks[-1])
    finally:
        done.set()
        monitor.join()


def run():
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.column import default_state
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.calculation_cache import calculation_signatures
    from temsim.simulation_pipeline import calculate
    from temsim.design_explorer import capture_design_snapshot, HighAccuracyRequest
    from temsim.design_experiments import recipe_from_snapshot, plan_parameter_sweep, SweepAxis
    from temsim.design_sweep_execution import execute_parameter_sweep
    from temsim.experiment_records import save_experiment, load_experiment
    state = default_state()
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    catalog.apply(state, selection)
    # Local execution budgets, not persisted source/optical defaults.
    state.electron_gun.emitter.ray_count = 9
    state.step_mm = state.history_step_mm = 2.5
    state.acceleration_backend = 'CPU'
    assert not state.sample.wave_enabled, 'This benchmark cannot enable paused wave physics'
    snapshot = capture_instrument_snapshot(state)
    first, cold = measure(lambda: calculate(snapshot.restore()))
    cold['calculated_products'] = sorted(first.calculated_products)
    cold['pipeline'] = first.performance
    assert first.wave_imaging is None
    same, edits = [], []
    for index in range(5):
        current = snapshot.restore()
        _, identity_timing = measure(lambda: calculation_signatures(current))
        reused, timing = measure(lambda: calculate(current, existing_result=first))
        timing.update(calculated_products=sorted(reused.calculated_products),
            reused_products=sorted(reused.reused_products), identity_lookup=identity_timing)
        same.append(timing)
        changed = snapshot.restore()
        lens = next(l for l in changed.lenses if l.key == 'projector_lens_2')
        lens.percent += .1 * (index + 1)
        changed_result, timing = measure(lambda: calculate(changed, existing_result=first))
        timing.update(calculated_products=sorted(changed_result.calculated_products),
            reused_products=sorted(changed_result.reused_products),
            changed_control='lenses[projector_lens_2].percent', value=lens.percent)
        edits.append(timing)
    recipe = recipe_from_snapshot(capture_design_snapshot(state, selection, slot='A',
        request=HighAccuracyRequest(9, 2.5)), name='Bounded CPU benchmark')
    lens = next(l for l in state.lenses if l.key == 'projector_lens_2')
    sweep = plan_parameter_sweep(recipe, (SweepAxis('lenses[projector_lens_2].percent',
        (lens.percent, lens.percent+.1, lens.percent+.2), '%'),))
    calls = []
    def observed(current, **kwargs):
        result = calculate(current, **kwargs)
        calls.append(result.signatures['column'])
        return result
    prefix, cancelled_timing = measure(lambda: execute_parameter_sweep(recipe, sweep,
        calculator=observed, cancel_requested=lambda: len(calls) >= 1))
    assert prefix.cancelled and prefix.completed_points == 1
    with TemporaryDirectory(prefix='temsim-report-benchmark-') as directory:
        path = Path(directory)/'study.temexp'
        _, write_timing = measure(lambda: save_experiment(path, recipe, sweep, prefix))
        loaded, read_timing = measure(lambda: load_experiment(path))
        archive_bytes = path.stat().st_size
        resumed, resume_timing = measure(lambda: execute_parameter_sweep(loaded[0], loaded[1],
            calculator=observed, resume=loaded[2]))
        assert not resumed.cancelled and resumed.completed_points == 3 and len(calls) == 3
        assert all(row.status == 'COMPLETE' for row in resumed.point_results)
        assert prefix.execution_identity == resumed.execution_identity
        assert prefix.point_results[0].request_signature == resumed.point_results[0].request_signature
        ui = measure_ui(Path(directory))
    return dict(schema='report-workflow-benchmark-v1', host=dict(platform=platform.platform(),
        processor=platform.processor(), python=platform.python_version(), logical_cpus=os.cpu_count(),
        physical_ram_bytes=psutil.virtual_memory().total, actual_backend='CPU', library_threads=4),
        input_identity=snapshot.digest, implementation=snapshot.implementation,
        rays=9, step_mm=2.5, initial=cold, repeated_product_reuse=same,
        repeated_reuse_median_s=median(r['wall_s'] for r in same), single_control_edits=edits,
        edited_median_s=median(r['wall_s'] for r in edits),
        sweep=dict(cancel_after_one=cancelled_timing, resume_remaining_two=resume_timing,
            write=write_timing, read=read_timing, archive_bytes=archive_bytes,
            retained_prefix_identical=True, execution_identity=resumed.execution_identity,
            points=[dict(index=r.point_index, status=r.status, numerical_status=r.numerical_status,
                         duration_s=r.duration_s, reused=list(r.reused_products)) for r in resumed.point_results]),
        presentation=ui,
        limits='Nine-ray classical software workflow only; no convergence, CUDA, image or OEM qualification. '
            'RSS sampled every 20 ms can miss short peaks. One cancellation/resume run has no percentile claim. '
            'UI uses offscreen empty-result task layouts; not native desktop or populated mesh performance.')


def measure_ui(directory):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from temsim.gui import main_window as shell, calculation_controller as controller, interactive_calculation as interactive
    from temsim.instrument_snapshot import capture_instrument_snapshot
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(directory/'isolated.ini'), QSettings.Format.IniFormat)
    with patch.object(shell, 'QSettings', return_value=settings), \
            patch.object(interactive, 'QSettings', return_value=settings), \
            patch.object(controller, 'default_artifact_cache_root', return_value=directory/'cache'), \
            patch.object(shell.MainWindow, 'INITIAL_PREVIEW_DELAY_MS', 600000):
        window, startup = measure(shell.MainWindow)
        window.preview_timer.stop()
        window.resize(1500, 920)
        window.show()
        app.processEvents()
        identity = capture_instrument_snapshot(window.state).digest
        samples = []
        with patch.object(window.calculations.pool, 'start', side_effect=AssertionError('Layout submitted physics')) as submit, \
                patch.object(window, 'restoreGeometry', return_value=True):
            for name in ('instrument', 'alignment', 'experiments', 'results') * 5:
                start = perf_counter()
                window.workspace_layouts.select('task_'+name)
                window.resize(1500, 920)
                app.processEvents()
                samples.append(perf_counter()-start)
            assert capture_instrument_snapshot(window.state).digest == identity
            count = submit.call_count
        window.close()
        app.processEvents()
    return dict(startup=startup, layout_samples_s=samples, median_s=median(samples),
        p95_s=float(np.percentile(samples, 95)), submitted_physics_calls=count,
        instrument_identity_unchanged=True, viewport_logical_px=[1500, 920])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    set_num_threads(4)
    with threadpool_limits(limits=4):
        receipt = run()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({key:receipt[key] for key in ('repeated_reuse_median_s', 'edited_median_s')}, indent=2))
