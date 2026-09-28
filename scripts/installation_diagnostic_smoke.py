"""Check an isolated installed wheel without starting coherent calculations.

Run with the installed interpreter's ``-I`` option, outside the checkout. This
bounded smoke executes the actual tip-origin classical optical column, a short
independent uniform-B diagnostic, safe diagnostic persistence, and both panels
offscreen. It is installation evidence, not full microscope qualification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import traceback

# Set the explicit one-thread smoke budget before importing numerical libraries.
for _name in ("TEMSIM_CPU_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["QT_QPA_PLATFORM"] = "offscreen"


def column_history_summary(simulation) -> dict:
    """Validate retained optical branches and distinguish specimen/end planes."""
    import numpy as np
    branches = (simulation.incident, *simulation.branches.values())
    for index, branch in enumerate(branches):
        z = np.asarray(branch.z)
        if z.ndim != 1 or not z.size or not np.isfinite(z).all():
            raise AssertionError(f"Invalid axial history in installed optical branch {index}")
        for name in ("x", "y", "tx", "ty"):
            if not np.isfinite(getattr(branch, name)).all():
                raise AssertionError(f"Nonfinite {name} in installed optical branch {index}")
    return {"incident_endpoint_mm": float(simulation.incident.z[-1]),
            "column_endpoint_mm": max(float(np.max(branch.z)) for branch in branches),
            "branches_checked": len(branches)}


def installation_checks(output_directory: Path) -> dict:
    import numpy as np
    import temsim
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.calculation_manifest import solver_source_identity
    from temsim.cpu_resources import numerical_job
    from temsim.electron_diagnostic_session import (
        DiagnosticDependency, DiagnosticDisplayState, DiagnosticElectronRecord,
        DiagnosticSession, load_diagnostic_session, save_diagnostic_session,
    )
    from temsim.gui.hardware_tuning_panel import HardwareTuningPanel
    from temsim.gui.magnetic_test_electron import TestElectronController
    from temsim.magnetic_test_particle import TestElectronSettings, trace_test_electron
    from temsim.optics.column import default_state
    from temsim.paths import CONFIG_ROOT
    from temsim.physics.simulation import run
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QDockWidget, QMainWindow

    if not sys.flags.isolated:
        raise RuntimeError("Use the installed interpreter with -I for this smoke")
    prefix = Path(sys.prefix).resolve()
    installed_module = Path(temsim.__file__).resolve()
    if not installed_module.is_relative_to(prefix):
        raise RuntimeError("temsim resolved outside the isolated installation prefix")
    if Path(CONFIG_ROOT).resolve() != prefix / "configs":
        raise RuntimeError("Configuration did not resolve to the installed wheel's configs")
    session_path = output_directory / "installation-electron.temdiag"
    if session_path.exists():
        raise ValueError("Choose a fresh output directory to preserve the previous diagnostic session")

    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    state.acceleration_enabled = False  # Select the reference CPU backend, not gun physics.
    state.acceleration_backend = "CPU"
    state.electron_gun.emitter.ray_count = 9
    state.step_mm = 5.
    state.history_step_mm = 5.
    if state.sample.wave_enabled or state.sample.stem_wave_enabled:
        raise AssertionError("Installed startup unexpectedly enabled coherent observables")
    if state.electron_gun.emitter.coherence is not None:
        raise AssertionError("Installed startup unexpectedly configured a coherent tip")
    with numerical_job(requested=1) as resources:
        start = perf_counter()
        simulation = run(state, optical_only=True)
        column_seconds = perf_counter() - start
    gun = simulation.gun_trace
    if gun is None or simulation.incident.x.shape[1] != 9:
        raise AssertionError("The installed column did not execute the requested tip population")
    column_history = column_history_summary(simulation)

    class UniformMagneticFixture:
        """Independent specified field, not a captured microscope source."""

        bounds_m = np.array(((-.01, -.01, -.01), (.01, .01, .01)))
        field = np.array((0., 0., .01))
        notes = ("Independent installation fixture: uniform Bz=0.01 T, E=0; not a microscope field capture.",)

        def contains(self, points):
            return ((points >= self.bounds_m[0]) & (points <= self.bounds_m[1])).all(axis=1)

        def field_at_global_positions_t(self, points):
            if not self.contains(points).all():
                raise AssertionError("Installation fixture sampled outside its declared domain")
            return np.broadcast_to(self.field, points.shape).copy()

    scene = UniformMagneticFixture()
    settings = TestElectronSettings(kinetic_energy_ev=200000., polar_angle_deg=1.,
                                    max_path_length_m=.001, step_m=1e-5, max_steps=1000)
    with numerical_job(requested=1):
        start = perf_counter()
        result = trace_test_electron(scene, settings, use_compiled=False)
        diagnostic_seconds = perf_counter() - start
    if not result.completed or result.reason != "path_limit":
        raise AssertionError("Installed reference diagnostic did not reach its declared path limit")
    np.testing.assert_allclose(result.kinetic_energy_ev, settings.kinetic_energy_ev, rtol=1e-12)
    if not 0 < result.steps <= settings.max_steps:
        raise AssertionError("Installed diagnostic has no accepted numerical steps")

    dependency = DiagnosticDependency(
        result_reference="installation-fixture:uniform-Bz-0.01-T-E0",
        provider_notes=scene.notes, bounds_m=scene.bounds_m,
    )
    session = DiagnosticSession(
        records=(DiagnosticElectronRecord(
            key="electron-1", label="Installation electron", colour="#ffd166",
            settings=settings, trajectory=result, trajectory_settings=settings, state="completed",
        ),), dependency=dependency,
        display=DiagnosticDisplayState(selected_key="electron-1", show_background=False),
    )
    save_diagnostic_session(session, session_path)
    restored = load_diagnostic_session(session_path)
    saved = restored.records[0]
    if saved.settings != settings or saved.trajectory_settings != settings:
        raise AssertionError("Diagnostic settings changed during installed archive roundtrip")
    for name in ("positions_m", "directions", "time_s", "path_length_m", "kinetic_energy_ev",
                 "electrostatic_potential_v", "speed_m_per_s", "momentum_kg_m_per_s"):
        np.testing.assert_array_equal(getattr(saved.trajectory, name), getattr(result, name))
        if getattr(saved.trajectory, name).flags.writeable:
            raise AssertionError("Loaded diagnostic arrays are not immutable")
    if (saved.trajectory.reason, saved.trajectory.completed) != (result.reason, result.completed):
        raise AssertionError("Diagnostic terminal status changed during archive roundtrip")

    app = QApplication.instance() or QApplication([])
    owner = QMainWindow()
    panel = HardwareTuningPanel(owner)
    controller = TestElectronController(owner)
    dock = QDockWidget("Virtual electrons", owner)
    dock.setObjectName("installationDiagnosticSmokeDock")
    dock.setWidget(controller.panel)
    owner.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
    owner.setCentralWidget(panel)
    try:
        before = state.to_dict()
        changes = []
        panel.runtime_changed.connect(changes.append)
        panel.set_state(state)
        panel.select_task("beam_shift")
        owner.resize(1200, 720)
        owner.show()
        controller.panel.show()
        app.processEvents()
        panel.select_task("beam_tilt")
        app.processEvents()
        if changes or state.to_dict() != before:
            raise AssertionError("Opening/selecting installed hardware tuning mutated machine state")
        if controller._worker is not None or controller._scene_worker is not None:
            raise AssertionError("Opening diagnostic controls unexpectedly started a calculation")
        if not panel.isVisible() or not controller.panel.isVisible():
            raise AssertionError("Installed GUI panels did not become visible offscreen")
    finally:
        controller.shutdown()
        owner.close()
        app.processEvents()

    return {
        "status": "PASS", "installed_module": str(installed_module), "python_prefix": str(prefix),
        "python_executable": sys.executable, "isolated": bool(sys.flags.isolated),
        "installed_config_root": str(Path(CONFIG_ROOT).resolve()),
        "solver_source_sha256": solver_source_identity(), "cpu_resources": resources.to_dict(),
        "particle_column": {
            "status": "PASS", "rays": 9, "seconds": column_seconds,
            "scope": "Tip emission, extraction, acceleration, gun focusing, apertures and optical column; no specimen signals",
            "gun_endpoint_mm": float(gun.z_mm[-1]),
            **column_history,
            "electrostatic_report_present": bool(getattr(gun, "electrostatic_model_report", None)),
        },
        "diagnostic": {
            "status": "PASS", "fixture": scene.notes[0], "seconds": diagnostic_seconds,
            "steps": int(result.steps), "reason": result.reason, "completed": bool(result.completed),
            "relative_energy_error": float(result.energy_invariant_relative_error),
            "full_field_identity": "UNKNOWN_SYNTHETIC_FIXTURE_NOT_REUSABLE_AS_MICROSCOPE_STATE",
        },
        "session": {"status": "PASS", "path": str(session_path.resolve()),
                    "bytes": session_path.stat().st_size, "exact_arrays": 8, "load_mode": "history only"},
        "gui": {"status": "PASS", "mode": "offscreen", "panels": ["Hardware tuning", "Virtual electrons"],
                "read_only_selection": True, "numerical_work_started_by_opening": False},
        "versions": {name: metadata.version(name) for name in ("numpy", "scipy", "PySide6", "tem-simulator-v2")},
        "coherent_execution": "NOT_RUN_PAUSED", "gpu_validation": "NOT_RUN",
        "full_microscope_qualification": "NOT_ESTABLISHED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists():
        parser.error("Choose a new output report to preserve previous installation evidence")
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "installation-diagnostic-smoke-v1", "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Isolated installed wheel: bounded classical column, independent diagnostic fixture, session and offscreen panels",
    }
    exit_code = 0
    try:
        receipt.update(installation_checks(destination.parent))
    except Exception:
        receipt.update(status="FAILED", traceback=traceback.format_exc())
        exit_code = 1
    receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
    destination.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "report": str(destination)}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
