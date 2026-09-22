"""Actual tip → extraction → acceleration → column → absorbing detector."""
from types import SimpleNamespace
import numpy as np
import pytest


def test_classical_surface_tip_axis_probe_traverses_full_column():
    from temsim.optics.column import default_state
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.gui.main_window import MainWindow
    from temsim.simulation_modes import switch_mode
    from temsim.physics.optical_tuning import prepare_tuning_snapshot
    from temsim.physics.simulation import run
    s = default_state()
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    catalog.apply(s, selection)
    proxy = SimpleNamespace(catalog=catalog,
                            _state_operating_mode_keys=MainWindow._state_operating_mode_keys)
    MainWindow._apply_state_operating_modes(proxy, s, selection)
    # This diagnostic probe belongs to the explicitly selected surface model.
    # The production default is planar classical emission, with no such probe.
    from temsim.optics.electron_gun.tip_assembly import model_from_part
    s.electron_gun.emitter.surface_model = model_from_part(s._resolved_assembly.part("feg_tip").data)
    assert s.electron_gun.emitter.surface_model.coherence is None
    switch_mode(s, "ideal")
    s.electron_gun.emitter.ray_count = 49
    s.step_mm = 1.
    prepare_tuning_snapshot(s, "Preview")
    result = run(s, optical_only=True)
    emitted = result.gun_trace.exit_bundle
    assert emitted.alive[-1] and emitted.weight[-1] == 0
    assert emitted.weight.sum() == pytest.approx(1.)
    assert result.incident.alive[-1]
    assert result.metrics["sample_support_probe_survivors"] == 1
    assert result.metrics["sample_beam_surviving_rays"] == 0
    assert result.metrics["sample_beam_surviving_fraction"] == 0.
    downstream = result.branches["000"]
    assert downstream.blocked_key[-1] == "bf"
    assert downstream.blocked_z[-1] == pytest.approx(next(d.z_mm for d in s.stem_detectors if d.key == "bf"))
    assert not downstream.alive[-1]
    assert np.isfinite(downstream.x[downstream.z <= downstream.blocked_z[-1], -1]).all()
