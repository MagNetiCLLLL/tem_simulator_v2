from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.optics.assembly_illumination import (
    TARGETS, assembly_combinations, seed_illumination, variable_lenses, load_targets, diameter_gate,
)
from temsim.optics.column import default_state
from temsim.optics.surface_probe_focus import SurfaceFocusMeasurement
from temsim.physics.beam_statistics import transverse_beam_statistics


def measurement(radius, slope):
    phi = np.linspace(0, 2*np.pi, 32, endpoint=False)
    s = transverse_beam_statistics(radius*np.cos(phi), radius*np.sin(phi),
        -slope*np.sin(phi), slope*np.cos(phi), weights=np.ones(32)/32)
    return SurfaceFocusMeasurement(100., s, 0., 2*slope*slope,
        (radius, radius, radius), .05, 32.)


def test_confirmed_targets_use_upper_surface_and_current_weighted_quantiles():
    nano = measurement(1e-9, np.tan(.030))
    micro = measurement(1e-6, np.tan(.0002))
    assert TARGETS['nano_probe'].accepts(nano)
    assert TARGETS['micro_probe'].accepts(micro)
    assert not TARGETS['micro_probe'].accepts(measurement(1e-6, np.tan(.000301)))
    assert not TARGETS['micro_probe'].accepts(measurement(2e-6, np.tan(.0002)))
    assert not TARGETS['nano_probe'].accepts(replace(nano, local_waist_offset_nm=2.))
    assert not TARGETS['micro_probe'].accepts(replace(micro, effective_rays=3.))
    assert not TARGETS['micro_probe'].accepts(replace(micro,
        statistics=replace(micro.statistics, radial_wavefront_curvature_per_m=26.)))


def test_all_supported_assemblies_are_audited_without_assumed_equivalence():
    catalog = AssemblyCatalog()
    combinations = assembly_combinations(catalog)
    assert len(combinations) == 60
    assert len(set(combinations)) == 60
    assert {c.gun for c in combinations} == {g.name for g in catalog.guns}


@pytest.mark.parametrize('column', ['C2', 'C3', 'C3 + Probe Corrector',
    'C3 + Image Corrector', 'C3 + Probe Corrector + Image Corrector'])
@pytest.mark.parametrize('mode', ['nano_probe', 'micro_probe'])
def test_seeds_preserve_gun_downstream_optics_and_installed_hardware(column, mode):
    s = default_state()
    AssemblyCatalog().apply(s, AssemblySelection('FEG', column, 'No Energy Filter'))
    gun_before = s.electron_gun.to_dict()
    downstream = [(l.key, l.percent, l.enabled) for l in s.lenses if l.z_mm > s.sample.z_mm]
    enabled = {l.key: l.enabled for l in s.lenses}
    positions = [(p.key, p.start_z_mm, p.end_z_mm) for p in s._resolved_assembly.parts]
    seed_illumination(s, mode)
    assert s.electron_gun.to_dict() == gun_before
    assert downstream == [(l.key, l.percent, l.enabled) for l in s.lenses if l.z_mm > s.sample.z_mm]
    assert enabled == {l.key: l.enabled for l in s.lenses}
    assert positions == [(p.key, p.start_z_mm, p.end_z_mm) for p in s._resolved_assembly.parts]
    keys = variable_lenses(s, mode)
    assert all(enabled[k] for k in keys)
    assert keys == ('mini_condenser', 'objective_lens')


def test_insufficient_current_is_not_a_valid_fit_residual():
    m = measurement(1e-6, .0002)
    with pytest.raises(ValueError, match='transmitted'):
        TARGETS['micro_probe'].residual(replace(m.statistics, surviving_rays=1))


def test_target_file_is_authoritative_and_rejects_invalid_values(tmp_path):
    assert load_targets() == TARGETS
    path = tmp_path / 'bad.toml'
    path.write_text('schema = "unknown"\n', encoding='utf-8')
    with pytest.raises(ValueError, match='schema'):
        load_targets(path)


def test_failed_budget_is_detached_and_counts_real_jacobian_probes(monkeypatch):
    import temsim.optics.assembly_illumination as module
    import temsim.optics.illumination_checkpoint as checkpoints
    calls = []
    class StubCheckpoint:
        prefix_z_mm = 1400.
        def __init__(self, state, keys, step_mm):
            pass
        def measure(self, vector):
            calls.append(tuple(vector))
            return measurement(1.5e-6, .0002)
    monkeypatch.setattr(checkpoints, 'IlluminationCheckpoint', StubCheckpoint)
    monkeypatch.setattr(module, 'measure_surface_focus', lambda *a, **k: measurement(1.5e-6, .0002))
    state = default_state()
    before = state.to_dict()
    candidate, report = module.calibrate_illumination(state, 'micro_probe', maximum_evaluations=1)
    assert state.to_dict() == before
    assert candidate is not state
    assert len(calls) == report['evaluations'] == 1
    assert report['status'] == 'NOT_QUALIFIED'
    assert not report['focus_angle_pass']
    assert 'budget exhausted' in report['attempts'][0]['error'].lower()


def test_fine_validation_cannot_publish_an_unresolved_candidate(monkeypatch):
    import temsim.optics.assembly_illumination as module
    import temsim.optics.illumination_checkpoint as checkpoints
    class StubCheckpoint:
        prefix_z_mm = 1400.
        def __init__(self, state, keys, step_mm):
            pass
        def measure(self, vector):
            return measurement(1e-6, .0002)
    monkeypatch.setattr(checkpoints, 'IlluminationCheckpoint', StubCheckpoint)
    monkeypatch.setattr(module, 'measure_surface_focus', lambda *a, **k: measurement(2e-6, .0002))
    _, report = module.calibrate_illumination(default_state(), 'micro_probe', maximum_evaluations=8)
    assert not report['focus_angle_pass']
    assert report['status'] == 'NOT_QUALIFIED'
    assert 'particle_sampling' in report['pending_gates']


@pytest.mark.parametrize('column,diameter_nm,expected', [
    ('C3 + Probe Corrector', .9, True), ('C3 + Probe Corrector', 1., False),
    ('C3', 2., True), ('C3', 10., True), ('C2', 10.1, False),
])
def test_diameter_gate_distinguishes_installed_corrector(column, diameter_nm, expected):
    state = default_state()
    AssemblyCatalog().apply(state, AssemblySelection('FEG', column, 'No Energy Filter'))
    m = measurement(diameter_nm*.5e-9, .03)
    # Isolate the threshold from trigonometric circle construction and its
    # one-ULP radial quantile error at exactly 10 nm.
    m = replace(m, statistics=replace(m.statistics, radius_95_m=diameter_nm/2e9))
    assert diameter_gate(state, m, 'nano_probe')['passed'] is expected
    assert diameter_gate(state, m, 'micro_probe')['passed']


def test_branch_search_keeps_searching_when_corrected_probe_is_too_large(monkeypatch):
    import temsim.optics.assembly_illumination as module
    import temsim.optics.illumination_checkpoint as checkpoints
    calls = []
    class StubCheckpoint:
        prefix_z_mm = 1400.
        def __init__(self, state, keys, step_mm):
            pass
        def measure(self, vector):
            calls.append(tuple(vector))
            return measurement((.75 if vector[0] == 35. else .4)*1e-9, np.tan(.030))
    monkeypatch.setattr(checkpoints, 'IlluminationCheckpoint', StubCheckpoint)
    monkeypatch.setattr(module, '_proposals', lambda *a, **k: [np.array([35.,69.]), np.array([40.,69.])])
    monkeypatch.setattr(module, '_focus_branch', lambda measure, target, seed, upper: seed)
    monkeypatch.setattr(module, 'measure_surface_focus', lambda *a, **k: measurement(.4e-9, np.tan(.030)))
    state = default_state()
    AssemblyCatalog().apply(state, AssemblySelection('FEG', 'C3 + Probe Corrector', 'No Energy Filter'))
    _, report = module.calibrate_illumination(state, 'nano_probe', search_branches=True)
    assert calls == [(35.,69.), (40.,69.)]
    assert report['diameter_gate']['passed']
    assert report['status'] == 'NOT_QUALIFIED'  # Independent gates still pending.


@pytest.mark.parametrize('diameter', [0., -1., np.nan, np.inf])
def test_invalid_physical_aperture_candidate_is_rejected_without_mutating_input(diameter):
    from temsim.optics.assembly_illumination import calibrate_illumination
    state = default_state()
    before = state.to_dict()
    with pytest.raises(ValueError, match='physical C2 aperture'):
        calibrate_illumination(state, 'nano_probe', c2_aperture_mm=diameter)
    assert state.to_dict() == before


def test_failed_physical_traces_also_consume_the_calibration_budget(monkeypatch):
    import temsim.optics.assembly_illumination as module
    import temsim.optics.illumination_checkpoint as checkpoints
    calls = []
    class StubCheckpoint:
        prefix_z_mm = 1400.
        def __init__(self, state, keys, step_mm):
            pass
        def measure(self, vector):
            calls.append(tuple(vector))
            raise ValueError('No finite surviving rays are available')
    monkeypatch.setattr(checkpoints, 'IlluminationCheckpoint', StubCheckpoint)
    monkeypatch.setattr(module, '_proposals', lambda *a, **k: [np.array([35.,69.]), np.array([40.,69.])])
    candidate, report = module.calibrate_illumination(default_state(), 'nano_probe',
        search_branches=True, maximum_evaluations=1)
    assert candidate is None
    assert len(calls) == report['evaluations'] == 1
    assert len(report['evaluation_failures']) == 1
    assert any('budget exhausted' in a.get('error', '').lower() for a in report['attempts'])


def test_historical_presets_are_visibly_distinguished_from_new_targets(qtbot):
    from temsim.gui.assembly_panel import AssemblyPanel
    catalog = AssemblyCatalog()
    panel = AssemblyPanel(catalog, catalog.default_selection())
    qtbot.addWidget(panel)
    assert panel.apply_operating_mode_button.text() == 'Apply stored lens preset'
    assert 'recalibration pending' in panel.operating_mode_status.text()
    assert 'Historical reference only' in panel.operating_mode_status.toolTip()


def test_candidate_polarity_is_detached_and_reported_for_exact_replay(monkeypatch):
    import temsim.optics.assembly_illumination as module
    import temsim.optics.illumination_checkpoint as checkpoints
    class StubCheckpoint:
        prefix_z_mm = 1400.
        def __init__(self, state, keys, step_mm):
            assert next(l for l in state.lenses if l.key=='mini_condenser').polarity == -1
        def measure(self, vector):
            return measurement(1e-6,.0002)
    monkeypatch.setattr(checkpoints,'IlluminationCheckpoint',StubCheckpoint)
    monkeypatch.setattr(module,'measure_surface_focus',lambda *a,**k:measurement(1e-6,.0002))
    state = default_state()
    before = state.to_dict()
    candidate, report = module.calibrate_illumination(state,'micro_probe',mini_polarity=-1)
    assert report['polarities']['mini_condenser']==-1
    assert state.to_dict()==before
    assert candidate is not state
    assert report['status']=='NOT_QUALIFIED'


@pytest.mark.parametrize('polarity',[0,2,-2,float('nan')])
def test_invalid_polarity_rejected_before_transport(polarity):
    from temsim.optics.assembly_illumination import calibrate_illumination
    with pytest.raises(ValueError,match='polarity'):
        calibrate_illumination(default_state(),'nano_probe',mini_polarity=polarity)


def test_focus_fit_aims_at_centre_not_first_point_inside_acceptance():
    from temsim.optics.assembly_illumination import _focus_branch
    calls=[]
    def measure(v):
        calls.append(tuple(v))
        # Exact focus at every point; primary controls alpha in mrad.
        return measurement(.1e-9,np.tan(v[0]*.001))
    result=_focus_branch(measure,TARGETS['nano_probe'],np.array([30.25,70.]),np.array([100.,100.]))
    assert len(calls)>1
    assert result[0]==pytest.approx(30.,abs=1e-5)
