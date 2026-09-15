"""Scalar report checks cannot turn a fitted or under-resolved spot into proof."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def gate(monkeypatch):
    scripts = Path(__file__).parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('gun_focus_report_gate',scripts/'summarise_gun_focus_validation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.focus_row


@pytest.fixture
def report():
    return {'focus':dict(statistics=dict(surviving_rays=1000,surviving_fraction=7.5e-5,
        convergence_95_rad=.03,radius_95_m=.7e-9,waist_offset_m=0.),
        effective_rays=800.,local_waist_offset_nm=0.,variance_second_derivative=.001,
        surface_z_mm=1599.1999975,step_mm=.025)}


def test_gate_accepts_resolved_comparisons_without_fitting(gate,report):
    ref = gate(report)
    other = deepcopy(report)
    other['focus']['statistics']['radius_95_m'] *= 1.05
    assert ref['passed'] and gate(other,ref)['passed']


@pytest.mark.parametrize('field,value', [('radius_95_m',1e-6),('convergence_95_rad',.034),
    ('surviving_fraction',0.),('waist_offset_m',2e-9)])
def test_gate_rejects_inadequate_physical_observables(gate,report,field,value):
    report['focus']['statistics'][field] = value
    assert not gate(report)['passed']


def test_gate_rejects_sampling_instability_and_tiny_effective_population(gate,report):
    ref = gate(report)
    report['focus']['statistics']['surviving_fraction'] *= 1.2
    assert not gate(report,ref)['passed']
    report['focus']['effective_rays'] = 3
    assert not gate(report)['passed']
