"""Input-only design export is not a cached downstream source."""
from dataclasses import replace
from zipfile import ZipFile

import pytest

from temsim.optics.column import default_state
from temsim.optics.gun_matching import candidate_with_gun_geometry, input_working_point
from temsim.working_point import WorkingPointCheckpoint


def test_design_package_restores_geometry_sampling_and_controls_without_cached_rays(tmp_path):
    original = default_state()
    s = candidate_with_gun_geometry(original, extractor_center_mm=2.1,
        lens_center_mm=8.2, accelerator_center_mm=218., dpa_center_mm=210.)
    s.electron_gun.electrostatic_lens.voltage_reference = 'tip'
    model = s.electron_gun.emitter.surface_model
    s.electron_gun.emitter.surface_model = replace(model, emission=replace(model.emission,
        angular_sampling='tangent_stratified_v2', angular_refinement_gain=80.,
        angular_stratum_allocation=(1,1,1,1,32,1,1,1,1), directions_per_position=72))
    s.objective_lens.percent = 68.89947594537043
    s._optical_tuning = True
    s._tuning_cancelled = lambda: False
    s.electron_gun._trace_cache = object()  # Must not be captured or restored.
    point = input_working_point(s,label='Detached classical gun design')
    assert s._optical_tuning and s._tuning_cancelled() is False
    assert not point.arrays
    assert point.observables.get('alpha95').status == 'NOT_COMPUTED'
    assert point.metadata['package_kind'] == 'INSTRUMENT_INPUTS_ONLY'
    path = tmp_path/'design.temwp'
    point.write_package(path)
    with ZipFile(path) as archive:
        assert archive.namelist() == ['manifest.json']
    restored = WorkingPointCheckpoint.read_package(path).compatible_state()
    assert not getattr(restored,'_optical_tuning',False)
    assert not hasattr(restored,'_tuning_cancelled')
    assert restored.electron_gun._trace_cache is None
    assert restored.electron_gun.emitter.surface_model == s.electron_gun.emitter.surface_model
    assert restored.objective_lens.percent == s.objective_lens.percent
    assert restored.electron_gun.electrostatic_lens.voltage_reference == 'tip'
    for key,z in [('feg_extractor',2.1),('feg_electrostatic_lens',8.2),
                  ('feg_accelerator',218.),('feg_dpa_aperture',210.)]:
        assert restored._resolved_assembly.part(key).center_z_mm == pytest.approx(z)
    assert original.electron_gun.electrostatic_lens.voltage_reference == 'extractor'


def test_input_package_rejects_an_empty_label():
    with pytest.raises(ValueError,match='label'):
        input_working_point(default_state(),label=' ')
