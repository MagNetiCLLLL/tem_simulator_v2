"""Matching geometry is exact; these tests do not qualify a propagated probe."""
from dataclasses import replace
import math

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.optics.tip_curvature_comparison import (
    flat_tip_d95_nm, matched_cap_angle_deg, matched_curved_candidate,
)
from temsim.optics.beam_path_audit import uniform_cap_footprint
from temsim.optics.electron_gun.tip_assembly import model_from_part, validate_tip_part
from temsim.instrument_snapshot import capture_instrument_snapshot


def test_flat_quantile_matches_emission_radial_cdf():
    emitter = default_state().electron_gun.emitter
    d95 = flat_tip_d95_nm(emitter)
    sigma = emitter.virtual_source_fwhm_nm / 2.354820045
    cdf = -math.expm1(-.5*(d95/2/sigma)**2) / -math.expm1(-4.5)
    assert cdf == pytest.approx(.95)
    assert d95 > emitter.virtual_source_fwhm_nm


@pytest.mark.parametrize("radius", [10.,100.,200.,500.,1e8])
def test_cap_quantile_inversion_and_depth(radius):
    state = default_state()
    model = model_from_part(state._resolved_assembly.part("feg_tip").data)
    d95 = flat_tip_d95_nm(state.electron_gun.emitter)
    angle = matched_cap_angle_deg(d95,radius)
    model = replace(model, geometry=replace(model.geometry,apex_radius_nm=radius,
        shank_length_um=max(1000.,radius*.002)),
        emission=replace(model.emission,cap_half_angle_deg=angle))
    assert uniform_cap_footprint(model)["diameter95_nm"] == pytest.approx(d95,rel=1e-13)


def test_candidate_changes_actual_geometry_but_not_original_or_emission_law():
    state = default_state()
    before = capture_instrument_snapshot(state).digest
    old_recipe = model_from_part(state._resolved_assembly.part("feg_tip").data)
    candidate = matched_curved_candidate(state,500.)
    model = candidate.electron_gun.emitter.surface_model
    assert capture_instrument_snapshot(state).digest == before
    assert model.geometry.apex_radius_nm == 500.
    assert model.emission == replace(old_recipe.emission,cap_half_angle_deg=model.emission.cap_half_angle_deg)
    assert model.emission.flux_electrons_per_nm2_s == old_recipe.emission.flux_electrons_per_nm2_s
    assert model.current_na < old_recipe.current_na
    part = candidate._resolved_assembly.part("feg_tip")
    validate_tip_part(part.data)
    assert part.data["tip_radius_nm"] == 500.
    module = next(m for m in candidate._resolved_assembly.modules if m.key == part.module_key)
    assert next(p for p in module.parts if p.key == part.key).data == part.data
    from temsim.column.module_assembly import ModulePart
    assert isinstance(next(p for p in module.parts if p.key == part.key),ModulePart)
    assert [l.percent for l in candidate.lenses] == [l.percent for l in state.lenses]
    assert candidate.electron_gun.extractor.voltage_kv == state.electron_gun.extractor.voltage_kv
    assert candidate.electron_gun.accelerator.stages == state.electron_gun.accelerator.stages
    assert capture_instrument_snapshot(candidate).restore().electron_gun.emitter.surface_model == model


@pytest.mark.parametrize("d,r", [(0.,100.),(10.,0.),(np.nan,100.),(10.,np.inf),(201.,100.)])
def test_invalid_match_is_rejected(d,r):
    with pytest.raises(ValueError):
        matched_cap_angle_deg(d,r)


def test_lower_curvature_input_package_round_trip_has_no_calculation_cache(tmp_path):
    from zipfile import ZipFile
    from temsim.optics.gun_matching import input_working_point
    from temsim.working_point import WorkingPointCheckpoint
    candidate = matched_curved_candidate(default_state(),500.)
    path = tmp_path / "matched-tip.temwp"
    input_working_point(candidate,label="Curvature comparison - NOT QUALIFIED").write_package(path)
    with ZipFile(path) as archive:
        assert archive.namelist() == ["manifest.json"]
    restored = WorkingPointCheckpoint.read_package(path).compatible_state()
    assert restored.electron_gun.emitter.surface_model == candidate.electron_gun.emitter.surface_model
    assert restored._resolved_assembly.part("feg_tip").data["tip_radius_nm"] == 500.
    assert restored.electron_gun._trace_cache is None
