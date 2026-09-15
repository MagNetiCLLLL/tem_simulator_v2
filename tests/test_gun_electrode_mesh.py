"""Geometry-aware resolution without moving any electrode or tip boundary."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.physics.grounded_tip_field import refine_electrode_axes, field_request
from temsim.physics.grounded_tip_field import grade_electrode_corners
from temsim.physics.grounded_tip_field import merge_axis_nodes
from temsim.optics.electron_gun.tip_assembly import model_from_part
from temsim import module_manifest


def test_roundoff_slivers_merge_to_exact_boundaries_not_average_positions():
    edge = .0039
    alias = np.nextafter(edge,np.inf)
    original = np.array([0.,1e-12,2e-12,1e-9,edge,.02])
    merged = merge_axis_nodes(original,[alias,edge-2*np.spacing(edge)],boundaries=[edge,.02])
    np.testing.assert_array_equal(merged,original)
    assert edge in merged and alias not in merged
    with pytest.raises(ValueError,match="physical boundaries"):
        merge_axis_nodes(original,boundaries=[edge,alias])


def test_overlapping_corner_grids_cannot_create_sub_ulp_sliver_elements():
    r = np.array([0.,1e-9,.002,.004,.03])
    z = np.array([-.001,0.,1e-9,.0001,.0041,.0042,.0122,.45])
    rings = [("extractor",.0001,.0041,.002,.01,4500.),
             ("lens",.0042,.0122,.004,.03,1100.)]
    rr,zz = grade_electrode_corners(r,z,rings,16)
    for nodes in (rr,zz):
        assert np.all(np.diff(nodes) > 32*np.spacing(np.maximum(abs(nodes[:-1]),abs(nodes[1:]))))
    for _,start,stop,inner,outer,_ in rings:
        assert start in zz and stop in zz and inner in rr and outer in rr
    assert 1e-9 in rr and 1e-9 in zz


def test_local_refinement_preserves_original_nodes_and_fringe_resolution():
    r = np.array([0., 1e-9, .001, .005, .01, .03])
    z = np.array([-.001, 0., 1e-9, .005, .01, .015, .05, .2])
    rings = [("lens", .01, .015, .002, .01, 4000.)]
    rr, zz = refine_electrode_axes(r, z, rings, 8)
    assert np.all(np.isin(r, rr)) and np.all(np.isin(z, zz))
    assert np.all(np.diff(rr) > 0) and np.all(np.diff(zz) > 0)
    within = (zz[:-1] < .023) & (zz[1:] > .002)
    assert np.max(np.diff(zz)[within]) <= .002/8*(1+1e-12)
    within = rr[1:] < .004
    assert np.max(np.diff(rr)[within]) <= .002/8*(1+1e-12)
    original_r, original_z = refine_electrode_axes(r, z, rings, 0)
    assert original_r is r and original_z is z


def test_corner_grading_keeps_original_domain_and_exact_metal_edges():
    r = np.array([0., 1e-9, .002, .01, .03])
    z = np.array([-.001, 0., 1e-9, .01, .015, .2])
    rings = [("lens", .01, .015, .002, .01, 4000.)]
    rr, zz = grade_electrode_corners(r, z, rings, 8)
    assert np.all(np.isin(r, rr)) and np.all(np.isin(z, zz))
    assert np.all(np.diff(rr) > 0) and np.all(np.diff(zz) > 0)
    assert (rr[0],rr[-1],zz[0],zz[-1]) == (r[0],r[-1],z[0],z[-1])
    assert np.min(abs(rr[rr != .002]-.002)) <= .002/64*(1+1e-12)
    assert np.min(abs(zz[zz != .01]-.01)) <= .002/64*(1+1e-12)
    assert grade_electrode_corners(r,z,rings,0)[0] is r


def test_corner_grading_is_optional_and_is_bound_to_executed_field_identity():
    from temsim.optics.column import default_state
    from temsim.optics.electron_gun.tip_surface import TipSurfaceModel
    state = default_state()
    gun = state.electron_gun
    gun.emitter.surface_model = model_from_part(module_manifest.part_data("gun/FEG.toml", "feg_tip"))
    original = gun.emitter.surface_model
    old_key, old_field = gun._cache_key(193), field_request(gun)
    assert "electrode_corner_cells" not in original.to_dict()["field_numerics"]
    changed = replace(original, field_numerics=replace(original.field_numerics, electrode_corner_cells=16))
    gun.emitter.surface_model = changed
    assert TipSurfaceModel.from_dict(changed.to_dict()) == changed
    assert gun._cache_key(193) != old_key
    assert field_request(gun) != old_field
    assert changed.geometry == original.geometry and changed.emission == original.emission
    for value in (True, -1, 3, 65, 8.):
        with pytest.raises(ValueError, match="corner"):
            replace(changed.field_numerics, electrode_corner_cells=value).validate()


def test_installed_tip_numerics_are_editable_without_changing_emission_or_geometry():
    from temsim import module_manifest
    from temsim.optics.column import default_state
    from temsim.instrument_snapshot import capture_instrument_snapshot
    part = module_manifest.part_data("gun/FEG.toml", "feg_tip")
    original = model_from_part(part)
    part["tip_field_axis_core_fraction"] = .005
    part["tip_field_electrode_cells_per_bore"] = 16
    part["tip_field_electrode_corner_cells"] = 16
    changed = model_from_part(part)
    assert changed.geometry == original.geometry and changed.emission == original.emission
    assert changed.field_numerics.axis_core_fraction == .005
    assert changed.field_numerics.electrode_cells_per_bore == 16
    assert changed.field_numerics.electrode_corner_cells == 16
    state = default_state()
    state.electron_gun.emitter.surface_model = original
    old = field_request(state.electron_gun)
    state.electron_gun.emitter.surface_model = changed
    request = field_request(state.electron_gun)
    assert old != request and old["rings"] == request["rings"]
    restored = capture_instrument_snapshot(state).restore()
    assert field_request(restored.electron_gun) == request
    for key in ("tip_field_axis_core_fraction", "tip_field_electrode_cells_per_bore", "tip_field_electrode_corner_cells"):
        del part[key]
    assert model_from_part(part) == original  # older assembly stays readable


def test_missing_optional_controls_materialize_without_rewriting_comments():
    import tomllib
    from temsim.paths import INSTRUMENT_CONFIG_ROOT
    from temsim.shared_tip import materialized_text
    from temsim import module_manifest
    path = INSTRUMENT_CONFIG_ROOT / "gun/FEG.toml"
    original = path.read_text(encoding="utf-8-sig")
    assert "tip_field_axis_core_fraction" not in original
    result = materialized_text(original, path)
    source_comments = [line for line in original.splitlines() if line.lstrip().startswith("#")]
    assert all(line in result for line in source_comments)
    doc = tomllib.loads(result)
    tip = next(part for part in doc["parts"] if part["key"] == "feg_tip")
    assert tip["tip_field_axis_core_fraction"] == .01
    assert tip["tip_field_electrode_cells_per_bore"] == 8
    assert materialized_text(result, path) == result
    with pytest.raises(ValueError, match="Missing TOML field"):
        module_manifest.stage_manifest_text(original, {("parts", "feg_tip", "misspelled_field"): 1})


@pytest.mark.parametrize("value", [True, 1, 3, 65, -1, 8.0])
def test_invalid_electrode_mesh_budget_is_rejected(value):
    model = model_from_part(module_manifest.part_data("gun/FEG.toml", "feg_tip"))
    with pytest.raises(ValueError, match="Electrode cells"):
        replace(model.field_numerics, electrode_cells_per_bore=value).validate()
