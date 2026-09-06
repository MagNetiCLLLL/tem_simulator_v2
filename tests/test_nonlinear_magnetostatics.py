"""Reference-data and synthetic numerical tests, not OEM lens validation."""

from copy import deepcopy
from dataclasses import replace
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from temsim.magnetic_materials import BHCurve, MU0, reference_materials, validate_bh_material, import_bh_csv
from temsim.physics.nonlinear_magnetostatics import solve_nonlinear, prepared_mesh
from temsim.physics.axisymmetric_magnetostatics import solve_axisymmetric, _part_mask
from temsim.physics.lens_field_provider import active_mapped_providers, FieldMapError, FrozenMappedField


def _shared():
    fixture = runpy.run_path(str(Path(__file__).with_name("test_magnetic_circuits.py")))
    state = fixture["_state"](fixture["_shared_rows"]())
    for row in state.lens_field_map_descriptors.values():
        row.update(solver="axisymmetric_nonlinear_fem", bh_material=reference_materials()[0], ampere_turns=7000)
    return state


def _linear_curve(mur=50):
    return dict(label="Synthetic linear B-H", source_url="test:analytic", source_sha256="0"*64,
                b_t=[0, 1, 10], h_a_per_m=[0, 1/(MU0*mur), 10/(MU0*mur)])


def test_reference_data_are_sourced_and_not_rescaled():
    material = reference_materials()[0]
    assert material["source_sha256"] == "ba0ca80ffae56440a9ad785c9089b0e7f63b45e1c7a73dd5c0bdc338beeb63b7"
    assert len(material["b_t"]) == 21
    assert material["b_t"][-1] == 2.56 and material["h_a_per_m"][-1] == 318310
    curve = BHCurve(material)
    h, slope = curve.evaluate(material["b_t"])
    np.testing.assert_allclose(h, material["h_a_per_m"], rtol=1e-14)
    assert np.all(slope > 0)
    with pytest.raises(ValueError, match="range exceeded"):
        curve.evaluate([2.6])
    h, slope = curve.evaluate([2.6], allow_iteration_extension=True)
    assert slope[0] == 1/MU0 and h[0] > 318310


@pytest.mark.parametrize("change", [
    {"b_t": [0, 1, 1]}, {"h_a_per_m": [0, 2, 1]}, {"b_t": [1, 2, 3]},
    {"h_a_per_m": [0, float("nan"), 3]}, {"source_sha256": "unknown"},
    {"model": "hysteresis"}, {"extrapolation": "clamp"},
])
def test_invalid_bh_tables_fail(change):
    with pytest.raises(ValueError):
        validate_bh_material(dict(_linear_curve(), **change))


def test_csv_units_hash_and_embedded_snapshot(tmp_path):
    path = tmp_path/"curve.csv"
    path.write_text("B_T,H_A_per_m\n0,0\n1,100\n2,10000\n", encoding="utf-8")
    row = import_bh_csv(path)
    assert row["b_t"] == [0, 1, 2]
    assert row["source_url"] == path.as_uri()
    path.write_text("B,H\n0,0\n1,100\n2,10000\n", encoding="utf-8")
    with pytest.raises(ValueError, match="headers"):
        import_bh_csv(path)
    assert row["h_a_per_m"] == [0, 100, 10000]


def test_nonlinear_linear_limit_and_mesh_cache():
    r, z = np.linspace(0, .04, 18), np.linspace(-.04, .04, 30)
    mask = lambda r, z: (r > .004) & (r < .012) & (abs(z) < .02)
    current = lambda r, z: np.where((r > .016) & (r < .024) & (abs(z) < .02), 2e5, 0)
    linear = solve_axisymmetric(r, z, lambda r, z: np.where(mask(r, z), 50, 1), current)
    nonlinear = solve_nonlinear(r, z, lambda r, z: np.where(mask(r, z), 0, -1), current, [_linear_curve()])
    np.testing.assert_allclose(nonlinear.a_phi_tm, linear.a_phi_tm, rtol=1e-10, atol=1e-15)
    assert nonlinear.iterations == 1 and nonlinear.relative_residual < 1e-7
    assert prepared_mesh(tuple(r), tuple(z)) is prepared_mesh(tuple(r), tuple(z))


def test_manufactured_nonlinear_potential_converges_on_mesh_refinement():
    # Exact A=k*r*(R^2-r^2)*(L^2-z^2), H=nu0*(1+B^2)*B.
    # Analytical curl(H) supplies J_phi; a dense synthetic B-H table represents
    # the monotone constitutive law, independently of the FEM discretisation.
    radius, length, k, nu0 = .05, .05, 1e5, 1/(MU0*50)
    b = np.linspace(0, 5, 2001)
    material = dict(_linear_curve(), b_t=b.tolist(), h_a_per_m=(nu0*(1+b*b)*b).tolist())
    def forcing(r, z):
        br = 2*k*r*(radius**2-r*r)*z
        bz = 2*k*(radius**2-2*r*r)*(length**2-z*z)
        br_z = 2*k*r*(radius**2-r*r)
        br_r = 2*k*(radius**2-3*r*r)*z
        bz_r = -8*k*r*(length**2-z*z)
        bz_z = -4*k*(radius**2-2*r*r)*z
        return nu0*((1+br*br+bz*bz)*(br_z-bz_r)
                    + 2*((br*br_z+bz*bz_z)*br-(br*br_r+bz*bz_r)*bz))
    errors = []
    for n in (16, 32):
        r, z = np.linspace(0, radius, n), np.linspace(-length, length, 2*n)
        result = solve_nonlinear(r, z, lambda r, z: np.zeros_like(r, int), forcing, [material])
        exact = k*r[:, None]*(radius**2-r[:, None]**2)*(length**2-z[None, :]**2)
        errors.append(np.linalg.norm(result.a_phi_tm-exact)/np.linalg.norm(exact))
        assert result.relative_residual < 1e-7
    assert errors[1] < errors[0]*.4
    assert errors[1] < .01


def test_failed_iterations_and_out_of_range_do_not_return_a_field():
    r, z = np.linspace(0, .03, 16), np.linspace(-.03, .03, 26)
    labels = lambda r, z: np.zeros_like(r, int)
    current = lambda r, z: np.full_like(r, 1e5)
    with pytest.raises(ValueError, match="did not converge"):
        solve_nonlinear(r, z, labels, current, reference_materials(), max_iterations=1)
    with pytest.raises(ValueError, match="range exceeded"):
        solve_nonlinear(r, z, labels, lambda r, z: np.full_like(r, 1e10), reference_materials())


def test_joint_field_is_counted_once_cached_and_recomputed_at_new_current():
    from temsim.physics.nonlinear_circuits import _joint_map
    state = _shared()
    _joint_map.cache_clear()
    providers = active_mapped_providers(state)
    assert _joint_map.cache_info().misses == 1
    assert providers[0].model_status == "joint_nonlinear_field"
    assert providers[1].model_status == "included_in_joint_nonlinear_field"
    assert not np.any(providers[1].field_map.components_t)
    frozen = FrozenMappedField.from_provider(providers[0])
    initial = frozen.field_at_global_positions_t([[0, 0, 0]])
    assert active_mapped_providers(state)[0] is providers[0]
    state.lenses[0].percent = 50
    with pytest.raises(FieldMapError, match="operating point changed"):
        providers[0].excitation_scale()
    changed = active_mapped_providers(state)[0]
    assert _joint_map.cache_info().misses == 2
    assert not np.allclose(changed.magnetic_field_t([0]), initial[:, 2])
    np.testing.assert_array_equal(frozen.field_at_global_positions_t([[0, 0, 0]]), initial)
    state.lenses[0].percent = 100
    restored = active_mapped_providers(state)[0]
    assert _joint_map.cache_info().misses == 2
    np.testing.assert_array_equal(restored.field_map.components_t, providers[0].field_map.components_t)


def test_reference_saturation_polarity_zero_and_non_superposition():
    state = _shared()
    full = active_mapped_providers(state)[0].field_map.components_t[1].copy()
    state.lenses[1].percent = 0
    a = active_mapped_providers(state)[0].field_map.components_t[1]
    state.lenses[0].percent, state.lenses[1].percent = 0, 100
    b = active_mapped_providers(state)[0].field_map.components_t[1]
    # This is a nonlinear identity check, not an assumed OEM field target.
    assert np.linalg.norm(full-a-b) > 1e-3*np.linalg.norm(full)
    for lens in state.lenses:
        lens.percent, lens.polarity = 100, -1
    negative = active_mapped_providers(state)[0].field_map.components_t[1]
    np.testing.assert_allclose(negative, -full, rtol=1e-8, atol=1e-10)
    for lens in state.lenses:
        lens.percent = 0
    assert not np.any(active_mapped_providers(state)[0].field_map.components_t)


def test_shared_recipe_mismatch_and_geometry_edit_invalidate():
    state = _shared()
    before = active_mapped_providers(state)[0]
    state.lens_field_map_descriptors["b"]["radial_nodes"] += 1
    with pytest.raises(ValueError, match="identical"):
        active_mapped_providers(state)
    state.lens_field_map_descriptors["b"]["radial_nodes"] -= 1
    state._resolved_assembly.parts = tuple(replace(p, data={**p.data, "magnetic_radial_profile_mm": [[0, 3.2, 12], [2, 3.2, 12]]}) if p.key == "pole1" else p for p in state._resolved_assembly.parts)
    after = active_mapped_providers(state)[0]
    assert after.binding.geometry_fingerprint != before.binding.geometry_fingerprint
    assert after.field_map.content_fingerprint != before.field_map.content_fingerprint


def test_objective_intervals_and_projector_taper_use_existing_geometry():
    from temsim.optics.column import default_state
    from temsim.physics.lens_field_provider import lens_geometry_binding, resolve_runtime_lens_field_provider
    from temsim.magnetic_geometry import objective_layer_intervals_mm
    state = default_state()
    lens = next(l for l in state.lenses if l.key == "objective_lens")
    state.lens_field_map_descriptors[lens.key] = dict(solver="axisymmetric_linear_fem", relative_permeability=100, ampere_turns=100, radial_nodes=22, axial_nodes=40, padding_factor=2)
    binding = lens_geometry_binding(state, lens.key, lens)
    parts = json.loads(binding.canonical_geometry_json)["lens_assembly"]["parts"]
    parent = state._resolved_assembly.part("objective_lens")
    coil = next(p for p in parts if p["key"] == "objective_lens_excitation_coil")
    expected = objective_layer_intervals_mm(parent.data, parent.start_z_mm, "magnetic_excitation_coil")
    assert np.allclose(coil["data"]["material_intervals_mm"], expected)
    assert not _part_mask(coil, np.array([.04]), np.array([lens.z_mm*1e-3]))[0]
    assert np.all(np.isfinite(resolve_runtime_lens_field_provider(state, lens.key, lens).magnetic_field_t([lens.z_mm])))
    # The generic tapered pole narrows at its face while preserving an open bore.
    pole = dict(key="upper", start_z_mm=0, end_z_mm=10, data=dict(
        mechanical_profile="magnetic_pole_piece", pole_piece_geometry_style="tapered_bore_pole",
        mechanical_outer_diameter_mm=20, mechanical_tip_diameter_mm=8,
        mechanical_bore_diameter_mm=4, pole_nose_axial_length_mm=4))
    np.testing.assert_array_equal(_part_mask(pole, np.array([.009, .009, .003, .001]), np.array([.002, .009, .009, .009])), [True, False, True, False])


def test_nonlinear_menu_recipe_roundtrip_without_solving(qtbot, monkeypatch):
    from temsim.gui.model_inspector import ModelInspectorPage
    from temsim.gui.simulation_menu import SimulationMenu
    from temsim.optics.column import default_state
    from temsim.optics.model import State
    from temsim.simulation_modes import switch_mode, mode_key
    from temsim.physics import nonlinear_circuits
    monkeypatch.setattr(nonlinear_circuits, "_joint_map", lambda *a: pytest.fail("UI must not solve"))
    state = default_state()
    for lens in state.lenses:
        lens.enabled = lens.key == "projector_lens_1"
    page = ModelInspectorPage(); qtbot.addWidget(page); page.set_state(state)
    page.lens.setCurrentIndex(page.lens.findData("projector_lens_1"))
    page.field_solver.setCurrentIndex(1)
    page.ampere_turns.setText("1000")
    with qtbot.waitSignal(page.changed):
        page._apply_field()
    menu = SimulationMenu(); qtbot.addWidget(menu); menu.set_state(state)
    assert menu.mode_actions["nonlinear_material"].isEnabled()
    assert not menu.mode_actions["coupled_multiphysics"].isEnabled()
    before = [l.percent for l in state.lenses]
    switch_mode(state, "nonlinear_material")
    restored = State.from_dict(state.to_dict())
    assert mode_key(restored) == "nonlinear_material"
    assert restored.lens_field_map_descriptors == state.lens_field_map_descriptors
    assert before == [l.percent for l in state.lenses]


def test_active_foreign_coil_cannot_be_ignored_after_cached_solve():
    fixture = runpy.run_path(str(Path(__file__).with_name("test_magnetic_circuits.py")))
    part = fixture["_part"]
    rows = fixture["_shared_rows"]()+[
        part("c", "magnetic_lens_assembly", 13, 20, 2, 30),
        part("coil_c", "magnetic_excitation_coil", 14, 19, 16, 20, "c", field_source_key="c"),
    ]
    state = fixture["_state"](rows)
    for key in ("a", "b"):
        state.lens_field_map_descriptors[key].update(solver="axisymmetric_nonlinear_fem", bh_material=reference_materials()[0])
    state.lens_field_map_descriptors.pop("c")
    state.lenses[2].enabled = False
    active_mapped_providers(state)
    state.lenses[2].enabled = True
    with pytest.raises(FieldMapError, match="Active coil c overlaps"):
        active_mapped_providers(state)


def test_monolithic_insert_accepts_joint_bh_and_no_empirical_cs_is_added():
    fixture = runpy.run_path(str(Path(__file__).with_name("test_magnetic_circuits.py")))
    part = fixture["_part"]
    rows = [part("a", "magnetic_lens_assembly", -10, 10, 2, 30,
                 magnetic_circuit_id="mono", magnetic_circuit_topology="monolithic_saturated_insert"),
            part("coil", "magnetic_excitation_coil", -8, 8, 16, 20, "a", field_source_key="a"),
            part("insert", "magnetic_lens_yoke", -10, 10, 4, 14, "a", magnetic_part_role="saturating_insert",
                 magnetic_radial_profile_mm=[[0, 2, 7], [8, 2, 2.5], [12, 2, 2.5], [20, 2, 7]])]
    state = fixture["_state"](rows)
    state.lens_field_map_descriptors["a"].update(solver="axisymmetric_nonlinear_fem", bh_material=reference_materials()[0])
    assert active_mapped_providers(state)[0].model_status == "joint_nonlinear_field"
    from temsim.physics.core import spherical_aberration_kick_m3
    assert not np.any(spherical_aberration_kick_m3(np.linspace(-20, 20, 20), state))


def test_material_snapshot_participates_in_cache_and_diagnostic_identities():
    from temsim.physics.nonlinear_circuits import nonlinear_state_fingerprint
    from temsim.optics.column import default_state
    from temsim.calculation_cache import calculation_signatures
    state = default_state()
    state.lens_field_map_descriptors["projector_lens_1"] = dict(
        solver="axisymmetric_nonlinear_fem", ampere_turns=100, bh_material=reference_materials()[0])
    identity, signatures = nonlinear_state_fingerprint(state), calculation_signatures(state)
    state.lens_field_map_descriptors["projector_lens_1"]["bh_material"]["h_a_per_m"][1] *= 1.01
    assert nonlinear_state_fingerprint(state) != identity
    assert calculation_signatures(state) != signatures


def test_geometry_objective_does_not_receive_extra_chromatic_kick():
    from temsim.optics.column import default_state
    from temsim.physics.chromatic import configured_objective_chromatic_focal_mm
    state = default_state()
    state.chromatic_aberration_enabled = True
    for solver in ("axisymmetric_linear_fem", "axisymmetric_nonlinear_fem"):
        state.lens_field_map_descriptors["objective_lens"] = dict(solver=solver)
        assert configured_objective_chromatic_focal_mm(state) is None
    assert state.chromatic_aberration_enabled


def test_field_diagnostics_identify_joint_maps_not_native_gaussians():
    from temsim.diagnostics import _field_formula, _field_model_semantics
    state = _shared()
    providers = active_mapped_providers(state)
    keys = [_field_formula(lens, provider)[0] for lens, provider in zip(state.lenses, providers)]
    assert keys == ["joint_nonlinear", "joint_nonlinear_member"]
    for key, provider in zip(keys, providers):
        support, status, coupling = _field_model_semantics(key, provider)
        assert "grid support" in support and "B-H" in status
        assert "complete current vector" in coupling
    generic = replace(providers[0], model_status="measured_or_fem_geometry_bound")
    assert _field_formula(state.lenses[0], generic)[0] == "solver_provider"
