"""Small, independent checks of the explicitly selected reference gun geometry.

These are topology, electrode-boundary and cache checks, not a qualification of
the patent's hardware, electron emission, or the complete microscope chain.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from temsim.physics import patent_gun_reference as reference
from temsim.physics import planar_gun_field as planar


CATHODE = "planar_equipotential"


class ReferenceLens(SimpleNamespace):
    def potential_rise_from_tip_v(self, extraction_kv, high_tension_kv):
        bases = {"tip": 0.0, "extractor": extraction_kv,
                 "ground": high_tension_kv}
        return (bases[self.voltage_reference] + self.voltage_kv) * 1000.0


@pytest.fixture
def reference_gun():
    return SimpleNamespace(
        emitter=SimpleNamespace(surface_model=None, curvature_nm_inv=0.0,
                                coherence=None, ray_count=11,
                                emission_current_na=2.0,
                                virtual_source_fwhm_nm=5.0),
        extractor=SimpleNamespace(voltage_kv=4.0),
        electrostatic_lens=ReferenceLens(voltage_kv=1.2,
                                         voltage_reference="extractor",
                                         potential_scale=4.22125),
        accelerator=SimpleNamespace(high_tension_kv=300.0,
                                     mechanical_center_from_tip_mm=5.0,
                                     mechanical_length_mm=2.0,
                                     stages=[]),
        exit_plane_z_mm=8.0, monochromator_installed=False,
        source_representation="classical_particles",
    )


def request(gun, **options):
    return reference.patent_reference_request(
        gun, cathode_boundary=CATHODE, **{"cells_per_bore": 4, **options})


def nodes_by_key(result):
    return {node["key"]: node for node in result["electrode_nodes"]}


def test_seven_nodes_and_five_equal_divider_intervals(reference_gun):
    result = request(reference_gun)
    nodes = nodes_by_key(result)
    assert set(nodes) == {"extractor", "control", "terminal"} | {
        f"accelerator:{index}" for index in range(1, 5)}
    assert len(result["electrode_nodes"]) == 7
    assert nodes["extractor"]["divider_index"] is None
    divider = sorted((node for node in nodes.values()
                      if node["divider_index"] is not None),
                     key=lambda node: node["divider_index"])
    assert [node["divider_index"] for node in divider] == list(range(6))
    assert nodes["extractor"]["potential_rise_v"] == 4000.0
    assert nodes["control"]["potential_rise_v"] == 5200.0
    assert nodes["terminal"]["potential_rise_v"] == 300000.0
    np.testing.assert_allclose(
        [node["potential_rise_v"] for node in divider],
        5200.0 + np.arange(6) * (300000.0 - 5200.0) / 5,
        rtol=0, atol=1e-9)


def test_control_change_reaches_divider_and_keeps_grounded_terminal(reference_gun):
    before = nodes_by_key(request(reference_gun))
    reference_gun.electrostatic_lens.voltage_kv += 0.5
    after = nodes_by_key(request(reference_gun))
    assert after["extractor"] == before["extractor"]
    for key, old in before.items():
        index = old["divider_index"]
        if index is not None:
            assert after[key]["potential_rise_v"] - old["potential_rise_v"] == pytest.approx(
                500.0 * (1.0 - index / 5), abs=1e-9)


def test_all_silhouette_pieces_keep_their_electrical_node(reference_gun):
    result = request(reference_gun)
    nodes = nodes_by_key(result)
    assert len(result["rings"]) == sum(len(n["pieces"]) for n in nodes.values())
    for node in nodes.values():
        pieces = [row for row in result["rings"] if row["key"] == node["key"]]
        assert pieces == node["pieces"]
        assert all(piece["potential_rise_v"] == node["potential_rise_v"]
                   for piece in pieces)
        assert node["minimum_bore_radius_m"] == min(piece["inner_m"] for piece in pieces)
        assert all(0 < piece["inner_m"] < piece["outer_m"] for piece in pieces)


def test_flat_comparison_changes_only_intermediate_contour(reference_gun):
    stepped, flat = request(reference_gun), request(reference_gun, contour="flat")
    s_nodes, f_nodes = nodes_by_key(stepped), nodes_by_key(flat)
    for key in ("extractor", "control", "terminal"):
        assert s_nodes[key] == f_nodes[key]
    for index in range(1, 5):
        key = f"accelerator:{index}"
        assert len(s_nodes[key]["pieces"]) > len(f_nodes[key]["pieces"])
        for quantity in ("reference_z_m", "potential_rise_v", "divider_index",
                         "minimum_bore_radius_m"):
            assert s_nodes[key][quantity] == f_nodes[key][quantity]
    assert stepped["domain"] == flat["domain"]
    assert planar.request_digest(stepped) != planar.request_digest(flat)


def test_reference_size_scales_isotropically_from_existing_terminal_anchor(reference_gun):
    before = request(reference_gun)
    scale = 1.2
    reference_gun.accelerator.mechanical_center_from_tip_mm *= scale
    reference_gun.accelerator.mechanical_length_mm *= scale
    reference_gun.exit_plane_z_mm *= scale
    after = request(reference_gun)
    for old, new in zip(before["rings"], after["rings"]):
        assert old["key"] == new["key"]
        assert old["potential_rise_v"] == new["potential_rise_v"]
        for key in ("start_m", "stop_m", "inner_m", "outer_m"):
            assert new[key] == pytest.approx(old[key] * scale, rel=1e-12)
    nodes = nodes_by_key(after)
    # The mechanical end anchors the downstream metal face, not its centre.
    assert max(piece["stop_m"] for piece in nodes["terminal"]["pieces"]) == pytest.approx(7.2e-3)
    assert after["normalization"]["terminal_end_m"] == pytest.approx(7.2e-3)


def test_mesh_retains_physical_faces_and_rejects_conflicting_shared_metal(reference_gun):
    result = request(reference_gun)
    radial, axial = reference.mesh_axes(result)
    assert np.all(np.diff(radial) > 0)
    assert np.all(np.diff(axial) > 0)
    for piece in result["rings"]:
        for name in ("inner_m", "outer_m"):
            assert piece[name] in radial
        for name in ("start_m", "stop_m"):
            assert piece[name] in axial
    # One conductor's adjacent rectangles are permitted. The same occupied
    # space cannot be assigned a second voltage, even in a modified request.
    fixed, values = reference.electrode_boundary_arrays(result, radial, axial)
    assert np.all(np.isfinite(values[fixed]))
    changed = copy.deepcopy(result)
    conflicting = copy.deepcopy(changed["rings"][0])
    conflicting["key"] = "conflicting_node"
    conflicting["potential_rise_v"] += 1.0
    changed["rings"].append(conflicting)
    with pytest.raises(ValueError):
        reference.electrode_boundary_arrays(changed, radial, axial)


def test_reference_request_does_not_change_default_source_or_field_provider():
    from temsim.optics.column import default_state
    from temsim.optics.electron_gun.electrostatic import FegElectrostaticField

    gun = default_state().electron_gun
    before = repr(vars(gun.emitter))
    stage_before = copy.deepcopy(gun.accelerator.stages)
    result = request(gun)
    assert len(result["electrode_nodes"]) == 7
    assert len(gun.accelerator.stages) == 10
    assert gun.accelerator.stages == stage_before
    assert repr(vars(gun.emitter)) == before
    assert isinstance(gun.electric_field, FegElectrostaticField)


@pytest.mark.parametrize("path,attribute,value", [
    ("extractor", "voltage_kv", 4.5),
    ("electrostatic_lens", "voltage_kv", 1.4),
    ("electrostatic_lens", "voltage_reference", "tip"),
    ("accelerator", "high_tension_kv", 200.0),
    ("accelerator", "mechanical_center_from_tip_mm", 5.2),
    ("accelerator", "mechanical_length_mm", 2.4),
    ("", "exit_plane_z_mm", 9.0),
])
def test_cache_binds_consumed_voltage_and_geometry(reference_gun, path, attribute, value):
    before = planar.request_digest(request(reference_gun))
    target = getattr(reference_gun, path) if path else reference_gun
    setattr(target, attribute, value)
    assert planar.request_digest(request(reference_gun)) != before


@pytest.mark.parametrize("options", [{"contour": "flat"}, {"cells_per_bore": 8},
                                        {"outer_factor": 3.0},
                                        {"exit_extension_mm": 1.0}])
def test_cache_binds_contour_numerical_grid_and_external_boundary(reference_gun, options):
    assert planar.request_digest(request(reference_gun)) != planar.request_digest(
        request(reference_gun, **options))


def test_static_field_does_not_consume_old_stage_layout_or_emission_samples(reference_gun):
    before = planar.request_digest(request(reference_gun))
    reference_gun.emitter.ray_count *= 2
    reference_gun.emitter.emission_current_na *= 3
    reference_gun.emitter.virtual_source_fwhm_nm *= 2
    reference_gun.electrostatic_lens.potential_scale *= 2
    reference_gun.accelerator.stages = [SimpleNamespace(center_from_tip_mm=4.0)] * 10
    assert planar.request_digest(request(reference_gun)) == before


@pytest.mark.parametrize("options", [
    {"contour": "unsupported"}, {"contour": None},
    {"cells_per_bore": 3}, {"cells_per_bore": 65}, {"cells_per_bore": True},
    {"outer_factor": 1.0}, {"outer_factor": np.nan},
    {"exit_extension_mm": -1.0}, {"exit_extension_mm": np.inf},
])
def test_invalid_reference_options_are_rejected(reference_gun, options):
    with pytest.raises((TypeError, ValueError)):
        request(reference_gun, **options)


def test_reference_requires_explicit_planar_cathode_choice(reference_gun):
    with pytest.raises(TypeError):
        reference.patent_reference_request(reference_gun)
    with pytest.raises(ValueError):
        reference.patent_reference_request(reference_gun, cathode_boundary="flat")


@pytest.mark.parametrize("path,attribute,value", [
    ("emitter", "surface_model", SimpleNamespace()),
    ("emitter", "curvature_nm_inv", 0.1),
    ("emitter", "coherence", SimpleNamespace()),
    ("", "source_representation", "coherent_wave"),
    ("", "monochromator_installed", True),
    ("accelerator", "mechanical_length_mm", -2.0),
    ("accelerator", "mechanical_center_from_tip_mm", np.nan),
    ("", "exit_plane_z_mm", 2.0),
])
def test_unsupported_source_or_invalid_domain_is_rejected(reference_gun, path, attribute, value):
    target = getattr(reference_gun, path) if path else reference_gun
    setattr(target, attribute, value)
    with pytest.raises(ValueError):
        request(reference_gun)


def test_small_reference_solve_preserves_node_voltages_and_cache(reference_gun, tmp_path):
    inputs_before = repr(reference_gun)
    with threadpool_limits(limits=1):
        field = reference.build_patent_gun_field(
            reference_gun, cathode_boundary=CATHODE, cells_per_bore=4,
            cache_dir=tmp_path)
    assert field.r.size * field.z.size < 100000
    assert field.report["linear_residual"] < 1e-9
    for node in field.request["electrode_nodes"]:
        for piece in node["pieces"]:
            query = [[0.5 * (piece["inner_m"] + piece["outer_m"]), 0.0,
                      0.5 * (piece["start_m"] + piece["stop_m"])]]
            potential, electric = field.interpolate(query)
            np.testing.assert_allclose(potential, node["potential_rise_v"],
                                       rtol=0, atol=1e-7)
            np.testing.assert_allclose(electric, 0.0, rtol=0, atol=0.1)
    loaded = planar.load_cached_field(field.request, tmp_path)
    assert loaded is not None
    np.testing.assert_array_equal(loaded.voltage, field.voltage)
    np.testing.assert_array_equal(loaded.r, field.r)
    np.testing.assert_array_equal(loaded.z, field.z)
    assert repr(reference_gun) == inputs_before
