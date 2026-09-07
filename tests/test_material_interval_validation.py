"""Overlap checks consume physical material sections, not display envelopes."""

import tomllib

import numpy as np
import pytest

from temsim.module_manifest import _validate_simple_magnetic_layer_geometry, validate_document
from temsim.part_model_3d import part_model_from_document
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.physics.axisymmetric_magnetostatics import _part_mask


def test_disjoint_projector_material_sections_match_renderer_and_fem():
    source = INSTRUMENT_CONFIG_ROOT / "project_and_recording_system/EnergyFilter.toml"
    document = tomllib.loads(source.read_text(encoding="utf-8"))
    by_key = {part["key"]: part for part in document["parts"]}
    yoke = by_key["intermediate_lens_yoke"]
    coil = by_key["intermediate_lens_excitation_coil"]
    yoke["mechanical_inner_diameter_mm"] = 68.0
    yoke["material_intervals_mm"] = [[137.5, 162.0], [343.0, 367.5]]
    model = part_model_from_document(document, yoke["key"])
    mesh = model.meshes[0]
    triangle_z = mesh.vertices[mesh.faces, 2]
    assert np.all((triangle_z.max(axis=1) <= 162.0) | (triangle_z.min(axis=1) >= 343.0))
    z = np.linspace(130.0, 380.0, 2501) * 1e-3
    r = np.full_like(z, 0.04)
    def mask(part):
        return _part_mask(dict(key=part["key"], start_z_mm=part["local_start_z_mm"],
                               end_z_mm=part["local_end_z_mm"], data=part), r, z)
    assert not np.any(mask(yoke) & mask(coil))
    validate_document(document)
    yoke["material_intervals_mm"][0][1] = 163.0
    with pytest.raises(ValueError, match="Mechanical radial layers overlap"):
        validate_document(document)


def _parts():
    parent = dict(key="lens", mechanical_profile="magnetic_lens_assembly", local_start_z_mm=0.0)
    def layer(key, profile, start, end, intervals=None):
        result = dict(key=key, parent_key="lens", mechanical_profile=profile,
                      local_start_z_mm=start, local_center_z_mm=(start + end) / 2,
                      local_end_z_mm=end, length_mm=end - start,
                      mechanical_inner_diameter_mm=20, mechanical_outer_diameter_mm=30,
                      vacuum_inner_diameter_mm=10, material_class="test_material")
        if intervals is not None:
            result["material_intervals_mm"] = intervals
        return result
    return [parent, layer("coil", "magnetic_excitation_coil", 0, 10, [[0, 2], [8, 10]]),
            layer("housing", "magnetic_lens_housing", 3, 7)]


def test_parent_owned_split_intervals_take_precedence_over_child_intervals():
    parts = _parts()
    parts[0].update(upper_yoke_start_local_z_mm=0, upper_yoke_end_local_z_mm=2,
                    lower_yoke_start_local_z_mm=8, lower_yoke_end_local_z_mm=10)
    parts[1]["material_intervals_mm"] = [[0, 10]]
    _validate_simple_magnetic_layer_geometry(parts)
    # Removing the parent's geometry makes the explicit child sections active.
    del parts[0]["upper_yoke_start_local_z_mm"]
    with pytest.raises(ValueError, match="Mechanical radial layers overlap"):
        _validate_simple_magnetic_layer_geometry(parts)


@pytest.mark.parametrize("intervals", [[], [[2, 1]], [[1, 1]], [[0, float("nan")]],
                                        [[0, float("inf")]], [[0]], [[0, 1, 2]], [[False, 1]], "bad"])
def test_invalid_explicit_material_sections_are_rejected(intervals):
    parts = _parts()
    parts[1]["material_intervals_mm"] = intervals
    with pytest.raises(ValueError, match="material_intervals_mm"):
        _validate_simple_magnetic_layer_geometry(parts)
