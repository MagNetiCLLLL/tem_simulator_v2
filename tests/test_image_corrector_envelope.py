"""Default image-corrector envelopes fit the existing column, in millimetres."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.assembly_model_3d import assembly_model_from_assembly
from temsim.component_keys import IMAGE_CORRECTOR_KEYS
from temsim.diagnostics import physical_layout_records
from temsim.module_manifest import read_document, validate_document
from temsim.optics.column import default_state
from temsim.physics.core import propagate


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    ("C3_ImageCorrector.toml", "C3 + Image Corrector"),
    ("C3_ProbeCorrector_ImageCorrector.toml", "C3 + Probe Corrector + Image Corrector"),
)
RESIZED = ("image_hp1_hexapole", "image_hp2_hexapole",
           "image_dph1_deflector", "image_dph2_deflector")


@pytest.mark.parametrize("filename,column", VARIANTS)
def test_image_corrector_default_envelopes_fit_column_and_keep_internal_clearance(filename, column):
    document = read_document(ROOT / "configs/instruments/column" / filename)
    validate_document(document)
    outer = document["geometry"]["magnetic_lens_external_diameter_mm"]
    by_key = {p["key"]: p for p in document["parts"]}
    for part in by_key.values():
        if part["key"] in IMAGE_CORRECTOR_KEYS or part.get("parent_key") in IMAGE_CORRECTOR_KEYS:
            if "mechanical_outer_diameter_mm" in part:
                assert part["mechanical_outer_diameter_mm"] <= outer
    for key in RESIZED:
        part = by_key[key]
        assert part["mechanical_outer_diameter_mm"] == outer == 180.0
        assert part["vacuum_inner_diameter_mm"] == 5.0
        assert part["mechanical_clear_bore_diameter_mm"] == 8.0
        assert part.get("effective_length_mm", part.get("effective_thickness_mm")) == 30.0
        if "active_diameter_mm" in part:
            assert part["active_diameter_mm"] == 152.0 < outer
        if part.get("parent_key"):
            parent = by_key[part["parent_key"]]
            assert parent["local_center_z_mm"] == part["local_center_z_mm"]
            assert parent["local_start_z_mm"] <= part["local_start_z_mm"]
            assert part["local_end_z_mm"] <= parent["local_end_z_mm"]


@pytest.mark.parametrize("filename,column", VARIANTS)
def test_resolved_2d_3d_geometry_agrees_and_outer_resize_does_not_change_ray_transport(filename, column):
    state = default_state()
    assembly = AssemblyCatalog().apply(state, AssemblySelection("FEG", column, "Energy Filter"))
    records = {r.key: r for r in physical_layout_records(SimpleNamespace(
        assembly=assembly, layout=state._resolved_optics_layout, state_snapshot=state))}
    model = assembly_model_from_assembly(assembly, angular_segments=16)
    meshes = {key: [m for m in model.meshes if m.key == key] for key in RESIZED}
    components = {c.key: c for c in state.corrector_elements}
    for key in RESIZED:
        assert records[key].outer_diameter_mm == 180.0
        assert components[key].mechanical_outer_diameter_mm == 180.0
        assert meshes[key], model.errors
        for mesh in meshes[key]:
            radius = np.linalg.norm(mesh.vertices[:, :2], axis=1)
            assert radius.max() == pytest.approx(90.0, abs=1e-9)

    # Exercise nonzero deflection and hexapole fields. Only the old mechanical
    # exterior is restored below, not a second optical calibration or source.
    components["image_dph1_deflector"].kick_x_mrad = 0.01
    state.step_mm = state.history_step_mm = 0.5
    zero = np.zeros(2)
    inputs = (state, components["image_hp1_hexapole"].z_mm - 1.0,
              components["image_hp2_hexapole"].z_mm + 1.0,
              np.array([-1e-7, 1e-7]), zero, zero, zero)
    resized = propagate(*inputs, include_spherical_aberration=True, include_hexapole=True)
    for key in RESIZED:
        components[key].mechanical_outer_diameter_mm = 220.0
    previous = propagate(*inputs, include_spherical_aberration=True, include_hexapole=True)
    for actual, expected in zip(resized, previous, strict=True):
        assert np.all(np.isfinite(actual))
        assert np.all(np.isfinite(expected))
        np.testing.assert_array_equal(actual, expected)
