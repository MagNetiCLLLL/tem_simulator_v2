"""WP-08/09 result provenance, narrow invalidation and old profile migration."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import json
import numpy as np
import pytest

from test_tem_flux_contract import tem_benchmark


def test_execution_export_matches_actual_nodes_and_keeps_verification_separate(tem_benchmark, tmp_path):
    f=tem_benchmark
    result=f.run()
    manifest=result.metrics["wave_execution_manifest"]
    operations={row["operation"] for row in manifest["nodes"]}
    assert "illumination" in operations and "specimen" in operations
    assert {row["node_id"] for row in result.metrics["camera_flux_ledger"]} <= operations
    assert manifest["source_energy_mode_count"]==1
    assert manifest["phonon_configuration_count"]==1
    assert manifest["verification"]["experimental_calibration"]=="NOT_RUN"
    assert manifest["verification"]["numerical_convergence"]=="NOT_RUN"
    assert "A5" in manifest["aberration_coverage"]["wave_terms"]
    assert result.metrics["image_formation_scope"]==manifest["summary"]
    # JSON roundtrip is exact; GUI and export do not build competing claims.
    assert json.loads(json.dumps(manifest))==manifest


def test_cad_display_changes_do_not_invalidate_scattering_or_field_geometry():
    from temsim.optics.column import default_state
    from temsim.calculation_cache import calculation_signatures
    from temsim.physics.lens_field_provider import lens_geometry_binding
    state=default_state()
    assembly=state._resolved_assembly
    lens=state.objective_lens
    before=calculation_signatures(state)
    binding=lens_geometry_binding(state,lens.key,lens)
    index=next(i for i,p in enumerate(assembly.parts) if p.parent_key==lens.key and p.data.get("mechanical_profile")=="magnetic_excitation_coil")
    parts=list(assembly.parts)
    parts[index]=replace(parts[index],data={**parts[index].data,"model_3d":{"features":[{"kind":"hole","diameter_mm":1}]}})
    state._resolved_assembly=replace(assembly,parts=tuple(parts))
    after=calculation_signatures(state)
    assert before["wave_source"]==after["wave_source"]
    assert before["incident"]==after["incident"]
    assert binding.geometry_fingerprint==lens_geometry_binding(state,lens.key,lens).geometry_fingerprint
    state.lens_field_map_descriptors[lens.key]={"solver":"axisymmetric_linear_fem","ampere_turns":250,"relative_permeability":100,"geometry_policy":"require_full_geometry"}
    from temsim.geometry_effects import admit_state_geometry
    with pytest.raises(ValueError,match="cannot consume"):
        admit_state_geometry(state)


def test_old_profile_is_rejected_before_applying_hardware_values(tmp_path):
    import tomllib, tomli_w
    from temsim.optics.column import default_state
    from temsim.profile_io import save_profile, read_profile
    from temsim.assembly_catalog import AssemblyCatalog
    state = default_state()
    path = tmp_path / "old.toml"
    save_profile(path, state, AssemblyCatalog().default_selection())
    document = tomllib.loads(path.read_text())
    document["format_version"] = 5
    path.write_text(tomli_w.dumps(document))
    before = deepcopy(state.to_dict())
    with pytest.raises(ValueError, match="Unsupported operating-profile format"):
        read_profile(path)
    assert state.to_dict() == before


def test_stem_execution_policy_and_budget_are_persisted_and_validated(tmp_path,qtbot):
    from temsim.optics.column import default_state
    from temsim.optics.model import State
    from temsim.runtime_parameters import runtime_targets,validate_runtime_assignment
    state=default_state()
    state.sample.stem_execution_policy="require_gpu"
    state.sample.stem_fourdstem_host_budget_mb=32
    restored=State.from_dict(state.to_dict())
    assert restored.sample.stem_execution_policy=="require_gpu"
    assert restored.sample.stem_fourdstem_host_budget_mb==32
    target=runtime_targets(state)["sample"]
    for field,value in (("stem_execution_policy","random"),("stem_fourdstem_host_budget_mb",0)):
        with pytest.raises(ValueError): validate_runtime_assignment(target,field,value)
