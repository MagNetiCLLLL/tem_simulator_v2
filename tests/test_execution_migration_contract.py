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


def test_old_profile_migration_preserves_explicit_hardware_and_sample(tmp_path):
    import tomllib, tomli_w
    from temsim.optics.column import default_state
    from temsim.profile_io import save_profile,read_profile,apply_profile_values
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.physics.illumination import default_illumination_config
    state=default_state()
    state.objective_lens.percent=68.
    state.sample.diameter_nm=10.; state.sample.thickness_nm=5.
    path=tmp_path/"legacy.toml"
    save_profile(path,state,AssemblyCatalog().default_selection())
    document=tomllib.loads(path.read_text())
    document["format_version"]=5
    document["sample_model"].pop("wave_illumination",None)
    for name in ("stem_execution_policy","stem_fourdstem_host_budget_mb"):
        document["devices"]["sample"].pop(name,None)
    path.write_text(tomli_w.dumps(document))
    before=[(l.key,l.percent,l.polarity,l.z_mm) for l in state.lenses]
    state.sample.wave_illumination=default_illumination_config()
    _,values=read_profile(path)
    assert apply_profile_values(state,values)==[]
    assert [(l.key,l.percent,l.polarity,l.z_mm) for l in state.lenses]==before
    assert state.sample.diameter_nm==10 and state.sample.thickness_nm==5
    assert state.sample.wave_illumination["model"]=="ray_conditioned_reduced_order"
    report=state._profile_migration_report
    assert report["from_version"]==5 and report["to_version"]==6 and report["notes"]


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
