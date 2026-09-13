"""Numerical partitions apply to one emission energy, never replace a mode."""
from dataclasses import asdict, replace

import numpy as np
import pytest

from temsim.immutable_json import json_digest
from temsim.optics.column import default_state
from temsim.optics.electron_gun.tip_surface import SurfaceCoherence, load_tip_surface_reference
from temsim.physics.occupied_axial_refinement import OccupiedAxialRefinement, refine_joint_mode
from temsim.physics.radial_gun_wave import RadialGunNumerics
from temsim.physics.surface_wave import SurfaceWaveNumerics
from test_occupied_axial_refinement import plan_and_boundary


SEED = ((0., 1., 0., .5, 1), (0., 1., .5, 1., 1))


def settings(**kwargs):
    return OccupiedAxialRefinement(enabled=True, strategy="spatial_embedded", integrator="cf6",
        initial_mesh=SEED, initial_mesh_energy_ev=.3, maximum_rounds=32, **kwargs)


def test_all_energies_execute_with_only_matching_mesh_hint():
    request = settings().validate()
    counts = []
    for energy in (.2, .3, .4):
        plan, boundary, _, calls = plan_and_boundary()
        plan["interval_z_nm"] = {0: (0., 1.)}
        local = request.for_energy(energy)
        assert bool(local.initial_mesh) == (energy == .3)
        assert local.enabled and local.tolerance == request.tolerance
        load, result, report = refine_joint_mode(plan, boundary, local)
        assert len(calls) >= 2 and report["rounds"][-1]["successive_uniform_checks"] == 2
        assert np.all(np.isfinite(load.propagate(result[1])))
        counts.append(report["evaluations"])
    assert counts[0] == counts[2] and counts[1] > 0
    assert request.initial_mesh == SEED
    assert json_digest(asdict(request)) != json_digest(asdict(replace(request, initial_mesh_energy_ev=.4)))


@pytest.mark.parametrize("value", [0, -1, True, "0.3", np.inf, np.nan])
def test_invalid_energy_scope_is_rejected(value):
    with pytest.raises(ValueError, match="finite positive emission energy"):
        replace(settings(), initial_mesh_energy_ev=value).validate()


def test_energy_scope_without_mesh_is_not_a_source_parameter():
    with pytest.raises(ValueError, match="scoped mesh seed"):
        replace(settings(), initial_mesh=()).validate()


@pytest.mark.parametrize("scope,message", [(None, "explicit emission energy"), (.5, "not one of")])
def test_multi_energy_source_rejects_ambiguous_or_unconsumed_hint(monkeypatch, scope, message):
    import temsim.physics.surface_gun_wave as module
    gun = default_state().electron_gun
    gun.emitter.surface_model = replace(load_tip_surface_reference(), coherence=SurfaceCoherence())
    # Input-wiring fixture only. No solved field is supplied or admitted.
    monkeypatch.setattr(module, "prepare_surface_problem", lambda *args, **kwargs: {"energies": [.2, .3, .4]})
    with pytest.raises(ValueError, match=message):
        module.build_surface_gun_checkpoint(gun,
            surface=SurfaceWaveNumerics(element_order=2, radial_nodes=97, axial_nodes=65, outer_radius_factor=4.),
            radial=RadialGunNumerics(radial_modes=8, potential_quadrature=32,
                occupied_refinement=replace(settings(), initial_mesh_energy_ev=scope)))
