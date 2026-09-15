"""Proposal mathematics and detached geometry; not full-beam qualification."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.constants import c, e, m_e

from temsim.optics.gun_matching import axis_variational_map, candidate_with_gun_geometry, _integrate
from temsim import module_manifest
from temsim.optics.electron_gun.tip_assembly import model_from_part


class AxialFixture:
    """Analytic electrostatic drift/acceleration, without an electron source."""
    z = np.array([0., .03, .1])
    r = np.array([0., 1e-6])

    def __init__(self, gradient=0.):
        self.gradient = gradient

    def _interpolate(self, positions):
        electric = np.zeros_like(positions)
        electric[:, 2] = -self.gradient
        return self.gradient*positions[:, 2], electric


@pytest.mark.parametrize("gradient", [0., 3e6])
def test_drift_and_relativistic_acceleration_use_canonical_momentum(gradient):
    energy, scale = .3, 1e-7
    matrix, report = axis_variational_map(AxialFixture(gradient),
        emission_energy_ev=energy, scale_m=scale, intervals=800)
    rest = m_e*c*c/e
    momentum = np.sqrt(energy*(energy+2*rest))
    if gradient:
        expected = momentum/(scale*gradient)*(np.arccosh(1+(energy+gradient*.1)/rest)
                                             - np.arccosh(1+energy/rest))
    else:
        expected = .1/scale
    assert matrix == pytest.approx(np.array([[1., expected], [0., 1.]]), rel=2e-8, abs=1e-10)
    assert report["determinant"] == pytest.approx(1.)
    assert "PROPOSAL_ONLY" in report["scope"]


def test_focusing_sign_and_fourth_order_refinement():
    angle = 2.3
    expected = np.array([[np.cos(angle), np.sin(angle)/angle],
                         [-angle*np.sin(angle), np.cos(angle)]])
    errors = []
    for steps in (20, 40):
        matrix = _integrate(np.tile([1., -angle**2], (2*steps+1, 1)), np.full(steps, 1/steps))
        errors.append(np.linalg.norm(matrix-expected))
        assert abs(np.linalg.det(matrix)-1) < 1e-6
    assert 14 < errors[0]/errors[1] < 18


def test_interior_observation_uses_executed_field_without_changing_source():
    matrix, report = axis_variational_map(AxialFixture(), emission_energy_ev=.3,
        scale_m=1e-7, intervals=800, end_m=.023)
    assert matrix == pytest.approx(np.array([[1., .023/1e-7], [0., 1.]]), rel=2e-8)
    assert report["end_m"] == .023
    for invalid in (0., -.01, .11, np.nan):
        with pytest.raises(ValueError, match="endpoint"):
            axis_variational_map(AxialFixture(), emission_energy_ev=.3, scale_m=1e-7,
                                 end_m=invalid)


@pytest.mark.parametrize("energy,scale,intervals", [(0.,1e-7,400),(.3,0.,400),
    (np.nan,1e-7,400),(.3,1e-7,True),(.3,1e-7,99)])
def test_invalid_variational_inputs_are_rejected(energy, scale, intervals):
    with pytest.raises(ValueError):
        axis_variational_map(AxialFixture(), emission_energy_ev=energy, scale_m=scale, intervals=intervals)


@pytest.mark.parametrize("extractor,centre", [(8.,22.8),(4.,14.8)])
def test_placement_moves_field_and_bore_together_without_mutating_original(extractor,centre):
    from temsim.optics.column import default_state
    from temsim.physics.grounded_tip_field import field_request
    from temsim.instrument_snapshot import capture_instrument_snapshot
    state = default_state()
    gun = state.electron_gun
    gun.extractor.voltage_kv = 4.5
    gun.electrostatic_lens.voltage_kv = 1.1
    gun.emitter.ray_count = 193
    gun.emitter.surface_model = model_from_part(module_manifest.part_data("gun/FEG.toml", "feg_tip"))
    gun.emitter.surface_model = replace(gun.emitter.surface_model,
        emission=replace(gun.emitter.surface_model.emission, spatial_sampling="apex_stratified_v1"))
    # Standalone gun identity initializes its declared vacuum context. Do this
    # before the immutability baseline; placement itself must not mutate it.
    original_key = gun._cache_key(193)
    before = capture_instrument_snapshot(state)
    candidate = candidate_with_gun_geometry(state, extractor_center_mm=extractor, lens_center_mm=centre)
    assembly = candidate._resolved_assembly
    assert candidate.electron_gun.emitter.surface_model == gun.emitter.surface_model
    assert candidate.electron_gun.emitter.ray_count == 193
    assert candidate.electron_gun.extractor.voltage_kv == 4.5
    assert candidate.electron_gun.electrostatic_lens.voltage_kv == 1.1
    assert candidate.electron_gun.electrostatic_lens.voltage_reference == gun.electrostatic_lens.voltage_reference
    assert candidate.electron_gun.extractor.mechanical_center_from_tip_mm == pytest.approx(extractor)
    assert candidate.electron_gun.electrostatic_lens.mechanical_center_from_tip_mm == pytest.approx(centre)
    moved = assembly.part("feg_electrostatic_lens")
    assert (moved.start_z_mm, moved.center_z_mm, moved.end_z_mm) == pytest.approx((centre-4,centre,centre+4))
    assert any(s.start_z_mm == pytest.approx(centre-4) and s.end_z_mm == pytest.approx(centre+4)
               and s.inner_diameter_mm == 8 for s in assembly.vacuum_bore_segments)
    for part in state._resolved_assembly.parts:
        if part.key not in {"feg_extractor", "feg_electrostatic_lens"}:
            assert assembly.part(part.key) == part
    assert field_request(candidate.electron_gun) != field_request(gun)
    assert candidate.electron_gun._cache_key(193) != original_key
    restored = capture_instrument_snapshot(candidate).restore()
    assert field_request(restored.electron_gun) == field_request(candidate.electron_gun)
    assert restored.electron_gun._cache_key(193) == candidate.electron_gun._cache_key(193)
    assert restored._resolved_assembly.vacuum_bore_segments == assembly.vacuum_bore_segments
    assert capture_instrument_snapshot(state).digest == before.digest


@pytest.mark.parametrize("ext,lens", [(2.,18.),(8.,13.),(8.,27.),(np.nan,18.),(8.,np.inf)])
def test_overlapping_or_nonfinite_placements_are_rejected(ext,lens):
    from temsim.optics.column import default_state
    with pytest.raises(ValueError):
        candidate_with_gun_geometry(default_state(), extractor_center_mm=ext, lens_center_mm=lens)


def test_accelerator_placement_moves_stages_and_attached_aperture_once():
    from temsim.optics.column import default_state
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.physics.grounded_tip_field import field_request
    state = default_state()
    state.electron_gun.emitter.surface_model = model_from_part(
        module_manifest.part_data("gun/FEG.toml", "feg_tip"))
    before = capture_instrument_snapshot(state).digest
    candidate = candidate_with_gun_geometry(state, extractor_center_mm=8., lens_center_mm=22.8,
                                           accelerator_center_mm=210.)
    old, new = state._resolved_assembly, candidate._resolved_assembly
    for key in ("feg_accelerator", "feg_dpa_aperture"):
        assert new.part(key).center_z_mm == pytest.approx(old.part(key).center_z_mm+10.)
        assert new.part(key).length_mm == old.part(key).length_mm
    assert [s.center_from_tip_mm for s in candidate.electron_gun.accelerator.stages] == pytest.approx(
        [s.center_from_tip_mm+10. for s in state.electron_gun.accelerator.stages])
    assert new.part("feg_deflector") == old.part("feg_deflector")
    assert field_request(capture_instrument_snapshot(candidate).restore().electron_gun) == field_request(candidate.electron_gun)
    assert capture_instrument_snapshot(state).digest == before


@pytest.mark.parametrize("accelerator", [195., 228., np.nan, np.inf])
def test_accelerator_overlap_and_nonfinite_are_rejected(accelerator):
    from temsim.optics.column import default_state
    with pytest.raises(ValueError):
        candidate_with_gun_geometry(default_state(), extractor_center_mm=8., lens_center_mm=22.8,
                                    accelerator_center_mm=accelerator)


def test_independent_dpa_placement_keeps_parent_wall_and_field_geometry_consistent():
    from temsim.optics.column import default_state
    from temsim.instrument_snapshot import capture_instrument_snapshot
    state = default_state()
    before = capture_instrument_snapshot(state).digest
    candidate = candidate_with_gun_geometry(state, extractor_center_mm=2.1, lens_center_mm=8.2,
                                           accelerator_center_mm=218., dpa_center_mm=210.)
    part = candidate._resolved_assembly.part("feg_dpa_aperture")
    assert (part.start_z_mm, part.center_z_mm, part.end_z_mm) == pytest.approx((209., 210., 211.))
    assert candidate.electron_gun.dpa_aperture.z_mm == pytest.approx(210.)
    assert part.data["parent_key"] == "feg_accelerator"
    assert any(row.start_z_mm == pytest.approx(209.) and row.end_z_mm == pytest.approx(211.)
               and row.inner_diameter_mm == 6. for row in candidate._resolved_assembly.vacuum_bore_segments)
    assert capture_instrument_snapshot(candidate).restore().electron_gun.dpa_aperture.z_mm == pytest.approx(210.)
    assert capture_instrument_snapshot(state).digest == before
    for centre in (20., 202.22222222222223, 400., np.nan):
        with pytest.raises(ValueError):
            candidate_with_gun_geometry(state, extractor_center_mm=2.1, lens_center_mm=8.2,
                                        accelerator_center_mm=218., dpa_center_mm=centre)
    from temsim.optics.beam_path_audit import optical_component_planes, require_same_topology
    original_planes = optical_component_planes(state, full_path=True)
    candidate_planes = optical_component_planes(candidate, full_path=True)
    assert "projector_lens_1" in dict(candidate_planes)
    assert "feg_dpa_aperture" not in dict(optical_component_planes(candidate))
    # A waist before the DPA must remain there, not just inside the same
    # distributed accelerator body. Geometry movement cannot weaken this gate.
    require_same_topology([41.5], [202.], original_planes, candidate_component_planes=candidate_planes)
    unmoved_dpa = candidate_with_gun_geometry(state, extractor_center_mm=2.1, lens_center_mm=8.2,
                                             accelerator_center_mm=218.)
    with pytest.raises(ValueError, match="topology changed"):
        require_same_topology([41.5], [202.], original_planes,
                             candidate_component_planes=optical_component_planes(unmoved_dpa, full_path=True))
