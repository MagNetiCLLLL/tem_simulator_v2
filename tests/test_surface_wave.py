"""Coherent surface segment tests; no full-column image admission is implied."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.electron_gun.tip_surface import SurfaceCoherence, load_tip_surface_reference
from temsim.optics.electron_gun.field_emission import FieldEmissionGun
from temsim.physics.surface_wave import (SurfaceWaveNumerics, structured_mesh, solve_driven_wave,
                                        compute_surface_wave, KINETIC_NM2_PER_EV)


def reference_gun():
    gun = FieldEmissionGun()
    gun.emitter.surface_model = replace(load_tip_surface_reference(), coherence=SurfaceCoherence(energy_rms_ev=0.))
    return gun


def plane_wave(nodes, step=False):
    # Refine both coordinates: elongated P1 triangles otherwise give a mixed
    # radial/axial error, not a one-dimensional second-order convergence test.
    radial = 7 if step else (nodes-1)//4+1
    points, triangles, edges, z = structured_mesh(np.linspace(0, 1., radial), np.zeros(radial), 3., nodes)
    edges["side"] = np.empty((0, 2), dtype=int)  # exact reflecting waveguide wall in this analytic fixture
    potential = np.where(points[:, 1] > 1.5, .9, 0.) if step else np.zeros(len(points))
    drive = np.zeros(len(points), complex)
    drive[np.arange(radial)*nodes] = 1
    psi, flux = solve_driven_wave(points, triangles, edges, potential, .3, drive)
    return z, psi.reshape(z.shape), flux


def test_uniform_waveguide_converges_to_analytic_complex_plane_wave():
    errors = []
    for n in (41, 81, 161):
        z, psi, flux = plane_wave(n)
        exact = np.exp(1j*np.sqrt(.3*KINETIC_NM2_PER_EV)*z)
        errors.append(np.max(abs(psi/psi[0, 0]-exact)))
        assert flux["balance_error"] < 1e-10
        assert flux["reflected"] < 1e-4
    assert errors[-1] < .005
    assert errors[0]/errors[1] > 2.5
    assert errors[1]/errors[2] > 2.5


def test_step_reflection_is_retained_and_not_renormalised_to_transmission_one():
    _, _, flux = plane_wave(301, step=True)
    assert flux["reflected"] == pytest.approx(1/9, abs=.006)
    assert flux["top"] == pytest.approx(8/9, abs=.006)
    assert flux["balance_error"] < 1e-10


def test_coherent_energy_quadrature_preserves_mean_width_and_is_not_one_phase():
    c = SurfaceCoherence()
    energy, weights = c.energy_quadrature(5)
    assert weights.sum() == pytest.approx(1)
    assert weights@energy == pytest.approx(c.mean_energy_ev)
    assert np.sqrt(weights@(energy-c.mean_energy_ev)**2) == pytest.approx(c.energy_rms_ev)
    with pytest.raises(ValueError, match="two energy samples"):
        c.energy_quadrature(1)


def test_source_model_serialises_coherence_and_rejects_fake_classical_samples():
    from temsim.optics.electron_gun.tip_surface import TipSurfaceModel
    gun = reference_gun()
    model = gun.emitter.surface_model
    assert TipSurfaceModel.from_dict(model.to_dict()) == model
    with pytest.raises(ValueError, match="propagated as a wave"):
        gun.emit(49)


def test_actual_tip_near_field_preserves_complex_fields_flux_and_exact_cache():
    gun = reference_gun()
    numerical = SurfaceWaveNumerics(radial_nodes=33, axial_nodes=65, energy_samples=1)
    result = compute_surface_wave(gun, numerical)
    assert result.modes[0].amplitude.shape == (33, 65)
    assert np.iscomplexobj(result.modes[0].amplitude)
    assert not result.modes[0].amplitude.flags.writeable
    assert result.z_nm[0, 0] == 0
    assert result.z_nm[-1, 0] < 0  # curved physical tip, not a disk
    assert result.modes[0].flux["balance_error"] < 1e-8
    assert compute_surface_wave(gun, numerical) is result
    gun.extractor.voltage_kv = 4.1
    changed = compute_surface_wave(gun, numerical)
    assert changed.digest != result.digest
    assert not np.allclose(changed.modes[0].amplitude, result.modes[0].amplitude)
    # Changing only the reference phase changes complex fields, not density.
    gun.extractor.voltage_kv = 4.
    gun.emitter.surface_model = replace(gun.emitter.surface_model,
        coherence=replace(gun.emitter.surface_model.coherence, edge_phase_rad=1.))
    phased = compute_surface_wave(gun, numerical)
    assert phased.digest != result.digest


def test_budget_and_cancellation_fail_before_field_allocation():
    gun = reference_gun()
    with pytest.raises(MemoryError, match="needs"):
        compute_surface_wave(gun, SurfaceWaveNumerics(maximum_working_bytes=1024))
    with pytest.raises(InterruptedError):
        compute_surface_wave(gun, cancelled=lambda: True)


def test_domain_cannot_jump_over_existing_electrodes():
    gun = reference_gun()
    with pytest.raises(ValueError, match="reaches an electrode"):
        compute_surface_wave(gun, SurfaceWaveNumerics(exit_height_nm=1e7))


def test_magnetic_fields_cannot_be_silently_discarded(monkeypatch):
    gun = reference_gun()
    monkeypatch.setattr(type(gun), "magnetic_field", property(lambda self: SimpleNamespace(
        field_at_global_positions_t=lambda p: np.ones_like(p)*.001)))
    with pytest.raises(ValueError, match="magnetic field is present"):
        compute_surface_wave(gun, SurfaceWaveNumerics(radial_nodes=9, axial_nodes=9))


def test_complex_drive_linearity_keeps_the_phase_reference():
    points, triangles, edges, _ = structured_mesh(np.linspace(0, 1, 5), np.zeros(5), 1., 41)
    drive = np.zeros(len(points), complex)
    drive[np.arange(5)*41] = 1
    a, _ = solve_driven_wave(points, triangles, edges, np.zeros(len(points)), .5, drive)
    b, _ = solve_driven_wave(points, triangles, edges, np.zeros(len(points)), .5, drive*np.exp(.7j))
    np.testing.assert_allclose(b, a*np.exp(.7j), atol=1e-12)
