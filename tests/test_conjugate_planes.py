"""Analytic conjugacy and cache-boundary checks; not full microscope validation."""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.cpu_resources import NumericalJobCancelled
from temsim.physics import conjugate_planes as cp
from temsim.physics.first_order import TransverseTransfer, PLANE_CANONICAL_MOMENTUM


def _drift(mm):
    result = np.eye(4)
    result[:2, 2:] = np.eye(2)*mm*1e-3
    return result


def _lens(fx_mm, fy_mm=None):
    result = np.eye(4)
    result[2:, :2] = -np.diag((1000./fx_mm, 1000./(fy_mm or fx_mm)))
    return result


def _thin_atlas(lenses=(), stop=125., *, step=.25, source_basis=None, g=None):
    # Public atlas input permits exact analytical maps. Lens-plane nodes are
    # right limits; checked image planes are in drifts, away from the kicks.
    z = np.linspace(0., stop, round(stop/step)+1)
    maps = []
    for target in z:
        matrix, previous = np.eye(4), 0.
        for plane, fx, fy in lenses:
            if plane > target:
                break
            matrix = _lens(fx, fy) @ _drift(plane-previous) @ matrix
            previous = plane
        matrix = _drift(target-previous) @ matrix
        maps.append(matrix if source_basis is None else matrix @ source_basis)
    return cp.ConjugateAtlas(z, maps, source_g_m1=g)


def _images(search):
    return [candidate for candidate in search.candidates if candidate.kind == "image"]


def test_free_drift_has_no_other_conjugate_even_between_nodes():
    atlas = _thin_atlas()
    for reference in (0., .01, 37.1234567, 124.99, 125.):
        assert cp.find_conjugate_planes(atlas, reference).candidates == ()


def test_two_f_lens_repeats_real_point_images_and_excludes_self():
    atlas = _thin_atlas(((20., 10., 10.), (60., 10., 10.), (100., 10., 10.)))
    first = _images(cp.find_conjugate_planes(atlas, 0.))
    assert [c.z_mm for c in first] == pytest.approx([40., 80., 120.], abs=1e-5)
    assert all(c.magnifications == pytest.approx((1., 1.), abs=1e-7) for c in first)
    later = _images(cp.find_conjugate_planes(atlas, 40.))
    assert [c.z_mm for c in later] == pytest.approx([0., 80., 120.], abs=1e-5)


def test_four_f_pair_has_image_and_non_grid_reference_rebases_correctly():
    atlas = _thin_atlas(((10., 10., 10.), (30., 10., 10.)), stop=50.)
    zero = _images(cp.find_conjugate_planes(atlas, 0.))
    assert len(zero) == 1
    assert zero[0].z_mm == pytest.approx(40., abs=1e-5)
    arbitrary = _images(cp.find_conjugate_planes(atlas, 3.312345))
    assert len(arbitrary) == 1
    assert arbitrary[0].z_mm == pytest.approx(43.312345, abs=1e-5)
    np.testing.assert_allclose(atlas.transfer_matrix(3.312345, 43.312345),
                               -np.eye(4), atol=1e-10)


def test_astigmatic_lens_reports_line_foci_not_two_dimensional_images():
    atlas = _thin_atlas(((20., 10., 12.),), stop=60.)
    result = cp.find_conjugate_planes(atlas, 0.)
    assert _images(result) == []
    line = [c.z_mm for c in result.candidates if c.kind == "line_focus"]
    assert line == pytest.approx([40., 50.], abs=1e-5)
    assert any(c.kind == "approximate" for c in result.candidates)
    assert all(c.residual_m_per_rad > atlas.image_tolerance_m_per_rad
               for c in result.candidates if c.kind == "line_focus")


def test_lens_with_object_at_focus_does_not_invent_finite_conjugate():
    atlas = _thin_atlas(((10., 10., 10.),), stop=50.)
    assert _images(cp.find_conjugate_planes(atlas, 0.)) == []


def test_common_input_gauge_cancels_but_selected_gauge_is_restored():
    basis = np.eye(4)
    basis[2, 1], basis[3, 0] = 20., -20.
    plain = _thin_atlas(((20., 10., 10.),), stop=60.)
    gauged = _thin_atlas(((20., 10., 10.),), stop=60., source_basis=basis,
                         g=np.full(241, 20.))
    ref, target = 3.123, 48.0
    mechanical = plain.transfer_matrix(ref, target)
    canonical = gauged.transfer_matrix(ref, target)
    np.testing.assert_allclose(canonical, mechanical @ basis, atol=1e-11)
    np.testing.assert_allclose(canonical[:2, 2:], mechanical[:2, 2:], atol=1e-12)
    assert not np.allclose(canonical[:2, :2], mechanical[:2, :2])
    expected = _images(cp.find_conjugate_planes(plain, ref))
    observed = _images(cp.find_conjugate_planes(gauged, ref))
    assert [c.z_mm for c in observed] == pytest.approx([c.z_mm for c in expected], abs=1e-5)


def test_full_source_gauge_gradient_is_retained_when_rebasing_between_cached_nodes():
    plain = _thin_atlas(((20., 10., 10.),), stop=60.)
    common_basis = np.eye(4)
    common_basis[2:, :2] = ((.3, 20.), (-19., -.2))
    c0 = np.array(((1.2, 8.), (-11., -.7)))
    derivative = np.array(((.01, .03), (.02, -.01)))
    source_c = c0+plain.z_mm[:, None, None]*derivative
    gauged = cp.ConjugateAtlas(plain.z_mm, plain.matrices@common_basis, source_c_m1=source_c)
    ref, target = 3.123, 48.
    selected_basis = np.eye(4)
    selected_basis[2:, :2] = c0+ref*derivative
    expected = plain.transfer_matrix(ref, target)@selected_basis
    np.testing.assert_allclose(gauged.transfer_matrix(ref, target), expected, atol=1e-11)
    assert gauged.source_c_m1[0, 0, 0] != 0.  # Cannot be represented by scalar Bz/2.
    with pytest.raises(ValueError):
        gauged.source_c_m1[0, 0, 0] = 0.
    expected_images = _images(cp.find_conjugate_planes(plain, ref))
    observed_images = _images(cp.find_conjugate_planes(gauged, ref))
    assert [c.z_mm for c in observed_images] == pytest.approx([c.z_mm for c in expected_images], abs=1e-5)


def _oscillator_atlas(step, omega=50., stop=150.):
    z = np.arange(0., stop+step/2, step)
    maps = []
    for value in z*1e-3:
        cosine, sine = np.cos(omega*value), np.sin(omega*value)
        maps.append(np.block([[np.eye(2)*cosine, np.eye(2)*sine/omega],
                              [-np.eye(2)*sine*omega, np.eye(2)*cosine]]))
    return cp.ConjugateAtlas(z, maps)


def test_cached_hermite_refinement_converges_for_continuous_focusing_field():
    expected = np.array([np.pi/50.*1000., 2*np.pi/50.*1000.])+1.123
    coarse = _images(cp.find_conjugate_planes(_oscillator_atlas(2.), 1.123))
    fine = _images(cp.find_conjugate_planes(_oscillator_atlas(.25), 1.123))
    coarse_error = np.max(np.abs(np.asarray([c.z_mm for c in coarse])-expected))
    fine_error = np.max(np.abs(np.asarray([c.z_mm for c in fine])-expected))
    assert fine_error < 1e-5
    assert fine_error < coarse_error


def test_unresolved_coarse_interpolation_is_not_certified_as_an_image():
    atlas = _oscillator_atlas(10., omega=100., stop=100.)
    result = cp.find_conjugate_planes(atlas, 4.123)
    assert any(c.kind == "approximate" for c in result.candidates)


def test_ill_conditioned_or_out_of_range_queries_fail_explicitly():
    atlas = _thin_atlas()
    for reference in (-1., 126., np.nan):
        with pytest.raises(ValueError, match="outside"):
            cp.find_conjugate_planes(atlas, reference)
    maps = np.repeat(np.diag((1e8, 1., 1e-8, 1.))[None], 3, axis=0)
    bad = cp.ConjugateAtlas(np.array([0., 1., 2.]), maps)
    with pytest.raises(ValueError, match="ill-conditioned"):
        cp.find_conjugate_planes(bad, 0.)


def test_zero_b_with_rank_deficient_a_is_not_advertised_as_an_image():
    # Deliberately malformed/collapsed target map: B=0 alone must not certify
    # an image when the stored position response loses one dimension.
    z = np.linspace(0., 20., 21)
    maps = []
    for value in z:
        matrix = np.eye(4)
        matrix[1, 1] = 1.-value/20.
        matrix[:2, 2:] = np.eye(2)*value*1e-3*(1.-value/20.)
        matrix[3, 1] = -50.
        matrix[2:, 2:] = np.eye(2)*(1.-value/10.)
        maps.append(matrix)
    result = cp.find_conjugate_planes(cp.ConjugateAtlas(z, maps), 0.)
    assert _images(result) == []
    target = next(candidate for candidate in result.candidates if candidate.z_mm == 20.)
    assert target.kind == "approximate"
    assert target.residual_m_per_rad == pytest.approx(0., abs=1e-15)
    assert "Position map A has rank 1" in target.detail


def test_read_only_atlas_and_query_cancellation():
    atlas = _thin_atlas()
    with pytest.raises(ValueError):
        atlas.matrices[0, 0, 0] = 5.
    with pytest.raises(NumericalJobCancelled):
        cp.find_conjugate_planes(atlas, 2., cancelled=lambda: True)


def test_captured_domain_and_backend_preference_are_preserved(monkeypatch):
    state = SimpleNamespace(
        sample=SimpleNamespace(z_mm=1001.), electron_gun=SimpleNamespace(exit_plane_z_mm=1000.),
        energy_filter_installed=True, energy_filter=SimpleNamespace(entrance_z_mm=1003.),
        beam_voltage_kv=300., lenses=(), compute_backend="gpu", _active_backends_used={"old GPU rays"},
    )
    receipt = {"optical_tuning": True, "sample_scattering_applied": False,
        "optical_execution_extent": {"coordinate_system": "column_axial_z_mm",
        "physics_scope": "optical_reference_without_specimen_interactions",
        "start_z_mm": 0., "completed_z_mm": 1005.}}
    result = SimpleNamespace(state_snapshot=state,
        simulation=SimpleNamespace(section_checkpoint=None, metrics=receipt))
    calls = []

    def maps(observer, source, targets):
        assert observer is not state and observer.compute_backend == "gpu"
        assert observer._active_backends_used == set()
        calls.append((source, tuple(targets)))
        return {z: TransverseTransfer(source, z, np.eye(2), np.eye(2)*(z-source)*1e-3,
            np.zeros((2, 2)), np.eye(2), input_basis=PLANE_CANONICAL_MOMENTUM) for z in targets}

    monkeypatch.setattr(cp, "canonical_transfers", maps)
    monkeypatch.setattr(cp, "_active_column_electric_field", lambda *_: None)
    monkeypatch.setattr(cp, "fields", lambda z, _: (np.zeros(len(z)), None, None))
    atlas = cp.build_conjugate_atlas(result)
    assert atlas.lower_z_mm == 1000.
    assert atlas.upper_z_mm < 1003.
    assert len(calls) == 1
    assert state._active_backends_used == {"old GPU rays"}
    for reference in (1000.3, 1001.5, 1002.75):
        assert cp.find_conjugate_planes(atlas, reference).candidates == ()
    assert len(calls) == 1  # continuous selected-Z movement never traces again


def test_missing_executed_receipt_is_not_treated_as_current_requested_range():
    result = SimpleNamespace(state_snapshot=SimpleNamespace(),
                             simulation=SimpleNamespace(metrics={}))
    with pytest.raises(ValueError, match="verified executed"):
        cp.build_conjugate_atlas(result)


def test_generalized_canonical_observer_preserves_specimen_wrapper_and_reference(monkeypatch):
    from temsim.optics import direct_alignment as da
    from temsim.physics.first_order import SPECIMEN_CANONICAL_MOMENTUM
    state = SimpleNamespace(sample=SimpleNamespace(z_mm=1000.), step_mm=.1,
                             beam_voltage_kv=300.)
    monkeypatch.setattr(da, "_active_column_electric_field", lambda *_: None)
    monkeypatch.setattr(da, "active_vector_providers", lambda *_: ())
    monkeypatch.setattr(da, "fields", lambda z, _: (np.zeros(len(z)),)*3)
    monkeypatch.setattr(da, "gun_paraxial_fields", lambda _, z: (np.zeros(len(z)),)*2)
    monkeypatch.setattr(da, "skew_quadrupole_field", lambda z, _: np.zeros(len(z)))
    monkeypatch.setattr(da, "column_dipole_fields", lambda *_: ())
    specimen = da.diffraction_transfers(state, (1001., 1003.))
    arbitrary = da.canonical_transfers(state, 900., (901., 903.))
    for offset in (1., 3.):
        old, new = specimen[1000.+offset], arbitrary[900.+offset]
        assert old.input_basis == SPECIMEN_CANONICAL_MOMENTUM
        assert new.input_basis == PLANE_CANONICAL_MOMENTUM
        np.testing.assert_allclose(new.matrix, old.matrix, atol=1e-14)
        np.testing.assert_allclose(new.matrix, _drift(offset), atol=1e-14)
    assert state.sample.z_mm == 1000.  # selected reference never moves physical sample
