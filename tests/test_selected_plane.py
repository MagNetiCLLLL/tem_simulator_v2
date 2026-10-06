"""Selected-Z software boundaries and bounded first-order optical references.

The lens-conjugacy fixture below uses independent thin-lens matrix algebra;
it tests classification/ownership, not full microscope qualification.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.cpu_resources import NumericalJobCancelled
from temsim.physics import selected_plane
from temsim.physics.first_order import TransverseTransfer


def _state():
    return SimpleNamespace(sample=SimpleNamespace(z_mm=1000.), beam_voltage_kv=300.,
        step_mm=.1, lenses=[SimpleNamespace(z_mm=1100., power_m1=0.)],
        projector_mode="diffraction", energy_filter_installed=False)


def _result(state=None, *, completed=1300.):
    state = _state() if state is None else state
    metrics = {"optical_tuning": True, "sample_scattering_applied": False,
        "optical_execution_extent": {"coordinate_system": "column_axial_z_mm",
            "physics_scope": "optical_reference_without_specimen_interactions",
            "start_z_mm": 0., "completed_z_mm": completed}}
    return SimpleNamespace(state_snapshot=state, simulation=SimpleNamespace(metrics=metrics,
        section_checkpoint=None, sample_to_analysis_transfer=None, optical_transfers=()))


def _map(source, target, a=1., b=.2):
    identity, zeros = np.eye(2), np.zeros((2, 2))
    return TransverseTransfer(source, target, a*identity, b*identity, zeros, identity,
                              input_basis="specimen_canonical_momentum")


@pytest.mark.parametrize(("z", "kind"), ((999., "upstream"), (1000., "specimen"),
    (1301., "not_calculated"), (-1., "not_calculated")))
def test_boundaries_never_trace_fields(monkeypatch, z, kind):
    monkeypatch.setattr(selected_plane, "diffraction_transfer",
                        lambda *_args: pytest.fail("Boundary status must not trace"))
    result = _result()
    assert selected_plane.plane_status_without_trace(result, z).kind == kind
    assert selected_plane.calculate_selected_plane(result, z).kind == kind


def test_requested_cutoff_or_ray_tail_is_not_an_executed_endpoint():
    result = _result()
    result.simulation.metrics = {"section_target_z_mm": 1800.}
    result.simulation.incident = SimpleNamespace(z=np.array([0., 1800.]))
    assert selected_plane.plane_status_without_trace(result, 1200.).kind == "unavailable"


def test_material_endpoint_is_not_truncated_to_incident_specimen_checkpoint():
    result = _result()
    result.simulation.section_checkpoint = SimpleNamespace(gun_trace=SimpleNamespace(z_mm=np.array([0., 450.])))
    result.simulation.metrics.update(section_target_z_mm=1300., section_resumable_through_z_mm=1250.)
    result.simulation.incident = SimpleNamespace(z=np.array([450., 1000.]))
    assert selected_plane.plane_status_without_trace(result, 1299.) is None
    assert selected_plane.plane_status_without_trace(result, 1301.).kind == "not_calculated"


@pytest.mark.parametrize("gun_trace", (None, SimpleNamespace(z_mm=np.array([]))))
def test_missing_executed_origin_cannot_admit_solver(monkeypatch, gun_trace):
    result = _result()
    result.simulation.section_checkpoint = SimpleNamespace(gun_trace=gun_trace)
    result.simulation.metrics.update(section_target_z_mm=1300.)
    monkeypatch.setattr(selected_plane, "diffraction_transfer",
                        lambda *_args: pytest.fail("Missing executed origin must not trace"))
    diagnostic = selected_plane.calculate_selected_plane(result, 1200.)
    assert diagnostic.kind == "unavailable"
    assert "origin and endpoint" in diagnostic.detail


@pytest.mark.parametrize("removed", ("beam_voltage_kv", "lenses", "step_mm"))
def test_incomplete_gui_fixture_does_not_admit_solver(removed):
    result = _result()
    delattr(result.state_snapshot, removed)
    assert selected_plane.plane_status_without_trace(result, 1200.).kind == "unavailable"


def test_missing_snapshot_and_nonfinite_z_are_explicit():
    result = _result()
    result.state_snapshot = None
    assert selected_plane.plane_status_without_trace(result, 1200.).kind == "unavailable"
    assert selected_plane.plane_status_without_trace(_result(), np.nan).kind == "unavailable"


@pytest.mark.parametrize("population", (0, 10))
@pytest.mark.parametrize("z", (1150., 1200.))
def test_filter_entrance_and_after_cannot_use_straight_column_map(monkeypatch, population, z):
    result = _result()
    result.state_snapshot.energy_filter_installed = True
    result.state_snapshot.energy_filter = SimpleNamespace(entrance_z_mm=1150.)
    result.simulation.metrics["sample_beam_surviving_rays"] = population
    result.simulation.sample_to_analysis_transfer = _map(1000., z, a=0.)
    monkeypatch.setattr(selected_plane, "diffraction_transfer",
                        lambda *_args: pytest.fail("Filter must not be bypassed"))
    assert selected_plane.calculate_selected_plane(result, z).kind == "unavailable"
    assert selected_plane.plane_status_without_trace(result, 1149.) is None


def test_unknown_installed_filter_boundary_is_unavailable():
    result = _result()
    result.state_snapshot.energy_filter_installed = True
    assert selected_plane.plane_status_without_trace(result, 1200.).kind == "unavailable"


@pytest.mark.parametrize(("a", "b", "kind"), ((1., 0., "image"), (0., .1, "diffraction"),
    (1., .1, "mixed"), (0., 0., "degenerate")))
def test_exact_saved_map_ignores_mode_label_and_needs_no_new_trace(monkeypatch, a, b, kind):
    result = _result()
    result.simulation.sample_to_analysis_transfer = _map(1000., 1200., a=a, b=b)
    result.state_snapshot.projector_mode = "image" if kind == "diffraction" else "diffraction"
    monkeypatch.setattr(selected_plane, "diffraction_transfer",
                        lambda *_args: pytest.fail("An exact saved map must be reused"))
    diagnostic = selected_plane.calculate_selected_plane(result, 1200.)
    assert diagnostic.kind == kind
    assert diagnostic.image_residual_m_per_rad == pytest.approx(abs(b))
    assert diagnostic.diffraction_residual == pytest.approx(abs(a))


def test_exact_canonical_analysis_can_be_reused_with_zero_particle_arrivals(monkeypatch):
    result = _result()
    result.simulation.sample_to_analysis_transfer = _map(1000., 1200., a=0.)
    result.simulation.metrics["sample_beam_surviving_rays"] = 0
    monkeypatch.setattr(selected_plane, "diffraction_transfer",
                        lambda *_args: pytest.fail("Exact canonical analysis should be reused"))
    diagnostic = selected_plane.calculate_selected_plane(result, 1200.)
    assert diagnostic.kind == "diffraction"
    assert "No tracked incident particles" in diagnostic.detail


def test_generic_optical_record_is_not_a_canonical_analysis_cache(monkeypatch):
    result = _result()
    result.simulation.optical_transfers = (SimpleNamespace(transfer=_map(1000., 1200., a=0.)),)
    calls = []

    def canonical_observer(state, target):
        calls.append((state, target))
        return _map(state.sample.z_mm, target, a=1.)

    monkeypatch.setattr(selected_plane, "diffraction_transfer", canonical_observer)
    diagnostic = selected_plane.calculate_selected_plane(result, 1200.)
    assert diagnostic.kind == "mixed"
    assert len(calls) == 1 and calls[0][1] == 1200.


@pytest.mark.parametrize(("source", "target"), ((999., 1200.), (1000., 1200.000000001)))
def test_reuse_requires_exact_source_and_target_and_preserves_snapshot(monkeypatch, source, target):
    result = _result()
    result.simulation.sample_to_analysis_transfer = _map(source, target, a=0.)
    received = []

    def observer(state, stop):
        start = state.sample.z_mm
        received.append((state, start, stop))
        state._last_ray_backend = "test observer metadata"
        return _map(start, stop)

    monkeypatch.setattr(selected_plane, "diffraction_transfer", observer)
    diagnostic = selected_plane.calculate_selected_plane(result, 1200.)
    assert diagnostic.kind == "mixed"
    assert len(received) == 1 and received[0][0] is not result.state_snapshot
    assert received[0][1:] == (1000., 1200.)
    assert not hasattr(result.state_snapshot, "_tuning_cancelled")
    assert not hasattr(result.state_snapshot, "_last_ray_backend")


def test_observer_detaches_mutable_runtime_containers_and_lens_metadata(monkeypatch):
    result = _result()
    snapshot = result.state_snapshot
    shared_field = SimpleNamespace(values=np.array([1., 2.]))
    shared_field.values.setflags(write=False)
    snapshot._active_backends_used = {"CUDA GPU"}
    snapshot._runtime_lens_field_provider_cache = {"lens": ("token", shared_field)}
    snapshot._runtime_nonlinear_provider_cache = {"lens": ("token", shared_field)}
    snapshot._field_provider_diagnostics = {"lens": {"scope": "accepted calculation"}}
    snapshot._lens_field_map_bindings = {"lens": shared_field}
    snapshot._objective_lens = snapshot.lenses[0]

    def observer(state, target):
        source = state.sample.z_mm
        assert state._objective_lens is state.lenses[0]
        assert state._objective_lens is not snapshot._objective_lens
        assert state._runtime_lens_field_provider_cache["lens"][1] is shared_field
        assert state._lens_field_map_bindings["lens"] is shared_field
        state._active_backends_used.add("Numba CPU")
        for name in ("_runtime_lens_field_provider_cache", "_runtime_nonlinear_provider_cache",
                     "_field_provider_diagnostics", "_lens_field_map_bindings"):
            getattr(state, name)["observer"] = "diagnostic-only metadata"
        state._objective_lens._image_plane_z_mm = target
        state._objective_lens.power_m1 = 1.
        return _map(source, target)

    monkeypatch.setattr(selected_plane, "diffraction_transfer", observer)
    assert selected_plane.calculate_selected_plane(result, 1200.).kind == "mixed"
    assert snapshot._active_backends_used == {"CUDA GPU"}
    for name in ("_runtime_lens_field_provider_cache", "_runtime_nonlinear_provider_cache",
                 "_field_provider_diagnostics", "_lens_field_map_bindings"):
        assert "observer" not in getattr(snapshot, name)
    assert not hasattr(snapshot._objective_lens, "_image_plane_z_mm")
    assert snapshot._objective_lens.power_m1 == 0.
    assert not shared_field.values.flags.writeable


def test_fixed_energy_and_geometry_lens_power_controls_plane_kind(monkeypatch):
    """Independent D(0.1 m) L(P) D(0.1 m), with all geometry held fixed."""
    result = _result()

    def thin_lens_observer(state, target):
        source = state.sample.z_mm
        lens = state.lenses[0]
        before, after = (lens.z_mm-source)*1e-3, (target-lens.z_mm)*1e-3
        drift_before = np.array([[1., before], [0., 1.]])
        drift_after = np.array([[1., after], [0., 1.]])
        lens_map = np.array([[1., 0.], [-lens.power_m1, 1.]])
        matrix = drift_after @ lens_map @ drift_before
        return _map(source, target, matrix[0, 0], matrix[0, 1])

    monkeypatch.setattr(selected_plane, "diffraction_transfer", thin_lens_observer)
    diagnostics = []
    for power in (0., 10., 20.):
        result.state_snapshot.lenses[0].power_m1 = power
        diagnostics.append(selected_plane.calculate_selected_plane(result, 1200.))
    assert [d.kind for d in diagnostics] == ["mixed", "diffraction", "image"]
    assert [d.image_residual_m_per_rad for d in diagnostics] == pytest.approx([.2, .1, 0.], abs=1e-15)
    assert result.state_snapshot.beam_voltage_kv == 300.
    assert result.state_snapshot.sample.z_mm == 1000.
    assert result.state_snapshot.lenses[0].z_mm == 1100.


def test_real_first_order_drift_observer_matches_analytic_reference():
    """Field-free limiting case of the actual observer, not a full tip-chain test."""
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.column import default_state

    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    # Retain the real observer's physical gun/provider, as the existing
    # first-order field-free fixture does. No gun transport or alternate
    # source is calculated in this local drift test.
    state.acceleration_enabled = False
    state.step_mm = .5
    state.history_step_mm = 10.
    state.energy_filter_installed = False
    for component in (*state.lenses, *state.stigmators, *state.corrector_elements):
        component.enabled = False
    result = _result(state, completed=float(state.sample.z_mm)+150.)
    target = float(state.sample.z_mm)+100.123456
    diagnostic = selected_plane.calculate_selected_plane(result, target)
    assert diagnostic.kind == "mixed", diagnostic.detail
    assert diagnostic.diffraction_residual == pytest.approx(1., abs=2e-8)
    assert diagnostic.image_residual_m_per_rad == pytest.approx(.100123456, abs=2e-8)
    assert diagnostic.z_mm == target


def test_nonzero_specimen_field_cached_and_fresh_canonical_responses_agree(monkeypatch):
    """Real bounded objective-field observer; no waves or full gun chain."""
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.column import default_state
    from temsim.optics.direct_alignment import diffraction_transfer
    from temsim.physics.core import fields
    from temsim.physics.first_order import trace_transverse_transfer

    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    state.acceleration_enabled = False
    state.energy_filter_installed = False
    state.step_mm = .025
    for component in (*state.lenses, *state.stigmators, *state.corrector_elements):
        component.enabled = False
    state.objective_lens.enabled = True
    state.objective_lens.percent = 69.
    source = float(state.sample.z_mm)
    target = source + 10.123456
    # The basis distinction must be exercised, not reduced to a vacuum case.
    assert abs(float(fields(np.array([source]), state)[0][0])) > 1e-6
    canonical = diffraction_transfer(state, target)
    mechanical = trace_transverse_transfer(state, source, target, maximum_step_mm=.025)
    assert np.linalg.norm(canonical.j_img-mechanical.j_img, ord=2) > 1e-7

    result = _result(state, completed=target+1.)
    result.simulation.sample_to_analysis_transfer = canonical
    monkeypatch.setattr(selected_plane, "diffraction_transfer",
                        lambda *_args: pytest.fail("Exact canonical analysis must be reused"))
    cached = selected_plane.calculate_selected_plane(result, target)

    # A mechanical record at the identical coordinates must not replace the
    # canonical analysis. Fresh evaluation uses the captured objective field.
    result.simulation.sample_to_analysis_transfer = None
    result.simulation.optical_transfers = (SimpleNamespace(transfer=mechanical),)
    monkeypatch.setattr(selected_plane, "diffraction_transfer", diffraction_transfer)
    fresh = selected_plane.calculate_selected_plane(result, target)
    assert fresh.kind == cached.kind != "unavailable", fresh.detail
    assert fresh.image_residual_m_per_rad == pytest.approx(cached.image_residual_m_per_rad, abs=1e-12)
    assert fresh.diffraction_residual == pytest.approx(cached.diffraction_residual, abs=1e-12)
    assert "canonical/Larmor" in fresh.detail


def test_cancel_before_work_never_traces(monkeypatch):
    monkeypatch.setattr(selected_plane, "diffraction_transfer",
                        lambda *_args: pytest.fail("Cancelled work must not trace"))
    with pytest.raises(NumericalJobCancelled):
        selected_plane.calculate_selected_plane(_result(), 1200., cancelled=lambda: True)


def test_cancelled_during_observer_cannot_publish_result(monkeypatch):
    cancelled = [False]

    def observer(state, target):
        cancelled[0] = True
        return _map(state.sample.z_mm, target)

    monkeypatch.setattr(selected_plane, "diffraction_transfer", observer)
    with pytest.raises(NumericalJobCancelled):
        selected_plane.calculate_selected_plane(_result(), 1200., cancelled=lambda: cancelled[0])


def test_solver_cancel_callback_raises_owned_cancellation(monkeypatch):
    cancelled = [False]

    def observer(state, *_args):
        cancelled[0] = True
        state._tuning_cancelled()
        pytest.fail("Cancellation callback must raise")

    monkeypatch.setattr(selected_plane, "diffraction_transfer", observer)
    with pytest.raises(NumericalJobCancelled):
        selected_plane.calculate_selected_plane(_result(), 1200., cancelled=lambda: cancelled[0])


def test_solver_error_is_explicitly_unavailable(monkeypatch):
    def observer(*_args):
        raise ValueError("Captured field boundary does not cover the query")

    monkeypatch.setattr(selected_plane, "diffraction_transfer", observer)
    diagnostic = selected_plane.calculate_selected_plane(_result(), 1200.)
    assert diagnostic.kind == "unavailable"
    assert "Captured field boundary" in diagnostic.detail
