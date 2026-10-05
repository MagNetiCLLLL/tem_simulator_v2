"""Bounded specimen-coordinate consistency, not full-column qualification."""

from types import SimpleNamespace

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.diagnostics import optical_transfer_records
from temsim.optics.column import default_state
from temsim.optics.direct_alignment import diffraction_transfer, diffraction_transfers
from temsim.physics import selected_plane
from temsim.physics.column_wave import _column_transports
from temsim.physics.core import build_propagation_plan, fields
from temsim.physics.first_order import TransverseTransfer, trace_transverse_transfer
from temsim.physics.scan_geometry import classify_sample_plane_transfer


CANONICAL = "specimen_canonical_momentum"
MECHANICAL = "mechanical_slopes"


def _bounded_state(*, objective_enabled):
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    state.acceleration_enabled = False
    state.energy_filter_installed = False
    state.image_corrector_installed = False
    state.recording_planes = ()
    state.step_mm = 0.025
    state.history_step_mm = 1.0
    for component in (
        *state.lenses, *state.stigmators, *state.corrector_elements, *state.deflectors,
    ):
        component.enabled = False
    state.objective_lens.enabled = objective_enabled
    state.objective_lens.percent = 69.0
    state.sync_objective()
    # Only the actual objective BFP is requested by this bounded diagnostic.
    # Suppress the additional image-plane output without altering its field.
    state.objective_image_plane_z_mm = None
    return state


def _result(state, target, *, saved=None, records=()):
    return SimpleNamespace(
        state_snapshot=state,
        simulation=SimpleNamespace(
            metrics={
                "optical_tuning": True,
                "sample_scattering_applied": False,
                "optical_execution_extent": {
                    "coordinate_system": "column_axial_z_mm",
                    "physics_scope": "optical_reference_without_specimen_interactions",
                    "start_z_mm": float(state.sample.z_mm),
                    "completed_z_mm": float(target) + 1.0,
                },
            },
            section_checkpoint=None,
            sample_to_analysis_transfer=saved,
            optical_transfers=records,
        ),
    )


def _assert_diagnostic(diagnostic, expected):
    kind, image_residual, diffraction_residual = expected
    assert diagnostic.kind == kind, diagnostic.detail
    assert diagnostic.image_residual_m_per_rad == pytest.approx(
        image_residual, abs=1.0e-12,
    )
    assert diagnostic.diffraction_residual == pytest.approx(
        diffraction_residual, abs=1.0e-12,
    )


def test_objective_bfp_named_selected_and_wave_position_blocks_agree(monkeypatch):
    state = _bounded_state(objective_enabled=True)
    source = float(state.sample.z_mm)
    target = float(state.objective_back_focal_plane_z_mm)
    assert 0.0 < target - source < 14.0
    assert abs(float(fields(np.array([source]), state)[0][0])) > 1.0e-6

    mechanical = trace_transverse_transfer(state, source, target, maximum_step_mm=0.025)
    original_mechanical = mechanical.matrix.copy()
    canonical = diffraction_transfer(state, target)
    assert mechanical.input_basis == MECHANICAL
    assert canonical.input_basis == CANONICAL
    assert np.linalg.norm(canonical.j_img - mechanical.j_img, ord=2) > 1.0e-7

    records = optical_transfer_records(state)
    assert len(records) == 1
    record = records[0]
    assert record.key == "objective_back_focal_plane"
    assert record.z_mm == target
    assert record.transfer.input_basis == CANONICAL
    np.testing.assert_allclose(record.transfer.j_img, canonical.j_img, atol=1.0e-12)
    np.testing.assert_allclose(
        record.transfer.j_diff_m_per_rad, canonical.j_diff_m_per_rad, atol=1.0e-12,
    )
    expected = classify_sample_plane_transfer(canonical)
    assert expected[0] == "diffraction"
    assert classify_sample_plane_transfer(record.transfer)[0] == expected[0]

    result = _result(state, target, saved=canonical, records=records)
    with monkeypatch.context() as cached:
        cached.setattr(
            selected_plane, "diffraction_transfer",
            lambda *_args: pytest.fail("Exact tagged canonical data must be reused"),
        )
        _assert_diagnostic(selected_plane.calculate_selected_plane(result, target), expected)
    result.simulation.sample_to_analysis_transfer = None
    result.simulation.optical_transfers = ()
    _assert_diagnostic(selected_plane.calculate_selected_plane(result, target), expected)

    # Independent midpoint-exponential wave Hamiltonian, restricted to this
    # same local interval and step. Its target momenta differ in convention,
    # so compare only the shared output-position blocks A and B.
    plan = build_propagation_plan(
        state, source, target, (), maximum_step_mm=0.025,
        include_spherical_aberration=False, include_hexapole=False,
    )
    wave_matrix = np.eye(4)
    for path in _column_transports(plan, state.beam_voltage_kv):
        wave_matrix = path.matrix @ wave_matrix
    assert np.max(np.abs(wave_matrix[:2, :2] - canonical.j_img)) <= 2.0e-5
    relative_angular_error = np.linalg.norm(
        wave_matrix[:2, 2:] - canonical.j_diff_m_per_rad, ord=2,
    ) / np.linalg.norm(canonical.j_diff_m_per_rad, ord=2)
    assert relative_angular_error <= 2.0e-5
    assert np.linalg.norm(wave_matrix[:2, :2], ord=2) <= 1.0e-3
    np.testing.assert_array_equal(mechanical.matrix, original_mechanical)
    assert mechanical.input_basis == MECHANICAL


def test_exact_saved_analysis_requires_canonical_basis_metadata(monkeypatch):
    state = SimpleNamespace(
        sample=SimpleNamespace(z_mm=1000.0), beam_voltage_kv=300.0,
        step_mm=0.025, lenses=(), energy_filter_installed=False,
    )
    source, target = 1000.0, 1001.0
    identity, zero = np.eye(2), np.zeros((2, 2))
    for a, b, kind in ((3.0, 0.0, "image"), (0.0, 0.2, "diffraction")):
        canonical = TransverseTransfer(
            source, target, a * identity, b * identity, zero, identity,
            input_basis=CANONICAL,
        )
        # Exact coordinates and an analysis-field name must not turn an
        # untagged mechanical map into canonical data. Its classification is
        # intentionally the opposite of the required canonical result.
        mechanical = TransverseTransfer(
            source, target, (0.0 if a else 3.0) * identity,
            (0.2 if a else 0.0) * identity, zero, identity,
        )
        assert mechanical.input_basis == MECHANICAL
        calls = []

        def observer(snapshot, stop):
            calls.append((snapshot, stop))
            return canonical

        monkeypatch.setattr(selected_plane, "diffraction_transfer", observer)
        result = _result(state, target, saved=mechanical)
        diagnostic = selected_plane.calculate_selected_plane(result, target)
        assert diagnostic.kind == kind
        assert len(calls) == 1 and calls[0][1] == target
        assert calls[0][0] is not state

        result.simulation.sample_to_analysis_transfer = canonical
        monkeypatch.setattr(
            selected_plane, "diffraction_transfer",
            lambda *_args: pytest.fail("Tagged exact canonical data must be reused"),
        )
        _assert_diagnostic(
            selected_plane.calculate_selected_plane(result, target),
            (kind, abs(b), abs(a)),
        )


def test_field_free_batch_singleton_and_mechanical_maps_share_exact_drift():
    state = _bounded_state(objective_enabled=False)
    source = float(state.sample.z_mm)
    targets = (source + 2.125, source + 5.0)
    batch = diffraction_transfers(state, targets)
    assert set(batch) == set(targets)
    for target in targets:
        expected = np.eye(4)
        expected[:2, 2:] = np.eye(2) * (target - source) * 1.0e-3
        singleton = diffraction_transfer(state, target)
        mechanical = trace_transverse_transfer(state, source, target, maximum_step_mm=0.025)
        assert singleton.input_basis == batch[target].input_basis == CANONICAL
        assert mechanical.input_basis == MECHANICAL
        for actual in (batch[target], singleton, mechanical):
            np.testing.assert_allclose(actual.matrix, expected, rtol=0.0, atol=2.0e-8)
        assert classify_sample_plane_transfer(singleton)[0] == "mixed"
        with pytest.raises(ValueError, match="canonical momentum"):
            classify_sample_plane_transfer(mechanical)


@pytest.mark.parametrize("later_field", ("quadrupole", "dipole", "electric"))
def test_later_general_field_does_not_change_the_upstream_canonical_solver(
    monkeypatch, later_field,
):
    from temsim.optics import direct_alignment as observer

    source, near, far = 0., 1., 4.
    state = SimpleNamespace(sample=SimpleNamespace(z_mm=source),
                            beam_voltage_kv=300., step_mm=.025)

    def bounded_fields(z_mm, _state):
        z = np.asarray(z_mm)
        # A nonzero, overlapping axial objective field exercises the Larmor
        # source convention. The transverse field exists only after near Z.
        magnetic = .7 / (1. + (z / 2.)**2)
        quadrupole = np.where((z >= 3.) & (z <= 3.1), 100., 0.)
        if later_field != "quadrupole":
            quadrupole = np.zeros_like(z)
        return magnetic, quadrupole, -quadrupole

    electric = SimpleNamespace(is_constant_on_interval=lambda start, stop: stop < 3.)
    monkeypatch.setattr(observer, "fields", bounded_fields)
    monkeypatch.setattr(observer, "active_mapped_providers", lambda _state: ())
    monkeypatch.setattr(observer, "_active_column_electric_field",
                        lambda _state, start, stop: electric
                        if later_field == "electric" and stop >= 3. else None)
    monkeypatch.setattr(observer, "_column_reference_momentum",
                        lambda *_: observer._electron_momentum_kg_m_s(300.))
    monkeypatch.setattr(observer, "gun_paraxial_fields",
                        lambda _state, z: (np.zeros_like(z), np.zeros_like(z)))
    monkeypatch.setattr(observer, "skew_quadrupole_field", lambda z, _state: np.zeros_like(z))
    dipole = SimpleNamespace(lower_m=.003, upper_m=.0031, bx_t=1e-4, by_t=0.)
    monkeypatch.setattr(observer, "column_dipole_fields",
                        lambda _state: (dipole,) if later_field == "dipole" else ())
    traced_targets = []

    def mechanical_observer(_state, start, targets, **_kwargs):
        targets = tuple(targets)
        traced_targets.append(targets)
        # Deliberately distinct maps expose an incorrect upstream fallback;
        # this fixture checks solver selection, not general-field integration.
        return {
            stop: TransverseTransfer(start, stop, np.eye(2),
                                     np.eye(2)*(stop-start)*1e-3,
                                     np.zeros((2, 2)), np.eye(2))
            for stop in targets
        }

    monkeypatch.setattr(observer, "trace_transverse_transfers", mechanical_observer)

    singleton = observer.diffraction_transfer(state, near)
    assert not traced_targets
    batch = observer.diffraction_transfers(state, (near, far))

    assert traced_targets == [(far,)]
    assert batch[near].input_basis == batch[far].input_basis == CANONICAL
    np.testing.assert_array_equal(batch[near].matrix, singleton.matrix)
    assert classify_sample_plane_transfer(batch[near]) == classify_sample_plane_transfer(singleton)
