"""Local canonical diagnostics retain placed main-column field coordinates."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.optics.direct_alignment import canonical_source_basis, canonical_transfers, _LiveFirstOrderModel
from temsim.physics.core import E, electron
from temsim.physics.posed_column_fields import capture_posed_column_fields
from test_posed_column_fields import component_state


@pytest.fixture(scope="module")
def native():
    return default_state()


def _translated_component(native, key="probe_qph2_quadrupole", shift_z_mm=100.):
    state, component = component_state(native, key, angle_mrad=8., offset_mm=.04)
    state._resolved_assembly = replace(state._resolved_assembly, parts=tuple(
        replace(part, data={**part.data, "offset_z_mm": shift_z_mm})
        if part.key == key else part for part in state._resolved_assembly.parts))
    return state, component


@pytest.mark.parametrize("key", ["condenser_stigmator", "probe_qph2_quadrupole", "probe_hp2_hexapole", "probe_dph2_deflector"])
def test_canonical_source_full_matrix_matches_independent_potential_difference(native, key):
    state, component = component_state(native, key, angle_mrad=12., offset_mm=.03)
    field, = capture_posed_column_fields(state)
    position = np.array((2e-6, -3e-6, component.z_mm*1e-3))
    momentum = electron(state)[1]
    basis = canonical_source_basis(state, component.z_mm, position_xy_m=position[:2],
                                   vector_providers=(), posed_fields=(field,))
    derivatives = []
    for axis in range(2):
        delta = np.eye(3)[axis]*1e-8
        derivatives.append((field.vector_potential_at_global_positions_t_m(position+delta)[:2]
                            -field.vector_potential_at_global_positions_t_m(position-delta)[:2])/(2e-8))
    expected = E/momentum*np.asarray(derivatives).T
    np.testing.assert_allclose(basis[2:, :2], expected, rtol=2e-7, atol=1e-12)
    assert np.linalg.norm(expected) > 0.
    np.testing.assert_array_equal(basis[:2], np.eye(4)[:2])
    np.testing.assert_array_equal(basis[2:, 2:], np.eye(2))


def test_same_plane_map_is_local_canonical_increment_basis_not_a_chief_shift(native):
    state, component = component_state(native, "probe_qph2_quadrupole", angle_mrad=8., offset_mm=.04)
    result = canonical_transfers(state, component.z_mm, (component.z_mm,))[component.z_mm]
    np.testing.assert_allclose(result.matrix, canonical_source_basis(state, component.z_mm))
    assert result.position_offset_m == result.angle_offset_rad == (0., 0.)


def test_placed_support_chooses_full_model_and_native_location_remains_field_free(native):
    state, component = _translated_component(native)
    lens = deepcopy(next(item for item in native.lenses if item.key == "projector_lens_2"))
    state.lenses = [lens]
    # A candidate round-lens variable is enough to construct the model; no
    # optimiser is executed, and the posed channel remains independently fixed.
    full = _LiveFirstOrderModel(state, component.z_mm+99., component.z_mm+101., (lens.key,), step_mm=.05)
    assert full.full_field_transfer
    before = _LiveFirstOrderModel(state, component.z_mm-.5, component.z_mm+.5, (lens.key,), step_mm=.05)
    assert not before.full_field_transfer
    np.testing.assert_array_equal(before.sx_m2, 0.)
    np.testing.assert_array_equal(before.sy_m2, 0.)


def test_no_ghost_quadrupole_after_the_physical_field_has_moved(native):
    state, component = _translated_component(native)
    source, target = component.z_mm-.5, component.z_mm+.5
    observed = canonical_transfers(state, source, (target,))[target]
    expected = np.eye(4)
    expected[0, 2] = expected[1, 3] = .001
    np.testing.assert_allclose(observed.matrix, expected, atol=1e-14)


def test_placed_support_uses_same_full_tracer_as_explicit_request(native):
    state, component = _translated_component(native)
    source, target = component.z_mm+99.8, component.z_mm+100.2
    automatic = canonical_transfers(state, source, (target,))[target]
    requested = canonical_transfers(state, source, (target,), stable_axisymmetric=False)[target]
    np.testing.assert_allclose(automatic.matrix, requested.matrix, rtol=0., atol=0.)
    np.testing.assert_allclose(automatic.position_offset_m, requested.position_offset_m, rtol=0., atol=0.)
    assert np.linalg.norm(automatic.matrix[2:, :2]) > 1e-5
