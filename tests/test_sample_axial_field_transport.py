import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.specimen.axial_field_transport import (
    advance_in_uniform_axial_field,
    axial_rotation_rate_rad_per_nm,
    sample_axial_field_diagnostic,
)


def test_uniform_axial_field_drift_is_reversible_and_energy_independent():
    position = np.asarray((12.0, -8.0, 3.0))
    direction = np.asarray((0.2, -0.1, 1.0))
    direction /= np.linalg.norm(direction)
    rate = axial_rotation_rate_rad_per_nm(1.7, 300_000.0)

    endpoint, final_direction = advance_in_uniform_axial_field(
        position,
        direction,
        25_000.0,
        rotation_rate_rad_per_nm=rate,
    )
    restored_position, restored_direction = advance_in_uniform_axial_field(
        endpoint,
        final_direction,
        -25_000.0,
        rotation_rate_rad_per_nm=rate,
    )

    assert np.linalg.norm(final_direction) == pytest.approx(1.0, abs=2.0e-15)
    np.testing.assert_allclose(restored_position, position, rtol=0.0, atol=2.0e-11)
    np.testing.assert_allclose(restored_direction, direction, rtol=0.0, atol=2.0e-15)
    assert final_direction[2] == pytest.approx(direction[2], abs=2.0e-15)


def test_zero_axial_field_is_exact_straight_drift():
    position = np.asarray((1.0, 2.0, 3.0))
    direction = np.asarray((0.1, -0.2, 1.0))
    direction /= np.linalg.norm(direction)

    endpoint, final_direction = advance_in_uniform_axial_field(
        position,
        direction,
        70.0,
        rotation_rate_rad_per_nm=0.0,
    )

    np.testing.assert_allclose(endpoint, position + 70.0 * direction)
    np.testing.assert_allclose(final_direction, direction)


def test_sample_field_uses_the_same_objective_provider_as_the_column_solver():
    catalog = AssemblyCatalog()
    state = default_state()
    catalog.apply(state, catalog.default_selection())

    diagnostic = sample_axial_field_diagnostic(state)
    expected_objective = float(
        state.objective_lens.magnetic_field_t((state.sample.z_mm,))[0]
    )

    assert diagnostic.total_field_t != 0.0
    assert diagnostic.objective_field_t == pytest.approx(expected_objective)
    assert dict(diagnostic.active_lens_contributions_t)["objective_lens"] == (
        pytest.approx(expected_objective)
    )
    assert diagnostic.transport_model == (
        "local_uniform_axial_Bz_exact_helical_drift"
    )
    assert diagnostic.geometry_material_coupled is False
