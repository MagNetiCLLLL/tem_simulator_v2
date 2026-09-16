import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.physics.analytic_gun_field import evaluate


def test_compiled_field_matches_numpy_at_edges_and_with_nondefault_controls():
    field = default_state().electron_gun.electric_field
    field.extractor.voltage_kv = 4.8
    field.extractor.field_center_offset_mm = .2
    field.electrostatic_lens.voltage_kv = 1.1
    field.electrostatic_lens.field_center_offset_mm = -.3
    field.accelerator.high_tension_kv = 200.
    field.accelerator.field_center_offset_mm = .4
    edges = [field.extractor.transition_start_mm+.2, field.extractor.transition_end_mm+.2]
    edges += [s.center_from_tip_mm+.4+sign*s.soft_edge_mm for s in field.accelerator.stages for sign in (-1, 1)]
    z = np.r_[np.linspace(-1, 500, 1003), edges,
              np.asarray(edges)-1e-8, np.asarray(edges)+1e-8]
    fast = field.axial_potential_v_and_derivatives_per_mm(z)
    field.compiled_axial = False
    reference = field.axial_potential_v_and_derivatives_per_mm(z)
    for a, b in zip(fast, reference):
        np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-8)
    positions = np.column_stack((np.linspace(-1e-4, 1e-4, len(z)), np.full(len(z), 1e-5), z*1e-3))
    reference_p = field.potential_v_at_global_positions(positions)
    reference_e = field.field_at_global_positions_v_per_m(positions)
    field.compiled_axial = True
    np.testing.assert_allclose(field.potential_v_at_global_positions(positions), reference_p, rtol=2e-12, atol=2e-8)
    np.testing.assert_allclose(field.field_at_global_positions_v_per_m(positions), reference_e, rtol=2e-12, atol=2e-5)


def test_custom_electrode_provider_is_not_bypassed():
    field = default_state().electron_gun.electric_field
    field.extractor.axial_potential_v_and_derivatives_per_mm = lambda z: (np.asarray(z)*0+1.,)*4
    assert evaluate(field, [0., 1.]) is None
