"""Profile normalization is a lens property, not a query-window operation.

These small field comparisons establish native/packed law agreement. They do
not qualify complete trajectories or measured magnetic material properties.
"""
from dataclasses import replace

import numpy as np
import pytest

from temsim.component_keys import (
    CONDENSER_LENS_1, MINI_CONDENSER, DIFFRACTION_LENS,
    INTERMEDIATE_LENS, PROJECTOR_LENS_1, PROJECTOR_LENS_2,
)
from temsim.optics.column import default_state
from temsim.optics.lens_focal_length import raw_unit_field_peak
from temsim.physics.compiled_magnetic_field import (
    njit, prepare_compiled_magnetic_sources, compiled_magnetic_batch,
)
from temsim.physics.instrument_magnetic import capture_instrument_magnetic_field


LENS_KEYS = (CONDENSER_LENS_1, MINI_CONDENSER, DIFFRACTION_LENS,
             INTERMEDIATE_LENS, PROJECTOR_LENS_1, PROJECTOR_LENS_2)


def _lens_fixture(key):
    state = default_state()
    lens = next(lens for lens in state.lenses if lens.key == key)
    # A non-unit, asymmetric profile catches both scalar normalization and a
    # constant peak assumed from the unmodified default Gaussian parameters.
    term = lens.gaussian[0]
    lens.gaussian = (replace(term, amplitude=2., sigma=.8, offset=-.6),
                     replace(term, amplitude=-.25, sigma=1.3, offset=.7))
    lens.percent, lens.enabled = 30., True
    native = state.condenser_system[key] if key == CONDENSER_LENS_1 else lens
    return state, lens, native


@pytest.mark.parametrize("key", LENS_KEYS)
def test_native_normalization_uses_fixed_profile_peak_for_every_query_shape(key):
    _state, lens, native = _lens_fixture(key)
    z = lens.z_mm + lens.a_mm*np.array([-4., -1.1, -.6, 0., .8, 3.])
    lens.normalise_profile_peak = False
    expected_raw = np.zeros_like(z)
    for term in lens.gaussian:
        center = lens.z_mm+term.offset*lens.a_mm
        expected_raw += term.amplitude*np.exp(-.5*((z-center)
                                                  / abs(term.sigma*lens.a_mm))**2)
    expected_raw *= float(lens.polarity)*lens.scale()
    np.testing.assert_allclose(native.magnetic_field_t(z), expected_raw, rtol=2e-14, atol=1e-15)
    expected = expected_raw/max(raw_unit_field_peak(lens), 1e-15)
    lens.normalise_profile_peak = True
    np.testing.assert_allclose(native.magnetic_field_t(z), expected, rtol=2e-14, atol=1e-15)
    for index, coordinate in enumerate(z):
        np.testing.assert_allclose(native.magnetic_field_t(coordinate), expected[index], rtol=2e-14, atol=1e-15)
        np.testing.assert_allclose(native.magnetic_field_t([coordinate]), expected[index:index+1], rtol=2e-14, atol=1e-15)
    np.testing.assert_allclose(native.magnetic_field_t(z[[4, 1, 5]]), expected[[4, 1, 5]], rtol=2e-14, atol=1e-15)
    expanded = np.r_[lens.z_mm+np.array([-100., 100.])*lens.a_mm, z]
    np.testing.assert_allclose(native.magnetic_field_t(expanded)[2:], expected, rtol=2e-14, atol=1e-15)


@pytest.mark.parametrize("key", LENS_KEYS)
@pytest.mark.skipif(njit is None, reason="Numba optional")
def test_shared_and_compiled_vector_fields_keep_the_same_fixed_normalization(key):
    state, lens, _native = _lens_fixture(key)
    lens.normalise_profile_peak = True
    complete = capture_instrument_magnetic_field(state)
    source = next(source for source in complete._sources if source.key == key)
    field = replace(complete, _sources=(source,))
    packed = prepare_compiled_magnetic_sources(field._sources)
    assert packed is not None
    z = (lens.z_mm+lens.a_mm*np.array([-4., -1.1, -.6, 0., .8, 3.]))*.001
    points = np.column_stack((np.full(len(z), 1e-6), np.full(len(z), -2e-6), z))
    expected = field.field_at_global_positions_t(points)
    # Br is obtained by the same declared finite-difference stencil in both
    # implementations; allow floating-point cancellation, not a model error.
    np.testing.assert_allclose(compiled_magnetic_batch(points, *packed), expected, rtol=2e-7, atol=1e-12)
    for index, point in enumerate(points):
        np.testing.assert_array_equal(field.field_at_global_positions_t(point[None]), expected[index:index+1])
        np.testing.assert_allclose(compiled_magnetic_batch(point[None], *packed), expected[index:index+1], rtol=2e-7, atol=1e-12)
    expanded = np.vstack((points, [[0., 0., 0.], [0., 0., 4.]]))
    np.testing.assert_array_equal(field.field_at_global_positions_t(expanded)[:len(points)], expected)
    np.testing.assert_allclose(compiled_magnetic_batch(expanded, *packed)[:len(points)], expected, rtol=2e-7, atol=1e-12)


def test_normalization_implementation_and_profile_are_bound_to_source_identity():
    from temsim.magnetic_field_scene import _implementation_inputs
    state, lens, _native = _lens_fixture(MINI_CONDENSER)
    before = capture_instrument_magnetic_field(state)
    original = next(source for source in before._sources if source.key == MINI_CONDENSER)
    provider = original.provider
    files = _implementation_inputs(type(provider), type(provider.native_provider))
    assert "optics/lens_focal_length.py" in files
    lens.normalise_profile_peak = True
    normal = capture_instrument_magnetic_field(state)
    assert normal.numerical_identity != before.numerical_identity
    lens.gaussian = (replace(lens.gaussian[0], amplitude=3.), *lens.gaussian[1:])
    changed = capture_instrument_magnetic_field(state)
    assert changed.numerical_identity != normal.numerical_identity
