"""A periodic reference crystal stays in sample coordinates as its ROI moves."""

import numpy as np
import pytest

from temsim.specimen.atomistic import build_equilibrium_atoms
from temsim.specimen.presets import load_specimen_preset


def _reference(**kwargs):
    pytest.importorskip("abtem")
    return build_equilibrium_atoms(
        load_specimen_preset("si_110"),
        thickness_angstrom=8.0,
        field_of_view_angstrom=16.0,
        **kwargs,
    )


@pytest.mark.parametrize("sample_centre,roi_centre", [
    ((0.0, 0.0), (1.7, -2.3)),
    ((8.1, -10.2), (11.6, -16.5)),
    ((-27.4, 43.6), (145.7, -218.9)),
])
def test_reference_lattice_fixed_in_lab_modulo_its_periodic_cell(sample_centre, roi_centre):
    baseline, periods = _reference()
    shifted, shifted_periods = _reference(
        specimen_centre_xy_angstrom=sample_centre,
        calculation_roi_centre_xy_angstrom=roi_centre,
    )
    np.testing.assert_array_equal(shifted.numbers, baseline.numbers)
    np.testing.assert_array_equal(shifted.cell.array, baseline.cell.array)
    assert shifted_periods == periods
    # Wrapping may choose another periodic image of a given atom. Its lab
    # position relative to the specimen must nevertheless stay unchanged.
    displacement = shifted.positions[:, :2] + roi_centre - sample_centre - baseline.positions[:, :2]
    periods_xy = shifted.cell.lengths()[:2]
    residual = displacement - np.rint(displacement / periods_xy) * periods_xy
    np.testing.assert_allclose(residual, 0.0, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(shifted.positions[:, 2], baseline.positions[:, 2])


def test_translating_sample_and_roi_together_does_not_move_reference_atoms():
    baseline, _ = _reference()
    translated, _ = _reference(
        specimen_centre_xy_angstrom=(200.0, -800.0),
        calculation_roi_centre_xy_angstrom=(200.0, -800.0),
    )
    np.testing.assert_array_equal(translated.positions, baseline.positions)


@pytest.mark.parametrize("options,match", [
    ({"specimen_centre_xy_angstrom": (np.nan, 0)}, "centres"),
    ({"calculation_roi_centre_xy_angstrom": (0, np.inf)}, "centres"),
    ({"calculation_roi_centre_xy_angstrom": (1, 2, 3)}, "centres"),
    ({"specimen_size_xy_angstrom": (0, 10)}, "X/Y size"),
    ({"specimen_size_xy_angstrom": (10, np.inf)}, "X/Y size"),
    ({"specimen_centre_xy_angstrom": (-1e308, 0),
      "calculation_roi_centre_xy_angstrom": (1e308, 0)}, "offset"),
])
def test_reference_roi_inputs_are_validated(options, match):
    with pytest.raises(ValueError, match=match):
        _reference(**options)
