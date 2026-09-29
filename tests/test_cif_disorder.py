"""CIF disorder must be validated before constructing atomistic potentials."""
from specimen_inputs import imported_sample, SI_CIF, AU_CIF
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from temsim.specimen.inelastic import real_inelastic_distribution
from temsim.specimen.rutherford import read_cif_composition












def _occupancy_cif(tmp_path, rows):
    path = tmp_path / "disorder.cif"
    path.write_text("""data_disorder
_cell_length_a 5
_cell_length_b 5
_cell_length_c 5
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_Int_Tables_number 1
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
""" + rows + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("rows", ["Si1 Si 0 0 0 .5", "Si1 Si 0 0 0 .5\nGe1 Ge 0 0 0 .5"])
def test_atomistic_potential_rejects_partial_or_mixed_sites_before_orthogonalizing(monkeypatch, tmp_path, rows):
    from temsim.specimen import atomistic

    path = _occupancy_cif(tmp_path, rows)
    composition = read_cif_composition(path)
    assert composition.partial_occupancy or composition.mixed_occupancy
    def no_potential(*_args, **_kwargs):
        raise AssertionError("Disorder must fail before constructing an atomistic potential")
    monkeypatch.setattr(atomistic, "_require_backend", lambda: (SimpleNamespace(orthogonalize_cell=no_potential), Atoms))
    with pytest.raises(ValueError, match="explicitly occupied/disordered supercell"):
        atomistic.build_cif_equilibrium_atoms(path, thickness_angstrom=50, field_of_view_angstrom=100)




