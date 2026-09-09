"""Read atomic structures from CIF and MCIF through the same CIF parser."""

from __future__ import annotations


def read_cif_atoms(path, *, index=-1):
    """Keep ASE's block selection while accepting the MCIF filename suffix.

    ASE does not register ``mcif`` as a format name. Its CIF reader supplies
    atomic positions and cell data; magnetic-moment scattering is not modelled.
    """
    from ase.io import read

    return read(path, index=index, format="cif")
