"""Explicit imported structures for tests; never called by the application."""
from pathlib import Path

CIF_DIRECTORY = Path(__file__).parent / "fixtures" / "cif"
SI_CIF = CIF_DIRECTORY / "Si.cif"
AU_CIF = CIF_DIRECTORY / "Au.cif"


def imported_sample(target, *, path=SI_CIF, zone=(1, 1, 0), in_plane=(1, -1, 0)):
    from temsim.specimen.cif_io import read_cif_atoms
    from temsim.specimen.geometry import quaternion_from_zone_axes, set_sample_orientation
    sample = getattr(target, "sample", target)
    sample.specimen_mode = "atomic"
    sample.inserted = True
    sample.cif_path = str(path)
    sample.zone_axis_uvw, sample.in_plane_axis_uvw = zone, in_plane
    atoms = read_cif_atoms(path)
    set_sample_orientation(sample, quaternion_from_zone_axes(atoms.cell.array, zone, in_plane))
    return sample
