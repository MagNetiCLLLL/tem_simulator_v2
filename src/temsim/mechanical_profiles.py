"""Canonical names for concentric electron-optical mechanical layers."""

MAGNETIC_LENS_ASSEMBLY = "magnetic_lens_assembly"
MAGNETIC_LENS_HOUSING = "magnetic_lens_housing"
MAGNETIC_LENS_YOKE = "magnetic_lens_yoke"
MAGNETIC_EXCITATION_COIL = "magnetic_excitation_coil"
MAGNETIC_POLE_PIECE = "magnetic_pole_piece"
ELECTROSTATIC_ELECTRODE_STACK = "electrostatic_electrode_stack"
VACUUM_LINER = "vacuum_liner"
VACUUM_BORE = "vacuum_bore"

MAGNETIC_LENS_MECHANICAL_PROFILES = frozenset({
    MAGNETIC_LENS_HOUSING,
    MAGNETIC_LENS_YOKE,
    MAGNETIC_EXCITATION_COIL,
})


def pole_piece_keys(lens_key: str) -> tuple[str, str]:
    """Return the canonical upstream/downstream pole-piece part keys."""

    key = str(lens_key)
    return f"{key}_upper_pole", f"{key}_lower_pole"


def lens_mechanical_part_keys(lens_key: str) -> tuple[str, str, str]:
    """Return the independent housing, yoke and coil keys for a lens."""

    key = str(lens_key)
    return (
        f"{key}_housing",
        f"{key}_yoke",
        f"{key}_excitation_coil",
    )
