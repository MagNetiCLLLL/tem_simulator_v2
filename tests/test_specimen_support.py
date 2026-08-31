import math
import tomllib
from pathlib import Path

import pytest

from temsim.specimen.support import (
    available_support_materials,
    available_support_meshes,
    load_support_catalog,
    resolve_support_grid,
)


PROJECT_ROOT = Path(__file__).parents[1]


def test_commercial_support_catalog_is_packaged_and_evidence_bounded():
    catalog = load_support_catalog()

    assert catalog.outer_diameter_mm == pytest.approx(3.05)
    assert catalog.foil_thickness_um == pytest.approx(25.0)
    assert catalog.rim_width_um == pytest.approx(375.0)
    assert catalog.geometry_status == "commercial_nominal"
    assert catalog.geometry_source_url.startswith("https://")
    assert dict(available_support_materials()) == {
        "vacuum": "Vacuum (virtual support)",
        "copper": "Copper",
        "gold": "Gold",
    }
    assert len(available_support_meshes()) == 10

    pyproject = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert pyproject["tool"]["setuptools"]["data-files"][
        "configs/specimen_supports"
    ] == ["configs/specimen_supports/*.toml"]


def test_square_mesh_classifies_opening_bar_rim_and_outside():
    grid = resolve_support_grid("copper", "square_200")

    assert grid.region_at_nm(0.0, 0.0) == "opening"
    assert grid.region_at_nm(60_000.0, 0.0) == "bar"
    assert grid.region_at_nm(1_300_000.0, 0.0) == "rim"
    assert grid.region_at_nm(1_600_000.0, 0.0) == "outside"
    assert grid.material_path_length_nm(0.0, 0.0) == 0.0
    assert grid.material_path_length_nm(60_000.0, 0.0) == pytest.approx(
        25_000.0
    )


def test_grid_offset_is_hole_centre_and_rotation_is_continuous():
    grid = resolve_support_grid("gold", "square_200")

    assert grid.region_at_nm(
        60_000.0, 0.0, offset_x_um=60.0
    ) == "opening"
    assert grid.region_at_nm(
        60_000.0, 0.0, rotation_deg=90.0
    ) == "bar"
    assert grid.region_at_nm(
        40_000.0, 40_000.0, rotation_deg=45.0
    ) == "bar"


def test_catalog_open_area_is_consistent_with_rounded_dimensions():
    catalog = load_support_catalog()

    for mesh in catalog.meshes.values():
        calculated = 100.0 * (mesh.hole_width_um / mesh.pitch_um) ** 2
        assert math.isclose(
            # The commercial table independently rounds pitch, hole, bar and
            # open area; the 150-mesh row is the limiting discrepancy.
            calculated, mesh.open_area_percent, abs_tol=6.0
        )


def test_virtual_vacuum_support_never_adds_a_material_path():
    grid = resolve_support_grid("vacuum", "square_500")

    assert grid.region_at_nm(0.0, 0.0) == "vacuum"
    assert grid.region_at_nm(1.0e9, -1.0e9) == "vacuum"
    assert grid.material_path_length_nm(0.0, 0.0) == 0.0
