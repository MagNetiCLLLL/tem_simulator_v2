import numpy as np
import pytest
from types import SimpleNamespace

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.diagnostics import (
    lens_field_records,
    physical_layout_records,
    ray_stop_records,
)
from temsim.optics.column import default_state
from temsim.physics.simulation import run
from temsim.simulation_pipeline import CalculationResult, aperture_stop_records


def _preview_result():
    state = default_state()
    catalog = AssemblyCatalog()
    assembly = catalog.apply(state, catalog.default_selection())
    state.electron_gun.emitter.ray_count = 25
    state.sample.diffraction_enabled = False
    layout = apply_physical_layout_to_state(state)
    simulation = run(state, resolved_layout=layout)
    return state, CalculationResult(
        simulation=simulation,
        energy_filter=None,
        state_snapshot=state,
        layout=layout,
        assembly=assembly,
        aperture_stops=aperture_stop_records(state),
    )


def test_physical_layout_records_use_resolved_geometry_and_optical_references():
    _state, result = _preview_result()

    records = physical_layout_records(result)
    objective = next(item for item in records if item.key == "objective_lens")
    projector_poles = [
        item for item in records
        if item.key.startswith((
            "diffraction_lens_",
            "intermediate_lens_",
            "projector_lens_",
        )) and item.key.endswith("_pole")
    ]

    assert len(records) == len(result.assembly.parts)
    assert objective.start_z_mm < objective.center_z_mm < objective.end_z_mm
    assert objective.outer_diameter_mm > objective.bore_diameter_mm > 0.0
    assert len(objective.optical_references_mm) >= 3
    assert len(projector_poles) == 8
    assert all(
        item.profile == "magnetic_pole_piece"
        for item in projector_poles
    )
    magnetic_lenses = [
        item for item in records
        if item.profile == "magnetic_lens_assembly"
    ]
    assert magnetic_lenses
    mechanical_layers = [
        item for item in records
        if item.profile in {
            "magnetic_lens_housing",
            "magnetic_lens_yoke",
            "magnetic_excitation_coil",
        }
    ]
    assert len(mechanical_layers) == 3 * len(magnetic_lenses)
    assert {
        item.profile for item in mechanical_layers
    } == {
        "magnetic_lens_housing",
        "magnetic_lens_yoke",
        "magnetic_excitation_coil",
    }
    assert result.assembly.vacuum_liner_segments
    assert all(
        segment.outer_diameter_mm > segment.inner_diameter_mm
        for segment in result.assembly.vacuum_liner_segments
    )


def test_individual_lens_fields_sum_to_solver_total():
    state, _result = _preview_result()
    z_mm = np.linspace(450.0, 2300.0, 500)

    total, records = lens_field_records(state, z_mm)

    assert records
    assert np.allclose(
        total,
        np.sum([record.field_t for record in records], axis=0),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    assert all(record.support_mm[0] <= record.support_mm[1] for record in records)
    assert any(record.peak_t > 0.0 for record in records)
    objective = next(record for record in records if record.key == "objective_lens")
    c1 = next(record for record in records if record.key == "condenser_lens_1")
    c2 = next(record for record in records if record.key == "condenser_lens_2")
    assert c1.formula_key == "peak_normalised_three_gaussian"
    assert c2.formula_key == "three_gaussian"
    assert objective.formula_key == "dual_pole_gaussian"
    assert len({c1.formula_colour, c2.formula_colour, objective.formula_colour}) == 3
    assert all(record.formula_expression for record in records)
    assert objective.spherical_aberration_mm == pytest.approx(1.2)
    assert objective.signed_field_integral_t_m > 0.0
    assert objective.larmor_rotation_deg > 0.0


def test_stop_diagnostics_report_exact_xy_radius_and_can_be_limited():
    _state, result = _preview_result()

    records = ray_stop_records(result.simulation)
    limited = ray_stop_records(result.simulation, maximum_records=6)

    assert records
    assert len(limited) <= 6
    for record in records:
        assert record.radial_mm == pytest.approx(
            np.hypot(record.x_mm, record.y_mm)
        )
        assert record.key
        assert np.isfinite(record.z_mm)


def test_all_catalog_assemblies_produce_layout_and_field_diagnostics():
    catalog = AssemblyCatalog()
    checked = 0
    for gun in catalog.guns:
        for column in catalog.columns:
            for recording in catalog.recording_systems:
                state = default_state()
                assembly = catalog.apply(state, AssemblySelection(
                    gun=gun.name,
                    column=column.name,
                    recording=recording.name,
                ))
                layout = apply_physical_layout_to_state(state)
                result = SimpleNamespace(assembly=assembly, layout=layout)
                records = physical_layout_records(result)
                child_keys = {
                    component.key: tuple(
                        part.key for part in assembly.parts
                        if part.parent_key == component.key
                        and part.key.endswith("_pole")
                    )
                    for component in layout
                }
                for component in layout:
                    shape = getattr(component, "mechanical_shape", None)
                    if getattr(shape, "profile", None) != "magnetic_lens_yoke":
                        continue
                    expected_poles = (
                        1
                        if component.key in {
                            "condenser_lens_1",
                            "condenser_lens_2",
                        }
                        else 2
                    )
                    assert len(child_keys[component.key]) == expected_poles
                total, lenses = lens_field_records(
                    state,
                    np.linspace(
                        0.0,
                        max(part.end_z_mm for part in assembly.parts),
                        24,
                    ),
                )
                assert len(records) == len(assembly.parts)
                assert total.shape == (24,)
                assert len(lenses) == len(state.lenses)
                checked += 1
    assert checked == 30
