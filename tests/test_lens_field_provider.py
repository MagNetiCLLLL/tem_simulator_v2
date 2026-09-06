from dataclasses import replace
import json

import numpy as np
import pytest

from temsim.calculation_cache import calculation_signatures
from temsim.optics.column import default_state
from temsim.optics.model import State
from temsim.physics.core import fields
from temsim.physics.lens_field_provider import (
    CoordinateRegistration,
    FieldMapProvenance,
    MagneticFieldMap,
    bind_imported_lens_field_map,
    calibrated_field_map,
    fit_axial_field_map_calibration,
    lens_geometry_binding,
    load_magnetic_field_map,
    resolve_runtime_lens_field_provider,
)
from temsim.specimen.axial_field_transport import (
    sample_axial_field_diagnostic,
)


def _objective(state):
    return next(lens for lens in state.lenses if lens.key == "objective_lens")


def _write_axisymmetric_map(path, state, lens, *, field_t=0.37):
    binding = lens_geometry_binding(state, lens.key, lens)
    geometry_source = getattr(lens, "lens", lens)
    z0_m = float(geometry_source.z_mm) * 1.0e-3
    r_m = np.asarray((0.0, 1.0e-3, 2.0e-3))
    z_m = z0_m + np.asarray((-2.0e-3, 0.0, 2.0e-3))
    br_t = np.zeros((r_m.size, z_m.size))
    bz_t = np.full_like(br_t, float(field_t))
    metadata = {
        "map_type": "axisymmetric_rz",
        "geometry_fingerprint": binding.geometry_fingerprint,
        "provenance_kind": "fem",
        "reference_excitation_percent": 100.0,
        "reference_polarity": 1,
    }
    np.savez(
        path,
        r_m=r_m,
        z_m=z_m,
        br_t=br_t,
        bz_t=bz_t,
        metadata_json=json.dumps(metadata),
    )
    return binding


def test_fields_reuses_provider_and_rejects_map_after_pole_geometry_change(
    tmp_path,
):
    state = default_state()
    lens = _objective(state)
    path = tmp_path / "objective_field.npz"
    binding = _write_axisymmetric_map(path, state, lens)
    field_map = load_magnetic_field_map(
        path,
        geometry_binding=binding,
        provenance_kind="fem",
        reference_excitation_percent=100.0,
        require_divergence=True,
    )
    bind_imported_lens_field_map(state, lens.key, field_map, native_provider=lens)

    first = resolve_runtime_lens_field_provider(state, lens.key, lens)
    second = resolve_runtime_lens_field_provider(state, lens.key, lens)
    assert first is second
    assert first.model_status == "measured_or_fem_geometry_bound"

    for other in state.lenses:
        other.enabled = other is lens
    expected = 0.37 * lens.percent / 100.0 * lens.polarity
    assert fields(np.asarray((lens.z_mm,)), state)[0][0] == pytest.approx(expected)

    assembly = state._resolved_assembly
    changed_parts = tuple(
        replace(
            part,
            data={**dict(part.data), "mechanical_tip_diameter_mm": 8.2},
        )
        if part.key == "objective_upper_pole"
        else part
        for part in assembly.parts
    )
    state._resolved_assembly = replace(assembly, parts=changed_parts)
    fallback = resolve_runtime_lens_field_provider(state, lens.key, lens)

    assert fallback is not first
    assert fallback.model_status == "provisional_geometry_bound_analytic_not_fem"
    assert fallback.fallback_reason == "imported_map_geometry_mismatch"
    diagnostic = state._field_provider_diagnostics[lens.key]
    assert diagnostic["mode"] == "provisional_analytic_fallback"
    assert "magnetostatic" not in diagnostic["model_status"]
    assert fallback.magnetic_field_t([lens.z_mm]) == pytest.approx(
        lens.magnetic_field_t([lens.z_mm])
    )


def test_objective_pole_profile_edit_stales_map_and_incident_signature(
    tmp_path,
):
    state = default_state()
    lens = _objective(state)
    path = tmp_path / "objective_profile_bound.npz"
    binding = _write_axisymmetric_map(path, state, lens)
    field_map = load_magnetic_field_map(
        path,
        geometry_binding=binding,
        provenance_kind="fem",
        reference_excitation_percent=100.0,
    )
    bind_imported_lens_field_map(state, lens.key, field_map, native_provider=lens)
    mapped = resolve_runtime_lens_field_provider(state, lens.key, lens)
    before = calculation_signatures(state)

    assembly = state._resolved_assembly
    changed_parts = tuple(
        replace(
            part,
            data={
                **dict(part.data),
                "mechanical_profile": "edited_objective_pole_profile",
            },
        )
        if part.key == "objective_upper_pole"
        else part
        for part in assembly.parts
    )
    state._resolved_assembly = replace(assembly, parts=changed_parts)

    fallback = resolve_runtime_lens_field_provider(state, lens.key, lens)
    after = calculation_signatures(state)

    assert mapped.model_status == "measured_or_fem_geometry_bound"
    assert fallback.model_status == "provisional_geometry_bound_analytic_not_fem"
    assert fallback.fallback_reason == "imported_map_geometry_mismatch"
    assert after["incident"] != before["incident"]


def test_projector_pole_edit_keeps_objective_map_and_incident_signature(
    tmp_path,
):
    state = default_state()
    lens = _objective(state)
    path = tmp_path / "objective_isolation.npz"
    binding = _write_axisymmetric_map(path, state, lens)
    field_map = load_magnetic_field_map(
        path,
        geometry_binding=binding,
        provenance_kind="fem",
        reference_excitation_percent=100.0,
    )
    bind_imported_lens_field_map(state, lens.key, field_map, native_provider=lens)
    mapped = resolve_runtime_lens_field_provider(state, lens.key, lens)
    before_binding = lens_geometry_binding(state, lens.key, lens)
    before = calculation_signatures(state)

    assembly = state._resolved_assembly
    changed_parts = tuple(
        replace(
            part,
            data={
                **dict(part.data),
                "pole_piece_geometry_style": "edited_projector_profile",
            },
        )
        if part.key == "projector_lens_1_upper_pole"
        else part
        for part in assembly.parts
    )
    state._resolved_assembly = replace(assembly, parts=changed_parts)

    still_mapped = resolve_runtime_lens_field_provider(state, lens.key, lens)
    after_binding = lens_geometry_binding(state, lens.key, lens)
    after = calculation_signatures(state)

    assert still_mapped is mapped
    assert still_mapped.model_status == "measured_or_fem_geometry_bound"
    assert after_binding.geometry_fingerprint == before_binding.geometry_fingerprint
    assert after["incident"] == before["incident"]


def test_shared_c1_c2_cartridge_stales_both_maps_but_projector_does_not(
    tmp_path,
):
    state = default_state()
    providers = {
        key: state.condenser_system[key]
        for key in ("condenser_lens_1", "condenser_lens_2")
    }
    mapped = {}
    bindings = {}
    for index, (key, provider) in enumerate(providers.items(), start=1):
        path = tmp_path / f"{key}_shared_cartridge_{index}.npz"
        bindings[key] = _write_axisymmetric_map(
            path,
            state,
            provider,
            field_t=0.1 * index,
        )
        field_map = load_magnetic_field_map(
            path,
            geometry_binding=bindings[key],
            provenance_kind="fem",
            reference_excitation_percent=100.0,
        )
        bind_imported_lens_field_map(
            state,
            key,
            field_map,
            native_provider=provider,
        )
        mapped[key] = resolve_runtime_lens_field_provider(
            state, key, provider
        )

    assembly = state._resolved_assembly
    downstream_parts = tuple(
        replace(
            part,
            data={
                **dict(part.data),
                "mechanical_outer_diameter_mm": (
                    float(dict(part.data)["mechanical_outer_diameter_mm"])
                    + 0.125
                ),
            },
        )
        if part.key == "projector_lens_1_upper_pole"
        else part
        for part in assembly.parts
    )
    state._resolved_assembly = replace(assembly, parts=downstream_parts)
    for key, provider in providers.items():
        still_mapped = resolve_runtime_lens_field_provider(
            state, key, provider
        )
        assert still_mapped is mapped[key]
        assert (
            lens_geometry_binding(state, key, provider).geometry_fingerprint
            == bindings[key].geometry_fingerprint
        )

    cartridge_parts = tuple(
        replace(
            part,
            data={
                **dict(part.data),
                "mechanical_outer_diameter_mm": (
                    float(dict(part.data)["mechanical_outer_diameter_mm"])
                    + 0.25
                ),
            },
        )
        if part.key == "c1_c2_pole_piece_cartridge"
        else part
        for part in state._resolved_assembly.parts
    )
    state._resolved_assembly = replace(
        state._resolved_assembly,
        parts=cartridge_parts,
    )
    for key, provider in providers.items():
        fallback = resolve_runtime_lens_field_provider(state, key, provider)
        assert fallback is not mapped[key]
        assert (
            fallback.model_status
            == "provisional_geometry_bound_analytic_not_fem"
        )
        assert fallback.fallback_reason == "imported_map_geometry_mismatch"
        assert (
            lens_geometry_binding(state, key, provider).geometry_fingerprint
            != bindings[key].geometry_fingerprint
        )


def test_field_map_descriptor_survives_state_worker_round_trip(tmp_path):
    state = default_state()
    lens = _objective(state)
    path = tmp_path / "worker_field.npz"
    binding = _write_axisymmetric_map(path, state, lens, field_t=0.22)
    field_map = load_magnetic_field_map(
        path,
        geometry_binding=binding,
        provenance_kind="fem",
        reference_excitation_percent=100.0,
    )
    bind_imported_lens_field_map(state, lens.key, field_map, native_provider=lens)

    worker_state = State.from_dict(state.to_dict())
    worker_lens = _objective(worker_state)
    provider = resolve_runtime_lens_field_provider(
        worker_state, worker_lens.key, worker_lens
    )

    assert provider.model_status == "measured_or_fem_geometry_bound"
    assert provider.field_map.provenance.source_sha256 == (
        field_map.provenance.source_sha256
    )
    assert worker_state.lens_field_map_descriptors[lens.key][
        "geometry_fingerprint"
    ] == binding.geometry_fingerprint


def test_specimen_diagnostic_uses_same_geometry_bound_map_as_column(tmp_path):
    state = default_state()
    lens = _objective(state)
    state.sample.z_mm = float(lens.z_mm)
    path = tmp_path / "sample_objective_field.npz"
    binding = _write_axisymmetric_map(path, state, lens, field_t=0.41)
    field_map = load_magnetic_field_map(
        path,
        geometry_binding=binding,
        provenance_kind="fem",
        reference_excitation_percent=100.0,
    )
    bind_imported_lens_field_map(
        state, lens.key, field_map, native_provider=lens
    )
    for candidate in state.lenses:
        candidate.enabled = candidate is lens

    column_value = float(fields((state.sample.z_mm,), state)[0][0])
    diagnostic = sample_axial_field_diagnostic(state)

    assert diagnostic.total_field_t == pytest.approx(column_value)
    assert diagnostic.objective_field_t == pytest.approx(column_value)
    assert dict(diagnostic.active_lens_contributions_t)[lens.key] == (
        pytest.approx(column_value)
    )
    assert diagnostic.geometry_material_coupled
    assert diagnostic.objective_field_model_status == (
        "measured_or_fem_geometry_bound"
    )
    assert str(path) in diagnostic.objective_field_source


def test_axial_measured_points_fit_scale_shift_and_report_residuals():
    z_axis = np.linspace(-4.0e-3, 4.0e-3, 81)
    radius = np.linspace(0.0, 1.0e-3, 5)
    sigma = 1.4e-3
    bz_axis = np.exp(-0.5 * (z_axis / sigma) ** 2)
    derivative = -z_axis / sigma**2 * bz_axis
    br = -0.5 * radius[:, None] * derivative[None, :]
    field_map = MagneticFieldMap(
        map_type="axisymmetric_rz",
        axes_m=(radius, z_axis),
        components_t=(br, np.broadcast_to(bz_axis, br.shape)),
        registration=CoordinateRegistration(),
        geometry_fingerprint="geometry-test",
        reference_excitation_percent=80.0,
        reference_polarity=1,
        provenance=FieldMapProvenance(
            kind="measured",
            source_path="measured-bz.csv",
            source_sha256="0" * 64,
            source_note="test instrument export",
        ),
    )
    measured_z = np.linspace(-2.0e-3, 2.0e-3, 31)
    known_shift = 0.31e-3
    positions = np.zeros((measured_z.size, 3))
    positions[:, 2] = measured_z - known_shift
    measured_bz = 1.25 * field_map.field_at_global_positions_t(positions)[:, 2]

    fit = fit_axial_field_map_calibration(
        field_map,
        measured_z,
        measured_bz,
        maximum_axial_shift_m=0.8e-3,
        measurement_reference="calibrated Hall-probe axial scan; test fixture",
    )
    calibrated = calibrated_field_map(field_map, fit)

    assert fit.axial_shift_m == pytest.approx(known_shift, abs=2.0e-7)
    assert fit.field_scale == pytest.approx(1.25, rel=2.0e-4)
    assert fit.relative_rms_residual < 2.0e-4
    assert fit.measurement_fingerprint
    assert calibrated.registration.origin_global_m[2] == pytest.approx(
        known_shift, abs=2.0e-7
    )
    assert calibrated.reference_excitation_percent == pytest.approx(64.0)
