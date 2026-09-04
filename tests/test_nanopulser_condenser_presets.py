from types import SimpleNamespace

import pytest

from temsim.operating_modes import load_operating_mode_catalog, mode_by_key
from temsim.optics import condenser_recalibration as recalibration
from temsim.optics.direct_alignment import DirectAlignmentResult


def _state():
    lenses = [SimpleNamespace(key=key, percent=value) for key, value in (
        ("condenser_lens_2", 25.0), ("condenser_lens_3", 21.0),
    )]
    state = SimpleNamespace(
        beam_blanked=True,
        nanopulser=SimpleNamespace(blanked=True),
        electron_gun=SimpleNamespace(
            emitter=SimpleNamespace(ray_count=15000),
            deflector=SimpleNamespace(
                upper_field_x_mt=0.12, upper_field_y_mt=-0.23,
                lower_field_x_mt=-0.34, lower_field_y_mt=0.45,
            ),
        ),
        lenses=lenses,
        _resolved_assembly=SimpleNamespace(parts=[SimpleNamespace(
            key="condenser_lens_1", start_z_mm=530.0, center_z_mm=580.0,
            end_z_mm=630.0, data={"field_half_width_mm": 4.0},
        )]),
    )
    state.to_dict = lambda: {
        "beam_voltage_kv": 300.0,
        "beam_blanked": state.beam_blanked,
        "nanopulser": {"blanked": state.nanopulser.blanked},
        "gun_alignment": vars(state.electron_gun.deflector).copy(),
    }
    return state


def _result(**changes):
    values = dict(
        key="nanoprobe_convergence", success=True, requested=25.0,
        achieved=24.9, unit="mrad", constraint_value=0.0000001,
        constraint_unit="mm",
        strengths={"condenser_lens_2": 24.2, "condenser_lens_3": 21.2},
        iterations=4, validation_step_mm=0.05, numerical_spread=0.0001,
        message="validated", convergence_95_mrad=24.9,
        illumination_diameter_95_um=0.001,
    )
    values.update(changes)
    return DirectAlignmentResult(**values)


@pytest.fixture(autouse=True)
def clear_calibration_cache(monkeypatch):
    import temsim.optics.direct_alignment as alignment

    monkeypatch.setattr(alignment, "_refine_nanoprobe_production_focus",
                        lambda _state, _definition, _target, vector, _step:
                        (vector, 0))
    recalibration._CACHE.clear()
    yield
    recalibration._CACHE.clear()


def test_preset_uses_transmitted_beam_and_replaces_old_geometry_metrics(monkeypatch):
    import temsim.optics.direct_alignment as alignment

    state = _state()
    calls = []

    def solve(candidate, key, target, *, definition):
        assert not candidate.beam_blanked
        assert not candidate.nanopulser.blanked
        assert candidate.electron_gun.emitter.ray_count == 512
        calls.append((key, target, definition.targets["validation_step_mm"]))
        return _result()

    monkeypatch.setattr(alignment, "apply_direct_alignment", solve)
    mode = mode_by_key("nano_probe")
    result = recalibration.recalibrate_nanopulser_condenser(
        state, mode, load_operating_mode_catalog(),
    )
    assert calls == [("nanoprobe_convergence", 25.0, 0.05)]
    assert state.beam_blanked
    assert state.nanopulser.blanked
    assert state.electron_gun.emitter.ray_count == 15000
    assert result.devices["condenser_lens_2"]["percent"] == 24.2
    assert result.targets["achieved_convergence_sem_angle_mrad"] == 24.9
    assert result.targets["achieved_waist_offset_nm"] == pytest.approx(0.1)
    assert "achieved_geometric_rms_radius_nm" not in result.targets
    assert "step_waist_difference_nm" not in result.targets
    assert mode.targets["achieved_convergence_sem_angle_mrad"] != 24.9


def test_exposure_gate_toggles_reuse_calibration_and_preserve_gun_alignment(monkeypatch):
    import temsim.optics.direct_alignment as alignment

    state = _state()
    gun_alignment = vars(state.electron_gun.deflector).copy()
    calls = []

    def solve(candidate, *_args, **_kwargs):
        assert not candidate.beam_blanked
        assert not candidate.nanopulser.blanked
        assert vars(candidate.electron_gun.deflector) == gun_alignment
        calls.append(True)
        return _result()

    monkeypatch.setattr(alignment, "apply_direct_alignment", solve)
    mode, catalog = mode_by_key("nano_probe"), load_operating_mode_catalog()
    first = recalibration.recalibrate_nanopulser_condenser(state, mode, catalog)
    for ordinary, nano in ((False, True), (True, False), (False, False)):
        state.beam_blanked, state.nanopulser.blanked = ordinary, nano
        result = recalibration.recalibrate_nanopulser_condenser(state, mode, catalog)
        assert result is first
        assert (state.beam_blanked, state.nanopulser.blanked) == (ordinary, nano)
        assert vars(state.electron_gun.deflector) == gun_alignment
    assert len(calls) == 1

    state.electron_gun.deflector.upper_field_x_mt += 0.1
    gun_alignment = vars(state.electron_gun.deflector).copy()
    recalibration.recalibrate_nanopulser_condenser(state, mode, catalog)
    assert len(calls) == 2


def test_identical_geometry_reuses_verified_preset_but_length_and_field_edits_recompute(monkeypatch):
    import temsim.optics.direct_alignment as alignment

    state = _state()
    calls = []
    monkeypatch.setattr(alignment, "apply_direct_alignment", lambda *_a, **_kw:
                        (calls.append(True) or _result()))
    mode, catalog = mode_by_key("nano_probe"), load_operating_mode_catalog()
    first = recalibration.recalibrate_nanopulser_condenser(state, mode, catalog)
    second = recalibration.recalibrate_nanopulser_condenser(state, mode, catalog)
    assert len(calls) == 1
    assert second is first
    assert state.lenses[0].percent == 24.2

    state._resolved_assembly.parts[0].center_z_mm += 10.0
    recalibration.recalibrate_nanopulser_condenser(state, mode, catalog)
    state._resolved_assembly.parts[0].data["field_half_width_mm"] = 4.5
    recalibration.recalibrate_nanopulser_condenser(state, mode, catalog)
    assert len(calls) == 3


def test_unreachable_preset_restores_blank_and_sampling_without_caching_failure(monkeypatch):
    import temsim.optics.direct_alignment as alignment

    state = _state()
    monkeypatch.setattr(alignment, "apply_direct_alignment", lambda *_a, **_kw:
                        _result(success=False, message="no surviving pupil"))
    with pytest.raises(ValueError, match="no surviving pupil"):
        recalibration.recalibrate_nanopulser_condenser(
            state, mode_by_key("nano_probe"), load_operating_mode_catalog(),
        )
    assert state.beam_blanked
    assert state.nanopulser.blanked
    assert state.electron_gun.emitter.ray_count == 15000
    assert not recalibration._CACHE


def test_unexpected_calibration_failure_restores_both_gates_and_alignment(monkeypatch):
    import temsim.optics.direct_alignment as alignment

    state = _state()
    before = state.to_dict()

    def fail(candidate, *_args, **_kwargs):
        assert not candidate.beam_blanked
        assert not candidate.nanopulser.blanked
        raise RuntimeError("source integration failed")

    monkeypatch.setattr(alignment, "apply_direct_alignment", fail)
    with pytest.raises(RuntimeError, match="source integration failed"):
        recalibration.recalibrate_nanopulser_condenser(
            state, mode_by_key("micro_probe"), load_operating_mode_catalog(),
        )
    assert state.to_dict() == before
    assert state.electron_gun.emitter.ray_count == 15000
    assert not recalibration._CACHE


def test_real_gun_blank_property_and_independent_nanopulser_gate_are_preserved(monkeypatch):
    import temsim.optics.direct_alignment as alignment
    from temsim.optics.column import default_state

    state = default_state()
    state.beam_blanked = True
    state.nanopulser.installed = True
    state.nanopulser.blanked = True
    deflector = state.electron_gun.deflector
    fields = ("upper_field_x_mt", "upper_field_y_mt",
              "lower_field_x_mt", "lower_field_y_mt")
    for field, value in zip(fields, (0.12, -0.23, -0.34, 0.45)):
        setattr(deflector, field, value)
    before_alignment = tuple(getattr(deflector, field) for field in fields)

    def solve(candidate, *_args, **_kwargs):
        assert not candidate.beam_blanked
        assert not candidate.electron_gun.deflector.beam_blanked
        assert not candidate.nanopulser.blanked
        assert tuple(getattr(deflector, field) for field in fields) == before_alignment
        return _result(key="microprobe_illumination")

    monkeypatch.setattr(alignment, "apply_direct_alignment", solve)
    recalibration.recalibrate_nanopulser_condenser(
        state, mode_by_key("micro_probe"), load_operating_mode_catalog(),
    )
    assert state.beam_blanked and deflector.beam_blanked
    assert state.nanopulser.blanked and state.nanopulser.installed
    assert tuple(getattr(deflector, field) for field in fields) == before_alignment


@pytest.mark.parametrize("mode,target", (
    ("nanoprobe_convergence", 25.0), ("microprobe_illumination", 2.0),
))
@pytest.mark.parametrize("outcome", ("success", "unreachable", "error"))
def test_general_condenser_alignment_opens_only_the_calculation_gates(
    monkeypatch, mode, target, outcome,
):
    import temsim.optics.direct_alignment as alignment

    state = _state()
    before = state.to_dict()

    def solve(candidate, *_args, **_kwargs):
        assert not candidate.beam_blanked
        assert not candidate.nanopulser.blanked
        assert vars(candidate.electron_gun.deflector) == before["gun_alignment"]
        if outcome == "error":
            raise RuntimeError("integration failed")
        return _result(success=outcome == "success")

    monkeypatch.setattr(alignment, "_solve_direct_alignment", solve)
    if outcome == "error":
        with pytest.raises(RuntimeError, match="integration failed"):
            alignment.apply_direct_alignment(state, mode, target)
    else:
        result = alignment.apply_direct_alignment(state, mode, target)
        assert result.success is (outcome == "success")
    assert state.to_dict() == before


def test_microprobe_metadata_reports_fresh_parallel_illumination(monkeypatch):
    import temsim.optics.direct_alignment as alignment

    monkeypatch.setattr(alignment, "apply_direct_alignment", lambda *_a, **_kw:
                        _result(key="microprobe_illumination", achieved=2.0,
                                convergence_95_mrad=0.25,
                                illumination_diameter_95_um=2.0,
                                constraint_value=0.01))
    result = recalibration.recalibrate_nanopulser_condenser(
        _state(), mode_by_key("micro_probe"), load_operating_mode_catalog(),
    )
    assert result.targets["achieved_wavefront_curvature_per_m"] == 0.01
    assert result.targets["achieved_illumination_diameter_95_um"] == 2.0
    assert result.calibration_status.startswith("computed_live_nanopulser")


def test_installed_pair_failure_restores_lenses_apertures_modes_and_detectors(monkeypatch):
    from dataclasses import replace
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.operating_modes import apply_operating_mode_pair
    from temsim.optics.column import default_state

    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, replace(catalog.default_selection(), beam_blanker="NanoPulser"))
    state.nanopulser.blanked = True
    state.electron_gun.electrostatic_lens.voltage_kv = 1.1
    before = state.to_dict()

    def fail(*_args):
        raise ValueError("new geometry does not meet the focus constraint")

    monkeypatch.setattr(recalibration, "recalibrate_nanopulser_condenser", fail)
    with pytest.raises(ValueError, match="new geometry"):
        apply_operating_mode_pair(state, "micro_probe", "imaging")
    assert state.to_dict() == before


def test_uninstalled_instrument_retains_its_existing_preset_path(monkeypatch):
    from temsim.operating_modes import apply_operating_mode_pair
    from temsim.optics.column import default_state

    state = default_state()

    def unexpected_solve(*_args):
        pytest.fail("Uninstalled NanoPulser must not trigger a new condenser solve")

    monkeypatch.setattr(recalibration, "recalibrate_nanopulser_condenser", unexpected_solve)
    result = apply_operating_mode_pair(state, "nano_probe", "diffraction")
    assert result.condenser is mode_by_key("nano_probe")


def test_condenser_candidate_search_respects_the_installed_blanking_stop():
    from dataclasses import replace
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.column import default_state
    from temsim.optics.direct_alignment import _CondenserMeasurementModel

    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, replace(catalog.default_selection(), beam_blanker="NanoPulser"))
    state.electron_gun.emitter.ray_count = 64
    state.nanopulser.aperture_radius_mm = 1.0e-12
    model = _CondenserMeasurementModel(state, step_mm=0.1)
    vector = [next(lens.percent for lens in state.lenses if lens.key == key)
              for key in ("condenser_lens_2", "condenser_lens_3")]
    with pytest.raises(ValueError, match="No finite surviving rays"):
        model.measure(vector)
