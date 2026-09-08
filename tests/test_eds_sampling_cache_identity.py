"""EDS-only overlap estimates invalidate spectra while retaining electron work."""
from types import SimpleNamespace
import tomllib

import pytest

import temsim.calculation_cache as cache
from temsim.optics.column import default_state
from temsim.assembly_catalog import AssemblyCatalog
from temsim.profile_io import apply_profile_values, read_profile, save_profile
from temsim.runtime_parameters import runtime_targets, validate_runtime_assignment
from temsim.specimen.interaction_engine import run_specimen_interactions
from temsim.specimen.interaction_types import (
    SpecimenInteractionRequest,
    SpecimenInteractionResult,
    SpecimenObservable,
)
from temsim.specimen.scene import SpecimenScene


EDS_PRODUCTS = {"request", "eds", "sample_region"}


def changed(before, after):
    return {name for name in before if before[name] != after[name]}


@pytest.mark.parametrize("wave_enabled", [False, True])
def test_eds_implementation_revision_keeps_electron_and_wave_caches(monkeypatch, wave_enabled):
    state = default_state()
    state.sample.stem_wave_enabled = wave_enabled
    current = cache.calculation_signatures(state)
    monkeypatch.setattr(cache, "_EDS_SIGNAL_SCHEMA", "older-eds-overlap-rule")
    previous = cache.calculation_signatures(state)
    assert changed(previous, current) == EDS_PRODUCTS
    assert {"elastic", "incident", "column", "wave", "wave_source",
            "scan_geometry", "stem", "stem_transport", "sample_downstream"} <= cache.matching_products(previous, current)


@pytest.mark.parametrize("field, before_value, after_value", [
    ("eds_overlap_sampling_enabled", True, False),
    ("eds_overlap_sampling_points", 256, 512),
])
@pytest.mark.parametrize("wave_enabled", [False, True])
def test_public_eds_sampling_parameters_only_change_spectrum_products(field, before_value, after_value, wave_enabled):
    state = default_state()
    state.sample.stem_wave_enabled = wave_enabled
    setattr(state.sample, field, before_value)
    previous = cache.calculation_signatures(state)
    setattr(state.sample, field, after_value)
    current = cache.calculation_signatures(state)
    assert changed(previous, current) == EDS_PRODUCTS


def test_eds_sampling_settings_survive_state_and_profile_serialization(tmp_path):
    state = default_state()
    assert state.sample.eds_overlap_sampling_enabled is True
    assert state.sample.eds_overlap_sampling_points == 256
    state.sample.eds_overlap_sampling_enabled = False
    state.sample.eds_overlap_sampling_points = 1024
    restored = type(state).from_dict(state.to_dict())
    assert restored.sample.eds_overlap_sampling_enabled is False
    assert restored.sample.eds_overlap_sampling_points == 1024
    path = tmp_path / "eds-settings.toml"
    save_profile(path, state, AssemblyCatalog().default_selection())
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    assert document["devices"]["sample"]["eds_overlap_sampling_enabled"] is False
    assert document["devices"]["sample"]["eds_overlap_sampling_points"] == 1024
    _, values = read_profile(path)
    loaded = default_state()
    skipped = apply_profile_values(loaded, values)
    assert not any("eds_overlap_sampling" in str(item) for item in skipped)
    assert loaded.sample.eds_overlap_sampling_enabled is False
    assert loaded.sample.eds_overlap_sampling_points == 1024


@pytest.mark.parametrize("value", [32, 256, 4096])
def test_overlap_sample_count_runtime_bounds_accept_endpoints(value):
    target = runtime_targets(default_state())["sample"]
    assert validate_runtime_assignment(target, "eds_overlap_sampling_points", value) == value


@pytest.mark.parametrize("value", [0, 31, 4097, True, 256.0, "256", float("nan"), float("inf"), None])
def test_overlap_sample_count_runtime_rejects_invalid_type_or_range(value):
    target = runtime_targets(default_state())["sample"]
    with pytest.raises(ValueError):
        validate_runtime_assignment(target, "eds_overlap_sampling_points", value)


@pytest.mark.parametrize("value", [True, False])
def test_overlap_enabled_runtime_requires_boolean(value):
    target = runtime_targets(default_state())["sample"]
    assert validate_runtime_assignment(target, "eds_overlap_sampling_enabled", value) is value
    with pytest.raises(ValueError, match="Boolean"):
        validate_runtime_assignment(target, "eds_overlap_sampling_enabled", int(value))


@pytest.mark.parametrize("request_eds", [False, True])
def test_mixed_specimen_result_drops_old_spectrum_but_reuses_particles_and_wave(monkeypatch, request_eds):
    import temsim.detector.eds_signal as eds_signal
    import temsim.specimen.elastic_transport as elastic_module

    state = default_state()
    simulation = SimpleNamespace(real_interactions=None)
    point = SpecimenInteractionRequest.eds_point(x_nm=0., y_nm=0.)
    transport = SimpleNamespace(
        eds_tracks=(), trajectories=(),
        metrics={"outcome_weight_fractions": {"transmitted": 1.}},
    )
    bundle = SimpleNamespace(rays=())
    wave = SimpleNamespace()
    old_spectrum = SimpleNamespace(elastic_transport=transport, metrics={}, lines=(), vacancies=())
    with monkeypatch.context() as old_version:
        old_version.setattr(cache, "_EDS_SIGNAL_SCHEMA", "older-eds-overlap-rule")
        old_signatures = cache.calculation_signatures(state)
    existing = SpecimenInteractionResult(
        request=point,
        completed_observables=frozenset({SpecimenObservable.COHERENT_ELASTIC_WAVE,
                                       SpecimenObservable.ELASTIC_TRANSPORT,
                                       SpecimenObservable.CHARACTERISTIC_X_RAY}),
        scene=SpecimenScene.from_state(state), incident_bundle=bundle,
        wave_imaging=wave, elastic_transport=transport, eds_spectrum=old_spectrum,
        metrics={"dependency_signatures": old_signatures},
    )
    calls = []
    replacement = SimpleNamespace(elastic_transport=transport, metrics={}, lines=(), vacancies=())

    def fake_eds(*_args, **kwargs):
        calls.append(kwargs)
        assert kwargs["elastic_transport"] is transport
        assert kwargs["incident_bundle"] is bundle
        return replacement

    monkeypatch.setattr(eds_signal, "simulate_eds_point", fake_eds)
    monkeypatch.setattr(elastic_module, "simulate_elastic_point_transport",
                        lambda *_args, **_kwargs: pytest.fail("Original electron transport must be retained"))
    request = point if request_eds else SpecimenInteractionRequest.elastic_point(x_nm=0., y_nm=0.)
    result = run_specimen_interactions(state, simulation, request,
                                      detector_geometry=object(), existing_result=existing)
    assert result.elastic_transport is transport
    assert result.incident_bundle is bundle
    assert result.wave_imaging is wave
    assert result.eds_spectrum is (replacement if request_eds else None)
    assert len(calls) == int(request_eds)
    assert SpecimenObservable.CHARACTERISTIC_X_RAY.value not in result.metrics["reused_observables"]
    if request_eds:
        assert result.metrics["calculated_observables_this_call"] == (SpecimenObservable.CHARACTERISTIC_X_RAY.value,)
    else:
        assert result.metrics["calculated_observables_this_call"] == ()
