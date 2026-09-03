from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.specimen.elastic_transport import (
    ElasticScatterEvent,
    IncidentElectronRay as ElasticIncidentElectronRay,
)
from temsim.specimen.interaction_engine import run_specimen_interactions
from temsim.specimen.interaction_types import (
    ConservationCheck,
    IncidentElectronRay,
    IncidentRayBundle,
    InteractionProcess,
    SpecimenInteractionRequest,
    SpecimenInteractionResult,
    SpecimenModelCoupling,
    SpecimenObservable,
)


def test_elastic_transport_reexports_the_canonical_incident_contract():
    assert ElasticIncidentElectronRay is IncidentElectronRay


def test_incident_bundle_validates_counts_units_and_energy_order():
    ray = IncidentElectronRay(
        source_ray_index=3,
        position_xy_nm=(1.0, -2.0),
        direction=(0.0, 0.0, 1.0),
        kinetic_energy_ev=300_000.0,
        weight=1.0,
    )
    bundle = IncidentRayBundle(
        rays=(ray,),
        emitted_ray_count=4,
        reaching_ray_count=1,
        surviving_fraction=0.25,
        original_centroid_nm=(1.0, -2.0),
        target_centroid_nm=(1.0, -2.0),
        chief_angle_mrad=(0.0, 0.0),
        energy_range_ev=(300_000.0, 300_000.0),
        boundary_z_mm=1.5,
    )

    assert bundle.rays == (ray,)
    with pytest.raises(ValueError, match="reaching-ray count"):
        IncidentRayBundle(
            rays=(ray,),
            emitted_ray_count=4,
            reaching_ray_count=0,
            surviving_fraction=0.25,
            original_centroid_nm=(1.0, -2.0),
            target_centroid_nm=(1.0, -2.0),
            chief_angle_mrad=(0.0, 0.0),
            energy_range_ev=(300_000.0, 300_000.0),
        )


def test_conservation_check_reports_a_physical_residual():
    check = ConservationCheck(
        name="test_probability",
        quantity="electron probability",
        unit="1",
        input_value=1.0,
        output_channels=(("forward", 0.8), ("backward", 0.2)),
    )

    assert check.output_value == pytest.approx(1.0)
    assert check.residual == pytest.approx(0.0)
    assert check.conserved

    failed = ConservationCheck(
        name="test_probability",
        quantity="electron probability",
        unit="1",
        input_value=1.0,
        output_channels=(("forward", 0.8), ("backward", 0.1)),
    )
    assert failed.residual == pytest.approx(-0.1)
    with pytest.raises(
        RuntimeError, match="failed electron probability conservation"
    ):
        failed.require_conserved()


def test_engine_attaches_existing_inelastic_budget_without_recalculation():
    distribution = SimpleNamespace(
        channels=(
            SimpleNamespace(
                key="real_zero_loss",
                probability=0.7,
                energy_loss_ev=0.0,
            ),
            SimpleNamespace(
                key="real_plasmon",
                probability=0.2,
                energy_loss_ev=15.0,
            ),
        ),
        absorbed_probability=0.1,
        beam_energy_kev=200.0,
    )
    simulation = SimpleNamespace(real_interactions=distribution)

    result = run_specimen_interactions(
        SimpleNamespace(),
        simulation,
        SpecimenInteractionRequest(),
    )

    assert result.inelastic_distribution is distribution
    assert result.completed_observables == frozenset(
        (SpecimenObservable.STOCHASTIC_INELASTIC,)
    )
    assert {check.name for check in result.conservation} == {
        "specimen_inelastic_probability",
        "specimen_inelastic_mean_energy_partition",
    }
    assert all(check.conserved for check in result.conservation)
    assert result.metrics["exclusive_electron_budget_owner_count"] == 1
    assert result.couplings[0].contributes_to_exclusive_electron_budget
    assert result.metrics["per_event_ledger_populated"] is False


def test_engine_real_inelastic_distribution_conserves_probability_and_energy():
    result = run_specimen_interactions(
        default_state(),
        SimpleNamespace(real_interactions=None),
        SpecimenInteractionRequest(
            observables=frozenset(
                (SpecimenObservable.STOCHASTIC_INELASTIC,)
            )
        ),
    )

    checks = {check.name: check for check in result.conservation}
    assert checks["specimen_inelastic_probability"].conserved
    assert checks["specimen_inelastic_mean_energy_partition"].conserved
    assert checks[
        "specimen_inelastic_mean_energy_partition"
    ].input_value == pytest.approx(300_000.0)
    assert result.scene is not None
    assert result.scene.source_key == "preset:si_110"


def test_engine_runs_explicit_eds_and_exposes_elastic_conservation(monkeypatch):
    event = ElasticScatterEvent(
        position_nm=(1.0, 2.0, 3.0),
        source_key="sample",
        material_key="specimen:si_110",
        atomic_number=14,
        theta_rad=0.02,
        phi_rad=0.5,
    )
    transport = SimpleNamespace(
        metrics={
            "electron_transport_model": "test_elastic_model",
            "incident_mean_energy_ev": 200_000.0,
            "outcome_weight_fractions": {
                "transmitted": 0.75,
                "backscattered": 0.20,
                "lateral_escape": 0.05,
            }
        },
        trajectories=(
            SimpleNamespace(
                events=(event,),
                incident_weight=1.0,
                initial_energy_ev=200_000.0,
                source_ray_index=7,
            ),
        ),
        material_flights=(
            SimpleNamespace(
                start_nm=(0.0, 0.0, -5.0),
                end_nm=(0.0, 0.0, 5.0),
                source_key="sample",
                material_key="specimen:si_110",
                history="straight_primary",
                source_ray_index=7,
            ),
        ),
        terminal_electrons=SimpleNamespace(
            kinetic_energy_ev=np.asarray((200_000.0, 200_000.0, 200_000.0)),
            weight=np.asarray((0.75, 0.20, 0.05)),
            outcome=("transmitted", "backscattered", "lateral_escape"),
        ),
    )
    vacancy = SimpleNamespace(
        vacancy_id="vacancy:track0:ray7:Z14:K",
        source_ray_index=7,
        source_key="sample",
        material_key="specimen:si_110",
        atomic_number=14,
        electron_history="straight_primary",
        electron_energy_ev=200_000.0,
        electron_weight=1.0,
        edge_energy_ev=1839.0,
        expected_occurrences_per_incident_electron=0.2,
        expected_vacancies=4.0,
        fluorescence_yield=0.5,
        auger_yield=0.4,
        unresolved_relaxation_yield=0.1,
    )
    line = SimpleNamespace(
        vacancy_id=vacancy.vacancy_id,
        transition="K-L3",
        energy_ev=1740.0,
        expected_emitted_photons=2.0,
    )
    spectrum = SimpleNamespace(
        elastic_transport=transport,
        metrics={"incident_electrons": 20.0},
        vacancies=(vacancy,),
        lines=(line,),
    )
    calls = []

    def fake_eds(state, geometry, **kwargs):
        calls.append((state, geometry, kwargs))
        return spectrum

    import temsim.detector.eds_signal as eds_signal

    monkeypatch.setattr(eds_signal, "simulate_eds_point", fake_eds)
    state = SimpleNamespace()
    simulation = SimpleNamespace(real_interactions=None)
    geometry = object()
    request = SpecimenInteractionRequest.eds_point(
        x_nm=2.0,
        y_nm=-3.0,
        dwell_time_s=4.0e-6,
        incident_electrons=20.0,
    )

    result = run_specimen_interactions(
        state,
        simulation,
        request,
        detector_geometry=geometry,
    )

    assert result.eds_spectrum is spectrum
    assert result.elastic_transport is transport
    assert result.completed_observables == frozenset(
        (
            SpecimenObservable.CHARACTERISTIC_X_RAY,
            SpecimenObservable.ELASTIC_TRANSPORT,
        )
    )
    assert result.conservation[0].conserved
    assert {check.name for check in result.conservation} == {
        "specimen_elastic_terminal_weight",
        "specimen_elastic_mean_energy",
        "eds_direct_vacancy_relaxation",
    }
    assert len(result.events) == 6
    assert result.events[0].process is InteractionProcess.ELASTIC_SCATTER
    assert result.events[0].event_id.startswith("elastic:")
    assert result.events[0].parent_electron_index == 7
    assert result.events[0].atomic_number == 14
    assert result.events[0].energy_transfer_ev == 0.0
    vacancy_event = result.events[1]
    radiative_event = result.events[2]
    photon_event = result.events[3]
    auger_event = result.events[4]
    unresolved_event = result.events[5]
    assert vacancy_event.process is InteractionProcess.CORE_IONISATION
    assert vacancy_event.event_id == vacancy.vacancy_id
    assert vacancy_event.position_nm[2] == pytest.approx(
        photon_event.position_nm[2]
    )
    assert -5.0 < vacancy_event.position_nm[2] < 5.0
    assert radiative_event.process is InteractionProcess.RADIATIVE_RELAXATION
    assert radiative_event.parent_event_id == vacancy_event.event_id
    assert photon_event.process is InteractionProcess.CHARACTERISTIC_X_RAY
    assert photon_event.parent_event_id == radiative_event.event_id
    assert photon_event.emitted_energy_ev == pytest.approx(1740.0)
    assert photon_event.expected_occurrences_per_source_electron == (
        pytest.approx(0.1)
    )
    assert auger_event.process is InteractionProcess.AUGER_RELAXATION
    assert auger_event.parent_event_id == vacancy_event.event_id
    assert auger_event.emitted_energy_ev is None
    assert unresolved_event.process is InteractionProcess.UNRESOLVED_RELAXATION
    assert unresolved_event.parent_event_id == vacancy_event.event_id
    assert unresolved_event.emitted_energy_ev is None
    assert result.metrics["per_event_ledger_populated"] is True
    assert result.metrics["eds_event_ledger_recomputed_cross_sections"] is False
    assert result.metrics[
        "event_ledger_expected_vacancies_per_source_electron"
    ] == pytest.approx(0.2)
    relaxation_check = next(
        check
        for check in result.conservation
        if check.name == "eds_direct_vacancy_relaxation"
    )
    assert relaxation_check.conserved
    assert relaxation_check.input_value == pytest.approx(4.0)
    assert result.metrics[
        "event_ledger_expected_radiative_relaxations_per_source_electron"
    ] == pytest.approx(0.1)
    assert result.metrics[
        "event_ledger_expected_auger_relaxations_per_source_electron"
    ] == pytest.approx(0.08)
    assert result.metrics[
        "event_ledger_expected_unresolved_relaxations_per_source_electron"
    ] == pytest.approx(0.02)
    assert spectrum.metrics["core_ionisation_accounting"].startswith(
        "derived relaxation observable"
    )
    assert not any(
        coupling.contributes_to_exclusive_electron_budget
        for coupling in result.couplings
    )
    assert calls == [
        (
            state,
            geometry,
            {
                "simulation": simulation,
                "x_nm": 2.0,
                "y_nm": -3.0,
                "dwell_time_s": 4.0e-6,
                "incident_electrons": 20.0,
            },
        )
    ]


def test_engine_can_run_elastic_transport_without_requesting_eds(monkeypatch):
    import temsim.specimen.elastic_transport as elastic_transport

    bundle = SimpleNamespace(rays=(object(),))
    transport = SimpleNamespace(
        metrics={"outcome_weight_fractions": {"transmitted": 1.0}}
    )
    calls = []

    def fake_bundle(state, simulation, **kwargs):
        calls.append(("bundle", state, simulation, kwargs))
        return bundle

    def fake_transport(state, **kwargs):
        calls.append(("transport", state, kwargs))
        return transport

    monkeypatch.setattr(
        elastic_transport, "incident_rays_from_simulation", fake_bundle
    )
    monkeypatch.setattr(
        elastic_transport, "simulate_elastic_point_transport", fake_transport
    )
    state = SimpleNamespace()
    simulation = SimpleNamespace(real_interactions=None)
    progress_callback = object()

    result = run_specimen_interactions(
        state,
        simulation,
        SpecimenInteractionRequest.elastic_point(x_nm=4.0, y_nm=5.0),
        progress_callback=progress_callback,
    )

    assert result.incident_bundle is bundle
    assert result.elastic_transport is transport
    assert result.eds_spectrum is None
    assert calls == [
        (
            "bundle",
            state,
            simulation,
            {"target_x_nm": 4.0, "target_y_nm": 5.0},
        ),
        (
            "transport",
            state,
            {
                "incident_rays": bundle.rays,
                "progress_callback": progress_callback,
            },
        ),
    ]


def test_engine_does_not_run_unrequested_wave(monkeypatch):
    import temsim.physics.wave_imaging as wave_imaging

    monkeypatch.setattr(
        wave_imaging,
        "simulate_wave_image",
        lambda *_args, **_kwargs: pytest.fail("wave calculation was not requested"),
    )

    result = run_specimen_interactions(
        SimpleNamespace(),
        SimpleNamespace(real_interactions=None),
        SpecimenInteractionRequest(),
    )

    assert result.wave_imaging is None
    assert not result.completed_observables


def test_engine_enriches_and_reuses_one_shared_specimen_result(monkeypatch):
    import temsim.detector.eds_signal as eds_signal
    import temsim.physics.wave_imaging as wave_imaging

    wave = object()
    base = SpecimenInteractionResult(
        request=SpecimenInteractionRequest.tem_wave(),
        completed_observables=frozenset(
            (SpecimenObservable.COHERENT_ELASTIC_WAVE,)
        ),
        wave_imaging=wave,
    )
    spectrum = SimpleNamespace(
        elastic_transport=None,
        metrics={},
        vacancies=(),
        lines=(),
    )
    calls = []

    monkeypatch.setattr(
        wave_imaging,
        "simulate_wave_image",
        lambda *_args, **_kwargs: pytest.fail(
            "the completed coherent wave must be reused"
        ),
    )

    def fake_eds(*args, **kwargs):
        calls.append((args, kwargs))
        return spectrum

    monkeypatch.setattr(eds_signal, "simulate_eds_point", fake_eds)
    state = SimpleNamespace()
    simulation = SimpleNamespace(real_interactions=None)
    request = SpecimenInteractionRequest.eds_point()

    enriched = run_specimen_interactions(
        state,
        simulation,
        request,
        detector_geometry=object(),
        existing_result=base,
    )

    assert enriched.wave_imaging is wave
    assert enriched.eds_spectrum is spectrum
    assert enriched.request.observables == frozenset(
        (
            SpecimenObservable.COHERENT_ELASTIC_WAVE,
            SpecimenObservable.CHARACTERISTIC_X_RAY,
        )
    )
    assert enriched.metrics["calculated_observables_this_call"] == (
        SpecimenObservable.CHARACTERISTIC_X_RAY.value,
    )
    assert enriched.metrics["reused_observables"] == (
        SpecimenObservable.COHERENT_ELASTIC_WAVE.value,
    )
    assert len(calls) == 1

    reused = run_specimen_interactions(
        state,
        simulation,
        request,
        existing_result=enriched,
    )

    assert reused.wave_imaging is wave
    assert reused.eds_spectrum is spectrum
    assert reused.metrics["calculated_observables_this_call"] == ()
    assert reused.metrics["reused_observables"] == (
        SpecimenObservable.CHARACTERISTIC_X_RAY.value,
        SpecimenObservable.COHERENT_ELASTIC_WAVE.value,
    )
    assert len(calls) == 1


def test_changed_point_request_recalculates_only_point_observables(monkeypatch):
    import temsim.detector.eds_signal as eds_signal

    first_spectrum = SimpleNamespace(
        elastic_transport=None,
        metrics={},
        vacancies=(),
        lines=(),
    )
    existing = SpecimenInteractionResult(
        request=SpecimenInteractionRequest.eds_point(x_nm=1.0, y_nm=2.0),
        completed_observables=frozenset(
            (SpecimenObservable.CHARACTERISTIC_X_RAY,)
        ),
        eds_spectrum=first_spectrum,
    )
    second_spectrum = SimpleNamespace(
        elastic_transport=None,
        metrics={},
        vacancies=(),
        lines=(),
    )
    calls = []

    def fake_eds(*args, **kwargs):
        calls.append((args, kwargs))
        return second_spectrum

    monkeypatch.setattr(eds_signal, "simulate_eds_point", fake_eds)
    request = SpecimenInteractionRequest.eds_point(x_nm=3.0, y_nm=2.0)
    result = run_specimen_interactions(
        SimpleNamespace(),
        SimpleNamespace(real_interactions=None),
        request,
        detector_geometry=object(),
        existing_result=existing,
    )

    assert result.eds_spectrum is second_spectrum
    assert result.request.point_x_nm == pytest.approx(3.0)
    assert result.metrics["existing_result_reused"] is False
    assert result.metrics["calculated_observables_this_call"] == (
        SpecimenObservable.CHARACTERISTIC_X_RAY.value,
    )
    assert len(calls) == 1


def test_result_rejects_two_additive_electron_population_owners():
    owner = SpecimenModelCoupling(
        solver="first",
        processes=(InteractionProcess.CORE_IONISATION,),
        representation="exclusive probability",
        event_resolution="aggregate",
        energy_coupling="mean loss",
        contributes_to_exclusive_electron_budget=True,
        double_count_rule="only owner",
    )
    duplicate = SpecimenModelCoupling(
        solver="second",
        processes=(InteractionProcess.PLASMON_LOSS,),
        representation="exclusive probability",
        event_resolution="aggregate",
        energy_coupling="mean loss",
        contributes_to_exclusive_electron_budget=True,
        double_count_rule="must not coexist",
    )

    with pytest.raises(ValueError, match="Only one solver"):
        SpecimenInteractionResult(
            request=SpecimenInteractionRequest(),
            completed_observables=frozenset(),
            couplings=(owner, duplicate),
        )
