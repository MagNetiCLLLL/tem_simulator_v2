"""Indexing stored flights preserves the full-scan deterministic event ledger."""

from types import SimpleNamespace

import pytest

from temsim.specimen import interaction_engine


def _flight(ray, *, source="sample", material="si", history="primary", start=(0., 0., 0.), end=(0., 0., 1.)):
    return SimpleNamespace(
        source_ray_index=ray, source_key=source, material_key=material,
        history=history, start_nm=start, end_nm=end,
    )


def _vacancy(identifier, ray, *, source="sample", material="si", history="primary"):
    return SimpleNamespace(
        vacancy_id=identifier, source_ray_index=ray, source_key=source,
        material_key=material, electron_history=history, atomic_number=14,
        electron_energy_ev=300000., electron_weight=0.1, edge_energy_ev=1839.,
        expected_occurrences_per_incident_electron=0.03,
        expected_vacancies=0.6, fluorescence_yield=0.5,
        auger_yield=0.4, unresolved_relaxation_yield=0.1,
    )


def _spectrum(vacancies, *, source_electrons=100.):
    return SimpleNamespace(
        vacancies=vacancies,
        lines=tuple(SimpleNamespace(
            vacancy_id=vacancy.vacancy_id, transition="K-L3", energy_ev=1740.,
            expected_emitted_photons=0.3,
        ) for vacancy in vacancies),
        metrics={"incident_electrons": 20.,
                 "source_electrons_before_column_losses": source_electrons},
    )


@pytest.mark.parametrize("source_electrons", [0., 100.])
def test_indexed_ledger_exactly_matches_original_full_flight_search(monkeypatch, source_electrons):
    duplicate = _flight(7, start=(2., -3., 1.), end=(2., -3., 4.))
    flights = (
        _flight(7, start=(1., 2., -4.), end=(1., 2., -2.)),
        _flight(8),
        _flight(7, source="grid"),
        duplicate,
        _flight(7, material="cu"),
        _flight(7, history="scattered"),
        _flight(7, end=(0., 0., 0.)),  # Zero-length flight remains ignored.
        duplicate,
        _flight(7, start=(-1., 2., 7.), end=(-1., 2., 7.5)),
    )
    vacancies = tuple(_vacancy(f"vacancy-{index}", 7) for index in range(35)) + (
        _vacancy("grid", 7, source="grid"),
        _vacancy("cu", 7, material="cu"),
        _vacancy("scattered", 7, history="scattered"),
        _vacancy("other-ray", 8),
        _vacancy("missing-ray", None),
        _vacancy("unmatched-ray", 99),
        _vacancy("unmatched-material", 7, material="au"),
    )
    transport = SimpleNamespace(material_flights=flights)
    spectrum = _spectrum(vacancies, source_electrons=source_electrons)
    indexed = interaction_engine._eds_event_ledger(spectrum, transport)
    original_sampler = interaction_engine._vacancy_position_nm
    monkeypatch.setattr(interaction_engine, "_vacancy_position_nm",
                        lambda vacancy, _subset: original_sampler(vacancy, flights))
    reference = interaction_engine._eds_event_ledger(spectrum, transport)
    # Exact equality includes order, IDs, positions, parent links and weights.
    assert indexed == reference
    assert indexed
    assert not any(event.event_id in {"missing-ray", "unmatched-ray", "unmatched-material"} for event in indexed)
    assert transport.material_flights is flights
    assert flights[3] is flights[7]


def test_position_search_work_is_bounded_by_matching_flights(monkeypatch):
    ray_count = 120
    flights = tuple(
        _flight(ray, start=(0., 0., float(part)), end=(0., 0., float(part + 1)))
        for part in range(3)
        for ray in range(ray_count)
    )
    vacancies = tuple(
        _vacancy(f"ray{ray}:shell{shell}", ray)
        for ray in range(ray_count)
        for shell in range(4)
    )
    original_sampler = interaction_engine._vacancy_position_nm
    inspected = 0
    received = []

    def counted(vacancy, subset):
        nonlocal inspected
        inspected += len(subset)
        received.append(tuple(flight.source_ray_index for flight in subset))
        return original_sampler(vacancy, subset)

    monkeypatch.setattr(interaction_engine, "_vacancy_position_nm", counted)
    events = interaction_engine._eds_event_ledger(
        _spectrum(vacancies), SimpleNamespace(material_flights=flights),
    )
    assert inspected == len(vacancies) * 3
    assert inspected == len(vacancies) * len(flights) // ray_count
    assert all(indices == (vacancy.source_ray_index,) * 3
               for indices, vacancy in zip(received, vacancies, strict=True))
    assert len(events) == len(vacancies) * 5


@pytest.mark.parametrize("vacancies", [(), (_vacancy("missing", None),)])
def test_empty_or_missing_source_vacancies_create_no_events(vacancies):
    assert interaction_engine._eds_event_ledger(
        _spectrum(vacancies), SimpleNamespace(material_flights=(_flight(1),)),
    ) == ()


def test_empty_transport_or_spectrum_retains_early_returns():
    spectrum = _spectrum((_vacancy("one", 1),))
    assert interaction_engine._eds_event_ledger(None, SimpleNamespace()) == ()
    assert interaction_engine._eds_event_ledger(spectrum, None) == ()
    assert interaction_engine._eds_event_ledger(spectrum, SimpleNamespace(material_flights=())) == ()
