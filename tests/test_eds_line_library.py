"""Offline database parity and signal-only EDS annotation aggregation."""

from dataclasses import FrozenInstanceError
from importlib.metadata import version
from types import SimpleNamespace

import numpy as np
import pytest
import xraylib

from temsim.detector import eds_line_library as library
from temsim.detector.eds_signal import _radiative_lines


def _previous_radiative_lines(z, shell):
    # Independent copy of the pre-extraction enumeration. Compound families
    # are absent, and the same conversion/order must remain in the signal path.
    orbitals = (
        "K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5",
        "N1", "N2", "N3", "N4", "N5", "N6", "N7",
        "O1", "O2", "O3", "O4", "O5", "O6", "O7",
        "P1", "P2", "P3", "P4", "P5", "Q1",
    )
    origin = orbitals[int(shell)]
    rows = []
    for destination in orbitals[int(shell) + 1:]:
        constant = getattr(xraylib, f"{origin}{destination}_LINE", None)
        if constant is None:
            continue
        try:
            energy = 1000.0 * float(xraylib.LineEnergy(int(z), constant))
            rate = float(xraylib.RadRate(int(z), constant))
        except ValueError:
            continue
        if energy > 0.0 and rate > 0.0:
            rows.append((f"{origin}-{destination}", energy, rate))
    return tuple(rows)


@pytest.mark.parametrize("z", [1, 2, 6, 14, 26, 29, 79, 92, 99])
def test_radiative_enumeration_exactly_preserves_previous_results(z):
    for shell in range(29):
        assert library.radiative_lines(z, shell) == _previous_radiative_lines(z, shell)
    assert _radiative_lines is library.radiative_lines


@pytest.mark.parametrize("z,symbol", [(14, "Si"), (26, "Fe")])
def test_common_ka1_label_uses_database_alias_and_ev_units(z, symbol):
    rows = library.element_lines(z)
    peak = next(row for row in rows if row.transition == "K-L3")
    assert xraylib.KA1_LINE == xraylib.KL3_LINE
    assert peak.energy_ev == 1000.0 * xraylib.LineEnergy(z, xraylib.KA1_LINE)
    assert peak.relative_rate == xraylib.RadRate(z, xraylib.KA1_LINE)
    assert peak.label == f"{symbol} Kα1 (K-L3)"
    assert peak.symbol == symbol
    assert peak.atomic_number == z
    assert peak.energy_ev > 1000
    assert len({row.transition for row in rows}) == len(rows)
    expected = tuple(row for shell in range(29) for row in _previous_radiative_lines(z, shell))
    assert tuple((row.transition, row.energy_ev, row.relative_rate) for row in rows) == expected
    with pytest.raises(FrozenInstanceError):
        peak.energy_ev = 0


def test_library_elements_and_provenance_are_offline_and_complete():
    entries = library.library_elements()
    assert len(entries) == 99
    assert entries[0] == (1, "H")
    assert entries[13] == (14, "Si")
    assert entries[25] == (26, "Fe")
    assert entries[-1] == (99, "Es")
    assert library.element_lines(1) == ()
    assert library.element_lines(2) == ()
    provenance = library.library_provenance()
    assert version("xraylib") in provenance
    assert "Offline" in provenance and "LineEnergy / RadRate" in provenance


@pytest.mark.parametrize("z", [0, 100, -1, 14.5, True])
def test_element_library_rejects_invalid_atomic_numbers(z):
    with pytest.raises(ValueError, match="atomic numbers"):
        library.element_lines(z)


def _signal(*, z=14, transition="K-L3", energy=1740.0, counts=1.0,
            source="sample", vacancy="vacancy-1"):
    return SimpleNamespace(
        atomic_number=z, transition=transition, energy_ev=energy,
        expected_detected_counts=counts, source_key=source, vacancy_id=vacancy,
    )


def test_vacancy_contributions_collapse_to_one_annotation_without_changing_counts():
    rows = [
        _signal(counts=2.5, vacancy="one"),
        _signal(counts=3.0, source="support", vacancy="two"),
        _signal(counts=0.5, vacancy="three"),
        _signal(z=26, transition="K-L3", energy=6403.9, counts=4.0, source="holder"),
    ]
    before = [vars(row).copy() for row in rows]
    annotations = library.aggregate_simulated_lines(iter(rows))
    assert len(annotations) == 2
    si, fe = annotations
    assert si.expected_counts == 6.0
    assert si.source_keys == ("sample", "support")
    assert si.label == "Si Kα1 (K-L3)"
    assert fe.expected_counts == 4.0
    assert fe.source_keys == ("holder",)
    assert sum(item.expected_counts for item in annotations) == sum(row.expected_detected_counts for row in rows)
    assert [vars(row) for row in rows] == before


def test_no_positive_signal_means_no_simulated_peak_or_reference_completion(monkeypatch):
    monkeypatch.setattr(library, "element_lines", lambda _z: pytest.fail("reference completion is not allowed"))
    assert library.aggregate_simulated_lines(()) == ()
    rows = [_signal(counts=value) for value in (0.0, -1.0, np.nan, np.inf)]
    rows += [_signal(energy=value) for value in (0.0, -1.0, np.nan, np.inf)]
    assert library.aggregate_simulated_lines(rows) == ()


def test_annotation_aggregation_does_not_merge_distinct_transitions_or_round_energies():
    rows = [
        _signal(transition="K-M2", energy=1835.9, counts=1),
        _signal(transition="K-M3", energy=1835.9, counts=2),
        _signal(transition="K-M3", energy=np.nextafter(1835.9, np.inf), counts=3),
    ]
    peaks = library.aggregate_simulated_lines(rows)
    assert len(peaks) == 3
    assert [peak.expected_counts for peak in peaks] == [1, 2, 3]


def test_uncommon_transition_uses_iupac_without_inventing_a_siegbahn_alias():
    peak, = library.aggregate_simulated_lines([_signal(transition="N1-O2", energy=100)])
    assert peak.label == "Si N1-O2"


def test_count_overflow_is_not_published_as_a_peak():
    with pytest.raises(ValueError, match="finite numeric range"):
        library.aggregate_simulated_lines([_signal(counts=1e308), _signal(counts=1e308)])
