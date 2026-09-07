"""Shared offline characteristic-line data and simulated-peak annotations.

Energies and radiative rates come directly from the installed xraylib tables.
Rates describe branching from the initial subshell, not elemental abundance
or expected detector intensity. Reference lines never create simulated counts.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version
import math
from typing import Protocol

import xraylib


_ORBITALS = (
    "K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5",
    "N1", "N2", "N3", "N4", "N5", "N6", "N7",
    "O1", "O2", "O3", "O4", "O5", "O6", "O7",
    "P1", "P2", "P3", "P4", "P5", "Q1",
)

# These labels rename individual transition constants only. Combined-family
# constants such as KA_LINE and LA_LINE are deliberately never enumerated.
_COMMON_ALIASES = (
    ("KA1_LINE", "Kα1"), ("KA2_LINE", "Kα2"), ("KA3_LINE", "Kα3"),
    ("KB1_LINE", "Kβ1"), ("KB2_LINE", "Kβ2"), ("KB3_LINE", "Kβ3"),
    ("KB4_LINE", "Kβ4"), ("KB5_LINE", "Kβ5"),
    ("LA1_LINE", "Lα1"), ("LA2_LINE", "Lα2"),
    ("LB1_LINE", "Lβ1"), ("LB2_LINE", "Lβ2"),
    ("LB3_LINE", "Lβ3"), ("LB4_LINE", "Lβ4"), ("LB5_LINE", "Lβ5"),
    ("LB6_LINE", "Lβ6"), ("LB7_LINE", "Lβ7"), ("LB9_LINE", "Lβ9"),
    ("LB10_LINE", "Lβ10"), ("LB15_LINE", "Lβ15"), ("LB17_LINE", "Lβ17"),
    ("LG1_LINE", "Lγ1"), ("LG2_LINE", "Lγ2"), ("LG3_LINE", "Lγ3"),
    ("LG4_LINE", "Lγ4"), ("LG5_LINE", "Lγ5"), ("LG6_LINE", "Lγ6"),
    ("MA1_LINE", "Mα1"), ("MA2_LINE", "Mα2"),
    ("MB_LINE", "Mβ"), ("MG_LINE", "Mγ"),
)


@dataclass(frozen=True, slots=True)
class CharacteristicLine:
    atomic_number: int
    symbol: str
    transition: str
    energy_ev: float
    relative_rate: float
    label: str


@dataclass(frozen=True, slots=True)
class PeakAnnotation:
    atomic_number: int
    symbol: str
    transition: str
    label: str
    energy_ev: float
    expected_counts: float
    source_keys: tuple[str, ...]


class _SimulatedLine(Protocol):
    atomic_number: int
    transition: str
    energy_ev: float
    expected_detected_counts: float
    source_key: str


@lru_cache(maxsize=None)
def radiative_lines(
    atomic_number: int, subshell_index: int
) -> tuple[tuple[str, float, float], ...]:
    """Return the signal model's unchanged IUPAC enumeration and arithmetic."""
    origin = _ORBITALS[int(subshell_index)]
    rows = []
    for destination in _ORBITALS[int(subshell_index) + 1 :]:
        constant_name = f"{origin}{destination}_LINE"
        constant = getattr(xraylib, constant_name, None)
        if constant is None:
            continue
        try:
            energy_ev = 1000.0 * float(
                xraylib.LineEnergy(int(atomic_number), constant)
            )
            rate = float(xraylib.RadRate(int(atomic_number), constant))
        except ValueError:
            continue
        if energy_ev > 0.0 and rate > 0.0:
            rows.append((f"{origin}-{destination}", energy_ev, rate))
    return tuple(rows)


def _atomic_number(value: int) -> int:
    z = int(value)
    if isinstance(value, bool) or z != value or not 1 <= z <= 99:
        raise ValueError("The EDS line library supports atomic numbers 1 through 99.")
    return z


@lru_cache(maxsize=1)
def library_elements() -> tuple[tuple[int, str], ...]:
    """All supported elements; an element can have no tabulated emission line."""
    return tuple((z, str(xraylib.AtomicNumberToSymbol(z))) for z in range(1, 100))


@lru_cache(maxsize=1)
def library_provenance() -> str:
    return (
        f"xraylib {version('xraylib')} | Offline LineEnergy / RadRate tables. "
        "Rates are branching probabilities per initial subshell vacancy."
    )


@lru_cache(maxsize=4096)
def _line_label(atomic_number: int, transition: str) -> str:
    symbol = str(xraylib.AtomicNumberToSymbol(atomic_number))
    constant = getattr(xraylib, transition.replace("-", "") + "_LINE", None)
    if constant is not None:
        for alias, name in _COMMON_ALIASES:
            if constant == getattr(xraylib, alias, None):
                return f"{symbol} {name} ({transition})"
    return f"{symbol} {transition}"


@lru_cache(maxsize=99)
def element_lines(atomic_number: int) -> tuple[CharacteristicLine, ...]:
    """Individual reference transitions, in subshell/destination table order."""
    z = _atomic_number(atomic_number)
    symbol = str(xraylib.AtomicNumberToSymbol(z))
    return tuple(
        CharacteristicLine(z, symbol, transition, energy, rate, _line_label(z, transition))
        for index in range(len(_ORBITALS))
        for transition, energy, rate in radiative_lines(z, index)
    )


def aggregate_simulated_lines(lines: Iterable[_SimulatedLine]) -> tuple[PeakAnnotation, ...]:
    """Combine vacancy contributions into labels using actual positive counts.

    Exact transition energies remain distinct: neither rounding, fitted peak
    identification nor reference-library completion is performed. Every input
    contribution is counted once, in its original order, without mutation.
    """
    totals: dict[tuple[int, str, float], float] = {}
    sources: dict[tuple[int, str, float], set[str]] = {}
    for line in lines:
        count = float(line.expected_detected_counts)
        energy = float(line.energy_ev)
        if not (math.isfinite(count) and count > 0 and math.isfinite(energy) and energy > 0):
            continue
        z = _atomic_number(line.atomic_number)
        key = (z, str(line.transition), energy)
        totals[key] = totals.get(key, 0.0) + count
        sources.setdefault(key, set()).add(str(line.source_key))
    result = []
    for (z, transition, energy), count in sorted(
        totals.items(), key=lambda item: (item[0][2], item[0][0], item[0][1])
    ):
        if not math.isfinite(count):
            raise ValueError("Aggregated EDS expected counts exceed the finite numeric range.")
        result.append(PeakAnnotation(
            z, str(xraylib.AtomicNumberToSymbol(z)), transition, _line_label(z, transition),
            energy, count, tuple(sorted(sources[(z, transition, energy)])),
        ))
    return tuple(result)
