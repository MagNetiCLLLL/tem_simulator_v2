"""Electron-impact inner-shell ionisation using Bote--Salvat fits.

The coefficient table covers K, L1-L3 and M1-M5 subshells where available
for Z=1..99. Cross sections are returned in square centimetres per atom.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path

from temsim.paths import EDS_PHYSICS_CONFIG_ROOT


SUBSHELL_NAMES = ("K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5")
ELECTRON_REST_ENERGY_EV = 5.10998918e5
BOHR_RADIUS_CM = 5.291772108e-9


@dataclass(frozen=True, slots=True)
class BoteElementData:
    atomic_number: int
    be: tuple[float, ...]
    anlj: tuple[float, ...]
    g: tuple[tuple[float, float, float, float], ...]
    edge_ev: tuple[float, ...]
    a: tuple[tuple[float, float, float, float, float], ...]

    @property
    def subshell_names(self) -> tuple[str, ...]:
        return SUBSHELL_NAMES[: len(self.edge_ev)]


def _coefficient_path(path: Path | None = None) -> Path:
    return Path(path or (EDS_PHYSICS_CONFIG_ROOT / "bote_salvat.json"))


@lru_cache(maxsize=1)
def load_bote_salvat_coefficients(
    path: Path | None = None,
) -> dict[int, BoteElementData]:
    source = _coefficient_path(path)
    with source.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    result: dict[int, BoteElementData] = {}
    for key, row in raw.items():
        atomic_number = int(key)
        be = tuple(float(value) for value in row["Be"])
        anlj = tuple(float(value) for value in row["Anlj"])
        edge = tuple(float(value) for value in row["edge_eV"])
        g = tuple(tuple(float(value) for value in values) for values in row["G"])
        a = tuple(tuple(float(value) for value in values) for values in row["A"])
        length = len(edge)
        if (
            not 1 <= atomic_number <= 99
            or not 1 <= length <= len(SUBSHELL_NAMES)
            or len(be) != length
            or len(anlj) != length
            or len(g) != length
            or len(a) != length
            or any(len(values) != 4 for values in g)
            or any(len(values) != 5 for values in a)
            or any(value <= 0.0 for value in edge)
        ):
            raise ValueError(
                f"Invalid Bote-Salvat coefficient row for Z={atomic_number}"
            )
        result[atomic_number] = BoteElementData(
            atomic_number=atomic_number,
            be=be,
            anlj=anlj,
            g=g,
            edge_ev=edge,
            a=a,
        )
    if set(result) != set(range(1, 100)):
        raise ValueError(
            "Bote-Salvat coefficient table must contain every Z from 1 to 99"
        )
    return result


def bote_element_data(atomic_number: int) -> BoteElementData:
    try:
        return load_bote_salvat_coefficients()[int(atomic_number)]
    except KeyError as exc:
        raise ValueError(
            "Bote-Salvat electron ionisation data cover Z=1..99"
        ) from exc


def _subshell_index(subshell: str | int) -> int:
    if isinstance(subshell, str):
        key = subshell.strip().upper().replace("_", "")
        try:
            return SUBSHELL_NAMES.index(key)
        except ValueError as exc:
            raise ValueError(f"Unknown inner subshell: {subshell}") from exc
    index = int(subshell)
    if not 0 <= index < len(SUBSHELL_NAMES):
        raise ValueError("Inner-subshell index must be between 0 and 8")
    return index


def bote_ionisation_cross_section_cm2(
    atomic_number: int,
    subshell: str | int,
    electron_energy_ev: float,
    *,
    edge_energy_ev: float | None = None,
) -> float:
    """Return the Bote--Salvat electron-impact ionisation cross section."""

    energy = float(electron_energy_ev)
    if not math.isfinite(energy) or energy < 0.0:
        raise ValueError("Electron energy must be finite and non-negative")
    data = bote_element_data(atomic_number)
    index = _subshell_index(subshell)
    if index >= len(data.edge_ev):
        return 0.0
    edge = (
        data.edge_ev[index]
        if edge_energy_ev is None
        else float(edge_energy_ev)
    )
    if not math.isfinite(edge) or edge <= 0.0:
        raise ValueError("Ionisation edge energy must be finite and positive")
    overvoltage = energy / edge
    if overvoltage <= 1.0:
        return 0.0
    if overvoltage <= 16.0:
        a0, a1, a2, a3, a4 = data.a[index]
        inverse_one_plus_u = 1.0 / (1.0 + overvoltage)
        fit = (
            a0
            + a1 * overvoltage
            + inverse_one_plus_u
            * (
                a2
                + inverse_one_plus_u**2
                * (a3 + inverse_one_plus_u**2 * a4)
            )
        )
        reduced = (overvoltage - 1.0) * (fit / overvoltage) ** 2
    else:
        beta_squared = (
            energy * (energy + 2.0 * ELECTRON_REST_ENERGY_EV)
            / (energy + ELECTRON_REST_ENERGY_EV) ** 2
        )
        scaled_momentum = (
            math.sqrt(
                energy * (energy + 2.0 * ELECTRON_REST_ENERGY_EV)
            )
            / ELECTRON_REST_ENERGY_EV
        )
        g0, g1, g2, g3 = data.g[index]
        fit = (
            (2.0 * math.log(scaled_momentum) - beta_squared)
            * (1.0 + g0 / scaled_momentum)
            + g1
            + g2
            * math.sqrt(
                ELECTRON_REST_ENERGY_EV
                / (energy + ELECTRON_REST_ENERGY_EV)
            )
            + g3 / scaled_momentum
        )
        factor = data.anlj[index] / beta_squared
        reduced = (
            factor
            * overvoltage
            / (overvoltage + data.be[index])
            * fit
        )
    cross_section = 4.0 * math.pi * BOHR_RADIUS_CM**2 * reduced
    if cross_section < 0.0 and cross_section > -1.0e-40:
        return 0.0
    if not math.isfinite(cross_section) or cross_section < 0.0:
        raise ValueError(
            "Bote-Salvat fit produced an invalid ionisation cross section"
        )
    return cross_section
