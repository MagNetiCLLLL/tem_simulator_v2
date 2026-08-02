"""Physical two-blade energy-selection slit."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from temsim.component_keys import ENERGY_FILTER_SLIT


@dataclass
class EnergySelectionSlitComponent:
    """Continuously adjustable piezo-driven mechanical slit."""

    name: str = "Energy Selection Slit"
    key: str = ENERGY_FILTER_SLIT
    inserted: bool = True
    distance_from_sector_exit_m: float = 205.0e-3
    gap_m: float = 36.0e-6
    centre_m: float = 0.0
    clear_height_m: float = 12.5e-3
    maximum_gap_m: float = 12.5e-3
    blade_thickness_m: float = 1.0e-3
    requested_width_ev: float = 10.0
    requested_centre_loss_ev: float = 0.0
    calibrated_dispersion_um_per_ev: float = 3.6

    KIND = "energy_selection_slit"
    INTERACTION_KIND = "two_blade_mechanical_stop"

    def __post_init__(self):
        if self.key != ENERGY_FILTER_SLIT:
            raise ValueError("Energy Selection Slit key is not canonical.")
        values = (
            self.distance_from_sector_exit_m,
            self.gap_m,
            self.centre_m,
            self.clear_height_m,
            self.maximum_gap_m,
            self.blade_thickness_m,
            self.requested_width_ev,
            self.requested_centre_loss_ev,
            self.calibrated_dispersion_um_per_ev,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Energy slit parameters must be finite.")
        if self.distance_from_sector_exit_m <= 0.0:
            raise ValueError("Energy slit must follow the sector exit.")
        if self.maximum_gap_m <= 0.0 or self.clear_height_m <= 0.0:
            raise ValueError("Energy slit clear dimensions must be positive.")
        if not 0.0 <= self.gap_m <= self.maximum_gap_m:
            raise ValueError("Energy slit gap exceeds its physical travel.")
        if self.blade_thickness_m <= 0.0:
            raise ValueError("Energy slit blades must have finite thickness.")
        if self.requested_width_ev < 0.0:
            raise ValueError("Requested energy width must not be negative.")
        if self.calibrated_dispersion_um_per_ev == 0.0:
            raise ValueError("Energy slit dispersion must be non-zero.")

    @property
    def lower_blade_edge_m(self):
        return float(self.centre_m) - 0.5 * float(self.gap_m)

    @property
    def upper_blade_edge_m(self):
        return float(self.centre_m) + 0.5 * float(self.gap_m)

    @property
    def derived_width_ev(self):
        return (
            float(self.gap_m)
            * 1.0e6
            / abs(float(self.calibrated_dispersion_um_per_ev))
        )

    @property
    def derived_centre_loss_ev(self):
        return (
            float(self.centre_m)
            * 1.0e6
            / float(self.calibrated_dispersion_um_per_ev)
        )

    def configure_energy_window(
        self,
        centre_loss_ev,
        width_ev,
        dispersion_um_per_ev=None,
    ):
        """Move both physical blades from a calibrated software request."""

        centre = float(centre_loss_ev)
        width = float(width_ev)
        dispersion = float(
            self.calibrated_dispersion_um_per_ev
            if dispersion_um_per_ev is None
            else dispersion_um_per_ev
        )
        if not all(math.isfinite(value) for value in (
            centre,
            width,
            dispersion,
        )):
            raise ValueError("Energy-window values must be finite.")
        if width < 0.0:
            raise ValueError("Energy-window width must not be negative.")
        if dispersion == 0.0:
            raise ValueError("Calibrated slit dispersion must be non-zero.")
        physical_gap = abs(dispersion) * width * 1.0e-6
        if physical_gap > self.maximum_gap_m:
            raise ValueError(
                "Requested energy window exceeds physical slit travel."
            )
        self.requested_centre_loss_ev = centre
        self.requested_width_ev = width
        self.calibrated_dispersion_um_per_ev = dispersion
        self.centre_m = dispersion * centre * 1.0e-6
        self.gap_m = physical_gap
        return self

    def transmission_mask(self, dispersive_m, non_dispersive_m):
        dispersive, non_dispersive = np.broadcast_arrays(
            np.asarray(dispersive_m, dtype=float),
            np.asarray(non_dispersive_m, dtype=float),
        )
        if not self.inserted:
            return np.ones(dispersive.shape, dtype=bool)
        return (
            (dispersive >= self.lower_blade_edge_m)
            & (dispersive <= self.upper_blade_edge_m)
            & (
                np.abs(non_dispersive)
                <= 0.5 * float(self.clear_height_m)
            )
        )


def create_energy_selection_slit(
    centre_loss_ev=0.0,
    width_ev=10.0,
    dispersion_um_per_ev=3.6,
    distance_from_sector_exit_m=205.0e-3,
):
    slit = EnergySelectionSlitComponent(
        distance_from_sector_exit_m=float(
            distance_from_sector_exit_m
        ),
        calibrated_dispersion_um_per_ev=float(
            dispersion_um_per_ev
        ),
    )
    return slit.configure_energy_window(
        centre_loss_ev,
        width_ev,
        dispersion_um_per_ev,
    )


def serialise_energy_selection_slit(slit):
    return {
        "name": slit.name,
        "key": slit.key,
        "inserted": bool(slit.inserted),
        "gap_m": float(slit.gap_m),
        "centre_m": float(slit.centre_m),
        "requested_width_ev": float(slit.requested_width_ev),
        "requested_centre_loss_ev": float(
            slit.requested_centre_loss_ev
        ),
        "calibrated_dispersion_um_per_ev": float(
            slit.calibrated_dispersion_um_per_ev
        ),
    }


def energy_selection_slit_from_dict(values, **legacy):
    if not isinstance(values, dict):
        return create_energy_selection_slit(**legacy)
    allowed = {
        "name",
        "key",
        "inserted",
        "gap_m",
        "centre_m",
        "requested_width_ev",
        "requested_centre_loss_ev",
        "calibrated_dispersion_um_per_ev",
    }
    return EnergySelectionSlitComponent(**{
        key: value
        for key, value in values.items()
        if key in allowed
    })
