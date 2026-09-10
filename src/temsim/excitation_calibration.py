"""Electrical control calibration, independent of the magnetic B(I) model.

Percent is a control coordinate. Even with a known N and I, saturation requires
a new joint field solve; this module never scales a saturated field.
"""

from dataclasses import asdict, dataclass
from datetime import date
import math

CALIBRATION_SCHEMA = "electrical-reference-v1"
SOURCE_KINDS = ("measured", "manufacturer", "literature", "estimated", "fitted")


@dataclass(frozen=True)
class ExcitationCalibration:
    ampere_turns_at_reference: float
    reference_excitation_percent: float = 100.0
    turns: int | None = None
    source_kind: str = "estimated"
    source: str = "Legacy prescribed ampere-turns; winding turns unavailable"
    source_date: str | None = None
    relative_uncertainty: float | None = None
    valid_current_range_A: tuple[float, float] | None = None

    def validate(self):
        if not (math.isfinite(self.ampere_turns_at_reference) and self.ampere_turns_at_reference >= 0
                and math.isfinite(self.reference_excitation_percent) and self.reference_excitation_percent > 0):
            raise ValueError("Reference ampere-turns must be nonnegative; reference percent must be positive")
        if self.turns is not None and (isinstance(self.turns, bool) or not math.isfinite(self.turns) or int(self.turns) != self.turns or self.turns < 1):
            raise ValueError("Known winding turns must be a positive integer; otherwise leave unavailable")
        if self.source_kind not in SOURCE_KINDS or not str(self.source).strip():
            raise ValueError("Calibration requires a source category and reference")
        if self.source_date is not None:
            date.fromisoformat(self.source_date)
        if self.relative_uncertainty is not None and not (math.isfinite(self.relative_uncertainty) and self.relative_uncertainty >= 0):
            raise ValueError("Relative uncertainty must be nonnegative or unavailable")
        if self.valid_current_range_A is not None:
            low, high = self.valid_current_range_A
            if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high):
                raise ValueError("Declare an increasing nonnegative current-magnitude validity range")
        return self

    @property
    def ampere_turns_at_100_percent(self):
        return self.ampere_turns_at_reference * 100 / self.reference_excitation_percent

    def at_control(self, percent, polarity=1, *, enabled=True):
        self.validate()
        percent = float(percent)
        if not math.isfinite(percent) or percent < 0 or polarity not in (-1, 1):
            raise ValueError("Control percent must be nonnegative with polarity +1 or -1")
        ni = self.ampere_turns_at_reference * percent / self.reference_excitation_percent * polarity if enabled else 0.0
        current = None if self.turns is None else ni / self.turns
        in_range = (None if current is None or self.valid_current_range_A is None else
                    self.valid_current_range_A[0] <= abs(current) <= self.valid_current_range_A[1])
        return {"schema": CALIBRATION_SCHEMA, "control_percent": percent, "polarity": polarity,
                "enabled": bool(enabled), "ampere_turns": ni, "turns": self.turns, "current_A": current,
                "current_status": "unavailable_unknown_turns" if current is None else self.source_kind,
                "within_calibrated_current_range": in_range, "provenance": asdict(self),
                "scope": "Electrical NI/I relation only; does not imply linear magnetic B response"}

    def control_for_current(self, current_A, *, maximum_percent=100., zero_polarity=1):
        self.validate()
        current = float(current_A)
        if self.turns is None:
            raise ValueError("Current control is unavailable without known winding turns")
        if not math.isfinite(current) or self.ampere_turns_at_reference <= 0:
            raise ValueError("Current inversion requires finite current and a nonzero reference excitation")
        percent = abs(current) * self.turns / self.ampere_turns_at_reference * self.reference_excitation_percent
        if not math.isfinite(maximum_percent) or not 0 <= percent <= maximum_percent:
            raise ValueError("Current exceeds the component's excitation limit")
        if self.valid_current_range_A is not None and not self.valid_current_range_A[0] <= abs(current) <= self.valid_current_range_A[1]:
            raise ValueError("Current is outside the declared calibration range")
        if zero_polarity not in (-1, 1):
            raise ValueError("Polarity must be +1 or -1")
        return percent, (-1 if current < 0 else 1 if current > 0 else zero_polarity)

    def to_dict(self):
        self.validate()
        return {key: value for key, value in asdict(self).items() if value is not None}


def calibration_from_recipe(recipe):
    """Legacy NI-at-100% remains valid, with no inferred current or turns."""
    raw = recipe.get("excitation_calibration")
    if raw is None:
        return ExcitationCalibration(float(recipe["ampere_turns"])).validate()
    calibration = ExcitationCalibration(**raw).validate()
    if "ampere_turns" in recipe and not math.isclose(float(recipe["ampere_turns"]), calibration.ampere_turns_at_100_percent, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Electrical calibration disagrees with prescribed ampere-turns at 100%")
    return calibration


def validate_excitation_recipe(recipe):
    if "excitation_calibration" in recipe:
        calibration_from_recipe(recipe)


def recipe_with_calibration(recipe, calibration):
    return {**recipe, "ampere_turns": calibration.ampere_turns_at_100_percent,
            "excitation_calibration": calibration.to_dict()}
