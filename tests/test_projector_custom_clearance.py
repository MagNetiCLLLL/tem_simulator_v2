"""Custom housing lengths retain optical centres and physical clearances."""
from copy import deepcopy
from pathlib import Path
import tomllib

import pytest

from temsim.module_manifest import (
    _validate_projector_lens_clearances,
    validate_document,
)


@pytest.fixture(params=("NoEnergyFilter.toml", "EnergyFilter.toml"))
def recording_document(request):
    path = (Path(__file__).parents[1] / "configs" / "instruments"
            / "project_and_recording_system" / request.param)
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _housing(document):
    return next(part for part in document["parts"]
                if part["key"] == "intermediate_lens_housing")


def _resize_housing(document, length):
    part = _housing(document)
    centre = part["local_center_z_mm"]
    part.update(length_mm=length, local_start_z_mm=centre - length / 2,
                local_end_z_mm=centre + length / 2)


def _actual_gaps(document):
    parts = {part["key"]: part for part in document["parts"]}
    keys = ("diffraction_lens_housing", "intermediate_lens_housing",
            "projector_lens_1_housing", "projector_lens_2_housing")
    return tuple(parts[downstream]["local_start_z_mm"] - parts[upstream]["local_end_z_mm"]
                 for upstream, downstream in zip(keys, keys[1:]))


@pytest.mark.parametrize("length, expected_gaps", (
    (225.0, (7.5, 7.5, 5.0)),
    (235.0, (2.5, 2.5, 5.0)),
    (240.0, (0.0, 0.0, 5.0)),
))
def test_custom_housing_lengths_keep_centres_and_other_parts(recording_document, length, expected_gaps):
    original = deepcopy(recording_document)
    assert _actual_gaps(original) == (5.0, 5.0, 5.0)
    _resize_housing(recording_document, length)
    edited = deepcopy(recording_document)

    validate_document(recording_document)

    assert recording_document == edited
    assert _actual_gaps(recording_document) == expected_gaps
    assert recording_document["geometry"] == original["geometry"]
    assert recording_document["geometry"]["projector_stack_inter_lens_gap_mm"] == 5.0
    original_parts = {part["key"]: part for part in original["parts"]}
    for part in recording_document["parts"]:
        before = original_parts[part["key"]]
        if part["key"] == "intermediate_lens_housing":
            assert part["local_center_z_mm"] == before["local_center_z_mm"] == 252.5
            assert {key for key in part if part[key] != before[key]} == {
                "length_mm", "local_start_z_mm", "local_end_z_mm",
            }
            if length == 225.0:
                assert (part["local_start_z_mm"], part["local_end_z_mm"]) == (140.0, 365.0)
        else:
            assert part == before


def test_custom_housing_length_still_rejects_overlap(recording_document):
    _resize_housing(recording_document, 241.0)
    assert _actual_gaps(recording_document) == (-0.5, -0.5, 5.0)
    with pytest.raises(ValueError, match="actual housing gap must be finite and non-negative"):
        validate_document(recording_document)


def test_custom_clearance_still_requires_uniform_vacuum_bore(recording_document):
    _resize_housing(recording_document, 225.0)
    _housing(recording_document)["vacuum_inner_diameter_mm"] = 19.0
    with pytest.raises(ValueError, match="vacuum ID .* does not match projector stack"):
        validate_document(recording_document)


@pytest.mark.parametrize("nominal", (-0.1, 10.1, float("inf"), float("nan")))
def test_nominal_gap_remains_finite_and_bounded(recording_document, nominal):
    recording_document["geometry"]["projector_stack_inter_lens_gap_mm"] = nominal
    with pytest.raises(ValueError, match="nominal inter-lens gap must be between 0 and 10 mm"):
        validate_document(recording_document)


@pytest.mark.parametrize("position", (float("inf"), float("nan")))
def test_actual_housing_gap_must_be_finite(recording_document, position):
    _housing(recording_document)["local_start_z_mm"] = position
    with pytest.raises(ValueError, match="actual housing gap must be finite and non-negative"):
        _validate_projector_lens_clearances(recording_document["parts"], recording_document["geometry"])
