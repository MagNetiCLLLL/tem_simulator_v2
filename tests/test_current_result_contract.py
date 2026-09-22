"""Current result producers and consumers share one unambiguous schema."""
import pytest

from temsim.optics.column import default_state
from temsim.simulation_pipeline import CalculationResult, aperture_stop_records


def test_unrecorded_result_has_independent_empty_signature_dictionary():
    first = CalculationResult(simulation=None, energy_filter=None)
    second = CalculationResult(simulation=None, energy_filter=None)
    first.signatures["request"] = "recorded"
    assert second.signatures == {}


@pytest.mark.parametrize("signatures", [None, [], "request"])
def test_result_rejects_invalid_signature_container(signatures):
    with pytest.raises(TypeError, match="dictionary"):
        CalculationResult(simulation=None, energy_filter=None, signatures=signatures)


@pytest.mark.parametrize("signatures", [{"request": None}, {"request": ""}, {1: "digest"}])
def test_result_rejects_invalid_signature_identity(signatures):
    with pytest.raises(ValueError, match="nonempty string"):
        CalculationResult(simulation=None, energy_filter=None, signatures=signatures)


def test_all_current_aperture_records_use_only_diameter():
    state = default_state()
    records = aperture_stop_records(state)
    assert records
    assert all("diameter_mm" in row and "radius_mm" not in row for row in records)
    runtime = {a.key: a for a in state.apertures}
    runtime.update({a.key: a for a in (state.electron_gun.dpa_aperture,
                                      state.electron_gun.c1_aperture)})
    assert all(row["diameter_mm"] == runtime[row["key"]].diameter_mm for row in records)
